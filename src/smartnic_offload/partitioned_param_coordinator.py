import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import logger

from dataclasses import dataclass
import collections
from collections import UserDict
from typing import Deque, Set
from torch.cuda import Event, Stream
import math
import types
from typing import Callable, Iterable
from enum import Enum
import functools
import itertools
from typing import List

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter
import torch.distributed as dist
from stage3_utils import * #parameterの持ち方が違うため独自のmemory_usage関数を呼ぶ
from partition_parameters import *

def debug_rank0(message: str) -> None:
    if dist.get_rank() == 0:
        logger.debug(message)


@instrument_w_nvtx
def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse),
                           sub_module.ds_external_parameters())


def iter_params(module: Module, recurse=False) -> Iterable[Parameter]:
    return map(lambda pair: pair[1], get_all_parameters(module, recurse))


class ZeRoTraceMode(Enum):
    # Record trace of the network during a single forward+backward (for training) or forward (for inference)
    RECORD = 1
    # Use recorded network trace to optimize current forward+backward or forward
    COMPLETE = 2
    # Recorded trace does not match current forward+backward or forward pass.
    INVALID = 3
    

class PartitionedParameterCoordinator:
    #目的: 「いま all-gather を非同期発行して**転送中（INFLIGHT）**のパラメータ」を管理するレジストリ。
    class __InflightParamRegistry(UserDict):
        """registry for parameters in flight"""
        def __setitem__(self,
                        param: Parameter,
                        handle: AllGatherCoalescedHandle) -> None:
            if param in self.data:
                raise RuntimeError(f"{param.ds_summary()} already in registry")
            if param.ds_status != ZeroParamStatus.INFLIGHT:
                raise RuntimeError(
                    f"attempted to add non-inflight parameter to registry {param.ds_summary()}"
                )
            self.data[param] = handle #キー/値: param: Parameter → handle: AllGatherCoalescedHandle
    
    @dataclass
    class __ParamInTrace:
        param: Parameter
        step_id_last_used_at: int

    def __init__(
        self,
        prefetch_bucket_sz: int, #プリフェッチのバッチ単位（バケツ）を要素数（numel）で表した上限。→ ここまでの“合計要素数”になるようにパラメータを束ねて先取り all-gather する。
        max_reuse_distance_in_numel: int, #再利用距離の上限（要素数ベース）。→ あるパラメータが次に使われるまでの「トレース距離」がこの閾値を超えたら、早めに release してメモリを空ける判断材料に。
        max_available_parameters_in_numel: int, #同時に“利用可能（AVAILABLE）”として保持してよい総要素数の上限。→ この上限を超えないように、古い／再利用距離が長いパラメータから解放
        allgather_stream: Stream, #all-gather 専用に使う CUDA ストリーム。→ 計算ストリームと分離して 通信と計算のオーバーラップを図る。
    ) -> None:
        self.__inflight_param_registry = __class__.__InflightParamRegistry() #非同期 all-gather 中の param → handle の対応表
        self.__step_id: int = 0 #いままでに“実行されたサブモジュール数”の通しカウンタ。→ forward/backward のモジュール呼び出しごとにインクリメント。→ トレース（使用順序の記録）や再利用距離の計算に使う

        self.__trace_mode: ZeRoTraceMode = ZeRoTraceMode.RECORD #トレースの動作モード。RECORD: 実行順と param 使用を記録する段階。実装によっては REPLAY: 記録済みの順序に従い先読みを最適化する段階。

        self.__submodule_order: Iterable[Module] = [] #実行されたサブモジュールの列（forward + backward）。→ トレース結果として並ぶ（順序ベースのプリフェッチ計画に利用）。
        self.__param_order: Iterable[__class__.__ParamInTrace] = [] #使用されたパラメータ列（param と最終使用 step のペア）。→ パラメータ単位の先読み／解放判断に使うトレース
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(
            lambda: int(-1e10)) #その param を最後にフェッチした step_id を記録（既定は非常に小さい負値）
        self.__step_id_module_fetched_for = collections.defaultdict(
            lambda: collections.deque()) #各モジュールについて「どの step_id でフェッチしたか」の履歴

        self.__n_available_params: int = 0 #現在 AVAILABLE（＝フル or 使える状態）として保持している総要素数のカウント
        self.__max_n_available_params: int = max_available_parameters_in_numel #AVAILABLE 総要素数の上限（= max_available_parameters_in_numel）。
        self.__max_reuse_dist_in_numel: int = max_reuse_distance_in_numel #再利用距離しきい値（要素数）。 → これより“遠い”将来にしか使わない param は先に release_sub_module/parameter の対象に。

        # side of the dequeue as they are fetched
        self.__param_queue: Deque[__class__.__ParamInTrace] = None #プリフェッチ予定の param キュー。→ 左からポップして fetch_sub_module / fetch_param を発行。→ トレースに基づき、先読み順序を制御。
        self.__prefetch_bucket_sz: int = prefetch_bucket_sz #プリフェッチ 1 回でまとめる合計要素数の上限。→ キューから足し込み、閾値までをひとかたまりの all-gatherとして投げる
        
        self.hierarchy: int = 0 #ログ/計測用のインデント階層や、階層型プリフェッチの段階を表すカウンタ

        # stream that will be used for allgather operations
        self.__allgather_stream: Stream = allgather_stream #all-gather を投げる CUDA ストリーム。→ 既定の計算ストリームと分離することで、通信と計算のオーバーラップを実現

        self.__ongoing_fetch_events: Deque[Event] = collections.deque() #現在キューに積まれている“フェッチ（all-gather / swap-in）イベント”のトラッキング
        self.__max_ongoing_fetch_events: int = 1024 #同時にキューできるフェッチイベント数の上限
        
    def _clear_trace_structures(self) -> None:
        self.__submodule_order = []
        self.__param_order = []
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(
            lambda: int(-1e10))
        self.__param_queue = None

    def is_complete_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.COMPLETE

    def is_invalid_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.INVALID

    def is_record_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.RECORD

    def _invalidate_trace(self) -> None:
        if self.is_invalid_trace():
            raise RuntimeError("attempted to invalidate already invalid trace")
        self.__trace_mode = ZeRoTraceMode.INVALID
        self._clear_trace_structures()
        
    #目的: すでに**完全トレース（COMPLETE）**がある場合に、実行順が前回と一致しているか検証。
    def trace_prologue(self, sub_module: Module) -> None:
        if self.is_complete_trace():
            # sub_module must match expectation else invalidate trace cache
            if sub_module != self.__submodule_order[self.__step_id]:
                expected_module_id = self.__submodule_order[self.__step_id].id
                debug_rank0(
                    f"Invalidate trace cache @ step {self.__step_id}: "
                    f"expected module {expected_module_id}, but got module {sub_module.id}"
                )
                self._invalidate_trace() #不一致なら「トレースキャッシュを無効化」し（_invalidate_trace()）、再学習フェーズへ戻す。

    #目的: トレース記録（RECORD）中に、実行されたサブモジュールを順に保存
    def record_module(self, sub_module: Module) -> None: 
        """adds sub module to trace"""
        if not self.is_record_trace():
            raise RuntimeError(
                f"attempted to record trace when status = {self.__trace_mode}")

        self.__submodule_order.append(sub_module)
        self.__step_id_module_fetched_for[sub_module.id].append(self.__step_id)
        
    #目的: モジュール → パラメータへの展開を、トレース列として記録
    def record_parameters(self, sub_module: Module) -> None:
        """adds sub module to trace"""
        if not self.is_record_trace():
            raise RuntimeError(
                f"attempted to record trace when status = {self.__trace_mode}")

        step_id = self.__step_id_module_fetched_for[sub_module.id].popleft()
        for param in sorted(set(iter_params(sub_module)), key=lambda p: p.ds_id):
            self.__param_order.append(
                __class__.__ParamInTrace(param=param,
                                         step_id_last_used_at=step_id))
            
    #目的: すでに溜めた __submodule_order から、パラメータ使用順トレース（__param_order）を組み立て直す。
    def construct_parameter_trace_from_module_trace(self):
        """use module trace to construct parameter trace"""
        self.__param_order = []
        for sub_module in self.__submodule_order:
            self.record_parameters(sub_module)

    #目的: 1 回の fwd+bwd が終わったタイミングで、トレース状態を更新・確定し、次ステップへ初期化。
    def reset_step(self) -> None:
        """indicate that we have completed one fwd+bwd for the model"""
        if self.__inflight_param_registry: #__inflight_param_registry が空であること（未完了の all-gather が残っていない）。
            raise RuntimeError(
                f"still have inflight params "
                f"{[p.ds_summary() for p in self.__inflight_param_registry.keys()]}")

        '''
        まだ COMPLETE でない場合、全 rank で
        __submodule_order の id 列
        __param_order の param.ds_id 列
        __param_order の step_id_last_used_at 列
        が一致しているか assert_ints_same_as_other_ranks(...) で検証。
        '''
        if not self.is_complete_trace():  # not self.trace_complete:
            # Make sure that recorded parameter and submodule orders are
            # identical across ranks
            assert_ints_same_as_other_ranks([m.id for m in self.__submodule_order])
            assert_ints_same_as_other_ranks([p.param.ds_id for p in self.__param_order])
            assert_ints_same_as_other_ranks(
                [p.step_id_last_used_at for p in self.__param_order])

            if self.is_record_trace():
                # Successfully recorded a trace ここでforward, backwardの順序を確定させて記録
                self.construct_parameter_trace_from_module_trace()
                self.__submodule_order = tuple(self.__submodule_order)  # freeze
                self.__param_order = tuple(self.__param_order)  # freeze
                self.__trace_mode = ZeRoTraceMode.COMPLETE
                print_rank_0(
                    f"completed record trace: {[m.id for m in self.__submodule_order]}",
                    force=False)
            else:
                # Enable trace recording for next forward/backward pass やり直し
                self.__trace_mode = ZeRoTraceMode.RECORD

        self.__param_queue = collections.deque(self.__param_order)  #__param_queue = deque(__param_order)（先読み対象のキュー）
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(
            lambda: int(-1e10))
        self.__step_id_module_fetched_for = collections.defaultdict(
            lambda: collections.deque())
        self.__step_id = 0
        self.__n_available_params = 0


    #以下二つはデバッグ用
    def _dump_params(self, tag, sub_module, params, step_id=None):
        if step_id is None:
            step_id = self.__step_id
        param_names = [debug_param2name_id(p) for p in params]
        print(
            f'{tag} step = {step_id} mod = {debug_module2name_id(sub_module)} p_names = {param_names}'
        )

    def _dump_param_ids(self, tag, mod_id, p_ids, step_id=None):
        if step_id is None:
            step_id = self.__step_id
        print(f'{tag} mod = {mod_id}, step = {step_id}, p_ids = {p_ids}')

    '''ここで使う直前にparameterをall_gatherする'''
    #この層を実行する直前に呼ばれる。必要パラメータの all-gather を起動 → 必要分が揃うまで待機 → 将来分を先読みする。
    @instrument_w_nvtx
    @torch.no_grad()
    def fetch_sub_module(self, current_submodule: Module) -> None:
        """This method does the following (in order):
        1. kick off fetch for parameters in immediately required sub module
        2. kick off fetch for next few parameters we will need later (prefetch)
        3. block on parameters in immediately required sub module
        """
        debug_rank0(
            f"{self.__step_id}: M{current_submodule.id}({type(current_submodule).__name__}) P{[p.ds_id for p in iter_params(current_submodule)]} "
            + str({
                "avail": f"{self.__n_available_params:.1e}",
                "queue_sz": f"{len(self.__param_queue or [])}",
                "inflight": [p.ds_id for p in self.__inflight_param_registry],
            }))

        params_to_fetch = frozenset(iter_params(current_submodule)) #現サブモジュールに属するパラメータ集合を不変セット化

        # kick off all gather for params in the immediately required submodule
        for param in params_to_fetch:
            debug_rank0(f"-fetch: {param.ds_summary()}")
        self.__all_gather_params(params_to_fetch) #__all_gather_params を呼び、未利用（NOT_AVAILABLE）のものに対して非同期AllGatherを投げる

        for param in params_to_fetch:
            param.ds_active_sub_modules.add(current_submodule.id) #各パラメータに「このサブモジュールが現利用中」であることを記録（後の解放判定に使う）
            debug_rank0(f"-wait: {param.ds_summary()}")
            if param in self.__inflight_param_registry: #AllGather中
                with torch.cuda.stream(self.__allgather_stream):
                    while self.__ongoing_fetch_events and self.__ongoing_fetch_events[0].query():
                        self.__ongoing_fetch_events.popleft() #先頭のCUDAイベントが完了済みならデキュー（古い完了イベントを掃除）。
                    if len(self.__ongoing_fetch_events) > self.__max_ongoing_fetch_events:
                        self.__ongoing_fetch_events.popleft().synchronize() #同時に保持するイベント数が上限超過なら、最古のイベントを同期（完了待ちしてから破棄）し、バックプレッシャをかける

                    self.__inflight_param_registry.pop(param).wait() #該当パラメータに紐づくAllGatherハンドルを取り出し、完了待ち（wait()）

                    event = Event()
                    event.record()
                    self.__ongoing_fetch_events.append(event) #今回のフェッチ完了の目印としてCUDAイベントを記録・キューに積む

            assert param.ds_status == ZeroParamStatus.AVAILABLE, param.ds_summary()
        torch.cuda.current_stream().wait_stream(self.__allgather_stream)

        # kick off parameter prefetches for upcoming modules
        # don't prefetch if we dont have a completed model trace
        if self.is_complete_trace(): #モデルのアクセス順トレースが完成している場合のみ、将来使うパラメータを先読み
            discarded_from_prefetch_queue = set()
            params_not_already_fetched = set(
                filter(
                    lambda p: self.__most_recent_step_id_param_fetched_for[p] < self.
                    __step_id,
                    params_to_fetch)) #今回必要なパラメータのうち「このステップでまだフェッチ扱いになっていない」ものを抽出
            #パラメトレース用キューの先頭から、今回必要分に相当する個数だけ取り除く（「今回で使う分は先読み対象から除外」する意味）。
            # ついでに「最後に使われたステップID」を更新
            while self.__param_queue and len(discarded_from_prefetch_queue) < len(
                    params_not_already_fetched):
                param_in_trace = self.__param_queue.popleft()
                self.__most_recent_step_id_param_fetched_for[
                    param_in_trace.param] = param_in_trace.step_id_last_used_at
                discarded_from_prefetch_queue.add(param_in_trace.param)

            if discarded_from_prefetch_queue != params_not_already_fetched:
                raise RuntimeError(
                    f"tracing error at step {self.__step_id}: \n"
                    f"module id: {current_submodule.id}, training: {current_submodule.training}\n"
                    f"expected the next {len(params_not_already_fetched)} parameters in the "
                    f"parameter fetch queue to be {tuple(p.ds_summary(use_debug_name=True) for p in params_not_already_fetched)} \n"
                    f"but got \n {tuple(p.ds_summary(use_debug_name=True) for p in discarded_from_prefetch_queue)}."
                )

            # kick off all gather for params in the next few submodules (prefetch)
            if self.__prefetch_bucket_sz > 0: #先読みを有効化している場合のみ進む
                max_params_to_prefetch = min(
                    self.__max_n_available_params - self.__n_available_params,
                    self.__prefetch_bucket_sz) #先読みできる上限（メモリ余力とプリフェッチバケットサイズの小さい方）
                params_to_prefetch = set()
                numel_prefetching = 0
                while self.__param_queue and numel_prefetching < max_params_to_prefetch: #トレースキューから将来使う順に取り出しつつ、先読み候補を集める
                    param_in_trace: __class__.__ParamInTrace = self.__param_queue.popleft(
                    )

                    do_prefetch = param_in_trace.param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                    if param_in_trace.param in params_to_prefetch: #まだ手元にない＆重複でないときだけ先読み対象
                        # Avoid duplicates
                        do_prefetch = False

                    self.__most_recent_step_id_param_fetched_for[param_in_trace.param] = \
                        max(self.__most_recent_step_id_param_fetched_for[param_in_trace.param],
                            param_in_trace.step_id_last_used_at) #先読み対象の「最後に使われたステップ」を最新に更新（スキップ検出・解放抑止に使う）

                    if do_prefetch:
                        params_to_prefetch.add(param_in_trace.param)
                        numel_prefetching += param_in_trace.param.ds_numel #先読み集合に追加し、先読み総要素数（numel）を加算。上限に達したらループ終了

                for param in params_to_prefetch:
                    debug_rank0(f"-prefetch: {param.ds_summary()}")
                self.__all_gather_params(params_to_prefetch) #prefetchしたパラメータのAllGather

        self.__step_id += 1
        
    # サブモジュールのパラメータを、条件を満たせば解放（＝再分割してメモリから退避）
    @instrument_w_nvtx
    @torch.no_grad()
    def release_sub_module(self, submodule: Module) -> None:
        """release the parameters of a sub module, assuming they meet conditions to
        be released."""
        params_to_release = (self.__params_to_release(submodule, self.__step_id)
                             if self.is_complete_trace() else set(p.ds_id for p in iter_params(submodule))) #releaseしてもいいパラメータを決定する
        for param in iter_params(submodule):
            param.ds_active_sub_modules.discard(submodule.id) #当該サブモジュールの「利用中フラグ」を解除
            if param.ds_id in params_to_release and not param.is_external_param:
                self.__release_param(param) #解放対象で、外部パラメータ（解放禁止）でなければ解放処理へ

    @instrument_w_nvtx
    @torch.no_grad()
    def release_and_reset_all(self, module: Module) -> None: #モジュール配下（再帰）の全パラメータを強制的に解放＆状態初期化
        """release all module parameters"""
        for param in iter_params(module, recurse=True):
            if param in self.__inflight_param_registry:
                raise RuntimeError(f"param {param.ds_summary()} still in flight")

            # there's a hook execution issue
            param.ds_active_sub_modules.clear()
            self.__release_param(param) #現在はアクティブサブモジュールが残っていてもクリアして解放（コメントは将来振る舞いの変更示唆)

        for param in iter_params(module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f"{param.ds_summary()} expected to be released")
            
    def __all_gather_params(self, params: Set[Parameter]) -> None:
        """for each partitioned parameter, kick off an async allgather and store
        the work handle for the in flight parameters."""
        partitioned_params = []
        for param in params:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                partitioned_params.append(param)
                self.__n_available_params += param.ds_numel

        if partitioned_params:
            with torch.cuda.stream(self.__allgather_stream):
                handle = partitioned_params[0].all_gather_coalesced(partitioned_params)

            for param in partitioned_params:
                assert param.ds_status == ZeroParamStatus.INFLIGHT, param.ds_summary()
                self.__inflight_param_registry[param] = handle
    
    @instrument_w_nvtx
    def __release_param(self, param: Parameter) -> None:
        if param.ds_status == ZeroParamStatus.AVAILABLE and not param.ds_active_sub_modules:
            debug_rank0(f"-release: {param.ds_summary()}")
            param.partition() #paramを分割して全体を解放
            self.__n_available_params -= param.ds_numel
            
    # ZeroOffload などクラス外から「解放すべき param の ds_id セット」を取得するための公開メソッド
    def params_to_release_for_submodule(self, submodule: Module) -> Set[int]:
        # 内部の trace 状態と step_id を使って計算する
        return self.__params_to_release(submodule, self.__step_id)
                
    @instrument_w_nvtx
    #@functools.lru_cache(maxsize=None)
    def __params_to_release(self,
                            submodule_to_release: Module,
                            step_id: int) -> Set[int]:
        if not self.is_complete_trace():
            raise RuntimeError("expected trace to be complete")

        params_to_release = set(p.ds_id for p in iter_params(submodule_to_release) if not p.ds_persist) #当該サブモジュールのパラメータのうち「永続（ds_persist）ではない」IDを初期候補にする
        # プリフェッチ時に「再利用がスキップされた」印がある (most_recent_step_id_param_fetched_for[param]
        # が現在ステップより未来) 場合は解放対象から外す。先読みスキップがあると解放後の prefetch 順が崩れ
        # 性能/バッファに悪影響するため。

        for param in iter_params(submodule_to_release):
            if self.__most_recent_step_id_param_fetched_for[param] > step_id:
                params_to_release.discard(param.ds_id)

        params_traversed = 0
        for module in self.__submodule_order[step_id:]:
            if params_traversed >= self.__max_reuse_dist_in_numel: #「この先すぐ（numelで測った距離制限以内）に再出現するパラメータ」は解放しない
                break
            for param in iter_params(module):
                params_to_release.discard(param.ds_id)
                params_traversed += param.ds_numel
        
        return params_to_release