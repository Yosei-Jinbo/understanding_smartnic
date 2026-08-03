import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import logger

import math
import types
from typing import Callable, Iterable
from enum import Enum
import functools
import itertools
from typing import List

import time

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter
import torch.distributed as dist
from stage3_utils import * #parameterの持ち方が違うため独自のmemory_usage関数を呼ぶ

# H2D/D2H の転送時間は CUDA Event ベースの計測 (_accumulate_* 系) で取得する。



class PartitionedParamStatus(Enum):
    # Partitioned parameters are present and ready for use
    AVAILABLE = 1

    # partitioned params are in some non-memory device
    NOT_AVAILABLE = 2

    # partitioned params are being read from some non-memory device.
    INFLIGHT = 3


class ZeroParamType(Enum):
    # same as regular pytorch parameters
    NORMAL = 1
    # parameters are partitioned across data parallel process
    PARTITIONED = 2
    # the parameter is held with a unique process rank
    # and is not available on all other process
    REMOTE = 3


class ZeroParamStatus(Enum):
    # parameters are fully present and ready for use on all processes
    AVAILABLE = 1
    # parameters are either partitioned or remote in some or all process
    NOT_AVAILABLE = 2
    # parameters are being gathered.
    INFLIGHT = 3
    

param_count = 0
partitioned_param_data_shape = [0]
zero_init_enabled = False

# ---- all_gather 前の param shard H2D の GPU 時間集計 (CUDA event based, per-rank ms) ----
PARAM_SHARD_H2D_TIME_MS: float = 0.0
PARAM_SHARD_H2D_CALLS: int = 0

def _accumulate_param_shard_h2d_time_ms(delta_ms: float) -> None:
    global PARAM_SHARD_H2D_TIME_MS, PARAM_SHARD_H2D_CALLS
    PARAM_SHARD_H2D_TIME_MS += float(delta_ms)
    PARAM_SHARD_H2D_CALLS += 1

def get_param_shard_h2d_time_ms() -> float:
    return float(PARAM_SHARD_H2D_TIME_MS)

def get_param_shard_h2d_calls() -> int:
    return int(PARAM_SHARD_H2D_CALLS)

def allgather_fn(output_tensor: torch.Tensor,
                 input_tensor: torch.Tensor,
                 group=None,
                 async_op: bool=False):
    """各 rank の input_tensor をフラットな output_tensor ([world_size * local_numel]) へ all-gather。
    async_op=True なら Work ハンドルを返す。"""

    if group is None:
        group = dist.group.WORLD

    world_size = dist.get_world_size(group=group)

    # 形状・dtype / device の軽い防御
    if output_tensor.numel() != input_tensor.numel() * world_size:
        raise ValueError(
            f"output numel={output_tensor.numel()} must be input numel * world_size "
            f"({input_tensor.numel()} * {world_size})"
        )
    if output_tensor.dtype != input_tensor.dtype:
        raise TypeError("output_tensor.dtype must match input_tensor.dtype")
    if output_tensor.device != input_tensor.device:
        raise TypeError("output_tensor.device must match input_tensor.device")

    # 1) 公開API (PyTorch 1.12+)
    if hasattr(dist, "all_gather_into_tensor"):
        return dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=group, async_op=async_op
        )

    # 2) 互換の base API（非公開・古いが多くの環境で利用可）
    if hasattr(dist, "_all_gather_base"):
        return dist._all_gather_base(
            output_tensor, input_tensor, group=group, async_op=async_op
        )

    # 3) 最後の手段：リスト版 all_gather（遅い・Pythonオーバーヘッド有り）
    # output_tensor を world_size 個に分割して受け皿にする
    chunks = list(torch.chunk(output_tensor, world_size, dim=0))
    # async_op=True のとき Work が返る。False のときは None。
    return dist.all_gather(
        chunks, input_tensor, group=group, async_op=async_op
    )
    

def _dist_allgather_fn(input_tensor: Tensor, output_tensor: Tensor, group=None):
    return instrument_w_nvtx(allgather_fn)(output_tensor,
                                                input_tensor,
                                                group=group,
                                                async_op=True)

def print_rank_0(message, debug=False, force=False):
    rank = dist.get_rank()
    if rank == 0 and (debug or force):
        print(message)
        
def debug_rank0(msg: str) -> None:
    if dist.get_rank() == 0:
        logger.debug(msg)
        
def is_zero_param(parameter):
    if not torch.is_tensor(parameter):
        return False
    return hasattr(parameter, "ds_id")

def _init_external_params(module):
    if not hasattr(module, '_external_params'):
        module._external_params = {}
        
        def external_parameters(self):
            return self._external_params.items()
        def all_parameters(self): #(自分のmoduleのパラメータ, 外部パラメータ)をつなげる
            return itertools.chain(self.named_parameters(self, recurse=False), external_parameters(self))
        
        module.ds_external_parameters = types.MethodType(external_parameters, module)
        module.all_parameters = types.MethodType(all_parameters, module)

#「このモジュールの forward 実行時には、この外部パラメータも必要だから事前に集めておいてね」と DeepSpeed に明示的に伝えます。
def register_external_parameter(module, parameter):
    if not hasattr(module, '_external_params'):
        _init_external_params(module) #moduleに外部パラメータを参照する辞書がないなら新しく作成
    
    key = id(parameter)
    module._external_params[key] = parameter #外部パラメータの登録
    
def unregister_external_parameter(module, parameter):
    key = id(parameter)
    del module._external_params[key]
    
_orig_torch_empty = torch.empty
_orig_torch_zeros = torch.zeros
_orig_torch_ones = torch.ones
_orig_torch_full = torch.full


def zero_wrapper_for_fp_tensor_constructor(fn: Callable,
                                           target_fp_dtype: torch.dtype) -> Callable:
    def wrapped_fn(*args, **kwargs) -> Tensor:
        if kwargs.get("device", None) is None:
            kwargs['device'] = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
        tensor: Tensor = fn(*args, **kwargs)
        if tensor.is_floating_point():
            tensor = tensor.to(target_fp_dtype)

        return tensor

    return wrapped_fn


def get_new_tensor_fn_for_dtype(dtype: torch.dtype) -> Callable:
    def new_tensor(cls, *args) -> Tensor:
        device = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
        tensor = _orig_torch_empty(0, device=device).new_empty(*args)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype)

        return tensor

    return new_tensor


# https://stackoverflow.com/a/63851681/9201239
def get_all_subclasses(cls):
    subclass_list = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


@instrument_w_nvtx
def free_param(param: Parameter) -> None:
    """Free underlying storage of a parameter."""
    assert not param.ds_active_sub_modules, param.ds_summary()
    if param.data.is_cuda:
        # need to make sure that we don't free the parameter while it is still
        # being used for computation
        param.data.record_stream(torch.cuda.current_stream())
    # param.data doesn't store anything meaningful in partitioned state
    param.data = torch.empty(0, dtype=param.dtype, device=param.device) #このデバイスはGPU (根拠: フルパラメータであるparam.dataはGPUにないとそもそも行列計算できないため)
    param.ds_status = ZeroParamStatus.NOT_AVAILABLE


reuse_buffers = False
temp_contiguous_tensor = None
empty_buffers = {}


#with ブロック中だけ全 nn.Module サブクラスの __init__ を差し替え、
#初期化直後に ZeRO-3 向け post-init (分割/収集の管理) を自動で挟み込む
class InsertPostInitMethodToModuleSubClasses(object):
    def __init__(self, enabled=True, dtype=None):
        self.enabled = enabled
        self._set_dtype(dtype)
    
    #有効化されていれば、以下の一時的な差し替えを行います（with を抜けると exit で元に戻す前提）
    def __enter__(self):
        global zero_init_enabled
        if not self.enabled:
            return
        zero_init_enabled = True
        
        #目的：Module.apply(fn) が子モジュールに対して後段初期化を掛ける際、シャード済み重みだと都合が悪いため、
        # 一時的に全量 gather → fn 実行 → rank0 から broadcast → 再分割という安全な手順に置き換える
        def apply_with_gather(orig_module_apply_fn: Callable) -> Callable:
            def get_wrapped_fn_to_apply(fn_to_apply: Callable) -> Callable:
                if hasattr(fn_to_apply, "wrapped"):
                    return fn_to_apply
                
                @functools.wraps(fn_to_apply)
                def wrapped_fn_to_apply(module_to_apply_fn_to: Module) -> None:
                    """apply 前に all-gather し、fn 適用後 rank0 から broadcast して再分割する。"""
                    if not all(is_zero_param(p) for p in module_to_apply_fn_to.parameters(recurse=False)):
                        raise RuntimeError(
                            f"not all parameters for {module_to_apply_fn_to.__class__.__name__}, "
                            f"were zero params, is it possible that the parameters were "
                            f"overwritten after they were initialized? "
                            f"params: {[p for p in module_to_apply_fn_to.parameters(recurse=False)]} "
                        )
                    params_to_apply_fn_to: Iterable[Parameter] = list(
                        sorted(module_to_apply_fn_to.parameters(recurse=False),
                               key=lambda p: p.ds_id))
                    
                    for param in params_to_apply_fn_to:
                        param.all_gather()
                        
                    fn_to_apply(module_to_apply_fn_to)
                    
                    for param in params_to_apply_fn_to:
                        dist.broadcast(param.data, 0, group=param.ds_process_group)
                        
                    for param in params_to_apply_fn_to:
                        param.partition(has_been_updated=True)
                        
                wrapped_fn_to_apply.wrapped = True
                
                return wrapped_fn_to_apply
            
            #実際の Module.apply を、「上記で生成した gather/broadcast/再分割付きの関数」で呼ぶように置き換える。
            @functools.wraps(orig_module_apply_fn)
            def wrapped_apply(module: Module, fn_to_apply: Callable) -> None:
                orig_module_apply_fn(module, get_wrapped_fn_to_apply(fn_to_apply))

            return wrapped_apply
        
        #目的：多重継承で親→子の順に __init__ が呼ばれる際、**実際の post-init を“子の __init__ 完了直後に 1 回だけ”**走らせる。
        def partition_after(f):
            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):

                # post_init は子の __init__ 完了直後に 1 回だけ走らせる (親・祖先の __init__ 後には走らせない)

                print_rank_0(f'Before initializing {module.__class__.__name__}',
                             force=False)

                is_child_module = False
                if not hasattr(module, "_ds_child_entered"):
                    # child's __init__ was called, since parents all see the same object they can now skip post_init
                    is_child_module = True
                    setattr(module, "_ds_child_entered", True)

                f(module, *args, **kwargs)

                if is_child_module:
                    # child's __init__ is done, now we can run a single post_init on the child object
                    delattr(module, "_ds_child_entered")

                    print_rank_0(f'Running post_init for {module.__class__.__name__}',
                                 force=False)
                    self._post_init_method(module)

                print_rank_0(
                    f'After initializing followed by post init for {module.__class__.__name__}',
                    force=False)

            return wrapper
        
        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = partition_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls.__init__ = partition_after(cls.__init__)
            
        # Replace .__init__() for all existing subclasses of torch.nn.Module recursively
        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            _enable_class(subclass)

        # holding onto some methods so we can put them back the way they were in __exit__
        torch.nn.modules.module.Module._old_init_subclass = torch.nn.modules.module.Module.__init_subclass__
        torch.nn.modules.module.Module._old_apply = torch.nn.modules.module.Module.apply
        torch.Tensor.__old_new__ = torch.Tensor.__new__

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)
        torch.nn.modules.module.Module.apply = apply_with_gather(
            torch.nn.modules.module.Module._old_apply)

        torch.Tensor.__new__ = get_new_tensor_fn_for_dtype(self.dtype)
        torch.empty = zero_wrapper_for_fp_tensor_constructor(_orig_torch_empty,
                                                             self.dtype)
        torch.zeros = zero_wrapper_for_fp_tensor_constructor(_orig_torch_zeros,
                                                             self.dtype)
        torch.ones = zero_wrapper_for_fp_tensor_constructor(_orig_torch_ones, self.dtype)
        torch.full = zero_wrapper_for_fp_tensor_constructor(_orig_torch_full, self.dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return

        shutdown_init_context()

        if dist.get_rank() == 0:
            logger.info("finished initializing model with %.2fB parameters",
                        param_count / 1e9)

        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def _set_dtype(self, dtype):
        self.dtype = dtype or torch.float32
        
def shutdown_init_context():
    global zero_init_enabled

    if not zero_init_enabled:
        return

    def _disable_class(cls):
        cls.__init__ = cls._old_init

    # Replace .__init__() for all existing subclasses of torch.nn.Module
    for subclass in get_all_subclasses(torch.nn.modules.module.Module):
        _disable_class(subclass)

    # putting methods back the way we found them
    torch.nn.modules.module.Module.__init_subclass__ = torch.nn.modules.module.Module._old_init_subclass
    torch.nn.modules.module.Module.apply = torch.nn.modules.module.Module._old_apply

    torch.Tensor.__new__ = torch.Tensor.__old_new__
    torch.empty = _orig_torch_empty
    torch.zeros = _orig_torch_zeros
    torch.ones = _orig_torch_ones
    torch.full = _orig_torch_full

    zero_init_enabled = False

'''どっかでwaitを呼ぼうね!! partitioned_param_coordinator.pyで呼ばれてます'''
class AllGatherHandle:
    def __init__(self, handle, param: Parameter, t_request=None, start_event=None) -> None:
        if param.ds_status != ZeroParamStatus.INFLIGHT:
            raise RuntimeError(f"expected param {param.ds_summary()} to be available")
        self.__handle = handle
        self.__param = param

    def wait(self) -> None:
        instrument_w_nvtx(self.__handle.wait)()
        self.__param.ds_status = ZeroParamStatus.AVAILABLE

class AllGatherCoalescedHandle:
    def __init__(self, allgather_handle, params: List[Parameter], partitions: List[Tensor], world_size: int, t_request=None, start_event=None) -> None:
        self.__allgather_handle = allgather_handle
        self.__params = params
        self.__partitions = partitions
        self.__world_size = world_size
        self.__complete = False
        for param in self.__params:
            if param.ds_status != ZeroParamStatus.INFLIGHT:
                raise RuntimeError(
                    f"expected param {param.ds_summary()} to not be available")

    @instrument_w_nvtx
    def wait(self) -> None:
        if self.__complete:
            return

        instrument_w_nvtx(self.__allgather_handle.wait)()

        param_offset = 0
        for param in self.__params:
            assert param.ds_status == ZeroParamStatus.INFLIGHT, f"expected param {param.ds_summary()} to be inflight"
            partitions: List[Tensor] = []
            for rank in range(self.__world_size):
                param_start = rank * param.ds_tensor.ds_numel
                if param_start < param.ds_numel:
                    part_to_copy = self.__partitions[rank].narrow(0, param_offset, min(param.ds_numel-param_start, param.ds_tensor.ds_numel))
                    partitions.append(part_to_copy)

            param.data = instrument_w_nvtx(torch.cat)(partitions).view(param.ds_shape)
            param.ds_status = ZeroParamStatus.AVAILABLE

            from common import debug_params as dbg
            dbg.log_ag_complete(param, tag="AG_COMPLETE(coalesced)")

            for part_to_copy in partitions:
                part_to_copy.record_stream(torch.cuda.current_stream())
            param_offset += param.ds_tensor.ds_numel

        self.__partitions = None   # flat_tensor への参照を解放し、GPU メモリの再利用を可能にする
        self.__complete = True

class CPUAllGatherCoalescedHandle:
    """cpu_full_param からのローカルコピーだけで all-gather を完了させるハンドル。
    通信は行わず、.wait() 時に CPU→GPU コピーして AVAILABLE にする。"""

    def __init__(self, params: List[Parameter], device: torch.device, t_request=None, start_event=None) -> None:
        self.__params = list(params)
        self.__device = device
        self.__complete = False

        # all_gather_coalesced 側で INFLIGHT にされている前提
        for p in self.__params:
            if p.ds_status != ZeroParamStatus.INFLIGHT:
                raise RuntimeError(
                    f"expected param {p.ds_summary()} to be inflight for CPUAllGatherCoalescedHandle"
                )

    @instrument_w_nvtx
    def wait(self) -> None:
        if self.__complete:
            return

        with torch.no_grad():
            for p in self.__params:
                cpu_full = getattr(p, "cpu_full_param", None)
                if cpu_full is None:
                    raise RuntimeError(
                        f"cpu_full_param is None for param {p.ds_summary()} in CPUAllGatherCoalescedHandle"
                    )
                if cpu_full.numel() != p.ds_numel:
                    raise RuntimeError(
                        f"numel mismatch in cpu_full_param for {p.ds_summary()}: "
                        f"cpu_full_param.numel={cpu_full.numel()} vs ds_numel={p.ds_numel}"
                    )

                # CPU -> GPU に復元（local_device か param.device に合わせる）
                full_gpu = cpu_full.to(self.__device, non_blocking=True)
                full_gpu = full_gpu.view(p.ds_shape)

                p.data = full_gpu
                p.ds_status = ZeroParamStatus.AVAILABLE

                from common import debug_params as dbg
                dbg.log_ag_complete(p, tag="AG_COMPLETE(cpu_full)")

        # H2D コピーは current stream 上で発行済み。後続の GPU op がストリーム順序で
        # 自然に完了を待つため、明示的な完了待ちは不要。
        self.__complete = True

#DeepSpeed ZeRO-3 の初期化コンテキスト (zero.Init): モデルを初期化しつつその場でパラメータを分割する
class Init(InsertPostInitMethodToModuleSubClasses):
    param_id = 0
    
    def __init__(self,
                 module=None,
                 data_parallel_group=None,
                 enabled=True,
                 remote_device=None,
                 pin_memory=False,
                 dtype=None):
        super().__init__(enabled=enabled,
                         dtype=dtype)
        #dist.is_initialized()は常にTrueになることを想定(外側のラッパーで初期化しておく)
        if data_parallel_group is None:
            self.ds_process_group = dist.group.WORLD
        else:
            self.ds_process_group = data_parallel_group

        self.rank = dist.get_rank(group=self.ds_process_group)
        self.world_size = dist.get_world_size(group=self.ds_process_group)
        
        self.local_device = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(self.local_device)
        
        self.remote_device = remote_device
        self.pin_memory = pin_memory
        
        self.param_swapper = None #Offloadはしないので無視する
        
        # If we are provided an already-allocated module to prepare.
        if module is not None:
            assert isinstance(module, torch.nn.Module)
            self._convert_to_zero_parameters(module.parameters(recurse=True))
            
        self.use_allgather_base = True #常にall_gather_baseが利用できることを想定
    
    #重みパラメータ分割に利用
    def _convert_to_zero_parameters(self, param_list):
        for param in param_list:
            if is_zero_param(param):
                continue
            self._convert_to_deepspeed_param(param) #paramにフィールド・動的なメソッドをはやす
            param.partition()
    
    #子モジュール直下のパラメータを ZeRO 分割管理に変換し、rank0 の値で揃えてシャーディング (post_init で 1 回だけ実行)
    def _post_init_method(self, module):
        print_rank_0(f'Converting Params in {module.__class__.__name__}', force=False)
        see_memory_usage(
            f"Before converting and partitioning parmas in {module.__class__.__name__}",
            force=False)
        
        global param_count
        for name, param in module.named_parameters(recurse=False): #直下パラメータの列挙（再帰しない）
            param_count += param.numel()
            if not is_zero_param(param):
                self._convert_to_deepspeed_param(param)
                print_rank_0(
                    f"Partitioning param {debug_param2name_id_shape(param)} module={debug_module2name(module)}"
                )
                
                if param.is_cuda:
                    dist.broadcast(param, 0, self.ds_process_group)
                else:
                    if dist.get_rank() == 0:
                        logger.warn(f"param `{name}` in {module.__class__.__name__} "
                                    f"not on GPU so was not broadcasted from rank 0")
                param.partition()
            see_memory_usage(
            f"Param count {param_count}. After converting and partitioning parmas in {module.__class__.__name__}",
            force=False)
                   
    #この関数でparamにメソッドやフィールドをはやす
    def _convert_to_deepspeed_param(self, param):
        #このパラメータは ZeRO により分割（シャード）管理されることを示す種別タグ。将来 Normal/Remote 等と分岐させるための識別。
        param.ds_param_type = ZeroParamType.PARTITIONED
        #ds_status はフル値の可用状態 (NOT_AVAILABLE→INFLIGHT→AVAILABLE)。shard しか持たない場合は
        #フル値が無いので NOT_AVAILABLE (shard 側の状態は ds_tensor.status が別に持つ)
        param.ds_status = ZeroParamStatus.AVAILABLE

        #元の完全テンソル形状（分割/パディング前の論理形状）。
        param.ds_shape = param.shape

        #元の要素数（パディングを除いた実数）。分割・フラット化時の整合に使う
        param.ds_numel = param.numel()

        #ローカル shard を保持する入れ物。後で “この rank が所有する断片（シャード）” テンソルが入る。
        #param.data は “フルサイズ” を指すハンドルとして使い分け、ローカル shard は ds_tensor に持つのが ZeRO-3 の流儀。
        param.ds_tensor = None

        #このパラメータを同時に必要としている forward 文脈（サブモジュール）集合。
        param.ds_active_sub_modules = set()

        #常駐（persist）フラグ。True の場合は学習中はレプリカで持ち、step 直前にだけ分割する等の最適化ポリシを示す（巨大埋め込み等の特例運用で利用）。 今は使わないかも
        param.ds_persist = False

        #外部パラメータ（所有モジュール≠利用モジュール）かどうか。register_external_parameter により利用側へ登録されると参照関係が追跡される。
        param.is_external_param = False

        #このパラメータの shard をやり取りするプロセスグループ(data_parallel_group)。all_gather や reduce_scatter はこのグループで行う。
        param.ds_process_group = self.ds_process_group
        
        # DeepSpeed Param ID
        param.ds_id = Init.param_id #Init.param_id: 付番用の グローバルな一意ID。各パラメータに ds_id として振る
        Init.param_id += 1
        
        param.cpu_full_param = None #CPUに全てのパラメータを保持する
        param.cpu_full_param_pool = None #CPUのフルパラメータをプールしておく・ピン止め森のコストは高いのでこのfull_param_poolを利用してピン止めコストを減らす
        
        #all_gather 前: param.data は free 済み (担当分は ds_tensor)。all_gather 後: param.data がフルサイズ
        def all_gather(param_list=None, async_op=False, hierarchy=0):
            cls = param
            if param_list is None:
                param_list = [cls]
            return self._all_gather(param_list, async_op=async_op, hierarchy=hierarchy)
        
        #複数パラメータの一括all_gather
        @instrument_w_nvtx
        def all_gather_coalesced(params: Iterable[Parameter], safe_mode: bool = False) -> AllGatherCoalescedHandle:
            for param in params:
                if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                    raise RuntimeError(param.ds_summary())
                param.ds_status = ZeroParamStatus.INFLIGHT #今からall_gatherするので通信中と変更する
            
            params = sorted(params, key=lambda p: p.ds_id) #全 rank で同じ順序に並べ替え。coalesced ではフラットに連結して1回の通信をするため、順序が違うと取り違え事故になる。
            debug_rank0(f"-allgather_coalesced: {[p.ds_id for p in params]}")

            # ENABLE_FULL_PARAM_TRANSFER=1 (デフォルト無効) かつ全 param が cpu_full_param を
            # 持つ場合は通信せず CPU→GPU コピーだけで済ませる
            if os.environ.get("ENABLE_FULL_PARAM_TRANSFER", "0") == "1" and \
               all(getattr(p, "cpu_full_param", None) is not None for p in params):
                t_request = time.perf_counter()
                return CPUAllGatherCoalescedHandle(params, self.local_device, t_request=t_request)

            if len(params) == 1:
                param, = params
                param_buffer = torch.empty(
                    math.ceil(param.ds_numel / self.world_size) * self.world_size,
                    dtype=param.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False
                ) #paramが1個だけならバッファを一つだけ確保してそれで通信する
                #_dist_allgather_fn はdist.all_gatherを呼び出す。自ランクのシャード→全ランク分を param_buffer に all-gather。
                _local_dev = torch.cuda.current_device()

                # --- measure CPU->GPU shard materialization time (H2D) for the single param ---
                # ds_tensor を CUDA に載せる処理を dist.all_gather と分離し、必要なら H2D 時間として加算
                _st = torch.cuda.Event(enable_timing=True)
                _ed = torch.cuda.Event(enable_timing=True)
                _st.record()
                shard_cuda = param.ds_tensor.to(_local_dev, non_blocking=True)
                _ed.record()

                # elapsed_time を取るには完了待ちが必要（ここで 1 回だけ同期）
                torch.cuda.synchronize()
                _accumulate_param_shard_h2d_time_ms(_st.elapsed_time(_ed))

                # --- measure dist allgather time separately (existing behavior) ---
                t_request = time.perf_counter()

                # _dist_allgather_fn は dist.all_gather を呼び出す。自ランクのシャード→全ランク分を param_buffer に all-gather。
                handle = _dist_allgather_fn(shard_cuda, param_buffer, self.ds_process_group)

                # 先頭から実長分だけ切り出し、元形状に view、元デバイスへ to。.data で param のストレージを差し替え
                param.data = param_buffer.narrow(0, 0, param.ds_numel).view(param.ds_shape).to(param.device)

                return AllGatherHandle(handle, param, t_request=t_request)  # このhandleが実行されるとparamにフルサイズの重みパラメータが入る
            else:
                #params に入っているのは「自分の rank が担当しているシャード（＝ds_tensor）を持っているパラメータだけ」
                partition_sz = sum(p.ds_tensor.ds_numel for p in params) #このランクが持つ全パラのシャード長の合計（連結後の 1ランク当たり長）
                #全ランクぶんの巨大フラット受け取りバッファを GPU に確保
                flat_tensor = torch.empty(partition_sz*self.world_size, dtype=get_only_unique_item(p.dtype for p in params), device=torch.cuda.current_device(), requires_grad=False)
                partitions: List[Parameter] = []
                for i in range(self.world_size):
                     #flat_tensor を rank ごとのスロットに narrow で切る（partitions[i] が rank i の連結結果置き場）
                    partitions.append(flat_tensor.narrow(0, partition_sz*i, partition_sz))
                    
                # 自ランクが持つ各パラのシャードを順序通りに cat して 1 本にし、自ランクのスロット
                # (partitions[self.rank]) に直接書き込む。
                # --- measure CPU->GPU shard materialization time (H2D) ---
                _local_dev = torch.cuda.current_device()
                _shards = []
                _ev_pairs = []
                for _p in params:
                    # H2D / D2D copy into local CUDA device; we account only when source is on CPU
                    _st = torch.cuda.Event(enable_timing=True)
                    _ed = torch.cuda.Event(enable_timing=True)
                    _st.record()
                    _tmp = _p.ds_tensor.to(_local_dev, non_blocking=True)
                    _ed.record()
                    _ev_pairs.append((_st, _ed))
                    _shards.append(_tmp)

                # synchronize once, then accumulate per-shard elapsed time
                # (elapsed_time は両イベント完了が必須。外すと RuntimeError になる)
                if _ev_pairs:
                    torch.cuda.synchronize()
                    for (_st, _ed) in _ev_pairs:
                        _accumulate_param_shard_h2d_time_ms(_st.elapsed_time(_ed))

                instrument_w_nvtx(torch.cat)(_shards, out=partitions[self.rank])

                t_request = time.perf_counter()
                handle = _dist_allgather_fn(partitions[self.rank], flat_tensor, self.ds_process_group)

                return AllGatherCoalescedHandle(
                    allgather_handle=handle,
                    params=params,
                    partitions=partitions,
                    world_size=self.world_size,
                    t_request=t_request,
                ) #このhandleが実行されるとparamにフルサイズの重みパラメータが入る
                    
        def partition(param_list=None, hierarchy=0, has_been_updated=False):
            cls = param
            print_rank_0(f"{'--'*hierarchy}----Partitioning param {debug_param2name_id_shape_device(cls)}")
            if param_list is None:
                param_list = [cls]
            self._partition(param_list, has_been_updated=has_been_updated)
            
        def reduce_gradients_at_owner(param_list=None, hierarchy=0):
            cls = param
            if param_list is None:
                param_list = [cls]
            print_rank_0(f"{'--'*hierarchy}----Reducing Gradients for param with ids {[param.ds_id for param in param_list]} to owner")
            self._reduce_scatter_gradients(param_list)
            
        def partition_gradients(param_list=None, partition_buffers=None, hierarchy=0, accumulate=False):
            cls = param
            print_rank_0(f"{'--'*hierarchy}----Partitioning param gradient with id {debug_param2name_id_shape_device(cls)}")
            if param_list is None:
                param_list = [cls]
                if isinstance(partition_buffers, torch.Tensor): #ただのテンソルならリスト表現にする
                    partition_buffers = [partition_buffers]
            
            self._partition_gradients(param_list, partition_buffers=partition_buffers, accumulate=accumulate)
        
        def aligned_size():
            return self._aligned_size(param)

        def padding_size():
            return self._padding_size(param)

        def partition_numel():
            return self._partition_numel(param)

        def item_override():
            param.all_gather()
            return param._orig_item()
        
        def ds_summary(slf: torch.Tensor, use_debug_name: bool = False) -> dict:
            return {
                "id": debug_param2name_id(slf) if use_debug_name else slf.ds_id,
                "status": slf.ds_status.name,
                "numel": slf.numel(),
                "ds_numel": slf.ds_numel,
                "shape": tuple(slf.shape),
                "ds_shape": tuple(slf.ds_shape),
                "requires_grad": slf.requires_grad,
                "grad_shape": tuple(slf.grad.shape) if slf.grad is not None else None,
                "persist": slf.ds_persist,
                "active_sub_modules": slf.ds_active_sub_modules,
            }

        def convert_to_zero_parameters(param_list):
            self._convert_to_zero_parameters(param_list) #_convert_to_deepspeed_paramを呼び出してparam.partition(この関数によりはやしたpartition()メソッド)を呼び出す

        #この関数内ではやしたall_gatherそれぞれ自分の持っているparamを全員に配りフルサイズのtensoを作り出す
        #param全体に触れるような関数を呼ぶときはこれでラップ数ることで自動的にall_gatherが呼ばれるようになる
        def allgather_before(func: Callable) -> Callable:
            def wrapped(*args, **kwargs):
                param.all_gather() 
                return func(*args, **kwargs)

            return wrapped
        
        # Collectives for gathering and partitioning parameters
        param.all_gather = all_gather
        param.all_gather_coalesced = all_gather_coalesced
        param.partition = partition

        # Collective for averaging gradients
        param.reduce_gradients_at_owner = reduce_gradients_at_owner
        param.partition_gradients = partition_gradients

        # Partitioning size utilities
        param.aligned_size = aligned_size
        param.padding_size = padding_size
        param.partition_numel = partition_numel
        param.ds_summary = types.MethodType(ds_summary, param)

        param.item = allgather_before(param.item)

        param.convert_to_zero_parameters = convert_to_zero_parameters
        
        
        
    def _aligned_size(self, param):
        return param.ds_numel + self._padding_size(param)

    def _padding_size(self, param):
        remainder = param.ds_numel % self.world_size
        return (self.world_size - remainder) if remainder else 0

    def _partition_numel(self, param):
        return param.ds_tensor.ds_numel
        
    '''_all_gatherなどの中身の実装を行っていく'''
    @instrument_w_nvtx
    def _all_gather(self, param_list, async_op=False, hierarchy=None):
        handles = [] #非同期時のハンドル格納
        all_gather_list = [] #同期一括の待ち行列, async_opによってどっちに入れるか決まる
        for param in param_list:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE: #自分のGPUにフルサイズがない
                if async_op:
                    handle = self._allgather_param(param, async_op, hierarchy=hierarchy) #単一パラの非同期 all-gather を起動
                    param.ds_status = ZeroParamStatus.INFLIGHT
                    handles.append(handle)
                else:
                    all_gather_list.append(param) #今自分のGPU上にあるならそのままall_gatherするリストに格納
        
        if not async_op:
            if len(param_list) == 1: #1つだけ
                ret_value = self._allgather_params(all_gather_list, hierarchy=hierarchy)
            else:
                ret_value = self._allgather_params_coalesced(all_gather_list, hierarchy) #同期的に行うときにしか出てこないよ~
            for param in all_gather_list:
                param.ds_status = ZeroParamStatus.AVAILABLE
            return ret_value
        
        return handles
    
    #重みパラメータ分割を行うメインの部分
    def _partition(self, param_list, force=False, has_been_updated=False):
        for param in param_list:
            self._partition_param(param, has_been_updated=has_been_updated)
            param.ds_status = ZeroParamStatus.NOT_AVAILABLE #分割したのでNOT AVAILABLE
        
    #1パラメータを「フルサイズ->シャード」へ切り出し
    @instrument_w_nvtx
    def _partition_param(self, param, buffer=None, has_been_updated=False):
        global reuse_buffers
        if param.ds_status is ZeroParamStatus.AVAILABLE: #フルサイズがすでにあるなら分割
            print_rank_0(f"Partitioning param id {param.ds_id} reuse buffers {reuse_buffers}", force=False)
            #ds_tensor が既にあり値も未更新なら、フル側 (param.data) を解放するだけでよい
            if param.ds_tensor is not None and not has_been_updated:
                see_memory_usage(f'Before partitioning param {param.ds_id} {param.shape}', force=False)
                free_param(param) #フルサイズのparamを開放して終わり
                see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}', force=False)
                return
        
        tensor_size = self._aligned_size(param) #paramをaligment制約を守った形でのサイズ
        partition_size = tensor_size // self.world_size #自分の担当分のサイズ
        
        if param.ds_tensor is None: #ds_tensor(分割済み)がないなら新しく分割
            partitioned_tensor = torch.empty(partition_size, dtype=param.dtype, device=self.remote_device)
            if self.pin_memory:
                partitioned_tensor = partitioned_tensor.pin_memory() #partitioned_tensorはGPU or CPU上で常に持っているのでpin_memoryでアクセスを高速化しておく
        partitioned_tensor.requires_grad = False
        param.ds_tensor = partitioned_tensor #ds_tensorは分割分を示している
        param.ds_tensor.ds_numel = partition_size
        param.ds_tensor.status = PartitionedParamStatus.AVAILABLE #分割片を保持している
        
        start = partition_size * self.rank
        end = partition_size * (self.rank + 1)
        one_dim_param = param.contiguous().view(-1) #contiguous()はparamをC-order(行優先)の1D Flat化テンソルに置く
        
        #自ランクの担当開始位置/終了位置を決定し、フルパラ param を 1次元化。
        if start < param.ds_numel and end <= param.ds_numel: #完全に担当範囲が実データ内なら、そのまま partition_size ぶんコピー
            src_tensor = one_dim_param.narrow(0, start, partition_size)
            param.ds_tensor.copy_(src_tensor)
        else: #末尾で実データが足りない（アラインのパディングがある）場合は、残り要素だけをコピーし、残りは未使用（0埋め相当の扱い
            if start < param.ds_numel:
                elements_to_copy = param.ds_numel - start
                param.ds_tensor.narrow(0, 0, elements_to_copy).copy_(one_dim_param.narrow(0, start, elements_to_copy))
                
        see_memory_usage(f'Before partitioning param {param.ds_id} {param.shape}', force=False)
        free_param(param) #フルサイズparamの解放
        see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}', force=False)
        print_rank_0(f"ID {param.ds_id} partitioned type {param.dtype} dev {param.device} shape {param.shape}")
    
    def _param_status(self, param):
        if param.ds_tensor is not None:
            print_rank_0(f"Param id {param.ds_id}, param status: {param.ds_status}, param numel {param.ds_numel}, partitioned numel {param.ds_tensor.numel()}, data numel {param.data.numel()}")
        else:
            print_rank_0(f"Param id {param.ds_id}, param status: {param.ds_status}, param numel {param.ds_numel}, partitioned ds_tensor {param.ds_tensor}, data numel {param.data.numel()}")
    
    #単一パラメータ専用の all-gather 実装, all_gather_base が使えるならそれを使い、なければ dist.all_gather（リスト版）で実現。
    def _allgather_param(self, param, async_op=False, hierarchy=0):
        partition_size = param.ds_tensor.ds_numel #paramの持つ要素数で自分のshard分
        tensor_size = partition_size * self.world_size
        aligned_param_size = self._aligned_size(param)
        
        print_rank_0(f"{'--'* hierarchy}---- Before allocating allgather param {debug_param2name_id_shape_status(param)} partition size={partition_size}")
        see_memory_usage(f'Before allocate allgather param {debug_param2name_id_shape_status(param)} partition_size={partition_size} ', force=False)
        
        flat_tensor = torch.zeros(aligned_param_size, dtype=param.dtype, device=param.device).view(-1)
        
        see_memory_usage(f'After allocate allgather param {debug_param2name_id_shape_status(param)} {aligned_param_size} {partition_size} ', force=False)
        #torch.cuda.synchronize() #単一プロセスでの同期, GPU kernelが終わるまではCPU実行を止める (torch.distributed.barrier()は複数プロセスのバリア同期)
        print_rank_0(f"{'--'* hierarchy}----allgather param with {debug_param2name_id_shape_status(param)} partition size={partition_size}")
        #_all_gather_base は出力をリストでなく 1 本のフラットテンソルに直接書くため高速・低オーバーヘッド
        handle = dist.all_gather_base(flat_tensor, param.ds_tensor.cuda(), group=self.ds_process_group, async_op=async_op)
        
        replicated_tensor = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape) #フルサイズのパラメータをreplicated_tensorに入れる
        param.data = replicated_tensor.data
        return handle
    
    def _allgather_params_coalesced(self, param_list, hierarchy=0):
        if len(param_list) == 0:
            return
        # collect local tensors and partition sizes
        partition_sizes = []
        local_tensors = []
        for param in param_list:
            partition_sizes.append(param.ds_tensor.ds_numel)
            local_tensors.append(param.ds_tensor.cuda())
        
        allgather_params = []
        for psize in partition_sizes:
            tensor_size = psize * self.world_size #フルサイズテンソルのサイズ
            flat_tensor = torch.empty(tensor_size, dtype=param_list[0].dtype, device=self.local_device).view(-1)
            flat_tensor.requires_grad = False
            allgather_params.append(flat_tensor)
            
        launch_handles = []
        for param_idx, param in enumerate(param_list):
            input_tensor = local_tensors[param_idx].view(-1)
            h = dist.all_gather_base(allgather_params[param_idx], input_tensor, group=self.ds_process_group, async_op=True)
            launch_handles.append(h)
        launch_handles[-1].wait()
        
        for i, param in enumerate(param_list):
            gathered_tensor = allgather_params[i]
            param.data = gathered_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape).data
        
        torch.cuda.synchronize() #async_op=Falseで呼ばれる関数のためちゃんと同期しておく
        return None
    
    def _allgather_params(self, param_list, hierarchy=0): #param_listのものすべてを1D flat tensorにして一気にall_gather通信
        if len(param_list) == 0:
            return
        
        partition_size = sum([param.ds_tensor.ds_numel for param in param_list])
        tensor_size = partition_size * self.world_size
        flat_tensor = torch.empty(tensor_size, dtype=param_list[0].dtype, device=self.local_device)
        flat_tensor.requires_grad = False
        partitions = [] #partitions: [[rank0,tensor0のshard, rank0,tensor1のshard], [rank1,tensor0のshard, rank1,tensor1のshard], ...]
        for i in range(self.world_size):
            start = partition_size * i
            partitions.append(flat_tensor.narrow(0, start, partition_size))
            if i == self.rank:
                offset = 0
                for param in param_list:
                    param_numel = param.ds_tensor.ds_numel
                    partitions[i].narrow(0, offset, param_numel).copy_(param.ds_tensor.data)
                    offset += param_numel
        dist.all_gather(partitions, partitions[self.rank], group=self.ds_process_group, async_op=False)
            
        param_offset = 0
        for param in param_list:
            param_partition_size = param.ds_tensor.ds_numel #自分の担当分のサイズ
            param_size  = param.ds_numel
            replicated_tensor = torch.empty(param.ds_shape, dtype=param.dtype, device=self.local_device)
                
            for i in range(self.world_size):
                start = i * partition_size
                param_start = i * param_partition_size
                if param_start < param_size: #paddingで埋められてるやつ
                    numel_to_copy = min(param_size - param_start, param_partition_size)
                    part_to_copy = partitions[i].narrow(0, param_offset, numel_to_copy)
                    replicated_tensor.view(-1).narrow(0, param_start, numel_to_copy).copy_(part_to_copy)
                
            param_offset += param.ds_tensor.ds_numel
            param.data = replicated_tensor.data
        return None
    
    def _reduce_scatter_gradients(self, param_list):
        handles_and_reduced_partitions = [] #非同期通信のハンドルと、出力バッファ（自ランクの縮約結果）を入れるリスト
        for param in param_list:
            assert param.grad.numel(
            ) == param.ds_numel, f"{param.grad.numel()} != {param.ds_numel} Cannot reduce scatter gradients whose size is not same as the params"

            handles_and_reduced_partitions.append(self._reduce_scatter_gradient(param))
        #param, (reduce_scatter, reduce済みの担当勾配)
        for param, (handle, reduced_partition) in zip(param_list, handles_and_reduced_partitions):
            if handle is not None:
                handle.wait() #GPUのreduceが終わるまで待つ
            partition_size = param.ds_tensor.ds_numel
            start = self.rank * partition_size
            end = start + partition_size
            #末尾以外の rank は出力が param.grad のビューに直書きされるためコピー不要。
            #末尾 rank (start < ds_numel < end) はパディング用一時バッファに出力されるため有効要素だけ書き戻す
            if start < param.ds_numel and end > param.ds_numel:
                elements = param.ds_numel - start
                param.grad.view(-1).narrow(0, start, elements).copy_(reduced_partition.narrow(0,0,elements)) #reduce済み勾配をparamにコピーする(自分の担当分のみ)

    def _reduce_scatter_gradient(self, param):
        partition_size = param.ds_tensor.ds_numel
        total_size = partition_size * self.world_size
        input_list = []
        for i in range(self.world_size):
            start = i * partition_size
            end = start + partition_size #自分の担当分のstart, end
            
            if start < param.ds_numel and end <= param.ds_numel:
                input = param.grad.view(-1).narrow(0, start, partition_size) #すべての担当分がinputに入れれる
            else: #paddinを行うので単純には入れれない
                input = torch.zeros(partition_size, dtype=param.dtype, device=param.device) #padding
                if start < param.ds_numel:
                    elements = param.ds_numel - start
                    input.narrow(0, 0, elements).copy_(param.grad.view(-1).narrow(0, start, elements))
            input_list.append(input)
        
        rank = dist.get_rank(group=self.ds_process_group)
        handle = dist.reduce_scatter(input_list[rank], input_list, group=self.ds_process_group, async_op=True)
        return handle, input_list[rank] #(そのパラメータのreduce_scatter, 自分の担当分のreduce済み勾配)
    
    def _partition_gradients(self, param_list, partition_buffers=None, accumulate=False):
        if partition_buffers is None:
            partition_buffers = [None] * len(param_list) #パラごとの**出力バッファ（分割勾配の置き場）**が来ていなければ、全件 None にする（＝各パラ内で自前確保）
        for param, partition_buffer in zip(param_list, partition_buffers):
            self._partition_gradient(param, partition_buffer=partition_buffer, accumulate=accumulate) #各paramについて1つずつgradient分割を行う
    
    #param.gradのうち自分の担当分のgradientのみを切り出して保存しておく
    def _partition_gradient(self, param, partition_buffer=None, accumulate=False):
        print_rank_0(f"Partitioning param {param.ds_id} gradient of size {param.grad.numel()} type {param.grad.dtype} part_size {param.ds_tensor.ds_numel}")
        see_memory_usage("Before partitioning gradients", force=False)
        partition_size = param.ds_tensor.ds_numel
        
        if partition_buffer is None:
            partition_buffer = torch.zeros(partition_size, dtype=param.dtype, device=param.device) #勾配を保存しておくためのバッファ領域を確保
        
        rank = dist.get_rank(group=self.ds_process_group)
        start = partition_size * rank
        end = start + partition_size ##担当する勾配のstart, end
        dest_tensor_full_buffer = partition_buffer.view(-1).narrow(0, 0, partition_size) #出力先を 1D 化し、先頭 partition_size 分を「このランクの受け取り領域」としてビュー化

        #実データが存在する要素数 elements を決定 (末尾パディングがあると elements < partition_size)
        if start < param.ds_numel:
            elements = min(param.ds_numel - start, partition_size)
            dest_tensor = dest_tensor_full_buffer.narrow(0, 0, elements)
            src_tensor = param.grad.view(-1).narrow(0, start, elements) #param.grad(この段階では全体を持っている)から自分の担当分の開始位置からelements分のみを切り出す
            
            if not accumulate:
                dest_tensor.copy_(src_tensor)
            elif src_tensor.device == dest_tensor.device:
                dest_tensor.add_(src_tensor) #勾配蓄積するなら足し合わせ、そうでないならそのまま上書きコピー
        
        #param.grad の実体ストレージを、分割済み（このランクぶん）に差し替える。
        # → 以降、このランクの param.grad は partition_size （または elements）しか持たない。
        param.grad.data = dest_tensor_full_buffer.data
        see_memory_usage("After partitioning gradients", force=False)