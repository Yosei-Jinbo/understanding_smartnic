import sys
import os
import argparse
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import logger
import torch
import torch.distributed as dist
import gc
import collections
from typing import Deque, Dict, Tuple
from torch.cuda import Event, Stream
from stage3_utils import * #parameterの持ち方が違うため独自のmemory_usage関数を呼ぶ
from partition_parameters import *
from partition_parameters import _DmaCompletionPoller
from parameter_offload import ZeroOffload
from torch._utils import _flatten_dense_tensors as flatten
from torch._utils import _unflatten_dense_tensors as unflatten
from torch.distributed import ProcessGroup
import math
import itertools
from typing import Deque, Dict, Tuple, List
from torch import Tensor
from torch.nn import Parameter
from deepspeed.ops.adam import DeepSpeedCPUAdam
from concurrent.futures import ThreadPoolExecutor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# ---- 転送内訳計測用 NVTX (XFER_NVTX=1 で有効) ----
# nsys の memcpy を NVTX 区間へ射影して「1step あたりの転送時間」を内訳付きで取るための計装。
# scripts/extract_transfer_per_step.py が 'xfer:*' 区間を参照する。
# per-submodule の USE_NVTX_RANGES とは独立にゲートする (単独で on にできるようにするため)。
_XFER_NVTX = os.environ.get("XFER_NVTX", "0") == "1"


def _xfer_push(label: str) -> None:
    if _XFER_NVTX:
        torch.cuda.nvtx.range_push(label)


def _xfer_pop() -> None:
    if _XFER_NVTX:
        torch.cuda.nvtx.range_pop()



# ------------------------------------------------------------
# CPU<->GPU copy time accounting (CUDA event based)
#   - We accumulate per-rank GPU time (ms) for:
#       * aggregated gradient GPU->CPU (D2H) offload (measurement)
# ------------------------------------------------------------
GRAD_OFFLOAD_D2H_TIME_MS: float = 0.0
GRAD_OFFLOAD_D2H_CALLS: int = 0

def _accumulate_grad_offload_d2h_time_ms(delta_ms: float) -> None:
    global GRAD_OFFLOAD_D2H_TIME_MS, GRAD_OFFLOAD_D2H_CALLS
    GRAD_OFFLOAD_D2H_TIME_MS += float(delta_ms)
    GRAD_OFFLOAD_D2H_CALLS += 1

def get_grad_offload_d2h_time_ms() -> float:
    return float(GRAD_OFFLOAD_D2H_TIME_MS)

def get_grad_offload_d2h_calls() -> int:
    return int(GRAD_OFFLOAD_D2H_CALLS)

# ------------------------------------------------------------
# Communication time accounting (CUDA event based)
#   - We accumulate per-rank GPU time (ms) for:
#       * gradient reduce_scatter (avg + scatter)
# ------------------------------------------------------------
REDUCE_SCATTER_TIME_MS: float = 0.0
REDUCE_SCATTER_CALLS: int = 0

def _accumulate_reduce_scatter_time_ms(delta_ms: float) -> None:
    global REDUCE_SCATTER_TIME_MS, REDUCE_SCATTER_CALLS
    REDUCE_SCATTER_TIME_MS += float(delta_ms)
    REDUCE_SCATTER_CALLS += 1

def get_reduce_scatter_time_ms() -> float:
    return float(REDUCE_SCATTER_TIME_MS)

def get_reduce_scatter_calls() -> int:
    return int(REDUCE_SCATTER_CALLS)

def print_rank_0(message, debug=False, force=False):
    rank = dist.get_rank()
    if rank == 0 and (debug or force):
        #print(message)
        pass
    
def _flatten(tensors):
    # dtype / device を揃えておくこと（ここでは fp32/CPU を想定）
    ts = [t.detach().contiguous() for t in tensors]
    return _flatten_dense_tensors(ts)

def _unflatten(flat, like_tensors):
    # like_tensors の shape に合わせて view を作る
    return _unflatten_dense_tensors(flat, [t.detach().contiguous() for t in like_tensors])


@instrument_w_nvtx
def _torch_reduce_scatter_fn(input_tensor: torch.Tensor,
                             output_tensor: torch.Tensor,
                             group=None,
                             async_op: bool = False,
                             prof: bool = False):
    """
    PyTorch標準の `torch.distributed.reduce_scatter` を用いた実装。
    既存の引数シグネチャ（DeepSpeed互換）を維持しつつ、中で標準APIを呼び出します。

    引数:
      - input_tensor: list/tuple のテンソル列 もしくは 1本のフラットテンソルを受け付ける
      - output_tensor: 本rankが受け取る1チャンク分の出力先テンソル
      - group: 通信に用いるプロセスグループ（Noneなら既定）
      - async_op: Trueで非同期実行（Workハンドルを返す）
      - prof: 互換性維持用のダミー引数（未使用）
    """
    # --- 入力の正規化（list/tupleならそのまま、単一テンソルならworld_size個に等分チャンク） ---
    if isinstance(input_tensor, (list, tuple)):
        input_list = list(input_tensor)
        if len(input_list) == 0:
            raise ValueError("input_tensor のリスト/タプルは空であってはなりません。")
    else:
        world_size = dist.get_world_size(group=group)
        if input_tensor.numel() % world_size != 0:
            raise ValueError(
                f"input_tensor.numel()={input_tensor.numel()} を "
                f"world_size={world_size} で等分できません。"
            )
        # 先頭次元(dim=0)で等分する前提（ZeROのフラット勾配/重みと整合）
        input_list = list(torch.chunk(input_tensor, world_size, dim=0))

    # --- 出力テンソル形状の簡易チェック（要素数がチャンクと一致している必要がある） ---
    if output_tensor.numel() != input_list[0].numel():
        raise ValueError(
            f"output_tensor の要素数 {output_tensor.numel()} が、"
            f"チャンク1個分の要素数 {input_list[0].numel()} と一致しません。"
        )

    # --- PyTorch標準 reduce_scatter をNVTX計測付きで呼び出し（演算は既定のSUM） ---
    return instrument_w_nvtx(dist.reduce_scatter)(
        output_tensor,       # 出力（本rankの受け取り先）
        input_list,          # 入力（全rank分のチャンク列）
        group=group,
        async_op=async_op
    )

#複数のテンソルをまとめて reduce-scatter（勾配の平均＋分割）するための高効率ユーティリティ関数
@instrument_w_nvtx
@torch.no_grad()
def reduce_scatter_coalesced(
    tensors: List[Tensor],
    group: ProcessGroup = None,
) -> List[Tensor]:
    this_rank = dist.get_rank(group)
    world_sz = dist.get_world_size(group)
    
    partition_lst_for_each_tensor = [None] * len(tensors)
    '''
    各テンソルを 1D に flatten。
    各テンソルを world_sz 個のチャンクに切り出す。割り切れないときは ceil で切り上げたサイズを使う（＝後で パディングが必要）。
    結果は partition_lst_for_each_tensor[tensor_idx][rank] で「tensor_idx のテンソルの rank 用チャンク」にアクセスできる。
    '''
    for tensor_idx, tensor in enumerate(tensors):
        flattened_tensor = tensor.view(-1) #flat化
        chunk_sz = math.ceil(tensor.numel() / world_sz) #1ランクあたりの目標チャンクサイズを計算(割り切れないときは切り上げ) → 不足分は後でパディング。
        #flat化したtensorをそれぞれのrankが担当する分に分けてリストとして保存しておく
        partition_lst_for_each_tensor[tensor_idx] = [flattened_tensor[rank*chunk_sz : (rank+1)*chunk_sz] for rank in range(0, world_sz)]
    
    padded_partition_sz_for_each_tensor = tuple(math.ceil(t.numel() / world_sz) for t in tensors) #padding
    if len(tensors) == 1 and tensors[0].numel() % world_sz == 0:
        tensor_partition_flat_buffer = tensors[0].view(-1)
    else: #paddingいる
        tensor_partitions_lst_with_padding = []
        for rank in range(world_sz):
            for tensor_idx in range(len(tensors)):
                #(rank0のチャンクの連続),(rank1のチャンクの連続) ...というようにrankがreduceする部分ごとにまとめる
                tensor_chunk = partition_lst_for_each_tensor[tensor_idx][rank]
                tensor_partitions_lst_with_padding.append(tensor_chunk)
                
                padding_sz = padded_partition_sz_for_each_tensor[tensor_idx] - tensor_chunk.numel()
                if padding_sz > 0:
                    tensor_partitions_lst_with_padding.append(torch.empty(padding_sz, dtype=tensor_chunk.dtype, device=tensor_chunk.device)) #paddingしている
        #インターリーブ＋パディングした全パーツを torch.cat で一つの大きな 1D バッファに結合。
        tensor_partition_flat_buffer = instrument_w_nvtx(torch.cat)(tensor_partitions_lst_with_padding)
    
    tensor_partition_flat_buffer.div_(world_sz)  # pre-divide
    #大きな入力バッファを rank 数で等分（world_sz 個の連続ブロックに分割, 各チャンクの中身はrankiが担当してreduce_scatterするもの全体）。
    #これは 出力バッファ群としても扱う（reduce-scatter の各ランク受け取り先をここから選ぶ）。
    tensor_partition_buffer_for_each_rank: List[Tensor] = torch.chunk(tensor_partition_flat_buffer, world_sz)
    
    from common import debug_params as dbg
    dbg.log_reduce_scatter(tensor_partition_flat_buffer, None, sub_group_id="flat_input")

    t_rs_start = time.perf_counter()
    # DMA ポーラー準備 (RS 出力テンソルのデバイスで)
    _rs_device = tensor_partition_buffer_for_each_rank[this_rank].device
    _rs_poller = _DmaCompletionPoller(_rs_device)

    # Phase 20: bracket the NCCL RS enqueue + wait-event for symmetric stall measurement
    try:
        from common import stall_event_tracker as _set  # type: ignore
    except Exception:
        _set = None
    if _set is not None and _set.is_enabled():
        try:
            from partition_parameters import _AG_PHASE as _PHASE  # type: ignore
        except Exception:
            _PHASE = "unknown"
        _tracker = _set.get_global()
        _stream = torch.cuda.current_stream()
        _rs_nbytes = int(tensor_partition_flat_buffer.numel() *
                         tensor_partition_flat_buffer.element_size())
        _h = _tracker.begin(_stream, _PHASE, op="rs", ds_id=-1, payload_bytes=_rs_nbytes)
        try:
            _torch_reduce_scatter_fn(tensor_partition_flat_buffer,
                                     tensor_partition_buffer_for_each_rank[this_rank],
                                     group=group)
        finally:
            _tracker.end(_h)
    else:
        _torch_reduce_scatter_fn(tensor_partition_flat_buffer, tensor_partition_buffer_for_each_rank[this_rank], group=group)
    # RS 完了後にフラグ書き込みを投入してポーリング
    _rs_poller.record()
    torch.cuda.nvtx.range_push("rs_wait")
    try:
        _rs_poller.join()
    finally:
        torch.cuda.nvtx.range_pop()
    t_rs_end = _rs_poller.t_complete or time.perf_counter()
    _accumulate_reduce_scatter_time_ms((t_rs_end - t_rs_start) * 1000.0)

    output_lst: List[Tensor] = [None] * len(tensors)
    offset = 0
    for tensor_idx in range(len(tensors)):
        output_lst[tensor_idx] = tensor_partition_buffer_for_each_rank[this_rank].narrow(
            0, offset, partition_lst_for_each_tensor[tensor_idx][this_rank].numel())
        offset += padded_partition_sz_for_each_tensor[tensor_idx]

    dbg.log_reduce_scatter(None, output_lst, sub_group_id="flat_output")

    return output_lst

    
'''
!!!!!! ここからCPUにオフロードするように準備
'''

def _adam_worker(pipe, fp32_weights, grad_bufs, optimizer_defaults,
                 sub_group_to_group_id, num_subgroups):
    """CPU Adam ワーカープロセス。Pipe 経由でコマンドを受信し shared memory 上の FP32 重みを更新。"""
    import os
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    optimizer = DeepSpeedCPUAdam(fp32_weights, **optimizer_defaults)
    for pg in optimizer.param_groups:
        pg['params'] = []

    pipe.send("ready")

    while True:
        cmd = pipe.recv()
        if cmd[0] == "shutdown":
            break
        elif cmd[0] == "set_affinity":
            try:
                os.sched_setaffinity(0, set(cmd[1]))
            except Exception:
                pass
        elif cmd[0] == "step":
            grad_idx = cmd[1]
            for i in range(num_subgroups):
                w = fp32_weights[i]
                w.grad = grad_bufs[grad_idx][i]
                gid = sub_group_to_group_id[i]
                optimizer.param_groups[gid]['params'] = [w]
                optimizer.step()
                optimizer.param_groups[gid]['params'] = []
                w.grad = None
            pipe.send("done")


class ZeroOptimizer3(object):
    def __init__(self,
                 module,
                 init_optimizer,
                 timers,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 #prefetch_bucket_size=50000000,
                 prefetch_bucket_size=0,
                 #max_reuse_distance=1000000000,
                 max_reuse_distance=0,
                 #max_live_parameters=1000000000,
                 max_live_parameters=0,
                 dp_process_group=None,
                 reduce_scatter=True,
                 overlap_comm=True,
                 sub_group_size=1000000000000,
                 gradient_accumulation_steps=1,
                 communication_data_type=torch.float32):
        
        see_memory_usage("Stage 3 initialize beginning", force=True)
        print_rank_0(f"initialized {__class__.__name__} with args: {locals()}",
                     force=False)
        if dist.get_rank() == 0:
            logger.info(f"Reduce bucket size {reduce_bucket_size}")
            logger.info(f"Prefetch bucket size {prefetch_bucket_size}")
            
        self.optimizer = init_optimizer
        self.flatten = flatten
        self.unflatten = unflatten
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype
        
        '''ZeRO Offloadのための追加分'''
        self.offload_optimizer = True
        self.offload_optimizer_pin_memory = True
        self.offload_param = True
        self.offload_param_pin_memory = True 
        
        self.parameter_offload = ZeroOffload(
            module=module,
            timers=timers,
            overlap_comm=overlap_comm,
            prefetch_bucket_size=prefetch_bucket_size,
            max_reuse_distance=max_reuse_distance,
            max_live_parameters=max_live_parameters)

        self.module = module
        
        self.deepspeed_adam_offload = (self.offload_optimizer
                                       and type(init_optimizer) == DeepSpeedCPUAdam) 
        
        self.device = torch.cuda.current_device() if not self.offload_optimizer else "cpu" #self.deviceがZeRO Offloadの時はCPUになる
        self.__reduce_and_partition_stream = Stream() if overlap_comm else torch.cuda.current_stream()
        self.local_device = torch.device("cuda") #現在のGPU
        
        self.timers = timers
        self.reduce_scatter = reduce_scatter
        self.dp_process_group = dp_process_group
        self.partition_count = dist.get_world_size(group=self.dp_process_group)
        
        #mpu周りは未実装
        self.model_parallel_group = None
        self.model_parallel_rank = 0
        
        self.communication_data_type = communication_data_type
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_step_id = 0
        self.reduce_bucket_size = int(reduce_bucket_size)
        
        self.fp16_groups = [] #各param_groupごとの16bit パラメータのリスト
        self.fp16_partitioned_groups = [] #各param_groupごとの16bit パラメータのリストをパーティションごとのリストにしたもの
        self.fp16_partitioned_groups_flat = [] #各param_groupごとの16bit パラメータのリストをパーティションごとのリストにしたもの(フラット化)
        self.fp16_partitioned_groups_flat_numel = [] #(要素数ごとに記録)
        
        self.param_groups_fp16_flat_cpu_memory = [] #CPU 側のピン留め（pinned）メモリ上にデフラグ済みで置いた FP16 フラットバッファ。転送やオフロード最適化用。
        
        '''Delayed Parameter Update 用変数'''
        self.fp32_partitioned_groups_flat = []     # 1本の FP32 重みバッファ (shared memory)
        self.fp32_grad_bufs = [[], []]             # 勾配のみ双バッファ (pinned + shared memory)
        self.grad_buf_switch = 0                   # backward が書く勾配バッファの選択
        self._adam_pipe_parent = None
        self._adam_process = None
        self._adam_thread = None   # DISABLE_ADAM_FORK=1 時の threading 経路用
        self._adam_step_pending = False
        
        #この変数いらないかも
        self.next_swappable_fp32_partitioned_groups = [] #FP32 パーティションの スワップ（入れ替え/オフロード）候補キュー。I/O と計算を重ねるための先読み・交換管理

        self.partition_size = [] #各パーティションの大きさ
        self.all_reduce_print = False
        self.prefetch_elements = int(prefetch_bucket_size)
        self.contiguous_gradients = contiguous_gradients
        self.groups_padding = [] #各sub_groupsについてaligment制約のためにpaddingした分
        
        self.sub_group_size = sub_group_size
        
        self.sub_group_to_group_id = {} #サブグループID → 親の param_group ID の 対応表
        see_memory_usage("Before creating fp16 partitions", force=False)
        self._create_fp16_partitions_with_defragmentation()
        num_fp16_subgroups = len(self.fp16_partitioned_groups_flat)
        see_memory_usage(f"After creating fp16 partitions: {num_fp16_subgroups}",
                         force=False)
        
        self.__params_in_ipg_bucket: List[Parameter] = [] #現在の IPG バケットに入っているパラメータの集合(勾配集約単位), parameter objectを格納
        self.is_gradient_accumulation_boundary: bool = True #勾配蓄積の境界かどうか。境界で reduce を走らせるか、次ステップへ持ち越すかの判断に使う
        
        self.__param_reduce_events: Deque[Event] = collections.deque() #NCCL などの 非同期 reduce イベントをためるキュー
        self.__max_param_reduce_events: int = 2
        self.param_dict = {}
        
        self.is_param_in_current_partition = {} # map between param_id and bool to specify if a param is in this partition
        
        self.extra_large_param_to_reduce = None
        self.grads_in_ipg_bucket = [] #IPG bucket内の勾配テンソル・対応パラメータのリスト
        self.params_in_ipg_bucket = []
        
        self.params_already_reduced = [] #reduce済みのパラメータかどうか, param_idをindexにしたリストで管理するらしい
        self.is_gradient_accumulation_boundary = True
        self._release_ipg_buffers() #ライフサイクル管理(いったん解放、次に必要になったらまた再確保)
        self.previous_reduced_grads = None
        
        self.param_id = {} #parameterのid -> 0スタートのunique番号
        
        count = 0
        for i, params_group in enumerate(self.fp16_groups): #sub_group単位の分割?
            for param in params_group: #sub_group内で各パラメータに対してforループ
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                count = count + 1
        
        largest_partitioned_param_numel = max([
            max([
                max(tensor.numel(), tensor.ds_numel) for tensor in fp16_partitioned_group
            ]) for fp16_partitioned_group in self.fp16_partitioned_groups
        ])
        print_rank_0(
            f'Largest partitioned param numel = {largest_partitioned_param_numel}',
            force=False)
        
        self._setup_for_real_optimizer()
        
        '''
        self.grad_position[param_id] = [
            int(group_id),
            int(current_offset),
            int(num_elements)
        ]
        何番目のパラメータが、どのsub_groupに属していて、sub_group上ではcurrent_offsetから始まり、num_elemnts個の要素を持っている
        '''
        self.grad_position = {}
        self.set_grad_positions() #上記の変数について全パラメータ分を一気に登録
        
        self.is_partition_reduced = {} #各パーティションがkの学習ステップ中にすでに通信(reduce / reduce_scatter)済みか
        self.is_grad_computed = {} #各Parameterの勾配がもう計算済みか（backward でフックが発火したか
        self.averaged_gradients = {} #このrankが最終的に必要とする平均済み勾配
        
        self.create_reduce_and_remove_grad_hooks()
        
        self.debug_fp16_grads = [{} for _ in self.fp16_groups]

        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f"After initializing ZeRO optimizer", force=True)
    
    def log_timers(self, timer_names):
        if self.timers is None:
            return

        self.timers.log(names=list(timer_names))

    def start_timers(self, timer_names):
        if self.timers is None:
            return

        for name in timer_names:
            self.timers(name).start()

    def stop_timers(self, timer_names):
        if self.timers is None:
            return

        for name in timer_names:
            self.timers(name).stop()
        
    def destroy(self):
        self.parameter_offload.destroy()
        
    def _get_param_coordinator(self, training):
        return self.parameter_offload.get_param_coordinator(training)
    
        
    #多数の小さな GPU テンソル(tensor)を一度 CPU に集めて 1 本の大きな連続バッファにまとめ直し、再び GPU に戻すことで、メモリ断片化を解消する
    #get_only_unique_item(list): listの要素がすべて同じときのみその要素を返す, 2つ以上あったらraise Error
    @staticmethod
    def defragment(tensors: List[Tensor]) -> Tensor:
        cpu_buffer = torch.empty(sum(p.numel() for p in tensors), dtype=get_only_unique_item(t.dtype for t in tensors), device="cpu") #CPU上のバッファに連続領域を取る
        tensor_infos: List[Tuple[Tensor, int, int]] = []
        orig_device = get_only_unique_item(t.device for t in tensors) #tensorがおかれているデバイス情報
        
        offset = 0
        for tensor in tensors:
            tensor_numel = tensor.numel()
            cpu_buffer.narrow(0, offset, tensor_numel).copy_(tensor) #cpu_bufferのスライス(ビュー)にtensorの内容をコピー(tensorとcpu_bufferは共有されてない)
            tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device) #tensor が参照する内部のストレージを、サイズ 0 の空 Tensor に差し替えてしまう
            tensor_infos.append((tensor, offset, tensor_numel))
            offset += tensor_numel
        
        gc.collect() #ガベージコレクションにより今までtensorがさしていた数値ストレージを開放する
        torch.cuda.empty_cache()
        
        device_buffer = cpu_buffer.to(orig_device) #CPU上に詰めた連続領域をGPUに書き戻し
        
        for tensor, offset, tensor_numel in tensor_infos:
            tensor.data = device_buffer.narrow(0, offset, tensor_numel) #GPU上に移した連続領域を参照できるようにする
        return device_buffer #GPU上連続領域
            
    #1つの param_group を「要素数（partition_numel）の合計が一定しきい値（sub_group_size）に達するまで」順に束ねて、小分けのサブグループ配列に分割す
    #ZeRO-3 の後続処理（フラット化・通信・オフロード・プリフェッチ）の処理単位を制御してメモリ/帯域を安定化
    def _create_fp16_sub_groups(self, params_group):
        params_group_numel = sum([param.partition_numel() for param in params_group])
        sub_group_size = self.sub_group_size
        if sub_group_size is None or sub_group_size >= params_group_numel:
            return [params_group] #1つのサブグループにしかならない
        
        sub_groups = []
        sub_group = []
        local_sub_group_size = 0
        for param in params_group:
            sub_group.append(param)
            local_sub_group_size += param.partition_numel()
            #閾値に達する or そのパラメータが最後の時はsub_groupとしてsub_groupsに加える
            if local_sub_group_size >= sub_group_size or id(param) == id(params_group[-1]):
                sub_groups.append(sub_group)
                sub_group = []
                local_sub_group_size = 0
                
        return sub_groups
    
    #(可能なら)各パラメータを1本のフラットCPUバッファへ順番に詰めなおし、元のparam.ds_tensorがそのフラット領域をさすように付け替える関数
    #self._move_to_flat_buffer(sub_group, fp16_partitioned_group_flat, avoid_copy=not self.offload_param)
    def _move_to_flat_buffer(self, param_list, flat_buffer, avoid_copy=False):
        if flat_buffer is None:
            return
        start = 0
        for param in param_list: #各paramごとに詰める要素、領域を決定している
            src = param.ds_tensor
            dest = flat_buffer.narrow(0, start, src.ds_numel)
            start = start + src.ds_numel
            if not avoid_copy:
                dest.data.copy_(src.data)
            src.data = dest.data
        
    def _create_param_groups_fp16_flat_cpu_memory(self):
        aggregate_param_count = 0
        for j, param_group in enumerate(self.optimizer.param_groups):
            params_in_group = sum([p.partition_numel() for p in param_group['params']]) #1列のバッファにするために要素数だけのサイズのバッファを取る
            flat_buffer_size = params_in_group
            #CPUに入るまでのデータはぎりぎりまでflat_buffer_sizeにつめる
            #そうでないデータはまだ処理しない(この関数ではCPUに連続バッファ領域を取るためだけの関数)
            aggregate_param_count += params_in_group
            if flat_buffer_size > 0:
                print_rank_0(f"group {j} flat buffer size {flat_buffer_size}", force=False)
                self.param_groups_fp16_flat_cpu_memory.append(
                    torch.empty(int(flat_buffer_size),
                                dtype=self.dtype,
                                pin_memory=True, device=self.device))
        
    def _create_fp16_partitions_with_defragmentation(self):
        dist.barrier() #CPU同期
        #param_group: self.optimizer.param_groupsを小さいグループに分割したリスト
        param_groups: List[List[Parameter]] = tuple(
            self._create_fp16_sub_groups(param_group["params"])
            for param_group in self.optimizer.param_groups)
        
        for param_group_idx, param_group in enumerate(param_groups):
            for sub_group in param_group:
                sub_group_idx = len(self.fp16_groups)
                self.fp16_groups.append(sub_group) #self.fp16_gropusにはsub_groupごとに分けたパラメータtensorを1つのリストにしている
                self.fp16_partitioned_groups.append([param.ds_tensor for param in sub_group])  #分割後のビュー (ds_tensor)を入れたfp16_partitioned_groups
                self.sub_group_to_group_id[sub_group_idx] = param_group_idx #sub group -> group mapping
                self.fp16_partitioned_groups_flat_numel.append(sum(p.partition_numel() for p in sub_group))
                rank_requires_padding = dist.get_rank(self.dp_process_group) == dist.get_world_size(self.dp_process_group) - 1
                self.groups_padding.append([p.padding_size() if rank_requires_padding else 0 for p in sub_group]) #最後のランクならpaddingを詰める
        
        self._create_param_groups_fp16_flat_cpu_memory() #CPUにオフロードするための連続領域の取得: self.param_groups_fp16_flat_cpu_memoryにその領域ができる
        for param_group_idx, param_group in enumerate(param_groups):
            flat_offset = 0
            for i, sub_group in enumerate(param_group):
                total_elements = sum(p.partition_numel() for p in sub_group) #sub_groupがどれだけの要素を持っているか
                fp16_partitioned_group_flat = self.param_groups_fp16_flat_cpu_memory[param_group_idx].narrow(0, flat_offset, total_elements) #すべてのsub_groupをCPU上にぶち込める
                self.fp16_partitioned_groups_flat.append(fp16_partitioned_group_flat) #CPU記憶領域のリスト, NVMeは積まれない
                flat_offset += total_elements
                self._move_to_flat_buffer(sub_group,
                                              fp16_partitioned_group_flat, avoid_copy=not self.offload_param)
        
        
    def _release_ipg_buffers(self):
        if self.contiguous_gradients:
            self.ipg_buffer = None
            
    
    def _create_fp32_partitions(self):
        '''FP32マスターのフラットバッファを1本だけ作成。shared memory で child process と共有。'''
        for i, tensor in enumerate(self.fp16_partitioned_groups_flat):
            src = self.fp16_partitioned_groups_flat[i].to(self.device).clone().float().detach()
            src.share_memory_()
            self.fp32_partitioned_groups_flat.append(src)
            self.fp32_partitioned_groups_flat[i].requires_grad = True
        for param_group in self.optimizer.param_groups:
            param_group['params'] = []

    def _optimizer_step(self, sub_group_id):
        param_group_id = self.sub_group_to_group_id[sub_group_id]
        fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
        self.optimizer.param_groups[param_group_id]['params'] = [fp32_param]
        self.optimizer.step()
        self.optimizer.param_groups[param_group_id]['params'] = []
    
    @staticmethod
    def _pin_shared_memory(tensor):
        """shared memory テンソルを cudaHostRegister で GPU pinned 登録する。"""
        cudart = torch.cuda.cudart()
        err = cudart.cudaHostRegister(tensor.data_ptr(), tensor.nelement() * tensor.element_size(), 1)
        if err != 0:
            print_rank_0(f"[WARN] cudaHostRegister failed (err={err})")

    def initialize_optimizer_states(self):
        num_subgroups = len(self.fp16_groups)
        gradient_dtype = self.fp32_partitioned_groups_flat[0].dtype

        for i, group in enumerate(self.fp16_groups):
            num_elements = int(self.fp16_partitioned_groups_flat_numel[i])
            see_memory_usage(f'[Begin] Initialize optimizer states {i} / {num_subgroups} subgroups, num_elems: {num_elements}', force=False)

            # 勾配の双バッファ: shared memory + cudaHostRegister で pinned 化
            buf0 = torch.zeros(num_elements, dtype=gradient_dtype, device=self.device)
            buf0.share_memory_()
            self._pin_shared_memory(buf0)
            self.fp32_grad_bufs[0].append(buf0)

            buf1 = torch.zeros(num_elements, dtype=gradient_dtype, device=self.device)
            buf1.share_memory_()
            self._pin_shared_memory(buf1)
            self.fp32_grad_bufs[1].append(buf1)

            self.fp32_partitioned_groups_flat[i].grad = buf0
            self._optimizer_step(i)

            see_memory_usage(f'[End] Initialize optimizer states {i} / {num_subgroups} subgroups, num_elems: {num_elements}', force=False)

        return

    '''
    実際にOptimizerを構築する前段階の「土台作り」をまとめて行う関数
    言い換えるとfp32パラメータのパーティション、オプティマイザ状態の初期化、勾配バッファの割り当て
    といった「ZeRO冗長化」+ オフロード前提の学習環境を作る
    '''
    def _setup_for_real_optimizer(self):
        see_memory_usage("Before creating fp32 partitions", force=False)
        self._create_fp32_partitions() #fp32マスターコピーを使うためのfp32パラメータパーティション作成
        see_memory_usage("After creating fp32 partitions", force=False)
        dist.barrier()
        
        see_memory_usage("Before initializing optimizer states", force=False)
        self.initialize_optimizer_states() #実際のoptimizer stateを初期化(自分の担当分のみの初期化) 登録されているのが自分の分のため
        see_memory_usage("After initializing optimizer states", force=False)
        dist.barrier()
        
        if dist.get_rank() == 0:
            logger.info(f"optimizer state initialized")
        
        if self.contiguous_gradients:
            self.__ipg_bucket_flat_buffer: Tensor = torch.empty(self.reduce_bucket_size, dtype=self.dtype, device=torch.cuda.current_device())
        
        grad_partitions_flat_buffer = None
        self.__param_id_to_grad_partition: Dict[int, Tensor] = {} #param_id => partition_flat_bufferのどの領域か?
        all_params = list(itertools.chain.from_iterable(self.fp16_groups)) #1列のリストにする
        grad_partitions_flat_buffer: Tensor = torch.zeros(sum(p.partition_numel() for p in all_params), dtype=self.dtype, device=self.device, pin_memory=True)
                
        offset = 0
        #param.ds_id: 各パラメータに割り振られた一意なid
        for param in all_params:
            self.__param_id_to_grad_partition[param.ds_id] = grad_partitions_flat_buffer.narrow(0, offset, param.partition_numel())
            offset += param.partition_numel()
        
    def set_grad_positions(self):
        for i, group in enumerate(self.fp16_groups):
            current_offset = 0
            for param in group:
                param_id = self.get_param_id(param)
                num_elements = param.partition_numel()

                self.grad_position[param_id] = [
                    int(i),
                    int(current_offset),
                    int(num_elements)
                ]
                #print(f"param id {param_id} i:{i}, ds_tensor {num_elements} numel {param.numel()}")
                current_offset += num_elements
        see_memory_usage(f"After Set Grad positions", force=False)
    
    
    def get_param_id(self, param):
        unique_id = id(param)
        return self.param_id[unique_id]
    
    @property
    def elements_in_ipg_bucket(self):
        return sum(p.ds_numel for p in self.__params_in_ipg_bucket)
    
    def report_ipg_memory_usage(self, tag, param_elems):
        elem_count = self.elements_in_ipg_bucket + param_elems
        percent_of_bucket_size = (100.0 * elem_count) // self.reduce_bucket_size
        see_memory_usage(
            f"{tag}: elems in_bucket {self.elements_in_ipg_bucket} param {param_elems} max_percent {percent_of_bucket_size}",
            force=False)
        
    @instrument_w_nvtx
    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage(f"In ipg_epilogue before reduce_ipg_grads", 0)
        self.__reduce_and_partition_ipg_grads()
        self.report_ipg_memory_usage(f"In ipg_epilogue after reduce_ipg_grads", 0)

        self.__reduce_and_partition_stream.synchronize()

        for i in range(len(self.params_already_reduced)):
            self.params_already_reduced[i] = False

        '''
        #if not self.offload_optimizer:
        for i, sub_group in enumerate(self.fp16_groups):
            self.averaged_gradients[i] = [
                self.__param_id_to_grad_partition[param.ds_id]
                if param.requires_grad else torch.zeros_like(param.ds_tensor)
                for param in sub_group
            ]
        '''
                
        self.micro_step_id += 1

    #DeepSpeedEngineにより呼ばれるよ
    def overlapping_partition_gradients_reduce_epilogue(self):
        self.independent_gradient_partition_epilogue()
    
    def create_reduce_and_remove_grad_hooks(self):
        print_rank_0(f'[Begin] Create gradient reduction hooks')
        self.grad_accs = []
        for i, param_group in enumerate(self.fp16_groups):
            for param in param_group:
                if param.requires_grad:
                    param.all_gather()
                    
                    def wrapper(param, i):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0] #計算ノード(勾配蓄積ノード)のタイミングでreduce_partition_and_remove_gradsが実行されるようにする

                        @instrument_w_nvtx
                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param, i)

                        grad_acc.register_hook(reduce_partition_and_remove_grads)
                        self.grad_accs.append(grad_acc)
                    
                    wrapper(param, i)
                    param.partition() #parameterを分割して持っておく
        print_rank_0(f'[End] Create gradient reduction hooks')
        
    def reduce_ready_partitions_and_remove_grads(self, param, i):
        self.reduce_independent_p_g_buckets_and_remove_grads(param, i)
        
    ###############Idependent Partition Gradient ########################
    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        if self.elements_in_ipg_bucket > 0 and self.elements_in_ipg_bucket + param.ds_numel > self.reduce_bucket_size:
            self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads",
                                         param.ds_numel)
            self.__reduce_and_partition_ipg_grads() #reduce_bucketサイズいっぱいになったら勾配をreduceする
            
        param_id = self.get_param_id(param)
        assert self.params_already_reduced[param_id] == False, \
            f"The parameter {param_id} has already been reduced. \
            Gradient computed twice for this partition. \
            Multiple gradient reduction is currently not supported"
            
        self.__add_grad_to_ipg_bucket(param) #paramをipg_bucketに詰める
        
    #IPG（intermediate gradient）バケットに溜めた勾配を 集約（reduce/average）してから、各ランクの担当分に分割（partition） するメソッド
    @instrument_w_nvtx #NVTX の範囲計測用デコレータ。プロファイラでこの関数の実行区間を可視化します。
    @torch.no_grad()
    def __reduce_and_partition_ipg_grads(self) -> None:
        if not self.__params_in_ipg_bucket: #空
            return
        
        for param in self.__params_in_ipg_bucket:
            if param.grad.numel() != param.ds_numel:
                raise RuntimeError(f"{param.grad.numel()} != {param.ds_numel} Cannot reduce scatter " f"gradients whose size is not same as the params")
        
        self.__params_in_ipg_bucket.sort(key=lambda p: p.ds_id) #連続領域になるようにid順にソート

        #以前に発行した CUDA イベント（後述）のうち、先頭から完了済み（query() が True）をデキューして掃除。
        while self.__param_reduce_events and self.__param_reduce_events[0].query():
            self.__param_reduce_events.popleft()
        #それでもイベントが多すぎる場合、最古のイベントを同期（synchronize()）して確実に完了させたうえで捨てる。
        if len(self.__param_reduce_events) > self.__max_param_reduce_events:
            self.__param_reduce_events.popleft().synchronize()
            
        with torch.cuda.stream(self.__reduce_and_partition_stream): #別ストリームで実行
            #reduce_scatterに相当する処理(自分の担当分のreduce済み勾配)
            grad_partitions = self.__avg_scatter_grads(self.__params_in_ipg_bucket)
            #上で得た grad_partitions を使って、各 param.grad に自ランク分を反映（書き戻し／差し替え）。
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            # 集約済み勾配 (fp16) を GPU→CPU に転送。このイベント区間で D2H 時間を計測する。
            self.__partition_grads(self.__params_in_ipg_bucket, grad_partitions)
            
            end_event.record()
            end_event.synchronize()
            _accumulate_grad_offload_d2h_time_ms(start_event.elapsed_time(end_event))

            self.__params_in_ipg_bucket.clear() #通信済みはクリアする
            event = Event()
            event.record()
            self.__param_reduce_events.append(event) #CUDA イベントを発行し、このストリーム上の直前までの処理が いつ完了したかを非同期に追跡できるようにする。

    @instrument_w_nvtx #NVTX の範囲計測用デコレータ。プロファイラでこの関数の実行区間を可視化します。
    @torch.no_grad()
    def __add_grad_to_ipg_bucket(self, param: Parameter) -> None: #このparamはフルサイズの勾配
        #wait_stream(A) を B ストリームの文脈で呼ぶと、
        #A 上の「それ以前に発行された全ての作業が完了したこと」を示すイベントを記録し、
        #B にそのイベントを待たせるので、B の以降の仕事は A の以前の仕事の完了後にだけ進むようになります。
        self.__reduce_and_partition_stream.wait_stream(torch.cuda.default_stream()) #通信・分割用ストリームが、**デフォルトストリーム（計算）**の処理完了を待つよう依存関係を張る。計算が終わって勾配が出来てから詰めるため。

        if self.contiguous_gradients and self.elements_in_ipg_bucket + param.grad.numel() < self.reduce_bucket_size:
            #連結勾配モードかつ、いまのバケット使用量 + この勾配の要素数がバケット容量未満なら、フラット連結バッファに詰め替える。
            with torch.cuda.stream(self.__reduce_and_partition_stream):
                #view_as(tensor): tensorと同じ形にreshapeするmethod
                new_grad_tensor = self.__ipg_bucket_flat_buffer.narrow(0, self.elements_in_ipg_bucket, param.grad.numel()).view_as(param.grad)
                new_grad_tensor.copy_(param.grad, non_blocking=True) #param.gradと同じ内容をコピー
                #そのテンソルのストレージ(実メモリ)を、指定したstreamが完了するまで解放・再利用させないように記録
                param.grad.record_stream(torch.cuda.current_stream()) 
                param.grad.data = new_grad_tensor #paramがipg_bucket_flat_bufferをさすようにする
                
        self.__params_in_ipg_bucket.append(param) #容量を超えていれば非連続領域に詰めるしかないのでcontiguous_gradientがTrueでもipg_bucketにそのまま詰める

    @instrument_w_nvtx
    def __avg_scatter_grads(self, params_to_reduce: List[Parameter]) -> List[Tensor]:
        dtype = get_only_unique_item(p.grad.dtype for p in params_to_reduce)
        full_grads_for_rank = [p.grad for p in params_to_reduce] #それぞれreduceする対象の勾配
        if self.communication_data_type == torch.float32:
            full_grads_for_rank = [g.float() for g in full_grads_for_rank]
            
        #複数テンソルをまとめてreduce_scatter, 返り値はreduce済み自分の担当勾配
        grad_partitions_for_rank = reduce_scatter_coalesced(full_grads_for_rank, self.dp_process_group)
        
        if self.communication_data_type == torch.float32:
            grad_partitions_for_rank = [g.to(dtype) for g in grad_partitions_for_rank]

        return grad_partitions_for_rank

    
    #self.__partition_grads(self.__params_in_ipg_bucket, grad_partitions)で呼ばれる(grad_partitions: 自分の担当の集約済み勾配のリスト)
    #reduce-scatter 済みの 各パラメータ用の勾配パーティション（grad_partitions）を、内部の保持先バッファへ コピー/加算 し、必要なら オフロード、最後に param.grad を解放
    #self.__partition_grads(self.__params_in_ipg_bucket, grad_partitions)で呼ばれる
    @instrument_w_nvtx
    def __partition_grads(self, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
        for param, grad_partition in zip(params_to_release, grad_partitions):
            if param.partition_numel() * dist.get_rank(self.dp_process_group) > param.ds_numel:
                # this grad partition is empty - don't need to do anything
                continue
            #grad_bufferはparamが含まれているdeviceをそのまま参照している
            grad_buffer = self.__param_id_to_grad_partition[param.ds_id].narrow(0, 0, grad_partition.numel())
            
            if self.micro_step_id == 0:  # don't accumulate, マイクロステップ最初（累積しない）なら、受け取った分割勾配を 上書きコピー。
                # D2H の時間は NVTX 区間 'xfer:grad_d2h' の memcpy 射影 (nsys) で取得する。
                _xfer_push("xfer:grad_d2h")
                try:
                    grad_buffer.copy_(grad_partition, non_blocking=True)
                finally:
                    _xfer_pop()
                grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
            elif grad_buffer.is_cuda:
                grad_buffer.add_(grad_partition) #それ以外で grad_buffer が CUDA 上なら、加算で勾配を累積（勾配蓄積）
            else:
                # if dst is CPU, copy first to src device, do the addition
                # there, then move back to dst. adding directly to cpu is very slow
                #grad_buffer が CPU 側のときは、直接 CPU で加算せず、いったん CUDA へ搬送 → 2) GPU 上で加算 → 3) 結果を元バッファへ反映。
                #さらに以降のために grad_buffer 参照を CUDA 側に切替（以後の処理を非同期・高速化）。
                cuda_grad_buffer = grad_buffer.to(grad_partition.device,
                                                  non_blocking=True) #.to(device) が デバイス間コピー（転送） を行います。, grad_partition(GPUでreduce_scatterを行いこの変数を作ったのでこの変数はGPU上に存在)
                cuda_grad_buffer.add_(grad_partition) #GPU上で加算
                _xfer_push("xfer:grad_d2h")
                try:
                    grad_buffer.copy_(cuda_grad_buffer, non_blocking=True) #CPU上のgrad_bufferにGPU上のgrad_bufferの内容をコピー
                finally:
                    _xfer_pop()
                # ensure grad buffer is a CUDA buffer to speed up the next few
                # operations and so it can be used asynchronously
                grad_buffer = cuda_grad_buffer
                
            if self.offload_optimizer:
                '''
                いま処理中の param の フラット化された FP32 勾配配列の中での位置情報を取得。
                i: どの グループ（partitioned group）に属するか
                dest_offset: そのフラット勾配バッファ内の 書き込み開始オフセット
                _: 使わない補助情報（長さなどが入っていることが多い）
                '''
                i, dest_offset, _ = self.grad_position[self.get_param_id(param)]
                offload_fp32_gradients = {}
                offload_fp32_offsets = {}
                if self.is_gradient_accumulation_boundary:
                    fp32_grad_tensor = self.fp32_grad_bufs[self.grad_buf_switch][i].narrow(0, dest_offset, grad_buffer.numel())
                    _xfer_push("xfer:grad_d2h")
                    try:
                        fp32_grad_tensor.copy_(grad_buffer)  # fp16 grad -> fp32 grad buf
                    finally:
                        _xfer_pop()
                
            param.grad.record_stream(torch.cuda.current_stream())
            param.grad = None
            
            
    '''ここからはbackward(), step()についての実装'''
    @instrument_w_nvtx
    def backward(self, loss, retain_graph=False):
        see_memory_usage(f"Before backward", force=False)
        loss.float().backward(retain_graph=retain_graph)
        self._get_param_coordinator(training=True).reset_step()
        
        
    @instrument_w_nvtx
    def _partition_all_parameters(self):
        self.parameter_offload.partition_all_parameters()
        
    def _pre_step(self):
        self.micro_step_id = 0
        print_rank_0(f"Inside Step function")
        see_memory_usage(f"In step before checking overflow", force=False)
        print_rank_0("Finished Tracing at Beginning of Step")
        
        self._get_param_coordinator(training=True).hierarchy = 0
        
        print_rank_0("Finished Tracing at Beginning of Step")
        
    @instrument_w_nvtx
    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        self.micro_step_id = 0
        
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    if p.grad is not None and p.grad.is_cuda:
                        p.grad.record_stream(torch.cuda.current_stream())
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
    
    '''
    @instrument_w_nvtx
    def _prepare_fp32_grad_for_sub_group(self, sub_group_id):
        partition_id = dist.get_rank(group=self.dp_process_group)
        #averaged_gradients[sub_group_id]（このサブグループの勾配シャード列）を flatten で1本の連続 1D テンソルに結合。
        # その後、FP32 マスタ重みフラット（fp32_partitioned_groups_flat[sub_group_id]）の dtype（通常 float32）に揃える。
        single_grad_partition = self.flatten(self.averaged_gradients[sub_group_id]).to(self.fp32_partitioned_groups_flat[sub_group_id].dtype)
        #FP32 マスタ重みフラットの .grad をこの連続フラット勾配に差し替え。→ Optimizer はここを読む
        # （あなたの _optimizer_step() が、まさにこの FP32 パラメータ1本を optimizer.param_groups[...]['params'] に入れて step します）。
        self.fp32_partitioned_groups_flat[sub_group_id].grad = single_grad_partition
        self.zero_grad()
        #record_stream は CUDA テンソルの“ストレージ寿命”を、指定したストリームの完了まで延長するための API です。順序（実行の前後関係）を作るものではなく、メモリの再利用・解放のタイミングを守らせる
        #current_stream()が終わるまではこの勾配ストレージ(averaged_gradients)を捨てないことを保証
        for grad in filter(lambda g: g.is_cuda, self.averaged_gradients[sub_group_id]):
            grad.record_stream(torch.cuda.current_stream())
            
        self.averaged_gradients[sub_group_id] = None
    '''
        
        
    @instrument_w_nvtx
    def _prepare_sub_group(self, sub_group_id, timer_names=set()):
        see_memory_usage(f'Before prepare optimizer sub group {sub_group_id}', force=False)
        
        #CPUでoptimizer更新を行うときはすでにfp32側に入っているはずなので前処理はいらない
        #self._prepare_fp32_grad_for_sub_group(sub_group_id) #GPU 常駐パスで、当該サブグループの FP32 マスタ側の .grad を使用可能に整える前処理
        
        see_memory_usage(f'After prepare optimizer sub group {sub_group_id}', force=False)
        
    '''
    self._unflatten_partitioned_parameters(sub_group_id) で、フラットバッファ上の各スライスを個々の Parameter shard の .data に再び張り直す（ビュー/コピー）処理を行います。
    これにより モデルパラメータ（分割 shard）が最新の値を指し、次イテレーションの forward/backward で使える状態になります。
    '''
    def _unflatten_partitioned_parameters(self, sub_group_id):
        updated_params = self.unflatten(self.fp16_partitioned_groups_flat[sub_group_id],
                                        self.fp16_partitioned_groups[sub_group_id])

        for partitioned_param, q in zip(self.fp16_partitioned_groups[sub_group_id], updated_params):
            partitioned_param.data = q.data
        
    @instrument_w_nvtx
    def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
        if self.fp16_partitioned_groups_flat[sub_group_id] is not None:
            self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
                self.fp32_partitioned_groups_flat[sub_group_id].data)
            self._unflatten_partitioned_parameters(sub_group_id)

    @instrument_w_nvtx
    def _release_sub_group(self, sub_group_id, timer_names=set()):
        
        see_memory_usage(f'Before release optimizer sub group {sub_group_id}', force=False)
        # get rid of the fp32 gradients. Not needed anymore
        #CPUにfp32 gradientが常駐しているのでNoneにして消さなくてもいい
        #self.fp32_partitioned_groups_flat[sub_group_id].grad = None

        see_memory_usage(f'After release optimizer sub group {sub_group_id}', force=False)
        
    @instrument_w_nvtx
    def _post_step(self, timer_names=set()):
        if self.offload_optimizer: #Offloadしないから今は無視
            #self.reset_cpu_buffers() #overflowなどのリセットだったので無視する
            pass
            
        #self.log_timers(timer_names)
        see_memory_usage('After zero_optimizer step', force=False)
        print_rank_0(f"------------------Finishing Step-----------------------")
        
    @instrument_w_nvtx
    def step(self, closure=None):
        self._pre_step() #self.micro_step=0, self._get_param_coordinator(training=True).hierarchy = 0
        self._partition_all_parameters() #モジュール配下（再帰）の全パラメータを強制的に解放＆状態初期化, forward, backwardが終了したのでもうパラメータは分割してもOK

        timer_names = set()
        #timer_names.add('optimizer_step')
        #self.start_timers(['optimizer_step'])

        from common import debug_params as dbg

        #update parameters one sub group at a time
        for sub_group_id, group in enumerate(self.fp16_groups):

            #prepare optimizer states, gradients and fp32 parameters for update
            self._prepare_sub_group(sub_group_id, timer_names) #fp32マスタコピーの登録

            dbg.log_param_update("BEFORE_STEP", sub_group_id, self.fp32_partitioned_groups_flat[sub_group_id])

            #apply the optimizer step on the sub group and copy fp32 parameters to fp16
            self._optimizer_step(sub_group_id) #sub_groupに対応するfp32マスタコピーを登録してそのままoptimizer.step()を行う

            dbg.log_param_update("AFTER_STEP", sub_group_id, self.fp32_partitioned_groups_flat[sub_group_id])

            #put fp16 parameters in appropriate location
            self._reassign_or_swap_out_partitioned_parameters(sub_group_id) #fp32 -> fp16へと重みパラメータを反映させる

            dbg.log_param_update("AFTER_COPY", sub_group_id,
                                 self.fp32_partitioned_groups_flat[sub_group_id],
                                 self.fp16_partitioned_groups_flat[sub_group_id])

            #release memory or swap out optimizer states of fp32 parameters
            self._release_sub_group(sub_group_id, timer_names) #fp32マスタコピーについての勾配を解放

        #self.stop_timers(['optimizer_step'])

        self._post_step(timer_names) #基本無視するかも


    '''Delayed Parameter Update周りの関数 (CPU Adam は child process で実行)'''

    @instrument_w_nvtx
    def _reassign_or_swap_out_partitioned_parameters_dpu(self, sub_group_id):
        if self.fp16_partitioned_groups_flat[sub_group_id] is not None:
            self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
                self.fp32_partitioned_groups_flat[sub_group_id].data)
            self._unflatten_partitioned_parameters(sub_group_id)

    @instrument_w_nvtx
    def step_dpu(self, first_step=False):
        """child process に Adam step コマンドを送信 (non-blocking)。"""
        from common import debug_params as dbg
        self._adam_step_pending = False
        if not first_step:
            dbg.log_param_update("DPU_BEFORE_STEP", 0, self.fp32_partitioned_groups_flat[0])
            self._adam_pipe_parent.send(("step", 1 - self.grad_buf_switch))
            self._adam_step_pending = True

    @instrument_w_nvtx
    def update_new_params(self):
        """child process の Adam 完了を待ち、FP32→FP16 を反映してバッファをトグル。"""
        from common import debug_params as dbg
        if self._adam_step_pending:
            result = self._adam_pipe_parent.recv()
            assert result == "done", f"unexpected message from adam worker: {result}"
        dbg.log_param_update("DPU_AFTER_STEP", 0, self.fp32_partitioned_groups_flat[0])
        for sub_group_id, group in enumerate(self.fp16_groups):
            self._reassign_or_swap_out_partitioned_parameters_dpu(sub_group_id)
        dbg.log_param_update("DPU_AFTER_COPY", 0,
                             self.fp32_partitioned_groups_flat[0],
                             self.fp16_partitioned_groups_flat[0])
        self.grad_buf_switch = 1 - self.grad_buf_switch

    def start_adam_process(self, cpu_affinity=None):
        """CPU Adam を実行する worker を起動する。

        DISABLE_ADAM_FORK=1: threading.Thread で同一プロセス内で起動(nsys/CUPTI 共存用)。
          DeepSpeedCPUAdam の step() は C++ 拡張で GIL を release するため、
          main thread(GPU kernel launch)との overlap は実質失われない。
        それ以外(default): multiprocessing.Process(fork) で別プロセス起動(従来通り)。
        """
        import os
        defaults = {k: v for k, v in self.optimizer.defaults.items()}

        if os.environ.get("DISABLE_ADAM_FORK", "0") == "1":
            import multiprocessing, threading
            parent_pipe, child_pipe = multiprocessing.Pipe()
            self._adam_pipe_parent = parent_pipe
            self._adam_thread = threading.Thread(
                target=_adam_worker,
                args=(child_pipe, self.fp32_partitioned_groups_flat,
                      self.fp32_grad_bufs, defaults,
                      dict(self.sub_group_to_group_id), len(self.fp16_groups)),
                daemon=True, name="adam-worker-thread")
            self._adam_thread.start()
            msg = parent_pipe.recv()
            assert msg == "ready", f"adam thread did not start: {msg}"
            self._adam_process = None
        else:
            import torch.multiprocessing as mp
            ctx = mp.get_context("fork")
            parent_pipe, child_pipe = ctx.Pipe()
            self._adam_pipe_parent = parent_pipe
            self._adam_process = ctx.Process(
                target=_adam_worker,
                args=(child_pipe, self.fp32_partitioned_groups_flat,
                      self.fp32_grad_bufs, defaults,
                      dict(self.sub_group_to_group_id), len(self.fp16_groups)),
                daemon=True)
            self._adam_process.start()
            msg = parent_pipe.recv()
            assert msg == "ready", f"adam worker did not start: {msg}"
            self._adam_thread = None

        if cpu_affinity is not None:
            parent_pipe.send(("set_affinity", list(cpu_affinity)))

    def stop_adam_process(self):
        """child process / thread を終了する。"""
        try:
            if self._adam_pipe_parent is not None:
                self._adam_pipe_parent.send(("shutdown",))
        except Exception:
            pass

        if self._adam_process is not None and self._adam_process.is_alive():
            try:
                self._adam_process.join(timeout=10)
            except Exception:
                pass
            if self._adam_process.is_alive():
                self._adam_process.terminate()

        if self._adam_thread is not None and self._adam_thread.is_alive():
            try:
                self._adam_thread.join(timeout=10)
            except Exception:
                pass