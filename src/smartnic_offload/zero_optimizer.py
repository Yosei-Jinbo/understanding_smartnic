import sys
import os
import argparse
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
from parameter_offload import ZeroOffload
from torch._utils import _flatten_dense_tensors as flatten
from torch._utils import _unflatten_dense_tensors as unflatten
from torch.distributed import ProcessGroup
import math
import itertools
from typing import Deque, Dict, Tuple, List, Optional
from torch import Tensor
from torch.nn import Parameter
from deepspeed.ops.adam import DeepSpeedCPUAdam
from concurrent.futures import ThreadPoolExecutor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
COMCH_HOST_DIR = THIS_DIR / "comch_mpi" / "host"
sys.path.insert(0, str(COMCH_HOST_DIR))

import doca_comch_client_pybind
from doca_comch_client_pybind import CollectiveCommunication

def print_rank_0(message, debug=False, force=False):
    rank = dist.get_rank()
    if rank == 0 and (debug or force):
        #print(message)
        pass
    
def _flatten(tensors):
    # dtype / device を揃えておくこと（ここでは fp32/CPU を想定）
    ts = [t.detach().contiguous() for t in tensors]
    return _flatten_dense_tensors(ts)

class _ComchHandleWork:
    """
    C 側の handle を保持して wait/test/release する Work。
    keep_alive: enqueue した src テンソル等を doorbell 完了 (=wait()) まで保持する
    """
    def __init__(self, handle: int, keep_alive=None) -> None:
        self._handle = int(handle)
        self._released = False
        self._completed = False
        self._keep_alive = keep_alive

    def wait(self):
        if not self._completed:
            doca_comch_client_pybind.comch_req_wait_py(self._handle)
            self._completed = True
            self._keep_alive = None  # DPU 読了済み。バッファを解放してよい
        if not self._released:
            doca_comch_client_pybind.comch_req_release_py(self._handle)
            self._released = True
        return None

    def __del__(self):
        # wait() せずに破棄された場合でもリークしないように release
        if (not self._released) and (self._handle is not None):
            try:
                doca_comch_client_pybind.comch_req_release_py(self._handle)
            except Exception:
                pass
            self._released = True

# =========================
# フラット版 ReduceScatter
# =========================
def reduce_scatter_flat_via_base(
    output_flat: torch.Tensor,
    input_flat: torch.Tensor,
    cid: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    world_size = dist.get_world_size()

    out_view = output_flat.contiguous().view(-1)
    in_view = input_flat.contiguous().view(-1)

    handle = doca_comch_client_pybind.ucp_collective_enqueue_py(
        cid, in_view, out_view, CollectiveCommunication.REDUCE_SCATTER
    )
    # in_view を doorbell 完了まで保持 (src が一時バッファの場合、参照を切らすと
    # DPU 読み取り前にアロケータが再利用する use-after-free になる)
    if not async_op:
        w = _ComchHandleWork(handle, keep_alive=(in_view,))
        w.wait()
        return None
    else:
        return _ComchHandleWork(handle, keep_alive=(in_view,))


#複数のテンソルをまとめて reduce-scatter（勾配の平均＋分割）するための高効率ユーティリティ関数
@instrument_w_nvtx
@torch.no_grad()
def reduce_scatter_coalesced(
    tensors: List[Tensor],
    group: ProcessGroup = None,
    cid: int = 0,
    output_buffer: Tensor = None,  # 外部出力バッファ (Noneなら内部で新規確保)
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
        tensor_partition_flat_buffer = instrument_w_nvtx(torch.cat)(tensor_partitions_lst_with_padding).contiguous()
    
    tensor_partition_flat_buffer.div_(world_sz)  # pre-divide
    
    # ==== reduce_scatter 実行 ====
    chunk_numel = tensor_partition_flat_buffer.numel() // world_sz

    # RS_USE_NCCL=1 で NCCL RS に切り替え (AG は DPU のまま)
    _rs_use_nccl = os.environ.get("RS_USE_NCCL", "0") == "1"

    if _rs_use_nccl:
        # NCCL 版: 入出力ともに GPU 上 (NCCL は GPU テンソルが必須)
        gpu_device = tensor_partition_flat_buffer.device
        # output_buffer が GPU 上で十分なサイズなら再利用、なければ新規確保
        if output_buffer is not None and output_buffer.is_cuda and output_buffer.numel() >= chunk_numel:
            output_flat = output_buffer.narrow(0, 0, chunk_numel).contiguous()
        else:
            output_flat = torch.empty(chunk_numel, dtype=tensor_partition_flat_buffer.dtype,
                                      device=gpu_device)
    else:
        # DOCA 版: 出力は CPU (DPU が CPU pinned memory に書き戻す)
        if output_buffer is not None:
            assert output_buffer.numel() >= chunk_numel, \
                f"output_buffer too small: {output_buffer.numel()} < {chunk_numel}"
            output_flat = output_buffer.narrow(0, 0, chunk_numel).contiguous()
        else:
            output_flat = torch.empty(
                chunk_numel,
                dtype=tensor_partition_flat_buffer.dtype,
                device="cpu",
            ).contiguous()

    from common import debug_params as dbg
    dbg.log_reduce_scatter(tensor_partition_flat_buffer, None, sub_group_id="flat_input")

    submit_fn = None
    src_ready_ev = None
    if _rs_use_nccl:
        # NCCL reduce_scatter (同期)
        input_chunks = list(torch.chunk(tensor_partition_flat_buffer, world_sz, dim=0))
        dist.reduce_scatter(output_flat, input_chunks, op=dist.ReduceOp.SUM, group=group)
        rs_handle = None
    else:
        # DOCA 版: src (cat + div_ の結果) は CUDA stream 上の非同期カーネルが書くが、
        # DPU の RDMA read は CUDA stream の順序保証の外にあるため、カーネル完了前に
        # enqueue すると DPU が計算前の内容を読む (NCCL RS は stream 順序で守られる)。
        # CPU をブロックしないよう、event を record して submit を遅延する:
        # 以後の maybe_submit() (勾配フック毎) で event.query() が立ったら enqueue、
        # wait() が先に来た場合のみ event を待ってから enqueue する。
        rs_handle = None
        src_ready_ev = Event()
        src_ready_ev.record()
        _src = tensor_partition_flat_buffer  # submit まで src への参照を保持

        def submit_fn(_out=output_flat, _in=_src, _cid=cid, _group=group):
            return reduce_scatter_flat_via_base(
                _out, _in, cid=_cid, group=_group, async_op=True)

    # RS handle を返す (呼び出し元が wait タイミングを制御)
    class _DeferredRsResult:
        """DOCA RS の submit (src ready 後) と wait を保持する。
        maybe_submit() は非ブロッキング: src を書く CUDA カーネルが完了していれば
        その場で DPU へ enqueue する。wait() は未 submit なら event を待って submit
        してから完了をブロック待ちする。in-flight は常に 1 個 (_pending_rs) なので
        遅延 submit でも rank 間の発行順序は保たれる。"""
        def __init__(self, handle, output_flat, output_lst, submit_fn=None, src_ready_ev=None):
            self._handle = handle
            self.output_flat = output_flat
            self.output_lst = output_lst
            self._waited = False
            self._submit_fn = submit_fn
            self._src_ready_ev = src_ready_ev

        def maybe_submit(self):
            if self._submit_fn is not None and self._src_ready_ev.query():
                self._handle = self._submit_fn()
                self._submit_fn = None

        def force_submit(self):
            """未 submit なら src ready を待って必ず submit する。
            別 collective (AG) の enqueue 直前に呼ばれ、全 rank の発行順序を
            決定論的に揃える (event 待ちは通常ゼロ〜サブ ms)。DPU が RS/AG を
            単一 msg_pool で直列実行するため、順序一致は必須 (ズレるとデッドロック)。"""
            if self._submit_fn is not None:
                self._src_ready_ev.synchronize()
                self._handle = self._submit_fn()
                self._submit_fn = None

        def wait(self):
            if self._waited:
                return self.output_lst
            self.force_submit()
            if self._handle is not None:
                torch.cuda.nvtx.range_push("rs_wait")
                try:
                    self._handle.wait()
                finally:
                    torch.cuda.nvtx.range_pop()
            self._waited = True
            return self.output_lst

    # ==== rank ごとの flat 出力から、各テンソルごとのチャンクを切り出す ====
    # output_flat の narrow はビュー (RS 完了後にデータが有効になる)
    output_lst: List[Tensor] = [None] * len(tensors)
    offset = 0
    for tensor_idx in range(len(tensors)):
        local_numel = partition_lst_for_each_tensor[tensor_idx][this_rank].numel()
        output_lst[tensor_idx] = output_flat.narrow(0, offset, local_numel)
        offset += padded_partition_sz_for_each_tensor[tensor_idx]

    dbg.log_reduce_scatter(None, output_lst, sub_group_id="flat_output")

    return _DeferredRsResult(rs_handle, output_flat, output_lst,
                             submit_fn=submit_fn, src_ready_ev=src_ready_ev)


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
                 reduce_bucket_size=0,
                 prefetch_bucket_size=0,
                 max_reuse_distance=0,
                 max_live_parameters=0,
                 dp_process_group=None,
                 reduce_scatter=True,
                 overlap_comm=True,
                 sub_group_size=0,
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

        # 遅延 submit 中の RS を AG enqueue 前に強制 submit するフックを登録。
        # これが無いと RS の submit 位置が GPU タイミング依存になり、rank 間で
        # collective の発行順序がズレて DPU ring が噛み合わない (RING_RECV Bad State)。
        import partition_parameters as _pp
        _pp.register_pre_enqueue_hook(self._force_submit_pending_rs)
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
                                pin_memory=True, device=self.device)) #CPU上にfp16_param_groupの大きい受け皿を作る
        
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
                                              fp16_partitioned_group_flat,
                                              avoid_copy=not self.offload_param)
        
        
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

        from common import debug_params as dbg
        if dbg.should_log():
            import torch.distributed as _dist
            _r = _dist.get_rank() if _dist.is_initialized() else 0
            _g = fp32_param.grad
            if _g is None:
                print(f"[DIAG step={dbg.STEP} rank={_r}] _optimizer_step sub_group={sub_group_id}: grad is None!", flush=True)
            else:
                _gf = _g.detach().float().flatten()
                print(f"[DIAG step={dbg.STEP} rank={_r}] _optimizer_step sub_group={sub_group_id}: "
                      f"grad shape={tuple(_g.shape)} norm={_gf.norm().item():.6f} "
                      f"grad_buf[0] same_storage={_g.data_ptr() == self.fp32_grad_bufs[0][sub_group_id].data_ptr()} "
                      f"grad_buf[1] same_storage={_g.data_ptr() == self.fp32_grad_bufs[1][sub_group_id].data_ptr()}",
                      flush=True)

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
        
        self.__param_id_to_grad_partition: Dict[int, Tensor] = {} #param_id => partition_flat_bufferのどの領域か?
        self.__param_id_to_grad_offset: Dict[int, int] = {}     #param_id => flat_buffer内のオフセット
        all_params = list(itertools.chain.from_iterable(self.fp16_groups)) #1列のリストにする
        self.__grad_partitions_flat_buffer: Tensor = torch.zeros(sum(p.partition_numel() for p in all_params), dtype=self.dtype, device=self.device, pin_memory=True)

        offset = 0
        #param.ds_id: 各パラメータに割り振られた一意なid
        for param in all_params:
            self.__param_id_to_grad_partition[param.ds_id] = self.__grad_partitions_flat_buffer.narrow(0, offset, param.partition_numel())
            self.__param_id_to_grad_offset[param.ds_id] = offset
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
    @instrument_w_nvtx
    @torch.no_grad()
    def __reduce_and_partition_ipg_grads(self) -> None:
        if not self.__params_in_ipg_bucket:
            return

        for param in self.__params_in_ipg_bucket:
            if param.grad.numel() != param.ds_numel:
                raise RuntimeError(f"{param.grad.numel()} != {param.ds_numel} Cannot reduce scatter " f"gradients whose size is not same as the params")

        self.__params_in_ipg_bucket.sort(key=lambda p: p.ds_id)

        while self.__param_reduce_events and self.__param_reduce_events[0].query():
            self.__param_reduce_events.popleft()
        if len(self.__param_reduce_events) > self.__max_param_reduce_events:
            self.__param_reduce_events.popleft().synchronize()

        # ---- RS wait 遅延: 前回の RS を wait + partition してから今回の RS を submit ----
        # 前回の deferred RS が残っていれば、ここで wait して partition
        if hasattr(self, '_pending_rs') and self._pending_rs is not None:
            prev_deferred, prev_params = self._pending_rs
            with torch.cuda.stream(self.__reduce_and_partition_stream):
                grad_partitions = prev_deferred.wait()
                if prev_deferred._needs_dtype_convert:
                    grad_partitions = [g.to(prev_deferred._target_dtype) for g in grad_partitions]
                # D2H (grad GPU->CPU) は nsys の memcpy トレースで観測する。
                self.__partition_grads(prev_params, grad_partitions)
                event = Event()
                event.record()
                self.__param_reduce_events.append(event)
            self._pending_rs = None

        # 今回の RS を submit (wait しない)
        with torch.cuda.stream(self.__reduce_and_partition_stream):
            deferred = self.__avg_scatter_grads(self.__params_in_ipg_bucket)

        # params と deferred を保存 (次回の呼び出し or flush で wait)
        self._pending_rs = (deferred, list(self.__params_in_ipg_bucket))
        self.__params_in_ipg_bucket.clear()

    @instrument_w_nvtx #NVTX の範囲計測用デコレータ。プロファイラでこの関数の実行区間を可視化します。
    @torch.no_grad()
    def _force_submit_pending_rs(self) -> None:
        """保留中の deferred RS を (src ready を待ってでも) 必ず submit する。
        partition_parameters の AG enqueue 直前フックから呼ばれる。"""
        pending = getattr(self, '_pending_rs', None)
        if pending is not None:
            pending[0].force_submit()

    def __add_grad_to_ipg_bucket(self, param: Parameter) -> None: #このparamはフルサイズの勾配
        # 保留中の deferred RS があれば、src ready (CUDA event) を非ブロッキングで
        # 確認してその場で DPU へ submit する (bwd 中に param 毎に呼ばれるため、
        # cat/div_ 完了のサブms後には submit される)
        if getattr(self, '_pending_rs', None) is not None:
            self._pending_rs[0].maybe_submit()

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
    def __avg_scatter_grads(self, params_to_reduce: List[Parameter]):
        """RS を非同期発行し、_DeferredRsResult を返す (wait は呼び出し元が制御)。"""
        dtype = get_only_unique_item(p.grad.dtype for p in params_to_reduce)
        full_grads_for_rank = [p.grad for p in params_to_reduce]
        if self.communication_data_type == torch.float32:
            full_grads_for_rank = [g.float() for g in full_grads_for_rank]

        cid = int(params_to_reduce[0].ds_id)

        output_buf = None
        _rs_use_nccl = os.environ.get("RS_USE_NCCL", "0") == "1"
        first_offset = self.__param_id_to_grad_offset[params_to_reduce[0].ds_id]
        total_numel = sum(p.partition_numel() for p in params_to_reduce)
        last_p = params_to_reduce[-1]
        expected_end = self.__param_id_to_grad_offset[last_p.ds_id] + last_p.partition_numel()
        if first_offset + total_numel == expected_end and not _rs_use_nccl:
            # DOCA RS: CPU バッファに直接書き込み可能
            output_buf = self.__grad_partitions_flat_buffer.narrow(0, first_offset, total_numel)
        # NCCL RS: output_buf=None → GPU 上に新規確保 (後で __partition_grads で CPU にコピー)

        # _DeferredRsResult を返す (submit のみ、wait は遅延)
        deferred = reduce_scatter_coalesced(
            full_grads_for_rank, self.dp_process_group, cid=cid,
            output_buffer=output_buf,
        )

        # dtype 変換情報を deferred に付与
        deferred._needs_dtype_convert = (self.communication_data_type == torch.float32)
        deferred._target_dtype = dtype
        return deferred

    
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
            
            same_buffer = (grad_buffer.data_ptr() == grad_partition.data_ptr())

            from common import debug_params as dbg
            if dbg.should_log():
                import torch.distributed as _dist
                _r = _dist.get_rank() if _dist.is_initialized() else 0
                _gb = grad_buffer.detach().float().flatten()
                _gp = grad_partition.detach().float().flatten()
                print(f"[DIAG step={dbg.STEP} rank={_r}] __partition_grads param_id={param.ds_id}: "
                      f"grad_buffer norm={_gb.norm().item():.6f} device={grad_buffer.device} "
                      f"grad_partition norm={_gp.norm().item():.6f} device={grad_partition.device} "
                      f"same_data_ptr={same_buffer}",
                      flush=True)

            if same_buffer:
                pass  # RS出力が直接grad_bufferに書き込まれている: コピー不要
            elif self.micro_step_id == 0:
                # fp16 のまま CPU pinned バッファへプレーン D2H する (高速なエンジンコピー)。
                # ここで grad_buffer = grad_partition (GPU) に付け替えて下流の
                # fp32_grad_tensor.copy_() に GPU fp16 を直接渡すと、型変換つき
                # cross-device コピー (~5.3GB/s) に落ちて __partition_grads が
                # +400ms/step 遅くなる (nsys 実測)。下流は CPU fp16 から読む。
                torch.cuda.nvtx.range_push("xfer:grad_d2h")
                try:
                    grad_buffer.copy_(grad_partition, non_blocking=True)
                finally:
                    torch.cuda.nvtx.range_pop()
            else:
                if grad_buffer.device != grad_partition.device:
                    cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
                    cuda_grad_buffer.add_(grad_partition)
                    torch.cuda.nvtx.range_push("xfer:grad_d2h")
                    try:
                        grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
                    finally:
                        torch.cuda.nvtx.range_pop()
                    grad_buffer = cuda_grad_buffer
                else:
                    grad_buffer.add_(grad_partition)
                
                
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
                    torch.cuda.nvtx.range_push("xfer:grad_d2h")
                    try:
                        fp32_grad_tensor.copy_(grad_buffer)  # fp16 grad -> fp32 grad buf
                    finally:
                        torch.cuda.nvtx.range_pop()
                
            param.grad.record_stream(torch.cuda.current_stream())
            param.grad = None
            
            
    '''ここからはbackward(), step()についての実装'''
    @instrument_w_nvtx
    def backward(self, loss, retain_graph=False):
        see_memory_usage(f"Before backward", force=False)
        loss.float().backward(retain_graph=retain_graph)
        self._get_param_coordinator(training=True).reset_step()
        
        
    @instrument_w_nvtx
    def _flush_pending_rs(self):
        """Flush any pending deferred RS (wait + partition grads)."""
        if hasattr(self, '_pending_rs') and self._pending_rs is not None:
            prev_deferred, prev_params = self._pending_rs
            with torch.cuda.stream(self.__reduce_and_partition_stream):
                grad_partitions = prev_deferred.wait()
                if prev_deferred._needs_dtype_convert:
                    grad_partitions = [g.to(prev_deferred._target_dtype) for g in grad_partitions]
                # D2H (grad GPU->CPU) は nsys の memcpy トレースで観測する。
                self.__partition_grads(prev_params, grad_partitions)
                event = Event()
                event.record()
                self.__param_reduce_events.append(event)
            self._pending_rs = None

    def _partition_all_parameters(self):
        self._flush_pending_rs()  # backward 末尾の RS を確実に完了
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
        
    @instrument_w_nvtx
    def _prepare_sub_group(self, sub_group_id, timer_names=set()):
        see_memory_usage(f'Before prepare optimizer sub group {sub_group_id}', force=False)
        
        #CPUでoptimizer更新を行うときはすでにfp32側に入っているはずなので前処理はいらない

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
        """通常経路 (非 DPU): backward が書いた勾配バッファで CPU Adam を同期実行し、
        同一 step 内で fp32 → fp16 への反映まで行う。"""
        self._pre_step()
        self._partition_all_parameters()

        timer_names = set()
        from common import debug_params as dbg
        for sub_group_id, group in enumerate(self.fp16_groups):
            dbg.log_param_update("BEFORE_STEP", sub_group_id, self.fp32_partitioned_groups_flat[sub_group_id])

        # backward が書き込んだ側の勾配バッファ (grad_buf_switch) で step し、完了を待つ
        self._adam_pipe_parent.send(("step", self.grad_buf_switch))
        result = self._adam_pipe_parent.recv()
        assert result == "done", f"unexpected message from adam worker: {result}"

        for sub_group_id, group in enumerate(self.fp16_groups):
            self._prepare_sub_group(sub_group_id, timer_names)
            dbg.log_param_update("AFTER_STEP", sub_group_id, self.fp32_partitioned_groups_flat[sub_group_id])
            self._reassign_or_swap_out_partitioned_parameters(sub_group_id)
            dbg.log_param_update("AFTER_COPY", sub_group_id,
                                 self.fp32_partitioned_groups_flat[sub_group_id],
                                 self.fp16_partitioned_groups_flat[sub_group_id])
            self._release_sub_group(sub_group_id, timer_names)

        self._post_step(timer_names)

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