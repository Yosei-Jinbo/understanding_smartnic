import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import logger
import torch
import torch.distributed as dist
import gc
import time
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

# ---- Per-RS detailed records ----
_RS_RECORDS: list = []  # [(nbytes, total_ms, enqueue_ms, wait_ms)]

def _record_rs_detail(nbytes: int, total_ms: float, enqueue_ms: float, wait_ms: float) -> None:
    _RS_RECORDS.append((nbytes, total_ms, enqueue_ms, wait_ms))

def print_rs_analysis(epoch: int) -> None:
    """Print per-RS analysis at epoch end."""
    records = list(_RS_RECORDS)
    if not records:
        return
    n = len(records)
    total_ms_all = sum(t for _, t, _, _ in records)
    avg_total = total_ms_all / n
    avg_enq = sum(e for _, _, e, _ in records) / n
    avg_wait = sum(w for _, _, _, w in records) / n
    avg_sz = sum(s for s, _, _, _ in records) / n
    sorted_total = sorted(t for _, t, _, _ in records)
    p50 = sorted_total[n // 2]
    p99 = sorted_total[int(n * 0.99)] if n > 1 else sorted_total[0]
    print(f"========== Per-RS Analysis (Epoch {epoch}) ==========", flush=True)
    print(f"  RS calls: {n} | total: {total_ms_all/1000:.1f}s", flush=True)
    print(f"  Per-RS: avg={avg_total:.3f}ms | p50={p50:.3f}ms | p99={p99:.3f}ms", flush=True)
    print(f"  Avg size: {avg_sz/1024/1024:.1f}MB", flush=True)
    print(f"  Enqueue overhead: avg={avg_enq:.3f}ms", flush=True)
    print(f"  Wait (DPU + completion): avg={avg_wait:.3f}ms", flush=True)
    print(f"{'=' * 55}", flush=True)

def reset_rs_records() -> None:
    _RS_RECORDS.clear()

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


def _scatter_flat_to_list(
    flat_output: torch.Tensor,
    output_tensors: List[torch.Tensor],
    like_tensor: torch.Tensor,
) -> None:
    world_size = len(output_tensors)
    numel_per_rank = like_tensor.numel()

    flat_2d = flat_output.view(world_size, numel_per_rank)

    for rank, out in enumerate(output_tensors):
        out.copy_(flat_2d[rank].view_as(out))

class _ComchHandleWork:
    """
    C 側の handle を保持して wait/test/release する Work。
    """
    def __init__(self, handle: int) -> None:
        self._handle = int(handle)
        self._released = False
        self._completed = False

    def wait(self):
        if not self._completed:
            doca_comch_client_pybind.comch_req_wait_py(self._handle)
            self._completed = True
        if not self._released:
            doca_comch_client_pybind.comch_req_release_py(self._handle)
            self._released = True
        return None

    def is_completed(self) -> bool:
        if self._completed:
            return True
        done = doca_comch_client_pybind.comch_req_test_py(self._handle)
        if done:
            self._completed = True
        return done

    def __del__(self):
        # wait() せずに破棄された場合でもリークしないように release
        if (not self._released) and (self._handle is not None):
            try:
                doca_comch_client_pybind.comch_req_release_py(self._handle)
            except Exception:
                pass
            self._released = True

def _scatter_flat_to_list(
    flat_output: torch.Tensor,
    output_tensors: List[torch.Tensor],
    like_tensor: torch.Tensor,
) -> None:
    world_size = len(output_tensors)
    numel_per_rank = like_tensor.numel()

    flat_2d = flat_output.view(world_size, numel_per_rank)

    for rank, out in enumerate(output_tensors):
        out.copy_(flat_2d[rank].view_as(out))

    
# =========================
# リスト版 ReduceScatter
# =========================
def reduce_scatter_list_via_base(
    output_tensor: torch.Tensor,
    input_tensors: List[torch.Tensor],
    cid: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    world_size = dist.get_world_size()

    if len(input_tensors) != world_size:
        raise ValueError(f"len(input_tensors)={len(input_tensors)} must equal world_size={world_size}")

    out_numel = output_tensor.numel()
    '''
    for i, tin in enumerate(input_tensors):
        if tin.numel() != out_numel:
            raise ValueError(f"input_tensors[{i}].numel()={tin.numel()} must equal output_tensor.numel()={out_numel}")
        if tin.dtype != output_tensor.dtype:
            raise TypeError("input_tensors[i].dtype must match output_tensor.dtype")
        if tin.device != output_tensor.device:
            raise TypeError("input_tensors[i].device must match output_tensor.device")
    '''

    flat_list = [t.contiguous().view(-1) for t in input_tensors]
    input_flat = torch.cat(flat_list, dim=0).contiguous()

    output_flat = torch.empty(out_numel, dtype=output_tensor.dtype, device=output_tensor.device).contiguous()

    handle = doca_comch_client_pybind.ucp_collective_enqueue_py(
        cid, input_flat, output_flat, CollectiveCommunication.REDUCE_SCATTER
    )

    class _ReduceScatterHandleWork(_ComchHandleWork):
        def __init__(self, h, out_flat, out_t):
            super().__init__(h)
            self._out_flat = out_flat
            self._out_t = out_t

        def wait(self):
            super().wait()
            self._out_t.copy_(self._out_flat.view_as(self._out_t))
            return None

    if not async_op:
        w = _ReduceScatterHandleWork(handle, output_flat, output_tensor)
        w.wait()
        return None
    else:
        return _ReduceScatterHandleWork(handle, output_flat, output_tensor)

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

    '''
    if in_view.numel() != out_view.numel() * world_size:
        raise ValueError("input_flat.numel() must be output_flat.numel() * world_size")
    if out_view.dtype != in_view.dtype:
        raise TypeError("output_flat.dtype must match input_flat.dtype")
    if out_view.device != in_view.device:
        raise TypeError("output_flat.device must match input_flat.device")
    '''

    handle = doca_comch_client_pybind.ucp_collective_enqueue_py(
        cid, in_view, out_view, CollectiveCommunication.REDUCE_SCATTER
    )
    if not async_op:
        w = _ComchHandleWork(handle)
        w.wait()
        return None
    else:
        return _ComchHandleWork(handle)

'''
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
'''

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

    t_rs_start = time.perf_counter()

    if _rs_use_nccl:
        # NCCL reduce_scatter (同期)
        input_chunks = list(torch.chunk(tensor_partition_flat_buffer, world_sz, dim=0))
        # Phase 20: bracket the NCCL RS for symmetric stall measurement
        # (rs_handle が None なので _DeferredRsResult.wait() の bracket が発火しない。
        #  ここで sync NCCL RS の cudaStreamWaitEvent enqueue 区間を計測する)
        try:
            from common import stall_event_tracker as _set_rs  # type: ignore
        except Exception:
            _set_rs = None
        if _set_rs is not None and _set_rs.is_enabled():
            try:
                from partition_parameters import _AG_PHASE as _PHASE_RS  # type: ignore
            except Exception:
                _PHASE_RS = "unknown"
            _tr_rs = _set_rs.get_global()
            _stream_rs = torch.cuda.current_stream()
            _rs_nbytes_inline = int(tensor_partition_flat_buffer.numel() *
                                     tensor_partition_flat_buffer.element_size())
            _h_rs = _tr_rs.begin(_stream_rs, _PHASE_RS, op="rs",
                                  ds_id=-1, payload_bytes=_rs_nbytes_inline)
            try:
                dist.reduce_scatter(output_flat, input_chunks, op=dist.ReduceOp.SUM, group=group)
            finally:
                _tr_rs.end(_h_rs)
        else:
            dist.reduce_scatter(output_flat, input_chunks, op=dist.ReduceOp.SUM, group=group)
        rs_handle = None
    else:
        # DOCA 版フラット reduce_scatter を非同期で発行
        rs_handle = reduce_scatter_flat_via_base(
            output_flat,
            tensor_partition_flat_buffer,
            cid=cid,
            group=group,
            async_op=True,
        )

    t_rs_enqueued = time.perf_counter()

    # RS handle を返す (呼び出し元が wait タイミングを制御)
    class _DeferredRsResult:
        """RS handle + 計測情報を保持。wait() で完了を待ち、出力テンソルを返す。"""
        def __init__(self, handle, output_flat, output_lst, t_start, t_enqueued, nbytes):
            self._handle = handle
            self.output_flat = output_flat
            self.output_lst = output_lst
            self._t_start = t_start
            self._t_enqueued = t_enqueued
            self._nbytes = nbytes
            self._waited = False

        def wait(self):
            if self._waited:
                return self.output_lst
            if self._handle is not None:
                torch.cuda.nvtx.range_push("rs_wait")
                try:
                    # Phase 20: bracket the actual stream wait op for symmetric stall measurement
                    try:
                        from common import stall_event_tracker as _set  # type: ignore
                    except Exception:
                        _set = None
                    if _set is not None and _set.is_enabled():
                        try:
                            from partition_parameters import _AG_PHASE as _PHASE  # type: ignore
                        except Exception:
                            _PHASE = "unknown"
                        tracker = _set.get_global()
                        stream = torch.cuda.current_stream()
                        h = tracker.begin(stream, _PHASE, op="rs",
                                          ds_id=-1, payload_bytes=int(self._nbytes))
                        try:
                            self._handle.wait()
                        finally:
                            tracker.end(h)
                    else:
                        self._handle.wait()
                finally:
                    torch.cuda.nvtx.range_pop()
            t_rs_end = time.perf_counter()
            total_ms = (t_rs_end - self._t_start) * 1000.0
            enqueue_ms = (self._t_enqueued - self._t_start) * 1000.0
            wait_ms = (t_rs_end - self._t_enqueued) * 1000.0
            _accumulate_reduce_scatter_time_ms(total_ms)
            _record_rs_detail(self._nbytes, total_ms, enqueue_ms, wait_ms)
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

    rs_nbytes = tensor_partition_flat_buffer.numel() * tensor_partition_flat_buffer.element_size()
    return _DeferredRsResult(rs_handle, output_flat, output_lst,
                             t_rs_start, t_rs_enqueued, rs_nbytes)

    '''
    #大きな入力バッファを rank 数で等分（world_sz 個の連続ブロックに分割, 各チャンクの中身はrankiが担当してreduce_scatterするもの全体）。
    #これは 出力バッファ群としても扱う（reduce-scatter の各ランク受け取り先をここから選ぶ）。
    tensor_partition_buffer_for_each_rank: List[Tensor] = torch.chunk(tensor_partition_flat_buffer, world_sz)
    _torch_reduce_scatter_fn(tensor_partition_flat_buffer, tensor_partition_buffer_for_each_rank[this_rank], group=group)
    
    output_lst: List[Tensor] = [None] * len(tensors)
    offset = 0
    for tensor_idx in range(len(tensors)):
        #各tensorのうち自分の担当テンソル分だけ実データのスライスを取り出してくる
        output_lst[tensor_idx] = tensor_partition_buffer_for_each_rank[this_rank].narrow(
            0, offset, partition_lst_for_each_tensor[tensor_idx][this_rank].numel())
        #次のテンソルに移るため、オフセットをパディング込みの長さぶんだけ進める, outputlistは連続領域としておく
        offset += padded_partition_sz_for_each_tensor[tensor_idx]

    return output_lst
    '''

    
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
                 #reduce_bucket_size=500000000,
                 reduce_bucket_size=0,
                 #prefetch_bucket_size=50000000,
                 prefetch_bucket_size=0,
                 #max_reuse_distance=1000000000,
                 max_reuse_distance=0,
                 #max_live_parameters=1000000000,
                 max_live_parameters=0,
                 dp_process_group=None,
                 reduce_scatter=True,
                 overlap_comm=True,
                 #sub_group_size=1000000000000,
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
            '''
            aggregate_param_count += params_in_group
            if flat_buffer_size > 0:
                print_rank_0(f"group {j} flat buffer size {flat_buffer_size}", force=False)
                self.param_groups_fp16_flat_cpu_memory.append(
                    torch.empty(int(flat_buffer_size),
                                dtype=self.dtype,
                                device=self.local_device)) #GPU memoryに載せる
            '''
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
                #self._move_to_flat_buffer(sub_group,
                #                              fp16_partitioned_group_flat, True)
        
        
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
                _rs_nccl = os.environ.get("RS_USE_NCCL", "0") == "1"
                if _rs_nccl:
                    d2h_start = torch.cuda.Event(enable_timing=True)
                    d2h_end = torch.cuda.Event(enable_timing=True)
                    d2h_start.record()
                self.__partition_grads(prev_params, grad_partitions)
                if _rs_nccl:
                    d2h_end.record()
                    d2h_end.synchronize()
                    _accumulate_grad_offload_d2h_time_ms(d2h_start.elapsed_time(d2h_end))
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

            # D2H 転送が必要か判定 (NCCL RS: grad_partition=GPU, grad_buffer=CPU)
            _is_d2h = (not same_buffer and grad_partition.is_cuda
                        and not grad_buffer.is_cuda)

            if same_buffer:
                pass  # RS出力が直接grad_bufferに書き込まれている: コピー不要
            elif self.micro_step_id == 0:
                # D2H の時間は NVTX 区間 'xfer:grad_d2h' の memcpy 射影 (nsys) で取得する。
                _xfer_push("xfer:grad_d2h")
                try:
                    grad_buffer.copy_(grad_partition, non_blocking=True)
                finally:
                    _xfer_pop()
            else:
                if grad_buffer.device != grad_partition.device:
                    cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
                    cuda_grad_buffer.add_(grad_partition)
                    _xfer_push("xfer:grad_d2h")
                    try:
                        grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
                    finally:
                        _xfer_pop()
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
    def _flush_pending_rs(self):
        """Flush any pending deferred RS (wait + partition grads)."""
        if hasattr(self, '_pending_rs') and self._pending_rs is not None:
            prev_deferred, prev_params = self._pending_rs
            with torch.cuda.stream(self.__reduce_and_partition_stream):
                grad_partitions = prev_deferred.wait()
                if prev_deferred._needs_dtype_convert:
                    grad_partitions = [g.to(prev_deferred._target_dtype) for g in grad_partitions]
                _rs_nccl = os.environ.get("RS_USE_NCCL", "0") == "1"
                if _rs_nccl:
                    d2h_start = torch.cuda.Event(enable_timing=True)
                    d2h_end = torch.cuda.Event(enable_timing=True)
                    d2h_start.record()
                self.__partition_grads(prev_params, grad_partitions)
                if _rs_nccl:
                    d2h_end.record()
                    d2h_end.synchronize()
                    _accumulate_grad_offload_d2h_time_ms(d2h_start.elapsed_time(d2h_end))
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
        self._flush_pending_rs()  # optimizer step 前に全 RS 完了を保証
        self._pre_step()
        self._partition_all_parameters()

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