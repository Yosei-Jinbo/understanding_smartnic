import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import logger

import math
import types
from typing import Callable, Iterable
from enum import Enum
import functools
import itertools
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter
import torch.distributed as dist
from stage3_utils import * #parameterの持ち方が違うため独自のmemory_usage関数を呼ぶ
import threading
import time

# ---- Completion poller gate (CUPTI/nsys との衝突回避用) ----
# DISABLE_COMPLETION_POLLER=1 で _CompletionPoller(DOCA)の使用を抑止する。
# _CompletionPoller / _NcclCompletionPoller は別スレッドから CUDA/DOCA API を
# 高頻度に叩くため、nsys の CUPTI hook と干渉して SIGSEGV を起こすことがある。
# 無効化すると block_ms / dpu_ms の detailed 統計が取れなくなるが、
# 学習の正しさと NVTX/CUDA Event 計測には影響しない。
# 注: _DmaCompletionPoller は ctypes 直読みなので CUPTI と衝突せず、gate 対象外。
_DISABLE_COMPLETION_POLLER = os.environ.get("DISABLE_COMPLETION_POLLER", "0") == "1"

# SMARTNIC_L2_PREWARM=1 で AG 完了後に param.data を一度 read する kernel を挟む。
# DOCA GPUDirect RDMA は L2 bypass で DRAM 直書きするため、後続 compute kernel が
# L2 miss を起こす仮説の検証用。NCCL の ring-AG kernel が実行する staging read と
# 等価な動作を挿入して、compute 時間が NCCL 配置と揃うかを確認する。
_SMARTNIC_L2_PREWARM = os.environ.get("SMARTNIC_L2_PREWARM", "0") == "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
COMCH_HOST_DIR = THIS_DIR / "comch_mpi" / "host"
sys.path.insert(0, str(COMCH_HOST_DIR))

import doca_comch_client_pybind
from doca_comch_client_pybind import CollectiveCommunication

global_all_gather_id = 0
global_reduce_scatter_id = 0

# ds_id → param info table for correlating DPU-side rank skew records with parameter names.
_ds_id_info = {}

def register_ag_param_names(model):
    """モデルを traverse して named_parameters の name を _ds_id_info に登録する。
    Init() で ds_id が付与された後に呼ぶこと。"""
    import torch.nn as _nn
    if not isinstance(model, _nn.Module):
        return
    for name, param in model.named_parameters():
        ds_id = getattr(param, 'ds_id', None)
        if ds_id is not None and ds_id in _ds_id_info:
            _ds_id_info[ds_id]['name'] = name

def dump_ds_id_info(filepath):
    """_ds_id_info を tab 区切りファイルに出力 (rank 0 のみ呼ぶこと)。"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    with open(filepath, 'w') as f:
        f.write("# ds_id → param info dump (P2: Phase 13-A outlier 解析用)\n")
        f.write(f"# total params: {len(_ds_id_info)}\n")
        f.write("# format: ds_id\tname\tshape\tnumel\tdtype\n")
        for ds_id in sorted(_ds_id_info.keys()):
            info = _ds_id_info[ds_id]
            f.write(f"{ds_id}\t{info.get('name', '<unknown>')}\t{info.get('shape', ())}\t{info.get('numel', 0)}\t{info.get('dtype', '?')}\n")
    print(f"[P2] Dumped {len(_ds_id_info)} ds_id entries to {filepath}")

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

# ------------------------------------------------------------
# CPU<->GPU copy time accounting (CUDA event based)
#   - We accumulate per-rank GPU time (ms) for:
#       * param shard CPU->GPU (H2D) before all_gather
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Communication time accounting (CUDA event based)
#   - We accumulate per-rank GPU time (ms) for:
#       * parameter all_gather (weight materialization)
# ------------------------------------------------------------
ALL_GATHER_CALLS: int = 0
ALL_GATHER_DPU_MS: float = 0.0       # DPU処理時間 (キュー待ち除去) の累計
ALL_GATHER_BLOCK_MS: float = 0.0      # ブロック時間 (wait_start〜wait_end) の累計
ALL_GATHER_BYTES: int = 0             # 転送バイト数の累計
ALL_GATHER_LAST_COMPLETE: float = 0.0 # 直前のAGの t_complete (キュー待ち除去用)
ALL_GATHER_LOCK = threading.Lock()

def _accumulate_all_gather_detailed(t_request: float, t_complete: float, block_ms: float, nbytes: int = 0) -> None:
    global ALL_GATHER_CALLS, ALL_GATHER_DPU_MS, ALL_GATHER_BLOCK_MS, ALL_GATHER_LAST_COMPLETE, ALL_GATHER_BYTES
    with ALL_GATHER_LOCK:
        # DPU処理時間 = t_complete - max(t_request, 直前のAG完了時刻)
        # キュー待ち (直前AGの処理が終わるまで待つ時間) を除去した純粋な処理時間
        dpu_start = max(t_request, ALL_GATHER_LAST_COMPLETE)
        dpu_ms = (t_complete - dpu_start) * 1000.0
        if dpu_ms < 0:
            dpu_ms = 0.0
        ALL_GATHER_DPU_MS += dpu_ms
        ALL_GATHER_BLOCK_MS += float(block_ms)
        ALL_GATHER_BYTES += nbytes
        ALL_GATHER_LAST_COMPLETE = t_complete
        ALL_GATHER_CALLS += 1

        # デバッグ: 各AGの詳細を出力
        #print(f"[AG#{ALL_GATHER_CALLS}] t_request={t_request:.6f} t_complete={t_complete:.6f} "
        #      f"dpu_start={dpu_start:.6f} dpu_ms={dpu_ms:.3f} block_ms={block_ms:.3f} "
        #      f"cum_dpu={ALL_GATHER_DPU_MS:.3f} cum_block={ALL_GATHER_BLOCK_MS:.3f}", flush=True)

def get_all_gather_calls() -> int:
    return int(ALL_GATHER_CALLS)

def get_all_gather_dpu_ms() -> float:
    return float(ALL_GATHER_DPU_MS)

def get_all_gather_block_ms() -> float:
    return float(ALL_GATHER_BLOCK_MS)

def get_all_gather_bytes() -> int:
    return int(ALL_GATHER_BYTES)

# ---- Forward / Backward AG block tracking ----
_AG_PHASE = "unknown"  # "forward" or "backward"
_AG_FWD_BLOCK_MS: float = 0.0
_AG_BWD_BLOCK_MS: float = 0.0
_AG_FWD_CALLS: int = 0
_AG_BWD_CALLS: int = 0

def set_ag_phase(phase: str) -> None:
    global _AG_PHASE
    _AG_PHASE = phase

def _accumulate_phase_block(block_ms: float) -> None:
    global _AG_FWD_BLOCK_MS, _AG_BWD_BLOCK_MS, _AG_FWD_CALLS, _AG_BWD_CALLS
    if _AG_PHASE == "forward":
        _AG_FWD_BLOCK_MS += block_ms
        _AG_FWD_CALLS += 1
    elif _AG_PHASE == "backward":
        _AG_BWD_BLOCK_MS += block_ms
        _AG_BWD_CALLS += 1

def get_ag_phase_stats() -> dict:
    return {
        "fwd_block_ms": _AG_FWD_BLOCK_MS, "fwd_calls": _AG_FWD_CALLS,
        "bwd_block_ms": _AG_BWD_BLOCK_MS, "bwd_calls": _AG_BWD_CALLS,
    }

def reset_ag_phase_stats() -> None:
    global _AG_FWD_BLOCK_MS, _AG_BWD_BLOCK_MS, _AG_FWD_CALLS, _AG_BWD_CALLS
    _AG_FWD_BLOCK_MS = _AG_BWD_BLOCK_MS = 0.0
    _AG_FWD_CALLS = _AG_BWD_CALLS = 0

# ---- Phase 20: per-AG/RS event-bracket stall tracker (両 mode 対称な真の stall 計測) ----
# tracker への参照は遅延 import (importlib loop 回避)。env gate (`MEASURE_AG_STALL=1`) は
# common/stall_event_tracker.py 側で判定する。
def _stall_bracket(wait_callable, params_list=None, op: str = "ag", num_bytes: int = 0):
    """Bracket the actual stream wait op with CUDA events for Phase 20 measurement.

    両 mode 対称: wait_callable() は内部で compute stream に sync op
    (cudaStreamWaitEvent / cuStreamWaitValue32) を enqueue する。その前後を
    CUDA event ペアで囲み、stream 上の真の stall ms を取得する。
    """
    try:
        from common import stall_event_tracker as _set  # type: ignore
    except Exception:
        return wait_callable()
    if not _set.is_enabled():
        return wait_callable()
    tracker = _set.get_global()
    if params_list:
        try:
            nbytes = sum(p.ds_numel * p.element_size() for p in params_list)
        except Exception:
            nbytes = num_bytes
        ds_id = getattr(params_list[0], "ds_id", -1) if params_list else -1
    else:
        nbytes = num_bytes
        ds_id = -1
    stream = torch.cuda.current_stream()
    handle = tracker.begin(stream, _AG_PHASE, op=op, ds_id=ds_id, payload_bytes=nbytes)
    try:
        return wait_callable()
    finally:
        tracker.end(handle)


# ---- Per-AG detailed records for size-based analysis ----
_AG_RECORDS: list = []   # [(nbytes, dpu_ms, block_ms, enqueue_ms, prefetch_lead_ms, wall_ms)]
_AG_RECORDS_LOCK = threading.Lock()

def _record_ag_detail(nbytes: int, dpu_ms: float, block_ms: float,
                      enqueue_ms: float = 0.0, prefetch_lead_ms: float = 0.0,
                      wall_ms: float = 0.0) -> None:
    with _AG_RECORDS_LOCK:
        _AG_RECORDS.append((nbytes, dpu_ms, block_ms, enqueue_ms, prefetch_lead_ms, wall_ms))

def print_ag_analysis(epoch: int) -> None:
    """Print per-AG size-bucket analysis at epoch end."""
    with _AG_RECORDS_LOCK:
        records = list(_AG_RECORDS)
    if not records:
        return

    import statistics
    # Size buckets: <1KB, 1-16KB, 16KB-1MB, 1-16MB, >16MB
    buckets = [
        ("<1KB",   0,        1024),
        ("1-16KB", 1024,     16384),
        ("16K-1M", 16384,    1048576),
        ("1-16MB", 1048576,  16777216),
        (">16MB",  16777216, float('inf')),
    ]

    print(f"========== Per-AG Analysis (Epoch {epoch}) ==========", flush=True)
    print(f"  AG calls: {len(records)}", flush=True)

    for label, lo, hi in buckets:
        group = [(nb, dpu, blk, enq, pfl, wl) for nb, dpu, blk, enq, pfl, wl in records if lo <= nb < hi]
        if not group:
            continue
        n = len(group)
        avg_dpu = sum(d for _, d, _, _, _, _ in group) / n
        avg_blk = sum(b for _, _, b, _, _, _ in group) / n
        avg_enq = sum(e for _, _, _, e, _, _ in group) / n
        avg_pfl = sum(p for _, _, _, _, p, _ in group) / n
        avg_wl  = sum(w for _, _, _, _, _, w in group) / n
        avg_sz  = sum(s for s, _, _, _, _, _ in group) / n
        p99_blk = sorted(b for _, _, b, _, _, _ in group)[int(n * 0.99)] if n > 1 else avg_blk
        p99_wl  = sorted(w for _, _, _, _, _, w in group)[int(n * 0.99)] if n > 1 else avg_wl
        print(f"  {label:>7s}: calls={n:>6d} | avg_size={avg_sz/1024:.1f}KB | "
              f"dpu={avg_dpu:.3f}ms | block={avg_blk:.3f}ms (p99={p99_blk:.3f}ms) | "
              f"wall={avg_wl:.3f}ms (p99={p99_wl:.3f}ms) | "
              f"prefetch_lead={avg_pfl:.1f}ms", flush=True)

    # Top 10 blocking AGs
    top_block = sorted(records, key=lambda r: r[2], reverse=True)[:10]
    print(f"  Top 10 blocking AGs:", flush=True)
    for nb, dpu, blk, enq, pfl, wl in top_block:
        print(f"    size={nb/1024:.1f}KB | dpu={dpu:.3f}ms | block={blk:.3f}ms | wall={wl:.3f}ms | prefetch_lead={pfl:.1f}ms", flush=True)
    print(f"{'=' * 55}", flush=True)

def reset_ag_records() -> None:
    with _AG_RECORDS_LOCK:
        _AG_RECORDS.clear()

FULL_PARAMETER_CALLS: int = 0
FULL_PARAMETER_BYTES: int = 0
FULL_PARAMETER_DMA_POLL_MS: float = 0.0  # DMA ポーリングによる合計時間
FULL_PARAMETER_PENDING_EVENTS: list = []  # [(start_event, end_event), ...]
FULL_PARAMETER_LOCK = threading.Lock()

def _record_full_parameter_events(start_event: torch.cuda.Event, end_event: torch.cuda.Event, nbytes: int = 0, dma_poll_ms: float = 0.0) -> None:
    global FULL_PARAMETER_CALLS, FULL_PARAMETER_BYTES, FULL_PARAMETER_DMA_POLL_MS
    with FULL_PARAMETER_LOCK:
        FULL_PARAMETER_PENDING_EVENTS.append((start_event, end_event))
        FULL_PARAMETER_CALLS += 1
        FULL_PARAMETER_BYTES += nbytes
        FULL_PARAMETER_DMA_POLL_MS += dma_poll_ms

def get_full_parameter_copy_ms() -> float:
    """エポック末に呼ばれる。未処理の CUDA event ペアを同期して elapsed_time を合計する。
    1エポックに1回の同期で済むため、per-call の同期オーバーヘッドがない。"""
    with FULL_PARAMETER_LOCK:
        if not FULL_PARAMETER_PENDING_EVENTS:
            return 0.0
        # 最後のイベントだけ同期すれば、それ以前の全イベントも完了している
        FULL_PARAMETER_PENDING_EVENTS[-1][1].synchronize()
        total_ms = 0.0
        for start_ev, end_ev in FULL_PARAMETER_PENDING_EVENTS:
            total_ms += start_ev.elapsed_time(end_ev)
        FULL_PARAMETER_PENDING_EVENTS.clear()
        return total_ms

def get_full_parameter_calls() -> int:
    return int(FULL_PARAMETER_CALLS)

def get_full_parameter_bytes() -> int:
    return int(FULL_PARAMETER_BYTES)

def get_full_parameter_dma_poll_ms() -> float:
    return float(FULL_PARAMETER_DMA_POLL_MS)

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

    Phase 14: gpu_flag_addr/gpu_flag_value が指定されると、wait() は
      1. cuStreamWaitValue32 を current stream に schedule (Python はブロックしない)
      2. comch_req_release_py で C 側 handle を解放 (DPU 側 ref はそのまま)
    の流れに切り替わる。Python は wait() でブロックしないので AG が compute と
    完全にオーバーラップする (NCCL と同じ semantics)。

    旧パス (gpu_flag_addr=None) は従来の comch_req_wait_py + release を使う。
    """
    def __init__(self, handle: int,
                 gpu_flag_addr: int = 0,
                 gpu_flag_value: int = 0,
                 gpu_flag_device: "torch.device | None" = None) -> None:
        self._handle = int(handle)
        self._released = False
        self._completed = False
        self._gpu_flag_addr = int(gpu_flag_addr)
        self._gpu_flag_value = int(gpu_flag_value)
        self._gpu_flag_device = gpu_flag_device
        self._gpu_flag_scheduled = False

    def use_gpu_flag(self) -> bool:
        return self._gpu_flag_addr != 0

    def wait(self):
        if self.use_gpu_flag():
            # Phase 14 path: schedule cuStreamWaitValue and return immediately
            if not self._gpu_flag_scheduled:
                from cuda_stream_wait import stream_wait_value_eq
                stream = torch.cuda.current_stream(self._gpu_flag_device)
                stream_wait_value_eq(stream, self._gpu_flag_addr, self._gpu_flag_value)
                self._gpu_flag_scheduled = True
            # Mark Python-side completed (the actual GPU sync happens via the stream)
            self._completed = True
            if not self._released:
                # Release the caller's refcnt; C-side completion thread will free
                # the resources when DPU writes the doorbell.
                doca_comch_client_pybind.comch_req_release_py(self._handle)
                self._released = True
            return None

        # Legacy path
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


class _AllGatherWork:
    """DOCA ALL_GATHER(リスト版) 用の Work ハンドル（PyTorch の Work 風インターフェース）"""

    def __init__(
        self,
        thread: threading.Thread,
        output_flat: torch.Tensor,
        output_tensors: List[torch.Tensor],
        like_tensor: torch.Tensor,
    ) -> None:
        self._thread = thread
        self._output_flat = output_flat
        self._output_tensors = output_tensors
        self._like_tensor = like_tensor
        self._completed = False

    def wait(self):
        if not self._completed:
            self._thread.join()
            _scatter_flat_to_list(self._output_flat, self._output_tensors, self._like_tensor)
            self._completed = True
        # PyTorch collective の仕様に合わせて None を返す
        return None

    def is_completed(self) -> bool:
        if self._completed:
            return True
        return not self._thread.is_alive()


class _ReduceScatterWork:
    """DOCA REDUCE_SCATTER(リスト版) 用の Work ハンドル"""

    def __init__(
        self,
        thread: threading.Thread,
        output_flat: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> None:
        self._thread = thread
        self._output_flat = output_flat
        self._output_tensor = output_tensor
        self._completed = False

    def wait(self):
        if not self._completed:
            self._thread.join()
            self._output_tensor.copy_(self._output_flat.view_as(self._output_tensor))
            self._completed = True
        return None

    def is_completed(self) -> bool:
        if self._completed:
            return True
        return not self._thread.is_alive()


# =========================
# リスト版 AllGather
# =========================
def all_gather_list_via_base(
    output_tensors: List[torch.Tensor],
    input_tensor: torch.Tensor,
    cid: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    world_size = dist.get_world_size()

    if len(output_tensors) != world_size:
        raise ValueError(f"len(output_tensors)={len(output_tensors)} must equal world_size={world_size}")

    '''
    for i, out in enumerate(output_tensors):
        if out.numel() != input_tensor.numel():
            raise ValueError(f"output_tensors[{i}].numel()={out.numel()} must equal input_tensor.numel()={input_tensor.numel()}")
        if out.dtype != input_tensor.dtype:
            raise TypeError("output_tensors[i].dtype must match input_tensor.dtype")
        #if out.device != input_tensor.device:
        #    raise TypeError("output_tensors[i].device must match input_tensor.device")
    '''

    input_flat = input_tensor.contiguous().view(-1)
    output_flat = torch.empty(
        world_size * input_flat.numel(),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    ).contiguous()

    # All paths go through enqueue to avoid ComCh contention with worker thread
    handle = doca_comch_client_pybind.ucp_collective_enqueue_py(
        cid, input_flat, output_flat, CollectiveCommunication.ALL_GATHER
    )

    class _AllGatherHandleWork(_ComchHandleWork):
        def __init__(self, h, out_flat, out_list, like):
            super().__init__(h)
            self._out_flat = out_flat
            self._out_list = out_list
            self._like = like

        def wait(self):
            super().wait()
            _scatter_flat_to_list(self._out_flat, self._out_list, self._like)
            return None

    if not async_op:
        w = _AllGatherHandleWork(handle, output_flat, output_tensors, input_tensor)
        w.wait()
        return None
    else:
        return _AllGatherHandleWork(handle, output_flat, output_tensors, input_tensor)
    
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
        #if tin.device != output_tensor.device:
        #    raise TypeError("input_tensors[i].device must match output_tensor.device")
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
# フラット版 AllGather
# =========================
def all_gather_flat_via_base(
    output_flat: torch.Tensor,
    input_flat: torch.Tensor,
    cid: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    world_size = dist.get_world_size()

    in_view = input_flat.contiguous().view(-1)
    out_view = output_flat.contiguous().view(-1)
    '''
    if out_view.numel() != in_view.numel() * world_size:
        raise ValueError("output_flat.numel() must be input_flat.numel() * world_size")
    if out_view.dtype != in_view.dtype:
        raise TypeError("output_flat.dtype must match input_flat.dtype")
    #if out_view.device != in_view.device:
    #    raise TypeError("output_flat.device must match input_flat.device")]
    '''

    #if dist.get_rank() == 0:
        #print(f"all_gather_flat_via_base: numel={in_view.numel() * 2} byte")

    # Phase 14: GPU flag pool が enable なら slot を取得して flag-aware enqueue を使う
    try:
        import gpu_flag_pool as _gfp
    except ImportError:
        _gfp = None

    if _gfp is not None and _gfp.is_enabled():
        pool = _gfp.get_global_pool()
        slot_id, gen, gpu_addr, expected = pool.acquire()
        handle = doca_comch_client_pybind.ucp_collective_enqueue_with_flag_py(
            cid, in_view, out_view, CollectiveCommunication.ALL_GATHER,
            gpu_addr, expected
        )
        work = _ComchHandleWork(handle,
                                gpu_flag_addr=gpu_addr,
                                gpu_flag_value=expected,
                                gpu_flag_device=out_view.device)
    else:
        handle = doca_comch_client_pybind.ucp_collective_enqueue_py(
            cid, in_view, out_view, CollectiveCommunication.ALL_GATHER
        )
        work = _ComchHandleWork(handle)

    if not async_op:
        work.wait()
        return None
    else:
        return work

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
    #if out_view.device != in_view.device:
    #    raise TypeError("output_flat.device must match input_flat.device")
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
    
def _dist_allgather_fn(
    input_flat: torch.Tensor,
    output_flat: torch.Tensor,
    cid: int,
    group=None,
):
    """
    ZeRO 内部で使う flat all_gather のラッパ。
    - input_flat : 各 rank の shard (1D)
    - output_flat: [world_size * shard] (1D)
    - cid        : collective を識別する一意 ID（param.ds_id など）
    """

    return all_gather_flat_via_base(
        output_flat=output_flat,
        input_flat=input_flat,
        cid=cid,
        group=group,
        async_op=True,  # ← 元コードどおり async にしておく
    )
    

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


'''
with InsertPostInitMethodToModuleSubClasses(...): のブロックに入っている間だけ、
すべての torch.nn.Module サブクラスの __init__ 振る舞いを差し替えて、
**初期化直後に ZeRO-3 向けの後処理（post-init：分割/収集の管理など）**を自動で挟み込みます。
メモリ効率の良いテンソル生成・Linear 演算への差し替えも同時に行います。
'''
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
                    """gathers parameters before calling apply function. afterwards
                    parameters are broadcasted to ensure consistency across all ranks
                    then re-partitioned.

                    takes the following steps:
                    1. allgathers parameters for the current module being worked on
                    2. calls the original function
                    3. broadcasts root rank's parameters to the other ranks
                    4. re-partitions the parameters
                    """
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

                # important logic: We want to run post_init only after child's __init__ is
                # completed, and do nothing after __init__ of any of its parents and grandparents in
                # the inheritance ancestry. This way the partitioning will need to happen only once
                # when the whole object is ready to be partitioned and not before. This is because
                # often the child module will need to tweak the weights - for example running a
                # custom weights init function. So if a parent created the weights param, the child
                # won't need to gather it in order to tweak it

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
            # print(f"subclass={subclass.__module__}.{subclass.__qualname__}")
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

    # un doing it here will undo it during training
    # if self.mem_efficient_linear:
    #    torch.nn.functional.linear = self.linear_bk
    #        if self.mem_efficient_linear:
    #            torch.nn.functional.linear = self.linear_bk

    zero_init_enabled = False

class _CompletionPoller:
    """バックグラウンドスレッドで is_completed() をポーリングし、
    DPU 完了通知の到着時刻 t_complete を記録する。
    100μs 間隔の sleep で GIL 負荷を最小化。"""

    _POLL_INTERVAL = 0.0001  # 100μs

    def __init__(self, handle):
        self._handle = handle
        self.t_complete = None
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        # _completed を変更しないよう comch_req_test_py を直接呼ぶ
        # (is_completed() を使うと _completed=True が設定され、
        #  メインスレッドの wait() が comch_req_wait_py をスキップしてしまう)
        handle_id = self._handle._handle
        while not doca_comch_client_pybind.comch_req_test_py(handle_id):
            time.sleep(self._POLL_INTERVAL)
        self.t_complete = time.perf_counter()

    def join(self):
        self._thread.join()


class _DmaCompletionPoller:
    """GPU→CPU pinned メモリへのフラグ書き込みをポーリングし、
    DMA コピー完了時刻を検知する。

    使い方: コピー投入後に record() を呼ぶ。
    record() は同一ストリーム上で GPU フラグに 1 を書き → pinned CPU にコピー。
    ポーリングスレッドが pinned メモリの値変化を検知して t_complete を記録。"""

    _POLL_INTERVAL = 0.00005  # 50μs

    def __init__(self, device):
        import ctypes
        self._flag_cpu = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._flag_ptr = ctypes.cast(self._flag_cpu.data_ptr(), ctypes.POINTER(ctypes.c_int32))
        self._device = device
        self.t_complete = None
        self._thread = None

    def record(self):
        """現在のストリームにフラグ書き込みを投入し、ポーリングを開始する。"""
        # 同一ストリーム上で: 先行する全コピーが完了 → GPU フラグ = 1 → CPU にコピー
        gpu_flag = torch.ones(1, dtype=torch.int32, device=self._device)
        self._flag_cpu.copy_(gpu_flag, non_blocking=True)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._flag_ptr[0] == 0:
            time.sleep(self._POLL_INTERVAL)
        self.t_complete = time.perf_counter()

    def join(self):
        if self._thread is not None:
            self._thread.join()


'''waitはpartitioned_param_coordinator.pyで呼ばれてます'''
class AllGatherHandle:
    def __init__(self, handle, param: Parameter, t_request=None, start_event=None) -> None:
        if param.ds_status != ZeroParamStatus.INFLIGHT:
            raise RuntimeError(f"expected param {param.ds_summary()} to be available")
        self.__handle = handle
        self.__param = param
        self.__t_request = t_request
        self.__use_poller = (
            t_request is not None
            and hasattr(handle, 'is_completed')
            and not _DISABLE_COMPLETION_POLLER
        )
        self.__use_gpu_flag = hasattr(handle, 'use_gpu_flag') and handle.use_gpu_flag()

    def wait(self) -> None:
        t_wait_start = time.perf_counter()

        if self.__use_gpu_flag:
            # Phase 14: schedule cuStreamWaitValue + return immediately.
            # Python は一切ブロックしない (block_ms = 0 として記録)。
            # Phase 20: bracket で compute stream 上の真の stall を測る
            _stall_bracket(instrument_w_nvtx(self.__handle.wait), [self.__param], op="ag")
            t_after = time.perf_counter()
            block_ms = (t_after - t_wait_start) * 1000.0  # この値は scheduling overhead のみ (~μs)
            if self.__t_request is not None:
                nbytes = self.__param.ds_numel * self.__param.element_size()
                prefetch_lead_ms = (t_wait_start - self.__t_request) * 1000.0
                wall_ms = (t_after - self.__t_request) * 1000.0
                _accumulate_all_gather_detailed(self.__t_request, t_after, block_ms, nbytes)
                _record_ag_detail(nbytes, 0.0, block_ms, 0.0, prefetch_lead_ms, wall_ms)
                _accumulate_phase_block(block_ms)
            self.__param.ds_status = ZeroParamStatus.AVAILABLE
            return

        if self.__use_poller:
            poller = _CompletionPoller(self.__handle)
            poller.join()
            t_complete = poller.t_complete or time.perf_counter()
            block_ms = (t_complete - t_wait_start) * 1000.0
            nbytes = self.__param.ds_numel * self.__param.element_size()
            dpu_start = max(self.__t_request, ALL_GATHER_LAST_COMPLETE)
            dpu_ms = (t_complete - dpu_start) * 1000.0
            prefetch_lead_ms = (t_wait_start - self.__t_request) * 1000.0
            wall_ms = (t_complete - self.__t_request) * 1000.0
            _accumulate_all_gather_detailed(self.__t_request, t_complete, block_ms, nbytes)
            _record_ag_detail(nbytes, max(0.0, dpu_ms), block_ms, 0.0, prefetch_lead_ms, wall_ms)
            _accumulate_phase_block(block_ms)

        # Phase 20: bracket the legacy / poller-completed wait path as well.
        # poller が完了してから handle.wait() が呼ばれるのでこれは release 中心の
        # 軽い処理だが、wait_op が再 enqueue されるケースもあるため対称に括る
        _stall_bracket(instrument_w_nvtx(self.__handle.wait), [self.__param], op="ag")
        self.__param.ds_status = ZeroParamStatus.AVAILABLE

class AllGatherCoalescedHandle:
    def __init__(self, allgather_handle, params: List[Parameter], partitions: List[Tensor], world_size: int, t_request=None, start_event=None) -> None:
        self.__allgather_handle = allgather_handle
        self.__params = params
        self.__partitions = partitions
        self.__world_size = world_size
        self.__t_request = t_request
        self.__complete = False
        self.__use_poller = (
            t_request is not None
            and hasattr(allgather_handle, 'is_completed')
            and not _DISABLE_COMPLETION_POLLER
        )
        self.__use_gpu_flag = hasattr(allgather_handle, 'use_gpu_flag') and allgather_handle.use_gpu_flag()
        for param in self.__params:
            if param.ds_status != ZeroParamStatus.INFLIGHT:
                raise RuntimeError(
                    f"expected param {param.ds_summary()} to not be available")

    @instrument_w_nvtx
    def wait(self) -> None:
        if self.__complete:
            return
        t_wait_start = time.perf_counter()

        if self.__use_gpu_flag:
            # Phase 14: schedule cuStreamWaitValue + return immediately
            # Phase 20: bracket で compute stream 上の真の stall を測る
            _stall_bracket(instrument_w_nvtx(self.__allgather_handle.wait),
                           list(self.__params), op="ag")
            t_after = time.perf_counter()
            block_ms = (t_after - t_wait_start) * 1000.0
            if self.__t_request is not None:
                nbytes = sum(p.ds_numel * p.element_size() for p in self.__params)
                prefetch_lead_ms = (t_wait_start - self.__t_request) * 1000.0
                wall_ms = (t_after - self.__t_request) * 1000.0
                _accumulate_all_gather_detailed(self.__t_request, t_after, block_ms, nbytes)
                _record_ag_detail(nbytes, 0.0, block_ms, 0.0, prefetch_lead_ms, wall_ms)
                _accumulate_phase_block(block_ms)
            self.__t_request = None
        elif self.__use_poller:
            poller = _CompletionPoller(self.__allgather_handle)
            poller.join()
            t_complete = poller.t_complete or time.perf_counter()
            block_ms = (t_complete - t_wait_start) * 1000.0
            nbytes = sum(p.ds_numel * p.element_size() for p in self.__params)
            dpu_start = max(self.__t_request, ALL_GATHER_LAST_COMPLETE)
            dpu_ms = (t_complete - dpu_start) * 1000.0
            prefetch_lead_ms = (t_wait_start - self.__t_request) * 1000.0
            wall_ms = (t_complete - self.__t_request) * 1000.0
            _accumulate_all_gather_detailed(self.__t_request, t_complete, block_ms, nbytes)
            _record_ag_detail(nbytes, max(0.0, dpu_ms), block_ms, 0.0, prefetch_lead_ms, wall_ms)
            _accumulate_phase_block(block_ms)
            self.__poller = None
            self.__t_request = None
            # release 処理 (ポーラー完了後なので安全)
            # Phase 20: 同じく bracket で計測 (poller 完了済みなので ev_ms ≈ 0 期待)
            _stall_bracket(instrument_w_nvtx(self.__allgather_handle.wait),
                           list(self.__params), op="ag")
        else:
            self.__t_request = None
            # release 処理 (ポーラー完了後なので安全)
            # Phase 20: bracket
            _stall_bracket(instrument_w_nvtx(self.__allgather_handle.wait),
                           list(self.__params), op="ag")

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

            if _SMARTNIC_L2_PREWARM:
                # L2 warming: NCCL staging kernel を mimic するため param.data を一度 read
                # compute stream 上で走る軽い reduction を挟んで L2 に乗せる
                torch.cuda.nvtx.range_push("l2_prewarm")
                _ = param.data.view(-1).sum()
                torch.cuda.nvtx.range_pop()

            from common import debug_params as dbg
            dbg.log_ag_complete(param, tag="AG_COMPLETE(coalesced)")

            for part_to_copy in partitions:
                part_to_copy.record_stream(torch.cuda.current_stream()) #現在ストリームでの使用が終わるまでバッファを生存させる（生存期間の正しい関連付け）。
            param_offset += param.ds_tensor.ds_numel

        self.__partitions = None   # flat_tensor への参照を解放し、GPU メモリの再利用を可能にする
        self.__complete = True

class CPUAllGatherCoalescedHandle:
    """cpu_full_param からのローカルコピーだけで all-gather を完了させるハンドル。

    - 通信は一切行わない
    - .wait() が呼ばれたタイミングで CPU→GPU コピーを行い、
      param.ds_status を AVAILABLE にする
    """

    def __init__(self, params: List[Parameter], device: torch.device, t_request=None, start_event=None) -> None:
        self.__params = list(params)
        self.__device = device
        self.__complete = False
        self.__t_request = t_request

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

        t_copy_start = time.perf_counter()

        # DMA 完了ポーラー準備
        dma_poller = _DmaCompletionPoller(self.__device)

        # CUDA event: コピー前に start を記録
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

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

        # CUDA event: コピー後に end を記録 (同期はエポック末に1回だけ)
        end_event.record()

        # メモリポーリング: コピー後に GPU→pinned CPU フラグ書き込みを投入
        dma_poller.record()
        dma_poller.join()
        t_complete = dma_poller.t_complete or time.perf_counter()
        dma_ms = (t_complete - t_copy_start) * 1000.0

        nbytes = sum(p.ds_numel * p.element_size() for p in self.__params)
        _record_full_parameter_events(start_event, end_event, nbytes, dma_ms)

        self.__complete = True

'''
これは DeepSpeed ZeRO-3 の 初期化コンテキスト (zero.Init) を実装したクラスです。
目的は「巨大モデルを効率的に初期化して、その場でパラメータを分割 (shard) する」ことです。
'''
# Replaces all parameters in module with Scattered Parameters
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
    
    #各“子モジュール直下（recurse=False）のパラメータを ZeRO-3 用の分割管理（Scattered/ZeroParam）に変換し、
    # rank0 の値でそろえてからシャーディングする処理です。
    # 実行は post_init タイミング（＝その子モジュールの __init__ が完全に終わった直後）で1回だけ走ります。
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
        '''
        ZeroParamStatus: 現在の可用状態。
        AVAILABLE: その rank に **完全復元（all-gather 済み）**で手元にある状態
        NOT_AVAILABLE: まだ完全には手元にない（分割の一部しかない/オフロード中など）
        INFLIGHT: 通信などで「復元中」
        '''
        '''
        GPU メモリに shard がないけど CPU にある、というケースについて
        param.ds_status（全体状態）は NOT_AVAILABLE
        → なぜなら「フルサイズのパラメータ」は CPU にも存在しない。持っているのは rank shard だけ。

        param.ds_tensor.status（shard 状態）は AVAILABLE
        → shard 自体は CPU メモリ上にあってアクセス可能。GPU 上にはないけど「NOT_AVAILABLE」ではない
        '''
         #パラメータ全体の可用状態。典型遷移は NOT_AVAILABLE（未復元/解放中）→ INFLIGHT（通信中）→ AVAILABLE（フル値が手元）。
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

        # P2: ds_id → param info マッピングに登録 (name は後で register_ag_param_names で埋める)
        if param.ds_id not in _ds_id_info:
            _ds_id_info[param.ds_id] = {
                'name': '<unnamed>',
                'shape': tuple(param.ds_shape),
                'numel': int(param.ds_numel),
                'dtype': str(param.dtype),
            }
        
        param.cpu_full_param = None #CPUに全てのパラメータを保持する
        param.cpu_full_param_pool = None #CPUのフルパラメータをプールしておく・ピン止め森のコストは高いのでこのfull_param_poolを利用してピン止めコストを減らす
        
        '''
        all_gather前後での引数paramについて
        all_gather 前: param.data → freeされているのでアクセス不可(param.ds_tensorに自分の担当分のみ保存されている)
        all_gather 後: param.data → 全 rank の shard を集めた「フルサイズ」テンソル。
        '''
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

            # ===== ここから CPU フルパラメータ優先パス =====
            # 環境変数 ENABLE_FULL_PARAM_TRANSFER=1 の場合のみ有効（デフォルト: 無効）
            # 全ての param が cpu_full_param を持っているなら、通信せずに CPU→GPU コピーだけを行う
            if os.environ.get("ENABLE_FULL_PARAM_TRANSFER", "0") == "1" and \
               all(getattr(p, "cpu_full_param", None) is not None for p in params):
                #print_rank_0(f"-allgather_coalesced (CPU full param path): {[p.ds_id for p in params]}", force=True)
                # Init.__init__ で定義した local_device を使って GPU に復元する
                t_request = time.perf_counter()
                return CPUAllGatherCoalescedHandle(params, self.local_device, t_request=t_request)
            
            #print_rank_0(f"-allgather_coaleseced: no cpu_data {[p.ds_id for p in params]}", force=True)
            
            if len(params) == 1:
                param, = params
                param_buffer = torch.empty(
                    math.ceil(param.ds_numel / self.world_size) * self.world_size,
                    dtype=param.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False
                ).contiguous() #paramが1個だけならバッファを一つだけ確保してそれで通信する
                #_dist_allgather_fn はdist.all_gatherを呼び出す。自ランクのシャード→全ランク分を param_buffer に all-gather。

                input_flat = param.ds_tensor.contiguous().view(-1)
                output_flat = param_buffer.contiguous().view(-1)

                t_request = time.perf_counter()

                # ★ DOCA 版 all_gather を async で起動
                handle = _dist_allgather_fn(
                    input_flat=input_flat,
                    output_flat=output_flat,
                    cid=param.ds_id,                 # ★ パラメータ固有 ID を collective ID に利用
                    group=self.ds_process_group,
                )

                #先頭から実長分だけ切り出し、元形状に view、元デバイスへ to。.data で param のストレージを差し替え(param_bufferに対する操作がparam.dataに対する操作になる)
                param.data = param_buffer.narrow(0, 0, param.ds_numel).view(param.ds_shape).to(param.device)
                return AllGatherHandle(handle, param, t_request=t_request) #このhandleが実行されるとparamにフルサイズの重みパラメータが入る
            else:
                #params に入っているのは「自分の rank が担当しているシャード（＝ds_tensor）を持っているパラメータだけ」
                partition_sz = sum(p.ds_tensor.ds_numel for p in params) #このランクが持つ全パラのシャード長の合計（連結後の 1ランク当たり長）
                #全ランクぶんの巨大フラット受け取りバッファを GPU に確保
                flat_tensor = torch.empty(partition_sz*self.world_size, dtype=get_only_unique_item(p.dtype for p in params), device=torch.cuda.current_device(), requires_grad=False).contiguous()
                flat_view = flat_tensor.view(-1)

                partitions: List[Parameter] = []
                for i in range(self.world_size):
                     #flat_tensor を rank ごとのスロットに narrow で切る（partitions[i] が rank i の連結結果置き場）
                    partitions.append(flat_tensor.narrow(0, partition_sz*i, partition_sz))

                # 自ランクが持つ各パラのシャードを順序通りに input_flat_cpu (pinned CPU) に詰めて DPU へ渡す
                input_flat_cpu = torch.empty(partition_sz, dtype=torch.float16, device="cpu", pin_memory=True).contiguous()
                offset = 0
                for p in params:
                    n = p.ds_tensor.ds_numel
                    input_flat_cpu.narrow(0, offset, n).copy_(p.ds_tensor.detach().view(-1))
                    offset += n

                # DOCA 版 all_gather を 1 回だけ発行する。coalesced 内の全パラを 1 collective に
                # 乗せるため cid は最小 ds_id を代表値として用いる。
                cid = min(p.ds_id for p in params)

                t_request = time.perf_counter()

                handle = _dist_allgather_fn(
                    #input_flat=partitions[self.rank],  # 各 rank の「自分のスロット」だけ送る
                    input_flat = input_flat_cpu,
                    output_flat=flat_view,            # 全 rank 分が入るフラットバッファ
                    cid=cid,
                    group=self.ds_process_group,
                )

                return AllGatherCoalescedHandle(
                    allgather_handle=handle,
                    params=params,
                    partitions=partitions,
                    world_size=self.world_size,
                    t_request=t_request
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
            '''
            すでに以前作った分割片（ds_tensor）が残っていて、かつ パラメータ値が直近で更新されていないなら、
            フルの param.data を解放（free_param）。
            ds_tensor はそのまま使えるので、ここで終了。
            つまり「シャードはもうある・中身も最新 → フル側だけ捨てればOK」という最短経路
            '''
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
        if dist.get_rank() == 0:
            pass
            #print(f"after _partition_param ID {param.ds_id} partitioned type {param.dtype} param.dev {param.device} param.ds_tensor.dev {param.ds_tensor.device} shape {param.shape}")
    
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
        torch.cuda.synchronize() #単一プロセスでの同期, GPU kernelが終わるまではCPU実行を止める (torch.distributed.barrier()は複数プロセスのバリア同期)
        print_rank_0(f"{'--'* hierarchy}----allgather param with {debug_param2name_id_shape_status(param)} partition size={partition_size}")
        '''
        _all_gather_base は PyTorch 1.8 以降で入った新しい実装で、従来の dist.all_gather と違い:
        出力を「リスト」ではなく 1つのフラットテンソルに直接書き込める。
        メモリアロケーションや Python ループを省けるので高速＆低オーバーヘッド
        '''
        '''
        handle = dist.all_gather_base(flat_tensor, param.ds_tensor.cuda(), group=self.ds_process_group, async_op=async_op)
        
        replicated_tensor = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape) #フルサイズのパラメータをreplicated_tensorに入れる
        param.data = replicated_tensor.data
        return handle
        '''
        # ===== ここを PyTorch → DOCA に差し替え =====
        # ds_tensor は remote_device 上にある可能性があるので、GPU へコピーして使う
        #input_tensor = param.ds_tensor.to(param.device)
        input_tensor = param.ds_tensor

        handle = all_gather_flat_via_base(
            output_flat=flat_tensor,
            input_flat=input_tensor,
            cid=param.ds_id,                # ★ param ごとの一意な ID を collective ID に利用
            group=self.ds_process_group,
            async_op=async_op,
        )

        # flat_tensor から実データ長だけ取り出して元 shape へ
        replicated_tensor = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape)
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
            local_tensors.append(param.ds_tensor)
        
        allgather_params = []
        for psize in partition_sizes:
            tensor_size = psize * self.world_size #フルサイズテンソルのサイズ
            flat_tensor = torch.empty(tensor_size, dtype=param_list[0].dtype, device=self.local_device).view(-1)
            flat_tensor.requires_grad = False
            allgather_params.append(flat_tensor)

        # ===== ここを dist.all_gather_base → DOCA all_gather_flat_via_base に変更 =====
        launch_handles = []
        for param_idx, param in enumerate(param_list):
            input_tensor = local_tensors[param_idx].view(-1)
            h = all_gather_flat_via_base(
                output_flat=allgather_params[param_idx],
                input_flat=input_tensor,
                cid=param.ds_id,
                group=self.ds_process_group,
                async_op=True,      # async_op=True のまま
            )
            launch_handles.append(h)
        #launch_handles[-1].wait()
        launch_handles[0].wait()

        for i, param in enumerate(param_list):
            gathered_tensor = allgather_params[i]
            param.data = gathered_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape).data

        torch.cuda.synchronize()
        return None
            
        '''
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
        '''
    
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
        #dist.all_gather(partitions, partitions[self.rank], group=self.ds_process_group, async_op=False)
        
        # ===== ここを PyTorch dist.all_gather → DOCA all_gather_list_via_base に変更 =====
        # collective ID は param_list[0].ds_id を代表として使う
        cid = param_list[0].ds_id

        all_gather_list_via_base(
            output_tensors=partitions,
            input_tensor=partitions[self.rank],
            cid=cid,
            group=self.ds_process_group,
            async_op=False,
        )
            
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
            #末尾以外の rank では、_reduce_scatter_gradient 内で 入力スライスを param.grad の該当ビューとして渡しているため、
            #**reduce_scatter の出力もそのビュー（= param.grad の一部）に“直書き”**される → コピー不要。
            #末尾 rank だけは パディング分の都合で “一時バッファ”を入力に使うため、出力もそこに書き込まれる → 後で有効要素だけ param.grad へ書き戻す。
            #start < param.ds_numel(全体サイズ) < end となっていると末尾が足りていない(paddingしたため)
            #reduced_partitionには[自分rankの集約済みgradient]のみが入っているので0スタートelements個だけcopyすればいい
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

        #このランクの開始位置が実データ範囲内なら、実データが存在する要素数 elements を決定
        # （末尾パディングがあると elements < partition_size）。
        #src_tensor＝フル勾配の該当スライス、dest_tensor＝出力先の該当スライス。
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