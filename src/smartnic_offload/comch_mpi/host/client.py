import os
import sys
import time
import argparse
import threading
from datetime import timedelta
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from mpi4py import MPI

import doca_comch_client_pybind
from doca_comch_client_pybind import CollectiveCommunication


# =============================================================================
# GPU "touch" utilities
# =============================================================================

def _cuda_touch_inplace(t: torch.Tensor, mode: str = "write_zero") -> None:
    """
    Force GPU memory to be actually backed/committed by issuing a real device op.

    mode:
      - "write_zero": t.zero_()   (writes all elements; most reliable)
      - "write_one":  t.fill_(1)
      - "read":       t.sum()     (reads; usually less reliable for commit than write)
    """
    if not t.is_cuda:
        return

    # Ensure contiguous to avoid surprises (not strictly necessary, but safer)
    if not t.is_contiguous():
        t = t.contiguous()

    if mode == "write_zero":
        t.zero_()
    elif mode == "write_one":
        t.fill_(1)
    elif mode == "read":
        _ = t.sum()
    else:
        raise ValueError(f"Unknown touch mode: {mode}")

    # Make sure the operation is completed before we proceed to ucp/DOCA calls
    torch.cuda.synchronize(t.device)


def _alloc_cuda_and_touch(shape, *, dtype, device, init: str) -> torch.Tensor:
    """
    Allocate CUDA tensor and touch immediately.

    init:
      - "zeros": allocate + zero touch
      - "empty_zero_touch": allocate empty then touch with zero_
      - "full": allocate and fill with constant (requires 'value' outside; use separately)
    """
    if init == "zeros":
        t = torch.zeros(shape, dtype=dtype, device=device)
        _cuda_touch_inplace(t, "write_zero")
        return t

    if init == "empty_zero_touch":
        t = torch.empty(shape, dtype=dtype, device=device)
        _cuda_touch_inplace(t, "write_zero")
        return t

    raise ValueError(f"Unknown init: {init}")


# =============================================================================
# Work handle wrappers (unchanged)
# =============================================================================

class _ComchHandleWork:
    """C 側の handle を保持して wait/test/release する Work。"""

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


# =============================================================================
# Collectives wrappers (touch added)
# =============================================================================

def all_gather_list_via_base(
    output_tensors: List[torch.Tensor],
    input_tensor: torch.Tensor,
    cid: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()

    if len(output_tensors) != world_size:
        raise ValueError(f"len(output_tensors)={len(output_tensors)} must equal world_size={world_size}")

    for i, out in enumerate(output_tensors):
        if out.numel() != input_tensor.numel():
            raise ValueError(
                f"output_tensors[{i}].numel()={out.numel()} must equal input_tensor.numel()={input_tensor.numel()}"
            )
        if out.dtype != input_tensor.dtype:
            raise TypeError("output_tensors[i].dtype must match input_tensor.dtype")
        if out.device != input_tensor.device:
            raise TypeError("output_tensors[i].device must match input_tensor.device")

    input_flat = input_tensor.contiguous().view(-1)
    output_flat = torch.empty(
        world_size * input_flat.numel(),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    ).contiguous()

    # ---- GPU touch (important for torch.empty) ----
    _cuda_touch_inplace(input_flat, "read")          # input is already filled usually; read touch is enough
    _cuda_touch_inplace(output_flat, "write_zero")   # output empty -> 반드시 write touch

    if not async_op:
        doca_comch_client_pybind.ucp_collective_request_py(
            cid, input_flat, output_flat, CollectiveCommunication.ALL_GATHER
        )
        _scatter_flat_to_list(output_flat, output_tensors, input_tensor)
        return None
    else:
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

        return _AllGatherHandleWork(handle, output_flat, output_tensors, input_tensor)


def reduce_scatter_list_via_base(
    output_tensor: torch.Tensor,
    input_tensors: List[torch.Tensor],
    cid: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()

    if len(input_tensors) != world_size:
        raise ValueError(f"len(input_tensors)={len(input_tensors)} must equal world_size={world_size}")

    out_numel = output_tensor.numel()
    for i, tin in enumerate(input_tensors):
        if tin.numel() != out_numel:
            raise ValueError(
                f"input_tensors[{i}].numel()={tin.numel()} must equal output_tensor.numel()={out_numel}"
            )
        if tin.dtype != output_tensor.dtype:
            raise TypeError("input_tensors[i].dtype must match output_tensor.dtype")
        if tin.device != output_tensor.device:
            raise TypeError("input_tensors[i].device must match output_tensor.device")

    flat_list = [t.contiguous().view(-1) for t in input_tensors]
    input_flat = torch.cat(flat_list, dim=0).contiguous()
    output_flat = torch.empty(out_numel, dtype=output_tensor.dtype, device=output_tensor.device).contiguous()

    # ---- GPU touch ----
    _cuda_touch_inplace(input_flat, "read")
    _cuda_touch_inplace(output_flat, "write_zero")

    if not async_op:
        doca_comch_client_pybind.ucp_collective_request_py(
            cid, input_flat, output_flat, CollectiveCommunication.REDUCE_SCATTER
        )
        output_tensor.copy_(output_flat.view_as(output_tensor))
        return None
    else:
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

        return _ReduceScatterHandleWork(handle, output_flat, output_tensor)


def all_gather_flat_via_base(
    output_flat: torch.Tensor,
    input_flat: torch.Tensor,
    cid: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()

    in_view = input_flat.view(-1)
    out_view = output_flat.view(-1)

    if out_view.numel() != in_view.numel() * world_size:
        raise ValueError("output_flat.numel() must be input_flat.numel() * world_size")
    if out_view.dtype != in_view.dtype:
        raise TypeError("output_flat.dtype must match input_flat.dtype")
    if out_view.device != in_view.device:
        raise TypeError("output_flat.device must match input_flat.device")

    # ---- GPU touch ----
    _cuda_touch_inplace(in_view, "read")
    _cuda_touch_inplace(out_view, "write_zero")

    if not async_op:
        doca_comch_client_pybind.ucp_collective_request_py(
            cid, in_view, out_view, CollectiveCommunication.ALL_GATHER
        )
        return None
    else:
        handle = doca_comch_client_pybind.ucp_collective_enqueue_py(
            cid, in_view, out_view, CollectiveCommunication.ALL_GATHER
        )
        return _ComchHandleWork(handle)


def reduce_scatter_flat_via_base(
    output_flat: torch.Tensor,
    input_flat: torch.Tensor,
    cid: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()

    out_view = output_flat.view(-1)
    in_view = input_flat.view(-1)

    if in_view.numel() != out_view.numel() * world_size:
        raise ValueError("input_flat.numel() must be output_flat.numel() * world_size")
    if out_view.dtype != in_view.dtype:
        raise TypeError("output_flat.dtype must match input_flat.dtype")
    if out_view.device != in_view.device:
        raise TypeError("output_flat.device must match input_flat.device")

    # ---- GPU touch ----
    _cuda_touch_inplace(in_view, "read")
    _cuda_touch_inplace(out_view, "write_zero")

    if not async_op:
        doca_comch_client_pybind.ucp_collective_request_py(
            cid, in_view, out_view, CollectiveCommunication.REDUCE_SCATTER
        )
        return None
    else:
        handle = doca_comch_client_pybind.ucp_collective_enqueue_py(
            cid, in_view, out_view, CollectiveCommunication.REDUCE_SCATTER
        )
        return _ComchHandleWork(handle)


def all_gather_cpu_full_flat_via_base(
    output_flat: torch.Tensor,
    input_flat: torch.Tensor,
    cpu_full_flat: torch.Tensor,
    cpu_full_flag: torch.Tensor,
    cid: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()

    in_view = input_flat.view(-1)
    out_view = output_flat.view(-1)
    flat_view = cpu_full_flat.view(-1)
    flag_view = cpu_full_flag.view(-1)

    if out_view.numel() != in_view.numel() * world_size:
        raise ValueError("output_flat.numel() must be input_flat.numel() * world_size")
    if out_view.dtype != in_view.dtype:
        raise TypeError("output_flat.dtype must match input_flat.dtype")

    # ---- GPU touch ----
    _cuda_touch_inplace(in_view, "read")
    _cuda_touch_inplace(out_view, "write_zero")

    if not async_op:
        doca_comch_client_pybind.ucp_collective_local_cpu_request_py(
            cid, in_view, out_view, cpu_full_flat, cpu_full_flag, CollectiveCommunication.ALL_GATHER
        )
        return None
    else:
        handle = doca_comch_client_pybind.ucp_collective_local_cpu_enqueue_py(
            cid, in_view, out_view, cpu_full_flat, cpu_full_flag, CollectiveCommunication.ALL_GATHER
        )
        return _ComchHandleWork(handle)
    
def all_gather_cpu_full_list_via_base(
    output_tensors: List[torch.Tensor],
    input_tensor: torch.Tensor,
    cpu_full_flat: torch.Tensor,
    cpu_full_flag: torch.Tensor,
    cid: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()

    flat_view = cpu_full_flat.view(-1)
    flag_view = cpu_full_flag.view(-1)

    input_flat = input_tensor.contiguous().view(-1)
    output_flat = torch.empty(
        world_size * input_flat.numel(),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    ).contiguous()

    # ---- GPU touch ----
    _cuda_touch_inplace(output_flat, "write_zero")

    if not async_op:
        doca_comch_client_pybind.ucp_collective_local_cpu_request_py(
            cid, input_flat, output_flat, cpu_full_flat, cpu_full_flag, CollectiveCommunication.ALL_GATHER
        )
        _scatter_flat_to_list(output_flat, output_tensors, input_tensor)
        return None
    else:
        handle = doca_comch_client_pybind.ucp_collective_local_cpu_enqueue_py(
            cid, input_flat, output_flat, cpu_full_flat, cpu_full_flag, CollectiveCommunication.ALL_GATHER
        )
        # wait() 完了時に scatter が必要なので、既存 Work と同様のラッパを作る
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

        return _AllGatherHandleWork(handle, output_flat, output_tensors, input_tensor)

# =============================================================================
# Benchmarks (DOCA)
# =============================================================================

def benchmark_doca_reduce_scatter_flat(
    device: torch.device,
    value: float,
    num_iters: int = 1000,
    num_warmup: int = 20,
):
    import statistics

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    sizes = [
        128,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        8388608,
    ]

    base_cid = 100
    all_values = comm.allgather(value)
    expected_sum = float(sum(all_values))

    if rank == 0:
        print("==== DOCA ReduceScatter (flat) benchmark ====")
        print(f"[CHECK] expected per-element value = {expected_sum}")
        print(f"[INFO] num_warmup={num_warmup}, num_iters={num_iters}")

    for idx, chunk in enumerate(sizes):
        cid = base_cid + idx

        input_flat = torch.full(
            (world_size * chunk,),
            value,
            dtype=torch.float16,
            device=device,
        ).contiguous()

        output_flat = torch.empty(
            (chunk,),
            dtype=torch.float16,
            device=device,
        ).contiguous()
        _cuda_touch_inplace(output_flat, "write_zero")

        # correctness
        comm.Barrier()
        reduce_scatter_flat_via_base(output_flat, input_flat, cid=cid, async_op=False)
        comm.Barrier()

        ok = torch.allclose(
            output_flat,
            torch.full_like(output_flat, expected_sum),
            rtol=1e-4,
            atol=1e-4,
        )
        if not ok:
            max_err = (output_flat - expected_sum).abs().max().item()
            print(f"[ERROR][DOCA RS flat] rank={rank}, N={chunk}: mismatch (max abs error={max_err})")
            torch.set_printoptions(threshold=float('inf'))
            print(f"output_flat={output_flat}")
            sys.exit()

        iters_this_size = 100 if chunk > 8388608 else num_iters
        times: list[float] = []
        for it in range(num_warmup + iters_this_size):
            comm.Barrier()
            t0 = time.time()

            reduce_scatter_flat_via_base(output_flat, input_flat, cid=cid, async_op=False)

            comm.Barrier()
            t1 = time.time()

            local_dt = t1 - t0
            max_dt = comm.allreduce(local_dt, op=MPI.MAX)
            if it >= num_warmup:
                times.append(max_dt)

        mean_dt = sum(times) / len(times)
        stddev_dt = statistics.stdev(times) if len(times) > 1 else 0.0
        min_dt = min(times)
        max_dt_val = max(times)
        sorted_times = sorted(times)
        p50 = sorted_times[len(sorted_times) // 2]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        if rank == 0:
            bytes_per_rank = chunk * 2
            total_bytes = bytes_per_rank * world_size
            bw_gbps = (total_bytes / mean_dt) / 1e9
            cv = (stddev_dt / mean_dt * 100) if mean_dt > 0 else 0
            print(
                f"[DOCA RS flat] N={chunk:>7d} | "
                f"avg={mean_dt * 1e3:8.3f} ms | "
                f"p50={p50 * 1e3:8.3f} ms | "
                f"p99={p99 * 1e3:8.3f} ms | "
                f"min={min_dt * 1e3:8.3f} ms | "
                f"max={max_dt_val * 1e3:8.3f} ms | "
                f"stddev={stddev_dt * 1e3:7.3f} ms ({cv:4.1f}%) | "
                f"BW={bw_gbps:7.2f} GB/s | correct={ok}"
            )


def benchmark_doca_all_gather_flat(
    device: torch.device,
    value: float,
    num_iters: int = 1000,
    num_warmup: int = 20,
):
    import statistics

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    sizes = [
        128,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        8388608,
    ]

    base_cid = 200
    all_values = comm.allgather(value)

    if rank == 0:
        print("==== DOCA AllGather (flat) benchmark ====")
        print(f"[INFO] num_warmup={num_warmup}, num_iters={num_iters}")

    for idx, chunk in enumerate(sizes):
        cid = base_cid + idx

        input_flat = torch.full(
            (chunk,),
            value,
            dtype=torch.float16,
            device=device,
        ).contiguous()

        # output_flat は確実に touch
        output_flat = torch.empty(
            (world_size * chunk,),
            dtype=torch.float16,
            device=device,
        ).contiguous()
        _cuda_touch_inplace(output_flat, "write_zero")

        # correctness
        comm.Barrier()
        all_gather_flat_via_base(output_flat, input_flat, cid=cid, async_op=False)
        comm.Barrier()

        out_2d = output_flat.view(world_size, chunk)
        ok = True
        for src_rank, v in enumerate(all_values):
            row = out_2d[src_rank]
            if not torch.allclose(row, torch.full_like(row, v), rtol=1e-4, atol=1e-4):
                ok = False
                max_err = (row - v).abs().max().item()
                if rank == 0:
                    print(
                        f"[ERROR][DOCA AG flat] rank={rank}, N={chunk}, src_rank={src_rank}: "
                        f"mismatch (max abs error={max_err})"
                    )
                    torch.set_printoptions(threshold=float('inf'))  # 省略なし
                    print(output_flat)
                sys.exit()
                break

        iters_this_size = 100 if chunk > 8388608 else num_iters
        times: list[float] = []
        for it in range(num_warmup + iters_this_size):
            comm.Barrier()
            t0 = time.time()

            all_gather_flat_via_base(output_flat, input_flat, cid=cid, async_op=False)

            comm.Barrier()
            t1 = time.time()

            local_dt = t1 - t0
            max_dt = comm.allreduce(local_dt, op=MPI.MAX)
            if it >= num_warmup:
                times.append(max_dt)

        mean_dt = sum(times) / len(times)
        stddev_dt = statistics.stdev(times) if len(times) > 1 else 0.0
        min_dt = min(times)
        max_dt_val = max(times)
        sorted_times = sorted(times)
        p50 = sorted_times[len(sorted_times) // 2]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        if rank == 0:
            total_bytes = chunk * world_size * 2  # AG: output per rank = total gathered data
            bw_gbps = (total_bytes / mean_dt) / 1e9
            cv = (stddev_dt / mean_dt * 100) if mean_dt > 0 else 0
            print(
                f"[DOCA AG flat] N={chunk:>7d} | "
                f"avg={mean_dt * 1e3:8.3f} ms | "
                f"p50={p50 * 1e3:8.3f} ms | "
                f"p99={p99 * 1e3:8.3f} ms | "
                f"min={min_dt * 1e3:8.3f} ms | "
                f"max={max_dt_val * 1e3:8.3f} ms | "
                f"stddev={stddev_dt * 1e3:7.3f} ms ({cv:4.1f}%) | "
                f"BW={bw_gbps:7.2f} GB/s | correct={ok}"
            )


# =============================================================================
# Benchmarks (Torch NCCL)
# =============================================================================

def _init_torch_distributed_nccl(device: torch.device, timeout_min: int = 15):
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if device.type != "cuda":
        raise RuntimeError(f"NCCL backend requires CUDA device, got {device}")

    local_rank = device.index if device.index is not None else 0

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    if "MASTER_ADDR" not in os.environ:
        raise RuntimeError("MASTER_ADDR is not set in environment")
    if "MASTER_PORT" not in os.environ:
        raise RuntimeError("MASTER_PORT is not set in environment")

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=timeout_min),
        device_id=local_rank,
    )


def benchmark_gpu_reduce_scatter_torch(
    device: torch.device,
    value: float,
    rank: int,
    world_size: int,
    num_iters: int = 1000,
    num_warmup: int = 20,
):
    import statistics
    comm = MPI.COMM_WORLD
    _init_torch_distributed_nccl(device)

    sizes = [
        128,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        8388608,
    ]

    all_values = comm.allgather(value)
    expected_sum = float(sum(all_values))

    if rank == 0:
        print("==== GPU ReduceScatter (torch.distributed, NCCL) benchmark ====")
        print(f"[CHECK] expected per-element value = {expected_sum}")
        print(f"[INFO] num_warmup={num_warmup}, num_iters={num_iters}")

    for chunk in sizes:
        # output
        output_tensor = torch.empty((chunk,), dtype=torch.float16, device=device)
        _cuda_touch_inplace(output_tensor, "write_zero")

        # inputs
        input_list = []
        for _ in range(world_size):
            t = torch.full((chunk,), value, dtype=torch.float16, device=device)
            # full() は書き込みを伴うので基本 touch 済みだが、念のため同期だけでも良い
            torch.cuda.synchronize(device)
            input_list.append(t)

        dist.barrier()
        comm.Barrier()
        dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
        dist.barrier()
        comm.Barrier()

        ok = torch.allclose(output_tensor, torch.full_like(output_tensor, expected_sum), rtol=1e-4, atol=1e-4)
        if not ok:
            max_err = (output_tensor - expected_sum).abs().max().item()
            print(f"[ERROR][GPU RS NCCL] rank={rank}, N={chunk}: mismatch (max abs error={max_err})")

        iters_this_size = 100 if chunk > 8388608 else num_iters
        times: list[float] = []
        for it in range(num_warmup + iters_this_size):
            torch.cuda.synchronize(device)
            comm.Barrier()
            t0 = time.time()

            dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)

            torch.cuda.synchronize(device)
            comm.Barrier()
            t1 = time.time()

            local_dt = t1 - t0
            max_dt = comm.allreduce(local_dt, op=MPI.MAX)
            if it >= num_warmup:
                times.append(max_dt)

        mean_dt = sum(times) / len(times)
        stddev_dt = statistics.stdev(times) if len(times) > 1 else 0.0
        min_dt = min(times)
        max_dt_val = max(times)
        sorted_times = sorted(times)
        p50 = sorted_times[len(sorted_times) // 2]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        if rank == 0:
            bytes_per_rank = chunk * 2
            total_bytes = bytes_per_rank * world_size
            bw_gbps = (total_bytes / mean_dt) / 1e9
            cv = (stddev_dt / mean_dt * 100) if mean_dt > 0 else 0
            print(
                f"[GPU RS NCCL] N={chunk:>7d} | "
                f"avg={mean_dt * 1e3:8.3f} ms | "
                f"p50={p50 * 1e3:8.3f} ms | "
                f"p99={p99 * 1e3:8.3f} ms | "
                f"min={min_dt * 1e3:8.3f} ms | "
                f"max={max_dt_val * 1e3:8.3f} ms | "
                f"stddev={stddev_dt * 1e3:7.3f} ms ({cv:4.1f}%) | "
                f"BW={bw_gbps:7.2f} GB/s | correct={ok}"
            )


def benchmark_gpu_all_gather_torch(
    device: torch.device,
    value: float,
    rank: int,
    world_size: int,
    num_iters: int = 1000,
    num_warmup: int = 20,
):
    import statistics
    comm = MPI.COMM_WORLD
    _init_torch_distributed_nccl(device)

    sizes = [
        128,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        8388608,
    ]

    all_values = comm.allgather(value)

    if rank == 0:
        print("==== GPU AllGather (torch.distributed, NCCL) benchmark ====")
        print(f"[INFO] num_warmup={num_warmup}, num_iters={num_iters}")

    for chunk in sizes:
        input_tensor = torch.full((chunk,), value, dtype=torch.float16, device=device)
        torch.cuda.synchronize(device)

        output_list = []
        for _ in range(world_size):
            t = torch.empty((chunk,), dtype=torch.float16, device=device)
            _cuda_touch_inplace(t, "write_zero")
            output_list.append(t)

        dist.barrier()
        comm.Barrier()
        dist.all_gather(output_list, input_tensor)
        dist.barrier()
        comm.Barrier()

        ok = True
        for src_rank, v in enumerate(all_values):
            row = output_list[src_rank]
            if not torch.allclose(row, torch.full_like(row, v), rtol=1e-4, atol=1e-4):
                ok = False
                max_err = (row - v).abs().max().item()
                print(
                    f"[ERROR][GPU AG NCCL] rank={rank}, N={chunk}, src_rank={src_rank}: "
                    f"mismatch (max abs error={max_err})"
                )
                break

        iters_this_size = 100 if chunk > 8388608 else num_iters
        times: list[float] = []
        for it in range(num_warmup + iters_this_size):
            torch.cuda.synchronize(device)
            comm.Barrier()
            t0 = time.time()

            dist.all_gather(output_list, input_tensor)

            torch.cuda.synchronize(device)
            comm.Barrier()
            t1 = time.time()

            local_dt = t1 - t0
            max_dt = comm.allreduce(local_dt, op=MPI.MAX)
            if it >= num_warmup:
                times.append(max_dt)

        mean_dt = sum(times) / len(times)
        stddev_dt = statistics.stdev(times) if len(times) > 1 else 0.0
        min_dt = min(times)
        max_dt_val = max(times)
        sorted_times = sorted(times)
        p50 = sorted_times[len(sorted_times) // 2]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        if rank == 0:
            total_bytes = chunk * world_size * 2  # AG: output per rank = total gathered data
            bw_gbps = (total_bytes / mean_dt) / 1e9
            cv = (stddev_dt / mean_dt * 100) if mean_dt > 0 else 0
            print(
                f"[GPU AG NCCL] N={chunk:>7d} | "
                f"avg={mean_dt * 1e3:8.3f} ms | "
                f"p50={p50 * 1e3:8.3f} ms | "
                f"p99={p99 * 1e3:8.3f} ms | "
                f"min={min_dt * 1e3:8.3f} ms | "
                f"max={max_dt_val * 1e3:8.3f} ms | "
                f"stddev={stddev_dt * 1e3:7.3f} ms ({cv:4.1f}%) | "
                f"BW={bw_gbps:7.2f} GB/s | correct={ok}"
            )

#######################################
###   SmartNIC中心CPUバッファ関数群   ###
#######################################
#新しく追加したフラッグポーリング用の関数
import ctypes
def poll_flag_seq(flag_tensor: torch.Tensor, expected: int,
                  timeout_s: float = 5.0,
                  sleep_us: int = 50,
                  max_sleep_us: int = 2000):
    """
    flag_tensor: CPU上の int32 (shape=(1,)) pinned推奨
    expected: 期待する seq 値
    """
    assert flag_tensor.device.type == "cpu"
    assert flag_tensor.dtype == torch.int32
    assert flag_tensor.numel() == 1

    flag_ptr = ctypes.cast(int(flag_tensor.data_ptr()), ctypes.POINTER(ctypes.c_int32))

    t0 = time.monotonic()
    cur_sleep = sleep_us
    while True:
        if flag_ptr[0] == expected:
            return True
        if (time.monotonic() - t0) > timeout_s:
            return False
        # 少し待ちながら（忙しすぎるスピンを避ける）
        time.sleep(cur_sleep / 1e6)
        cur_sleep = min(cur_sleep * 2, max_sleep_us)


def test_all_gather_full_cpu(
    device: torch.device,
    value: float,
    num_iters: int = 10,
    num_warmup: int = 10,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    base_cid = 200
    all_values = comm.allgather(value)
    
    if rank == 0:
        print("==== DOCA AllGather FULL CPU (flat) benchmark ====")
        print(f"[INFO] num_warmup={num_warmup}, num_iters={num_iters}")
    #テストコード
    chunk = 1048576*2
    cid = base_cid
    #1. 各種メモリ領域の定義 (面倒なので一回投げます)
    input_flat = torch.full(
        (chunk,),
        value,
        dtype=torch.float16,
        device="cpu",
        pin_memory=True,
    ).contiguous()
    
    output_flat = torch.empty(
        (world_size * chunk,),
        dtype=torch.float16,
        device=device,
    ).contiguous()
    _cuda_touch_inplace(output_flat, "write_zero")
    
    full_param_flat = torch.empty(
        (world_size * chunk,),
        dtype=torch.float16,
        device="cpu",
        pin_memory=True,
    ).contiguous()
    full_param_flat.zero_()
    
    # 受信側/送信側どちらでも「RDMAで見えるフラグ領域」として確保
    full_param_flag = torch.zeros(
        (1,),
        dtype=torch.int32,
        device="cpu",
        pin_memory=True,   # CUDA転送にも使うなら有益。RDMAだけなら必須ではないケースもある
    ).contiguous()
    # クリア
    full_param_flag[0] = 0
    
    #2. DOCA側のAllGather➡full_param移動の関数を投げる
    comm.Barrier()
    all_gather_cpu_full_flat_via_base(output_flat, input_flat, full_param_flat, full_param_flag, cid=cid, async_op=False)
    comm.Barrier()
        
    #3. 終了処理 ➡ AllGatherの結果
    out_2d = output_flat.view(world_size, chunk)
    ok = True
    for src_rank, v in enumerate(all_values):
        row = out_2d[src_rank]
        if not torch.allclose(row, torch.full_like(row, v), rtol=1e-4, atol=1e-4):
            ok = False
            max_err = (row - v).abs().max().item()
            if rank == 0:
                print(
                    f"[ERROR][DOCA AG flat] rank={rank}, N={chunk}, src_rank={src_rank}: "
                    f"mismatch (max abs error={max_err})"
                )
                torch.set_printoptions(threshold=float('inf'))  # 省略なし
                print(output_flat)
            sys.exit()
            break
    
    #4. フラッグがついたかのポーリング
    ok = poll_flag_seq(full_param_flag, 1)
    if not ok:
        if rank == 0:
            print(f"[TIMEOUT] flag={int(full_param_flag.item())}")
        comm.Abort(1)
            
    #5. CPUメモリ領域の確認 output flatとまったく同じ値になっているはず
    # output_flat を CPU 側に持ってくる（GPUなら pinned staging を使う）
    if output_flat.device.type != "cpu":
        output_cpu = torch.empty_like(
            full_param_flat,  # shape/dtype を揃える
            device="cpu",
            pin_memory=True,
        )
        output_cpu.copy_(output_flat, non_blocking=True)
        if output_flat.is_cuda:
            torch.cuda.synchronize(device=output_flat.device)
    else:
        output_cpu = output_flat

    # 比較（基本は allclose。bitwise 完全一致を期待するなら torch.equal も可）
    if not torch.allclose(full_param_flat, output_cpu, rtol=1e-4, atol=1e-4):
        diff = (full_param_flat - output_cpu).abs()
        max_err = diff.max().item()
        idx = int(diff.argmax().item())

        if rank == 0:
            print(f"[ERROR][FULL_PARAM] mismatch: max abs err={max_err} at flat idx={idx}")

            # 周辺を少しだけ表示（float16のままだと見にくいのでfloat32にする）
            s = max(0, idx - 8)
            e = min(diff.numel(), idx + 8)
            print("full_param_flat[s:e] =", full_param_flat[s:e].to(torch.float32))
            print("output_cpu[s:e]      =", output_cpu[s:e].to(torch.float32))

        comm.Abort(1)
    else:
        if rank == 0:
            print("[OK] full_param_flat matches output_flat (within tol)")


def test_all_gather_full_cpu_multi_params(
    device: torch.device,
    value: float,
    num_iters: int = 10,
    num_warmup: int = 10,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    base_cid = 200
    all_values = comm.allgather(value)
    
    if rank == 0:
        print("==== DOCA AllGather FULL CPU (flat) benchmark ====")
        print(f"[INFO] num_warmup={num_warmup}, num_iters={num_iters}")
    #テストコード
    chunk = 16
    cid = base_cid
    #1. 各種メモリ領域の定義
    #1-1 各paramsのサイズを保持した配列を定義
    params_sizes = [chunk, chunk, chunk, chunk]
    partition_sz = sum(params_sizes)
    
    #出力用のテンソル
    output_flat = torch.empty(partition_sz*world_size, dtype=torch.float16, device=torch.cuda.current_device(), requires_grad=False).contiguous()
    flat_view = output_flat.view(-1)
    
    input_params = [
        torch.full((sz,), value * (i + 1), dtype=torch.float16, device="cpu", pin_memory=True).contiguous()
        for i, sz in enumerate(params_sizes)
    ]
    input_flat_cpu = torch.empty(partition_sz, dtype=torch.float16, device="cpu", pin_memory=True).contiguous()
    offset = 0
    for p in input_params:
        n = p.numel()
        input_flat_cpu.narrow(0, offset, n).copy_(p.detach().view(-1))
        offset += n
    
    # full_param_flats は出力テンソルと同形の連続領域を確保し、view として各 param に切り出す。
    full_param_flats = torch.empty(partition_sz*world_size, dtype=torch.float16, device="cpu", requires_grad=False).contiguous()
    #input_flat_cpuと同様にfull_param_flatsからviewとして切り出す
    full_param_2d = full_param_flats.view(world_size, partition_sz)                                     
    local_cpu_full_params = []                                                                          
    offset = 0                                                                                          
    for sz in params_sizes:                                                                             
        # shape: (world_size, sz) — param p_idx の全ランク分                                            
        local_cpu_full_params.append(full_param_2d[:, offset:offset + sz])                              
        offset += sz

    # full_param_flagsを連続領域として一括確保し、各パラメータ分をviewで切り出す
    # 各パラメータにつきint32を1要素ずつ持つ
    num_params = len(params_sizes)
    full_param_flags = torch.zeros(
        (num_params,),
        dtype=torch.int32,
        device="cpu",
        pin_memory=True,
    ).contiguous()
    local_cpu_full_param_flags = [
        full_param_flags.narrow(0, i, 1) for i in range(num_params)
    ]
    
    #2. DOCA側のAllGather➡full_param移動の関数を投げる
    #ここは修正なし (既存APIで実行する)
    comm.Barrier()
    all_gather_cpu_full_flat_via_base(flat_view, input_flat_cpu, full_param_flats, full_param_flags, cid=cid, async_op=False)
    comm.Barrier()
        
    #3. 終了処理 ➡ AllGatherの結果を各パラメータごとに確認
    for p_idx, sz in enumerate(params_sizes):
        param_offset = sum(params_sizes[:p_idx])
        ok = True
        for src_rank, v in enumerate(all_values):
            start = src_rank * partition_sz + param_offset
            row = output_flat[start : start + sz]
            if not torch.allclose(row, torch.full_like(row, v * (p_idx + 1)), rtol=1e-4, atol=1e-4):
                ok = False
                max_err = (row - v).abs().max().item()
                if rank == 0:
                    print(
                        f"[ERROR][DOCA AG flat] rank={rank}, param={p_idx}, src_rank={src_rank}: "
                        f"mismatch (max abs error={max_err})"
                        f"output_flat={output_flat}"
                    )
                sys.exit()
                break
        if ok and rank == 0:
            print(f"[OK] AllGather output check passed for param {p_idx}")

    #4. フラッグがついたかのポーリング（各パラメータごと）
    for p_idx in range(num_params):
        ok = poll_flag_seq(local_cpu_full_param_flags[p_idx], 1)
        if not ok:
            if rank == 0:
                print(f"[TIMEOUT] param={p_idx}, flag={int(local_cpu_full_param_flags[p_idx].item())}")
            comm.Abort(1)

    #5. CPUメモリ領域の確認（各パラメータごとにlocal_cpu_full_paramsとGPU出力を比較）
    for p_idx, sz in enumerate(params_sizes):
        cpu_full = local_cpu_full_params[p_idx]  # size: sz * world_size

        # output_flatから該当パラメータ分をランクごとに抽出して連結
        param_offset = sum(params_sizes[:p_idx])
        gpu_slices = []
        for src_rank in range(world_size):
            start = src_rank * partition_sz + param_offset
            gpu_slices.append(output_flat[start : start + sz])
        gpu_full = torch.stack(gpu_slices)  # shape: (world_size, sz)

        # GPU→CPUへ転送
        if gpu_full.device.type != "cpu":
            output_cpu = torch.empty_like(cpu_full, device="cpu", pin_memory=True)
            output_cpu.copy_(gpu_full, non_blocking=True)
            if gpu_full.is_cuda:
                torch.cuda.synchronize(device=gpu_full.device)
        else:
            output_cpu = gpu_full

        if not torch.allclose(cpu_full, output_cpu, rtol=1e-4, atol=1e-4):
            diff = (cpu_full - output_cpu).abs()
            max_err = diff.max().item()
            idx = int(diff.argmax().item())
            if rank == 0:
                print(f"[ERROR][FULL_PARAM] param={p_idx}: mismatch, max abs err={max_err} at idx={idx}")
                s = max(0, idx - 8)
                e = min(diff.numel(), idx + 8)
                print(f"  cpu_full[{s}:{e}] =", cpu_full[s:e].to(torch.float32))
                print(f"  output_cpu[{s}:{e}] =", output_cpu[s:e].to(torch.float32))
            comm.Abort(1)
        else:
            if rank == 0:
                print(f"[OK] full_param matches output for param {p_idx}"
                      f"cpu_full={cpu_full}")

def test_all_gather_full_cpu_list(
    device: torch.device,
    value: float,
    num_iters: int = 10,
    num_warmup: int = 10,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    base_cid = 200
    all_values = comm.allgather(value)
    
    if rank == 0:
        print("==== DOCA AllGather FULL CPU LIST (flat) benchmark ====")
        print(f"[INFO] num_warmup={num_warmup}, num_iters={num_iters}")
    #テストコード
    chunk = 1048576//4
    cid = base_cid
    #1. 各種メモリ領域の定義
    #1-1 各paramsのサイズを保持した配列を定義
    params_sizes = [chunk, chunk, chunk, chunk]
    partition_sz = sum(params_sizes)
    tensor_size = partition_sz * world_size
    
    param_list = [
            torch.full((sz,), value * (i + 1), dtype=torch.float16, device="cpu", pin_memory=True).contiguous()
            for i, sz in enumerate(params_sizes)
        ]
    
    flat_tensor = torch.empty(tensor_size, dtype=torch.float16, device=torch.cuda.current_device()).contiguous()
    partitions = []
    for i in range(world_size):
        start = partition_sz * i
        partitions.append(flat_tensor.narrow(0, start, partition_sz))
        if i == rank:
            offset = 0
            for param in param_list:
                param_numel = param.numel()
                partitions[i].narrow(0, offset, param_numel).copy_(param.data)
                offset += param_numel
                
    # full_param_flats は出力テンソルと同形の連続領域を確保し、view として各 param に切り出す。
    full_param_flats = torch.empty(partition_sz*world_size, dtype=torch.float16, device="cpu", requires_grad=False).contiguous()
    #input_flat_cpuと同様にfull_param_flatsからviewとして切り出す
    full_param_2d = full_param_flats.view(world_size, partition_sz)                                     
    local_cpu_full_params = []                                                                          
    offset = 0                                                                                          
    for sz in params_sizes:                                                                             
        # shape: (world_size, sz) — param p_idx の全ランク分                                            
        local_cpu_full_params.append(full_param_2d[:, offset:offset + sz])                              
        offset += sz

    # full_param_flagsを連続領域として一括確保し、各パラメータ分をviewで切り出す
    # 各パラメータにつきint32を1要素ずつ持つ
    num_params = len(params_sizes)
    full_param_flags = torch.zeros(
        (num_params,),
        dtype=torch.int32,
        device="cpu",
        pin_memory=True,
    ).contiguous()
    local_cpu_full_param_flags = [
        full_param_flags.narrow(0, i, 1) for i in range(num_params)
    ]
    
    all_gather_cpu_full_list_via_base(
            output_tensors=partitions,
            input_tensor=partitions[rank],
            cpu_full_flat=full_param_flats, 
            cpu_full_flag=full_param_flags,
            cid=cid,
            async_op=False,
        )
    
    #3. AllGatherの結果を各パラメータごとに確認（GPU側: flat_tensor）
    for p_idx, sz in enumerate(params_sizes):
        param_offset = sum(params_sizes[:p_idx])
        ok = True
        for src_rank, v in enumerate(all_values):
            start = src_rank * partition_sz + param_offset
            row = flat_tensor[start : start + sz]
            if not torch.allclose(row, torch.full_like(row, v * (p_idx + 1)), rtol=1e-4, atol=1e-4):
                ok = False
                max_err = (row - v * (p_idx + 1)).abs().max().item()
                if rank == 0:
                    print(
                        f"[ERROR][DOCA AG list] rank={rank}, param={p_idx}, src_rank={src_rank}: "
                        f"mismatch (max abs error={max_err})"
                    )
                sys.exit()
                break
        if ok and rank == 0:
            print(f"[OK] AllGather list output check passed for param {p_idx}")

    #4. フラッグがついたかのポーリング（各パラメータごと）
    for p_idx in range(num_params):
        ok = poll_flag_seq(local_cpu_full_param_flags[p_idx], 1)
        if not ok:
            if rank == 0:
                print(f"[TIMEOUT] param={p_idx}, flag={int(local_cpu_full_param_flags[p_idx].item())}")
            comm.Abort(1)

    #5. CPUメモリ領域の確認（各パラメータごとにlocal_cpu_full_paramsとGPU出力を比較）
    for p_idx, sz in enumerate(params_sizes):
        cpu_full = local_cpu_full_params[p_idx]  # shape: (world_size, sz)

        # flat_tensorから該当パラメータ分をランクごとに抽出して連結
        param_offset = sum(params_sizes[:p_idx])
        gpu_slices = []
        for src_rank in range(world_size):
            start = src_rank * partition_sz + param_offset
            gpu_slices.append(flat_tensor[start : start + sz])
        gpu_full = torch.stack(gpu_slices)  # shape: (world_size, sz)

        # GPU→CPUへ転送
        if gpu_full.device.type != "cpu":
            output_cpu = torch.empty_like(cpu_full, device="cpu", pin_memory=True)
            output_cpu.copy_(gpu_full, non_blocking=True)
            if gpu_full.is_cuda:
                torch.cuda.synchronize(device=gpu_full.device)
        else:
            output_cpu = gpu_full

        if not torch.allclose(cpu_full, output_cpu, rtol=1e-4, atol=1e-4):
            diff = (cpu_full - output_cpu).abs()
            max_err = diff.max().item()
            idx = int(diff.argmax().item())
            if rank == 0:
                print(f"[ERROR][FULL_PARAM list] param={p_idx}: mismatch, max abs err={max_err} at idx={idx}")
                s = max(0, idx - 8)
                e = min(diff.numel(), idx + 8)
                print(f"  cpu_full[{s}:{e}] =", cpu_full[s:e].to(torch.float32))
                print(f"  output_cpu[{s}:{e}] =", output_cpu[s:e].to(torch.float32))
            comm.Abort(1)
        else:
            if rank == 0:
                print(f"[OK] full_param list matches output for param {p_idx}")

# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server_name")
    parser.add_argument("pci_addr")
    parser.add_argument("device_name")
    parser.add_argument("--run", choices=["doca_rs", "doca_ag", "nccl_rs", "nccl_ag", "all"], default="all")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No CUDA devices available")

    local_rank = rank % num_gpus
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    value = float(10 ** rank)

    # ---- DOCA comch init ----
    doca_comch_client_pybind.doca_comch_client_init_py(args.server_name, args.pci_addr)

    id0 = 0
    doca_comch_client_pybind.ucp_connect_host_dpu_request_py(id0)
    doca_comch_client_pybind.ucp_create_ring_request_py(id0)
    
    time.sleep(1)


    # ---- run benchmarks ----
    if args.run in ("doca_rs", "all"):
        benchmark_doca_reduce_scatter_flat(device=device, value=value)
        pass

    if args.run in ("doca_ag", "all"):
        benchmark_doca_all_gather_flat(device=device, value=value)
        pass
    
    if args.run in ("nccl_rs", "all"):
        benchmark_gpu_reduce_scatter_torch(device=device, value=value, rank=rank, world_size=world_size)
        pass

    if args.run in ("nccl_ag", "all"):
        benchmark_gpu_all_gather_torch(device=device, value=value, rank=rank, world_size=world_size)
        pass

    # ---- finalize ----
    comm.Barrier()
    if rank == 0:
        try:
            input("Enterキーを押すと続行します...")
        except EOFError:
            print("stdin が接続されていないので、そのまま続行します")

    comm.Barrier()
    doca_comch_client_pybind.doca_mpi_finalize_py()


if __name__ == "__main__":
    main()
