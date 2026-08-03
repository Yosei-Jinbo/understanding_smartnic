# 集合通信 (AllGather / ReduceScatter) 単体のベンチマーク兼 correctness テスト。
# DOCA (SmartNIC オフロード) 経路と torch.distributed (NCCL) 経路を同一条件で計測する。
#
# comch_mpi/host/client.py のテストハーネスを整理した後継。pybind のクリーン API
# (enqueue + wait + release) のみを使う。
#
# 起動は run_collective_benchmark.sh を参照 (mpirun --app 経由、4 rank / 2 node)。

import os
import sys
import time
import argparse
import statistics
from datetime import timedelta
from typing import Optional

from pathlib import Path

# src/smartnic_offload/comch_mpi/host の pybind モジュールを import する
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
COMCH_HOST_DIR = REPO_ROOT / "src" / "smartnic_offload" / "comch_mpi" / "host"
sys.path.insert(0, str(COMCH_HOST_DIR))

import torch
import torch.distributed as dist
from mpi4py import MPI

import doca_comch_client_pybind
from doca_comch_client_pybind import CollectiveCommunication


# ベンチマーク対象のメッセージサイズ (per-rank 要素数, fp16)
SIZES = [
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


def _cuda_touch_inplace(t: torch.Tensor, mode: str = "write_zero") -> None:
    """
    torch.empty で確保した GPU メモリを実際に commit させるために
    実 device op を発行する。DOCA (RDMA) に渡す前に必須。
    """
    if not t.is_cuda:
        return

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

    torch.cuda.synchronize(t.device)


# =============================================================================
# DOCA collective wrappers (enqueue + wait + release)
# =============================================================================

class _ComchHandleWork:
    """C 側の handle を保持して wait/release する Work。"""

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

    def __del__(self):
        if not self._released:
            try:
                doca_comch_client_pybind.comch_req_release_py(self._handle)
            except Exception:
                pass
            self._released = True


def _collective_flat_via_base(
    op: CollectiveCommunication,
    output_flat: torch.Tensor,
    input_flat: torch.Tensor,
    cid: int,
    async_op: bool = False,
) -> Optional[_ComchHandleWork]:
    """AllGather / ReduceScatter 共通の enqueue 経路。

    サイズ関係は op で決まる:
      AG: output.numel == input.numel * world_size
      RS: input.numel  == output.numel * world_size
    """
    world_size = MPI.COMM_WORLD.Get_size()

    in_view = input_flat.view(-1)
    out_view = output_flat.view(-1)

    if op == CollectiveCommunication.ALL_GATHER:
        if out_view.numel() != in_view.numel() * world_size:
            raise ValueError("output_flat.numel() must be input_flat.numel() * world_size")
    else:
        if in_view.numel() != out_view.numel() * world_size:
            raise ValueError("input_flat.numel() must be output_flat.numel() * world_size")
    if out_view.dtype != in_view.dtype:
        raise TypeError("output_flat.dtype must match input_flat.dtype")
    if out_view.device != in_view.device:
        raise TypeError("output_flat.device must match input_flat.device")

    _cuda_touch_inplace(in_view, "read")
    _cuda_touch_inplace(out_view, "write_zero")

    handle = doca_comch_client_pybind.ucp_collective_enqueue_py(
        cid, in_view, out_view, op
    )
    work = _ComchHandleWork(handle)
    if async_op:
        return work
    work.wait()
    return None


def all_gather_flat_via_base(output_flat, input_flat, cid, async_op=False):
    return _collective_flat_via_base(
        CollectiveCommunication.ALL_GATHER, output_flat, input_flat, cid, async_op
    )


def reduce_scatter_flat_via_base(output_flat, input_flat, cid, async_op=False):
    return _collective_flat_via_base(
        CollectiveCommunication.REDUCE_SCATTER, output_flat, input_flat, cid, async_op
    )


# =============================================================================
# 計測・出力の共通部
# =============================================================================

def _measure_and_report(
    tag: str,
    chunk: int,
    fn,
    ok: bool,
    total_bytes: int,
    num_iters: int,
    num_warmup: int,
):
    """fn() を warmup + iters 回実行し、rank 間 max 時間の統計を rank0 が出力する。"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    times: list[float] = []
    for it in range(num_warmup + num_iters):
        comm.Barrier()
        t0 = time.time()

        fn()

        comm.Barrier()
        t1 = time.time()

        local_dt = t1 - t0
        max_dt = comm.allreduce(local_dt, op=MPI.MAX)
        if it >= num_warmup:
            times.append(max_dt)

    if rank != 0:
        return

    mean_dt = sum(times) / len(times)
    stddev_dt = statistics.stdev(times) if len(times) > 1 else 0.0
    sorted_times = sorted(times)
    p50 = sorted_times[len(sorted_times) // 2]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]
    bw_gbps = (total_bytes / mean_dt) / 1e9
    cv = (stddev_dt / mean_dt * 100) if mean_dt > 0 else 0
    print(
        f"[{tag}] N={chunk:>7d} | "
        f"avg={mean_dt * 1e3:8.3f} ms | "
        f"p50={p50 * 1e3:8.3f} ms | "
        f"p99={p99 * 1e3:8.3f} ms | "
        f"min={min(times) * 1e3:8.3f} ms | "
        f"max={max(times) * 1e3:8.3f} ms | "
        f"stddev={stddev_dt * 1e3:7.3f} ms ({cv:4.1f}%) | "
        f"BW={bw_gbps:7.2f} GB/s | correct={ok}"
    )


def _fail(tag: str, rank: int, chunk: int, max_err: float, detail: str = "") -> None:
    print(f"[ERROR][{tag}] rank={rank}, N={chunk}{detail}: mismatch (max abs error={max_err})")
    sys.exit(1)


# =============================================================================
# Benchmarks (DOCA)
# =============================================================================

def benchmark_doca_reduce_scatter_flat(
    device: torch.device,
    value: float,
    num_iters: int = 1000,
    num_warmup: int = 20,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    base_cid = 100
    expected_sum = float(sum(comm.allgather(value)))

    if rank == 0:
        print("==== DOCA ReduceScatter (flat) benchmark ====")
        print(f"[CHECK] expected per-element value = {expected_sum}")
        print(f"[INFO] num_warmup={num_warmup}, num_iters={num_iters}")

    for idx, chunk in enumerate(SIZES):
        cid = base_cid + idx

        input_flat = torch.full(
            (world_size * chunk,), value, dtype=torch.float16, device=device
        ).contiguous()
        output_flat = torch.empty(
            (chunk,), dtype=torch.float16, device=device
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
            _fail("DOCA RS flat", rank, chunk, max_err)

        _measure_and_report(
            "DOCA RS flat",
            chunk,
            lambda: reduce_scatter_flat_via_base(output_flat, input_flat, cid=cid, async_op=False),
            ok,
            total_bytes=chunk * 2 * world_size,
            num_iters=num_iters,
            num_warmup=num_warmup,
        )


def benchmark_doca_all_gather_flat(
    device: torch.device,
    value: float,
    num_iters: int = 1000,
    num_warmup: int = 20,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    base_cid = 200
    all_values = comm.allgather(value)

    if rank == 0:
        print("==== DOCA AllGather (flat) benchmark ====")
        print(f"[INFO] num_warmup={num_warmup}, num_iters={num_iters}")

    for idx, chunk in enumerate(SIZES):
        cid = base_cid + idx

        input_flat = torch.full(
            (chunk,), value, dtype=torch.float16, device=device
        ).contiguous()
        output_flat = torch.empty(
            (world_size * chunk,), dtype=torch.float16, device=device
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
                max_err = (row - v).abs().max().item()
                _fail("DOCA AG flat", rank, chunk, max_err, detail=f", src_rank={src_rank}")

        _measure_and_report(
            "DOCA AG flat",
            chunk,
            lambda: all_gather_flat_via_base(output_flat, input_flat, cid=cid, async_op=False),
            ok,
            total_bytes=chunk * world_size * 2,
            num_iters=num_iters,
            num_warmup=num_warmup,
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


def benchmark_nccl_reduce_scatter(
    device: torch.device,
    value: float,
    num_iters: int = 1000,
    num_warmup: int = 20,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    _init_torch_distributed_nccl(device)

    expected_sum = float(sum(comm.allgather(value)))

    if rank == 0:
        print("==== NCCL ReduceScatter (torch.distributed) benchmark ====")
        print(f"[CHECK] expected per-element value = {expected_sum}")
        print(f"[INFO] num_warmup={num_warmup}, num_iters={num_iters}")

    for chunk in SIZES:
        output_tensor = torch.empty((chunk,), dtype=torch.float16, device=device)
        _cuda_touch_inplace(output_tensor, "write_zero")

        input_list = [
            torch.full((chunk,), value, dtype=torch.float16, device=device)
            for _ in range(world_size)
        ]
        torch.cuda.synchronize(device)

        dist.barrier()
        comm.Barrier()
        dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
        dist.barrier()
        comm.Barrier()

        ok = torch.allclose(output_tensor, torch.full_like(output_tensor, expected_sum), rtol=1e-4, atol=1e-4)
        if not ok:
            max_err = (output_tensor - expected_sum).abs().max().item()
            print(f"[ERROR][NCCL RS] rank={rank}, N={chunk}: mismatch (max abs error={max_err})")

        def run_once():
            torch.cuda.synchronize(device)
            dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)

        _measure_and_report(
            "NCCL RS",
            chunk,
            run_once,
            ok,
            total_bytes=chunk * 2 * world_size,
            num_iters=num_iters,
            num_warmup=num_warmup,
        )


def benchmark_nccl_all_gather(
    device: torch.device,
    value: float,
    num_iters: int = 1000,
    num_warmup: int = 20,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    _init_torch_distributed_nccl(device)

    all_values = comm.allgather(value)

    if rank == 0:
        print("==== NCCL AllGather (torch.distributed) benchmark ====")
        print(f"[INFO] num_warmup={num_warmup}, num_iters={num_iters}")

    for chunk in SIZES:
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
                    f"[ERROR][NCCL AG] rank={rank}, N={chunk}, src_rank={src_rank}: "
                    f"mismatch (max abs error={max_err})"
                )
                break

        def run_once():
            torch.cuda.synchronize(device)
            dist.all_gather(output_list, input_tensor)
            torch.cuda.synchronize(device)

        _measure_and_report(
            "NCCL AG",
            chunk,
            run_once,
            ok,
            total_bytes=chunk * world_size * 2,
            num_iters=num_iters,
            num_warmup=num_warmup,
        )


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="集合通信単体ベンチマーク (DOCA vs NCCL)"
    )
    parser.add_argument("server_name")
    parser.add_argument("pci_addr")
    parser.add_argument("device_name")
    parser.add_argument("--run", choices=["doca_rs", "doca_ag", "nccl_rs", "nccl_ag", "all"], default="all")
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--num-warmup", type=int, default=20)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No CUDA devices available")

    local_rank = rank % num_gpus
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # rank ごとに一意な値 (correctness チェックで送信元を判別できる)
    value = float(10 ** rank)

    # ---- DOCA comch init (run_zero_mpi.py と同一手順) ----
    doca_comch_client_pybind.doca_comch_client_init_py(args.server_name, args.pci_addr)

    id0 = 0
    doca_comch_client_pybind.ucp_connect_host_dpu_request_py(id0)
    doca_comch_client_pybind.ucp_create_ring_request_py(id0)

    # DPU 側の ring 構築 (4 rank の集団処理) 完了待ちの保険。
    # create_ring 自体は ComCh 応答で返るが、DPU 内部のリング接続確立が
    # わずかに遅れて最初の collective と競合した実績があるため 1 秒置く。
    comm.Barrier()
    time.sleep(1)

    # ---- run benchmarks ----
    if args.run in ("doca_rs", "all"):
        benchmark_doca_reduce_scatter_flat(
            device=device, value=value,
            num_iters=args.num_iters, num_warmup=args.num_warmup,
        )

    if args.run in ("doca_ag", "all"):
        benchmark_doca_all_gather_flat(
            device=device, value=value,
            num_iters=args.num_iters, num_warmup=args.num_warmup,
        )

    if args.run in ("nccl_rs", "all"):
        benchmark_nccl_reduce_scatter(
            device=device, value=value,
            num_iters=args.num_iters, num_warmup=args.num_warmup,
        )

    if args.run in ("nccl_ag", "all"):
        benchmark_nccl_all_gather(
            device=device, value=value,
            num_iters=args.num_iters, num_warmup=args.num_warmup,
        )

    comm.Barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    if rank == 0:
        print("[done] collective benchmark finished")


if __name__ == "__main__":
    main()
