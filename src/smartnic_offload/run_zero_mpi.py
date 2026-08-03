"""mpirun 用ランチャ。mpi4py で rank/world_size を取得し、torch.distributed が
期待する環境変数 (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT) を
セットしてから run_zero.run_zero() を呼ぶ。
"""

import os
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
COMCH_HOST_DIR = THIS_DIR / "comch_mpi" / "host"
sys.path.insert(0, str(COMCH_HOST_DIR))

import doca_comch_client_pybind
from doca_comch_client_pybind import CollectiveCommunication

from mpi4py import MPI
import run_zero


def setup_env_from_mpi(master_addr=None, master_port=None):
    """
    MPI(COMM_WORLD) から rank / world_size を取得し、
    torchrun が本来セットする環境変数を上書きする。
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # 単ノード or 各ノード同数 GPU 前提
    local_rank = rank
    try:
        import torch

        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            local_rank = rank % num_gpus
    except Exception:
        # torch がまだ使えない環境でも一応動くようにしておく
        pass

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    if master_addr is not None:
        os.environ["MASTER_ADDR"] = master_addr
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    if master_port is not None:
        os.environ["MASTER_PORT"] = str(master_port)
    else:
        os.environ.setdefault("MASTER_PORT", "29500")

    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server_name")
    parser.add_argument("pci_addr")
    parser.add_argument("device_name")

    parser.add_argument(
        "--profiler",
        action="store_true",
        help="Enable profiler (run_zero.py の引数と同じ)",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument(
        "--master_addr",
        type=str,
        default=None,
        help="MASTER_ADDR として使う IP/ホスト名 (torchrun の --master_addr 相当)",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default=None,
        help="MASTER_PORT として使う ポート番号 (torchrun の --master_port 相当)",
    )
    parser.add_argument("--model", type=str, default="vit_l_16")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--measure-iters", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--debug-params", action="store_true")
    parser.add_argument("--reduce-bucket-size", type=float, default=1e8)
    parser.add_argument("--prefetch-bucket-size", type=float, default=1e8)
    parser.add_argument("--max-reuse-distance", type=float, default=0)
    parser.add_argument("--max-live-parameters", type=float, default=1.5e8)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_env_from_mpi(
        master_addr=args.master_addr,
        master_port=args.master_port,
    )
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    process_per_node = 2
    local_rank = rank % process_per_node
    
    # mpirun --bind-to core がプロセスを1コアに制限している場合、
    # 全コアにアクセスできるように解除する (run_zero.py が NUMA-aware に再配分)
    try:
        all_cpus = set(range(os.cpu_count()))
        os.sched_setaffinity(0, all_cpus)
    except Exception:
        pass

    doca_comch_client_pybind.doca_comch_client_init_py(args.server_name, args.pci_addr)
    id0 = 0
    doca_comch_client_pybind.ucp_connect_host_dpu_request_py(id0)
    doca_comch_client_pybind.ucp_create_ring_request_py(id0)

    if args.debug_params:
        from common import debug_params as dbg
        dbg.enable()

    run_zero.run_zero(
        use_profiler=args.profiler,
        use_bf16=args.bf16,
        use_ema=args.ema,
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        seq_len=args.seq_len,
        num_epochs=args.epochs,
        reduce_bucket_size=int(args.reduce_bucket_size),
        prefetch_bucket_size=int(args.prefetch_bucket_size),
        max_reuse_distance=int(args.max_reuse_distance),
        max_live_parameters=int(args.max_live_parameters),
    )


if __name__ == "__main__":
    main()
