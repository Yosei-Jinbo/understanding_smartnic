# run_zero_mpi.py
"""
mpirun 用ランチャ

- mpirun + mpi4py で rank / world_size を取得
- それを torch.distributed が期待する環境変数
  (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT)
  に詰めてから、元の run_zero.run_zero() を呼び出す。

元の run_zero.py は torchrun 前提のままで OK。
"""

import os
import argparse
import sys
from mpi4py import MPI  # mpirun でランク情報を取る
import run_zero  # 同じディレクトリの run_zero.py をインポート


def setup_env_from_mpi(master_addr=None, master_port=None):
    """
    MPI(COMM_WORLD) から rank / world_size を取得し、
    torchrun が本来セットする環境変数を上書きする。
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # 単ノード or 各ノード同数 GPU 前提:
    # local_rank は「GPU 枚数で割った余り」にする
    local_rank = rank
    try:
        import torch

        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            local_rank = rank % num_gpus
    except Exception:
        # torch がまだ使えない環境でも一応動くようにしておく
        pass

    # torchrun が渡してくれるはずの環境変数を、自分で埋める
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # MASTER_ADDR / MASTER_PORT は優先順位:
    #   1) 関数引数 (CLI で渡した値)
    #   2) すでに環境変数に入っている値
    #   3) デフォルト値
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
    args = parser.parse_args()

    # ここで MPI から env を準備
    rank, world_size, local_rank = setup_env_from_mpi(
        master_addr=args.master_addr,
        master_port=args.master_port,
    )
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    process_per_node = 2
    local_rank = rank % process_per_node

    run_zero.run_zero(args.profiler)


if __name__ == "__main__":
    main()
