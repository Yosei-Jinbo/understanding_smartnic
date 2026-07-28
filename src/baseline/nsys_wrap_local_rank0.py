#!/usr/bin/env python3
"""[非推奨] torchrun 用の nsys ラッパ: LOCAL_RANK==0 の process だけ nsys profile で再 exec する。

**このファイルは使わないこと。** scripts/smartnic_offload/nsys_wrap.sh に統一済み。

非推奨の理由 (2026-07-28):
  1. LOCAL_RANK==0 しか包まないため、4 ランク中 2 ランクしかレポートが取れない。
  2. --capture-range-end=stop-shutdown を使っている。これは cudaProfilerStop() の
     瞬間に対象アプリを強制終了するため、複数ランクを包むと
     「最初に到達したランクが死ぬ → 残りが集団通信で待つ → mpirun/torchrun が
     ジョブ全体を SIGKILL → 書き出し中の nsys が道連れ」となる。
  3. 下記 docstring の「CUPTI 同居 SIGSEGV」という設計前提は、2026-07-28 に
     4 ランク同時 profiling が成功したことで否定された。

以下は歴史的経緯としての元の説明。


背景:
  `nsys profile torchrun --nproc_per_node=N ...` のように nsys が torchrun を包むと、
  torchrun が spawn した N 個の Python 子プロセス全部に CUPTI がロードされる。
  同一ノードで複数プロセスが cuEventQuery を叩き合うと CUPTI 内部で SIGSEGV が発生する
  (cudart + nccl Work の isCompleted 経路で再現)。
  smartnic_offload 側 (mpirun appfile) は rank 0 だけ nsys を頭につけているため競合しない。
  このラッパはその振る舞いを torchrun でも再現する。

使い方:
  NSYS_OUTPUT_BASE=logs/nsys/<label> \
      torchrun --nnodes=2 --nproc_per_node=2 ... \
          nsys_wrap_local_rank0.py run_zero.py --model ... [args]

  - LOCAL_RANK==0 の process: nsys profile 付きで python を exec
  - 他の LOCAL_RANK: 素の python で exec (CUPTI なし)

環境変数:
  NSYS_OUTPUT_BASE      nsys -o の出力先 (ホスト名/node_rank は自動付与)
  NSYS_EXTRA_ARGS       nsys profile に渡す追加引数 (任意, スペース区切り)
"""
import os
import sys
import shlex


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    python = sys.executable or "python"
    script_and_args = sys.argv[1:]

    if local_rank == 0:
        node_rank = os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", "0"))
        host = os.uname().nodename
        base = os.environ.get("NSYS_OUTPUT_BASE")
        if base:
            outfile = f"{base}_{host}_node{node_rank}"
        else:
            outfile = f"logs/nsys/profile_{host}_node{node_rank}"
        os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
        extra = shlex.split(os.environ.get("NSYS_EXTRA_ARGS", ""))
        # --trace/--sample/--cpuctxsw は NSYS_EXTRA_ARGS で重複指定すると壊れるため、
        # ここでは env で上書き可能にする (CPU トレースラン用: NSYS_TRACE=cuda,nvtx,osrt など)。
        cmd = [
            "/usr/local/cuda/bin/nsys", "profile",
            "--trace=" + os.environ.get("NSYS_TRACE", "cuda,nvtx"),
            "--sample=" + os.environ.get("NSYS_SAMPLE", "none"),
            "--cpuctxsw=" + os.environ.get("NSYS_CPUCTXSW", "none"),
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop-shutdown",
            "--force-overwrite=true",
            "-o", outfile,
            *extra,
            python, *script_and_args,
        ]
        print(f"[nsys_wrap] LOCAL_RANK=0 -> exec nsys: {' '.join(cmd)}", file=sys.stderr)
        os.execvp(cmd[0], cmd)
    else:
        cmd = [python, *script_and_args]
        # keep quiet for non-profiled ranks
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
