#!/usr/bin/env python3
"""nsys レポートから ZeRO-3 baseline の「1ステップあたりの通信/転送時間」を抽出する。

抽出する GPU 実行時間 (すべて 1 step あたり = 合計 ÷ N):
  AllGather      : ncclDevKernel_AllGather*      カーネル
  ReduceScatter  : ncclDevKernel_ReduceScatter*  カーネル
  full param     : NVTX 'xfer:full_param_h2d' / 'xfer:full_param_d2h' に射影した memcpy
                   (ENABLE_FULL_PARAM_TRANSFER=0 の run では 0)
  H2D  (Host->Device, copyKind=1): 合計と内訳
        - param_shard : NVTX 'xfer:param_shard_h2d'   (AllGather 前の重みシャード持ち上げ)
        - other       : 入力データ等、上記以外の H2D
  D2H  (Device->Host, copyKind=2): 合計と内訳
        - grad        : NVTX 'xfer:grad_d2h'          (ReduceScatter 後の勾配 GPU->CPU)
        - other       : 上記以外の D2H
  (参考) DtoD (copyKind=8): GPU 内コピー (NCCL 内部 / torch.cat 等)。H2D/D2H とは別枠。

N (ステップ数) = NVTX 'ZeroWrapperExample.forward' の数。
memcpy の NVTX 帰属は correlationId 経由 (memcpy を launch した runtime API の時刻が
その NVTX 区間内にあるかで判定) なので、非同期コピーでも正しく射影される。

使い方:
  python3 scripts/extract_transfer_per_step.py --sqlite <path>.sqlite
  python3 scripts/extract_transfer_per_step.py --nsys-rep <path>.nsys-rep   # 自動で sqlite を生成
  オプション: --iters N (forward NVTX が無い場合の N 上書き) / --json out.json
"""
import argparse
import bisect
import json
import os
import sqlite3
import subprocess
import sys


def ensure_sqlite(rep, sqlite_path=None):
    if sqlite_path is None:
        base = rep[:-len(".nsys-rep")] if rep.endswith(".nsys-rep") else rep
        sqlite_path = base + ".sqlite"
    if not os.path.exists(sqlite_path):
        nsys = os.environ.get("NSYS_BIN", "/usr/local/cuda/bin/nsys")
        subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true",
                        "-o", sqlite_path, rep], check=True)
    return sqlite_path


def kernel_ms(con, like):
    d = con.execute(
        """SELECT SUM(k.end - k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON k.shortName = s.id WHERE s.value LIKE ?""",
        (like,)).fetchone()[0]
    return (d or 0) / 1e6


def memcpy_total_ms(con, copy_kind):
    d = con.execute(
        "SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = ?",
        (copy_kind,)).fetchone()[0]
    return (d or 0) / 1e6


def memcpy_in_nvtx_ms(con, copy_kind, nvtx_text):
    """copyKind の memcpy のうち、それを launch した runtime API の時刻が
    NVTX 区間 nvtx_text 内にある分の合計 (ms)。"""
    rng = con.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text = ? AND end IS NOT NULL ORDER BY start",
        (nvtx_text,)).fetchall()
    if not rng:
        return 0.0
    starts = [s for s, _ in rng]
    ends = [e for _, e in rng]
    rows = con.execute(
        """SELECT r.start, (m.end - m.start) FROM CUPTI_ACTIVITY_KIND_MEMCPY m
           JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON m.correlationId = r.correlationId
           WHERE m.copyKind = ?""",
        (copy_kind,)).fetchall()
    tot = 0.0
    for t, dur in rows:
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and t <= ends[i]:
            tot += dur
    return tot / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", help="nsys export 済み sqlite")
    ap.add_argument("--nsys-rep", help=".nsys-rep (指定時は自動で sqlite を生成)")
    ap.add_argument("--iters", type=int, default=None,
                    help="ステップ数 N の上書き (forward NVTX が無い場合)")
    ap.add_argument("--json", help="結果を JSON で書き出す先")
    args = ap.parse_args()

    sq = args.sqlite or (ensure_sqlite(args.nsys_rep) if args.nsys_rep else None)
    if not sq or not os.path.exists(sq):
        sys.exit("--sqlite か --nsys-rep を指定してください")

    con = sqlite3.connect(sq)
    n = args.iters or con.execute(
        "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text = 'ZeroWrapperExample.forward'").fetchone()[0] or 0
    if not n:
        sys.exit("N=0 (forward NVTX が見つからない)。--iters で指定してください")

    ag = kernel_ms(con, "ncclDevKernel_AllGather%")
    rs = kernel_ms(con, "ncclDevKernel_ReduceScatter%")
    h2d = memcpy_total_ms(con, 1)
    d2h = memcpy_total_ms(con, 2)
    dtod = memcpy_total_ms(con, 8)
    fp_h2d = memcpy_in_nvtx_ms(con, 1, "xfer:full_param_h2d")
    fp_d2h = memcpy_in_nvtx_ms(con, 2, "xfer:full_param_d2h")
    ps_h2d = memcpy_in_nvtx_ms(con, 1, "xfer:param_shard_h2d")
    grad_d2h = memcpy_in_nvtx_ms(con, 2, "xfer:grad_d2h")
    other_h2d = max(0.0, h2d - ps_h2d - fp_h2d)
    other_d2h = max(0.0, d2h - grad_d2h - fp_d2h)
    con.close()

    rows = [
        ("AllGather (NCCL kernel)", ag),
        ("ReduceScatter (NCCL kernel)", rs),
        ("full_param H2D (xfer:full_param_h2d)", fp_h2d),
        ("full_param D2H (xfer:full_param_d2h)", fp_d2h),
        ("H2D total", h2d),
        ("  H2D param_shard (xfer:param_shard_h2d)", ps_h2d),
        ("  H2D other (input data 等)", other_h2d),
        ("D2H total", d2h),
        ("  D2H grad (xfer:grad_d2h)", grad_d2h),
        ("  D2H other", other_d2h),
        ("(参考) DtoD total", dtod),
    ]

    print(f"sqlite: {sq}")
    print(f"steps N = {n}  (NVTX 'ZeroWrapperExample.forward')")
    print(f"{'metric':<42}{'total (ms)':>12}{'per-step (ms)':>15}")
    print("-" * 69)
    for name, tot in rows:
        print(f"{name:<42}{tot:>12.3f}{tot / n:>15.3f}")

    if args.json:
        out = {"_steps": n,
               "_metrics": {name.strip(): {"total_ms": tot, "per_step_ms": tot / n}
                            for name, tot in rows}}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
