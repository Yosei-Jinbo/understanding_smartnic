#!/usr/bin/env python3
"""全構成×全指標を、計測手法ごとに正しいソースから抽出・統合して出力する。

ソースの振り分け (EXPERIMENT.md §4.5 の手法):
  Step / Grad GPUtoCPU / Param CPUtoGPU : 非nsys host ログ
      step = step_total mean,  Grad = grad D2H ÷ N_host,  Param = param H2D ÷ N_host
  AllGather:
      smartnic / ag_smartnic : DPU union tracker "AG busy" ÷ N_nsys   (DPU ログ)
      prefetch               : nsys ncclDevKernel_AllGather ÷ N_nsys  (nsys)
  ReduceScatter:
      smartnic               : DPU union tracker "RS busy" ÷ N_nsys   (DPU ログ)
      ag_smartnic / prefetch : nsys ncclDevKernel_ReduceScatter ÷ N_nsys (nsys)

iter 数:
  N_host = 非nsys host ログの step_total count
  N_nsys = nsys sqlite の forward NVTX 数 (無ければ --iters)。DPU は同一 nsys run 前提で N_nsys を使用。

使い方:
  python3 scripts/extract_all_metrics.py \
      --host-dir logs/smartnic_offload --host-dir logs/baseline \
      --nsys-dir logs/smartnic_offload/nsys --dpu-dir logs/dpu [--iters 25]
"""
import argparse
import glob
import os
import re
import sqlite3
import subprocess

MODELS = ["opt_1.3b", "deberta_xl", "vit_l_16"]
CONFIGS = ["smartnic", "ag_smartnic", "prefetch"]
AG_SRC = {"smartnic": "dpu", "ag_smartnic": "dpu", "prefetch": "nsys"}
RS_SRC = {"smartnic": "dpu", "ag_smartnic": "nsys", "prefetch": "nsys"}
NSYS_MODEL = {"opt_1.3b": "opt", "deberta_xl": "deberta_xl", "vit_l_16": "vit_l_16"}


# ---------- host (非nsys) ----------
def find_host_log(host_dirs, model, cfg):
    for d in host_dirs:
        for name in (f"{model}_{cfg}.log", f"host_{model}_{cfg}.log"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


def parse_host(path):
    t = open(path, "r", errors="replace").read()
    o = {}
    m = list(re.finditer(r"step_total:\s*count=\s*(\d+).*?mean=([\d.]+)ms", t))
    if m:
        o["N"] = int(m[-1].group(1)); o["step"] = float(m[-1].group(2))
    for k, pat in (("grad", r"grad D2H:\s*([\d.]+)s"), ("param", r"param H2D:\s*([\d.]+)s")):
        mm = list(re.finditer(pat, t))
        if mm:
            o[k] = float(mm[-1].group(1))
    return o


# ---------- nsys ----------
# nsys 出力の命名は 2 系統ある:
#   smartnic 側 (appfile の -o):  <nm>_<cfg>_rank0.{sqlite,nsys-rep}
#   baseline 側 (run_*_nsys.sh):  <nm>_buf_<cfg>_<hostname>_node<N>.{sqlite,nsys-rep}
# 後者はホスト名/ノード番号が入るので glob で拾う。
def _nsys_basename_patterns(model, cfg):
    nm = NSYS_MODEL[model]
    return [f"{nm}_{cfg}_rank0",              # smartnic 系
            f"{nm}_buf_{cfg}_*_node*",        # baseline 系 (prefetch)
            f"{nm}_{cfg}_*_node*"]            # 予備


def ensure_sqlite(nsys_dirs, model, cfg):
    """nsys_dirs (複数可) から該当 sqlite を探す。無ければ .nsys-rep から生成する。"""
    if isinstance(nsys_dirs, str):
        nsys_dirs = [nsys_dirs]
    reps = []
    for d in nsys_dirs:
        if not d:
            continue
        for pat in _nsys_basename_patterns(model, cfg):
            hits = sorted(glob.glob(os.path.join(d, pat + ".sqlite")))
            if hits:
                return hits[0]
            reps += sorted(glob.glob(os.path.join(d, pat + ".nsys-rep")))
    if reps:
        rep = reps[0]
        nsys = os.environ.get("NSYS_BIN", "/usr/local/cuda/bin/nsys")
        subprocess.run([nsys, "stats", "--force-export=true",
                        "--report", "cuda_gpu_kern_sum", rep],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sq = rep[: -len(".nsys-rep")] + ".sqlite"
        return sq if os.path.exists(sq) else None
    return None


def nsys_kernel_ms_per(sqlite_path, like):
    """ncclDevKernel_* 合計(ms) と forward NVTX 数(cap) を返す。"""
    con = sqlite3.connect(sqlite_path)
    cap = con.execute(
        "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text='ZeroWrapperExample.forward'").fetchone()[0] or 0
    tot = con.execute("""SELECT SUM(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k
                         JOIN StringIds s ON k.shortName=s.id WHERE s.value LIKE ?""",
                      (like,)).fetchone()[0] or 0
    con.close()
    return tot / 1e6, cap


# ---------- DPU union tracker ----------
def find_dpu_log(dpu_dir, model, cfg):
    if not dpu_dir:
        return None
    for name in (f"dpu_{model}_{cfg}_nsys.log", f"dpu_{model}_{cfg}.log"):
        p = os.path.join(dpu_dir, name)
        if os.path.exists(p):
            return p
    return None


def parse_dpu(path):
    t = open(path, "r", errors="replace").read()
    o = {}
    m = re.search(r"AG  busy \(union\):\s*([\d.]+)\s*s", t)
    if m:
        o["ag_s"] = float(m.group(1))
    m = re.search(r"RS  busy \(union\):\s*([\d.]+)\s*s", t)
    if m:
        o["rs_s"] = float(m.group(1))
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host-dir", action="append", default=[], help="非nsys host ログ dir (複数可)")
    ap.add_argument("--nsys-dir", action="append", default=[],
                    help="nsys sqlite/.nsys-rep dir (複数可)")
    ap.add_argument("--dpu-dir", default=None, help="DPU ログ dir")
    ap.add_argument("--iters", type=int, default=25, help="nsys/DPU の iter 既定値 (sqlite cap 優先)")
    ap.add_argument("--only", default=None, help="model,config を1件だけ (例 deberta_xl,smartnic)")
    args = ap.parse_args()
    host_dirs = args.host_dir or ["logs/smartnic_offload", "logs/baseline"]

    targets = [(m, c) for m in MODELS for c in CONFIGS]
    if args.only:
        mm, cc = args.only.split(","); targets = [(mm, cc)]

    def fmt(x):
        return f"{x:8.1f}" if x is not None else "     N/A"

    print(f"{'config':24s}{'metric':6s}{'value(ms)':>10}  source")
    print("-" * 56)
    for (model, cfg) in targets:
        # --- host (step/grad/param) ---
        hp = find_host_log(host_dirs, model, cfg)
        h = parse_host(hp) if hp else {}
        Nh = h.get("N")
        step = h.get("step")
        grad = (h["grad"] * 1000.0 / Nh) if (Nh and "grad" in h) else None
        param = (h["param"] * 1000.0 / Nh) if (Nh and "param" in h) else None

        # --- nsys (cap + kernels) ---
        sq = ensure_sqlite(args.nsys_dir, model, cfg) if args.nsys_dir else None

        cap = None
        ag_nsys = rs_nsys = None
        if sq:
            ag_ms, cap = nsys_kernel_ms_per(sq, "ncclDevKernel_AllGather%")
            rs_ms, cap2 = nsys_kernel_ms_per(sq, "ncclDevKernel_ReduceScatter%")
            cap = cap or cap2 or args.iters
            if cap:
                ag_nsys = ag_ms / cap
                rs_nsys = rs_ms / cap
        N_nsys = cap or args.iters

        # --- DPU union tracker ---
        dp = find_dpu_log(args.dpu_dir, model, cfg)
        d = parse_dpu(dp) if dp else {}
        ag_dpu = (d["ag_s"] * 1000.0 / N_nsys) if ("ag_s" in d and N_nsys) else None
        rs_dpu = (d["rs_s"] * 1000.0 / N_nsys) if ("rs_s" in d and N_nsys) else None

        # --- route AG / RS ---
        if AG_SRC[cfg] == "dpu":
            ag, ag_src = ag_dpu, f"DPU÷{N_nsys}"
        else:
            ag, ag_src = ag_nsys, f"nsysAGk÷{N_nsys}"
        if RS_SRC[cfg] == "dpu":
            rs, rs_src = rs_dpu, f"DPU÷{N_nsys}"
        else:
            rs, rs_src = rs_nsys, f"nsysRSk÷{N_nsys}"

        # データが全く無い構成はスキップ
        if step is None and ag is None and rs is None and grad is None and param is None:
            continue

        name = f"{model}/{cfg}"
        sh = f"host÷{Nh}" if Nh else "host:NA"
        print(f"{name:24s}{'step':6s}{fmt(step)}  {sh}")
        print(f"{'':24s}{'AG':6s}{fmt(ag)}  {ag_src}")
        print(f"{'':24s}{'RS':6s}{fmt(rs)}  {rs_src}")
        print(f"{'':24s}{'Grad':6s}{fmt(grad)}  {sh}")
        print(f"{'':24s}{'Param':6s}{fmt(param)}  {sh}")
        print()


if __name__ == "__main__":
    main()
