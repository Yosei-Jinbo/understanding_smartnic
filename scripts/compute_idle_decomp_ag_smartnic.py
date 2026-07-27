#!/usr/bin/env python3
"""ag_smartnic の nsys トレースから compute(busy) / idle 時間を分解して出す。

HasegawaLab/performance_evaluation/figures/compute_idle_decomp.py のロジックを
そのまま移植し、CONFIG_PATHS ハードコードをやめて任意のファイルを引数で渡せる
ようにしたもの

各 NVTX 区間 (ZeroWrapperExample.{forward|backward}) 内で:
  busy = compute ストリームの GPU カーネルが動いていた時間
  idle = wall - busy。さらに優先度 AG > RS > memcpy > DPU-AG-wait > host で分解
  ag_smartnic では AG は DPU オフロードなので idle の大半は dpu_ag_wait / host になる

使い方:
  python3 scripts/compute_idle_decomp_ag_smartnic.py \
      --sqlite logs/smartnic_offload/nsys/<model>_ag_smartnic_rank0.sqlite [--label deberta_xl]
  # .nsys-rep しか無い場合 (自動で sqlite 生成):
  python3 scripts/compute_idle_decomp_ag_smartnic.py \
      --nsys-rep logs/smartnic_offload/nsys/<model>_ag_smartnic_rank0.nsys-rep
  # JSON 出力したい場合: --json out.json
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from bisect import bisect_left, bisect_right

PHASES = {
    "fwd": "ZeroWrapperExample.forward",
    "bwd": "ZeroWrapperExample.backward",
}


# ----------------------------------------------------------------------------
# 以下 compute_idle_decomp.py からの移植 (ロジック不変)
# ----------------------------------------------------------------------------
def fetch_intervals(conn, query, params=()):
    rows = conn.execute(query, params).fetchall()
    return [(s, e) for (s, e) in rows if e is not None and e > s]


def find_compute_stream(conn):
    rows = conn.execute("""
        SELECT k.streamId, COUNT(*) AS n
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE s.value NOT LIKE 'ncclDevKernel%'
          AND s.value NOT LIKE 'CatArrayBatchedCopy%'
        GROUP BY k.streamId ORDER BY n DESC LIMIT 1
    """).fetchall()
    if not rows:
        raise RuntimeError("no compute stream found")
    return rows[0][0]


def merge_intervals(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def split_arrays(ivs):
    return [s for s, _ in ivs], [e for _, e in ivs]


def clip_to_window(ivs, starts, ends, ws, we):
    if not ivs:
        return []
    lo = bisect_right(ends, ws)
    hi = bisect_left(starts, we)
    out = []
    for i in range(lo, hi):
        s, e = ivs[i]
        a = max(s, ws)
        b = min(e, we)
        if b > a:
            out.append((a, b))
    return out


def sweep_idle_classify(idle_subs, ag_clipped, rs_clipped, mem_clipped, dpu_clipped):
    """idle 区間を AG > RS > mem > DPU-AG-wait > host の優先度で分類。"""
    ag = rs = mem = dpu = host = 0
    if not idle_subs:
        return 0, 0, 0, 0, 0

    ag_starts, ag_ends = split_arrays(ag_clipped)
    rs_starts, rs_ends = split_arrays(rs_clipped)
    mem_starts, mem_ends = split_arrays(mem_clipped)
    dpu_starts, dpu_ends = split_arrays(dpu_clipped)

    def collect_events(ivs, starts, ends, k_idx, ws, we, events):
        if not ivs:
            return
        lo = bisect_right(ends, ws)
        hi = bisect_left(starts, we)
        for i in range(lo, hi):
            s, e = ivs[i]
            a = max(s, ws)
            b = min(e, we)
            if b > a:
                events.append((a, +1, k_idx))
                events.append((b, -1, k_idx))

    for ws, we in idle_subs:
        events = []
        collect_events(ag_clipped,  ag_starts,  ag_ends,  0, ws, we, events)
        collect_events(rs_clipped,  rs_starts,  rs_ends,  1, ws, we, events)
        collect_events(mem_clipped, mem_starts, mem_ends, 2, ws, we, events)
        collect_events(dpu_clipped, dpu_starts, dpu_ends, 3, ws, we, events)
        events.sort()

        active = [0, 0, 0, 0]
        prev_t = ws

        def credit(dt):
            nonlocal ag, rs, mem, dpu, host
            if active[0] > 0:
                ag += dt
            elif active[1] > 0:
                rs += dt
            elif active[2] > 0:
                mem += dt
            elif active[3] > 0:
                dpu += dt
            else:
                host += dt

        for t, delta, k in events:
            if t > prev_t:
                credit(t - prev_t)
                prev_t = t
            active[k] += delta
        if we > prev_t:
            credit(we - prev_t)

    return ag, rs, mem, dpu, host


def analyze_window(window_start, window_end,
                   compute_ivs, compute_starts, compute_ends,
                   ag_ivs, ag_starts, ag_ends,
                   rs_ivs, rs_starts, rs_ends,
                   mem_ivs, mem_starts, mem_ends,
                   dpu_ivs, dpu_starts, dpu_ends):
    busy_clipped = clip_to_window(compute_ivs, compute_starts, compute_ends,
                                  window_start, window_end)
    busy = sum(b - a for a, b in busy_clipped)

    idle_subs = []
    cursor = window_start
    for a, b in busy_clipped:
        if a > cursor:
            idle_subs.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < window_end:
        idle_subs.append((cursor, window_end))

    if not idle_subs:
        return busy, 0, 0, 0, 0, 0

    ag_w = clip_to_window(ag_ivs, ag_starts, ag_ends, window_start, window_end)
    rs_w = clip_to_window(rs_ivs, rs_starts, rs_ends, window_start, window_end)
    mem_w = clip_to_window(mem_ivs, mem_starts, mem_ends, window_start, window_end)
    dpu_w = clip_to_window(dpu_ivs, dpu_starts, dpu_ends, window_start, window_end)

    ag, rs, mem, dpu, host = sweep_idle_classify(idle_subs, ag_w, rs_w, mem_w, dpu_w)
    return busy, ag, rs, mem, dpu, host


def fetch_dpu_ag_wait_gaps(conn, csid):
    """compute ストリームの kernel-gap のうち cuStreamWaitValue32 を含むもの
    (= DPU AllGather 完了待ち) を返す。"""
    rows = conn.execute("""
        SELECT k.start, k.end, r.start AS host_start
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
        WHERE k.streamId = ?
        ORDER BY r.start
    """, (csid,)).fetchall()
    if not rows:
        return []
    host_starts = [r[2] for r in rows]
    gpu_starts = [r[0] for r in rows]
    gpu_ends = [r[1] for r in rows]

    waits = conn.execute("""
        SELECT r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value = 'cuStreamWaitValue32_v2'
        ORDER BY r.start
    """).fetchall()
    if not waits:
        return []

    marked = set()
    for (T,) in waits:
        i = bisect_right(host_starts, T) - 1
        if i < 0 or i >= len(host_starts) - 1:
            continue
        marked.add(i)

    gaps = []
    for i in sorted(marked):
        ws = gpu_ends[i]
        we = gpu_starts[i + 1]
        if we > ws:
            gaps.append((ws, we))
    return merge_intervals(gaps)


def analyze_sqlite(path):
    conn = sqlite3.connect(path)
    csid = find_compute_stream(conn)

    compute_ivs = merge_intervals(fetch_intervals(conn, """
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId = ?
    """, (csid,)))
    dpu_wait_ivs = fetch_dpu_ag_wait_gaps(conn, csid)
    ag_ivs = merge_intervals(fetch_intervals(conn, """
        SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE s.value LIKE 'ncclDevKernel_AllGather%'
    """))
    rs_ivs = merge_intervals(fetch_intervals(conn, """
        SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE s.value LIKE 'ncclDevKernel_ReduceScatter%'
    """))
    mem_ivs = merge_intervals(fetch_intervals(conn, """
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
    """))

    compute_starts, compute_ends = split_arrays(compute_ivs)
    ag_starts, ag_ends = split_arrays(ag_ivs)
    rs_starts, rs_ends = split_arrays(rs_ivs)
    mem_starts, mem_ends = split_arrays(mem_ivs)
    dpu_starts, dpu_ends = split_arrays(dpu_wait_ivs)

    out = {}
    for phase, label in PHASES.items():
        nvtx = conn.execute("""
            SELECT start, end FROM NVTX_EVENTS
            WHERE text = ? AND end IS NOT NULL ORDER BY start
        """, (label,)).fetchall()
        wall = busy = ag = rs = mem = dpu = host = 0
        for s, e in nvtx:
            wall += e - s
            b, a, r, m, d, h = analyze_window(
                s, e,
                compute_ivs, compute_starts, compute_ends,
                ag_ivs, ag_starts, ag_ends,
                rs_ivs, rs_starts, rs_ends,
                mem_ivs, mem_starts, mem_ends,
                dpu_wait_ivs, dpu_starts, dpu_ends,
            )
            busy += b; ag += a; rs += r; mem += m; dpu += d; host += h
        out[phase] = dict(n=len(nvtx), wall=wall, busy=busy, idle=wall - busy,
                          ag=ag, rs=rs, mem=mem, all_idle=dpu + host,
                          dpu_ag_wait=dpu, host_overhead=host)
    conn.close()
    return out


# ----------------------------------------------------------------------------
def ensure_sqlite(nsys_rep):
    base = nsys_rep[:-len(".nsys-rep")] if nsys_rep.endswith(".nsys-rep") else nsys_rep
    sqlite_path = base + ".sqlite"
    if not os.path.exists(sqlite_path):
        nsys = os.environ.get("NSYS_BIN", "/usr/local/cuda/bin/nsys")
        subprocess.run([nsys, "stats", "--force-export=true",
                        "--report", "cuda_gpu_kern_sum", nsys_rep],
                       check=True, stdout=subprocess.DEVNULL)
    return sqlite_path


def _ms(ns):
    return ns / 1e6


def report(label, per_phase):
    cols = ["wall", "busy", "idle", "ag", "rs", "mem", "dpu_ag_wait", "host_overhead"]
    print(f"=== ({label!r}, 'ag_smartnic') compute/idle 分解 ===")
    for phase in ("fwd", "bwd"):
        d = per_phase[phase]
        n = d["n"] or 1
        print(f"\n[{phase}] n={d['n']} steps")
        print(f"  {'(ms total)':<16}{'(ms/step)':>12}")
        for c in cols:
            print(f"  {c:<14}{_ms(d[c]):>10.1f}  {_ms(d[c])/n:>10.2f}")
    # fwd+bwd 合算 (1step あたり)
    n = (per_phase['fwd']['n'] or 1)
    print(f"\n[fwd+bwd] 1step あたり (÷{n})")
    for c in cols:
        tot = per_phase['fwd'][c] + per_phase['bwd'][c]
        print(f"  {c:<14}{_ms(tot)/n:>10.2f} ms")
    # 要約: busy(compute) と idle
    busy = (per_phase['fwd']['busy'] + per_phase['bwd']['busy'])
    idle = (per_phase['fwd']['idle'] + per_phase['bwd']['idle'])
    wall = (per_phase['fwd']['wall'] + per_phase['bwd']['wall'])
    print(f"\n  => compute(busy) = {_ms(busy)/n:.1f} ms/step, "
          f"idle = {_ms(idle)/n:.1f} ms/step "
          f"(wall fwd+bwd = {_ms(wall)/n:.1f} ms/step)")

    # ------------------------------------------------------------------
    # figure_pdf/resource_busy_idle_*.pdf / resource_idle_decomp_*.pdf と同じ集計
    #   plot_resource_liberation_split.py: (fwd+bwd) を合算し ÷n (per iter)
    #   busy_idle  : Compute Kernel(busy) + Idle
    #   idle_decomp: AllGather(ag+dpu_ag_wait) / ReduceScatter(rs) /
    #                Memcpy(mem) / Other(host_overhead)   ※合計 = idle
    # ------------------------------------------------------------------
    def agg(k):
        return _ms(per_phase['fwd'][k] + per_phase['bwd'][k]) / n
    print("\n  --- resource_busy_idle_*.pdf と整合 (ms/iter) ---")
    print(f"    Compute Kernel : {agg('busy'):>8.1f}")
    print(f"    Idle           : {agg('idle'):>8.1f}")
    print("  --- resource_idle_decomp_*.pdf と整合 (ms/iter, 合計=Idle) ---")
    ag_dec = agg('ag') + agg('dpu_ag_wait')
    print(f"    AllGather      : {ag_dec:>8.1f}   (ag + dpu_ag_wait)")
    print(f"    ReduceScatter  : {agg('rs'):>8.1f}")
    print(f"    Memcpy         : {agg('mem'):>8.1f}")
    print(f"    Other          : {agg('host_overhead'):>8.1f}   (host_overhead)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sqlite", help="nsys sqlite (*.sqlite)")
    g.add_argument("--nsys-rep", help="nsys レポート (*.nsys-rep) — sqlite を自動生成")
    ap.add_argument("--label", default=None, help="出力ラベル (例: deberta_xl)")
    ap.add_argument("--json", default=None, nargs="?", const="-",
                    help="結果を JSON 出力。パス省略時または '-' で標準出力に出す")
    args = ap.parse_args()

    sqlite_path = args.sqlite or ensure_sqlite(args.nsys_rep)
    if not os.path.exists(sqlite_path):
        ap.error(f"sqlite not found: {sqlite_path}")

    label = args.label or re.sub(
        r"(_ag_smartnic.*|_rank\d+.*)$", "",
        re.sub(r"^host_", "", os.path.basename(sqlite_path)))

    per_phase = analyze_sqlite(sqlite_path)
    report(label, per_phase)

    if args.json is not None:
        pretty = {ph: {k: (_ms(v) if k != "n" else v) for k, v in d.items()}
                  for ph, d in per_phase.items()}
        payload = {label: {"ag_smartnic": pretty}}
        if args.json == "-":
            print()
            json.dump(payload, sys.stdout, indent=2)
            print()
        else:
            with open(args.json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
