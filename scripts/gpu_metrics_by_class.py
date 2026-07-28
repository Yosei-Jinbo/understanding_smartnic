#!/usr/bin/env python3
"""nsys の GPU メトリクスを「何が走っている区間か」で分類して集計する。

査読 C5（通信カーネルが計算資源を奪っているのか）への直接的な証拠を出すのが目的。
セッション前半では ncu が重いため「同一形状カーネルの実行時間差」という間接推定を
使ったが、*_nsys 系のレポートには --gpu-metrics-set=ga10x が入っており、
SM 利用率も DRAM 帯域も 10kHz で直接取れている（1 ファイルあたり約 1857 万サンプル）。

■ 区間の分類（排他）
    compute_only  主ストリームの計算カーネルのみ（通信カーネルなし）＝ 干渉なしの基準
    compute+comm  計算カーネルと NCCL カーネルが**同時に**走っている ＝ 干渉あり
    comm_only     NCCL カーネルのみ ＝ 通信カーネル単体の資源消費
    dpu_wait      DPU 完了待ち（オフロード時。GPU ストリームは止まっている）
    idle          どれでもない

  compute_only と compute+comm を比べれば「通信が同居したとき計算が何を失うか」、
  comm_only を見れば「通信カーネル自体がどれだけ SM を占有するか」が分かる。

■ 主要指標
    SMs Active           SM が 1 つ以上の warp を持っている割合
    SM Issue             発行スロット利用率（実際に命令を出せているか）
    Tensor Active        Tensor Core 稼働率
    Compute Warps Avg    実効オキュパンシ
    Unallocated Warps    ★SM は Active だが warp が埋まっていない量。
                          通信カーネルが SM を押さえて計算 warp を入れられない
                          状態の直接的な指標
    DRAM R/W BW          メモリ帯域。SM 競合ではなく帯域競合かの切り分けに使う

■ 注意
    10kHz = 100us 間隔のサンプリングなので、個々のカーネル単位の値は出せない。
    区間をまとめた統計としてのみ意味を持つ。

使い方:
    python3 scripts/gpu_metrics_by_class.py --json out.json a.sqlite b.sqlite ...
"""
import argparse
import json
import os
import sqlite3
import sys
from bisect import bisect_left, bisect_right

K = "CUPTI_ACTIVITY_KIND_KERNEL"

# 見たい指標。metricName の**完全一致**で引く。
#
# [Avg] / [Avg Warps per Cycle] 系は使わない:
#   - 前方一致だと "[Avg]" が "[Avg Warps per Cycle]" にも当たる
#   - value が INTEGER 列に別スケールで格納されており、そのまま平均すると
#     数百万や負値になる（実測で確認）
# [Throughput %] 系は 0..100 に正規化されていて素直に平均できる。
METRICS = [
    ("SMs Active %",      "SMs Active [Throughput %]"),
    ("SM Issue %",        "SM Issue [Throughput %]"),
    ("Tensor Active %",   "Tensor Active [Throughput %]"),
    ("ComputeWarps %",    "Compute Warps in Flight [Throughput %]"),
    ("UnallocWarps %",    "Unallocated Warps in Active SMs [Throughput %]"),
    ("DRAM Read %",       "DRAM Read Bandwidth [Throughput %]"),
    ("DRAM Write %",      "DRAM Write Bandwidth [Throughput %]"),
]


def merge(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def sub(base, cut):
    out, j = [], 0
    for a, b in base:
        cur = a
        while j < len(cut) and cut[j][1] <= cur:
            j += 1
        k = j
        while k < len(cut) and cut[k][0] < b:
            if cut[k][0] > cur:
                out.append([cur, min(cut[k][0], b)])
            cur = max(cur, cut[k][1])
            k += 1
        if cur < b:
            out.append([cur, b])
    return [x for x in out if x[1] > x[0]]


def intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append([lo, hi])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def clip(iv, win):
    out, j = [], 0
    for a, b in iv:
        while j < len(win) and win[j][1] <= a:
            j += 1
        k = j
        while k < len(win) and win[k][0] < b:
            lo, hi = max(a, win[k][0]), min(b, win[k][1])
            if hi > lo:
                out.append([lo, hi])
            k += 1
    return merge(out)


def total(iv):
    return sum(b - a for a, b in iv)


def sample_stats(ts, vals, intervals):
    """区間集合に入るサンプルの平均・件数を返す。

    サンプルは時刻でソート済みなので、区間ごとに二分探索で範囲を取る。
    全サンプルを区間数だけ走査すると 1857万 × 区間数になって終わらない。
    """
    n = 0
    s = 0.0
    for a, b in intervals:
        i = bisect_left(ts, a)
        j = bisect_right(ts, b)
        if j > i:
            n += j - i
            s += sum(vals[i:j])
    return (s / n if n else None), n


def analyze(db_path):
    c = sqlite3.connect(db_path)
    tabs = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "GPU_METRICS" not in tabs:
        raise SystemExit(f"{db_path}: GPU_METRICS が無い（NSYS_GPU_METRICS=0 で採取された？）")

    steps = c.execute(
        "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text='ZeroWrapperExample.forward'").fetchone()[0]

    main = c.execute(f"""
        SELECT k.streamId FROM {K} k JOIN StringIds s ON k.shortName=s.id
        WHERE s.value NOT LIKE 'ncclDevKernel%' AND s.value NOT LIKE 'CatArrayBatchedCopy%'
        GROUP BY k.streamId ORDER BY COUNT(*) DESC LIMIT 1""").fetchone()[0]

    def kern(where, args=()):
        return merge([[s, e] for s, e in c.execute(
            f"SELECT k.start,k.end FROM {K} k JOIN StringIds s ON k.shortName=s.id WHERE {where}",
            args)])

    busy = kern("k.streamId=?", (main,))
    ag = kern("s.value LIKE 'ncclDevKernel_AllGather%'")
    rs = kern("s.value LIKE 'ncclDevKernel_ReduceScatter%'")
    nccl = kern("s.value LIKE 'ncclDevKernel%'")

    # 解析窓は forward+backward（step: 系の最適化区間は計算資源の議論に無関係）
    win = merge([[s, e] for s, e in c.execute(
        "SELECT start,end FROM NVTX_EVENTS WHERE text IN "
        "('ZeroWrapperExample.forward','ZeroWrapperExample.backward') AND end IS NOT NULL")])

    busy = clip(busy, win)
    ag = clip(ag, win)
    rs = clip(rs, win)
    nccl = clip(nccl, win)

    # DPU 完了待ち: 主ストリームのカーネル間ギャップのうち、
    # cuStreamWaitValue32 を挟んだもの（API 区間では 0.3ms にしかならない）
    dpu = []
    ks = list(c.execute(f"""
        SELECT k.start,k.end,r.start FROM {K} k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
        WHERE k.streamId=? ORDER BY r.start""", (main,)))
    if ks:
        hs = [x[2] for x in ks]
        for (t,) in c.execute("""
                SELECT r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN StringIds s ON r.nameId=s.id
                WHERE s.value='cuStreamWaitValue32_v2' ORDER BY r.start"""):
            i = bisect_right(hs, t) - 1
            if 0 <= i < len(ks) - 1 and ks[i + 1][0] > ks[i][1]:
                dpu.append([ks[i][1], ks[i + 1][0]])
        dpu = clip(merge(dpu), win)

    # 他ストリームのカーネル（torch.cat 等）。idle に混ぜると
    # 「GPU 完全停止のはずなのに SMs Active が数 %」という誤読になる。
    other_ks = clip(kern("k.streamId<>? AND s.value NOT LIKE 'ncclDevKernel%'", (main,)), win)
    other_ks = sub(other_ks, busy)

    classes = {
        "compute_only": sub(busy, nccl),
        "compute+comm": intersect(busy, nccl),
        "comm_only": sub(nccl, busy),
        "comm_only:AG": sub(ag, busy),
        "comm_only:RS": sub(rs, busy),
        "other_stream": sub(other_ks, nccl),
        "dpu_wait": sub(sub(sub(dpu, busy), nccl), other_ks),
        "idle": sub(sub(sub(win, busy), nccl), other_ks),
    }

    # metricId 解決（完全一致）
    mid = {}
    names = {name: i for i, name in
             c.execute("SELECT metricId,metricName FROM TARGET_INFO_GPU_METRICS")}
    for label, exact in METRICS:
        if exact in names:
            mid[label] = names[exact]
        else:
            sys.stderr.write(f"  [warn] 指標が見つからない: {exact}\n")

    out = {"file": os.path.basename(db_path), "steps": steps,
           "durations": {k: total(v) / 1e6 / steps for k, v in classes.items()},
           "metrics": {}}

    for label, _ in METRICS:
        if label not in mid:
            continue
        rows = list(c.execute(
            "SELECT timestamp,value FROM GPU_METRICS WHERE metricId=? ORDER BY timestamp",
            (mid[label],)))
        ts = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        out["metrics"][label] = {}
        for cname, iv in classes.items():
            if not iv:
                continue
            m, n = sample_stats(ts, vals, iv)
            if n:
                out["metrics"][label][cname] = {"mean": m, "n": n}
    c.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = []
    for f in a.sqlite:
        sys.stderr.write(f"[解析中] {os.path.basename(f)}\n")
        sys.stderr.flush()
        res.append(analyze(f))
    if a.json:
        json.dump({"results": res}, open(a.json, "w"), ensure_ascii=False)
        sys.stderr.write(f"[JSON] {a.json}\n")
    for r in res:
        print(f"\n=== {r['file']} (steps={r['steps']}) ===")
        print("  区間長 [ms/step]: " +
              "  ".join(f"{k}={v:.1f}" for k, v in r["durations"].items() if v > 0.01))
        for label, per in r["metrics"].items():
            print(f"  {label:18s}" +
                  "  ".join(f"{k}={v['mean']:.1f}" for k, v in per.items()))


if __name__ == "__main__":
    main()
