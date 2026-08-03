#!/usr/bin/env python3
"""通信が遅れているのは「カーネルが遅い」からか「発行が遅い」からかを分離する。

これまでに分かっていること:
  - 通信カーネルは grid=2 blocks しか使わず、SM 競合は起きていない
  - 計算と重なってもカーネル実行時間は変わらない（キュー遅延も 1us）
  - にもかかわらず通信の 71〜91% が露出し、GPU は 23〜60% 完全停止している

残る説明は「通信ストリームが連続的に走っていない」＝発行側の問題である。
本スクリプトはそれを直接測る。

測るもの:
  [1] 通信ストリームの稼働率 (duty cycle)
        span   = ステップ内で最初の通信カーネル開始〜最後の終了
        busy   = 通信カーネルの実行時間の総和
        duty   = busy / span
      duty が低ければ、通信ストリームは「待たされている」のではなく
      「仕事を与えられていない」。ネットワークが遊んでいる。

  [2] 発行ラグ
        前の通信カーネルが終わってから、次の通信カーネルが
        cudaLaunchKernel で投入されるまでの時間。
        ここが大きければ、ホストが次の通信を出すのが遅い。

  [3] ギャップ中にホストが何をしていたか
        通信ストリームが空いている区間を NVTX で属性づけする。
        AllGather 完了待ちの関数名が出れば、
        「ホストが AG N の完了を待ってから AG N+1 を出している」
        ＝ 直列化していることの直接証拠になる。

制約:
  SO(AG) では AllGather が DPU 側に出るため nsys からは見えない。
  AllGather については ZO のみ解釈可能。ReduceScatter は両方で見られる。

使い方:
    python3 scripts/comm_stream_gaps.py --json out.json a.sqlite ...
"""
import argparse
import json
import os
import sqlite3
import statistics
import sys
from bisect import bisect_right

K = "CUPTI_ACTIVITY_KIND_KERNEL"


def merge(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def inter(a, b):
    i = j = t = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            t += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


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


def analyze(db_path, topn=6):
    c = sqlite3.connect(db_path)
    steps = c.execute(
        "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text='ZeroWrapperExample.forward'").fetchone()[0]
    fstarts = sorted(s for (s,) in c.execute(
        "SELECT start FROM NVTX_EVENTS WHERE text='ZeroWrapperExample.forward' AND end IS NOT NULL"))

    out = {"file": os.path.basename(db_path), "steps": steps, "kinds": {}}

    for kind, pat in (("AllGather", "ncclDevKernel_AllGather%"),
                      ("ReduceScatter", "ncclDevKernel_ReduceScatter%")):
        rows = list(c.execute(f"""
            SELECT k.start, k.end, r.end
            FROM {K} k JOIN StringIds s ON k.shortName=s.id
            LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
            WHERE s.value LIKE ? ORDER BY k.start""", (pat,)))
        if len(rows) < 20:
            continue

        # ステップごとに分けて duty cycle と発行ラグを出す
        per_step = {}
        for ks, ke, rend in rows:
            si = bisect_right(fstarts, ks) - 1
            per_step.setdefault(si, []).append((ks, ke, rend))

        duties, spans, busies, gap_tot = [], [], [], []
        lags = []
        gap_iv_all = []
        for si, lst in per_step.items():
            if si < 0 or len(lst) < 2:
                continue
            lst.sort()
            iv = merge([[a, b] for a, b, _ in lst])
            span = iv[-1][1] - iv[0][0]
            busy = sum(b - a for a, b in iv)
            if span <= 0:
                continue
            spans.append(span / 1e6)
            busies.append(busy / 1e6)
            duties.append(busy / span)
            gap_tot.append((span - busy) / 1e6)
            # ギャップ区間
            gap_iv_all += sub([[iv[0][0], iv[-1][1]]], iv)
            # 発行ラグ: 前カーネル終了 → 次カーネルの launch API 終了
            for i in range(1, len(lst)):
                prev_end = lst[i - 1][1]
                rend = lst[i][2]
                if rend is not None:
                    lags.append((rend - prev_end) / 1e6)   # ms（負なら先行発行済み）

        gap_iv_all = merge(gap_iv_all)

        # ギャップ中のホスト位置（内側 NVTX 優先で排他割当）
        nv = {}
        for (t,) in c.execute(
                "SELECT DISTINCT text FROM NVTX_EVENTS WHERE text IS NOT NULL AND end IS NOT NULL"):
            ivx = [[s, e] for s, e in c.execute(
                "SELECT start,end FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL", (t,))]
            if ivx:
                nv[t] = (statistics.median(b - a for a, b in ivx), merge(ivx))
        rows_nv, rem = [], gap_iv_all
        for t in sorted(nv, key=lambda x: nv[x][0]):
            ov = inter(rem, nv[t][1])
            if ov > 0:
                rows_nv.append((ov / 1e6 / steps, t))
                rem = sub(rem, nv[t][1])
            if not rem:
                break

        out["kinds"][kind] = {
            "n_per_step": len(rows) / steps,
            "busy_ms_per_step": statistics.mean(busies) if busies else None,
            "span_ms_per_step": statistics.mean(spans) if spans else None,
            "duty": statistics.mean(duties) if duties else None,
            "gap_ms_per_step": statistics.mean(gap_tot) if gap_tot else None,
            "lag_median_ms": statistics.median(lags) if lags else None,
            "lag_p90_ms": (statistics.quantiles(lags, n=10)[8] if len(lags) > 10 else None),
            "lag_neg_frac": (sum(1 for x in lags if x < 0) / len(lags)) if lags else None,
            "gap_host": [{"name": t, "ms": v} for v, t in sorted(rows_nv, reverse=True)[:topn]],
        }
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
    for r in res:
        print(f"\n=== {r['file']} ===")
        for kind, d in r["kinds"].items():
            print(f"  [{kind}] {d['n_per_step']:.0f} 回/step")
            print(f"    busy={d['busy_ms_per_step']:.1f}ms  span={d['span_ms_per_step']:.1f}ms  "
                  f"duty={d['duty']*100:.1f}%  gap={d['gap_ms_per_step']:.1f}ms")
            print(f"    発行ラグ 中央値={d['lag_median_ms']:.3f}ms  p90={d['lag_p90_ms']}  "
                  f"先行発行済みの割合={d['lag_neg_frac']}")
            for g in d["gap_host"]:
                print(f"      ギャップ中: {g['name'][:50]:50s} {g['ms']:8.2f} ms/step")


if __name__ == "__main__":
    main()
