#!/usr/bin/env python3
"""計算カーネルが通信カーネルを締め出していないかを、nsys から直接検証する。

背景:
  gpu_metrics_by_class.py で「通信カーネルは SM をほとんど占有しない
  (SMs Active 4〜10%)」ことが分かった。NCCL カーネルは実測で grid=2 blocks /
  544 threads しか要求していないためである。
  しかし逆方向 —— 計算カーネルが SM を埋め尽くしていて、たった 2 ブロックの
  通信カーネルすらスケジュールされずに待たされている —— は別の問題であり、
  こちらの方が「オフロードで通信が速くなる」理由になりうる。

2 つの独立した証拠を出す:

  [A] 投入→開始のキュー遅延
      cudaLaunchKernel が返ってから、カーネルが実際に走り出すまでの時間。
      SM が埋まっていてブロックを置けなければ、ここが伸びる。
      計算カーネルが走っている最中に投入されたものと、GPU が空いている
      ときに投入されたものを比較する。
      ※ 交絡: NCCL は専用ストリームにあり、計算ストリームからの event 待ちが
        入りうる。その分は SM 競合と区別できないので、値は上限として読む。

  [B] 同一メッセージの実行時間比較（自然実験）
      ZeRO-3 では毎ステップ同じ順序で同じパラメータを AllGather するので、
      「ステップ内で k 番目の通信カーネル」は 25 ステップとも同じサイズになる。
      同じ k のインスタンス同士で、計算と重なった回と重ならなかった回の
      実行時間を比べれば、メッセージサイズを制御した比較ができる。
      サイズ情報が nsys に無いという制約を、この対応づけで回避している。

使い方:
    python3 scripts/comm_kernel_crowding.py --json out.json a.sqlite b.sqlite ...
"""
import argparse
import json
import os
import sqlite3
import statistics
import sys
from bisect import bisect_left, bisect_right

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


class Cover:
    """merge 済み区間との交差長と、点が中に入るかを高速に返す。"""

    def __init__(self, iv):
        self.s = [x[0] for x in iv]
        self.e = [x[1] for x in iv]
        self.c = [0] * (len(iv) + 1)
        for i, x in enumerate(iv):
            self.c[i + 1] = self.c[i] + (x[1] - x[0])

    def _upto(self, t):
        i = bisect_right(self.s, t) - 1
        if i < 0:
            return 0
        return self.c[i] + max(0, min(t, self.e[i]) - self.s[i])

    def overlap(self, a, b):
        return self._upto(b) - self._upto(a)

    def contains(self, t):
        i = bisect_right(self.s, t) - 1
        return i >= 0 and t < self.e[i]


def analyze(db_path):
    c = sqlite3.connect(db_path)
    steps = c.execute(
        "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text='ZeroWrapperExample.forward'").fetchone()[0]
    main = c.execute(f"""
        SELECT k.streamId FROM {K} k JOIN StringIds s ON k.shortName=s.id
        WHERE s.value NOT LIKE 'ncclDevKernel%' AND s.value NOT LIKE 'CatArrayBatchedCopy%'
        GROUP BY k.streamId ORDER BY COUNT(*) DESC LIMIT 1""").fetchone()[0]

    busy = Cover(merge([[s, e] for s, e in c.execute(
        f"SELECT start,end FROM {K} WHERE streamId=?", (main,))]))

    # 解析窓: forward+backward
    win = merge([[s, e] for s, e in c.execute(
        "SELECT start,end FROM NVTX_EVENTS WHERE text IN "
        "('ZeroWrapperExample.forward','ZeroWrapperExample.backward') AND end IS NOT NULL")])
    wincov = Cover(win)

    # ステップ境界（forward の開始時刻）
    fstarts = sorted(s for (s,) in c.execute(
        "SELECT start FROM NVTX_EVENTS WHERE text='ZeroWrapperExample.forward' AND end IS NOT NULL"))

    out = {"file": os.path.basename(db_path), "steps": steps, "kinds": {}}

    for kind, pat in (("AllGather", "ncclDevKernel_AllGather%"),
                      ("ReduceScatter", "ncclDevKernel_ReduceScatter%")):
        rows = list(c.execute(f"""
            SELECT k.start, k.end, r.end
            FROM {K} k JOIN StringIds s ON k.shortName=s.id
            LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId = r.correlationId
            WHERE s.value LIKE ? ORDER BY k.start""", (pat,)))
        rows = [r for r in rows if wincov.contains(r[0])]
        if len(rows) < 20:
            continue

        # ---- [A] キュー遅延 -------------------------------------------------
        busy_delays, free_delays = [], []
        for ks, ke, rend in rows:
            if rend is None or ks < rend:
                continue
            d = (ks - rend) / 1e3            # us
            (busy_delays if busy.contains(rend) else free_delays).append(d)

        # ---- [B] 同一メッセージの実行時間 -----------------------------------
        # ステップ内の通し番号を振る（毎ステップ同じ順序で同じサイズが流れる前提）
        by_ordinal = {}
        counter = {}
        for ks, ke, _ in rows:
            si = bisect_right(fstarts, ks) - 1
            n = counter.get(si, 0)
            counter[si] = n + 1
            ov = busy.overlap(ks, ke) / max(1, ke - ks)
            by_ordinal.setdefault(n, []).append((ov, (ke - ks) / 1e3))

        # ステップ内の位置ごとに重なり条件がほぼ決定的なため、
        # 「同一 ordinal 内で重なり小 vs 大」を直接比べると群が 1 つも成立しない
        # （実測で 0 群）。そこで ordinal ごとの中央値で実行時間を正規化し、
        # サイズ差を消したうえで全インスタンスをプールして重なり率と対応づける。
        pooled = []      # (重なり率, 正規化実行時間)
        for n, lst in by_ordinal.items():
            if len(lst) < 5:
                continue
            m = statistics.median(d for _, d in lst)
            if m <= 0:
                continue
            for ov, d in lst:
                pooled.append((ov, d / m))

        def bin_mean(lo, hi):
            v = [r for ov, r in pooled if lo <= ov < hi]
            return (statistics.median(v), len(v)) if len(v) >= 10 else (None, len(v))

        bins = {"重なり<0.05": bin_mean(0.0, 0.05),
                "0.05-0.5": bin_mean(0.05, 0.5),
                "0.5-0.95": bin_mean(0.5, 0.95),
                ">=0.95": bin_mean(0.95, 1.01)}

        ovs = [ov for lst in by_ordinal.values() for ov, _ in lst]
        out["kinds"][kind] = {
            "n_kernels": len(rows),
            "mean_overlap_frac": statistics.mean(ovs) if ovs else None,
            "queue_delay_busy_us": statistics.median(busy_delays) if busy_delays else None,
            "queue_delay_free_us": statistics.median(free_delays) if free_delays else None,
            "n_busy": len(busy_delays), "n_free": len(free_delays),
            "n_pooled": len(pooled),
            "norm_dur_by_overlap": {k: {"median": v[0], "n": v[1]} for k, v in bins.items()},
            "busy_launch_frac": (len(busy_delays) / (len(busy_delays) + len(free_delays))
                                 if (busy_delays or free_delays) else None),
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
            print(f"  [{kind}] n={d['n_kernels']} 平均重なり率={d['mean_overlap_frac']:.2f}")
            print(f"    キュー遅延  計算中={d['queue_delay_busy_us']}us (n={d['n_busy']})  "
                  f"空き={d['queue_delay_free_us']}us (n={d['n_free']})")
            print(f"    投入時に計算中だった割合={d['busy_launch_frac']}")
            print(f"    サイズ正規化した実行時間 (1.0=その位置の中央値, n={d['n_pooled']}):")
            for k, v in d["norm_dur_by_overlap"].items():
                if v["median"] is not None:
                    print(f"       {k:12s} {v['median']:.3f}  (n={v['n']})")


if __name__ == "__main__":
    main()
