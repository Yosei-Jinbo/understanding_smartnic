#!/usr/bin/env python3
"""nsys sqlite を全ランク分まとめて解析し、計算 / アイドル / "Other" を排他分解する。

idle_decomp_detailed.py（rank0 単独）からの拡張点:
  1. 複数ランクを一度に処理し、ランク間のばらつき（min/max）まで出す
     → 2026-07-28 の実測で step:barrier に最大 741ms のスキューがあり、
       rank0 だけを見た分解は代表性に欠けることが判明したため。
  2. compute(busy) をカーネル名で内訳分解する（何を計算しているのか）
  3. アイドル側の各クラスもカーネル名まで下ろす（ag/rs/other_stream_kernel）
  4. "Other"(true_idle) を NVTX で特定したうえで、意味カテゴリに集約する
  5. host_api を API 名で内訳分解する（起動オーバーヘッドの正体）
  6. --json で機械可読出力（ノードをまたいでマージするため）

区間演算の方針:
  - すべて排他割当。上のクラスに取られた区間は下のクラスには渡さない。
  - クラス順序は「GPU が実際に仕事をしているもの」→「待ち」の順。
  - 最後に残った区間が true_idle（全ストリームでカーネルも memcpy も無い）＝ Fig.7 の "Other"。

使い方:
    python3 scripts/idle_decomp_allrank.py --label opt_ag_smartnic a.sqlite b.sqlite ...
    python3 scripts/idle_decomp_allrank.py --label X --json out.json *.sqlite
"""
import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
from bisect import bisect_right

K = "CUPTI_ACTIVITY_KIND_KERNEL"


# ---------------------------------------------------------------- 区間演算
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
    """base から cut を除いた区間。どちらも merge 済みであること。"""
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


def inter(a, b):
    """交差の総長。"""
    i = j = t = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            t += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


def clip(iv, win):
    """iv を win で切る。二重ループだと区間数の積になるので二点走査で線形にする。"""
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


class Cover:
    """merge 済み区間集合との交差長を O(log n) で返す累積和。

    名前別内訳では「数百万個のカーネル/API 区間」×「数千個の窓区間」を突き合わせるため、
    素直な二重ループだと現実的な時間で終わらない（実測で 10 分超えて未完了）。
    窓側を累積長にしておき、二分探索 2 回の差分で交差長を出す。
    """

    def __init__(self, win):
        self.s = [w[0] for w in win]
        self.e = [w[1] for w in win]
        self.c = [0] * (len(win) + 1)
        for i, w in enumerate(win):
            self.c[i + 1] = self.c[i] + (w[1] - w[0])

    def _upto(self, x):
        i = bisect_right(self.s, x) - 1
        if i < 0:
            return 0
        return self.c[i] + max(0, min(x, self.e[i]) - self.s[i])

    def inter(self, a, b):
        return self._upto(b) - self._upto(a)


# ---------------------------------------------------------------- 意味カテゴリ
# true_idle 中のホスト位置(NVTX)を、査読の議論単位に集約する。
# 順序が優先度。最初にマッチしたものを採用する。
CATEGORIES = [
    ("AG関連(パラメータ取得)",
     r"all_gather|allgather|fetch_sub_module|_all_gather_params|prefetch|"
     r"AllGatherCoalesced|param_coordinator"),
    ("勾配処理",
     r"grad|reduce_scatter|reduce_and_remove|partition_grads|ipg_|bucket"),
    ("パラメータ管理(解放/分割)",
     r"free_param|release|partition_param|_partition|padding"),
    ("フック/フレーム",
     r"forward|backward|hook|module|Zero.*Example"),
]


def categorize(text):
    for name, pat in CATEGORIES:
        if re.search(pat, text, re.I):
            return name
    return "その他"


# ---------------------------------------------------------------- 1 ランク解析
def analyze(db_path, topn=8):
    c = sqlite3.connect(db_path)
    c.execute("PRAGMA temp_store=MEMORY")

    def nvtx(text):
        return merge([[s, e] for s, e in c.execute(
            "SELECT start,end FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL", (text,))])

    steps = c.execute(
        "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text='ZeroWrapperExample.forward'"
    ).fetchone()[0]
    if not steps:
        raise SystemExit(f"{db_path}: ZeroWrapperExample.forward の NVTX が無い")

    # 主計算ストリーム = nccl/cat 以外のカーネルが最も多いストリーム
    main = c.execute(f"""
        SELECT k.streamId FROM {K} k JOIN StringIds s ON k.shortName=s.id
        WHERE s.value NOT LIKE 'ncclDevKernel%' AND s.value NOT LIKE 'CatArrayBatchedCopy%'
        GROUP BY k.streamId ORDER BY COUNT(*) DESC LIMIT 1""").fetchone()[0]

    def kern(where, args=()):
        return merge([[s, e] for s, e in c.execute(
            f"SELECT k.start,k.end FROM {K} k JOIN StringIds s ON k.shortName=s.id WHERE {where}",
            args)])

    def kern_named(where, args=()):
        return [(s, e, n) for s, e, n in c.execute(
            f"SELECT k.start,k.end,s.value FROM {K} k JOIN StringIds s ON k.shortName=s.id WHERE {where}",
            args)]

    # dpu_ag_wait: cuStreamWaitValue32 の API 区間は 0.3ms にしかならないので、
    # 「待ちを挟んだ主ストリームのカーネル間ギャップ」で測る（実測でこれが正しい）。
    def dpu_gaps():
        ks = list(c.execute(f"""
            SELECT k.start,k.end,r.start FROM {K} k
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
            WHERE k.streamId=? ORDER BY r.start""", (main,)))
        if not ks:
            return []
        hs = [x[2] for x in ks]
        g = []
        for (t,) in c.execute("""
                SELECT r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN StringIds s ON r.nameId=s.id
                WHERE s.value='cuStreamWaitValue32_v2' ORDER BY r.start"""):
            i = bisect_right(hs, t) - 1
            if 0 <= i < len(ks) - 1 and ks[i + 1][0] > ks[i][1]:
                g.append([ks[i][1], ks[i + 1][0]])
        return merge(g)

    ag = kern("s.value LIKE 'ncclDevKernel_AllGather%'")
    rs = kern("s.value LIKE 'ncclDevKernel_ReduceScatter%'")
    nccl_o = kern("s.value LIKE 'ncclDevKernel%' AND s.value NOT LIKE '%AllGather%' "
                  "AND s.value NOT LIKE '%ReduceScatter%'")
    memcpy = merge([[s, e] for s, e in
                    c.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY")])
    dpu = dpu_gaps()
    other_ks = kern("k.streamId<>? AND s.value NOT LIKE 'ncclDevKernel%'", (main,))
    hostapi = merge([[s, e] for s, e in
                     c.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_RUNTIME")])
    CLS = [("ag(AllGatherカーネル)", ag), ("rs(ReduceScatterカーネル)", rs),
           ("nccl_other", nccl_o), ("mem(memcpy)", memcpy),
           ("dpu_ag_wait", dpu), ("other_stream_kernel", other_ks),
           ("host_api(起動オーバーヘッド)", hostapi)]
    BUSY = kern("k.streamId=?", (main,))

    # 名前つき（内訳用）
    busy_named = kern_named("k.streamId=?", (main,))
    other_named = kern_named("k.streamId<>? AND s.value NOT LIKE 'ncclDevKernel%'", (main,))
    api_named = [(s, e, n) for s, e, n in c.execute(
        "SELECT r.start,r.end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId=s.id")]

    # NVTX: 中央値が短い＝より内側とみなし、内側優先で排他割当する
    nv = {}
    for (t,) in c.execute(
            "SELECT DISTINCT text FROM NVTX_EVENTS WHERE text IS NOT NULL AND end IS NOT NULL"):
        iv = [[s, e] for s, e in c.execute(
            "SELECT start,end FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL", (t,))]
        if iv:
            nv[t] = (statistics.median(b - a for a, b in iv), merge(iv))
    order_nv = sorted(nv, key=lambda t: nv[t][0])

    def named_breakdown(named, win, limit=topn):
        """(start,end,name) 群を win で切って名前別に集計。"""
        if not win:
            return []
        cov = Cover(win)
        agg = {}
        for s, e, n in named:
            v = cov.inter(s, e)
            if v > 0:
                a = agg.setdefault(n, [0, 0])
                a[0] += v
                a[1] += 1
        rows = sorted(agg.items(), key=lambda kv: -kv[1][0])[:limit]
        return [{"name": n, "ms": v[0] / 1e6 / steps, "count": v[1] / steps} for n, v in rows]

    phases = [("forward", nvtx("ZeroWrapperExample.forward")),
              ("backward", nvtx("ZeroWrapperExample.backward"))]
    for t in ("step:opt_dpu", "step:opt", "step:update_new_params", "step:barrier"):
        iv = nvtx(t)
        if iv:
            phases.append((t, iv))

    out = {"file": os.path.basename(db_path), "steps": steps, "phases": {}}
    ms = lambda x: x / 1e6 / steps

    for pname, win in phases:
        if not win:
            continue
        sys.stderr.write(f"    - {pname}\n")
        sys.stderr.flush()
        busy = clip(BUSY, win)
        rem = sub(win, busy)
        rec = {"wall": ms(total(win)), "compute": ms(total(busy)),
               "idle": ms(total(rem)), "classes": {}, "class_detail": {}}
        for cname, iv in CLS:
            ov = inter(rem, iv)
            if ov > 0:
                rec["classes"][cname] = ms(ov)
                # 「アイドル中にそのクラスが占めていた区間」に限定して名前別内訳を出す。
                # rem はこのクラスを引く前の残余なので、そのまま窓に使えば
                # 「アイドル中に動いていた分」だけが集計される。
                if cname == "other_stream_kernel":
                    rec["class_detail"][cname] = named_breakdown(other_named, rem, 5)
                elif cname.startswith("host_api"):
                    rec["class_detail"][cname] = named_breakdown(api_named, rem, 6)
            rem = sub(rem, iv)
        ti = total(rem)
        rec["true_idle"] = ms(ti)
        rec["compute_detail"] = named_breakdown(busy_named, busy, topn)

        # true_idle 中のホスト位置
        rows, cats = [], {}
        r2 = rem
        for t in order_nv:
            ov = inter(r2, nv[t][1])
            if ov > 0:
                rows.append((ms(ov), t))
                cats[categorize(t)] = cats.get(categorize(t), 0) + ms(ov)
                r2 = sub(r2, nv[t][1])
            if not r2:
                break
        rec["true_idle_nvtx"] = [{"name": t, "ms": v}
                                 for v, t in sorted(rows, reverse=True)[:topn]]
        rec["true_idle_cat"] = dict(sorted(cats.items(), key=lambda kv: -kv[1]))
        rec["true_idle_outside_nvtx"] = ms(total(r2))
        out["phases"][pname] = rec

    c.close()
    return out


# ---------------------------------------------------------------- 出力
def rank_of(fname):
    m = re.search(r"rank(\d+)", fname)
    return int(m.group(1)) if m else -1


def fmt_spread(vals):
    """4 ランクの平均と範囲。ランク間のばらつきを隠さない。"""
    if not vals:
        return "-"
    m = statistics.mean(vals)
    if len(vals) == 1:
        return f"{m:8.2f}"
    return f"{m:8.2f}  [{min(vals):7.2f},{max(vals):7.2f}]"


def report(label, results):
    results = sorted(results, key=lambda r: rank_of(r["file"]))
    ranks = [rank_of(r["file"]) for r in results]
    print("=" * 78)
    print(f"=== {label}   ranks={ranks}  steps={results[0]['steps']}   単位: ms/step")
    print("=" * 78)

    pnames = []
    for r in results:
        for p in r["phases"]:
            if p not in pnames:
                pnames.append(p)

    for p in pnames:
        recs = [r["phases"][p] for r in results if p in r["phases"]]
        if not recs:
            continue
        print(f"\n■ {p}")
        print(f"  {'wall':34s}{fmt_spread([x['wall'] for x in recs])}")
        print(f"  {'├ compute (GPU が計算中)':34s}{fmt_spread([x['compute'] for x in recs])}")
        print(f"  {'└ idle    (計算していない)':34s}{fmt_spread([x['idle'] for x in recs])}")

        print(f"\n  [1] idle の排他分解")
        keys = []
        for x in recs:
            for k in x["classes"]:
                if k not in keys:
                    keys.append(k)
        for k in keys:
            print(f"      {k:32s}{fmt_spread([x['classes'].get(k, 0.0) for x in recs])}")
        print(f"      {'true_idle (GPU 完全停止)=Other':32s}"
              f"{fmt_spread([x['true_idle'] for x in recs])}")

        print(f"\n  [2] compute の内訳（主ストリームのカーネル）")
        agg = {}
        for x in recs:
            for d in x["compute_detail"]:
                a = agg.setdefault(d["name"], [])
                a.append(d["ms"])
        for n, v in sorted(agg.items(), key=lambda kv: -statistics.mean(kv[1]))[:6]:
            print(f"      {n[:32]:32s}{fmt_spread(v)}")

        for cname in ("other_stream_kernel", "host_api(起動オーバーヘッド)"):
            agg = {}
            for x in recs:
                for d in x["class_detail"].get(cname, []):
                    agg.setdefault(d["name"], []).append(d["ms"])
            if agg:
                print(f"\n  [3] {cname} の内訳")
                for n, v in sorted(agg.items(), key=lambda kv: -statistics.mean(kv[1]))[:5]:
                    print(f"      {n[:32]:32s}{fmt_spread(v)}")

        print(f"\n  [4] true_idle(=Other) 中にホストがいた場所")
        agg = {}
        for x in recs:
            for k, v in x["true_idle_cat"].items():
                agg.setdefault(k, []).append(v)
        for n, v in sorted(agg.items(), key=lambda kv: -statistics.mean(kv[1])):
            print(f"      {n:32s}{fmt_spread(v)}")
        agg = {}
        for x in recs:
            for d in x["true_idle_nvtx"]:
                agg.setdefault(d["name"], []).append(d["ms"])
        print(f"      -- 関数レベル（内側 NVTX 優先の排他割当）--")
        for n, v in sorted(agg.items(), key=lambda kv: -statistics.mean(kv[1]))[:8]:
            print(f"        {n[:44]:44s}{fmt_spread(v)}")
        print(f"        {'(NVTX 外)':44s}"
              f"{fmt_spread([x['true_idle_outside_nvtx'] for x in recs])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="+")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--topn", type=int, default=8)
    a = ap.parse_args()

    results = []
    for f in a.sqlite:
        sys.stderr.write(f"[解析中] {f}\n")
        sys.stderr.flush()
        results.append(analyze(f, a.topn))

    if a.json:
        with open(a.json, "w") as fp:
            json.dump({"label": a.label, "results": results}, fp, ensure_ascii=False)
        sys.stderr.write(f"[JSON] {a.json}\n")
    report(a.label or "result", results)


if __name__ == "__main__":
    main()
