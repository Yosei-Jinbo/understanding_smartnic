#!/usr/bin/env python3
"""nsys sqlite から計算時間・通信時間・オーバーラップ (隠蔽率) を算出し表にする。

分類の定義:
  compute = ncclDevKernel 以外の全 GPU カーネルの union
            (DtoD memcpy は GPU 内コピーなので compute にも comm にも入れない)
  comm    = AG + RS + その他 nccl の union。チャネル別の計上は:
              AG = ncclDevKernel_AllGather* + HtoD memcpy (param ステージング)
                   [+ SO では DPU-AG (下記)]
              RS = ncclDevKernel_ReduceScatter* + DtoH memcpy (勾配 D2H)
              ※ 転送 (memcpy) は対応する collective の一部として AG/RS に含める
              - [SO のみ] DPU-AG: GPU からは見えないため、cuStreamWaitValue32
                起点のカーネル間ギャップ = GPU が実際に停止した時間で計上する。
                定義上ほぼ 100% 露出扱いになるため、SO の隠蔽率は「下限」
  overlap = compute ∩ comm。隠蔽率 = overlap / comm、露出 = comm − overlap

集計窓:
  最初の ZeroWrapperExample.forward 開始 〜 最後の ZeroOptimizer3.zero_grad 終了。
  全区間をこの窓にクリップし、ms/step は NVTX の forward 数 (=step 数) で割る。

注意 (解釈上の制約):
  - ZO と SO は別ランク・別ノードの実行なので、比率・構造の比較に使い、
    wall の高速化率そのものには使わない
  - DPU-AG の wait ギャップはカーネル再投入レイテンシ (~数十 µs) を含む。
    中央値が ~0.02 ms ならその AG は実質隠蔽済みと読む

使い方:
    python3 scripts/comm_compute_overlap.py <sqlite> [<sqlite> ...]
    最後に全トレースを列に並べた markdown 表を出力する。
"""
import sqlite3
import statistics
import sys
from bisect import bisect_right

K = "CUPTI_ACTIVITY_KIND_KERNEL"


# ---------------- 区間演算 ----------------
def merge(iv):
    """区間リストを昇順マージして重複を潰す。"""
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def inter(a, b):
    """マージ済み区間リスト同士の交差時間 (両方が同時に走っていた時間)。"""
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


def clip(iv, w):
    """区間リストを集計窓 w=(start,end) 内に切り詰める。"""
    return merge([[max(a, w[0]), min(b, w[1])] for a, b in iv if b > w[0] and a < w[1]])


def tot(iv):
    return sum(b - a for a, b in iv)


# ---------------- トレース 1 本の解析 ----------------
def analyze(db):
    c = sqlite3.connect(db)
    r = {"name": db.split("/")[-1].replace(".sqlite", "")}

    # 集計窓と step 数 (forward の NVTX 数 = step 数)
    N = c.execute(
        "SELECT COUNT(*) FROM NVTX_EVENTS WHERE text='ZeroWrapperExample.forward'"
    ).fetchone()[0]
    w0 = c.execute(
        "SELECT MIN(start) FROM NVTX_EVENTS WHERE text='ZeroWrapperExample.forward'"
    ).fetchone()[0]
    w1 = c.execute(
        "SELECT MAX(end) FROM NVTX_EVENTS WHERE text='ZeroOptimizer3.zero_grad' AND end IS NOT NULL"
    ).fetchone()[0]
    win = (w0, w1)
    r["steps"], r["wall"] = N, w1 - w0

    # メイン計算ストリーム = 非 nccl カーネルが最多のストリーム
    # (DPU wait ギャップの検出に使う。CatArrayBatchedCopy はコピー系なので除外して判定)
    main = c.execute(
        f"""SELECT k.streamId FROM {K} k JOIN StringIds s ON k.shortName=s.id
            WHERE s.value NOT LIKE 'ncclDevKernel%' AND s.value NOT LIKE 'CatArrayBatchedCopy%'
            GROUP BY k.streamId ORDER BY COUNT(*) DESC LIMIT 1"""
    ).fetchone()[0]

    def kern(where, args=()):
        return clip(
            merge(
                [
                    [s, e]
                    for s, e in c.execute(
                        f"SELECT k.start,k.end FROM {K} k JOIN StringIds s ON k.shortName=s.id WHERE {where}",
                        args,
                    )
                ]
            ),
            win,
        )

    def memcpy(kind):
        return clip(
            merge(
                [
                    [s, e]
                    for s, e in c.execute(
                        "SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=?",
                        (kind,),
                    )
                ]
            ),
            win,
        )

    # 計算と通信の構成要素
    compute = kern("s.value NOT LIKE 'ncclDevKernel%'")
    ag = kern("s.value LIKE 'ncclDevKernel_AllGather%'")
    rs = kern("s.value LIKE 'ncclDevKernel_ReduceScatter%'")
    onc = kern(
        "s.value LIKE 'ncclDevKernel%' AND s.value NOT LIKE '%AllGather%' AND s.value NOT LIKE '%ReduceScatter%'"
    )
    h2d = memcpy(1)  # copyKind=1: HtoD (param ステージング)
    d2h = memcpy(2)  # copyKind=2: DtoH (勾配 D2H)

    # ---- DPU-AG (SO のみ): wait ギャップと発行→完了スパン ----
    # メインストリームのカーネルを「ホスト発行時刻」順に並べ、
    # cuStreamWaitValue32 の直後に来るカーネル間ギャップ = GPU が DPU 完了を待って
    # 停止した区間、とみなす (idle_decomp_detailed.py と同じ方法)。
    ks = list(
        c.execute(
            f"""SELECT k.start,k.end,r.start FROM {K} k
                JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
                WHERE k.streamId=? ORDER BY r.start""",
            (main,),
        )
    )
    hs = [x[2] for x in ks]
    wait_ts = [
        t
        for (t,) in c.execute(
            """SELECT r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r
               JOIN StringIds s ON r.nameId=s.id
               WHERE s.value='cuStreamWaitValue32_v2' ORDER BY r.start"""
        )
    ]
    gaps_raw = []  # (gap start, gap end): wait ごとの停止区間 (長さ 0 も保持)
    for t in wait_ts:
        i = bisect_right(hs, t) - 1
        if 0 <= i < len(ks) - 1:
            gaps_raw.append((ks[i][1], ks[i + 1][0]))
    dpu_wait = clip(merge([[a, b] for a, b in gaps_raw if b > a]), win)

    # ---- チャネル別グループ (転送 memcpy を対応する collective に計上) ----
    # AG = AG カーネル + param HtoD (+ SO では DPU-AG)、RS = RS カーネル + 勾配 DtoH
    # グループ内の重なりを二重に数えないよう union で足す
    ag_g = merge(ag + h2d + dpu_wait)
    rs_g = merge(rs + d2h)

    # ---- comm union と overlap ----
    comm = merge(ag_g + rs_g + onc)

    r.update(
        compute=tot(compute),
        ag=tot(ag),
        rs=tot(rs),
        onc=tot(onc),
        h2d=tot(h2d),
        d2h=tot(d2h),
        dpu_wait=tot(dpu_wait),
        ag_g=tot(ag_g),
        rs_g=tot(rs_g),
        n_wait=len(gaps_raw),
        comm=tot(comm),
        ov=inter(compute, comm),
        wait_med=statistics.median(b - a for a, b in gaps_raw) if gaps_raw else None,
    )
    c.close()
    return r


# ---------------- 表出力 ----------------
def fmt_table(results):
    """全トレースを列に並べた markdown 表を返す。単位は ms/step (率は %)。"""

    def ms(r, key):
        v = r.get(key)
        return f"{v / 1e6 / r['steps']:.1f}" if v else "—"

    def pct(num, den):
        return f"{num / den * 100:.1f}%" if den else "—"

    rows = [
        ("steps", lambda r: str(r["steps"])),
        ("wall [ms/step]", lambda r: ms(r, "wall")),
        ("compute", lambda r: f"{ms(r, 'compute')} ({pct(r['compute'], r['wall'])})"),
        ("AG 合計", lambda r: ms(r, "ag_g")),
        ("  AG kernel", lambda r: ms(r, "ag")),
        ("  param HtoD", lambda r: ms(r, "h2d")),
        ("  DPU-AG wait", lambda r: ms(r, "dpu_wait")),
        ("RS 合計", lambda r: ms(r, "rs_g")),
        ("  RS kernel", lambda r: ms(r, "rs")),
        ("  grad DtoH", lambda r: ms(r, "d2h")),
        ("nccl other", lambda r: ms(r, "onc")),
        ("comm", lambda r: f"{ms(r, 'comm')} ({pct(r['comm'], r['wall'])})"),
        ("  隠蔽率", lambda r: pct(r["ov"], r["comm"])),
        ("  露出", lambda r: f"{(r['comm'] - r['ov']) / 1e6 / r['steps']:.1f}"),
    ]
    names = [r["name"] for r in results]
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, f in rows:
        lines.append(f"| {label} | " + " | ".join(f(r) for r in results) + " |")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results = []
    for db in sys.argv[1:]:
        r = analyze(db)
        results.append(r)
        # トレースごとの補足情報 (表に入らない診断値)
        print(f"# {r['name']}: steps={r['steps']}", file=sys.stderr)
        if r["n_wait"]:
            print(
                f"#   DPU-AG: waits={r['n_wait']} wait中央値={r['wait_med'] / 1e6:.2f}ms",
                file=sys.stderr,
            )
    print(fmt_table(results))


if __name__ == "__main__":
    main()
