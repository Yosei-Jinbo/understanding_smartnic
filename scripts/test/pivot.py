#!/usr/bin/env python3
"""results.csv を「RS/AG × N を行、各設定を列」にピボットする。

sweep_multicore.sh / sweep_ablation.sh が出す long 形式 CSV を読み、
各 (collective, size) を 1 行、各設定を 1 列にした見やすい表を作る。

使い方:
    pivot.py <results.csv> [metric]
      metric: avg_ms | p50_ms | p99_ms | bw_gbps  (既定 p50_ms)

出力:
    - 標準出力に整列表示
    - 同ディレクトリに pivot_<metric>.csv
設定(列)の識別:
    - config 列があれば（アブレーション）その名前を列名に
    - 無ければ（マルチコア）"C{comm}K{compute}" を列名に
"""
import sys, csv, os
from collections import OrderedDict

if len(sys.argv) < 2:
    sys.exit("usage: pivot.py <results.csv> [metric=p50_ms]")
path = sys.argv[1]
metric = sys.argv[2] if len(sys.argv) > 2 else "p50_ms"

rows = list(csv.DictReader(open(path)))
if not rows:
    sys.exit("empty csv: " + path)
if metric not in rows[0]:
    sys.exit(f"metric '{metric}' not in columns: {list(rows[0].keys())}")


def config_key(r):
    if r.get("config"):                                   # アブレーション形式
        return r["config"]
    if r.get("comm_cores") is not None:                   # マルチコア形式
        return f"C{r['comm_cores']}K{r['compute_cores']}"
    if r.get("rail") is not None:                         # rail×piece 形式
        return f"{r['rail']}:p{r.get('ag_piece_max', '')}"
    return "cfg"


# 列（設定）: 出現順を保持
configs = list(OrderedDict((config_key(r), None) for r in rows))

# 行キー: RS を先、N 昇順
def rowkey(r):
    return (0 if r["coll"] == "RS" else 1, int(r["N"]))

rowmeta = OrderedDict()   # rowkey -> (coll, size_label)
cell = {}
for r in rows:
    rk = rowkey(r)
    rowmeta[rk] = (r["coll"], r.get("size_label", r["N"]))
    cell[(rk, config_key(r))] = r.get(metric, "")
rowkeys = sorted(rowmeta.keys())

header = ["coll", "size"] + configs
table = [header]
for rk in rowkeys:
    coll, size = rowmeta[rk]
    table.append([coll, size] + [cell.get((rk, c), "") for c in configs])

# CSV 書き出し
outp = os.path.join(os.path.dirname(os.path.abspath(path)), f"pivot_{metric}.csv")
with open(outp, "w", newline="") as f:
    csv.writer(f).writerows(table)

# 整列して標準出力
widths = [max(len(str(row[i])) for row in table) for i in range(len(header))]
print(f"=== pivot ({metric}): 行=RS/AG×N, 列=設定 ===")
for row in table:
    print("  ".join(str(v).rjust(widths[i]) for i, v in enumerate(row)))
print(f"\n[written] {outp}")
