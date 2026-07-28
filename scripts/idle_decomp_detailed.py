#!/usr/bin/env python3
"""nsys sqlite から fwd/bwd の idle を排他分解し、true_idle 中のホスト位置まで出す。

既存の compute_idle_decomp_ag_smartnic.py に対する改良点（すべて実測で必要と判明）:
  1. フェーズ別に分解する（fwd と bwd をまとめない）
  2. ncclDevKernel_AllReduce 等を取りこぼさない（nccl_other クラス）
  3. dpu_ag_wait はカーネル間ギャップで取る（API 区間だと 0.30ms にしかならない）
  4. true_idle 中のホスト位置を NVTX で特定（中央値が短い＝内側の順に排他割当）

使い方:
    python3 scripts/idle_decomp_detailed.py <nsys sqlite>
"""
import sqlite3, sys, statistics
from bisect import bisect_right
def merge(iv):
    iv=sorted(iv); out=[]
    for a,b in iv:
        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out
def sub(base,cut):
    out,j=[],0
    for a,b in base:
        cur=a
        while j<len(cut) and cut[j][1]<=cur: j+=1
        k=j
        while k<len(cut) and cut[k][0]<b:
            if cut[k][0]>cur: out.append([cur,min(cut[k][0],b)])
            cur=max(cur,cut[k][1]); k+=1
        if cur<b: out.append([cur,b])
    return [x for x in out if x[1]>x[0]]
def inter(a,b):
    i=j=t=0
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]); hi=min(a[i][1],b[j][1])
        if hi>lo: t+=hi-lo
        if a[i][1]<b[j][1]: i+=1
        else: j+=1
    return t
def clip(iv,win): return merge([[max(a,w[0]),min(b,w[1])] for a,b in iv for w in win if b>w[0] and a<w[1]])

db=sys.argv[1]; c=sqlite3.connect(db); K="CUPTI_ACTIVITY_KIND_KERNEL"
N=c.execute("SELECT COUNT(*) FROM NVTX_EVENTS WHERE text='ZeroWrapperExample.forward'").fetchone()[0]
main=c.execute(f"""SELECT k.streamId FROM {K} k JOIN StringIds s ON k.shortName=s.id
  WHERE s.value NOT LIKE 'ncclDevKernel%' AND s.value NOT LIKE 'CatArrayBatchedCopy%'
  GROUP BY k.streamId ORDER BY COUNT(*) DESC LIMIT 1""").fetchone()[0]
def kern(w,a=()): return merge([[s,e] for s,e in c.execute(f"SELECT k.start,k.end FROM {K} k JOIN StringIds s ON k.shortName=s.id WHERE {w}",a)])
def dpu_gaps():
    ks=list(c.execute(f"""SELECT k.start,k.end,r.start FROM {K} k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
        ON k.correlationId=r.correlationId WHERE k.streamId=? ORDER BY r.start""",(main,)))
    if not ks: return []
    hs=[x[2] for x in ks]; g=[]
    for (t,) in c.execute("""SELECT r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
                             WHERE s.value='cuStreamWaitValue32_v2' ORDER BY r.start"""):
        i=bisect_right(hs,t)-1
        if 0<=i<len(ks)-1 and ks[i+1][0]>ks[i][1]: g.append([ks[i][1],ks[i+1][0]])
    return merge(g)

CLS=[("ag",kern("s.value LIKE 'ncclDevKernel_AllGather%'")),
     ("rs",kern("s.value LIKE 'ncclDevKernel_ReduceScatter%'")),
     ("nccl_other(AllReduce等)",kern("s.value LIKE 'ncclDevKernel%' AND s.value NOT LIKE '%AllGather%' AND s.value NOT LIKE '%ReduceScatter%'")),
     ("mem(memcpy)",merge([[s,e] for s,e in c.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY")])),
     ("dpu_ag_wait",dpu_gaps()),
     ("other_stream_kernel",kern("k.streamId<>? AND s.value NOT LIKE 'ncclDevKernel%'",(main,))),
     ("host_api",merge([[s,e] for s,e in c.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_RUNTIME")]))]
BUSY=kern("k.streamId=?",(main,))

# NVTX を「内側優先」で排他割当するため、平均長の短い順に並べる
nv={}
for t, in c.execute("SELECT DISTINCT text FROM NVTX_EVENTS WHERE text IS NOT NULL AND end IS NOT NULL"):
    iv=[[s,e] for s,e in c.execute("SELECT start,end FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL",(t,))]
    if iv: nv[t]=(statistics.median(b-a for a,b in iv), merge(iv))
order_nv=sorted(nv, key=lambda t: nv[t][0])

print(f"=== {db.split('/')[-1]}  (steps={N}) ===")
for phase in ("ZeroWrapperExample.forward","ZeroWrapperExample.backward"):
    win=merge([[s,e] for s,e in c.execute("SELECT start,end FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL",(phase,))])
    wall=sum(b-a for a,b in win); busy=clip(BUSY,win); rem=sub(win,busy)
    ms=lambda x:x/1e6/N
    print(f"\n--- {phase.split('.')[1]}  wall={ms(wall):.2f}  compute={ms(sum(b-a for a,b in busy)):.2f}  idle={ms(sum(b-a for a,b in rem)):.2f} [ms/step] ---")
    for name,iv in CLS:
        ov=inter(rem,iv); rem=sub(rem,iv)
        if ov>0: print(f"      {name:26s}{ms(ov):9.2f}")
    ti=sum(b-a for a,b in rem)
    print(f"      {'true_idle(GPU 完全停止)':26s}{ms(ti):9.2f}")
    if ti>0:
        print(f"      └ true_idle 中のホスト位置（内側 NVTX 優先の排他割当）")
        r2=rem; rows=[]
        for t in order_nv:
            ov=inter(r2,nv[t][1])
            if ov>0: rows.append((ms(ov),t)); r2=sub(r2,nv[t][1])
            if not r2: break
        for v,t in sorted(rows,reverse=True)[:8]:
            print(f"          {t[:46]:46s}{v:8.2f} ({v/ms(ti)*100:4.1f}%)")
        if r2: print(f"          {'(NVTX 外)':46s}{ms(sum(b-a for a,b in r2)):8.2f}")
