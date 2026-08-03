# ../common/submodule_timing.py
from __future__ import annotations

import math
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional, List

try:
    import torch
except Exception:
    torch = None  # type: ignore


@dataclass
class Stat:
    """集計統計: count/total/mean/std/min/max + 固定長サンプルによる p50/p90/p99 近似。
    最初の warmup_n サンプルは統計から除外する。"""
    warmup_n: int = 10
    seen: int = 0  # 受け取ったサンプル数（warmup含む）

    # 集計対象（warmup後のみ）
    count: int = 0
    total_s: float = 0.0
    total_sq_s: float = 0.0
    min_s: float = float("inf")
    max_s: float = 0.0

    # 分位点推定のための固定長サンプル（リングバッファ）
    sample_cap: int = 1024
    samples: List[float] = field(default_factory=list)

    def add(self, x: float) -> None:
        """1サンプル（秒）を加算する。warmup_n 個は統計に入れずに捨てる。"""
        if x is None:
            return
        x = float(x)
        if x < 0:
            return

        self.seen += 1
        if self.seen <= self.warmup_n:
            return

        self.count += 1
        self.total_s += x
        self.total_sq_s += x * x
        if x < self.min_s:
            self.min_s = x
        if x > self.max_s:
            self.max_s = x

        if self.sample_cap > 0:
            if len(self.samples) < self.sample_cap:
                self.samples.append(x)
            else:
                # count を用いた循環上書き（warmup後のcountに対して）
                idx = (self.count - 1) % self.sample_cap
                self.samples[idx] = x

    @property
    def avg_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0

    @property
    def std_s(self) -> float:
        """母標準偏差（unbiased=False 相当）"""
        if self.count <= 1:
            return 0.0
        mean = self.avg_s
        var = (self.total_sq_s / self.count) - (mean * mean)
        return math.sqrt(var) if var > 0.0 else 0.0

    @property
    def cv(self) -> float:
        """変動係数 = std/mean"""
        m = self.avg_s
        return (self.std_s / m) if m > 0 else 0.0

    def quantile_s(self, q: float) -> float:
        """samples から近似分位点を返す（q in [0,1]）"""
        if not self.samples:
            return 0.0
        xs = sorted(self.samples)
        k = int(round(q * (len(xs) - 1)))
        k = max(0, min(k, len(xs) - 1))
        return xs[k]


# 集計は (phase, module_id) 単位。phase: "fwd_fetch", "bwd_exec" など任意に追加可
STATS: Dict[Tuple[str, int], Stat] = {}

# wallclock 計測中: (phase, module_id, call_id) -> t_start
_INFLIGHT_CPU: Dict[Tuple[str, int, int], float] = {}

# gpu event 計測中: (phase, module_id, call_id) -> start_event
_INFLIGHT_GPU: Dict[Tuple[str, int, int], Any] = {}

# end 時点では同期せず pending に積み、snapshot/pretty_print 時に flush。要素: (phase, mid, start_ev, end_ev)
_PENDING_GPU: List[Tuple[str, int, Any, Any]] = []

# module.id -> module info
MODULE_INFO: Dict[int, Dict[str, Optional[str]]] = {}

_LOCK = threading.Lock()


def _get_module_id(sub_module: Any) -> int:
    mid = getattr(sub_module, "id", None)
    if mid is None:
        raise RuntimeError("sub_module.id is not set. Ensure _register_hooks_recursively sets module.id.")
    return int(mid)


def register_module_info(sub_module: Any) -> None:
    """module.id -> (name, class) の対応を保存する。"""
    mid = _get_module_id(sub_module)
    cls = sub_module.__class__
    name = getattr(sub_module, "_timing_name", None)
    with _LOCK:
        MODULE_INFO[mid] = {
            "name": name,
            "class": f"{cls.__module__}.{cls.__qualname__}",
        }


def get_module_info(mid: int) -> Optional[Dict[str, Optional[str]]]:
    with _LOCK:
        return MODULE_INFO.get(int(mid))


def _is_exec_phase(phase: str) -> bool:
    # "_exec"/"_stall" は wallclock でなく CUDA Event (GPU 側 elapsed_time) で測る必要がある
    return phase.endswith("_exec") or phase.endswith("_stall")


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _get_or_create_stat_locked(phase: str, mid: int) -> Stat:
    st = STATS.get((phase, mid))
    if st is None:
        st = Stat()  # warmup_n=10 がデフォルト
        STATS[(phase, mid)] = st
    return st


def _flush_pending_gpu_locked() -> None:
    """
    _LOCK を取った状態で呼ばれる前提。
    pending の end_event を同期して elapsed を確定し、STATS に加算する。
    """
    if not _PENDING_GPU:
        return
    if not _cuda_available():
        _PENDING_GPU.clear()
        return

    pending = list(_PENDING_GPU)
    _PENDING_GPU.clear()

    # まとめて synchronize（これが唯一の同期点）
    for _, _, _, end_ev in pending:
        end_ev.synchronize()

    # elapsed_time は ms を返す
    for phase, mid, start_ev, end_ev in pending:
        elapsed_ms = start_ev.elapsed_time(end_ev)
        elapsed_s = float(elapsed_ms) / 1000.0
        st = _get_or_create_stat_locked(phase, mid)
        st.add(elapsed_s)


def start(phase: str, sub_module: Any, *, call_id: int) -> None:
    """
    - *_fetch は wallclock
    - *_exec は CUDA Event（同期は後でまとめて）
    """
    mid = _get_module_id(sub_module)
    key = (phase, mid, int(call_id))

    with _LOCK:
        if _is_exec_phase(phase):
            if key in _INFLIGHT_GPU:
                return  # 重複 start 防止

            if not _cuda_available():
                # フォールバック: wallclock
                _INFLIGHT_CPU[key] = time.perf_counter()
                return

            start_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()  # current stream
            _INFLIGHT_GPU[key] = start_ev
        else:
            if key in _INFLIGHT_CPU:
                return
            _INFLIGHT_CPU[key] = time.perf_counter()


def end(phase: str, sub_module: Any, *, call_id: int) -> float:
    """
    戻り値:
      - fetch: 確定した elapsed(s)
      - exec : ここでは確定できないので 0.0（flush 後に STATS へ反映）
    """
    mid = _get_module_id(sub_module)
    key = (phase, mid, int(call_id))

    with _LOCK:
        if _is_exec_phase(phase):
            start_ev = _INFLIGHT_GPU.pop(key, None)
            if start_ev is None:
                # フォールバック (wallclock start で入っていた場合)
                t0 = _INFLIGHT_CPU.pop(key, None)
                if t0 is None:
                    return 0.0
                elapsed = time.perf_counter() - t0
                st = _get_or_create_stat_locked(phase, mid)
                st.add(elapsed)
                return elapsed

            end_ev = torch.cuda.Event(enable_timing=True)
            end_ev.record()
            _PENDING_GPU.append((phase, mid, start_ev, end_ev))
            return 0.0

        # wallclock phase
        t0 = _INFLIGHT_CPU.pop(key, None)
        if t0 is None:
            return 0.0
        elapsed = time.perf_counter() - t0
        st = _get_or_create_stat_locked(phase, mid)
        st.add(elapsed)
        return elapsed


def snapshot(flush_cuda: bool = True) -> Dict[Tuple[str, int], Stat]:
    with _LOCK:
        if flush_cuda:
            _flush_pending_gpu_locked()
        return dict(STATS)


def reset() -> None:
    with _LOCK:
        STATS.clear()
        _INFLIGHT_CPU.clear()
        _INFLIGHT_GPU.clear()
        _PENDING_GPU.clear()
        MODULE_INFO.clear()


def compute_pure_exec_by_hierarchy(
    exec_phase: str = "bwd_exec",
    stall_phase: str = "bwd_wait_stall",
) -> Dict[int, Dict[str, Any]]:
    """`_timing_name` の階層から pure_exec(M) = exec(M) - Σ 子孫の wait_stall を算出。
    親の exec ウィンドウには子孫の fetch stall が混入するため差し引き、subtree の純粋カーネル時間を得る。
    Returns: {module_id: {name, class, exec_total_s/mean_s/count, stall_self_s,
    stall_in_subtree_s, pure_exec_total_s/mean_s}}"""
    with _LOCK:
        _flush_pending_gpu_locked()

        # (phase, mid) -> Stat をコピー
        local_stats: Dict[Tuple[str, int], Stat] = dict(STATS)
        local_info: Dict[int, Dict[str, Optional[str]]] = dict(MODULE_INFO)

    # mid -> name のマップ(None/empty を別扱いしやすい形で)
    mid_to_name: Dict[int, str] = {}
    for mid, info in local_info.items():
        name = info.get("name")
        mid_to_name[mid] = name if name else ""

    result: Dict[int, Dict[str, Any]] = {}

    for mid, name in mid_to_name.items():
        info = local_info.get(mid, {})
        cls = info.get("class") or ""

        exec_st = local_stats.get((exec_phase, mid))
        stall_self_st = local_stats.get((stall_phase, mid))

        exec_total_s = exec_st.total_s if exec_st is not None else 0.0
        exec_count = exec_st.count if exec_st is not None else 0
        exec_mean_s = (exec_total_s / exec_count) if exec_count else 0.0
        stall_self_s = stall_self_st.total_s if stall_self_st is not None else 0.0

        # descendants: _timing_name が f"{name}." で始まる他モジュール (name=="" root は自分以外全て)
        stall_in_subtree_s = 0.0
        if name:
            prefix = name + "."
            for other_mid, other_name in mid_to_name.items():
                if other_mid == mid:
                    continue
                if other_name and other_name.startswith(prefix):
                    st = local_stats.get((stall_phase, other_mid))
                    if st is not None:
                        stall_in_subtree_s += st.total_s
        else:
            for other_mid in mid_to_name.keys():
                if other_mid == mid:
                    continue
                st = local_stats.get((stall_phase, other_mid))
                if st is not None:
                    stall_in_subtree_s += st.total_s

        pure_exec_total_s = exec_total_s - stall_in_subtree_s
        pure_exec_mean_s = (pure_exec_total_s / exec_count) if exec_count else 0.0

        result[mid] = {
            "name": name,
            "class": cls,
            "exec_total_s": exec_total_s,
            "exec_mean_s": exec_mean_s,
            "exec_count": exec_count,
            "stall_self_s": stall_self_s,
            "stall_in_subtree_s": stall_in_subtree_s,
            "pure_exec_total_s": pure_exec_total_s,
            "pure_exec_mean_s": pure_exec_mean_s,
        }

    return result


def pretty_print_pure_exec(
    exec_phase: str = "bwd_exec",
    stall_phase: str = "bwd_wait_stall",
    limit: int = 200,
    sort_by: str = "pure_exec_total_s",
) -> None:
    """
    compute_pure_exec_by_hierarchy の結果を人間可読な形で print する。

    sort_by: "pure_exec_total_s" | "exec_total_s" | "stall_in_subtree_s" | "stall_self_s"
    """
    data = compute_pure_exec_by_hierarchy(exec_phase=exec_phase, stall_phase=stall_phase)

    rows = list(data.items())
    rows.sort(key=lambda kv: kv[1].get(sort_by, 0.0), reverse=True)

    print(f"========== pure_exec breakdown (exec={exec_phase}, stall={stall_phase}) ==========")
    print(
        f"{'id':>6} {'count':>7} "
        f"{'exec_total_ms':>14} {'stall_subtree_ms':>18} "
        f"{'pure_exec_total_ms':>20} {'pure_exec_mean_ms':>20} "
        f"{'stall_self_ms':>14}  name"
    )
    s2ms = 1000.0
    for mid, d in rows[:limit]:
        print(
            f"{mid:>6} {d['exec_count']:>7} "
            f"{d['exec_total_s']*s2ms:>14.3f} {d['stall_in_subtree_s']*s2ms:>18.3f} "
            f"{d['pure_exec_total_s']*s2ms:>20.3f} {d['pure_exec_mean_s']*s2ms:>20.3f} "
            f"{d['stall_self_s']*s2ms:>14.3f}  {d['name']}"
        )
    print(f"{'=' * 80}")


def pretty_print(limit: int = 200, sort_by: str = "total_s") -> None:
    """module.id 単位の集計を表示（モジュール名/型も併記）。
    sort_by: total_s (default) / count / avg_s / std_s / p90 / p99 / cv / seen (warmup 含む)"""
    with _LOCK:
        _flush_pending_gpu_locked()

        rows = []
        for (phase, mid), st in STATS.items():
            info = MODULE_INFO.get(mid, {})
            name = info.get("name")
            cls = info.get("class")

            min_s = st.min_s if st.count else 0.0
            max_s = st.max_s if st.count else 0.0
            p50 = st.quantile_s(0.50)
            p90 = st.quantile_s(0.90)
            p99 = st.quantile_s(0.99)

            rows.append((
                phase, mid, name, cls,
                st.seen, st.warmup_n,
                st.count, st.total_s, st.avg_s, st.std_s, st.cv,
                min_s, p50, p90, p99, max_s
            ))

    if sort_by == "seen":
        rows.sort(key=lambda x: x[4], reverse=True)
    elif sort_by == "count":
        rows.sort(key=lambda x: x[6], reverse=True)
    elif sort_by == "avg_s":
        rows.sort(key=lambda x: x[8], reverse=True)
    elif sort_by == "std_s":
        rows.sort(key=lambda x: x[9], reverse=True)
    elif sort_by == "cv":
        rows.sort(key=lambda x: x[10], reverse=True)
    elif sort_by == "p90":
        rows.sort(key=lambda x: x[13], reverse=True)
    elif sort_by == "p99":
        rows.sort(key=lambda x: x[14], reverse=True)
    else:
        rows.sort(key=lambda x: x[7], reverse=True)  # total_s

    print(
        f"{'phase':<10} {'id':>6} {'seen':>8} {'wup':>4} {'count':>8} "
        f"{'total_s':>12} {'mean_s':>12} {'std_s':>12} {'cv':>10} "
        f"{'min_s':>12} {'p50_s':>12} {'p90_s':>12} {'p99_s':>12} {'max_s':>12}  "
        f"{'name':<40} {'class'}"
    )
    for (phase, mid, name, cls, seen, wup, cnt, total_s, mean_s, std_s, cv, min_s, p50, p90, p99, max_s) in rows[:limit]:
        name_s = (name or "")[:40]
        cls_s = cls or ""
        print(
            f"{phase:<10} {mid:>6} {seen:>8} {wup:>4} {cnt:>8} "
            f"{total_s:>12.6f} {mean_s:>12.6f} {std_s:>12.6f} {cv:>10.3f} "
            f"{min_s:>12.6f} {p50:>12.6f} {p90:>12.6f} {p99:>12.6f} {max_s:>12.6f}  "
            f"{name_s:<40} {cls_s}"
        )
