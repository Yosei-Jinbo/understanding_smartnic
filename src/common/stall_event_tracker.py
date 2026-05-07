"""Phase 20 (Phase 1): per-AG/RS stall event-bracket tracker.

両 mode 対称な「真の AG/RS stall」測定:
  - PF (NCCL): handle.wait() = cudaStreamWaitEvent on NCCL completion event
  - SO (DOCA): cuStreamWaitValue32 on DPU-written flag

どちらも compute stream に積まれる sync op なので、その op を CUDA event ペアで
括れば「op が compute stream の head に到達してから抜けるまでの GPU 上経過時間」
= 真の stall 時間が両 mode 同じ定義で取れる。

使い方:
    from common import stall_event_tracker as _set
    if _set.is_enabled():
        h = _set.get_global().begin(stream, phase, op="ag",
                                     ds_id=..., payload_bytes=...)
        wait_op()                       # cudaStreamWaitEvent / cuStreamWaitValue32
        _set.get_global().end(h)

Env gates:
    MEASURE_AG_STALL=1            # 計測有効化 (デフォルト無効)
    AG_STALL_DUMP_PATH=<path>     # 出力 JSON path (省略時 stall_events.json)

注意:
  - PF mode では DISABLE_COMPLETION_POLLER=1 必須。
    poller がいると handle.wait() が host 側 polling 経路となり
    cudaStreamWaitEvent を発行しない (= ev_ms ≈ 0 で計測不能)。
  - 計測 overhead: ev0/ev1.record() 各 ~1µs (host) + ~1µs (GPU stream op)。
    75 AG/iter で ~0.3 ms/iter (基底 1100 ms の 0.03%)、無視可。
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore


_ENABLED = os.environ.get("MEASURE_AG_STALL", "0") == "1"


def is_enabled() -> bool:
    return _ENABLED


# Per-handle tuple layout: (ev0, ev1, phase, op, ag_id, ds_id, bytes, iter_idx, stream)
_Handle = Tuple[Any, Any, str, str, int, int, int, int, Any]


class StallEventTracker:
    """Per-AG/RS event-bracket measurement of GPU-side stall time."""

    def __init__(self) -> None:
        self._pending: List[_Handle] = []
        self._records: List[Dict[str, Any]] = []
        self._iter_idx: int = -1
        self._epoch_idx: int = 0
        self._counter: int = 0
        self._lock = threading.Lock()
        # When set, drain() is short-circuited (used during finalize race conditions)
        self._frozen: bool = False

    # -------------------------------------------------------------------
    # Public state setters
    # -------------------------------------------------------------------

    def set_iter(self, idx: int) -> None:
        with self._lock:
            self._iter_idx = idx

    def set_epoch(self, idx: int) -> None:
        with self._lock:
            self._epoch_idx = idx

    def get_iter(self) -> int:
        with self._lock:
            return self._iter_idx

    # -------------------------------------------------------------------
    # Bracketing
    # -------------------------------------------------------------------

    def begin(self,
              stream: Any,
              phase: str,
              op: str = "ag",
              ds_id: int = -1,
              payload_bytes: int = 0) -> Optional[_Handle]:
        """Record ev0 immediately on `stream`. Return handle for end().

        例外時は None を返し、bracket 全体を no-op 化する (学習を壊さない)。
        """
        if not _ENABLED or torch is None:
            return None
        try:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record(stream)
        except Exception:
            return None
        with self._lock:
            ag_id = self._counter
            self._counter += 1
            iter_idx = self._iter_idx
        return (ev0, ev1, phase, op, ag_id, ds_id, int(payload_bytes), iter_idx, stream)

    def end(self, handle: Optional[_Handle]) -> None:
        """Record ev1 on the same stream and queue for later drain.

        例外を吸収する (学習側の例外を suppress しないため)。
        """
        if handle is None:
            return
        ev0, ev1, phase, op, ag_id, ds_id, nbytes, iter_idx, stream = handle
        try:
            ev1.record(stream)
        except Exception:
            return
        with self._lock:
            self._pending.append(handle)

    # -------------------------------------------------------------------
    # Drain / dump
    # -------------------------------------------------------------------

    def drain(self) -> None:
        """Sync once, compute elapsed_time for all pending events, store records."""
        if not _ENABLED or torch is None:
            return
        with self._lock:
            if self._frozen or not self._pending:
                return
            pending = list(self._pending)
            self._pending.clear()
        try:
            torch.cuda.synchronize()
        except Exception:
            return
        with self._lock:
            for ev0, ev1, phase, op, ag_id, ds_id, nbytes, iter_idx, _stream in pending:
                try:
                    ms = float(ev0.elapsed_time(ev1))
                except Exception:
                    continue
                if ms < 0.0:
                    ms = 0.0
                self._records.append({
                    "epoch": self._epoch_idx,
                    "iter": iter_idx,
                    "phase": phase,
                    "op": op,
                    "ag_id": ag_id,
                    "ds_id": ds_id,
                    "bytes": nbytes,
                    "ms": ms,
                })

    def freeze(self) -> None:
        with self._lock:
            self._frozen = True

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._pending.clear()
            self._counter = 0
            self._iter_idx = -1

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------

    def dump_json(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not _ENABLED:
            return
        self.drain()
        with self._lock:
            payload = {
                "extra": extra or {},
                "records": list(self._records),
            }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Aggregate by (phase, op): count / total / mean / p50 / p95 / max."""
        if not _ENABLED:
            return {}
        self.drain()
        groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        with self._lock:
            recs = list(self._records)
        for r in recs:
            groups[(r["phase"], r["op"])].append(r["ms"])
        out: Dict[str, Dict[str, float]] = {}
        for (phase, op), vs in groups.items():
            vs.sort()
            n = len(vs)
            if n == 0:
                continue
            out[f"{phase}.{op}"] = {
                "count": float(n),
                "total_ms": sum(vs),
                "mean_ms": sum(vs) / n,
                "p50_ms": vs[n // 2],
                "p95_ms": vs[min(int(n * 0.95), n - 1)],
                "max_ms": vs[-1],
            }
        return out


# -------------------------------------------------------------------
# Module-level singleton
# -------------------------------------------------------------------

_GLOBAL: Optional[StallEventTracker] = None
_GLOBAL_LOCK = threading.Lock()


def get_global() -> StallEventTracker:
    global _GLOBAL
    if _GLOBAL is None:
        with _GLOBAL_LOCK:
            if _GLOBAL is None:
                _GLOBAL = StallEventTracker()
    return _GLOBAL


# -------------------------------------------------------------------
# Convenience
# -------------------------------------------------------------------

def default_dump_path(rank: int = 0, label: str = "stall_events") -> str:
    """Return env-overridable default path for the JSON dump."""
    p = os.environ.get("AG_STALL_DUMP_PATH")
    if p:
        return p
    return f"logs/{label}_rank{rank}.json"
