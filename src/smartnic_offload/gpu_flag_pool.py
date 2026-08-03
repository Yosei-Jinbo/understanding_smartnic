"""Completion flag pool — int32 array used as DPU→host completion flags.

Each AG request acquires a slot, gets a unique generation value, and:
  - sends (host_addr, generation) to DPU as part of the cmd
  - schedules cuStreamWaitValue32(EQ, generation) on the consumer compute stream

After the AG completes, DPU writes `generation` to the slot via RDMA Write
(before the doorbell). The compute stream's wait then unblocks, and downstream
kernels execute. The Python thread never blocks in wait().

Memory placement: **GPU device memory** (default)。
  - cuStreamWaitValue32 が GPU ローカルメモリを読むため wait overhead は ~0.1ms/AG。
    pinned host 配置だと PCIe 越しポーリングになり ~1.8ms/AG の GPU stall が乗る。
  - DPU からの書き込みは AG dst バッファと同じ Cross-GVMI 経路。
  - keep_alive の解放判定 (partition_parameters._sweep_flag_keepalive) は
    サイドストリームで flag 配列を pinned staging へスナップショットして読む
    (compute stream をブロックしない。スナップショットが古い分は解放が
    遅れるだけで安全側)。
  - USE_GPU_FLAG_POOL_PCI=0 で pinned host 配置に切替 (UVA で直接読める)。

Slot recycling:
  - Pool size POOL_SIZE (default 4096) >> peak in-flight AGs (~32)
  - Slots are reused round-robin
  - Each reuse increments the slot's generation, so the (slot_id, gen) pair is
    unique across the lifetime of training. cuStreamWaitValue(EQ, gen) for an
    OLD generation is fine because by the time a slot is reused, the GPU has
    already consumed the old AG result.

Why generation as the value (not 1):
  - Each AG using the same slot must have a UNIQUE wait value to avoid the
    consumer stream matching a stale/future flag write.
  - generation is monotonic and unique per slot.
"""

from __future__ import annotations

import os
import threading
from typing import Tuple

import torch


def _use_gpu_memory() -> bool:
    """USE_GPU_FLAG_POOL_PCI=0 で pinned host memory に切替 (既定は GPU memory)。"""
    return os.environ.get("USE_GPU_FLAG_POOL_PCI", "1") != "0"


class GpuFlagPool:
    """int32 completion flags (GPU device memory 既定, POOL_SIZE entries)."""

    def __init__(self, device: torch.device, size: int = 4096):
        if size <= 0 or (size & (size - 1)) != 0:
            raise ValueError(f"GpuFlagPool size must be positive power of 2, got {size}")
        self._device = device
        self._size = size
        self._mask = size - 1
        self._on_gpu = _use_gpu_memory()
        if self._on_gpu:
            self._flags = torch.zeros(size, dtype=torch.int32, device=device)
            # flag_reached 用スナップショット (サイドストリームで D2H コピーして読む)
            self._staging = torch.zeros(size, dtype=torch.int32, pin_memory=True)
            self._copy_stream = torch.cuda.Stream(device=device)
        else:
            # pinned host memory (UVA accessible from device)
            self._flags = torch.zeros(size, dtype=torch.int32, pin_memory=True)
            self._staging = None
            self._copy_stream = None
        self._base_addr = self._flags.data_ptr()
        self._gens = [0] * size  # per-slot monotonic generation counter
        self._next = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return self._size

    @property
    def base_addr(self) -> int:
        return self._base_addr

    @property
    def total_bytes(self) -> int:
        return self._size * 4

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def on_gpu(self) -> bool:
        return self._on_gpu

    def acquire(self) -> Tuple[int, int, int, int]:
        """Acquire the next slot.

        Returns:
            (slot_id, generation, flag_addr, expected_value)
            - slot_id: index in [0, size)
            - generation: monotonic counter for this slot (also = expected_value)
            - flag_addr: address of the int32 slot (GPU device ptr / pinned host UVA)
            - expected_value: value to wait for (== generation)
        """
        with self._lock:
            slot_id = self._next
            self._next = (self._next + 1) & self._mask
            self._gens[slot_id] += 1
            gen = self._gens[slot_id]
        if gen >= (1 << 30):
            raise OverflowError(f"slot {slot_id} generation overflow at {gen}")
        flag_addr = self._base_addr + slot_id * 4
        return slot_id, gen, flag_addr, gen

    def refresh_flags(self) -> None:
        """flag 配列のスナップショットを更新する (GPU 配置時のみ実処理)。
        サイドストリームで 16KB を D2H コピーするだけなので compute stream は
        ブロックしない。flag_reached はこのスナップショットを読む。"""
        if not self._on_gpu:
            return
        with torch.cuda.stream(self._copy_stream):
            self._staging.copy_(self._flags, non_blocking=True)
        self._copy_stream.synchronize()

    def flag_reached(self, slot_id: int, gen: int) -> bool:
        """slot の flag が gen 以上か (= その AG の DPU 側処理が完了済みか)。
        GPU 配置時は直近の refresh_flags() スナップショットを読む
        (古い場合は False 側に倒れるだけで安全)。"""
        src = self._staging if self._on_gpu else self._flags
        return int(src[slot_id]) >= gen


# Global singleton (set by run_zero.py at startup)
_global_pool: "GpuFlagPool | None" = None


def set_global_pool(pool: "GpuFlagPool | None") -> None:
    global _global_pool
    _global_pool = pool


def get_global_pool() -> "GpuFlagPool | None":
    return _global_pool


def is_enabled() -> bool:
    return _global_pool is not None
