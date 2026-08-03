"""Completion flag pool — int32 array used as DPU→host completion flags.

Each AG request acquires a slot, gets a unique generation value, and:
  - sends (host_addr, generation) to DPU as part of the cmd
  - schedules cuStreamWaitValue32(EQ, generation) on the consumer compute stream

After the AG completes, DPU writes `generation` to the slot via RDMA Write
(before the doorbell). The compute stream's wait then unblocks, and downstream
kernels execute. The Python thread never blocks in wait().

Memory placement: **pinned host memory** (fixed).
  - CUDA UVA makes data_ptr() valid as a device pointer, so
    cuStreamWaitValue32 can wait on it from the GPU.
  - The DPU writes it via the ordinary RDMA rkey path (same as the doorbell).
  - Being host memory, Python can also read the flags cheaply — used to decide
    when enqueue-time keep_alive buffers can be released (see
    partition_parameters._sweep_flag_keepalive).

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

import threading
from typing import Tuple

import torch


class GpuFlagPool:
    """int32 completion flags (pinned host memory, POOL_SIZE entries)."""

    def __init__(self, device: torch.device, size: int = 4096):
        if size <= 0 or (size & (size - 1)) != 0:
            raise ValueError(f"GpuFlagPool size must be positive power of 2, got {size}")
        self._device = device
        self._size = size
        self._mask = size - 1
        # pinned host memory (UVA accessible from device)
        self._flags = torch.zeros(size, dtype=torch.int32, pin_memory=True)
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

    def acquire(self) -> Tuple[int, int, int, int]:
        """Acquire the next slot.

        Returns:
            (slot_id, generation, flag_addr, expected_value)
            - slot_id: index in [0, size)
            - generation: monotonic counter for this slot (also = expected_value)
            - flag_addr: address of the int32 slot (pinned host; UVA device ptr)
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

    def flag_reached(self, slot_id: int, gen: int) -> bool:
        """slot の flag が gen 以上か (= その AG の DPU 側処理が完了済みか)。
        pinned host memory の読みなので GPU 同期なしで安価。"""
        return int(self._flags[slot_id]) >= gen


# Global singleton (set by run_zero.py at startup)
_global_pool: "GpuFlagPool | None" = None


def set_global_pool(pool: "GpuFlagPool | None") -> None:
    global _global_pool
    _global_pool = pool


def get_global_pool() -> "GpuFlagPool | None":
    return _global_pool


def is_enabled() -> bool:
    return _global_pool is not None
