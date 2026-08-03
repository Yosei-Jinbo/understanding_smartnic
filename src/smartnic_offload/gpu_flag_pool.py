"""Completion flag pool — int32 array used as DPU→host completion flags.

Each AG request acquires a slot, gets a unique generation value, and:
  - sends (host_addr, generation) to DPU as part of the cmd
  - schedules cuStreamWaitValue32(EQ, generation) on the consumer compute stream

After the AG completes, DPU writes `generation` to the slot via RDMA Write
(before the doorbell). The compute stream's wait then unblocks, and downstream
kernels execute. The Python thread never blocks in wait().

Memory placement: pinned host memory 固定。
  - CUDA UVA で data_ptr() がそのまま device ポインタとして使え、
    keep_alive の解放判定 (partition_parameters._sweep_flag_keepalive) も
    Python から直接安価に読める。
  - GPU device memory 配置は wait overhead は小さいが、dual-rail (既定) では
    rail1 経由の AG データ書き込みと flag 書き込みの到着順序が保証されず
    精度が壊れる (vit E1 で実測) ため採用しない。
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
