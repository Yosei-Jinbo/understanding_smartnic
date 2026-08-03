"""Completion flag pool — int32 array used as DPU→host completion flags.

各 AG は slot を取得して (host_addr, generation) を DPU に送り、consumer stream に
cuStreamWaitValue32(EQ, generation) を積む。AG 完了時に DPU が RDMA Write で
generation を書き込み、待ちが解ける。generation は slot ごとに単調増加なので
stale/future な flag write と誤マッチしない。
DPU が GPU メモリへ書くため host mmap は DOCA_ACCESS_FLAG_PCI_READ_WRITE
(Cross-GVMI) が必要。USE_GPU_FLAG_POOL_PCI=0 で pinned host memory に切替。
"""

from __future__ import annotations

import os
import threading
from typing import Tuple

import torch


def _use_gpu_memory() -> bool:
    """USE_GPU_FLAG_POOL_PCI=0 で pinned host memory に切替 (デフォルトは GPU memory)。"""
    v = os.environ.get("USE_GPU_FLAG_POOL_PCI", "1")
    return v != "0"


class GpuFlagPool:
    """int32 completion flag pool. GPU device memory 上に置き、DPU は Cross-GVMI
    (doca_mmap_export_pci) 経由で書き込む。USE_GPU_FLAG_POOL_PCI=0 で pinned host に切替。
    """

    def __init__(self, device: torch.device, size: int = 4096):
        if size <= 0 or (size & (size - 1)) != 0:
            raise ValueError(f"GpuFlagPool size must be positive power of 2, got {size}")
        self._device = device
        self._size = size
        self._mask = size - 1
        self._use_gpu = _use_gpu_memory()
        if self._use_gpu:
            self._flags = torch.zeros(size, dtype=torch.int32, device=device)
        else:
            # fallback: pinned host memory (UVA accessible from device)
            self._flags = torch.zeros(size, dtype=torch.int32, pin_memory=True)
        self._base_addr = self._flags.data_ptr()
        self._gens = [0] * size  # per-slot monotonic generation counter
        self._next = 0
        self._lock = threading.Lock()

    @property
    def is_gpu_memory(self) -> bool:
        return self._use_gpu

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

        Returns (slot_id, generation, gpu_addr, expected_value);
        expected_value == generation (slot ごとの単調カウンタ)。
        """
        with self._lock:
            slot_id = self._next
            self._next = (self._next + 1) & self._mask
            self._gens[slot_id] += 1
            gen = self._gens[slot_id]
        if gen >= (1 << 30):
            raise OverflowError(f"slot {slot_id} generation overflow at {gen}")
        gpu_addr = self._base_addr + slot_id * 4
        return slot_id, gen, gpu_addr, gen

    def reset(self) -> None:
        """Reset all flags/generations. outstanding wait が残っていると永久待ちに
        なるため、待ちが無いときのみ呼ぶこと。
        """
        with self._lock:
            self._flags.zero_()
            self._gens = [0] * self._size
            self._next = 0


# Global singleton (set by run_zero.py at startup)
_global_pool: "GpuFlagPool | None" = None


def set_global_pool(pool: "GpuFlagPool | None") -> None:
    global _global_pool
    _global_pool = pool


def get_global_pool() -> "GpuFlagPool | None":
    return _global_pool


def is_enabled() -> bool:
    return _global_pool is not None
