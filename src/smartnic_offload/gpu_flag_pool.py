"""Phase 14/15: completion flag pool — int32 array used as DPU→host completion flags.

Each AG request acquires a slot, gets a unique generation value, and:
  - sends (host_addr, generation) to DPU as part of the cmd
  - schedules cuStreamWaitValue32(EQ, generation) on the consumer compute stream

After the AG completes, DPU writes `generation` to the slot via RDMA Write.
The compute stream's wait then unblocks, and downstream kernels execute.

**Phase 15 memory placement**: **GPU device memory** (was pinned host memory).

Phase 14 originally used pinned host memory with the assumption that
cuStreamWaitValue32 polling via CUDA UVA would add negligible overhead (~10us).
**Empirically this was wrong**: the per-AG wait stall on pinned host memory is
~1.8ms, which accumulates to ~168ms/step of extra GPU stall time during fwd.

Phase 15 moves the flag to GPU memory so that cuStreamWaitValue32 reads a
GPU-local memory location (native GPU semaphore), with ~0.1ms overhead per AG.

For the DPU to write to GPU memory, the host mmap must carry
DOCA_ACCESS_FLAG_PCI_READ_WRITE (Cross-GVMI). This is the same mechanism that
AG dst buffers already use successfully.

**Fallback**: if `USE_GPU_FLAG_POOL_PCI=0` is set in the environment, this
module falls back to pinned host memory (the Phase 14 path).

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

Why monotonic counter (not free-list):
  - O(1) acquire, no synchronization needed for slot return
  - Pool size is far larger than peak in-flight, so wraparound is safe
"""

from __future__ import annotations

import os
import threading
from typing import Tuple

import torch


def _use_gpu_memory() -> bool:
    """Phase 15 デフォルトは GPU memory。USE_GPU_FLAG_POOL_PCI=0 で pinned host に戻す。"""
    v = os.environ.get("USE_GPU_FLAG_POOL_PCI", "1")
    return v != "0"


class GpuFlagPool:
    """Allocate an int32 tensor of POOL_SIZE entries used as completion flags.

    Phase 15: the pool lives in **GPU device memory** (was pinned host memory).
    Cross-GVMI (doca_mmap_export_pci) is used so that the DPU can write the
    flag directly into GPU memory via PCIe, and cuStreamWaitValue32 reads it
    as a GPU-local memory location (no PCIe polling).

    Fallback to pinned host memory via env var `USE_GPU_FLAG_POOL_PCI=0`.
    """

    def __init__(self, device: torch.device, size: int = 4096):
        if size <= 0 or (size & (size - 1)) != 0:
            raise ValueError(f"GpuFlagPool size must be positive power of 2, got {size}")
        self._device = device
        self._size = size
        self._mask = size - 1
        self._use_gpu = _use_gpu_memory()
        if self._use_gpu:
            # Phase 15: GPU memory — DPU writes via Cross-GVMI (export_pci),
            # cuStreamWaitValue32 reads GPU-local memory (native semaphore).
            self._flags = torch.zeros(size, dtype=torch.int32, device=device)
        else:
            # Phase 14 fallback: pinned host memory (UVA accessible from device).
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

        Returns:
            (slot_id, generation, gpu_addr, expected_value)
            - slot_id: index in [0, size)
            - generation: monotonic counter for this slot (also = expected_value)
            - gpu_addr: GPU device address of the int32 slot
            - expected_value: value to wait for (== generation)
        """
        with self._lock:
            slot_id = self._next
            self._next = (self._next + 1) & self._mask
            self._gens[slot_id] += 1
            gen = self._gens[slot_id]
        # int32 max は 0x7FFFFFFF。各 slot ごとに ~4096 AG / step × 1000 step で
        # まだまだ余裕。万一上限に近づいたら再起動を促すアラートを将来追加可。
        if gen >= (1 << 30):
            # very rare; almost never reached in practice
            raise OverflowError(f"slot {slot_id} generation overflow at {gen}")
        gpu_addr = self._base_addr + slot_id * 4
        return slot_id, gen, gpu_addr, gen

    def reset(self) -> None:
        """Reset all flags to 0 and zero generations.

        Used as a debugging escape hatch / hang recovery. After reset, no
        previously-issued waits will fire (they would all wait forever),
        so this should only be called when there are no outstanding waits.
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
