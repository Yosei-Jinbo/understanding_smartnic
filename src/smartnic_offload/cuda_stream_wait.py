"""cuStreamWaitValue32 wrapper for PyTorch CUDA streams.

DPU が AG 完了時に GPU メモリ上の int32 フラグに値を書き込み、PyTorch の compute
stream はその値を待ってから後続の compute kernel を実行する。

これにより:
  - Python thread は wait() でブロックしなくなる
  - GPU の HW semaphore で AG completion と compute が同期する
  - NCCL が CUDA stream で実現しているのと同じ semantics
"""

from __future__ import annotations

import ctypes
from typing import Optional

# --- libcuda binding ---------------------------------------------------------

_libcuda: Optional[ctypes.CDLL] = None
_cuStreamWaitValue32 = None
_init_done = False
_init_error: Optional[str] = None


def _ensure_init() -> None:
    global _libcuda, _cuStreamWaitValue32, _init_done, _init_error
    if _init_done:
        return
    try:
        _libcuda = ctypes.CDLL("libcuda.so.1")
        # CUresult cuStreamWaitValue32_v2(CUstream, CUdeviceptr, cuuint32_t, unsigned int)
        _cuStreamWaitValue32 = _libcuda.cuStreamWaitValue32_v2
        _cuStreamWaitValue32.restype = ctypes.c_int
        _cuStreamWaitValue32.argtypes = [
            ctypes.c_void_p,  # CUstream
            ctypes.c_uint64,  # CUdeviceptr
            ctypes.c_uint32,  # value
            ctypes.c_uint32,  # flags
        ]
    except Exception as e:
        _init_error = repr(e)
    _init_done = True


CU_STREAM_WAIT_VALUE_GEQ = 0
CU_STREAM_WAIT_VALUE_EQ = 1
CU_STREAM_WAIT_VALUE_AND = 2
CU_STREAM_WAIT_VALUE_NOR = 3
CU_STREAM_WAIT_VALUE_FLUSH = 1 << 30  # not supported on A4000


def is_available() -> bool:
    _ensure_init()
    return _cuStreamWaitValue32 is not None


def init_error() -> Optional[str]:
    _ensure_init()
    return _init_error


def stream_wait_value_eq(stream, gpu_addr: int, value: int) -> None:
    """Block the given PyTorch CUDA stream until *gpu_addr == value.

    The wait is enqueued on the stream as a hardware semaphore op; the calling
    Python thread does NOT block.

    Args:
        stream: torch.cuda.Stream (or torch.cuda.streams.Stream)
        gpu_addr: GPU device address (uint64) of an int32 location
        value: 32-bit value to wait for (EQ comparison)
    """
    _ensure_init()
    if _cuStreamWaitValue32 is None:
        raise RuntimeError(f"cuStreamWaitValue32 not available: {_init_error}")
    # PyTorch Stream exposes the underlying CUstream as `cuda_stream` (uintptr).
    cu_stream = ctypes.c_void_p(stream.cuda_stream)
    res = _cuStreamWaitValue32(
        cu_stream,
        gpu_addr,
        value,
        CU_STREAM_WAIT_VALUE_EQ,
    )
    if res != 0:
        raise RuntimeError(
            f"cuStreamWaitValue32 failed: code={res} addr=0x{gpu_addr:x} value={value}"
        )


def stream_wait_value_geq(stream, gpu_addr: int, value: int) -> None:
    """GEQ variant — block until *gpu_addr >= value."""
    _ensure_init()
    if _cuStreamWaitValue32 is None:
        raise RuntimeError(f"cuStreamWaitValue32 not available: {_init_error}")
    cu_stream = ctypes.c_void_p(stream.cuda_stream)
    res = _cuStreamWaitValue32(
        cu_stream,
        gpu_addr,
        value,
        CU_STREAM_WAIT_VALUE_GEQ,
    )
    if res != 0:
        raise RuntimeError(
            f"cuStreamWaitValue32 (GEQ) failed: code={res} addr=0x{gpu_addr:x} value={value}"
        )
