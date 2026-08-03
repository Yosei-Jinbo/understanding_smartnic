# 訓練補助関数(損失計算、評価、ログ出力などすべての手法で利用するような関数をこのファイルにまとめておく)

import os
import time
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from typing import Optional
from .log_utils import logger
import logging


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            #logger.info(message)
            pass
    else:
        #logger.info(message)
        pass


class SynchronizedWallClockTimer:
    # start()/stop() のたびに torch.cuda.synchronize() が入り、launch-ahead や
    # プリフェッチとのオーバーラップを潰して計測対象の挙動自体を変えてしまうため、
    # デフォルトでは無効 (ENABLE_STEP_TIMERS=1 でのみ有効化)。
    # 無効時は start/stop/log がすべて no-op になり同期は一切発生しない。
    class Timer:
        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            assert not self.started_
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self, reset=False, record=False): #ZeRO Stage3のためについている引数
            assert self.started_
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            started_ = self.started_
            if self.started_:
                self.stop()
            elapsed_ = self.elapsed_
            if reset:
                self.reset()
            if started_:
                self.start()
            return elapsed_

    class NullTimer:
        def start(self):
            pass

        def stop(self, reset=False, record=False):
            pass

        def reset(self):
            pass

        def elapsed(self, reset=True):
            return 0.0

    def __init__(self, enabled=None):
        if enabled is None:
            enabled = os.environ.get("ENABLE_STEP_TIMERS", "0") == "1"
        self.enabled = enabled
        self.timers = {}
        self._null_timer = self.NullTimer()

    def __call__(self, name):
        if not self.enabled:
            return self._null_timer
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    @staticmethod
    def _tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    @staticmethod
    def memory_usage(model: torch.nn.Module = None,
                     device: int | torch.device | None = None) -> str:
        if not torch.cuda.is_available():
            return "CUDA not available."

        if device is None:
            dev = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            dev = device.index if device.type == "cuda" else torch.cuda.current_device()
        else:
            dev = device

        alloc_b      = torch.cuda.memory_allocated(dev)
        reserved_b   = torch.cuda.memory_reserved(dev)
        max_alloc_b  = torch.cuda.max_memory_allocated(dev)
        max_res_b    = torch.cuda.max_memory_reserved(dev)

        GB = 1024 ** 3

        head = (f"alloc={alloc_b/GB:.5f} GB | max_alloc={max_alloc_b/GB:.5f} GB | "
                f"reserved={reserved_b/GB:.5f} GB | max_reserved={max_res_b/GB:.5f} GB")

        if model is None:
            return head

        param_b = grad_b = buf_b = 0
        for p in model.parameters():
            if p.is_cuda and (p.device.index == dev):
                param_b += SynchronizedWallClockTimer._tensor_bytes(p)
                if p.grad is not None and p.grad.is_cuda:
                    grad_b += SynchronizedWallClockTimer._tensor_bytes(p.grad)
        for b in model.buffers():
            if b.is_cuda and (b.device.index == dev):
                buf_b += SynchronizedWallClockTimer._tensor_bytes(b)

        other_gb = max(0.0, (alloc_b - (param_b + grad_b + buf_b)) / GB)

        detail = (f"params={param_b/GB:.5f} GB | grads={grad_b/GB:.5f} GB | "
                  f"buffers={buf_b/GB:.5f} GB | other={other_gb:.5f} GB")

        return f"{head} | {detail}"

    def log(self,
            names,
            normalizer: float = 1.0,
            reset: bool = True,
            memory_breakdown: bool = False,
            model: torch.nn.Module | None = None,
            device: int | torch.device | None = None):
        if not self.enabled:
            return
        assert normalizer > 0.0
        s = 'time (ms)'
        for name in names:
            elapsed_ms = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            s += f" | {name}: {elapsed_ms:.2f}"

        if memory_breakdown:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            s += " | " + self.memory_usage(model=model, device=device)

        print_rank_0(s)
        

def memory_usage_rank(model: Optional[torch.nn.Module] = None,
                      rank: Optional[int] = None,
                      optimizer: Optional[torch.optim.Optimizer] = None,
                      human_readable: bool = True) -> str:
    if not torch.cuda.is_available():
        return "CUDA not available."

    if rank is None:
        rank = torch.cuda.current_device()
    dev = torch.device(f"cuda:{rank}")
    GB = 1024 ** 3

    def fmt_gb(x):
        return f"{x/GB:.5f} GB" if human_readable else str(x)

    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    alloc_b     = torch.cuda.memory_allocated(dev)
    reserv_b    = torch.cuda.memory_reserved(dev)
    max_alloc_b = torch.cuda.max_memory_allocated(dev)
    max_reserv_b= torch.cuda.max_memory_reserved(dev)

    parts = [f"[cuda:{rank}]",
             f"mem_allocated={fmt_gb(alloc_b)}",
             f"mem_reserved={fmt_gb(reserv_b)}",
             f"max_mem_allocated={fmt_gb(max_alloc_b)}",
             f"max_mem_reserved={fmt_gb(max_reserv_b)}"]

    param_b = grad_b = buf_b = opt_state_b = 0

    if model is not None:
        param_b = sum(tensor_bytes(p) for p in model.parameters()
                      if p.is_cuda and p.device.index == rank)
        grad_b  = sum(tensor_bytes(p.grad) for p in model.parameters()
                      if (p.grad is not None and p.grad.is_cuda and p.grad.device.index == rank))
        buf_b   = sum(tensor_bytes(b) for b in model.buffers()
                      if b.is_cuda and b.device.index == rank)

    if optimizer is not None:
        for state in optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor) and v.is_cuda and v.device.index == rank:
                    opt_state_b += tensor_bytes(v)

    other_gb = (alloc_b - (param_b + grad_b + buf_b + opt_state_b)) / GB

    parts.extend([
        f"params={param_b/GB:.5f} GB",
        f"grads={grad_b/GB:.5f} GB",
        f"buffers={buf_b/GB:.5f} GB",
        f"opt_state={opt_state_b/GB:.5f} GB",
        f"other={other_gb:.5f} GB" if human_readable else str(other_gb)
    ])

    return " | ".join(parts)

import os, time, glob
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule as prof_schedule
def start_profiler(
    log_dir: str = "./profiler_log",
    use_cuda: bool = True,
    *,
    # 収集ウィンドウ（短くするほど小さくなる）
    wait: int = 20,      # 収集しない期間
    warmup: int = 5,     # 計測のウォームアップ
    active: int = 3,    # 実際に保存する期間（ここだけファイルに出る）
    repeat: int = 1,     # 周回回数
    # 出力と収集項目（軽量設定）
    use_tensorboard: bool = True,  # TrueにするとTBへ出力（サイズは増えがち）
    record_shapes: bool = False,
    with_stack: bool = False,
    profile_memory: bool = False,
    with_modules: bool = False,
    # ログ数の上限（古いトレースを自動で消す）
    keep_last_n_traces: int = 3,
):
    os.makedirs(log_dir, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    sch = prof_schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

    def _on_trace_ready_tb(p):
        # TensorBoard向け（サイズは大きめ）
        return tensorboard_trace_handler(log_dir)(p)

    def _on_trace_ready_chrome(p):
        # 軽量な1ファイル出力（Chrome trace）
        ts = int(time.time())
        path = os.path.join(log_dir, f"trace_{ts}.json")
        p.export_chrome_trace(path)

        # 古いファイルを削除して増えすぎを防止
        traces = sorted(glob.glob(os.path.join(log_dir, "trace_*.json")))
        if keep_last_n_traces is not None and len(traces) > keep_last_n_traces:
            for old in traces[:-keep_last_n_traces]:
                try:
                    os.remove(old)
                except Exception:
                    pass

    on_trace_ready = _on_trace_ready_tb if use_tensorboard else _on_trace_ready_chrome

    return profile(
        activities=activities,
        schedule=sch,
        on_trace_ready=on_trace_ready,
        record_shapes=record_shapes,
        with_stack=with_stack,
        profile_memory=profile_memory,
        with_modules=with_modules,
    )
