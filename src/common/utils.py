# 訓練補助関数(損失計算、評価、ログ出力などすべての手法で利用するような関数をこのファイルにまとめておく)

import time
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from .log_utils import logger
import logging
from statistics import median
import psutil


# =========================
# 低レベル: GPUメモリ内訳取得
# =========================
def _tensor_bytes(t: torch.Tensor) -> int:
    """テンソルが占有するバイト数 (= 要素数 * 要素サイズ)"""
    return t.numel() * t.element_size()

def gpu_memory_breakdown(model: Optional[torch.nn.Module] = None, device: Optional[int] = None) -> dict:
    """
    現在の GPU メモリ状態を辞書で返す（GB単位）
    """
    if not torch.cuda.is_available():
        return {}

    dev = device if device is not None else torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev)
    reserv = torch.cuda.memory_reserved(dev)
    max_alloc = torch.cuda.max_memory_allocated(dev)
    max_reserv = torch.cuda.max_memory_reserved(dev)

    param_b = grad_b = buf_b = 0
    if model is not None:
        for p in model.parameters():
            if p.is_cuda and (p.device.index == dev if isinstance(dev, int) else p.device == dev):
                param_b += _tensor_bytes(p)
                if p.grad is not None and p.grad.is_cuda:
                    grad_b += _tensor_bytes(p.grad)
        for b in model.buffers():
            if b.is_cuda and (b.device.index == dev if isinstance(dev, int) else b.device == dev):
                buf_b += _tensor_bytes(b)

    other_b = max(0, alloc - (param_b + grad_b + buf_b))

    GB = 1024 ** 3
    return {
        "alloc_GB": alloc / GB,
        "reserved_GB": reserv / GB,
        "max_alloc_GB": max_alloc / GB,
        "max_reserved_GB": max_reserv / GB,
        "params_GB": param_b / GB,
        "grads_GB": grad_b / GB,
        "buffers_GB": buf_b / GB,
        "other_GB": other_b / GB,
    }

# =========================
# 1. スループット精密計測: ThroughputMeter
# =========================
class ThroughputMeter:
    def __init__(self,
                 warmup_steps: int = 5,
                 steps_per_output: int = 100,
                 log_fn=None,
                 report_vm_swap: bool = True,
                 report_gpu_mem: bool = True,
                 device: Optional[int] = None,
                 model: Optional[torch.nn.Module] = None):
        self.warmup_steps = warmup_steps
        self.steps_per_output = steps_per_output
        self.log = log_fn or logger.info
        self.report_vm_swap = report_vm_swap
        self.report_gpu_mem = report_gpu_mem
        self.device = device
        self.model = model

        self._skipped_steps = 0
        self._warmup_done = False
        self.steps = 0
        self.num_samples = 0
        self._step_times = []

    def _rank0(self) -> bool:
        return (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

    def start(self):
        self._skipped_steps = 0
        self._warmup_done = False
        self.steps = 0
        self.num_samples = 0
        self._step_times = []

    def update(self, batch_size: int, step_time: float):
        if not self._warmup_done:
            self._skipped_steps += 1
            if self._skipped_steps >= self.warmup_steps:
                self._warmup_done = True
                self.steps = 0
                self.num_samples = 0
                self._step_times.clear()
            return

        self.num_samples += batch_size
        self.steps += 1
        self._step_times.append(step_time)

        if self.steps_per_output > 0 and (self.steps % self.steps_per_output == 0):
            self.report(prefix=f"[step {self.steps}]")

    def compute(self):
        if self.steps == 0:
            return 0.0, 0.0
        total_time = sum(self._step_times)
        return self.num_samples / total_time, total_time / self.steps

    def percentiles(self) -> dict:
        if not self._step_times:
            return {"p50": 0.0, "p95": 0.0, "max": 0.0}
        sorted_t = sorted(self._step_times)
        p50 = median(sorted_t)
        p95 = sorted_t[int(len(sorted_t) * 0.95) - 1] if len(sorted_t) > 1 else sorted_t[0]
        return {"p50": p50, "p95": p95, "max": max(sorted_t)}

    def report(self, prefix: str = ""):
        if not self._rank0():
            return

        sps, avg_step = self.compute()
        p = self.percentiles()

        self.log(f"{prefix} Throughput: {sps:.2f} samples/sec | "
                 f"Avg step: {avg_step:.4f}s | "
                 f"p50: {p['p50']:.4f}s | p95: {p['p95']:.4f}s | max: {p['max']:.4f}s | "
                 f"steps: {self.steps}")

        if self.report_vm_swap:
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            self.log(f"{prefix} VM: {vm.percent:.1f}% | Swap: {sw.percent:.1f}%")

        if self.report_gpu_mem and torch.cuda.is_available():
            torch.cuda.synchronize()
            d = gpu_memory_breakdown(self.model, self.device)
            if d:
                self.log(
                    f"{prefix} GPU mem: "
                    f"alloc={d['alloc_GB']:.5f}GB (max={d['max_alloc_GB']:.5f}GB) | "
                    f"reserved={d['reserved_GB']:.5f}GB (max={d['max_reserved_GB']:.5f}GB) | "
                    f"params={d['params_GB']:.5f}GB grads={d['grads_GB']:.5f}GB "
                    f"buffers={d['buffers_GB']:.5f}GB other={d['other_GB']:.5f}GB"
                )

    def summary(self):
        self.report(prefix="[summary]")

    def __call__(self, batch_size: int):
        return _StepTimer(self, batch_size)


class _StepTimer:
    def __init__(self, meter: ThroughputMeter, batch_size: int):
        self.meter = meter
        self.batch_size = batch_size
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        self.meter.update(self.batch_size, elapsed)
        

# ================================
# 2. 時間計測タイマー
# ================================    
def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            #logger.info(message)
            pass
    else:
        #logger.info(message)
        pass


class SynchronizedWallClockTimer:
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

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
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
        

# ================================
# 3. GPUメモリ詳細（GB表記）
# ================================

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

def see_memory_usage(message, model: torch.nn.Module = None):
    if torch.distributed.is_initialized() and not torch.distributed.get_rank() == 0:
        return

    logger.info(message)

    GB = 1024 ** 3

    alloc_b     = torch.cuda.memory_allocated()
    reserv_b    = torch.cuda.memory_reserved()
    max_alloc_b = torch.cuda.max_memory_allocated()
    max_reserv_b= torch.cuda.max_memory_reserved()

    logger.info("Memory Allocated: %.5f GB", alloc_b / GB)
    logger.info("Max Memory Allocated: %.5f GB", max_alloc_b / GB)
    logger.info("Memory Reserved (cached): %.5f GB", reserv_b / GB)
    logger.info("Max Memory Reserved: %.5f GB", max_reserv_b / GB)

    if model is not None:
        def tensor_bytes(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()

        param_b = sum(tensor_bytes(p) for p in model.parameters() if p.is_cuda)
        grad_b  = sum(tensor_bytes(p.grad) for p in model.parameters()
                      if p.grad is not None and p.grad.is_cuda)
        buf_b   = sum(tensor_bytes(b) for b in model.buffers() if b.is_cuda)

        other_gb = (alloc_b - (param_b + grad_b + buf_b)) / GB

        logger.info("Params: %.5f GB", param_b / GB)
        logger.info("Grads: %.5f GB", grad_b / GB)
        logger.info("Buffers: %.5f GB", buf_b / GB)
        logger.info("Other: %.5f GB", other_gb)
    
    
# ================================
# 4. 通信時間・通信帯域 (Profiler)
# ================================
'''
import os, time, glob
from contextlib import contextmanager
import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    tensorboard_trace_handler,
    schedule as prof_schedule,
    record_function,
)

def start_profiler(
    log_dir: str = "./profiler_log",
    use_cuda: bool = True,
    *,
    # 収集ウィンドウ（短くするほど小さくなる）
    wait: int = 20,      # 収集しない期間（スケジュール）
    warmup: int = 5,     # 計測のウォームアップ
    active: int = 10,    # 実際に保存する期間（ここだけファイルに出る）
    repeat: int = 1,     # 周回回数
    # 出力と収集項目
    output_mode: str = "chrome",  # "tb" | "chrome" | "both"
    record_shapes: bool = False,
    with_stack: bool = False,
    profile_memory: bool = False,   # PyTorch側メモリ計測
    with_modules: bool = False,
    # Chrome trace まわり
    chrome_pattern: str = "trace_{ts}.json",
    keep_last_n_traces: int = 5,   # 古いトレース自動削除
):
    """
    Returns:
        torch.profiler.profile コンテキストマネージャ
    """
    os.makedirs(log_dir, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    sch = prof_schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    wrote = {"done": False}  # クロージャで可変参照
    
    # ---- on_trace_ready handlers ----
    def _save_tb(p):
        # TensorBoard向け（サイズはやや大きめ）
        tensorboard_trace_handler(log_dir)(p)

    def _save_chrome(p):
        # 軽量な1ファイル出力（Chrome trace）
        ts = int(time.time())
        path = os.path.join(log_dir, chrome_pattern.format(ts=ts))
        p.export_chrome_trace(path)

        # 古いファイルを削除して増えすぎを防止
        traces = sorted(glob.glob(os.path.join(log_dir, "trace_*.json")))
        if keep_last_n_traces is not None and len(traces) > keep_last_n_traces:
            for old in traces[:-keep_last_n_traces]:
                try:
                    os.remove(old)
                except Exception:
                    pass

    def _save_tb_once(p):
        if wrote["done"]:
            return
        tensorboard_trace_handler(log_dir)(p)
        wrote["done"] = True

    def _save_chrome_once(p):
        if wrote["done"]:
            return
        ts = int(time.time())
        path = os.path.join(log_dir, f"trace_{ts}.json")
        p.export_chrome_trace(path)
        wrote["done"] = True

    if output_mode == "tb":
        on_trace_ready = _save_tb_once
    elif output_mode == "chrome":
        on_trace_ready = _save_chrome_once
    elif output_mode == "both":
        def _both_once(p):
            if wrote["done"]:
                return
            tensorboard_trace_handler(log_dir)(p)
            ts = int(time.time())
            p.export_chrome_trace(os.path.join(log_dir, f"trace_{ts}.json"))
            wrote["done"] = True
        on_trace_ready = _both_once
    else:
        raise ValueError(f"Unknown output_mode: {output_mode} (expected 'tb'|'chrome'|'both')")

    return profile(
        activities=activities,
        schedule=sch,
        on_trace_ready=on_trace_ready,
        record_shapes=record_shapes,
        with_stack=with_stack,
        profile_memory=profile_memory,
        with_modules=with_modules,
    )
'''
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


# ================================
# 5. 学習品質
# ================================
from torch import amp
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(labels)

def evaluate(model, dataloader, device):
    model.eval()
    total_acc, total_loss, count = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = None
            with amp.autocast("cuda", dtype=torch.bfloat16): #ZeRO Stage 3と相性よくないのでZeROは別で全部bfloat16で評価
                outputs = model(images)
                loss = criterion(outputs.float(), labels)
            total_loss += loss.item() * labels.size(0)

            _, preds = torch.max(outputs, dim=1)
            total_acc += torch.sum(preds == labels).item()
            count += labels.size(0)

    return total_loss / count, total_acc / count

def evaluate_zero3(model, dataloader, device):
    model.eval()
    total_acc, total_loss, count = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            # ★ 修正点：float() をやめて bfloat16 に統一
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)  # BF16入力 × BF16重みでOK
            loss = criterion(outputs.float(), labels)  # loss計算のみFP32

            total_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            total_acc += (preds == labels).sum().item()
            count += labels.size(0)

    return total_loss / count, total_acc / count

def plot_loss_curve(train_losses, val_losses=None):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    if val_losses is not None:
        plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


###########################################
#NVTX PROFILER
###########################################
# --- 追加: 先頭の import に追記 ---
from torch.profiler import record_function  # PyTorch Profiler用の区間名

# --- 追加: 超シンプルな NVTX ヘルパ（ガード無しで短縮）---
from contextlib import contextmanager
@contextmanager
def nvtx_range(msg: str):
    torch.cuda.nvtx.range_push(msg)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()