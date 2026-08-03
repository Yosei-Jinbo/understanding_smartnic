# run_zero.py
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import contextlib
from datetime import timedelta
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from common.dataset import get_datasets
from common.model import get_benchmark_model
from common.utils import (
    memory_usage_rank,
    start_profiler,
    SynchronizedWallClockTimer,
)
from zero_wrapper_example import ZeroWrapperExample
from deepspeed.ops.adam import DeepSpeedCPUAdam
from mpi4py import MPI
import glob
import re

# ------------------------------------------------------------
# ロギング
# ------------------------------------------------------------
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def print_rank_0(message: str):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)


# ------------------------------------------------------------
# 分散初期化
# ------------------------------------------------------------
def setup_from_env():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    prop = torch.cuda.get_device_properties(local_rank)
    print(f"[GPU] local_rank={local_rank} name={prop.name}")
    print(f"[GPU] pci_bus_id attr exists? {hasattr(prop, 'pci_bus_id')}")
    if hasattr(prop, "pci_bus_id"):
        print(f"[GPU] pci_bus_id={prop.pci_bus_id}")

    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=15),
        device_id=local_rank,
    )
    return local_rank, rank, world_size


def cleanup():
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def warmup_cpuadam_once(comm, rank):
    if rank == 0:
        try:
            print("[CPUAdam warmup] rank 0: start", flush=True)
            model = torch.nn.Linear(10, 10)
            _ = DeepSpeedCPUAdam(model.parameters(), lr=1e-3)
            print("[CPUAdam warmup] rank 0: done", flush=True)
        except Exception as e:
            print("[CPUAdam warmup] rank 0: FAILED:", e, flush=True)
    comm.Barrier()


def _read_first_int(path: str, default: int = -1) -> int:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return default


def get_allowed_cpus():
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        return list(range(os.cpu_count() or 1))


def _get_numa_cpus(numa_node: int):
    """指定 NUMA ノードの CPU リストを sysfs から取得。"""
    try:
        path = f"/sys/devices/system/node/node{numa_node}/cpulist"
        with open(path) as f:
            text = f.read().strip()
        cpus = []
        for part in text.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                cpus.extend(range(int(lo), int(hi) + 1))
            else:
                cpus.append(int(part))
        return sorted(cpus)
    except Exception:
        return []


def choose_rank_cpu_sets(rank: int, world_size: int, allowed_cpus,
                         cpu_thread_cores: int = 2,
                         gpu_numa_node: int = -1, local_rank: int = -1,
                         local_world_size: int = -1):
    """GPU と同じ NUMA ノードの CPU を local_rank で分割する。"""
    if gpu_numa_node >= 0:
        numa_cpus = _get_numa_cpus(gpu_numa_node)
        numa_allowed = sorted(set(numa_cpus) & set(allowed_cpus))
        if numa_allowed and local_rank >= 0 and local_world_size > 0:
            n = len(numa_allowed)
            base = n // local_world_size
            rem = n % local_world_size
            start = local_rank * base + min(local_rank, rem)
            size = base + (1 if local_rank < rem else 0)
            rset = numa_allowed[start : start + size] or numa_allowed
            k = min(cpu_thread_cores, max(1, len(rset) - 1))
            cpu_set = rset[-k:]
            main_set = rset[:-k] if len(rset) > k else rset
            return main_set, cpu_set

    # フォールバック
    n = len(allowed_cpus)
    if n <= 0:
        return [0], [0]
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    size = base + (1 if rank < rem else 0)
    rset = allowed_cpus[start : start + size] or allowed_cpus
    k = min(cpu_thread_cores, max(1, len(rset) // 2))
    cpu_set = rset[-k:]
    main_set = rset[:-k] if len(rset) > k else rset
    return main_set, cpu_set

def _nvml_try_get_bus_id(local_rank: int) -> str:
    """
    NVMLが使えるなら GPU index -> PCI BusId (BDF) を取得する。
    例: '00000000:65:00.0' or '0000:65:00.0'
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        pci = pynvml.nvmlDeviceGetPciInfo(h)
        bus_id = pci.busId.decode() if isinstance(pci.busId, (bytes, bytearray)) else str(pci.busId)
        # '00000000:65:00.0' -> '0000:65:00.0'
        if bus_id.startswith("00000000:"):
            bus_id = "0000:" + bus_id.split(":", 1)[1]
        # 念のため正規化
        m = re.match(r"^([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])$", bus_id)
        if m:
            return f"{m.group(1).lower()}:{m.group(2).lower()}:{m.group(3).lower()}.{m.group(4)}"
        return bus_id
    except Exception:
        return ""


def _sysfs_find_bdf_by_bus_only(bus_dec: int) -> str:
    """
    PyTorch が返す pci_bus_id が「バス番号(10進)のみ」だった場合の救済。
    /sys/bus/pci/devices/* の BDF を走査して、bus が一致するものを探す。
    例: bus_dec=101 -> bus_hex='65' -> '0000:65:00.0' 等を返す。

    同一 bus に複数デバイスがある環境では曖昧になり得る (その場合は NVML 経由を推奨)。
    """
    bus_hex = f"{bus_dec:02x}"
    cands = glob.glob(f"/sys/bus/pci/devices/*:{bus_hex}:*")
    # cands の例: '/sys/bus/pci/devices/0000:65:00.0'
    if not cands:
        return ""
    # numa_node が -1 でないものを優先（取れない環境もある）
    best = ""
    for p in sorted(cands):
        nn = _read_first_int(os.path.join(p, "numa_node"), default=-1)
        if nn >= 0:
            best = p
            break
    if not best:
        best = sorted(cands)[0]
    return os.path.basename(best)


def get_gpu_numa_node(local_rank: int) -> int:
    """
    GPU が属する NUMA node を推定。
    優先順:
      1) NVML で正しい BDF を取って sysfs を引く
      2) PyTorch の pci_bus_id が整数(=bus番号)なら sysfs を走査して BDF を推定
    """
    # 1) NVML
    bdf = _nvml_try_get_bus_id(local_rank)
    if bdf:
        p = f"/sys/bus/pci/devices/{bdf}"
        if os.path.exists(p):
            return _read_first_int(os.path.join(p, "numa_node"), default=-1)

    # 2) PyTorch fallback（あなたのログのケース：23, 101）
    try:
        prop = torch.cuda.get_device_properties(local_rank)
        if hasattr(prop, "pci_bus_id"):
            bus = prop.pci_bus_id
            # bus が int の想定
            if isinstance(bus, int):
                bdf2 = _sysfs_find_bdf_by_bus_only(bus)
                if bdf2:
                    p2 = f"/sys/bus/pci/devices/{bdf2}"
                    return _read_first_int(os.path.join(p2, "numa_node"), default=-1)
    except Exception:
        pass

    return -1


# ------------------------------------------------------------
# 学習本体を構成するヘルパ (モジュールレベル)
# ------------------------------------------------------------

# Causal LM / MLM モデル判定
CAUSAL_LM_MODELS = {"opt-1.3b", "opt_1.3b", "llama-3b", "llama_3b", "llama-3.2-3b", "llama-7b", "llama_7b", "llama-2-7b"}
MLM_MODELS = {"deberta-xl", "deberta_xl"}


def _build_datasets(model_name, dataset_name, batch_size, num_workers, seq_len,
                    is_causal_lm, is_mlm):
    """モデル種別に応じて (train_dataset, test_dataset) を返す。"""
    if model_name.lower() == "tinynn":
        num_train_samples = 256
        num_test_samples = 64
        train_dataset = TensorDataset(
            torch.randn(num_train_samples, 4),
            torch.randint(0, 4, (num_train_samples,)),
        )
        test_dataset = TensorDataset(
            torch.randn(num_test_samples, 4),
            torch.randint(0, 4, (num_test_samples,)),
        )
    elif is_causal_lm:
        from common.text_dataset import get_causal_lm_datasets
        train_dataset, test_dataset, _tokenizer = get_causal_lm_datasets(
            dataset_name=dataset_name,
            model_name=model_name,
            seq_len=seq_len,
        )
    elif is_mlm:
        from common.text_dataset import get_mlm_datasets
        train_dataset, test_dataset, _tokenizer = get_mlm_datasets(
            dataset_name=dataset_name,
            model_name=model_name,
            seq_len=seq_len,
        )
    else:
        train_dataset, test_dataset = get_datasets(
            dataset_name,
            batch_size,
            num_workers,
            resize_to_imagenet=True, #ViT用
            #resize_to_imagenet=False, #ResNet用
        )
    return train_dataset, test_dataset


def _build_data_loaders(train_dataset, test_dataset, batch_size, num_workers,
                        rank, world_size):
    """train は rank 分割 (DistributedSampler)、test は全 rank 複製で
    (train_sampler, train_loader, test_loader) を返す。"""
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_sampler, train_loader, test_loader


def _bind_cpu_and_comch_cores(rank, world_size, local_rank, num_workers):
    """rank → GPU → NUMA node に基づき CPU set を分割し、ComCh worker/poller 用
    コアを環境変数で C 側に伝える。(main_cpu_set, cpu_thread_set) を返す。"""
    gpu_node = get_gpu_numa_node(local_rank)
    allowed_cpus = get_allowed_cpus()
    cpu_thread_cores = int(os.environ.get("CPU_THREAD_CORES", "2"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE",
                           os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", "1")))
    main_cpu_set, cpu_thread_set = choose_rank_cpu_sets(
        rank=rank, world_size=world_size, allowed_cpus=allowed_cpus,
        cpu_thread_cores=cpu_thread_cores,
        gpu_numa_node=gpu_node, local_rank=local_rank,
        local_world_size=local_world_size,
    )
    # COMCH worker/poller 用コアを main_cpu_set の末尾2つから割り当て
    # main_cpu_set = [12,13,14,15] → worker=14, poller=15, main残り=[12,13]
    if len(main_cpu_set) >= 4:
        comch_worker_core = main_cpu_set[-2]
        comch_poller_core = main_cpu_set[-1]
        main_cpu_set_reduced = main_cpu_set[:-2]
    elif len(main_cpu_set) >= 2:
        comch_worker_core = main_cpu_set[-2]
        comch_poller_core = main_cpu_set[-1]
        main_cpu_set_reduced = main_cpu_set  # コア不足時は共有
    else:
        comch_worker_core = main_cpu_set[0] if main_cpu_set else 0
        comch_poller_core = main_cpu_set[0] if main_cpu_set else 0
        main_cpu_set_reduced = main_cpu_set

    os.environ["COMCH_WORKER_CORE"] = str(comch_worker_core)
    os.environ["COMCH_POLLER_CORE"] = str(comch_poller_core)

    try:
        os.sched_setaffinity(0, set(main_cpu_set_reduced))
    except Exception as e:
        print_rank_0(f"[AFFINITY] process setaffinity failed: {e}")

    print_rank_0(
        f"[COMCH] worker_core={comch_worker_core} poller_core={comch_poller_core} "
        f"main_cores={main_cpu_set_reduced}"
    )
    print_rank_0(
        f"[BIND] rank={rank} local_rank(gpu)={local_rank} gpu_numa_node={gpu_node} "
        f"main_cpu_set={main_cpu_set} cpu_thread_set={cpu_thread_set} num_workers={num_workers}"
    )
    return main_cpu_set, cpu_thread_set


def _move_batch_to_device(batch, device, is_causal_lm, is_mlm):
    """バッチを device に転送する。LM 系は dict、画像分類は (images, labels)。"""
    if is_causal_lm or is_mlm:
        return {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
            "labels": batch["labels"].to(device, non_blocking=True),
        }
    images, labels = batch
    return (images.to(device, non_blocking=True).half(),  # モデルは fp16
            labels.to(device, non_blocking=True))


def _forward_loss(zero_model, loss_fn, inputs, is_causal_lm, is_mlm):
    """forward を実行して float の loss を返す。"""
    if is_causal_lm or is_mlm:
        output = zero_model.forward(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"],
                                    labels=inputs["labels"])
        return output.loss.float()
    images, labels = inputs
    logits = zero_model.forward(images)
    return loss_fn(logits.float(), labels)


def _format_step_stats(label, vals_sec, warmup=0):
    """統計情報を1行にフォーマット。warmup ステップを除外。"""
    import numpy as np
    a = np.array(vals_sec[warmup:]) * 1000.0  # ms
    if len(a) == 0:
        return None
    return (f"{label:>10}: "
            f"count={len(a):>4} (warmup={warmup}) | "
            f"total={a.sum():.1f}ms  mean={a.mean():.2f}ms  std={a.std():.2f}ms | "
            f"min={a.min():.2f}ms  p50={np.median(a):.2f}ms  "
            f"p90={np.percentile(a,90):.2f}ms  p99={np.percentile(a,99):.2f}ms  "
            f"max={a.max():.2f}ms")


class StepTimeStats:
    """ステップ単位の実行時間を記録・集計する。
    全エポック通算で先頭 warmup_steps ステップ分は記録しない。"""

    KEYS = ("fwd", "bwd", "opt", "opt_dpu", "step_total")

    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps
        self.epoch_times = {}       # key -> list of float (seconds), per-epoch
        self.all_times = {}         # key -> list of float (seconds), across all epochs
        self.global_step_count = 0  # 全エポック通算のステップカウンタ（ウォームアップ判定用）

    def record(self, key, elapsed_sec):
        if self.global_step_count < self.warmup_steps:
            return
        self.epoch_times.setdefault(key, []).append(elapsed_sec)

    def finish_step(self):
        self.global_step_count += 1

    def start_epoch(self):
        self.epoch_times = {}

    def accumulate_epoch(self):
        """エポックのデータを全エポック通算に蓄積（ウォームアップは記録時点で除外済み）。"""
        for key, vals in self.epoch_times.items():
            self.all_times.setdefault(key, []).extend(vals)

    def print_epoch(self, epoch_num):
        print(f"========== Per-Step Timing Statistics (Epoch {epoch_num}) ==========")
        for key in self.KEYS:
            vals = self.epoch_times.get(key)
            if not vals:
                continue
            line = _format_step_stats(f"[E{epoch_num}] {key}", vals, warmup=0)
            if line:
                print(line)
        print("=" * 80)

    def print_all(self):
        print(f"========== Per-Step Timing Statistics (All Epochs, global warmup={self.warmup_steps} excluded) ==========")
        for key in self.KEYS:
            vals = self.all_times.get(key)
            if not vals:
                continue
            line = _format_step_stats(f"[All] {key}", vals, warmup=0)
            if line:
                print(line)
        print("=" * 80)


def _train_step_normal(zero_model, loss_fn, inputs, timers, stats,
                       is_causal_lm, is_mlm):
    """通常経路: fwd → bwd → 同期 CPU Adam step (同一 step 内で反映)。"""
    timers("fwd").start()
    t0 = time.perf_counter()
    loss = _forward_loss(zero_model, loss_fn, inputs, is_causal_lm, is_mlm)
    timers("fwd").stop()
    stats.record("fwd", time.perf_counter() - t0)

    timers("bwd").start()
    t0 = time.perf_counter()
    zero_model.backward(loss)
    timers("bwd").stop()
    stats.record("bwd", time.perf_counter() - t0)

    timers("opt").start()
    t0 = time.perf_counter()
    zero_model.step()
    timers("opt").stop()
    stats.record("opt", time.perf_counter() - t0)


def _train_step_dpu(zero_model, loss_fn, inputs, timers, stats,
                    is_causal_lm, is_mlm):
    """DPU 経路: 前 step の勾配で先に Adam step を発行し (delayed 1-step)、
    fwd/bwd と並行実行。boundary で子プロセスの完了を待って fp32→fp16 を反映。"""
    timers("opt_dpu").start()
    t0 = time.perf_counter()
    boundary_flag = zero_model.step_dpu()
    timers("opt_dpu").stop()
    stats.record("opt_dpu", time.perf_counter() - t0)

    timers("fwd").start()
    t0 = time.perf_counter()
    loss = _forward_loss(zero_model, loss_fn, inputs, is_causal_lm, is_mlm)
    timers("fwd").stop()
    stats.record("fwd", time.perf_counter() - t0)

    timers("bwd").start()
    t0 = time.perf_counter()
    zero_model.backward(loss)

    if hasattr(zero_model, "optimizer") and hasattr(zero_model.optimizer, "_partition_all_parameters"):
        zero_model.optimizer._partition_all_parameters()

    timers("bwd").stop()
    stats.record("bwd", time.perf_counter() - t0)

    if boundary_flag:
        zero_model.optimizer.update_new_params()


@torch.no_grad()
def _evaluate_testset(zero_model, test_loader, device, loss_fn, is_causal_lm, is_mlm):
    """test_loader でテストセット全体を評価し (loss, accuracy) を返す。
    ZeRO-3 では forward が param を all-gather する (SmartNIC 版では DOCA AG 経路)。
    no_grad 中は _end_of_forward_hook が推論用 coordinator を各 forward 後に
    reset するため、訓練用 coordinator の trace は壊れない。all_reduce(SUM) は
    test_loader が rank 分割済みでも複製でも正しい平均を返す
    (分子分母とも同じ倍率で相殺)。

    accuracy の定義:
      画像分類     : サンプル単位の top-1 正答率。
      causal LM    : 次トークン予測のトークン精度 (logits[:, :-1] を labels[:, 1:] と
                     比較。labels==-100 の位置は無視)。loss/精度ともトークン重み平均。
      MLM          : マスク位置のトークン精度 (labels!=-100 の位置のみ)。
    """
    zero_model.eval()
    correct = 0.0
    loss_sum = 0.0
    total = 0.0
    for batch in test_loader:
        inputs = _move_batch_to_device(batch, device, is_causal_lm, is_mlm)
        if is_causal_lm or is_mlm:
            labels = inputs["labels"]
            output = zero_model.forward(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"],
                                        labels=labels)
            logits = output.logits
            if is_causal_lm:
                # 次トークン予測: 位置 t の logit で t+1 を当てる
                preds = logits[:, :-1, :].argmax(dim=-1)
                tgt = labels[:, 1:]
            else:  # MLM: マスク位置をその場で予測
                preds = logits.argmax(dim=-1)
                tgt = labels
            mask = tgt != -100          # loss 無視インデックスを除外
            n = float(mask.sum().item())
            correct += (preds[mask] == tgt[mask]).sum().item()
            loss_sum += output.loss.float().item() * n
            total += n
        else:
            images, labels = inputs
            logits = zero_model.forward(images)
            loss = loss_fn(logits.float(), labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    zero_model.train()
    reduced = torch.tensor([loss_sum, float(correct), float(total)],
                           dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    _loss, _correct, _total = reduced.tolist()
    _total = max(1.0, _total)
    return _loss / _total, _correct / _total


# ------------------------------------------------------------
# 学習本体
# ------------------------------------------------------------
def run_zero(use_profiler=False,
             model_name="vit_l_16", dataset_name="cifar10", batch_size=64,
             warmup_iters=None, measure_iters=None, seq_len=1024, num_epochs=None,
             reduce_bucket_size=int(1e8), prefetch_bucket_size=int(1e8),
             max_reuse_distance=0, max_live_parameters=int(1.5e8),
             eval_accuracy=False):
    profiler = None
    try:
        local_rank, rank, world_size = setup_from_env()
        comm = MPI.COMM_WORLD
        warmup_cpuadam_once(comm, local_rank)

        device = torch.device(f"cuda:{local_rank}")

        # ------------------------
        # config
        # ------------------------
        num_workers = 0

        is_causal_lm = model_name.lower() in CAUSAL_LM_MODELS
        is_mlm = model_name.lower() in MLM_MODELS

        # イテレーションベースの計測制御
        if measure_iters is not None:
            _warmup = warmup_iters if warmup_iters is not None else 0
            total_target_iters = _warmup + measure_iters
            WARMUP_STEPS = _warmup
            # nsys profile 区間 (default: measure iter 全部). hang 回避用に末尾を削れる。
            # 例: NSYS_PROFILE_MEASURE_ITERS=25 measure_iters=30 → warmup 後 25 iter capture、
            #     残り 5 iter は nsys 抜きで走る (hang してもレポートは保存済み)。
            _nsys_profile_measure = int(os.environ.get("NSYS_PROFILE_MEASURE_ITERS", str(measure_iters)))
            _nsys_profile_end_iter = _warmup + min(_nsys_profile_measure, measure_iters)
            epochs = num_epochs if num_epochs is not None else 10
        else:
            total_target_iters = None
            _nsys_profile_end_iter = None
            WARMUP_STEPS = int(os.environ.get("TIMING_WARMUP_STEPS", "5"))
            epochs = num_epochs if num_epochs is not None else 5

        # このエポック以下 (0-indexed) は通常経路 (同期 CPU Adam)、超えたら DPU
        # (delayed parameter update) 経路。baseline と同じ環境変数で制御する。
        # デフォルト -1 = 全エポック DPU (従来の smartnic_offload の挙動)。
        # 例: DPU_THRESHOLD=7 → epoch 0-7 通常 / 8以降 DPU。常時通常なら十分大きい値を指定。
        dpu_threshold = int(os.environ.get("DPU_THRESHOLD", "100000000"))

        global_iter = 0

        print_rank_0(
            f"[CONFIG] model={model_name} dataset={dataset_name} batch_size={batch_size} "
            f"epochs={epochs} warmup_iters={warmup_iters} measure_iters={measure_iters} "
            f"is_causal_lm={is_causal_lm} dpu_threshold={dpu_threshold}"
        )

        # rank -> GPU -> NUMA node binding (cpu set 分割) + ComCh コア割り当て
        main_cpu_set, cpu_thread_set = _bind_cpu_and_comch_cores(
            rank, world_size, local_rank, num_workers)

        # ------------------------
        # dataset
        # ------------------------
        train_dataset, test_dataset = _build_datasets(
            model_name, dataset_name, batch_size, num_workers, seq_len,
            is_causal_lm, is_mlm)
        train_sampler, train_loader, test_loader = _build_data_loaders(
            train_dataset, test_dataset, batch_size, num_workers, rank, world_size)

        # ------------------------
        # model
        # ------------------------
        model = get_benchmark_model(model_name)
        model.half()  # FP16に変換してからGPUに転送（大規模モデルのメモリ節約）
        model = model.to(device)
        model_parameters = model.parameters()
        #通常だとmodelがbfloat16で登録されるとAdamもbf16で登録される。
        #しかし、ZeROの実装ではoptimizerをfp32で登録しなおすのでこのままの実装でおけ
        optimizer = DeepSpeedCPUAdam(model_parameters, lr=1e-4, eps=1e-5)
        loss_fn = nn.CrossEntropyLoss()

        # ユーザ実装の ZeRO ラッパ（Stage 3 を想定）
        zero_model = ZeroWrapperExample(
            model, optimizer=optimizer, model_parameters=model_parameters,
            reduce_bucket_size=reduce_bucket_size,
            prefetch_bucket_size=prefetch_bucket_size,
            max_reuse_distance=max_reuse_distance,
            max_live_parameters=max_live_parameters,
        )

        # GPU flag pool 初期化 (デフォルト有効。USE_GPU_FLAG_WAIT=0 で無効化)。
        # AG completion を Python thread でなく GPU compute stream で同期する:
        # cuStreamWaitValue32 が schedule され、Python の wait() は即 return する。
        if os.environ.get("USE_GPU_FLAG_WAIT", "1") == "1":
            try:
                import gpu_flag_pool as _gfp
                import cuda_stream_wait as _csw
                if not _csw.is_available():
                    if rank == 0:
                        print(f"[FlagPool] cuStreamWaitValue not available: {_csw.init_error()}")
                else:
                    pool_size = int(os.environ.get("GPU_FLAG_POOL_SIZE", "4096"))
                    pool = _gfp.GpuFlagPool(device=device, size=pool_size)
                    _gfp.set_global_pool(pool)
                    # ホスト C 側に登録 → DPU に INIT_FLAG_POOL を送信
                    try:
                        import doca_comch_client_pybind as _dcp
                        _dcp.register_flag_pool_py(pool.base_addr, pool.total_bytes)
                    except Exception as _re:
                        if rank == 0:
                            print(f"[FlagPool] register_flag_pool_py failed: {_re} (disabling)")
                        _gfp.set_global_pool(None)
                    else:
                        if rank == 0:
                            print(f"[FlagPool] GPU flag wait enabled: size={pool.size} addr=0x{pool.base_addr:x} storage=pinned host")
            except Exception as _e:
                if rank == 0:
                    print(f"[FlagPool] init failed: {_e}")

        # CPU Adam を別プロセスで起動
        zero_model.start_adam_process(cpu_affinity=cpu_thread_set)

        timers = SynchronizedWallClockTimer()
        timers("opt_dpu")

        timer_every = 10000
        since_last_timer = 0

        if use_profiler:
            log_dir = f"./profiler_log/rank_{rank}"
            os.makedirs(log_dir, exist_ok=True)
            profiler = start_profiler(log_dir=log_dir, use_cuda=True)

        stats = StepTimeStats(WARMUP_STEPS)
        loss_history, accuracy_history = [], []

        # ------------------------
        # Training loop
        # ------------------------
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            zero_model.train()
            use_dpu = epoch > dpu_threshold

            total_steps = len(train_loader)
            next_report = 0.1
            start_t = time.time()
            stats.start_epoch()

            ctx = profiler if profiler else contextlib.nullcontext()
            with ctx as prof:
                for step, batch in enumerate(train_loader, 1):
                    inputs = _move_batch_to_device(batch, device, is_causal_lm, is_mlm)

                    step_t0 = time.perf_counter()

                    # nsys --capture-range=cudaProfilerApi 用: warmup 完了時にキャプチャ開始
                    if total_target_iters is not None and global_iter == WARMUP_STEPS:
                        torch.cuda.synchronize()
                        torch.cuda.profiler.start()

                    # debug step counter
                    from common import debug_params as dbg
                    dbg.set_step(step + epoch * total_steps)

                    optimizer.zero_grad(set_to_none=True)

                    if use_dpu:
                        _train_step_dpu(zero_model, loss_fn, inputs, timers, stats,
                                        is_causal_lm, is_mlm)
                    else:
                        _train_step_normal(zero_model, loss_fn, inputs, timers, stats,
                                           is_causal_lm, is_mlm)

                    stats.record("step_total", time.perf_counter() - step_t0)
                    stats.finish_step()

                    # イテレーションベースの計測制御
                    global_iter += 1
                    # nsys profile の終了: default では total target と同時だが、
                    # NSYS_PROFILE_MEASURE_ITERS で早めに切って末尾の hang を回避できる
                    if total_target_iters is not None and global_iter == _nsys_profile_end_iter:
                        torch.cuda.synchronize()
                        # nsys 版は capture-range-end=stop-shutdown でこの直後にプロセスが
                        # kill され、エポック末の集計ログに到達しない。MEASURE_HOST_XFER=1 の
                        # ときは、ここ (計測区間終了時) で通常ログ相当の集計を出力しておく。
                        if rank == 0 and os.environ.get("MEASURE_HOST_XFER", "0") == "1":
                            print(f"[Epoch {epoch+1}/{epochs}] | train time: {time.time() - start_t:.2f}s "
                                  f"(nsys measure cutoff @ iter {global_iter})")
                            stats.print_epoch(epoch + 1)
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                            sys.stdout.flush()
                        torch.cuda.profiler.stop()
                    if total_target_iters is not None and global_iter >= total_target_iters:
                        print_rank_0(f"Reached target iterations ({global_iter}/{total_target_iters}), stopping.")
                        break

                    if profiler:
                        prof.step()

                    if stats.global_step_count >= WARMUP_STEPS:
                        if since_last_timer % timer_every == 0:
                            timer_names = ["fwd", "bwd", "opt_dpu"] if use_dpu else ["fwd", "bwd", "opt"]
                            timers.log(
                                names=timer_names,
                                normalizer=max(1, float(timer_every)),
                                reset=True,
                                memory_breakdown=True,
                                model=model,
                                device=device,
                            )
                        since_last_timer += 1

                    progress = step / total_steps
                    if progress >= next_report or step == total_steps:
                        print_rank_0(
                            f"Epoch {epoch+1}/{epochs} Progress: {progress*100:.1f}% "
                            f"(step {step}/{total_steps})"
                        )
                        next_report += 0.1

            # ---- 各エポック終了後にテストセットで評価し accuracy_history に記録 ----
            # --eval-accuracy 指定時のみ。全 rank が collective forward + all_reduce に
            # 参加するため if rank==0 の外で呼ぶ。
            if eval_accuracy:
                test_loss, test_acc = _evaluate_testset(
                    zero_model, test_loader, device, loss_fn, is_causal_lm, is_mlm)
                loss_history.append(test_loss)
                accuracy_history.append(test_acc)
                print_rank_0(f"[Epoch {epoch+1}/{epochs}] test_loss={test_loss:.4f} "
                             f"test_acc={test_acc:.4f}")

            if rank == 0:
                print(f"[Epoch {epoch+1}/{epochs}] | train time: {time.time() - start_t:.2f}s")
                stats.print_epoch(epoch + 1)
                stats.accumulate_epoch()

            # イテレーションベースの計測: エポックループもbreak
            if total_target_iters is not None and global_iter >= total_target_iters:
                break

        if rank == 0:
            stats.print_all()

            try:
                print(memory_usage_rank(model, optimizer=optimizer))
            except Exception:
                pass
            if eval_accuracy:
                print("Loss history:", loss_history)
                print("Accuracy history:", accuracy_history)

    finally:
        if profiler and hasattr(profiler, "stop"):
            profiler.stop()
        zero_model.stop_adam_process()
        cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--model", type=str, default="vit_l_16",
                        help="Model name (vit_l_16, opt-1.3b, llama-7b, etc.)")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name (cifar10, wikitext-103, etc.)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-iters", type=int, default=None,
                        help="ウォームアップイテレーション数 (計測から除外)")
    parser.add_argument("--measure-iters", type=int, default=None,
                        help="計測イテレーション数 (指定時はイテレーションベースで停止)")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Causal LM のシーケンス長 (default: 1024)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="エポック数 (--measure-iters未指定時のデフォルトは5)")
    parser.add_argument("--debug-params", action="store_true",
                        help="パラメータ検証デバッグ出力を有効化 (最初の数ステップのみ)")
    parser.add_argument("--reduce-bucket-size", type=float, default=1e8)
    parser.add_argument("--prefetch-bucket-size", type=float, default=1e8)
    parser.add_argument("--max-reuse-distance", type=float, default=0)
    parser.add_argument("--max-live-parameters", type=float, default=1.5e8)
    parser.add_argument("--eval-accuracy", action="store_true",
                        help="各エポック後に test_loader で accuracy を評価する "
                             "(--epochs と併用。--measure-iters 指定時は途中打ち切りで評価に到達しない)")
    args = parser.parse_args()

    if args.debug_params:
        from common import debug_params as dbg
        dbg.enable()

    run_zero(
        use_profiler=args.profiler,
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        seq_len=args.seq_len,
        num_epochs=args.epochs,
        reduce_bucket_size=int(args.reduce_bucket_size),
        prefetch_bucket_size=int(args.prefetch_bucket_size),
        max_reuse_distance=int(args.max_reuse_distance),
        max_live_parameters=int(args.max_live_parameters),
        eval_accuracy=args.eval_accuracy,
    )


if __name__ == "__main__":
    main()