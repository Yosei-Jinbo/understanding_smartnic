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
import nvtx as pnvtx

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
from transformers import DataCollatorWithPadding
from common.text_dataset import get_text_datasets

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


# ---- ステップ内訳 NVTX (STEP_NVTX=1 で有効): compute_idle_decomp の "Other" 区間を細分する計装 ----
_STEP_NVTX = os.environ.get("STEP_NVTX", "0") == "1"


def _step_push(label: str) -> None:
    if _STEP_NVTX:
        torch.cuda.nvtx.range_push(label)


def _step_pop() -> None:
    if _STEP_NVTX:
        torch.cuda.nvtx.range_pop()


def instrument_w_nvtx(func):
    if hasattr(torch.cuda.nvtx, "range"):

        def wrapped_fn(*args, **kwargs):
            with torch.cuda.nvtx.range(func.__qualname__):
                return func(*args, **kwargs)

        return wrapped_fn
    else:
        return func


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


def set_seed(seed: int, rank: int):
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_final_params(model, rank: int, path_prefix: str):
    """訓練後の各パラメータのローカルシャード (ds_tensor) を ds_id 順に連結し、
    ビット再現性検証用に <path_prefix>.rank<rank>.pt へ保存 + sha256 を表示する。
    数値に影響しないはずの変更の前後で、このハッシュが完全一致することを
    確認するためのもの。"""
    import hashlib
    shards = []
    for p in sorted(model.parameters(), key=lambda x: getattr(x, "ds_id", -1)):
        t = getattr(p, "ds_tensor", None)
        src = t if t is not None else p.data
        shards.append(src.detach().to(torch.float32).cpu().contiguous().view(-1))
    flat = torch.cat(shards) if shards else torch.empty(0)
    out = f"{path_prefix}.rank{rank}.pt"
    torch.save(flat, out)
    h = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
    print(f"[DUMP_FINAL_PARAMS] rank={rank} numel={flat.numel()} "
          f"sha256={h} -> {out}", flush=True)


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

# ------------------------------------------------------------
# 学習本体を構成するヘルパ (モジュールレベル)
# ------------------------------------------------------------

# Causal LM / MLM モデル判定
CAUSAL_LM_MODELS = {"opt-1.3b", "opt_1.3b", "llama-3b", "llama_3b", "llama-7b", "llama_7b", "llama-2-7b"}
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


def _bind_cpu_sets(rank, world_size, local_rank, num_workers):
    """rank → GPU → NUMA node に基づき CPU set を分割し、main 側に setaffinity する。
    (main_cpu_set, cpu_thread_set) を返す。
    この環境では GPU は常に NUMA node 1 に接続されているため直書きする。"""
    gpu_node = 1
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
    try:
        os.sched_setaffinity(0, set(main_cpu_set))
    except Exception as e:
        print_rank_0(f"[AFFINITY] process setaffinity failed: {e}")

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


def _print_all_comm_stats():
    """通信/転送の時間・回数・バイトはコード内計測を廃止し nsys に一本化した。"""
    pass


def _train_step_normal(zero_model, loss_fn, inputs, timers, stats,
                       is_causal_lm, is_mlm):
    """通常経路: fwd → bwd → 同期 Adam step (同一 step 内で反映)。"""
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
    _step_push("step:opt")
    try:
        zero_model.step()
    finally:
        _step_pop()
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
        _step_push("step:update_new_params")
        try:
            zero_model.optimizer.update_new_params()
        finally:
            _step_pop()


@torch.no_grad()
def _evaluate_testset(zero_model, test_loader, device, loss_fn, is_causal_lm, is_mlm):
    """テストセット全体を評価し (loss, accuracy) を返す。no_grad 中は推論用 coordinator が
    forward 後に reset されるため訓練用 trace は壊れない。accuracy は画像=top-1、
    causal LM=次トークン精度、MLM=マスク位置精度 (labels==-100 は無視)。"""
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


def _build_lr_scheduler(optimizer, use_ema, lr_decay):
    """--ema / --lr-decay 指定時に ExponentialLR を構築 (未指定なら None、既定 gamma=0.95)。
    DPU (遅延1step更新) の精度ダメージは lr に比例するため減衰が崩落回避に効く。
    スケジュール計算は親の carrier optimizer で行い、worker へは set_lr で毎エポック転送する。"""
    if not use_ema and lr_decay == 1.0:
        return None
    _gamma = lr_decay if lr_decay != 1.0 else 0.95
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=_gamma)
    print_rank_0(f"[LR] ExponentialLR gamma={_gamma}")
    # carrier optimizer の param_groups は空なのでこの step() は no-op だが、
    # scheduler のカウンタが進み "lr_scheduler.step() before optimizer.step()" 警告を抑止できる。
    optimizer.step()
    return lr_scheduler


# ------------------------------------------------------------
# 学習本体
# ------------------------------------------------------------
def run_zero(use_profiler=False, use_bf16=False, use_ema=False,
             model_name="vit_l_16", dataset_name="cifar10", batch_size=64,
             warmup_iters=None, measure_iters=None, seq_len=1024, num_epochs=None,
             reduce_bucket_size=int(1e8), prefetch_bucket_size=int(1e8),
             max_reuse_distance=0, max_live_parameters=int(1.5e8),
             eval_accuracy=False, lr_decay=1.0, no_offload=False):
    profiler = None
    zero_model = None
    try:
        local_rank, rank, world_size = setup_from_env()
        comm = MPI.COMM_WORLD
        warmup_cpuadam_once(comm, local_rank)

        # 再現性テスト用: TEST_SEED があれば全 rank 同一シードで固定し、CUDA カーネルも
        # 決定論化する (dataset の torch.randn / モデル初期化 / cuDNN・cublas を固定)。
        # 大きいモデル (vit 等) の run 間ビット再現性を確保するためのもの。通常実行には無影響。
        # cublas 決定論には CUBLAS_WORKSPACE_CONFIG=:4096:8 を環境変数で渡す必要がある。
        _test_seed = os.environ.get("TEST_SEED")
        if _test_seed is not None:
            _s = int(_test_seed)
            torch.manual_seed(_s)
            torch.cuda.manual_seed_all(_s)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # warn_only=True: 決定論実装が無い op は例外にせず警告のみ (落とさない)
            torch.use_deterministic_algorithms(True, warn_only=True)

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
            epochs = num_epochs if num_epochs is not None else 9999
        else:
            total_target_iters = None
            _nsys_profile_end_iter = None
            WARMUP_STEPS = int(os.environ.get("TIMING_WARMUP_STEPS", "5"))
            epochs = num_epochs if num_epochs is not None else 5

        # このエポック以下は正常経路 (同期 step)、超えたら DPU (delayed parameter update) 経路。
        # 既定 10**9 = 全エポック通常経路。環境変数で上書き可能 (例: DPU_THRESHOLD=-1 で全エポック DPU)。
        dpu_threshold = int(os.environ.get("DPU_THRESHOLD", "1000000000"))
        if no_offload:
            # 純粋 ZeRO-3 (GPU) モード: CPU Adam worker がないため DPU は使えない。
            if "DPU_THRESHOLD" in os.environ:
                print_rank_0("[WARN] --no-offload では DPU は無効です (DPU_THRESHOLD は無視)")
            dpu_threshold = 10**9
        global_iter = 0

        print_rank_0(
            f"[CONFIG] model={model_name} dataset={dataset_name} batch_size={batch_size} "
            f"epochs={epochs} warmup_iters={warmup_iters} measure_iters={measure_iters} "
            f"is_causal_lm={is_causal_lm} "
            f"mode={'zero3-gpu (no-offload)' if no_offload else 'zero-offload (cpu)'}"
        )

        # rank -> GPU -> NUMA node binding (cpu set 分割)
        main_cpu_set, cpu_thread_set = _bind_cpu_sets(
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
            offload=not no_offload,
        )

        # CPU Adam を別プロセスで起動 (--no-offload 時は optimizer 側で no-op)
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

        lr_scheduler = _build_lr_scheduler(optimizer, use_ema, lr_decay)

        # ------------------------
        # Training loop
        # ------------------------
        # Phase 20: AG/RS stall event-bracket tracker (env: MEASURE_AG_STALL=1)
        try:
            from common import stall_event_tracker as _set
            _stall_tracker = _set.get_global() if _set.is_enabled() else None
        except Exception:
            _stall_tracker = None
        if _stall_tracker is not None:
            print_rank_0("[Phase 20] AG/RS stall event-bracket tracker ENABLED "
                         "(MEASURE_AG_STALL=1).")

        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            zero_model.train()
            use_dpu = epoch > dpu_threshold
            if _stall_tracker is not None:
                _stall_tracker.set_epoch(epoch)

            if lr_scheduler is not None:
                _lr_e = optimizer.param_groups[0]["lr"]
                if hasattr(zero_model.optimizer, "set_lr"):
                    zero_model.optimizer.set_lr(_lr_e)
                else:
                    print_rank_0("[LR][WARN] optimizer に set_lr がないため "
                                 "worker への lr 反映はスキップ")
                print_rank_0(f"[LR] epoch {epoch+1}: lr={_lr_e:.3e}")

            total_steps = len(train_loader)
            next_report = 0.1
            start_t = time.time()
            stats.start_epoch()

            ctx = profiler if profiler else contextlib.nullcontext()
            with ctx as prof:
                for step, batch in enumerate(train_loader, 1):
                    inputs = _move_batch_to_device(batch, device, is_causal_lm, is_mlm)

                    step_t0 = time.perf_counter()

                    # Phase 20: tracker に現 iter を伝える
                    if _stall_tracker is not None:
                        _stall_tracker.set_iter(global_iter)

                    # nsys --capture-range=cudaProfilerApi 用: warmup 完了時にキャプチャ開始
                    if total_target_iters is not None and global_iter == WARMUP_STEPS:
                        torch.cuda.synchronize()
                        # 全ランク profiling では、ランク間の到達スキュー (実測で最大 741ms) が
                        # そのままキャプチャ窓のズレになり、「誰が誰を待っているか」を
                        # タイムライン上で突き合わせられなくなる。窓の開始を揃える。
                        #
                        # baseline は torchrun 起動 (mpirun ではない) のため、
                        # MPI.COMM_WORLD は各プロセスで size=1 の singleton になり
                        # comm.Barrier() は何も同期しない (実測確認済み)。
                        # ここは必ず torch.distributed 側の barrier を使うこと。
                        dist.barrier()
                        torch.cuda.profiler.start()

                    # debug step counter
                    from common import debug_params as dbg
                    dbg.set_step(step + epoch * total_steps)

                    # 注: ここで optimizer.zero_grad は呼ばない。run_zero.py の optimizer
                    # (DeepSpeedCPUAdam) は param_groups が空にされる設定キャリアで no-op。
                    # 実際の勾配クリアは wrapper 内 _take_model_step(_dpu) が呼ぶ
                    # ZeroOptimizer3.zero_grad (fp16 param の .grad=None 化) が行う。
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
                            _print_all_comm_stats()
                            sys.stdout.flush()
                        dist.barrier()  # キャプチャ窓の終端も揃える (start 側と対)
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

            if lr_scheduler is not None:
                lr_scheduler.step()

            # イテレーションベースの計測: エポックループもbreak
            if total_target_iters is not None and global_iter >= total_target_iters:
                break

        if rank == 0:
            stats.print_all()
            _print_all_comm_stats()

            try:
                print(memory_usage_rank(model, optimizer=optimizer))
            except Exception:
                pass
            print("Loss history:", loss_history)
            print("Accuracy history:", accuracy_history)

        _dump_path = os.environ.get("DUMP_FINAL_PARAMS")
        if _dump_path:
            dump_final_params(model, rank, _dump_path)

    finally:
        if profiler and hasattr(profiler, "stop"):
            profiler.stop()
        # 初期化途中 (例: model.to(device) の CUDA OOM) で抜けると zero_model は未生成。
        # ここで例外を投げると元の例外が握り潰され、真因が追えなくなる。
        if zero_model is not None:
            try:
                zero_model.stop_adam_process()
            except Exception as _e:
                print(f"[warn] stop_adam_process failed: {_e}", flush=True)
        cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 autocast (default: fp32)")
    parser.add_argument("--ema", action="store_true",
                        help="PyTorch 標準 ExponentialLR による学習率スケジュールを有効化 "
                             "(長期学習向け既定: gamma=0.95)")
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
                             "(画像=分類精度 / テキスト=トークン精度)。既定は無効。")
    parser.add_argument("--lr-decay", type=float, default=1.0,
                        help="ExponentialLR の gamma を明示指定 (指定するとスケジュール有効化。"
                             "--ema 使用時の既定は 0.95)。")
    parser.add_argument("--no-offload", action="store_true",
                        help="CPU オフロードを無効化し、純粋な ZeRO-3 (全 GPU 常駐, "
                             "in-process GPU Adam) で学習する。DPU は使えない。")
    args = parser.parse_args()

    if args.debug_params:
        from common import debug_params as dbg
        dbg.enable()

    run_zero(
        use_profiler=args.profiler,
        use_bf16=args.bf16,
        use_ema=args.ema,
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
        lr_decay=args.lr_decay,
        no_offload=args.no_offload,
    )


if __name__ == "__main__":
    main()