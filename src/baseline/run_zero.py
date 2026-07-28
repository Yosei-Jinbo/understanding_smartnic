# run_zero.py
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import contextlib
from datetime import timedelta
import threading
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import nvtx as pnvtx

from common.dataset import get_datasets
from common.model import get_benchmark_model
from common.utils import (
    ThroughputMeter,
    memory_usage_rank,
    start_profiler,
    evaluate_zero3,
    SynchronizedWallClockTimer,
)
from zero_wrapper_example import ZeroWrapperExample
from deepspeed.ops.adam import DeepSpeedCPUAdam
from mpi4py import MPI
import glob
import re
from common import submodule_timing as smt
from transformers import DataCollatorWithPadding
from common.text_dataset import get_text_datasets

# ------------------------------------------------------------
# ロギング
# ------------------------------------------------------------
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---- ステップ内訳 NVTX (STEP_NVTX=1 で有効) ----
# compute_idle_decomp の "Other"(=host_overhead) に丸め込まれている区間を細分するための計装。
# 査読 C3「Fig.7 の Other を SM 競合緩和とホスト/起動オーバーヘッドに分離できるか」に対応。
# xfer: 系 (XFER_NVTX) とは独立にゲートする。
_STEP_NVTX = os.environ.get("STEP_NVTX", "0") == "1"


def _step_push(label: str) -> None:
    if _STEP_NVTX:
        torch.cuda.nvtx.range_push(label)


def _step_pop() -> None:
    if _STEP_NVTX:
        torch.cuda.nvtx.range_pop()


def print_rank_0(message: str):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)


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


def lower_thread_priority(nice=10):
    try:
        tid = threading.get_native_id()
        os.setpriority(os.PRIO_PROCESS, tid, nice)
    except Exception:
        pass

def _read_first_int(path: str, default: int = -1) -> int:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return default


def _cuda_pci_bus_id_hex(local_rank: int) -> str:
    """
    PyTorch の device properties から PCI bus id を得る。
    返り値例: '0000:65:00.0'
    """
    try:
        prop = torch.cuda.get_device_properties(local_rank)
        # PyTorch によっては pci_bus_id がある
        if hasattr(prop, "pci_bus_id"):
            return str(prop.pci_bus_id)
        # ない場合は NVML 等が必要になるが、ここではフォールバック
    except Exception:
        pass
    return ""


def _pci_sysfs_path_from_bus_id(bus_id: str) -> str:
    """
    /sys/bus/pci/devices/<bus_id>
    """
    if not bus_id:
        return ""
    p = f"/sys/bus/pci/devices/{bus_id}"
    return p if os.path.exists(p) else ""


def get_gpu_numa_node(local_rank: int) -> int:
    """
    GPU の PCI デバイスが属する NUMA node を sysfs から推定。
    失敗時は -1 を返す。
    """
    bus_id = _cuda_pci_bus_id_hex(local_rank)
    sysfs = _pci_sysfs_path_from_bus_id(bus_id)
    if not sysfs:
        return -1
    return _read_first_int(os.path.join(sysfs, "numa_node"), default=-1)


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

def set_affinity_for_tid(tid: int, cpu_ids):
    try:
        os.sched_setaffinity(tid, set(cpu_ids))
        return True
    except Exception as e:
        print_rank_0(f"[AFFINITY] failed tid={tid} cpus={cpu_ids}: {e}")
        return False


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

    注意: 同一busに複数デバイスがある環境では曖昧になり得る。
          その場合は NVML 経由を推奨。
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

def summarize_by_submodule():
    stats = smt.snapshot(flush_cuda=True)

    phases = (
        "fwd_fetch", "fwd_wait_stall", "fwd_exec",
        "bwd_fetch", "bwd_wait_stall", "bwd_exec",
    )

    # module_id を集める
    module_ids = {mid for (_, mid) in stats.keys()}

    out = {}

    def _stat_to_dict_ms(st):
        s2ms = 1000.0

        def q(p):
            return (st.quantile_s(p) * s2ms) if hasattr(st, "quantile_s") else 0.0

        # submodule_timing.Stat（warmup拡張版）に対応
        return {
            "seen": getattr(st, "seen", st.count),     # warmup含む観測回数
            "warmup_n": getattr(st, "warmup_n", 0),
            "count": st.count,                         # warmup除外後のサンプル数

            # ---- time: all in ms ----
            "total_ms": st.total_s * s2ms,
            "avg_ms": st.avg_s * s2ms,
            "std_ms": getattr(st, "std_s", 0.0) * s2ms,
            "cv": getattr(st, "cv", 0.0),              # 変動係数は無次元なのでそのまま

            "min_ms": (st.min_s * s2ms if st.count else 0.0),
            "p50_ms": q(0.50),
            "p90_ms": q(0.90),
            "p99_ms": q(0.99),
            "max_ms": (st.max_s * s2ms if st.count else 0.0),
        }

    for mid in sorted(module_ids):
        info = smt.get_module_info(mid) or {}
        name = info.get("name") or ""
        cls = info.get("class") or ""
        mkey = f"id={mid} name='{name}' class='{cls}'"

        out[mkey] = {ph: None for ph in phases}

        for ph in phases:
            st = stats.get((ph, mid))
            if st is None:
                continue
            out[mkey][ph] = _stat_to_dict_ms(st)

    return out

# ------------------------------------------------------------
# 学習本体
# ------------------------------------------------------------
def run_zero(use_profiler=False, use_bf16=False, use_ema=False,
             model_name="vit_l_16", dataset_name="cifar10", batch_size=64,
             warmup_iters=None, measure_iters=None, seq_len=1024, num_epochs=None,
             reduce_bucket_size=int(1e8), prefetch_bucket_size=int(1e8),
             max_reuse_distance=0, max_live_parameters=int(1.5e8)):
    profiler = None
    zero_model = None
    try:
        local_rank, rank, world_size = setup_from_env()
        comm = MPI.COMM_WORLD
        warmup_cpuadam_once(comm, local_rank)

        device = torch.device(f"cuda:{local_rank}")

        # ------------------------
        # config
        # ------------------------
        num_workers = 0
        #batch_size = 4
        #num_workers = 0
        #model_name = "tinynn"
        #num_classes = 4
        #input_size = 4
        #num_train_samples = 16
        #num_test_samples = 16
        #epochs = 2

        # Causal LM モデル判定
        CAUSAL_LM_MODELS = {"opt-1.3b", "opt_1.3b", "llama-3b", "llama_3b", "llama-7b", "llama_7b", "llama-2-7b"}
        MLM_MODELS = {"deberta-xl", "deberta_xl"}
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

        DPU_THRESHOLD = -1
        global_iter = 0

        print_rank_0(
            f"[CONFIG] model={model_name} dataset={dataset_name} batch_size={batch_size} "
            f"epochs={epochs} warmup_iters={warmup_iters} measure_iters={measure_iters} "
            f"is_causal_lm={is_causal_lm}"
        )

        # ----------------------------------------------------------
        # rank -> GPU -> NUMA node binding (cpu set 分割)
        # ----------------------------------------------------------
        gpu_node = get_gpu_numa_node(local_rank)
        allowed_cpus = get_allowed_cpus()
        CPU_THREAD_CORES = int(os.environ.get("CPU_THREAD_CORES", "2"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE",
                               os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", "1")))
        main_cpu_set, cpu_thread_set = choose_rank_cpu_sets(
            rank=rank, world_size=world_size, allowed_cpus=allowed_cpus,
            cpu_thread_cores=CPU_THREAD_CORES,
            gpu_numa_node=gpu_node, local_rank=local_rank,
            local_world_size=local_world_size,
        )
        try:
            os.sched_setaffinity(0, set(main_cpu_set))
        except Exception as e:
            print_rank_0(f"[AFFINITY] process setaffinity failed: {e}")

        # DataLoader worker 数を main_cpu_set に合わせて制限
        # (main_cpu_set が 3 コアなら、worker=2 程度が妥当)
        #num_workers = max(0, min(num_workers, max(0, len(main_cpu_set) - 1)))
        print_rank_0(
            f"[BIND] rank={rank} local_rank(gpu)={local_rank} gpu_numa_node={gpu_node} "
            f"main_cpu_set={main_cpu_set} cpu_thread_set={cpu_thread_set} num_workers={num_workers}"
        )

        # ------------------------
        # dataset
        # ------------------------
        if model_name.lower() == "tinynn":
            # TinyNN用ダミーデータ (input=4, num_classes=4)
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

        # ------------------------
        # model
        # ------------------------
        model = get_benchmark_model(model_name)
        model.half()  # FP16に変換してからGPUに転送（大規模モデルのメモリ節約）
        model = model.to(device)
        model_parameters = model.parameters()
        #通常だとmodelがbfloat16で登録されるとAdamもbf16で登録される。
        #しかし、ZeROの実装ではoptimizerをfp32で登録しなおすのでこのままの実装でおけ
        optimizer = DeepSpeedCPUAdam(model_parameters, lr=3e-4, eps=1e-5)
        loss_fn = nn.CrossEntropyLoss()
        
        # -------------------------
        # デバッグ用にモデルを出力
        # -------------------------
        #def is_rank0() -> bool:
        #    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

        #@torch.no_grad()
        #def print_all_params_rank0(model: torch.nn.Module):
        #    if not is_rank0():
        #        return
        #    for name, p in model.named_parameters():
        #        t = p.detach()
        #        print(f"=== {name} | shape={tuple(t.shape)} dtype={t.dtype} device={t.device} ===")
        #        print(t)  # ← ここで全要素をそのまま出力
        #        print()
                
        #print_all_params_rank0(model)

        # ユーザ実装の ZeRO ラッパ（Stage 3 を想定）
        zero_model = ZeroWrapperExample(
            model, optimizer=optimizer, model_parameters=model_parameters,
            reduce_bucket_size=reduce_bucket_size,
            prefetch_bucket_size=prefetch_bucket_size,
            max_reuse_distance=max_reuse_distance,
            max_live_parameters=max_live_parameters,
        )

        # CPU Adam を別プロセスで起動
        zero_model.start_adam_process(cpu_affinity=cpu_thread_set)

        timers = SynchronizedWallClockTimer()
        timers("opt_dpu")

        throughput_every = 10000
        timer_every = 10000
        since_last_timer = 0

        try:
            throughput_meter = ThroughputMeter(
                warmup_steps=5,
                steps_per_output=throughput_every,
                device=device,
                model=model,
            )
        except TypeError:
            throughput_meter = ThroughputMeter(
                warmup_steps=5,
                steps_per_output=throughput_every,
            )
        throughput_meter.start()

        if use_profiler:
            log_dir = f"./profiler_log/rank_{rank}"
            os.makedirs(log_dir, exist_ok=True)
            profiler = start_profiler(log_dir=log_dir, use_cuda=True)

        loss_history, accuracy_history = [], []
        
        # ------------------------------------------------------------
        # Communication/Transfer stats (epoch delta, no reset)
        # ------------------------------------------------------------
        import zero_optimizer as _zo
        import partition_parameters as _pp

        def _get_float(fn, default=0.0):
            try:
                return float(fn())
            except Exception:
                return float(default)

        # prev cumulative (process lifetime)
        prev_rs_ms = _get_float(_zo.get_reduce_scatter_time_ms, 0.0)
        prev_rs_calls = _get_float(_zo.get_reduce_scatter_calls, 0.0)
        prev_ag_calls = _get_float(_pp.get_all_gather_calls, 0.0)
        prev_fp_calls = _get_float(_pp.get_full_parameter_calls, 0.0)

        get_ag_dpu_ms = getattr(_pp, "get_all_gather_dpu_ms", None)
        get_ag_block_ms = getattr(_pp, "get_all_gather_block_ms", None)
        get_ag_bytes = getattr(_pp, "get_all_gather_bytes", None)
        get_fp_copy_ms = getattr(_pp, "get_full_parameter_copy_ms", None)
        get_fp_bytes = getattr(_pp, "get_full_parameter_bytes", None)
        prev_ag_dpu_ms = _get_float(get_ag_dpu_ms, 0.0) if get_ag_dpu_ms else 0.0
        prev_ag_block_ms = _get_float(get_ag_block_ms, 0.0) if get_ag_block_ms else 0.0
        prev_ag_bytes = 0
        prev_fp_bytes = 0
        prev_fp_copy_ms = 0.0

        get_grad_d2h_ms = getattr(_zo, "get_grad_offload_d2h_time_ms", None)
        get_grad_d2h_calls = getattr(_zo, "get_grad_offload_d2h_calls", None)
        get_param_h2d_ms = getattr(_pp, "get_param_shard_h2d_time_ms", None)
        get_param_h2d_calls = getattr(_pp, "get_param_shard_h2d_calls", None)

        prev_grad_d2h_ms = _get_float(get_grad_d2h_ms, 0.0) if get_grad_d2h_ms else 0.0
        prev_grad_d2h_calls = _get_float(get_grad_d2h_calls, 0.0) if get_grad_d2h_calls else 0.0
        prev_param_h2d_ms = _get_float(get_param_h2d_ms, 0.0) if get_param_h2d_ms else 0.0
        prev_param_h2d_calls = _get_float(get_param_h2d_calls, 0.0) if get_param_h2d_calls else 0.0
        
        # ステップ単位の時間を記録
        step_times = {}       # key -> list of float (seconds), per-epoch
        all_step_times = {}   # key -> list of float (seconds), across all epochs
        global_step_count = 0  # 全エポック通算のステップカウンタ（ウォームアップ判定用）

        def _record_step_time(key, elapsed_sec):
            # ウォームアップ中（global_step_count < WARMUP_STEPS）は記録しない
            if global_step_count < WARMUP_STEPS:
                return
            if key not in step_times:
                step_times[key] = []
            step_times[key].append(elapsed_sec)

        def _format_stats(label, vals_sec, warmup=0):
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

        def _print_step_stats(epoch_num):
            """エポック単位の統計情報を出力する（ウォームアップはグローバルで1回のみ除外済み）。"""
            print(f"========== Per-Step Timing Statistics (Epoch {epoch_num}) ==========")
            for key in ["fwd", "bwd", "opt", "opt_dpu", "step_total"]:
                vals = step_times.get(key)
                if not vals:
                    continue
                line = _format_stats(f"[E{epoch_num}] {key}", vals, warmup=0)
                if line:
                    print(line)
            print("=" * 80)

        def _print_all_epochs_stats():
            """全エポック通算の統計情報を出力する。"""
            print(f"========== Per-Step Timing Statistics (All Epochs, global warmup={WARMUP_STEPS} excluded) ==========")
            for key in ["fwd", "bwd", "opt", "opt_dpu", "step_total"]:
                vals = all_step_times.get(key)
                if not vals:
                    continue
                line = _format_stats(f"[All] {key}", vals, warmup=0)
                if line:
                    print(line)
            print("=" * 80)

        def _print_all_comm_stats():
            """全エポック通算の通信統計を出力する。"""
            try:
                tot_rs_ms = _get_float(_zo.get_reduce_scatter_time_ms, 0.0)
                tot_rs_calls = int(_get_float(_zo.get_reduce_scatter_calls, 0.0))
                tot_ag_calls = int(_get_float(_pp.get_all_gather_calls, 0.0))
                tot_fp_calls = int(_get_float(_pp.get_full_parameter_calls, 0.0))
                tot_ag_dpu_ms = _get_float(get_ag_dpu_ms, 0.0) if get_ag_dpu_ms else 0.0
                tot_ag_block_ms = _get_float(get_ag_block_ms, 0.0) if get_ag_block_ms else 0.0
                tot_fp_copy_ms = _get_float(get_fp_copy_ms, 0.0) if get_fp_copy_ms else 0.0
                tot_grad_d2h_ms = _get_float(get_grad_d2h_ms, 0.0) if get_grad_d2h_ms else 0.0
                tot_grad_d2h_calls = int(_get_float(get_grad_d2h_calls, 0.0)) if get_grad_d2h_calls else 0
                tot_param_h2d_ms = _get_float(get_param_h2d_ms, 0.0) if get_param_h2d_ms else 0.0
                tot_param_h2d_calls = int(_get_float(get_param_h2d_calls, 0.0)) if get_param_h2d_calls else 0
                _get_ag_bytes = getattr(_pp, "get_all_gather_bytes", None)
                _get_fp_bytes = getattr(_pp, "get_full_parameter_bytes", None)
                tot_ag_bytes = _get_float(_get_ag_bytes, 0) if _get_ag_bytes else 0
                tot_fp_bytes = _get_float(_get_fp_bytes, 0) if _get_fp_bytes else 0

                print("========== ZeRO-3 Communication / Transfer (All Epochs total) ==========")
                print(f"[All] reduce_scatter:    {tot_rs_ms/1000.0:.6f}s, calls={tot_rs_calls}")
                print(f"[All] all_gather (NCCL): calls={tot_ag_calls}, {tot_ag_bytes/1e9:.3f}GB")
                print(f"[All]   AG 通信処理:     {tot_ag_dpu_ms/1000.0:.6f}s")
                print(f"[All]   AG block:        {tot_ag_block_ms/1000.0:.6f}s")
                print(f"[All] full param (CPU→GPU): calls={tot_fp_calls}, {tot_fp_bytes/1e9:.3f}GB")
                print(f"[All]   FP コピー:       {tot_fp_copy_ms/1000.0:.6f}s")
                print(f"[All] grad D2H:          {tot_grad_d2h_ms/1000.0:.6f}s, calls={tot_grad_d2h_calls}")
                print(f"[All] param H2D:         {tot_param_h2d_ms/1000.0:.6f}s, calls={tot_param_h2d_calls}")
                print("=" * 80)
            except Exception:
                pass

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
            if _stall_tracker is not None:
                _stall_tracker.set_epoch(epoch)

            total_steps = len(train_loader)
            next_report = 0.1
            start_t = time.time()
            step_times.clear()

            ctx = profiler if profiler else contextlib.nullcontext()
            with ctx as prof:
                for step, batch in enumerate(train_loader, 1):
                    if is_causal_lm or is_mlm:
                        input_ids = batch["input_ids"].to(device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                        lm_labels = batch["labels"].to(device, non_blocking=True)
                        bsz = input_ids.size(0)
                    else:
                        images, labels = batch
                        images = images.to(device, non_blocking=True).half()
                        labels = labels.to(device, non_blocking=True)
                        bsz = labels.size(0)

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

                    if epoch <= DPU_THRESHOLD:
                        # ----------------------------------------------------------
                        # ★重要：通常側でも zero_grad を明示（ラッパ依存を排除）
                        # ----------------------------------------------------------
                        optimizer.zero_grad(set_to_none=True)

                        _pp.set_ag_phase("forward")
                        timers("fwd").start()
                        t0 = time.perf_counter()
                        with throughput_meter(batch_size=bsz):
                            if is_causal_lm or is_mlm:
                                output = zero_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels)
                                loss = output.loss.float()
                            else:
                                logits = zero_model.forward(images)
                                loss = loss_fn(logits.float(), labels)
                        timers("fwd").stop()
                        _record_step_time("fwd", time.perf_counter() - t0)

                        _pp.set_ag_phase("backward")
                        timers("bwd").start()
                        t0 = time.perf_counter()
                        zero_model.backward(loss)
                        timers("bwd").stop()
                        _record_step_time("bwd", time.perf_counter() - t0)
                        _pp.set_ag_phase("unknown")

                        timers("opt").start()
                        t0 = time.perf_counter()
                        _step_push("step:opt")
                        try:
                            zero_model.step()
                        finally:
                            _step_pop()
                        timers("opt").stop()
                        _record_step_time("opt", time.perf_counter() - t0)

                    # ----------------------
                    # DPU オフロード版 (CPU Adam は別プロセスで実行)
                    # ----------------------
                    else:
                        optimizer.zero_grad(set_to_none=True)

                        timers("opt_dpu").start()
                        t0 = time.perf_counter()
                        boundary_flag = zero_model.step_dpu()
                        timers("opt_dpu").stop()
                        _record_step_time("opt_dpu", time.perf_counter() - t0)

                        _pp.set_ag_phase("forward")
                        timers("fwd").start()
                        t0 = time.perf_counter()
                        with throughput_meter(batch_size=bsz):
                            if is_causal_lm or is_mlm:
                                output = zero_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels)
                                loss = output.loss.float()
                            else:
                                logits = zero_model.forward(images)
                                loss = loss_fn(logits.float(), labels)
                        timers("fwd").stop()
                        _record_step_time("fwd", time.perf_counter() - t0)

                        _pp.set_ag_phase("backward")
                        timers("bwd").start()
                        t0 = time.perf_counter()
                        zero_model.backward(loss)

                        if hasattr(zero_model, "optimizer") and hasattr(zero_model.optimizer, "_partition_all_parameters"):
                            zero_model.optimizer._partition_all_parameters()

                        timers("bwd").stop()
                        _record_step_time("bwd", time.perf_counter() - t0)
                        _pp.set_ag_phase("unknown")

                        if boundary_flag:
                            _step_push("step:update_new_params")
                            try:
                                zero_model.optimizer.update_new_params()
                            finally:
                                _step_pop()

                    _record_step_time("step_total", time.perf_counter() - step_t0)
                    global_step_count += 1

                    # イテレーションベースの計測制御
                    global_iter += 1
                    # nsys profile の終了: default では total target と同時だが、
                    # NSYS_PROFILE_MEASURE_ITERS で早めに切って末尾の hang を回避できる
                    if total_target_iters is not None and global_iter == _nsys_profile_end_iter:
                        torch.cuda.synchronize()
                        dist.barrier()  # キャプチャ窓の終端も揃える (start 側と対)
                        torch.cuda.profiler.stop()
                    if total_target_iters is not None and global_iter >= total_target_iters:
                        print_rank_0(f"Reached target iterations ({global_iter}/{total_target_iters}), stopping.")
                        break

                    if profiler:
                        prof.step()

                    if getattr(throughput_meter, "_warmup_done", False):
                        if since_last_timer % timer_every == 0:
                            timer_names = ["fwd", "bwd", "opt"] if epoch <= DPU_THRESHOLD else ["fwd", "bwd", "opt_dpu"]
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

            if rank == 0:
                print(f"[Epoch {epoch+1}/{epochs}] | train time: {time.time() - start_t:.2f}s")
                _print_step_stats(epoch + 1)

                # ウォームアップ除外後のデータを全エポック通算に蓄積
                for key, vals in step_times.items():
                    if key not in all_step_times:
                        all_step_times[key] = []
                    all_step_times[key].extend(vals)

            # ------------------------------------------------------------
            # Communication / Transfer time (epoch delta, no reset)
            # ------------------------------------------------------------
            if rank == 0:
                # ensure kernels done before reading stats
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

                cur_rs_ms = _get_float(_zo.get_reduce_scatter_time_ms, 0.0)
                cur_rs_calls = _get_float(_zo.get_reduce_scatter_calls, 0.0)
                cur_ag_calls = _get_float(_pp.get_all_gather_calls, 0.0)
                cur_fp_calls = _get_float(_pp.get_full_parameter_calls, 0.0)

                cur_grad_d2h_ms = _get_float(get_grad_d2h_ms, 0.0) if get_grad_d2h_ms else 0.0
                cur_grad_d2h_calls = _get_float(get_grad_d2h_calls, 0.0) if get_grad_d2h_calls else 0.0
                cur_param_h2d_ms = _get_float(get_param_h2d_ms, 0.0) if get_param_h2d_ms else 0.0
                cur_param_h2d_calls = _get_float(get_param_h2d_calls, 0.0) if get_param_h2d_calls else 0.0

                cur_ag_dpu_ms = _get_float(get_ag_dpu_ms, 0.0) if get_ag_dpu_ms else 0.0
                cur_ag_block_ms = _get_float(get_ag_block_ms, 0.0) if get_ag_block_ms else 0.0

                cur_fp_copy_ms = _get_float(get_fp_copy_ms, 0.0) if get_fp_copy_ms else 0.0
                cur_ag_bytes = _get_float(get_ag_bytes, 0) if get_ag_bytes else 0
                cur_fp_bytes = _get_float(get_fp_bytes, 0) if get_fp_bytes else 0

                # epoch deltas
                epoch_rs_ms = max(0.0, cur_rs_ms - prev_rs_ms)
                epoch_rs_calls = max(0.0, cur_rs_calls - prev_rs_calls)
                epoch_ag_calls = max(0.0, cur_ag_calls - prev_ag_calls)
                epoch_fp_calls = max(0.0, cur_fp_calls - prev_fp_calls)

                epoch_grad_d2h_ms = max(0.0, cur_grad_d2h_ms - prev_grad_d2h_ms)
                epoch_grad_d2h_calls = max(0.0, cur_grad_d2h_calls - prev_grad_d2h_calls)
                epoch_param_h2d_ms = max(0.0, cur_param_h2d_ms - prev_param_h2d_ms)
                epoch_param_h2d_calls = max(0.0, cur_param_h2d_calls - prev_param_h2d_calls)

                epoch_ag_dpu_ms = max(0.0, cur_ag_dpu_ms - prev_ag_dpu_ms)
                epoch_ag_block_ms = max(0.0, cur_ag_block_ms - prev_ag_block_ms)
                epoch_ag_bytes = max(0, cur_ag_bytes - prev_ag_bytes)
                epoch_fp_bytes = max(0, cur_fp_bytes - prev_fp_bytes)
                epoch_fp_copy_ms = max(0.0, cur_fp_copy_ms - prev_fp_copy_ms)

                print("========== ZeRO-3 Communication / Transfer (epoch delta) ==========")
                print(f"[Epoch {epoch+1}] reduce_scatter:    {epoch_rs_ms/1000.0:.6f}s, calls={int(epoch_rs_calls)}")
                print(f"[Epoch {epoch+1}] all_gather (NCCL): calls={int(epoch_ag_calls)}, {epoch_ag_bytes/1e9:.3f}GB")
                print(f"[Epoch {epoch+1}]   AG 通信処理:     {epoch_ag_dpu_ms/1000.0:.6f}s")
                print(f"[Epoch {epoch+1}]   AG block:        {epoch_ag_block_ms/1000.0:.6f}s")
                print(f"[Epoch {epoch+1}] full param (CPU→GPU): calls={int(epoch_fp_calls)}, {epoch_fp_bytes/1e9:.3f}GB")
                print(f"[Epoch {epoch+1}]   FP コピー:       {epoch_fp_copy_ms/1000.0:.6f}s")
                print(f"[Epoch {epoch+1}] grad D2H:          {epoch_grad_d2h_ms/1000.0:.6f}s, calls={int(epoch_grad_d2h_calls)}")
                print(f"[Epoch {epoch+1}] param H2D:         {epoch_param_h2d_ms/1000.0:.6f}s, calls={int(epoch_param_h2d_calls)}")
                print("===================================================================")

                # Per-AG detailed analysis
                try:
                    _pp.print_ag_analysis(epoch + 1)
                    _pp.reset_ag_records()
                except Exception:
                    pass

                # Forward/Backward AG block breakdown
                try:
                    ps = _pp.get_ag_phase_stats()
                    print(f"========== AG Block by Phase (Epoch {epoch+1}) ==========")
                    print(f"  Forward:  block={ps['fwd_block_ms']/1000:.3f}s  calls={ps['fwd_calls']}"
                          f"  avg={ps['fwd_block_ms']/max(1,ps['fwd_calls']):.3f}ms/call")
                    print(f"  Backward: block={ps['bwd_block_ms']/1000:.3f}s  calls={ps['bwd_calls']}"
                          f"  avg={ps['bwd_block_ms']/max(1,ps['bwd_calls']):.3f}ms/call")
                    print(f"{'=' * 55}")
                    _pp.reset_ag_phase_stats()
                except Exception:
                    pass

                # update prev for next epoch
                prev_rs_ms = cur_rs_ms
                prev_rs_calls, prev_ag_calls, prev_fp_calls = cur_rs_calls, cur_ag_calls, cur_fp_calls
                prev_grad_d2h_ms, prev_grad_d2h_calls = cur_grad_d2h_ms, cur_grad_d2h_calls
                prev_param_h2d_ms, prev_param_h2d_calls = cur_param_h2d_ms, cur_param_h2d_calls
                prev_ag_dpu_ms, prev_ag_block_ms = cur_ag_dpu_ms, cur_ag_block_ms
                prev_ag_bytes, prev_fp_bytes = cur_ag_bytes, cur_fp_bytes
                prev_fp_copy_ms = cur_fp_copy_ms

            # イテレーションベースの計測: エポックループもbreak
            if total_target_iters is not None and global_iter >= total_target_iters:
                break

        if rank == 0:
            _print_all_epochs_stats()
            _print_all_comm_stats()
            throughput_meter.summary()

            try:
                print(memory_usage_rank(model, optimizer=optimizer))
            except Exception:
                pass
            print("Loss history:", loss_history)
            print("Accuracy history:", accuracy_history)
            
            '''
            summary = summarize_by_submodule()

            # 見やすさのため、fwd_exec の total が大きい順に並べる（無い場合は 0）
            def _get_total(v, phase):
                d = v.get(phase)
                return float(d["total_ms"]) if d else 0.0

            items = list(summary.items())
            items.sort(key=lambda kv: _get_total(kv[1], "fwd_exec"), reverse=True)

            def _fmt_phase(ph_name, d):
                if d is None:
                    return f"  {ph_name:<8}: (no data)"
                return (
                    f"  {ph_name:<8}: "
                    f"seen={d['seen']:>4} wup={d['warmup_n']:>2} count={d['count']:>4} | "
                    f"total={d['total_ms']:.3f}ms mean={d['avg_ms']:.3f}ms "
                    f"std={d['std_ms']:.3f}ms cv={d['cv']:.3f} | "
                    f"min={d['min_ms']:.3f}ms p50={d['p50_ms']:.3f}ms "
                    f"p90={d['p90_ms']:.3f}ms p99={d['p99_ms']:.3f}ms max={d['max_ms']:.3f}ms"
                )

            for mkey, v in items:
                print("module:", mkey)
                print(_fmt_phase("fwd_fetch",      v.get("fwd_fetch")))
                print(_fmt_phase("fwd_wait_stall", v.get("fwd_wait_stall")))
                print(_fmt_phase("fwd_exec",       v.get("fwd_exec")))
                print(_fmt_phase("bwd_fetch",      v.get("bwd_fetch")))
                print(_fmt_phase("bwd_wait_stall", v.get("bwd_wait_stall")))
                print(_fmt_phase("bwd_exec",       v.get("bwd_exec")))

            # ---- pure_exec サマリ (子孫の wait_stall を差し引いた純粋カーネル時間) ----
            # 葉モジュール: pure_exec == exec (純粋カーネル時間そのもの)
            # 親モジュール: pure_exec == subtree 内の純粋カーネル時間総和
            try:
                smt.pretty_print_pure_exec(
                    exec_phase="bwd_exec",
                    stall_phase="bwd_wait_stall",
                    sort_by="exec_total_s",
                )
                smt.pretty_print_pure_exec(
                    exec_phase="fwd_exec",
                    stall_phase="fwd_wait_stall",
                    sort_by="exec_total_s",
                )
            except Exception as _e:
                print(f"[warn] pretty_print_pure_exec failed: {_e}")

            # ---- Phase 20: AG/RS stall event-bracket dump ----
            if _stall_tracker is not None:
                try:
                    label = os.environ.get("AG_STALL_LABEL", "stall_events")
                    out_path = os.environ.get(
                        "AG_STALL_DUMP_PATH",
                        f"logs/{label}_rank{rank}.json")
                    extra = {
                        "rank": rank,
                        "world_size": world_size,
                        "config": "cpu_buffering",
                        "model": model_name,
                        "warmup_iters": int(WARMUP_STEPS),
                        "total_target_iters": int(total_target_iters) if total_target_iters else None,
                        "disable_completion_poller": os.environ.get(
                            "DISABLE_COMPLETION_POLLER", "0"),
                    }
                    _stall_tracker.dump_json(out_path, extra=extra)
                    print(f"[Phase 20] AG/RS stall events dumped to {out_path}")
                    summ = _stall_tracker.summary()
                    if summ:
                        print("========== Phase 20: AG/RS stall summary (event-bracket) ==========")
                        for k in sorted(summ.keys()):
                            v = summ[k]
                            print(f"  {k:<24} count={int(v['count']):>5} "
                                  f"total={v['total_ms']:>9.1f}ms "
                                  f"mean={v['mean_ms']:>7.3f}ms "
                                  f"p50={v['p50_ms']:>7.3f}ms "
                                  f"p95={v['p95_ms']:>7.3f}ms "
                                  f"max={v['max_ms']:>7.3f}ms")
                        print("=" * 70)
                except Exception as _e:
                    print(f"[warn] Phase 20 stall dump failed: {_e}")
            '''

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
    parser.add_argument("--ema", action="store_true")
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
    )


if __name__ == "__main__":
    main()