import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import SynchronizedWallClockTimer

import torch
import torch.distributed as dist
from torch.nn.modules import Module
import logging
from torch._utils import _unflatten_dense_tensors
from torch._utils import _flatten_dense_tensors
from zero_optimizer import ZeroOptimizer3
from partition_parameters import ZeroParamStatus
from stage3_utils import *

MEMORY_OPT_ALLREDUCE_SIZE = 500000000

FORWARD_MICRO_TIMER = 'forward_microstep'
FORWARD_GLOBAL_TIMER = 'forward'
BACKWARD_MICRO_TIMER = 'backward_microstep'
BACKWARD_GLOBAL_TIMER = 'backward'
BACKWARD_INNER_MICRO_TIMER = 'backward_inner_microstep'
BACKWARD_INNER_GLOBAL_TIMER = 'backward_inner'
BACKWARD_REDUCE_MICRO_TIMER = 'backward_allreduce_microstep'
BACKWARD_REDUCE_GLOBAL_TIMER = 'backward_allreduce'
STEP_MICRO_TIMER = 'step_microstep'
STEP_GLOBAL_TIMER = 'step'


class EngineTimers(object):
    r"""Wallclock timers for DeepSpeedEngine"""
    def __init__(self, enable_micro_timers, enable_global_timers):
        self.forward_timers = []
        self.backward_timers = []
        self.backward_inner_timers = []
        self.backward_reduce_timers = []
        self.step_timers = []
        self.global_timers = []
        self.micro_timers = []

        if enable_micro_timers:
            self.forward_timers += [FORWARD_MICRO_TIMER]
            self.backward_timers += [BACKWARD_MICRO_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_MICRO_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_MICRO_TIMER]
            self.step_timers += [STEP_MICRO_TIMER]
            self.micro_timers += [
                FORWARD_MICRO_TIMER,
                BACKWARD_MICRO_TIMER,
                BACKWARD_INNER_MICRO_TIMER,
                BACKWARD_REDUCE_MICRO_TIMER,
                STEP_MICRO_TIMER
            ]

        if enable_global_timers:
            self.forward_timers += [FORWARD_GLOBAL_TIMER]
            self.backward_timers += [BACKWARD_GLOBAL_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_GLOBAL_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_GLOBAL_TIMER]
            self.step_timers += [STEP_GLOBAL_TIMER]
            self.global_timers += [
                FORWARD_GLOBAL_TIMER,
                BACKWARD_GLOBAL_TIMER,
                BACKWARD_INNER_GLOBAL_TIMER,
                BACKWARD_REDUCE_GLOBAL_TIMER,
                STEP_GLOBAL_TIMER
            ]

def instrument_w_nvtx(func):
    """decorator that causes an NVTX range to be recorded for the duration of the
    function call."""
    if hasattr(torch.cuda.nvtx, "range"):

        def wrapped_fn(*args, **kwargs):
            with torch.cuda.nvtx.range(func.__qualname__):
                return func(*args, **kwargs)

        return wrapped_fn
    else:
        return func

def split_half_float_double_sparse(tensors):
    supported_types = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.DoubleTensor",
        "torch.cuda.BFloat16Tensor",
    ]

    for t in tensors:
        assert t.type() in supported_types, f"attempting to reduce an unsupported grad type: {t.type()}"

    buckets = []
    for i, dtype in enumerate(supported_types):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append((dtype, bucket))
    return buckets


module_names = {}
param_names = {}

def debug_extract_module_and_param_names(model):
    # extract the fully qualified names as soon as the model is acquired
    global module_names
    global param_names
    module_names = {module: name for name, module in model.named_modules()}
    param_names = {param: name for name, param in model.named_parameters()}


#ZeRO Stage3
class ZeroWrapperExample(Module):
    def __init__(self, model, optimizer, model_parameters, dist_init_required=None,
                 reduce_bucket_size=int(1e8), prefetch_bucket_size=int(1e8),
                 max_reuse_distance=0, max_live_parameters=int(1.5e8),
                 offload=True):
        super().__init__()
        self.client_optimizer = optimizer
        self.client_model_parameter = model_parameters
        self._offload = offload  # False: 純粋な ZeRO-3 (GPU 常駐, in-process Adam)
        self._reduce_bucket_size = int(reduce_bucket_size)
        self._prefetch_bucket_size = int(prefetch_bucket_size)
        self._max_reuse_distance = int(max_reuse_distance)
        self._max_live_parameters = int(max_live_parameters)
        self.data_parallel_group = None
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_steps = 0
        self.dpu_steps = 0 ##DPU上で必要になるパラメータ, DPUでは各エポックで最初のパラメータ勾配を捨てる
        self.global_samples = 0
        self.micro_steps = 0
        self.skipped_steps = 0
        self.gas_boundary_ctr = 0
        self.gradient_average = True
        self.enable_backward_allreduce = True
        self.dist_backend = "nccl"

        self._is_gradient_accumulation_boundary = None
        
        # for debug purposes - can then debug print: debug_get_module_name(module)
        debug_extract_module_and_param_names(model)
        
        # needed for zero_to_fp32 weights reconstruction to remap nameless data to state_dict
        self.param_names = {param: name for name, param in model.named_parameters()}
        
        if dist_init_required is None:
            dist_init_required = not dist.is_initialized()

            if dist_init_required is False:
                assert (
                    dist.is_initialized() is True
                ), "Torch distributed not initialized. Please set dist_init_required to True or initialize before calling deepspeed.initialize()"
            else:
                if not dist.is_initialized():
                    dist.init_process_group(backend=self.dist_backend)
                    
        self._init_distributed(dist_init_required) #GPUのセット、rankやworld_sizeの設定 自分で足したよ, 本家にはないよ
                    
        self._configure_distributed_model(model)
        self._get_model_parameters()
                    
        self.optimizer = None
        self.basic_optimizer = None
        
        self.is_boundary = False #DPUを行ったかどうか
        
        if model_parameters or optimizer:
            self._configure_optimizer(optimizer, model_parameters)
        #optimizerは必ず引数として与えるようにする!!!

        self._get_model_parameters()
        
    def memory_breakdown(self):
        return True
        
    def wall_clock_breakdown(self):
        return True
        
    def zero_optimization(self):
        return True #今回はZeRO Stage3を必ず利用するためTrueを返すようにしている
        
    def _init_distributed(self, dist_init_required: bool):
        if self.local_rank >= 0:
            # GPU をこのプロセスが担当するデバイスに固定
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)

            # 必要ならここでプロセスグループ初期化（任意）
            if dist_init_required and not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            # 分散情報
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
            self.global_rank = dist.get_rank() if dist.is_initialized() else 0

            logging.info(f"Set device to local rank {self.local_rank} within node.")

            # ★ データ並列グループとサイズ（PyTorch 既定の WORLD を採用）
            #    独自のサブグループを作らない限り、WORLD を “DP グループ” として扱って問題ありません。
            self.data_parallel_group = dist.group.WORLD if dist.is_initialized() else None
            self.dp_world_size = self.world_size

        else:
            # 単機（非分散）
            self.world_size = 1
            self.global_rank = 0
            self.device = torch.device("cuda")
            # ★ 非分散時のフォールバック
            self.data_parallel_group = None
            self.dp_world_size = 1
    
    #ユーザーが渡した model を self.module として管理下に置く（以降、このラッパ参照を通じて処理）
    def _set_client_model(self, model):
        # register client model in _modules so that nn.module methods work correctly
        modules = self.__dict__.get('_modules')
        modules['module'] = model # (1) nn.Module のサブモジュール登録
        self.__dict__['module'] = model # (2) 属性アクセス用に __dict__ にも入れる, ここでself.moduleがmodelを参照するようにする
        
    def _broadcast_model(self):
        def is_replicated(p):
            if hasattr(p, "ds_status") and p.ds_status is not ZeroParamStatus.AVAILABLE:
                return False
            return True
        
        for p in self.module.parameters():
            #MOEは利用しない
            src_rank = 0 #コピー元のパラメータはすべてrank0から取ることにする
            if torch.is_tensor(p) and is_replicated(p):
                with torch.no_grad():
                    dist.broadcast(p, src_rank, group=self.data_parallel_group)
    
    def _configure_distributed_model(self, model):
        self._set_client_model(model)
        #model parameterのtype調整はDeepSpeed外で行う, ただし、ZeRO Optimizerのcommunication typeに注意!!!
        self.module.to(self.device) #ここもOffloadを実装するときには変えてね
        #ampの確認が必要だが、bf16では基本的に利用しないので本家とは違いチェックはしない
        self._broadcast_model()
        
    def _get_model_parameters(self):
        #auto tuining用の関数、DeepSpeed の autotuning は、与えたモデル／クラスタ条件で実行が安定して速くなる構成を自動探索するための仕組み
        #今回の実装では利用しないのでそのままpassにしている
        pass
    
    #Optimizer に渡した param_groups の中に、同じ Parameter が重複登録されていないかを検査して止める関数
    def _check_for_duplicates(self, optimizer):
        for name, param in self.module.named_parameters():
            param_id = id(param) #id(param) は Python 組み込み関数 id() で、そのオブジェクト（ここでは param）の 同一性（identity）を表す整数を返します
            
            def ids_list(group):
                return [id(param) for param in group]
            
            occurrence = sum([
                ids_list(group['params']).count(param_id)
                if param_id in ids_list(group['params']) else 0
                for group in optimizer.param_groups
            ])
            assert occurrence <= 1, f"Parameter with name: {name} occurs multiple times in optimizer.param_groups. Make sure it only appears once to prevent undefined behaviour."
    
    def _configure_zero_optimizer(self, optimizer):
        timers = None
        self.contiguous_gradients = True
        self.reduce_bucket_size: int = self._reduce_bucket_size
        self.prefetch_bucket_size: int = self._prefetch_bucket_size
        self.max_reuse_distance: int = self._max_reuse_distance
        self.max_live_parameters: int = self._max_live_parameters
        self.reduce_scatter: bool = True
        self.overlap_comm: bool = True #ここは調整が用必要らしい
        self.sub_group_size: int = int(1e7)
        self.gradient_accumulation_steps: int = 1 #勾配蓄積(デフォルトはなしにしておく)
        self.communication_data_type = torch.float16 #現在の実装ではbf16を利用することを想定
        
        optimizer = ZeroOptimizer3(
            self.module,
            optimizer,
            timers=timers,
            contiguous_gradients=self.contiguous_gradients,
            reduce_bucket_size=self.reduce_bucket_size,
            prefetch_bucket_size=self.prefetch_bucket_size,
            max_reuse_distance=self.max_reuse_distance,
            max_live_parameters=self.max_live_parameters,
            dp_process_group=self.data_parallel_group,
            reduce_scatter=self.reduce_scatter,
            overlap_comm=self.overlap_comm,
            sub_group_size=self.sub_group_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            communication_data_type=self.communication_data_type,
            offload=self._offload
        )
        return optimizer
    
    def _configure_optimizer(self, client_optimizer, model_parameters):
        #optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)として渡されたものをclient_optimizerとして利用することを想定
        client_optimizer.param_groups[:] = [
                    pg for pg in client_optimizer.param_groups if len(pg["params"]) != 0
                ] #"params": [] なグループを掃除して後段の処理を安定化。
        basic_optimizer = client_optimizer
        self._check_for_duplicates(basic_optimizer)

        self.basic_optimizer = basic_optimizer
        
        if self.zero_optimization():
            self.optimizer = self._configure_zero_optimizer(basic_optimizer)


    def train(self):
        self.module.train()
    
    def eval(self):
        self.module.train(False)
        
    @instrument_w_nvtx
    def forward(self, *inputs, **kwargs):
        for module in self.module.modules():
            module._parameters._in_forward = True
            pass

        loss = self.module(*inputs, **kwargs)

        for module in self.module.modules():
            module._parameters._in_forward = False

        # Profile-only (NSYS_SYNC_RANGES=1): forward NVTX range を全カーネル完了まで開けておく
        if os.environ.get("NSYS_SYNC_RANGES", "0") == "1":
            torch.cuda.synchronize()

        return loss
    
    
    def _scale_loss_by_gas(self, prescaled_loss):
        if isinstance(prescaled_loss, torch.Tensor):
            scaled_loss = prescaled_loss / self.gradient_accumulation_steps
        elif isinstance(prescaled_loss, tuple) or isinstance(prescaled_loss, list):
            scaled_loss = []
            for l in prescaled_loss:
                if isinstance(l, torch.Tensor):
                    scaled_loss.append(l / self.gradient_accumulation_steps)
                else:
                    scaled_loss.append(l)
        else:
            scaled_loss = prescaled_loss

        return scaled_loss
    
    def is_gradient_accumulation_boundary(self):
        if self._is_gradient_accumulation_boundary is None:
            return (self.micro_steps + 1) % \
                self.gradient_accumulation_steps == 0
        else:
            return self._is_gradient_accumulation_boundary
    
    @instrument_w_nvtx
    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
        # ZeRO stage 2 communicates during non gradient accumulation boundaries as well
        self.optimizer.overlapping_partition_gradients_reduce_epilogue()
            
    @instrument_w_nvtx
    def backward(self,
                 loss, 
                 allreduce_gradients=True,
                 release_loss=False,
                 retain_graph=False,
                 scale_wrt_gas=True):
        #ZeRO Stage3 では backward 中の reduce は計算グラフ上のフックで発火する

        # scale loss w.r.t. gradient accumulation if needed
        if self.gradient_accumulation_steps > 1 and scale_wrt_gas:
            loss = self._scale_loss_by_gas(loss.float())

        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
        self.optimizer.backward(loss, retain_graph=retain_graph)

        if allreduce_gradients:
            # Traditional code path that allreduces the module parameter grads
            self.allreduce_gradients()

        # Profile-only (NSYS_SYNC_RANGES=1): backward NVTX range を全カーネル完了まで開けておく
        if os.environ.get("NSYS_SYNC_RANGES", "0") == "1":
            torch.cuda.synchronize()

        return loss
    
    
    def _take_model_step(self, lr_kwargs, block_eigenvalue={}):
        self.optimizer.step()
        self.optimizer.zero_grad() #パラメータ更新用の処理をしてからgradをNoneにしているので関係ない
        self.global_steps += 1
    
    def step(self, lr_kwargs=None):
        if self.is_gradient_accumulation_boundary():
            self.gas_boundary_ctr += 1
            self._take_model_step(lr_kwargs)

        self.micro_steps += 1
        
    def _take_model_step_dpu(self, lr_kwargs, block_eigenvalue={}):
        self.optimizer.step_dpu(first_step = (self.dpu_steps == 0)) #DPUで初めのステップはパラメータ更新のための勾配がないのでパラメータ更新はせずに捨てる
        self.optimizer.zero_grad() #パラメータ更新用の処理をしてからgradをNoneにしているので関係ない
        self.global_steps += 1
        self.dpu_steps += 1
            
    def step_dpu(self, lr_kwargs=None):
        self.is_boundary = False
        if self.is_gradient_accumulation_boundary():
            self.gas_boundary_ctr += 1
            self.is_boundary = True
            self._take_model_step_dpu(lr_kwargs)

        self.micro_steps += 1
        return self.is_boundary

    def start_adam_process(self, cpu_affinity=None):
        self.optimizer.start_adam_process(cpu_affinity)

    def stop_adam_process(self):
        self.optimizer.stop_adam_process()

    def _start_timers(self, timer_names):
        pass

    def _stop_timers(self, timer_names):
        pass