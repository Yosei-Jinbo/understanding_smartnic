import sys
import os
import torch
from torch.cuda import Stream
from collections import OrderedDict
from partition_parameters import _init_external_params
from partition_parameters import *
from partitioned_param_coordinator import PartitionedParameterCoordinator, iter_params
from stage3_utils import * #parameterの持ち方が違うため独自のmemory_usage関数を呼ぶ
from common import submodule_timing as smt
import time

# H2D/D2H の転送時間は CUDA Event ベースの計測で取得する。



# ---- NVTX per-submodule ranges (USE_NVTX_RANGES=1 で有効) ----
# nsys profile + `nsys stats --report nvtxkernsum` で
# fwd:<name>, fwd_fetch:<name>, fwd_exec:<name>, bwd:<name>, bwd_fetch:<name>, bwd_exec:<name>
# のレンジ別に CUDA kernel 実時間を集計できる。
_NVTX_ENABLED = os.environ.get("USE_NVTX_RANGES", "0") == "1"

def _nvtx_name(sub_module) -> str:
    name = getattr(sub_module, "_timing_name", None)
    if name:
        return name
    return sub_module.__class__.__name__

def _nvtx_push(label: str) -> None:
    if _NVTX_ENABLED:
        torch.cuda.nvtx.range_push(label)

def _nvtx_pop() -> None:
    if _NVTX_ENABLED:
        torch.cuda.nvtx.range_pop()

FWD_MODULE_STACK = list()

def attach_module_names(model: torch.nn.Module) -> None:
    for name, m in model.named_modules():
        m._timing_name = name

def is_builtin_type(obj):
    # https://stackoverflow.com/a/17795199
    return obj.__class__.__module__ == '__builtin__' or obj.__class__.__module__ == "builtins"

#外部パラメータを参照できるようにparamのアクセス方法を変更
class ZeroOrderedDict(OrderedDict):
    def __init__(self, parent_module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent_module = parent_module
        self._in_forward = False
        
    def __getitem__(self, key):
        param = super().__getitem__(key)
        if param is None:
            return param
        
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if self._parent_module._parameters._in_forward:
                register_external_parameter(FWD_MODULE_STACK[-1], param)
                param.all_gather()
                print_rank_0(
                    f'Registering external parameter from getter {key} ds_id = {param.ds_id}',
                    force=False)
        return param
    
#module の _parameters 辞書を cls (ZeroOrderedDict) に置き換え、アクセスをフックする
def _inject_parameters(module, cls):
    for module in module.modules():
        if cls == ZeroOrderedDict:
            new_param = cls(parent_module=module)
        else:
            new_param = cls()
            
        for key, param in module._parameters.items():
            new_param[key] = param
        module._parameters = new_param

#apply torch.autograd.Function that calls a backward_function to tensors in output
def _apply_to_tensors_only(module, functional, backward_function, outputs):
    if isinstance(outputs, (tuple, list)):
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(module, functional, backward_function, output)
            touched_outputs.append(touched_output)
        return outputs.__class__(touched_outputs)
    elif isinstance(outputs, dict):
        for key in outputs.keys():
            outputs[key] = _apply_to_tensors_only(module, functional, backward_function, outputs[key])
        return outputs
    
    elif type(outputs) is torch.Tensor:
        return functional.apply(module, backward_function, outputs)
    else:
        if not is_builtin_type(outputs):
            logger.warning(
                f"A module has unknown inputs or outputs type ({type(outputs)}) and the tensors embedded in it cannot be detected. "
                "The ZeRO-3 hooks designed to trigger before or after backward pass of the module relies on knowing the input and "
                "output tensors and therefore may not get triggered properly.")
        return outputs
    
#for each tensor in outputs run the forward_function and register backward_function as hook
def _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function, output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        forward_function(outputs)
        if outputs.requires_grad:
            outputs.register_hook(backward_function)
        return outputs
    else:
        return outputs

#「そのモジュールの逆伝播が始まる直前に実行したい処理」を差し込む“入口フック”
class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        # 複数出力/複数回ラップ時の重複適用を数える参照カウンタ
        if not hasattr(module, "applied_pre_backward_ref_cnt"):
            module.applied_pre_backward_ref_cnt = 0
        module.applied_pre_backward_ref_cnt += 1
        outputs = outputs.detach()
        return outputs
    
    @staticmethod
    def backward(ctx, *args):
        ctx.pre_backward_function(ctx.module)
        # 非 Tensor 引数 (module, fn) は None、outputs への勾配はそのまま上流へ通す
        return (None, None) + args

#「そのモジュールの逆伝播がすべて終わった時点で"一度だけ"実行したい処理」を差し込む“出口フック”
class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        if output.requires_grad: #leafノードならpost backwardを行う (parameterのreleaseのためleafのノード以外でやると爆発)
            module.ds_grads_remaining += 1 #“このモジュール出力の勾配があと何回来るか”の残数
            ctx.pre_backward_function = pre_backward_function
        output = output.detach()
        return output
    
    @staticmethod
    def backward(ctx, *args):
        ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
        if ctx.module.ds_grads_remaining == 0:
            ctx.pre_backward_function(ctx.module)
        return (None, None) + args
    

#Stage3で常に呼ばれるクラス(ここでの処理でパラメータの分割処理を呼び出している)
class ZeroOffload(object):
    def __init__(self,
                 module,
                 timers,
                 overlap_comm=True,
                 prefetch_bucket_size=0,
                 max_reuse_distance=0,
                 max_live_parameters=0,
                 offload_param=True):

        see_memory_usage("ZeRoOffload initialize [begin]", force=True)
        print_rank_0(f"initialized {__class__.__name__} with args: {locals()}", force=False)

        self.module = module
        attach_module_names(self.module)
        self.dtype = list(module.parameters())[0].dtype
        # offload_param=True: 分割パラメータ (ds_tensor) を CPU pinned に常駐 (ZeRO-Offload)
        # offload_param=False: GPU に常駐 (純粋な ZeRO-3 ベースライン)
        self.offload_device = "cpu" if offload_param else "cuda"
        self.offload_param_pin_memory = offload_param
        
        self._convert_to_zero_parameters(module)  # ここでパラメータを分割 (shard 化)
        
        for m in module.modules():
            _init_external_params(m)

        _inject_parameters(module, ZeroOrderedDict)
        
        self.param_coordinators = {}
        self._prefetch_bucket_sz = int(prefetch_bucket_size)
        self._max_reuse_distance_in_numel = int(max_reuse_distance)
        self._max_available_parameters_in_numel = int(max_live_parameters)
        self.__allgather_stream = Stream() if overlap_comm else torch.cuda.default_stream()
        
        self.forward_hooks = []
        self.backward_hooks = []
        self.setup_zero_stage3_hooks()  # forward/backward 時の fetch/release フックを登録
        print_rank_0(
            f'Created module hooks: forward = {len(self.forward_hooks)}, backward = {len(self.backward_hooks)}',
            force=False)
        see_memory_usage("ZeRoOffload initialize [end]", force=True)
        
    
    def _convert_to_zero_parameters(self, module):
        non_zero_params = [p for p in module.parameters() if not is_zero_param(p)]
        if non_zero_params:
            zero_params = [p for p in module.parameters() if is_zero_param(p)]
            if zero_params:
                zero_params[0].convert_to_zero_parameters(param_list=non_zero_params)
            else:
                group = None
                Init(module=module,
                     data_parallel_group=group,
                     dtype=self.dtype,
                     remote_device=self.offload_device,
                     pin_memory=self.offload_param_pin_memory)
    
    @instrument_w_nvtx
    def partition_all_parameters(self):
        self.get_param_coordinator(training=self.module.training).release_and_reset_all(self.module)
        for param in iter_params(self.module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f"{param.ds_summary()} expected to be released")
            
    def get_param_coordinator(self, training):
        if not training in self.param_coordinators:
            # いつ all-gather / partition するかは PartitionedParameterCoordinator が決める
            self.param_coordinators[training] = PartitionedParameterCoordinator(
                prefetch_bucket_sz=self._prefetch_bucket_sz,
                max_reuse_distance_in_numel=self._max_reuse_distance_in_numel,
                max_available_parameters_in_numel=self._max_available_parameters_in_numel,
                allgather_stream=self.__allgather_stream,
            )
        return self.param_coordinators[training]
    
    def destroy(self):
        self._remove_module_hooks()

    def _remove_module_hooks(self):
        num_forward_hooks = len(self.forward_hooks)
        num_backward_hooks = len(self.backward_hooks)
        
        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()
            
        print_rank_0(f'Deleted module hooks: forward = {num_forward_hooks}, backward = {num_backward_hooks}',force=False)
    
    
    def setup_zero_stage3_hooks(self):
        self.hierarchy = 0

        @instrument_w_nvtx
        def _end_of_forward_hook(module, *args):
            if not torch._C.is_grad_enabled():
                self.get_param_coordinator(training=False).reset_step()

        self._register_hooks_recursively(self.module)
        self.module.register_forward_hook(_end_of_forward_hook)

        # Add top module to stack trace
        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(self.module)
        
    def _register_hooks_recursively(self, module, count=[0]):
        my_count = count[0]
        module.id = my_count
        
        smt.register_module_info(module)

        module._timing_call_counter = 0
        module._timing_fwd_call_stack = []      # forward 呼び出し順に push
        module._timing_bwd_active_call = None   # backward 中に参照する call_id
        
        for child in module.children():
            count[0] = count[0] + 1
            self._register_hooks_recursively(child, count=count)
        
        @instrument_w_nvtx
        def _pre_forward_module_hook(module, *args):
            module._timing_call_counter += 1
            call_id = module._timing_call_counter
            module._timing_active_fwd_call = call_id
            module._timing_fwd_call_stack.append(call_id)
            
            self.pre_sub_module_forward_function(module)
            
        @instrument_w_nvtx
        def _post_forward_module_hook(module, input, output):
            global FWD_MODULE_STACK
            FWD_MODULE_STACK.pop()
            
            if output is None:
                output = []
            elif not isinstance(output, (list, tuple)):
                if torch.is_tensor(output):
                    output = [output]
                else:
                    outputs = []
                    output = output if isinstance(output, dict) else vars(output)
                    for name, val in output.items():
                        if not name.startswith('__') and torch.is_tensor(val):
                            outputs.append(val)
                    output = outputs
            
            for item in filter(lambda item: is_zero_param(item), output):
                if not any(id(item) in m._external_params for m in FWD_MODULE_STACK):
                    item.is_external_param = True
                    module_to_register = FWD_MODULE_STACK[-1]
                    register_external_parameter(module_to_register, item)
                    print_rank_0(f'Registering dangling parameter for module {module_to_register.__class__.__name__}, ds_id = {item.ds_id}.', force=False)
                    
                    if id(item) in module._external_params:
                        print_rank_0(f'  Unregistering nested dangling parameter from module {module.__class__.__name__}, ds_id = {item.ds_id}', force=False)
                        #今終わったモジュール自身にも同じ外部登録が重複している場合は、内側の登録を解除（外側モジュールで一括管理させるため二重管理を回避）
                        unregister_external_parameter(module, item)
                    
                    item.all_gather()
            self.post_sub_module_forward_function(module)
            
        def _pre_backward_module_hook(module, inputs, output):
            @instrument_w_nvtx
            def _run_before_backward_function(sub_module):
                #同一層の 複数回 forward→1 回 backward のケース（Albert 等）に備え、参照カウントで必要回数だけプリフェッチを行う設計
                if sub_module.applied_pre_backward_ref_cnt > 0:
                    call_id = sub_module._timing_fwd_call_stack[-1] if sub_module._timing_fwd_call_stack else None
                    sub_module._timing_bwd_active_call = call_id
                    if call_id is not None:
                        last = getattr(sub_module, "_timing_last_bwd_pre_call", None)
                        if last == call_id:
                            return
                        sub_module._timing_last_bwd_pre_call = call_id
                    
                    self.pre_sub_module_backward_function(sub_module)
                    sub_module.applied_pre_backward_ref_cnt -= 1
                    
            return _apply_to_tensors_only(module,
                                        PreBackwardFunction,
                                        _run_before_backward_function,
                                        output)
            
        
        #This is an alternate to doing _post_backward_module_hook
        #it uses tensor.register_hook instead of using torch.autograd.Function
        def _alternate_post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0


            def _run_after_backward_hook(*unused):
                module.ds_grads_remaining = module.ds_grads_remaining - 1
                if module.ds_grads_remaining == 0:
                    self.post_sub_module_backward_function(module)

            def _run_before_forward_function(input):
                if input.requires_grad:
                    module.ds_grads_remaining += 1

            return _apply_forward_and_backward_to_tensors_only(
                module,
                _run_before_forward_function,
                _run_after_backward_hook,
                inputs)
            
        def _post_backward_module_hook(module, inputs): #こちらは PostBackwardFunction を使う標準ルート
            module.ds_grads_remaining = 0
            @instrument_w_nvtx
            def _run_after_backward_function(sub_module):
                if sub_module.ds_grads_remaining == 0:
                    call_id = getattr(sub_module, "_timing_bwd_active_call", None)
                    if call_id is None and sub_module._timing_fwd_call_stack:
                        call_id = sub_module._timing_fwd_call_stack[-1]
                        sub_module._timing_bwd_active_call = call_id

                    self.post_sub_module_backward_function(sub_module)

                    if sub_module._timing_fwd_call_stack:
                        sub_module._timing_fwd_call_stack.pop()
                    sub_module._timing_bwd_active_call = None
            
            return _apply_to_tensors_only(module,
                                          PostBackwardFunction,
                                          _run_after_backward_function,
                                          inputs)
            
        # Pre forward hook
        self.forward_hooks.append(module.register_forward_pre_hook(_pre_forward_module_hook))
        # Post forward hook
        self.forward_hooks.append(module.register_forward_hook(_post_forward_module_hook))
        # Pre backward hook
        self.backward_hooks.append(module.register_forward_hook(_pre_backward_module_hook))
        # post backward hook
        self.backward_hooks.append(module.register_forward_pre_hook(_post_backward_module_hook))
        
        
        
    def _ensure_pinned_cpu_full_param(self, param):
        """param.cpu_full_param を pinned CPU かつ param.data と同 shape/dtype の
        バッファとして確保 or 再利用し、そのテンソルを返す。"""
        cpu_full = getattr(param, "cpu_full_param", None)
        pool_buf = getattr(param, "cpu_full_param_pool", None)

        if cpu_full is None and pool_buf is not None:
            cpu_full = pool_buf

        needs_new = (
            cpu_full is None
            or cpu_full.device.type != "cpu"
            or not cpu_full.is_pinned()
            or cpu_full.shape != param.data.shape
            or cpu_full.dtype != param.data.dtype
        )

        if needs_new:
            cpu_full = torch.empty_like(
                param.data,
                device="cpu",
                pin_memory=True,
            )
            param.cpu_full_param_pool = cpu_full

        param.cpu_full_param = cpu_full

        return cpu_full
    
    @torch.no_grad()
    def pre_sub_module_forward_function(self, sub_module):
        _nvtx_name_s = _nvtx_name(sub_module)
        _nvtx_push(f"fwd:{_nvtx_name_s}")
        see_memory_usage(f"Before sub module function {sub_module.__class__.__name__}", force=False)

        from common import debug_params as dbg
        dbg.log_before_forward(sub_module)

        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(sub_module)

        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            param_coordinator.record_module(sub_module)

        call_id = getattr(sub_module, "_timing_active_fwd_call", None)
        # fwd_fetch: CPU wallclock (fetch_sub_module の CPU 実行時間)
        # fwd_wait_stall: CUDA Event on compute stream
        #   fetch_sub_module は内部で wait_stream(allgather_stream) を enqueue するので、
        #   その前後に CUDA Event を挟めば compute stream が fetch 完了を待って
        #   stall していた時間 = GPU が fetch のために idle だった時間 が取れる。
        if call_id is not None:
            smt.start("fwd_fetch", sub_module, call_id=call_id)
            smt.start("fwd_wait_stall", sub_module, call_id=call_id)
        _nvtx_push(f"fwd_fetch:{_nvtx_name_s}")
        param_coordinator.fetch_sub_module(sub_module)
        _nvtx_pop()  # fwd_fetch
        if call_id is not None:
            smt.end("fwd_wait_stall", sub_module, call_id=call_id)
            smt.end("fwd_fetch", sub_module, call_id=call_id)
            

        if call_id is not None:
            smt.start("fwd_exec", sub_module, call_id=call_id)
        _nvtx_push(f"fwd_exec:{_nvtx_name_s}")


    @torch.no_grad()
    def post_sub_module_forward_function(self, sub_module):
        _nvtx_pop()  # fwd_exec
        call_id = getattr(sub_module, "_timing_active_fwd_call", None)
        if call_id is not None:
            smt.end("fwd_exec", sub_module, call_id=call_id)
            
        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        params_to_release = (
                        param_coordinator.params_to_release_for_submodule(sub_module)
                        if param_coordinator.is_complete_trace()
                        else set(p.ds_id for p in iter_params(sub_module))
                    )
        _enable_fp = os.environ.get("ENABLE_FULL_PARAM_TRANSFER", "0") == "1"
        for param in iter_params(sub_module):
            if param.ds_id in params_to_release and not param.is_external_param:
                if _enable_fp:
                    cpu_full = self._ensure_pinned_cpu_full_param(param)
                    # 後で CPU から読む前にどこかで同期されていればOK
                    cpu_full.copy_(param.data, non_blocking=True)
        
                    
        param_coordinator.release_sub_module(sub_module)
        see_memory_usage(f"After sub module function {sub_module.__class__.__name__}  {sub_module.id} after release", force=False)
        _nvtx_pop()  # fwd


    @torch.no_grad()
    def pre_sub_module_backward_function(self, sub_module):
        _nvtx_name_s = _nvtx_name(sub_module)
        _nvtx_push(f"bwd:{_nvtx_name_s}")
        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            param_coordinator.record_module(sub_module)

        # bwd_fetch: CPU wallclock
        # bwd_wait_stall: CUDA Event (compute stream の GPU 側 stall 時間)
        call_id = getattr(sub_module, "_timing_bwd_active_call", None)
        if call_id is not None:
            smt.start("bwd_fetch", sub_module, call_id=call_id)
            smt.start("bwd_wait_stall", sub_module, call_id=call_id)
        _nvtx_push(f"bwd_fetch:{_nvtx_name_s}")
        param_coordinator.fetch_sub_module(sub_module)
        _nvtx_pop()  # bwd_fetch
        if call_id is not None:
            smt.end("bwd_wait_stall", sub_module, call_id=call_id)
            smt.end("bwd_fetch", sub_module, call_id=call_id)

        from common import debug_params as dbg
        dbg.log_before_backward(sub_module)

        if call_id is not None:
            smt.start("bwd_exec", sub_module, call_id=call_id)
        _nvtx_push(f"bwd_exec:{_nvtx_name_s}")


    @torch.no_grad()
    def post_sub_module_backward_function(self, sub_module):
        _nvtx_pop()  # bwd_exec
        call_id = getattr(sub_module, "_timing_bwd_active_call", None)
        if call_id is not None:
            smt.end("bwd_exec", sub_module, call_id=call_id)
            
                   
        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        params_to_release = (
                        param_coordinator.params_to_release_for_submodule(sub_module)
                        if param_coordinator.is_complete_trace()
                        else set(p.ds_id for p in iter_params(sub_module))
                    )

        # フリーせずにパラメータのストレージを再利用する
        for param in iter_params(sub_module):
            if param.ds_id in params_to_release and not param.is_external_param:
                cpu_full = getattr(param, "cpu_full_param", None)
                if cpu_full is not None:
                    pool_buf = getattr(param, "cpu_full_param_pool", None)
                    if pool_buf is None or pool_buf.data_ptr() != cpu_full.data_ptr():
                        param.cpu_full_param_pool = cpu_full

                # 次ステップでは「値」は保持しないので None にする
                param.cpu_full_param = None
        
                
        #GPUメモリ側の解放
        self.get_param_coordinator(training=sub_module.training).release_sub_module(sub_module)

        see_memory_usage(f"After sub module backward function {sub_module.__class__.__name__} {sub_module.id} after release", force=False)
        _nvtx_pop()  # bwd