import os
import sys
import torch
from torch.cuda import Stream
from collections import OrderedDict
from partition_parameters import _init_external_params
from partition_parameters import *
from partitioned_param_coordinator import PartitionedParameterCoordinator, iter_params
from stage3_utils import * #parameterの持ち方が違うため独自のmemory_usage関数を呼ぶ
from common import submodule_timing as smt

# NVTX per-submodule ranges (USE_NVTX_RANGES=1): fwd/bwd の fetch/exec を
# レンジ別に nsys で集計できるようにする。
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
        
        #全体として持っていない
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if self._parent_module._parameters._in_forward:
                register_external_parameter(FWD_MODULE_STACK[-1], param) #単純にそのモジュールのもつ外部パラメータ辞書にそのparam情報を加えるだけ, そのforward stackが実行中なのにparamがないということは外部パラメータであるため
                param.all_gather()
                print_rank_0(
                    f'Registering external parameter from getter {key} ds_id = {param.ds_id}',
                    force=False)
        return param
    
#module の _parameters を cls (ZeroOrderedDict) に置き換え、アクセスをフックする
def _inject_parameters(module, cls):
    for module in module.modules():
        if cls == ZeroOrderedDict:
            new_param = cls(parent_module=module)
        else:
            new_param = cls()
            
        for key, param in module._parameters.items():
            new_param[key] = param
        module._parameters = new_param

#outputs 内の Tensor に functional.apply(...) でノードを注入し、backward 到達時に
#backward_function が先に実行されるようにする
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
    
#outputs 内の各 Tensor に forward_function を適用し、backward_function を grad hook として登録
def _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function, output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        forward_function(outputs)
        if outputs.requires_grad: #そのノードに勾配が逆流して流れてくるか
            outputs.register_hook(backward_function) #そのテンソルに対する勾配が“計算され、集約し終わった瞬間”にbackward_functionが呼ばれる
        return outputs
    else:
        return outputs

#モジュールの逆伝播が始まる直前に pre_backward_function を実行する“入口フック”
#(backward 前のパラメータ AllGather 発火用)
class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        # 適用回数の参照カウンタ (出力が複数テンソルの場合の重複呼び出し制御用)
        if not hasattr(module, "applied_pre_backward_ref_cnt"):
            module.applied_pre_backward_ref_cnt = 0
        module.applied_pre_backward_ref_cnt += 1
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        ctx.pre_backward_function(ctx.module)
        # 非 Tensor 引数 (module, fn) には None、outputs には勾配をそのまま返す
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
        if ctx.module.ds_grads_remaining == 0: #モジュールの逆伝播が完全に終わった後の後始末
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
                 max_live_parameters=0):
        
        see_memory_usage("ZeRoOffload initialize [begin]", force=True)
        print_rank_0(f"initialized {__class__.__name__} with args: {locals()}", force=False)
        
        self.module = module
        attach_module_names(self.module)
        self.dtype = list(module.parameters())[0].dtype
        self.offload_device = None #GPUに分割したパラメータは常駐させておく
        self.offload_device = "cpu"
        self.offload_param_pin_memory = True
        
        self._convert_to_zero_parameters(module) #ここでパラメータの分割をしている, 引数は本家よりも消している
        
        for m in module.modules():
            _init_external_params(m)

        _inject_parameters(module, ZeroOrderedDict) #外部パラメータの登録
        
        self.param_coordinators = {}
        self._prefetch_bucket_sz = int(prefetch_bucket_size)
        self._max_reuse_distance_in_numel = int(max_reuse_distance)
        self._max_available_parameters_in_numel = int(max_live_parameters)
        self.__allgather_stream = Stream() if overlap_comm else torch.cuda.default_stream()
        
        self.forward_hooks = []
        self.backward_hooks = []
        self.setup_zero_stage3_hooks() #ここでforward, backward時にパラメータ分割するようなスケジューリングを登録している
        print_rank_0(
            f'Created module hooks: forward = {len(self.forward_hooks)}, backward = {len(self.backward_hooks)}',
            force=False)
        see_memory_usage("ZeRoOffload initialize [end]", force=True)
        
    
    def _convert_to_zero_parameters(self, module):
        non_zero_params = [p for p in module.parameters() if not is_zero_param(p)] #まだ ZeRO 管理下に入っていない（通常の torch.nn.Parameter のまま）パラメータを集めています。
        if non_zero_params:
            zero_params = [p for p in module.parameters() if is_zero_param(p)] #すでに ZeRO 化されたパラメータがある場合は、それを利用して non_zero_params をまとめて ZeRO 化 (if文)
            if zero_params:
                zero_params[0].convert_to_zero_parameters(param_list=non_zero_params) #ここでpartition_parameters.pyのconvert_to_zero_parameters.pyを呼び出す
            else: #まだZeRO化が終わっていないものに対して
                group = None
                Init(module=module,
                     data_parallel_group=group,
                     dtype=self.dtype,
                     remote_device=self.offload_device,
                     pin_memory=self.offload_param_pin_memory) #class Init(InsertPostInitMethodToModuleSubClasses), partition_parameters.pyのこのクラスを呼び出す
    
    @instrument_w_nvtx
    def partition_all_parameters(self): #モジュール配下（再帰）の全パラメータを強制的に解放＆状態初期化
        self.get_param_coordinator(training=self.module.training).release_and_reset_all(self.module)
        for param in iter_params(self.module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f"{param.ds_summary()} expected to be released")
            
    def get_param_coordinator(self, training):
        if not training in self.param_coordinators:
            #PartitionedParameterCoordinatorにより重みパラメータがいつall-gather、いつpartitionするかを決定する
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
        self.hierarchy = 0 #self.hierarchy を 0 に初期化。→ ZeRO Stage 3 ではモジュール階層ごとにパラメータを管理する仕組みがあり、そのレベルを追跡するためのフィールド。

        @instrument_w_nvtx
        def _end_of_forward_hook(module, *args):
            if not torch._C.is_grad_enabled(): #勾配計算が無効 → 推論モード
                self.get_param_coordinator(training=False).reset_step() #推論モードのときにパラメータコーディネータのステップをリセットする

        self._register_hooks_recursively(self.module)
        self.module.register_forward_hook(_end_of_forward_hook) #これがforward 実行後に呼ばれる, register_forward_hook()はPyTorchのnn.Moduleで定義されている

        # Add top module to stack trace
        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(self.module)
        
        #forward, backwardの再帰的な登録
    def _register_hooks_recursively(self, module, count=[0]):
        my_count = count[0]
        module.id = my_count
        
        smt.register_module_info(module)

        module._timing_call_counter = 0
        module._timing_fwd_call_stack = []      # forward 呼び出し順に push
        module._timing_bwd_active_call = None   # backward 中に参照する call_id
        
        for child in module.children():
            count[0] = count[0] + 1
            #直下の子モジュールを列挙し、通し番号をインクリメントして再帰的に同じ処理（子にも ID を振ってフック登録）。
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
            FWD_MODULE_STACK.pop() #forward 終了時に実行中モジュールのスタック（FWD_MODULE_STACK）からこのモジュールを pop

            #フックの output を扱いやすい「Tensor のリスト」に正規化するブロック。
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
                    output = outputs #Tensorだけを抽出してリスト化
            
            for item in filter(lambda item: is_zero_param(item), output): #ZeRO対応のパラメータのみについて
                #その item が、現在 forward 中のどのモジュール（スタック上）からも「外部パラメータ」として登録されていないなら
                if not any(id(item) in m._external_params for m in FWD_MODULE_STACK):
                    item.is_external_param = True
                    module_to_register = FWD_MODULE_STACK[-1]
                    register_external_parameter(module_to_register, item)
                    print_rank_0(f'Registering dangling parameter for module {module_to_register.__class__.__name__}, ds_id = {item.ds_id}.', force=False)
                    
                    if id(item) in module._external_params:
                        print_rank_0(f'  Unregistering nested dangling parameter from module {module.__class__.__name__}, ds_id = {item.ds_id}', force=False)
                        #今終わったモジュール自身にも同じ外部登録が重複している場合は、内側の登録を解除（外側モジュールで一括管理させるため二重管理を回避）
                        unregister_external_parameter(module, item)
                    
                    item.all_gather() #ZeRO で**分割保持されているパラメータを集約（all-gather）**し、完全な Tensorにしておく（外部で使われても問題ないように）
            self.post_sub_module_forward_function(module)
            
        def _pre_backward_module_hook(module, inputs, output):
            @instrument_w_nvtx
            def _run_before_backward_function(sub_module):
                #同一層の複数回 forward→1回 backward (Albert 等) に備え、参照カウントで必要回数だけプリフェッチ
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
                    
            #output 内の Tensor に PreBackwardFunction を差し込む
            return _apply_to_tensors_only(module,
                                        PreBackwardFunction,
                                        _run_before_backward_function,
                                        output)
            
        
        #_post_backward_module_hook の代替 (tensor.register_hook 版、未使用)
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
                if sub_module.ds_grads_remaining == 0: #全部の勾配が消化済みなら、後処理を実行
                    call_id = getattr(sub_module, "_timing_bwd_active_call", None)
                    if call_id is None and sub_module._timing_fwd_call_stack:
                        call_id = sub_module._timing_fwd_call_stack[-1]
                        sub_module._timing_bwd_active_call = call_id

                    self.post_sub_module_backward_function(sub_module)

                    if sub_module._timing_fwd_call_stack:
                        sub_module._timing_fwd_call_stack.pop()
                    sub_module._timing_bwd_active_call = None
                    
            #入力 Tensor に PostBackward の autograd.Function を差し込み、backward の最後に後処理が呼ばれるようにする
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
        """param.cpu_full_param として使う pinned CPU tensor (shape/dtype 一致) を
        確保 or プールから再利用して返す。"""
        # 現在アクティブなバッファ
        cpu_full = getattr(param, "cpu_full_param", None)
        # プールしてある pinned バッファ
        pool_buf = getattr(param, "cpu_full_param_pool", None)

        # アクティブ側が None なら、まずプールから昇格して使う
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
            # 初回 or 条件不一致なら新しい pinned CPU テンソルを作る
            cpu_full = torch.empty_like(
                param.data,
                device="cpu",
                pin_memory=True,
            )
            # 新しく作った pinned バッファはプールに保持しておく
            param.cpu_full_param_pool = cpu_full

        # このステップでは cpu_full_param として使う
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
        FWD_MODULE_STACK.append(sub_module) #現在実行中のモジュールとしてスタックに積む（ネスト追跡用）

        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            param_coordinator.record_module(sub_module)

        call_id = getattr(sub_module, "_timing_active_fwd_call", None)
        # fwd_fetch: CPU wallclock / fwd_wait_stall: CUDA Event で compute stream が
        # fetch 完了待ちで idle だった時間を測る
        if call_id is not None:
            smt.start("fwd_fetch", sub_module, call_id=call_id)
            smt.start("fwd_wait_stall", sub_module, call_id=call_id)
        _nvtx_push(f"fwd_fetch:{_nvtx_name_s}")
        param_coordinator.fetch_sub_module(sub_module) #このサブモジュールに必要なパラメータ shard をフェッチ（GPU/CPU/NVMe からオンデマンド搬入）
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
                    ) #releaseしてもいいパラメータを決定する
        _enable_fp = os.environ.get("ENABLE_FULL_PARAM_TRANSFER", "0") == "1"
        for param in iter_params(sub_module):
            if param.ds_id in params_to_release and not param.is_external_param:
                if _enable_fp:
                    # Pinned CPU バッファを用意 or 再利用
                    cpu_full = self._ensure_pinned_cpu_full_param(param)
                    # GPU → pinned CPU への非同期コピー（DMA）
                    # 後で CPU から読む前にどこかで同期されていればOK
                    cpu_full.copy_(param.data, non_blocking=True)
                    # param.cpu_full_param はずっと保持して再利用する
        param_coordinator.release_sub_module(sub_module) #このサブモジュールが使ったパラメータ shard を解放（必要に応じて GPU→CPU/NVMe に戻すなど）
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

        # bwd_fetch: CPU wallclock / bwd_wait_stall: CUDA Event (compute stream の stall 時間)
        call_id = getattr(sub_module, "_timing_bwd_active_call", None)
        if call_id is not None:
            smt.start("bwd_fetch", sub_module, call_id=call_id)
            smt.start("bwd_wait_stall", sub_module, call_id=call_id)
        _nvtx_push(f"bwd_fetch:{_nvtx_name_s}")
        param_coordinator.fetch_sub_module(sub_module) #fetch
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
                    ) #releaseしてもいいパラメータを決定する

        # freeせずにパラメータのストレージを再利用する
        for param in iter_params(sub_module):
            if param.ds_id in params_to_release and not param.is_external_param:
                # 今ステップで使ったバッファをプールに退避しておく
                cpu_full = getattr(param, "cpu_full_param", None)
                if cpu_full is not None:
                    pool_buf = getattr(param, "cpu_full_param_pool", None)
                    # まだプールが無い or 別バッファならプールを更新
                    if pool_buf is None or pool_buf.data_ptr() != cpu_full.data_ptr():
                        param.cpu_full_param_pool = cpu_full

                # 次ステップでは「値」は保持しないので None にする
                param.cpu_full_param = None
        
                
        #GPUメモリ側の解放
        self.get_param_coordinator(training=sub_module.training).release_sub_module(sub_module) #解放

        see_memory_usage(f"After sub module backward function {sub_module.__class__.__name__} {sub_module.id} after release", force=False)
        _nvtx_pop()  # bwd