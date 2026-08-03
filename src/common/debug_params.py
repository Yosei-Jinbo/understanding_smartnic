"""ZeRO Stage 3 パラメータ検証用デバッグモジュール（--debug-params で有効化）。"""

import os
import torch
import torch.distributed as dist

# 環境変数で制御可能:
#   DEBUG_PARAMS=1            : --debug-params を付けなくても有効化
#   DEBUG_PARAMS_MAX_STEPS=N  : 最初の N ステップだけ出力 (default 3)
#   DEBUG_PARAMS_MAX_VALS=N   : 各テンソルで出力する要素数 (default 8, 0 以下なら全要素)
# 基本的にはtinyNNでこの環境変数を立てて、正常に集合通信ができているのかを検証する
ENABLED = os.environ.get("DEBUG_PARAMS", "0") == "1"
STEP = 0
MAX_STEPS = int(os.environ.get("DEBUG_PARAMS_MAX_STEPS", "3"))   # 最初の N ステップのみ出力
MAX_VALS = int(os.environ.get("DEBUG_PARAMS_MAX_VALS", "8"))     # 出力する値の最大数 (0 以下で全要素)


def enable(flag=True):
    global ENABLED
    ENABLED = flag


def set_step(step):
    global STEP
    STEP = step


def should_log():
    return ENABLED and STEP <= MAX_STEPS


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _fmt(t, max_vals=None):
    """テンソルの統計情報と値を文字列化 (max_vals<=0 なら全要素)"""
    if max_vals is None:
        max_vals = MAX_VALS
    if t is None:
        return "None"
    t_flat = t.detach().float().flatten()
    n = t_flat.numel() if max_vals <= 0 else min(t_flat.numel(), max_vals)
    vals = ", ".join(f"{v:.6f}" for v in t_flat[:n].tolist())
    suffix = f" ...({t_flat.numel()} total)" if n < t_flat.numel() else ""
    return f"shape={tuple(t.shape)} norm={t_flat.norm().item():.6f} mean={t_flat.mean().item():.6f} [{vals}{suffix}]"


def log_before_forward(sub_module):
    """順伝播前: サブモジュールの各パラメータのローカルパーティション (ds_tensor) を出力"""
    if not should_log():
        return
    rank = _rank()
    mod_name = f"{sub_module.__class__.__name__}({getattr(sub_module, 'id', '?')})"
    for name, param in sub_module.named_parameters(recurse=False):
        ds = getattr(param, "ds_tensor", None)
        print(f"[DEBUG step={STEP} rank={rank}] BEFORE_FWD {mod_name}.{name} partition={_fmt(ds)}", flush=True)


def log_ag_complete(param, tag="AG_COMPLETE"):
    """AllGather 完了後: フルパラメータ (param.data) を出力"""
    if not should_log():
        return
    rank = _rank()
    ds_id = getattr(param, "ds_id", "?")
    print(f"[DEBUG step={STEP} rank={rank}] {tag} ds_id={ds_id} full_param={_fmt(param.data)}", flush=True)


def log_before_backward(sub_module):
    """逆伝播前: フルパラメータ or パーティションを出力"""
    if not should_log():
        return
    rank = _rank()
    mod_name = f"{sub_module.__class__.__name__}({getattr(sub_module, 'id', '?')})"
    for name, param in sub_module.named_parameters(recurse=False):
        status = getattr(param, "ds_status", None)
        if status is not None and status.name == "AVAILABLE":
            print(f"[DEBUG step={STEP} rank={rank}] BEFORE_BWD {mod_name}.{name} full_param={_fmt(param.data)}", flush=True)
        else:
            ds = getattr(param, "ds_tensor", None)
            print(f"[DEBUG step={STEP} rank={rank}] BEFORE_BWD {mod_name}.{name} partition={_fmt(ds)}", flush=True)


def log_reduce_scatter(pre_rs_tensor, post_rs_tensors, sub_group_id):
    """ReduceScatter 前後の勾配を出力 (pre / post のうち渡された方だけ出す)"""
    if not should_log():
        return
    rank = _rank()
    if pre_rs_tensor is not None:
        print(f"[DEBUG step={STEP} rank={rank}] PRE_RS  sub_group={sub_group_id} {_fmt(pre_rs_tensor)}", flush=True)
    if post_rs_tensors is not None:
        for i, t in enumerate(post_rs_tensors):
            print(f"[DEBUG step={STEP} rank={rank}] POST_RS sub_group={sub_group_id} part[{i}] {_fmt(t)}", flush=True)


def log_param_update(tag, sub_group_id, fp32_tensor, fp16_tensor=None):
    """パラメータ更新の前後を出力 (通常 step / DPU step)"""
    if not should_log():
        return
    rank = _rank()
    print(f"[DEBUG step={STEP} rank={rank}] {tag} sub_group={sub_group_id} fp32={_fmt(fp32_tensor)}", flush=True)
    if fp16_tensor is not None:
        print(f"[DEBUG step={STEP} rank={rank}] {tag} sub_group={sub_group_id} fp16={_fmt(fp16_tensor)}", flush=True)
