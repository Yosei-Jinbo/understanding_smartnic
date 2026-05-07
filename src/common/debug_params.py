# common/debug_params.py
"""
ZeRO Stage 3 パラメータ検証用デバッグモジュール。

--debug-params フラグで有効化すると、各ステップで以下を出力:
  1. BEFORE_FWD:  順伝播前のローカルパーティション (ds_tensor)
  2. AG_COMPLETE: AllGather完了後のフルパラメータ (param.data)
  3. BEFORE_BWD:  逆伝播前のパラメータ (フルまたはパーティション)
  4. PRE_RS:      ReduceScatter前の勾配
  5. POST_RS:     ReduceScatter後の集約済み勾配
  6. BEFORE_STEP: optimizer.step() 前の FP32 マスター重み
  7. AFTER_STEP:  optimizer.step() 後の FP32 マスター重み
  8. AFTER_COPY:  FP32→FP16 コピー後の FP16 パーティション
"""

import torch
import torch.distributed as dist

ENABLED = False
STEP = 0
MAX_STEPS = 3   # 最初の N ステップのみ出力
MAX_VALS = 8    # 出力する値の最大数


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


def _fmt(t, max_vals=MAX_VALS):
    """テンソルの統計情報と先頭値を文字列化"""
    if t is None:
        return "None"
    t_flat = t.detach().float().flatten()
    n = min(len(t_flat), max_vals)
    vals = ", ".join(f"{v:.6f}" for v in t_flat[:n].tolist())
    suffix = f" ...({t_flat.numel()} total)" if t_flat.numel() > n else ""
    return f"shape={tuple(t.shape)} norm={t_flat.norm().item():.6f} mean={t_flat.mean().item():.6f} [{vals}{suffix}]"


# ========== Public API ==========

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
    """ReduceScatter 前後の勾配を出力"""
    if not should_log():
        return
    rank = _rank()
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
