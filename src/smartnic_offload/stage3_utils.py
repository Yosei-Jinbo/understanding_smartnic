import torch
import psutil
import gc
import torch.distributed as dist
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import logger
from typing import Deque, Dict, Tuple, List

""" debug utils """

# For lazy import with printflock()
fcntl = None

# for debug purposes map module and param objects to their fully qualified names
module_names = {}
param_names = {}


def debug_extract_module_and_param_names(model):
    # extract the fully qualified names as soon as the model is acquired
    global module_names
    global param_names
    module_names = {module: name for name, module in model.named_modules()}
    param_names = {param: name for name, param in model.named_parameters()}


def debug_module2name(module):
    if module in module_names:
        return module_names[module]
    else:
        return "unknown"


def debug_module2name_id(module):
    return f"name={debug_module2name(module)} id={module.id}"


def debug_module2name_class(module):
    return f"name={debug_module2name(module)} {module.__class__.__name__}"


def debug_param2name(param):
    if param in param_names:
        return param_names[param]
    else:
        return "unknown"


def debug_param2name_id(param):
    return f"name={debug_param2name(param)} id={param.ds_id}"


def debug_param2name_id_shape(param):
    return f"name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape}"


def debug_param2name_id_shape_device(param):
    return f"name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape} device={param.device}"


def debug_param2name_id_numel(param):
    return f"name={debug_param2name(param)} id={param.ds_id} numel={param.numel()}"


def debug_param2name_id_shape_status(param):
    return f"name={debug_param2name(param)} id={param.ds_id} shape={param.data.shape} status={param.ds_status}"


def printflock(*msgs):
    """

    For printing messages for all concurrent gpus w/o getting interleaved text.

    This is useful when debugging issues where multi-gpus don't sync.

    1. Enable the force debug in say partitioning and zero3 files
    2. Override the usual versions with ::

        def print_rank_0(message, debug=False, force=False):
            rank = deepspeed.comm.get_rank()
            printflock(f"[{rank}] {message}")
    3. run the program and you get both logs non-interleaved

    But this makes it very difficult to make sense of the output, so the ``log_rank_file`` helper
    function might be more useful, as it's easier to send each log stream into a separate file and
    then compare those.

    """
    global fcntl
    if fcntl == None:
        import fcntl

    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


fh = None


def log_rank_file(rank, *msgs):
    """
    Print to a log file of the given rank

    This is useful for debugging hanging in sync processes. Here is a possible workflow:

    1. Enable the force debug in say partitioning and zero3 files
    2. Override the usual versions of print_rank_0 in those files with ::

        def print_rank_0(message, debug=False, force=False):
            rank = deepspeed.comm.get_rank()
            log_rank_file(rank, message)

    3. run the program
    4. fix up the expected differences, e.g. different cuda numbers ::

        perl -pi -e 's|cuda:1|cuda:0|' log_rank_*

    5. now diff and see where names and ids diverge - you will find where the gpus don't do the same
    work (e.g. when some layers get conditionally skipped on one gpu but not all)

        diff -u log_rank_0.txt log_rank_1.txt | less

    """
    global fh
    if fh is None:
        fh = open(f"log_rank_{rank}.txt", "w")
    for m in msgs:
        fh.write(f"{m}\n")
    fh.flush()


def print_backward_tensors(tensor):
    def _print_bwd_tensors(grad_fn):
        print(f"Backward tensors in {grad_fn}")
        for funcs in grad_fn.next_functions:
            if funcs[0]:
                try:
                    tensor = getattr(funcs[0], 'variable')
                    print(funcs[0])
                    print(
                        f"Tensor - id: {id(tensor)}, shape: {tensor.shape}, data: {tensor}, grad: {tensor.grad}"
                    )
                except AttributeError as e:
                    _print_bwd_tensors(funcs[0])

    if hasattr(tensor, 'grad_fn'):
        _print_bwd_tensors(tensor.grad_fn)


def see_memory_usage(message, force=False):
    if not force:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    logger.info(message)
    logger.info(
        f"MA {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        CA {round(torch.cuda.memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        Max_CA {round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024))} GB ")

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logger.info(
        f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()


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
    
def get_only_unique_item(items):
    item_set = set(items)
    if len(item_set) != 1:
        raise RuntimeError(f"expected there to be only one unique element in {items}")
    unique_item, = item_set

    return unique_item

def get_lst_from_rank0(lst: List[int]) -> None:
    """
    NOTE: creates both communication and synchronization overhead so should be used
    sparingly
    """
    lst_tensor = torch.tensor(
        lst if dist.get_rank() == 0 else [-1] * len(lst),
        dtype=int,
        # device=torch.cuda.current_device(),
        device=torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"])),
        requires_grad=False,
    )
    dist.broadcast(lst_tensor, src=0, async_op=False)

    return list(lst_tensor.cpu().numpy())

@instrument_w_nvtx
def assert_ints_same_as_other_ranks(ints: List[int]) -> None:
    """
    NOTE: creates both communication and synchronization overhead so should be
    used sparingly

    takes a list of ints from each rank and ensures that they are the same
    across ranks, throwing an exception if they are not.
    """
    rank0_ints = get_lst_from_rank0(ints)
    if ints != rank0_ints:
        raise RuntimeError(f"disagreement between rank0 and rank{dist.get_rank()}: "
                           f"rank0: {rank0_ints}, rank{dist.get_rank()}: {ints}")