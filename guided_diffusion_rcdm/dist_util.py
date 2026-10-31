"""
Helpers for distributed training.
"""

import io
import os
import socket
import sys
from contextlib import nullcontext
import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3
_DEVICE = None

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        th.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    device = set_device(getattr(args, "device", "auto"))
    if device.type != "cuda":
        world_size = int(os.environ.get("WORLD_SIZE", getattr(args, "world_size", 1)))
        if world_size != 1:
            raise ValueError("Multi-device training requires CUDA.")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        args.device = device
        print(f"Training on {device} without distributed initialization.")
        return

    # launched with torch.distributed.launch
    if 'SLURM_PROCID' in os.environ and args.submitit:
        print("Distributed")
    # launched with submitit on a slurm cluster
    #if 'SLURM_PROCID' in os.environ and args.submitit:
    #    args.rank = int(os.environ['SLURM_PROCID'])
    #    args.gpu = args.rank % torch.cuda.device_count()
    # launched with torch.distributed.launch
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    #elif 'SLURM_PROCID' in os.environ:
    #    args.rank = int(os.environ['SLURM_PROCID'])
    #    args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif th.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    args.device = set_device(f"cuda:{args.gpu}")
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def dev():
    """
    Get the configured training or sampling device.
    """
    if _DEVICE is None:
        return set_device()
    return _DEVICE


def set_device(device="auto"):
    global _DEVICE
    if device == "auto":
        mps = getattr(th.backends, "mps", None)
        device = (
            "cuda" if th.cuda.is_available()
            else "mps" if mps is not None and mps.is_available()
            else "cpu"
        )
    device = th.device(device)
    if device.type == "cuda":
        if not th.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        device = th.device("cuda", device.index if device.index is not None else th.cuda.current_device())
        th.cuda.set_device(device)
    elif device.type == "mps":
        mps = getattr(th.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("MPS was requested but is not available.")
    elif device.type != "cpu":
        raise ValueError(f"Unsupported device: {device}")
    _DEVICE = device
    return device


def autocast(use_fp16=False):
    if dev().type == "cuda":
        return th.autocast("cuda", enabled=use_fp16)
    if use_fp16:
        raise ValueError("FP16 training is only supported on CUDA; use FP32 on MPS/CPU.")
    return nullcontext()

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not is_dist_avail_and_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
