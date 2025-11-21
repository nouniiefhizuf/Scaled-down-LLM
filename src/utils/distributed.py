import os
import torch
import torch.distributed as dist
from typing import Tuple

def setup_distributed() -> Tuple[int, int, int]:
    """
    Initializes the distributed process group.
    Returns: (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    else:
        # Single GPU / CPU fallback
        return 0, 1, 0

def cleanup_distributed() -> None:
    """Destroys the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0