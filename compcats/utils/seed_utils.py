import random
import numpy as np
import torch

def seed_everything(seed: int = 0) -> int:
    """
    Seed everything for reproducibility.
    Args:
        seed (int): seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
