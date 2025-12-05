"""
Random seed fixing utility for reproducibility.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Fix random seeds for reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed}")