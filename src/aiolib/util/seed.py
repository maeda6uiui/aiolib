import numpy as np
import random
import torch

def set_seed(seed:int):
    """
    シードを設定する。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
