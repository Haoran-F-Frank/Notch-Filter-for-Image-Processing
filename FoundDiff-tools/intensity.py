"""Shared HU <-> model [0,1] intensity mapping."""

import numpy as np
import torch

# norm = clip((hu + 2000) / 3000, 0, 1)  ->  HU in [-2000, +1000]
# hu   = norm * 3000 - 2000
HU_ADD = 2000
HU_RANGE = 3000


def hu_to_model_norm(hu):
    return np.clip((np.asarray(hu, dtype=np.float32) + HU_ADD) / HU_RANGE, 0.0, 1.0)


def model_norm_to_hu(norm):
    return np.asarray(norm, dtype=np.float32) * HU_RANGE - HU_ADD


def preprocess_hu_slice(hu_slice):
    norm = hu_to_model_norm(hu_slice)
    tensor = torch.from_numpy(norm.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.unsqueeze(0)
