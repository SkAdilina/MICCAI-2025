import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

#epsilon = torch.tensor(1e-8, dtype=torch.float32)
epsilon = torch.tensor(0.0, dtype=torch.float32)

def clear_gpu_memory():
    torch.cuda.empty_cache()
    
def dice_score(pred, target, smooth=1e-6):
    """
    Calculate the Dice score between predicted and target tensors.

    Args:
    pred (torch.Tensor): Predicted tensor. Expected to be probabilities (e.g., after a sigmoid or softmax layer).
    target (torch.Tensor): Ground truth tensor. Expected to be binary (0s and 1s).
    smooth (float): Smoothing factor to avoid division by zero.

    Returns:
    float: Dice score.
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice