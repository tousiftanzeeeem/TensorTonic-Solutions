import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    dk = K.shape[-1]
    weights = Q @ K.transpose(-2,-1)
    score = F.softmax(weights/(math.sqrt(dk)),dim=-1) @ V
    return score