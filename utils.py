from functorch import vmap
import numpy as np
import torch
from dmff_torch.settings import DO_JIT

def jit_condition(*args, **kwargs):
    def jit_deco(func):
        if DO_JIT:
            return torch.jit.script(func, *args, **kwargs)
        else:
            return func
    return jit_deco

def jit_condition_trace(*args, **kwargs):
    def jit_deco(func):
        if DO_JIT:
            return torch.jit.trace(func, *args, **kwargs)
        else:
            return func
    return jit_deco

@vmap
def regularize_pairs(p):
    # using vmap; we view 2-d array with only its element (1-d array, exampe p = p[m]), but dp is same as  p[:,0] - p[:,1]
    dp = p[1] - p[0]
    dp = torch.where(dp > torch.tensor(0, dtype=torch.int32, device=dp.device), torch.tensor(0,dtype=torch.int32, device=dp.device), torch.tensor(1, dtype=torch.int32, device=dp.device))
    # vmap don't support .item on a Tensor, for nopbc system, no buffer atoms 
    #dp_vec = torch.tensor([dp, 2 * dp])
    p[0] = p[0] - dp
    p[1] = p[1] - dp * 2
    return p 

@vmap
def pair_buffer_scales(p):
    dp = p[0] - p[1]
    return torch.where(dp < torch.tensor(0, dtype=torch.int32, device=dp.device), torch.tensor(1, dtype=torch.int32, device=dp.device), torch.tensor(0, dtype=torch.int32, device=dp.device))
