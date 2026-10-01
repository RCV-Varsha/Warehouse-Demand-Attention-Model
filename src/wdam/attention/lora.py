import math
import torch
from torch import nn
from .core import attention


class LoRAAttention(nn.Module):
    """Row-vector convention: Delta W = B @ A, B zero-initialized."""
    def __init__(self,base,rank=2):
        super().__init__()
        self.base=base
        for p in base.parameters(): p.requires_grad_(False)
        d,dk=base.W_Q.shape; dv=base.W_V.shape[1]
        self.B_Q=nn.Parameter(torch.zeros(d,rank))
        self.A_Q=nn.Parameter(torch.randn(rank,dk)/math.sqrt(rank))
        self.B_V=nn.Parameter(torch.zeros(d,rank))
        self.A_V=nn.Parameter(torch.randn(rank,dv)/math.sqrt(rank))

    def forward(self,x,**kwargs):
        return attention(x,self.base.W_Q+self.B_Q@self.A_Q,self.base.W_K,
                         self.base.W_V+self.B_V@self.A_V,scale=self.base.scale,**kwargs)
