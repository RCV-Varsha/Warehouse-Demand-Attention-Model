"""Single-head attention with visible raw tensor operations."""
import math
import torch
from torch import nn
from .softmax import stable_softmax


def attention(X, W_Q, W_K, W_V, scale=True, causal=False, temperature=1., mask=None):
    assert X.ndim == 3
    B, n, d = X.shape
    assert W_Q.shape[0] == W_K.shape[0] == W_V.shape[0] == d
    assert W_Q.shape == W_K.shape and temperature > 0
    dk, dv = W_Q.shape[1], W_V.shape[1]
    Q = X @ W_Q                             # (B,n,dk): queries
    assert Q.shape == (B,n,dk)
    K = X @ W_K                             # (B,n,dk): keys
    assert K.shape == (B,n,dk)
    V = X @ W_V                             # (B,n,dv): values
    assert V.shape == (B,n,dv)
    S = Q @ K.transpose(-2, -1)              # (B,n,n): raw similarities
    assert S.shape == (B,n,n)
    S_scaled = S / math.sqrt(dk) if scale else S
    assert S_scaled.shape == (B,n,n)
    logits = S_scaled / temperature
    if causal:
        future = torch.ones(n,n,dtype=torch.bool,device=X.device).triu(1)
        logits = logits.masked_fill(future, -torch.inf)
    if mask is not None:
        logits = logits.masked_fill(~mask, -torch.inf)
    A = stable_softmax(logits, dim=-1)       # normalize over source positions
    assert A.shape == (B,n,n)
    Y = A @ V                               # (B,n,dv): weighted values
    assert Y.shape == (B,n,dv)
    return Y, dict(X=X,Q=Q,K=K,V=V,S=S,S_scaled=S_scaled,logits=logits,A=A,Y=Y)


class Attention(nn.Module):
    def __init__(self, d, dk=None, dv=None, scale=True, init=1.):
        super().__init__()
        dk, dv = dk or d, dv or d
        self.W_Q = nn.Parameter(torch.randn(d,dk)*init/math.sqrt(d))
        self.W_K = nn.Parameter(torch.randn(d,dk)*init/math.sqrt(d))
        self.W_V = nn.Parameter(torch.randn(d,dv)/math.sqrt(d))
        self.scale = scale

    def forward(self, x, **kwargs):
        return attention(x, self.W_Q, self.W_K, self.W_V, scale=self.scale, **kwargs)
