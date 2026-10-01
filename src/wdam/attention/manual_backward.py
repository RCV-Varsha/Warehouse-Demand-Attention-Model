import math


def backward(cache, dY, W_Q, W_K, W_V, scale=True, temperature=1.):
    """VJP for an unmasked attention call; sums shared-weight gradients over batch."""
    X,Q,K,V,A = [cache[k] for k in ('X','Q','K','V','A')]
    dV = A.transpose(-2,-1) @ dY
    dA = dY @ V.transpose(-2,-1)
    dlogits = A * (dA - (dA*A).sum(-1,keepdim=True))
    dS = dlogits / temperature / (math.sqrt(Q.shape[-1]) if scale else 1)
    dQ, dK = dS @ K, dS.transpose(-2,-1) @ Q
    return dict(X=dQ @ W_Q.T + dK @ W_K.T + dV @ W_V.T,
                W_Q=(X.transpose(-2,-1) @ dQ).sum(0),
                W_K=(X.transpose(-2,-1) @ dK).sum(0),
                W_V=(X.transpose(-2,-1) @ dV).sum(0))
