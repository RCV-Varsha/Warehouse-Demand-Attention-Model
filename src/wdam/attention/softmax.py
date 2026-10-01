import torch


def naive_softmax(x, dim=-1):
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


def stable_softmax(x, dim=-1):
    """All -inf rows return zeros with zero gradients; +inf/NaN are rejected."""
    if torch.isnan(x).any() or torch.isposinf(x).any():
        raise ValueError('Logits must be finite or masked with -inf')
    maximum = x.max(dim=dim, keepdim=True).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    e = (x-maximum).exp()
    denominator = e.sum(dim=dim, keepdim=True)
    return e / denominator.clamp_min(torch.finfo(x.dtype).tiny)
