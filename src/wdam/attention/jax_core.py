"""Functional JAX counterpart; framework import is optional."""
import jax.numpy as jnp


def attention(x,wq,wk,wv,scale=True,causal=False,temperature=1.,mask=None):
    q=x@wq; k=x@wk; v=x@wv
    s=q@jnp.swapaxes(k,-2,-1)
    ss=s/jnp.sqrt(q.shape[-1]) if scale else s
    logits=ss/temperature
    if causal: logits=jnp.where(jnp.triu(jnp.ones(s.shape[-2:],dtype=bool),1),-jnp.inf,logits)
    if mask is not None: logits=jnp.where(mask,logits,-jnp.inf)
    maximum=jnp.max(logits,axis=-1,keepdims=True)
    maximum=jnp.where(jnp.isfinite(maximum),maximum,0.)
    e=jnp.exp(logits-maximum)
    a=e/jnp.maximum(e.sum(axis=-1,keepdims=True),jnp.finfo(e.dtype).tiny)
    y=a@v
    return y,dict(Q=q,K=k,V=v,S=s,S_scaled=ss,A=a,Y=y)
