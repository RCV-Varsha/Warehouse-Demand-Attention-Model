import numpy as np
import torch
from wdam.utils import cli,seed_all,save
from wdam.attention.core import attention as torch_attention


def run(seed):
    import jax
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_platform_name','cpu')
    import jax.numpy as jnp
    from wdam.attention.jax_core import attention
    seed_all(seed)
    ts=[torch.randn(*s,dtype=torch.float64,requires_grad=True) for s in [(2,4,3),(3,2),(3,2),(3,3)]]
    js=[jnp.array(t.detach().numpy()) for t in ts]
    rows=[]
    for causal in [False,True]:
        ty,_=torch_attention(*ts,causal=causal)
        jy,_=attention(*js,causal=causal)
        tg=torch.autograd.grad(.5*ty.square().sum(),ts)
        jg=jax.grad(lambda *args:.5*jnp.sum(attention(*args,causal=causal)[0]**2),argnums=(0,1,2,3))(*js)
        compiled=jax.jit(lambda *args:attention(*args,causal=causal)[0])(*js)
        rows.append(dict(causal=causal,output_max_abs=float(np.max(np.abs(np.asarray(jy)-ty.detach().numpy()))),
            jit_max_abs=float(np.max(np.abs(np.asarray(compiled)-np.asarray(jy)))),
            gradient_max_abs={name:float(np.max(np.abs(np.asarray(j)-t.numpy()))) for name,j,t in zip(['X','W_Q','W_K','W_V'],jg,tg)}))
    # JAX randomness is explicit, unlike PyTorch's seeded global generator.
    key=jax.random.PRNGKey(seed); sample=jax.random.normal(key,(3,),dtype=jnp.float64)
    save('jax_compare.json',dict(jax=jax.__version__,device=str(jax.devices()[0]),comparisons=rows,explicit_rng_sample=np.asarray(sample).tolist()))
    assert max(r['output_max_abs'] for r in rows)<1e-10
    assert max(v for r in rows for v in r['gradient_max_abs'].values())<1e-10


if __name__=='__main__': cli(run)
