import torch
from wdam.attention.softmax import stable_softmax, naive_softmax
from wdam.attention.core import attention
from wdam.utils import cli, seed_all, save
from wdam.viz import plt, finish


def run(seed):
    seed_all(seed)
    rows=[]
    for dtype in (torch.float16,torch.float32,torch.float64):
        for scale in [1,10,100,1000]:
            s=(torch.tensor([[1.,2.,3.]],dtype=dtype)*scale).requires_grad_()
            a=stable_softmax(s)
            loss=(a*torch.tensor([[1.,-1.,2.]],dtype=dtype)).sum()
            grad=torch.autograd.grad(loss,s)[0]
            rows.append(dict(dtype=str(dtype),scale=scale,naive_finite=bool(torch.isfinite(naive_softmax(s)).all()),
                stable_finite=bool(torch.isfinite(a).all()),entropy=float(-(a*a.clamp_min(torch.finfo(dtype).tiny).log()).sum().detach()),
                gradient_norm=float(grad.norm())))
    tensors=[torch.randn(*s,dtype=torch.float64)*.5 for s in [(2,5,4),(4,4),(4,4),(4,4)]]
    ref=attention(*tensors)[0]
    precision={str(dt):float((attention(*[t.to(dt) for t in tensors])[0].double()-ref).abs().max()) for dt in [torch.float16,torch.float32,torch.float64]}
    masked=stable_softmax(torch.full((2,3),-torch.inf))
    assert torch.equal(masked,torch.zeros_like(masked))
    temperatures=[]
    for temp in [.1,.3,1.,3.,10.]:
        _,c=attention(*tensors,temperature=temp)
        a=c['A']
        temperatures.append(dict(temperature=temp,entropy=float(-(a*a.clamp_min(1e-30).log()).sum(-1).mean()),rank=int(torch.linalg.matrix_rank(a[0]))))
    result=dict(softmax=rows,attention_output_max_abs=precision,all_masked=masked.tolist(),temperature=temperatures)
    save('numerical_stability.json',result)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for dtype in ('torch.float16','torch.float32','torch.float64'):
        r=[x for x in rows if x['dtype']==dtype]
        ax[0].semilogx([x['scale'] for x in r],[x['entropy'] for x in r],label=dtype)
        ax[1].loglog([x['scale'] for x in r],[max(x['gradient_norm'],1e-300) for x in r],label=dtype)
    ax[0].set(xlabel='Logit multiplier',ylabel='Entropy'); ax[1].set(xlabel='Logit multiplier',ylabel='Gradient norm (zeros clipped for log axis)'); ax[0].legend(); finish('numerical_stability.png')
    plt.semilogx([r['temperature'] for r in temperatures],[r['entropy'] for r in temperatures],marker='o'); plt.xlabel('Temperature'); plt.ylabel('Entropy'); finish('temperature.png')
    return result


if __name__=='__main__': cli(run)
