import numpy as np
import torch
from wdam.attention.core import attention
from wdam.attention.manual_backward import backward
from wdam.utils import cli, seed_all, save
from wdam.viz import plt, finish


def relative(a,b):
    return float((a-b).norm() / max(a.norm(), b.norm(), 1e-30))


def run(seed):
    records=[]
    manual=[]
    for dtype in (torch.float64,torch.float32):
        seed_all(seed)
        tensors=[(torch.randn(*shape,dtype=dtype)*.4).requires_grad_()
                 for shape in [(1,3,2),(2,2),(2,2),(2,2)]]
        def loss(*args):
            y,_=attention(*args)
            return .5*y.square().sum()
        L=loss(*tensors)
        gradients=torch.autograd.grad(L,tensors)
        y,cache=attention(*tensors)
        mg=backward(cache,y,*tensors[1:])
        for name,g in zip(['X','W_Q','W_K','W_V'],gradients):
            manual.append(dict(dtype=str(dtype),parameter=name,relative=relative(g,mg[name]),
                               max_abs=float((g-mg[name]).abs().max().detach())))
        for epsilon in [10.**(-i) for i in range(2,9)]:
            for j,(name,g) in enumerate(zip(['X','W_Q','W_K','W_V'],gradients)):
                numerical=torch.empty_like(tensors[j])
                for idx in range(tensors[j].numel()):
                    plus=[t.detach().clone() for t in tensors]
                    minus=[t.detach().clone() for t in tensors]
                    plus[j].view(-1)[idx]+=epsilon
                    minus[j].view(-1)[idx]-=epsilon
                    numerical.view(-1)[idx]=(loss(*plus)-loss(*minus))/(2*epsilon)
                records.append(dict(dtype=str(dtype),epsilon=epsilon,parameter=name,
                    analytical=g.flatten().tolist(),numerical=numerical.flatten().tolist(),
                    relative=relative(g,numerical),max_abs=float((g-numerical).abs().max()),
                    manual_relative=relative(mg[name].detach(),numerical)))
        if dtype==torch.float64:
            passed=torch.autograd.gradcheck(loss,tuple(tensors))
    result=dict(loss='0.5 * sum(Y**2)',sweep=records,manual=manual,autograd_gradcheck=passed)
    save('gradcheck.json',result)
    for dtype in ('torch.float64','torch.float32'):
        for name in ['X','W_Q','W_K','W_V']:
            rs=[r for r in records if r['dtype']==dtype and r['parameter']==name]
            plt.loglog([r['epsilon'] for r in rs],[r['relative'] for r in rs],label=f'{dtype[6:]} {name}')
    plt.legend(fontsize=7); plt.xlabel('Central difference epsilon'); plt.ylabel('Relative gradient error'); finish('gradcheck_eps_sweep.png')
    assert passed
    assert max(r['relative'] for r in manual if r['dtype']=='torch.float64') < 1e-8
    assert max(r['relative'] for r in records if r['dtype']=='torch.float64' and r['epsilon']==1e-5) < 1e-6
    return result


if __name__=='__main__': cli(run)
