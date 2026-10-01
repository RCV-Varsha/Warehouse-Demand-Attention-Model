import math
import pytest
import torch
from wdam.attention.core import attention
from wdam.attention.softmax import stable_softmax,naive_softmax
from wdam.attention.manual_backward import backward


@pytest.mark.parametrize('b,n,d,dk,dv',[(1,3,2,4,3),(2,5,4,3,2),(3,1,8,2,7)])
def test_shapes_convex_and_equivariance(b,n,d,dk,dv):
    torch.manual_seed(0)
    x=torch.randn(b,n,d,dtype=torch.float64)
    ws=[torch.randn(d,k,dtype=torch.float64) for k in [dk,dk,dv]]
    y,c=attention(x,*ws)
    assert y.shape==(b,n,dv)
    assert c['Q'].shape==c['K'].shape==(b,n,dk)
    assert c['A'].shape==c['S'].shape==(b,n,n)
    assert c['V'].shape==(b,n,dv)
    assert torch.all(y>=c['V'].min(1,keepdim=True).values-1e-12)
    assert torch.all(y<=c['V'].max(1,keepdim=True).values+1e-12)
    permutation=torch.randperm(n)
    yp,_=attention(x[:,permutation],*ws)
    torch.testing.assert_close(yp,y[:,permutation])


def test_softmax_invariants():
    x=torch.tensor([[1.,2.,-1.],[-2.,.5,1.]],dtype=torch.float64)
    a=stable_softmax(x)
    torch.testing.assert_close(a.sum(-1),torch.ones(2,dtype=x.dtype))
    assert (a>=0).all()
    torch.testing.assert_close(a,naive_softmax(x))
    torch.testing.assert_close(a,stable_softmax(x+12345))
    assert torch.isfinite(stable_softmax(x*10000)).all()


def test_mask_and_scale():
    torch.manual_seed(1); x=torch.randn(2,4,3); ws=[torch.randn(3,5) for _ in range(3)]
    _,scaled=attention(x,*ws,causal=True); _,raw=attention(x,*ws,scale=False)
    assert (scaled['A'].triu(1)==0).all()
    torch.testing.assert_close(scaled['S_scaled']*math.sqrt(5),raw['S_scaled'])
    s=torch.full((2,3),-torch.inf,requires_grad=True)
    a=stable_softmax(s); a.sum().backward()
    assert (a==0).all() and (s.grad==0).all()
    with pytest.raises(ValueError): stable_softmax(torch.tensor([float('inf')]))


def test_gradients_manual_and_finite_difference():
    torch.manual_seed(2)
    ts=[torch.randn(*s,dtype=torch.float64,requires_grad=True)*.3 for s in [(1,3,2),(2,2),(2,2),(2,2)]]
    y,c=attention(*ts); gradients=torch.autograd.grad(.5*y.square().sum(),ts)
    manual=backward(c,y,*ts[1:])
    assert torch.autograd.gradcheck(lambda *t:attention(*t)[0],tuple(ts))
    for j,name in enumerate(['X','W_Q','W_K','W_V']):
        torch.testing.assert_close(manual[name],gradients[j],rtol=1e-8,atol=1e-12)
        g=torch.empty_like(ts[j]); eps=1e-5
        for idx in range(g.numel()):
            plus=[t.detach().clone() for t in ts]; minus=[t.detach().clone() for t in ts]
            plus[j].view(-1)[idx]+=eps; minus[j].view(-1)[idx]-=eps
            g.view(-1)[idx]=(.5*attention(*plus)[0].square().sum()-.5*attention(*minus)[0].square().sum())/(2*eps)
        assert float((g-gradients[j]).norm()/gradients[j].norm())<1e-6


def test_lora_zero_initialization_and_freezing():
    from wdam.attention.core import Attention
    from wdam.attention.lora import LoRAAttention
    torch.manual_seed(0); base=Attention(4); adapted=LoRAAttention(base,2); x=torch.randn(2,3,4)
    torch.testing.assert_close(base(x)[0],adapted(x)[0])
    adapted(x)[0].square().sum().backward()
    assert base.W_Q.grad is None
    assert adapted.B_Q.grad.norm()>0


def assert_normalized(a):
    torch.testing.assert_close(a.sum(-1),torch.ones(a.shape[:-1],dtype=a.dtype))


def test_mutation_wrong_softmax_axis():
    x=torch.tensor([[1.,2.,3.],[-1.,0.,1.]])
    assert_normalized(stable_softmax(x,-1))
    with pytest.raises(AssertionError): assert_normalized(stable_softmax(x,-2))


def test_mutation_missing_transpose():
    q=torch.randn(2,5,3); k=torch.randn(2,5,3)
    assert (q@k.transpose(-1,-2)).shape==(2,5,5)
    with pytest.raises(RuntimeError): _=q@k


def test_mutation_missing_scale():
    q=torch.randn(2,5,4); k=torch.randn(2,5,4); s=q@k.transpose(-1,-2)
    def contract(actual): torch.testing.assert_close(actual,s/2)
    contract(s/2)
    with pytest.raises(AssertionError): contract(s)
