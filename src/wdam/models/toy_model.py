import torch
from torch import nn
from wdam.attention.core import Attention


def batch(rng, count, keys=8, values=4, pairs=4):
    """Bound key/value records followed by query; key identities never repeat."""
    key=torch.rand(count,keys,generator=rng).argsort(-1)[:,:pairs]
    value=torch.randint(values,(count,pairs),generator=rng)
    index=torch.randint(pairs,(count,),generator=rng)
    x=torch.zeros(count,pairs+1,keys+values+1)
    x[:,:pairs,:keys]=torch.nn.functional.one_hot(key,keys)
    x[:,:pairs,keys:keys+values]=torch.nn.functional.one_hot(value,values)
    x[:,-1,:keys]=torch.nn.functional.one_hot(key[torch.arange(count),index],keys)
    x[:,-1,-1]=1
    return x,value[torch.arange(count),index],index


class ToyModel(nn.Module):
    def __init__(self,d_model=16,dk=16,scale=True,init=1.,mean_pool=False):
        super().__init__()
        self.embed=nn.Linear(13,d_model,bias=False)
        self.attention=Attention(d_model,dk,d_model,scale,init)
        self.head=nn.Linear(d_model,4)
        self.mean_pool=mean_pool

    def forward(self,x):
        h=self.embed(x)
        if self.mean_pool:
            return self.head(h.mean(1)),None
        mask=torch.ones(h.shape[1],h.shape[1],dtype=torch.bool)
        mask[:,-1]=False
        y,c=self.attention(h,mask=mask)
        return self.head(y[:,-1]),c
