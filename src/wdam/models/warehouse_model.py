import torch
from torch import nn
from wdam.attention.core import Attention


class WarehouseModel(nn.Module):
    def __init__(self,d_model=16,scale=True,pooling='last'):
        super().__init__()
        assert d_model <= 64 and pooling in ('last','learned')
        self.embed=nn.Linear(5,d_model)
        self.position=nn.Parameter(torch.randn(24,d_model)*.05)
        self.attention=Attention(d_model,scale=scale)
        self.query=nn.Parameter(torch.randn(1,1,d_model)*.05) if pooling=='learned' else None
        self.head=nn.Sequential(nn.Linear(d_model,d_model),nn.Tanh(),nn.Linear(d_model,1))

    def forward(self,x):
        h=self.embed(x)+self.position
        mask=None
        if self.query is not None:
            h=torch.cat([h,self.query.expand(len(x),-1,-1)],1)
            mask=torch.ones(25,25,dtype=torch.bool,device=x.device); mask[:,-1]=False
        y,c=self.attention(h,mask=mask)
        return self.head(y[:,-1]).squeeze(-1),c
