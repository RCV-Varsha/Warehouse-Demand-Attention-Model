import copy
import numpy as np
import torch
from wdam.utils import config, seed_all
from wdam.models.toy_model import ToyModel, batch


def diagnostics(model, cache):
    if cache is None:
        return dict(entropy=0.,max_weight=0.,grad_q=0.,grad_k=0.,logit_variance=0.)
    a=cache['A'][:,-1,:]
    return dict(entropy=float(-(a*a.clamp_min(1e-30).log()).sum(-1).mean().detach()),
        max_weight=float(a.max(-1).values.mean().detach()),
        grad_q=float(model.attention.W_Q.grad.norm()) if model.attention.W_Q.grad is not None else 0.,
        grad_k=float(model.attention.W_K.grad.norm()) if model.attention.W_K.grad is not None else 0.,
        logit_variance=float(cache['S_scaled'][:,-1,:-1].var().detach()))


def train_toy(seed,dk=16,scale=True,init=1.,lr=None,mean_pool=False):
    seed_all(seed)
    cfg=config('toy.yaml')
    model=ToyModel(cfg['d_model'],dk,scale,init,mean_pool)
    opt=torch.optim.Adam(model.parameters(),lr=lr or cfg['lr'])
    rng=torch.Generator().manual_seed(seed+1000)
    vx,vy,match=batch(torch.Generator().manual_seed(seed+2000),cfg['validation'])
    logs=[]
    for step in range(cfg['steps']+1):
        x,y,_=batch(rng,cfg['batch'])
        opt.zero_grad()
        logits,c=model(x)
        loss=torch.nn.functional.cross_entropy(logits,y)
        loss.backward()
        diag=diagnostics(model,c)
        with torch.no_grad():
            vl,vc=model(vx)
            val_loss=float(torch.nn.functional.cross_entropy(vl,vy))
            accuracy=float((vl.argmax(-1)==vy).float().mean())
        logs.append(dict(step=step,loss=float(loss.detach()),val_loss=val_loss,accuracy=accuracy,**diag))
        if step < cfg['steps']: opt.step()
    threshold=next((r['step'] for r in logs if r['accuracy'] >= cfg['threshold']),None)
    result=dict(seed=seed,dk=dk,scale=scale,init=init,lr=lr or cfg['lr'],mean_pool=mean_pool,
                config=cfg,initial=logs[0],final=logs[-1],steps_to_threshold=threshold,logs=logs)
    if not mean_pool:
        result['heatmap']=vc['A'][0].tolist()
        result['matching_position']=int(match[0])
        result['matching_mass']=float(vc['A'][torch.arange(len(match)),-1,match].mean())
    return model,result


def predict(model,x,batch_size=512):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+batch_size]))[0].numpy()
                               for i in range(0,len(x),batch_size)])


def fit_warehouse(model,train,val,scaler,epochs=None,lr=None):
    cfg=config('warehouse_model.yaml')
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr or cfg['lr'])
    best=float('inf'); state=copy.deepcopy(model.state_dict()); patience=0; logs=[]
    x,y=torch.from_numpy(train['x']),torch.from_numpy(train['y'])
    for epoch in range(epochs or cfg['epochs']):
        model.train(); losses=[]; ds=[]
        for ids in torch.randperm(len(x)).split(cfg['batch']):
            opt.zero_grad(); pred,c=model(x[ids]); loss=(pred-y[ids]).square().mean()
            loss.backward()
            if hasattr(model.attention,'W_Q'): ds.append(diagnostics(model,c))
            opt.step(); losses.append(float(loss.detach()))
        mae=float(np.mean(np.abs(predict(model,val['x'])-val['y']))*scaler.std)
        logs.append(dict(epoch=epoch,loss=float(np.mean(losses)),val_mae=mae,
                         **({k:float(np.mean([d[k] for d in ds])) for k in ds[0]} if ds else {})))
        if mae < best:
            best=mae; state=copy.deepcopy(model.state_dict()); patience=0
        else:
            patience+=1
        if patience>=cfg['patience']: break
    model.load_state_dict(state)
    return logs
