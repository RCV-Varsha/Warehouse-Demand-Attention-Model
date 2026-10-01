import copy
import numpy as np
from wdam.utils import cli,seed_all,config,save
from wdam.data.generator import generate
from wdam.data.windows import windows,bounds
from wdam.attention.lora import LoRAAttention
from wdam.training import fit_warehouse,predict
from experiments.warehouse_train_eval import load
from wdam.viz import plt,finish


def run(seed):
    rows=[]; cfg=config('warehouse_model.yaml')
    for off in config('experiments.yaml')['seed_offsets']:
        s=seed+off; _,_,scaler,base=load(s)
        shifted=generate(config('data_shift.yaml'),s); b=bounds(len(shifted['y']))
        train=windows(shifted,scaler,0,cfg['adapt_samples']+24)
        val=windows(shifted,scaler,*b['val']); test=windows(shifted,scaler,*b['test'])
        for method in ['none','full','lora']:
            seed_all(s); model=copy.deepcopy(base); logs=[]
            if method=='none':
                for p in model.parameters(): p.requires_grad_(False)
            if method=='lora':
                for p in model.parameters(): p.requires_grad_(False)
                model.attention=LoRAAttention(model.attention,cfg['rank'])
            if method!='none': logs=fit_warehouse(model,train,val,scaler,epochs=cfg['adapt_epochs'],lr=.001)
            p=predict(model,test['x'])*scaler.std+scaler.mean
            rows.append(dict(seed=s,method=method,mae=float(np.mean(np.abs(p-test['raw_y']))),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),training=logs,
                adaptation_targets=[int(train['t'][0]),int(train['t'][-1])],validation_targets=[int(val['t'][0]),int(val['t'][-1])],test_targets=[int(test['t'][0]),int(test['t'][-1])]))
    groups={m:dict(mae_mean=float(np.mean([r['mae'] for r in rows if r['method']==m])),mae_std=float(np.std([r['mae'] for r in rows if r['method']==m],ddof=1)),trainable=next(r['trainable'] for r in rows if r['method']==m)) for m in ['none','full','lora']}
    save('lora_adaptation.json',dict(runs=rows,groups=groups))
    plt.bar(list(groups),[g['mae_mean'] for g in groups.values()],yerr=[g['mae_std'] for g in groups.values()]); plt.ylabel('Shifted-test MAE'); finish('lora_adaptation.png')


if __name__=='__main__': cli(run)
