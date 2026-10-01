import numpy as np
from wdam.utils import cli,config,save
from wdam.data.generator import generate
from wdam.data.windows import windows,bounds,slices
from wdam.baselines import simple,Ridge
from wdam.training import predict
from wdam.metrics import sliced,seasonal_scale
from experiments.warehouse_train_eval import load
from wdam.viz import plt,finish


def shift_configs():
    base=config('data_train.yaml'); combined=config('data_shift.yaml')
    return dict(base=base,noise={**base,'noise0':combined['noise0'],'noise1':combined['noise1']},
        magnitude={**base,'spike_median':combined['spike_median']},rate={**base,'spike_rate':combined['spike_rate']},combined=combined)


def run(seed):
    rows=[]
    for off in config('experiments.yaml')['seed_offsets']:
        s=seed+off; base,splits,scaler,model=load(s)
        ridge=Ridge().fit(splits['train'],config('warehouse_model.yaml')['ridge_alpha'])
        denom=seasonal_scale(base['y'][:bounds(len(base['y']))['train'][1]])
        for name,c in shift_configs().items():
            d=generate(c,s); w=windows(d,scaler,*bounds(len(d['y']))['test'])
            preds={**simple(w['x']),'ridge':ridge.predict(w),'attention':predict(model,w['x'])}
            scores={k:sliced(w['raw_y'],v*scaler.std+scaler.mean,denom,slices(d,w['t'])) for k,v in preds.items()}
            rows.append(dict(seed=s,distribution=name,metrics=scores,
                outside_train_range=float(np.mean((w['x'][:,:,0]<splits['train']['x'][:,:,0].min())|(w['x'][:,:,0]>splits['train']['x'][:,:,0].max())))))
    groups={}
    for distribution in shift_configs():
        rs=[r for r in rows if r['distribution']==distribution]
        groups[distribution]={m:dict(mean=float(np.mean([r['metrics'][m]['all']['mae'] for r in rs])),std=float(np.std([r['metrics'][m]['all']['mae'] for r in rs],ddof=1))) for m in rs[0]['metrics']}
    save('generalization_shift.json',dict(configs=shift_configs(),runs=rows,groups=groups))
    fig,ax=plt.subplots(figsize=(10,5))
    for m in groups['base']:
        ax.errorbar(list(groups),[groups[d][m]['mean'] for d in groups],yerr=[groups[d][m]['std'] for d in groups],marker='o',label=m)
    ax.set_ylabel('Test MAE (orders)'); ax.legend(); finish('generalization_shift.png')
    return groups


if __name__=='__main__': cli(run)
