import numpy as np
import torch
from wdam.utils import ROOT,cli,config,seed_all,save
from wdam.data.generator import generate
from wdam.data.windows import prepare,slices,bounds
from wdam.models.warehouse_model import WarehouseModel
from wdam.training import fit_warehouse,predict
from wdam.baselines import simple,Ridge
from wdam.metrics import seasonal_scale,sliced
from wdam.viz import plt,finish


def load(seed,pooling='last',scale=True):
    data=generate(config('data_train.yaml'),seed); splits,scaler=prepare(data)
    model=WarehouseModel(config('warehouse_model.yaml')['d_model'],scale,pooling)
    model.load_state_dict(torch.load(ROOT/'data'/f'model_{seed}_{pooling}_{scale}.pt',weights_only=True))
    return data,splits,scaler,model


def baseline_predictions(splits):
    ridge=Ridge().fit(splits['train'],config('warehouse_model.yaml')['ridge_alpha'])
    return {**simple(splits['test']['x']),'ridge':ridge.predict(splits['test'])}


def run(seed):
    cfg=config('warehouse_model.yaml'); rows=[]
    (ROOT/'data').mkdir(exist_ok=True)
    for off in config('experiments.yaml')['seed_offsets']:
        s=seed+off; data=generate(config('data_train.yaml'),s); splits,scaler=prepare(data)
        test=splits['test']; denominator=seasonal_scale(data['y'][:bounds(len(data['y']))['train'][1]])
        preds=baseline_predictions(splits)
        training={}
        for pooling,scale in [('last',True),('last',False),('learned',True)]:
            seed_all(s)
            model=WarehouseModel(cfg['d_model'],scale,pooling)
            logs=fit_warehouse(model,splits['train'],splits['val'],scaler)
            name='attention' if pooling=='last' and scale else ('unscaled' if not scale else 'learned_pool')
            preds[name]=predict(model,test['x']); training[name]=logs
            torch.save(model.state_dict(),ROOT/'data'/f'model_{s}_{pooling}_{scale}.pt')
        raw={k:v*scaler.std+scaler.mean for k,v in preds.items()}
        scores={k:sliced(test['raw_y'],v,denominator,slices(data,test['t'])) for k,v in raw.items()}
        rows.append(dict(seed=s,metrics=scores,training=training,scaler=vars(scaler),mase_denominator=denominator,
                         split_target_ranges={k:[int(w['t'][0]),int(w['t'][-1])] for k,w in splits.items()}))
        np.savez_compressed(ROOT/'data'/f'predictions_{s}.npz',t=test['t'],actual=test['raw_y'],**raw)
    groups={}
    for name in rows[0]['metrics']:
        vals=[r['metrics'][name]['all']['mae'] for r in rows]
        delta=[r['metrics']['attention']['all']['mae']-r['metrics'][name]['all']['mae'] for r in rows]
        pct=[100*d/v for d,v in zip(delta,vals)]
        groups[name]=dict(mae_mean=float(np.mean(vals)),mae_std=float(np.std(vals,ddof=1)),
            attention_minus_baseline=float(np.mean(delta)),paired_difference_std=float(np.std(delta,ddof=1)),
            percent_difference=float(np.mean(pct)),percent_difference_std=float(np.std(pct,ddof=1)),
            rmse_mean=float(np.mean([r['metrics'][name]['all']['rmse'] for r in rows])),
            rmse_std=float(np.std([r['metrics'][name]['all']['rmse'] for r in rows],ddof=1)),
            mase_mean=float(np.mean([r['metrics'][name]['all']['mase'] for r in rows])),
            mase_std=float(np.std([r['metrics'][name]['all']['mase'] for r in rows],ddof=1)))
    save('warehouse.json',dict(config=cfg,runs=rows,groups=groups))
    plt.figure(figsize=(10,4)); plt.bar(list(groups),[g['mae_mean'] for g in groups.values()],yerr=[g['mae_std'] for g in groups.values()]); plt.xticks(rotation=20); plt.ylabel('Test MAE (orders)'); finish('warehouse_comparison.png')
    for name,logs in rows[0]['training'].items(): plt.plot([r['val_mae'] for r in logs],label=name)
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Validation MAE'); finish('warehouse_training.png')
    return groups


if __name__=='__main__': cli(run)
