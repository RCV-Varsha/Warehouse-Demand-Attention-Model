import numpy as np
from wdam.training import train_toy
from wdam.utils import cli,config,save
from wdam.viz import plt,finish


def run(seed):
    cfg=config('experiments.yaml'); rows=[]; groups=[]
    for dk in cfg['key_dims']:
        for scale in [False,True]:
            runs=[]
            for off in cfg['seed_offsets']:
                _,r=train_toy(seed+off,dk=dk,scale=scale)
                r.pop('logs'); runs.append(r); rows.append(r)
            g=dict(dk=dk,scale=scale)
            for phase,key in [('final','loss'),('final','accuracy'),('initial','entropy'),('final','entropy'),('initial','grad_q'),('initial','grad_k'),('initial','logit_variance')]:
                values=[r[phase][key] for r in runs]
                g[phase+'_'+key]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)))
            achieved=[r['steps_to_threshold'] for r in runs if r['steps_to_threshold'] is not None]
            g['threshold_reached']=len(achieved)
            g['steps_to_threshold_mean_if_reached']=float(np.mean(achieved)) if achieved else None
            g['steps_to_threshold_std_if_reached']=float(np.std(achieved,ddof=1)) if len(achieved)>1 else None
            groups.append(g)
    save('ablation_scaling.json',dict(runs=rows,groups=groups))
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for scale in [False,True]:
        gs=[g for g in groups if g['scale']==scale]
        for panel,key in zip(ax,['final_accuracy','initial_entropy','initial_logit_variance']):
            panel.errorbar([g['dk'] for g in gs],[g[key]['mean'] for g in gs],yerr=[g[key]['std'] for g in gs],label=str(scale),marker='o'); panel.set(xscale='log',xlabel='Key dimension',ylabel=key)
    ax[0].legend(title='Scaled'); finish('ablation_scaling.png')
    return groups


if __name__=='__main__': cli(run)
