from wdam.training import train_toy
from wdam.utils import cli,config,save
from wdam.viz import plt,finish


def run(seed):
    cfg=config('experiments.yaml'); rows=[]
    for lr in cfg['learning_rates']:
        for init in cfg['init_scales']:
            _,r=train_toy(seed,lr=lr,init=init); rows.append(r)
    save('training_dynamics.json',rows)
    fig,ax=plt.subplots(2,3,figsize=(14,8))
    for r in rows:
        for panel,key in zip(ax.flat,['loss','val_loss','grad_q','grad_k','entropy','max_weight']):
            panel.plot([s[key] for s in r['logs']],label=f"lr={r['lr']}, init={r['init']}"); panel.set(xlabel='Step',ylabel=key)
    ax[0,0].legend(fontsize=6); finish('training_dynamics.png')


if __name__=='__main__': cli(run)
