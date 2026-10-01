from wdam.training import train_toy
from wdam.utils import cli,save
from wdam.viz import plt,finish


def run(seed):
    _,a=train_toy(seed)
    _,b=train_toy(seed,mean_pool=True)
    save('toy_task.json',dict(attention=a,mean_pool=b,chance=.25))
    for name,r in [('attention',a),('mean pool',b)]:
        plt.plot([v['loss'] for v in r['logs']],label=name+' train')
        plt.plot([v['val_loss'] for v in r['logs']],label=name+' val',alpha=.7)
    plt.legend(); plt.xlabel('Step'); plt.ylabel('Cross entropy'); finish('toy_curves.png')
    plt.imshow(a['heatmap'],vmin=0,vmax=1); plt.colorbar(); plt.xlabel('Source position'); plt.ylabel('Query position'); plt.title(f"Query matches source {a['matching_position']}"); finish('toy_attention.png')
    return a


if __name__=='__main__': cli(run)
