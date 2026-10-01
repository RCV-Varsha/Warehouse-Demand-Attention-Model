import numpy as np
import torch
from wdam.utils import cli,config,save
from experiments.warehouse_train_eval import load
from wdam.viz import plt,finish


def run(seed):
    rows=[]
    for off in config('experiments.yaml')['seed_offsets']:
        data,splits,scaler,model=load(seed+off); w=splits['test']; weights=[]
        with torch.no_grad():
            for chunk in np.array_split(w['x'],4): weights.append(model(torch.from_numpy(chunk))[1]['A'][:,-1].numpy())
        a=np.concatenate(weights); by_hour=np.stack([a[w['t']%24==h].mean(0) for h in range(24)])
        with torch.no_grad(): full=model(torch.from_numpy(w['x'][:1]))[1]['A'][0].numpy()
        rows.append(dict(seed=seed+off,mean_by_position=a.mean(0).tolist(),mean_by_hour=by_hour.tolist(),
                         h3_mass=float(a[:,[0,21,22,23]].sum(-1).mean()),sample=a[:5].tolist(),full_sample=full.tolist()))
    save('attention_viz.json',dict(runs=rows,h3_mass_mean=float(np.mean([r['h3_mass'] for r in rows]))))
    r=rows[0]
    plt.imshow(r['mean_by_hour'],aspect='auto'); plt.colorbar(); plt.xlabel('Position (lag 24 to lag 1)'); plt.ylabel('Target hour'); finish('attention_by_hour.png')
    plt.plot(range(24,0,-1),r['mean_by_position']); plt.xlabel('Lag'); plt.ylabel('Average attention'); finish('attention_by_lag.png')
    plt.imshow(r['sample'],aspect='auto'); plt.colorbar(); plt.xlabel('Position'); plt.ylabel('Test sample'); finish('attention_samples.png')
    plt.imshow(r['full_sample']); plt.colorbar(); plt.xlabel('Source position'); plt.ylabel('Query position'); finish('attention_individual.png')


if __name__=='__main__': cli(run)
