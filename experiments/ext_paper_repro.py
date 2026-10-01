import numpy as np
from wdam.utils import cli,config,save
from wdam.viz import plt,finish


def run(seed):
    cfg=config('experiments.yaml'); rng=np.random.default_rng(seed); rows=[]
    for dk in cfg['key_dims']:
        q=rng.normal(size=(cfg['paper_samples'],dk)); k=rng.normal(size=q.shape)
        s=(q*k).sum(-1)
        rows.append(dict(dk=dk,variance=float(s.var()),scaled_variance=float((s/np.sqrt(dk)).var()),samples=len(s)))
    save('paper_reproduction.json',dict(source='https://arxiv.org/html/1706.03762#S3.SS2.SSS1',section='3.2.1 footnote',rows=rows))
    plt.loglog([r['dk'] for r in rows],[r['variance'] for r in rows],marker='o',label='measured raw'); plt.loglog(cfg['key_dims'],cfg['key_dims'],ls='--',label='theory raw'); plt.loglog([r['dk'] for r in rows],[r['scaled_variance'] for r in rows],marker='o',label='measured scaled'); plt.axhline(1,ls='--',color='gray'); plt.legend(); plt.xlabel('Key dimension'); plt.ylabel('Variance'); finish('paper_reproduction.png')


if __name__=='__main__': cli(run)
