import time
import gzip
import json
import torch
from torch.profiler import profile,ProfilerActivity
from wdam.attention.core import attention
from wdam.attention.softmax import stable_softmax,naive_softmax
from wdam.utils import ROOT,cli,seed_all,config,save
from wdam.viz import plt,finish


def run(seed):
    seed_all(seed); cfg=config('experiments.yaml'); rows=[]
    for dtype in [torch.float32,torch.float64]:
        for n in cfg['profile_lengths']:
            x=torch.randn(1,n,16,dtype=dtype); ws=[torch.randn(16,16,dtype=dtype)*.1 for _ in range(3)]
            for _ in range(3): attention(x,*ws)
            start=time.perf_counter()
            for _ in range(cfg['profile_repeats']): attention(x,*ws)
            elapsed=(time.perf_counter()-start)/cfg['profile_repeats']
            with profile(activities=[ProfilerActivity.CPU],profile_memory=True,record_shapes=True,with_stack=True) as p:
                attention(x,*ws)
            (ROOT/'tmp').mkdir(exist_ok=True)
            trace=ROOT/'tmp'/'memory.raw.json.gz'
            p.export_memory_timeline(str(trace))
            with gzip.open(trace,'rt') as stream: events=json.load(stream)
            baseline=sum(size for _,action,size,_ in events if action==1)
            live=baseline; peak=baseline
            for _,action,size,_ in events:
                if action==2: live+=size
                elif action==4: live-=size
                peak=max(peak,live)
            peak-=baseline
            # Explicit live output storage is also reported; this is not peak RSS.
            y,c=attention(x,*ws)
            storages={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in c.values()}
            s=c['S_scaled']; costs={}
            for name,fn in [('naive',naive_softmax),('stable',stable_softmax)]:
                start=time.perf_counter()
                for _ in range(cfg['profile_repeats']): fn(s)
                costs[name]=(time.perf_counter()-start)/cfg['profile_repeats']
            rows.append(dict(dtype=str(dtype),n=n,seconds=elapsed,allocator_peak_bytes=peak,
                retained_tensor_bytes=sum(storages.values()),softmax_seconds=costs))
    save('profiling.json',dict(measurement='CPU profiler allocation-event high-water mark; excludes process/library RSS; wall times are nondeterministic',rows=rows))
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for dtype in ['torch.float32','torch.float64']:
        rs=[r for r in rows if r['dtype']==dtype]
        ax[0].loglog([r['n'] for r in rs],[r['seconds'] for r in rs],marker='o',label=dtype)
        ax[1].loglog([r['n'] for r in rs],[r['allocator_peak_bytes'] for r in rs],marker='o',label=dtype)
    ax[0].set(xlabel='Sequence length',ylabel='Seconds / forward'); ax[1].set(xlabel='Sequence length',ylabel='Allocator peak bytes'); ax[0].legend(); finish('profiling.png')


if __name__=='__main__': cli(run)
