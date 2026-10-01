import json
import torch
from wdam.utils import ROOT,cli
from wdam.attention.core import attention


def run(seed):
    def read(name): return json.loads((ROOT/'results'/name).read_text())
    x=torch.randn(1,4,3); weights=[torch.randn(3,2) for _ in range(3)]; _,cache=attention(x,*weights)
    print('1. Attention walkthrough: src/wdam/attention/core.py')
    print({k:list(v.shape) for k,v in cache.items()}); print('plot: results/toy_attention.png')
    g=read('gradcheck.json'); print('2. Gradcheck:',g['autograd_gradcheck'],'plot: results/gradcheck_eps_sweep.png')
    t=read('toy_task.json'); print('3. Toy accuracy:',{k:t[k]['final']['accuracy'] for k in ['attention','mean_pool']},'plot: results/toy_curves.png')
    print('4. Warehouse MAE:',{k:v['mae_mean'] for k,v in read('warehouse.json')['groups'].items()},'plot: results/warehouse_comparison.png')
    h=read('hypotheses.json'); print('5. Scaling prediction: H1 in docs/PHASE0_DESIGN.md; observed:',h['evidence']['H1'],'supported:',h['supported']['H1'],'plot: results/ablation_scaling.png')
    s=read('generalization_shift.json')['groups']; print('6. Shift attention MAE:',{k:v['attention']['mean'] for k,v in s.items()},'plot: results/generalization_shift.png')
    f=read('failure_analysis.json')['cases']; print('7. Failure:',[{k:c[k] for k in ['label','target_hour','error','delta_prediction','delta_attention_l1']} for c in f],'plot: results/failure_counterfactual.png')


if __name__=='__main__': cli(run)
