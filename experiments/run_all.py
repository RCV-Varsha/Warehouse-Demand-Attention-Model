"""Regenerate every result in dependency order. No test-set hyperparameter tuning."""
import importlib
import json
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wdam.utils import ROOT,cli,save,hashes,environment,seed_all

STAGES=['dataset','gradcheck','numerical_stability','toy_task','training_dynamics',
        'ablation_scaling','warehouse_train_eval','generalization_shift','failure_analysis',
        'attention_viz','ext_lora_adaptation','ext_jax_compare','ext_profiling','ext_paper_repro']


def run(seed):
    times={}
    for stage in STAGES:
        print(f'Running {stage}',flush=True); seed_all(seed); environment(seed)
        started=time.perf_counter()
        importlib.import_module('experiments.'+stage).run(seed)
        times[stage]=time.perf_counter()-started
    save('runtime.json',dict(stage_seconds=times,total_seconds=sum(times.values())))
    summarize(seed)
    from scripts.fill_tables import run as fill
    fill(seed)
    from scripts.audit import run as audit
    audit(seed)


def summarize(seed):
    def read(name): return json.loads((ROOT/'results'/name).read_text())
    summary=dict(seed=seed,config_hashes=hashes(),warehouse=read('warehouse.json')['groups'],
        toy={name:read('toy_task.json')[name]['final']['accuracy'] for name in ['attention','mean_pool']},
        gradcheck=read('gradcheck.json')['autograd_gradcheck'],shift=read('generalization_shift.json')['groups'],
        ablation=read('ablation_scaling.json')['groups'],lora=read('lora_adaptation.json')['groups'],
        attention_mass=read('attention_viz.json')['h3_mass_mean'],paper=read('paper_reproduction.json')['rows'],
        jax=read('jax_compare.json')['comparisons'])
    save('summary.json',summary)
    return summary


if __name__=='__main__': cli(run)
