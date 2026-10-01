"""Generate measured tables and explicit hypothesis decisions from result files."""
import json
import re
import numpy as np
from wdam.utils import ROOT,cli,save


def read(name): return json.loads((ROOT/'results'/name).read_text())


def inject(path,key,content):
    p=ROOT/path
    source=p.read_text(encoding='utf-8')
    pattern=rf'<!-- AUTO:{key} -->.*?<!-- /AUTO:{key} -->'
    text=f'<!-- AUTO:{key} -->\n{content}\n<!-- /AUTO:{key} -->'
    assert re.search(pattern,source,re.S),f'Missing {key} in {path}'
    p.write_text(re.sub(pattern,lambda _:text,source,flags=re.S),encoding='utf-8')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def run(seed):
    summary=read('summary.json'); w=summary['warehouse']; ab=read('ablation_scaling.json'); shift=summary['shift']; toy=read('toy_task.json')
    warehouse=table(['Model','MAE mean ± sample std','RMSE mean ± std','MASE mean ± std','Attention minus model MAE','Difference % ± std'],
        [[k,f"{v['mae_mean']:.3f} ± {v['mae_std']:.3f}",f"{v['rmse_mean']:.3f} ± {v['rmse_std']:.3f}",f"{v['mase_mean']:.3f} ± {v['mase_std']:.3f}",f"{v['attention_minus_baseline']:.3f}",f"{v['percent_difference']:.2f} ± {v['percent_difference_std']:.2f}"] for k,v in w.items()])
    inject('README.md','SUMMARY',warehouse+'\n\nSource: `results/warehouse.json`. Negative differences favor attention.')
    inject('docs/EXPERIMENTS.md','WAREHOUSE',warehouse)
    scaled={g['dk']:g for g in ab['groups'] if g['scale']}; unscaled={g['dk']:g for g in ab['groups'] if not g['scale']}
    h1entropy=all(unscaled[d]['initial_entropy']['mean']<scaled[d]['initial_entropy']['mean'] for d in [64,256])
    h1accuracy=all(unscaled[d]['final_accuracy']['mean']<scaled[d]['final_accuracy']['mean'] for d in [64,256])
    h1variance=all(abs(r['scaled_variance']-1)<.1 for r in summary['paper'])
    h2=all(w['attention']['mae_mean']<=.9*w[b]['mae_mean'] for b in ['last_value','moving_average']) and all(abs(w['attention']['mae_mean']/w[b]['mae_mean']-1)<=.15 for b in ['seasonal_naive','ridge'])
    shifts=read('generalization_shift.json')['runs']
    def slice_mean(dist,name,model='attention'):
        return float(np.mean([r['metrics'][model][name]['mae'] for r in shifts if r['distribution']==dist]))
    degradation={m:shift['combined'][m]['mean']/shift['base'][m]['mean']-1 for m in shift['base']}
    spike_deg=slice_mean('combined','spike')/slice_mean('base','spike')-1
    normal_deg=slice_mean('combined','normal')/slice_mean('base','normal')-1
    wr=read('warehouse.json')['runs']
    slice_rows=[]
    for name in wr[0]['metrics']['attention']:
        entries=[r['metrics']['attention'][name] for r in wr]
        entry=dict(slice=name,counts=[e['count'] for e in entries])
        for metric in ['mae','rmse','mase','bias']:
            vals=[e[metric] for e in entries if e[metric] is not None]
            entry[metric+'_mean']=float(np.mean(vals)) if vals else None
            entry[metric+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        slice_rows.append(entry)
    save('slice_summary.json',slice_rows)
    onset=float(np.mean([r['metrics']['attention']['onset']['mae'] for r in wr])); normal=float(np.mean([r['metrics']['attention']['normal']['mae'] for r in wr])); post_bias=float(np.mean([r['metrics']['attention']['post_spike']['bias'] for r in wr]))
    decisions=dict(H1=h1entropy and h1accuracy and h1variance,H2=h2,H3=summary['attention_mass']>.25,
        H4=all(v>=.2 for v in degradation.values()) and spike_deg>normal_deg,H5=onset>=2*normal and post_bias>0,
        H6=summary['toy']['attention']>.9 and summary['toy']['mean_pool']<.6,
        H7=summary['lora']['lora']['trainable']<summary['lora']['full']['trainable'] and summary['lora']['lora']['mae_mean']<summary['lora']['none']['mae_mean'])
    details=dict(H1=f'Entropy condition: {h1entropy}; accuracy condition: {h1accuracy}; variance condition: {h1variance}',
        H2='Attention/reference MAE ratios: '+', '.join(f"{name}={w['attention']['mae_mean']/w[name]['mae_mean']:.3f}" for name in ['last_value','moving_average','seasonal_naive','ridge']),
        H3=f"Selected-lag mass {summary['attention_mass']:.4f}",
        H4=f'Combined relative MAE changes {degradation}; spike {spike_deg:.3f}, normal {normal_deg:.3f}',
        H5=f'Onset MAE {onset:.3f}; normal MAE {normal:.3f}; post-onset bias {post_bias:.3f}',
        H6=f"Attention accuracy {summary['toy']['attention']:.4f}; mean-pool accuracy {summary['toy']['mean_pool']:.4f}",
        H7=f"LoRA MAE {summary['lora']['lora']['mae_mean']:.3f}; no adaptation {summary['lora']['none']['mae_mean']:.3f}")
    save('hypotheses.json',dict(supported=decisions,evidence=details,criterion='Conjunction of all predeclared conditions; descriptive seed means, not a significance test'))
    design=(ROOT/'docs/PHASE0_DESIGN.md').read_text()
    hypotheses={m.group(1):m.group(2) for m in re.finditer(r'- (H\d): (.+)',design)}
    htable=table(['Hypothesis (verbatim)','Observation','Decision'],[[hypotheses[k],details[k],'Supported' if v else 'Not supported in full'] for k,v in decisions.items()])
    inject('docs/EXPERIMENTS.md','HYPOTHESES',htable)
    inject('docs/EXPERIMENTS.md','ABLATION',table(['d_k','Scaled','Final accuracy ± std','Initial entropy ± std','Initial logit variance','Reached threshold'],
        [[g['dk'],g['scale'],f"{g['final_accuracy']['mean']:.4f} ± {g['final_accuracy']['std']:.4f}",f"{g['initial_entropy']['mean']:.4f} ± {g['initial_entropy']['std']:.4f}",f"{g['initial_logit_variance']['mean']:.5f}",g['threshold_reached']] for g in ab['groups']]))
    inject('docs/EXPERIMENTS.md','SHIFT',table(['Distribution']+list(shift['base']),[[dist]+[f"{vals[m]['mean']:.3f} ± {vals[m]['std']:.3f}" for m in shift['base']] for dist,vals in shift.items()]))
    grad=read('gradcheck.json'); num=read('numerical_stability.json'); dyn=read('training_dynamics.json')
    sections=[]
    sections.append('Primary warehouse model by slice: '+table(['Slice','Counts by seed','MAE ± std','RMSE ± std','MASE ± std','Bias mean'],[[r['slice'],r['counts'],f"{r['mae_mean']:.3f} ± {r['mae_std']:.3f}",f"{r['rmse_mean']:.3f} ± {r['rmse_std']:.3f}",f"{r['mase_mean']:.3f} ± {r['mase_std']:.3f}",f"{r['bias_mean']:.3f}"] for r in slice_rows]))
    sections.append('Gradient checks: '+table(['dtype','parameter','manual relative error'],[[r['dtype'],r['parameter'],f"{r['relative']:.3e}"] for r in grad['manual']]))
    sections.append('Numerical stability: '+table(['dtype','logit multiplier','naive finite','stable finite','entropy','gradient norm'],[[r[k] for k in ['dtype','scale','naive_finite','stable_finite','entropy','gradient_norm']] for r in num['softmax']]))
    sections.append('Toy: '+table(['Model','Initial val loss','Final val loss','Final accuracy'],[[k,toy[k]['initial']['val_loss'],toy[k]['final']['val_loss'],toy[k]['final']['accuracy']] for k in ['attention','mean_pool']]))
    sections.append('Training dynamics: '+table(['LR','Q/K init scale','Initial entropy','Initial Q gradient','Final val loss','Final accuracy'],[[r['lr'],r['init'],r['initial']['entropy'],r['initial']['grad_q'],r['final']['val_loss'],r['final']['accuracy']] for r in dyn]))
    reference=next(r for r in dyn if r['lr']==.01 and r['init']==1.)
    high_lr=next(r for r in dyn if r['lr']==.1 and r['init']==1.)
    large_init=next(r for r in dyn if r['lr']==.01 and r['init']==10.)
    sections.append(f"At fixed Q/K init scale, raising LR from {reference['lr']} to {high_lr['lr']} changes final validation accuracy from {reference['final']['accuracy']:.4f} to {high_lr['final']['accuracy']:.4f}. At fixed LR {reference['lr']}, increasing init scale from {reference['init']} to {large_init['init']} lowers initial entropy from {reference['initial']['entropy']:.4f} to {large_init['initial']['entropy']:.4f}, yet final accuracy is {large_init['final']['accuracy']:.4f}. Thus initial saturation alone does not establish failed learning; the optimizer can alter the projection scale. These outcomes follow directly from the dynamics traces.")
    sections.append('Adaptation: '+table(['Method','Trainable parameters','MAE ± std','Paired MAE change ± std'],[[m,g['trainable'],f"{g['mae_mean']:.3f} ± {g['mae_std']:.3f}",f"{g['paired_difference_mean']:.4f} ± {g['paired_difference_std']:.4f}"] for m,g in summary['lora'].items()]))
    lg=summary['lora']['lora']
    sections.append('The LoRA mean change is smaller than its paired-seed standard deviation; under the predeclared descriptive rule the practical improvement is inconclusive.' if abs(lg['paired_difference_mean'])<lg['paired_difference_std'] else 'The LoRA mean change exceeds its paired-seed standard deviation; this remains descriptive evidence, not a formal significance test.')
    sections.append('JAX: '+table(['Causal','Output max error','Largest gradient error'],[[r['causal'],r['output_max_abs'],max(r['gradient_max_abs'].values())] for r in summary['jax']]))
    sections.append('Paper reproduction: '+table(['d_k','Raw variance','Scaled variance'],[[r['dk'],r['variance'],r['scaled_variance']] for r in summary['paper']]))
    inject('docs/EXPERIMENTS.md','DETAILS','\n\n'.join(sections))
    failures=read('failure_analysis.json')['cases']
    ftable=table(['Case','Hour','Target is onset','Actual','Predicted','Error','Input spike attention mass','Counterfactual delta prediction','Attention L1 delta'],[[c[k] for k in ['label','target_hour','target_is_onset','actual','prediction','error','spike_attention_mass','delta_prediction','delta_attention_l1']] for c in failures])
    c=failures[1]
    direction='raises' if c['delta_prediction']>0 else 'lowers'
    observation=f"For the selected historical-spike case, removing the input spike {direction} the forecast by {abs(c['delta_prediction']):.3f} orders, while the original absolute error is {abs(c['error']):.3f}. "
    observation+=('This is inconsistent with a simple story that the past spike only pushes the prediction upward. The target is a new unpredictable onset; the counterfactual does not remove that target event.' if c['delta_prediction']>0 and c['target_is_onset'] else 'The intervention measures sensitivity of the fitted function and does not establish real-world causality.')
    observation+=f" Across seeds, post-onset mean signed bias is {post_bias:.3f} orders (results/hypotheses.json); "+('underprediction contradicts the positive-bias component of H5.' if post_bias<0 else 'the sign agrees with the positive-bias component of H5.')
    inject('docs/FAILURE_ANALYSIS.md','FAILURE',ftable+'\n\n'+observation)
    inject('docs/REFLECTION.md','OUTCOMES',table(['Hypothesis','Decision'],[[k,'Supported' if v else 'Not supported in full'] for k,v in decisions.items()]))
    if (ROOT/'results/runtime.json').exists():
        runtime=read('runtime.json')
        inject('README.md','RUNTIME',f"Measured experiment-stage CPU wall time: {runtime['total_seconds']/60:.2f} minutes on the environment in `results/environment.json`. Source: `results/runtime.json`. Download/setup and final test/report overhead are excluded; timings vary.")


if __name__=='__main__': cli(run)
