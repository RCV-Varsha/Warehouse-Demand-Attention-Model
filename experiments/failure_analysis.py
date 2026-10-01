import numpy as np
import torch
from wdam.utils import cli,save
from wdam.training import predict
from experiments.warehouse_train_eval import load
from wdam.viz import plt,finish


def run(seed):
    data,splits,scaler,model=load(seed); w=splits['test']
    pred=predict(model,w['x'])*scaler.std+scaler.mean; errors=pred-w['raw_y']
    worst=int(np.argmax(np.abs(errors)))
    has_spike=data['spike_active'][w['idx']].any(-1)
    candidates=np.flatnonzero(has_spike)
    selected=int(candidates[np.argmax(np.abs(errors[candidates]))])
    cases=[]
    for label,i in [('worst_overall',worst),('worst_with_input_spike',selected)]:
        x=torch.from_numpy(w['x'][i:i+1].copy()); cf=x.clone(); ids=w['idx'][i]
        positions=data['spike_active'][ids]
        cf[0,positions,0]=torch.from_numpy(((data['despiked'][ids[positions]]-scaler.mean)/scaler.std).astype('float32'))
        with torch.no_grad():
            p,c=model(x); cp,cc=model(cf)
        a,ca=c['A'][0,-1].numpy(),cc['A'][0,-1].numpy()
        cases.append(dict(label=label,target_hour=int(w['t'][i]),actual=float(w['raw_y'][i]),prediction=float(pred[i]),error=float(errors[i]),
            target_is_onset=bool(data['spike_onset'][w['t'][i]]),input_spike_positions=np.flatnonzero(positions).tolist(),
            spike_attention_mass=float(a[positions].sum()),counterfactual_prediction=float(cp[0]*scaler.std+scaler.mean),
            delta_prediction=float((cp[0]-p[0])*scaler.std),delta_attention_l1=float(np.abs(ca-a).sum()),
            attention=a.tolist(),counterfactual_attention=ca.tolist(),input_demand=data['y'][ids].tolist(),
            despiked_input=data['despiked'][ids].tolist()))
    bins=[]
    for h in range(25):
        m=data['since_spike'][w['t']]==h
        bins.append(dict(hours_since_onset=h,count=int(m.sum()),mae=float(np.abs(errors[m]).mean()) if m.any() else None,bias=float(errors[m].mean()) if m.any() else None))
    save('failure_analysis.json',dict(seed=seed,cases=cases,event_error=bins))
    plt.plot([b['hours_since_onset'] for b in bins],[b['mae'] for b in bins],marker='o'); plt.xlabel('Hours since spike onset'); plt.ylabel('MAE (orders)'); finish('failure_event_error.png')
    fig,ax=plt.subplots(2,1,figsize=(10,6))
    c=cases[1]; ax[0].plot(c['input_demand'],label='observed'); ax[0].plot(c['despiked_input'],label='de-spiked'); ax[0].legend(); ax[0].set_ylabel('Input orders')
    ax[1].plot(c['attention'],label='original'); ax[1].plot(c['counterfactual_attention'],label='counterfactual'); ax[1].legend(); ax[1].set(xlabel='Input position (old to recent)',ylabel='Attention'); finish('failure_counterfactual.png')
    return cases


if __name__=='__main__': cli(run)
