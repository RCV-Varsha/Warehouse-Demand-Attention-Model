import numpy as np
from wdam.utils import ROOT, config, save, cli
from wdam.data.generator import generate
from wdam.viz import plt, finish


def run(seed):
    (ROOT / 'data').mkdir(exist_ok=True)
    d = generate(config('data_train.yaml'), seed)
    np.savez_compressed(ROOT / 'data' / f'data_train_{seed}.npz', **d)
    np.savez_compressed(ROOT / 'data' / f'data_shift_{seed}.npz', **generate(config('data_shift.yaml'), seed))
    y = d['y']-d['y'].mean()
    acf = np.array([np.dot(y[:len(y)-lag], y[lag:]) / np.dot(y,y) for lag in range(337)])
    save('dataset.json', dict(seed=seed, config=config('data_train.yaml'), acf=acf.tolist(),
         mean=float(d['y'].mean()), std=float(d['y'].std()), spike_onsets=int(d['spike_onset'].sum()),
         drop_onsets=int(d['drop_onset'].sum())))
    fig, ax = plt.subplots(4, 1, figsize=(10,8))
    ax[0].plot(d['y'][:168], label='orders'); ax[0].plot(d['mean'][:168], label='event-free mean'); ax[0].legend()
    ax[1].plot(d['trend'][:168], label='trend'); ax[1].plot((d['mean']-d['trend'])[:168], label='seasonal contribution'); ax[1].legend()
    ax[2].plot((d['mean']*d['spike']*d['drop'])[:168], label='spike contribution'); ax[2].plot(d['noise'][:168], label='noise'); ax[2].legend()
    ax[3].plot(d['drop'][:168]); ax[3].set(xlabel='Hour', ylabel='Drop multiplier')
    finish('dataset_components.png')
    plt.plot(acf); plt.xlabel('Lag (hours)'); plt.ylabel('Biased sample ACF'); finish('dataset_acf.png')
    plt.hist(d['y'], bins=60); plt.xlabel('Orders'); plt.ylabel('Count'); finish('dataset_histogram.png')
    plt.figure(figsize=(10,3)); plt.plot(d['y'], lw=.5); ix=np.flatnonzero(d['spike_onset']); plt.scatter(ix,d['y'][ix],c='r',s=10,label='Spike onset'); plt.legend(); plt.xlabel('Hour'); plt.ylabel('Orders'); finish('dataset_events.png')


if __name__ == '__main__': cli(run)
