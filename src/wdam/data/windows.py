from dataclasses import dataclass
import numpy as np


@dataclass
class Scaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, y):
        return cls(float(np.mean(y)), float(max(np.std(y), 1e-8)))


def bounds(n):
    a, b = int(n * .70), int(n * .85)
    return {'train': (0, a), 'val': (a+24, b), 'test': (b+24, n)}


def windows(data, scaler, start, end):
    t = np.arange(start+24, end)
    idx = t[:, None] - np.arange(24, 0, -1)
    h, d = data['hour'][idx], data['weekday'][idx]
    x = np.stack([(data['y'][idx]-scaler.mean)/scaler.std,
                  np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                  np.sin(2*np.pi*d/7), np.cos(2*np.pi*d/7)], -1).astype('float32')
    return dict(x=x, y=((data['y'][t]-scaler.mean)/scaler.std).astype('float32'),
                raw_y=data['y'][t], t=t, idx=idx)


def prepare(data):
    blocks = bounds(len(data['y']))
    scaler = Scaler.fit(data['y'][:blocks['train'][1]])
    return {name: windows(data, scaler, *b) for name, b in blocks.items()}, scaler


def slices(data, t):
    return dict(normal=~(data['spike_active'][t] | (data['drop'][t] < 1)),
                spike=data['spike_active'][t], onset=data['spike_onset'][t],
                post_spike=data['post_spike'][t], drop=data['drop'][t] < 1,
                weekday=data['weekday'][t] < 5, weekend=data['weekday'][t] >= 5)
