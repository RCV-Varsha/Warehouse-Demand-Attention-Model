"""Independent random streams keep one-factor distribution shifts paired."""
import argparse
from pathlib import Path
import numpy as np
import yaml
from wdam.utils import seed_all, environment


def generate(c, seed):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    sr, sm, dr, dm, nr = streams
    n = c['hours']
    t = np.arange(n)
    hour, weekday = t % 24, (t // 24) % 7
    trend = c['base'] * (1 + c['trend'] * t / n)
    daily = sum(a * np.cos(2 * np.pi * k * (hour - c['daily_center']) / 24)
                for k, a in enumerate(c['daily_amplitudes'], 1))
    weekly = np.asarray(c['weekday'])[weekday]
    mean = trend * (1 + daily) * weekly
    spike_count = sr.poisson(c['spike_rate'] / 24, n)
    spike = np.zeros(n)
    since = np.full(n, n, dtype=int)
    for i in np.flatnonzero(spike_count):
        amplitude = sm.lognormal(np.log(c['spike_median']), c['spike_logstd'], spike_count[i]).sum()
        end = min(n, i + c['spike_duration'])
        spike[i:end] += amplitude * np.exp(-np.arange(end-i) / c['spike_tau'])
    last = -n
    for i in range(n):
        if spike_count[i]:
            last = i
        since[i] = i-last
    drop_count = dr.poisson(c['drop_rate'] / 24, n)
    drop = np.ones(n)
    for i in np.flatnonzero(drop_count):
        for _ in range(drop_count[i]):
            duration = dm.integers(c['drop_duration'][0], c['drop_duration'][1]+1)
            drop[i:min(n,i+duration)] *= dm.uniform(c['drop_min'], c['drop_max'])
    noise = nr.normal(size=n) * (c['noise0'] + c['noise1'] * mean)
    y = np.maximum(0, mean * (1+spike) * drop + noise)
    return dict(t=t, y=y, trend=trend, daily=daily, weekly=weekly, mean=mean,
                spike=spike, drop=drop, noise=noise, spike_onset=spike_count > 0,
                drop_onset=drop_count > 0, spike_active=spike > 0,
                post_spike=(since >= 1) & (since <= 3), since_spike=since,
                hour=hour, weekday=weekday,
                despiked=np.maximum(0, mean * drop + noise))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='data')
    a = p.parse_args()
    seed_all(a.seed)
    environment(a.seed)
    c = yaml.safe_load(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / (Path(a.config).stem + f'_{a.seed}.npz'), **generate(c, a.seed))


if __name__ == '__main__':
    main()
