import argparse
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def config(name):
    return yaml.safe_load((ROOT / 'configs' / name).read_text())


def save(name, value):
    path = ROOT / 'results' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def environment(seed):
    save('environment.json', dict(python=platform.python_version(), torch=torch.__version__,
         numpy=np.__version__, os=platform.platform(), seed=seed, device='cpu', threads=1))


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'configs').glob('*.yaml'))}


def cli(fn):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    seed_all(args.seed)
    environment(args.seed)
    fn(args.seed)
