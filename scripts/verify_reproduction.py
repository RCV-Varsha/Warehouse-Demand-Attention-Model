import hashlib
import subprocess
import sys
from wdam.utils import ROOT,cli,save


def run(seed):
    subprocess.run([sys.executable,'-m','scripts.manage','reset','--seed',str(seed)],check=True,cwd=ROOT)
    first=(ROOT/'results/summary.json').read_bytes()
    subprocess.run([sys.executable,'-m','experiments.run_all','--seed',str(seed)],check=True,cwd=ROOT)
    second=(ROOT/'results/summary.json').read_bytes()
    save('reproduction.json',dict(seed=seed,identical=first==second,
         first_sha256=hashlib.sha256(first).hexdigest(),second_sha256=hashlib.sha256(second).hexdigest(),
         commands=['python -m scripts.manage reset --seed '+str(seed),'python -m experiments.run_all --seed '+str(seed)]))
    assert first==second,'Deterministic summary changed!'


if __name__=='__main__': cli(run)
