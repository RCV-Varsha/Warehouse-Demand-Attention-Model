import argparse
import shutil
from wdam.utils import ROOT,seed_all


def clean():
    root=ROOT.resolve()
    for name in ['data','results']:
        target=(root/name).resolve()
        if target.parent!=root or target.name!=name or target.is_symlink():
            raise RuntimeError(f'Unsafe generated directory: {target}')
        if target.exists(): shutil.rmtree(target)


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['clean','reset']); p.add_argument('--seed',type=int,default=0)
    a=p.parse_args(); seed_all(a.seed); clean()
    if a.action=='reset':
        from experiments.run_all import run
        run(a.seed)


if __name__=='__main__': main()
