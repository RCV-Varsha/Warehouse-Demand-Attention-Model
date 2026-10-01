import hashlib
import subprocess
import sys
import os
import shutil
from wdam.utils import ROOT,cli,save


def run(seed):
    env=dict(os.environ)
    env['PATH']=str(ROOT/'.venv'/('Scripts' if os.name=='nt' else 'bin'))+os.pathsep+env.get('PATH','')
    make=os.environ.get('WDAM_MAKE') or shutil.which('make',path=env['PATH']) or shutil.which('gmake',path=env['PATH'])
    commands=([[make,'reset','SEED='+str(seed)],[make,'all','SEED='+str(seed)]] if make else
              [[sys.executable,'-m','scripts.manage','reset','--seed',str(seed)],
               [sys.executable,'-m','experiments.run_all','--seed',str(seed)]])
    subprocess.run(commands[0],check=True,cwd=ROOT,env=env)
    first=(ROOT/'results/summary.json').read_bytes()
    subprocess.run(commands[1],check=True,cwd=ROOT,env=env)
    second=(ROOT/'results/summary.json').read_bytes()
    save('reproduction.json',dict(seed=seed,identical=first==second,
         first_sha256=hashlib.sha256(first).hexdigest(),second_sha256=hashlib.sha256(second).hexdigest(),
         commands=commands,gnu_make=bool(make)))
    assert first==second,'Deterministic summary changed!'


if __name__=='__main__': cli(run)
