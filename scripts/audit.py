import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from wdam.utils import ROOT,cli,save


def run(seed):
    (ROOT/'tmp').mkdir(exist_ok=True)
    result=subprocess.run([sys.executable,'-m','pytest','-q','--junitxml=tmp/tests.xml'],cwd=ROOT,capture_output=True,text=True)
    forbidden=re.compile(r'MultiheadAttention|\bTransformer\w*|scaled_dot_product_attention|(?:torch|F)\.softmax|huggingface|einops',re.I)
    matches=[]
    for path in (ROOT/'src').rglob('*.py'):
        for line,text in enumerate(path.read_text().splitlines(),1):
            if forbidden.search(text): matches.append(dict(path=str(path.relative_to(ROOT)),line=line,text=text))
    all_json=list((ROOT/'results').glob('*.json'))
    for p in all_json: json.loads(p.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
    mutations=[]
    if (ROOT/'tmp/tests.xml').exists():
        tree=ET.parse(ROOT/'tmp/tests.xml')
        mutations=[dict(name=e.attrib['name'],passed=not any(c.tag in ['failure','error','skipped'] for c in e)) for e in tree.iter('testcase') if 'mutation' in e.attrib['name']]
    evidence=dict(seed=seed,pytest_returncode=result.returncode,pytest_output=result.stdout,
                  forbidden_matches=matches,mutation_tests=mutations,json_files_checked=len(all_json))
    save('verification.json',evidence)
    print(result.stdout)
    assert result.returncode==0,result.stderr
    assert not matches
    assert len(mutations)==3 and all(m['passed'] for m in mutations)


if __name__=='__main__': cli(run)
