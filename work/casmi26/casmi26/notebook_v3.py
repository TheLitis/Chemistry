"""Self-contained, no-network inference notebook for v3 model assets."""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import zipfile


def build_notebook(output):
    buffer=io.BytesIO();package=Path(__file__).resolve().parent
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(package.glob('*.py')):z.writestr('casmi26/'+path.name,path.read_bytes())
    payload=base64.b64encode(buffer.getvalue()).decode()
    source="""from pathlib import Path
import base64, io, json, os, subprocess, sys, tempfile, zipfile
INPUT = Path(os.environ.get('CASMI_INPUT_ROOT', '/kaggle/input'))
WORK = Path(os.environ.get('CASMI_WORK_ROOT', '/kaggle/working'))
WORK.mkdir(parents=True, exist_ok=True)
matches = [p.parent for p in INPUT.rglob('test.parquet') if (p.parent/'train.parquet').is_file()]
if len(matches) != 1: raise RuntimeError('Attach exactly one competition dataset')
DATA = matches[0]
if os.environ.get('CASMI_BUNDLE_DIR'):
    ASSETS = Path(os.environ['CASMI_BUNDLE_DIR'])
else:
    bundles = list(INPUT.rglob('v3-bundle.json'))
    if len(bundles) != 1: raise RuntimeError('Attach exactly one v3 bundle')
    ASSETS = bundles[0].parent
"""
    source+='SOURCE = '+repr(payload)+'\n'
    source+="""with tempfile.TemporaryDirectory(prefix='casmi-v3-') as tmp:
    TEMP = Path(tmp); CODE = TEMP/'code'; CODE.mkdir(); DEPS = TEMP/'deps'; DEPS.mkdir()
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE))) as z:
        for member in z.infolist():
            p = Path(member.filename)
            if p.is_absolute() or '..' in p.parts or p.parts[0]!='casmi26' or p.suffix!='.py':
                raise RuntimeError('Invalid source member')
        z.extractall(CODE)
    if sys.platform.startswith('linux'):
        wheels = ASSETS/'wheels' if (ASSETS/'wheels').is_dir() else ASSETS
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--find-links',str(wheels),
            '--target',str(DEPS),'numpy==2.3.5','rdkit==2026.3.3','pyarrow==21.0.0'],check=True)
    launcher = CODE/'launch.py'
    launcher.write_text('import sys\\nsys.path[:0]='+repr([str(CODE),str(DEPS)])+
                       '\\nfrom casmi26.inference_v3 import main\\nraise SystemExit(main())\\n')
    args = [sys.executable,str(launcher),'--test',str(DATA/'test.parquet'),'--train',str(DATA/'train.parquet'),
            '--bundle',str(ASSETS),'--output',str(WORK/'submission.csv')]
    if (DATA/'sample_submission.csv').is_file(): args+=['--sample-submission',str(DATA/'sample_submission.csv')]
    env = dict(os.environ); env['OPENBLAS_NUM_THREADS']='4'; env['OMP_NUM_THREADS']='4'
    subprocess.run(args,check=True,env=env)
report = json.loads((WORK/'submission.csv.report.json').read_text())
print({k:v for k,v in report.items() if k!='details'})
"""
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
      'cells':[{'cell_type':'markdown','id':'description','metadata':{},'source':['# CASMI26 v3: fractional masses + complementary fingerprints\n',
        'Private train-only assets; Internet disabled. Predictions use the current hidden-rerun mount. This is a catalog model, not a score guarantee.']},
       {'cell_type':'code','id':'predict','metadata':{},'source':source.splitlines(True),'outputs':[],'execution_count':None}]}
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<notebook>','exec')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
    return output
