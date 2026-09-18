"""Offline notebook for the frozen R07 plus R08B forward-ranking candidate."""
from __future__ import annotations
import json
from pathlib import Path
import re


def build_notebook(path, *, manifest_sha256):
    if not isinstance(manifest_sha256,str) or not re.fullmatch(r'[0-9a-f]{64}',manifest_sha256):
        raise ValueError('A verified asset-manifest SHA256 is required')
    cells=[]
    def code(text):
        cells.append({'cell_type':'code','id':'code-'+str(len(cells)), 'metadata':{},
                      'source':text.splitlines(True),'execution_count':None,'outputs':[]})
    cells.append({'cell_type':'markdown','id':'intro','metadata':{},'source':[
        '# CASMI26 R08B: fixed forward-assisted R07\n',
        'Unchanged R07 full-refit weights and complete COCONUT snapshot; pinned FIORA evidence contributes at most 0.25. '
        'Exact numerical top-25 screening, not chemical certainty or de novo generation. No hidden answers or cached test predictions.\n']})
    code('EXPECTED_ASSET_MANIFEST = '+repr(manifest_sha256)+'\n')
    code("""from pathlib import Path
import os,sys,io,json,hashlib,zipfile,subprocess,shutil
INPUT=Path(os.environ.get('CASMI_INPUT_ROOT','/kaggle/input'))
WORK=Path(os.environ.get('CASMI_WORK_ROOT','/kaggle/working'));WORK.mkdir(parents=True,exist_ok=True)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
assets=list(INPUT.rglob('forward-assets.json'))
if len(assets)!=1 or digest(assets[0])!=EXPECTED_ASSET_MANIFEST:raise RuntimeError('Frozen forward asset manifest missing or modified')
ASSETS=assets[0].parent;manifest=json.loads(assets[0].read_text())
if manifest.get('format')!=8 or manifest.get('contains_test_ids_or_predictions') is not False:raise RuntimeError('Wrong forward asset contract')
for name,h in manifest['files'].items():
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or not (ASSETS/p).is_file() or digest(ASSETS/p)!=h:raise RuntimeError('Asset integrity failure: '+name)
mounts=[p.parent for p in INPUT.rglob('test.parquet') if (p.parent/'train.parquet').is_file()]
if len(mounts)!=1:raise RuntimeError('Expected one current competition mount')
DATA=mounts[0]
models=list(INPUT.rglob('r07-bundle.json'))
if len(models)!=1 or digest(models[0])!=manifest['base_bundle_manifest_sha256']:raise RuntimeError('Wrong frozen R07 bundle')
BUNDLE=models[0].parent
if not (BUNDLE/'coconut.zip').is_file() and (BUNDLE/'coconut.snapshot').is_file():
    restored=WORK/'r07_base';restored.mkdir(exist_ok=True)
    for name in ('r07-bundle.json','model.npz','catalog.json','fingerprints.npy'):shutil.copyfile(BUNDLE/name,restored/name)
    shutil.copyfile(BUNDLE/'coconut.snapshot',restored/'coconut.zip');BUNDLE=restored
CODE=WORK/'forward_source';CODE.mkdir(exist_ok=True)
with zipfile.ZipFile(ASSETS/'forward.snapshot') as z:
    for info in z.infolist():
        p=Path(info.filename)
        if p.is_absolute() or '..' in p.parts or info.is_dir():raise RuntimeError('Unsafe source archive path')
        if p.parts[0] not in ('casmi26','fiora') and info.filename!='FIORA-LICENSE':raise RuntimeError('Unexpected source member')
        if not (p.suffix=='.py' or p.name.endswith(('_state.pt','_params.json')) or p.name=='FIORA-LICENSE'):raise RuntimeError('Unexpected source type')
    z.extractall(CODE)
for name,h in manifest['source_files'].items():
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or digest(CODE/p)!=h:raise RuntimeError('Extracted source differs')
""")
    code("""DEPS=WORK/'forward_deps';DEPS.mkdir(exist_ok=True)
if not sys.platform.startswith('linux'):raise RuntimeError('This preview is pinned to the Linux notebook image')
python_tag='cp'+str(sys.version_info.major)+str(sys.version_info.minor)
wheel_snapshot=ASSETS/('wheels-'+python_tag+'.snapshot')
if not wheel_snapshot.is_file() or python_tag not in manifest['wheel_files']:raise RuntimeError('Unsupported Python runtime '+python_tag)
WHEELS=WORK/'forward_wheels'/python_tag;WHEELS.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(wheel_snapshot) as z:
    if set(z.namelist())!=set(manifest['wheel_files'][python_tag]):raise RuntimeError('Wheel manifest differs')
    for name in z.namelist():
        if Path(name).name!=name or not name.endswith('.whl'):raise RuntimeError('Unsafe wheel member')
    z.extractall(WHEELS)
for name,h in manifest['wheel_files'][python_tag].items():
    if digest(WHEELS/name)!=h:raise RuntimeError('Extracted dependency hash mismatch')
base_wheels=sorted({p.parent for p in INPUT.rglob('*.whl')})
command=[sys.executable,'-m','pip','install','--no-index','--target',str(DEPS),'--upgrade']
for folder in base_wheels:command+=['--find-links',str(folder)]
command+=['numpy==2.3.5','rdkit==2026.3.3','pyarrow==21.0.0']
subprocess.run(command,check=True,capture_output=True,text=True)
# Install the already-resolved, hashed Linux wheel closure without network/dependency resolution.
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(DEPS),'--upgrade']+
               [str(p) for p in sorted(WHEELS.glob('*.whl'))],check=True,capture_output=True,text=True)
env={k:v for k,v in os.environ.items() if not any(word in k.upper() for word in ('TOKEN','SECRET','PASSWORD','KAGGLE_KEY'))}
env.update(PYTHONPATH=str(DEPS)+os.pathsep+str(CODE),PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
           OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MPLBACKEND='Agg',PYTHONWARNINGS='ignore')
check="import torch,rdkit,numpy,pyarrow;from fiora.cli.predict import build_metabolites;assert rdkit.__version__=='2026.03.3';assert numpy.__version__=='2.3.5';assert pyarrow.__version__=='21.0.0';print(torch.__version__)"
version=subprocess.run([sys.executable,'-c',check],check=True,env=env,capture_output=True,text=True)
print('TORCH_RUNTIME',version.stdout.strip())
""")
    code("""launcher=WORK/'predict_forward.py'
launcher.write_text("from casmi26.forward_release import main\\nif __name__=='__main__':\\n    raise SystemExit(main())\\n",encoding='utf-8')
output=WORK/'submission.csv'
command=[sys.executable,str(launcher),'--test',str(DATA/'test.parquet'),'--train',str(DATA/'train.parquet'),
         '--bundle',str(BUNDLE),'--output',str(output),'--model-path',str(CODE/'fiora/resources/models/fiora_OS_v1.0.0.pt'),
         '--cache',str(WORK/'forward-cache.sqlite'),'--device','cpu','--workers','4']
if (DATA/'sample_submission.csv').is_file():command+=['--sample-submission',str(DATA/'sample_submission.csv')]
r=subprocess.run(command,check=True,env=env,capture_output=True,text=True)
print(r.stdout[-8000:]);print(r.stderr[-1500:])
report=json.loads((WORK/'submission.csv.report.json').read_text())
assert report['format']==8 and report['test_labels_used'] is False and report['empty_candidate_rows']==[]
assert report['forward']['all_top25_numerically_certified'] is True
assert report['forward']['weight']==0.25 and report['forward']['feature']=='cosine_nearest'
assert digest(output)==report['submission_sha256']
print(json.dumps({k:report[k] for k in ('prediction_count','test_spectra','submission_sha256','forward','seconds_total')},indent=2))
""")
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
            'language_info':{'name':'python'}},'cells':cells}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(book,indent=1)+'\n',encoding='utf-8');return path
