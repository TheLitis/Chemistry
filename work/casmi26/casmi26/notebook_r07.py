"""Build an offline Kaggle notebook containing versioned R07 inference source."""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import zipfile


def build_notebook(path):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in sorted(Path(__file__).parent.glob('*.py')):
            member=zipfile.ZipInfo('casmi26/'+file.name,date_time=(2026,9,17,0,0,0))
            member.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(member,file.read_bytes().replace(b'\r\n',b'\n'))
    cells=[]
    def code(text):
        cells.append({'cell_type':'code','id':'code-'+str(len(cells)),'metadata':{},'source':text.splitlines(True),'execution_count':None,'outputs':[]})
    cells.append({'cell_type':'markdown','id':'intro','metadata':{},'source':['# CASMI26 R07\n','Fixed timsTOF-calibrated hybrid with full-corpus refit and a complete offline COCONUT snapshot. Catalog retrieval, not de novo generation. Inputs and IDs are discovered from the current competition mount.\n']})
    code('SOURCE_B64 = '+repr(base64.b64encode(buf.getvalue()).decode())+'\n')
    code("""from pathlib import Path
import os,sys,io,json,zipfile,base64,subprocess
INPUT=Path(os.environ.get('CASMI_INPUT_ROOT','/kaggle/input'))
WORK=Path(os.environ.get('CASMI_WORK_ROOT','/kaggle/working'));WORK.mkdir(parents=True,exist_ok=True)
mounts=[p.parent for p in INPUT.rglob('test.parquet') if (p.parent/'train.parquet').is_file()]
if len(mounts)!=1:raise RuntimeError('Expected a single competition data mount')
DATA=mounts[0]
manifests=list(INPUT.rglob('r07-bundle.json'))
if len(manifests)!=1:raise RuntimeError('Expected a single R07 model bundle')
BUNDLE=manifests[0].parent
# Some dataset services unpack .zip inputs. Transport the unchanged ZIP bytes
# under an opaque extension and restore the exact manifest name locally.
if not (BUNDLE/'coconut.zip').is_file() and (BUNDLE/'coconut.snapshot').is_file():
    import shutil
    restored=WORK/'r07_bundle';restored.mkdir(exist_ok=True)
    for name in ('r07-bundle.json','model.npz','catalog.json','fingerprints.npy'):
        shutil.copyfile(BUNDLE/name,restored/name)
    shutil.copyfile(BUNDLE/'coconut.snapshot',restored/'coconut.zip')
    BUNDLE=restored
CODE=WORK/'r07_code';CODE.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as z:
    for item in z.infolist():
        p=Path(item.filename)
        if p.is_absolute() or '..' in p.parts or p.parts[0]!='casmi26' or p.suffix!='.py':
            raise RuntimeError('Unsafe source archive member')
    z.extractall(CODE)
DEPS=WORK/'r07_deps';DEPS.mkdir(exist_ok=True)
if sys.platform.startswith('linux'):
    wheels=sorted({p.parent for p in INPUT.rglob('*.whl')})
    if not wheels:raise RuntimeError('Pinned offline dependency wheels missing')
    command=[sys.executable,'-m','pip','install','--no-index','--target',str(DEPS),'--upgrade']
    for folder in wheels:command+=['--find-links',str(folder)]
    command+=['numpy==2.3.5','rdkit==2026.3.3','pyarrow==21.0.0']
    subprocess.run(command,check=True,capture_output=True,text=True)
env=dict(os.environ,PYTHONPATH=str(DEPS)+os.pathsep+str(CODE),PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
subprocess.run([sys.executable,'-c',"import rdkit,numpy,pyarrow;assert rdkit.__version__=='2026.03.3';assert numpy.__version__=='2.3.5';assert pyarrow.__version__=='21.0.0'"],check=True,env=env)
""")
    code("""launcher=WORK/'predict_r07.py'
launcher.write_text("from casmi26.r07_release import main\\nif __name__=='__main__':\\n    raise SystemExit(main())\\n",encoding='utf-8')
output=WORK/'submission.csv'
command=[sys.executable,str(launcher),'--test',str(DATA/'test.parquet'),'--train',str(DATA/'train.parquet'),'--bundle',str(BUNDLE),'--output',str(output),'--workers','4']
if (DATA/'sample_submission.csv').is_file():command+=['--sample-submission',str(DATA/'sample_submission.csv')]
run=subprocess.run(command,check=True,env=env,capture_output=True,text=True)
print(run.stdout[-6000:])
report=json.loads((WORK/'submission.csv.report.json').read_text())
assert report['format']==7 and report['test_labels_used'] is False
assert report['empty_candidate_rows']==[]
print(json.dumps({k:report[k] for k in ('prediction_count','test_spectra','submission_sha256','mass_incompatible_fallbacks')},indent=2))
""")
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},'cells':cells}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return path
