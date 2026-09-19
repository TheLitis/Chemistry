"""Run the optional dataframe-based model diagnostic in isolated dependencies."""
from __future__ import annotations
import importlib.util,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path

PACKAGES=('pandas==2.3.3','python-dateutil==2.9.0.post0','pytz==2025.2','tzdata==2025.2','six==1.17.0')

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
 if Path(sys.executable).resolve()!=py.resolve():
  return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
 repo=Path(__file__).resolve().parents[1];out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
 root=state/'artifacts/casmi26/baseline328-diagnostic/models';root.mkdir(parents=True,exist_ok=True)
 deps=state/'envs/casmi26-condition-check/deps';deps.mkdir(parents=True,exist_ok=True)
 clean={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','PASSWORD','SECRET','KAGGLE','GH_'))}
 clean.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
 marker=deps.parent/'installed.json'
 if not marker.is_file():
  cmd=[str(py),'-m','pip','install','--disable-pip-version-check','--no-deps','--only-binary=:all:','--target',str(deps),'--upgrade',*PACKAGES]
  r=subprocess.run(cmd,env=clean,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
  (out/'optional-deps.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
  if r.returncode:raise RuntimeError('Optional dependency setup failed; original environment untouched')
  marker.write_text(json.dumps({'packages':PACKAGES,'original_environment_changed':False})+'\n',encoding='utf-8')
 else:
  if json.loads(marker.read_text())['packages']!=list(PACKAGES):raise ValueError('Optional dependency identity changed')
 paths=[str(deps),str(repo/'tasks'),str(repo/'work/casmi26')]
 prefix='import sys;sys.path[:0]='+repr(paths)+';'
 checks=prefix+'import pandas,pytest;assert pandas.__version__=="2.3.3";raise SystemExit(pytest.main('+repr(['-q',str(repo/'work/casmi26/tests/test_conditioned_encoder.py'),str(repo/'work/casmi26/tests/test_conditioned_check.py')])+'))'
 r=subprocess.run([str(py),'-c',checks],env=clean,cwd=repo,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
 (out/'tests.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
 if r.returncode or 'skipped' in r.stdout:raise RuntimeError('Required model-check tests did not all run')
 check=load('check',repo/'tasks/casmi_conditioned_encoder_check.py')
 prep=load('prep',repo/'tasks/casmi_prepare.py');env=prep.kaggle_environment(state,dict(os.environ));env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
 identity={}
 for name,h in check.WEIGHTS.items():
  p=root/name
  if not p.exists():
   if shutil.disk_usage(root).free<2*check.BYTE_SIZE+1024**3:raise ValueError('Insufficient free space')
   cmd=[str(py),'-c','from kaggle.cli import main;main()','datasets','download','-d',check.REF,'-f',name,'-p',str(root),'-q']
   r=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
   if r.returncode:raise RuntimeError('Public checkpoint download failed: '+name)
   if not p.exists() and p.with_suffix(p.suffix+'.zip').is_file():
    with zipfile.ZipFile(p.with_suffix(p.suffix+'.zip')) as z:
     entries=[i for i in z.infolist() if i.filename==name and i.file_size==check.BYTE_SIZE]
     if len(entries)!=1:raise ValueError('Unexpected public checkpoint archive')
     p.write_bytes(z.read(entries[0]))
  if p.stat().st_size!=check.BYTE_SIZE or check.digest(p)!=h:raise ValueError('Checkpoint differs from the scored baseline')
  identity[name]={'sha256':h,'bytes':p.stat().st_size}
 check.dump(out/'checkpoint-identity.json',identity)
 code=prefix+'from pathlib import Path;from casmi_conditioned_encoder_check import execute;execute(*map(Path,'+repr([str(state),str(repo),str(out),str(root)])+'))'
 r=subprocess.run([str(py),'-c',code],env=clean,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=1200)
 (out/'model-check.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
 if r.returncode:raise RuntimeError('Isolated real-weight check failed; inspect model-check.log')
 return 0

if __name__=='__main__':raise SystemExit(main())
