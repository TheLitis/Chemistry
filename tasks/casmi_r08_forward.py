"""Scoped pinned forward-model execution; no Kaggle writes and no CASMI refitting."""
from __future__ import annotations
import argparse,datetime as dt,gzip,hashlib,importlib.util,json,os,shutil,sqlite3,subprocess,sys,time,zipfile,zlib
from pathlib import Path

SOURCE='e19ef82c9a6cb9dbac92bce23e914008f1aeb44e'
ARCHIVE_HASH='5fde1a6c7cf19e99aa655ae83d793292fe9e9c8feca2c499bbffe028fdce7538'


def setup(state,repo,root,out):
    from casmi26.production import sha256,write_json
    material=state/'artifacts/casmi26/public-model-probe-20260917'
    archive=material/'fiora.zip';source=root/('fiora-'+SOURCE);deps=root/'deps';deps.mkdir(parents=True,exist_ok=True)
    if sha256(archive)!=ARCHIVE_HASH:raise RuntimeError('Pinned public source archive changed')
    with zipfile.ZipFile(archive) as z:
        selected=[i for i in z.infolist() if not i.is_dir() and (i.filename.endswith(('.py','_state.pt','_params.json','.csv','.mgf','LICENSE')))]
        for i in selected:
            p=Path(i.filename)
            if p.is_absolute() or '..' in p.parts or ':' in i.filename or p.parts[0]!='fiora-'+SOURCE:
                raise ValueError('Unsafe public archive member')
            dest=root/p
            body=z.read(i)
            if dest.exists() and dest.read_bytes()!=body:raise RuntimeError('Existing third-party source changed')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(body)
    base=env_clean();base['PYTHONPATH']=os.pathsep.join([str(deps),str(source),str(repo/'work/casmi26')])
    wheels=material/'wheels'
    pure=sorted(wheels.glob('*.whl'))
    if len(pure)!=4:raise RuntimeError('Expected four previously staged public wheels')
    wheel_records=json.loads((material/'probe.json').read_text())['public_wheel_download']['files']
    hashes={r['name']:r['sha256'] for r in wheel_records}
    if any(hashes.get(p.name)!=sha256(p) for p in pure):raise RuntimeError('Public dependency wheel hash mismatch')
    install=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(deps),'--upgrade']+[str(p) for p in pure],
        env=base,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    (out/'pure-deps.log').write_text(install.stdout+'\n'+install.stderr,encoding='utf-8')
    if install.returncode:raise RuntimeError('Isolated dependency staging failed')
    allowed={
      'pandas':'pandas==2.3.3','matplotlib':'matplotlib==3.10.8','IPython':'ipython==9.9.0',
      'scipy':'scipy==1.17.0','sklearn':'scikit-learn==1.8.0','numba':'numba==0.63.1',
      'regex':'regex==2025.11.3','requests':'requests==2.32.5','tqdm':'tqdm==4.67.1',
      'networkx':'networkx==3.6.1','psutil':'psutil==7.2.1','jinja2':'Jinja2==3.1.6',
      'pyparsing':'pyparsing==3.3.2','aiohttp':'aiohttp==3.13.3'}
    probe=subprocess.run([sys.executable,'-c','import importlib.util,json;print(json.dumps([m for m in '+repr(list(allowed))+' if importlib.util.find_spec(m) is None]))'],
        env=base,capture_output=True,text=True,timeout=30,check=True)
    missing=json.loads(probe.stdout)
    if missing:
        constraints=root/'protected-versions.txt'
        constraints.write_text('numpy==2.3.5\nrdkit==2026.3.3\npyarrow==21.0.0\n',encoding='utf-8')
        r=subprocess.run([sys.executable,'-m','pip','install','--only-binary=:all:','--target',str(deps),
             '--constraint',str(constraints)]+[allowed[m] for m in missing],env=base,stdin=subprocess.DEVNULL,
             capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
        (out/'extra-deps.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
        if r.returncode:raise RuntimeError('Isolated additional dependencies failed')
    imports="import torch,numpy,rdkit,pyarrow,pandas;from fiora.cli.predict import build_metabolites;assert numpy.__version__=='2.3.5';assert rdkit.__version__=='2026.03.3';assert pyarrow.__version__=='21.0.0';print(torch.__version__)"
    r=subprocess.run([sys.executable,'-c',imports],env=base,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=90)
    (out/'imports.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
    if r.returncode:raise RuntimeError('Pinned inference environment does not import; no model executed')
    identity={'source_commit':SOURCE,'source_archive_sha256':ARCHIVE_HASH,'extra_packages_requested':[allowed[m] for m in missing],
              'base_environment_modified':False,'third_party_pickle_loaded':False,'torch':r.stdout.strip()}
    write_json(root/'environment.json',identity)
    return base


def env_clean():
    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','SECRET','PASSWORD','KAGGLE_KEY','GH_TOKEN'))}
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MPLBACKEND='Agg',PYTHONWARNINGS='ignore')
    return env


def execute(root,out,device):
    import numpy as np,pandas as pd,torch,rdkit
    from casmi26.production import sha256,write_json
    from casmi26.fiora_adapter import ForwardModel,MODEL_HASH
    from casmi26.forward_ranking import ENERGIES
    source=root/('fiora-'+SOURCE)
    model_path=source/'fiora/resources/models/fiora_OS_v1.0.0.pt'
    torch.set_num_threads(4)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ForwardModel(model_path,device=device)
    examples=pd.read_csv(source/'examples/example_input.csv')
    maxerror=0.;conditions=0;timings={}
    for cached in (False,True):
        started=time.monotonic();outputs=[]
        for row in examples.to_dict('records'):
            predictions=model.predict(row['SMILES'],[row['Precursor_type']],ENERGIES,cache_graph=cached)
            outputs.append(predictions);conditions+=len(predictions)
        if not cached:reference=outputs
        else:
            for a,b in zip(reference,outputs):
                for condition in a:
                    if a[condition].shape!=b[condition].shape:raise RuntimeError('Cached simulation changed peak count')
                    err=float(np.max(abs(a[condition]-b[condition]),initial=0));maxerror=max(maxerror,err)
                    if not np.allclose(a[condition],b[condition],rtol=2e-6,atol=2e-7):raise RuntimeError('Cached simulation failed author-model parity')
        timings['cached' if cached else 'full']=time.monotonic()-started
    parity={'max_absolute_error':maxerror,'conditions_each':conditions//2,'seconds':timings,
            'speedup':timings['full']/timings['cached'],'device':device,'torch':torch.__version__,'rdkit':rdkit.__version__}
    write_json(root/'graph-parity.json',parity);write_json(out/'graph-parity.json',parity)
    prepared=json.loads((root/'prepared.json').read_text())
    for name,digest in prepared['files'].items():
        if sha256(root/name)!=digest:raise RuntimeError('Prepared candidate inputs changed')
    requests=json.loads((root/'requests.json').read_text())
    identity={'source':SOURCE,'model':MODEL_HASH,'requests':sha256(root/'requests.json'),'energies':list(ENERGIES),
              'torch':torch.__version__,'rdkit':rdkit.__version__,'device':device,'adapter':sha256(Path(__file__).resolve().parents[1]/'work/casmi26/casmi26/fiora_adapter.py')}
    db=sqlite3.connect(root/'simulations.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS predictions(k TEXT PRIMARY KEY,content BLOB NOT NULL)')
    signature=json.dumps(identity,sort_keys=True);prev=db.execute('SELECT v FROM meta WHERE k="signature"').fetchone()
    if prev and prev[0]!=signature:raise RuntimeError('Cannot mix forward runs from different model/source/environment')
    if not prev:db.execute('INSERT INTO meta VALUES(?,?)',('signature',signature));db.commit()
    existing={r[0] for r in db.execute('SELECT k FROM predictions')};started=time.monotonic();created=0
    for request in requests:
        key=request['key']
        if key in existing:continue
        record={'key':key,'smiles':request['smiles'],'modes':request['modes']}
        try:
            if request['modes']:
                spectra=model.predict(request['smiles'],request['modes'],ENERGIES)
                record['spectra']=[{'adduct':m,'energy':e,'peaks':p.tolist()} for (m,e),p in spectra.items()]
                record['status']='predicted'
            else:record.update(spectra=[],status='no_supported_modes')
        except (ValueError,RuntimeError,AssertionError,IndexError,KeyError) as exc:
            record.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:300],spectra=[])
        db.execute('INSERT INTO predictions VALUES(?,?)',(key,zlib.compress(json.dumps(record,allow_nan=False).encode(),1)));db.commit();created+=1
        if created%100==0:print('R08_SIMULATED '+str(created+len(existing))+'/'+str(len(requests)),flush=True)
    statuses={};conditions=0
    for (blob,) in db.execute('SELECT content FROM predictions'):
        r=json.loads(zlib.decompress(blob));statuses[r['status']]=statuses.get(r['status'],0)+1;conditions+=len(r['spectra'])
    db.close()
    result={'status':'simulation_completed','created_this_pass':created,'structures':len(requests),'statuses':statuses,
            'spectra':conditions,'seconds':time.monotonic()-started,'identity':identity,'parity':parity,'new_submissions':0}
    write_json(root/'simulation.json',result);write_json(out/'simulation.json',result)
    print('R08_SIMULATION_BEGIN\n'+json.dumps(result,indent=2)+'\nR08_SIMULATION_END',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--configured',action='store_true');parser.add_argument('--device',default='auto');a=parser.parse_args()
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env=env_clean())
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    root=state/'artifacts/casmi26/research-r08/forward-v1';out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    if not (root/'prepared.json').is_file():raise RuntimeError('Frozen R08 preparation not complete')
    if not a.configured:
        env=setup(state,repo,root,out)
        env['FIORA_TEST_MODEL']=str(root/('fiora-'+SOURCE)/'fiora/resources/models/fiora_OS_v1.0.0.pt')
        tests=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=360)
        (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8')
        if tests.returncode:raise RuntimeError('Forward tests failed before simulation')
        return subprocess.call([str(py),str(Path(__file__).resolve()),'--configured','--device',a.device],env=env)
    execute(root,out,a.device);return 0


if __name__=='__main__':raise SystemExit(main())
