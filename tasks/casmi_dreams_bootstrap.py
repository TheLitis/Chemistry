"""Pinned public DreaMS TorchScript acquisition and exact-peak embedding smoke.

This uses the author's current HF distribution (referenced by pinned source),
not the unavailable legacy Zenodo endpoint. It never changes the incumbent.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile

MODEL='DreaMS_embedding_model_torchscript.pt'
SETTINGS='DreaMS_embedding_model_torchscript_settings.json'
SOURCE_COMMIT='dbec3a0b514a99e5056cfccde4559fda8cfe8129'
HOSTS={'huggingface.co','cdn-lfs.huggingface.co','cdn-lfs-us-1.hf.co','cdn-lfs-eu-1.hf.co','cas-bridge.xethub.hf.co'}


def validate_url(url):
    p=urllib.parse.urlsplit(url)
    if p.scheme!='https' or p.hostname not in HOSTS or p.username or p.password or p.port not in (None,443):
        raise ValueError('Unapproved public model host')
    return url


def select_asset(meta):
    if meta.get('id') not in (None,'roman-bushuiev/DreaMS') or meta.get('cardData',{}).get('license')!='mit':
        raise ValueError('Wrong author model or missing MIT weight license')
    rev=meta.get('sha','')
    if not re.fullmatch('[0-9a-f]{40}',rev):raise ValueError('Immutable model revision missing')
    rows=[r for r in meta.get('siblings',[]) if r.get('rfilename')==MODEL]
    if len(rows)!=1:raise ValueError('TorchScript model missing or duplicated')
    lfs=rows[0].get('lfs',{});size=lfs.get('size');sha=lfs.get('sha256','')
    if not re.fullmatch('[0-9a-f]{64}',sha) or type(size) is not int or not 0<size<600*1024**2:
        raise ValueError('Publisher size/hash contract unavailable')
    return {'revision':rev,'sha256':sha,'bytes':size,'file':MODEL}


def pack_spectrum(peaks,precursor,n=100):
    import numpy as np
    p=np.asarray(peaks,dtype='f8')
    if p.ndim!=2 or p.shape[1]!=2 or not len(p) or not np.isfinite(p).all() or np.any(p[:,0]<=0) or np.any(p[:,1]<0) or p[:,1].max()<=0:
        raise ValueError('Invalid peaks')
    if not np.isfinite(precursor) or precursor<=0 or type(n) is not int or n<1:raise ValueError('Invalid precursor/budget')
    chosen=np.sort(np.argsort(p[:,1])[-n:]);p=p[chosen].copy();p[:,1]/=p[:,1].max()
    out=np.zeros((n+1,2),dtype='f4');out[0]=[precursor,1.1];out[1:len(p)+1]=p
    return out


def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def smoke(root,output,state):
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    identity=json.loads((root/'identity.json').read_text())
    if sha256(root/MODEL)!=identity['sha256']:raise ValueError('Model bytes changed')
    settings=json.loads((root/SETTINGS).read_text())
    print('PUBLISHED_MODEL_SETTINGS '+json.dumps(settings),flush=True)
    model=torch.jit.load(str(root/MODEL),map_location='cuda' if torch.cuda.is_available() else 'cpu').eval()
    param=next(model.parameters());device,dtype=param.device,param.dtype
    for key,expected in [('n_highest_peaks',100),('prec_intens',1.1),('normalize_mzs',False)]:
        if key in settings and settings[key]!=expected:raise ValueError('Unexpected published preprocessing: '+key)
    inv=json.loads((state/'artifacts/casmi26/research-r07/inventory.json').read_text());train=Path(inv['train_path'])
    rows=[]
    columns=['ingest_lib','adduct','precursor_mz','ms2_mzs','ms2_normalized_intensities']
    for b in pq.ParquetFile(train).iter_batches(batch_size=32768,columns=columns):
        for r in b.to_pylist():
            if r['ingest_lib']=='enveda-np-examples' and r['adduct']=='[M+H]+':
                rows.append(r)
                if len(rows)==64:break
        if len(rows)==64:break
    if len(rows)!=64:raise ValueError('Missing fixed training-spectrum smoke sample')
    inputs=np.stack([pack_spectrum(np.column_stack((r['ms2_mzs'],r['ms2_normalized_intensities'])),r['precursor_mz']) for r in rows])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    def infer(x,batch):
        result=[]
        with torch.inference_mode():
            for i in range(0,len(x),batch):
                value=model(torch.as_tensor(x[i:i+batch],device=device,dtype=dtype))
                if not isinstance(value,torch.Tensor):raise ValueError('Unexpected author model output')
                result.append(value.detach().cpu().float().numpy())
        return np.concatenate(result)
    started=time.monotonic();vectors=infer(inputs,4)
    if vectors.shape!=(64,1024) or not np.isfinite(vectors).all():raise ValueError('Invalid embeddings')
    one=infer(inputs[:4],1);delta=float(np.abs(one-vectors[:4]).max())
    if delta>1e-3:raise ValueError('Batch parity error')
    toys=np.stack([pack_spectrum([[50.125,1],[50.375,1]],200.5),pack_spectrum([[50.1875,1],[50.3125,1]],200.5)])
    toy=infer(toys,2)
    np.save(output/'smoke-embeddings.npy',vectors);np.save(output/'inputs.npy',inputs)
    report={'status':'author_torchscript_executed','identity':identity,'settings':settings,'device':str(device),
        'dtype':str(dtype),'parameters':sum(p.numel() for p in model.parameters()),'schema':str(model.forward.schema),
        'spectra':64,'embedding_shape':list(vectors.shape),'batch_parity_max_absolute_error':delta,
        'within_bin_synthetic_pair_embedding_l2':float(np.linalg.norm(toy[0]-toy[1])),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        'seconds_for_smoke_inference':time.monotonic()-started,'input_transform':'author SpectrumPreprocessor default reproduced',
        'test_spectra_read':False,'new_training':False,'new_submissions':0,'accuracy_established':False,
        'pretraining_overlap_with_CASMI_not_audited':True,'incumbent_changed':False}
    write(output/'smoke.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'tasks'))
    import casmi_foundation_probe as fetch
    from public_tls import verified_context
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root=state/'data/external/dreams-torchscript';root.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)>1 and sys.argv[1]=='_smoke':smoke(root,out,state);return 0
    result={'new_submissions':0,'new_training':False,'incumbent_changed':False,'source_commit':SOURCE_COMMIT}
    check=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests/test_dreams_bootstrap.py')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    (out/'tests.log').write_text(check.stdout+'\n'+check.stderr,encoding='utf-8')
    if check.returncode:raise RuntimeError('Bootstrap tests failed')
    class Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl):
            validate_url(newurl);return super().redirect_request(req,fp,code,msg,headers,newurl)
    opener=urllib.request.build_opener(Redirect(),urllib.request.HTTPSHandler(context=verified_context()))
    def download(url,dest,limit,expected=None):
        validate_url(url)
        if dest.exists() and expected and sha256(dest)==expected:return {'reused':True,'sha256':expected}
        temp=dest.with_suffix(dest.suffix+'.checked')
        with opener.open(urllib.request.Request(url,headers={'User-Agent':'CASMI-authorized-research/1.0'}),timeout=60) as r:
            validate_url(r.geturl());info=fetch.copy_public(r,temp,limit)
        if expected and info['sha256']!=expected:temp.unlink();raise ValueError('HF publisher SHA256 mismatch')
        os.replace(temp,dest);return info
    download('https://huggingface.co/api/models/roman-bushuiev/DreaMS?blobs=true',root/'metadata.json',2*1024**2)
    meta=json.loads((root/'metadata.json').read_text());identity=select_asset(meta);write(root/'identity.json',identity)
    base='https://huggingface.co/roman-bushuiev/DreaMS/resolve/'+identity['revision']+'/'
    result['asset']=download(base+MODEL,root/MODEL,identity['bytes'],identity['sha256'])
    result['settings']=download(base+SETTINGS,root/SETTINGS,128*1024)
    with zipfile.ZipFile(root/MODEL) as z:
        if sum(i.file_size for i in z.infolist())>800*1024**2:raise ValueError('Unexpected model expansion')
        code={i.filename:z.read(i).decode('utf-8') for i in z.infolist() if '/code/' in i.filename and i.filename.endswith('.py') and i.file_size<1024**2}
        if not code:raise ValueError('Not an inspectable TorchScript archive')
        (out/'torchscript-source.json').write_text(json.dumps(code,indent=2),encoding='utf-8')
    env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('TOKEN','PASSWORD','SECRET','KAGGLE','GH_'))}
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    run=subprocess.run([str(py),str(Path(__file__).resolve()),'_smoke'],env=env,stdin=subprocess.DEVNULL,
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=900)
    (out/'smoke.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8');result['smoke_exit_code']=run.returncode
    result['checked_utc']=dt.datetime.now(dt.timezone.utc).isoformat();result['identity']=identity
    for n in ('metadata.json','identity.json',SETTINGS):
        (out/n).write_bytes((root/n).read_bytes())
    write(out/'bootstrap.json',result);write(root/'bootstrap.json',result)
    if run.returncode:raise RuntimeError('Public model acquired; isolated smoke failed, see smoke.log')
    return 0


if __name__=='__main__':raise SystemExit(main())
