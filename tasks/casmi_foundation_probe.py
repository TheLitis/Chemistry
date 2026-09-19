"""Public SSL-model acquisition and current competition inventory; no scoring writes.

Only a published, explicitly permissively licensed SSL backbone is selected.
No downloaded code or checkpoint is executed, and no environment is installed.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

DREAMS_COMMIT='dbec3a0b514a99e5056cfccde4559fda8cfe8129'
SLUG='enveda-CASMI26-molecule-id-mass-spectra'
HOSTS={'zenodo.org','www.zenodo.org','codeload.github.com','raw.githubusercontent.com'}
ALLOWED_LICENSES={'cc-by-4.0','cc0-1.0','mit','apache-2.0'}


def check_url(url):
    p=urllib.parse.urlsplit(url)
    if p.scheme!='https' or p.hostname not in HOSTS or p.username or p.password or p.port not in (None,443):
        raise ValueError('Outside permitted public-download hosts')
    return url


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        check_url(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def ssl_entry(meta):
    license_id=meta.get('metadata',{}).get('license',{}).get('id')
    if license_id not in ALLOWED_LICENSES:raise ValueError('SSL weights lack an accepted permissive license')
    rows=[r for r in meta.get('files',[]) if r.get('key')=='ssl_model.ckpt']
    if len(rows)!=1:raise ValueError('Exactly one documented SSL backbone required')
    r=rows[0];size=r.get('size');checksum=r.get('checksum','')
    if type(size) is not int or not 0<size<=3*1024**3:raise ValueError('Unknown or excessive model size')
    if not checksum.startswith('md5:') or len(checksum)!=36:raise ValueError('Publisher checksum missing')
    int(checksum[4:],16);check_url(r.get('links',{}).get('self',''));return r


def copy_public(stream,destination,limit,expected=None):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=destination.name+'.',suffix='.part',dir=destination.parent)
    size=0;sha=hashlib.sha256();md5=hashlib.md5();start=time.monotonic()
    try:
        with os.fdopen(fd,'wb') as f:
            while True:
                if time.monotonic()-start>1500:raise TimeoutError('Bounded acquisition exceeded 25 minutes')
                block=stream.read(min(4*1024**2,limit-size+1))
                if not block:break
                size+=len(block)
                if size>limit:raise ValueError('Download exceeds byte budget')
                f.write(block);sha.update(block);md5.update(block)
        if expected is not None and expected!='md5:'+md5.hexdigest():raise ValueError('Publisher checksum mismatch')
        os.replace(tmp,destination)
        return {'bytes':size,'sha256':sha.hexdigest(),'md5':md5.hexdigest(),'seconds':time.monotonic()-start}
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def parse_public_result(text):
    lines=[s for s in text.splitlines() if s.strip()]
    if not lines:raise ValueError('Empty public API response')
    value=json.loads(lines[-1])
    if not isinstance(value,dict):raise ValueError('Expected final JSON object')
    return value


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root=state/'data/external/dreams-ssl'/DREAMS_COMMIT;root.mkdir(parents=True,exist_ok=True)
    report={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'commit':os.environ.get('GITHUB_SHA'),
        'downloaded_code_executed':False,'checkpoint_executed':False,'new_submissions':0,'new_training':False,
        'installed_environment_changed':False,'incumbent_changed':False,'root':str(root),'downloads':{}}
    test=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests/test_foundation_probe.py')],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    (out/'tests.log').write_text(test.stdout+'\n'+test.stderr,encoding='utf-8')
    if test.returncode:raise RuntimeError('Acquisition contract tests failed')
    report['tests']=test.stdout.strip()
    tls=load('public_tls',repo/'tasks/public_tls.py')
    opener=urllib.request.build_opener(PublicRedirect(),urllib.request.HTTPSHandler(context=tls.verified_context()))
    def get(name,url,limit,expected=None):
        check_url(url);dest=root/name
        if dest.exists():
            with dest.open('rb') as f:
                h=hashlib.file_digest(f,'md5').hexdigest()
            if expected is not None and expected!='md5:'+h:raise ValueError('Existing public file checksum differs')
            return {'reused':True,'bytes':dest.stat().st_size,'md5':h,'path':str(dest)}
        print('PUBLIC_ACQUIRE '+name,flush=True)
        req=urllib.request.Request(url,headers={'User-Agent':'CASMI-authorized-research/1.0'})
        with opener.open(req,timeout=60) as r:
            check_url(r.geturl());declared=r.headers.get('Content-Length')
            if declared and int(declared)>limit:raise ValueError('Declared public content exceeds budget')
            info=copy_public(r,dest,limit,expected)
        return {**info,'path':str(dest)}
    prep=load('prep',repo/'tasks/casmi_prepare.py');shared=load('shared',repo/'tasks/casmi_r07_submission.py')
    env=prep.kaggle_environment(state,dict(os.environ));env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    current={}
    public_api="""import inspect,json
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate();out={}
def serial(v):
 if hasattr(v,'to_dict'):return v.to_dict()
 if hasattr(v,'__dict__'):return {k:x for k,x in vars(v).items() if not k.startswith('_')}
 if isinstance(v,(list,tuple)):return [serial(x) for x in v]
 return str(v)
for name,method in [('leaderboard','competition_leaderboard_view'),('files','competition_list_files')]:
 f=getattr(api,method,None)
 if f is None:out[name]={'not_available':True};continue
 sig=inspect.signature(f);kw={'page_size':20 if name=='leaderboard' else 100} if 'page_size' in sig.parameters else {}
 try:out[name]={'signature':str(sig),'data':serial(f('enveda-CASMI26-molecule-id-mass-spectra',**kw))}
 except Exception as exc:out[name]={'signature':str(sig),'error_type':type(exc).__name__}
print(json.dumps(out,default=serial))
"""
    for name,code in [('account',shared.API_READ),('public_metadata',public_api)]:
        r=subprocess.run([str(py),'-c',code],env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,
            encoding='utf-8',errors='replace',timeout=180)
        current[name]={'exit_code':r.returncode}
        if r.returncode==0:
            try:current[name]['data']=parse_public_result(r.stdout)
            except ValueError:
                current[name]['error']='No final JSON object; read failure does not authorize any write'
                current[name]['stdout_excerpt']=prep.redact(r.stdout,env)[:4000]
        else:current[name]['error']=prep.redact(r.stderr,env)[:1000]
    (out/'kaggle-current.json').write_text(json.dumps(current,indent=2,default=str)+'\n',encoding='utf-8')
    report['current_kaggle_read_completed']=True
    import torch
    report['compute']={'torch':torch.__version__,'cuda':torch.cuda.is_available(),'free_disk_bytes':shutil.disk_usage(root).free}
    if torch.cuda.is_available():report['compute']['gpu']=torch.cuda.get_device_name(0)
    for name,url,limit in [
        ('zenodo.json','https://zenodo.org/api/records/10997887',5*1024**2),
        ('source.zip','https://codeload.github.com/pluskal-lab/DreaMS/zip/'+DREAMS_COMMIT,256*1024**2)]:
        try:report['downloads'][name]=get(name,url,limit)
        except Exception as exc:report['downloads'][name]={'error_type':type(exc).__name__,'http_status':getattr(exc,'code',None)}
    try:
        meta=json.loads((root/'zenodo.json').read_text(encoding='utf-8'))
        report['publisher']={'title':meta.get('metadata',{}).get('title'),'license':meta.get('metadata',{}).get('license'),
            'files':[{'key':r.get('key'),'size':r.get('size'),'checksum':r.get('checksum')} for r in meta.get('files',[])]}
        r=ssl_entry(meta)
        if shutil.disk_usage(root).free < r['size']+5*1024**3:raise ValueError('Insufficient space for bounded weights')
        report['downloads']['ssl_model.ckpt']=get('ssl_model.ckpt',r['links']['self'],r['size'],r['checksum'])
    except Exception as exc:report['weights_blocked']={'error_type':type(exc).__name__,'reason':str(exc)[:250]}
    if (root/'source.zip').exists():
        with zipfile.ZipFile(root/'source.zip') as z:
            names=[n for n in z.namelist() if n.endswith(('/LICENSE','/setup.py','/dreams/api.py','/dreams/models/dreams/dreams.py','/dreams/utils/misc.py'))]
            for n in names:
                info=z.getinfo(n)
                if info.file_size<512*1024:
                    p=out/'source-inspection'/Path(n).name;p.parent.mkdir(exist_ok=True);p.write_bytes(z.read(n))
        shutil.copy2(root/'source.zip',out/'dreams-source.zip')
    for n in ('zenodo.json',):
        if (root/n).exists():shutil.copy2(root/n,out/n)
    report['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    text=json.dumps(report,indent=2,allow_nan=False)+'\n';(out/'probe.json').write_text(text,encoding='utf-8');(root/'probe.json').write_text(text,encoding='utf-8')
    print('FOUNDATION_PROBE_COMPLETE\n'+text,flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
