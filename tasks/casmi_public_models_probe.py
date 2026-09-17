"""Fetch pinned public model material for sandbox inspection, never execute it.

Only project-scoped files are written. No CASMI data, model weights, credentials,
Kaggle settings, or installed PC packages are changed. Pip DOWNLOAD is limited to
public pure-python wheels; downloaded checkpoint objects are never unpickled.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

FIORA_COMMIT = 'e19ef82c9a6cb9dbac92bce23e914008f1aeb44e'
ALLOWED_HOSTS = {'codeload.github.com','raw.githubusercontent.com','zenodo.org','www.zenodo.org','pan.sjtu.edu.cn'}
WHEELS = ['torch-geometric==2.6.1','dill==0.4.0','treelib==1.8.0','spectrum-utils==0.4.2']


def check_url(url: str) -> bool:
    p = urllib.parse.urlsplit(url)
    if p.scheme != 'https' or p.hostname not in ALLOWED_HOSTS or p.username or p.password or p.port not in (None,443):
        raise ValueError('Not an allowed public model source')
    return True


def copy_bounded(stream, destination: Path, limit: int) -> dict:
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name+'.', suffix='.part', dir=destination.parent)
    digest=hashlib.sha256();count=0;start=time.monotonic()
    try:
        with os.fdopen(fd,'wb') as out:
            while True:
                if time.monotonic()-start>300:raise TimeoutError('Public download time budget exceeded')
                chunk=stream.read(min(1024*1024,limit-count+1))
                if not chunk:break
                count+=len(chunk)
                if count>limit:raise ValueError('Public file exceeds bounded download size')
                out.write(chunk);digest.update(chunk)
        os.replace(temporary,destination)
        return {'bytes':count,'sha256':digest.hexdigest()}
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def inspect_archive(path: Path) -> dict:
    result={'model_assets':[],'parameters':{},'license_text':None,'checkpoint_executed':False}
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist())>1024**3:raise ValueError('Expanded source exceeds size budget')
        seen=set()
        for item in z.infolist():
            name=item.filename.replace('\\','/');parts=PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or ':' in name or stat.S_ISLNK(item.external_attr>>16):
                raise ValueError('Unsafe archive member')
            if name.lower() in seen:raise ValueError('Duplicate archive member')
            seen.add(name.lower())
            if name.endswith(('.pt','.ckpt','.safetensors')):
                result['model_assets'].append({'path':name,'bytes':item.file_size})
            if name.endswith('_params.json') and item.file_size<128*1024:
                result['parameters'][name]=json.loads(z.read(item))
            if len(parts.parts)==2 and parts.name.upper() in ('LICENSE','LICENSE.TXT'):
                result['license_text']=z.read(item).decode('utf-8')
        result['members']=len(z.infolist())
    return result


def summarize_zenodo(data: dict) -> dict:
    meta=data.get('metadata',{})
    return {'title':meta.get('title'),'license':meta.get('license'),
            'files':[{'key':r.get('key'),'bytes':r.get('size'),'checksum':r.get('checksum'),
                      'url':r.get('links',{}).get('self')} for r in data.get('files',[])],
            'checkpoint_downloaded':False}


def main() -> int:
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1'})
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('public_tls',repo/'tasks/public_tls.py')
    tls=importlib.util.module_from_spec(spec);spec.loader.exec_module(tls)
    context=tls.verified_context()
    root=state/'artifacts/casmi26/public-model-probe-20260917';root.mkdir(parents=True,exist_ok=True)
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    report={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'commit':os.environ.get('GITHUB_SHA'),
            'third_party_code_executed':False,'checkpoint_executed':False,'installed_packages_changed':False,
            'test_data_read':False,'new_submissions':0,'downloads':{}}
    sources={
        'fiora.zip':('https://codeload.github.com/BAMeScience/fiora/zip/'+FIORA_COMMIT,256*1024**2),
        'frigid-zenodo.json':('https://zenodo.org/api/records/19685145',4*1024**2),
        'msgpt-public-page.html':('https://pan.sjtu.edu.cn/web/share/8f63823e4d82280636e537d1c789128d',2*1024**2),
    }
    for name,(url,limit) in sources.items():
        info={'source':url,'filename':name}
        try:
            check_url(url)
            request=urllib.request.Request(url,headers={'User-Agent':'CASMI-public-source-probe/1.0'})
            with urllib.request.urlopen(request,context=context,timeout=30) as response:
                check_url(response.geturl())
                declared=response.headers.get('Content-Length')
                if declared and int(declared)>limit:raise ValueError('Public response exceeds size budget')
                info.update(status=response.status,content_type=response.headers.get('Content-Type'))
                info.update(copy_bounded(response,root/name,limit))
            if name=='fiora.zip':info['inspection']=inspect_archive(root/name)
            elif name=='frigid-zenodo.json':info['inspection']=summarize_zenodo(json.loads((root/name).read_text()))
            else:info['html_only_not_checkpoint']=True
        except Exception as exc:
            info['error_type']=type(exc).__name__
            if isinstance(exc,urllib.error.HTTPError):info['status']=exc.code
        report['downloads'][name]=info
        print('PUBLIC_SOURCE '+name+' '+str(info.get('status',info.get('error_type'))),flush=True)
    wheels=root/'wheels';wheels.mkdir(exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('PIP_','KAGGLE_','GITHUB_TOKEN','GH_TOKEN'))}
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK='1')
    command=[str(python),'-m','pip','download','--index-url','https://pypi.org/simple','--no-deps',
             '--only-binary=:all:','--dest',str(wheels)]+WHEELS
    try:
        r=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,
                         encoding='utf-8',errors='replace',timeout=180)
        report['public_wheel_download']={'exit_code':r.returncode,'requested':WHEELS,
            'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(wheels.glob('*.whl'))]}
    except subprocess.TimeoutExpired:
        report['public_wheel_download']={'error_type':'TimeoutExpired','requested':WHEELS}
    text=json.dumps(report,indent=2,ensure_ascii=True)+'\n'
    (root/'probe.json').write_text(text,encoding='utf-8');(out/'probe.json').write_text(text,encoding='utf-8')
    with zipfile.ZipFile(out/'public-model-material.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in ('fiora.zip','frigid-zenodo.json','msgpt-public-page.html','probe.json'):
            path=root/name
            if path.is_file():z.write(path,name)
        for path in sorted(wheels.glob('*.whl')):z.write(path,'wheels/'+path.name)
    print('PUBLIC_MODEL_PROBE_BEGIN\n'+text+'PUBLIC_MODEL_PROBE_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
