"""Read current account results and kernel version metadata, with no Kaggle writes."""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

KERNEL = 'thelindortis/casmi26-r12-ion-view-diagnostic'


def last_json(text):
    for line in reversed(text.splitlines()):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('Expected final JSON object')
            return value
    raise ValueError('Empty read response')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    state = Path(os.environ['CHEMISTRY_STATE_ROOT'])
    py = state / 'envs/casmi26/python.exe'
    if Path(sys.executable).resolve() != py.resolve():
        return subprocess.call([str(py), str(Path(__file__).resolve())],
                               env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})
    repo = Path(__file__).resolve().parents[1]
    out = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    out.mkdir(parents=True, exist_ok=True)
    prep = load('r12_inventory_prep', repo / 'tasks/casmi_prepare.py')
    shared = load('r12_inventory_shared', repo / 'tasks/casmi_r07_submission.py')
    env = prep.kaggle_environment(state, dict(os.environ))
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    code = '''import inspect,json
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
def plain(value):
 if value is None or isinstance(value,(bool,int,float,str)):return value
 if isinstance(value,(list,tuple)):return [plain(x) for x in value]
 if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
 if hasattr(value,'to_dict'):return plain(value.to_dict())
 return str(value)
result={}
for name in ['kernels_status','kernels_pull','competition_submit','competition_submit_code']:
 method=getattr(api,name,None)
 if method is not None:
  source=inspect.getsource(method)
  result[name]={'signature':str(inspect.signature(method)),'source':source[:40000]}
try:result['current_worker_status']=plain(api.kernels_status('thelindortis/casmi26-r12-ion-view-diagnostic'))
except Exception as exc:result['status_read_error']=type(exc).__name__
from kaggle.api.kaggle_api_extended import ApiGetKernelRequest
import hashlib
with api.build_kaggle_client() as client:
 for suffix in ['', '/1']:
  req=ApiGetKernelRequest();req.user_name='thelindortis';req.kernel_slug='casmi26-r12-ion-view-diagnostic'+suffix
  try:
   response=client.kernels.kernels_api_client.get_kernel(req)
   data=plain(response)
   if not isinstance(data,dict):raise ValueError('Unknown response schema')
   blob=data.get('blob',{})
   source=blob.pop('source',None)
   if source is None:source=response.blob.source
   notebook=json.loads(source)
   cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
   data['all_code_sha256']=hashlib.sha256(json.dumps(cells,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
   result['kernel_latest' if not suffix else 'kernel_version_1']=data
  except Exception as exc:result['kernel_read_error'+suffix.replace('/','_')]=type(exc).__name__+': '+str(exc)[:200]
print(json.dumps(result))
'''
    report = {'new_submissions': 0, 'new_notebook_versions': 0,
              'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'commit': os.environ.get('GITHUB_SHA')}
    for name, source in [('account', shared.API_READ), ('kernel-api', code)]:
        run = subprocess.run([str(py), '-c', source], env=env, stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
        if run.returncode:
            (out / (name + '-error.log')).write_text(prep.redact(run.stderr, env), encoding='utf-8')
            raise RuntimeError('Read-only inventory failed: ' + name)
        value = last_json(run.stdout)
        (out / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        if name == 'account':
            if value['limits']['numTotal'] != len(value['history']):
                raise ValueError('Incomplete account history')
            report['budget'] = shared.budget_decision(value['history'], value['limits'], dt.datetime.now(dt.timezone.utc))
            report['history'] = value['history']
    (out / 'remote-inventory.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('R12_REMOTE_READ ' + json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
