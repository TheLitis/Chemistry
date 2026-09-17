"""Read-only reconciliation of R07 files; never changes a sealed experiment."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);repo=Path(__file__).resolve().parents[1]
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root=state/'artifacts/casmi26/research-r07/full-system-v1'
    result={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'commit':os.environ.get('GITHUB_SHA'),
            'root':str(root),'files':[],'source':{},'new_training':False,'new_submissions':0}
    allowed={'protocol.json','report.json','audit-ranks.json','selection-before-audit.json','mass-diagnostics.json',
             'v1-independent.npz.training.json','v3-independent.npz.training.json','v1-independent.training.json',
             'v3-independent.training.json','external-manifest.json','calibration-keys.json'}
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if not p.is_file():continue
            item={'name':p.name,'bytes':p.stat().st_size,'mtime_utc':dt.datetime.fromtimestamp(p.stat().st_mtime,dt.timezone.utc).isoformat()}
            if p.suffix in ('.json','.npz'):item['sha256']=digest(p)
            result['files'].append(item)
            if p.name in allowed:
                shutil.copy2(p,out/p.name)
                if p.name in ('report.json','protocol.json','v1-independent.training.json','v3-independent.training.json'):
                    data=json.loads(p.read_text(encoding='utf-8-sig'))
                    result[p.name]=data if p.name!='protocol.json' else {k:v for k,v in data.items() if not k.endswith('_keys')}
    rels=['tasks/casmi_r07_domain.py']+['work/casmi26/casmi26/'+n for n in ('target_domain.py','production.py','learning.py','model_v3.py','features_v3.py','portable.py','catalog_candidates.py','ranking.py')]
    for relative in rels:
        p=repo/relative;b=p.read_bytes()
        result['source'][relative]={'sha256':digest(p),'lf_sha256':hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()}
    # Only collect process lines containing this exact research entry point.
    ps="$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match 'casmi_r07_domain.py|run_r07.py') } | Select-Object ProcessId,ParentProcessId,Name,CommandLine) | ConvertTo-Json -Depth 3 -Compress"
    p=subprocess.run(['powershell.exe','-NoProfile','-Command',ps],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
    result['r07_processes_query_ok']=p.returncode==0
    if p.returncode==0:
        try:result['r07_processes']=json.loads(p.stdout.strip()) if p.stdout.strip() else []
        except ValueError:result['r07_processes']='unparsed'
    (out/'reconcile.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('R07_RECONCILE_BEGIN\n'+json.dumps(result,indent=2)+'\nR07_RECONCILE_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
