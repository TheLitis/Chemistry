"""Verify durable delivery files and parse the user-only credential helper.

No credential values are read or emitted; the staging helper is not executed.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    from casmi26.portable import verify_bundle
    artifacts=state/'artifacts/casmi26/official-v1';output=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    report=json.loads((artifacts/'delivery-report.json').read_text())
    if report['status']!='local_trained_notebook_completed_official_score_unverified':raise ValueError('Delivery did not complete')
    bundle=verify_bundle(artifacts/'bundle')
    if sha256(artifacts/'submission.csv')!=report['prediction']['submission_sha256']:raise ValueError('Submission changed')
    if sha256(Path(report['delivery']['path']))!=report['delivery']['sha256']:raise ValueError('Delivery ZIP changed')
    helper=repo/'runner/stage-kaggle-access.ps1'
    script="$tokens=$null;$errors=$null;[System.Management.Automation.Language.Parser]::ParseFile('"+str(helper).replace("'","''")+"',[ref]$tokens,[ref]$errors)|Out-Null;if($errors.Count){$errors|ForEach-Object{Write-Error $_.Message};exit 1};Write-Output 'POWERSHELL_PARSE_OK'"
    parsed=subprocess.run(['powershell.exe','-NoLogo','-NoProfile','-Command',script],capture_output=True,text=True,encoding='utf-8',errors='replace')
    if parsed.returncode:raise RuntimeError('Credential helper PowerShell syntax invalid: '+parsed.stderr)
    print(parsed.stdout,flush=True)
    tests=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
    print(tests.stdout,flush=True)
    if tests.returncode:raise RuntimeError('Final software tests failed: '+tests.stderr)
    brief={k:v for k,v in report.items() if k in ('status','official_score','submission_made','submission_validation','notebook_execution','wheel_resolution','paths','delivery','kaggle_access')}
    brief.update(request_id=os.environ.get('CHEMISTRY_REQUEST_ID'),verified_current_commit=os.environ.get('GITHUB_SHA'),
                 test_result=tests.stdout.strip(),powershell_helper_syntax_verified=True,credentials_staged_by_this_run=False,
                 trained_bundle_hashes_verified=True,submission_hash_verified=True,zip_hash_verified=True)
    for name in ('casmi26-submission.ipynb','submission.csv','submission.csv.report.json','delivery-report.json','validation.json'):
        shutil.copy2(artifacts/name,output/name)
    write_json(artifacts/'verified-delivery-status.json',brief);write_json(output/'verified-delivery-status.json',brief)
    print('VERIFIED_DELIVERY_BEGIN\n'+json.dumps(brief,indent=2)+'\nVERIFIED_DELIVERY_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
