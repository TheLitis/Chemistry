"""Resolve already provisioned Kaggle access in the service and named user profile.

Only exact Kaggle environment values and official data filenames are checked.
Secrets remain in local process memory; no browser cookies, credential vaults,
email, or unrelated user files are read. This does not create credentials,
accept rules, submit predictions, or change Windows permissions.
"""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil

AUTH_KEYS=('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_CONFIG_DIR')
OFFICIAL_NAMES=('train.parquet','test.parquet','sample_submission.csv')


def select_environment(inherited: dict, candidates: list[tuple[str,dict]]) -> tuple[dict,str]:
    env=dict(inherited)
    def complete(values):
        return bool(values.get('KAGGLE_API_TOKEN') or
                    (values.get('KAGGLE_USERNAME') and values.get('KAGGLE_KEY')))
    if complete(env):return env,'process'
    for source,values in candidates:
        if complete(values):
            for key in AUTH_KEYS[:3]:env.pop(key,None)
            env.update({k:v for k,v in values.items() if k in AUTH_KEYS and v})
            return env,source
    # A directory can still contain the standard CLI credential file. Never
    # combine a username from one source with a key from another source.
    if not env.get('KAGGLE_CONFIG_DIR'):
        for _,values in candidates:
            if values.get('KAGGLE_CONFIG_DIR'):
                env['KAGGLE_CONFIG_DIR']=values['KAGGLE_CONFIG_DIR'];break
    return env,'not_provisioned'


def windows_candidates(user_root: Path) -> tuple[list[tuple[str,dict]],list[str]]:
    if os.name!='nt':return [],['not_windows']
    import winreg
    warnings=[]
    def read_values(hive,path):
        out={}
        try:
            with winreg.OpenKey(hive,path,0,winreg.KEY_READ) as key:
                for name in AUTH_KEYS:
                    try:
                        value,kind=winreg.QueryValueEx(key,name)
                        if isinstance(value,str) and value.strip():out[name]=value
                    except FileNotFoundError:pass
        except OSError as exc:
            warnings.append(type(exc).__name__)
        return out
    sid=None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r'SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList',0,winreg.KEY_READ) as root:
            for index in range(winreg.QueryInfoKey(root)[0]):
                name=winreg.EnumKey(root,index)
                try:
                    with winreg.OpenKey(root,name) as item:
                        profile,_=winreg.QueryValueEx(item,'ProfileImagePath')
                    if os.path.normcase(os.path.expandvars(profile))==os.path.normcase(str(user_root)):
                        sid=name;break
                except OSError:continue
    except OSError as exc:warnings.append(type(exc).__name__)
    user=read_values(winreg.HKEY_USERS,sid+r'\Environment') if sid else {}
    machine=read_values(winreg.HKEY_LOCAL_MACHINE,
                        r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment')
    return [('named_user_environment',user),('machine_environment',machine)],warnings


def official_files(roots: list[Path]) -> list[dict]:
    found=[];seen=set()
    for root in roots:
        for name in OFFICIAL_NAMES:
            path=root/name
            try:
                if path.is_file() and not path.is_symlink() and str(path.resolve()) not in seen:
                    seen.add(str(path.resolve()))
                    found.append({'path':str(path),'bytes':path.stat().st_size})
            except OSError:pass
    return found


def main() -> int:
    repo=Path(__file__).resolve().parents[1]
    state=Path(os.environ['CHEMISTRY_STATE_ROOT'])
    output=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
    user=Path(r'C:\Users\loval')
    candidates,warnings=windows_candidates(user)
    env,source=select_environment(dict(os.environ),candidates)
    spec=importlib.util.spec_from_file_location('prepare',repo/'tasks/casmi_prepare.py')
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    env=prepare.kaggle_environment(state,env)
    env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
    python=state/'envs/casmi26/python.exe'
    tests=prepare.run([str(python),'-m','pytest','-q',str(repo/'work/casmi26/tests')],
                      env=env,timeout=240,log=output/'access-tests.log',expose=True)
    if tests['exit_code']:raise RuntimeError('Updated pipeline tests failed')
    cli=[str(python),'-c','from kaggle.cli import main; main()']
    result=prepare.run(cli+['competitions','files',prepare.SLUG,'--page-size','200','-v'],
                       env=env,timeout=60,log=output/'kaggle-access.log',expose=False)
    roots=[user/'Downloads',user/'Chemistry',user/'Chemistry/data',
           user/'Chemistry/work/casmi26/data',state/'data/casmi26',state/'data/casmi26/raw']
    report={'request_id':os.environ.get('CHEMISTRY_REQUEST_ID'),
            'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'credential_source_kind':source,
            'credential_values_logged':False,'registry_warnings':sorted(set(warnings)),
            'tests_exit_code':tests['exit_code'],
            'kaggle_file_listing_succeeded':result['exit_code']==0,
            'kaggle_file_listing_exit_code':result['exit_code'],
            'official_files_in_scoped_locations':official_files(roots),
            'official_score':None,'rules_accepted_by_script':False,'submission_made':False}
    if result['exit_code']==0:
        if shutil.disk_usage(state).free<30*2**30:raise RuntimeError('Less than 30 GiB free')
        target=state/'data/casmi26/raw';target.mkdir(parents=True,exist_ok=True)
        download=prepare.run(cli+['competitions','download',prepare.SLUG,'-p',str(target)],
                             env=env,timeout=1800,log=output/'kaggle-download.log',expose=False)
        report['download_exit_code']=download['exit_code']
        report['download_directory']=str(target)
    report['status']='authorized_access_verified' if result['exit_code']==0 else 'no_authorized_file_access_verified'
    text=json.dumps(report,indent=2,ensure_ascii=True,allow_nan=False)
    (output/'access-report.json').write_text(text+'\n',encoding='utf-8')
    destination=state/'artifacts/casmi26';destination.mkdir(parents=True,exist_ok=True)
    (destination/'access-report.json').write_text(text+'\n',encoding='utf-8')
    print('CASMI_ACCESS_REPORT_BEGIN\n'+text+'\nCASMI_ACCESS_REPORT_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
