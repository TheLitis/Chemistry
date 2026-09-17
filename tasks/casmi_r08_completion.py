"""Verify completed R08 evidence, inspect source lineage, and read Kaggle scores.

No training, dataset/kernel uploads, submissions, service or permission changes.
The promotion gate is fixed before this continuation reads the study results.
"""
from __future__ import annotations
from collections import Counter
import datetime as dt
import importlib.util,json,math,os,shutil,subprocess,sys
from pathlib import Path


def promotion_gate(report):
    checks={}
    try:
        fresh=report['cohorts']['fresh_timsTOF_transfer'];old=report['cohorts']['reused_np_diagnostic']
        numbers=[report['selected']['weight']]
        numbers += [fresh[c]['paired_effect']['ci95'][0] for c in ('available','absent','external_recovery')]
        numbers += [old['available']['paired_effect']['delta_mrr']]
        checks['finite']=all(math.isfinite(float(x)) for x in numbers)
        checks['nonzero_calibrated_forward_weight']=float(numbers[0])>0
        checks['fresh_absent_positive_interval']=float(numbers[2])>0
        checks['fresh_external_positive_interval']=float(numbers[3])>0
        checks['fresh_reference_noninferiority_0_02']=float(numbers[1])>-.02
        checks['reused_np_reference_point_regression_at_most_0_01']=float(numbers[4])>=-.01
    except (KeyError,IndexError,TypeError,ValueError):checks['complete_valid_report']=False
    return {'eligible':bool(checks) and all(checks.values()),'checks':checks,
            'meaning':'Evidence gate for a new candidate, not proof of hidden performance or public-model train disjointness',
            'uses_official_score_for_selection':False}


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);python=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=python.resolve():
        return subprocess.call([str(python),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'tasks'));sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    import casmi_r08_verify as independent
    root=state/'artifacts/casmi26/research-r08/forward-v1'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    checked=independent.verify(root)
    write_json(out/'independent-verification.json',checked);write_json(root/'independent-verification.json',checked)
    report=json.loads((root/'report.json').read_text());plan=json.loads((root/'protocol.json').read_text())
    gate=promotion_gate(report)
    write_json(out/'promotion-gate.json',gate);write_json(root/'promotion-gate.json',gate)
    source=root/'fiora-e19ef82c9a6cb9dbac92bce23e914008f1aeb44e'
    paths=[root/'deps',source,repo/'work/casmi26',repo/'tasks']
    helper=load('python_paths',repo/'tasks/r08_python_command.py')
    forward=load('r08_forward',repo/'tasks/casmi_r08_forward.py')
    clean=forward.env_clean();clean['FIORA_TEST_MODEL']=str(source/'fiora/resources/models/fiora_OS_v1.0.0.pt')
    tests=subprocess.run(helper.python_command(python,paths,'import pytest,sys;raise SystemExit(pytest.main(sys.argv[1:]))',
        ['-q',str(repo/'work/casmi26/tests')]),env=clean,stdin=subprocess.DEVNULL,capture_output=True,text=True,
        encoding='utf-8',errors='replace',timeout=600)
    (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8')
    if tests.returncode:raise RuntimeError('Tests failed after numerical verification')
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    inventory=json.loads((state/'artifacts/casmi26/research-r07/inventory.json').read_text())
    train=Path(inventory['train_path'])
    if sha256(train)!=plan['source_train']:raise RuntimeError('Lineage corpus differs from sealed experiment')
    catalog=json.loads((state/'cache/casmi26/highres-v3/catalog.json').read_text())
    keys=set(plan['calibration_keys']+plan['reused_np_audit_keys']+plan['fresh_transfer_keys'])
    lookup={r[0]:r[2] for r in catalog if r[2] in keys};sources={k:Counter() for k in keys}
    values=pa.array(list(lookup),type=pa.string());scanned=0
    with pq.ParquetFile(train) as f:
        for batch in f.iter_batches(batch_size=65536,columns=['normalized_smiles','ingest_lib']):
            scanned+=len(batch)
            part=batch.filter(pc.is_in(batch.column('normalized_smiles'),value_set=values))
            for raw,lib in zip(part.column('normalized_smiles').to_pylist(),part.column('ingest_lib').to_pylist()):
                sources[lookup[raw]][str(lib)]+=1
    raw={r['key']:r for r in json.loads((root/'audit-ranks.json').read_text())}
    cohorts={}
    for label,ks in [('reused_np',plan['reused_np_audit_keys']),('fresh_timsTOF',plan['fresh_transfer_keys'])]:
        cohorts[label]={}
        for present in (False,True):
            ids=[k for k in ks if ('pluskal_ms2' in sources[k])==present];mm={}
            if ids:
                for case in ('available','absent','external_recovery'):
                    b=[raw[k]['baseline'][case] for k in ids];s=[raw[k]['selected'][case] for k in ids]
                    mm[case]={'baseline':independent.metrics(b),'selected':independent.metrics(s),'paired':independent.paired(b,s)}
            cohorts[label]['pluskal_present' if present else 'pluskal_not_found']={'molecules':len(ids),'metrics':mm}
    lineage={'scanned_rows':scanned,'catalog_sha256':sha256(state/'cache/casmi26/highres-v3/catalog.json'),
        'cohorts':cohorts,'sources_per_query':{k:dict(v) for k,v in sources.items()},
        'training_library_declared_by_fiora':'MSnLib v7','exact_forward_training_keys_available':False,
        'source_presence_is_not_exact_model_training_membership':True,
        'source_absence_does_not_prove_pretraining_disjointness':True,'model_selection_changed':False}
    write_json(out/'source-lineage.json',lineage);write_json(root/'source-lineage.json',lineage)
    prep=load('prepare',repo/'tasks/casmi_prepare.py');access=load('submission_reader',repo/'tasks/casmi_r07_submission.py')
    env=prep.kaggle_environment(state,dict(os.environ));env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    api=subprocess.run([str(python),'-c',access.API_READ],env=env,stdin=subprocess.DEVNULL,
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
    snapshot={'exit_code':api.returncode,'read_only':True}
    if api.returncode==0:snapshot['data']=json.loads(api.stdout)
    else:snapshot['error']=prep.redact(api.stderr,env)[:1000]
    write_json(out/'kaggle-status.json',snapshot)
    champion=state/'artifacts/casmi26/final-r07-v1/bundle'
    unchanged=sha256(champion/'model.npz')=='4a05a97b65276df6558e4c156f2129b5463b758f4ea730b0f1f93e99acf055a6'
    if not unchanged:raise RuntimeError('Frozen R07 model hash changed')
    for name in ('protocol.json','prepared.json','report.json','audit-ranks.json','evidence.json.gz',
        'forward-features.json.gz','selection-before-transfer-audit.json','simulation.json','graph-parity.json','requests.json','replay.json','failures.json'):
        shutil.copy2(root/name,out/name)
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip',
                    '--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    result={'status':'completed','commit':os.environ.get('GITHUB_SHA'),'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'verification':checked,'promotion_gate':gate,'tests':tests.stdout.strip(),'r07_model_unchanged':unchanged,
        'kaggle_read':snapshot,'new_training':False,'new_uploads':0,'new_submissions':0,
        'lineage_scope':'Exact membership only in the supplied CASMI source snapshot; not a reconstruction of FIORA training'}
    write_json(out/'completion.json',result);write_json(root/'completion.json',result)
    print('R08_COMPLETION_BEGIN\n'+json.dumps(result,indent=2)+'\nR08_COMPLETION_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
