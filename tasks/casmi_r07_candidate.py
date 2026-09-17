"""Adopt verified completed R07 evidence, refit the fixed winner, package and test.

No Kaggle upload or submission. The immutable local-package audit is preserved.
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
import time
import zipfile

EXPECTED_REPORT='3775a0fb2d78766149d7cbc89a4b603e69840befdcae7de11e5d19c39d97408e'
EXPECTED_IMPLEMENTATION='11a479c7e6677c891fba38cad2a9d03551dff6478c656f5b26d605f81327cbd2'
SOURCE_NAMES=['tasks/casmi_r07_domain.py']+['work/casmi26/casmi26/'+n for n in ('target_domain.py','production.py','learning.py','model_v3.py','features_v3.py','portable.py','catalog_candidates.py','ranking.py')]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.production import sha256,write_json
    from casmi26.target_domain import training_rows,metrics
    from casmi26.ranking import paired_effect
    from casmi26.learning import train_arrays
    from casmi26.pipeline_v3 import export_old_model
    from casmi26.r07_release import validate_selection,verify_bundle,infer,source_identity,FILES
    from casmi26.metric import require_official_rdkit,structure_key
    import numpy as np
    require_official_rdkit();start=time.monotonic()
    art=state/'artifacts/casmi26';study=art/'research-r07/full-system-v1';root=art/'final-r07-v1';root.mkdir(parents=True,exist_ok=True)
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    if sha256(study/'report.json')!=EXPECTED_REPORT:raise RuntimeError('Audited study changed')
    report=json.loads((study/'report.json').read_text());protocol=json.loads((study/'protocol.json').read_text())
    selection=json.loads((study/'selection-before-audit.json').read_text());ranks=json.loads((study/'audit-ranks.json').read_text())
    for name,h in report['artifact_hashes'].items():
        if sha256(study/name)!=h:raise RuntimeError('Research artifact hash mismatch: '+name)
    # Original archive used LF for its two added files and the preserved bytes for
    # pre-existing source. Reconstruct exactly that identity, not a weaker guard.
    historical=[]
    for name in SOURCE_NAMES:
        content=(repo/name).read_bytes()
        if name in ('tasks/casmi_r07_domain.py','work/casmi26/casmi26/target_domain.py'):
            content=content.replace(b'\r\n',b'\n')
        historical.append(hashlib.sha256(content).hexdigest())
    if hashlib.sha256(''.join(historical).encode()).hexdigest()!=EXPECTED_IMPLEMENTATION:
        raise RuntimeError('Execution source differs from the audited package beyond line endings')
    if report['source']['implementation_sha256']!=EXPECTED_IMPLEMENTATION or report['warmstart'] is not None:
        raise RuntimeError('Study provenance mismatch')
    cal=protocol['calibration_keys'];audit=protocol['audit_keys']
    if set(cal)&set(audit) or [r['key'] for r in ranks]!=audit or len(audit)!=170:
        raise RuntimeError('Split/rank identity changed')
    if report['selected']!=selection['selected']['configuration'] or not selection['selected_before_audit']:
        raise RuntimeError('Selection was not frozen')
    config=validate_selection(report['selected'],report['mass_offset_ppm'])
    for case,v in report['audit'].items():
        br=[r['baseline'][case] for r in ranks];sr=[r['selected'][case] for r in ranks]
        for key,values in (('baseline',br),('selected',sr)):
            for m,x in metrics(values).items():
                if abs(x-v[key][m])>1e-12:raise RuntimeError('Inconsistent reported metric')
        rr=lambda a:np.where(np.array(a)>0,1/np.maximum(a,1),0.)
        effect=paired_effect(rr(br),rr(sr))
        if not np.allclose(effect['ci95'],v['paired']['ci95'],rtol=0,atol=1e-12):raise RuntimeError('Inconsistent paired interval')
    checks=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=360)
    (out/'tests.log').write_text(checks.stdout+'\n'+checks.stderr,encoding='utf-8');print(checks.stdout,flush=True)
    if checks.returncode:raise RuntimeError('Candidate tests failed')
    cache=state/'cache/casmi26/highres-v3';catalog=json.loads((cache/'catalog.json').read_text())
    counts=np.load(cache/'counts.npy',allow_pickle=False);packed=np.load(cache/'targets.npy',allow_pickle=False,mmap_mode='r')
    target=set(cal)|set(audit);hold=training_rows(catalog,counts,target)
    keyhash=hashlib.sha256('\n'.join(sorted(catalog[i][2] for i in hold)).encode()).hexdigest()
    if keyhash!=report['source']['training_key_sha256']:raise RuntimeError('Holdout fitting keys differ')
    for name,k in (('catalog.json','catalog_sha256'),('targets.npy','targets_sha256')):
        if sha256(cache/name)!=report['source'][k]:raise RuntimeError('Candidate corpus changed')
    prepared=json.loads((cache/'prepared-v3.json').read_text())
    for name,h in prepared['files'].items():
        if sha256(cache/name)!=h:raise RuntimeError('Prepared cache changed: '+name)
    ids=training_rows(catalog,counts,set(),exclude_hash_holdout=False)
    keyhash_full=hashlib.sha256('\n'.join(sorted(catalog[i][2] for i in ids)).encode()).hexdigest()
    bundle=root/'bundle';bundle.mkdir(exist_ok=True)
    identity={'study_report_sha256':EXPECTED_REPORT,'train_sha256':report['source']['train_sha256'],
        'training_keys_sha256':keyhash_full,'training_rows':len(ids),'epochs':30,'seed':1729,'learning_rate':.002,
        'hidden':512,'batch_size':256,'warmstart':None,'selection':config,'mass_offset_ppm':report['mass_offset_ppm']}
    mp=bundle/'model.npz';meta=root/'refit.json'
    if mp.exists():
        old=json.loads(meta.read_text())
        if old['identity']!=identity or old['model_sha256']!=sha256(mp):raise RuntimeError('Refit checkpoint mismatch')
        training=old['training']
    else:
        x=np.load(cache/'features.npy',allow_pickle=False,mmap_mode='r')
        print('R07_FULL_REFIT_START '+str(len(ids)),flush=True)
        legacy=root/'full-refit-original.npz'
        if legacy.exists():raise RuntimeError('Partial refit requires explicit reconciliation; not overwriting')
        training=train_arrays(np.asarray(x[ids,:4104]),np.unpackbits(packed[ids,:256],axis=1).astype(np.float32),
            legacy,epochs=30,hidden=512,batch_size=256,device='cuda',seed=1729,learning_rate=.002)
        export_old_model(legacy,mp)
        write_json(meta,{'identity':identity,'training':training,'model_sha256':sha256(mp)})
    snapshot=state/'data/external/coconut-2026-08/coconut_csv_lite-08-2026.zip'
    if sha256(snapshot)!=report['source']['coconut_snapshot_sha256']:raise RuntimeError('External snapshot changed')
    shutil.copy2(cache/'catalog.json',bundle/'catalog.json')
    np.save(bundle/'fingerprints.npy',np.array(packed[:,:256]))
    shutil.copy2(snapshot,bundle/'coconut.zip')
    manifest={'format':7,'algorithm':'r07-selected-v1-hybrid','feature_version':'official-corpus-v1',
        'selection':config,'mass_offset_ppm':report['mass_offset_ppm'],'mass_scale_ppm':5.,
        'train_sha256':report['source']['train_sha256'],'study_report_sha256':EXPECTED_REPORT,
        'weights_kind':'full-corpus-refit-not-the-audit-checkpoint','refit':identity,
        'files':{n:sha256(bundle/n) for n in FILES},'contains_test_ids_or_predictions':False,
        'external_source':{'name':'COCONUT','release':'2026-08','license':'CC0','snapshot_sha256':report['source']['coconut_snapshot_sha256']},
        'official_score':None,'candidate_not_champion':True}
    write_json(bundle/'r07-bundle.json',manifest);verify_bundle(bundle)
    data=load('staged',repo/'tasks/casmi_staged.py').find_dataset(state/'data/external')
    output=root/'submission.csv'
    print('R07_VISIBLE_INFERENCE_START',flush=True)
    inference=infer(data/'test.parquet',data/'train.parquet',bundle,output,template=data/'sample_submission.csv',workers=8)
    import csv
    with output.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        keys=[structure_key(s) for s in row['smiles'].split(';')]
        if not 1<=len(keys)<=25 or None in keys or len(set(keys))!=len(keys):raise RuntimeError('Invalid or duplicate structural guesses')
    launcher=root/'predict_r07.py'
    launcher.write_text("from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent/'work/casmi26'))\nif __name__=='__main__':\n    from casmi26.r07_release import main\n    raise SystemExit(main())\n",encoding='utf-8')
    source=root/'work/casmi26/casmi26';source.mkdir(parents=True,exist_ok=True)
    for p in (repo/'work/casmi26/casmi26').glob('*.py'):shutil.copy2(p,source/p.name)
    shutil.copy2(repo/'work/casmi26/requirements.txt',root/'requirements.txt')
    readme='R07 final candidate for this iteration. Full-corpus refit of the calibration-selected V1-fingerprint hybrid with an offline COCONUT snapshot. Not de novo and not an officially evaluated champion. The held-out timsTOF study is separate from these refit weights. Run predict_r07.py --test <current test.parquet> --train <train.parquet> --bundle bundle --output submission.csv. Existing pinned CASMI Python environment required. No credentials or account tools are included.\n'
    (root/'README.txt').write_text(readme,encoding='utf-8')
    (root/'SOURCE-LICENSES.txt').write_text('External structure data: COCONUT, August 2026 snapshot, CC0. Source: https://coconut.naturalproducts.net/download . Python dependency licenses remain those of their respective projects. Competition-supplied structures are not represented as CC0. This archive is private research output, not a relicensing of competition data.\n',encoding='utf-8')
    archive=root/'casmi-r07-candidate.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(bundle.iterdir()):
            if p.is_file():z.write(p,p.relative_to(root).as_posix())
        for p in sorted(source.glob('*.py')):z.write(p,p.relative_to(root).as_posix())
        for name in ('predict_r07.py','requirements.txt','README.txt','SOURCE-LICENSES.txt'):z.write(root/name,name)
    release={'status':'candidate_built_and_visible_output_verified','checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'commit':os.environ.get('GITHUB_SHA'),'adopted_preexisting_local_study':True,'study_report_sha256':EXPECTED_REPORT,
        'study_source_equivalent_after_line_endings':True,'study_protocol_overwritten':False,
        'canonical_study_source_sha256':source_identity(repo,SOURCE_NAMES),'independent_audit_recomputed':True,
        'tests':checks.stdout.strip(),'selection':config,'refit':{'identity':identity,'training':training,'model_sha256':sha256(mp)},
        'bundle_manifest_sha256':sha256(bundle/'r07-bundle.json'),'inference':{k:v for k,v in inference.items() if k!='details'},
        'total_structural_guesses':sum(len(r['smiles'].split(';')) for r in rows),'guesses_valid_and_distinct':True,
        'release_directory':str(root),'archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha256(archive)},
        'new_submissions':0,'new_kaggle_uploads':0,'official_score_for_candidate':None,'original_champion_replaced':False,
        'seconds':time.monotonic()-start}
    write_json(root/'release.json',release);write_json(out/'release.json',release)
    for name in ('report.json','audit-ranks.json','protocol.json','mass-diagnostics.json'):shutil.copy2(study/name,out/('study-'+name))
    shutil.copy2(output,out/output.name);shutil.copy2(output.with_suffix('.csv.report.json'),out/'submission.csv.report.json')
    subprocess.run(['git','-c','safe.directory='+str(repo),'-C',str(repo),'archive','--format=zip','--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=90)
    print('R07_CANDIDATE_BEGIN\n'+json.dumps(release,indent=2)+'\nR07_CANDIDATE_END',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
