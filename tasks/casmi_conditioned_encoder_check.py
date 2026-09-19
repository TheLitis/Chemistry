"""Engineering check of conditioned inference using exact public FPNet weights.

Only tensor checkpoints and allowlisted definitions from the reviewed notebook
are loaded. No third-party notebook, installer, pickle, or test labels execute
on ChemistryPC. Training examples here test input behavior, not accuracy.
"""
from __future__ import annotations
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

SOURCE_HASH='69005db439e20c5dba730779d7332934e57cfd36c08292722ed2d4a3337cfd1c'
REF='prvsiyan/casmi26-fp-models-v2'
WEIGHTS={
 'fp_single_s2.pt':'c19aa29016ed4d7a67bac38105b8fa5076c696c29eb66a504558c9b2e52b614a',
 'fp_merged_m1.pt':'261529c823825cbe2028684bbf7214a95a7e631ca9f656336d577c507cce8294',
}
BYTE_SIZE=144110255
SEED='conditioned-encoder-engineering-20260919'
NAMES=('instr_family','prep_peaks','SinEmb','Block','FPNet','_merge_peaks',
       'model_logits','_logits_from','_logits_raw')


def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(p,data):
    Path(p).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def definition_ast(text):
    tree=ast.parse(text);chosen=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in NAMES]
    if sorted(n.name for n in chosen)!=sorted(NAMES):raise ValueError('Missing or duplicated reviewed definitions')
    constants={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('ADDUCT_LIST','INSTR_LIST','MAX_PEAKS_NN'):
            if n.targets[0].id in constants:raise ValueError('Duplicate model constant')
            constants[n.targets[0].id]=ast.literal_eval(n.value)
    if set(constants)!={'ADDUCT_LIST','INSTR_LIST','MAX_PEAKS_NN'} or constants['MAX_PEAKS_NN']!=128:
        raise ValueError('Unreviewed model preprocessing constants')
    return ast.fix_missing_locations(ast.Module(body=chosen,type_ignores=[])),constants


def reviewed_namespace(path):
    if digest(path)!=SOURCE_HASH:raise ValueError('Author notebook bytes changed')
    import numpy as np
    import torch
    book=json.loads(Path(path).read_text(encoding='utf-8'))
    text='\n\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
    tree,constants=definition_ast(text)
    namespace={'np':np,'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'math':math,**constants}
    namespace['ADDUCT_IX']={a:i for i,a in enumerate(constants['ADDUCT_LIST'])}
    namespace['INSTR_IX']={a:i for i,a in enumerate(constants['INSTR_LIST'])}
    namespace['_MODEL']=None
    exec(compile(tree,'<reviewed-author-model-definitions>','exec'),namespace)
    return namespace


def execute(state,repo,out,root):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.conditioned_encoder import model_logits as corrected
    source=state/'artifacts/casmi26/fundamental-audit-20260919/haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline.ipynb'
    ns=reviewed_namespace(source)
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2)
    nets=[];nbits=None;params=[]
    for name,h in WEIGHTS.items():
        p=root/name
        if p.stat().st_size!=BYTE_SIZE or digest(p)!=h:raise ValueError('Checkpoint integrity changed')
        ck=torch.load(p,map_location='cpu',weights_only=True)
        if not isinstance(ck,dict) or any(k not in ck for k in ('model','nbits','d','layers')):raise ValueError('Unexpected tensor checkpoint')
        if nbits is not None and ck['nbits']!=nbits:raise ValueError('Fingerprint dimensions differ')
        nbits=int(ck['nbits']);net=ns['FPNet'](nbits,d=int(ck['d']),layers=int(ck['layers']))
        net.load_state_dict(ck['model'],strict=True);net.to(device).eval();nets.append(net)
        params.append(sum(p.numel() for p in net.parameters()));del ck
    bundle=([nets[0]],[nets[1]],device,nbits);ns['_MODEL']=bundle
    inv=json.loads((state/'artifacts/casmi26/research-r07/inventory.json').read_text())
    columns=['inchikey14','adduct','ionization_mode','instrument_type','precursor_mz','ms2_mzs','ms2_normalized_intensities','collision_energy_ev']
    frame=pq.read_table(inv['train_path'],columns=columns,filters=[('ingest_lib','==','enveda-np-examples')]).to_pandas()
    keys=sorted(set(frame.inchikey14),key=lambda k:hashlib.sha256((SEED+':'+str(k)).encode()).digest())[:32]
    if len(keys)!=32:raise ValueError('Insufficient deterministic training smoke groups')
    if device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    started=time.monotonic();records=[];arrays={}
    def new(g):return corrected(g,bundle,ns['prep_peaks'],ns['_merge_peaks'],ns['ADDUCT_IX'],ns['instr_family'])
    for i,key in enumerate(keys):
        group=frame.loc[frame.inchikey14==key].reset_index(drop=True)
        with torch.inference_mode():
            a=ns['model_logits'](group);rev=ns['model_logits'](group.iloc[::-1]);b=new(group);br=new(group.iloc[::-1])
            solo=ns['model_logits'](group.iloc[:1]);solo_new=new(group.iloc[:1])
        if any(x is None or x.shape!=(nbits,) or not np.isfinite(x).all() for x in (a,rev,b,br,solo,solo_new)):
            raise ValueError('Model produced invalid diagnostics')
        row={'key':str(key),'spectra':len(group),'adducts':int(group.adduct.nunique()),'polarities':int(group.ionization_mode.nunique()),
             'original_reverse_max_abs':float(np.max(abs(a-rev))), 'corrected_reverse_max_abs':float(np.max(abs(b-br))),
             'single_spectrum_parity_max_abs':float(np.max(abs(solo-solo_new))),
             'changed_logits_l2':float(np.linalg.norm(a-b)),'changed_logits_max_abs':float(np.max(abs(a-b)))}
        if row['corrected_reverse_max_abs']>1e-3 or row['single_spectrum_parity_max_abs']>1e-3:
            raise ValueError('Corrected invariance or one-spectrum parity failed')
        records.append(row)
        for label,value in [('original',a),('original_reversed',rev),('conditioned',b),('conditioned_reversed',br)]:arrays[f'{i}_{label}']=value
        if (i+1)%8==0:print('CONDITION_CHECK '+str(i+1),flush=True)
    if device=='cuda':torch.cuda.synchronize()
    np.savez_compressed(out/'logits.npz',**arrays)
    report={'status':'real_weights_behavior_verified','checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_sha256':SOURCE_HASH,'checkpoint_sha256':WEIGHTS,'torch':torch.__version__,'numpy':np.__version__,
        'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'parameters_per_model':params,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
        'groups':len(records),'spectra':sum(r['spectra'] for r in records),'seed':SEED,'records':records,
        'original_order_changes_gt_1e_3':sum(r['original_reverse_max_abs']>1e-3 for r in records),
        'corrected_order_changes_gt_1e_3':sum(r['corrected_reverse_max_abs']>1e-3 for r in records),
        'max_corrected_reverse_error':max(r['corrected_reverse_max_abs'] for r in records),
        'max_single_spectrum_error':max(r['single_spectrum_parity_max_abs'] for r in records),
        'seconds':time.monotonic()-started,'new_training':False,'new_submissions':0,'incumbent_changed':False,
        'accuracy_established':False,'test_data_read':False,'tensor_only_checkpoint_loading':True,
        'notebook_top_level_executed':False,'allowed_definitions':list(NAMES),
        'limitations':['Training examples test engineering properties, not held-out chemical accuracy.',
            'Author training augmentation could mix different conditions; repaired inference changes its distribution.',
            'Improved input consistency is not evidence of better MRR; the full ranker must be evaluated with changed inputs.',
            'No new model, fingerprint coordinates, mass filter, fragmentation scores or ranking parameters were trained.']}
    dump(out/'condition-check.json',report);dump(root/'condition-check.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


def main():
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
    repo=Path(__file__).resolve().parents[1];out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    root=state/'artifacts/casmi26/baseline328-diagnostic/models';root.mkdir(parents=True,exist_ok=True)
    if sys.argv[1:]==['_execute']:execute(state,repo,out,root);return 0
    check=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests/test_conditioned_encoder.py'),
                          str(repo/'work/casmi26/tests/test_conditioned_check.py')],cwd=repo,stdin=subprocess.DEVNULL,
                         capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    (out/'tests.log').write_text(check.stdout+'\n'+check.stderr,encoding='utf-8')
    if check.returncode:raise RuntimeError('Engineering tests failed before model acquisition')
    spec=importlib.util.spec_from_file_location('prep',repo/'tasks/casmi_prepare.py');prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    env=prep.kaggle_environment(state,dict(os.environ));env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    acquisition={}
    for name,h in WEIGHTS.items():
        p=root/name
        if not p.exists():
            if shutil.disk_usage(root).free<2*BYTE_SIZE+1024**3:raise ValueError('Insufficient disk headroom')
            cmd=[str(py),'-c','from kaggle.cli import main;main()','datasets','download','-d',REF,'-f',name,'-p',str(root),'-q']
            r=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
            if r.returncode:raise RuntimeError('Public tensor acquisition failed: '+name)
            if not p.exists() and p.with_suffix(p.suffix+'.zip').is_file():
                with zipfile.ZipFile(p.with_suffix(p.suffix+'.zip')) as z:
                    entries=[i for i in z.infolist() if i.filename==name and i.file_size==BYTE_SIZE]
                    if len(entries)!=1:raise ValueError('Unexpected tensor checkpoint archive')
                    p.write_bytes(z.read(entries[0]))
        if p.stat().st_size!=BYTE_SIZE or digest(p)!=h:raise ValueError('Public checkpoint differs from scored snapshot')
        acquisition[name]={'sha256':h,'bytes':p.stat().st_size}
    dump(out/'checkpoint-identity.json',acquisition)
    clean={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN','PASSWORD','SECRET','KAGGLE','GH_'))}
    clean.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    r=subprocess.run([str(py),str(Path(__file__).resolve()),'_execute'],env=clean,stdin=subprocess.DEVNULL,
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=1200)
    (out/'model-check.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
    if r.returncode:raise RuntimeError('Isolated model behavior check failed; no fallback to unsafe loading')
    return 0

if __name__=='__main__':raise SystemExit(main())
