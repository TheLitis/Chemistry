"""Build a diagnostic notebook; no Kaggle writes and no competition submission.

Inputs are the already reviewed, scored copy plus our experimental view helpers.
Every candidate set, public weight and ranker remains fixed across four branches.
The ordinary submission.csv is deliberately left on the frozen baseline branch.
"""
from pathlib import Path
import hashlib,json,copy,zipfile

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'baseline-anchor.ipynb'
ANCHOR='f80c2d7725fdf6288e1c788f599f39ec82530778d5e4bb9c5de1a059af370f6e'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==ANCHOR
book=json.loads(SOURCE.read_text())
original=copy.deepcopy(book)
preview=json.loads((ROOT/'evidence/anchor-preview.json').read_text())
asset_hashes={Path(n).name:v['sha256'] for n,v in preview['input_assets'].items()}

intro='''# R12: ion-view ablation against the scored 0.328 control\n
This is an unscored diagnostic experiment, not a claimed improvement.\n
Baseline scientific implementation: haideptry, *Fast Spectral Cosine Baseline*,
public V17 (Apache-2.0), using public models/features by prvsiyan. The scored
private copy obtained 0.328; the author's published 0.339 is not our result.\n
Only the neural evidence input construction varies. Candidates, library and
analog matches, graph evidence, fingerprint weights and eight rankers stay
identical. Fixed branches: frozen; aligned_rows; single_only; ion_views.\n
The `submission.csv` file retains the frozen branch. No controller automatically
submits an alternative or selects a winner from visible test data. Comparison
reports contain agreement and input-contract diagnostics, NOT test accuracy.\n
An ion-consistent input can change a merged network's training distribution;
its benefit must be measured, not assumed from physical plausibility.\n'''
book['cells'][0]['source']=intro.splitlines(True)
# Check the exact numeric assets BEFORE any scientific code/pickle is executed.
asset_check='''
import glob as _r12_glob, hashlib as _r12_hashlib, pathlib as _r12_pathlib
_R12_ASSET_HASHES = ASSETS
for _name,_expected_hash in _R12_ASSET_HASHES.items():
    _hits=sorted(_r12_glob.glob('/kaggle/input/**/'+_name,recursive=True),key=len)
    assert _hits, 'Required baseline asset missing: '+_name
    with open(_hits[0],'rb') as _fh:
        assert _r12_hashlib.file_digest(_fh,'sha256').hexdigest()==_expected_hash, 'Changed baseline asset: '+_name
'''.replace('ASSETS',repr(asset_hashes))
book['cells'][1]['source']+=asset_check.splitlines(True)

views=(ROOT/'src/spectral_views.py').read_text();models=(ROOT/'src/ion_model_views.py').read_text()
embedded='''
import types as _r12_types, sys as _r12_sys
_r12_views=_r12_types.ModuleType('spectral_views')
exec(compile(VIEWS,'<spectral_views>','exec'),_r12_views.__dict__)
_r12_sys.modules['spectral_views']=_r12_views
_r12_models=_r12_types.ModuleType('ion_model_views')
exec(compile(MODELS,'<ion_model_views>','exec'),_r12_models.__dict__)
_R12_VARIANTS=('aligned_rows','single_only','ion_views')
'''.replace('VIEWS',repr(views)).replace('MODELS',repr(models))
helper={'cell_type':'code','metadata':{},'source':embedded.splitlines(True),'outputs':[],'execution_count':None}

main=''.join(book['cells'][11]['source'])
def once(old,new):
    global main
    assert main.count(old)==1,repr(old)
    main=main.replace(old,new)
once('rows, diag = [], []','rows, diag = [], []\n_r12_rows={v:[] for v in _R12_VARIANTS}\n_r12_diagnostics=[]')
once('    smis = []','    smis = []\n    _r12_lists={}')
once('            p = rank_proba(X)', '''            p = rank_proba(X)
            # Same candidate indices and all non-neural evidence in every branch.
            _frozen_order=np.argsort(-p)[:CFG.TOPN]
            _d={'molecule_id':str(mid),'candidates':len(cand),'spectra':len(sub),
                'adducts':sorted(set(sub.adduct)),'modes':sorted(set(sub.ionization_mode)),'variants':{}}
            for _variant in _R12_VARIANTS:
                _zv=_r12_models.model_logits_variant(globals(),sub,_variant)
                assert _zv is not None and np.isfinite(_zv).all(), 'Missing neural branch'
                _xv=rank_features(cfp,lv,afp,np.array(sims,np.float32),_zv,fsc)[:,:NFEAT]
                _pv=rank_proba(_xv)
                assert np.isfinite(_pv).all()
                _ov=np.argsort(-_pv)[:CFG.TOPN]
                _r12_lists[_variant]=[pool.smiles[cand[i]] for i in _ov]
                _d['variants'][_variant]={'same_top1':bool(_ov[0]==_frozen_order[0]),
                    'same_top25_order':bool(np.array_equal(_ov,_frozen_order)),
                    'fp_logits_l2':float(np.linalg.norm(np.asarray(_zv)-np.asarray(zlog))),
                    'shared_top25':len(set(map(int,_ov))&set(map(int,_frozen_order)))}
            _r12_diagnostics.append(_d)''')
once("    rows.append((mid, ';'.join(smis[:CFG.TOPN])))",'''    rows.append((mid, ';'.join(smis[:CFG.TOPN])))
    for _variant in _R12_VARIANTS:
        _guesses=_r12_lists.get(_variant,smis)
        _r12_rows[_variant].append((mid,';'.join(_guesses[:CFG.TOPN])))''')
book['cells'][11]['source']=main.splitlines(True)

footer='''
import json as _r12_json, hashlib as _r12_h, pathlib as _r12_p
import rdkit as _r12_rdkit
assert _r12_rdkit.__version__=='2026.03.3'
assert _MODEL is not None and len(RANKERS)==8
_expected_ids=set(map(str,te.molecule_id))
_report={'experiment':'R12-fixed-public-baseline-ion-views','status':'diagnostic_completed',
    'incumbent_public_score':0.328,'new_official_score':None,'accuracy_established':False,
    'same_candidates_and_non_neural_features':True,'new_weight_training':False,
    'rankers_refit_as_in_baseline':True,'test_labels_used':False,'new_submissions':0,
    'baseline_source_sha256':ANCHOR_VALUE,'variants':{},'details':_r12_diagnostics,
    'limitations':['Visible test predictions do not measure hidden accuracy.',
        'Author neural pretraining membership has not been independently reconstructed.',
        'Ion-consistent merging may differ from the merged head training distribution.',
        'No alternative is automatically promoted.']}
assert set(map(str,submission.molecule_id))==_expected_ids and not submission.molecule_id.duplicated().any()
_report['baseline_csv_sha256']=_r12_h.sha256(_r12_p.Path('submission.csv').read_bytes()).hexdigest()
for _variant in _R12_VARIANTS:
    _df=pd.DataFrame(_r12_rows[_variant],columns=['molecule_id','smiles'])
    _df['smiles']=_df.smiles.apply(pad_candidates)
    assert set(map(str,_df.molecule_id))==_expected_ids and not _df.molecule_id.duplicated().any()
    assert _df.smiles.notna().all()
    for _s in _df.smiles:
        _ss=_s.split(';')
        assert 1<=len(_ss)<=25 and all(_x and Chem.MolFromSmiles(_x) is not None for _x in _ss)
    _path='ablation-'+_variant+'.csv';_df.to_csv(_path,index=False)
    _v=[d['variants'][_variant] for d in _r12_diagnostics]
    _report['variants'][_variant]={'rows':len(_df),'evaluated_groups':len(_v),
        'changed_top1':sum(not r['same_top1'] for r in _v),
        'changed_top25_order':sum(not r['same_top25_order'] for r in _v),
        'csv_sha256':_r12_h.sha256(_r12_p.Path(_path).read_bytes()).hexdigest()}
_report['seconds']=_repro_time.monotonic()-_REPRO_START
_r12_p.Path('ion-ablation.json').write_text(_r12_json.dumps(_report,indent=2,allow_nan=False)+'\\n',encoding='utf-8')
print(_r12_json.dumps({k:v for k,v in _report.items() if k!='details'},indent=2))
'''.replace('ANCHOR_VALUE',repr(ANCHOR))
# No dashboard and no stale reproduction-identity assertions.
book['cells']=book['cells'][:11]+[helper,book['cells'][11],{'cell_type':'code','metadata':{},'source':footer.splitlines(True),'outputs':[],'execution_count':None}]
for i,c in enumerate(book['cells']):
    c['id']='r12-'+str(i)
    if c['cell_type']=='code':
        c['outputs']=[];c['execution_count']=None
        compile(''.join(c['source']),'<cell-'+str(i)+'>','exec')
out=ROOT/'r12-ion-view-ablation.ipynb';out.write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n')
meta={'id':'thelindortis/casmi26-r12-ion-view-diagnostic','title':'CASMI26 R12 Ion View Diagnostic','code_file':out.name,
      'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,
      'dataset_sources':['prvsiyan/casmi26-fp-models-v2','aidensong123/casmi26-offline-rdkit-2026033',
                         'prvsiyan/casmi26-ranker-features','prvsiyan/coconut-casmi26-candidates'],
      'competition_sources':['enveda-CASMI26-molecule-id-mass-spectra'],'kernel_sources':[],'model_sources':[]}
(ROOT/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
manifest={'status':'built_not_executed_on_real_models','source_sha256':ANCHOR,
          'notebook_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'assets':asset_hashes,
          'baseline_scientific_cells_unchanged_except_diagnostic_loop':True,
          'private_notebook_uploaded':False,'contest_submission_made':False}
(ROOT/'evidence/notebook-build.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='assets'},indent=2))
