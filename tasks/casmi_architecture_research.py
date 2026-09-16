"""Execute the sealed R06 architecture tournament on ChemistryPC; no submissions."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('screen','refine','audit','all'),default='all');args=parser.parse_args()
    state=Path(os.environ['CHEMISTRY_STATE_ROOT']);py=state/'envs/casmi26/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),str(Path(__file__).resolve())]+sys.argv[1:],env={**os.environ,
            'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    if not torch.cuda.is_available():raise RuntimeError('R06 requires the authorized local CUDA device')
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'work/casmi26'))
    from casmi26.architectures import ARCHITECTURES,hard_negative_indices,rank_metrics
    from casmi26.architecture_training import Data,TorchAnchor,train_trial,predict_trial,evaluate_predictions,rank_from_basis
    from casmi26.production import sha256,write_json,is_validation
    from casmi26.ranking import paired_effect
    from casmi26.metric import require_official_rdkit
    require_official_rdkit()
    art=state/'artifacts/casmi26';root=art/'research-r06';cache=state/'cache/casmi26/architecture-r06'
    out=Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();plan=json.loads((root/'protocol.json').read_text());prepared=json.loads((cache/'prepared.json').read_text())
    if prepared['source']!=plan['source']:raise RuntimeError('Protocol and cache lineage mismatch')
    for name,digest in prepared['files'].items():
        if sha256(cache/name)!=digest:raise RuntimeError('Corrupted R06 cache: '+name)
    anchor_path=art/'research-v3/multitarget.npz'
    if sha256(anchor_path)!=plan['source']['anchor_sha256']:raise RuntimeError('Frozen anchor changed')
    checks=subprocess.run([str(py),'-m','pytest','-q',str(repo/'work/casmi26/tests')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300)
    (out/'tests.log').write_text(checks.stdout+'\n'+checks.stderr,encoding='utf-8');print(checks.stdout,flush=True)
    if checks.returncode:raise RuntimeError('Tests failed before training')
    data=Data(cache);anchor=TorchAnchor(anchor_path).cuda().eval()
    old=state/'cache/casmi26/highres-v3';catalog=json.loads((old/'catalog.json').read_text());packed=np.load(old/'targets.npy',mmap_mode='r',allow_pickle=False)
    if sha256(old/'catalog.json')!=plan['source']['catalog_sha256'] or sha256(old/'targets.npy')!=plan['source']['targets_sha256']:
        raise RuntimeError('Candidate universe changed')
    trainids=np.array([data.index[k] for k in plan['training_keys']],dtype=np.int64)
    if any(is_validation(data.keys[int(i)]) for i in trainids):raise RuntimeError('Training contains a heldout molecular key')
    if (data.view_counts[trainids]<1).any():raise RuntimeError('Some predeclared training keys have no valid observations')
    splitids={name:[data.index[k] for k in plan[name+'_keys']] for name in ('screen','selection','audit')}
    if set(trainids)&set(splitids['screen']+splitids['selection']+splitids['audit']):raise RuntimeError('Split overlap')
    configs={name:{'architecture':name,'loss':'balanced'} for name in ARCHITECTURES}
    configs['massset_hybrid+rank']={'architecture':'massset_hybrid','loss':'rank'}
    configs['massset_hybrid+unweighted']={'architecture':'massset_hybrid','loss':'unweighted'}
    trialdir=root/'trials';trialdir.mkdir(exist_ok=True)
    grid={'weights':[[.5,.25,.25],[1.,0.,0.],[1.,1.,1.]],'sigma_ppm':[None,2.,5.,10.,20.],
          'pooling':['early','late_probability'],'finalist_ensembles':['individual','top_two_equal','two_seeds_equal'],
          'no_change_anchor_included':True,'chosen_without_audit_results':True,
          'seed_replication':'repeat the complete screening-plus-refinement exposure with the second seed',
          'no_all_acquisition_late_fusion_in_cache':True}
    gridpath=root/'selection-grid-before-training.json'
    if gridpath.exists() and json.loads(gridpath.read_text())!=grid:raise RuntimeError('Selection grid changed')
    write_json(gridpath,grid)
    masses=np.array([catalog[int(i)][3] for i in data.catalog_indices])
    screenids=trainids[:plan['screen_training_molecules']]
    def fit(name,config,ids,epochs,seed,initial=None):
        negatives=hard_negative_indices(masses,data.keys,ids,7) if config['loss']=='rank' else None
        return train_trial(data,anchor,ids,config,trialdir/(name+'.npz'),epochs=epochs,seed=seed,
             initial=initial,negatives=negatives,batch_size=plan['batch_size'],lr=plan['learning_rate'])
    def measure(ids,path=None,budget=3,pooling='early'):
        logits,mass=predict_trial(data,anchor,ids,path,budget=budget,pooling=pooling)
        values=evaluate_predictions(logits,mass,[data.keys[i] for i in ids],catalog,packed)
        return {**rank_metrics(values['ranks']),'candidate_coverage':float(np.mean(values['coverage']))},values
    if args.stage in ('screen','all'):
        file=root/'screen-results.json'
        if not file.exists():
            baseline,_=measure(splitids['screen']);results={};failures={}
            for name,config in configs.items():
                try:
                    training=fit('screen-'+name,config,screenids,plan['screen_epochs'],plan['screen_seed'])
                    metric,detail=measure(splitids['screen'],trialdir/('screen-'+name+'.npz'))
                    results[name]={'training':training,'screen':metric,'ranks':detail['ranks']}
                    print('R06_SCREEN_RESULT '+json.dumps({'configuration':name,**metric,'train_seconds':training['seconds']}),flush=True)
                except (RuntimeError,ValueError) as exc:
                    failures[name]={'type':type(exc).__name__,'message':str(exc)[:1200]}
                    print('R06_SCREEN_FAILED '+name+' '+str(exc)[:300],flush=True);torch.cuda.empty_cache()
                write_json(root/'screen-progress.json',{'results':results,'failures':failures})
            if len(results)<2:raise RuntimeError('Fewer than two valid trained configurations')
            order=sorted(results,key=lambda name:(-results[name]['screen']['mrr_at_25'],name))
            summary={'status':'completed','baseline':baseline,'results':results,'failures':failures,'advance':order[:2],
                     'screen_keys_sha256':hashlib.sha256('\n'.join(plan['screen_keys']).encode()).hexdigest(),
                     'selected_before_refinement_and_audit':True,'audit_data_evaluated':False,'new_submissions':0}
            write_json(file,summary)
        shutil.copy2(file,out/file.name)
        print('R06_SCREEN_COMPLETE '+json.dumps({k:v for k,v in json.loads(file.read_text()).items() if k not in ('results',)}),flush=True)
    if args.stage in ('refine','all'):
        file=root/'refinement-results.json'
        if not file.exists():
            screen=json.loads((root/'screen-results.json').read_text());results={};finalpaths={}
            for name in screen['advance']:
                config=configs[name];initial=trialdir/('screen-'+name+'.npz')
                r=fit('full-'+name,config,trainids,plan['refinement_epochs'],plan['screen_seed'],initial)
                results['full-'+name]=r;finalpaths[name]='full-'+name+'.npz'
            chosen=screen['advance'][0];config=configs[chosen]
            fit('replica-screen',config,screenids,plan['screen_epochs'],plan['replication_seed'])
            results['replica-full']=fit('replica-full',config,trainids,plan['refinement_epochs'],plan['replication_seed'],trialdir/'replica-screen.npz')
            finalpaths['replica']='replica-full.npz'
            write_json(file,{'status':'completed','finalpaths':finalpaths,'results':results,'best_screen_configuration':chosen,
                             'audit_data_evaluated':False,'new_submissions':0})
        shutil.copy2(file,out/file.name);print('R06_REFINEMENT_COMPLETE',flush=True)
    if args.stage in ('audit','all'):
        file=root/'report.json'
        if file.exists():
            shutil.copy2(file,out/'report.json');print('R06_FINISHED_RESULTS_REUSED',flush=True);return 0
        refinement=json.loads((root/'refinement-results.json').read_text());screen=json.loads((root/'screen-results.json').read_text())
        paths=refinement['finalpaths'];first,second=screen['advance']
        combinations={'anchor':[],first:[paths[first]],second:[paths[second]],'replica':[paths['replica']],
                      'ensemble_top_two':[paths[first],paths[second]],'ensemble_two_seeds':[paths[first],paths['replica']]}
        for name in set(p for pp in combinations.values() for p in pp):
            meta=json.loads((trialdir/name).with_suffix('.json').read_text())
            if meta['checkpoint_sha256']!=sha256(trialdir/name) or meta['mask_sha256']!=sha256((trialdir/name).with_suffix('.active.npy')):
                raise RuntimeError('Trained trial changed before selection')
        ids=splitids['selection'];truth=[data.keys[i] for i in ids];calculated={};calmetrics={};calpaths={}
        for pooling in grid['pooling']:
            base,mass=predict_trial(data,anchor,ids,pooling=pooling);prediction={'anchor':base}
            for name in set(p for pp in combinations.values() for p in pp):
                prediction[name],_=predict_trial(data,anchor,ids,trialdir/name,pooling=pooling)
            for label,names in combinations.items():
                logits=np.mean([prediction[n] for n in names],axis=0) if names else base
                values=evaluate_predictions(logits,mass,truth,catalog,packed,return_basis=True)
                for weights in grid['weights']:
                    for sigma in grid['sigma_ppm']:
                        spec={'component':label,'models':names,'pooling':pooling,'weights':weights,'sigma_ppm':sigma}
                        key=json.dumps(spec,sort_keys=True,separators=(',',':'))
                        ranks=[rank_from_basis(*b,weights,sigma) for b in values['basis']]
                        calmetrics[key]=rank_metrics(ranks);calpaths[key]=spec
        winner=max(calmetrics,key=lambda k:calmetrics[k]['mrr_at_25'])
        selection={'configuration':calpaths[winner],'calibration':calmetrics[winner],'grid_size':len(calmetrics),
                   'all_calibration_results':[{'configuration':calpaths[k],'metrics':v} for k,v in calmetrics.items()],
                   'audit_keys_sha256':hashlib.sha256('\n'.join(plan['audit_keys']).encode()).hexdigest(),
                   'selected_before_audit':True,'model_hashes':{n:sha256(trialdir/n) for n in calpaths[winner]['models']}}
        selection_path=root/'selection-before-audit.json'
        if selection_path.exists():
            if json.loads(selection_path.read_text())!=selection:raise RuntimeError('Sealed selection differs; do not retune the audit')
        else:write_json(selection_path,selection)
        print('R06_LOCKED_SELECTION '+json.dumps(selection['configuration']),flush=True)
        selected=selection['configuration'];audit={};raw=[];ids=splitids['audit'];truth=plan['audit_keys']
        for budget in (1,3,'all'):
            pooling=selected['pooling'] if budget!='all' else 'early'
            base,mass=predict_trial(data,anchor,ids,budget=budget,pooling='early')
            baseline=evaluate_predictions(base,mass,truth,catalog,packed)
            if selected['models']:
                preds=[predict_trial(data,anchor,ids,trialdir/name,budget=budget,pooling=pooling)[0] for name in selected['models']]
                logits=np.mean(preds,axis=0)
            else:logits=predict_trial(data,anchor,ids,budget=budget,pooling=pooling)[0]
            values=evaluate_predictions(logits,mass,truth,catalog,packed,weights=selected['weights'],sigma_ppm=selected['sigma_ppm'])
            a=np.array(baseline['ranks']);b=np.array(values['ranks']);ra=np.where(a>0,1/np.maximum(a,1),0);rb=np.where(b>0,1/np.maximum(b,1),0)
            audit[str(budget)]={'anchor':rank_metrics(a),'selected':rank_metrics(b),'paired':paired_effect(ra,rb),
                'candidate_coverage':float(np.mean(values['coverage'])),'mean_candidates':float(np.mean(values['candidate_counts'])),
                'pooling':pooling,'all_budget_is_diagnostic_only':budget=='all','strata':{}}
            for label,flag in [('bruker',1),('positive',2),('negative',4),('np_source',8)]:
                ii=[q for q,i in enumerate(ids) if int(data.strata[i])&flag]
                if ii:audit[str(budget)]['strata'][label]={'anchor':rank_metrics(a[ii]),'selected':rank_metrics(b[ii])}
            raw.extend({'key':k,'budget':str(budget),'anchor_rank':int(x),'selected_rank':int(y),'in_catalog_window':bool(c)} for k,x,y,c in zip(truth,a,b,values['coverage']))
            print('R06_AUDIT '+str(budget)+' '+json.dumps({k:v for k,v in audit[str(budget)].items() if k!='strata'}),flush=True)
        report={'status':'completed','experiment':'R06-architecture-tournament','commit':os.environ.get('GITHUB_SHA'),
                'architectures':len(ARCHITECTURES),'screen_configurations':len(configs),'screen_completed':len(screen['results']),
                'screen_failures':screen['failures'],'screen_training_molecules':len(screenids),'refinement_training_molecules':len(trainids),
                'screen_molecules':len(plan['screen_keys']),'selection_molecules':len(plan['selection_keys']),'audit_molecules':len(ids),
                'excluded_prior_keys':plan['excluded_prior_keys'],'training_audit_key_overlap':0,
                'screen_metrics':{k:v['screen'] for k,v in screen['results'].items()},
                'screen_training':{k:v['training'] for k,v in screen['results'].items()},'refinement':refinement,
                'selection':{k:v for k,v in selection.items() if k!='all_calibration_results'},'audit':audit,
                'supported_improvement':audit['3']['paired']['ci95'][0]>0,'new_submissions':0,'official_score':None,
                'test_data_read':False,'production_champion_changed':False,'tests':checks.stdout.strip(),
                'stage_seconds':time.monotonic()-started,
                'limitations':['Finite compact residual architecture comparison on a common pretrained anchor, not all possible architectures.',
                    'Candidate catalog contains known structures; this is not a de novo generation benchmark.',
                    'Raw set branch retains top 32 peaks from up to three acquisitions; vector input compresses spectra.',
                    'Validation acceptance does not use the true mass; empty or uncovered candidate sets count as failures.',
                    'All-budget diagnostic uses all-spectrum vector means, at most three token acquisitions, and early pooling.',
                    'Bruker/NP source flags mean at least one available spectrum, not pure independent instrument cohorts.',
                    'Calibration grid is separate from the 2048-key audit; public Kaggle scores were not used for selection.']}
        write_json(file,report);write_json(root/'audit-ranks.json',raw)
        write_json(root/'calibration-keys.json',plan['screen_keys']+plan['selection_keys'])
        for name in ('report.json','selection-before-audit.json','audit-ranks.json','screen-results.json','refinement-results.json'):
            shutil.copy2(root/name,out/name)
        print('R06_REPORT_BEGIN\n'+json.dumps({k:v for k,v in report.items() if k not in ('screen_training','refinement')},indent=2)+'\nR06_REPORT_END',flush=True)
    subprocess.run(['git','-C',str(repo),'archive','--format=zip','--output='+str(out/'source.zip'),'HEAD'],check=True,timeout=60)
    return 0


if __name__=='__main__':raise SystemExit(main())
