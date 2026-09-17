import gzip,json,os,sqlite3,subprocess,sys,zlib
from pathlib import Path


def test_evaluator_freezes_on_calibration_and_replays_identically(tmp_path):
    from casmi26.production import sha256
    records=[];requests=[];predictions=[]
    for key,good in [('calibration',True),('old-audit',False),('fresh-audit',False)]:
        wrong=key+'-wrong'
        case={'keys':[wrong,key],'scores':[.6,.5]}
        records.append({'key':key,'queries':[{'peaks':[[20,1]],'precursor':100,'adduct':'[M+H]+','ce':[20]}],
                        'frontier':[key,wrong],'cases':{name:dict(case) for name in ['available','absent','external_recovery']}})
        for k,mz in [(key,20 if good else 30),(wrong,30 if good else 20)]:
            requests.append({'key':k})
            predictions.append({'key':k,'status':'predicted','spectra':[{'adduct':'[M+H]+','energy':20,'peaks':[[mz,1]]}]})
    plan={'experiment':'software-fixture','calibration_keys':['calibration'],'reused_np_audit_keys':['old-audit'],
          'fresh_transfer_keys':['fresh-audit'],'features':['cosine_nearest'],'forward_weights':[0.,1.],
          'shortlist_per_regime':2,'fitting_overlap':0}
    (tmp_path/'protocol.json').write_text(json.dumps(plan));(tmp_path/'requests.json').write_text(json.dumps(requests))
    (tmp_path/'evidence.json.gz').write_bytes(gzip.compress(json.dumps(records).encode(),mtime=0))
    (tmp_path/'prepared.json').write_text(json.dumps({'files':{n:sha256(tmp_path/n) for n in ['protocol.json','requests.json','evidence.json.gz']}}))
    (tmp_path/'simulation.json').write_text(json.dumps({'status':'simulation_completed','identity':{'requests':sha256(tmp_path/'requests.json')}}))
    (tmp_path/'graph-parity.json').write_text('{}')
    with sqlite3.connect(tmp_path/'simulations.sqlite') as db:
        db.execute('CREATE TABLE predictions(k TEXT PRIMARY KEY,content BLOB)')
        db.executemany('INSERT INTO predictions VALUES(?,?)',[(r['key'],zlib.compress(json.dumps(r).encode())) for r in predictions])
    entry=Path(__file__).resolve().parents[3]/'tasks/casmi_r08_evaluate.py'
    p=subprocess.run([sys.executable,str(entry),'--root',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['selected']['weight']==1.
    fresh=report['cohorts']['fresh_timsTOF_transfer']['absent']
    assert fresh['selected']['mrr_at_25']==.5
    selection=(tmp_path/'selection-before-transfer-audit.json').read_bytes()
    import time
    time.sleep(1.1)
    p=subprocess.run([sys.executable,str(entry),'--root',str(tmp_path)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert (tmp_path/'selection-before-transfer-audit.json').read_bytes()==selection
