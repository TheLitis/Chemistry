from pathlib import Path
import csv
import json
import numpy as np
import pytest


def test_confident_entrypoint_exists():
    assert (Path(__file__).resolve().parents[1]/'casmi26/confident.py').is_file()


def setup_case(tmp_path, exact_reference, favored='COC'):
    pa=pytest.importorskip('pyarrow');import pyarrow.parquet as pq
    from casmi26 import production as p,portable
    raw=[p.molecule_record(s) for s in ('CCO','COC')];raw.sort(key=lambda r:(r[3],r[2],r[0]))
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'catalog.json').write_text(json.dumps([list(r[:4]) for r in raw]))
    packed=np.array([np.frombuffer(bytes.fromhex(r[4]),dtype=np.uint8) for r in raw])
    np.save(bundle/'fingerprints.npy',packed)
    target=np.unpackbits(packed[next(i for i,r in enumerate(raw) if r[0]==favored)])
    np.savez(bundle/'model.npz',version=np.array(1),w1=np.zeros((2,4104),np.float32),b1=np.zeros(2,np.float32),
             w2=np.zeros((2048,2),np.float32),b2=np.where(target,3.,-3.).astype(np.float32))
    def row(smi,peaks):
        return dict(normalized_smiles=smi,ms2_mzs=peaks,ms2_normalized_intensities=[10.,1.],
                    precursor_mz=47.0491412786,adduct='[M+H]+',collision_energy_ev=[20.])
    rows=[row('CCO',[31.,29.])]
    if exact_reference:rows.append(row('COC',[45.,29.]))
    train=tmp_path/'train.parquet';pq.write_table(pa.Table.from_pylist(rows),train)
    test=tmp_path/'test.parquet';query={**row('UNTRUSTED_ANSWER',[45.,29.]),'molecule_id':'hidden-new'}
    pq.write_table(pa.Table.from_pylist([query]),test)
    (bundle/'bundle.json').write_text(json.dumps({'format':1,'version':p.VERSION,
         'source':{'train_sha256':p.sha256(train)},'files':{n:p.sha256(bundle/n) for n in portable.ASSET_NAMES}}))
    return test,train,bundle


def test_exact_reference_can_override_neural(tmp_path):
    test,train,bundle=setup_case(tmp_path,True,'CCO')
    from casmi26.confident import infer
    out=tmp_path/'submission.csv';report=infer(test,train,bundle,out,threshold=.95,margin=.05)
    rows=list(csv.DictReader(out.open()))
    assert rows[0]['smiles'].split(';')[0]=='COC'
    assert report['gate_activations']==1 and report['official_score'] is None


def test_absent_reference_uses_neural_and_current_hidden_ids(tmp_path):
    test,train,bundle=setup_case(tmp_path,False)
    from casmi26.confident import infer
    template=tmp_path/'sample_submission.csv';template.write_text('molecule_id,smiles\nold,CCO\n')
    out=tmp_path/'submission.csv';report=infer(test,train,bundle,out,template)
    rows=list(csv.DictReader(out.open()))
    assert rows[0]['molecule_id']=='hidden-new'
    assert rows[0]['smiles'].split(';')[0]=='COC'
    assert report['gate_activations']==0 and report['test_spectra']==1


def test_confident_pipeline_cannot_overwrite_input(tmp_path):
    test,train,bundle=setup_case(tmp_path,False)
    from casmi26.confident import infer
    with pytest.raises(ValueError,match='overwrite'):infer(test,train,bundle,test)
