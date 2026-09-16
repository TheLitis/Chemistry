import csv
import json
import math
import numpy as np
import pytest
from test_pipeline import fixture_data, row, write_jsonl


def test_official_schema_and_energy_list():
    from casmi26.spectra import from_row
    s=from_row({'molecule_id':'m_01','spectrum_id':'s_45', 'precursor_mz':47.04914127862,
       'adduct':'[M+H]+', 'ms2_mzs':[29.,31.], 'ms2_normalized_intensities':[.1,1.],
       'normalized_smiles':'CCO','collision_energy_ev':[20,40,60],
       'collision_energy_orig_units':'eV', 'ionization_mode':'positive'}, labeled=True)
    assert s.compound_id=='m_01'
    assert s.smiles=='CCO'
    assert s.collision_energies == (20.,40.,60.)
    assert s.collision_energy is None


@pytest.mark.parametrize('adduct,delta', [('[M-H2O+H]+',1.007276466621-18.01056468403),
    ('[M-2H2O+H]+',1.007276466621-2*18.01056468403),
    ('[M-H2O-H]-',-18.01056468403-1.007276466621),
    ('[M+CH2O2-H]-',46.0054793036-1.007276466621)])
def test_documented_adducts(adduct,delta):
    from casmi26.chemistry import neutral_mass
    assert neutral_mass(350+delta,adduct)==pytest.approx(350,abs=1e-8)


def test_metric_tautomer_and_stereo_equivalence():
    from casmi26.metric import structure_key
    assert structure_key('CC(=O)C') == structure_key('C=C(O)C')
    assert structure_key('C[C@H](O)F') == structure_key('C[C@@H](O)F')
    assert structure_key('CCO') != structure_key('COC')
    assert len(structure_key('CCO')) == 14
    assert structure_key('not-smiles') is None


def test_mrr_first_correct_rank_not_tanimoto():
    from casmi26.metric import mrr_at_25
    result=mrr_at_25({'a':'CCO','b':'CC(=O)C','c':'CCN'},
                     {'a':['COC','CCO'], 'b':['C=C(O)C'], 'c':['CO']})
    assert result['mrr_at_25']==pytest.approx(.5)
    assert result['top1_accuracy']==pytest.approx(1/3)
    assert result['recall_at_25']==pytest.approx(2/3)
    assert result['official_score'] is None


def test_mrr_preserves_invalid_and_duplicate_ranks():
    from casmi26.metric import mrr_at_25
    r=mrr_at_25({'a':'CCO'},{'a':['bad','COC','COC','CCO']})
    assert r['mrr_at_25']==.25
    with pytest.raises(ValueError): mrr_at_25({'a':'CCO'},{'b':['CCO']})
    with pytest.raises(ValueError): mrr_at_25({'a':'CCO'},{'a':['C']*26})


def test_deduplicate_equivalent_tautomers_not_just_strings():
    from casmi26.metric import distinct_guesses
    assert len(distinct_guesses(['CC(=O)C','C=C(O)C','CCO','bad'],25)) == 2


def test_ranked_submission_semicolons_and_all_molecules(tmp_path):
    from casmi26.engine import Config,predict
    train,test,template=fixture_data(tmp_path)
    template.write_text('molecule_id,smiles\n001,C\n002,C\n')
    out=tmp_path/'submission.csv'
    result=predict(test,train,template,out,config=Config(top_k=25))
    rows=list(csv.DictReader(out.open()))
    assert rows[0]['smiles'].split(';')[0]=='CCO'
    assert set(rows[0]['smiles'].split(';'))=={'CCO','COC'}
    assert rows[1]['smiles'].split(';')[0]=='COC'
    assert result['metric_contract']=='MRR@25/tautomer-InChIKey14'
    assert result['official_score'] is None


def test_training_curation_is_counted_not_silent(tmp_path):
    from casmi26.engine import Config,predict
    train,test,template=fixture_data(tmp_path)
    with train.open('a') as f:
        f.write(json.dumps(row('bad1',[(31,1)],'bad-smiles'))+'\n')
        f.write(json.dumps(row('bad2',[(31,1)],'CCO',precursor=500))+'\n')
        f.write(json.dumps(row('bad3',[(31,1)],'CCO',adduct='[M+UNSUPPORTED]+'))+'\n')
    report=predict(test,train,template,tmp_path/'submission.csv',config=Config(curate_training=True))
    counts=report['training_curation']
    assert counts['seen']==6
    assert counts['accepted']==3
    assert sum(counts['rejected'].values())==3


def test_tautomer_equivalent_structures_share_fold():
    from casmi26.engine import structure_fold
    for n in (3,5,7,11):
        assert structure_fold('CC(=O)C',n)==structure_fold('C=C(O)C',n)


def test_metric_version_guard():
    from casmi26.metric import require_official_rdkit
    from rdkit import rdBase
    if rdBase.rdkitVersion != '2026.03.3':
        with pytest.raises(RuntimeError,match='2026.03.3'): require_official_rdkit()
    else:
        require_official_rdkit()


def test_test_polarity_disagreement_is_not_silent():
    from casmi26.spectra import from_row
    r=row('a',[(31,1)],ionization_mode='negative')
    with pytest.raises(ValueError,match='polarity'):from_row(r)


def test_offline_notebook_embeds_source_and_never_downloads_at_inference(tmp_path):
    from casmi26.notebook import build_notebook
    p=tmp_path/'submission.ipynb'
    build_notebook(p)
    doc=json.loads(p.read_text())
    assert doc['nbformat']==4
    code='\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code')
    assert '--no-index' in code
    assert 'molecule_id' in code and 'submission.csv' in code
    assert '--allow-rdkit-version-mismatch' not in code
    for cell in doc['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'cell','exec')
