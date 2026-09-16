from pathlib import Path
import pytest
from test_pipeline import row,write_jsonl


def test_training_groups_tautomers_and_splits_without_leakage(tmp_path):
    from casmi26.training import prepare_examples
    p=write_jsonl(tmp_path/'a.jsonl',[row('a',[(31,1)],'CCO'),row('b',[(29,1)],'OCC'),row('c',[(31,1)],'COC')])
    examples,stats=prepare_examples(p,max_molecules=20)
    assert len(examples)==2
    assert sorted(x['spectra_count'] for x in examples.values())==[1,2]
    assert stats['accepted_spectra']==3


def test_catalog_evaluation_includes_failures_in_denominator():
    from casmi26.training import summarize_ranks
    r=summarize_ranks([1,2,None,4])
    assert r['mrr_at_25']==(1+.5+0+.25)/4
    assert r['top1_accuracy']==.25
    assert r['recall_at_25']==.75


def test_molecule_sampling_is_independent_of_input_order(tmp_path):
    from casmi26.training import prepare_examples
    from casmi26.chemistry import info, PROTON
    structures=['CCO','COC','CCN','CCC','CCCO','CC(C)O','CCCN','CCCC']
    rows=[row(str(i),[(20,1)],s,precursor=info(s).mass+PROTON) for i,s in enumerate(structures)]
    a,_=prepare_examples(write_jsonl(tmp_path/'a.jsonl',rows),max_molecules=3)
    b,_=prepare_examples(write_jsonl(tmp_path/'b.jsonl',list(reversed(rows))),max_molecules=3)
    assert set(a)==set(b)
