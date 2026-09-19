import importlib.util
from pathlib import Path
import pytest


def mod():
    p=Path(__file__).resolve().parents[3]/'tasks/casmi_public_delivery.py'
    assert p.is_file(),'Canonical delivery not implemented'
    s=importlib.util.spec_from_file_location('delivery',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def book():return {'cells':[{'cell_type':'code','source':['x=1\n'],'outputs':[]} ]}


def metadata(m):
    return {'id':m.CANONICAL,'id_no':123,'is_private':True,'enable_internet':False,
            'competition_sources':[m.COMPETITION],'dataset_sources':list(m.INPUTS),'kernel_sources':[],'model_sources':[]}


def test_canonical_name_requires_both_metadata_and_exact_code():
    m=mod();got=m.canonical_identity(book(),book(),metadata(m))
    assert got['kernel']==m.CANONICAL and got['id_no']==123


@pytest.mark.parametrize('field,value',[('id','other/notebook'),('is_private',False),('enable_internet',True),('id_no',0),('dataset_sources',['unreviewed/file'])])
def test_reject_wrong_or_public_target(field,value):
    m=mod();d=metadata(m);d[field]=value
    with pytest.raises(ValueError):m.canonical_identity(book(),book(),d)


def test_changed_scientific_or_audit_code_is_not_reconciled():
    m=mod();b=book();b['cells'][0]['source']=['x=2\n']
    with pytest.raises(ValueError):m.canonical_identity(book(),b,metadata(m))


def test_observed_outputs_not_treated_as_changed_source():
    m=mod();b=book();b['cells'][0]['outputs']=[{'text':'anything'}]
    assert m.canonical_identity(book(),b,metadata(m))['kernel']==m.CANONICAL
