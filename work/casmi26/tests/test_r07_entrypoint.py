import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest


def fixture(tmp_path):
    p=Path(__file__).with_name('test_r07_release.py')
    spec=importlib.util.spec_from_file_location('r07_fixture',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.fixture(tmp_path)


@pytest.mark.parametrize('joined',[False,True])
def test_root_command_runs_r07_without_legacy_dispatch(tmp_path,joined):
    bundle,train,test,_=fixture(tmp_path)
    entry=Path(__file__).resolve().parents[3]/'predict.py';output=tmp_path/'out.csv'
    flag=['--bundle='+str(bundle)] if joined else ['--bundle',str(bundle)]
    r=subprocess.run([sys.executable,str(entry),'--test',str(test),'--train',str(train),
       '--output',str(output),'--workers','1']+flag,capture_output=True,text=True,cwd=tmp_path)
    assert r.returncode==0,r.stderr
    data=json.loads(output.with_suffix('.csv.report.json').read_text())
    assert data['format']==7 and data['test_spectra']==3 and data['prediction_count']==2


def test_ambiguous_r07_legacy_bundle_is_not_guessed(tmp_path):
    bundle,_,_,_=fixture(tmp_path)
    (bundle/'bundle.json').write_text('{}')
    entry=Path(__file__).resolve().parents[3]/'predict.py'
    r=subprocess.run([sys.executable,str(entry),'--bundle',str(bundle),'--help'],capture_output=True,text=True)
    assert r.returncode==2 and 'ambiguous' in r.stderr.lower()
