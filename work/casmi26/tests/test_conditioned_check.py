import ast
import importlib.util
from pathlib import Path
import pytest

def mod():
 p=Path(__file__).resolve().parents[3]/'tasks/casmi_conditioned_encoder_check.py'
 s=importlib.util.spec_from_file_location('conditioned_check',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def text():
 c=mod()
 return 'MAX_PEAKS_NN = 128\nADDUCT_LIST = ["<unk>"]\nINSTR_LIST = ["other"]\n'+'\n'.join('def '+name+'():\n pass\n' for name in c.NAMES)

def test_top_level_calls_are_never_selected():
 c=mod();tree,constants=c.definition_ast(text()+'\nraise RuntimeError("DO_NOT_EXECUTE")\n')
 assert all(isinstance(n,(ast.FunctionDef,ast.ClassDef)) for n in tree.body)
 assert len(tree.body)==len(c.NAMES)
 compile(tree,'<isolated>','exec')

@pytest.mark.parametrize('kind',['missing','duplicate','budget'])
def test_reviewed_source_contract_rejects_drift(kind):
 bad={'missing':text().replace('def FPNet','def NotFPNet'),'duplicate':text()+'\ndef FPNet():\n pass\n','budget':text().replace('128','256')}[kind]
 with pytest.raises(ValueError):mod().definition_ast(bad)

def test_refuses_unhashed_notebook_before_load(tmp_path):
 p=tmp_path/'source.ipynb';p.write_text('{}')
 with pytest.raises(ValueError,match='bytes'):mod().reviewed_namespace(p)
