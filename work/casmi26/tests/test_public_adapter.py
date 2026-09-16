import importlib.util
import json
from pathlib import Path


def test_unknown_external_collision_energy_is_not_claimed_as_ev(tmp_path):
    source = Path(__file__).resolve().parents[3]/'tasks/casmi_progress.py'
    spec = importlib.util.spec_from_file_location('public_task', source)
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    (tmp_path/'MassSpecGym.tsv').write_text(
        'identifier\tmzs\tintensities\tsmiles\tprecursor_mz\tadduct\tinchikey\tcollision_energy\n'
        'x\t29,31\t0.1,1\tCCO\t47.04914127862\t[M+H]+\tLFQSCWFLJHTTHZ-UHFFFAOYSA-N\t35\n')
    path, evidence = task.public_subset(tmp_path)
    row = json.loads(path.read_text().splitlines()[0])
    assert row['collision_energy_ev'] == []
    assert row['collision_energy_orig'] == '35'
    assert row['collision_energy_orig_units'] == 'unknown'
    assert evidence['rows'] == 1


def test_user_mirror_permission_failure_keeps_persistent_outputs(tmp_path,monkeypatch):
    source = Path(__file__).resolve().parents[3]/'tasks/casmi_progress.py'
    spec=importlib.util.spec_from_file_location('public_task_delivery',source)
    task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
    assert hasattr(task,'deliver_artifacts')
    artifact=tmp_path/'source';artifact.mkdir()
    (artifact/'model.npz').write_bytes(b'example')
    target=tmp_path/'denied'
    original=Path.mkdir
    def deny(self,*args,**kwargs):
        if self==target:raise PermissionError('denied')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'mkdir',deny)
    result=task.deliver_artifacts(artifact,target)
    assert result['mirror_copy_ok'] is False
    assert result['artifact_directory']==str(artifact)
    assert (artifact/'model.npz').read_bytes()==b'example'
