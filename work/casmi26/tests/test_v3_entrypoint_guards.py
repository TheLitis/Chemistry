"""Executable entry-point and data-lineage guards, using real Parquet fixtures."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def fixture(tmp_path):
    spec = importlib.util.spec_from_file_location('v3_fixture', Path(__file__).with_name('test_v3.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture_v3(tmp_path)


def rewrite_manifest(bundle, **changes):
    path = bundle / 'v3-bundle.json'
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize('joined', [False, True])
def test_root_entrypoint_recognizes_v3_bundle(tmp_path, joined):
    data, bundle, _ = fixture(tmp_path)
    output = tmp_path / 'out.csv'
    entry = Path(__file__).resolve().parents[3] / 'predict.py'
    flag = ['--bundle=' + str(bundle)] if joined else ['--bundle', str(bundle)]
    command = [sys.executable, str(entry), '--test', str(data/'test.parquet'),
               '--train', str(data/'train.parquet'), '--output', str(output),
               '--sample-submission', str(data/'sample_submission.csv')] + flag
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.with_suffix('.csv.report.json').read_text())
    assert report['model_format'] == 3 and report['test_spectra'] == 3
    assert [row['molecule_id'] for row in csv.DictReader(output.open())] == ['001', '000']


@pytest.mark.parametrize('kind', ['ambiguous', 'missing'])
def test_root_entrypoint_never_guesses_unknown_bundle(tmp_path, kind):
    _, bundle, _ = fixture(tmp_path)
    if kind == 'ambiguous':
        (bundle/'bundle.json').write_text('{}')
    else:
        (bundle/'v3-bundle.json').unlink()
    entry = Path(__file__).resolve().parents[3] / 'predict.py'
    result = subprocess.run([sys.executable, str(entry), '--bundle', str(bundle), '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert kind in result.stderr.lower()


@pytest.mark.parametrize('version', [None, 'different-feature-order'])
def test_bundle_rejects_missing_or_wrong_feature_version(tmp_path, version):
    from casmi26.pipeline_v3 import verify_bundle
    _, bundle, _ = fixture(tmp_path)
    rewrite_manifest(bundle, feature_version=version)
    with pytest.raises(ValueError, match='feature version'):
        verify_bundle(bundle)


@pytest.mark.parametrize('field,value', [('gate_threshold', -0.1), ('gate_threshold', 1.1),
                                        ('gate_threshold', float('nan')), ('gate_margin', -1),
                                        ('gate_margin', float('inf'))])
def test_bundle_rejects_invalid_confidence_configuration(tmp_path, field, value):
    from casmi26.pipeline_v3 import verify_bundle
    _, bundle, _ = fixture(tmp_path)
    rewrite_manifest(bundle, mode='confidence', gate_threshold=.95, gate_margin=.05)
    rewrite_manifest(bundle, **{field: value})
    with pytest.raises(ValueError, match='confidence'):
        verify_bundle(bundle)


@pytest.mark.parametrize('copy_input', [False, True])
def test_inference_refuses_reference_query_overlap(tmp_path, copy_input):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from casmi26.inference_v3 import infer
    from casmi26.production import sha256
    data, bundle, _ = fixture(tmp_path)
    table = pq.read_table(data/'train.parquet')
    hybrid = data/'hybrid.parquet'
    pq.write_table(table.append_column('molecule_id', pa.array(['000', '001'])), hybrid)
    query = hybrid
    if copy_input:
        query = data/'duplicate-content.parquet'
        shutil.copy2(hybrid, query)
    rewrite_manifest(bundle, train_sha256=sha256(hybrid))
    with pytest.raises(ValueError, match='overlap'):
        infer(query, hybrid, bundle, tmp_path/'out.csv')
    assert not (tmp_path/'out.csv').exists()


@pytest.mark.parametrize('bad_id', [None, '', '   '])
def test_inference_rejects_missing_compound_identifiers(tmp_path, bad_id):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from casmi26.inference_v3 import infer
    data, bundle, _ = fixture(tmp_path)
    table = pq.read_table(data/'test.parquet')
    table = table.set_column(table.schema.get_field_index('molecule_id'), 'molecule_id',
                             pa.array([bad_id, '001', '000'], type=pa.string()))
    pq.write_table(table, data/'test.parquet')
    with pytest.raises(ValueError, match='molecule ID'):
        infer(data/'test.parquet', data/'train.parquet', bundle, tmp_path/'out.csv')
    assert not (tmp_path/'out.csv').exists()
