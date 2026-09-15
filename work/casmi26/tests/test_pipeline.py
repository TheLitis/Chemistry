import csv
import json
from pathlib import Path

import numpy as np
import pytest


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return path


def row(cid, peaks, smiles=None, precursor=47.04914127862, adduct='[M+H]+', **extra):
    r = dict(compound_id=cid, precursor_mz=precursor, adduct=adduct,
             mz=[p[0] for p in peaks], intensity=[p[1] for p in peaks], **extra)
    if smiles is not None:
        r['smiles'] = smiles
    return r


def test_canonical_graph_not_string_or_fingerprint():
    from casmi26.chemistry import canonical, exact_match
    assert canonical('OCC') == canonical('CCO')
    assert exact_match('OCC', 'CCO')
    assert not exact_match('COC', 'CCO')
    assert exact_match('C[C@H](O)F', 'C[C@@H](O)F')
    assert not exact_match('[13CH3]CO', 'CCO')
    with pytest.raises(ValueError):
        canonical('not-a-molecule')


def test_neutral_mass_adducts():
    from casmi26.chemistry import neutral_mass, info
    m = info('CCO').mass
    assert neutral_mass(m + 1.007276466621, '[M+H]+') == pytest.approx(m)
    assert neutral_mass(m - 1.007276466621, '[M-H]-') == pytest.approx(m)
    assert neutral_mass((m + 2 * 1.007276466621) / 2, '[M+2H]2+') == pytest.approx(m)
    assert neutral_mass(m + 22.989220702, '[M+Na]+') == pytest.approx(m)
    with pytest.raises(ValueError, match='adduct'):
        neutral_mass(100, 'unknown')


def test_clean_peak_arrays_and_label_noninterference():
    from casmi26.spectra import from_row
    a = from_row(row('x', [(31, 4), (20, 0), (31, 5)]))
    b = from_row(row('x', [(31, 4), (20, 0), (31, 5)], smiles='PRIVATE-ANSWER'))
    assert a.compound_id == b.compound_id
    np.testing.assert_equal(a.peaks, b.peaks)
    assert a.peaks.tolist() == [[31.0, 1.0]]
    assert a.smiles is None
    with pytest.raises(ValueError):
        from_row(row('x', [(float('nan'), 2)]))
    with pytest.raises(ValueError):
        from_row(dict(compound_id='x', precursor_mz=100, adduct='[M+H]+', mz=[1, 2], intensity=[1]))


def test_cosine_one_to_one_scale_and_tolerance():
    from casmi26.spectra import cosine
    a = np.array([[10.0, 1.0], [20, 4.0]])
    assert cosine(a, a) == pytest.approx(1)
    assert cosine(a, np.array([[50., 1.]])) == 0
    assert cosine(np.array([[10., 1.], [10.001, 1.]]), np.array([[10., 1.]]), da=.01) < 1
    assert cosine(a, a * [1, 50]) == pytest.approx(1)


def test_reader_jsonl_mgf_and_columns(tmp_path):
    from casmi26.spectra import read_spectra
    p = write_jsonl(tmp_path / 'test.jsonl', [row('01', [(31, 1)]), row('01', [(29, 1)])])
    assert [s.compound_id for s in read_spectra(p)] == ['01', '01']
    mgf = tmp_path / 'test.mgf'
    mgf.write_text('BEGIN IONS\nCOMPOUND_ID=01\nPEPMASS=47.04914127862\nADDUCT=[M+H]+\n31 1\nEND IONS\n')
    assert list(read_spectra(mgf))[0].compound_id == '01'
    custom = write_jsonl(tmp_path / 'custom.jsonl', [dict(target='b', parent=100, ion='[M-H]-', peaks=[[30, 2]])])
    s = list(read_spectra(custom, columns={'compound_id': 'target', 'precursor_mz': 'parent', 'adduct': 'ion'}))[0]
    assert s.compound_id == 'b'
    with pytest.raises(ValueError, match='compound'):
        list(read_spectra(write_jsonl(tmp_path / 'bad.jsonl', [dict(precursor_mz=100, adduct='[M+H]+', peaks=[[30, 2]])])))


def test_generated_isomers_are_new_valid_and_same_formula():
    from casmi26.chemistry import proposals, info, canonical
    candidates = list(proposals('CCO', limit=20))
    assert canonical('COC') in candidates
    assert canonical('CCO') not in candidates
    assert all(info(s).formula == 'C2H6O' for s in candidates)


def fixture_data(tmp_path):
    train = write_jsonl(tmp_path / 'train.jsonl', [
        row('train_a', [(31, 10), (29, 1)], 'CCO'),
        row('train_b', [(45, 10), (29, 1)], 'COC'),
        row('train_c', [(60, 10)], 'CCCO', precursor=61.06479134262),
    ])
    test = write_jsonl(tmp_path / 'test.jsonl', [
        row('002', [(45, 10), (29, 1)], 'CCO'),
        row('001', [(31, 10), (29, 1)], 'COC'),
        row('001', [(31, 20), (29, 2)], 'COC'),
    ])
    template = tmp_path / 'sample_submission.csv'
    template.write_text('compound_id,smiles\n001,ANSWER-MUST-BE-IGNORED\n002,ANSWER-MUST-BE-IGNORED\n', encoding='utf-8')
    return train, test, template


def test_end_to_end_preserves_order_ids_all_spectra_and_ignores_labels(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    output = tmp_path / 'submission.csv'
    report = predict(test, train, template, output, cache=tmp_path / 'cache.sqlite')
    with output.open() as f:
        rows = list(csv.DictReader(f))
    assert rows == [dict(compound_id='001', smiles='CCO'), dict(compound_id='002', smiles='COC')]
    assert report['official_score'] is None
    assert report['compounds']['001']['spectra_used'] == 2
    assert report['compounds']['002']['spectra_used'] == 1
    assert len(report['inputs']['train']['sha256']) == 64
    assert output.with_suffix('.csv.report.json').exists()
    # Cached and non-cached outputs must match, not silently rebuild differently.
    report2 = predict(test, train, template, output, cache=tmp_path / 'cache.sqlite')
    assert report2['cache_reused'] is True


def test_multispectrum_aggregation_not_first_spectrum_only(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    write_jsonl(test, [row('001', [(45, 10), (29, 1)]), row('001', [(31, 10), (29, 1)]),
                      row('001', [(31, 10), (29, 1)]), row('002', [(45, 10), (29, 1)])])
    predict(test, train, template, tmp_path / 'sub.csv', cache=tmp_path / 'cache.sqlite')
    assert list(csv.DictReader((tmp_path / 'sub.csv').open()))[0]['smiles'] == 'CCO'


def test_inconsistent_mass_or_missing_group_does_not_write_submission(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    write_jsonl(test, [row('001', [(31, 1)]), row('001', [(31, 1)], precursor=200), row('002', [(45, 1)])])
    output = tmp_path / 'no.csv'
    with pytest.raises(ValueError, match='inconsistent'):
        predict(test, train, template, output, cache=tmp_path / 'cache.sqlite')
    assert not output.exists()
    write_jsonl(test, [row('001', [(31, 1)])])
    with pytest.raises(ValueError, match='IDs'):
        predict(test, train, template, output, cache=tmp_path / 'cache.sqlite')


def test_cannot_overwrite_inputs(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    with pytest.raises(ValueError, match='overwrite'):
        predict(test, train, template, test, cache=tmp_path / 'cache.sqlite')


def test_no_mass_compatible_candidates_is_not_dummy_carbon(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    write_jsonl(test, [row('001', [(31, 1)], precursor=999), row('002', [(45, 1)])])
    with pytest.raises(ValueError, match='candidate'):
        predict(test, train, template, tmp_path / 'no.csv', cache=tmp_path / 'cache.sqlite')
    assert not (tmp_path / 'no.csv').exists()


def test_duplicate_template_ids_rejected(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    template.write_text('compound_id,smiles\n001,C\n001,C\n')
    with pytest.raises(ValueError, match='duplicate'):
        predict(test, train, template, tmp_path / 'no.csv', cache=tmp_path / 'cache.sqlite')


def test_invalid_reference_not_silently_dropped(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    write_jsonl(train, [row('a', [(31, 1)], 'bogus')])
    with pytest.raises(ValueError):
        predict(test, train, template, tmp_path / 'no.csv', cache=tmp_path / 'cache.sqlite')


def test_cache_invalidation_by_content(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    first = predict(test, train, template, tmp_path / 's.csv', cache=tmp_path / 'c.sqlite')
    with train.open('a') as stream:
        stream.write(json.dumps(row('extra', [(30, 1)], 'CCO')) + '\n')
    second = predict(test, train, template, tmp_path / 's.csv', cache=tmp_path / 'c.sqlite')
    assert second['cache_reused'] is False
    assert first['inputs']['train']['sha256'] != second['inputs']['train']['sha256']
    assert second['reference_spectra'] == first['reference_spectra'] + 1


def test_unsupported_template_schema_is_not_guessed(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    template.write_text('id,smiles_1,smiles_2\n001,C,C\n')
    with pytest.raises(ValueError, match='schema'):
        predict(test, train, template, tmp_path / 's.csv')


def test_no_grouping_by_mass(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    report = predict(test, train, template, tmp_path / 's.csv')
    assert report['prediction_count'] == 2
    assert set(report['compounds']) == {'001', '002'}


def test_graph_search_is_integrated_and_bounded(tmp_path):
    from casmi26.engine import Config, predict
    train, test, template = fixture_data(tmp_path)
    write_jsonl(train, [row('a', [(31, 10)], 'CCO')])
    report = predict(test, train, template, tmp_path / 's.csv', config=Config(isomer_budget=4))
    d = report['compounds']['001']
    assert 1 <= d['generated_candidate_count'] <= 4
    assert any(c['smiles'] == 'COC' for c in d['top_candidates_for_audit_only'])
    assert d['score_is_probability'] is False


def test_same_graph_always_in_same_validation_fold():
    from casmi26.engine import structure_fold
    assert structure_fold('CCO') == structure_fold('OCC')
    assert structure_fold('C[C@H](O)F') == structure_fold('C[C@@H](O)F')
    with pytest.raises(ValueError):
        structure_fold('C', 1)


def test_csv_peak_arrays(tmp_path):
    from casmi26.spectra import read_spectra
    path = tmp_path / 'spectra.csv'
    r = row('0001', [(20, 2), (31, 1)])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(r))
        writer.writeheader()
        writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()})
    assert list(read_spectra(path))[0].compound_id == '0001'


def test_parquet_lists(tmp_path):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    from casmi26.spectra import read_spectra
    path = tmp_path / 'spectra.parquet'
    pq.write_table(pa.Table.from_pylist([row('01', [(31, 2), (29, 1)])]), path)
    assert list(read_spectra(path))[0].compound_id == '01'


def test_cli_exact_entrypoint_and_failure_exit_code(tmp_path):
    import subprocess
    import sys
    train, test, template = fixture_data(tmp_path)
    entry = Path(__file__).resolve().parents[3] / 'predict.py'
    output = tmp_path / 'out.csv'
    p = subprocess.run([sys.executable, str(entry), '--test', str(test), '--output', str(output)],
                       capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)['official_score'] is None
    assert output.exists()
    bad = subprocess.run([sys.executable, str(entry), '--test', str(tmp_path / 'absent'),
                          '--output', str(tmp_path / 'bad.csv')], capture_output=True, text=True, cwd=tmp_path)
    assert bad.returncode == 2
    assert not (tmp_path / 'bad.csv').exists()


def test_output_cannot_contaminate_input_folder(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    target = tmp_path / 'queries'
    target.mkdir()
    test.rename(target / test.name)
    with pytest.raises(ValueError, match='contaminate'):
        predict(target, train, template, target / 's.csv')


def test_opposite_polarities_do_not_directly_match():
    from casmi26.spectra import spectral_score, from_row
    a = from_row(row('a', [(31, 1)]))
    b = from_row(row('b', [(31, 1)], precursor=45.034588345, adduct='[M-H]-'))
    assert spectral_score(a, b) == 0


def test_cosine_bounds_on_random_spectra():
    from casmi26.spectra import cosine
    rng = np.random.default_rng(18)
    for _ in range(30):
        a = np.column_stack((rng.uniform(10, 20, 30), rng.uniform(0, 100, 30)))
        b = np.column_stack((rng.uniform(10, 20, 30), rng.uniform(0, 100, 30)))
        assert 0 <= cosine(a, b, da=.5) <= 1


def test_reference_and_test_files_cannot_overlap(tmp_path):
    from casmi26.engine import predict
    train, test, template = fixture_data(tmp_path)
    with pytest.raises(ValueError, match='overlap'):
        predict(test, test, template, tmp_path / 's.csv')
