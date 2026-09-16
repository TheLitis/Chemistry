import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[3] / 'tasks/casmi_rank_research.py'
    assert path.exists(), 'Frozen ranker experiment implementation is absent'
    spec = importlib.util.spec_from_file_location('rank_research_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_distinct_key_rank_and_truncation():
    m = module()
    assert m.rank_of(np.array([4., 3., 2.]), ['A', 'A', 'B'], 'B') == 2
    assert m.rank_of(np.arange(30.)[::-1], [str(i) for i in range(30)], '26') == 0
    assert m.rank_of(np.array([]), [], 'none') == 0


def test_weight_correction_recovers_weighted_bce_optimum():
    m = module()
    true = np.array([.1, .7, .3])
    weights = np.array([10., 2., 5.])
    weighted = weights * true / (1 - true + weights * true)
    np.testing.assert_allclose(m.unweight(weighted, weights), true)


def test_scores_are_finite_and_truth_is_not_an_input():
    m = module()
    p = np.array([.99, .01, .5])
    fps = np.array([[1., 0., 0.], [0., 1., 0.]])
    results = m.base_scores(p, fps, np.ones(3))
    for name, values in results.items():
        assert values.shape == (2,) and np.isfinite(values).all(), name
        assert values[0] > values[1], name


def test_pool_split_excludes_previous_queries_and_duplicates():
    m = module()
    calibration, audit = m.split_keys(['A', 'A', 'B', 'C', 'D', 'E', 'F'], {'A'}, 2, 2)
    assert len(calibration) == len(audit) == 2
    assert not set(calibration) & set(audit)
    assert 'A' not in calibration + audit
    assert (calibration, audit) == m.split_keys(list(reversed(['A', 'B', 'C', 'D', 'E', 'F'])), {'A'}, 2, 2)


def test_small_pool_does_not_silently_reduce_audit_size():
    with pytest.raises(ValueError):
        module().split_keys(['A', 'B'], set(), 2, 2)


def test_metrics_count_misses_as_zero():
    result = module().metrics([1, 2, 0, 26])
    assert result['mrr_at_25'] == .375
    assert result['top1_accuracy'] == .25
    assert result['recall_at_25'] == .5


def test_paired_interval_exact_when_no_difference():
    result = module().paired_interval([1, 2, 0], [1, 2, 0], repeats=50)
    assert result == {'delta_mrr': 0., 'ci95': [0., 0.], 'bootstrap_repeats': 50}


def test_mass_prior_uses_only_observed_mass():
    m = module()
    z = m.mass_prior(np.array([100., 100.001, 100.01]), 100.)
    assert z[0] == 0 and z[0] > z[1] > z[2]
