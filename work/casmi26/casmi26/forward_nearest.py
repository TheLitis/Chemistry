"""Optional exact selected-feature evaluator; not enabled in frozen R08B."""
from __future__ import annotations
import numpy as np
from .forward_ranking import SUPPORTED_ADDUCTS, clean_peaks, compare_spectra


def pool_cosine_nearest_only(queries, predictions):
    """Exact selected feature, omitting unused max-grid matching/entropy work.

    Keep validation of every supported prediction, including unselected energies.
    This is not enabled in the frozen R08B release; numerical parity must be
    verified before changing a packaged inference implementation.
    """
    values = []
    for query in queries:
        mode = query['adduct']
        if mode not in SUPPORTED_ADDUCTS:
            continue
        candidates = sorted((float(e), p) for (m, e), p in predictions.items()
                            if m == mode and p is not None)
        if not candidates:
            continue
        clean_peaks(query['peaks'], query['precursor'])
        for _, predicted in candidates:
            clean_peaks(predicted, query['precursor'])
        energies = np.array([e for e, _ in candidates])
        collision = query.get('ce')
        if not isinstance(collision, (list, tuple, np.ndarray)):
            collision = [] if collision is None else [collision]
        collision = [float(e) for e in collision
                     if e is not None and np.isfinite(e) and float(e) >= 0]
        indices = ([int(np.argmin(abs(energies - e))) for e in collision]
                   if collision else list(range(len(candidates))))
        parts = []
        for i in indices:
            peaks = np.asarray(candidates[i][1], dtype=np.float64).reshape(-1, 2)
            if len(peaks) and peaks[:, 1].sum() > 0:
                peaks = peaks.copy()
                peaks[:, 1] /= peaks[:, 1].sum()
                parts.append(peaks)
        mixture = np.concatenate(parts) if parts else np.empty((0, 2))
        values.append(compare_spectra(query['peaks'], mixture, query['precursor'])['cosine'])
    return float(np.mean(values)) if values else None
