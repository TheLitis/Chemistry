"""Streaming input adapters and one-to-one spectral matching."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .chemistry import adduct_spec, canonical, neutral_mass

ALIASES = {
    'compound_id': ('compound_id', 'molecule_id', 'compound', 'molecule', 'sample_id', 'id'),
    'precursor_mz': ('precursor_mz', 'precursor_mass', 'pepmass', 'parent_mz'),
    'adduct': ('adduct', 'precursor_type', 'ion_type'),
    'mz': ('mz', 'mzs', 'm/z', 'mz_array'),
    'intensity': ('intensity', 'intensities', 'intensity_array'),
    'peaks': ('peaks', 'spectrum'),
    'smiles': ('smiles', 'canonical_smiles'),
    'collision_energy': ('collision_energy', 'ce'),
}
EXTENSIONS = {'.jsonl', '.ndjson', '.mgf', '.csv', '.tsv', '.parquet'}


def data_files(path: Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Data path not found: {path}')
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.suffix.lower() in EXTENSIONS)
    files = [p for p in files if 'sample_submission' not in p.name and not p.name.endswith('.report.json')]
    if not files:
        raise ValueError(f'No supported data files in {path}')
    return files


def records(path: Path) -> Iterator[dict]:
    for file in data_files(path):
        suffix = file.suffix.lower()
        if suffix in ('.jsonl', '.ndjson'):
            with file.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
        elif suffix in ('.csv', '.tsv'):
            with file.open(encoding='utf-8-sig', newline='') as stream:
                yield from csv.DictReader(stream, delimiter='\t' if suffix == '.tsv' else ',')
        elif suffix == '.parquet':
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError('Parquet requires pyarrow: python -m pip install pyarrow') from exc
            with pq.ParquetFile(file) as parquet:
                for batch in parquet.iter_batches(batch_size=1024):
                    yield from batch.to_pylist()
        elif suffix == '.mgf':
            current = None
            with file.open(encoding='utf-8-sig') as stream:
                for text in stream:
                    text = text.strip()
                    if not text or text.startswith(('#', ';')):
                        continue
                    if text.upper() == 'BEGIN IONS':
                        if current is not None:
                            raise ValueError('Nested MGF BEGIN IONS')
                        current = {'peaks': []}
                    elif text.upper() == 'END IONS':
                        if current is None:
                            raise ValueError('MGF END IONS without BEGIN IONS')
                        yield current
                        current = None
                    elif current is not None and '=' in text:
                        key, value = text.split('=', 1)
                        current[key.lower()] = value.split()[0] if key.upper() == 'PEPMASS' else value
                    elif current is not None:
                        current['peaks'].append([float(v) for v in text.split()[:2]])
            if current is not None:
                raise ValueError(f'Unclosed MGF spectrum in {file}')
        else:
            raise ValueError(f'Unsupported input extension: {suffix}')


def value(row: dict, key: str, columns: dict | None = None, required: bool = True):
    lower = {str(k).lower(): v for k, v in row.items()}
    keys = (columns[key].lower(),) if columns and key in columns else ALIASES[key]
    for name in keys:
        if name in lower and lower[name] is not None:
            v = lower[name]
            if not isinstance(v, str) or v.strip():
                return v
    if required:
        raise ValueError(f'Missing {key}; supply an explicit --columns mapping for the real schema.')
    return None


@dataclass
class Spectrum:
    compound_id: str
    precursor_mz: float
    adduct: str
    peaks: np.ndarray
    collision_energy: float | None = None
    smiles: str | None = None

    @property
    def neutral(self) -> float:
        return neutral_mass(self.precursor_mz, self.adduct)

    @property
    def mode(self) -> int:
        return 1 if adduct_spec(self.adduct)[1] > 0 else -1


def decode_array(v):
    return json.loads(v) if isinstance(v, str) else v


def from_row(row: dict, columns: dict | None = None, labeled: bool = False) -> Spectrum:
    smiles = canonical(str(value(row, 'smiles', columns))) if labeled else None
    compound_id = value(row, 'compound_id', columns, required=not labeled)
    if compound_id is None:
        compound_id = smiles  # Only reference data may use its known graph as group key.
    precursor = float(value(row, 'precursor_mz', columns))
    adduct = str(value(row, 'adduct', columns)).replace(' ', '').replace('−', '-')
    neutral_mass(precursor, adduct)  # Reject unknown adducts, NaN and nonpositive masses.
    combined = value(row, 'peaks', columns, required=False)
    if combined is not None:
        raw = np.asarray(decode_array(combined), dtype=np.float64)
    else:
        mz = np.asarray(decode_array(value(row, 'mz', columns)), dtype=np.float64)
        intensity = np.asarray(decode_array(value(row, 'intensity', columns)), dtype=np.float64)
        if mz.ndim != 1 or intensity.ndim != 1 or len(mz) != len(intensity):
            raise ValueError('m/z and intensity arrays must be equal-length vectors')
        raw = np.column_stack((mz, intensity))
    if raw.ndim != 2 or raw.shape[1] != 2 or not len(raw):
        raise ValueError('Peaks must be a nonempty array of [m/z, intensity] pairs')
    if not np.isfinite(raw).all() or (raw[:, 0] <= 0).any() or (raw[:, 1] < 0).any():
        raise ValueError('Peaks contain nonfinite/invalid masses or intensities')
    raw = raw[raw[:, 1] > 0]
    if not len(raw):
        raise ValueError('Spectrum has no positive intensities')
    mz, inverse = np.unique(raw[:, 0], return_inverse=True)
    intensities = np.bincount(inverse, weights=raw[:, 1])
    peaks = np.column_stack((mz, intensities / intensities.sum()))
    ce = value(row, 'collision_energy', columns, required=False)
    try:
        ce = float(ce) if ce is not None else None
        if ce is not None and not np.isfinite(ce):
            ce = None
    except (ValueError, TypeError):
        ce = None  # NCE strings are not silently equated with numeric eV values.
    return Spectrum(str(compound_id), precursor, adduct, peaks, ce, smiles)


def read_spectra(path: Path, columns: dict | None = None, labeled: bool = False) -> Iterator[Spectrum]:
    for number, row in enumerate(records(path), 1):
        try:
            yield from_row(row, columns, labeled)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'{path}, record {number}: {exc}') from exc


def cosine(a: np.ndarray, b: np.ndarray, ppm: float = 20, da: float = .01) -> float:
    """Square-root-intensity cosine with a deterministic greedy one-to-one match."""
    if not len(a) or not len(b):
        return 0.0
    a, b = a[np.argsort(a[:, 0])], b[np.argsort(b[:, 0])]
    av, bv = np.sqrt(a[:, 1]), np.sqrt(b[:, 1])
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if not denom:
        return 0.0
    edges = []
    for i, (mz, _) in enumerate(a):
        tol = max(da, mz * ppm * 1e-6)
        start, end = np.searchsorted(b[:, 0], [mz-tol, mz+tol], side='left')
        # searchsorted side='right' for the upper endpoint includes equality.
        end = np.searchsorted(b[:, 0], mz+tol, side='right')
        edges.extend((float(av[i]*bv[j]), i, j) for j in range(start, end))
    used_a, used_b, total = set(), set(), 0.0
    for weight, i, j in sorted(edges, reverse=True):
        if i not in used_a and j not in used_b:
            total += weight
            used_a.add(i)
            used_b.add(j)
    return min(1.0, max(0.0, total / denom))


def spectral_score(query: Spectrum, reference: Spectrum, ppm: float = 20, da: float = .01) -> float:
    if query.mode != reference.mode:
        return 0.0
    a = query.peaks[query.peaks[:, 0] < query.precursor_mz - da]
    b = reference.peaks[reference.peaks[:, 0] < reference.precursor_mz - da]
    direct = cosine(a, b, ppm, da)
    loss = 0.0
    if adduct_spec(query.adduct)[:2] == adduct_spec(reference.adduct)[:2]:
        loss = cosine(np.column_stack((query.precursor_mz-a[:, 0], a[:, 1])),
                      np.column_stack((reference.precursor_mz-b[:, 0], b[:, 1])), ppm, da)
    return (0.85*direct + 0.15*loss) * (1.0 if query.adduct == reference.adduct else .8)
