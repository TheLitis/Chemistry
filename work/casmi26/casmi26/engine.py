"""Disk-backed reference search, joint spectrum decisions and strict output."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import zlib

import numpy as np
from rdkit import rdBase

from . import __version__
from .chemistry import canonical, fragment_explanation, info, proposals
from .spectra import Spectrum, data_files, read_spectra, records, spectral_score, value

INDEX_VERSION = 1


@dataclass(frozen=True)
class Config:
    precursor_ppm: float = 15.0
    precursor_da: float = .003
    fragment_ppm: float = 20.0
    fragment_da: float = .01
    # The unvalidated graph heuristic is opt-in; default scores use references only.
    isomer_budget: int = 0
    fragment_weight: float = .3

    def __post_init__(self):
        for v in (self.precursor_ppm, self.precursor_da, self.fragment_ppm, self.fragment_da):
            if not np.isfinite(v) or v <= 0:
                raise ValueError('Mass tolerances must be finite and positive')
        if self.isomer_budget < 0 or not 0 <= self.fragment_weight <= 1:
            raise ValueError('Invalid isomer budget or fragment weight')


def digest(path: Path) -> dict:
    files = data_files(path) if Path(path).is_dir() else [Path(path)]
    aggregate = hashlib.sha256()
    size = 0
    for file in files:
        h = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                size += len(chunk)
                h.update(chunk)
        if len(files) == 1:
            aggregate = h
        else:
            aggregate.update(str(file.relative_to(path)).replace('\\', '/').encode())
            aggregate.update(b'\0' + h.digest())
    return {'sha256': aggregate.hexdigest(), 'files': len(files), 'bytes': size}


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _prepare_library(train: Path, cache: Path, columns: dict, candidates: Path | None):
    identity = {
        'index_version': INDEX_VERSION, 'rdkit': rdBase.rdkitVersion,
        'columns': columns, 'train': digest(train),
        'candidates': digest(candidates) if candidates else None,
    }
    signature = json.dumps(identity, sort_keys=True)
    if cache.exists():
        db = sqlite3.connect(cache)
        try:
            result = db.execute('SELECT value FROM meta WHERE key="signature"').fetchone()
            if result and result[0] == signature:
                return db, identity, True
        except sqlite3.DatabaseError:
            pass
        db.close()
    cache.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=cache.name+'.', suffix='.tmp', dir=cache.parent)
    os.close(fd)
    db = sqlite3.connect(name)
    try:
        db.executescript('''
            CREATE TABLE molecules(smiles TEXT PRIMARY KEY, mass REAL NOT NULL, formula TEXT NOT NULL);
            CREATE INDEX mass_index ON molecules(mass);
            CREATE TABLE spectra(smiles TEXT NOT NULL, precursor REAL NOT NULL,
                                 adduct TEXT NOT NULL, ce REAL, peaks BLOB NOT NULL);
            CREATE INDEX spectrum_molecule ON spectra(smiles);
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        n = 0
        for s in read_spectra(train, columns, labeled=True):
            m = info(s.smiles)
            db.execute('INSERT OR IGNORE INTO molecules VALUES(?,?,?)', (m.smiles, m.mass, m.formula))
            blob = zlib.compress(s.peaks.astype('<f8').tobytes(), 1)
            db.execute('INSERT INTO spectra VALUES(?,?,?,?,?)',
                       (m.smiles, s.precursor_mz, s.adduct, s.collision_energy, blob))
            n += 1
            if n % 5000 == 0:
                db.commit()
        if not n:
            raise ValueError('Reference library contains no spectra')
        if candidates:
            for row in records(candidates):
                m = info(str(value(row, 'smiles', columns)))
                db.execute('INSERT OR IGNORE INTO molecules VALUES(?,?,?)', (m.smiles, m.mass, m.formula))
        db.execute('INSERT INTO meta VALUES(?,?)', ('signature', signature))
        db.execute('INSERT INTO meta VALUES(?,?)', ('spectra', str(n)))
        db.commit()
        db.close()
        os.replace(name, cache)
    except Exception:
        db.close()
        if os.path.exists(name):
            os.unlink(name)
        raise
    return sqlite3.connect(cache), identity, False


def _template(path: Path, id_column: str | None, prediction_column: str | None):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        if not header or len(header) != 2 or len(set(header)) != 2:
            raise ValueError('Unsupported submission schema: require exactly one ID and one SMILES column; '
                             'verify the actual competition contract before adapting this writer.')
        pred = prediction_column or next((h for h in header if h.lower() in ('smiles', 'structure')), None)
        if pred is None or pred not in header:
            raise ValueError('SMILES submission column is unknown; use --prediction-column from the official template')
        key = id_column or next(h for h in header if h != pred)
        if key not in header or key == pred:
            raise ValueError('Invalid submission ID column')
        ids = [r[key] for r in reader]
    if not ids or any(x is None or not x.strip() for x in ids):
        raise ValueError('Template has empty IDs')
    if len(set(ids)) != len(ids):
        raise ValueError('Template has duplicate IDs')
    return header, key, pred, ids


def _group_candidates(db, group: list[Spectrum], config: Config):
    masses = np.array([s.neutral for s in group])
    center = float(np.median(masses))
    tolerance = max(config.precursor_da, center * config.precursor_ppm * 1e-6)
    if np.any(np.abs(masses-center) > tolerance):
        raise ValueError(f'Compound {group[0].compound_id}: inconsistent neutral masses across spectra; '
                         'check grouping and adduct annotations')
    lower = max(masses-tolerance)
    upper = min(masses+tolerance)
    candidates = db.execute('SELECT smiles FROM molecules WHERE mass BETWEEN ? AND ? ORDER BY smiles',
                            (float(lower), float(upper))).fetchall()
    if not candidates:
        raise ValueError(f'Compound {group[0].compound_id}: no mass-compatible structural candidate. '
                         'No dummy structure or relaxed-mass guess was written. '
                         'This version needs a broader permitted structure catalog or a generative model.')
    return [r[0] for r in candidates], center


def _rank(db, group, config: Config):
    candidates, mass = _group_candidates(db, group, config)
    scores = {}
    for smiles in candidates:
        by_query = np.zeros(len(group), dtype=float)
        refs = db.execute('SELECT precursor, adduct, ce, peaks FROM spectra WHERE smiles=?', (smiles,))
        for precursor, adduct, ce, blob in refs:
            peaks = np.frombuffer(zlib.decompress(blob), dtype='<f8').reshape(-1, 2)
            reference = Spectrum('', precursor, adduct, peaks, ce, smiles)
            for i, query in enumerate(group):
                by_query[i] = max(by_query[i], spectral_score(query, reference,
                                                              config.fragment_ppm, config.fragment_da))
        scores[smiles] = {'reference_score': float(by_query.mean()), 'source': 'reference_or_catalog'}
    generated = set()
    if config.isomer_budget:
        # Seed from the strongest available structures; do not read unknown labels.
        seeds = sorted(scores, key=lambda s: (-scores[s]['reference_score'], s))[:8]
        for seed in seeds:
            for s in proposals(seed, config.isomer_budget):
                if s not in scores:
                    generated.add(s)
                    scores[s] = {'reference_score': 0.0, 'source': 'bounded_graph_proposal'}
                if len(generated) >= config.isomer_budget:
                    break
            if len(generated) >= config.isomer_budget:
                break
    for s, record in scores.items():
        explained = float(np.mean([fragment_explanation(s, q, config.fragment_ppm, config.fragment_da)
                                   for q in group])) if config.isomer_budget else 0.0
        record['fragment_heuristic'] = explained
        w = config.fragment_weight if config.isomer_budget else 0.0
        record['score'] = (1-w)*record['reference_score'] + w*explained
    ranking = sorted(scores, key=lambda s: (-scores[s]['score'], s))
    best = ranking[0]
    return best, {
        'spectra_used': len(group), 'neutral_mass': mass,
        'candidate_count': len(scores), 'generated_candidate_count': len(generated),
        'selected_source': scores[best]['source'], 'score_is_probability': False,
        'top1_score': scores[best]['score'],
        'score_margin': scores[best]['score']-scores[ranking[1]]['score'] if len(ranking) > 1 else None,
        'zero_spectral_evidence': all(r['reference_score'] == 0 for r in scores.values()),
        'top_candidates_for_audit_only': [{'smiles': s, **scores[s]} for s in ranking[:10]],
    }


def predict(test: Path, train: Path, sample_submission: Path, output: Path, *,
            cache: Path | None = None, columns: dict | None = None, config: Config | None = None,
            candidates: Path | None = None, id_column: str | None = None,
            prediction_column: str | None = None) -> dict:
    test, train, sample_submission, output = map(Path, (test, train, sample_submission, output))
    cache = Path(cache) if cache else output.parent / 'artifacts' / 'library.sqlite'
    candidates = Path(candidates) if candidates else None
    sidecar = output.with_suffix(output.suffix+'.report.json')
    config, columns = config or Config(), columns or {}
    test_paths = {p.resolve() for p in data_files(test)}
    train_paths = {p.resolve() for p in data_files(train)}
    candidate_paths = {p.resolve() for p in data_files(candidates)} if candidates else set()
    if test_paths.intersection(train_paths | candidate_paths):
        raise ValueError('Reference/candidate files overlap test files; refusing potential label leakage')
    source_paths = test_paths | train_paths | candidate_paths | {sample_submission.resolve()}
    products = [output.resolve(), sidecar.resolve(), cache.resolve()]
    if len(set(products)) != len(products) or source_paths.intersection(products):
        raise ValueError('Output/cache would overwrite an input or another output')
    # Do not write products inside an input directory, where a later run would
    # ingest its own output and contaminate the input manifest.
    for source in (test, train, candidates):
        if source and source.is_dir() and any(source.resolve() in p.parents for p in products):
            raise ValueError('Output/cache inside an input directory would overwrite/contaminate data')
    header, key, pred, expected = _template(sample_submission, id_column, prediction_column)
    groups = defaultdict(list)
    for spectrum in read_spectra(test, columns, labeled=False):
        groups[spectrum.compound_id].append(spectrum)
    if set(groups) != set(expected):
        missing = sorted(set(expected)-set(groups))[:10]
        extra = sorted(set(groups)-set(expected))[:10]
        raise ValueError(f'Test and submission IDs differ: missing={missing}, extra={extra}. '
                         'Set the compound_id mapping from the official grouping schema; do not group by mass.')
    db, identity, reused = _prepare_library(train, cache, columns, candidates)
    predictions, details = {}, {}
    try:
        for compound_id in expected:
            predictions[compound_id], details[compound_id] = _rank(db, groups[compound_id], config)
        reference_count = int(db.execute('SELECT value FROM meta WHERE key="spectra"').fetchone()[0])
        structure_count = db.execute('SELECT COUNT(*) FROM molecules').fetchone()[0]
    finally:
        db.close()
    report = {
        'status': 'predictions_generated_not_competition_validated', 'version': __version__,
        'official_score': None, 'official_metric_verified': False,
        'official_schema_verified': False, 'uses_test_labels': False,
        'config': asdict(config), 'columns': columns, 'cache_reused': reused,
        'inputs': {'train': identity['train'], 'test': digest(test), 'template': digest(sample_submission),
                   'candidates': identity['candidates']},
        'rdkit_version': rdBase.rdkitVersion,
        'reference_spectra': reference_count, 'reference_structures': structure_count,
        'prediction_count': len(predictions), 'compounds': details,
        'limitations': [
            'No competitive pretrained generative model or validated forward model is included.',
            'Default prediction selects only from the available mass-compatible structure catalog.',
            'Optional graph proposals are bounded, formula-preserving and heuristically ranked.',
            'Scores/margins are not correctness probabilities. Format matching is not Kaggle acceptance.',
        ],
    }
    import io
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: k, pred: canonical(predictions[k])} for k in expected)
    text = buffer.getvalue()
    report['submission_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    # Write CSV last; a failed prediction never produces a partial submission.
    atomic_text(sidecar, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    atomic_text(output, text)
    return report


def structure_fold(smiles: str, folds: int = 5) -> int:
    """Group splitting helper: every spectrum of the same 2D graph stays together."""
    if folds < 2:
        raise ValueError('At least two folds required')
    return int.from_bytes(hashlib.sha256(canonical(smiles).encode()).digest()[:8], 'big') % folds
