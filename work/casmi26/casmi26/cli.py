from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .engine import Config, predict


def find_data(path: Path, stem: str) -> Path:
    for parent in (path if path.is_dir() else path.parent, path.parent, Path('data')):
        for suffix in ('', '.parquet', '.jsonl', '.mgf', '.csv', '.tsv'):
            candidate = parent / (stem + suffix)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f'{stem} not found alongside {path}; pass its explicit path')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='MS/MS compound groups -> ranked 2D structures per molecule. '
                                                 'Experimental; no measured official score.')
    parser.add_argument('--test', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--train', type=Path, help='Reference spectra; auto-detected alongside test')
    parser.add_argument('--sample-submission', type=Path, help='Actual competition template; auto-detected')
    parser.add_argument('--candidates', type=Path, help='Optional permitted local SMILES catalog, no test answers')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--columns', type=Path, help='JSON mapping canonical input fields to actual column names')
    parser.add_argument('--id-column')
    parser.add_argument('--prediction-column')
    parser.add_argument('--precursor-ppm', type=float, default=15)
    parser.add_argument('--precursor-da', type=float, default=.003)
    parser.add_argument('--fragment-ppm', type=float, default=20)
    parser.add_argument('--fragment-da', type=float, default=.01)
    parser.add_argument('--isomer-budget', type=int, default=0,
                        help='Opt-in bounded graph proposals; unvalidated heuristic, 0 disables')
    parser.add_argument('--fragment-weight', type=float, default=.3)
    parser.add_argument('--model', type=Path, help='Trained numeric fingerprint model (.npz)')
    parser.add_argument('--neural-weight', type=float, default=.5)
    parser.add_argument('--top-k', type=int, default=25)
    parser.add_argument('--strict-training', action='store_true', help='Stop rather than count/reject invalid reference rows')
    parser.add_argument('--allow-rdkit-version-mismatch', action='store_true', help='Development only; not competition matching')
    args = parser.parse_args(argv)
    try:
        if not args.allow_rdkit_version_mismatch:
            from .metric import require_official_rdkit
            require_official_rdkit()
        train = args.train or find_data(args.test, 'train')
        template = args.sample_submission or find_data(args.test, 'sample_submission')
        columns = json.loads(args.columns.read_text(encoding='utf-8-sig')) if args.columns else {}
        if not isinstance(columns, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in columns.items()):
            raise ValueError('--columns must be a JSON object mapping string field names to string column names')
        report = predict(args.test, train, template, args.output, cache=args.cache, columns=columns,
                         candidates=args.candidates, id_column=args.id_column, prediction_column=args.prediction_column, model=args.model,
                         config=Config(args.precursor_ppm, args.precursor_da, args.fragment_ppm,
                                       args.fragment_da, args.isomer_budget, args.fragment_weight,
                                       top_k=args.top_k, curate_training=not args.strict_training, neural_weight=args.neural_weight))
        print(json.dumps({'output': str(args.output), 'predictions': report['prediction_count'],
                          'official_score': None, 'status': report['status']}, ensure_ascii=False))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f'Prediction stopped: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
