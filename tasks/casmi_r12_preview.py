"""Publish/collect one private R12 diagnostic. Never makes a contest submission."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = 'r12-ion-view-preview-v1'
KERNEL = 'thelindortis/casmi26-r12-ion-view-diagnostic'
COMPETITION = 'enveda-CASMI26-molecule-id-mass-spectra'
DATASETS = ('prvsiyan/casmi26-fp-models-v2', 'aidensong123/casmi26-offline-rdkit-2026033',
            'prvsiyan/casmi26-ranker-features', 'prvsiyan/coconut-casmi26-candidates')
ANCHOR_SHA = 'f80c2d7725fdf6288e1c788f599f39ec82530778d5e4bb9c5de1a059af370f6e'
BASE_CSV_SHA = '2badfe4735facab3d2da4387a4ce96b0032446be4cd1514854adb84fef7be6b9'
VARIANTS = ('aligned_rows', 'single_only', 'ion_views')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def worker_state(text):
    hit = re.search(r'KernelWorkerStatus\.(RUNNING|QUEUED|COMPLETE|ERROR|CANCELLED)\b', text, re.I)
    if not hit:
        raise ValueError('No explicit Kaggle worker status')
    return hit.group(1).lower()


def may_publish(journal):
    return not journal.get('publish_attempted', False)


def code_hash(book):
    cells = [''.join(c['source']) for c in book['cells'] if c['cell_type'] == 'code']
    return hashlib.sha256(json.dumps(cells, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def verify_remote(expected, actual, metadata):
    if code_hash(expected) != code_hash(actual):
        raise ValueError('Remote notebook code differs')
    if (metadata.get('id') != KERNEL or type(metadata.get('id_no')) is not int
        or metadata['id_no'] <= 0 or metadata.get('is_private') not in (True, 'true')
        or metadata.get('enable_internet') not in (False, 'false')
        or set(metadata.get('dataset_sources', [])) != set(DATASETS)
        or metadata.get('competition_sources') != [COMPETITION]
        or metadata.get('kernel_sources') or metadata.get('model_sources')):
        raise ValueError('Remote notebook identity, privacy or inputs differ')
    return {'id': KERNEL, 'id_no': metadata['id_no'], 'all_code_sha256': code_hash(actual)}


def read_predictions(path):
    result = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['molecule_id', 'smiles']:
            raise ValueError('Unexpected CSV columns')
        for row in reader:
            key, text = row['molecule_id'], row['smiles']
            tokens = text.split(';') if isinstance(text, str) else []
            if not key or key in result or not 1 <= len(tokens) <= 25 or any(not s.strip() for s in tokens):
                raise ValueError('Duplicate/missing ID or invalid output field')
            result[key] = tokens
    if not result:
        raise ValueError('Empty predictions')
    return result


def compare_predictions(base, alternative):
    if set(base) != set(alternative):
        raise ValueError('Alternative IDs differ from current frozen branch')
    return {'rows': len(base), 'changed_top1': sum(base[k][0] != alternative[k][0] for k in base),
            'changed_rows': sum(base[k] != alternative[k] for k in base)}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(repo, state, root, out):
    source = state / 'artifacts/casmi26/public-baseline-v17-repro-20260919'
    anchor = source / 'notebook/baseline.ipynb'
    if digest(anchor) != ANCHOR_SHA:
        raise ValueError('Scored source differs from R12 anchor')
    preview = source / 'output/reproduction.json'
    if read(preview).get('submission_sha256') != BASE_CSV_SHA:
        raise ValueError('Missing confirmed baseline preview')
    bundle = root / 'notebook'
    bundle.mkdir(exist_ok=True)
    support = repo / 'work/casmi26/research/r12-ion-views'
    for relative in ('build_ablation.py', 'src/spectral_views.py', 'src/ion_model_views.py'):
        dest = bundle / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(support / relative, dest)
    shutil.copy2(anchor, bundle / 'baseline-anchor.ipynb')
    (bundle / 'evidence').mkdir(exist_ok=True)
    shutil.copy2(preview, bundle / 'evidence/anchor-preview.json')
    result = subprocess.run([sys.executable, str(bundle / 'build_ablation.py')],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
    (out / 'build.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Notebook build failed')
    notebook = bundle / 'r12-ion-view-ablation.ipynb'
    book = read(notebook)
    for cell in book['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<r12-cell>', 'exec')
    original = read(anchor)
    for i in range(3, 11):
        if original['cells'][i]['source'] != book['cells'][i]['source']:
            raise ValueError('Non-neural scientific cell was modified')
    meta = read(bundle / 'kernel-metadata.json')
    if meta.get('id') != KERNEL or not meta.get('is_private') or meta.get('enable_internet'):
        raise ValueError('Invalid build metadata')
    return {'notebook_sha256': digest(notebook), 'all_code_sha256': code_hash(book),
            'metadata_sha256': digest(bundle / 'kernel-metadata.json'),
            'anchor_sha256': ANCHOR_SHA, 'new_training': False, 'new_submissions': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('publish', 'collect'), required=True)
    args = parser.parse_args()
    state = Path(os.environ['CHEMISTRY_STATE_ROOT'])
    py = state / 'envs/casmi26/python.exe'
    if Path(sys.executable).resolve() != py.resolve():
        return subprocess.call([str(py), str(Path(__file__).resolve())] + sys.argv[1:],
                               env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})
    repo = Path(__file__).resolve().parents[1]
    out = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT'])
    out.mkdir(parents=True, exist_ok=True)
    root = state / 'artifacts/casmi26' / ROOT
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'action.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    journal_path = root / 'journal.json'
    journal = read(journal_path) if journal_path.exists() else {}
    report = {'stage': args.stage, 'new_notebook_versions': 0, 'new_submissions': 0,
              'incumbent_changed': False, 'official_score': None, 'commit': os.environ.get('GITHUB_SHA')}
    prep = load('r12_prep', repo / 'tasks/casmi_prepare.py')
    env = prep.kaggle_environment(state, dict(os.environ))
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8')

    def cli(name, arguments, timeout=180):
        r = subprocess.run([str(py), '-c', 'from kaggle.cli import main;main()'] + arguments,
                           env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=timeout)
        text = prep.redact(r.stdout + '\n' + r.stderr, env)
        text = re.sub(r'(https?://[^\s?]+)\?[^\s]+', r'\1?[QUERY_REDACTED]', text)
        (out / (name + '.log')).write_text(text, encoding='utf-8')
        return r.returncode, text

    try:
        if args.stage == 'publish' and may_publish(journal):
            rc, text = cli('existing-preview', ['kernels', 'status', KERNEL])
            if rc == 0 or not re.search(r'404|not found|does not exist', text, re.I):
                raise RuntimeError('Remote absence not established; no new publication attempted')
            journal.update(prepare(repo, state, root, out))
            journal['publish_attempted'] = True
            journal['publish_attempted_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            dump(journal_path, journal)
            rc, text = cli('one-private-push', ['kernels', 'push', '-p', str(root / 'notebook'), '-t', '32400'])
            report['new_notebook_versions'] = 1 if rc == 0 else 0
            hit = re.search(r'Kernel version\s+(\d+)', text, re.I)
            if rc or hit is None or int(hit.group(1)) != 1:
                raise RuntimeError('Publication failed/ambiguous or not version 1; do not push again')
            journal['kernel_version'] = 1
            dump(journal_path, journal)
        if journal.get('kernel_version') != 1:
            raise ValueError('First publication not confirmed; no repeat allowed')
        notebook = root / 'notebook/r12-ion-view-ablation.ipynb'
        if digest(notebook) != journal['notebook_sha256']:
            raise ValueError('Frozen diagnostic notebook changed locally')
        remote = root / 'remote-source'
        remote.mkdir(exist_ok=True)
        rc, _ = cli('read-back', ['kernels', 'pull', KERNEL, '-p', str(remote), '--metadata'])
        if rc:
            raise RuntimeError('Cannot read back canonical notebook')
        metadata = read(remote / 'kernel-metadata.json')
        actual = remote / Path(metadata['code_file']).name
        identity = verify_remote(read(notebook), read(actual), metadata)
        if journal.get('remote_identity', identity) != identity:
            raise ValueError('Canonical notebook identity changed')
        journal['remote_identity'] = identity
        dump(journal_path, journal)
        rc, text = cli('worker-status', ['kernels', 'status', KERNEL])
        if rc:
            raise RuntimeError('Worker status unavailable')
        status = worker_state(text)
        report['worker_status'] = status
        if status in ('error', 'cancelled'):
            cli('failed-preview-output', ['kernels', 'output', KERNEL, '-p', str(out / 'preview-failure')], timeout=600)
            raise RuntimeError('Diagnostic notebook failed; no contest attempt was made')
        if status != 'complete':
            report['status'] = 'preview_pending'
        else:
            output = root / 'output'
            output.mkdir(exist_ok=True)
            pattern = r'(^|/)(submission\.csv|ablation-(aligned_rows|single_only|ion_views)\.csv|ion-ablation\.json)$'
            rc, _ = cli('download-output', ['kernels', 'output', KERNEL, '-p', str(output),
                                          '--file-pattern', pattern, '--page-size', '200'], timeout=600)
            if rc:
                raise RuntimeError('Could not collect diagnostic files')
            result = read(output / 'ion-ablation.json')
            if (result.get('status') != 'diagnostic_completed' or result.get('new_submissions') != 0
                or result.get('test_labels_used') is not False or result.get('new_official_score') is not None
                or result.get('baseline_source_sha256') != ANCHOR_SHA):
                raise ValueError('Unexpected diagnostic lineage')
            base = read_predictions(output / 'submission.csv')
            checks = {}
            from rdkit import Chem
            for variant in ('frozen',) + VARIANTS:
                filename = 'submission.csv' if variant == 'frozen' else 'ablation-' + variant + '.csv'
                path = output / filename
                rows = read_predictions(path)
                checks[variant] = compare_predictions(base, rows)
                invalid = sum(Chem.MolFromSmiles(s) is None for values in rows.values() for s in values)
                if invalid:
                    raise ValueError('Invalid chemical structures in ' + variant)
                expected = result['baseline_csv_sha256'] if variant == 'frozen' else result['variants'][variant]['csv_sha256']
                if digest(path) != expected:
                    raise ValueError('Downloaded CSV hash differs')
                checks[variant]['sha256'] = digest(path)
                checks[variant]['valid_structures'] = True
                shutil.copy2(path, out / filename)
            shutil.copy2(output / 'ion-ablation.json', out / 'ion-ablation.json')
            report.update(status='diagnostic_collected', checks=checks, notebook_result=result,
                          baseline_byte_identical=digest(output / 'submission.csv') == BASE_CSV_SHA,
                          accuracy_established=False)
            journal['collected'] = True
            dump(journal_path, journal)
        report['journal'] = journal
        report['checked_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        dump(out / 'r12-preview-status.json', report)
        dump(root / 'latest-status.json', report)
        print('R12_PREVIEW_RESULT ' + json.dumps({k: v for k, v in report.items() if k != 'notebook_result'}), flush=True)
        return 0
    except Exception as exc:
        report.update(status='blocked', error_type=type(exc).__name__, error=str(exc), journal=journal)
        dump(out / 'r12-preview-status.json', report)
        print('R12_PREVIEW_BLOCKED ' + str(exc), flush=True)
        return 2
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
