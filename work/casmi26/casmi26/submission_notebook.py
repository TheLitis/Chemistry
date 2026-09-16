"""Build an offline Kaggle inference notebook with embedded, inspectable source."""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import zipfile


def build_notebook(output: Path) -> Path:
    package=Path(__file__).resolve().parent
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.glob('*.py')):
            archive.writestr('casmi26/'+path.name,path.read_bytes())
    source64=base64.b64encode(stream.getvalue()).decode('ascii')
    cells=[]
    def md(text):
        cells.append({'cell_type':'markdown','id':f'cell-{len(cells)}','metadata':{},'source':text.splitlines(True)})
    def code(text):
        cells.append({'cell_type':'code','id':f'cell-{len(cells)}','metadata':{},'source':text.splitlines(True),
                      'execution_count':None,'outputs':[]})
    md('# CASMI26 — offline structure ranking\n\nAttach the competition and the private model-assets dataset; disable Internet. '
       'This notebook predicts from the **current test mount**, including the hidden rerun. '
       'No test answers or precomputed test predictions are embedded. '
       'This is a catalog-retrieval model, not a guarantee of exact recovery or a measured Kaggle score.\n')
    code("""from pathlib import Path
import os, sys, json, base64, io, zipfile, subprocess
INPUT_ROOT = Path(os.environ.get('CASMI_INPUT_ROOT', '/kaggle/input'))
WORK_ROOT = Path(os.environ.get('CASMI_WORK_ROOT', '/kaggle/working'))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
datasets = [p.parent for p in INPUT_ROOT.rglob('test.parquet') if (p.parent/'train.parquet').is_file()]
if len(datasets) != 1:
    raise RuntimeError(f'Expected one current competition dataset, found {len(datasets)}')
DATA = datasets[0]
current_test = DATA/'test.parquet'
if os.environ.get('CASMI_BUNDLE_DIR'):
    ASSETS = Path(os.environ['CASMI_BUNDLE_DIR'])
else:
    bundles = list(INPUT_ROOT.rglob('bundle.json'))
    if len(bundles) != 1:
        raise RuntimeError('Attach exactly one model-assets dataset containing bundle.json')
    ASSETS = bundles[0].parent
print('Current input:', current_test)
print('Model assets:', ASSETS)
""")
    md('## Reconstruct the versioned inference code\nThe following archive contains Python source only. Training arrays, credentials and test identifiers are not included.\n')
    code(f"SOURCE_ZIP_BASE64 = {source64!r}\n"+"""CODE = WORK_ROOT/'_casmi26_code'
CODE.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ZIP_BASE64))) as archive:
    for member in archive.infolist():
        path = Path(member.filename)
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'casmi26' or path.suffix != '.py':
            raise RuntimeError('Unexpected source archive member')
    archive.extractall(CODE)
print('Inference source reconstructed in', CODE)
""")
    md('## Offline dependencies\nLinux wheels in the assets must match the notebook Python version. Installation uses only local files. '
       'The Windows verification run instead uses the already pinned isolated scientific environment.\n')
    code("""DEPS = WORK_ROOT/'_casmi26_deps'
DEPS.mkdir(exist_ok=True)
if sys.platform.startswith('linux'):
    wheel_dir = ASSETS/'wheels'
    if not wheel_dir.is_dir():
        raise RuntimeError('Missing offline Linux wheels')
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
        '--find-links', str(wheel_dir), '--target', str(DEPS), '--upgrade',
        'numpy==2.3.5', 'rdkit==2026.3.3', 'pyarrow==21.0.0'],
        check=True, capture_output=True, text=True)
    print(result.stdout[-4000:])
launcher = CODE/'run_inference.py'
launcher.write_text('import sys\\nsys.path[:0] = '+repr([str(DEPS),str(CODE)])+
    '\\nfrom casmi26.portable import main\\nraise SystemExit(main())\\n', encoding='utf-8')
env = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4',
           PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
check = 'import sys;sys.path[:0]='+repr([str(DEPS),str(CODE)])+';from casmi26.metric import require_official_rdkit;require_official_rdkit()'
subprocess.run([sys.executable, '-c', check], check=True, env=env)
""")
    md('## Predict every molecule from all its available spectra\nReference candidates are rebuilt for these inputs. '
       'The visible template supplies ordering only when its IDs match the current test; hidden IDs come from the hidden test itself.\n')
    code("""command = [sys.executable, str(launcher), '--test', str(current_test),
           '--train', str(DATA/'train.parquet'), '--bundle', str(ASSETS),
           '--output', str(WORK_ROOT/'submission.csv')]
if (DATA/'sample_submission.csv').is_file():
    command += ['--sample-submission', str(DATA/'sample_submission.csv')]
result = subprocess.run(command, check=True, env=env, capture_output=True, text=True)
print(result.stdout[-8000:])
""")
    md('## Output checks\nA complete CSV is not a correctness score. Kaggle alone evaluates the hidden answers.\n')
    code("""report = json.loads((WORK_ROOT/'submission.csv.report.json').read_text())
print(json.dumps({k:report[k] for k in ('prediction_count','test_spectra','submission_sha256',
    'mass_incompatible_fallbacks','official_score')}, indent=2))
assert (WORK_ROOT/'submission.csv').stat().st_size > 0
""")
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
          'language_info':{'name':'python'}},'cells':cells}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return output
