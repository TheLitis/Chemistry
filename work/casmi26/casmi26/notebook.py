"""Build a source-embedded offline Kaggle notebook. No credentials are bundled."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def build_notebook(output: Path) -> None:
    root=Path(__file__).resolve().parent
    memory=io.BytesIO()
    with ZipFile(memory,'w',ZIP_DEFLATED) as z:
        for file in sorted(root.glob('*.py')):
            z.writestr('casmi26/'+file.name,file.read_bytes())
    archive=memory.getvalue();encoded=base64.b64encode(archive).decode('ascii')
    first='''from pathlib import Path
import sys, subprocess, importlib.metadata
required = {'rdkit': '2026.3.3', 'numpy': '2.3.5', 'pyarrow': '21.0.0'}
def matches(package, version):
    try:
        return importlib.metadata.version(package) == version
    except importlib.metadata.PackageNotFoundError:
        return False
if not all(matches(k,v) for k,v in required.items()):
    wheel_dirs = sorted({str(p.parent) for p in Path('/kaggle/input').rglob('*.whl')})
    if not wheel_dirs:
        raise RuntimeError('Attach the offline wheelhouse dataset; network installation is disabled.')
    args = [sys.executable, '-m', 'pip', 'install', '--no-index', '--only-binary=:all:',
            '--target', '/kaggle/working/casmi-deps', '--upgrade']
    for folder in wheel_dirs:
        args += ['--find-links', folder]
    subprocess.run(args + [k+'=='+v for k,v in required.items()], check=True)
    sys.path.insert(0, '/kaggle/working/casmi-deps')
'''
    second=f'''import base64, hashlib, io, zipfile
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(archive).hexdigest()!r}
source_root = Path('/kaggle/working/casmi-source')
source_root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    for member in bundle.infolist():
        target = (source_root/member.filename).resolve()
        if source_root.resolve() not in target.parents:
            raise RuntimeError('Unsafe source archive path')
    bundle.extractall(source_root)
sys.path.insert(0, str(source_root))
from casmi26.metric import require_official_rdkit
require_official_rdkit()
'''
    third='''from casmi26.cli import main
# Select only an unambiguous official train/test/template mount. Do not hardcode IDs.
roots = sorted({p.parent for p in Path('/kaggle/input').rglob('test.parquet')
                if (p.parent/'train.parquet').exists() and (p.parent/'sample_submission.csv').exists()})
if len(roots) != 1:
    raise RuntimeError('Expected one competition data mount, found '+str(roots))
data = roots[0]
models = list(Path('/kaggle/input').rglob('fingerprint-public.npz'))
if len(models) > 1:
    raise RuntimeError('Multiple model artifacts; attach only the intended model dataset')
args = ['--test', str(data/'test.parquet'), '--train', str(data/'train.parquet'),
        '--sample-submission', str(data/'sample_submission.csv'),
        '--output', '/kaggle/working/submission.csv', '--top-k', '25',
        '--id-column', 'molecule_id', '--prediction-column', 'smiles']
if models:
    catalog = models[0].with_suffix('.catalog.csv')
    if not catalog.exists():
        raise RuntimeError('The attached model needs its associated structure catalog')
    args += ['--model', str(models[0]), '--candidates', str(catalog)]
code = main(args)
if code:
    raise RuntimeError('Inference failed; no valid new submission was produced')
# Hidden test IDs and spectra are used at rerun time; no visible-test answer lookup.
import csv
with open('/kaggle/working/submission.csv', newline='') as f:
    rows = list(csv.DictReader(f))
assert rows and all(1 <= len(r['smiles'].split(';')) <= 25 for r in rows)
print('Predicted molecules:', len(rows))
'''
    cells=[{'cell_type':'markdown','metadata':{},'source':[
        '# CASMI26 compound-level structure prediction\n',
        'Run with **Internet disabled**. Attach the official competition data and offline dependencies.\n',
        'Optional fingerprint-public.npz and its catalog enable the external-data ranker.\n',
        'This notebook is not evidence of a hidden-test score. The current generator remains limited.\n']}]
    cells += [{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
               'source':code.splitlines(keepends=True)} for code in (first,second,third)]
    doc={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                 'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(doc,indent=1)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    build_notebook(p.parse_args().output)


if __name__=='__main__':main()
