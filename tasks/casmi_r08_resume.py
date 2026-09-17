"""Resume the sealed R07-vs-FIORA study using explicit isolated import paths.

Reads existing prepared inputs and pinned dependencies. No installation,
model fitting, input changes, Kaggle writes, or service/ACL changes.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = 'e19ef82c9a6cb9dbac92bce23e914008f1aeb44e'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    state = Path(os.environ['CHEMISTRY_STATE_ROOT'])
    repo = Path(__file__).resolve().parents[1]
    out = Path(os.environ['CHEMISTRY_REQUEST_OUTPUT']); out.mkdir(parents=True, exist_ok=True)
    root = state/'artifacts/casmi26/research-r08/forward-v1'
    python = state/'envs/casmi26/python.exe'
    helper = load('r08_paths', repo/'tasks/r08_python_command.py')
    paths = [root/'deps', root/('fiora-'+SOURCE), repo/'work/casmi26', repo/'tasks']
    task = load('r08_existing', repo/'tasks/casmi_r08_forward.py')
    env = task.env_clean()
    model = paths[1]/'fiora/resources/models/fiora_OS_v1.0.0.pt'
    env['FIORA_TEST_MODEL'] = str(model)
    # Store only committed project source, never local model/provider files.
    subprocess.run(['git', '-c', 'safe.directory='+str(repo), '-C', str(repo),
                    'archive', '--format=zip', '--output='+str(out/'source.zip'), 'HEAD'],
                   check=True, timeout=60)
    started = time.monotonic()
    def run(name, code, arguments=(), timeout=300, stream=False):
        command = helper.python_command(python, paths, code, arguments)
        if stream:
            result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, timeout=timeout)
        else:
            result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
            (out/(name+'.log')).write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
            print(result.stdout[-3000:], flush=True)
            if result.returncode: print(result.stderr[-3000:], flush=True)
        if result.returncode: raise RuntimeError(name+' failed; source inputs and checkpoints preserved')
    run('explicit-imports', "import sys,json,pandas,numpy,rdkit,pyarrow,torch; from fiora.cli.predict import build_metabolites; from casmi26.fiora_adapter import verify_model_files; from pathlib import Path; import os; verify_model_files(Path(os.environ['FIORA_TEST_MODEL'])); assert numpy.__version__=='2.3.5'; assert rdkit.__version__=='2026.03.3'; assert pyarrow.__version__=='21.0.0'; print(json.dumps({'isolated':sys.flags.isolated,'ignore_environment':sys.flags.ignore_environment,'pandas_file':pandas.__file__,'torch':torch.__version__,'numpy':numpy.__version__,'rdkit':rdkit.__version__}))", timeout=90)
    run('tests', "import pytest,sys; raise SystemExit(pytest.main(sys.argv[1:]))",
        ['-q', str(repo/'work/casmi26/tests')], timeout=480)
    # execute(), not main(), avoids the old child's ignored PYTHONPATH path.
    code = "import sys; from pathlib import Path; import casmi_r08_forward; casmi_r08_forward.execute(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])"
    run('simulation', code, [str(root), str(out), args.device], timeout=5400, stream=True)
    evaluation = out/'evaluation'; evaluation.mkdir(exist_ok=True)
    run('evaluation', "import casmi_r08_evaluate; raise SystemExit(casmi_r08_evaluate.main())",
        ['--root', str(root), '--output', str(evaluation)], timeout=3600, stream=True)
    report = {'status':'completed', 'commit':os.environ.get('GITHUB_SHA'),
              'device':args.device, 'seconds':time.monotonic()-started,
              'base_python_config_changed':False,'new_submissions':0,'model_refit':False,
              'evaluation_report':str(root/'report.json')}
    (out/'resume.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('R08_RESUME_COMPLETE '+json.dumps(report),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
