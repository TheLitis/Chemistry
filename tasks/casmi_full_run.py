"""Run the reviewed scientific experiment and package only its successful output."""
from pathlib import Path
import subprocess
import sys


def main():
    tasks=Path(__file__).resolve().parent
    for name in ('casmi_progress.py','casmi_package.py'):
        subprocess.run([sys.executable,str(tasks/name)],check=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
