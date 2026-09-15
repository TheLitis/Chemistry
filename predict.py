"""Repository-root entry point; delivered to the PC by the existing file bridge."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'work' / 'casmi26'))

if __name__ == '__main__':
    from casmi26.cli import main
    raise SystemExit(main())
