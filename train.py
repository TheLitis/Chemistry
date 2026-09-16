"""Repository-root training entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'work' / 'casmi26'))

if __name__ == '__main__':
    from casmi26.training import main
    raise SystemExit(main())
