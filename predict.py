"""Repository-root inference entry point (legacy and trained-bundle interfaces)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'work' / 'casmi26'))

if __name__ == '__main__':
    arguments = sys.argv[1:]
    if '--production' in arguments or any(a == '--bundle' or a.startswith('--bundle=') for a in arguments):
        from casmi26.portable import main
        arguments = [a for a in arguments if a != '--production']
    else:
        from casmi26.cli import main
    raise SystemExit(main(arguments))
