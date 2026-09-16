"""One repository-root entry point for legacy, V1/V2, and V3 inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'work' / 'casmi26'))


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--bundle', type=Path)
    options, _ = parser.parse_known_args(arguments)
    production = '--production' in arguments or options.bundle is not None
    arguments = [value for value in arguments if value != '--production']
    if options.bundle is not None:
        folder = options.bundle
        manifests = [name for name in ('bundle.json', 'v3-bundle.json') if (folder / name).is_file()]
        if len(manifests) > 1:
            parser.error('Ambiguous model bundle: V1/V2 and V3 manifests coexist; select one versioned folder')
        if not manifests:
            parser.error('Missing model manifest: expected bundle.json or v3-bundle.json in ' + str(folder))
        if manifests[0] == 'v3-bundle.json':
            from casmi26.inference_v3 import main as predict_main
        else:
            from casmi26.portable import main as predict_main
    elif production:
        from casmi26.portable import main as predict_main
    else:
        from casmi26.cli import main as predict_main
    try:
        return predict_main(arguments)
    except (OSError, ValueError) as exc:
        print('Prediction stopped: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
