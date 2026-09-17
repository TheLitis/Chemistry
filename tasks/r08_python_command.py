"""Explicit, per-process imports for the embedded isolated Windows runtime.

Never edits Python's ._pth, registry, global site-packages or user configuration.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable


def python_command(python: str | Path, paths: Iterable[str | Path], code: str,
                   arguments: Iterable[str] = ()) -> list[str]:
    resolved = []
    for value in paths:
        path = Path(value).resolve()
        if not path.is_dir():
            raise FileNotFoundError('Missing explicit Python import directory: ' + str(path))
        if str(path) not in resolved:
            resolved.append(str(path))
    if not isinstance(code, str) or not code.strip():
        raise ValueError('A nonempty Python program is required')
    bootstrap = 'import sys; sys.path[:0] = ' + repr(resolved) + '\n' + code
    return [str(python), '-I', '-c', bootstrap, *map(str, arguments)]
