# -*- coding: utf-8 -*-
"""GRDL version helpers.

Uses ``pyproject.toml`` as the source of truth when running from a source
checkout, and falls back to installed package metadata otherwise.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import tomllib


def _version_from_pyproject() -> str | None:
    """Read the version from the repository's pyproject.toml when present."""
    pyproject_path = Path(__file__).resolve().parent.parent / 'pyproject.toml'
    if not pyproject_path.is_file():
        return None

    try:
        with pyproject_path.open('rb') as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None

    version = data.get('project', {}).get('version')
    return version if isinstance(version, str) and version else None


def get_version() -> str:
    """Resolve the GRDL version from source metadata or installed metadata."""
    version = _version_from_pyproject()
    if version is not None:
        return version

    try:
        return importlib.metadata.version('grdl')
    except importlib.metadata.PackageNotFoundError:
        return 'unknown'


__version__ = get_version()