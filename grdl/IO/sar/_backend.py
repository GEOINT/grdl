# -*- coding: utf-8 -*-
"""
SAR Backend Detection - Detect available SAR reading libraries.

Probes for sarkit and sarpy at import time. Provides helper functions
that SAR readers use to select the best available backend or raise
clear errors when no backend is available.

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-09

Modified
--------
2026-02-09
"""

_HAS_SARKIT = False
_HAS_SARPY = False

try:
    import sarkit  # noqa: F401
    _HAS_SARKIT = True
except ImportError:
    pass

try:
    import sarpy  # noqa: F401
    _HAS_SARPY = True
except ImportError:
    pass


def require_sar_backend(format_name: str) -> str:
    """Return the best available SAR backend name.

    Parameters
    ----------
    format_name : str
        Human-readable format name for error messages (e.g. ``'SICD'``).

    Returns
    -------
    str
        ``'sarkit'`` or ``'sarpy'``.

    Raises
    ------
    ImportError
        If neither sarkit nor sarpy is installed.
    """
    if _HAS_SARKIT:
        return 'sarkit'
    if _HAS_SARPY:
        return 'sarpy'
    raise ImportError(
        f"Reading {format_name} files requires sarkit or sarpy. "
        f"Install with: pip install sarkit"
    )


def require_sarkit(format_name: str) -> None:
    """Require sarkit specifically (no sarpy fallback).

    Used for formats where sarpy does not provide adequate support
    (CRSD, SIDD).

    Parameters
    ----------
    format_name : str
        Human-readable format name for error messages.

    Raises
    ------
    ImportError
        If sarkit is not installed.
    """
    if not _HAS_SARKIT:
        raise ImportError(
            f"Reading {format_name} files requires sarkit. "
            f"Install with: pip install sarkit"
        )
