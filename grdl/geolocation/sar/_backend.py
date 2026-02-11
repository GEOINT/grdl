# -*- coding: utf-8 -*-
"""
SAR Geolocation Backend Detection - Detect available SAR projection libraries.

Probes for sarpy (specifically ``sarpy.geometry.point_projection``) and
sarkit at import time. Provides boolean flags and a helper function that
SAR geolocation implementations use to select the best available projection
backend or raise clear errors when no backend is available.

Dependencies
------------
sarkit
sarpy

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
2026-02-11

Modified
--------
2026-02-11
"""

_HAS_SARKIT = False
_HAS_SARPY = False

try:
    import sarkit  # noqa: F401
    _HAS_SARKIT = True
except ImportError:
    pass

try:
    from sarpy.geometry import point_projection  # noqa: F401
    _HAS_SARPY = True
except ImportError:
    pass


def require_projection_backend(format_name: str) -> str:
    """Return the best available SAR projection backend name.

    Prefers sarkit when both are available. Falls back to sarpy if sarkit
    is not installed.

    Parameters
    ----------
    format_name : str
        Human-readable format name for error messages (e.g. ``'SICD'``,
        ``'CPHD'``).

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
        f"Geolocation for {format_name} imagery requires sarkit or sarpy "
        f"(sarpy.geometry.point_projection). "
        f"Install with: pip install sarkit"
    )
