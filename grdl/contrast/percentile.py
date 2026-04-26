# -*- coding: utf-8 -*-
"""
Percentile-based contrast stretch — re-export shim.

Canonical home is :mod:`grdl.contrast`; the implementation continues to
live in :mod:`grdl.image_processing.intensity` for backward compatibility
with existing callers.

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-26

Modified
--------
2026-04-26
"""

from grdl.image_processing.intensity import PercentileStretch  # noqa: F401

__all__ = ['PercentileStretch']
