# -*- coding: utf-8 -*-
"""
GMTI submodule - STANAG 4607 Ground Moving Target Indicator IO.

Reader, writer, and helpers for STANAG 4607 GMTI data. Provides:

- ``STANAG4607Reader`` — parse a 4607 file into typed segment dataclasses.
- ``STANAG4607Writer`` — write typed segment dataclasses back to a file.
- ``open_gmti(path)`` — convenience factory mirroring ``open_sar``,
  ``open_eo``, etc.
- Helpers for dwell footprint geometry, ground-relative velocity,
  target-report filtering, and quick-look summary statistics.
- ``build_steering_matrix_from_cphd_metadata`` — derive a scene-
  projected per-channel steering matrix from the CPHD ``Antenna``
  section. Given a set of scene cross-range offsets from the SRP (one
  per image column), it evaluates each channel's ``Array``/``Element``
  gain/phase polynomials at the SRP-relative look-direction cosines of
  those scene points and returns a matrix whose columns align with the
  image cross-range bins. Use this as the column dictionary for STAP
  detectors driven by metadata rather than data.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-29

Modified
--------
2026-04-29
"""

from grdl.IO.gmti.cphd_steering import (
    CPHDMetadataSteering,
    build_steering_matrix_from_cphd_metadata,
)
from grdl.IO.gmti.helpers import (
    dwell_footprint_polygon,
    filter_target_reports,
    ground_relative_velocity,
    summarize,
)
from grdl.IO.gmti.stanag4607 import STANAG4607Reader, open_gmti
from grdl.IO.gmti.stanag4607_writer import STANAG4607Writer

__all__ = [
    'STANAG4607Reader',
    'STANAG4607Writer',
    'open_gmti',
    'dwell_footprint_polygon',
    'ground_relative_velocity',
    'filter_target_reports',
    'summarize',
    'CPHDMetadataSteering',
    'build_steering_matrix_from_cphd_metadata',
]
