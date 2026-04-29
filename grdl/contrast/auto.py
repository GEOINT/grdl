# -*- coding: utf-8 -*-
"""
Auto-select a contrast operator from imagery metadata.

Inspects an :class:`~grdl.IO.models.base.ImageMetadata` and returns the
name of the :mod:`grdl.contrast` operator best suited for view-time
display of that modality.  Mapping is intentionally conservative — when
the metadata type is unrecognized it falls back to ``'percentile'``,
which works on every modality without tuning.

Recommended operators by modality
---------------------------------
SAR (SICD, SIDD, CPHD, CRSD, NISAR, Sentinel-1 SLC, TerraSAR, BIOMASS)
    ``'brighter'`` — Mangis density preset (``dmin=60``, ``mmult=40``).
    Brighter than vanilla ``MangisDensity`` for typical SAR scenes;
    works well without per-image tuning.

EO NITF (RPC/RSM-tagged optical)
    ``'gamma'`` — pooled 2/99 percentile normalization with gamma = 1.4
    for typical 8-bit display.

MSI / HSI (Sentinel-2, VIIRS, ASTER)
    ``'percentile'`` — per-band 2/99 clip; predictable across bands.

GeoTIFF / unknown
    ``'percentile'`` — universal safe default.

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

# Standard library
from typing import Any


# Metadata class names that indicate each modality.  Detection by class
# name avoids importing the concrete metadata classes here (no circular
# dependency, no extras-gating).
_SAR_METADATA = frozenset({
    'SICDMetadata',
    'SIDDMetadata',
    'CPHDMetadata',
    'CRSDMetadata',
    'BIOMASSMetadata',
    'Sentinel1SLCMetadata',
    'TerraSARMetadata',
    'NISARMetadata',
})

_MSI_METADATA = frozenset({
    'Sentinel2Metadata',
    'VIIRSMetadata',
    'ASTERMetadata',
})

_EO_METADATA = frozenset({
    'EONITFMetadata',
})


def auto_select(metadata: Any) -> str:
    """Return the recommended ``grdl.contrast`` operator name.

    Parameters
    ----------
    metadata : ImageMetadata or any object
        Reader metadata object (e.g., ``reader.metadata``).  ``None`` is
        accepted and falls through to the universal default.

    Returns
    -------
    str
        Operator name — one of the choices documented in
        :mod:`grdl.contrast` (``'nrl'``, ``'gamma'``, ``'percentile'``,
        etc.).  Always safe to feed to a downstream dispatcher.

    Examples
    --------
    >>> from grdl.IO.generic import open_any
    >>> from grdl.contrast import auto_select
    >>> reader = open_any('/data/scene.nitf')
    >>> auto_select(reader.metadata)
    'brighter'
    """
    if metadata is None:
        return 'percentile'

    type_name = type(metadata).__name__

    if type_name in _SAR_METADATA:
        return 'brighter'
    if type_name in _EO_METADATA:
        return 'gamma'
    if type_name in _MSI_METADATA:
        return 'percentile'

    # Unknown — universal safe default.
    return 'percentile'
