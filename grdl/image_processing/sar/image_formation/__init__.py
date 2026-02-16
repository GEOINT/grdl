# -*- coding: utf-8 -*-
"""
SAR Image Formation - Transform phase history data into complex SAR imagery.

Provides the Polar Format Algorithm (PFA) pipeline for forming complex SAR
images from CPHD (Compensated Phase History Data). The pipeline includes:

- ``CollectionGeometry``: Compute coordinate systems, angles, and ARP
  polynomials from CPHD per-vector parameters.
- ``PolarGrid``: Compute k-space annulus bounds and resampled grid dimensions.
- ``PolarFormatAlgorithm``: Range/azimuth interpolation and IFFT to form
  the complex SAR image.

Each module is independently usable for intermediate inspection (tapout
points) or can be composed for end-to-end image formation.

Dependencies
------------
scipy

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
2026-02-12

Modified
--------
2026-02-12
"""

from grdl.image_processing.sar.image_formation.base import (
    ImageFormationAlgorithm,
)
from grdl.image_processing.sar.image_formation.geometry import (
    CollectionGeometry,
)
from grdl.image_processing.sar.image_formation.polar_grid import PolarGrid
from grdl.image_processing.sar.image_formation.pfa import (
    PolarFormatAlgorithm,
)
from grdl.image_processing.sar.image_formation.subaperture import (
    SubaperturePartitioner,
)
from grdl.image_processing.sar.image_formation.stripmap_pfa import (
    StripmapPFA,
)
from grdl.image_processing.sar.image_formation.rda import (
    RangeDopplerAlgorithm,
)
from grdl.image_processing.sar.image_formation.ffbp import (
    FastBackProjection,
)
__all__ = [
    'ImageFormationAlgorithm',
    'CollectionGeometry',
    'PolarGrid',
    'PolarFormatAlgorithm',
    'SubaperturePartitioner',
    'StripmapPFA',
    'RangeDopplerAlgorithm',
    'FastBackProjection',
]
