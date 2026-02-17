# -*- coding: utf-8 -*-
"""
SAR Image Processing - SAR-specific transforms requiring sensor metadata.

Provides image processing operations that operate on complex SAR imagery
and require SAR metadata (bandwidth, weight functions, spatial frequency
parameters) from the reader. All processors inherit from ``ImageProcessor``
and accept ``SICDMetadata`` at construction time.

Oversampling-aware: operations account for signal bandwidth occupying
only a fraction of the DFT extent. Accepts both numpy arrays and
PyTorch tensors (GPU-accelerated FFTs); always returns numpy arrays.

Key Classes
-----------
- SublookDecomposition: Split complex SAR image into N sub-aperture looks
- CSIProcessor: Coherent Shape Index (color sub-aperture) RGB composite

Usage
-----
    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.image_processing.sar import SublookDecomposition
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     image = reader.read_full()
    ...     sublook = SublookDecomposition(reader.metadata, num_looks=3,
    ...                                    overlap=0.1, deweight=True)
    ...     looks = sublook.decompose(image)   # (3, rows, cols) complex
    ...     mag = sublook.to_magnitude(looks)  # (3, rows, cols) float
    ...     db = sublook.to_db(looks)          # (3, rows, cols) float

Dependencies
------------
torch (optional, for GPU-accelerated FFT path)

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
2026-02-10

Modified
--------
2026-02-12
"""

from grdl.image_processing.sar.sublook import SublookDecomposition
from grdl.image_processing.sar.multilook import MultilookDecomposition
from grdl.image_processing.sar.csi import CSIProcessor
from grdl.image_processing.sar.image_formation import (
    ImageFormationAlgorithm,
    CollectionGeometry,
    PolarGrid,
    PolarFormatAlgorithm,
    SubaperturePartitioner,
    StripmapPFA,
    RangeDopplerAlgorithm,
    FastBackProjection,
)

__all__ = [
    'SublookDecomposition',
    'MultilookDecomposition',
    'CSIProcessor',
    'ImageFormationAlgorithm',
    'CollectionGeometry',
    'PolarGrid',
    'PolarFormatAlgorithm',
    'SubaperturePartitioner',
    'StripmapPFA',
    'RangeDopplerAlgorithm',
    'FastBackProjection',
]
