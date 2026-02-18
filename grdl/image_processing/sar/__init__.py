# -*- coding: utf-8 -*-
"""
SAR Image Processing - SAR-specific transforms requiring sensor metadata.

Provides image processing operations that operate on complex SAR imagery
and require SAR metadata (bandwidth, weight functions, spatial frequency
parameters) from the reader. All processors accept ``SICDMetadata`` at
construction time.

Oversampling-aware: operations account for signal bandwidth occupying
only a fraction of the DFT extent. Accepts both numpy arrays and
PyTorch tensors (GPU-accelerated FFTs); always returns numpy arrays.

Sub-aperture Decomposition
    ``SublookDecomposition`` — 1D spectral splitting along azimuth or
    range into N sub-aperture looks. Output: ``(N, rows, cols)``.

    ``MultilookDecomposition`` — 2D spectral splitting into an M x N
    grid of sub-aperture looks using ``Tiler`` from ``grdl.data_prep``
    for frequency bin partitioning. Output: ``(M, N, rows, cols)``.

    ``CSIProcessor`` — Coherent Shape Index (color sub-aperture) RGB
    composite. Man-made targets appear coloured; natural clutter greyscale.

Image Formation (``image_formation/``)
    ``PolarFormatAlgorithm`` — Spotlight PFA (polar-to-rectangular
    resampling + 2D IFFT).

    ``StripmapPFA`` — Stripmap PFA variant with subaperture stitching.

    ``RangeDopplerAlgorithm`` — Range-Doppler Algorithm for stripmap SAR
    (range compression, RCMC, azimuth compression).

    ``FastBackProjection`` — FFT-accelerated backprojection with
    subaperture leaf-tree structure.

    ``CollectionGeometry`` — Coordinate systems and geometry from CPHD
    per-vector parameters.

    ``PolarGrid`` — Polar k-space grid computation for PFA.

    ``SubaperturePartitioner`` — Partition aperture for FFBP processing.

Usage
-----
1D sublook decomposition:

    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.image_processing.sar import SublookDecomposition
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     image = reader.read_full()
    ...     sublook = SublookDecomposition(reader.metadata, num_looks=3,
    ...                                    overlap=0.1, deweight=True)
    ...     looks = sublook.decompose(image)   # (3, rows, cols) complex
    ...     db = sublook.to_db(looks)          # (3, rows, cols) float

2D multilook decomposition:

    >>> from grdl.image_processing.sar import MultilookDecomposition
    >>>
    >>> with SICDReader('image.nitf') as reader:
    ...     image = reader.read_full()
    ...     ml = MultilookDecomposition(reader.metadata,
    ...                                looks_rg=3, looks_az=3)
    ...     grid = ml.decompose(image)     # (3, 3, rows, cols) complex
    ...     flat = ml.to_flat_stack(grid)  # (9, rows, cols) complex

Dependencies
------------
torch (optional, for GPU-accelerated FFT path)

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
2026-02-10

Modified
--------
2026-02-17
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
