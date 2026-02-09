# -*- coding: utf-8 -*-
"""
Analyze Particles - Port of ImageJ's Analyze > Analyze Particles.

Performs connected component analysis on a binary or thresholded
image and measures geometric properties (area, centroid, bounding box,
circularity, aspect ratio, solidity) for each detected particle
(connected component). Optionally filters particles by size and
circularity.

Particularly useful for:
- Counting and measuring ships in thresholded SAR maritime imagery
- Building footprint analysis from classified PAN/EO data
- Vegetation patch statistics from NDVI-derived masks (MSI)
- Thermal hotspot characterization (size, shape, distribution)
- Crater or feature counting in planetary remote sensing
- Target detection validation by measuring detected object properties

Attribution
-----------
ImageJ implementation: Wayne Rasband (NIH).
Source: ``ij/plugin/filter/ParticleAnalyzer.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

Dependencies
------------
scipy

Author
------
Jason Fritz

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

# Standard library
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np
from scipy.ndimage import label, find_objects

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags


def _measure_particle(region_mask: np.ndarray,
                      offset_r: int, offset_c: int) -> Dict[str, float]:
    """Measure geometric properties of a single particle.

    Parameters
    ----------
    region_mask : np.ndarray
        2D boolean mask of the particle (cropped to bounding box).
    offset_r : int
        Row offset of the bounding box in the full image.
    offset_c : int
        Column offset of the bounding box in the full image.

    Returns
    -------
    dict
        Measurement dictionary with keys: area, centroid_row,
        centroid_col, bbox_row, bbox_col, bbox_height, bbox_width,
        perimeter, circularity, aspect_ratio, solidity.
    """
    area = float(region_mask.sum())
    if area == 0:
        return {}

    # Centroid (in full-image coordinates)
    coords = np.argwhere(region_mask)
    centroid_r = coords[:, 0].mean() + offset_r
    centroid_c = coords[:, 1].mean() + offset_c

    # Bounding box
    bbox_h, bbox_w = region_mask.shape

    # Perimeter: count boundary pixels (foreground pixels with at
    # least one 4-connected background neighbor)
    padded = np.pad(region_mask, 1, mode='constant', constant_values=False)
    interior = (
        padded[1:-1, 1:-1] &
        padded[:-2, 1:-1] &  # north
        padded[2:, 1:-1] &   # south
        padded[1:-1, :-2] &  # west
        padded[1:-1, 2:]     # east
    )
    boundary = region_mask & ~interior
    perimeter = float(boundary.sum())

    # Circularity: 4*pi*area / perimeter^2
    if perimeter > 0:
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        circularity = min(circularity, 1.0)
    else:
        circularity = 1.0

    # Aspect ratio of bounding box
    aspect_ratio = float(bbox_w) / float(bbox_h) if bbox_h > 0 else 1.0

    # Solidity: area / convex hull area (approximated by bounding box)
    bbox_area = float(bbox_h * bbox_w)
    solidity = area / bbox_area if bbox_area > 0 else 1.0

    return {
        'area': area,
        'centroid_row': centroid_r,
        'centroid_col': centroid_c,
        'bbox_row': float(offset_r),
        'bbox_col': float(offset_c),
        'bbox_height': float(bbox_h),
        'bbox_width': float(bbox_w),
        'perimeter': perimeter,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
    }


@processor_tags(modalities=['SAR', 'PAN', 'EO', 'MSI', 'HSI', 'thermal'],
                category='analyze')
@processor_version('1.54j')
class AnalyzeParticles(ImageTransform):
    """Connected component analysis with measurements, ported from ImageJ 1.54j.

    Labels connected components in a binary image and measures
    geometric properties of each particle. Particles can be filtered
    by area and circularity ranges.

    Parameters
    ----------
    min_area : float
        Minimum particle area in pixels. Particles smaller than this
        are excluded. Default is 0 (no minimum).
    max_area : float
        Maximum particle area in pixels. Particles larger than this
        are excluded. Default is ``inf`` (no maximum).
    min_circularity : float
        Minimum circularity (0 to 1). Default is 0.0.
    max_circularity : float
        Maximum circularity (0 to 1). Default is 1.0.
    connectivity : int
        Connectivity for labeling: 4 (face-connected) or 8
        (face+corner). Default is 8, matching ImageJ.
    output_mode : str
        Output format:

        - ``'labels'`` (default): Labeled image where each accepted
          particle has a unique integer value.
        - ``'mask'``: Binary mask of accepted particles.
        - ``'outlines'``: Binary outlines (perimeters) of accepted
          particles.

    Attributes
    ----------
    results_ : list of dict
        List of measurement dictionaries (one per accepted particle).
        Set after ``apply()`` is called. Each dict has keys:
        ``area``, ``centroid_row``, ``centroid_col``, ``bbox_row``,
        ``bbox_col``, ``bbox_height``, ``bbox_width``, ``perimeter``,
        ``circularity``, ``aspect_ratio``, ``solidity``.
    n_particles_ : int
        Number of accepted particles. Set after ``apply()``.

    Notes
    -----
    Port of ``ij/plugin/filter/ParticleAnalyzer.java`` from ImageJ
    1.54j (public domain). Original by Wayne Rasband.

    The input should be a binary image (0 = background, nonzero =
    foreground).

    Examples
    --------
    Count and measure all particles:

    >>> from grdl.imagej import AnalyzeParticles
    >>> ap = AnalyzeParticles(min_area=10)
    >>> labeled = ap.apply(binary_mask)
    >>> print(f"Found {ap.n_particles_} particles")
    >>> for p in ap.results_:
    ...     print(f"  Area={p['area']:.0f}, Circ={p['circularity']:.2f}")

    Filter by size and circularity:

    >>> ap = AnalyzeParticles(min_area=50, max_area=5000,
    ...                       min_circularity=0.5)
    >>> mask = ap.apply(detections, output_mode='mask')
    """

    __imagej_source__ = 'ij/plugin/filter/ParticleAnalyzer.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = False

    OUTPUT_MODES = ('labels', 'mask', 'outlines')

    def __init__(
        self,
        min_area: float = 0,
        max_area: float = np.inf,
        min_circularity: float = 0.0,
        max_circularity: float = 1.0,
        connectivity: int = 8,
        output_mode: str = 'labels',
    ) -> None:
        if min_area < 0:
            raise ValueError(f"min_area must be >= 0, got {min_area}")
        if max_area < min_area:
            raise ValueError(
                f"max_area ({max_area}) must be >= min_area ({min_area})"
            )
        if not (0.0 <= min_circularity <= 1.0):
            raise ValueError(
                f"min_circularity must be in [0, 1], got {min_circularity}"
            )
        if not (0.0 <= max_circularity <= 1.0):
            raise ValueError(
                f"max_circularity must be in [0, 1], got {max_circularity}"
            )
        if connectivity not in (4, 8):
            raise ValueError(
                f"connectivity must be 4 or 8, got {connectivity}"
            )
        out_lower = output_mode.lower()
        if out_lower not in self.OUTPUT_MODES:
            raise ValueError(
                f"Unknown output_mode '{output_mode}'. "
                f"Must be one of {self.OUTPUT_MODES}"
            )

        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.max_circularity = max_circularity
        self.connectivity = connectivity
        self.output_mode = out_lower
        self.results_: List[Dict[str, float]] = []
        self.n_particles_: int = 0

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Analyze particles in a binary image.

        Parameters
        ----------
        source : np.ndarray
            2D binary image. Shape ``(rows, cols)``. Non-zero pixels
            are foreground.

        Returns
        -------
        np.ndarray
            Result image (dtype float64), format depends on
            ``output_mode``.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        binary = source.astype(np.float64) > 0

        if not binary.any():
            self.results_ = []
            self.n_particles_ = 0
            return np.zeros_like(source, dtype=np.float64)

        # Label connected components
        struct = np.ones((3, 3)) if self.connectivity == 8 else None
        labeled, n_labels = label(binary, structure=struct)
        slices = find_objects(labeled)

        # Measure and filter particles
        self.results_ = []
        accepted_labels = set()
        new_label = 0

        output = np.zeros_like(source, dtype=np.float64)

        for i, sl in enumerate(slices):
            if sl is None:
                continue
            lbl = i + 1
            region_mask = labeled[sl] == lbl

            offset_r = sl[0].start
            offset_c = sl[1].start

            measurements = _measure_particle(region_mask, offset_r, offset_c)
            if not measurements:
                continue

            area = measurements['area']
            circ = measurements['circularity']

            # Filter by area and circularity
            if area < self.min_area or area > self.max_area:
                continue
            if circ < self.min_circularity or circ > self.max_circularity:
                continue

            new_label += 1
            accepted_labels.add(lbl)
            measurements['label'] = float(new_label)
            self.results_.append(measurements)

            if self.output_mode == 'labels':
                output[sl][region_mask] = float(new_label)
            elif self.output_mode == 'mask':
                output[sl][region_mask] = 1.0
            elif self.output_mode == 'outlines':
                # Compute boundary of this particle
                padded = np.pad(region_mask, 1, mode='constant',
                                constant_values=False)
                interior = (
                    padded[1:-1, 1:-1] &
                    padded[:-2, 1:-1] &
                    padded[2:, 1:-1] &
                    padded[1:-1, :-2] &
                    padded[1:-1, 2:]
                )
                boundary = region_mask & ~interior
                output[sl][boundary] = 1.0

        self.n_particles_ = new_label
        return output
