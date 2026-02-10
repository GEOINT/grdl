# -*- coding: utf-8 -*-
"""
Skeletonize - Port of ImageJ's Process > Binary > Skeletonize.

Reduces binary foreground regions to 1-pixel-wide skeletal lines while
preserving topology (connectivity). Uses the Zhang-Suen thinning
algorithm, which iteratively removes boundary pixels that are not
needed to maintain connectivity.

Particularly useful for:
- Extracting road networks from classified PAN/EO imagery
- River/stream centerline extraction from water masks (MSI/SAR)
- Runway and taxiway extraction from airport segmentation maps
- Extracting ridge lines from DEM-derived binary masks
- Reducing detected linear features (pipelines, power lines) to paths
- Simplifying building outlines to structural axes

Attribution
-----------
ImageJ implementation: Wayne Rasband (NIH), based on the lookup-table
thinning approach. Source: ``ij/process/BinaryProcessor.java``
(``skeletonize()`` method) in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

Algorithm reference: Zhang & Suen, "A Fast Parallel Algorithm for
Thinning Digital Patterns", CACM 27(3), 1984, pp. 236-239.

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
from typing import Any

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags


def _neighbors_count(img: np.ndarray, r: int, c: int) -> int:
    """Count the number of non-zero 8-connected neighbors."""
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if img[r + dr, c + dc]:
                count += 1
    return count


def _transitions(img: np.ndarray, r: int, c: int) -> int:
    """Count 0-to-1 transitions in the ordered neighbor sequence P2..P9."""
    # Neighbors in clockwise order: P2,P3,P4,P5,P6,P7,P8,P9
    p = [
        img[r - 1, c],      # P2 (north)
        img[r - 1, c + 1],  # P3 (northeast)
        img[r, c + 1],      # P4 (east)
        img[r + 1, c + 1],  # P5 (southeast)
        img[r + 1, c],      # P6 (south)
        img[r + 1, c - 1],  # P7 (southwest)
        img[r, c - 1],      # P8 (west)
        img[r - 1, c - 1],  # P9 (northwest)
    ]
    count = 0
    for i in range(8):
        if p[i] == 0 and p[(i + 1) % 8] == 1:
            count += 1
    return count


def _zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Apply the Zhang-Suen thinning algorithm.

    Parameters
    ----------
    binary : np.ndarray
        2D binary image (0 and 1), padded with a 1-pixel border of 0s.

    Returns
    -------
    np.ndarray
        Thinned (skeletonized) image, same shape as input.
    """
    img = binary.copy()
    rows, cols = img.shape

    while True:
        # Sub-iteration 1
        to_remove = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if img[r, c] == 0:
                    continue
                p2 = img[r - 1, c]
                p4 = img[r, c + 1]
                p6 = img[r + 1, c]
                p8 = img[r, c - 1]

                n = _neighbors_count(img, r, c)
                t = _transitions(img, r, c)

                if (2 <= n <= 6 and t == 1 and
                        p2 * p4 * p6 == 0 and
                        p4 * p6 * p8 == 0):
                    to_remove.append((r, c))

        for r, c in to_remove:
            img[r, c] = 0

        # Sub-iteration 2
        to_remove2 = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if img[r, c] == 0:
                    continue
                p2 = img[r - 1, c]
                p4 = img[r, c + 1]
                p6 = img[r + 1, c]
                p8 = img[r, c - 1]

                n = _neighbors_count(img, r, c)
                t = _transitions(img, r, c)

                if (2 <= n <= 6 and t == 1 and
                        p2 * p4 * p8 == 0 and
                        p2 * p6 * p8 == 0):
                    to_remove2.append((r, c))

        for r, c in to_remove2:
            img[r, c] = 0

        if len(to_remove) == 0 and len(to_remove2) == 0:
            break

    return img


@processor_tags(modalities=['SAR', 'PAN', 'EO', 'MSI', 'HSI', 'thermal'],
                category='binary')
@processor_version('1.54j')
class Skeletonize(ImageTransform):
    """Binary skeletonization (thinning), ported from ImageJ 1.54j.

    Reduces binary foreground objects to 1-pixel-wide skeletal lines
    while preserving topology. Uses the Zhang-Suen parallel thinning
    algorithm, which iteratively removes boundary pixels that satisfy
    specific connectivity-preserving conditions.

    Notes
    -----
    Port of the ``skeletonize()`` method from
    ``ij/process/BinaryProcessor.java`` in ImageJ 1.54j (public domain).
    Original by Wayne Rasband.

    The input should be a binary image (0 = background, nonzero =
    foreground). Non-binary images are thresholded at > 0.

    The algorithm preserves 8-connected topology: connected objects
    remain connected, and holes remain holes. Isolated single pixels
    are preserved. The result is a topological skeleton (medial axis).

    For very large images, this implementation may be slow due to the
    iterative pixel-scanning nature of Zhang-Suen. Consider
    downsampling large images before skeletonization.

    Examples
    --------
    Extract road centerlines from a binary road mask:

    >>> from grdl.imagej import Skeletonize
    >>> skel = Skeletonize()
    >>> centerlines = skel.apply(road_mask)

    Extract river network skeleton:

    >>> skel = Skeletonize()
    >>> river_skeleton = skel.apply(water_mask)
    """

    __imagej_source__ = 'ij/process/BinaryProcessor.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = False

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply skeletonization to a binary image.

        Parameters
        ----------
        source : np.ndarray
            2D binary image. Shape ``(rows, cols)``. Non-zero pixels
            are foreground.

        Returns
        -------
        np.ndarray
            Skeletonized image, dtype float64, values 0.0 or 1.0.
            Same shape as input.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        binary = (source.astype(np.float64) > 0).astype(np.uint8)

        if not binary.any():
            return np.zeros_like(source, dtype=np.float64)

        # Pad with a 1-pixel border of zeros
        padded = np.pad(binary, 1, mode='constant', constant_values=0)
        skeleton = _zhang_suen(padded)

        # Remove padding
        result = skeleton[1:-1, 1:-1]
        return result.astype(np.float64)
