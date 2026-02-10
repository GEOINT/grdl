# -*- coding: utf-8 -*-
"""
Watershed - Port of ImageJ's Process > Binary > Watershed.

Implements the classic watershed segmentation algorithm for splitting
touching binary objects. The algorithm computes an Euclidean Distance
Map (EDM) of the binary foreground, identifies local maxima as seeds,
and grows regions from those seeds to find watershed lines that
separate touching objects.

Particularly useful for:
- Separating touching ships in SAR maritime surveillance imagery
- Splitting merged building footprints in PAN/EO classification maps
- Dividing clustered vegetation patches in NDVI-derived masks
- Separating overlapping thermal hotspots
- Post-processing binary segmentation of closely spaced targets
- Refining connected component analysis for counting applications

Attribution
-----------
ImageJ implementation: Wayne Rasband (NIH), based on the algorithm by
Soille & Vincent. Source: ``ij/plugin/filter/EDM.java`` (watershed
method) and ``ij/process/BinaryProcessor.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

Algorithm reference: Soille & Vincent, "Determining Watersheds in
Digital Pictures via Flooding Simulations", Proc. SPIE 1360, 1990.

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
from typing import Any

# Third-party
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    label,
    maximum_filter,
    binary_dilation,
    generate_binary_structure,
)

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


def _find_seeds(edt: np.ndarray, min_distance: int = 2) -> np.ndarray:
    """Find seed points as local maxima of the distance transform.

    Parameters
    ----------
    edt : np.ndarray
        Euclidean distance transform of the binary image.
    min_distance : int
        Minimum pixel distance between seeds.

    Returns
    -------
    np.ndarray
        Boolean mask of seed locations.
    """
    size = 2 * min_distance + 1
    local_max = maximum_filter(edt, size=size)
    seeds = (edt == local_max) & (edt > 0)
    return seeds


def _watershed_from_markers(edt: np.ndarray, markers: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
    """Grow watershed regions from labeled markers on inverted EDT.

    Uses a priority-queue-free approach: iteratively dilates each
    labeled region into unlabeled foreground pixels, prioritizing
    by descending EDT value (closest to center first).

    Parameters
    ----------
    edt : np.ndarray
        Euclidean distance transform.
    markers : np.ndarray
        Labeled seed image (0 = unlabeled, >0 = region label).
    mask : np.ndarray
        Binary foreground mask.

    Returns
    -------
    np.ndarray
        Labeled image with watershed lines (value 0) between regions.
    """
    result = markers.copy()
    struct = generate_binary_structure(2, 2)  # 8-connected

    # Sort unlabeled foreground pixels by EDT (descending)
    unlabeled = mask & (result == 0)
    coords = np.argwhere(unlabeled)
    if len(coords) == 0:
        return result

    edt_vals = edt[coords[:, 0], coords[:, 1]]
    order = np.argsort(-edt_vals)
    coords = coords[order]

    # Iterative region growing
    for _ in range(max(edt.shape)):
        changed = False
        for r, c in coords:
            if result[r, c] != 0:
                continue
            if not mask[r, c]:
                continue

            # Check 8-connected neighbors
            neighbors = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                        if result[nr, nc] > 0:
                            neighbors.add(result[nr, nc])

            if len(neighbors) == 1:
                result[r, c] = neighbors.pop()
                changed = True
            elif len(neighbors) > 1:
                # Watershed line: leave as 0
                pass

        if not changed:
            break

    return result


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.MSI, IM.HSI, IM.SWIR, IM.MWIR, IM.LWIR],
                category=PC.SEGMENTATION)
@processor_version('1.54j')
class Watershed(ImageTransform):
    """Binary watershed segmentation, ported from ImageJ 1.54j.

    Splits touching objects in a binary image by computing the
    Euclidean Distance Map, finding local maxima as seeds, and
    growing watershed regions. The result is a labeled image where
    each separated object has a unique integer label and watershed
    lines between objects have value 0.

    Parameters
    ----------
    min_seed_distance : int
        Minimum pixel distance between watershed seeds. Larger
        values merge nearby seeds and produce fewer splits.
        Default is 2, matching ImageJ's behavior.
    output_mode : str
        Output format. One of:

        - ``'labels'`` (default): Each object gets a unique integer
          label. Watershed lines are 0.
        - ``'lines'``: Binary image of watershed lines only (1.0
          at watershed lines, 0.0 elsewhere).
        - ``'binary'``: Input binary image with watershed lines
          removed (objects separated by 1-pixel gaps).

    Notes
    -----
    Port of the watershed method from ``ij/plugin/filter/EDM.java``
    in ImageJ 1.54j (public domain). Original by Wayne Rasband.

    The input should be a binary image (0 = background, nonzero =
    foreground). Non-binary images are thresholded at > 0.

    Examples
    --------
    Separate touching objects and count them:

    >>> from grdl.imagej import Watershed
    >>> ws = Watershed(output_mode='labels')
    >>> labeled = ws.apply(binary_mask)
    >>> n_objects = int(labeled.max())

    Get watershed lines to overlay on the original image:

    >>> ws = Watershed(output_mode='lines')
    >>> lines = ws.apply(binary_mask)

    Split touching objects preserving binary format:

    >>> ws = Watershed(output_mode='binary')
    >>> separated = ws.apply(touching_objects)
    """

    __imagej_source__ = 'ij/plugin/filter/EDM.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = False

    OUTPUT_MODES = ('labels', 'lines', 'binary')

    def __init__(
        self,
        min_seed_distance: int = 2,
        output_mode: str = 'labels',
    ) -> None:
        if min_seed_distance < 1:
            raise ValueError(
                f"min_seed_distance must be >= 1, got {min_seed_distance}"
            )
        out_lower = output_mode.lower()
        if out_lower not in self.OUTPUT_MODES:
            raise ValueError(
                f"Unknown output_mode '{output_mode}'. "
                f"Must be one of {self.OUTPUT_MODES}"
            )

        self.min_seed_distance = min_seed_distance
        self.output_mode = out_lower

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply watershed segmentation to a binary image.

        Parameters
        ----------
        source : np.ndarray
            2D binary image. Shape ``(rows, cols)``. Non-zero pixels
            are treated as foreground.

        Returns
        -------
        np.ndarray
            Result depends on ``output_mode``:

            - ``'labels'``: int-valued labeled image (float64).
            - ``'lines'``: binary watershed lines (float64, 0/1).
            - ``'binary'``: separated binary objects (float64, 0/1).

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        mask = source.astype(np.float64) > 0

        # Trivial case: no foreground
        if not mask.any():
            return np.zeros_like(source, dtype=np.float64)

        # Euclidean distance transform
        edt = distance_transform_edt(mask)

        # Find seeds as local maxima of EDT
        seeds = _find_seeds(edt, self.min_seed_distance)

        # Label the seeds
        labeled_seeds, n_seeds = label(seeds)

        if n_seeds <= 1:
            # Nothing to split
            if self.output_mode == 'labels':
                return mask.astype(np.float64)
            elif self.output_mode == 'lines':
                return np.zeros_like(source, dtype=np.float64)
            else:
                return mask.astype(np.float64)

        # Grow watershed regions
        labeled = _watershed_from_markers(edt, labeled_seeds, mask)

        if self.output_mode == 'labels':
            return labeled.astype(np.float64)
        elif self.output_mode == 'lines':
            lines = mask & (labeled == 0)
            return lines.astype(np.float64)
        else:  # binary
            separated = mask & (labeled > 0)
            return separated.astype(np.float64)
