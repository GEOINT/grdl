# -*- coding: utf-8 -*-
"""
Statistical Region Merging - Port of Fiji's SRM plugin.

Implements the Statistical Region Merging (SRM) algorithm for
image segmentation. SRM iteratively merges adjacent pixel regions
based on a statistical test that determines whether the mean values
of two adjacent regions are close enough to belong to the same
segment. The ``Q`` parameter controls the coarseness: smaller Q
produces fewer (larger) regions, larger Q produces more (smaller)
regions with finer detail.

Particularly useful for:
- Land cover segmentation from MSI/HSI satellite imagery
- Object-based image analysis (OBIA) of high-resolution PAN/EO
- SAR image segmentation for target region extraction
- Thermal imagery zoning (temperature regions)
- Superpixel generation for machine learning feature extraction
- Forest/vegetation boundary delineation from multispectral data

Attribution
-----------
Algorithm: R. Nock and F. Nielsen, "Statistical Region Merging",
IEEE Transactions on Pattern Analysis and Machine Intelligence,
28(8):1452-1458, 2006.

Fiji implementation: Johannes Schindelin (Max Planck Institute of
Molecular Cell Biology and Genetics). Source:
``Statistical_Region_Merging.java`` (Fiji, GPL-2). This is an
independent NumPy reimplementation following the published algorithm.

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
2026-02-06

Modified
--------
2026-02-06
"""

# Standard library
from typing import Any

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version


class _UnionFind:
    """Weighted union-find (disjoint set) with path compression.

    Used internally by SRM to track which pixels belong to which region.
    """

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        """Find root with path compression."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, a: int, b: int) -> bool:
        """Merge sets containing a and b. Returns True if merged."""
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        # Union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def get_size(self, x: int) -> int:
        return self.size[self.find(x)]


def _srm_predicate(
    mean_a: float, mean_b: float,
    size_a: int, size_b: int,
    q: float, g: float,
) -> bool:
    """SRM merge predicate (Nock & Nielsen 2006, Eq. 4).

    Two regions should merge if their mean difference is below a
    threshold that depends on region sizes and the Q parameter.

    Parameters
    ----------
    mean_a, mean_b : float
        Mean intensity of the two regions.
    size_a, size_b : int
        Number of pixels in each region.
    q : float
        Coarseness parameter.
    g : float
        Maximum intensity range of the image.

    Returns
    -------
    bool
        True if regions should be merged.
    """
    # Threshold from Eq. 4: sqrt(b(R)) where b(R) = g^2/(2*Q*|R|) * ln(2/delta)
    # delta = 1/(6*n^2) for n pixels, simplified to use ln(6*n^2)
    # Practical threshold per region:
    threshold_a = g * np.sqrt(2.0 / (q * size_a))
    threshold_b = g * np.sqrt(2.0 / (q * size_b))
    return abs(mean_a - mean_b) <= (threshold_a + threshold_b)


@processor_version('1.0')
class StatisticalRegionMerging(ImageTransform):
    """Statistical Region Merging segmentation, ported from Fiji.

    Segments an image into homogeneous regions by iteratively merging
    adjacent regions whose mean values are statistically similar.

    Parameters
    ----------
    Q : float
        Coarseness parameter controlling the number of output regions.

        - Small Q (e.g., 5-25): Few large regions, coarse segmentation.
        - Medium Q (e.g., 50-100): Moderate detail.
        - Large Q (e.g., 200-500): Many small regions, fine detail.

        Fiji default is 25. Must be > 0.

    output : str
        Output format:

        - ``'labels'``: Integer label map where each region has a unique
          ID. Default.
        - ``'mean'``: Each pixel replaced by the mean intensity of its
          region (smoothed/segmented image).

    Notes
    -----
    Independent reimplementation of the Nock & Nielsen (2006) algorithm
    following the Fiji plugin by Johannes Schindelin
    (``Statistical_Region_Merging.java``, GPL-2).

    The algorithm sorts all neighboring pixel pairs by their intensity
    difference, then iterates through pairs in order (smallest
    difference first), merging regions that pass the statistical test.
    This greedy approach produces results consistent with the
    theoretical guarantees of the SRM framework.

    Computational complexity is O(n log n) for n pixels (dominated
    by the edge sort step).

    Examples
    --------
    Segment a multispectral band into land cover regions:

    >>> from grdl.imagej import StatisticalRegionMerging
    >>> srm = StatisticalRegionMerging(Q=50)
    >>> labels = srm.apply(msi_band)
    >>> n_regions = int(labels.max())

    Generate a smoothed/segmented version:

    >>> srm = StatisticalRegionMerging(Q=25, output='mean')
    >>> smoothed = srm.apply(pan_image)
    """

    __imagej_source__ = 'fiji/Statistical_Region_Merging.java'
    __imagej_version__ = '1.0'

    def __init__(
        self,
        Q: float = 25.0,
        output: str = 'labels',
    ) -> None:
        if Q <= 0:
            raise ValueError(f"Q must be > 0, got {Q}")
        if output not in ('labels', 'mean'):
            raise ValueError(
                f"output must be 'labels' or 'mean', got {output!r}"
            )
        self.Q = Q
        self.output = output

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply SRM segmentation to a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            If ``output='labels'``: float64 label map (1-indexed).
            If ``output='mean'``: float64 image with per-region means.
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

        image = source.astype(np.float64)
        rows, cols = image.shape
        n = rows * cols

        # Intensity range
        g = image.max() - image.min()
        if g < 1e-15:
            # Uniform image → single region
            if self.output == 'labels':
                return np.ones_like(image)
            else:
                return image.copy()

        # Build edge list: all 4-connected neighbor pairs
        # Each edge: (pixel_a_idx, pixel_b_idx, abs_diff)
        flat = image.ravel()

        edges_list = []

        # Horizontal edges (each pixel with its right neighbor)
        for r in range(rows):
            for c in range(cols - 1):
                idx_a = r * cols + c
                idx_b = r * cols + c + 1
                diff = abs(flat[idx_a] - flat[idx_b])
                edges_list.append((diff, idx_a, idx_b))

        # Vertical edges (each pixel with its bottom neighbor)
        for r in range(rows - 1):
            for c in range(cols):
                idx_a = r * cols + c
                idx_b = (r + 1) * cols + c
                diff = abs(flat[idx_a] - flat[idx_b])
                edges_list.append((diff, idx_a, idx_b))

        # Sort edges by difference (ascending) -- greedy merging
        edges_list.sort(key=lambda e: e[0])

        # Initialize union-find and per-region statistics
        uf = _UnionFind(n)
        region_sum = flat.copy()  # Sum of intensities per region

        # Merge pass
        q = self.Q
        for diff, idx_a, idx_b in edges_list:
            ra = uf.find(idx_a)
            rb = uf.find(idx_b)
            if ra == rb:
                continue

            size_a = uf.get_size(ra)
            size_b = uf.get_size(rb)
            mean_a = region_sum[ra] / size_a
            mean_b = region_sum[rb] / size_b

            if _srm_predicate(mean_a, mean_b, size_a, size_b, q, g):
                # Merge: update sum, union
                total_sum = region_sum[ra] + region_sum[rb]
                uf.union(ra, rb)
                new_root = uf.find(ra)
                region_sum[new_root] = total_sum

        # Build output
        if self.output == 'labels':
            # Relabel regions to consecutive integers
            label_map = np.zeros(n, dtype=np.float64)
            root_to_label = {}
            next_label = 1
            for i in range(n):
                root = uf.find(i)
                if root not in root_to_label:
                    root_to_label[root] = next_label
                    next_label += 1
                label_map[i] = root_to_label[root]
            return label_map.reshape(rows, cols)

        else:  # 'mean'
            mean_map = np.zeros(n, dtype=np.float64)
            for i in range(n):
                root = uf.find(i)
                mean_map[i] = region_sum[root] / uf.get_size(root)
            return mean_map.reshape(rows, cols)
