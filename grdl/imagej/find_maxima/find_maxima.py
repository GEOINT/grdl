# -*- coding: utf-8 -*-
"""
Find Maxima - Port of ImageJ's MaximumFinder plugin.

Detects local intensity maxima (peaks) in an image using
prominence-based filtering. A maximum is considered significant
only if it stands out from its surroundings by at least a specified
prominence (noise tolerance). This prevents noise-induced false peaks.

Particularly useful for:
- Target/bright-point detection in SAR amplitude imagery
- Star/point source detection in nighttime PAN imagery
- Thermal hotspot detection in IR imagery
- Peak detection in spectral response curves (HSI)
- Ship detection in SAR ocean imagery
- Building/structure detection in high-resolution PAN/EO imagery

Attribution
-----------
ImageJ implementation: Michael Schmid (Vienna University of Technology).
Source: ``ij/plugin/filter/MaximumFinder.java`` in ImageJ 1.54j.
ImageJ 1.x source is in the public domain.

The prominence-based approach is based on topographic prominence
concepts from terrain analysis.

Dependencies
------------
scipy

Author
------
Steven Siebert

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
from typing import Any, Tuple

# Third-party
import numpy as np
from scipy.ndimage import maximum_filter, label

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.versioning import processor_version, processor_tags


@processor_tags(modalities=['SAR', 'PAN', 'EO', 'thermal'], category='find_maxima')
@processor_version('1.54j')
class FindMaxima(ImageTransform):
    """Prominence-based local maximum detection, ported from ImageJ 1.54j.

    Finds local maxima that exceed their surroundings by at least
    ``prominence`` intensity units. Returns either a binary point map
    or a count map (for multi-point maxima at plateaus).

    Parameters
    ----------
    prominence : float
        Minimum intensity difference between a maximum and its
        surrounding saddle point. Higher values detect only the most
        prominent peaks; lower values detect more (including noisy)
        peaks. ImageJ calls this "Noise tolerance". Default 10.0.
    output : str
        Output format:

        - ``'point_map'``: Binary image with 1.0 at detected maxima,
          0.0 elsewhere. Default.
        - ``'count_map'``: Each detected maximum gets a unique integer
          label (connected-component labeling of maxima regions).

    exclude_on_edges : bool
        If True, exclude maxima that touch the image border.
        Default False.

    Notes
    -----
    Port of ``ij/plugin/filter/MaximumFinder.java`` from ImageJ 1.54j
    (public domain). Original implementation by Michael Schmid.

    This is a simplified implementation that uses local maximum
    filtering followed by prominence thresholding. The full ImageJ
    ``MaximumFinder`` uses a more sophisticated flooding algorithm;
    this port captures the core behavior for most remote sensing
    use cases.

    Examples
    --------
    Detect bright targets in SAR imagery:

    >>> from grdl.imagej import FindMaxima
    >>> fm = FindMaxima(prominence=20.0)
    >>> peaks = fm.apply(sar_amplitude)
    >>> peak_locations = np.argwhere(peaks > 0)

    Count distinct peaks:

    >>> fm = FindMaxima(prominence=15.0, output='count_map')
    >>> labels = fm.apply(pan_image)
    >>> n_peaks = int(labels.max())
    """

    __imagej_source__ = 'ij/plugin/filter/MaximumFinder.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = False

    def __init__(
        self,
        prominence: float = 10.0,
        output: str = 'point_map',
        exclude_on_edges: bool = False,
    ) -> None:
        if prominence < 0:
            raise ValueError(f"prominence must be >= 0, got {prominence}")
        if output not in ('point_map', 'count_map'):
            raise ValueError(
                f"output must be 'point_map' or 'count_map', got {output!r}"
            )

        self.prominence = prominence
        self.output = output
        self.exclude_on_edges = exclude_on_edges

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Detect local maxima in a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            If ``output='point_map'``: binary float64 image with 1.0
            at maxima, 0.0 elsewhere.
            If ``output='count_map'``: integer-labeled image where
            each connected maximum region has a unique positive integer.

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

        # Step 1: Find local maxima (3x3 neighborhood)
        local_max = maximum_filter(image, size=3, mode='nearest')
        is_max = (image == local_max)

        # Step 2: Prominence filtering
        # For each candidate local maximum, estimate the "saddle level"
        # (highest surrounding value reachable without crossing the peak)
        # and require: peak_value - saddle_level >= prominence.
        #
        # Approach: suppress all local maxima from the image, then use
        # maximum_filter on the suppressed image to find the highest
        # non-peak value in each neighborhood. This gives the saddle
        # level without the peak contaminating its own background.
        if self.prominence > 0:
            from scipy.ndimage import minimum_filter as min_filt

            # Create a suppressed image: replace local max pixels with
            # the minimum of their immediate non-max neighbors.
            # This removes the peak so background estimation is clean.
            suppressed = image.copy()
            # For max pixels, replace with the local minimum in 3x3
            local_min = min_filt(image, size=3, mode='nearest')
            suppressed[is_max] = local_min[is_max]

            # Now find the highest "saddle" value around each peak
            # using progressively larger neighborhoods
            bg_size = max(5, int(self.prominence) * 2 + 1)
            bg_size = min(bg_size, min(rows, cols))
            if bg_size % 2 == 0:
                bg_size += 1

            saddle = maximum_filter(
                suppressed, size=bg_size, mode='nearest'
            )

            is_prominent = (image - saddle) >= self.prominence
            is_max = is_max & is_prominent

        # Step 3: Exclude edge maxima
        if self.exclude_on_edges:
            is_max[0, :] = False
            is_max[-1, :] = False
            is_max[:, 0] = False
            is_max[:, -1] = False

        # Step 4: Output
        if self.output == 'count_map':
            # Label connected maxima regions
            labeled, n_features = label(is_max)
            return labeled.astype(np.float64)
        else:
            return is_max.astype(np.float64)

    def find_peaks(self, source: np.ndarray) -> np.ndarray:
        """Convenience method returning peak coordinates.

        Parameters
        ----------
        source : np.ndarray
            2D image array.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, 2)`` with ``[row, col]`` coordinates
            of detected maxima. Empty ``(0, 2)`` array if no peaks found.
        """
        point_map = self.apply(source)
        coords = np.argwhere(point_map > 0)
        if coords.size == 0:
            return np.empty((0, 2), dtype=int)
        return coords
