# -*- coding: utf-8 -*-
"""
CFAR Detector Base - Template-method base class for CFAR detection.

Provides ``CFARDetector``, an abstract ``ImageDetector`` subclass that
implements the full CFAR detection pipeline.  Concrete subclasses
override only ``_estimate_background()`` to supply their specific
background estimation strategy.

The pipeline executed by ``detect()`` is:

1. Resolve tunable parameters (``_resolve_params``).
2. Estimate local background via ``_estimate_background()`` (abstract hook).
3. Compute adaptive threshold from background statistics and PFA.
4. Binary detection mask → connected-component labeling.
5. Filter clusters by minimum area and build ``DetectionSet``.

Also provides two vectorized background estimation helpers used by
the concrete subclasses:

- ``_annular_stats``: mean and std of an annular training ring via
  uniform_filter subtraction (O(N), separable).
- ``_quadrant_means``: mean of four quadrant regions of the training
  annulus via a summed area table (O(N)).

Dependencies
------------
scipy
shapely

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
2026-02-16

Modified
--------
2026-02-16
"""

# Standard library
from abc import abstractmethod
from typing import Annotated, Any, List, Optional, Tuple

# Third-party
import numpy as np
from scipy.ndimage import find_objects, label, uniform_filter
from scipy.special import erfinv
from shapely.geometry import Point, box

# GRDL internal
from grdl.exceptions import ValidationError
from grdl.image_processing.detection.base import ImageDetector
from grdl.image_processing.detection.models import Detection, DetectionSet
from grdl.image_processing.detection.fields import Fields
from grdl.image_processing.params import Desc, Options, Range
from grdl.image_processing.detection.cfar._validation import (
    validate_assumption,
    validate_cfar_window,
    validate_pfa,
)


# ===================================================================
# Vectorized background estimation helpers
# ===================================================================

def _annular_stats(
    image: np.ndarray,
    guard_cells: int,
    training_cells: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of an annular training ring.

    Uses the identity ``annulus_sum = outer_sum - guard_sum``, computed
    via two calls to ``uniform_filter`` (separable, O(N) each).

    Parameters
    ----------
    image : np.ndarray
        2D float64 image.
    guard_cells : int
        Guard band half-width.
    training_cells : int
        Training annulus half-width.

    Returns
    -------
    bg_mean : np.ndarray
        Local background mean, same shape as *image*.
    bg_std : np.ndarray
        Local background standard deviation, same shape as *image*.
    """
    outer_size = 2 * training_cells + 1
    guard_size = 2 * guard_cells + 1
    outer_count = outer_size * outer_size
    guard_count = guard_size * guard_size
    train_count = outer_count - guard_count

    outer_mean = uniform_filter(image, size=outer_size, mode='reflect')
    guard_mean = uniform_filter(image, size=guard_size, mode='reflect')
    bg_mean = (outer_mean * outer_count - guard_mean * guard_count) / train_count

    outer_sq = uniform_filter(image * image, size=outer_size, mode='reflect')
    guard_sq = uniform_filter(image * image, size=guard_size, mode='reflect')
    bg_sq_mean = (outer_sq * outer_count - guard_sq * guard_count) / train_count

    bg_var = bg_sq_mean - bg_mean * bg_mean
    np.maximum(bg_var, 0.0, out=bg_var)
    bg_std = np.sqrt(bg_var)

    return bg_mean, bg_std


def _quadrant_means(
    image: np.ndarray,
    guard_cells: int,
    training_cells: int,
) -> List[np.ndarray]:
    """Compute mean of four quadrant regions of the training annulus.

    Each quadrant is the rectangular region between the guard boundary
    and the training boundary in one corner direction.  Uses a summed
    area table (SAT) with reflected padding for O(1) per-pixel lookups.

    Quadrant layout around pixel ``(r, c)``::

        +---+-------+---+
        | Q0|       | Q1|   Q0 = upper-left    Q1 = upper-right
        +---+       +---+
        |   | guard |   |
        +---+       +---+
        | Q2|       | Q3|   Q2 = lower-left    Q3 = lower-right
        +---+-------+---+

    Parameters
    ----------
    image : np.ndarray
        2D float64 image.
    guard_cells : int
        Guard band half-width.
    training_cells : int
        Training annulus half-width.

    Returns
    -------
    list of np.ndarray
        Four arrays (Q0, Q1, Q2, Q3), each same shape as *image*.
    """
    G = guard_cells
    T = training_cells
    rows, cols = image.shape

    # Pad and build summed area table
    padded = np.pad(image, T, mode='reflect').astype(np.float64)
    # SAT with a leading row/col of zeros for the standard rectangle formula
    sat = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    sat = np.pad(sat, ((1, 0), (1, 0)), mode='constant', constant_values=0)

    # In padded coordinates, pixel (r, c) in the original image is at
    # (r + T, c + T).  The SAT has an extra leading row/col of zeros,
    # so SAT pixel (r + T + 1, c + T + 1) corresponds to the cumulative
    # sum up to and including padded[r + T, c + T].

    def _rect_mean(r1: np.ndarray, c1: np.ndarray,
                   r2: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """Rectangle mean from SAT.  Coordinates are in padded space."""
        # SAT formula: sum = SAT[r2+1, c2+1] - SAT[r1, c2+1] - SAT[r2+1, c1] + SAT[r1, c1]
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        s = (sat[r2 + 1, c2 + 1] - sat[r1, c2 + 1]
             - sat[r2 + 1, c1] + sat[r1, c1])
        return s / area

    # Row/col index grids in padded coordinates
    rr = np.arange(rows) + T  # padded row indices for original pixels
    cc = np.arange(cols) + T
    # Broadcast to 2D: rr[:, None] and cc[None, :]

    rr2d = rr[:, np.newaxis].astype(np.intp)
    cc2d = cc[np.newaxis, :].astype(np.intp)

    # Quadrant boundaries in padded coordinates:
    # Q0 (upper-left): rows [r-T, r-G-1], cols [c-T, c-G-1]
    q0 = _rect_mean(rr2d - T, cc2d - T, rr2d - G - 1, cc2d - G - 1)
    # Q1 (upper-right): rows [r-T, r-G-1], cols [c+G+1, c+T]
    q1 = _rect_mean(rr2d - T, cc2d + G + 1, rr2d - G - 1, cc2d + T)
    # Q2 (lower-left): rows [r+G+1, r+T], cols [c-T, c-G-1]
    q2 = _rect_mean(rr2d + G + 1, cc2d - T, rr2d + T, cc2d - G - 1)
    # Q3 (lower-right): rows [r+G+1, r+T], cols [c+G+1, c+T]
    q3 = _rect_mean(rr2d + G + 1, cc2d + G + 1, rr2d + T, cc2d + T)

    return [q0, q1, q2, q3]


# ===================================================================
# CFARDetector base class
# ===================================================================

class CFARDetector(ImageDetector):
    """Abstract base class for CFAR (Constant False Alarm Rate) detectors.

    Implements the full CFAR detection pipeline as a template method.
    Concrete subclasses override ``_estimate_background()`` to provide
    their specific background estimation strategy (cell-averaging,
    greatest-of, smallest-of, ordered-statistics, etc.).

    The detector operates on any 2D array the user passes — dB-scale
    or linear-scale — and uses the ``assumption`` parameter to select
    the appropriate threshold computation.

    Parameters
    ----------
    guard_cells : int
        Half-width of the guard band around the cell under test.
    training_cells : int
        Half-width of the training annulus outside the guard band.
    pfa : float
        Probability of false alarm.  Lower values yield fewer false
        detections but may miss weaker targets.
    min_pixels : int
        Minimum connected-component area in pixels.  Clusters smaller
        than this are rejected as noise.
    assumption : str
        Statistical model for threshold computation:
        ``'gaussian'`` for dB-domain input (threshold = mean + alpha * std),
        ``'exponential'`` for linear-domain input (threshold = alpha * mean).
    """

    guard_cells: Annotated[int, Range(min=1, max=20),
                           Desc('Guard band half-width (pixels)')] = 3
    training_cells: Annotated[int, Range(min=2, max=50),
                              Desc('Training annulus half-width (pixels)')] = 12
    pfa: Annotated[float, Range(min=1e-12, max=0.5),
                   Desc('Probability of false alarm')] = 1e-3
    min_pixels: Annotated[int, Range(min=1, max=1000),
                          Desc('Minimum cluster area (pixels)')] = 9
    assumption: Annotated[str, Options('gaussian', 'exponential'),
                          Desc('Statistical model for threshold')] = 'gaussian'

    def __init__(
        self,
        guard_cells: int = 3,
        training_cells: int = 12,
        pfa: float = 1e-3,
        min_pixels: int = 9,
        assumption: str = 'gaussian',
    ) -> None:
        validate_cfar_window(guard_cells, training_cells)
        validate_pfa(pfa)
        validate_assumption(assumption)
        self.guard_cells = guard_cells
        self.training_cells = training_cells
        self.pfa = pfa
        self.min_pixels = min_pixels
        self.assumption = assumption

    @property
    def output_fields(self) -> Tuple[str, ...]:
        """Declare output fields using the GRDL data dictionary."""
        return (
            Fields.sar.SIGMA0,
            Fields.identity.IS_TARGET,
            Fields.physical.AREA,
            Fields.physical.LENGTH,
            Fields.physical.WIDTH,
        )

    # ----- Abstract hook for subclasses ---------------------------------

    @abstractmethod
    def _estimate_background(
        self,
        image: np.ndarray,
        guard_cells: int,
        training_cells: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate local background statistics.

        Parameters
        ----------
        image : np.ndarray
            2D float64 image.
        guard_cells : int
            Guard band half-width.
        training_cells : int
            Training annulus half-width.

        Returns
        -------
        bg_mean : np.ndarray
            Background level estimate, same shape as *image*.
        bg_std : np.ndarray
            Background standard deviation estimate, same shape as
            *image*.  May be ignored under the ``'exponential'``
            assumption.
        """
        ...

    # ----- Shared pipeline methods --------------------------------------

    @staticmethod
    def _compute_threshold(
        bg_mean: np.ndarray,
        bg_std: np.ndarray,
        pfa: float,
        training_cells: int,
        guard_cells: int,
        assumption: str,
    ) -> np.ndarray:
        """Compute adaptive detection threshold from background statistics.

        Parameters
        ----------
        bg_mean : np.ndarray
            Background level estimate.
        bg_std : np.ndarray
            Background standard deviation estimate.
        pfa : float
            Probability of false alarm.
        training_cells : int
            Training annulus half-width.
        guard_cells : int
            Guard band half-width.
        assumption : str
            ``'gaussian'`` or ``'exponential'``.

        Returns
        -------
        np.ndarray
            Pixel-wise detection threshold, same shape as bg_mean.
        """
        if assumption == 'gaussian':
            alpha = np.sqrt(2.0) * float(erfinv(1.0 - 2.0 * pfa))
            return bg_mean + alpha * bg_std
        else:
            # Exponential model: alpha = N * (pfa^(-1/N) - 1)
            outer_size = 2 * training_cells + 1
            guard_size = 2 * guard_cells + 1
            N = outer_size * outer_size - guard_size * guard_size
            exponent = -1.0 / N
            pfa_term = pfa ** exponent
            if pfa_term > 1e15:
                raise ValidationError(
                    f"Threshold overflow: pfa={pfa} with N={N} training "
                    f"cells yields pfa^(-1/N)={pfa_term:.2e}. Increase "
                    f"training_cells or raise pfa."
                )
            alpha = N * (pfa_term - 1.0)
            return alpha * bg_mean

    @staticmethod
    def _build_detections(
        image: np.ndarray,
        threshold: np.ndarray,
        labeled: np.ndarray,
        n_components: int,
        min_pixels: int,
        assumption: str,
        geolocation: Optional[Any],
    ) -> List[Detection]:
        """Build Detection objects from labeled connected components.

        Fully vectorized: uses ``np.bincount`` for pixel counts and
        intensity sums, ``find_objects`` for bounding boxes.  The
        Python loop iterates only over surviving components.

        Parameters
        ----------
        image : np.ndarray
            2D input image (same domain as threshold).
        threshold : np.ndarray
            Pixel-wise detection threshold.
        labeled : np.ndarray
            Connected-component label array from ``scipy.ndimage.label``.
        n_components : int
            Number of connected components.
        min_pixels : int
            Minimum cluster area in pixels.
        assumption : str
            Statistical model, used for confidence normalization.
        geolocation : optional
            Geolocation object for pixel-to-geographic transforms.

        Returns
        -------
        List[Detection]
        """
        if n_components == 0:
            return []

        flat_labels = labeled.ravel()
        pixel_counts = np.bincount(flat_labels, minlength=n_components + 1)
        intensity_sums = np.bincount(
            flat_labels, weights=image.ravel(), minlength=n_components + 1,
        )
        slices = find_objects(labeled)

        detections: List[Detection] = []
        for comp_id in range(1, n_components + 1):
            n_px = int(pixel_counts[comp_id])
            if n_px < min_pixels:
                continue

            row_slice, col_slice = slices[comp_id - 1]
            row_min = row_slice.start
            row_max = row_slice.stop - 1
            col_min = col_slice.start
            col_max = col_slice.stop - 1

            mean_val = float(intensity_sums[comp_id] / n_px)
            height_px = row_max - row_min + 1
            width_px = col_max - col_min + 1

            pixel_geom = box(
                float(col_min), float(row_min),
                float(col_max + 1), float(row_max + 1),
            )

            geo_geom = None
            if geolocation is not None:
                center_row = (row_min + row_max) / 2.0
                center_col = (col_min + col_max) / 2.0
                lat, lon, _ = geolocation.image_to_latlon(
                    center_row, center_col,
                )
                geo_geom = Point(float(lon), float(lat))

            properties = {
                Fields.sar.SIGMA0: round(mean_val, 2),
                Fields.identity.IS_TARGET: True,
                Fields.physical.AREA: float(n_px),
                Fields.physical.LENGTH: float(height_px),
                Fields.physical.WIDTH: float(width_px),
            }

            # Confidence: excess above threshold, normalized
            center_r = (row_min + row_max) // 2
            center_c = (col_min + col_max) // 2
            excess = image[center_r, center_c] - threshold[center_r, center_c]
            if assumption == 'gaussian':
                confidence = float(np.clip(excess / 20.0, 0.0, 1.0))
            else:
                bg_val = threshold[center_r, center_c]
                confidence = float(np.clip(
                    excess / (bg_val + np.finfo(np.float64).tiny), 0.0, 1.0,
                ))

            detections.append(Detection(
                pixel_geometry=pixel_geom,
                properties=properties,
                confidence=confidence,
                geo_geometry=geo_geom,
            ))

        return detections

    # ----- Template method: detect() ------------------------------------

    def detect(
        self,
        source: np.ndarray,
        geolocation: Optional[Any] = None,
        **kwargs: Any,
    ) -> DetectionSet:
        """Run CFAR detection on a 2D image.

        Parameters
        ----------
        source : np.ndarray
            2D image array, shape ``(rows, cols)``.  Can be dB-scale
            (use ``assumption='gaussian'``) or linear-scale (use
            ``assumption='exponential'``).
        geolocation : Geolocation, optional
            For pixel-to-geographic coordinate transforms on detections.

        Returns
        -------
        DetectionSet
            Bounding-box detections for bright-target candidates.
        """
        params = self._resolve_params(kwargs)
        guard = params['guard_cells']
        train = params['training_cells']
        pfa = params['pfa']
        min_px = params['min_pixels']
        assumption = params['assumption']

        validate_cfar_window(guard, train)
        validate_pfa(pfa)
        validate_assumption(assumption)

        if source.ndim != 2:
            raise ValidationError(
                f"CFAR detector requires 2D input, got shape {source.shape}"
            )

        img = source.astype(np.float64)

        # Step 1: Background estimation (abstract hook)
        bg_mean, bg_std = self._estimate_background(img, guard, train)

        # Step 2: Adaptive threshold
        threshold = self._compute_threshold(
            bg_mean, bg_std, pfa, train, guard, assumption,
        )

        # Step 3: Binary mask and connected components
        detection_mask = img > threshold
        labeled_arr, n_components = label(detection_mask)

        # Step 4: Build detections
        detections = self._build_detections(
            img, threshold, labeled_arr, n_components,
            min_px, assumption, geolocation,
        )

        return DetectionSet(
            detections=detections,
            detector_name=self.__class__.__name__,
            detector_version=self.__processor_version__,
            output_fields=self.output_fields,
            metadata={
                'guard_cells': guard,
                'training_cells': train,
                'pfa': pfa,
                'min_pixels': min_px,
                'assumption': assumption,
                'image_shape': list(source.shape),
                'n_components_raw': n_components,
            },
        )
