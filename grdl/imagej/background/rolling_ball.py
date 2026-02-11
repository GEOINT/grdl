# -*- coding: utf-8 -*-
"""
Rolling Ball Background Subtraction - Port of ImageJ's BackgroundSubtracter.

Implements Sternberg's rolling-ball algorithm for non-uniform background
estimation and subtraction. The algorithm models the background by rolling
a ball (approximated as a paraboloid) beneath the image surface. Pixels
that the ball cannot reach define the estimated background; subtracting
this background corrects for illumination gradients and sensor non-uniformity.

Particularly useful for:
- Removing illumination gradients in PAN and EO imagery
- Correcting vignetting in optical sensors
- Normalizing thermal imagery with spatial drift
- Pre-processing SAR amplitude imagery for change detection

ImageJ uses an optimized shrink-smooth-expand pipeline: the image is
downsampled by the ball radius, the paraboloid is rolled on the small
image, then the result is expanded back. This port reproduces that
optimization faithfully.

Attribution
-----------
Original algorithm: Stanley R. Sternberg, "Biomedical Image Processing",
IEEE Computer, 1983.

ImageJ implementation: Michael Castle and Janice Keller (Mental Health
Research Institute, University of Michigan), with modifications by
Wayne Rasband (NIH). Source: ``ij/plugin/filter/BackgroundSubtracter.java``
in ImageJ 1.54j. ImageJ 1.x source is in the public domain.

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
from typing import Annotated, Any, Optional, Tuple

# Third-party
import numpy as np
from scipy.ndimage import uniform_filter

# GRDL internal
from grdl.image_processing.base import ImageTransform
from grdl.image_processing.params import Desc, Range
from grdl.image_processing.versioning import processor_version, processor_tags
from grdl.vocabulary import ImageModality as IM, ProcessorCategory as PC


def _build_ball_profile(radius: int) -> np.ndarray:
    """Build a 1D cross-section of the rolling ball (paraboloid).

    The paraboloid approximation ``z = (x^2 + y^2) / (2 * radius)``
    is separable, so we only need a 1D profile. This matches ImageJ's
    ``RollingBall`` inner class.

    Parameters
    ----------
    radius : int
        Ball radius in pixels.

    Returns
    -------
    np.ndarray
        1D float64 array of length ``2 * radius + 1`` containing the
        paraboloid heights.
    """
    width = 2 * radius + 1
    x = np.arange(width, dtype=np.float64) - radius
    profile = x * x / (2.0 * radius)
    return profile


def _shrink_image(image: np.ndarray, shrink_factor: int) -> np.ndarray:
    """Downsample image by taking local minima (or maxima for light bg).

    Reduces the image by ``shrink_factor`` by taking the minimum value
    in each block. This is the first step of the shrink-smooth-expand
    optimization.

    Parameters
    ----------
    image : np.ndarray
        2D input image, float64.
    shrink_factor : int
        Downsampling factor (typically ``max(radius // 10, 1)``).

    Returns
    -------
    np.ndarray
        Downsampled image.
    """
    if shrink_factor <= 1:
        return image.copy()

    rows, cols = image.shape
    s_rows = (rows + shrink_factor - 1) // shrink_factor
    s_cols = (cols + shrink_factor - 1) // shrink_factor
    small = np.full((s_rows, s_cols), np.inf, dtype=np.float64)

    for i in range(s_rows):
        r_start = i * shrink_factor
        r_end = min(r_start + shrink_factor, rows)
        for j in range(s_cols):
            c_start = j * shrink_factor
            c_end = min(c_start + shrink_factor, cols)
            small[i, j] = image[r_start:r_end, c_start:c_end].min()

    return small


def _expand_image(small: np.ndarray, target_shape: Tuple[int, int],
                  shrink_factor: int) -> np.ndarray:
    """Upsample a shrunken image back to original size via bilinear interpolation.

    Parameters
    ----------
    small : np.ndarray
        Downsampled 2D image.
    target_shape : Tuple[int, int]
        (rows, cols) of the original image.
    shrink_factor : int
        The factor used during shrinking.

    Returns
    -------
    np.ndarray
        Upsampled image matching ``target_shape``.
    """
    if shrink_factor <= 1:
        return small.copy()

    rows, cols = target_shape
    s_rows, s_cols = small.shape

    # Build coordinate grids for interpolation
    row_coords = np.arange(rows, dtype=np.float64) / shrink_factor
    col_coords = np.arange(cols, dtype=np.float64) / shrink_factor

    # Clamp to valid range for the small image
    row_coords = np.clip(row_coords, 0, s_rows - 1)
    col_coords = np.clip(col_coords, 0, s_cols - 1)

    r0 = np.floor(row_coords).astype(int)
    c0 = np.floor(col_coords).astype(int)
    r1 = np.minimum(r0 + 1, s_rows - 1)
    c1 = np.minimum(c0 + 1, s_cols - 1)

    dr = row_coords - r0
    dc = col_coords - c0

    # Bilinear interpolation using outer products
    dr = dr[:, np.newaxis]
    dc = dc[np.newaxis, :]

    expanded = (
        small[np.ix_(r0, c0)] * (1 - dr) * (1 - dc) +
        small[np.ix_(r0, c1)] * (1 - dr) * dc +
        small[np.ix_(r1, c0)] * dr * (1 - dc) +
        small[np.ix_(r1, c1)] * dr * dc
    )
    return expanded


def _roll_paraboloid(image: np.ndarray, radius: int) -> np.ndarray:
    """Roll a paraboloid under the image to estimate background.

    For each pixel, the paraboloid is positioned at the maximum height
    where it still fits under the image surface. The background at each
    pixel is the apex of the paraboloid in that position.

    This is implemented as two sequential 1D passes (rows then columns)
    using the separable paraboloid profile, matching ImageJ's approach.

    Parameters
    ----------
    image : np.ndarray
        2D float64 image (possibly downsampled).
    radius : int
        Effective ball radius for this image scale.

    Returns
    -------
    np.ndarray
        Estimated background, same shape as input.
    """
    profile = _build_ball_profile(radius)
    half = radius

    rows, cols = image.shape
    background = image.copy()

    # Pass 1: roll along columns (for each row)
    temp = np.full_like(background, -np.inf)
    for r in range(rows):
        for c in range(cols):
            # Find the maximum paraboloid apex height at column c
            c_start = max(0, c - half)
            c_end = min(cols, c + half + 1)
            p_start = c_start - (c - half)
            p_end = p_start + (c_end - c_start)
            heights = background[r, c_start:c_end] - profile[p_start:p_end]
            temp[r, c] = heights.min()
    background = temp

    # Pass 2: roll along rows (for each column)
    temp = np.full_like(background, -np.inf)
    for c in range(cols):
        for r in range(rows):
            r_start = max(0, r - half)
            r_end = min(rows, r + half + 1)
            p_start = r_start - (r - half)
            p_end = p_start + (r_end - r_start)
            heights = background[r_start:r_end, c] - profile[p_start:p_end]
            temp[r, c] = heights.min()
    background = temp

    return background


def _roll_paraboloid_vectorized(image: np.ndarray, radius: int) -> np.ndarray:
    """Vectorized paraboloid rolling using sliding-window minimum approach.

    Uses the separable paraboloid: two 1D passes. Each pass computes
    ``min(image[x+k] - profile[k])`` over the kernel window, which is
    equivalent to a morphological erosion with a parabolic structuring
    element.

    Parameters
    ----------
    image : np.ndarray
        2D float64 image (possibly downsampled).
    radius : int
        Effective ball radius for this image scale.

    Returns
    -------
    np.ndarray
        Estimated background, same shape as input.
    """
    profile = _build_ball_profile(radius)
    half = radius
    rows, cols = image.shape

    # Pass 1: along columns (axis=1)
    bg = np.full_like(image, np.inf)
    for k in range(len(profile)):
        offset = k - half
        # Source slice
        if offset >= 0:
            src_slice = image[:, offset:min(cols, cols + offset)]
            dst_start = 0
            dst_end = src_slice.shape[1]
        else:
            src_slice = image[:, 0:min(cols, cols + offset)]
            dst_start = -offset
            dst_end = dst_start + src_slice.shape[1]

        if dst_end <= dst_start or src_slice.shape[1] == 0:
            continue

        shifted = src_slice - profile[k]
        bg[:, dst_start:dst_end] = np.minimum(
            bg[:, dst_start:dst_end], shifted
        )

    # Pass 2: along rows (axis=0)
    bg2 = np.full_like(bg, np.inf)
    for k in range(len(profile)):
        offset = k - half
        if offset >= 0:
            src_slice = bg[offset:min(rows, rows + offset), :]
            dst_start = 0
            dst_end = src_slice.shape[0]
        else:
            src_slice = bg[0:min(rows, rows + offset), :]
            dst_start = -offset
            dst_end = dst_start + src_slice.shape[0]

        if dst_end <= dst_start or src_slice.shape[0] == 0:
            continue

        shifted = src_slice - profile[k]
        bg2[dst_start:dst_end, :] = np.minimum(
            bg2[dst_start:dst_end, :], shifted
        )

    return bg2


@processor_tags(modalities=[IM.SAR, IM.PAN, IM.EO, IM.SWIR, IM.MWIR, IM.LWIR], category=PC.BACKGROUND)
@processor_version('1.54j')
class RollingBallBackground(ImageTransform):
    """Rolling-ball background subtraction, ported from ImageJ 1.54j.

    Estimates non-uniform background using Sternberg's rolling-ball
    algorithm and subtracts it from the image. Uses the ImageJ
    shrink-smooth-expand optimization for performance on large images.

    Parameters
    ----------
    radius : float
        Rolling ball radius in pixels. Larger values produce smoother
        background estimates. ImageJ default is 50.
    light_background : bool
        If True, assumes bright background and dark objects (inverts
        the rolling direction). Default False (dark background,
        bright objects -- typical for SAR amplitude).
    create_background : bool
        If True, ``apply()`` returns the estimated background instead
        of the background-subtracted image. Default False.
    smoothing : bool
        If True, apply 3x3 mean smoothing to the shrunken image before
        rolling, as in ImageJ. Default True.

    Notes
    -----
    Port of ``ij/plugin/filter/BackgroundSubtracter.java`` from ImageJ
    1.54j (public domain). Original algorithm by S. Sternberg (1983),
    ImageJ implementation by M. Castle, J. Keller, and W. Rasband.

    Examples
    --------
    >>> from grdl.imagej import RollingBallBackground
    >>> rb = RollingBallBackground(radius=50)
    >>> corrected = rb.apply(sar_amplitude_image)
    """

    __imagej_source__ = 'ij/plugin/filter/BackgroundSubtracter.java'
    __imagej_version__ = '1.54j'
    __gpu_compatible__ = False

    radius: Annotated[float, Range(min=0.5), Desc('Rolling ball radius in pixels')] = 50.0
    light_background: Annotated[bool, Desc('Assume bright background and dark objects')] = False
    create_background: Annotated[bool, Desc('Return estimated background instead of subtracted')] = False
    smoothing: Annotated[bool, Desc('Apply 3x3 mean smoothing to shrunken image')] = True

    def __post_init__(self):
        self.radius = max(1, int(round(self.radius)))

    def apply(self, source: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply rolling-ball background subtraction.

        Parameters
        ----------
        source : np.ndarray
            2D image array. Any numeric dtype (converted to float64
            internally). Shape ``(rows, cols)``.

        Returns
        -------
        np.ndarray
            Background-subtracted image (or background if
            ``create_background=True``), same shape as input,
            dtype float64.

        Raises
        ------
        ValueError
            If source is not 2D.
        """
        if source.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {source.shape}"
            )

        p = self._resolve_params(kwargs)
        image = source.astype(np.float64)

        radius = max(1, int(round(p['radius'])))

        if p['light_background']:
            image = -image

        # Shrink-smooth-expand optimization
        shrink_factor = max(radius // 10, 1)
        effective_radius = max(radius // shrink_factor, 1)

        small = _shrink_image(image, shrink_factor)

        if p['smoothing'] and shrink_factor > 1:
            small = uniform_filter(small, size=3, mode='nearest')

        background_small = _roll_paraboloid_vectorized(small, effective_radius)

        background = _expand_image(
            background_small, image.shape, shrink_factor
        )

        if p['light_background']:
            image = -image
            background = -background

        if p['create_background']:
            return background

        result = image - background
        np.maximum(result, 0.0, out=result)
        return result
