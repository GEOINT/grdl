# -*- coding: utf-8 -*-
"""Polarimetric Parameters - Scalar and local SAR image statistics.

``complex_correlation`` computes a global coefficient or a local 2D map.
Differential phase, linear depolarization ratio, and coefficient of variation
are reserved entry points and are not implemented yet.

Dependencies
------------
scipy

Author
------
Jason Fritz, PhD
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-11-09

Modified
--------
2026-11-10
"""

from numbers import Integral

import numpy as np

from grdl.image_processing.base import ImageTransform
from grdl.image_processing.filters.base import PolarimetricFilter
from grdl.image_processing.filters.linear import boxcar_mean

__all__ = [
    'complex_correlation',
    'differential_phase',
    'linear_depolarization_ratio',
    'coefficient_of_variation',
]


def complex_correlation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    return_map: bool = False,
    window_size: int = 7,
    mode: str = 'reflect',
    spatial_filter: ImageTransform | None = None,
) -> complex | np.ndarray:
    """Return the uncentered normalized complex correlation of two arrays.

    Computes ``sum(first * conj(second)) / sqrt(sum(abs(first)**2) *
    sum(abs(second)**2))`` over all samples, without subtracting means,
    masking samples. With ``return_map=True``, replaces each sum by a local
    boxcar mean, or uses ``spatial_filter`` on the second-order products
    before normalization. Inputs are assumed to describe corresponding
    (coregistered) samples. No registration is performed.

    Parameters
    ----------
    first, second : np.ndarray
        Equal-shaped, nonempty real or complex numeric arrays of any dimension.
        All samples must be finite. Scalar output requires nonzero energy
        in each array; map output or a spatial filter requires 2D inputs.
    return_map : bool
        If False (default), return one global complex coefficient. If True,
        return a complex128 array with the same shape as the inputs.
    window_size : int
        Positive odd side length of the local square window, default 7.
        Used only for map output without a spatial filter. A unit window measures phase alone,
        with unit magnitude wherever both samples have nonzero energy.
    mode : str
        Boxcar boundary handling: 'reflect' (default), 'constant' (zero
        padding), 'nearest', 'wrap', or 'mirror'. Used only for map output
        without a spatial filter.
    spatial_filter : ImageTransform, optional
        Configured GRDL filter replacing boxcar averaging. A
        ``PolarimetricFilter`` (e.g. RefinedLeeFilter, IDANFilter, WishartFilter)
        jointly filters the 2x2 covariance matrix via
        ``apply(matrix, input_kind='matrix', matrix_type='C2')``. Other
        transforms (e.g. MeanFilter or GaussianFilter) filter the complex
        cross-product and each real power image separately. The filter must
        preserve shape, finite values, complex cross-products, and valid
        covariance statistics. It owns its window and boundary settings;
        ``window_size`` and ``mode`` are ignored. Scalar output pools the
        filtered products over the image before normalization, rather than
        averaging normalized correlation values. Input amplitudes share one
        scale so joint filters retain the relative channel powers.

    Returns
    -------
    complex or np.ndarray
        Normalized coefficient, with magnitude at most one up to roundoff.
        If ``second = first * exp(1j * phase)``, the result is
        ``exp(-1j * phase)`` wherever the coefficient is defined. Map pixels
        whose window has zero energy in either input are complex NaN.
        Computation uses complex128 precision.

    Raises
    ------
    TypeError
        If either array has a nonnumeric dtype or the filter is not an
        ``ImageTransform`` instance.
    ValueError
        If shapes differ, arrays are empty, samples are nonfinite, or either
        array has zero energy for scalar output. Also raised for invalid
        map/filter dimensions, window size, boundary mode, or filter output.
    """
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape:
        raise ValueError(f'Input shapes differ: {first.shape} and {second.shape}')
    if first.size == 0:
        raise ValueError('Inputs must be nonempty')
    for array in (first, second):
        if array.dtype.kind not in 'iufc':
            raise TypeError('Inputs must have real or complex numeric dtypes')
        if not np.isfinite(array).all():
            raise ValueError('Inputs must contain only finite values')

    if spatial_filter is not None and not isinstance(spatial_filter, ImageTransform):
        raise TypeError('spatial_filter must be an ImageTransform instance')
    if spatial_filter is not None and first.ndim != 2:
        raise ValueError('Spatial filtering requires 2D inputs')
    if return_map:
        if first.ndim != 2:
            raise ValueError('Map output requires 2D inputs')
        if spatial_filter is None and (
                isinstance(window_size, bool) or not isinstance(window_size, Integral)
                or window_size < 1 or window_size % 2 == 0):
            raise ValueError('Map window_size must be a positive odd integer')

    first = first.astype(np.complex128)
    second = second.astype(np.complex128)
    first_scale = max(np.max(np.abs(np.real(first))), np.max(np.abs(np.imag(first))))
    second_scale = max(np.max(np.abs(np.real(second))), np.max(np.abs(np.imag(second))))
    if spatial_filter is None and not return_map:
        if first_scale == 0 or second_scale == 0:
            raise ValueError('Each input must have nonzero energy')
        first = first / first_scale
        second = second / second_scale
        return complex(
            np.vdot(second, first) / np.linalg.norm(first) / np.linalg.norm(second)
        )
    if spatial_filter is not None:
        common_scale = max(first_scale, second_scale)
        first = first / common_scale if common_scale > 0 else first
        second = second / common_scale if common_scale > 0 else second
        cross = first * np.conj(second)
        first_power = np.abs(first)**2
        second_power = np.abs(second)**2
        if isinstance(spatial_filter, PolarimetricFilter):
            matrix = np.empty((2, 2, *first.shape), dtype=np.complex128)
            matrix[0, 0] = first_power
            matrix[1, 1] = second_power
            matrix[0, 1] = cross
            matrix[1, 0] = np.conj(cross)
            filtered = np.asarray(spatial_filter.apply(
                matrix, input_kind='matrix', matrix_type='C2',
            ))
            if filtered.shape != matrix.shape or not np.isfinite(filtered).all():
                raise ValueError('Spatial filter must preserve matrix shape and finite values')
            cross = filtered[0, 1]
            first_power = np.real(filtered[0, 0])
            second_power = np.real(filtered[1, 1])
        else:
            cross, first_power, second_power = (
                np.asarray(spatial_filter.apply(term))
                for term in (cross, first_power, second_power)
            )
            if any(term.shape != first.shape or not np.isfinite(term).all()
                   for term in (cross, first_power, second_power)):
                raise ValueError('Spatial filter must preserve image shape and finite values')
            if np.iscomplexobj(first_power) or np.iscomplexobj(second_power):
                raise ValueError('Spatial filter must return real power images')
    else:
        first = first / first_scale if first_scale > 0 else first
        second = second / second_scale if second_scale > 0 else second
        cross = boxcar_mean(first * np.conj(second), window_size, mode=mode)
        first_power = boxcar_mean(np.abs(first)**2, window_size, mode=mode)
        second_power = boxcar_mean(np.abs(second)**2, window_size, mode=mode)
    if not return_map:
        pooled_first = float(np.mean(first_power))
        pooled_second = float(np.mean(second_power))
        if pooled_first <= 0 or pooled_second <= 0:
            raise ValueError('Each filtered input must have nonzero energy')
        return complex(np.mean(cross) / np.sqrt(pooled_first) / np.sqrt(pooled_second))
    denominator = np.sqrt(np.maximum(first_power, 0)) * np.sqrt(
        np.maximum(second_power, 0)
    )
    result = np.full(first.shape, complex(np.nan, np.nan), dtype=np.complex128)
    np.divide(cross, denominator, out=result, where=denominator > 0)
    return result


def differential_phase(
    first: np.ndarray,
    second: np.ndarray,
    *,
    return_map: bool = False,
    window_size: int = 7,
    mode: str = 'reflect',
    spatial_filter: ImageTransform | None = None,
) -> float | np.ndarray:
    """Placeholder for differential phase; reserve scalar or same-shaped map output.

    Output and spatial-filter options follow ``complex_correlation``.
    This function always raises ``NotImplementedError`` until implemented.
    """
    raise NotImplementedError('differential_phase is not implemented yet')


def linear_depolarization_ratio(
    cross_pol: np.ndarray,
    co_pol: np.ndarray,
    *,
    return_map: bool = False,
    window_size: int = 7,
    mode: str = 'reflect',
    spatial_filter: ImageTransform | None = None,
) -> float | np.ndarray:
    """Placeholder for cross/co-pol power ratio; reserve scalar or map output.

    Output and spatial-filter options follow ``complex_correlation``.
    This function always raises ``NotImplementedError`` until implemented.
    """
    raise NotImplementedError('linear_depolarization_ratio is not implemented yet')


def coefficient_of_variation(
    image: np.ndarray,
    *,
    return_map: bool = False,
    window_size: int = 7,
    mode: str = 'reflect',
    spatial_filter: ImageTransform | None = None,
) -> float | np.ndarray:
    """Placeholder for coefficient of variation; reserve scalar or map output.

    Output and spatial-filter options follow ``complex_correlation``.
    This function always raises ``NotImplementedError`` until implemented.
    """
    raise NotImplementedError('coefficient_of_variation is not implemented yet')
