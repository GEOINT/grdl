# -*- coding: utf-8 -*-
"""Tests for scalar and local SAR polarimetric parameters.

Dependencies
------------
pytest
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

import numpy as np
import pytest

from grdl.image_processing.filters import GaussianFilter, MeanFilter, RefinedLeeFilter
from grdl.image_processing.sar import (
    coefficient_of_variation,
    complex_correlation,
    differential_phase,
    linear_depolarization_ratio,
)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_correlation_matches_definition(dtype):
    rng = np.random.default_rng(42)
    first = (rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))).astype(dtype)
    second = (rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))).astype(dtype)
    first_reference = first.astype(np.complex128)
    second_reference = second.astype(np.complex128)
    expected = np.sum(first_reference * np.conj(second_reference)) / np.sqrt(
        np.sum(np.abs(first_reference)**2) * np.sum(np.abs(second_reference)**2)
    )
    first_before, second_before = first.copy(), second.copy()
    result = complex_correlation(first, second)
    assert isinstance(result, complex)
    assert result == pytest.approx(expected)
    assert complex_correlation(second, first) == pytest.approx(result.conjugate())
    np.testing.assert_array_equal(first, first_before)
    np.testing.assert_array_equal(second, second_before)


def test_correlation_phase_and_gain():
    first = np.array([[1 + 2j, 3 - 1j], [0, -2j]])
    phase = 0.7
    second = 3 * first * np.exp(1j * phase)
    assert complex_correlation(first, second) == pytest.approx(np.exp(-1j * phase))


def test_correlation_is_uncentered():
    first = np.full((2, 3), 2 + 1j)
    assert complex_correlation(first, first) == pytest.approx(1 + 0j)


def test_correlation_orthogonal():
    assert complex_correlation(np.array([1, 1j]), np.array([1, -1j])) == 0j


@pytest.mark.parametrize('scale', [1e-300, 1e300])
def test_correlation_extreme_amplitudes(scale):
    first = np.array([1 + 2j, -3 + 1j])
    second = np.array([2 - 1j, 1 + 3j])
    assert complex_correlation(first * scale, second / scale) == pytest.approx(
        complex_correlation(first, second)
    )


@pytest.mark.parametrize('value', [
    complex(1e-310, 1e-310), complex(np.nextafter(0., 1.), 0.),
    complex(1e308, 1e308),
])
@pytest.mark.parametrize('return_map', [False, True])
@pytest.mark.parametrize('filter_type', [None, MeanFilter, RefinedLeeFilter])
def test_correlation_self_at_float64_limits(value, return_map, filter_type):
    image = np.full((9, 9), value)
    filtering = filter_type() if filter_type is not None else None
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result = complex_correlation(
            image, image, return_map=return_map, spatial_filter=filtering,
        )
    np.testing.assert_allclose(result, 1, atol=1e-14)


@pytest.mark.parametrize('return_map', [False, True])
@pytest.mark.parametrize('filter_type', [MeanFilter, GaussianFilter])
@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_linear_filter_handles_extreme_channel_power_ratios(return_map, filter_type, scale):
    first = np.full((9, 9), scale, dtype=complex)
    second = np.full((9, 9), np.exp(0.7j) / scale)
    result = complex_correlation(
        first, second, spatial_filter=filter_type(), return_map=return_map,
    )
    np.testing.assert_allclose(result, np.exp(-0.7j), atol=1e-14)


@pytest.mark.parametrize('return_map', [False, True])
@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_joint_filter_rejects_unrepresentable_power_ratios(monkeypatch, return_map, scale):
    filtering = RefinedLeeFilter()
    monkeypatch.setattr(filtering, 'apply', lambda *args, **kwargs: pytest.fail(
        'Unrepresentable covariance must be rejected before joint filtering'
    ))
    with pytest.raises(ValueError, match='dynamic range.*common-scale covariance'):
        complex_correlation(
            np.full((9, 9), scale), np.full((9, 9), 1 / scale),
            spatial_filter=filtering, return_map=return_map,
        )


@pytest.mark.parametrize('filter_type', [None, MeanFilter, RefinedLeeFilter])
def test_signed_zero_energy_contract(filter_type):
    first = np.full((9, 9), complex(-0., -0.))
    second = np.ones_like(first)
    filtering = filter_type() if filter_type is not None else None
    with pytest.raises(ValueError, match='nonzero energy'):
        complex_correlation(first, second, spatial_filter=filtering)
    result = complex_correlation(first, second, return_map=True, spatial_filter=filtering)
    assert np.isnan(result.real).all()
    assert np.isnan(result.imag).all()


@pytest.mark.parametrize('shape', [(), (4,), (2, 2), (1, 2, 2)])
def test_correlation_accepts_any_dimension(shape):
    first = np.full(shape, 1 + 2j)
    assert complex_correlation(first, first) == pytest.approx(1)


@pytest.mark.parametrize('first, second, message', [
    (np.ones((2, 3)), np.ones((3, 2)), 'shapes differ'),
    (np.array([]), np.array([]), 'nonempty'),
    (np.zeros(2), np.ones(2), 'nonzero energy'),
    (np.ones(2), np.zeros(2), 'nonzero energy'),
    (np.array([np.nan]), np.ones(1), 'finite'),
    (np.ones(1), np.array([np.inf]), 'finite'),
    (np.array([complex(1, np.inf)]), np.ones(1), 'finite'),
])
def test_correlation_rejects_undefined_inputs(first, second, message):
    with pytest.raises(ValueError, match=message):
        complex_correlation(first, second)


@pytest.mark.parametrize('invalid', [np.array(['a']), np.array([object()])])
def test_correlation_rejects_nonnumeric_inputs(invalid):
    with pytest.raises(TypeError, match='numeric'):
        complex_correlation(invalid, np.ones(1))


@pytest.mark.parametrize('function, count', [
    (differential_phase, 2),
    (linear_depolarization_ratio, 2),
    (coefficient_of_variation, 1),
])
@pytest.mark.parametrize('return_map', [False, True])
def test_placeholders_raise(function, count, return_map):
    with pytest.raises(NotImplementedError, match='not implemented yet'):
        function(
            *(np.ones((2, 2), dtype=complex) for _ in range(count)),
            return_map=return_map, window_size=3, mode='reflect',
            spatial_filter=MeanFilter(kernel_size=3),
        )


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('mode, padding', [
    ('reflect', 'symmetric'), ('constant', 'constant'),
    ('nearest', 'edge'), ('wrap', 'wrap'), ('mirror', 'reflect'),
])
def test_correlation_map_matches_local_definition(dtype, mode, padding):
    rng = np.random.default_rng(42)
    first = (rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))).astype(dtype)
    second = (rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))).astype(dtype)
    first_before, second_before = first.copy(), second.copy()
    result = complex_correlation(first, second, return_map=True, window_size=3, mode=mode)
    padded_first = np.pad(first.astype(np.complex128), 1, mode=padding)
    padded_second = np.pad(second.astype(np.complex128), 1, mode=padding)
    expected = np.empty(first.shape, dtype=np.complex128)
    for row, col in np.ndindex(first.shape):
        local_first = padded_first[row:row + 3, col:col + 3]
        local_second = padded_second[row:row + 3, col:col + 3]
        expected[row, col] = np.sum(local_first * np.conj(local_second)) / np.sqrt(
            np.sum(np.abs(local_first)**2) * np.sum(np.abs(local_second)**2)
        )
    assert result.shape == first.shape
    assert result.dtype == np.complex128
    np.testing.assert_allclose(result, expected, atol=1e-14)
    np.testing.assert_array_equal(first, first_before)
    np.testing.assert_array_equal(second, second_before)


@pytest.mark.parametrize('scale', [1e-300, 1, 1e300])
def test_correlation_map_phase_and_gain(scale):
    first = np.arange(1, 21).reshape(4, 5) * (1 + 2j)
    second = 3 * first * np.exp(0.7j)
    result = complex_correlation(first * scale, second / scale, return_map=True)
    np.testing.assert_allclose(result, np.exp(-0.7j), atol=1e-14)


def test_correlation_map_unit_window():
    first = np.array([[1 + 1j, 2], [0, -1j]])
    second = np.ones((2, 2), dtype=complex)
    result = complex_correlation(first, second, return_map=True, window_size=1)
    np.testing.assert_allclose(result[first != 0], first[first != 0] / np.abs(first[first != 0]))
    assert np.isnan(result[1, 0])


def test_correlation_map_zero_energy_neighborhoods():
    first = np.zeros((7, 7), dtype=complex)
    first[3, 3] = 1
    result = complex_correlation(first, np.ones_like(first), return_map=True, window_size=3)
    assert np.isnan(result[0, 0])
    assert result[3, 3] == pytest.approx(1 / 3)
    assert np.isnan(complex_correlation(np.zeros_like(first), first, return_map=True)).all()


@pytest.mark.parametrize('window_size', [0, -1, 2, 1.5, True])
def test_correlation_map_rejects_invalid_window(window_size):
    with pytest.raises(ValueError, match='positive odd integer'):
        complex_correlation(np.ones((2, 3)), np.ones((2, 3)),
                            return_map=True, window_size=window_size)


@pytest.mark.parametrize('shape', [(), (4,), (1, 2, 2)])
def test_correlation_map_requires_2d(shape):
    with pytest.raises(ValueError, match='2D'):
        complex_correlation(np.ones(shape), np.ones(shape), return_map=True)


def test_correlation_map_rejects_invalid_mode():
    with pytest.raises(ValueError, match='boundary mode'):
        complex_correlation(np.ones((2, 3)), np.ones((2, 3)), return_map=True, mode='invalid')


@pytest.mark.parametrize('return_map', [False, True])
@pytest.mark.parametrize('filter_type', [MeanFilter, GaussianFilter, RefinedLeeFilter])
def test_correlation_spatial_filter_uses_second_order_products(filter_type, return_map):
    rng = np.random.default_rng(42)
    first = rng.normal(size=(11, 12)) + 1j * rng.normal(size=(11, 12))
    second = 3 * (rng.normal(size=(11, 12)) + 1j * rng.normal(size=(11, 12)))
    first_before, second_before = first.copy(), second.copy()
    filtering = filter_type()
    scale = max(np.abs(first.real).max(), np.abs(first.imag).max(),
                np.abs(second.real).max(), np.abs(second.imag).max())
    scaled_first, scaled_second = first / scale, second / scale
    if filter_type is RefinedLeeFilter:
        matrix = np.empty((2, 2, *first.shape), dtype=complex)
        matrix[0, 0] = np.abs(scaled_first)**2
        matrix[1, 1] = np.abs(scaled_second)**2
        matrix[0, 1] = scaled_first * np.conj(scaled_second)
        matrix[1, 0] = np.conj(matrix[0, 1])
        filtered = filtering.apply(matrix, input_kind='matrix', matrix_type='C2')
        cross, first_power, second_power = filtered[0, 1], filtered[0, 0].real, filtered[1, 1].real
    else:
        cross = filtering.apply(scaled_first * np.conj(scaled_second))
        first_power = filtering.apply(np.abs(scaled_first)**2)
        second_power = filtering.apply(np.abs(scaled_second)**2)
    if return_map:
        expected = cross / np.sqrt(first_power) / np.sqrt(second_power)
    else:
        expected = np.mean(cross) / np.sqrt(np.mean(first_power)) / np.sqrt(np.mean(second_power))
    result = complex_correlation(first, second, return_map=return_map,
                                 spatial_filter=filtering, window_size=0, mode='ignored')
    np.testing.assert_allclose(result, expected, atol=1e-14)
    np.testing.assert_array_equal(first, first_before)
    np.testing.assert_array_equal(second, second_before)


def test_mean_filter_matches_default_boxcar():
    rng = np.random.default_rng(7)
    first = rng.normal(size=(8, 9)) + 1j * rng.normal(size=(8, 9))
    second = rng.normal(size=(8, 9)) + 1j * rng.normal(size=(8, 9))
    expected = complex_correlation(first, second, return_map=True, window_size=3)
    result = complex_correlation(first, second, return_map=True,
                                 spatial_filter=MeanFilter(kernel_size=3))
    np.testing.assert_allclose(result, expected, atol=1e-14)


@pytest.mark.parametrize('return_map', [False, True])
def test_spatial_filter_requires_2d(return_map):
    with pytest.raises(ValueError, match='2D'):
        complex_correlation(np.ones(5), np.ones(5), return_map=return_map,
                            spatial_filter=GaussianFilter())


def test_spatial_filter_rejects_invalid_type():
    with pytest.raises(TypeError, match='ImageTransform'):
        complex_correlation(np.ones((2, 3)), np.ones((2, 3)), spatial_filter=object())


@pytest.mark.parametrize('invalid_result', [np.ones((1, 1)), np.full((2, 3), np.nan)])
def test_spatial_filter_validates_output(monkeypatch, invalid_result):
    filtering = MeanFilter(kernel_size=3)
    monkeypatch.setattr(filtering, 'apply', lambda source: invalid_result)
    with pytest.raises(ValueError, match='shape and finite'):
        complex_correlation(np.ones((2, 3)), np.ones((2, 3)), spatial_filter=filtering)


@pytest.mark.parametrize('return_map', [False, True])
@pytest.mark.parametrize('invalid, message', [
    ('first_imaginary', 'real covariance diagonal'),
    ('second_imaginary', 'real covariance diagonal'),
    ('negative', 'nonnegative'),
    ('asymmetric', 'Hermitian'),
    ('non_psd', 'positive semidefinite'),
    ('nonfinite', 'shape and finite'),
    ('shape', 'shape and finite'),
])
def test_joint_filter_rejects_invalid_covariance(monkeypatch, return_map, invalid, message):
    matrix = np.ones((2, 2, 3, 4), dtype=complex)
    if invalid == 'first_imaginary':
        matrix[0, 0] += 1j
    elif invalid == 'second_imaginary':
        matrix[1, 1] += 1j
    elif invalid == 'negative':
        matrix[0, 0] = -1
    elif invalid == 'asymmetric':
        matrix[1, 0] = 0
    elif invalid == 'non_psd':
        matrix[0, 1] = matrix[1, 0] = 2
    elif invalid == 'nonfinite':
        matrix[0, 1] = np.inf
    else:
        matrix = matrix[:, :, :1]
    filtering = RefinedLeeFilter()
    monkeypatch.setattr(filtering, 'apply', lambda *args, **kwargs: matrix)
    with pytest.raises(ValueError, match=message):
        complex_correlation(
            np.ones((3, 4)), np.ones((3, 4)),
            spatial_filter=filtering, return_map=return_map,
        )


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_joint_filter_accepts_covariance_roundoff(monkeypatch, dtype):
    matrix = np.ones((2, 2, 3, 4), dtype=dtype)
    epsilon = np.finfo(matrix.real.dtype).eps
    matrix[0, 0] += epsilon * 1j
    matrix[1, 0] += epsilon * 1j
    filtering = RefinedLeeFilter()
    monkeypatch.setattr(filtering, 'apply', lambda *args, **kwargs: matrix)
    result = complex_correlation(
        np.ones((3, 4)), np.ones((3, 4)), spatial_filter=filtering,
    )
    assert result == pytest.approx(1)


@pytest.mark.parametrize('return_map', [False, True])
@pytest.mark.parametrize('power, cross, message', [
    (-1., 0., 'nonnegative'), (1., 2., 'positive semidefinite'),
])
def test_independent_filter_rejects_invalid_covariance(
    monkeypatch, return_map, power, cross, message,
):
    filtering = MeanFilter()
    outputs = iter([
        np.full((3, 4), cross, dtype=complex),
        np.full((3, 4), power), np.ones((3, 4)),
    ])
    monkeypatch.setattr(filtering, 'apply', lambda source: next(outputs))
    with pytest.raises(ValueError, match=message):
        complex_correlation(
            np.ones((3, 4)), np.ones((3, 4)),
            spatial_filter=filtering, return_map=return_map,
        )
