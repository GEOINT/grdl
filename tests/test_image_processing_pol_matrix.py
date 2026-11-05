# -*- coding: utf-8 -*-
"""
Unit tests for CovarianceMatrix and CoherencyMatrix.

Covers:
- C2 and C3 shape and Hermitian property
- T2 and T3 shape and Hermitian property
- Diagonal elements are real-valued
- Trace equality between C3 and T3 (both equal span)
- CYX cube auto-extraction via execute()
- execute() metadata stamping (axis_order, channel_metadata, bands)
- Direct kwargs interface (shh, shv, svh, svv)
- Window size validation
"""

import numpy as np
import pytest
import dataclasses

from grdl.IO.models.base import ChannelMetadata, ImageMetadata
from grdl.image_processing.decomposition.pol_matrix import (
    CovarianceMatrix,
    CoherencyMatrix,
    StokesVector,
    KennaughMatrix,
    pack_matrix,
    unpack_matrix,
    prepare_polarimetric_matrix,
    stokes_from_c2,
    stokes_degree_of_polarization,
    c2_degree_of_polarization,
)
from grdl.exceptions import ValidationError
from grdl.image_processing.decomposition.dual_pol_base import DualPolDecompositionBase
from grdl.image_processing.decomposition.dop_dp import DegreeOfPolarizationDP
from grdl.image_processing.decomposition.dop_cp import CompactPolDegreeOfPolarization
from scipy.ndimage import uniform_filter


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('window', [1, 3, 7])
def test_canonical_stokes_precision_and_sign(dtype, window):
    rng = np.random.default_rng(103)
    fields = (rng.normal(size=(2, 9, 11)) + 1j * rng.normal(size=(2, 9, 11))).astype(dtype)
    fields[:, 0, 0] = [1, 1j]
    result = StokesVector(window).compute(fields[0], fields[1])
    c11 = _reference_element(fields[0], fields[0], window).real
    c12 = _reference_element(fields[0], fields[1], window)
    c22 = _reference_element(fields[1], fields[1], window).real
    expected = np.stack([c11 + c22, c11 - c22, 2 * c12.real, -2 * c12.imag])
    assert result.dtype == np.finfo(dtype).dtype
    tolerance = 2e-6 if dtype == np.complex64 else 5e-14
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
    dual = DualPolDecompositionBase._normalized_stokes(c11, c12, c22)
    np.testing.assert_allclose(np.stack(dual[:4]), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(
        dual[-1], DualPolDecompositionBase._normalized_stokes(c11, c12.conj(), c22)[-1])
    circular = StokesVector(1).compute(np.ones((2, 3), dtype=dtype),
                                     np.full((2, 3), 1j, dtype=dtype))
    np.testing.assert_array_equal(circular[3], 2)
    reversed_fields = StokesVector(1).compute(np.full((2, 3), 1j, dtype=dtype),
                                            np.ones((2, 3), dtype=dtype))
    np.testing.assert_array_equal(reversed_fields[3], -2)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_two_component_dop_power_extremes(dtype):
    real_dtype = np.finfo(dtype).dtype
    powers = np.array([0, 1e-30, 1, 1e30] if dtype == np.complex64 else
                      [0, 1e-200, 1, 1e200], dtype=real_dtype)[None, :]
    fields = np.sqrt(powers).astype(dtype)
    zeros = np.zeros_like(fields)
    proc = StokesVector(1)
    with np.errstate(all='raise'):
        stokes = proc.compute(fields, zeros)
        result = proc.degree_of_polarization(stokes)
        dual = DegreeOfPolarizationDP(window_size=1).decompose_dual(fields, zeros)['dop']
        compact = CompactPolDegreeOfPolarization(window_size=1).decompose_compact(
            powers, zeros, zeros, np.zeros_like(powers))['dop']
    for dop in (result, dual, compact):
        assert dop.dtype == real_dtype
        np.testing.assert_allclose(dop, [[0, 1, 1, 1]], rtol=2e-6, atol=2e-6)
    with np.errstate(all='raise'):
        incoherent = np.stack([2 * powers, np.zeros_like(powers),
                               np.zeros_like(powers), np.zeros_like(powers)])
        result = proc.degree_of_polarization(incoherent)
        compact = CompactPolDegreeOfPolarization(window_size=1).decompose_compact(
            powers, zeros, zeros, powers)['dop']
    np.testing.assert_array_equal(result, 0)
    np.testing.assert_array_equal(compact, 0)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_explicit_c21_is_not_replaced(dtype):
    first, cross, lower, second = [np.full((3, 5), value, dtype=dtype)
                                  for value in (3, .4 + .7j, .1 - .2j, 2)]
    result = stokes_from_c2(first, cross, second, lower)
    for actual, expected in zip(result, (5, 1, .5, -.9), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    expected = np.sqrt((1 - 4 * (first * second - cross * lower) / (first + second) ** 2).real)
    actual = c2_degree_of_polarization(first, cross, second, lower)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert actual.dtype == np.finfo(dtype).dtype
    assert not np.allclose(actual, c2_degree_of_polarization(first, cross, second))


def test_malformed_power_policies_are_preserved():
    first = np.array([[0., 1., 0., -3., np.nan]])
    second = np.array([[0., -1., 0., -2., 2.]])
    cross = np.array([[0., 0., 1., 0., 0.]])
    lower = -cross
    np.testing.assert_allclose(c2_degree_of_polarization(first, cross, second, lower),
                               [[0, 1, 0, .2, np.nan]], atol=1e-14, equal_nan=True)
    power = np.array([[0., 0., -1., np.nan]])
    polarized = np.array([[0., 1., 1., 1.]])
    zeros = np.zeros_like(power)
    np.testing.assert_array_equal(stokes_degree_of_polarization(power, polarized, zeros, zeros),
                                  [[0, 1, 0, np.nan]])


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('layout', ['CYX', 'YXC'])
def test_stokes_execute_precision_metadata(dtype, layout):
    source = np.stack([np.ones((9, 11), dtype=dtype), np.full((9, 11), 1j, dtype=dtype)])
    metadata = ImageMetadata(format='test', rows=9, cols=11, bands=2,
                             dtype=np.dtype(dtype).name, axis_order=layout)
    if layout == 'YXC':
        source = source.transpose(1, 2, 0)
    result, updated = StokesVector(1).execute(metadata, source)
    assert result.dtype == np.finfo(dtype).dtype
    assert updated.dtype == str(result.dtype)
    assert updated.axis_order == 'CYX'
    assert (updated.bands, updated.rows, updated.cols) == (4, 9, 11)
    assert [channel.name for channel in updated.channel_metadata] == ['S0', 'S1', 'S2', 'S3']
    assert all(channel.source_indices == [0, 1] for channel in updated.channel_metadata)
    np.testing.assert_array_equal(result[3], 2)


def _reference_target(channels, kind):
    channels = [channel.astype(np.result_type(*channels), copy=False) for channel in channels]
    if len(channels) == 2:
        if kind == 'C':
            return channels
        return [(channels[0] + channels[1]) / np.sqrt(2.0),
                (channels[0] - channels[1]) / np.sqrt(2.0)]
    cross = channels[1] if len(channels) == 3 else (channels[1] + channels[2]) / 2.0
    if kind == 'C':
        return [channels[0], np.sqrt(2.0) * cross, channels[-1]]
    return [(channels[0] + channels[-1]) / np.sqrt(2.0),
            (channels[0] - channels[-1]) / np.sqrt(2.0), np.sqrt(2.0) * cross]


def _reference_element(first, second, window):
    product = first * second.conj()
    return (uniform_filter(product.real, size=window, mode='reflect')
            + 1j * uniform_filter(product.imag, size=window, mode='reflect'))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('window', [1, 3, 7])
@pytest.mark.parametrize('count', [2, 3, 4])
@pytest.mark.parametrize('processor,kind', [(CovarianceMatrix, 'C'), (CoherencyMatrix, 'T')])
def test_matrix_element_reference_and_precision(dtype, window, count, processor, kind):
    rng = np.random.default_rng(43)
    channels = (rng.normal(size=(count, 9, 11)) + 1j * rng.normal(size=(count, 9, 11))).astype(dtype)
    result = processor(window_size=window).compute(channels)
    assert result.dtype == dtype
    components = _reference_target(channels, kind)
    tolerance = 2e-6 if dtype == np.complex64 else 5e-14
    for row, first in enumerate(components):
        for col, second in enumerate(components):
            expected = _reference_element(first, second, window)
            np.testing.assert_allclose(result[row, col], expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(result[row, row].imag, 0)


@pytest.mark.parametrize('dtypes', [(np.complex64, np.complex64),
                                   (np.complex128, np.complex128),
                                   (np.complex64, np.complex128)])
@pytest.mark.parametrize('window', [1, 3, 7])
@pytest.mark.parametrize('kind', ['C', 'T'])
def test_dual_elements_reference_precision_without_matrix(monkeypatch, dtypes, window, kind):
    from grdl.image_processing.decomposition import pol_matrix

    def reject_matrix(*args, **kwargs):
        raise AssertionError('Dual helpers must not allocate a full matrix')

    rng = np.random.default_rng(71)
    channels = [(rng.normal(size=(9, 11)) + 1j * rng.normal(size=(9, 11))).astype(dtype)
                for dtype in dtypes]
    processor = CovarianceMatrix if kind == 'C' else CoherencyMatrix
    matrix = processor(window_size=window).compute(np.stack(channels))
    monkeypatch.setattr(pol_matrix, 'compute_matrix', reject_matrix)
    compute = (DualPolDecompositionBase._compute_c2 if kind == 'C'
               else DualPolDecompositionBase._compute_t2)
    first, cross, second = compute(channels[0], channels[1], window)
    dtype = np.result_type(*channels)
    assert cross.dtype == dtype
    assert first.dtype == second.dtype == np.finfo(dtype).dtype
    components = _reference_target(channels, kind)
    tolerance = 2e-6 if dtype == np.complex64 else 5e-14
    for result, row, col in ((first, 0, 0), (cross, 0, 1), (second, 1, 1)):
        expected = _reference_element(components[row], components[col], window)
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(result, matrix[row, col])


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('kind,count', [('C2', 2), ('T2', 2), ('C3', 3),
                                      ('T3', 3), ('C3', 4), ('T3', 4)])
def test_raw_matrix_preparation_precision(dtype, kind, count):
    rng = np.random.default_rng(83)
    channels = (rng.normal(size=(count, 9, 11)) + 1j * rng.normal(size=(count, 9, 11))).astype(dtype)
    result = prepare_polarimetric_matrix(channels, input_kind='channels', matrix_type=kind)
    processor = CovarianceMatrix if kind.startswith('C') else CoherencyMatrix
    np.testing.assert_array_equal(result, processor(window_size=1).compute(channels))
    assert result.dtype == dtype


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('processor', [CovarianceMatrix, CoherencyMatrix])
def test_matrix_element_gpu_parity(dtype, processor):
    cupy = pytest.importorskip('cupy')
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('No CUDA device available')
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA runtime unavailable')
    rng = np.random.default_rng(89)
    channels = (rng.normal(size=(4, 9, 11)) + 1j * rng.normal(size=(4, 9, 11))).astype(dtype)
    proc = processor(window_size=3)
    result = proc.compute(cupy.asarray(channels))
    assert isinstance(result, cupy.ndarray)
    assert result.dtype == dtype
    tolerance = 2e-6 if dtype == np.complex64 else 5e-14
    np.testing.assert_allclose(cupy.asnumpy(result), proc.compute(channels),
                               rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('window', [1, 7])
def test_stokes_gpu_precision_and_power_policy(dtype, window):
    cupy = pytest.importorskip('cupy')
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('No CUDA device available')
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA runtime unavailable')
    rng = np.random.default_rng(109)
    fields = (rng.normal(size=(2, 9, 11)) + 1j * rng.normal(size=(2, 9, 11))).astype(dtype)
    device_fields = cupy.asarray(fields)
    processor = StokesVector(window)
    actual = processor.compute(device_fields[0], device_fields[1])
    expected = processor.compute(fields[0], fields[1])
    assert isinstance(actual, cupy.ndarray)
    assert actual.dtype == np.finfo(dtype).dtype
    tolerance = 2e-6 if dtype == np.complex64 else 5e-14
    np.testing.assert_allclose(cupy.asnumpy(actual), expected,
                               rtol=tolerance, atol=tolerance)
    dop = processor.degree_of_polarization(actual)
    assert isinstance(dop, cupy.ndarray)
    assert dop.dtype == actual.dtype
    np.testing.assert_allclose(cupy.asnumpy(dop), processor.degree_of_polarization(expected),
                               rtol=tolerance, atol=tolerance)
    zero = cupy.zeros((4, 3, 5), dtype=actual.dtype)
    np.testing.assert_array_equal(cupy.asnumpy(processor.degree_of_polarization(zero)), 0)
    np.testing.assert_array_equal(cupy.asnumpy(device_fields), fields)


@pytest.mark.parametrize('processor', [CovarianceMatrix, CoherencyMatrix])
def test_matrix_execute_preserves_complex128_metadata(processor):
    channels = np.ones((2, 5, 7), dtype=np.complex128)
    metadata = ImageMetadata(format='array', rows=5, cols=7, bands=2,
                             dtype='complex128', axis_order='CYX')
    result, updated = processor().execute(metadata, channels)
    assert result.dtype == np.complex128
    assert updated.dtype == 'complex128'


@pytest.mark.parametrize('processor', [CovarianceMatrix, CoherencyMatrix, KennaughMatrix])
@pytest.mark.parametrize('axis_order', ['CYX', 'YXC', None])
@pytest.mark.parametrize('descriptor', ['polarization', 'role', 'txrx'])
def test_shuffled_matrix_descriptors(quad_pol, quad_meta, processor, axis_order, descriptor):
    order = [3, 2, 0, 1]
    roles = ['co_pol_a', 'cross_pol', 'cross_pol_2', 'co_pol_b']
    tokens = ['XX', 'XY', 'YX', 'YY']
    descriptors = []
    for slot, index in enumerate(order):
        descriptors.append(ChannelMetadata(
            index=slot, name=f'band_{slot}', source_indices=[90 + index],
            polarization=tokens[index] if descriptor == 'polarization' else None,
            role=roles[index] if descriptor == 'role' else 'measurement',
            tx_polarization=tokens[index][0] if descriptor == 'txrx' else None,
            rcv_polarization=tokens[index][1] if descriptor == 'txrx' else None,
        ))
    metadata = dataclasses.replace(quad_meta, axis_order=axis_order, channel_metadata=descriptors)
    cube = np.stack(quad_pol)[order]
    if axis_order == 'YXC':
        cube = cube.transpose(1, 2, 0)
    proc = processor(window_size=3)
    result, updated = proc.execute(metadata, cube)
    np.testing.assert_array_equal(result, proc.compute(np.stack(quad_pol)))
    assert updated.dtype == str(result.dtype)
    if processor is CovarianceMatrix:
        assert updated.channel_metadata[0].source_indices == [2]
        assert updated.channel_metadata[4].source_indices == [3, 1]
        assert updated.channel_metadata[1].source_indices == [2, 3, 1]
    elif processor is CoherencyMatrix:
        assert updated.channel_metadata[0].source_indices == [2, 0]
        assert updated.channel_metadata[8].source_indices == [3, 1]
        assert updated.channel_metadata[2].source_indices == [2, 0, 3, 1]
    else:
        assert all(channel.source_indices == [2, 3, 1, 0]
                   for channel in updated.channel_metadata)


@pytest.mark.parametrize('labels,selected', [
    (['V', 'H'], [1, 0]), (['Y', 'X'], [1, 0]), (['L', 'R'], [1, 0]),
    (['VV', 'HH'], [1, 0]), (['VV', 'HV', 'HH'], [2, 0]),
    (['VH', 'VV', 'HH', 'HV'], [2, 1]),
])
@pytest.mark.parametrize('axis_order', ['CYX', 'YXC'])
def test_stokes_labelled_fields(quad_meta, labels, selected, axis_order):
    cube = np.stack([_complex(index) for index in range(len(labels))])
    expected = StokesVector(3).compute(cube[selected[0]], cube[selected[1]])
    metadata = dataclasses.replace(quad_meta, bands=len(labels), axis_order=axis_order,
                                  channel_metadata=[ChannelMetadata(index=index, name=label)
                                                    for index, label in enumerate(labels)])
    if axis_order == 'YXC':
        cube = cube.transpose(1, 2, 0)
    result, updated = StokesVector(3).execute(metadata, cube)
    np.testing.assert_array_equal(result, expected)
    assert all(channel.source_indices == selected for channel in updated.channel_metadata)
    assert updated.dtype == str(result.dtype)


@pytest.mark.parametrize('processor', [CovarianceMatrix, CoherencyMatrix, KennaughMatrix, StokesVector])
def test_explicit_arrays_bypass_bad_source(quad_pol, quad_meta, processor):
    metadata = dataclasses.replace(quad_meta, axis_order='CCYX', rows=1, cols=1,
                                  channel_metadata=[ChannelMetadata(index=0, name='C11')])
    proc = processor(3)
    inputs = (dict(zip(('shh', 'shv', 'svh', 'svv'), quad_pol, strict=True))
              if processor is not StokesVector else {'e_x': quad_pol[0], 'e_y': quad_pol[3]})
    result, updated = proc.execute(metadata, np.zeros((2, 3, 4)), **inputs)
    expected = (proc.compute(np.stack(quad_pol)) if processor is not StokesVector else
                proc.compute(quad_pol[0], quad_pol[3]))
    np.testing.assert_array_equal(result, expected)
    assert (updated.rows, updated.cols) == SHAPE
    assert all(channel.source_indices == [] for channel in updated.channel_metadata)
    incomplete = {'shv': quad_pol[1]} if processor is not StokesVector else {'e_y': quad_pol[3]}
    with pytest.raises(ValueError, match='Explicit|Both'):
        proc.execute(quad_meta, np.stack(quad_pol), **incomplete)


@pytest.mark.parametrize('labels', [
    ['HH', 'HH', 'VH', 'VV'], ['HH', 'HV', 'VH', 'band'],
    ['HH', 'XY', 'VH', 'VV'], ['C11', 'C12_real', 'C12_imag', 'C22'],
])
def test_matrix_rejects_invalid_descriptors(quad_pol, quad_meta, labels):
    metadata = dataclasses.replace(quad_meta, channel_metadata=[
        ChannelMetadata(index=index, name=label) for index, label in enumerate(labels)
    ])
    with pytest.raises(ValueError):
        CovarianceMatrix().execute(metadata, np.stack(quad_pol))


def test_matrix_rejects_ambiguous_layout_and_unlabelled_surplus(quad_meta):
    metadata = dataclasses.replace(quad_meta, rows=4, cols=4, axis_order=None)
    with pytest.raises(ValueError, match='Ambiguous'):
        CovarianceMatrix().execute(metadata, np.ones((4, 4, 4), dtype=np.complex64))
    metadata = dataclasses.replace(quad_meta, bands=5, channel_metadata=None)
    with pytest.raises(ValueError, match='Unlabelled'):
        CovarianceMatrix().execute(metadata, np.ones((5, *SHAPE), dtype=np.complex64))


def test_prepare_preserves_raw_counts_finite_and_basis_contract(quad_pol):
    raw = np.stack(quad_pol)
    shuffled = raw[[3, 2, 0, 1]]
    descriptors = [ChannelMetadata(index=index, name=name)
                   for index, name in enumerate(('VV', 'VH', 'HH', 'HV'))]
    expected = CovarianceMatrix().compute(raw)
    actual = prepare_polarimetric_matrix(shuffled, input_kind='channels', matrix_type='C3',
                                        channel_metadata=descriptors)
    np.testing.assert_array_equal(actual, expected)
    packed = pack_matrix(actual)
    np.testing.assert_allclose(unpack_matrix(packed), actual, atol=3e-7)
    np.testing.assert_array_equal(pack_matrix(unpack_matrix(packed)), packed)
    assert prepare_polarimetric_matrix(actual, input_kind='matrix', matrix_type='T3') is actual
    for kind in ('C4', 'T4', 'C2'):
        with pytest.raises(ValidationError):
            prepare_polarimetric_matrix(raw, input_kind='channels', matrix_type=kind)
    with pytest.raises(ValidationError):
        prepare_polarimetric_matrix(np.concatenate([raw, raw[:1]]),
                                   input_kind='channels', matrix_type='C3')
    with pytest.raises(ValidationError, match='incompatible raw count'):
        prepare_polarimetric_matrix(raw, input_kind='channels', matrix_type='C3',
                                   channel_metadata=[ChannelMetadata(index=index, name=name)
                                                     for index, name in enumerate(
                                                         ('HH', 'HV', 'band', 'band'))])
    invalid = raw.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValidationError, match='non-finite'):
        prepare_polarimetric_matrix(invalid, input_kind='channels', matrix_type='C3')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
SHAPE = (32, 32)


def _complex(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(SHAPE) + 1j * rng.standard_normal(SHAPE)).astype(np.complex64)


@pytest.fixture
def quad_pol():
    return _complex(0), _complex(1), _complex(2), _complex(3)


@pytest.fixture
def dual_pol():
    return _complex(10), _complex(11)


@pytest.fixture
def quad_meta():
    return ImageMetadata(
        format='test', rows=SHAPE[0], cols=SHAPE[1],
        dtype='complex64', bands=4, axis_order='CYX',
        channel_metadata=[
            ChannelMetadata(index=0, name='HH'),
            ChannelMetadata(index=1, name='HV'),
            ChannelMetadata(index=2, name='VH'),
            ChannelMetadata(index=3, name='VV'),
        ],
    )


@pytest.fixture
def dual_meta():
    return ImageMetadata(
        format='test', rows=SHAPE[0], cols=SHAPE[1],
        dtype='complex64', bands=2, axis_order='CYX',
        channel_metadata=[
            ChannelMetadata(index=0, name='VV'),
            ChannelMetadata(index=1, name='VH'),
        ],
    )


# ---------------------------------------------------------------------------
# CovarianceMatrix
# ---------------------------------------------------------------------------

class TestCovarianceMatrix:

    def test_default_window_size_is_one(self):
        assert CovarianceMatrix().window_size == 1

    def test_c3_shape(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        C = CovarianceMatrix(window_size=3).compute(np.stack([shh, shv, svh, svv], axis=0))
        assert C.shape == (3, 3, *SHAPE)

    def test_c2_shape(self, dual_pol):
        s_co, s_cross = dual_pol
        C = CovarianceMatrix(window_size=3).compute(np.stack([s_co, s_cross], axis=0))
        assert C.shape == (2, 2, *SHAPE)

    def test_c3_hermitian(self, quad_pol):
        """C[i,j] = conj(C[j,i]) everywhere."""
        shh, shv, svh, svv = quad_pol
        C = CovarianceMatrix(window_size=3).compute(np.stack([shh, shv, svh, svv], axis=0))
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(C[i, j], np.conj(C[j, i]), atol=1e-6)

    def test_c3_diagonal_real(self, quad_pol):
        """Diagonal elements should be real (zero imaginary part)."""
        shh, shv, svh, svv = quad_pol
        C = CovarianceMatrix(window_size=3).compute(np.stack([shh, shv, svh, svv], axis=0))
        for i in range(3):
            np.testing.assert_allclose(C[i, i].imag, 0.0, atol=1e-6)

    def test_c3_diagonal_nonnegative(self, quad_pol):
        """Power elements must be >= 0."""
        shh, shv, svh, svv = quad_pol
        C = CovarianceMatrix(window_size=3).compute(np.stack([shh, shv, svh, svv], axis=0))
        for i in range(3):
            assert np.all(C[i, i].real >= -1e-6)

    def test_execute_cyx_extraction(self, quad_pol, quad_meta):
        cube = np.stack(quad_pol, axis=0)  # (4, rows, cols)
        cmat = CovarianceMatrix(window_size=3)
        C, out_meta = cmat.execute(quad_meta, cube)
        assert C.shape == (3, 3, *SHAPE)
        assert out_meta.axis_order == 'CCYX'
        assert out_meta.bands == 9
        assert out_meta.channel_metadata is not None
        assert len(out_meta.channel_metadata) == 9
        assert out_meta.channel_metadata[0].name == 'C[0,0]'
        assert out_meta.channel_metadata[4].name == 'C[1,1]'

    def test_execute_dual_pol_cyx(self, dual_pol, dual_meta):
        cube = np.stack(dual_pol, axis=0)  # (2, rows, cols)
        cmat = CovarianceMatrix(window_size=3)
        C, out_meta = cmat.execute(dual_meta, cube)
        assert C.shape == (2, 2, *SHAPE)
        assert out_meta.axis_order == 'CCYX'
        assert out_meta.bands == 4

    def test_execute_three_channel_cube(self, quad_pol, quad_meta):
        shh, shv, _, svv = quad_pol
        cube = np.stack([shh, shv, svv], axis=0)
        meta3 = dataclasses.replace(
            quad_meta,
            bands=3,
            channel_metadata=[
                ChannelMetadata(index=0, name='HH'),
                ChannelMetadata(index=1, name='HV'),
                ChannelMetadata(index=2, name='VV'),
            ],
        )
        cmat = CovarianceMatrix(window_size=3)
        C, out_meta = cmat.execute(meta3, cube)
        assert C.shape == (3, 3, *SHAPE)
        assert out_meta.axis_order == 'CCYX'

    def test_execute_generic_kwargs_aliases(self, quad_pol, quad_meta):
        shh, shv, svh, svv = quad_pol
        cmat = CovarianceMatrix(window_size=3)
        C, _ = cmat.execute(
            quad_meta,
            np.zeros((SHAPE[0], SHAPE[1])),
            co_pol_a=shh,
            cross_pol=shv,
            cross_pol_2=svh,
            co_pol_b=svv,
        )
        assert C.shape == (3, 3, *SHAPE)

    def test_execute_role_based_extraction_generic_names(self, quad_pol, quad_meta):
        shh, shv, _, svv = quad_pol
        cube = np.stack([shh, shv, svv], axis=0)
        meta3 = dataclasses.replace(
            quad_meta,
            bands=3,
            channel_metadata=[
                ChannelMetadata(index=0, name='chan_a', role='co_pol'),
                ChannelMetadata(index=1, name='chan_b', role='cross_pol'),
                ChannelMetadata(index=2, name='chan_c', role='co_pol'),
            ],
        )
        cmat = CovarianceMatrix(window_size=3)
        C, out_meta = cmat.execute(meta3, cube)
        assert C.shape == (3, 3, *SHAPE)
        assert out_meta.axis_order == 'CCYX'

    def test_execute_kwargs_interface(self, quad_pol, quad_meta):
        shh, shv, svh, svv = quad_pol
        cmat = CovarianceMatrix(window_size=3)
        C, _ = cmat.execute(quad_meta, np.zeros((SHAPE[0], SHAPE[1])),
                            shh=shh, shv=shv, svh=svh, svv=svv)
        assert C.shape == (3, 3, *SHAPE)

    def test_repr(self):
        assert repr(CovarianceMatrix(7)) == 'CovarianceMatrix(window_size=7)'

    def test_invalid_window_even(self):
        with pytest.raises(ValueError, match='odd'):
            CovarianceMatrix(window_size=4)

    def test_invalid_window_small(self):
        with pytest.raises(ValueError, match='odd'):
            CovarianceMatrix(window_size=2)


# ---------------------------------------------------------------------------
# CoherencyMatrix
# ---------------------------------------------------------------------------

class TestCoherencyMatrix:

    def test_default_window_size_is_one(self):
        assert CoherencyMatrix().window_size == 1

    def test_t3_shape(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        T = CoherencyMatrix(window_size=3).compute(np.stack([shh, shv, svh, svv], axis=0))
        assert T.shape == (3, 3, *SHAPE)

    def test_t2_shape(self, dual_pol):
        s_co, s_cross = dual_pol
        T = CoherencyMatrix(window_size=3).compute(np.stack([s_co, s_cross], axis=0))
        assert T.shape == (2, 2, *SHAPE)

    def test_t3_hermitian(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        T = CoherencyMatrix(window_size=3).compute(np.stack([shh, shv, svh, svv], axis=0))
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(T[i, j], np.conj(T[j, i]), atol=1e-6)

    def test_t3_diagonal_real(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        T = CoherencyMatrix(window_size=3).compute(np.stack([shh, shv, svh, svv], axis=0))
        for i in range(3):
            np.testing.assert_allclose(T[i, i].imag, 0.0, atol=1e-6)

    def test_trace_equals_c3_trace(self, quad_pol):
        """Under reciprocity trace(T3) = trace(C3) = span."""
        shh, shv, svh, svv = quad_pol
        ws = 3
        chans = np.stack([shh, shv, svh, svv], axis=0)
        C = CovarianceMatrix(ws).compute(chans)
        T = CoherencyMatrix(ws).compute(chans)
        trace_c = sum(C[i, i].real for i in range(3))
        trace_t = sum(T[i, i].real for i in range(3))
        np.testing.assert_allclose(trace_c, trace_t, rtol=1e-4)

    def test_execute_cyx_extraction(self, quad_pol, quad_meta):
        cube = np.stack(quad_pol, axis=0)
        tmat = CoherencyMatrix(window_size=3)
        T, out_meta = tmat.execute(quad_meta, cube)
        assert T.shape == (3, 3, *SHAPE)
        assert out_meta.axis_order == 'CCYX'
        assert out_meta.bands == 9
        assert out_meta.channel_metadata[0].name == 'T[0,0]'

    def test_execute_dual_pol_cyx(self, dual_pol, dual_meta):
        cube = np.stack(dual_pol, axis=0)
        tmat = CoherencyMatrix(window_size=3)
        T, out_meta = tmat.execute(dual_meta, cube)
        assert T.shape == (2, 2, *SHAPE)
        assert out_meta.axis_order == 'CCYX'
        assert out_meta.bands == 4

    def test_repr(self):
        assert repr(CoherencyMatrix(5)) == 'CoherencyMatrix(window_size=5)'

    def test_invalid_window_even(self):
        with pytest.raises(ValueError, match='odd'):
            CoherencyMatrix(window_size=6)


# ---------------------------------------------------------------------------
# Import sanity
# ---------------------------------------------------------------------------

def test_importable_from_decomposition_package():
    from grdl.image_processing.decomposition import (
        CovarianceMatrix, CoherencyMatrix, StokesVector, KennaughMatrix,
    )
    assert CovarianceMatrix is not None
    assert CoherencyMatrix is not None
    assert StokesVector is not None
    assert KennaughMatrix is not None


# ---------------------------------------------------------------------------
# StokesVector
# ---------------------------------------------------------------------------

class TestStokesVector:

    def test_default_window_size_is_one(self):
        assert StokesVector().window_size == 1

    def test_shape_dual_pol(self, dual_pol):
        e_h, e_v = dual_pol
        sv = StokesVector(window_size=3)
        S = sv.compute(e_h, e_v)
        assert S.shape == (4, *SHAPE)

    def test_s0_non_negative(self, dual_pol):
        e_h, e_v = dual_pol
        sv = StokesVector(window_size=3)
        S = sv.compute(e_h, e_v)
        assert np.all(S[0] >= 0)

    def test_dtype_float32(self, dual_pol):
        e_h, e_v = dual_pol
        S = StokesVector(window_size=3).compute(e_h, e_v)
        assert S.dtype == np.float32

    def test_degree_of_polarization_bounds(self, dual_pol):
        e_h, e_v = dual_pol
        sv = StokesVector(window_size=3)
        S = sv.compute(e_h, e_v)
        dop = sv.degree_of_polarization(S)
        assert np.all(dop >= 0)
        assert np.all(dop <= 1.0 + 1e-6)

    def test_fully_polarized_wave(self):
        # Single complex constant in one component → DoP ≈ 1
        ones = np.ones(SHAPE, dtype=np.complex64)
        zeros = np.zeros(SHAPE, dtype=np.complex64)
        sv = StokesVector(window_size=3)
        S = sv.compute(ones, zeros)
        dop = sv.degree_of_polarization(S)
        np.testing.assert_allclose(dop, 1.0, atol=1e-5)

    def test_execute_quad_pol_uses_shh_svv(self, quad_pol, quad_meta):
        shh, shv, svh, svv = quad_pol
        cube = np.stack(quad_pol, axis=0)
        sv = StokesVector(window_size=3)
        S, out_meta = sv.execute(quad_meta, cube)
        # compute(shh, svv) directly → should match execute output
        S_direct = sv.compute(shh, svv)
        np.testing.assert_array_equal(S, S_direct)

    def test_execute_dual_pol(self, dual_pol, dual_meta):
        e_h, e_v = dual_pol
        cube = np.stack(dual_pol, axis=0)
        sv = StokesVector(window_size=3)
        S, out_meta = sv.execute(dual_meta, cube)
        assert S.shape == (4, *SHAPE)

    def test_execute_metadata_axis_order(self, quad_pol, quad_meta):
        cube = np.stack(quad_pol, axis=0)
        _, out_meta = StokesVector(window_size=3).execute(quad_meta, cube)
        assert out_meta.axis_order == 'CYX'
        assert out_meta.bands == 4

    def test_execute_channel_names(self, quad_pol, quad_meta):
        cube = np.stack(quad_pol, axis=0)
        _, out_meta = StokesVector(window_size=3).execute(quad_meta, cube)
        names = [cm.name for cm in out_meta.channel_metadata]
        assert names == ['S0', 'S1', 'S2', 'S3']

    def test_execute_channel_roles(self, quad_pol, quad_meta):
        cube = np.stack(quad_pol, axis=0)
        _, out_meta = StokesVector(window_size=3).execute(quad_meta, cube)
        for cm in out_meta.channel_metadata:
            assert cm.role == 'stokes'

    def test_repr(self):
        assert repr(StokesVector(5)) == 'StokesVector(window_size=5)'

    def test_invalid_window_even(self):
        with pytest.raises(ValueError, match='odd'):
            StokesVector(window_size=4)

    def test_invalid_window_too_small(self):
        with pytest.raises(ValueError):
            StokesVector(window_size=0)

    def test_s2_s3_identity_channel(self):
        # When e_h and e_v are identical: S3 = -2 Im(e*conj(e)) = 0
        e = _complex(99)
        sv = StokesVector(window_size=3)
        S = sv.compute(e, e)
        np.testing.assert_allclose(S[3], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# KennaughMatrix
# ---------------------------------------------------------------------------

class TestKennaughMatrix:

    def test_default_window_size_is_one(self):
        assert KennaughMatrix().window_size == 1

    def test_shape_quad_pol(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        K = KennaughMatrix(window_size=3).compute(np.stack([shh, shv, svh, svv], axis=0))
        assert K.shape == (4, 4, *SHAPE)

    def test_dtype_float32(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        K = KennaughMatrix(window_size=3).compute(np.stack([shh, shv, svh, svv], axis=0))
        assert K.dtype == np.float32

    def test_symmetric(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        K = KennaughMatrix(window_size=3).compute(np.stack([shh, shv, svh, svv], axis=0))
        # K[i,j] == K[j,i] for all pixels
        for i in range(4):
            for j in range(i + 1, 4):
                np.testing.assert_allclose(K[i, j], K[j, i], atol=1e-5)

    def test_diagonal_positive_semidefinite(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        K = KennaughMatrix(window_size=3).compute(np.stack([shh, shv, svh, svv], axis=0))
        for i in range(4):
            assert np.all(K[i, i] >= -1e-5)

    def test_reciprocity_zeroes_fourth_row_col(self):
        # Under perfect reciprocity (SVH = SHV), K[3,:] and K[:,3] should be ~0
        rng = np.random.default_rng(7)
        shh = (rng.standard_normal(SHAPE) + 1j * rng.standard_normal(SHAPE)).astype(np.complex64)
        svv = (rng.standard_normal(SHAPE) + 1j * rng.standard_normal(SHAPE)).astype(np.complex64)
        shv = (rng.standard_normal(SHAPE) + 1j * rng.standard_normal(SHAPE)).astype(np.complex64)
        # svh = shv → perfect reciprocity
        K = KennaughMatrix(window_size=3).compute(np.stack([shh, shv, shv, svv], axis=0))
        # The 4th row / col (index 3) should be near zero
        np.testing.assert_allclose(K[3, :], 0.0, atol=1e-4)
        np.testing.assert_allclose(K[:, 3], 0.0, atol=1e-4)

    def test_surface_scatterer_physics(self):
        # Pure surface: SHH = SVV, SHV = SVH = 0
        # → K[1,1] = K[2,2] = K[3,3] = 0, K[0,0] > 0
        a = (np.ones(SHAPE) + 0j).astype(np.complex64)
        zeros = np.zeros(SHAPE, dtype=np.complex64)
        K = KennaughMatrix(window_size=3).compute(np.stack([a, zeros, zeros, a], axis=0))
        np.testing.assert_allclose(K[1, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(K[2, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(K[3, 3], 0.0, atol=1e-5)
        assert np.all(K[0, 0] > 0)

    def test_reciprocity_default_svh(self, quad_pol):
        # Passing svh=None should behave identically to svh=shv
        shh, shv, _, svv = quad_pol
        K_none = KennaughMatrix(window_size=3).compute(np.stack([shh, shv, svv], axis=0))
        K_expl = KennaughMatrix(window_size=3).compute(np.stack([shh, shv, shv, svv], axis=0))
        np.testing.assert_array_equal(K_none, K_expl)

    def test_missing_svv_raises(self, quad_pol):
        sxx, sxy = quad_pol[0], quad_pol[1]
        with pytest.raises(ValueError, match='channels'):
            KennaughMatrix(window_size=3).compute(np.stack([sxx, sxy], axis=0))

    def test_execute_cyx_extraction(self, quad_pol, quad_meta):
        cube = np.stack(quad_pol, axis=0)
        km = KennaughMatrix(window_size=3)
        K, out_meta = km.execute(quad_meta, cube)
        assert K.shape == (4, 4, *SHAPE)
        assert out_meta.axis_order == 'KKYX'
        assert out_meta.bands == 16

    def test_execute_channel_names(self, quad_pol, quad_meta):
        cube = np.stack(quad_pol, axis=0)
        _, out_meta = KennaughMatrix(window_size=3).execute(quad_meta, cube)
        assert out_meta.channel_metadata[0].name == 'K[0,0]'
        assert out_meta.channel_metadata[5].name == 'K[1,1]'

    def test_repr(self):
        assert repr(KennaughMatrix(5)) == 'KennaughMatrix(window_size=5)'

    def test_invalid_window_even(self):
        with pytest.raises(ValueError, match='odd'):
            KennaughMatrix(window_size=6)
