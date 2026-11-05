# -*- coding: utf-8 -*-
"""Tests for compact-pol decomposition processors."""

import dataclasses

import numpy as np
import pytest

from grdl.IO.models.base import ChannelMetadata, ImageMetadata
from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.decomposition.dop_cp import CompactPolDegreeOfPolarization
from grdl.image_processing.decomposition.model_free_cp import CompactPolModelFree3C
from grdl.image_processing.decomposition.s_omega_cp import CompactPolSOmega
from grdl.image_processing.decomposition.m_chi_cp import CompactPolMChi
from grdl.image_processing.decomposition.m_delta_cp import CompactPolMDelta
from grdl.image_processing.decomposition.pol_matrix import StokesVector


@pytest.mark.parametrize('chi', [-45.0, 0.0, 45.0])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_compact_canonical_stokes(chi, dtype):
    fields = [np.ones((3, 5), dtype=dtype), np.full((3, 5), 1j, dtype=dtype)]
    c11, c22 = np.abs(fields[0]) ** 2, np.abs(fields[1]) ** 2
    c12 = fields[0] * fields[1].conj()
    proc = CompactPolMChi(window_size=1, chi=chi)
    result = proc._stokes_from_c2(c11, c12, c12.conj(), c22, window_size=1)
    np.testing.assert_array_equal(np.stack(result), StokesVector(1).compute(*fields))
    assert all(value.dtype == np.finfo(dtype).dtype for value in result)


@pytest.mark.parametrize('processor,names,negative,nonnegative', [
    (CompactPolMChi, ('surface', 'double_bounce', 'volume', 'm_cp', 'chi_cp'),
     (1.2840106300379734, .4986815597658628, 1.7614293638687224,
      .3794733192202055, -23.774922228554335),
     (.4986815597658628, 1.2840106300379734, 1.7614293638687224,
      .3794733192202055, 23.774922228554335)),
    (CompactPolMDelta, ('surface', 'double_bounce', 'volume', 'm_cp', 'delta_cp'),
     (1.3313042723701018, .35354706966702604, 1.2455186477726803,
      .3794733192202055, 60.25511870305777),
     (.35354706966702604, 1.3313042723701018, 1.2455186477726803,
      .3794733192202055, -60.25511870305777)),
    (CompactPolModelFree3C, ('surface', 'double_bounce', 'volume', 'theta_cp'),
     (1.4470100397602799, .450356556340748, 3.102633403898972, 15.843631736063516),
     (.450356556340748, 1.4470100397602799, 3.102633403898972, -15.843631736063516)),
    (CompactPolSOmega, ('surface', 'double_bounce', 'volume'),
     (2.1264842659301326, .44958338727379543, 2.423932346796072),
     (.44958338727379543, 2.1264842659301326, 2.423932346796072)),
])
@pytest.mark.parametrize('chi', [-45., 0., 45.])
def test_compact_physical_outputs_before_stokes_refactor(processor, names, negative,
                                                        nonnegative, chi):
    """Freeze outputs captured before canonical Stokes/handedness separation."""
    elements = [np.full((3, 5), value) for value in (3., .4 + .7j, .4 - .7j, 2.)]
    result = processor(window_size=1, chi=chi).decompose_compact(*elements)
    expected = negative if chi < 0 else nonnegative
    if processor is CompactPolSOmega and chi == 0:
        expected = (.4494733192202055, 1.9649988441973174, 2.585527836582477)
    assert set(result) == set(names)
    for name, value in zip(names, expected, strict=True):
        np.testing.assert_allclose(result[name], value, rtol=1e-12, atol=1e-12)


@pytest.fixture
def c2_random():
    rng = np.random.default_rng(7)
    shape = (48, 48)
    c11 = np.abs(rng.standard_normal(shape)).astype(np.float32) + 1.0
    c22 = np.abs(rng.standard_normal(shape)).astype(np.float32) + 0.5
    c12 = (0.2 * rng.standard_normal(shape) + 0.2j * rng.standard_normal(shape)).astype(np.complex64)
    c21 = np.conj(c12)
    return c11, c12, c21, c22


@pytest.fixture
def c2_metadata():
    return ImageMetadata(
        format='C2',
        rows=48,
        cols=48,
        bands=4,
        dtype='complex64',
        axis_order='CYX',
        channel_metadata=[
            ChannelMetadata(index=0, name='C11', role='matrix'),
            ChannelMetadata(index=1, name='C12_real', role='matrix'),
            ChannelMetadata(index=2, name='C12_imag', role='matrix'),
            ChannelMetadata(index=3, name='C22', role='matrix'),
        ],
    )


def _cube_from_c2(c2_random):
    c11, c12, _, c22 = c2_random
    return np.stack([c11, c12.real, c12.imag, c22], axis=0)


@pytest.mark.parametrize('axis_order', ['CYX', 'YXC'])
def test_shuffled_packed_c2(c2_random, c2_metadata, axis_order):
    order = [3, 2, 0, 1]
    cube = _cube_from_c2(c2_random)[order]
    metadata = dataclasses.replace(
        c2_metadata, axis_order=axis_order,
        channel_metadata=[dataclasses.replace(c2_metadata.channel_metadata[index], index=slot)
                          for slot, index in enumerate(order)],
    )
    if axis_order == 'YXC':
        cube = cube.transpose(1, 2, 0)
    proc = CompactPolMChi(window_size=3)
    result, updated = proc.execute(metadata, cube)
    expected = proc.decompose_compact(*c2_random)
    for name in result:
        np.testing.assert_array_equal(result[name], expected[name])
    assert updated.axis_order == 'CYX'
    assert updated.dtype == str(np.result_type(*(value.dtype for value in result.values())))
    assert [channel.name for channel in updated.channel_metadata] == list(result)
    assert all(channel.source_indices == [2, 3, 1, 0] for channel in updated.channel_metadata)


def test_compact_rejects_raw_labels(c2_random, c2_metadata):
    metadata = dataclasses.replace(c2_metadata, channel_metadata=[
        ChannelMetadata(index=index, name=name)
        for index, name in enumerate(('HH', 'HV', 'VH', 'VV'))
    ])
    with pytest.raises(ValueError, match='Raw scattering'):
        CompactPolMChi().execute(metadata, _cube_from_c2(c2_random))


def test_explicit_compact_bypasses_source(c2_random, c2_metadata):
    metadata = dataclasses.replace(c2_metadata, axis_order='CCYX', rows=1, cols=1)
    proc = CompactPolMChi(window_size=3)
    inputs = dict(zip(('c11', 'c12', 'c21', 'c22'), c2_random, strict=True))
    result, updated = proc.execute(metadata, np.zeros((2, 3, 4)), **inputs)
    expected = proc.decompose_compact(*c2_random)
    for name in result:
        np.testing.assert_array_equal(result[name], expected[name])
    assert updated.rows == 48
    assert all(channel.source_indices == [] for channel in updated.channel_metadata)
    with pytest.raises(ValueError, match='Explicit compact-pol'):
        proc.execute(c2_metadata, _cube_from_c2(c2_random), c22=c2_random[-1])


class TestCompactPolBase:
    def test_inherits_polarimetric_base(self):
        proc = CompactPolDegreeOfPolarization()
        assert isinstance(proc, PolarimetricDecomposition)

    def test_execute_from_cube(self, c2_random, c2_metadata):
        cube = _cube_from_c2(c2_random)
        proc = CompactPolDegreeOfPolarization(window_size=3)
        result, updated = proc.execute(c2_metadata, cube)
        assert set(result.keys()) == {'dop'}
        assert updated.bands == 1

    def test_decompose_bridges_to_compact_interface(self, c2_random):
        proc = CompactPolDegreeOfPolarization(window_size=3)
        result = proc.decompose(*c2_random)
        assert set(result.keys()) == {'dop'}


class TestDegreeOfPolarizationCP:
    def test_bounds(self, c2_random):
        proc = CompactPolDegreeOfPolarization(window_size=3)
        result = proc.decompose_compact(*c2_random)
        dop = result['dop']
        assert dop.shape == c2_random[0].shape
        assert np.all(dop[np.isfinite(dop)] >= 0.0)
        assert np.all(dop[np.isfinite(dop)] <= 1.0)


class TestModelFree3CCP:
    def test_outputs_and_ranges(self, c2_random):
        proc = CompactPolModelFree3C(window_size=3)
        result = proc.decompose_compact(*c2_random)
        assert set(result.keys()) == {'surface', 'double_bounce', 'volume', 'theta_cp'}
        for key in ('surface', 'double_bounce', 'volume'):
            assert result[key].shape == c2_random[0].shape
            assert np.all(result[key][np.isfinite(result[key])] >= 0.0)
        theta = result['theta_cp']
        assert np.all(np.isfinite(theta))


class TestSOmegaCP:
    def test_outputs_and_nonnegative(self, c2_random):
        proc = CompactPolSOmega(window_size=3)
        result = proc.decompose_compact(*c2_random)
        assert set(result.keys()) == {'surface', 'double_bounce', 'volume'}
        for key in result:
            assert np.all(result[key][np.isfinite(result[key])] >= 0.0)


class TestMChiCP:
    def test_outputs(self, c2_random):
        proc = CompactPolMChi(window_size=3)
        result = proc.decompose_compact(*c2_random)
        assert set(result.keys()) == {'surface', 'double_bounce', 'volume', 'm_cp', 'chi_cp'}
        assert np.all(result['m_cp'][np.isfinite(result['m_cp'])] >= 0.0)
        assert np.all(result['m_cp'][np.isfinite(result['m_cp'])] <= 1.0)


class TestMDeltaCP:
    def test_outputs(self, c2_random):
        proc = CompactPolMDelta(window_size=3)
        result = proc.decompose_compact(*c2_random)
        assert set(result.keys()) == {'surface', 'double_bounce', 'volume', 'm_cp', 'delta_cp'}
        assert np.all(result['m_cp'][np.isfinite(result['m_cp'])] >= 0.0)
        assert np.all(result['m_cp'][np.isfinite(result['m_cp'])] <= 1.0)