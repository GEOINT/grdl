# -*- coding: utf-8 -*-
"""Tests for compact-pol decomposition processors."""

import numpy as np
import pytest

from grdl.IO.models.base import ChannelMetadata, ImageMetadata
from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.decomposition.dop_cp import CompactPolDegreeOfPolarization
from grdl.image_processing.decomposition.model_free_cp import CompactPolModelFree3C
from grdl.image_processing.decomposition.s_omega_cp import CompactPolSOmega
from grdl.image_processing.decomposition.m_chi_cp import CompactPolMChi
from grdl.image_processing.decomposition.m_delta_cp import CompactPolMDelta


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