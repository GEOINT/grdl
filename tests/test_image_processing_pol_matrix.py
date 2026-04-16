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

from grdl.IO.models.base import ChannelMetadata, ImageMetadata
from grdl.image_processing.decomposition.pol_matrix import (
    CovarianceMatrix,
    CoherencyMatrix,
    StokesVector,
    KennaughMatrix,
)


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

    def test_c3_shape(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        C = CovarianceMatrix(window_size=3).compute(shh, shv, svh, svv)
        assert C.shape == (3, 3, *SHAPE)

    def test_c2_shape(self, dual_pol):
        s_co, s_cross = dual_pol
        C = CovarianceMatrix(window_size=3).compute(s_co, s_cross)
        assert C.shape == (2, 2, *SHAPE)

    def test_c3_hermitian(self, quad_pol):
        """C[i,j] = conj(C[j,i]) everywhere."""
        shh, shv, svh, svv = quad_pol
        C = CovarianceMatrix(window_size=3).compute(shh, shv, svh, svv)
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(C[i, j], np.conj(C[j, i]), atol=1e-6)

    def test_c3_diagonal_real(self, quad_pol):
        """Diagonal elements should be real (zero imaginary part)."""
        shh, shv, svh, svv = quad_pol
        C = CovarianceMatrix(window_size=3).compute(shh, shv, svh, svv)
        for i in range(3):
            np.testing.assert_allclose(C[i, i].imag, 0.0, atol=1e-6)

    def test_c3_diagonal_nonnegative(self, quad_pol):
        """Power elements must be >= 0."""
        shh, shv, svh, svv = quad_pol
        C = CovarianceMatrix(window_size=3).compute(shh, shv, svh, svv)
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

    def test_t3_shape(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        T = CoherencyMatrix(window_size=3).compute(shh, shv, svh, svv)
        assert T.shape == (3, 3, *SHAPE)

    def test_t2_shape(self, dual_pol):
        s_co, s_cross = dual_pol
        T = CoherencyMatrix(window_size=3).compute(s_co, s_cross)
        assert T.shape == (2, 2, *SHAPE)

    def test_t3_hermitian(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        T = CoherencyMatrix(window_size=3).compute(shh, shv, svh, svv)
        for i in range(3):
            for j in range(3):
                np.testing.assert_allclose(T[i, j], np.conj(T[j, i]), atol=1e-6)

    def test_t3_diagonal_real(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        T = CoherencyMatrix(window_size=3).compute(shh, shv, svh, svv)
        for i in range(3):
            np.testing.assert_allclose(T[i, i].imag, 0.0, atol=1e-6)

    def test_trace_equals_c3_trace(self, quad_pol):
        """Under reciprocity trace(T3) = trace(C3) = span."""
        shh, shv, svh, svv = quad_pol
        ws = 3
        C = CovarianceMatrix(ws).compute(shh, shv, svh, svv)
        T = CoherencyMatrix(ws).compute(shh, shv, svh, svv)
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
            StokesVector(window_size=1)

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

    def test_shape_quad_pol(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        K = KennaughMatrix(window_size=3).compute(shh, shv, svh, svv)
        assert K.shape == (4, 4, *SHAPE)

    def test_dtype_float32(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        K = KennaughMatrix(window_size=3).compute(shh, shv, svh, svv)
        assert K.dtype == np.float32

    def test_symmetric(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        K = KennaughMatrix(window_size=3).compute(shh, shv, svh, svv)
        # K[i,j] == K[j,i] for all pixels
        for i in range(4):
            for j in range(i + 1, 4):
                np.testing.assert_allclose(K[i, j], K[j, i], atol=1e-5)

    def test_diagonal_positive_semidefinite(self, quad_pol):
        shh, shv, svh, svv = quad_pol
        K = KennaughMatrix(window_size=3).compute(shh, shv, svh, svv)
        for i in range(4):
            assert np.all(K[i, i] >= -1e-5)

    def test_reciprocity_zeroes_fourth_row_col(self):
        # Under perfect reciprocity (SVH = SHV), K[3,:] and K[:,3] should be ~0
        rng = np.random.default_rng(7)
        shh = (rng.standard_normal(SHAPE) + 1j * rng.standard_normal(SHAPE)).astype(np.complex64)
        svv = (rng.standard_normal(SHAPE) + 1j * rng.standard_normal(SHAPE)).astype(np.complex64)
        shv = (rng.standard_normal(SHAPE) + 1j * rng.standard_normal(SHAPE)).astype(np.complex64)
        # svh = shv → perfect reciprocity
        K = KennaughMatrix(window_size=3).compute(shh, shv, shv, svv)
        # The 4th row / col (index 3) should be near zero
        np.testing.assert_allclose(K[3, :], 0.0, atol=1e-4)
        np.testing.assert_allclose(K[:, 3], 0.0, atol=1e-4)

    def test_surface_scatterer_physics(self):
        # Pure surface: SHH = SVV, SHV = SVH = 0
        # → K[1,1] = K[2,2] = K[3,3] = 0, K[0,0] > 0
        a = (np.ones(SHAPE) + 0j).astype(np.complex64)
        zeros = np.zeros(SHAPE, dtype=np.complex64)
        K = KennaughMatrix(window_size=3).compute(a, zeros, zeros, a)
        np.testing.assert_allclose(K[1, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(K[2, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(K[3, 3], 0.0, atol=1e-5)
        assert np.all(K[0, 0] > 0)

    def test_reciprocity_default_svh(self, quad_pol):
        # Passing svh=None should behave identically to svh=shv
        shh, shv, _, svv = quad_pol
        K_none = KennaughMatrix(window_size=3).compute(shh, shv, None, svv)
        K_expl = KennaughMatrix(window_size=3).compute(shh, shv, shv, svv)
        np.testing.assert_array_equal(K_none, K_expl)

    def test_missing_svv_raises(self, quad_pol):
        shh, shv, svh, _ = quad_pol
        with pytest.raises(ValueError, match='svv'):
            KennaughMatrix(window_size=3).compute(shh, shv, svh, svv=None)

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
