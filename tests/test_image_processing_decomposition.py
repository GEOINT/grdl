# -*- coding: utf-8 -*-
"""
Polarimetric Decomposition Tests.

Tests for the PolarimetricDecomposition ABC and PauliDecomposition
concrete class. All tests use synthetic complex arrays -- no real
imagery required.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-01-30

Modified
--------
2026-01-30
"""

import numpy as np
import pytest

from grdl.IO.models.base import ChannelMetadata, ImageMetadata
from grdl.IO.models.polarization import (
    channel_axis,
    resolve_packed_c2_indices,
    resolve_raw_indices,
)
from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.decomposition.pauli import PauliDecomposition
from grdl.image_processing.decomposition.pol_matrix import CoherencyMatrix, CovarianceMatrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pauli():
    """PauliDecomposition instance."""
    return PauliDecomposition()


@pytest.fixture
def quad_pol_random():
    """Random 64x64 quad-pol complex arrays."""
    rng = np.random.default_rng(42)
    shape = (64, 64)

    def _random_complex(s):
        return (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(
            np.complex128
        )

    shh = _random_complex(shape)
    shv = _random_complex(shape) * 0.3
    svh = shv + _random_complex(shape) * 0.01  # near-reciprocal
    svv = _random_complex(shape)
    return shh, shv, svh, svv


@pytest.fixture
def quad_pol_known():
    """Single-pixel arrays with analytically verifiable values.

    S_HH = 3+0j, S_HV = 1+0j, S_VH = 1+0j, S_VV = 1+0j

    Expected Pauli components:
        surface       = (3 + 1) / sqrt(2) = 4/sqrt(2) = 2*sqrt(2)
        double_bounce = (3 - 1) / sqrt(2) = 2/sqrt(2) = sqrt(2)
        volume        = (1 + 1) / sqrt(2) = 2/sqrt(2) = sqrt(2)
    """
    shh = np.array([[3.0 + 0j]], dtype=np.complex128)
    shv = np.array([[1.0 + 0j]], dtype=np.complex128)
    svh = np.array([[1.0 + 0j]], dtype=np.complex128)
    svv = np.array([[1.0 + 0j]], dtype=np.complex128)
    return shh, shv, svh, svv


@pytest.fixture
def quad_pol_phased():
    """Arrays with non-trivial phase for interference testing.

    S_HH = 1+0j (0 deg), S_VV = 0+1j (90 deg)
    The 90-degree phase offset between HH and VV should produce
    partial constructive/destructive interference in surface and
    double_bounce components.
    """
    shh = np.array([[1.0 + 0j]], dtype=np.complex128)
    shv = np.array([[0.5 + 0.5j]], dtype=np.complex128)
    svh = np.array([[0.5 + 0.5j]], dtype=np.complex128)
    svv = np.array([[0.0 + 1.0j]], dtype=np.complex128)
    return shh, shv, svh, svv


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------

class TestABCContract:
    """Verify PauliDecomposition satisfies the ABC contract."""

    def test_execute_incomplete_explicit_fails_before_decompose(self, pauli, monkeypatch):
        def unexpected_decompose(*args):
            pytest.fail('Incomplete explicit input reached numerical decomposition')

        monkeypatch.setattr(pauli, 'decompose', unexpected_decompose)
        metadata = ImageMetadata(format='test', rows=8, cols=8, bands=4,
                                 dtype='complex64', axis_order='CYX')
        with pytest.raises(ValueError, match='Explicit full-pol'):
            pauli.execute(metadata, np.ones((4, 8, 8), dtype=np.complex64),
                          shv=np.ones((8, 8), dtype=np.complex64))

    def test_is_subclass(self):
        assert issubclass(PauliDecomposition, PolarimetricDecomposition)

    def test_instance(self, pauli):
        assert isinstance(pauli, PolarimetricDecomposition)

    def test_component_names(self, pauli):
        names = pauli.component_names
        assert isinstance(names, tuple)
        assert len(names) == 3
        assert set(names) == {'surface', 'double_bounce', 'volume'}

    def test_repr(self, pauli):
        assert repr(pauli) == "PauliDecomposition(window_size=1)"

    def test_execute_sets_pauli_lineage(self, pauli):
        meta = ImageMetadata(
            format='test', rows=8, cols=8, dtype='complex64',
            bands=4, axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=0, name='HH', polarization='HH'),
                ChannelMetadata(index=1, name='HV', polarization='HV'),
                ChannelMetadata(index=2, name='VH', polarization='VH'),
                ChannelMetadata(index=3, name='VV', polarization='VV'),
            ],
        )
        source = np.zeros((4, 8, 8), dtype=np.complex64)
        source[0] = 1
        source[1] = 2
        source[2] = 3
        source[3] = 4
        _, out_meta = pauli.execute(meta, source)
        assert out_meta.bands == 3
        assert out_meta.channel_metadata[0].name == 'surface'
        assert out_meta.channel_metadata[0].source_indices == [0, 3]
        assert out_meta.channel_metadata[2].name == 'volume'
        assert out_meta.channel_metadata[2].source_indices == [1, 2]


# ---------------------------------------------------------------------------
# Decomposition correctness
# ---------------------------------------------------------------------------

class TestDecompose:
    """Test decompose() output correctness."""

    def test_known_values(self, pauli, quad_pol_known):
        shh, shv, svh, svv = quad_pol_known
        c = pauli.decompose(shh, shv, svh, svv)

        sqrt2 = np.sqrt(2.0)
        assert c['surface'][0, 0] == pytest.approx(2 * sqrt2, abs=1e-10)
        assert c['double_bounce'][0, 0] == pytest.approx(sqrt2, abs=1e-10)
        assert c['volume'][0, 0] == pytest.approx(sqrt2, abs=1e-10)

    def test_output_is_complex(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        for name in pauli.component_names:
            assert np.iscomplexobj(c[name]), f"{name} should be complex"

    def test_output_shape(self, pauli, quad_pol_random):
        shh = quad_pol_random[0]
        c = pauli.decompose(*quad_pol_random)
        for name in pauli.component_names:
            assert c[name].shape == shh.shape

    def test_dict_keys(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        assert set(c.keys()) == {'surface', 'double_bounce', 'volume'}

    def test_normalization_factor(self, pauli, quad_pol_known):
        """Verify the 1/sqrt(2) normalization is applied."""
        shh, shv, svh, svv = quad_pol_known

        c = pauli.decompose(shh, shv, svh, svv)

        # Without normalization, surface would be (3+1) = 4
        # With normalization, surface = 4/sqrt(2) = 2*sqrt(2) ~ 2.828
        assert abs(c['surface'][0, 0]) < 4.0
        assert abs(c['surface'][0, 0]) == pytest.approx(
            4.0 / np.sqrt(2.0), abs=1e-10
        )

    def test_reciprocity_power_conservation(self, pauli):
        """Under reciprocity (S_HV == S_VH), total Pauli power == span."""
        rng = np.random.default_rng(123)
        shape = (32, 32)

        def _cplx(s):
            return (rng.standard_normal(s) + 1j * rng.standard_normal(s))

        shh = _cplx(shape)
        shv = _cplx(shape)
        svh = shv.copy()  # exact reciprocity
        svv = _cplx(shape)

        c = pauli.decompose(shh, shv, svh, svv)
        pauli_power = (
            np.abs(c['surface']) ** 2
            + np.abs(c['double_bounce']) ** 2
            + np.abs(c['volume']) ** 2
        )
        span = (
            np.abs(shh) ** 2
            + np.abs(svv) ** 2
            + 2 * np.abs(shv) ** 2
        )
        np.testing.assert_allclose(pauli_power, span, rtol=1e-10)

    def test_monostatic_volume_simplification(self, pauli):
        """Under reciprocity, volume = sqrt(2) * S_HV."""
        rng = np.random.default_rng(456)
        shape = (16, 16)
        shh = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        shv = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        svh = shv.copy()
        svv = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

        c = pauli.decompose(shh, shv, svh, svv)
        expected_volume = np.sqrt(2.0) * shv
        np.testing.assert_allclose(c['volume'], expected_volume, rtol=1e-10)

    def test_window_size_boxcar_runs(self, quad_pol_random):
        pauli = PauliDecomposition(window_size=7)
        c = pauli.decompose(*quad_pol_random)
        assert set(c.keys()) == {'surface', 'double_bounce', 'volume'}
        assert c['surface'].shape == quad_pol_random[0].shape

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError, match=r'odd integer in \[1, 31\]'):
            PauliDecomposition(window_size=2)

    def test_decompose_from_t3_returns_real_magnitude(self, quad_pol_random):
        shh, shv, svh, svv = quad_pol_random
        t3 = CoherencyMatrix(window_size=7).compute(
            np.stack([shh, shv, svh, svv], axis=0)
        )

        c = PauliDecomposition().decompose_from_t3(t3)
        assert set(c.keys()) == {'surface', 'double_bounce', 'volume'}
        for name in c:
            assert not np.iscomplexobj(c[name])
            assert np.nanmin(c[name]) >= 0.0

    def test_decompose_from_c3_returns_real_magnitude(self, quad_pol_random):
        shh, shv, svh, svv = quad_pol_random
        c3 = CovarianceMatrix(window_size=7).compute(
            np.stack([shh, shv, svh, svv], axis=0)
        )

        c = PauliDecomposition().decompose_from_c3(c3)
        assert set(c.keys()) == {'surface', 'double_bounce', 'volume'}
        for name in c:
            assert not np.iscomplexobj(c[name])
            assert np.nanmin(c[name]) >= 0.0

    # --- monostatic / single cross-pol tests --------------------------------

    def test_monostatic_shv_only_matches_reciprocal(self, pauli):
        """decompose(shh, shv=x, svv=svv) matches decompose(shh, x, x, svv)."""
        rng = np.random.default_rng(7)
        shape = (16, 16)
        shh = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        shv = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        svv = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

        c_mono = pauli.decompose(shh, shv=shv, svv=svv)
        c_full = pauli.decompose(shh, shv, shv, svv)  # svh == shv

        np.testing.assert_allclose(c_mono['surface'], c_full['surface'])
        np.testing.assert_allclose(c_mono['double_bounce'], c_full['double_bounce'])
        np.testing.assert_allclose(c_mono['volume'], c_full['volume'])

    def test_monostatic_svh_only_matches_reciprocal(self, pauli):
        """decompose(shh, svh=x, svv=svv) matches decompose(shh, x, x, svv)."""
        rng = np.random.default_rng(8)
        shape = (16, 16)
        shh = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        svh = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        svv = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

        c_mono = pauli.decompose(shh, svh=svh, svv=svv)
        c_full = pauli.decompose(shh, svh, svh, svv)  # shv == svh

        np.testing.assert_allclose(c_mono['surface'], c_full['surface'])
        np.testing.assert_allclose(c_mono['double_bounce'], c_full['double_bounce'])
        np.testing.assert_allclose(c_mono['volume'], c_full['volume'])

    def test_monostatic_volume_equals_sqrt2_times_shv(self, pauli):
        """volume = sqrt(2) * shv when only shv is provided."""
        rng = np.random.default_rng(9)
        shape = (8, 8)
        shh = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        shv = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        svv = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

        c = pauli.decompose(shh, shv=shv, svv=svv)
        np.testing.assert_allclose(c['volume'], np.sqrt(2.0) * shv, rtol=1e-10)

    def test_monostatic_missing_both_cross_pol_raises(self, pauli):
        """Omitting both shv and svh must raise ValueError."""
        rng = np.random.default_rng(10)
        shape = (8, 8)
        shh = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        svv = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

        with pytest.raises(ValueError, match="At least one cross-pol"):
            pauli.decompose(shh, svv=svv)

    def test_monostatic_output_shape_matches_input(self, pauli):
        """Monostatic path preserves shape."""
        rng = np.random.default_rng(11)
        shape = (20, 30)
        shh = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        shv = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        svv = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

        c = pauli.decompose(shh, shv=shv, svv=svv)
        for name in pauli.component_names:
            assert c[name].shape == shape


# ---------------------------------------------------------------------------
# Phase preservation
# ---------------------------------------------------------------------------

class TestPhasePreservation:
    """Verify complex mixing and phase are correctly handled."""

    def test_phase_in_surface_component(self, pauli, quad_pol_phased):
        """S_HH=1+0j, S_VV=0+1j → surface has non-trivial phase."""
        shh, shv, svh, svv = quad_pol_phased
        c = pauli.decompose(shh, shv, svh, svv)

        surface = c['surface'][0, 0]
        # (1+0j) + (0+1j) = 1+1j → phase = 45 deg
        expected = (1.0 + 1.0j) / np.sqrt(2.0)
        assert surface == pytest.approx(expected, abs=1e-10)

    def test_phase_in_double_bounce(self, pauli, quad_pol_phased):
        """S_HH=1+0j, S_VV=0+1j → double_bounce has different phase."""
        shh, shv, svh, svv = quad_pol_phased
        c = pauli.decompose(shh, shv, svh, svv)

        db = c['double_bounce'][0, 0]
        # (1+0j) - (0+1j) = 1-1j → phase = -45 deg
        expected = (1.0 - 1.0j) / np.sqrt(2.0)
        assert db == pytest.approx(expected, abs=1e-10)

    def test_surface_and_double_bounce_magnitudes_equal(
        self, pauli, quad_pol_phased
    ):
        """When |S_HH| == |S_VV|, surface and double_bounce have equal
        magnitude (only the phase differs)."""
        shh, shv, svh, svv = quad_pol_phased
        c = pauli.decompose(shh, shv, svh, svv)

        assert abs(c['surface'][0, 0]) == pytest.approx(
            abs(c['double_bounce'][0, 0]), abs=1e-10
        )

    def test_constructive_interference(self, pauli):
        """HH and VV in phase → large surface, small double_bounce."""
        shh = np.array([[1.0 + 0j]], dtype=np.complex128)
        svv = np.array([[1.0 + 0j]], dtype=np.complex128)
        shv = np.array([[0.1 + 0j]], dtype=np.complex128)
        svh = np.array([[0.1 + 0j]], dtype=np.complex128)

        c = pauli.decompose(shh, shv, svh, svv)

        # HH + VV = 2 (constructive), HH - VV = 0 (destructive)
        assert abs(c['surface'][0, 0]) == pytest.approx(
            2.0 / np.sqrt(2.0), abs=1e-10
        )
        assert abs(c['double_bounce'][0, 0]) == pytest.approx(0.0, abs=1e-10)

    def test_destructive_interference(self, pauli):
        """HH and VV 180 deg out of phase → small surface, large double_bounce."""
        shh = np.array([[1.0 + 0j]], dtype=np.complex128)
        svv = np.array([[-1.0 + 0j]], dtype=np.complex128)
        shv = np.array([[0.1 + 0j]], dtype=np.complex128)
        svh = np.array([[0.1 + 0j]], dtype=np.complex128)

        c = pauli.decompose(shh, shv, svh, svv)

        # HH + VV = 0 (destructive), HH - VV = 2 (constructive)
        assert abs(c['surface'][0, 0]) == pytest.approx(0.0, abs=1e-10)
        assert abs(c['double_bounce'][0, 0]) == pytest.approx(
            2.0 / np.sqrt(2.0), abs=1e-10
        )

    def test_phase_angles_preserved(self, pauli):
        """Output phase angles match expected complex arithmetic."""
        # Pure imaginary S_HH, pure real S_VV
        shh = np.array([[0.0 + 2.0j]], dtype=np.complex128)
        svv = np.array([[2.0 + 0.0j]], dtype=np.complex128)
        shv = np.array([[0.0 + 0.0j]], dtype=np.complex128)
        svh = np.array([[0.0 + 0.0j]], dtype=np.complex128)

        c = pauli.decompose(shh, shv, svh, svv)

        # surface = (2j + 2) / sqrt(2) → phase = 45 deg
        surface_phase = np.angle(c['surface'][0, 0])
        assert surface_phase == pytest.approx(np.pi / 4, abs=1e-10)

        # double_bounce = (2j - 2) / sqrt(2) → phase = 135 deg
        db_phase = np.angle(c['double_bounce'][0, 0])
        assert db_phase == pytest.approx(3 * np.pi / 4, abs=1e-10)


# ---------------------------------------------------------------------------
# dtype preservation
# ---------------------------------------------------------------------------

class TestDtypePreservation:
    """Verify input dtype is preserved through computation."""

    def test_complex128_preserved(self, pauli):
        shape = (8, 8)
        arr = np.ones(shape, dtype=np.complex128)
        c = pauli.decompose(arr, arr, arr, arr)
        for name in pauli.component_names:
            assert c[name].dtype == np.complex128

    def test_complex64_preserved(self, pauli):
        shape = (8, 8)
        arr = np.ones(shape, dtype=np.complex64)
        c = pauli.decompose(arr, arr, arr, arr)
        for name in pauli.component_names:
            assert c[name].dtype == np.complex64


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Verify fail-fast input validation."""

    def test_real_input_raises_type_error(self, pauli):
        arr = np.ones((8, 8), dtype=np.float64)
        cplx = np.ones((8, 8), dtype=np.complex128)
        with pytest.raises(TypeError, match="complex"):
            pauli.decompose(arr, cplx, cplx, cplx)

    def test_non_array_raises_type_error(self, pauli):
        cplx = np.ones((8, 8), dtype=np.complex128)
        with pytest.raises(TypeError, match="ndarray"):
            pauli.decompose([[1 + 0j]], cplx, cplx, cplx)

    def test_1d_raises_value_error(self, pauli):
        arr = np.ones(8, dtype=np.complex128)
        with pytest.raises(ValueError, match="2D"):
            pauli.decompose(arr, arr, arr, arr)

    def test_3d_raises_value_error(self, pauli):
        arr = np.ones((2, 4, 4), dtype=np.complex128)
        with pytest.raises(ValueError, match="2D"):
            pauli.decompose(arr, arr, arr, arr)

    def test_shape_mismatch_raises_value_error(self, pauli):
        a = np.ones((8, 8), dtype=np.complex128)
        b = np.ones((8, 10), dtype=np.complex128)
        with pytest.raises(ValueError, match="same shape"):
            pauli.decompose(a, a, a, b)


# ---------------------------------------------------------------------------
# Conversion methods
# ---------------------------------------------------------------------------

class TestConversions:
    """Test to_power, to_magnitude, to_db, to_rgb."""

    def test_to_power_non_negative(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        pwr = pauli.to_power(c)
        for name in pauli.component_names:
            assert np.all(pwr[name] >= 0)
            assert not np.iscomplexobj(pwr[name])

    def test_to_power_values(self, pauli, quad_pol_known):
        c = pauli.decompose(*quad_pol_known)
        pwr = pauli.to_power(c)
        # surface power = |2*sqrt(2)|^2 = 8
        assert pwr['surface'][0, 0] == pytest.approx(8.0, abs=1e-10)
        # double_bounce power = |sqrt(2)|^2 = 2
        assert pwr['double_bounce'][0, 0] == pytest.approx(2.0, abs=1e-10)

    def test_to_magnitude_non_negative(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        mag = pauli.to_magnitude(c)
        for name in pauli.component_names:
            assert np.all(mag[name] >= 0)
            assert not np.iscomplexobj(mag[name])

    def test_to_db_finite(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        db = pauli.to_db(c)
        for name in pauli.component_names:
            assert np.all(np.isfinite(db[name]))
            assert not np.iscomplexobj(db[name])

    def test_to_db_floor(self, pauli):
        """Verify dB floor is applied."""
        arr = np.array([[1e-20 + 0j]], dtype=np.complex128)
        c = pauli.decompose(arr, arr, arr, arr)
        db = pauli.to_db(c, floor=-40.0)
        for name in pauli.component_names:
            assert db[name][0, 0] >= -40.0

    def test_to_rgb_shape(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        rgb, meta = pauli.to_rgb(c)
        rows, cols = quad_pol_random[0].shape
        assert rgb.shape == (3, rows, cols)
        assert meta.rows == rows
        assert meta.cols == cols
        assert meta.bands == 3

    def test_to_rgb_range(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        rgb, _ = pauli.to_rgb(c)
        assert np.all(rgb >= 0.0)
        assert np.all(rgb <= 1.0)

    def test_to_rgb_dtype(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        rgb, meta = pauli.to_rgb(c)
        assert rgb.dtype == np.float32
        assert meta.dtype == 'float32'

    def test_to_rgb_representations(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        for rep in ('db', 'magnitude', 'power'):
            rgb, _ = pauli.to_rgb(c, representation=rep)
            assert rgb.shape[0] == 3
            assert np.all(rgb >= 0.0)
            assert np.all(rgb <= 1.0)

    def test_to_rgb_invalid_representation(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        with pytest.raises(ValueError, match="representation"):
            pauli.to_rgb(c, representation='invalid')

    def test_to_rgb_missing_keys(self, pauli):
        with pytest.raises(ValueError, match="Missing"):
            pauli.to_rgb({'surface': np.zeros((2, 2))})


def _phase2_metadata(labels, *, order: str | None = 'CYX', rows=7, cols=9):
    return ImageMetadata(
        format='test', rows=rows, cols=cols, dtype='complex64', bands=len(labels),
        axis_order=order,
        channel_metadata=[ChannelMetadata(index=index, name=label)
                          for index, label in enumerate(labels)],
    )


@pytest.mark.parametrize('basis', ['HV', 'XY', 'RL'])
def test_phase2_shuffled_raw_indices(basis):
    first, second = basis
    labels = [second * 2, second + first, first * 2, first + second]
    meta = _phase2_metadata(labels)
    assert resolve_raw_indices(4, meta, 'full') == (2, 3, 1, 0)
    assert resolve_raw_indices(4, meta, 'matrix') == (2, 3, 1, 0)
    assert resolve_raw_indices(4, meta, 'dual') == (2, 3)
    assert resolve_raw_indices(4, meta, 'stokes') == (2, 0)


def test_phase2_named_role_order():
    meta = _phase2_metadata(['band_0', 'band_1', 'band_2', 'band_3'])
    assert meta.channel_metadata is not None
    for channel, role in zip(meta.channel_metadata,
                             ['co_pol_b', 'cross_pol_2', 'co_pol_a', 'cross_pol'], strict=True):
        channel.role = role
    assert resolve_raw_indices(4, meta, 'full') == (2, 3, 1, 0)


@pytest.mark.parametrize('labels, error', [
    (['HH', 'unknown', 'VH', 'VV'], 'Missing'),
    (['HH', 'HV', 'HV', 'VV'], 'Duplicate'),
    (['HH', 'HV', 'LR', 'LL'], 'Mixed'),
    (['C11', 'C12_real', 'C12_imag', 'C22'], 'Packed'),
])
def test_phase2_invalid_raw_descriptors(labels, error):
    with pytest.raises(ValueError, match=error):
        resolve_raw_indices(4, _phase2_metadata(labels), 'full')


@pytest.mark.parametrize('field, value', [
    ('polarization', 'HV'), ('role', 'cross_pol'),
])
def test_phase2_conflicting_descriptor(field, value):
    meta = _phase2_metadata(['HH', 'HV', 'VH', 'VV'])
    assert meta.channel_metadata is not None
    setattr(meta.channel_metadata[0], field, value)
    with pytest.raises(ValueError, match='Conflicting'):
        resolve_raw_indices(4, meta, 'full')


def test_phase2_txrx_and_explicit_polarization():
    meta = _phase2_metadata(['band_0', 'band_1'])
    assert meta.channel_metadata is not None
    meta.channel_metadata[0].polarization = 'RHC:RHC'
    meta.channel_metadata[1].tx_polarization = 'RHC'
    meta.channel_metadata[1].rcv_polarization = 'LHC'
    assert resolve_raw_indices(2, meta, 'dual') == (0, 1)
    meta.channel_metadata[1].polarization = 'LR'
    with pytest.raises(ValueError, match='Conflicting'):
        resolve_raw_indices(2, meta)


@pytest.mark.parametrize('count', [2, 3, 4])
def test_phase2_canonical_unlabelled(count):
    meta = _phase2_metadata([f'band_{index}' for index in range(count)])
    assert resolve_raw_indices(count, meta) == tuple(range(count))
    assert resolve_raw_indices(count, meta, 'stokes') == (0, count - 1)


def test_phase2_surplus_requires_labels():
    meta = _phase2_metadata(['VV', 'VH', 'noise', 'HH', 'HV'])
    assert resolve_raw_indices(5, meta, 'full') == (3, 4, 1, 0)
    meta.channel_metadata = None
    with pytest.raises(ValueError, match='canonical'):
        resolve_raw_indices(5, meta, 'full')


def test_phase2_reciprocal_and_dual_requirements():
    meta = _phase2_metadata(['VV', 'VH', 'HH'])
    assert resolve_raw_indices(3, meta) == (2, 1, 0)
    assert resolve_raw_indices(3, meta, 'dual') == (2, 1)
    assert resolve_raw_indices(3, meta, 'stokes') == (2, 0)
    with pytest.raises(ValueError, match='Missing'):
        resolve_raw_indices(3, meta, 'full')
    meta = _phase2_metadata(['HV', 'HH'])
    assert resolve_raw_indices(2, meta) == (1, 0)


@pytest.mark.parametrize('order', ['CYX', 'YXC', None])
def test_phase2_execute_reorders_views_and_lineage(order, pauli):
    meta = _phase2_metadata(['VV', 'VH', 'HH', 'HV'], order=order)
    cube = np.broadcast_to(np.array([4, 3, 1, 2], dtype=np.complex128)[:, None, None],
                           (4, 7, 9)).copy()
    source = cube.transpose(1, 2, 0) if order == 'YXC' else cube
    before = source.copy()
    result, updated = pauli.execute(meta, source)
    expected = pauli.decompose(cube[2], cube[3], cube[1], cube[0])
    for name in result:
        np.testing.assert_array_equal(result[name], expected[name])
    np.testing.assert_array_equal(source, before)
    assert updated.channel_metadata[0].source_indices == [2, 0]
    assert updated.channel_metadata[2].source_indices == [3, 1]
    assert updated.dtype == 'complex128'
    assert updated.axis_order == 'CYX'


def test_phase2_explicit_kwargs_bypass_source_shape(pauli):
    meta = _phase2_metadata(['HH', 'HV', 'VH', 'VV'], rows=4, cols=4, order=None)
    field = np.ones((7, 9), dtype=np.complex128)
    _, updated = pauli.execute(meta, np.empty((4, 4, 4)),
                              shh=field, shv=field, svh=field, svv=field)
    assert (updated.rows, updated.cols) == field.shape
    assert all(channel.source_indices == [] for channel in updated.channel_metadata)


def test_phase2_layout_ambiguity_and_consistency():
    meta = _phase2_metadata(['HH', 'HV', 'VH', 'VV'], rows=4, cols=4, order=None)
    with pytest.raises(ValueError, match='Ambiguous'):
        channel_axis((4, 4, 4), meta)
    meta.axis_order = 'YXC'
    assert channel_axis((4, 4, 4), meta) == 2
    meta.bands = 3
    with pytest.raises(ValueError, match='inconsistent'):
        channel_axis((4, 4, 4), meta)


def test_phase2_packed_c2_independent_descriptors():
    meta = _phase2_metadata(['C22', 'Im(C12)', 'C11', 'Re(C12)'])
    assert resolve_packed_c2_indices(4, meta) == (2, 3, 1, 0)
    with pytest.raises(ValueError, match='Packed'):
        resolve_raw_indices(4, meta)
    with pytest.raises(ValueError, match='Raw'):
        resolve_packed_c2_indices(4, _phase2_metadata(['HH', 'HV', 'VH', 'VV']))


@pytest.mark.parametrize('basis', ['HV', 'XY', 'RL'])
def test_phase2_single_field_stokes_pair(basis):
    meta = _phase2_metadata(list(basis[::-1]))
    assert resolve_raw_indices(2, meta, 'stokes') == (1, 0)
    assert meta.channel_metadata is not None
    for channel in meta.channel_metadata:
        channel.polarization = channel.name
        channel.name = 'measurement'
    assert resolve_raw_indices(2, meta, 'stokes') == (1, 0)
    with pytest.raises(ValueError, match='Single-field'):
        resolve_raw_indices(2, meta, 'dual')


def test_phase2_scattering_aliases():
    meta = _phase2_metadata(['S_VV', 'SVH', 'S_HH', 'SHV'])
    assert resolve_raw_indices(4, meta, 'full') == (2, 3, 1, 0)


@pytest.mark.parametrize('role', ['covariance', 'coherency'])
def test_phase2_matrix_roles_never_raw_fallback(role):
    meta = _phase2_metadata(['band_0', 'band_1', 'band_2', 'band_3'])
    assert meta.channel_metadata is not None
    for channel in meta.channel_metadata:
        channel.role = role
    with pytest.raises(ValueError, match='Packed'):
        resolve_raw_indices(4, meta)


def test_phase2_duplicate_named_roles():
    meta = _phase2_metadata(['band_0', 'band_1', 'band_2', 'band_3'])
    assert meta.channel_metadata is not None
    for channel, role in zip(meta.channel_metadata,
                             ['co_pol_a', 'co_pol_a', 'cross_pol', 'co_pol_b'], strict=True):
        channel.role = role
    with pytest.raises(ValueError, match='Duplicate'):
        resolve_raw_indices(4, meta)


@pytest.mark.parametrize('labels, error', [
    (['C11', 'C12_real', 'unknown', 'C22'], 'Missing'),
    (['C11', 'C12_real', 'C12_real', 'C22'], 'Duplicate'),
    (['T11', 'T12_real', 'T12_imag', 'T22'], 'Unsupported'),
])
def test_phase2_invalid_packed_descriptors(labels, error):
    with pytest.raises(ValueError, match=error):
        resolve_packed_c2_indices(4, _phase2_metadata(labels))


def test_phase2_layout_infers_from_spatial_shape():
    meta = _phase2_metadata(['HH', 'HV', 'VH', 'VV'], order=None, rows=4, cols=9)
    assert channel_axis((4, 9, 4), meta) == 2
    meta.bands = None
    meta.channel_metadata = None
    assert channel_axis((4, 4, 9), meta) == 0
    with pytest.raises(ValueError, match='inconsistent'):
        channel_axis((4, 8, 9), meta)


def test_phase2_execute_requires_actual_full_channels(pauli):
    meta = _phase2_metadata(['HH', 'HV', 'VV'])
    with pytest.raises(ValueError, match='Missing'):
        pauli.execute(meta, np.ones((3, 7, 9), dtype=np.complex64))


def test_phase2_execute_selects_labelled_surplus(pauli):
    meta = _phase2_metadata(['VV', 'VH', 'noise', 'HH', 'HV'])
    cube = np.ones((5, 7, 9), dtype=np.complex64)
    _, updated = pauli.execute(meta, cube)
    assert updated.channel_metadata[0].source_indices == [3, 0]
    assert updated.channel_metadata[2].source_indices == [4, 1]
