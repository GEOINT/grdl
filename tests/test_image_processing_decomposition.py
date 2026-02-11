# -*- coding: utf-8 -*-
"""
Polarimetric Decomposition Tests.

Tests for the PolarimetricDecomposition ABC and PauliDecomposition
concrete class. All tests use synthetic complex arrays -- no real
imagery required.

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
2026-01-30

Modified
--------
2026-01-30
"""

import numpy as np
import pytest

from grdl.image_processing.decomposition.base import PolarimetricDecomposition
from grdl.image_processing.decomposition.pauli import PauliDecomposition


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
        assert repr(pauli) == "PauliDecomposition()"


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
        rgb = pauli.to_rgb(c)
        rows, cols = quad_pol_random[0].shape
        assert rgb.shape == (rows, cols, 3)

    def test_to_rgb_range(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        rgb = pauli.to_rgb(c)
        assert np.all(rgb >= 0.0)
        assert np.all(rgb <= 1.0)

    def test_to_rgb_dtype(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        rgb = pauli.to_rgb(c)
        assert rgb.dtype == np.float32

    def test_to_rgb_representations(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        for rep in ('db', 'magnitude', 'power'):
            rgb = pauli.to_rgb(c, representation=rep)
            assert rgb.shape[2] == 3
            assert np.all(rgb >= 0.0)
            assert np.all(rgb <= 1.0)

    def test_to_rgb_invalid_representation(self, pauli, quad_pol_random):
        c = pauli.decompose(*quad_pol_random)
        with pytest.raises(ValueError, match="representation"):
            pauli.to_rgb(c, representation='invalid')

    def test_to_rgb_missing_keys(self, pauli):
        with pytest.raises(ValueError, match="Missing"):
            pauli.to_rgb({'surface': np.zeros((2, 2))})
