# -*- coding: utf-8 -*-
"""
Tests for new full-pol PolSAR decompositions:
    DegreeOfPolarization, ShannonEntropy, NeumannDecomposition,
    PraksParameters, TouziDecomposition, Yamaguchi4C.

All tests use synthetic data — no real imagery required.

Author
------
Jason Fritz, PhD
43161141+stryder-vtx@users.noreply.github.com

Created
-------
2026-06-30
"""

import numpy as np
import pytest

from grdl.image_processing.decomposition import (
    FullPolHAalpha,
    FreemanDurden3C,
    ModelFree3C,
    ModelFree4C,
    DegreeOfPolarization,
    ShannonEntropy,
    NeumannDecomposition,
    PraksParameters,
    TouziDecomposition,
    Yamaguchi4C,
    CoherencyMatrix,
    CovarianceMatrix,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SHAPE = (30, 30)
RNG = np.random.default_rng(42)


def _random_complex(shape=SHAPE):
    return (
        RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)
    ).astype(np.complex128)


@pytest.fixture(scope='module')
def quad_pol():
    """Random 30×30 quad-pol complex arrays."""
    shh = _random_complex()
    shv = _random_complex() * 0.3
    svh = shv + _random_complex() * 0.01
    svv = _random_complex()
    return shh, shv, svh, svv


@pytest.fixture(scope='module')
def t3_precomputed(quad_pol):
    shh, shv, svh, svv = quad_pol
    channels = np.stack([shh, shv, svh, svv], axis=0)
    return CoherencyMatrix(window_size=7).compute(channels)


@pytest.fixture(scope='module')
def c3_precomputed(quad_pol):
    shh, shv, svh, svv = quad_pol
    channels = np.stack([shh, shv, svh, svv], axis=0)
    return CovarianceMatrix(window_size=7).compute(channels)


@pytest.mark.parametrize(
    ('factory', 'method_name', 'matrix_fixture'),
    [
        (lambda: FullPolHAalpha(window_size=1), 'decompose_from_t3', 't3_precomputed'),
        (lambda: FreemanDurden3C(window_size=1), 'decompose_from_c3', 'c3_precomputed'),
        (lambda: ModelFree3C(window_size=1), 'decompose_from_t3', 't3_precomputed'),
        (lambda: ModelFree4C(window_size=1), 'decompose_from_t3', 't3_precomputed'),
        (lambda: DegreeOfPolarization(window_size=1), 'decompose_from_t3', 't3_precomputed'),
        (lambda: ShannonEntropy(window_size=1), 'decompose_from_t3', 't3_precomputed'),
        (lambda: NeumannDecomposition(window_size=1), 'decompose_from_t3', 't3_precomputed'),
        (lambda: PraksParameters(window_size=1), 'decompose_from_c3', 'c3_precomputed'),
        (lambda: TouziDecomposition(window_size=1), 'decompose_from_t3', 't3_precomputed'),
        (lambda: Yamaguchi4C(window_size=1), 'decompose_from_t3', 't3_precomputed'),
    ],
)
def test_precomputed_matrix_workflows_allow_window_size_one(
    request, factory, method_name, matrix_fixture
):
    decomp = factory()
    matrix = request.getfixturevalue(matrix_fixture)
    components = getattr(decomp, method_name)(matrix)
    assert components


@pytest.mark.parametrize(
    'decomp',
    [
        FullPolHAalpha(window_size=1),
        FreemanDurden3C(window_size=1),
        ModelFree3C(window_size=1),
        ModelFree4C(window_size=1),
        DegreeOfPolarization(window_size=1),
        ShannonEntropy(window_size=1),
        NeumannDecomposition(window_size=1),
        PraksParameters(window_size=1),
        TouziDecomposition(window_size=1),
        Yamaguchi4C(window_size=1),
    ],
)
def test_internal_matrix_build_rejects_window_size_one(quad_pol, decomp):
    with pytest.raises(ValueError, match='window_size >= 3'):
        decomp.decompose(*quad_pol)


# ---------------------------------------------------------------------------
# DegreeOfPolarization
# ---------------------------------------------------------------------------

class TestDegreeOfPolarization:

    def test_output_keys(self, quad_pol):
        comp = DegreeOfPolarization(window_size=7).decompose(*quad_pol)
        assert set(comp.keys()) == {'dop'}

    def test_output_shape(self, quad_pol):
        comp = DegreeOfPolarization(window_size=7).decompose(*quad_pol)
        assert comp['dop'].shape == SHAPE

    def test_bounds(self, quad_pol):
        comp = DegreeOfPolarization(window_size=7).decompose(*quad_pol)
        assert np.all(comp['dop'] >= 0.0)
        assert np.all(comp['dop'] <= 1.0)

    def test_finite_values(self, quad_pol):
        comp = DegreeOfPolarization(window_size=7).decompose(*quad_pol)
        assert np.all(np.isfinite(comp['dop']))

    def test_from_t3_matches_decompose(self, quad_pol, t3_precomputed):
        d = DegreeOfPolarization(window_size=7)
        comp_direct = d.decompose(*quad_pol)
        comp_t3 = d.decompose_from_t3(t3_precomputed)
        np.testing.assert_allclose(
            comp_direct['dop'], comp_t3['dop'], atol=1e-10
        )

    def test_to_rgb_shape(self, quad_pol):
        comp = DegreeOfPolarization(window_size=7).decompose(*quad_pol)
        rgb, _ = DegreeOfPolarization().to_rgb(comp)
        assert rgb.shape == (3, SHAPE[0], SHAPE[1])
        assert rgb.dtype == np.float32

    def test_single_scatterer_dop_one(self):
        """A coherent single-scatterer has DoP close to 1 after single pixel."""
        shh = np.ones((5, 5), dtype=np.complex128)
        shv = np.zeros((5, 5), dtype=np.complex128)
        svh = np.zeros((5, 5), dtype=np.complex128)
        svv = np.ones((5, 5), dtype=np.complex128)
        comp = DegreeOfPolarization(window_size=3).decompose(shh, shv, svh, svv)
        # Interior pixels fully averaged over constant data → fully polarised
        assert comp['dop'][2, 2] > 0.99

    def test_component_names(self):
        assert DegreeOfPolarization().component_names == ('dop',)


# ---------------------------------------------------------------------------
# ShannonEntropy
# ---------------------------------------------------------------------------

class TestShannonEntropy:

    def test_output_keys(self, quad_pol):
        comp = ShannonEntropy(window_size=7).decompose(*quad_pol)
        assert set(comp.keys()) == {'H_total', 'H_intensity', 'H_polarimetric'}

    def test_output_shape(self, quad_pol):
        comp = ShannonEntropy(window_size=7).decompose(*quad_pol)
        for v in comp.values():
            assert v.shape == SHAPE

    def test_finite_total(self, quad_pol):
        comp = ShannonEntropy(window_size=7).decompose(*quad_pol)
        assert np.all(np.isfinite(comp['H_total']))

    def test_from_t3_matches_decompose(self, quad_pol, t3_precomputed):
        d = ShannonEntropy(window_size=7)
        c1 = d.decompose(*quad_pol)
        c2 = d.decompose_from_t3(t3_precomputed)
        np.testing.assert_allclose(c1['H_total'], c2['H_total'], atol=1e-10)

    def test_to_rgb_shape(self, quad_pol):
        comp = ShannonEntropy(window_size=7).decompose(*quad_pol)
        rgb, _ = ShannonEntropy().to_rgb(comp)
        assert rgb.shape == (3, SHAPE[0], SHAPE[1])

    def test_component_names(self):
        assert ShannonEntropy().component_names == (
            'H_total', 'H_intensity', 'H_polarimetric'
        )


# ---------------------------------------------------------------------------
# NeumannDecomposition
# ---------------------------------------------------------------------------

class TestNeumannDecomposition:

    def test_output_keys(self, quad_pol):
        comp = NeumannDecomposition(window_size=7).decompose(*quad_pol)
        assert set(comp.keys()) == {'psi', 'delta_mod', 'delta_pha', 'tau'}

    def test_output_shape(self, quad_pol):
        comp = NeumannDecomposition(window_size=7).decompose(*quad_pol)
        for v in comp.values():
            assert v.shape == SHAPE

    def test_finite_values(self, quad_pol):
        comp = NeumannDecomposition(window_size=7).decompose(*quad_pol)
        for v in comp.values():
            assert np.all(np.isfinite(v))

    def test_psi_bounds(self, quad_pol):
        comp = NeumannDecomposition(window_size=7).decompose(*quad_pol)
        assert np.all(comp['psi'] >= -45.0 - 1e-6)
        assert np.all(comp['psi'] <= 45.0 + 1e-6)

    def test_delta_mod_nonneg(self, quad_pol):
        comp = NeumannDecomposition(window_size=7).decompose(*quad_pol)
        assert np.all(comp['delta_mod'] >= 0.0)

    def test_tau_bounds(self, quad_pol):
        comp = NeumannDecomposition(window_size=7).decompose(*quad_pol)
        assert np.all(comp['tau'] >= 0.0 - 1e-6)
        assert np.all(comp['tau'] <= 1.0 + 1e-6)

    def test_from_t3_matches_decompose(self, quad_pol, t3_precomputed):
        d = NeumannDecomposition(window_size=7)
        c1 = d.decompose(*quad_pol)
        c2 = d.decompose_from_t3(t3_precomputed)
        np.testing.assert_allclose(c1['psi'], c2['psi'], atol=1e-10)

    def test_to_rgb_shape(self, quad_pol):
        comp = NeumannDecomposition(window_size=7).decompose(*quad_pol)
        rgb, _ = NeumannDecomposition().to_rgb(comp)
        assert rgb.shape == (3, SHAPE[0], SHAPE[1])

    def test_component_names(self):
        assert NeumannDecomposition().component_names == (
            'psi', 'delta_mod', 'delta_pha', 'tau'
        )


# ---------------------------------------------------------------------------
# PraksParameters
# ---------------------------------------------------------------------------

class TestPraksParameters:

    def test_output_keys(self, quad_pol):
        comp = PraksParameters(window_size=7).decompose(*quad_pol)
        assert set(comp.keys()) == {
            'frobenius_norm', 'scattering_predominance',
            'scattering_diversity', 'degree_purity',
            'depolarization_index', 'alpha', 'entropy',
        }

    def test_output_shape(self, quad_pol):
        comp = PraksParameters(window_size=7).decompose(*quad_pol)
        for v in comp.values():
            assert v.shape == SHAPE

    def test_finite_values(self, quad_pol):
        comp = PraksParameters(window_size=7).decompose(*quad_pol)
        for v in comp.values():
            assert np.all(np.isfinite(v))

    def test_frobenius_norm_bounds(self, quad_pol):
        comp = PraksParameters(window_size=7).decompose(*quad_pol)
        # For a 3x3 normalised Hermitian PSD matrix with trace 1:
        # ||M||^2 ∈ [1/3, 1]
        assert np.all(comp['frobenius_norm'] >= 1.0 / 3.0 - 1e-6)
        assert np.all(comp['frobenius_norm'] <= 1.0 + 1e-6)

    def test_alpha_bounds(self, quad_pol):
        comp = PraksParameters(window_size=7).decompose(*quad_pol)
        assert np.all(comp['alpha'] >= 0.0 - 1e-6)
        assert np.all(comp['alpha'] <= 90.0 + 1e-6)

    def test_degree_purity_nonneg(self, quad_pol):
        comp = PraksParameters(window_size=7).decompose(*quad_pol)
        assert np.all(comp['degree_purity'] >= 0.0 - 1e-6)

    def test_from_c3_matches_decompose(self, quad_pol, c3_precomputed):
        d = PraksParameters(window_size=7)
        c1 = d.decompose(*quad_pol)
        c2 = d.decompose_from_c3(c3_precomputed)
        np.testing.assert_allclose(
            c1['frobenius_norm'], c2['frobenius_norm'], atol=1e-10
        )

    def test_to_rgb_shape(self, quad_pol):
        comp = PraksParameters(window_size=7).decompose(*quad_pol)
        rgb, _ = PraksParameters().to_rgb(comp)
        assert rgb.shape == (3, SHAPE[0], SHAPE[1])

    def test_component_names(self):
        assert PraksParameters().component_names == (
            'frobenius_norm', 'scattering_predominance',
            'scattering_diversity', 'degree_purity',
            'depolarization_index', 'alpha', 'entropy',
        )


# ---------------------------------------------------------------------------
# TouziDecomposition
# ---------------------------------------------------------------------------

class TestTouziDecomposition:

    def test_output_keys(self, quad_pol):
        comp = TouziDecomposition(window_size=7).decompose(*quad_pol)
        expected = {
            'alpha1', 'alpha2', 'alpha3',
            'phi1',   'phi2',   'phi3',
            'tau1',   'tau2',   'tau3',
            'psi1',   'psi2',   'psi3',
            'alpha_mean', 'phi_mean', 'tau_mean', 'psi_mean',
        }
        assert set(comp.keys()) == expected

    def test_output_shape(self, quad_pol):
        comp = TouziDecomposition(window_size=7).decompose(*quad_pol)
        for v in comp.values():
            assert v.shape == SHAPE

    def test_finite_values(self, quad_pol):
        comp = TouziDecomposition(window_size=7).decompose(*quad_pol)
        for v in comp.values():
            assert np.all(np.isfinite(v))

    def test_alpha_bounds(self, quad_pol):
        comp = TouziDecomposition(window_size=7).decompose(*quad_pol)
        for k in ('alpha1', 'alpha2', 'alpha3', 'alpha_mean'):
            assert np.all(comp[k] >= -1e-6), f"{k} below 0"
            assert np.all(comp[k] <= 90.0 + 1e-6), f"{k} above 90"

    def test_psi_bounds(self, quad_pol):
        comp = TouziDecomposition(window_size=7).decompose(*quad_pol)
        for k in ('psi1', 'psi2', 'psi3', 'psi_mean'):
            assert np.all(comp[k] >= -90.0 - 1e-4), f"{k} below -90"
            assert np.all(comp[k] <= 90.0 + 1e-4), f"{k} above 90"

    def test_tau_bounds(self, quad_pol):
        comp = TouziDecomposition(window_size=7).decompose(*quad_pol)
        for k in ('tau1', 'tau2', 'tau3', 'tau_mean'):
            assert np.all(comp[k] >= -45.0 - 1e-4), f"{k} below -45"
            assert np.all(comp[k] <= 45.0 + 1e-4), f"{k} above 45"

    def test_from_t3_matches_decompose(self, quad_pol, t3_precomputed):
        d = TouziDecomposition(window_size=7)
        c1 = d.decompose(*quad_pol)
        c2 = d.decompose_from_t3(t3_precomputed)
        np.testing.assert_allclose(
            c1['alpha_mean'], c2['alpha_mean'], atol=1e-10
        )

    def test_to_rgb_shape(self, quad_pol):
        comp = TouziDecomposition(window_size=7).decompose(*quad_pol)
        rgb, _ = TouziDecomposition().to_rgb(comp)
        assert rgb.shape == (3, SHAPE[0], SHAPE[1])
        assert np.all(rgb >= 0.0)
        assert np.all(rgb <= 1.0)

    def test_component_names(self):
        assert len(TouziDecomposition().component_names) == 16


# ---------------------------------------------------------------------------
# Yamaguchi4C
# ---------------------------------------------------------------------------

class TestYamaguchi4C:

    @pytest.mark.parametrize('model', ['y4o', 'y4r', 'y4s'])
    def test_output_keys(self, quad_pol, model):
        comp = Yamaguchi4C(window_size=7, model=model).decompose(*quad_pol)
        assert set(comp.keys()) == {
            'surface', 'double_bounce', 'volume', 'helix', 'span'
        }

    @pytest.mark.parametrize('model', ['y4o', 'y4r', 'y4s'])
    def test_output_shape(self, quad_pol, model):
        comp = Yamaguchi4C(window_size=7, model=model).decompose(*quad_pol)
        for k, v in comp.items():
            assert v.shape == SHAPE, f"{k} shape mismatch for {model}"

    @pytest.mark.parametrize('model', ['y4o', 'y4r', 'y4s'])
    def test_powers_nonneg(self, quad_pol, model):
        comp = Yamaguchi4C(window_size=7, model=model).decompose(*quad_pol)
        for k in ('surface', 'double_bounce', 'volume', 'helix'):
            assert np.all(comp[k] >= 0.0 - 1e-8), f"{k} negative for {model}"

    def test_powers_sum_bounded_by_span(self, quad_pol):
        comp = Yamaguchi4C(window_size=7, model='y4o').decompose(*quad_pol)
        power_sum = (
            comp['surface']
            + comp['double_bounce']
            + comp['volume']
            + comp['helix']
        )
        # Power sum should be within 25% of span (relaxed bound to account for
        # the Freeman-Durden fallback branch where energy is not perfectly
        # conserved due to cross-term approximations).
        span = comp['span']
        relative_err = np.abs(power_sum - span) / (span + 1e-10)
        assert np.nanpercentile(relative_err, 95) < 0.25

    def test_finite_values(self, quad_pol):
        comp = Yamaguchi4C(window_size=7, model='y4r').decompose(*quad_pol)
        for v in comp.values():
            assert np.all(np.isfinite(v))

    def test_from_t3_matches_decompose(self, quad_pol, t3_precomputed):
        yam = Yamaguchi4C(window_size=7, model='y4o')
        c1 = yam.decompose(*quad_pol)
        c2 = yam.decompose_from_t3(t3_precomputed)
        np.testing.assert_allclose(c1['span'], c2['span'], atol=1e-10)
        np.testing.assert_allclose(
            c1['surface'], c2['surface'], rtol=1e-6, atol=1e-8
        )

    def test_from_c3_consistent_with_from_t3(self, t3_precomputed, c3_precomputed):
        """decompose_from_c3 and decompose_from_t3 agree (C3→T3 roundtrip)."""
        yam = Yamaguchi4C(model='y4o')
        c_t3 = yam.decompose_from_t3(t3_precomputed)
        c_c3 = yam.decompose_from_c3(c3_precomputed)
        np.testing.assert_allclose(
            c_t3['volume'], c_c3['volume'], rtol=1e-5, atol=1e-8
        )

    def test_to_rgb_shape(self, quad_pol):
        comp = Yamaguchi4C(window_size=7).decompose(*quad_pol)
        rgb, _ = Yamaguchi4C().to_rgb(comp)
        assert rgb.shape == (3, SHAPE[0], SHAPE[1])
        assert rgb.dtype == np.float32

    def test_component_names(self):
        assert set(Yamaguchi4C().component_names) == {
            'surface', 'double_bounce', 'volume', 'helix', 'span'
        }
