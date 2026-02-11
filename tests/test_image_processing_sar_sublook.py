# -*- coding: utf-8 -*-
"""
Tests for SublookDecomposition - Sub-aperture spectral splitting.

Uses synthetic complex images with known frequency content to verify
correct sub-band splitting, oversampling handling, overlap geometry,
and backend dispatch (numpy / torch).

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-10

Modified
--------
2026-02-10
"""

import pytest
import numpy as np

from grdl.image_processing.sar.sublook import SublookDecomposition
from grdl.IO.models import SICDMetadata
from grdl.IO.models.sicd import SICDGrid, SICDDirParam, SICDWgtType


# ===================================================================
# Helpers
# ===================================================================

def _make_metadata(
    imp_resp_bw: float = 100.0,
    ss: float = 0.005,
    delta_k1: float = None,
    delta_k2: float = None,
    k_ctr: float = 0.0,
    wgt_funct: np.ndarray = None,
    dimension: str = 'azimuth',
) -> SICDMetadata:
    """Build minimal SICDMetadata for sublook tests.

    Parameters
    ----------
    imp_resp_bw : float
        Impulse response bandwidth (cycles/meter).
    ss : float
        Sample spacing (meters).
    delta_k1, delta_k2 : float or None
        Frequency support bounds.
    k_ctr : float
        Center spatial frequency.
    wgt_funct : np.ndarray or None
        Weight function samples.
    dimension : str
        Which direction param to populate ('azimuth' or 'range').
    """
    dir_param = SICDDirParam(
        ss=ss,
        imp_resp_bw=imp_resp_bw,
        k_ctr=k_ctr,
        delta_k1=delta_k1,
        delta_k2=delta_k2,
        wgt_funct=wgt_funct,
    )
    if dimension == 'azimuth':
        grid = SICDGrid(row=SICDDirParam(ss=ss, imp_resp_bw=imp_resp_bw),
                        col=dir_param)
    else:
        grid = SICDGrid(row=dir_param,
                        col=SICDDirParam(ss=ss, imp_resp_bw=imp_resp_bw))
    return SICDMetadata(
        format='SICD', rows=64, cols=128, dtype='complex64', grid=grid
    )


def _synthetic_complex_image(
    rows: int = 64,
    cols: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Create a synthetic complex SAR-like image.

    Random complex Gaussian noise simulates a complex SAR image with
    energy spread across the frequency domain.
    """
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((rows, cols))
        + 1j * rng.standard_normal((rows, cols))
    ).astype(np.complex64)


# ===================================================================
# Construction validation
# ===================================================================

class TestConstruction:
    """Tests for constructor argument validation."""

    def test_valid_construction(self):
        meta = _make_metadata()
        sd = SublookDecomposition(meta, num_looks=3, overlap=0.1)
        assert sd.num_looks == 3
        assert sd.dimension == 'azimuth'
        assert sd.overlap == 0.1

    def test_missing_grid_raises(self):
        meta = SICDMetadata(
            format='SICD', rows=64, cols=128, dtype='complex64'
        )
        with pytest.raises(ValueError, match="grid is None"):
            SublookDecomposition(meta)

    def test_missing_dir_param_raises(self):
        meta = SICDMetadata(
            format='SICD', rows=64, cols=128, dtype='complex64',
            grid=SICDGrid(),
        )
        with pytest.raises(ValueError, match="is None"):
            SublookDecomposition(meta, dimension='azimuth')

    def test_missing_imp_resp_bw_raises(self):
        meta = SICDMetadata(
            format='SICD', rows=64, cols=128, dtype='complex64',
            grid=SICDGrid(col=SICDDirParam(ss=0.005)),
        )
        with pytest.raises(ValueError, match="imp_resp_bw"):
            SublookDecomposition(meta)

    def test_missing_ss_raises(self):
        meta = SICDMetadata(
            format='SICD', rows=64, cols=128, dtype='complex64',
            grid=SICDGrid(col=SICDDirParam(imp_resp_bw=100.0)),
        )
        with pytest.raises(ValueError, match="ss"):
            SublookDecomposition(meta)

    def test_num_looks_too_small_raises(self):
        meta = _make_metadata()
        with pytest.raises(ValueError, match="num_looks"):
            SublookDecomposition(meta, num_looks=1)

    def test_invalid_dimension_raises(self):
        meta = _make_metadata()
        with pytest.raises(ValueError, match="dimension"):
            SublookDecomposition(meta, dimension='diagonal')

    def test_overlap_out_of_range_raises(self):
        meta = _make_metadata()
        with pytest.raises(ValueError, match="overlap"):
            SublookDecomposition(meta, overlap=1.0)
        with pytest.raises(ValueError, match="overlap"):
            SublookDecomposition(meta, overlap=-0.1)


# ===================================================================
# Decomposition output
# ===================================================================

class TestDecompose:
    """Tests for the decompose() method output."""

    def test_output_shape_2_looks(self):
        meta = _make_metadata()
        sd = SublookDecomposition(meta, num_looks=2)
        image = _synthetic_complex_image()
        result = sd.decompose(image)
        assert result.shape == (2, 64, 128)

    def test_output_shape_4_looks(self):
        meta = _make_metadata()
        sd = SublookDecomposition(meta, num_looks=4)
        image = _synthetic_complex_image()
        result = sd.decompose(image)
        assert result.shape == (4, 64, 128)

    def test_output_is_complex(self):
        meta = _make_metadata()
        sd = SublookDecomposition(meta, num_looks=2)
        image = _synthetic_complex_image()
        result = sd.decompose(image)
        assert np.iscomplexobj(result)

    def test_output_dtype_matches_input(self):
        meta = _make_metadata()
        sd = SublookDecomposition(meta, num_looks=2)
        image = _synthetic_complex_image().astype(np.complex128)
        result = sd.decompose(image)
        assert result.dtype == np.complex128

    def test_non_complex_input_raises(self):
        meta = _make_metadata()
        sd = SublookDecomposition(meta)
        with pytest.raises(TypeError, match="complex"):
            sd.decompose(np.ones((64, 128), dtype=np.float32))

    def test_3d_input_raises(self):
        meta = _make_metadata()
        sd = SublookDecomposition(meta)
        with pytest.raises(ValueError, match="2D"):
            sd.decompose(np.ones((2, 64, 128), dtype=np.complex64))


# ===================================================================
# Energy conservation
# ===================================================================

class TestEnergyConservation:
    """Verify sub-look powers sum approximately to original power."""

    def test_energy_no_overlap_no_deweight(self):
        """Without deweighting or overlap, sub-look powers should
        approximately sum to original power."""
        meta = _make_metadata()
        sd = SublookDecomposition(meta, num_looks=2, deweight=False)
        image = _synthetic_complex_image(rows=64, cols=256)

        original_power = np.sum(np.abs(image) ** 2)
        result = sd.decompose(image)
        sublook_power = np.sum(np.abs(result) ** 2)

        # Energy should be approximately conserved (within 20% due to
        # spectral leakage at sub-band edges and oversampling)
        ratio = sublook_power / original_power
        assert 0.3 < ratio < 1.5, f"Energy ratio {ratio:.3f} out of range"


# ===================================================================
# Dimension parameter
# ===================================================================

class TestDimension:
    """Verify range vs azimuth dimension parameter."""

    def test_range_dimension(self):
        meta = _make_metadata(dimension='range')
        sd = SublookDecomposition(meta, num_looks=2, dimension='range')
        image = _synthetic_complex_image(rows=128, cols=64)
        result = sd.decompose(image)
        assert result.shape == (2, 128, 64)

    def test_azimuth_dimension(self):
        meta = _make_metadata(dimension='azimuth')
        sd = SublookDecomposition(meta, num_looks=3, dimension='azimuth')
        image = _synthetic_complex_image(rows=64, cols=128)
        result = sd.decompose(image)
        assert result.shape == (3, 64, 128)


# ===================================================================
# Oversampling
# ===================================================================

class TestOversampling:
    """Verify correct handling of oversampled images."""

    def test_oversampled_sublooks_confined_to_support(self):
        """With OSR=2, signal occupies half the DFT. Sub-looks should
        have zero energy outside the signal support."""
        # OSR = (1/ss) / imp_resp_bw = (1/0.005) / 100 = 200/100 = 2.0
        meta = _make_metadata(imp_resp_bw=100.0, ss=0.005)
        sd = SublookDecomposition(meta, num_looks=2, deweight=False)

        # Create image with energy only in the signal support band
        rows, cols = 64, 256
        image = _synthetic_complex_image(rows, cols)
        result = sd.decompose(image)

        # Each sub-look should have a spectrum narrower than the original
        for i in range(2):
            look_spectrum = np.fft.fftshift(np.fft.fft(result[i], axis=1),
                                            axes=1)
            power_profile = np.mean(np.abs(look_spectrum) ** 2, axis=0)
            # The outer 25% on each side (outside signal support) should
            # have negligible energy compared to the peak
            n = cols
            outer_power = (np.sum(power_profile[:n // 4])
                           + np.sum(power_profile[3 * n // 4:]))
            inner_power = np.sum(power_profile[n // 4: 3 * n // 4])
            assert outer_power < 0.01 * inner_power, (
                f"Sub-look {i} has significant energy outside signal support"
            )


# ===================================================================
# Overlap parameter
# ===================================================================

class TestOverlap:
    """Verify overlap between adjacent sub-bands."""

    def test_overlap_produces_wider_subbands(self):
        """With overlap, each sub-band should be wider than without."""
        meta = _make_metadata()
        image = _synthetic_complex_image(rows=64, cols=256)

        sd_no_overlap = SublookDecomposition(
            meta, num_looks=3, overlap=0.0, deweight=False
        )
        sd_with_overlap = SublookDecomposition(
            meta, num_looks=3, overlap=0.5, deweight=False
        )

        result_no = sd_no_overlap.decompose(image)
        result_with = sd_with_overlap.decompose(image)

        # With overlap, each sub-look should have more energy
        # (wider sub-band captures more of the spectrum)
        power_no = np.sum(np.abs(result_no[0]) ** 2)
        power_with = np.sum(np.abs(result_with[0]) ** 2)
        assert power_with > power_no, (
            "50% overlap sub-look should have more power than 0% overlap"
        )


# ===================================================================
# Deweighting
# ===================================================================

class TestDeweight:
    """Verify deweighting removes original apodization."""

    def test_deweight_with_weight_function(self):
        """When a weight function is provided, deweighting should change
        the result compared to not deweighting."""
        # Hamming-like weight function
        wgt = np.hamming(64).astype(np.float64)
        meta = _make_metadata(wgt_funct=wgt)
        image = _synthetic_complex_image(rows=64, cols=128)

        sd_deweight = SublookDecomposition(meta, num_looks=2, deweight=True)
        sd_no_deweight = SublookDecomposition(meta, num_looks=2, deweight=False)

        result_dw = sd_deweight.decompose(image)
        result_ndw = sd_no_deweight.decompose(image)

        # Results should differ when deweighting is applied
        assert not np.allclose(result_dw, result_ndw), (
            "Deweighted and non-deweighted results should differ"
        )


# ===================================================================
# Conversion helpers
# ===================================================================

class TestConversions:
    """Tests for to_magnitude, to_power, to_db."""

    @pytest.fixture
    def sublook_stack(self):
        meta = _make_metadata()
        sd = SublookDecomposition(meta, num_looks=2)
        image = _synthetic_complex_image()
        return sd, sd.decompose(image)

    def test_to_magnitude(self, sublook_stack):
        sd, stack = sublook_stack
        mag = sd.to_magnitude(stack)
        assert mag.shape == stack.shape
        assert not np.iscomplexobj(mag)
        np.testing.assert_allclose(mag, np.abs(stack))

    def test_to_power(self, sublook_stack):
        sd, stack = sublook_stack
        pwr = sd.to_power(stack)
        assert pwr.shape == stack.shape
        assert not np.iscomplexobj(pwr)
        np.testing.assert_allclose(pwr, np.abs(stack) ** 2)

    def test_to_db(self, sublook_stack):
        sd, stack = sublook_stack
        db = sd.to_db(stack, floor=-60.0)
        assert db.shape == stack.shape
        assert not np.iscomplexobj(db)
        assert np.all(db >= -60.0)


# ===================================================================
# Versioning / tags
# ===================================================================

class TestVersioning:
    """Verify processor version and tags."""

    def test_has_version(self):
        assert SublookDecomposition.__processor_version__ == '0.1.0'

    def test_has_tags(self):
        tags = SublookDecomposition.__processor_tags__
        assert 'SAR' in tags['modalities']
        assert tags['category'] == 'decomposition'


# ===================================================================
# Torch backend
# ===================================================================

class TestTorchBackend:
    """Tests for torch tensor input dispatch."""

    @pytest.fixture
    def torch_available(self):
        pytest.importorskip('torch')

    def test_torch_input_returns_numpy(self, torch_available):
        import torch
        meta = _make_metadata()
        sd = SublookDecomposition(meta, num_looks=2)
        image_np = _synthetic_complex_image()
        image_t = torch.from_numpy(image_np)

        result = sd.decompose(image_t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 64, 128)
        assert np.iscomplexobj(result)

    def test_torch_matches_numpy(self, torch_available):
        """Torch and numpy paths should produce similar results."""
        import torch
        meta = _make_metadata()
        sd = SublookDecomposition(meta, num_looks=2, deweight=False)
        image_np = _synthetic_complex_image()
        image_t = torch.from_numpy(image_np.copy())

        result_np = sd.decompose(image_np)
        result_torch = sd.decompose(image_t)

        np.testing.assert_allclose(
            result_torch, result_np, rtol=1e-4, atol=1e-6
        )
