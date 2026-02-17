# -*- coding: utf-8 -*-
"""
Tests for MultilookDecomposition - 2D sub-aperture spectral splitting.

Uses synthetic complex images with known frequency content to verify
correct 2D sub-band splitting, Tiler integration, oversampling handling,
overlap geometry, deweighting, and backend dispatch (numpy / torch).

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
2026-02-17

Modified
--------
2026-02-17
"""

import pytest
import numpy as np

from grdl.image_processing.sar.multilook import MultilookDecomposition
from grdl.IO.models import SICDMetadata
from grdl.IO.models.sicd import SICDGrid, SICDDirParam


# ===================================================================
# Helpers
# ===================================================================

def _make_metadata(
    rg_imp_resp_bw: float = 100.0,
    rg_ss: float = 0.005,
    rg_delta_k1: float = None,
    rg_delta_k2: float = None,
    rg_wgt_funct: np.ndarray = None,
    az_imp_resp_bw: float = 80.0,
    az_ss: float = 0.008,
    az_delta_k1: float = None,
    az_delta_k2: float = None,
    az_wgt_funct: np.ndarray = None,
) -> SICDMetadata:
    """Build minimal SICDMetadata for multilook tests.

    Default parameters:
      - Range: imp_resp_bw=100, ss=0.005 => OSR = (1/0.005)/100 = 2.0
      - Azimuth: imp_resp_bw=80, ss=0.008 => OSR = (1/0.008)/80 = 1.5625
    """
    row_param = SICDDirParam(
        ss=rg_ss,
        imp_resp_bw=rg_imp_resp_bw,
        delta_k1=rg_delta_k1,
        delta_k2=rg_delta_k2,
        wgt_funct=rg_wgt_funct,
    )
    col_param = SICDDirParam(
        ss=az_ss,
        imp_resp_bw=az_imp_resp_bw,
        delta_k1=az_delta_k1,
        delta_k2=az_delta_k2,
        wgt_funct=az_wgt_funct,
    )
    grid = SICDGrid(row=row_param, col=col_param)
    return SICDMetadata(
        format='SICD', rows=64, cols=128, dtype='complex64', grid=grid,
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

    def test_valid_3x3(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=3, looks_az=3)
        assert ml.looks_rg == 3
        assert ml.looks_az == 3
        assert ml.overlap == 0.0
        assert ml.grid_shape == (3, 3)

    def test_valid_asymmetric(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=2, looks_az=4, overlap=0.3)
        assert ml.grid_shape == (2, 4)
        assert ml.overlap == 0.3

    def test_degenerate_1xN(self):
        """1 x N grid (range-only splitting) should be valid."""
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=1, looks_az=3)
        assert ml.grid_shape == (1, 3)

    def test_degenerate_Mx1(self):
        """M x 1 grid (azimuth-only splitting) should be valid."""
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=3, looks_az=1)
        assert ml.grid_shape == (3, 1)

    def test_total_looks_too_small_raises(self):
        meta = _make_metadata()
        with pytest.raises(ValueError, match="Total looks"):
            MultilookDecomposition(meta, looks_rg=1, looks_az=1)

    def test_missing_grid_raises(self):
        meta = SICDMetadata(
            format='SICD', rows=64, cols=128, dtype='complex64',
        )
        with pytest.raises(ValueError, match="grid is None"):
            MultilookDecomposition(meta)

    def test_missing_row_param_raises(self):
        meta = SICDMetadata(
            format='SICD', rows=64, cols=128, dtype='complex64',
            grid=SICDGrid(col=SICDDirParam(ss=0.008, imp_resp_bw=80.0)),
        )
        with pytest.raises(ValueError, match="row is None"):
            MultilookDecomposition(meta)

    def test_missing_col_param_raises(self):
        meta = SICDMetadata(
            format='SICD', rows=64, cols=128, dtype='complex64',
            grid=SICDGrid(row=SICDDirParam(ss=0.005, imp_resp_bw=100.0)),
        )
        with pytest.raises(ValueError, match="col is None"):
            MultilookDecomposition(meta)

    def test_missing_row_imp_resp_bw_raises(self):
        meta = SICDMetadata(
            format='SICD', rows=64, cols=128, dtype='complex64',
            grid=SICDGrid(
                row=SICDDirParam(ss=0.005),
                col=SICDDirParam(ss=0.008, imp_resp_bw=80.0),
            ),
        )
        with pytest.raises(ValueError, match="row imp_resp_bw"):
            MultilookDecomposition(meta)

    def test_missing_col_ss_raises(self):
        meta = SICDMetadata(
            format='SICD', rows=64, cols=128, dtype='complex64',
            grid=SICDGrid(
                row=SICDDirParam(ss=0.005, imp_resp_bw=100.0),
                col=SICDDirParam(imp_resp_bw=80.0),
            ),
        )
        with pytest.raises(ValueError, match="col ss"):
            MultilookDecomposition(meta)

    def test_overlap_too_high_raises(self):
        meta = _make_metadata()
        with pytest.raises(ValueError, match="overlap"):
            MultilookDecomposition(meta, overlap=1.0)

    def test_overlap_negative_raises(self):
        meta = _make_metadata()
        with pytest.raises(ValueError, match="overlap"):
            MultilookDecomposition(meta, overlap=-0.1)


# ===================================================================
# Decompose output
# ===================================================================

class TestDecompose:
    """Tests for the decompose() method output shape and type."""

    def test_output_shape_3x3(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=3, looks_az=3)
        image = _synthetic_complex_image()
        result = ml.decompose(image)
        assert result.shape == (3, 3, 64, 128)

    def test_output_shape_2x4(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=2, looks_az=4)
        image = _synthetic_complex_image()
        result = ml.decompose(image)
        assert result.shape == (2, 4, 64, 128)

    def test_output_shape_1x3(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=1, looks_az=3)
        image = _synthetic_complex_image()
        result = ml.decompose(image)
        assert result.shape == (1, 3, 64, 128)

    def test_output_is_complex(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=2, looks_az=2)
        image = _synthetic_complex_image()
        result = ml.decompose(image)
        assert np.iscomplexobj(result)

    def test_output_dtype_matches_input(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=2, looks_az=2)
        image = _synthetic_complex_image().astype(np.complex128)
        result = ml.decompose(image)
        assert result.dtype == np.complex128

    def test_non_complex_input_raises(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta)
        with pytest.raises(TypeError, match="complex"):
            ml.decompose(np.ones((64, 128), dtype=np.float32))

    def test_3d_input_raises(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta)
        with pytest.raises(ValueError, match="2D"):
            ml.decompose(np.ones((2, 64, 128), dtype=np.complex64))

    def test_non_ndarray_input_raises(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta)
        with pytest.raises(TypeError, match="ndarray"):
            ml.decompose([[1 + 0j, 2 + 0j]])


# ===================================================================
# Tiler integration
# ===================================================================

class TestTilerIntegration:
    """Verify the Tiler produces the expected number of regions."""

    def test_tiler_region_count_3x3(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=3, looks_az=3)
        image = _synthetic_complex_image()
        rows, cols = image.shape

        rg_start, rg_stop, _ = ml._compute_support_geometry(
            rows, ml._rg_imp_resp_bw, ml._rg_ss,
            ml._rg_delta_k1, ml._rg_delta_k2,
        )
        az_start, az_stop, _ = ml._compute_support_geometry(
            cols, ml._az_imp_resp_bw, ml._az_ss,
            ml._az_delta_k1, ml._az_delta_k2,
        )
        tiler = ml._build_tiler(rg_stop - rg_start, az_stop - az_start)
        regions = tiler.tile_positions()
        assert len(regions) == 9

    def test_tiler_region_count_2x4(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=2, looks_az=4)
        image = _synthetic_complex_image()
        rows, cols = image.shape

        rg_start, rg_stop, _ = ml._compute_support_geometry(
            rows, ml._rg_imp_resp_bw, ml._rg_ss,
            ml._rg_delta_k1, ml._rg_delta_k2,
        )
        az_start, az_stop, _ = ml._compute_support_geometry(
            cols, ml._az_imp_resp_bw, ml._az_ss,
            ml._az_delta_k1, ml._az_delta_k2,
        )
        tiler = ml._build_tiler(rg_stop - rg_start, az_stop - az_start)
        regions = tiler.tile_positions()
        assert len(regions) == 8

    def test_tiler_regions_within_support(self):
        """All tile regions should be within the support dimensions."""
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=3, looks_az=3)
        image = _synthetic_complex_image()
        rows, cols = image.shape

        rg_start, rg_stop, _ = ml._compute_support_geometry(
            rows, ml._rg_imp_resp_bw, ml._rg_ss,
            ml._rg_delta_k1, ml._rg_delta_k2,
        )
        az_start, az_stop, _ = ml._compute_support_geometry(
            cols, ml._az_imp_resp_bw, ml._az_ss,
            ml._az_delta_k1, ml._az_delta_k2,
        )
        n_rg = rg_stop - rg_start
        n_az = az_stop - az_start
        tiler = ml._build_tiler(n_rg, n_az)
        for region in tiler.tile_positions():
            assert 0 <= region.row_start < region.row_end <= n_rg
            assert 0 <= region.col_start < region.col_end <= n_az


# ===================================================================
# Energy conservation
# ===================================================================

class TestEnergyConservation:
    """Verify sub-look powers sum approximately to original power."""

    def test_energy_no_overlap_no_deweight(self):
        """Without deweighting or overlap, sub-look powers should
        approximately sum to original power."""
        meta = _make_metadata()
        ml = MultilookDecomposition(
            meta, looks_rg=2, looks_az=2, overlap=0.0, deweight=False,
        )
        image = _synthetic_complex_image(rows=64, cols=128)

        original_power = np.sum(np.abs(image) ** 2)
        result = ml.decompose(image)
        sublook_power = np.sum(np.abs(result) ** 2)

        # Energy should be approximately conserved (within a factor due to
        # spectral leakage at sub-band edges and oversampling)
        ratio = sublook_power / original_power
        assert 0.3 < ratio < 1.5, f"Energy ratio {ratio:.3f} out of range"


# ===================================================================
# Overlap parameter
# ===================================================================

class TestOverlap:
    """Verify overlap between adjacent sub-bands."""

    def test_overlap_produces_wider_subbands(self):
        """With overlap, each sub-band should be wider (more energy)."""
        meta = _make_metadata()
        image = _synthetic_complex_image(rows=64, cols=128)

        ml_no = MultilookDecomposition(
            meta, looks_rg=3, looks_az=3, overlap=0.0, deweight=False,
        )
        ml_with = MultilookDecomposition(
            meta, looks_rg=3, looks_az=3, overlap=0.5, deweight=False,
        )

        result_no = ml_no.decompose(image)
        result_with = ml_with.decompose(image)

        # Center sub-look (1,1) should have more energy with overlap
        power_no = np.sum(np.abs(result_no[1, 1]) ** 2)
        power_with = np.sum(np.abs(result_with[1, 1]) ** 2)
        assert power_with > power_no, (
            "50% overlap sub-look should have more power than 0% overlap"
        )


# ===================================================================
# Deweighting
# ===================================================================

class TestDeweight:
    """Verify deweighting removes original apodization."""

    def test_deweight_with_weight_function(self):
        """Deweighting should change the result vs not deweighting."""
        rg_wgt = np.hamming(64).astype(np.float64)
        az_wgt = np.hanning(64).astype(np.float64)
        meta = _make_metadata(rg_wgt_funct=rg_wgt, az_wgt_funct=az_wgt)
        image = _synthetic_complex_image(rows=64, cols=128)

        ml_dw = MultilookDecomposition(
            meta, looks_rg=2, looks_az=2, deweight=True,
        )
        ml_ndw = MultilookDecomposition(
            meta, looks_rg=2, looks_az=2, deweight=False,
        )

        result_dw = ml_dw.decompose(image)
        result_ndw = ml_ndw.decompose(image)

        assert not np.allclose(result_dw, result_ndw), (
            "Deweighted and non-deweighted results should differ"
        )

    def test_deweight_disabled_ignores_weights(self):
        """When deweight=False, weight functions are ignored."""
        rg_wgt = np.hamming(64).astype(np.float64)
        az_wgt = np.hanning(64).astype(np.float64)
        meta_with = _make_metadata(rg_wgt_funct=rg_wgt, az_wgt_funct=az_wgt)
        meta_without = _make_metadata()
        image = _synthetic_complex_image(rows=64, cols=128)

        ml_with = MultilookDecomposition(
            meta_with, looks_rg=2, looks_az=2, deweight=False,
        )
        ml_without = MultilookDecomposition(
            meta_without, looks_rg=2, looks_az=2, deweight=False,
        )

        result_with = ml_with.decompose(image)
        result_without = ml_without.decompose(image)

        np.testing.assert_allclose(result_with, result_without, atol=1e-6)

    def test_deweight_one_dimension_only(self):
        """If only one dimension has weights, the other is identity."""
        rg_wgt = np.hamming(64).astype(np.float64)
        meta = _make_metadata(rg_wgt_funct=rg_wgt)
        image = _synthetic_complex_image(rows=64, cols=128)

        ml = MultilookDecomposition(
            meta, looks_rg=2, looks_az=2, deweight=True,
        )
        # Should not raise — azimuth gets identity weights
        result = ml.decompose(image)
        assert result.shape == (2, 2, 64, 128)


# ===================================================================
# Conversion helpers
# ===================================================================

class TestConversions:
    """Tests for to_magnitude, to_power, to_db, to_flat_stack."""

    @pytest.fixture
    def multilook_grid(self):
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=2, looks_az=3)
        image = _synthetic_complex_image()
        return ml, ml.decompose(image)

    def test_to_magnitude(self, multilook_grid):
        ml, grid = multilook_grid
        mag = ml.to_magnitude(grid)
        assert mag.shape == grid.shape
        assert not np.iscomplexobj(mag)
        np.testing.assert_allclose(mag, np.abs(grid))

    def test_to_power(self, multilook_grid):
        ml, grid = multilook_grid
        pwr = ml.to_power(grid)
        assert pwr.shape == grid.shape
        assert not np.iscomplexobj(pwr)
        np.testing.assert_allclose(pwr, np.abs(grid) ** 2)

    def test_to_db(self, multilook_grid):
        ml, grid = multilook_grid
        db = ml.to_db(grid, floor=-60.0)
        assert db.shape == grid.shape
        assert not np.iscomplexobj(db)
        assert np.all(db >= -60.0)

    def test_to_db_default_floor(self, multilook_grid):
        ml, grid = multilook_grid
        db = ml.to_db(grid)
        assert np.all(db >= -50.0)

    def test_to_flat_stack(self, multilook_grid):
        ml, grid = multilook_grid
        flat = ml.to_flat_stack(grid)
        assert flat.shape == (6, 64, 128)
        assert flat.dtype == grid.dtype

    def test_to_flat_stack_order(self, multilook_grid):
        """Flat stack should be row-major: [rg0_az0, rg0_az1, ...]."""
        ml, grid = multilook_grid
        flat = ml.to_flat_stack(grid)
        # First sub-look in flat = grid[0, 0]
        np.testing.assert_array_equal(flat[0], grid[0, 0])
        # Second sub-look in flat = grid[0, 1]
        np.testing.assert_array_equal(flat[1], grid[0, 1])
        # Last sub-look = grid[1, 2]
        np.testing.assert_array_equal(flat[5], grid[1, 2])


# ===================================================================
# Support geometry
# ===================================================================

class TestSupportGeometry:
    """Tests for _compute_support_geometry static method."""

    def test_symmetric_support_without_delta_k(self):
        """Without delta_k, support should be centered and symmetric."""
        start, stop, osr = MultilookDecomposition._compute_support_geometry(
            n_samples=256, imp_resp_bw=100.0, ss=0.005,
            delta_k1=None, delta_k2=None,
        )
        # OSR = (1/0.005)/100 = 2.0, so support = 256/2 = 128 bins
        n_support = stop - start
        assert n_support == 128
        # Should be centered: starts at (256 - 128) / 2 = 64
        assert start == 64
        assert stop == 192

    def test_support_with_delta_k(self):
        """With delta_k, support should respect the specified offsets."""
        # k_full = 1/0.005 = 200 cycles/m
        # bw = 100 cycles/m
        # n_samples * ss = 256 * 0.005 = 1.28 m (bin_per_k)
        # delta_k1 = -50 => start_f = -50 * 1.28 = -64
        # delta_k2 = +50 => stop_f = +50 * 1.28 = +64
        start, stop, osr = MultilookDecomposition._compute_support_geometry(
            n_samples=256, imp_resp_bw=100.0, ss=0.005,
            delta_k1=-50.0, delta_k2=50.0,
        )
        n_support = stop - start
        assert n_support == 128
        assert osr == 2.0

    def test_osr_computation(self):
        """OSR should be k_full / imp_resp_bw."""
        _, _, osr = MultilookDecomposition._compute_support_geometry(
            n_samples=128, imp_resp_bw=80.0, ss=0.008,
            delta_k1=None, delta_k2=None,
        )
        # OSR = (1/0.008) / 80 = 125 / 80 = 1.5625
        assert abs(osr - 1.5625) < 1e-10


# ===================================================================
# Versioning / tags
# ===================================================================

class TestVersioning:
    """Verify processor version and tags."""

    def test_has_version(self):
        assert MultilookDecomposition.__processor_version__ == '0.1.0'

    def test_has_tags(self):
        from grdl.vocabulary import ImageModality, ProcessorCategory
        tags = MultilookDecomposition.__processor_tags__
        assert ImageModality.SAR in tags['modalities']
        assert tags['category'] == ProcessorCategory.STACKS

    def test_gpu_compatible_flag(self):
        assert MultilookDecomposition.__gpu_compatible__ is True


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
        ml = MultilookDecomposition(meta, looks_rg=2, looks_az=2)
        image_np = _synthetic_complex_image()
        image_t = torch.from_numpy(image_np)

        result = ml.decompose(image_t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2, 64, 128)
        assert np.iscomplexobj(result)

    def test_torch_matches_numpy(self, torch_available):
        """Torch and numpy paths should produce similar results."""
        import torch
        meta = _make_metadata()
        ml = MultilookDecomposition(
            meta, looks_rg=2, looks_az=2, deweight=False,
        )
        image_np = _synthetic_complex_image()
        image_t = torch.from_numpy(image_np.copy())

        result_np = ml.decompose(image_np)
        result_torch = ml.decompose(image_t)

        np.testing.assert_allclose(
            result_torch, result_np, rtol=1e-4, atol=1e-6,
        )

    def test_torch_non_complex_raises(self, torch_available):
        import torch
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=2, looks_az=2)
        with pytest.raises(TypeError, match="complex"):
            ml.decompose(torch.ones(64, 128))

    def test_torch_3d_raises(self, torch_available):
        import torch
        meta = _make_metadata()
        ml = MultilookDecomposition(meta, looks_rg=2, looks_az=2)
        with pytest.raises(ValueError, match="2D"):
            ml.decompose(torch.ones(2, 64, 128, dtype=torch.complex64))
