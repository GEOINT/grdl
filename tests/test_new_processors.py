# -*- coding: utf-8 -*-
"""
Tests for newly added GRDL processors:
- Edge: SobelFilter, LaplacianFilter, UnsharpMaskFilter, HighPassFilter
- SAR Speckle: FrostFilter, KuanFilter, BoxcarFilter
- Fusion: BroveyPanSharpenTransform, SimpleMeanFusionTransform
- Spectral: SpectralIndexTransform, compute_spectral_index
"""

import numpy as np
import pytest

from grdl.image_processing.filters.edge import (
    HighPassFilter,
    LaplacianFilter,
    SobelFilter,
    UnsharpMaskFilter,
)
from grdl.image_processing.filters.sar_speckle import (
    BoxcarFilter,
    FrostFilter,
    KuanFilter,
)
from grdl.image_processing.fusion import (
    BroveyPanSharpenTransform,
    SimpleMeanFusionTransform,
    brovey_pan_sharpen,
    simple_mean_fusion,
)
from grdl.image_processing.spectral import (
    SpectralIndexTransform,
    compute_spectral_index,
)


class TestEdgeFilters:
    def test_sobel_gradient_components(self):
        arr = np.random.rand(20, 20).astype(np.float32)
        f_mag = SobelFilter(output_type="magnitude")
        out_mag = f_mag.apply(arr)
        assert out_mag.shape == arr.shape

        f_gx = SobelFilter(output_type="gradient_x")
        out_gx = f_gx.apply(arr)
        assert out_gx.shape == arr.shape

        f_gy = SobelFilter(output_type="gradient_y")
        out_gy = f_gy.apply(arr)
        assert out_gy.shape == arr.shape

    def test_laplacian_filter(self):
        arr = np.random.rand(20, 20).astype(np.float32)
        f = LaplacianFilter(mode="reflect")
        out = f.apply(arr)
        assert out.shape == arr.shape

    def test_unsharp_mask(self):
        arr = np.random.rand(20, 20).astype(np.float32)
        f = UnsharpMaskFilter(radius=1.0, amount=2.0)
        out = f.apply(arr)
        assert out.shape == arr.shape

    def test_high_pass_filter(self):
        arr = np.random.rand(20, 20).astype(np.float32)
        f = HighPassFilter(cutoff=2.0)
        out = f.apply(arr)
        assert out.shape == arr.shape


class TestSARSpeckleFilters:
    def test_frost_filter(self):
        arr = np.ones((25, 25), dtype=np.float32) * 10.0
        f = FrostFilter(window_size=5, damping_factor=2.0)
        out = f.apply(arr)
        assert out.shape == arr.shape
        assert np.allclose(out, 10.0, atol=1e-3)

    def test_kuan_filter(self):
        arr = np.ones((25, 25), dtype=np.float32) * 10.0
        f = KuanFilter(window_size=5, num_looks=2)
        out = f.apply(arr)
        assert out.shape == arr.shape
        assert np.allclose(out, 10.0, atol=1e-3)

    def test_boxcar_filter(self):
        arr = np.ones((15, 15), dtype=np.float32) * 5.0
        f = BoxcarFilter(window_size=3)
        out = f.apply(arr)
        assert out.shape == arr.shape
        assert np.allclose(out, 5.0, atol=1e-3)


class TestFusionTransforms:
    def test_brovey_fusion(self):
        pan = np.ones((16, 16), dtype=np.float32) * 4.0
        ms = np.ones((4, 16, 16), dtype=np.float32) * 2.0
        f = BroveyPanSharpenTransform(pan_weight=1.0)
        out = f.apply(ms, pan=pan)
        assert out.shape == (4, 16, 16)
        # Ratio: 4.0 / 2.0 = 2.0, Fused: 2.0 * 2.0 = 4.0
        assert np.allclose(out, 4.0, atol=1e-3)

    def test_simple_mean_fusion(self):
        pan = np.ones((16, 16), dtype=np.float32) * 6.0
        ms = np.ones((4, 16, 16), dtype=np.float32) * 2.0
        f = SimpleMeanFusionTransform(pan_weight=0.5)
        out = f.apply(ms, pan=pan)
        assert out.shape == (4, 16, 16)
        # 0.5 * 2.0 + 0.5 * 6.0 = 4.0
        assert np.allclose(out, 4.0, atol=1e-3)


class TestSpectralTransforms:
    def test_spectral_indices(self):
        cube = np.zeros((6, 20, 20), dtype=np.float32)
        cube[2] = 0.2  # Red
        cube[3] = 0.8  # NIR
        cube[1] = 0.5  # Green
        cube[5] = 0.1  # SWIR

        ndvi = compute_spectral_index(cube, index_type="NDVI", nir_idx=3, red_idx=2)
        # (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
        assert np.allclose(ndvi, 0.6, atol=1e-3)

        ndwi = compute_spectral_index(cube, index_type="NDWI", green_idx=1, nir_idx=3)
        # (0.5 - 0.8) / (0.5 + 0.8) = -0.3 / 1.3
        assert np.allclose(ndwi, -0.3 / 1.3, atol=1e-3)

        t = SpectralIndexTransform(index_type="NDVI", nir_idx=3, red_idx=2)
        out_t = t.apply(cube)
        assert np.allclose(out_t, 0.6, atol=1e-3)
