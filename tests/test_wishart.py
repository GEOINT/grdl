#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for WishartFilter - validates basic functionality.

This script creates synthetic C2 covariance matrix data and applies
the Wishart filter to verify it works correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add grdl to path if needed
grdl_path = Path(__file__).parent.parent.parent.parent
if str(grdl_path) not in sys.path:
    sys.path.insert(0, str(grdl_path))

from grdl.image_processing.filters.wishart import WishartFilter


def create_synthetic_c2_covmat(rows: int = 256, cols: int = 256, add_speckle: bool = True) -> np.ndarray:
    """Create a synthetic C2 covariance matrix image.
    
    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    add_speckle : bool
        Whether to add multiplicative speckle noise.
    
    Returns
    -------
    np.ndarray
        Synthetic C2 image with shape (4, rows, cols).
    """
    # Create a simple scene with different regions
    scene = np.zeros((rows, cols), dtype=np.float32)
    
    # Add regions with different backscatter
    scene[:rows//2, :cols//2] = 1.0  # Low backscatter
    scene[:rows//2, cols//2:] = 3.0  # Medium backscatter
    scene[rows//2:, :cols//2] = 5.0  # High backscatter
    scene[rows//2:, cols//2:] = 2.0  # Medium-low backscatter
    
    # Add smooth transitions (edges)
    scene = smooth_edges(scene, 20)
    
    # Create C2 matrix: [C11, C12_real, C12_imag, C22]
    c11 = scene.copy()
    c22 = scene * 0.8  # Different polarization response
    
    # Add correlation structure
    c12_real = np.sqrt(c11 * c22) * 0.3 * np.cos(2 * np.pi * np.arange(cols) / cols)[None, :]
    c12_imag = np.sqrt(c11 * c22) * 0.2 * np.sin(2 * np.pi * np.arange(rows) / rows)[:, None]
    
    # Stack into C2 format
    covmat = np.stack([c11, c12_real, c12_imag, c22], axis=0)
    
    # Add speckle noise
    if add_speckle:
        enl = 4.0  # Equivalent number of looks
        for i in range(4):
            # Multiplicative gamma-distributed speckle
            speckle = np.random.gamma(enl, 1.0 / enl, size=(rows, cols))
            covmat[i] *= speckle
    
    return covmat.astype(np.float32)


def smooth_edges(image: np.ndarray, width: int) -> np.ndarray:
    """Add smooth transitions at region boundaries.
    
    Parameters
    ----------
    image : np.ndarray
        Input image with sharp boundaries.
    width : int
        Width of the transition zone in pixels.
    
    Returns
    -------
    np.ndarray
        Image with smoothed edges.
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image, sigma=width/3.0)


def compute_metrics(original: np.ndarray, filtered: np.ndarray) -> dict:
    """Compute quality metrics for filtered image.
    
    Parameters
    ----------
    original : np.ndarray
        Original covariance matrix image.
    filtered : np.ndarray
        Filtered covariance matrix image.
    
    Returns
    -------
    dict
        Dictionary of metrics.
    """
    # Use C11 (first band) as intensity proxy
    orig_int = original[0]
    filt_int = filtered[0]
    
    # Coefficient of variation (speckle index)
    orig_cv = np.std(orig_int) / (np.mean(orig_int) + 1e-10)
    filt_cv = np.std(filt_int) / (np.mean(filt_int) + 1e-10)
    
    # Edge preservation (correlation)
    correlation = np.corrcoef(orig_int.ravel(), filt_int.ravel())[0, 1]
    
    # Mean preservation
    orig_mean = np.mean(orig_int)
    filt_mean = np.mean(filt_int)
    mean_ratio = filt_mean / (orig_mean + 1e-10)
    
    return {
        'original_cv': orig_cv,
        'filtered_cv': filt_cv,
        'cv_reduction': (orig_cv - filt_cv) / (orig_cv + 1e-10),
        'correlation': correlation,
        'mean_ratio': mean_ratio
    }


def test_c2_filtering():
    """Test C2 (dual-pol) filtering."""
    print("=" * 60)
    print("Testing WishartFilter with C2 (dual-pol) data")
    print("=" * 60)
    
    # Create synthetic C2 data
    print("\n1. Creating synthetic C2 covariance matrix image...")
    covmat = create_synthetic_c2_covmat(256, 256, add_speckle=True)
    print(f"   Shape: {covmat.shape}")
    print(f"   C11 range: [{covmat[0].min():.3f}, {covmat[0].max():.3f}]")
    print(f"   C22 range: [{covmat[3].min():.3f}, {covmat[3].max():.3f}]")
    
    # Apply Wishart filter with auto-detected matrix type
    print("\n2. Applying Wishart filter (auto-detect matrix type)...")
    filt = WishartFilter(kernel_size=7, enl=0.0, matrix_type='auto')
    filtered = filt.apply(covmat)
    print(f"   Output shape: {filtered.shape}")
    print(f"   Filtered C11 range: [{filtered[0].min():.3f}, {filtered[0].max():.3f}]")
    
    # Compute metrics
    print("\n3. Computing quality metrics...")
    metrics = compute_metrics(covmat, filtered)
    print(f"   Original CV (C11): {metrics['original_cv']:.4f}")
    print(f"   Filtered CV (C11): {metrics['filtered_cv']:.4f}")
    print(f"   CV reduction: {metrics['cv_reduction']*100:.1f}%")
    print(f"   Correlation: {metrics['correlation']:.4f}")
    print(f"   Mean preservation: {metrics['mean_ratio']:.4f}")
    
    # Verify output properties
    print("\n4. Verifying output properties...")
    assert filtered.shape == covmat.shape, "Output shape mismatch"
    assert np.all(np.isfinite(filtered)), "Output contains NaN or Inf"
    assert np.all(filtered[0] >= 0), "C11 should be non-negative"
    assert np.all(filtered[3] >= 0), "C22 should be non-negative"
    print("   ✓ All checks passed")
    
    return True


def test_explicit_matrix_types():
    """Test explicit matrix type specification."""
    print("\n" + "=" * 60)
    print("Testing explicit matrix type specification")
    print("=" * 60)
    
    # Test C2
    print("\n1. Testing with matrix_type='C2'...")
    covmat_c2 = create_synthetic_c2_covmat(128, 128, add_speckle=True)
    filt_c2 = WishartFilter(kernel_size=5, enl=4.0, matrix_type='C2')
    filtered_c2 = filt_c2.apply(covmat_c2)
    print(f"   Input shape: {covmat_c2.shape}, Output shape: {filtered_c2.shape}")
    assert filtered_c2.shape == covmat_c2.shape
    print("   ✓ C2 filtering successful")
    
    # Test with wrong matrix type (should raise error)
    print("\n2. Testing error handling with wrong matrix type...")
    try:
        filt_wrong = WishartFilter(kernel_size=5, matrix_type='C3')
        filt_wrong.apply(covmat_c2)  # Should fail - C2 has 4 bands, C3 expects 9
        print("   ✗ Expected error was not raised")
        return False
    except ValueError as e:
        print(f"   ✓ Expected error caught: {e}")
    
    return True


def test_edge_preservation():
    """Test edge preservation capabilities."""
    print("\n" + "=" * 60)
    print("Testing edge preservation")
    print("=" * 60)
    
    # Create data with sharp edges
    print("\n1. Creating scene with sharp edges...")
    rows, cols = 200, 200
    scene = np.ones((rows, cols), dtype=np.float32) * 2.0
    scene[rows//3:2*rows//3, cols//3:2*cols//3] = 10.0  # Bright square
    
    # Create C2 matrix
    c11 = scene.copy()
    c22 = scene * 0.9
    c12_real = np.zeros_like(scene)
    c12_imag = np.zeros_like(scene)
    
    # Add speckle
    for band in [c11, c22]:
        speckle = np.random.gamma(4.0, 0.25, size=band.shape)
        band *= speckle
    
    covmat = np.stack([c11, c12_real, c12_imag, c22], axis=0).astype(np.float32)
    
    # Apply filter with different sigma_range values
    print("\n2. Testing different sigma_range values...")
    for sigma_range in [1.0, 3.0, 5.0]:
        filt = WishartFilter(kernel_size=7, enl=4.0, sigma_range=sigma_range)
        filtered = filt.apply(covmat)
        metrics = compute_metrics(covmat, filtered)
        print(f"   sigma_range={sigma_range:.1f}: CV reduction={metrics['cv_reduction']*100:.1f}%, "
              f"correlation={metrics['correlation']:.3f}")
    
    print("\n   ✓ Edge preservation test complete")
    return True


def main():
    """Run all tests."""
    print("\n")
    print("#" * 60)
    print("# WishartFilter Test Suite")
    print("#" * 60)
    
    try:
        # Run tests
        success = True
        success &= test_c2_filtering()
        success &= test_explicit_matrix_types()
        success &= test_edge_preservation()
        
        # Summary
        print("\n" + "=" * 60)
        if success:
            print("✓ All tests PASSED")
        else:
            print("✗ Some tests FAILED")
        print("=" * 60)
        print()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n✗ Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
