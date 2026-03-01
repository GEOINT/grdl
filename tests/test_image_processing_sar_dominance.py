# -*- coding: utf-8 -*-
"""
Tests for sub-aperture dominance features.

Verifies ``compute_dominance``, ``compute_sublook_entropy``, and the
``DominanceFeatures`` ImageTransform wrapper using synthetic sub-look
stacks with known power distributions.

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
2026-02-26

Modified
--------
2026-02-26
"""

import pytest
import numpy as np

from grdl.image_processing.sar.dominance import (
    compute_dominance,
    compute_sublook_entropy,
    DominanceFeatures,
)
from grdl.IO.models import SICDMetadata
from grdl.IO.models.sicd import SICDGrid, SICDDirParam


# ===================================================================
# Helpers
# ===================================================================

def _make_metadata(
    imp_resp_bw: float = 100.0,
    ss: float = 0.005,
) -> SICDMetadata:
    """Build minimal SICDMetadata for dominance tests."""
    dir_param = SICDDirParam(ss=ss, imp_resp_bw=imp_resp_bw)
    grid = SICDGrid(
        row=SICDDirParam(ss=ss, imp_resp_bw=imp_resp_bw),
        col=dir_param,
    )
    return SICDMetadata(
        format='SICD', rows=64, cols=64, dtype='complex64', grid=grid,
    )


def _uniform_stack(n_looks: int = 7, rows: int = 32, cols: int = 32,
                   seed: int = 42) -> np.ndarray:
    """Sublook stack with roughly equal power in every look (clutter)."""
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_looks, rows, cols))
        + 1j * rng.standard_normal((n_looks, rows, cols))
    ).astype(np.complex64)


def _concentrated_stack(n_looks: int = 7, rows: int = 32, cols: int = 32,
                        bright_look: int = 3,
                        bright_factor: float = 50.0) -> np.ndarray:
    """Sublook stack where one look dominates (target-like)."""
    stack = np.ones((n_looks, rows, cols), dtype=np.complex64) * 0.01
    stack[bright_look] = bright_factor + 0j
    return stack


# ===================================================================
# compute_dominance
# ===================================================================

class TestComputeDominance:
    """Tests for the compute_dominance pure function."""

    def test_output_shape(self):
        stack = _uniform_stack(n_looks=5, rows=20, cols=30)
        result = compute_dominance(stack, window_size=3, dom_window=2)
        assert result.shape == (20, 30)

    def test_uniform_power_low_dominance(self):
        """Uniform power across looks should give dominance near floor."""
        stack = _uniform_stack(n_looks=7, rows=32, cols=32)
        dom = compute_dominance(stack, window_size=5, dom_window=3)
        floor = 3.0 / 7.0
        # Mean dominance should be close to floor for uniform power
        assert np.mean(dom) < floor + 0.15

    def test_concentrated_power_high_dominance(self):
        """Power in one look should give dominance near 1.0."""
        stack = _concentrated_stack(n_looks=7, bright_look=3,
                                    bright_factor=100.0)
        dom = compute_dominance(stack, window_size=3, dom_window=3)
        # Bright pixel region should be near 1.0
        assert np.mean(dom) > 0.8

    def test_values_bounded(self):
        """Dominance must be in (0, 1.0]."""
        stack = _uniform_stack(n_looks=7)
        dom = compute_dominance(stack, window_size=3, dom_window=3)
        assert np.all(dom > 0.0)
        assert np.all(dom <= 1.0 + 1e-6)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            compute_dominance(np.zeros((32, 32)))

    def test_dom_window_exceeds_looks_raises(self):
        stack = _uniform_stack(n_looks=3)
        with pytest.raises(ValueError, match="dom_window"):
            compute_dominance(stack, dom_window=5)

    def test_dom_window_equals_looks(self):
        """When dom_window == n_looks, dominance should be 1.0 everywhere."""
        stack = _uniform_stack(n_looks=5)
        dom = compute_dominance(stack, window_size=3, dom_window=5)
        np.testing.assert_allclose(dom, 1.0, atol=1e-6)


# ===================================================================
# compute_sublook_entropy
# ===================================================================

class TestComputeSublookEntropy:
    """Tests for the compute_sublook_entropy pure function."""

    def test_output_shape(self):
        stack = _uniform_stack(n_looks=5, rows=20, cols=30)
        result = compute_sublook_entropy(stack, window_size=3)
        assert result.shape == (20, 30)

    def test_uniform_power_high_entropy(self):
        """Uniform power should give entropy near log(n_looks)."""
        stack = _uniform_stack(n_looks=7, rows=32, cols=32)
        ent = compute_sublook_entropy(stack, window_size=5)
        max_entropy = np.log(7)
        # Should be close to maximum
        assert np.mean(ent) > max_entropy * 0.8

    def test_concentrated_power_low_entropy(self):
        """Power in one look should give low entropy."""
        stack = _concentrated_stack(n_looks=7, bright_look=3,
                                    bright_factor=1000.0)
        ent = compute_sublook_entropy(stack, window_size=3)
        max_entropy = np.log(7)
        # Should be much lower than maximum
        assert np.mean(ent) < max_entropy * 0.3

    def test_entropy_non_negative(self):
        stack = _uniform_stack(n_looks=5)
        ent = compute_sublook_entropy(stack, window_size=3)
        assert np.all(ent >= -1e-10)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            compute_sublook_entropy(np.zeros((32, 32)))


# ===================================================================
# DominanceFeatures (ImageTransform wrapper)
# ===================================================================

class TestDominanceFeatures:
    """Tests for the DominanceFeatures ImageTransform class."""

    def test_output_shape(self):
        meta = _make_metadata()
        rng = np.random.default_rng(99)
        image = (rng.standard_normal((64, 64))
                 + 1j * rng.standard_normal((64, 64))).astype(np.complex64)

        dom = DominanceFeatures(meta, num_looks=3, window_size=3,
                                dom_window=2)
        result = dom.apply(image)
        assert result.shape == (2, 64, 64)

    def test_channels_are_dominance_and_entropy(self):
        meta = _make_metadata()
        rng = np.random.default_rng(99)
        image = (rng.standard_normal((64, 64))
                 + 1j * rng.standard_normal((64, 64))).astype(np.complex64)

        dom = DominanceFeatures(meta, num_looks=3, window_size=3,
                                dom_window=2)
        result = dom.apply(image)

        # Channel 0 = dominance: bounded in (0, 1]
        assert np.all(result[0] > 0.0)
        assert np.all(result[0] <= 1.0 + 1e-6)

        # Channel 1 = entropy: non-negative
        assert np.all(result[1] >= -1e-10)

    def test_invalid_dimension_raises(self):
        meta = _make_metadata()
        with pytest.raises(ValueError, match="dimension"):
            DominanceFeatures(meta, dimension='time')

    def test_dom_window_exceeds_looks_raises(self):
        meta = _make_metadata()
        with pytest.raises(ValueError, match="dom_window"):
            DominanceFeatures(meta, num_looks=3, dom_window=5)

    def test_processor_version(self):
        assert hasattr(DominanceFeatures, '__processor_version__')
        assert DominanceFeatures.__processor_version__ == '1.0.0'

    def test_processor_tags(self):
        tags = DominanceFeatures.__processor_tags__
        assert 'SAR' in str(tags.get('modalities', ''))

    def test_runtime_param_override(self):
        """Tunable params can be overridden at apply() time."""
        meta = _make_metadata()
        rng = np.random.default_rng(99)
        image = (rng.standard_normal((64, 64))
                 + 1j * rng.standard_normal((64, 64))).astype(np.complex64)

        dom = DominanceFeatures(meta, num_looks=3, window_size=3,
                                dom_window=2)
        # Override window_size at apply time
        result = dom.apply(image, window_size=5)
        assert result.shape == (2, 64, 64)
