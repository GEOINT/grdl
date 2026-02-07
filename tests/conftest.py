# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for GRDL test suite.

Provides reusable synthetic test images and common test utilities
used across multiple test modules.

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-06

Modified
--------
2026-02-06
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic random number generator (seed=42)."""
    return np.random.RandomState(42)


@pytest.fixture
def flat_image():
    """50x50 uniform image at value 100.0."""
    return np.full((50, 50), 100.0)


@pytest.fixture
def gradient_image():
    """100x100 horizontal gradient from 0 to 200."""
    rows, cols = 100, 100
    return np.tile(np.linspace(0, 200, cols), (rows, 1))


@pytest.fixture
def step_edge_image():
    """40x80 image with vertical step edge at column 40."""
    image = np.zeros((40, 80))
    image[:, 40:] = 200.0
    return image


@pytest.fixture
def salt_pepper_image(rng):
    """50x50 image at 100.0 with 5% salt-and-pepper noise."""
    image = np.full((50, 50), 100.0)
    salt = rng.rand(50, 50) < 0.05
    image[salt] = 255.0
    pepper = rng.rand(50, 50) < 0.05
    image[pepper] = 0.0
    return image


@pytest.fixture
def random_image(rng):
    """30x30 random image in [0, 200]."""
    return rng.rand(30, 30) * 200


@pytest.fixture
def binary_image():
    """30x30 binary image with 10x10 bright square at center."""
    image = np.zeros((30, 30))
    image[10:20, 10:20] = 1.0
    return image


@pytest.fixture
def single_peak_image():
    """50x50 zeros with a single bright peak at [25, 25]."""
    image = np.zeros((50, 50))
    image[25, 25] = 100.0
    return image


@pytest.fixture
def two_region_image():
    """20x40 image: left half = 0, right half = 200."""
    image = np.zeros((20, 40))
    image[:, 20:] = 200.0
    return image
