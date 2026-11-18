"""Tests for coherent orthogonal-linear polarization basis correction."""

import numpy as np
import pytest

from grdl.image_processing.sar import rotate_quad_pol_to_hv


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('angle_deg', [0.0, 13.5, -27.0, 45.0, 90.0])
def test_recovers_ground_scattering_matrix(dtype, angle_deg):
    rng = np.random.default_rng(42)
    ground = (rng.normal(size=(2, 2, 3, 4)) + 1j * rng.normal(size=(2, 2, 3, 4))).astype(dtype)
    radians = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]])
    platform = np.einsum('ia,ij...,jb->ab...', rotation, ground, rotation).astype(dtype)
    before = platform.copy()
    result = rotate_quad_pol_to_hv(*platform.reshape(4, 3, 4), angle_deg)
    np.testing.assert_allclose(np.stack(result).reshape(ground.shape), ground, atol=1e-6)
    np.testing.assert_array_equal(platform, before)
    assert all(channel.dtype == dtype for channel in result)
    np.testing.assert_allclose(
        np.sum(np.abs(result)**2, axis=0),
        np.sum(np.abs(platform)**2, axis=(0, 1)), rtol=1e-6,
    )


def test_signed_rotation_of_horizontal_target():
    ones = np.ones((2, 3), dtype=complex)
    zeros = np.zeros_like(ones)
    shh, shv, svh, svv = rotate_quad_pol_to_hv(ones, zeros, zeros, zeros, 30)
    np.testing.assert_allclose(shh, 0.75)
    np.testing.assert_allclose(shv, np.sqrt(3) / 4)
    np.testing.assert_allclose(svh, shv)
    np.testing.assert_allclose(svv, 0.25)


def test_per_pixel_angles_and_inverse():
    rng = np.random.default_rng(7)
    channels = rng.normal(size=(4, 2, 3)) + 1j * rng.normal(size=(4, 2, 3))
    angles = np.array([[0, -20, 45], [12, 90, 180]])
    rotated = rotate_quad_pol_to_hv(*channels, angles)
    recovered = rotate_quad_pol_to_hv(*rotated, -angles, basis='HV')
    np.testing.assert_allclose(recovered, channels, atol=1e-14)
    assert not np.allclose(rotated[1], rotated[2])
    broadcast = rotate_quad_pol_to_hv(*channels, angles[0])
    for index in range(3):
        expected = rotate_quad_pol_to_hv(*channels[:, :, index], angles[0, index])
        np.testing.assert_allclose(np.asarray(broadcast)[:, :, index], expected)


@pytest.mark.parametrize('basis', ['RL', 'LR', 'RH', 'XV', 'elliptical', 'nonorthogonal'])
def test_rejects_non_linear_or_mixed_basis(basis):
    channel = np.ones((2, 3), dtype=complex)
    with pytest.raises(ValueError, match='orthogonal linear'):
        rotate_quad_pol_to_hv(channel, channel, channel, channel, 0, basis=basis)


@pytest.mark.parametrize('invalid, exception, message', [
    (np.ones((2, 3)), TypeError, 'complex'),
    (np.ones((3, 2), dtype=complex), ValueError, 'shapes'),
    (np.empty((0, 3), dtype=complex), ValueError, 'nonempty'),
    (np.full((2, 3), np.nan + 1j), ValueError, 'finite'),
    (np.full((2, 3), np.inf + 1j), ValueError, 'finite'),
])
def test_rejects_invalid_channel(invalid, exception, message):
    channel = np.ones((2, 3), dtype=complex)
    with pytest.raises(exception, match=message):
        rotate_quad_pol_to_hv(channel, invalid, channel, channel, 0)


@pytest.mark.parametrize('angle, exception, message', [
    (np.nan, ValueError, 'finite'),
    (np.inf, ValueError, 'finite'),
    (1j, TypeError, 'real numeric'),
    ('10', TypeError, 'real numeric'),
    (True, TypeError, 'real numeric'),
    (np.zeros((4, 2, 3)), ValueError, 'broadcast'),
])
def test_rejects_invalid_angle(angle, exception, message):
    channel = np.ones((2, 3), dtype=complex)
    with pytest.raises(exception, match=message):
        rotate_quad_pol_to_hv(channel, channel, channel, channel, angle)