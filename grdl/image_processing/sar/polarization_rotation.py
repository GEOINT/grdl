# -*- coding: utf-8 -*-
"""
Coherent quad-pol rotation from an orthogonal linear basis to ground H/V.

Applies a known geometric polarization-basis rotation to calibrated complex
scattering channels. This is polarization synthesis in an orthogonal linear
basis, not an estimator of antenna angle, terrain orientation or Faraday
rotation. The basis-change principle is described in [1]; the derivation
below fixes the axis, angle and channel-order conventions used here.

Algorithm
---------
Let c = cos(theta), s = sin(theta), X = c H + s V, and
Y = -s H + c V. The columns of R = [[c, -s], [s, c]] are the
platform basis vectors expressed in ground H/V coordinates. Thus electric
field components satisfy e_ground = R @ e_platform.

For a receive-first scattering matrix A, e_received = A @ e_transmitted
gives A_ground = R @ A_platform @ R.T, since R is real orthogonal.
This module instead stores transmit-first channels, S = A.T; transposing
the identity yields the same S_ground = R @ S_platform @ R.T. No
reciprocity assumption (XY = YX) is needed. Expanding the products gives::

    HH = c*c*XX - c*s*(XY + YX) + s*s*YY
    HV = c*s*(XX - YY) + c*c*XY - s*s*YX
    VH = c*s*(XX - YY) - s*s*XY + c*c*YX
    VV = s*s*XX + c*s*(XY + YX) + c*c*YY

The inverse uses -theta. Orthogonality preserves total scattering power,
|HH|**2 + |HV|**2 + |VH|**2 + |VV|**2, up to roundoff. Ground H/V must
be defined in the same transverse polarization frame as the supplied angle;
this transform does not perform image georeferencing or terrain correction.

References
----------
.. [1] J.-S. Lee and E. Pottier, *Polarimetric Radar Imaging: From Basics
       to Applications*, CRC Press, 2009, Chapter 1 (radar polarimetry,
       scattering matrices and polarization synthesis).
       https://doi.org/10.1201/9781420054989

Author
------
Jason Fritz, PhD
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-11-10

Modified
--------
2026-11-10
"""

import numpy as np


def rotate_quad_pol_to_hv(
    sxx: np.ndarray,
    sxy: np.ndarray,
    syx: np.ndarray,
    syy: np.ndarray,
    angle_deg: float | np.ndarray,
    *,
    basis: str = 'XY',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate four coherent scattering channels into ground-referenced H/V.

    Parameters
    ----------
    sxx, sxy, syx, syy : np.ndarray
        Nonempty complex arrays with identical shapes, ordered XX, XY, YX,
        YY (or platform HH, HV, VH, VV for ``basis='HV'``). The first label
        denotes transmit and the second receive. Channels must be spatially
        co-registered and relatively phase/amplitude calibrated. Both transmit
        and receive must use the same orthogonal linear basis, with
        X = cos(angle) H + sin(angle) V and
        Y = -sin(angle) H + cos(angle) V. Orthogonality and calibration
        are acquisition preconditions; they cannot be inferred from samples.
        Intensities, Stokes vectors and covariance matrices are not inputs.
    angle_deg : float or np.ndarray
        Signed angle of X (platform H) from ground H toward ground V, in
        degrees. A scalar or an array broadcastable to the channel shape.
        With ``get_antenna_polarization``, use the X-axis angle, not the
        absolute Y-axis angle: subtract 90 degrees from a Y angle. The
        rotation offset returned by ``get_antenna_polarization_from_file``
        already makes that adjustment. Angle estimates are only as spatially
        representative as the supplied metadata/geometry.
    basis : {'XY', 'HV'}, optional
        Declares an orthogonal linear input basis. Circular, elliptical,
        mixed and nonorthogonal bases are unsupported.

    Returns
    -------
    tuple of np.ndarray
        Ground-referenced (HH, HV, VH, VV), with the original shape and common
        complex precision. Inputs are not modified; HV and VH remain distinct.

    Notes
    -----
    For the transmit-first matrix S = [[XX, XY], [YX, YY]], this computes
    ``S_ground = R @ S @ R.T``, where
    ``R = [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]``.
    This is a geometric linear-basis correction, not an estimated terrain
    orientation or Faraday correction. A single angle assumes identical
    transmit and receive frame rotations. Full Stokes parameters alone do
    not replace the four coherent scattering channels required here.
    See the module's Algorithm and References sections for the derivation
    and the polarization-synthesis reference underlying this transform.

    Raises
    ------
    TypeError
        If channels are not complex NumPy arrays or angles are not real numeric.
    ValueError
        If the basis, shapes, finite values or angle broadcasting are invalid.
    """
    # A basis label declares the acquisition axes; samples cannot prove them.
    if basis not in ('XY', 'HV'):
        raise ValueError("basis must be 'XY' or 'HV' (orthogonal linear polarization)")
    channels = (sxx, sxy, syx, syy)
    for channel in channels:
        if not isinstance(channel, np.ndarray) or not np.iscomplexobj(channel):
            raise TypeError('Quad-pol channels must be complex NumPy arrays')
        if channel.size == 0:
            raise ValueError('Quad-pol channels must be nonempty')
        if channel.shape != sxx.shape:
            raise ValueError('Quad-pol channel shapes must match')
        if not np.all(np.isfinite(channel)):
            raise ValueError('Quad-pol channels must contain only finite values')

    # Broadcast the geometric angle without introducing extra channel axes.
    angles = np.asarray(angle_deg)
    if angles.dtype.kind not in 'iuf':
        raise TypeError('angle_deg must be real numeric degrees')
    if not np.all(np.isfinite(angles)):
        raise ValueError('angle_deg must contain only finite values')
    try:
        angles = np.broadcast_to(angles, sxx.shape)
    except ValueError as error:
        raise ValueError('angle_deg must broadcast to the channel shape') from error

    # Evaluate trigonometry in float64, then retain common channel precision.
    dtype = np.result_type(*(channel.dtype for channel in channels), np.complex64)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    radians = np.deg2rad(angles.astype(np.float64))
    cosine = np.cos(radians).astype(real_dtype)
    sine = np.sin(radians).astype(real_dtype)
    cosine_squared = cosine * cosine
    sine_squared = sine * sine
    mixed = cosine * sine
    # Expand R @ S @ R.T without allocating a per-pixel 2-by-2 matrix stack.
    # Keep XY and YX separate: this transform does not enforce reciprocity.
    shh = cosine_squared * sxx - mixed * (sxy + syx) + sine_squared * syy
    shv = mixed * (sxx - syy) + cosine_squared * sxy - sine_squared * syx
    svh = mixed * (sxx - syy) - sine_squared * sxy + cosine_squared * syx
    svv = sine_squared * sxx + mixed * (sxy + syx) + cosine_squared * syy
    return shh, shv, svh, svv