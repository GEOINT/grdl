# -*- coding: utf-8 -*-
"""
Metadata-derived steering matrix from a CPHD ``Antenna`` section.

The CPHD standard stores per-channel receive antenna patterns in
**antenna coordinates** -- a complex aperture response

    G_c(DCx, DCy) = 10**(gain_dB_c(DCx, DCy) / 20)
                    * exp(j * phase_rad_c(DCx, DCy))

with

    gain_dB_c   = Array.GainPoly  + Element.GainPoly       # dB
    phase_rad_c = Array.PhasePoly + Element.PhasePoly      # rad

evaluated at antenna direction cosines ``(DCx, DCy)`` measured in the
channel's Antenna Coordinate Frame (ACF). The pattern is therefore
"at the antenna" -- to use it as a per-channel steering dictionary on
the ground, it has to be **projected** to scene coordinates.

This module does the projection. For each direction cosine sample
``DCx`` near the antenna's electrical boresight ``EB.dcx_poly[0]``
(at fixed ``DCy = EB.dcy_poly[0]``, i.e. on the boresight scan line),
it ray-traces from the channel's APC in ECF, intersects the planar
reference surface anchored at SRP, and reports the ground intercept's
cross-range offset from SRP along ``u_iax``. The per-channel pattern
is evaluated at the same ``(DCx, DCy)`` for every channel (or, when
the channels have non-coincident APCs, at each channel's own
``(DCx_c, DCy_c)`` derived from the unit vector APC_c -> ground point).

The result is a steering matrix indexed by ground cross-range -- a
ground beam pattern, not an antenna-frame curve -- ready to be used as
the column dictionary for a STAP detector.

Dependencies
------------
None beyond numpy.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-06

Modified
--------
2026-05-06
"""

# Standard library
from dataclasses import dataclass
from typing import Optional

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.cphd import CPHDMetadata


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CPHDMetadataSteering:
    """Output of :func:`build_steering_matrix_from_cphd_metadata`.

    Parameters
    ----------
    s : np.ndarray, shape ``(Nc, n_scan)``, complex128
        Column-normalised, phase-anchored steering matrix. Each column
        is the per-channel complex response to a target sitting at the
        ground intercept of the corresponding ``DCx`` scan sample.
    dcx_scan : np.ndarray, shape ``(n_scan,)``, float64
        ``DCx`` values used in the scan, in the reference channel's
        ACF, centred on the reference channel's electrical boresight.
    dcy_scan : float
        Fixed ``DCy`` used for the scan (the reference channel's
        ``EB.dcy_poly[0]``).
    ground_pts_ecf : np.ndarray, shape ``(n_scan, 3)``, float64
        Per-scan ECF intercept of the reference channel's ray on the
        planar reference surface (the antenna's ground beam center
        line).
    xr_offsets_m : np.ndarray, shape ``(n_scan,)``, float64
        Cross-range offset of each ground intercept from SRP, along
        the planar reference surface ``u_iax`` (metres).
    boresight_xr_offset_m : float
        Cross-range offset of the EB ground intercept from SRP. Tells
        you whether the metadata-claimed beam centre lands inside the
        imaged scene; large absolute values indicate a mismatch
        between the metadata and the SRP.
    gain_dB : np.ndarray, shape ``(Nc, n_scan)``, float64
        Per-channel gain in dB at each scan column, before column
        normalisation. Useful for sanity-checking that the polynomial
        was evaluated inside its design domain.
    """

    s: np.ndarray
    dcx_scan: np.ndarray
    dcy_scan: float
    ground_pts_ecf: np.ndarray
    xr_offsets_m: np.ndarray
    boresight_xr_offset_m: float
    gain_dB: np.ndarray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _eval_axis_poly(coefs: Optional[np.ndarray], t: float) -> np.ndarray:
    """Evaluate a CPHD axis polynomial ``(K, 3)`` at scalar time ``t``.

    Parameters
    ----------
    coefs : np.ndarray or None
        Coefficient array of shape ``(K, 3)``. ``coefs[k]`` is the
        ``k``-th order coefficient (3D vector) and the polynomial value
        at ``t`` is ``sum_k coefs[k] * t**k``. ``None`` is treated as
        a missing axis and raises.
    t : float
        Time argument relative to the CPHD collection reference time.

    Returns
    -------
    np.ndarray, shape ``(3,)``
        Polynomial value as a 3D vector.
    """
    if coefs is None:
        raise ValueError("axis polynomial is None")
    coefs = np.asarray(coefs, dtype=np.float64)
    if coefs.ndim != 2 or coefs.shape[1] != 3:
        raise ValueError(
            f"axis polynomial must be shape (K, 3); got {coefs.shape}"
        )
    K = coefs.shape[0]
    out = np.zeros(3, dtype=np.float64)
    tk = 1.0
    for k in range(K):
        out += coefs[k] * tk
        tk *= t
    return out


def _ray_plane_intercept(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """Intersect a ray ``origin + t * direction`` with a plane.

    Parameters
    ----------
    origin : np.ndarray, shape ``(3,)``
        Ray origin in ECF.
    direction : np.ndarray, shape ``(N, 3)`` or ``(3,)``
        Ray directions (will be normalised by the caller).
    plane_point : np.ndarray, shape ``(3,)``
        A point on the plane (e.g. SRP).
    plane_normal : np.ndarray, shape ``(3,)``
        Plane normal.

    Returns
    -------
    np.ndarray
        Intercepts, shape matching the broadcast of inputs minus the
        last axis. NaN where the ray is parallel to the plane.
    """
    direction = np.atleast_2d(np.asarray(direction, dtype=np.float64))
    n = np.asarray(plane_normal, dtype=np.float64)
    denom = direction @ n
    num = float((np.asarray(plane_point) - np.asarray(origin)) @ n)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(np.abs(denom) > 0, num / denom, np.nan)
    pts = np.asarray(origin)[None, :] + t[:, None] * direction
    return pts


def _polyval2d(
    poly: Optional[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Evaluate a CPHD ``Poly2D`` (raw ``(I, J)`` array) at arrays."""
    if poly is None:
        return np.zeros_like(x, dtype=np.float64)
    return np.polynomial.polynomial.polyval2d(
        x, y, np.asarray(poly, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def build_steering_matrix_from_cphd_metadata(
    metadata: CPHDMetadata,
    rcv_pos_ecf: np.ndarray,
    srp_ecf: np.ndarray,
    n_scan: int = 51,
    dcx_extent: float = 0.005,
    pulse_time: float = 0.0,
    ref_channel: int = 0,
    ground_plane_normal: Optional[np.ndarray] = None,
) -> CPHDMetadataSteering:
    """Build a per-channel steering matrix from CPHD ``Antenna`` metadata,
    projected to the local horizontal ground plane at SRP.

    The scan walks ``DCx`` across ``[EB.dcx - dcx_extent, EB.dcx +
    dcx_extent]`` (in the reference channel's ACF) at fixed ``DCy =
    EB.dcy``. Each scan ``DCx`` is converted into an ECF unit vector
    via the reference channel's ACF, the ray from the reference
    channel's APC is intersected with the local horizontal ground
    plane at SRP, and the ground point is reported as a cross-range
    offset from SRP along the in-plane projection of ``u_iax``. For
    every channel (including the reference), the per-channel pattern
    is evaluated at the ``(DCx_c, DCy_c)`` corresponding to that
    ground point in **that channel's** ACF -- so non-coincident APCs
    (with different ACF orientations) are handled correctly.

    Notes
    -----
    The CPHD ``ReferenceSurface/Planar`` (``u_iax``, ``u_iay``) defines
    the simulator's image-formation plane, which for many collections
    is the **slant** plane (containing APC and SRP) rather than a
    ground plane. Using ``u_iax × u_iay`` as the ground normal would
    produce a plane that the boresight ray runs nearly parallel to,
    giving meaningless intercepts. So the default ground plane here is
    the local horizontal at SRP (normal = SRP/|SRP|). The cross-range
    axis on that plane is the projection of ``u_iax`` onto it.

    Parameters
    ----------
    metadata : CPHDMetadata
        Must contain ``antenna`` (with ant_coord_frames, ant_phase_centers,
        ant_patterns), ``channel_section`` (per-channel rcv_apc_id and
        rcv_apat_id refs), and ``scene_coordinates.reference_surface.planar``
        (u_iax) for the cross-range axis direction.
    rcv_pos_ecf : np.ndarray, shape ``(Nc, 3)``
        Per-channel APC ECF position at the chosen pulse. Typically
        ``pvp.rcv_pos[mid_pulse]`` from each channel's PVP load.
    srp_ecf : np.ndarray, shape ``(3,)``
        SRP ECF at the chosen pulse (anchors the ground plane).
    n_scan : int
        Number of ``DCx`` scan points. Default 51.
    dcx_extent : float
        Half-extent of the ``DCx`` scan around ``EB.dcx`` (radians).
        Default ``0.005`` (≈ ±0.3 deg, sized for a high-gain main
        lobe). Increase to walk into the sidelobes.
    pulse_time : float
        Time argument (seconds since CPHD collection reference) used
        when evaluating ``ant_coord_frames[*].x_axis_poly`` /
        ``y_axis_poly``. Default 0 = use the constant term, which is
        accurate to <0.001 fractional error for typical airborne CPI
        durations.
    ref_channel : int
        Channel index whose ACF / EB anchors the scan. Default 0.
    ground_plane_normal : np.ndarray, shape ``(3,)``, optional
        Override the local-horizontal-at-SRP default. Pass an explicit
        normal (will be normalised) if the simulator's reference
        surface is in fact a ground plane and you want to use it.

    Returns
    -------
    CPHDMetadataSteering
        See :class:`CPHDMetadataSteering`.

    Raises
    ------
    ValueError
        On any missing required metadata section, missing
        per-channel reference, or non-conformant array shape.
    """
    if metadata.antenna is None or not metadata.antenna.ant_patterns:
        raise ValueError(
            "metadata.antenna is missing or has no ant_patterns"
        )
    if metadata.channel_section is None:
        raise ValueError(
            "metadata.channel_section is required to resolve per-channel "
            "rcv_apat_id / rcv_apc_id"
        )
    sc = metadata.scene_coordinates
    if (sc is None or sc.reference_surface is None
            or sc.reference_surface.planar is None
            or sc.reference_surface.planar.u_iax is None):
        raise ValueError(
            "metadata.scene_coordinates.reference_surface.planar.u_iax "
            "is required for the ground cross-range axis"
        )

    rcv_pos_ecf = np.asarray(rcv_pos_ecf, dtype=np.float64)
    srp_ecf = np.asarray(srp_ecf, dtype=np.float64)
    if rcv_pos_ecf.ndim != 2 or rcv_pos_ecf.shape[1] != 3:
        raise ValueError(
            f"rcv_pos_ecf must be shape (Nc, 3); got {rcv_pos_ecf.shape}"
        )
    if srp_ecf.shape != (3,):
        raise ValueError(
            f"srp_ecf must be shape (3,); got {srp_ecf.shape}"
        )

    Nc = metadata.num_channels
    if rcv_pos_ecf.shape[0] != Nc:
        raise ValueError(
            f"rcv_pos_ecf has {rcv_pos_ecf.shape[0]} rows but metadata "
            f"reports {Nc} channels"
        )
    if not 0 <= ref_channel < Nc:
        raise ValueError(
            f"ref_channel={ref_channel} is out of range for {Nc} channels"
        )

    u_iax_raw = np.asarray(
        sc.reference_surface.planar.u_iax, dtype=np.float64,
    )

    # Ground plane: default = local horizontal at SRP (normal = SRP-radial).
    # The CPHD planar reference surface (u_iax, u_iay) is often the slant
    # plane, not a ground plane, so we don't use u_iax x u_iay here.
    if ground_plane_normal is None:
        srp_norm = np.linalg.norm(srp_ecf)
        if srp_norm == 0.0:
            raise ValueError("srp_ecf has zero norm; cannot derive local up")
        plane_normal = srp_ecf / srp_norm
    else:
        plane_normal = np.asarray(ground_plane_normal, dtype=np.float64)
        n_norm = np.linalg.norm(plane_normal)
        if n_norm == 0.0:
            raise ValueError("ground_plane_normal has zero norm")
        plane_normal = plane_normal / n_norm

    # Cross-range direction on the ground plane: u_iax projected onto plane.
    u_iax = u_iax_raw - (u_iax_raw @ plane_normal) * plane_normal
    u_iax_norm = np.linalg.norm(u_iax)
    if u_iax_norm == 0.0:
        raise ValueError(
            "u_iax is parallel to ground_plane_normal; cannot define "
            "cross-range axis"
        )
    u_iax = u_iax / u_iax_norm

    patterns_by_id = {p.identifier: p for p in metadata.antenna.ant_patterns}
    apcs_by_id = {a.identifier: a for a in metadata.antenna.ant_phase_centers}
    acfs_by_id = {a.identifier: a for a in metadata.antenna.ant_coord_frames}

    def _channel_axes(ch_idx: int):
        ch_par = metadata.channel_section.parameters[ch_idx]
        if (ch_par.antenna is None
                or ch_par.antenna.rcv_apat_id is None
                or ch_par.antenna.rcv_apc_id is None):
            raise ValueError(
                f"channel {ch_idx} has no rcv_apat_id / rcv_apc_id"
            )
        pat = patterns_by_id.get(ch_par.antenna.rcv_apat_id)
        apc = apcs_by_id.get(ch_par.antenna.rcv_apc_id)
        if pat is None or apc is None:
            raise ValueError(
                f"channel {ch_idx} references missing antenna IDs "
                f"(rcv_apat_id={ch_par.antenna.rcv_apat_id!r}, "
                f"rcv_apc_id={ch_par.antenna.rcv_apc_id!r})"
            )
        acf = acfs_by_id.get(apc.acf_id)
        if acf is None:
            raise ValueError(
                f"channel {ch_idx} references missing acf_id={apc.acf_id!r}"
            )
        x_hat = _eval_axis_poly(acf.x_axis_poly, pulse_time)
        x_hat = x_hat / np.linalg.norm(x_hat)
        y_hat = _eval_axis_poly(acf.y_axis_poly, pulse_time)
        y_hat = y_hat / np.linalg.norm(y_hat)
        return pat, apc, x_hat, y_hat

    # ----- Reference channel: build the DCx scan around its EB -----
    pat0, _, x0, y0 = _channel_axes(ref_channel)
    z0 = np.cross(x0, y0)
    z0 = z0 / np.linalg.norm(z0)

    eb_dcx0 = (
        float(np.asarray(pat0.eb.dcx_poly)[0])
        if pat0.eb is not None and pat0.eb.dcx_poly is not None else 0.0
    )
    eb_dcy0 = (
        float(np.asarray(pat0.eb.dcy_poly)[0])
        if pat0.eb is not None and pat0.eb.dcy_poly is not None else 0.0
    )
    dcx_scan = np.linspace(
        eb_dcx0 - dcx_extent, eb_dcx0 + dcx_extent, n_scan,
    )
    dcy_scan = eb_dcy0

    # ----- Build ECF directions and ray-trace to the planar surface -----
    # Reference-channel direction vectors at each scan DCx (DCy fixed).
    dcz_sq = 1.0 - dcx_scan ** 2 - dcy_scan ** 2
    if np.any(dcz_sq < 0):
        raise ValueError(
            "scan (DCx, DCy) exceeds unit-sphere bounds; reduce dcx_extent"
        )
    dcz_scan = np.sqrt(dcz_sq)
    # direction[k] = DCx*x0 + DCy*y0 + DCz*z0
    direction = (
        dcx_scan[:, None] * x0[None, :]
        + dcy_scan * y0[None, :]
        + dcz_scan[:, None] * z0[None, :]
    )                                                      # (n_scan, 3)
    apc0 = rcv_pos_ecf[ref_channel]
    ground_pts = _ray_plane_intercept(
        apc0, direction, srp_ecf, plane_normal,
    )                                                      # (n_scan, 3)
    xr_offsets = (ground_pts - srp_ecf[None, :]) @ u_iax    # (n_scan,)

    # Boresight ground intercept (DCx = EB.dcx, DCy = EB.dcy):
    bs_dir = (
        eb_dcx0 * x0
        + eb_dcy0 * y0
        + np.sqrt(max(0.0, 1.0 - eb_dcx0 ** 2 - eb_dcy0 ** 2)) * z0
    )
    bs_intercept = _ray_plane_intercept(
        apc0, bs_dir[None, :], srp_ecf, plane_normal,
    )[0]
    boresight_xr = float((bs_intercept - srp_ecf) @ u_iax)

    # ----- Per-channel pattern evaluation -----
    s = np.empty((Nc, n_scan), dtype=np.complex128)
    gain_dB_all = np.empty((Nc, n_scan), dtype=np.float64)
    for c in range(Nc):
        pat_c, _, x_c, y_c = _channel_axes(c)
        apc_c = rcv_pos_ecf[c]
        r = ground_pts - apc_c[None, :]                    # (n_scan, 3)
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)
        r_norm = np.where(r_norm > 0, r_norm, 1.0)
        r_hat = r / r_norm
        dcx_c = r_hat @ x_c
        dcy_c = r_hat @ y_c

        gain_db = np.zeros(n_scan, dtype=np.float64)
        phase_rad = np.zeros(n_scan, dtype=np.float64)
        for src in (pat_c.array, pat_c.element):
            if src is None:
                continue
            gain_db = gain_db + _polyval2d(src.gain_poly, dcx_c, dcy_c)
            phase_rad = phase_rad + _polyval2d(src.phase_poly, dcx_c, dcy_c)

        s[c] = (10.0 ** (gain_db / 20.0)) * np.exp(1j * phase_rad)
        gain_dB_all[c] = gain_db

    # Phase-anchor at ref_channel so its phase is real-positive
    ref = s[ref_channel]
    ref_mag = np.abs(ref)
    phase_corr = np.where(
        ref_mag > 0,
        np.conj(ref) / np.where(ref_mag > 0, ref_mag, 1.0),
        1.0,
    )
    s = s * phase_corr[None, :]

    # L2-normalise columns
    l2 = np.linalg.norm(s, axis=0, keepdims=True)
    l2 = np.where(l2 > 0, l2, 1.0)
    s = s / l2

    return CPHDMetadataSteering(
        s=s,
        dcx_scan=dcx_scan,
        dcy_scan=float(dcy_scan),
        ground_pts_ecf=ground_pts,
        xr_offsets_m=xr_offsets,
        boresight_xr_offset_m=boresight_xr,
        gain_dB=gain_dB_all,
    )
