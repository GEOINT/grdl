"""
Metadata-derived steering matrix from a CPHD ``Antenna`` section.

The CPHD standard stores per-channel receive antenna patterns in
**antenna coordinates** -- a complex aperture response

    G_c(DCx, DCy) = 10**(gain_dB_c(DCx, DCy) / 20)
                    * exp(j * 2*pi * phase_cycles_c(DCx, DCy))

with

    gain_dB_c       = Array.GainPoly  + Element.GainPoly       # dB
    phase_cycles_c  = Array.PhasePoly + Element.PhasePoly      # cycles

evaluated at antenna direction cosines ``(DCx, DCy)`` measured in the
channel's Antenna Coordinate Frame (ACF). The pattern is therefore
"at the antenna" -- to use it as a per-channel steering dictionary on
the ground, it has to be **projected** to scene coordinates.

This module does the projection, indexing the result directly by the
**image cross-range bin**. The caller supplies a set of cross-range
offsets from the SRP (in metres, along the planar reference surface
``u_iax``); each offset defines a scene point on the reference surface,
the per-channel antenna pattern is evaluated at the look direction from
the (shared) reference-channel APC to that scene point, and the result
is a steering matrix whose columns line up one-to-one with the image
cross-range columns the STAP cube was formed on. That alignment is what
lets the matrix serve as the column dictionary for a metadata-driven
STAP detector.

Polynomial input convention
---------------------------
The CPHD spec (§8.2.3) defines the gain polynomial as gain in direction
``(DCX, DCY)`` relative to mechanical boresight, with the EB applying an
electronic shift. The natural mathematical input is therefore the
look-direction direction-cosines *relative to the polynomial's reference
point*. Two conventions are in use in practice:

* EB-relative (textbook):
  ``δDCX = look·uACX - EB_DCX(t)``, ``δDCY = look·uACY - EB_DCY(t)``.
  Correct when the simulator stored the polynomial centred on the EB
  direction.

* SRP-relative (this builder):
  ``δDCX = look·uACX - LOS_at_SRP·uACX``,
  ``δDCY = look·uACY - LOS_at_SRP·uACY``.
  Correct when the simulator stored the polynomial centred on the SRP
  look direction (the actual imaged scene anchor), regardless of where
  the EB is steered. Aspen-style simulators typically do this.

The two reduce to the same thing when the SRP is at the antenna
boresight footprint (``LOS_at_SRP ≈ EB direction``), which is the common
case. They diverge when the SRP is not at boresight; for those datasets
(e.g. wide-area collections where the SRP is just an image anchor), only
SRP-relative produces polynomial inputs inside the polynomial's mainlobe
domain.

Per-channel differences are often encoded in the ``DCY`` coefficients,
so both ``δDCX`` and ``δDCY`` are passed to the polynomial (full 2-D
evaluation). Some simulators additionally use the axis order
``exponent1 = DCY power``, ``exponent2 = DCX power`` -- opposite the
spec's literal reading -- which places the channel-distinguishing
coefficients along the axis the cross-range scan actually sweeps. The
``swap_poly_axes`` flag (default ``True``) selects that convention.

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
2026-06-17
"""

# Standard library
from collections.abc import Sequence
from dataclasses import dataclass

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
    s : np.ndarray, shape ``(Nc, n_xr)``, complex128
        Column-normalised steering matrix. Each column is the
        per-channel complex response to a target sitting at the scene
        cross-range offset of the corresponding image column. Columns
        are L2-unit-norm across channels, matching the convention of a
        data-estimated steering matrix so the two can be overlaid.
    delta_dcx : np.ndarray, shape ``(n_xr,)``, float64
        SRP-relative ``DCX`` look-direction cosine at each scene column,
        in the reference channel's ACF.
    delta_dcy : np.ndarray, shape ``(n_xr,)``, float64
        SRP-relative ``DCY`` look-direction cosine at each scene column.
    xr_offsets_m : np.ndarray, shape ``(n_xr,)``, float64
        The scene cross-range offsets from SRP (metres) that were
        supplied by the caller, echoed back.
    scene_pts_ecf : np.ndarray, shape ``(n_xr, 3)``, float64
        ECF position of each scene point on the reference surface.
    gain_dB : np.ndarray, shape ``(Nc, n_xr)``, float64
        Per-channel gain in dB at each column, before column
        normalisation. Useful for sanity-checking that the polynomial
        was evaluated inside its design domain.
    dcx_at_srp : float
        ``DCX`` of the look direction to the SRP (the polynomial origin).
    dcy_at_srp : float
        ``DCY`` of the look direction to the SRP.
    """

    s: np.ndarray
    delta_dcx: np.ndarray
    delta_dcy: np.ndarray
    xr_offsets_m: np.ndarray
    scene_pts_ecf: np.ndarray
    gain_dB: np.ndarray
    dcx_at_srp: float
    dcy_at_srp: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _eval_axis_poly(coefs: np.ndarray | None, t: float) -> np.ndarray:
    """Evaluate a CPHD axis polynomial ``(K, 3)`` at scalar time ``t``.

    Returns the polynomial value ``sum_k coefs[k] * t**k`` as a 3-vector.
    """
    if coefs is None:
        raise ValueError("axis polynomial is None")
    return np.polynomial.polynomial.polyval(
        t, np.asarray(coefs, dtype=np.float64),
    )


def _polyval2d(
    poly: np.ndarray | None,
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
    xr_offsets_m: np.ndarray,
    *,
    eval_time: float = 0.0,
    ref_channel: int = 0,
    channel_indices: Sequence[int] | None = None,
    apply_apc_offset: bool = True,
    swap_poly_axes: bool = True,
    verbose: bool = False,
) -> CPHDMetadataSteering:
    """Per-channel steering matrix from CPHD metadata, projected to scene.

    Per-channel antenna **beam intensity** projected to scene
    cross-range. For a squinted multi-channel array (e.g. left / centre /
    right beams), each channel has its own ``GainPoly``; when those
    polynomials peak at offset ``DCY`` values relative to the look
    direction at the SRP, the beams land at different cross-range bins on
    the ground.

    The scene points are ``SRP + xr_offsets_m[k] * u_iax`` on the planar
    reference surface. A single shared origin -- the look direction from
    the reference channel's APC to the SRP -- defines the polynomial's
    ``(0, 0)``; each scene column's look direction is taken relative to
    it (``δDCX``, ``δDCY``). Every channel evaluates its own
    ``Array``/``Element`` gain & phase polynomials at that shared
    ``(δDCX, δDCY)`` trajectory. ``PhasePoly`` is in cycles → radians via
    ``2π``; no slant-range path-delay phase is added. Each cross-range
    column is L2-normalised across channels.

    Parameters
    ----------
    metadata : CPHDMetadata
        Must contain ``antenna`` (ant_coord_frames, ant_phase_centers,
        ant_patterns), ``channel_section`` (per-channel rcv_apc_id /
        rcv_apat_id refs), and
        ``scene_coordinates.reference_surface.planar.u_iax``.
    rcv_pos_ecf : np.ndarray, shape ``(Nc, 3)``
        Per-channel APC ECF position at the chosen pulse. Only the
        reference channel's row anchors the shared look-direction
        trajectory (this matches the validated GMTI convention, in which
        all channels share the reference APC); the other rows are
        accepted for shape symmetry.
    srp_ecf : np.ndarray, shape ``(3,)``
        SRP ECF at the chosen pulse (the scene anchor).
    xr_offsets_m : np.ndarray, shape ``(n_xr,)``
        Scene cross-range offsets from SRP along ``u_iax`` (metres), one
        per image cross-range column. The returned matrix columns align
        with these.
    eval_time : float
        Time argument (seconds since the CPHD collection reference) for
        evaluating ``ant_coord_frames[*].x_axis_poly`` / ``y_axis_poly``.
        Use the receive time of the reference channel's centre pulse.
    ref_channel : int
        Channel that supplies the shared APC, ACF, and SRP look direction
        that define the polynomial origin. Default 0.
    channel_indices : sequence of int, optional
        Maps local channel ``c`` -> the index into
        ``channel_section.parameters``. Required when the metadata
        enumerates more channels than this CPI uses (multi-CPI files).
        Defaults to identity.
    apply_apc_offset : bool
        Apply the reference APC's ``apc_xyz`` offset (in ACF axes) to the
        ECF APC before computing look directions. Default ``True``.
    swap_poly_axes : bool
        Evaluate gain/phase polynomials as ``polyval2d(δDCY, δDCX, poly)``
        (the axis order used by Aspen-style simulators) rather than
        ``polyval2d(δDCX, δDCY, poly)``. Default ``True``.
    verbose : bool
        Print per-channel diagnostics. Default ``False``.

    Returns
    -------
    CPHDMetadataSteering
        See :class:`CPHDMetadataSteering`.

    Raises
    ------
    ValueError
        On any missing required metadata section, missing per-channel
        reference, or non-conformant array shape.
    """
    if metadata.antenna is None or not metadata.antenna.ant_patterns:
        raise ValueError(
            "metadata.antenna is missing or has no ant_patterns -- "
            "cannot build metadata-derived steering."
        )
    if metadata.channel_section is None:
        raise ValueError(
            "metadata.channel_section is required to resolve per-channel "
            "rcv_apat_id / rcv_apc_id."
        )
    sc = metadata.scene_coordinates
    if (sc is None or sc.reference_surface is None
            or sc.reference_surface.planar is None
            or sc.reference_surface.planar.u_iax is None):
        raise ValueError(
            "metadata.scene_coordinates.reference_surface.planar.u_iax "
            "is required to project the pattern to scene cross-range."
        )

    u_iax = np.asarray(
        sc.reference_surface.planar.u_iax, dtype=np.float64,
    )
    u_iax = u_iax / np.linalg.norm(u_iax)

    rcv_pos_ecf = np.asarray(rcv_pos_ecf, dtype=np.float64)
    srp_ecf = np.asarray(srp_ecf, dtype=np.float64)
    xr_offsets_m = np.asarray(xr_offsets_m, dtype=np.float64)
    if rcv_pos_ecf.ndim != 2 or rcv_pos_ecf.shape[1] != 3:
        raise ValueError(
            f"rcv_pos_ecf must be shape (Nc, 3); got {rcv_pos_ecf.shape}"
        )
    if srp_ecf.shape != (3,):
        raise ValueError(f"srp_ecf must be shape (3,); got {srp_ecf.shape}")

    patterns_by_id = {p.identifier: p for p in metadata.antenna.ant_patterns}
    apcs_by_id = {a.identifier: a for a in metadata.antenna.ant_phase_centers}
    acfs_by_id = {a.identifier: a for a in metadata.antenna.ant_coord_frames}

    Nc = rcv_pos_ecf.shape[0]
    if channel_indices is not None:
        if len(channel_indices) != Nc:
            raise ValueError(
                f"channel_indices length {len(channel_indices)} does not "
                f"match rcv_pos_ecf channel count {Nc}"
            )
        global_idx = list(channel_indices)
    else:
        global_idx = list(range(Nc))

    if not 0 <= ref_channel < Nc:
        raise ValueError(
            f"ref_channel={ref_channel} out of range for Nc={Nc}."
        )

    n_xr = xr_offsets_m.shape[0]
    scene_pts = srp_ecf[None, :] + xr_offsets_m[:, None] * u_iax[None, :]

    # ---- Single shared ACF + APC + LOS-at-SRP from ref_channel. The
    # polynomial inputs are SRP-relative, so the geometry that defines
    # the "(0, 0)" of the polynomial is the look direction at the SRP --
    # one shared origin for all channels.
    ch_par_ref = metadata.channel_section.parameters[global_idx[ref_channel]]
    if (ch_par_ref.antenna is None
            or ch_par_ref.antenna.rcv_apc_id is None):
        raise ValueError(
            f"Reference channel {ref_channel} missing rcv_apc_id."
        )
    apc_ref_meta = apcs_by_id.get(ch_par_ref.antenna.rcv_apc_id)
    if apc_ref_meta is None:
        raise ValueError(
            f"Reference channel {ref_channel} references missing "
            f"rcv_apc_id={ch_par_ref.antenna.rcv_apc_id!r}."
        )
    acf_ref = acfs_by_id.get(apc_ref_meta.acf_id)
    if (acf_ref is None or acf_ref.x_axis_poly is None
            or acf_ref.y_axis_poly is None):
        raise ValueError(
            f"Reference channel {ref_channel} has no usable ACF "
            f"(acf_id={apc_ref_meta.acf_id!r})."
        )

    x_hat = _eval_axis_poly(acf_ref.x_axis_poly, eval_time)
    y_hat = _eval_axis_poly(acf_ref.y_axis_poly, eval_time)
    x_hat = x_hat / np.linalg.norm(x_hat)
    y_hat = y_hat / np.linalg.norm(y_hat)

    apc_ecf = rcv_pos_ecf[ref_channel].copy()
    if apply_apc_offset and apc_ref_meta.apc_xyz is not None:
        off = np.asarray(apc_ref_meta.apc_xyz, dtype=np.float64)
        if np.any(off != 0):
            z_hat = np.cross(x_hat, y_hat)
            z_hat = z_hat / np.linalg.norm(z_hat)
            apc_ecf = apc_ecf + (
                off[0] * x_hat + off[1] * y_hat + off[2] * z_hat
            )

    los_at_srp = srp_ecf - apc_ecf
    los_at_srp = los_at_srp / np.linalg.norm(los_at_srp)
    dcx_at_srp = float(los_at_srp @ x_hat)
    dcy_at_srp = float(los_at_srp @ y_hat)

    # Per-scene-bin look-direction cosines, then SRP-relative shift.
    r_vec = scene_pts - apc_ecf[None, :]                      # (n_xr, 3)
    r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r_norm = np.where(r_norm > 0, r_norm, 1.0)
    r_hat = r_vec / r_norm
    delta_dcx = (r_hat @ x_hat) - dcx_at_srp                  # (n_xr,)
    delta_dcy = (r_hat @ y_hat) - dcy_at_srp                  # (n_xr,)

    if verbose:
        print(
            "  [build_steering_matrix_from_cphd_metadata] SRP-relative "
            "polynomial input:"
        )
        print(
            f"    DCX_at_SRP={dcx_at_srp:+.5f}   DCY_at_SRP={dcy_at_srp:+.5f}"
        )
        print(
            f"    δDCX scan:  [{delta_dcx.min():+.5f}, {delta_dcx.max():+.5f}]"
            f"   span={delta_dcx.max() - delta_dcx.min():+.5f}"
        )
        print(
            f"    δDCY scan:  [{delta_dcy.min():+.5f}, {delta_dcy.max():+.5f}]"
            f"   span={delta_dcy.max() - delta_dcy.min():+.5f}"
        )

    # ---- Per-channel: each channel evaluates its own GainPoly /
    # PhasePoly at the SHARED (δDCX, δDCY) trajectory. PhasePoly is in
    # cycles → radians via 2π. No slant-range phase added.
    if swap_poly_axes:
        ax0, ax1 = delta_dcy, delta_dcx
    else:
        ax0, ax1 = delta_dcx, delta_dcy

    s = np.empty((Nc, n_xr), dtype=np.complex128)
    gain_dB_all = np.empty((Nc, n_xr), dtype=np.float64)
    diag: list = []
    for c in range(Nc):
        ch_par = metadata.channel_section.parameters[global_idx[c]]
        if (ch_par.antenna is None
                or ch_par.antenna.rcv_apat_id is None):
            raise ValueError(
                f"Channel {c} missing rcv_apat_id; cannot build steering "
                f"from metadata."
            )
        pat_c = patterns_by_id.get(ch_par.antenna.rcv_apat_id)
        if pat_c is None:
            raise ValueError(
                f"Channel {c} references missing rcv_apat_id="
                f"{ch_par.antenna.rcv_apat_id!r}."
            )

        gain_db = np.zeros(n_xr, dtype=np.float64)
        phase_cycles = np.zeros(n_xr, dtype=np.float64)
        for src in (pat_c.array, pat_c.element):
            if src is None:
                continue
            if src.gain_poly is not None:
                gain_db = gain_db + _polyval2d(src.gain_poly, ax0, ax1)
            if src.phase_poly is not None:
                phase_cycles = phase_cycles + _polyval2d(
                    src.phase_poly, ax0, ax1,
                )

        s[c] = (
            (10.0 ** (gain_db / 20.0))
            * np.exp(1j * 2.0 * np.pi * phase_cycles)
        )
        gain_dB_all[c] = gain_db
        diag.append((
            float(gain_db.min()), float(gain_db.max()),
            int(np.argmax(gain_db)),
        ))

    if verbose:
        for c in range(Nc):
            gd_lo, gd_hi, peak_bin = diag[c]
            print(
                f"    ch {c}: gain_dB=[{gd_lo:+.2f}, {gd_hi:+.2f}]   "
                f"peak at bin {peak_bin} / {n_xr - 1}   "
                f"scene xr offset = {xr_offsets_m[peak_bin]:+.0f} m"
            )

    # Per-cross-range-bin L2 normalisation: each column ``s[:, k]`` is
    # unit-norm across channels.
    l2 = np.linalg.norm(s, axis=0, keepdims=True)
    l2 = np.where(l2 > 0, l2, 1.0)
    s = s / l2

    return CPHDMetadataSteering(
        s=s,
        delta_dcx=delta_dcx,
        delta_dcy=delta_dcy,
        xr_offsets_m=xr_offsets_m,
        scene_pts_ecf=scene_pts,
        gain_dB=gain_dB_all,
        dcx_at_srp=dcx_at_srp,
        dcy_at_srp=dcy_at_srp,
    )
