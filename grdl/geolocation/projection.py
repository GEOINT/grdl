# -*- coding: utf-8 -*-
"""
R/Rdot Projection Engine - Native SICD Volume 3 geolocation.

Implements the image-to-ground and ground-to-image projection algorithms
from NGA.STND.0024-1 Volume 3 (SICD Image Projections Description).
This is a standalone, backend-free implementation that operates entirely
on GRDL metadata types -- no sarpy or sarkit dependency required.

The pipeline converts image pixel coordinates to slant range (R) and
range rate (Rdot), then intersects the R/Rdot contour with a reference
surface (constant HAE plane or DEM) to produce ground coordinates.

Key Classes
-----------
COAProjection
    Central hub: image pixel → (R, Rdot, time_coa, arp, varp).
    Dispatches to formation-specific projectors (PFA, INCA, RgAzComp,
    PLANE) based on the image formation algorithm.

Key Functions
-------------
image_to_ground_plane
    R/Rdot contour intersection with an arbitrary plane.
image_to_ground_hae
    Iterative projection to constant height above WGS-84 ellipsoid.
ground_to_image
    Inverse: ECF ground point → image pixel coordinates.

Dependencies
------------
scipy (for iterative convergence only)

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
2026-03-16

Modified
--------
2026-04-17  Enforce RgAzComp validation; derive AzSF from KazPoly fallback.
2026-03-31  Add nan_fill_height parameter to image_to_ground_hae.
2026-03-27  Add numba dispatch for image_to_ground_plane and wgs84_norm.
2026-03-22  Update coordinate function calls to (N, M) stacked convention.
2026-03-16
"""

# Standard library
from typing import Callable, List, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.common import Poly1D, Poly2D, XYZ, XYZPoly
from grdl.geolocation.coordinates import (
    WGS84_A,
    WGS84_B,
    WGS84_E1_SQ,
    ecef_to_geodetic,
    geodetic_to_ecef,
)

# Optional numba acceleration for R/Rdot algebra
try:
    from grdl.geolocation._numba_projection import (
        image_to_ground_plane_fast as _itgp_fast,
        wgs84_norm_fast as _wgs84_norm_fast,
    )
    _HAS_NUMBA_PROJ = True
except ImportError:
    _HAS_NUMBA_PROJ = False


# ── WGS-84 helpers ───────────────────────────────────────────────────


def wgs84_norm(ecf: np.ndarray) -> np.ndarray:
    """Unit normal to the WGS-84 ellipsoid at an ECF point.

    Parameters
    ----------
    ecf : np.ndarray
        ECF coordinates, shape ``(3,)`` or ``(N, 3)``.

    Returns
    -------
    np.ndarray
        Unit normal vector(s), same shape as input.
    """
    ecf = np.asarray(ecf, dtype=np.float64)

    # Try numba-accelerated path for large batches
    if _HAS_NUMBA_PROJ:
        fast = _wgs84_norm_fast(ecf)
        if fast is not None:
            return fast

    scale = np.array([1.0 / WGS84_A ** 2,
                      1.0 / WGS84_A ** 2,
                      1.0 / WGS84_B ** 2])
    n = ecf * scale
    mag = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / mag


# ── Formation-specific R/Rdot projectors ─────────────────────────────


def _pfa_projector(
    scp_ecf: np.ndarray,
    polar_ang_poly: Poly1D,
    spatial_freq_sf_poly: Poly1D,
    polar_ang_ref_time: float,
    row_ss: float,
    col_ss: float,
    row_sgn: int,
    col_sgn: int,
    row_kctr: float,
    col_kctr: float,
) -> Callable:
    """Build PFA-specific image-to-R/Rdot closure.

    Implements the PFA projection per SICD Volume 3, Section 4.1,
    matching sarpy's ``pfa_projection`` algorithm exactly.

    The row_transform and col_transform (in meters) are rotated into
    the polar aperture (Ka, Kc) frame using the polar angle at COA
    time.  Range and range-rate offsets from SCP are computed via
    the spatial frequency scale factor and its derivative.
    """
    # Pre-compute derivative polynomials
    polar_ang_poly_der = polar_ang_poly.derivative(1)
    spatial_freq_sf_poly_der = spatial_freq_sf_poly.derivative(1)

    def project(
        row_t: np.ndarray,
        col_t: np.ndarray,
        time_coa: np.ndarray,
        arp_coa: np.ndarray,
        varp_coa: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # R and Rdot at SCP
        arp_minus_scp = arp_coa - scp_ecf
        r_scp = np.linalg.norm(arp_minus_scp, axis=-1)
        rdot_scp = np.sum(varp_coa * arp_minus_scp, axis=-1) / r_scp

        # Polar angle and derivatives at COA time
        theta = polar_ang_poly(time_coa)
        dtheta_dt = polar_ang_poly_der(time_coa)

        # Spatial frequency scale factor and derivative wrt polar angle
        k_sf = spatial_freq_sf_poly(theta)
        dk_sf_dtheta = spatial_freq_sf_poly_der(theta)

        # Rotate (row, col) transforms into polar aperture frame
        # dPhiDKa: phase slope along aperture (range-like)
        # dPhiDKc: phase slope cross aperture (azimuth-like)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        dphi_dka = row_t * cos_theta + col_t * sin_theta
        dphi_dkc = -row_t * sin_theta + col_t * cos_theta

        # Range offset from SCP
        delta_r = k_sf * dphi_dka

        # Range-rate offset from SCP:
        # d(deltaR)/dtheta * dtheta/dt
        d_delta_r_dtheta = dk_sf_dtheta * dphi_dka + k_sf * dphi_dkc
        delta_rdot = d_delta_r_dtheta * dtheta_dt

        return r_scp + delta_r, rdot_scp + delta_rdot

    return project


def _inca_projector(
    time_ca_poly: Poly1D,
    r_ca_scp: float,
    d_rate_sf_poly: Poly2D,
) -> Callable:
    """Build INCA (RMA) image-to-R/Rdot closure.

    Parameters follow SICD Volume 3, Section 4.2.
    """
    def project(
        row_t: np.ndarray,
        col_t: np.ndarray,
        time_coa: np.ndarray,
        arp_coa: np.ndarray,
        varp_coa: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Range at closest approach
        r_ca = r_ca_scp + row_t

        # Time of closest approach
        t_ca = time_ca_poly(col_t)

        # Velocity magnitude squared
        vel2 = np.sum(varp_coa ** 2, axis=-1)

        # Doppler Rate Scale Factor
        drsf = d_rate_sf_poly(row_t, col_t)

        # Time offset from closest approach
        dt = time_coa - t_ca

        # Range at COA: R² = R_CA² + DRSF * v² * dt²
        r = np.sqrt(r_ca ** 2 + drsf * vel2 * dt ** 2)

        # Range rate at COA
        rdot = (drsf / r) * vel2 * dt

        return r, rdot

    return project


def _rgazcomp_projector(
    scp_ecf: np.ndarray,
    az_sf: float,
) -> Callable:
    """Build RgAzComp image-to-R/Rdot closure.

    Parameters follow SICD Volume 3, Section 4.3.
    """
    def project(
        row_t: np.ndarray,
        col_t: np.ndarray,
        time_coa: np.ndarray,
        arp_coa: np.ndarray,
        varp_coa: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # R and Rdot at SCP
        delta_arp = arp_coa - scp_ecf
        r_scp = np.linalg.norm(delta_arp, axis=-1)
        u_rng = delta_arp / r_scp[..., np.newaxis]
        rdot_scp = np.sum(varp_coa * u_rng, axis=-1)

        # Range offset = range coordinate directly
        delta_r = row_t

        # Rdot offset via azimuth scale factor
        vel_mag = np.linalg.norm(varp_coa, axis=-1)
        delta_rdot = -vel_mag * az_sf * col_t

        return r_scp + delta_r, rdot_scp + delta_rdot

    return project


def _plane_projector(
    scp_ecf: np.ndarray,
    u_row: np.ndarray,
    u_col: np.ndarray,
    row_ss: float,
    col_ss: float,
) -> Callable:
    """Build PLANE-type image-to-R/Rdot closure.

    Used for XRGYCR, XCTYAT, and PLANE grid types, and also for
    SIDD PlaneProjection R/Rdot refinement.
    """
    def project(
        row_t: np.ndarray,
        col_t: np.ndarray,
        time_coa: np.ndarray,
        arp_coa: np.ndarray,
        varp_coa: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Image Plane Point in ECF
        # row_t and col_t arrive already in meters from COAProjection
        ipp = (scp_ecf
               + np.outer(row_t, u_row)
               + np.outer(col_t, u_col))

        # Slant range: ||ARP - IPP||
        delta = arp_coa - ipp
        r = np.linalg.norm(delta, axis=-1)

        # Range rate: dot(VARP, ARP - IPP) / R
        rdot = np.sum(varp_coa * delta, axis=-1) / r

        return r, rdot

    return project


# ── COAProjection ────────────────────────────────────────────────────


class COAProjection:
    """Image pixel to R/Rdot projection via Center of Aperture geometry.

    This class encapsulates the SICD Volume 3 projection model.  Given
    image pixel coordinates it computes:

    - ``time_coa`` — center-of-aperture time (seconds from collection start)
    - ``arp_coa`` — aperture reference point position at COA (ECF meters)
    - ``varp_coa`` — aperture velocity at COA (ECF m/s)
    - ``r`` — slant range to target at COA (meters)
    - ``rdot`` — range rate at COA (m/s)

    Parameters
    ----------
    time_coa_poly : Poly2D
        COA time as function of image coordinates.
    arp_poly : XYZPoly
        ARP position polynomial in ECF as function of time.
    method_projection : callable
        Formation-specific projector (PFA, INCA, RgAzComp, or PLANE).
    scp_pixel : tuple of float
        (row, col) of the Scene Center Point in the image.
    row_ss : float
        Row sample spacing (meters).
    col_ss : float
        Column sample spacing (meters).
    first_row : int
        First row offset for sub-images.
    first_col : int
        First column offset for sub-images.
    delta_arp : np.ndarray, optional
        ARP position correction in ECF (meters), shape ``(3,)``.
    delta_varp : np.ndarray, optional
        ARP velocity correction in ECF (m/s), shape ``(3,)``.
    range_bias : float
        Range bias correction (meters).
    """

    def __init__(
        self,
        time_coa_poly: Poly2D,
        arp_poly: XYZPoly,
        method_projection: Callable,
        scp_pixel: Tuple[float, float],
        row_ss: float,
        col_ss: float,
        first_row: int = 0,
        first_col: int = 0,
        delta_arp: Optional[np.ndarray] = None,
        delta_varp: Optional[np.ndarray] = None,
        range_bias: float = 0.0,
    ) -> None:
        self._time_coa_poly = time_coa_poly
        self._arp_poly = arp_poly
        self._vel_poly = arp_poly.derivative(1)
        self._method_projection = method_projection
        self._scp_row = scp_pixel[0]
        self._scp_col = scp_pixel[1]
        self._row_ss = row_ss
        self._col_ss = col_ss
        self._first_row = first_row
        self._first_col = first_col
        self._delta_arp = (delta_arp if delta_arp is not None
                           else np.zeros(3))
        self._delta_varp = (delta_varp if delta_varp is not None
                            else np.zeros(3))
        self._range_bias = range_bias

    def projection(
        self, im_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """Project image coordinates to R/Rdot space.

        Parameters
        ----------
        im_points : np.ndarray
            Image coordinates, shape ``(N, 2)`` as ``[row, col]``.

        Returns
        -------
        r : np.ndarray
            Slant range at COA (meters), shape ``(N,)``.
        rdot : np.ndarray
            Range rate at COA (m/s), shape ``(N,)``.
        time_coa : np.ndarray
            Center of aperture time (seconds), shape ``(N,)``.
        arp_coa : np.ndarray
            ARP position at COA (ECF meters), shape ``(N, 3)``.
        varp_coa : np.ndarray
            ARP velocity at COA (ECF m/s), shape ``(N, 3)``.
        """
        im_points = np.atleast_2d(im_points)
        rows = im_points[:, 0].astype(np.float64)
        cols = im_points[:, 1].astype(np.float64)

        # Transform to image coordinates relative to SCP
        # Account for sub-image offset (FirstRow/FirstCol)
        row_t = (rows + self._first_row - self._scp_row) * self._row_ss
        col_t = (cols + self._first_col - self._scp_col) * self._col_ss

        # COA time from TimeCOAPoly (evaluated at transformed coords)
        # TimeCOAPoly is defined over (row_transform, col_transform)
        # in sample-spacing units, so divide back out
        row_poly = rows + self._first_row - self._scp_row
        col_poly = cols + self._first_col - self._scp_col
        time_coa = self._time_coa_poly(row_poly, col_poly)

        # ARP position and velocity at COA time
        arp_coa = self._arp_poly(time_coa) + self._delta_arp
        varp_coa = self._vel_poly(time_coa) + self._delta_varp

        # Formation-specific R/Rdot projection
        r, rdot = self._method_projection(
            row_t, col_t, time_coa, arp_coa, varp_coa,
        )

        # Apply range bias
        r = r + self._range_bias

        return r, rdot, time_coa, arp_coa, varp_coa

    @classmethod
    def from_sicd(
        cls,
        metadata: 'SICDMetadata',
        delta_arp: Optional[np.ndarray] = None,
        delta_varp: Optional[np.ndarray] = None,
        range_bias: float = 0.0,
    ) -> 'COAProjection':
        """Construct from a GRDL SICDMetadata object.

        Selects the correct formation-specific projector based on
        ``ImageFormation.image_form_algo`` and ``Grid.type``.

        Parameters
        ----------
        metadata : SICDMetadata
            Fully populated SICD metadata.
        delta_arp : np.ndarray, optional
            ARP position correction (ECF meters).
        delta_varp : np.ndarray, optional
            ARP velocity correction (ECF m/s).
        range_bias : float
            Range bias correction (meters).

        Returns
        -------
        COAProjection
        """
        # Validate required sections
        grid = metadata.grid
        pos = metadata.position
        img = metadata.image_data
        geo = metadata.geo_data

        if grid is None or grid.time_coa_poly is None:
            raise ValueError("Grid.TimeCOAPoly is required")
        if pos is None or pos.arp_poly is None:
            raise ValueError("Position.ARPPoly is required")
        if img is None or img.scp_pixel is None:
            raise ValueError("ImageData.SCPPixel is required")
        if geo is None or geo.scp is None or geo.scp.ecf is None:
            raise ValueError("GeoData.SCP.ECF is required")

        scp_ecf = geo.scp.ecf.to_array()
        scp_pixel = (img.scp_pixel.row, img.scp_pixel.col)
        row_ss = grid.row.ss if grid.row else 1.0
        col_ss = grid.col.ss if grid.col else 1.0
        first_row = img.first_row if img.first_row else 0
        first_col = img.first_col if img.first_col else 0

        # Determine formation-specific projector
        algo = None
        if metadata.image_formation is not None:
            algo = metadata.image_formation.image_form_algo

        grid_type = grid.type if grid.type else ''

        if algo == 'PFA' and metadata.pfa is not None:
            pfa = metadata.pfa
            if (pfa.polar_ang_poly is None
                    or pfa.spatial_freq_sf_poly is None):
                raise ValueError(
                    "PFA.PolarAngPoly and SpatialFreqSFPoly required")
            row_sgn = grid.row.sgn if (grid.row and grid.row.sgn) else 1
            col_sgn = grid.col.sgn if (grid.col and grid.col.sgn) else 1
            row_kctr = grid.row.k_ctr if (grid.row and grid.row.k_ctr) else 0
            col_kctr = grid.col.k_ctr if (grid.col and grid.col.k_ctr) else 0
            projector = _pfa_projector(
                scp_ecf=scp_ecf,
                polar_ang_poly=pfa.polar_ang_poly,
                spatial_freq_sf_poly=pfa.spatial_freq_sf_poly,
                polar_ang_ref_time=(pfa.polar_ang_ref_time
                                    if pfa.polar_ang_ref_time else 0.0),
                row_ss=row_ss,
                col_ss=col_ss,
                row_sgn=row_sgn,
                col_sgn=col_sgn,
                row_kctr=row_kctr,
                col_kctr=col_kctr,
            )
        elif (grid_type.upper() == 'RGZERO'
              and metadata.rma is not None
              and metadata.rma.inca is not None):
            inca = metadata.rma.inca
            if (inca.time_ca_poly is None
                    or inca.r_ca_scp is None
                    or inca.d_rate_sf_poly is None):
                raise ValueError(
                    "RMA.INCA: TimeCAPoly, R_CA_SCP, DRateSFPoly "
                    "required")
            projector = _inca_projector(
                time_ca_poly=inca.time_ca_poly,
                r_ca_scp=inca.r_ca_scp,
                d_rate_sf_poly=inca.d_rate_sf_poly,
            )
        elif algo == 'RGAZCOMP':
            # Per SICD Vol3 §4.3: RgAzComp requires AzSF (or KazPoly from
            # which AzSF can be evaluated at the SCP COA time).
            if metadata.rg_az_comp is None:
                raise ValueError(
                    "ImageFormation.ImageFormAlgo='RGAZCOMP' but "
                    "RgAzComp metadata section is missing")
            az_sf = metadata.rg_az_comp.az_sf
            if az_sf is None and metadata.rg_az_comp.kaz_poly is not None:
                # KazPoly is evaluated at SCP COA time; AzSF is its value
                # (SICD Vol3 §4.3 / Vol1 §15: cos(DCA_COA) = AzSF * az).
                # TimeCOAPoly is defined over (row_offset, col_offset)
                # relative to SCP, so at the SCP the offsets are (0, 0).
                t_scp_coa = float(grid.time_coa_poly(0.0, 0.0))
                az_sf = float(metadata.rg_az_comp.kaz_poly(t_scp_coa))
            if az_sf is None:
                raise ValueError(
                    "RgAzComp requires AzSF or KazPoly per SICD Vol3 §4.3")
            projector = _rgazcomp_projector(
                scp_ecf=scp_ecf,
                az_sf=az_sf,
            )
        else:
            # PLANE / XRGYCR / XCTYAT or fallback
            u_row = (grid.row.uvect_ecf.to_array()
                     if grid.row and grid.row.uvect_ecf else
                     np.array([1.0, 0.0, 0.0]))
            u_col = (grid.col.uvect_ecf.to_array()
                     if grid.col and grid.col.uvect_ecf else
                     np.array([0.0, 1.0, 0.0]))
            projector = _plane_projector(
                scp_ecf=scp_ecf,
                u_row=u_row,
                u_col=u_col,
                row_ss=row_ss,
                col_ss=col_ss,
            )

        return cls(
            time_coa_poly=grid.time_coa_poly,
            arp_poly=pos.arp_poly,
            method_projection=projector,
            scp_pixel=scp_pixel,
            row_ss=row_ss,
            col_ss=col_ss,
            first_row=first_row,
            first_col=first_col,
            delta_arp=delta_arp,
            delta_varp=delta_varp,
            range_bias=range_bias,
        )

    @classmethod
    def from_sidd(
        cls,
        metadata: 'SIDDMetadata',
    ) -> 'COAProjection':
        """Construct from a GRDL SIDDMetadata object.

        Uses the Measurement block's TimeCOAPoly and ARPPoly with
        a PLANE-type projector for R/Rdot refinement of grid-based
        SIDD geolocation.

        Parameters
        ----------
        metadata : SIDDMetadata
            Fully populated SIDD metadata.

        Returns
        -------
        COAProjection
        """
        meas = metadata.measurement
        if meas is None:
            raise ValueError("SIDD Measurement block is required")
        if meas.arp_poly is None:
            raise ValueError("Measurement.ARPPoly is required for "
                             "R/Rdot projection")

        proj = meas.plane_projection
        if proj is None:
            raise ValueError("Measurement.PlaneProjection is required")
        if proj.time_coa_poly is None:
            raise ValueError("PlaneProjection.TimeCOAPoly is required "
                             "for R/Rdot projection")
        if proj.reference_point is None:
            raise ValueError("PlaneProjection.ReferencePoint is required")

        ref_ecf = proj.reference_point.ecef.to_array()
        ref_pix = proj.reference_point.point
        scp_pixel = (ref_pix.row, ref_pix.col)
        row_ss = proj.sample_spacing.row if proj.sample_spacing else 1.0
        col_ss = proj.sample_spacing.col if proj.sample_spacing else 1.0

        pp = proj.product_plane
        u_row = (pp.row_unit_vector.to_array()
                 if pp and pp.row_unit_vector else
                 np.array([1.0, 0.0, 0.0]))
        u_col = (pp.col_unit_vector.to_array()
                 if pp and pp.col_unit_vector else
                 np.array([0.0, 1.0, 0.0]))

        projector = _plane_projector(
            scp_ecf=ref_ecf,
            u_row=u_row,
            u_col=u_col,
            row_ss=row_ss,
            col_ss=col_ss,
        )

        return cls(
            time_coa_poly=proj.time_coa_poly,
            arp_poly=meas.arp_poly,
            method_projection=projector,
            scp_pixel=scp_pixel,
            row_ss=row_ss,
            col_ss=col_ss,
        )


# ── Image-to-ground projection ───────────────────────────────────────


def image_to_ground_plane(
    r: np.ndarray,
    rdot: np.ndarray,
    arp: np.ndarray,
    varp: np.ndarray,
    gref: np.ndarray,
    u_z: np.ndarray,
) -> np.ndarray:
    """Intersect R/Rdot contour with an arbitrary ground plane.

    Implements SICD Volume 3, Section 5.1 — the core geometric
    transformation that solves for the unique point on the R/Rdot
    cone that lies on a given plane.

    Parameters
    ----------
    r : np.ndarray
        Slant range (meters), shape ``(N,)``.
    rdot : np.ndarray
        Range rate (m/s), shape ``(N,)``.
    arp : np.ndarray
        ARP position at COA (ECF meters), shape ``(N, 3)``.
    varp : np.ndarray
        ARP velocity at COA (ECF m/s), shape ``(N, 3)``.
    gref : np.ndarray
        Ground reference point on the plane (ECF meters), shape ``(3,)``.
    u_z : np.ndarray
        Unit normal to the ground plane (away from Earth), shape ``(3,)``
        or ``(N, 3)``.

    Returns
    -------
    np.ndarray
        Ground points on the plane (ECF meters), shape ``(N, 3)``.
        Points where the R/Rdot contour does not intersect the plane
        are filled with NaN.
    """
    # Try numba-accelerated path for large batches
    if _HAS_NUMBA_PROJ:
        fast_result = _itgp_fast(r, rdot, arp, varp, gref, u_z)
        if fast_result is not None:
            return fast_result

    r = np.atleast_1d(r).astype(np.float64)
    rdot = np.atleast_1d(rdot).astype(np.float64)
    arp = np.atleast_2d(arp).astype(np.float64)
    varp = np.atleast_2d(varp).astype(np.float64)
    gref = np.asarray(gref, dtype=np.float64)
    u_z = np.asarray(u_z, dtype=np.float64)
    if u_z.ndim == 1:
        u_z = np.broadcast_to(u_z, arp.shape)

    n = len(r)
    result = np.full((n, 3), np.nan)

    # ARP height above plane
    arp_gref = arp - gref
    arp_z = np.sum(arp_gref * u_z, axis=-1)

    # Check feasibility: ARP must be closer than range
    feasible = np.abs(arp_z) < r

    if not np.any(feasible):
        return result

    # Ground range on the plane
    gd = np.sqrt(np.maximum(r ** 2 - arp_z ** 2, 0.0))

    # Grazing angle components
    cos_graz = gd / np.maximum(r, 1e-10)
    sin_graz = arp_z / np.maximum(r, 1e-10)

    # Velocity decomposition relative to the plane
    v_z = np.sum(varp * u_z, axis=-1)
    v_horiz = varp - v_z[..., np.newaxis] * u_z
    v_x = np.linalg.norm(v_horiz, axis=-1)

    # Unit vectors in the ground plane
    # uX: horizontal velocity direction
    u_x = v_horiz / np.maximum(v_x[..., np.newaxis], 1e-10)
    # uY: perpendicular in plane (cross(uZ, uX))
    u_y = np.cross(u_z, u_x)

    # Cosine of azimuth angle in the ground plane
    cos_az = (-rdot + v_z * sin_graz) / np.maximum(
        v_x * cos_graz, 1e-10)

    # Check feasibility: |cos_az| must be <= 1
    feasible &= np.abs(cos_az) <= 1.0 + 1e-10
    cos_az = np.clip(cos_az, -1.0, 1.0)

    # Determine LOOK (side of track)
    look_vec = np.cross(arp_gref, varp)
    look = np.sign(np.sum(look_vec * u_z, axis=-1))

    # Sine of azimuth (sign from LOOK)
    sin_az = look * np.sqrt(np.maximum(1.0 - cos_az ** 2, 0.0))

    # ARP nadir on the plane
    a_gpn = arp - arp_z[..., np.newaxis] * u_z

    # Ground point
    gpp = (a_gpn
           + gd[..., np.newaxis] * cos_az[..., np.newaxis] * u_x
           + gd[..., np.newaxis] * sin_az[..., np.newaxis] * u_y)

    result[feasible] = gpp[feasible]
    return result


# Block size for the spatial partition inside
# :func:`image_to_ground_hae`. Batches above this count are split into
# blocks of nearby image pixels, each solved with its own ground
# reference plane. Mirrors sarpy's per-block strategy and avoids the
# accuracy/iteration-count penalty of a single global ``gref`` across
# wide swaths.
_HAE_BLOCK_SIZE = 4096

# Pixel size (in image rows/cols) of one spatial bucket used to
# assign points to blocks. A 2048×2048 region maps to ~one ``gref``
# plane; far smaller than typical SICD scene extents so per-block
# iteration converges in 2-3 passes.
_HAE_BLOCK_PIXELS = 2048


def _quantize_latlon(
    lats: np.ndarray, lons: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize (lat, lon) to ~1 m resolution for DEM-query caching.

    1e-5 degrees ≈ 1.1 m at the equator — finer than DTED/SRTM cell
    spacing, so quantization does not change interpolated values
    while making (lat, lon) hashable for the per-call cache.
    """
    return (
        np.round(lats * 1e5).astype(np.int64),
        np.round(lons * 1e5).astype(np.int64),
    )


def _cached_dem_query(
    elevation_model: object,
    lats: np.ndarray,
    lons: np.ndarray,
    cache: dict,
) -> np.ndarray:
    """Query ``elevation_model.get_elevation`` with per-call memoization.

    The R/Rdot loop re-queries (lat, lon) clusters across 2-10
    iterations.  ``cache`` is a plain dict keyed on quantized
    coordinates; misses are batched into a single DEM call and the
    results are merged back in original order.
    """
    qlat, qlon = _quantize_latlon(lats, lons)
    n = lats.shape[0]
    out = np.empty(n, dtype=np.float64)
    miss_idx = []
    for i in range(n):
        key = (int(qlat[i]), int(qlon[i]))
        hit = cache.get(key)
        if hit is None:
            miss_idx.append(i)
        else:
            out[i] = hit
    if miss_idx:
        miss = np.asarray(miss_idx, dtype=np.intp)
        dem_h = elevation_model.get_elevation(lats[miss], lons[miss])
        if isinstance(dem_h, (int, float)):
            dem_h = np.full(miss.size, float(dem_h), dtype=np.float64)
        else:
            dem_h = np.asarray(dem_h, dtype=np.float64)
        for k, i in enumerate(miss_idx):
            cache[(int(qlat[i]), int(qlon[i]))] = dem_h[k]
            out[i] = dem_h[k]
    return out


# Default altitude step for the DEM vertical-step search.  Matches
# sarpy's ``vertical_step_size`` parameter on
# ``image_to_ground_dem``.  Tighter values trade speed for accuracy
# of the bracket linear-interpolation between adjacent altitude
# samples; 10 m matches what sarpy ships.
_DEM_VERTICAL_STEP_M = 10.0


def _image_to_ground_hae_perform(
    r: np.ndarray,
    rdot: np.ndarray,
    arp_coa: np.ndarray,
    varp_coa: np.ndarray,
    ref_point: np.ndarray,
    ugpn: np.ndarray,
    hae0: float,
    ref_hae: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Per-point R/Rdot projection onto the HAE = ``hae0`` surface.

    Mirrors sarpy's ``_image_to_ground_hae_perform`` (NGA SICD Vol 3
    section 9).  Iterates the plane intersection with a *per-point*
    ``gref`` so every point ends up exactly on the ellipsoidal
    ``hae0`` surface — a single plane only matches at one point
    because the ellipsoid curves underneath it.  A final slant-plane
    projection corrects each point to its precise hae0 height.

    Used as the inner kernel for both the constant-HAE path and the
    vertical-step DEM search in :func:`_solve_block_dem`.
    """
    look = np.sign(np.sum(
        np.cross(arp_coa, varp_coa) * (ref_point - arp_coa), axis=1,
    ))
    gref = ref_point - (ref_hae - hae0) * ugpn
    gpp = None
    delta_hae = None
    for _ in range(max_iter):
        gpp = image_to_ground_plane(
            r, rdot, arp_coa, varp_coa, gref, ugpn,
        )
        gpp_llh = ecef_to_geodetic(gpp)
        delta_hae = gpp_llh[:, 2] - hae0
        gref = gpp - delta_hae[:, np.newaxis] * ugpn
        if np.max(np.abs(delta_hae)) < tol:
            break

    # Slant-plane correction: project from gpp along the R/Rdot
    # slant plane normal to the hae0 surface so each point sits at
    # exactly hae0.
    uspn = np.cross(varp_coa, gpp - arp_coa) * look[:, np.newaxis]
    uspn /= np.linalg.norm(uspn, axis=-1, keepdims=True)
    sf = np.sum(ugpn * uspn, axis=-1)
    slp = gpp - uspn * (delta_hae / sf)[:, np.newaxis]
    spp_llh = ecef_to_geodetic(slp)
    spp_llh[:, 2] = hae0
    return geodetic_to_ecef(spp_llh)


def _solve_block_dem(
    r: np.ndarray,
    rdot: np.ndarray,
    arp_coa: np.ndarray,
    varp_coa: np.ndarray,
    hae: float,
    elevation_model: Optional[object],
    nan_fill_height: Optional[float],
    scp_ecf: Optional[np.ndarray],
    max_iter: int,
    tol: float,
    dem_cache: Optional[dict] = None,
) -> np.ndarray:
    """R/Rdot iteration for one spatial block.

    Two distinct paths:

    * **No DEM** — converge ``gref`` position iteratively on a single
      plane at the input ``hae``.  The convergence test on gref
      shift in ECF meters reliably triggers after 2-3 iterations
      (the original height-error test never converged for non-trivial
      ellipsoid curvature, so the loop wasted 10 iterations every
      time).
    * **DEM attached** — vertical-step search (ported from sarpy's
      ``image_to_ground_dem``).  Sweep ``this_hae`` from above
      max(DEM) downward; each step re-projects active points onto
      the surface at ``this_hae`` and computes
      ``diff = h_projected - DEM(lat, lon)``.  The transition from
      diff > 0 to diff < 0 brackets the R/Rdot cone's intersection
      with terrain; linear interpolation between the bracketing
      altitudes places the ground point.  This is what sarpy does
      and matches its output to within meters at the median (vs the
      tens-of-meters error of a block-mean-plane approximation).
    """
    n = r.shape[0]
    if scp_ecf is not None:
        ref_point = np.asarray(scp_ecf, dtype=np.float64)
    else:
        ref_point = np.mean(arp_coa, axis=0)
    ref_llh = ecef_to_geodetic(ref_point)
    ref_hae = float(ref_llh[2])
    ugpn = wgs84_norm(ref_point)

    if dem_cache is None:
        dem_cache = {}

    # ── No-DEM path: single plane, iterate gref position. ──────────
    if elevation_model is None:
        gref = geodetic_to_ecef(
            np.array([ref_llh[0], ref_llh[1], hae]),
        )
        for _iteration in range(max_iter):
            u_z = wgs84_norm(gref)
            gpp = image_to_ground_plane(
                r, rdot, arp_coa, varp_coa, gref, u_z,
            )
            valid = ~np.any(np.isnan(gpp), axis=-1)
            if not np.any(valid):
                break
            gref_mean_ecf = np.mean(gpp[valid], axis=0)
            geo_mean = ecef_to_geodetic(gref_mean_ecf)
            new_gref = geodetic_to_ecef(
                np.array([geo_mean[0], geo_mean[1], hae]),
            )
            if np.linalg.norm(new_gref - gref) < tol:
                gref = new_gref
                break
            gref = new_gref
        u_slant = wgs84_norm(gref)
        return image_to_ground_plane(
            r, rdot, arp_coa, varp_coa, gref, u_slant,
        )

    # ── DEM path: vertical-step search (sarpy-style). ──────────────
    # 1) Rough projection at ref_hae → coarse lat/lon for envelope.
    rough_ecf = _image_to_ground_hae_perform(
        r, rdot, arp_coa, varp_coa, ref_point, ugpn,
        ref_hae, ref_hae, max_iter, tol,
    )
    rough_llh = ecef_to_geodetic(rough_ecf)
    dem_rough = _cached_dem_query(
        elevation_model, rough_llh[:, 0], rough_llh[:, 1], dem_cache,
    )
    nan_mask = np.isnan(dem_rough)
    dem_rough[nan_mask] = (
        nan_fill_height if nan_fill_height is not None else hae
    )
    min_dem = float(np.min(dem_rough)) - 10.0
    max_dem = float(np.max(dem_rough)) + 10.0

    # 2) Flat-terrain fast path: skip the sweep entirely.
    if max_dem - min_dem < _DEM_VERTICAL_STEP_M:
        return _image_to_ground_hae_perform(
            r, rdot, arp_coa, varp_coa, ref_point, ugpn,
            max_dem, ref_hae, max_iter, tol,
        )

    # 3) Initial step at max_dem so all points start above terrain.
    out = np.zeros((n, 3), dtype=np.float64)
    cont_mask = np.ones(n, dtype=bool)
    this_hae = max_dem
    previous_ecf = _image_to_ground_hae_perform(
        r, rdot, arp_coa, varp_coa, ref_point, ugpn,
        this_hae, ref_hae, max_iter, tol,
    )
    previous_llh = ecef_to_geodetic(previous_ecf)
    dem_prev = _cached_dem_query(
        elevation_model,
        previous_llh[:, 0], previous_llh[:, 1], dem_cache,
    )
    nan_mask = np.isnan(dem_prev)
    dem_prev[nan_mask] = (
        nan_fill_height if nan_fill_height is not None else hae
    )
    previous_diff = previous_llh[:, 2] - dem_prev

    # 4) Vertical sweep until every point brackets the terrain.
    safety = int(
        np.ceil((max_dem - min_dem) / _DEM_VERTICAL_STEP_M),
    ) + 5
    for _sweep in range(safety):
        if not np.any(cont_mask):
            break
        this_hae -= _DEM_VERTICAL_STEP_M

        active_idx = np.flatnonzero(cont_mask)
        this_ecf = _image_to_ground_hae_perform(
            r[cont_mask], rdot[cont_mask],
            arp_coa[cont_mask], varp_coa[cont_mask],
            ref_point, ugpn, this_hae, ref_hae, max_iter, tol,
        )
        this_llh = ecef_to_geodetic(this_ecf)
        dem_this = _cached_dem_query(
            elevation_model,
            this_llh[:, 0], this_llh[:, 1], dem_cache,
        )
        nan_mask = np.isnan(dem_this)
        dem_this[nan_mask] = (
            nan_fill_height if nan_fill_height is not None else hae
        )
        this_diff = this_llh[:, 2] - dem_this

        # Bracket detected where diff flips positive -> negative.
        bracketed = np.isfinite(this_diff) & (this_diff < 0)
        if np.any(bracketed):
            d0 = previous_diff[bracketed]
            d1 = np.abs(this_diff[bracketed])
            interp = (
                d1[:, np.newaxis] * previous_ecf[bracketed]
                + d0[:, np.newaxis] * this_ecf[bracketed]
            ) / (d0 + d1)[:, np.newaxis]
            finalize_idx = active_idx[bracketed]
            out[finalize_idx] = interp
            cont_mask[finalize_idx] = False

            # Compact: keep only non-bracketed for next step.
            keep = ~bracketed
            previous_ecf = this_ecf[keep]
            previous_diff = this_diff[keep]
        else:
            previous_ecf = this_ecf
            previous_diff = this_diff

    # Any unbracketed points (rare; only when terrain extends past
    # min_dem - safety_extra * step) get the last projection.
    if np.any(cont_mask):
        out[cont_mask] = previous_ecf
    return out


def _partition_blocks(
    im_points: np.ndarray,
    block_pixels: int,
    block_size: int,
) -> List[np.ndarray]:
    """Bucket image points into spatial blocks.

    Each block is the set of points falling into one
    ``block_pixels × block_pixels`` image-coordinate cell, optionally
    further split if a single cell exceeds ``block_size`` points.
    Returns a list of index arrays into ``im_points``.

    Adjacent pixels stay in the same block, so a per-block ``gref``
    converges fast even at scene edges.
    """
    n = im_points.shape[0]
    if n <= block_size:
        return [np.arange(n, dtype=np.intp)]
    row_bucket = np.floor(im_points[:, 0] / block_pixels).astype(np.int64)
    col_bucket = np.floor(im_points[:, 1] / block_pixels).astype(np.int64)
    order = np.lexsort((col_bucket, row_bucket))
    sorted_row = row_bucket[order]
    sorted_col = col_bucket[order]
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (sorted_row[1:] != sorted_row[:-1]) | (
        sorted_col[1:] != sorted_col[:-1]
    )
    starts = np.flatnonzero(change)
    ends = np.concatenate([starts[1:], [n]])
    blocks: List[np.ndarray] = []
    for s, e in zip(starts, ends):
        idx = order[s:e]
        if idx.size <= block_size:
            blocks.append(idx)
        else:
            # Split oversized cells along their natural order.
            for split in range(0, idx.size, block_size):
                blocks.append(idx[split:split + block_size])
    return blocks


def image_to_ground_hae(
    coa_proj: COAProjection,
    im_points: np.ndarray,
    hae: float = 0.0,
    scp_ecf: Optional[np.ndarray] = None,
    elevation_model: Optional[object] = None,
    max_iter: int = 10,
    tol: float = 1e-3,
    nan_fill_height: Optional[float] = None,
) -> np.ndarray:
    """Project image points to a height surface via R/Rdot iteration.

    Implements the SICD Volume 3 iterative projection algorithm.
    Supports three modes:

    - **Constant HAE** (default): All points projected to the same
      height above the WGS-84 ellipsoid.
    - **DEM surface**: When ``elevation_model`` is provided, the DEM
      height is queried at each candidate (lat, lon) during every
      iteration.  The DEM height feeds back into the R/Rdot cone
      intersection, producing terrain-corrected ground points.
    - **Per-point HAE**: Not yet supported; use DEM mode instead.

    The DEM-integrated iteration follows the same pattern as sarpy's
    ``image_to_ground_dem``: project → query DEM → adjust reference
    surface → repeat until height converges.

    For large batches (N > ``_HAE_BLOCK_SIZE``), points are partitioned
    into spatial blocks and each block iterates with its own
    ``gref`` plane.  This matches sarpy's block-local strategy and
    avoids the accuracy loss a single global plane suffers at the
    edges of wide swaths.  The output shape and per-point semantics
    are unchanged.

    Parameters
    ----------
    coa_proj : COAProjection
        Configured projection object.
    im_points : np.ndarray
        Image coordinates, shape ``(N, 2)`` as ``[row, col]``.
    hae : float
        Target height above WGS-84 (meters).  Used as the initial
        guess and as fallback when ``elevation_model`` returns NaN
        and ``nan_fill_height`` is not provided.
    scp_ecf : np.ndarray, optional
        Scene Center Point in ECF (meters), shape ``(3,)``.
    elevation_model : ElevationModel, optional
        DEM/DTED elevation model with ``get_elevation(lat, lon)``
        returning HAE (meters).  When provided, the target height
        at each point is queried from the DEM at every iteration.
    max_iter : int
        Maximum iterations (default 10).
    tol : float
        Convergence tolerance on height (meters, default 1 mm).
    nan_fill_height : float, optional
        Height (meters HAE) used to fill NaN gaps in DEM queries.
        When ``None``, falls back to ``hae``.  Callers should pass
        the sensor reference height (e.g., SCP HAE) so that DEM
        gaps are filled consistently across all geolocation types.

    Returns
    -------
    np.ndarray
        Ground points in ECF (meters), shape ``(N, 3)``.
    """
    im_points = np.atleast_2d(im_points)
    n = im_points.shape[0]

    # Get R/Rdot parameters from COAProjection (single batched call).
    r, rdot, time_coa, arp_coa, varp_coa = coa_proj.projection(im_points)

    blocks = _partition_blocks(
        im_points, _HAE_BLOCK_PIXELS, _HAE_BLOCK_SIZE,
    )
    # Single-block fast path preserves the exact previous behaviour
    # so small-N callers and tests see no semantic change.
    if len(blocks) == 1:
        return _solve_block_dem(
            r, rdot, arp_coa, varp_coa, hae, elevation_model,
            nan_fill_height, scp_ecf, max_iter, tol,
        )

    # Multi-block path: every block shares the caller-supplied
    # ``scp_ecf`` as its *initial* gref so the first iteration starts
    # from a sane reference.  After that each block's gref evolves
    # independently to the mean of *its own* projected points — that
    # is the block-local part.  A per-call DEM cache amortises
    # overlapping (lat, lon) hits at block borders.
    result = np.empty((n, 3), dtype=np.float64)
    dem_cache: dict = {}
    for idx in blocks:
        block_result = _solve_block_dem(
            r[idx], rdot[idx], arp_coa[idx], varp_coa[idx], hae,
            elevation_model, nan_fill_height,
            scp_ecf=scp_ecf,
            max_iter=max_iter, tol=tol, dem_cache=dem_cache,
        )
        result[idx] = block_result
    return result


def image_to_ground_dem(
    coa_proj: COAProjection,
    im_points: np.ndarray,
    elevation_model: object,
    scp_ecf: Optional[np.ndarray] = None,
    initial_hae: float = 0.0,
    max_iter: int = 10,
    tol: float = 1e-3,
) -> np.ndarray:
    """Project image points to terrain surface via DEM intersection.

    Iterative algorithm: projects at an initial HAE, looks up the DEM
    height at the projected (lat, lon), then re-projects at the DEM
    height until convergence.  This correctly accounts for height-
    dependent projection geometry (parallax, R/Rdot cone shape).

    Parameters
    ----------
    coa_proj : COAProjection
        Configured projection object.
    im_points : np.ndarray
        Image coordinates, shape ``(N, 2)`` as ``[row, col]``.
    elevation_model : ElevationModel
        DEM/DTED elevation model with ``get_elevation(lat, lon)`` method.
    scp_ecf : np.ndarray, optional
        Scene Center Point in ECF (meters) for initial reference.
    initial_hae : float
        Initial height guess (meters HAE).  Using the mean terrain
        height improves convergence speed.
    max_iter : int
        Maximum iterations (default 10).
    tol : float
        Convergence tolerance on height (meters, default 1 mm).

    Returns
    -------
    np.ndarray
        Ground points in ECF (meters), shape ``(N, 3)``.
    """
    im_points = np.atleast_2d(im_points)
    n = im_points.shape[0]
    hae = np.full(n, initial_hae)

    for iteration in range(max_iter):
        # Project at current HAE estimates (per-point)
        # Use the mean HAE for the bulk projection, then refine
        mean_hae = float(np.mean(hae))
        gpp = image_to_ground_hae(
            coa_proj, im_points, hae=mean_hae,
            scp_ecf=scp_ecf, max_iter=5, tol=tol * 0.1,
        )

        # Convert to geodetic
        valid = ~np.any(np.isnan(gpp), axis=-1)
        if not np.any(valid):
            break

        geo_dem = ecef_to_geodetic(gpp[valid])
        lats, lons, heights = geo_dem[:, 0], geo_dem[:, 1], geo_dem[:, 2]

        # Look up DEM height at projected positions (public API handles
        # geoid correction, scalar/array dispatch, and NaN transparently)
        dem_heights = np.asarray(
            elevation_model.get_elevation(lats, lons), dtype=np.float64)

        # Replace NaN DEM values with current HAE estimate
        nan_mask = np.isnan(dem_heights)
        dem_heights[nan_mask] = hae[valid][nan_mask]

        # Check convergence
        height_err = np.max(np.abs(hae[valid] - dem_heights))
        hae[valid] = dem_heights

        if height_err < tol:
            break

    # Final projection at converged per-point HAE
    # Process individually for per-point height support
    result = np.full((n, 3), np.nan)
    for i in range(n):
        gpp_i = image_to_ground_hae(
            coa_proj, im_points[i:i + 1], hae=float(hae[i]),
            scp_ecf=scp_ecf, max_iter=5, tol=tol * 0.1,
        )
        result[i] = gpp_i[0]

    return result


def ground_to_image(
    coa_proj: COAProjection,
    ground_ecf: np.ndarray,
    scp_ecf: np.ndarray,
    u_row: np.ndarray,
    u_col: np.ndarray,
    row_ss: float,
    col_ss: float,
    scp_pixel: Tuple[float, float],
    max_iter: int = 10,
    tol: float = 1e-2,
) -> np.ndarray:
    """Project ground ECF points to image pixel coordinates.

    Fully vectorized SICD Volume 3, Section 6 algorithm.  All N
    points are processed simultaneously per iteration.  Follows
    sarpy's ``_ground_to_image`` pattern:

    1. Project ECF ground points onto the image plane.
    2. Convert to pixel coordinates.
    3. R/Rdot forward those pixels back to the ground plane
       containing the original scene points.
    4. Compute displacement, adjust ground points, repeat.

    Parameters
    ----------
    coa_proj : COAProjection
        Configured projection object.
    ground_ecf : np.ndarray
        Ground points in ECF (meters), shape ``(N, 3)``.
    scp_ecf : np.ndarray
        Scene Center Point in ECF (meters), shape ``(3,)``.
    u_row : np.ndarray
        Image row unit vector (ECF), shape ``(3,)``.
    u_col : np.ndarray
        Image col unit vector (ECF), shape ``(3,)``.
    row_ss : float
        Row sample spacing (meters).
    col_ss : float
        Column sample spacing (meters).
    scp_pixel : tuple of float
        (row, col) of SCP in image.
    max_iter : int
        Maximum iterations (default 10).
    tol : float
        Ground plane displacement tolerance (meters, default 1 cm).

    Returns
    -------
    np.ndarray
        Image pixel coordinates, shape ``(N, 2)`` as ``[row, col]``.
    """
    ground_ecf = np.atleast_2d(ground_ecf).astype(np.float64)
    scp_ecf = np.asarray(scp_ecf, dtype=np.float64)
    u_row = np.asarray(u_row, dtype=np.float64)
    u_col = np.asarray(u_col, dtype=np.float64)
    scp_pixel = np.asarray(scp_pixel, dtype=np.float64)

    n = ground_ecf.shape[0]

    # Image Plane Normal
    u_ipn = np.cross(u_row, u_col)
    u_ipn = u_ipn / np.linalg.norm(u_ipn)

    # Ground Plane Normal (WGS-84 normal at SCP)
    u_gpn = wgs84_norm(scp_ecf)

    # Slant Plane Normal = IPN; scale factor
    u_spn = u_ipn
    sf = float(np.dot(u_spn, u_ipn))

    # Handle non-orthogonal row/col
    cos_theta = np.dot(u_row, u_col)
    sin_theta = np.sqrt(max(1.0 - cos_theta * cos_theta, 1e-30))
    ipp_transform = np.array(
        [[1, -cos_theta], [-cos_theta, 1]],
        dtype=np.float64) / (sin_theta * sin_theta)
    matrix_transform = np.column_stack([u_row, u_col]) @ ipp_transform

    g_n = ground_ecf.copy()
    im_points = np.zeros((n, 2), dtype=np.float64)

    for _it in range(max_iter):
        # Project ground points onto image plane
        dist_n = np.dot(scp_ecf - g_n, u_ipn) / sf
        i_n = g_n + np.outer(dist_n, u_spn)

        # Convert to pixel coordinates
        delta_ipp = i_n - scp_ecf
        ip_iter = delta_ipp @ matrix_transform
        im_points[:, 0] = ip_iter[:, 0] / row_ss + scp_pixel[0]
        im_points[:, 1] = ip_iter[:, 1] / col_ss + scp_pixel[1]

        # R/Rdot forward to ground plane at scene points
        r, rdot, _, arp_coa, varp_coa = coa_proj.projection(im_points)
        p_n = image_to_ground_plane(
            r, rdot, arp_coa, varp_coa, g_n, u_gpn)

        # Displacement and adjustment
        diff_n = ground_ecf - p_n
        delta_gpn = np.linalg.norm(diff_n, axis=1)
        g_n += diff_n

        if np.all(delta_gpn < tol):
            break

    return im_points
