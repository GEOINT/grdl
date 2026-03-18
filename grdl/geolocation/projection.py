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
duane.d.smalley@gmail.com

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
2026-03-16
"""

# Standard library
from typing import Callable, Optional, Tuple

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

    Parameters follow SICD Volume 3, Section 4.1.
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

        # Polar angle and spatial frequency scale factor
        polar_ang = polar_ang_poly(time_coa)
        k_sf = spatial_freq_sf_poly(polar_ang)

        # Polar angle rate: d(polar_ang)/dt
        dpolar_dt = polar_ang_poly.derivative_eval(time_coa)

        # Spatial frequency offsets from SCP
        # Row and col in spatial frequency domain
        delta_kcol = col_t * (2 * np.pi / col_ss) * col_sgn
        delta_krow = row_t * (2 * np.pi / row_ss) * row_sgn

        # Along-aperture and cross-aperture spatial frequencies
        cos_pa = np.cos(polar_ang)
        sin_pa = np.sin(polar_ang)
        k_a = col_kctr + delta_kcol  # along-aperture (azimuth)
        k_c = row_kctr + delta_krow  # cross-aperture (range)

        # Range offset from SCP via spatial frequency mapping
        # k_sf maps spatial frequency to slant range
        delta_r = k_sf * (k_c / row_kctr - 1.0) * r_scp

        # Rdot offset from SCP
        # d(delta_r)/d(theta) * d(theta)/dt
        delta_rdot = k_sf * (k_a / col_kctr - 1.0) * r_scp * dpolar_dt

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
        elif (algo == 'RGAZCOMP'
              and metadata.rg_az_comp is not None
              and metadata.rg_az_comp.az_sf is not None):
            projector = _rgazcomp_projector(
                scp_ecf=scp_ecf,
                az_sf=metadata.rg_az_comp.az_sf,
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


def image_to_ground_hae(
    coa_proj: COAProjection,
    im_points: np.ndarray,
    hae: float = 0.0,
    scp_ecf: Optional[np.ndarray] = None,
    max_iter: int = 10,
    tol: float = 1e-3,
) -> np.ndarray:
    """Project image points to constant HAE above WGS-84 ellipsoid.

    Iterative algorithm from SICD Volume 3, Section 5.2.  Uses
    successive R/Rdot contour intersections with tangent planes
    until the projected height converges to the target HAE.

    Parameters
    ----------
    coa_proj : COAProjection
        Configured projection object.
    im_points : np.ndarray
        Image coordinates, shape ``(N, 2)`` as ``[row, col]``.
    hae : float
        Target height above WGS-84 ellipsoid (meters).
    scp_ecf : np.ndarray, optional
        Scene Center Point in ECF (meters), shape ``(3,)``.  Used as
        the initial ground reference for iteration.  If not provided,
        the ARP nadir is used (slower convergence for side-looking).
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

    # Get R/Rdot parameters from COAProjection
    r, rdot, time_coa, arp_coa, varp_coa = coa_proj.projection(im_points)

    # Initial ground reference: prefer SCP, fall back to ARP nadir
    if scp_ecf is not None:
        ref_ecf = np.asarray(scp_ecf, dtype=np.float64)
    else:
        ref_ecf = np.mean(arp_coa, axis=0)
    lat0, lon0, _ = ecef_to_geodetic(
        np.array([ref_ecf[0]]),
        np.array([ref_ecf[1]]),
        np.array([ref_ecf[2]]),
    )
    gref_x, gref_y, gref_z = geodetic_to_ecef(lat0, lon0, np.array([hae]))
    gref = np.array([gref_x[0], gref_y[0], gref_z[0]])

    # Iterate: project to tangent plane, check height, adjust
    for iteration in range(max_iter):
        # Normal to ellipsoid at current ground reference
        u_z = wgs84_norm(gref)

        # Project R/Rdot contour to plane at gref
        gpp = image_to_ground_plane(r, rdot, arp_coa, varp_coa, gref, u_z)

        # Check convergence: compute HAE of projected points
        valid = ~np.any(np.isnan(gpp), axis=-1)
        if not np.any(valid):
            break

        lats, lons, heights = ecef_to_geodetic(
            gpp[valid, 0], gpp[valid, 1], gpp[valid, 2])
        height_err = np.max(np.abs(heights - hae))

        if height_err < tol:
            break

        # Update ground reference to mean of projected points
        gref_mean = np.mean(gpp[valid], axis=0)
        lat_m, lon_m, _ = ecef_to_geodetic(
            np.array([gref_mean[0]]),
            np.array([gref_mean[1]]),
            np.array([gref_mean[2]]),
        )
        gx, gy, gz = geodetic_to_ecef(lat_m, lon_m, np.array([hae]))
        gref = np.array([gx[0], gy[0], gz[0]])

    # Final projection with converged reference using slant plane
    # normal for sub-meter accuracy (SICD Vol 3 recommendation)
    u_slant = wgs84_norm(gref)
    result = image_to_ground_plane(
        r, rdot, arp_coa, varp_coa, gref, u_slant)

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

        lats, lons, heights = ecef_to_geodetic(
            gpp[valid, 0], gpp[valid, 1], gpp[valid, 2])

        # Look up DEM height at projected positions
        dem_heights = elevation_model._get_elevation_array(lats, lons)

        # Apply geoid correction if available
        if hasattr(elevation_model, '_geoid') and elevation_model._geoid is not None:
            dem_heights = dem_heights + elevation_model._geoid.get_undulation(
                lats, lons)

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
    hae: Optional[float] = None,
    max_iter: int = 10,
    tol: float = 1e-3,
) -> np.ndarray:
    """Project ground ECF points to image pixel coordinates.

    Iterative algorithm from SICD Volume 3, Section 6.  Projects
    ground points onto the image plane, computes R/Rdot contour,
    projects back to ground at the original HAE, and iterates
    until convergence.

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
    hae : float, optional
        Target HAE for reprojection.  If None, uses per-point height.
    max_iter : int
        Maximum iterations (default 10).
    tol : float
        Convergence tolerance (meters, default 1 mm).

    Returns
    -------
    np.ndarray
        Image pixel coordinates, shape ``(N, 2)`` as ``[row, col]``.
    """
    ground_ecf = np.atleast_2d(ground_ecf).astype(np.float64)
    scp_ecf = np.asarray(scp_ecf, dtype=np.float64)
    u_row = np.asarray(u_row, dtype=np.float64)
    u_col = np.asarray(u_col, dtype=np.float64)
    n = ground_ecf.shape[0]

    # Image Plane Normal
    u_ipn = np.cross(u_row, u_col)
    u_ipn = u_ipn / np.linalg.norm(u_ipn)

    # Get height of each ground point for reprojection
    if hae is None:
        _, _, pt_heights = ecef_to_geodetic(
            ground_ecf[:, 0], ground_ecf[:, 1], ground_ecf[:, 2])
    else:
        pt_heights = np.full(n, hae)

    # Initial: project ground points onto image plane
    delta_g = ground_ecf - scp_ecf
    row_proj = np.dot(delta_g, u_row) / row_ss + scp_pixel[0]
    col_proj = np.dot(delta_g, u_col) / col_ss + scp_pixel[1]
    im_points = np.column_stack([row_proj, col_proj])

    for _ in range(max_iter):
        # Forward project these image points to ground at target HAE
        gpp_arr = np.zeros((n, 3))
        for i in range(n):
            gpp_i = image_to_ground_hae(
                coa_proj,
                im_points[i:i + 1],
                hae=pt_heights[i],
                scp_ecf=scp_ecf,
                max_iter=5,
                tol=tol * 0.1,
            )
            gpp_arr[i] = gpp_i[0]

        # Compute error
        err = ground_ecf - gpp_arr
        err_mag = np.linalg.norm(err, axis=-1)

        if np.max(err_mag) < tol:
            break

        # Update: project error onto image plane and adjust
        d_row = np.dot(err, u_row) / row_ss
        d_col = np.dot(err, u_col) / col_ss
        im_points[:, 0] += d_row
        im_points[:, 1] += d_col

    return im_points
