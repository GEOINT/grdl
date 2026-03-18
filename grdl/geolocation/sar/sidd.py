# -*- coding: utf-8 -*-
"""
SIDD Geolocation - Coordinate transforms for SIDD derived SAR products.

Implements the SIDD Volume 1 (NGA.STND.0025-1) pixel-to-geographic and
geographic-to-pixel projection equations for all three grid types:

- **PGD** (Planar Gridded Display / PlaneProjection): Pixel grid sampled
  on a plane in ECEF with constant sample spacing.  Forward and inverse
  are exact linear transforms.
- **GGD** (Geodetic Gridded Display / GeographicProjection): Pixel grid
  sampled in latitude/longitude with constant angular spacing.
- **CGD** (Cylindrical Gridded Display / CylindricalProjection): Pixel
  grid sampled on a cylindrical surface aligned with the stripmap
  direction.

When ``TimeCOAPoly`` and ``ARPPoly`` are present in the Measurement
block, R/Rdot refinement is used for sub-meter accuracy.  The grid
projection provides the initial approximation; the native R/Rdot engine
(``COAProjection``) then iteratively projects onto the WGS-84 surface
at the target HAE, correcting for Earth curvature and terrain.

All transforms follow the standard's sensor-model-grid convention::

    r' = r - r0       (translate to sensor model coordinates)
    c' = c - c0
    dr = Δr * r'      (physical distance / angular measure)
    dc = Δc * c'

Dependencies
------------
(none beyond numpy and scipy, which are core GRDL deps)

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
2026-03-07

Modified
--------
2026-03-16  Add R/Rdot refinement via COAProjection when TimeCOAPoly
            and ARPPoly are available.
2026-03-07
"""

# Standard library
from typing import Optional, Tuple, Union, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import Geolocation

if TYPE_CHECKING:
    from grdl.IO.models.sidd import SIDDMetadata


# ===================================================================
# WGS-84 constants (NGA.STND.0025-1 Section 3.6)
# ===================================================================

_WGS84_A = 6378137.0                  # semi-major axis (m)
_WGS84_B = 6356752.314245179          # semi-minor axis (m)
_WGS84_F = 1.0 / 298.257223563        # flattening
_WGS84_E1_SQ = 2 * _WGS84_F - _WGS84_F ** 2  # first eccentricity squared
_WGS84_E2_SQ = (_WGS84_A ** 2 - _WGS84_B ** 2) / _WGS84_B ** 2  # second ecc sq


def _geodetic_to_ecef(
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic (lat, lon, height) to ECEF (X, Y, Z).

    Implements Section 3.6 of the SIDD standard.

    Parameters
    ----------
    lats : np.ndarray
        Latitudes in degrees.
    lons : np.ndarray
        Longitudes in degrees.
    heights : np.ndarray
        Heights above WGS-84 ellipsoid in meters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (X, Y, Z) ECEF coordinates in meters.
    """
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Radius of curvature in the prime vertical
    rc = _WGS84_A / np.sqrt(1.0 - _WGS84_E1_SQ * sin_lat ** 2)

    x = (rc + heights) * cos_lat * cos_lon
    y = (rc + heights) * cos_lat * sin_lon
    z = ((_WGS84_B ** 2 / _WGS84_A ** 2) * rc + heights) * sin_lat

    return x, y, z


def _ecef_to_geodetic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF (X, Y, Z) to geodetic (lat, lon, height).

    Implements Section 3.7 of the SIDD standard (iterative method).

    Parameters
    ----------
    x, y, z : np.ndarray
        ECEF coordinates in meters.
    max_iter : int
        Maximum iterations for latitude convergence.
    tol : float
        Convergence tolerance on tan(latitude).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (lats, lons, heights) in degrees and meters.
    """
    lon = np.arctan2(y, x)

    dxy = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(_WGS84_A * z, _WGS84_B * dxy)

    # Initial latitude estimate (Bowring's formula)
    tan_lat = (
        (z + _WGS84_E2_SQ * _WGS84_B * np.sin(theta) ** 3)
        / (dxy - _WGS84_E1_SQ * _WGS84_A * np.cos(theta) ** 3)
    )

    # Iterative refinement (Section 3.7)
    for _ in range(max_iter):
        tan_lat_prev = tan_lat
        # Reduced latitude step
        tan_lat_reduced = (1.0 - _WGS84_F) * tan_lat
        lat_reduced = np.arctan(tan_lat_reduced)
        # Compute geocentric latitude for next iteration
        sin_reduced = np.sin(lat_reduced)
        cos_reduced = np.cos(lat_reduced)
        theta_new = np.arctan2(
            _WGS84_A * z, _WGS84_B * dxy
        )
        tan_lat = (
            (z + _WGS84_E2_SQ * _WGS84_B * np.sin(theta_new) ** 3)
            / (dxy - _WGS84_E1_SQ * _WGS84_A * np.cos(theta_new) ** 3)
        )
        if np.all(np.abs(tan_lat - tan_lat_prev) < tol):
            break

    lat = np.arctan(tan_lat)

    # Height above ellipsoid
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    rc = _WGS84_A / np.sqrt(1.0 - _WGS84_E1_SQ * sin_lat ** 2)

    # Avoid division by zero near poles
    with np.errstate(invalid='ignore', divide='ignore'):
        h_equatorial = dxy / cos_lat - rc
        h_polar = np.abs(z) / np.where(
            np.abs(sin_lat) > 0, np.abs(sin_lat), 1.0
        ) - (_WGS84_B ** 2 / _WGS84_A ** 2) * rc
    height = np.where(np.abs(cos_lat) > 1e-10, h_equatorial, h_polar)

    return np.degrees(lat), np.degrees(lon), height


# ===================================================================
# SIDDGeolocation
# ===================================================================

class SIDDGeolocation(Geolocation):
    """Geolocation for SIDD (Sensor Independent Derived Data) imagery.

    Implements the SIDD Volume 1 (NGA.STND.0025-1) projection equations
    for converting between image pixel coordinates and geographic
    coordinates.  Supports PlaneProjection (PGD), GeographicProjection
    (GGD), and CylindricalProjection (CGD) grid types.

    The most common case is **PlaneProjection**, where the image is
    sampled on a plane in ECEF space.  Forward and inverse transforms
    are exact (no iteration required for the pixel ↔ ECEF step).

    When ``TimeCOAPoly`` and ``ARPPoly`` are present in the SIDD
    Measurement block, R/Rdot refinement is automatically enabled for
    maximum positional accuracy.  The grid projection provides the
    initial approximation; the native R/Rdot engine then iteratively
    projects onto the WGS-84 surface at the target HAE.

    Parameters
    ----------
    metadata : SIDDMetadata
        Typed SIDD metadata.  Must have ``measurement`` with a
        supported projection type.
    refine : bool
        Enable R/Rdot refinement when ``TimeCOAPoly`` and ``ARPPoly``
        are available (default ``True``).  Set to ``False`` to force
        grid-only projection.

    Attributes
    ----------
    metadata : SIDDMetadata
        The SIDD metadata.
    projection_type : str
        Active projection type (``'PlaneProjection'``,
        ``'GeographicProjection'``, ``'CylindricalProjection'``).
    has_rdot : bool
        Whether R/Rdot refinement is active.

    Raises
    ------
    ValueError
        If the metadata does not contain a supported projection.

    Examples
    --------
    >>> from grdl.IO.sar import SIDDReader
    >>> from grdl.geolocation.sar.sidd import SIDDGeolocation
    >>> with SIDDReader('product.nitf') as reader:
    ...     geo = SIDDGeolocation.from_reader(reader)
    ...     lat, lon, h = geo.image_to_latlon(6000, 6000)
    ...     row, col = geo.latlon_to_image(lat, lon, h)
    """

    def __init__(
        self,
        metadata: 'SIDDMetadata',
        refine: bool = True,
        dem_path: Optional[str] = None,
        geoid_path: Optional[str] = None,
    ) -> None:
        self.metadata = metadata

        meas = metadata.measurement
        if meas is None:
            raise ValueError(
                "SIDDMetadata.measurement is required for geolocation"
            )

        self.projection_type = meas.projection_type or ''

        if self.projection_type == 'PlaneProjection':
            self._init_plane(meas)
        elif self.projection_type == 'GeographicProjection':
            self._init_geographic(meas)
        elif self.projection_type == 'CylindricalProjection':
            self._init_cylindrical(meas)
        else:
            raise ValueError(
                f"Unsupported SIDD projection type: "
                f"{self.projection_type!r}.  Supported: "
                f"PlaneProjection, GeographicProjection, "
                f"CylindricalProjection."
            )

        # R/Rdot refinement: requires TimeCOAPoly + ARPPoly
        self.has_rdot = False
        self._coa_proj = None
        self._scp_ecf = None
        if refine:
            self._init_rdot_refinement(meas)

        shape = (metadata.rows, metadata.cols)
        super().__init__(
            shape, crs='WGS84', dem_path=dem_path, geoid_path=geoid_path
        )

    # ------------------------------------------------------------------
    # Initializers per projection type
    # ------------------------------------------------------------------

    def _init_plane(self, meas: object) -> None:
        """Extract PGD (PlaneProjection) parameters.

        Parameters
        ----------
        meas : SIDDMeasurement
            Measurement section with plane_projection populated.
        """
        pp = meas.plane_projection
        if pp is None:
            raise ValueError(
                "PlaneProjection declared but plane_projection is None"
            )
        rp = pp.reference_point
        if rp is None or rp.ecef is None or rp.point is None:
            raise ValueError(
                "PlaneProjection requires reference_point with ecef and "
                "pixel coordinates"
            )
        if pp.sample_spacing is None:
            raise ValueError(
                "PlaneProjection requires sample_spacing"
            )
        plane = pp.product_plane
        if plane is None or plane.row_unit_vector is None or plane.col_unit_vector is None:
            raise ValueError(
                "PlaneProjection requires product_plane with "
                "row_unit_vector and col_unit_vector"
            )

        # Reference point in ECEF (meters)
        self._srp = np.array(
            [rp.ecef.x, rp.ecef.y, rp.ecef.z], dtype=np.float64
        )
        # Reference point pixel coordinates
        self._r0 = float(rp.point.row)
        self._c0 = float(rp.point.col)
        # Sample spacing (meters/pixel)
        self._dr = float(pp.sample_spacing.row)
        self._dc = float(pp.sample_spacing.col)
        # Unit vectors in ECEF
        self._row_vec = np.array(
            [plane.row_unit_vector.x,
             plane.row_unit_vector.y,
             plane.row_unit_vector.z],
            dtype=np.float64,
        )
        self._col_vec = np.array(
            [plane.col_unit_vector.x,
             plane.col_unit_vector.y,
             plane.col_unit_vector.z],
            dtype=np.float64,
        )

    def _init_geographic(self, meas: object) -> None:
        """Extract GGD (GeographicProjection) parameters.

        Uses the reference point's geodetic coordinates and sample
        spacing in arc-seconds to build the simple angular mapping.

        Parameters
        ----------
        meas : SIDDMeasurement
            Measurement section.
        """
        # GeographicProjection uses the same reference_point +
        # sample_spacing convention; stored on plane_projection or
        # directly on the measurement object.  Try plane_projection
        # first (sarpy/sarkit populate it there), then fall back.
        pp = meas.plane_projection
        if pp is None:
            raise ValueError(
                "GeographicProjection requires projection parameters"
            )
        rp = pp.reference_point
        if rp is None or rp.ecef is None or rp.point is None:
            raise ValueError(
                "GeographicProjection requires reference_point"
            )
        if pp.sample_spacing is None:
            raise ValueError(
                "GeographicProjection requires sample_spacing"
            )

        # Convert ECEF reference point to geodetic
        ecef = np.array([rp.ecef.x, rp.ecef.y, rp.ecef.z])
        lat0, lon0, h0 = _ecef_to_geodetic(
            np.array([ecef[0]]),
            np.array([ecef[1]]),
            np.array([ecef[2]]),
        )
        self._lat0 = float(lat0[0])
        self._lon0 = float(lon0[0])
        self._h0 = float(h0[0])
        self._r0 = float(rp.point.row)
        self._c0 = float(rp.point.col)
        # Sample spacing in arc-seconds
        self._dr = float(pp.sample_spacing.row)
        self._dc = float(pp.sample_spacing.col)

    def _init_cylindrical(self, meas: object) -> None:
        """Extract CGD (CylindricalProjection) parameters.

        Parameters
        ----------
        meas : SIDDMeasurement
            Measurement section.
        """
        pp = meas.plane_projection
        if pp is None:
            raise ValueError(
                "CylindricalProjection requires projection parameters"
            )
        rp = pp.reference_point
        if rp is None or rp.ecef is None or rp.point is None:
            raise ValueError(
                "CylindricalProjection requires reference_point"
            )
        if pp.sample_spacing is None:
            raise ValueError(
                "CylindricalProjection requires sample_spacing"
            )
        plane = pp.product_plane
        if plane is None or plane.row_unit_vector is None or plane.col_unit_vector is None:
            raise ValueError(
                "CylindricalProjection requires product_plane "
                "with row_unit_vector and col_unit_vector"
            )

        self._srp = np.array(
            [rp.ecef.x, rp.ecef.y, rp.ecef.z], dtype=np.float64
        )
        self._r0 = float(rp.point.row)
        self._c0 = float(rp.point.col)
        self._dr = float(pp.sample_spacing.row)
        self._dc = float(pp.sample_spacing.col)
        self._row_vec = np.array(
            [plane.row_unit_vector.x,
             plane.row_unit_vector.y,
             plane.row_unit_vector.z],
            dtype=np.float64,
        )
        self._col_vec = np.array(
            [plane.col_unit_vector.x,
             plane.col_unit_vector.y,
             plane.col_unit_vector.z],
            dtype=np.float64,
        )

        # Compute cylinder radius Rs.  If not supplied in metadata,
        # compute from inflated WGS-84 ellipsoid at reference point.
        # (Section 2.8: "If a cylinder radius is not supplied, then a
        #  radius is computed by an inflated ellipsoid.")
        lat0, _, h0 = _ecef_to_geodetic(
            np.array([self._srp[0]]),
            np.array([self._srp[1]]),
            np.array([self._srp[2]]),
        )
        sin_lat = np.sin(np.radians(lat0[0]))
        rc = _WGS84_A / np.sqrt(1.0 - _WGS84_E1_SQ * sin_lat ** 2)
        self._rs = rc + float(h0[0])

        # U' = Z_PGD = row_vec × col_vec (normal to the plane)
        self._u_prime = np.cross(self._row_vec, self._col_vec)
        norm = np.linalg.norm(self._u_prime)
        if norm > 0:
            self._u_prime /= norm

    # ------------------------------------------------------------------
    # R/Rdot refinement initializer
    # ------------------------------------------------------------------

    def _init_rdot_refinement(self, meas: object) -> None:
        """Attempt to build R/Rdot refinement from Measurement params.

        Sets ``self.has_rdot = True`` and populates ``self._coa_proj``
        if both ``TimeCOAPoly`` and ``ARPPoly`` are present.  Otherwise
        silently leaves refinement disabled.

        Parameters
        ----------
        meas : SIDDMeasurement
            Measurement section.
        """
        # Check for required parameters
        pp = meas.plane_projection
        if pp is None or pp.time_coa_poly is None:
            return
        if pp.time_coa_poly.coefs is None:
            return
        if meas.arp_poly is None:
            return
        if (meas.arp_poly.x is None or meas.arp_poly.y is None
                or meas.arp_poly.z is None):
            return

        try:
            from grdl.geolocation.projection import COAProjection
            self._coa_proj = COAProjection.from_sidd(self.metadata)
            # Store SCP ECF for initial reference in HAE iteration
            rp = pp.reference_point
            if rp is not None and rp.ecef is not None:
                self._scp_ecf = np.array(
                    [rp.ecef.x, rp.ecef.y, rp.ecef.z], dtype=np.float64)
            self.has_rdot = True
        except (ValueError, AttributeError):
            # Missing fields — fall back to grid-only
            pass

    # ------------------------------------------------------------------
    # Forward: pixel → lat/lon  (array interface for base class)
    # ------------------------------------------------------------------

    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform pixel coordinates to geographic coordinates.

        Always uses the grid projection for consistency with the
        inverse (``_latlon_to_image_array``).  The grid forward and
        inverse operate on the same surface (the product plane),
        guaranteeing exact round-trips.

        R/Rdot refinement, when available, is exposed via
        :meth:`image_to_latlon_precise` for point queries that need
        sub-meter accuracy on the WGS-84 ellipsoid rather than on
        the product plane.

        Parameters
        ----------
        rows : np.ndarray
            Row pixel coordinates (1D array, float64).
        cols : np.ndarray
            Column pixel coordinates (1D array, float64).
        height : float, default=0.0
            Height above WGS-84 ellipsoid (meters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) in WGS-84 coordinates.
        """
        if self.projection_type == 'PlaneProjection':
            return self._plane_to_latlon(rows, cols)
        elif self.projection_type == 'GeographicProjection':
            return self._geographic_to_latlon(rows, cols)
        elif self.projection_type == 'CylindricalProjection':
            return self._cylindrical_to_latlon(rows, cols)
        raise NotImplementedError(
            f"Forward projection not implemented for "
            f"{self.projection_type!r}"
        )

    def image_to_latlon_precise(
        self,
        row: float,
        col: float,
        height: float = 0.0,
    ) -> Tuple[float, float, float]:
        """R/Rdot-refined forward projection for precise point queries.

        Projects onto the WGS-84 ellipsoid at the given HAE using the
        full R/Rdot sensor model.  More accurate than the grid
        projection away from the reference point, but NOT consistent
        with ``latlon_to_image`` (which uses the grid inverse).

        Use this for geolocation accuracy assessment, not for ortho
        mapping.  Requires ``has_rdot=True``.

        Parameters
        ----------
        row : float
            Row pixel coordinate.
        col : float
            Column pixel coordinate.
        height : float
            Height above WGS-84 (meters).

        Returns
        -------
        Tuple[float, float, float]
            (lat, lon, height) in WGS-84.

        Raises
        ------
        RuntimeError
            If R/Rdot refinement is not available.
        """
        if not self.has_rdot:
            raise RuntimeError(
                "R/Rdot refinement not available (TimeCOAPoly or "
                "ARPPoly missing from SIDD Measurement block)")
        rows = np.array([float(row)])
        cols = np.array([float(col)])
        lats, lons, heights = self._image_to_latlon_rdot(
            rows, cols, height)
        return float(lats[0]), float(lons[0]), float(heights[0])

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform geographic coordinates to pixel coordinates.

        Uses the fast vectorized grid inverse by default.  When R/Rdot
        refinement is available **and** per-point heights differ from
        the product plane (i.e., a non-scalar ``height`` array is
        passed), falls back to R/Rdot iteration to correct for
        elevation-induced parallax.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees (1D array, float64).
        lons : np.ndarray
            Longitudes in degrees (1D array, float64).
        height : float or np.ndarray, default=0.0
            Height above WGS-84 ellipsoid (meters).  Scalar uses
            the fast grid path.  An array of per-point heights
            triggers R/Rdot refinement when available.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays.
        """
        # Use R/Rdot only when per-point heights vary (DEM terrain)
        if self.has_rdot and np.ndim(height) > 0:
            h_arr = np.asarray(height, dtype=np.float64)
            if np.ptp(h_arr) > 1.0:  # >1 m variation → parallax matters
                return self._latlon_to_image_rdot(lats, lons, height)

        # Fast vectorized grid inverse (exact on the product plane)
        if self.projection_type == 'PlaneProjection':
            return self._latlon_to_plane(lats, lons, height)
        elif self.projection_type == 'GeographicProjection':
            return self._latlon_to_geographic(lats, lons)
        elif self.projection_type == 'CylindricalProjection':
            return self._latlon_to_cylindrical(lats, lons, height)
        raise NotImplementedError(
            f"Inverse projection not implemented for "
            f"{self.projection_type!r}"
        )

    # ------------------------------------------------------------------
    # R/Rdot refined projection
    # ------------------------------------------------------------------

    def _image_to_latlon_rdot(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """R/Rdot refined forward projection: pixel → geodetic.

        Uses the native R/Rdot engine (``COAProjection`` +
        ``image_to_ground_hae``) for precise projection.
        """
        from grdl.geolocation.projection import image_to_ground_hae

        im_points = np.column_stack([rows, cols])
        gpp = image_to_ground_hae(
            self._coa_proj,
            im_points,
            hae=height,
            scp_ecf=self._scp_ecf,
        )
        return _ecef_to_geodetic(gpp[:, 0], gpp[:, 1], gpp[:, 2])

    def _latlon_to_image_rdot(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """R/Rdot refined inverse projection: geodetic → pixel.

        Uses the native R/Rdot engine (``ground_to_image``) for
        precise inverse projection.
        """
        from grdl.geolocation.projection import ground_to_image

        if np.ndim(height) == 0:
            h = np.full_like(lats, float(height))
        else:
            h = np.asarray(height, dtype=np.float64)

        x, y, z = _geodetic_to_ecef(lats, lons, h)
        ground_ecf = np.column_stack([x, y, z])

        # Use grid parameters for the image plane
        pp = self.metadata.measurement.plane_projection
        plane = pp.product_plane
        u_row = np.array(
            [plane.row_unit_vector.x, plane.row_unit_vector.y,
             plane.row_unit_vector.z], dtype=np.float64)
        u_col = np.array(
            [plane.col_unit_vector.x, plane.col_unit_vector.y,
             plane.col_unit_vector.z], dtype=np.float64)

        im_points = ground_to_image(
            self._coa_proj,
            ground_ecf,
            scp_ecf=self._scp_ecf,
            u_row=u_row,
            u_col=u_col,
            row_ss=self._dr,
            col_ss=self._dc,
            scp_pixel=(self._r0, self._c0),
        )
        return im_points[:, 0], im_points[:, 1]

    # ------------------------------------------------------------------
    # PGD (PlaneProjection)  — Section 3.2 / 3.3
    # ------------------------------------------------------------------

    def _plane_to_latlon(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """PGD forward: pixel → ECEF → geodetic.

        Section 3.2:
            P_ECEF = P_PGD + dr * R_PGD + dc * C_PGD
        where dr = Δr * (r - r0), dc = Δc * (c - c0).
        """
        # Sensor model distances (meters)
        dr = self._dr * (rows - self._r0)  # (N,)
        dc = self._dc * (cols - self._c0)  # (N,)

        # ECEF position for each pixel  (N, 3)
        x = self._srp[0] + dr * self._row_vec[0] + dc * self._col_vec[0]
        y = self._srp[1] + dr * self._row_vec[1] + dc * self._col_vec[1]
        z = self._srp[2] + dr * self._row_vec[2] + dc * self._col_vec[2]

        return _ecef_to_geodetic(x, y, z)

    def _latlon_to_plane(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """PGD inverse: geodetic → ECEF → pixel.

        Section 3.3:
            r = r0 + ((P_ECEF - P_PGD) · R_PGD) / Δr
            c = c0 + ((P_ECEF - P_PGD) · C_PGD) / Δc
        """
        if np.ndim(height) == 0:
            h = np.full_like(lats, float(height))
        else:
            h = np.asarray(height, dtype=np.float64)

        x, y, z = _geodetic_to_ecef(lats, lons, h)

        # Offset from reference point
        dx = x - self._srp[0]
        dy = y - self._srp[1]
        dz = z - self._srp[2]

        # Dot products with unit vectors
        dot_row = dx * self._row_vec[0] + dy * self._row_vec[1] + dz * self._row_vec[2]
        dot_col = dx * self._col_vec[0] + dy * self._col_vec[1] + dz * self._col_vec[2]

        rows = self._r0 + dot_row / self._dr
        cols = self._c0 + dot_col / self._dc

        return rows, cols

    # ------------------------------------------------------------------
    # GGD (GeographicProjection)  — Section 3.4 / 3.5
    # ------------------------------------------------------------------

    def _geographic_to_latlon(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """GGD forward: pixel → geodetic.

        Section 3.4:
            φ = φ0 - dr / 3600
            λ = λ0 + dc / 3600
            h = h0
        where dr = Δr * (r - r0), dc = Δc * (c - c0), spacing in arc-sec.
        """
        dr = self._dr * (rows - self._r0)
        dc = self._dc * (cols - self._c0)

        lats = self._lat0 - dr / 3600.0
        lons = self._lon0 + dc / 3600.0
        heights = np.full_like(lats, self._h0)

        return lats, lons, heights

    def _latlon_to_geographic(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GGD inverse: geodetic → pixel.

        Section 3.5:
            r = r0 + 3600 * (φ0 - φ) / Δr
            c = c0 + 3600 * (λ - λ0) / Δc
        """
        rows = self._r0 + 3600.0 * (self._lat0 - lats) / self._dr
        cols = self._c0 + 3600.0 * (lons - self._lon0) / self._dc

        return rows, cols

    # ------------------------------------------------------------------
    # CGD (CylindricalProjection)  — Section 3.8 / 3.9
    # ------------------------------------------------------------------

    def _cylindrical_to_latlon(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CGD forward: pixel → ECEF → geodetic.

        Section 3.8:
            θ = dc / R_S
            P_ECEF = P_CGD + dr*R_CGD + R_S*sin(θ)*C_CGD
                     + R_S*(cos(θ) - 1)*U'
        """
        dr = self._dr * (rows - self._r0)
        dc = self._dc * (cols - self._c0)
        theta = dc / self._rs

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        x = (self._srp[0]
             + dr * self._row_vec[0]
             + self._rs * sin_t * self._col_vec[0]
             + self._rs * (cos_t - 1.0) * self._u_prime[0])
        y = (self._srp[1]
             + dr * self._row_vec[1]
             + self._rs * sin_t * self._col_vec[1]
             + self._rs * (cos_t - 1.0) * self._u_prime[1])
        z = (self._srp[2]
             + dr * self._row_vec[2]
             + self._rs * sin_t * self._col_vec[2]
             + self._rs * (cos_t - 1.0) * self._u_prime[2])

        return _ecef_to_geodetic(x, y, z)

    def _latlon_to_cylindrical(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CGD inverse: geodetic → ECEF → pixel.

        Section 3.9:
            r = r0 + ((P_ECEF - P_CGD) · R_CGD) / Δr
            cc = (P_ECEF - P_CGD) · C_CGD
            cu = (P_ECEF - P_CGD) · U'
            θ  = atan2(cc, cu + R_S)
            c  = c0 + R_S * θ / Δc
        """
        if np.ndim(height) == 0:
            h = np.full_like(lats, float(height))
        else:
            h = np.asarray(height, dtype=np.float64)

        x, y, z = _geodetic_to_ecef(lats, lons, h)

        dx = x - self._srp[0]
        dy = y - self._srp[1]
        dz = z - self._srp[2]

        dot_row = dx * self._row_vec[0] + dy * self._row_vec[1] + dz * self._row_vec[2]
        cc = dx * self._col_vec[0] + dy * self._col_vec[1] + dz * self._col_vec[2]
        cu = dx * self._u_prime[0] + dy * self._u_prime[1] + dz * self._u_prime[2]

        theta = np.arctan2(cc, cu + self._rs)

        rows = self._r0 + dot_row / self._dr
        cols = self._c0 + self._rs * theta / self._dc

        return rows, cols

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_reader(cls, reader: object) -> 'SIDDGeolocation':
        """Create SIDDGeolocation from a SIDDReader.

        Parameters
        ----------
        reader : SIDDReader
            An open SIDD reader with populated metadata.

        Returns
        -------
        SIDDGeolocation
            Configured geolocation object.

        Raises
        ------
        ValueError
            If the reader is not a SIDDReader or the metadata lacks
            a supported projection.
        TypeError
            If the reader type is wrong.

        Examples
        --------
        >>> from grdl.IO.sar import SIDDReader
        >>> from grdl.geolocation.sar.sidd import SIDDGeolocation
        >>> with SIDDReader('product.nitf') as reader:
        ...     geo = SIDDGeolocation.from_reader(reader)
        ...     lat, lon, h = geo.image_to_latlon(500, 1000)
        """
        if type(reader).__name__ != 'SIDDReader':
            raise TypeError(
                f"Expected SIDDReader, got {type(reader).__name__}"
            )
        return cls(metadata=reader.metadata)
