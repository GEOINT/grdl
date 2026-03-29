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
2026-03-27  Add per-point ellipsoid normal method (_latlon_to_image_rdot_ppn)
            and per_point_normal constructor parameter.
2026-03-22  Update coordinate function calls to (N, M) stacked convention.
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
from grdl.geolocation.coordinates import (
    WGS84_A as _WGS84_A,
    WGS84_B as _WGS84_B,
    WGS84_F as _WGS84_F,
    WGS84_E1_SQ as _WGS84_E1_SQ,
    WGS84_E2_SQ as _WGS84_E2_SQ,
    geodetic_to_ecef as _geodetic_to_ecef,
    ecef_to_geodetic as _ecef_to_geodetic,
)

if TYPE_CHECKING:
    from grdl.IO.models.sidd import SIDDMetadata


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
    backend : str
        Projection backend (always ``'native'`` for SIDD).

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
        per_point_normal: bool = True,
    ) -> None:
        self.metadata = metadata
        self.backend = 'native'
        self._use_ppn = per_point_normal

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

        # Compute default HAE from the measurement reference point.
        # This is used as the projection surface when no DEM or
        # explicit height is provided — avoids projecting at h=0
        # when the scene is at significant altitude.
        self._default_hae = 0.0
        if self.projection_type == 'PlaneProjection' and hasattr(self, '_srp'):
            # _srp is guaranteed set by _init_plane; derive HAE from it
            _geo = _ecef_to_geodetic(self._srp)
            self._default_hae = float(_geo[2])
        else:
            pp = meas.plane_projection
            if pp is not None and pp.reference_point is not None:
                rp_ecf = pp.reference_point.ecef
                if rp_ecf is not None:
                    _geo = _ecef_to_geodetic(
                        np.array([rp_ecf.x, rp_ecf.y, rp_ecf.z]))
                    self._default_hae = float(_geo[2])

        # R/Rdot refinement: requires TimeCOAPoly + ARPPoly
        self.has_rdot = False
        self._coa_proj = None
        self._scp_ecf = None
        if refine:
            self._init_rdot_refinement(meas)

        # When R/Rdot is active, _image_to_latlon_rdot_fwd passes
        # self.elevation into image_to_ground_hae() which does per-point
        # DEM iteration internally.  Also, _latlon_to_image_array handles
        # DEM lookup when has_rdot is True.  Tell the base class not to
        # add redundant DEM loops.
        self._handles_dem_internally = self.has_rdot

        shape = (metadata.rows, metadata.cols)
        super().__init__(
            shape, crs='WGS84', dem_path=dem_path, geoid_path=geoid_path
        )

    # ------------------------------------------------------------------
    # Standardized public properties
    # ------------------------------------------------------------------

    @property
    def default_hae(self) -> float:
        """Default height above ellipsoid from the measurement reference point.

        Returns
        -------
        float
            Reference point height above WGS-84 ellipsoid in meters.
        """
        return self._default_hae

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
        _geo = _ecef_to_geodetic(ecef)
        self._lat0 = float(_geo[0])
        self._lon0 = float(_geo[1])
        self._h0 = float(_geo[2])
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
        _geo_cyl = _ecef_to_geodetic(self._srp)
        sin_lat = np.sin(np.radians(_geo_cyl[0]))
        rc = _WGS84_A / np.sqrt(1.0 - _WGS84_E1_SQ * sin_lat ** 2)
        self._rs = rc + float(_geo_cyl[2])

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
        # Check for required parameters — use getattr to tolerate
        # duck-typed metadata objects that may omit optional fields.
        pp = getattr(meas, 'plane_projection', None)
        if pp is None:
            return
        time_coa = getattr(pp, 'time_coa_poly', None)
        if time_coa is None or getattr(time_coa, 'coefs', None) is None:
            return
        arp = getattr(meas, 'arp_poly', None)
        if arp is None:
            return
        if (getattr(arp, 'x', None) is None
                or getattr(arp, 'y', None) is None
                or getattr(arp, 'z', None) is None):
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
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform pixel coordinates to geographic coordinates.

        Model selection (determined at init, used consistently):

        - **R/Rdot + DEM**: Projects to terrain surface via R/Rdot
          with DEM heights fed into the iteration loop.
        - **R/Rdot only**: Projects to constant HAE (SCP height).
        - **Grid only**: PlaneProjection / GGD / CGD (no R/Rdot).
        """
        if self.has_rdot:
            if np.ndim(height) > 0:
                h_arr = np.asarray(height, dtype=np.float64)
                hae = float(np.mean(h_arr)) if np.any(h_arr != 0.0) \
                    else self._default_hae
            else:
                hae = float(height) if height != 0.0 else self._default_hae
            return self._image_to_latlon_rdot_fwd(rows, cols, hae)

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

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform geographic coordinates to pixel coordinates.

        Uses the same model as ``_image_to_latlon_array`` for
        consistency:

        - **R/Rdot**: Vectorized scene-to-image (SICD Vol 3 Sec 6).
          Heights: DEM → explicit → SCP HAE.
        - **Grid only**: Fast vectorized plane inverse.
        """
        if self.has_rdot:
            # Height chain: explicit array → DEM → SCP HAE.
            # When the caller passes a per-pixel height array (e.g.
            # from Orthorectifier's DEM lookup), use it directly.
            # Only fall back to self.elevation when height is scalar 0.
            if np.ndim(height) > 0:
                h_arr = np.asarray(height, dtype=np.float64)
            elif self.elevation is not None and height == 0.0:
                h_arr = self.elevation.get_elevation(lats, lons)
                if isinstance(h_arr, (int, float)):
                    h_arr = np.full_like(lats, float(h_arr))
                nan_mask = np.isnan(h_arr)
                if np.any(nan_mask):
                    h_arr[nan_mask] = self._default_hae
            else:
                h = height if height != 0.0 else self._default_hae
                h_arr = np.full_like(lats, float(h))
            if self._use_ppn:
                return self._latlon_to_image_rdot_ppn(
                    lats, lons, h_arr)
            return self._latlon_to_image_rdot(lats, lons, h_arr)

        # Grid-only path: DEM → explicit height → _default_hae.
        # Terrain height matters for PlaneProjection — a ground point
        # at a different height than the reference plane projects to a
        # different pixel.  Error scales with
        # (h_terrain - h_plane) * tan(incidence_angle) / sample_spacing.
        if np.ndim(height) > 0:
            h = np.asarray(height, dtype=np.float64)
        elif self.elevation is not None and height == 0.0:
            h = self.elevation.get_elevation(lats, lons)
            if isinstance(h, (int, float)):
                h = np.full_like(lats, float(h))
            nan_mask = np.isnan(h)
            if np.any(nan_mask):
                h[nan_mask] = self._default_hae
        else:
            h = height if height != 0.0 else self._default_hae
        if self.projection_type == 'PlaneProjection':
            return self._latlon_to_plane(lats, lons, h)
        elif self.projection_type == 'GeographicProjection':
            return self._latlon_to_geographic(lats, lons)
        elif self.projection_type == 'CylindricalProjection':
            return self._latlon_to_cylindrical(lats, lons, h)
        raise NotImplementedError(
            f"Inverse projection not implemented for "
            f"{self.projection_type!r}"
        )

    def _latlon_to_image_rdot(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        heights: np.ndarray,
        max_iter: int = 10,
        tol: float = 1e-2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """R/Rdot scene-to-image per SICD Volume 3, Section 6.

        Fully vectorized iterative projection following the same
        algorithm as sarpy's ``_ground_to_image``:

        1. Project ECF ground points onto the image plane.
        2. Convert to pixel coordinates.
        3. Forward-project those pixels back to the ground plane
           containing the original scene points via R/Rdot.
        4. Compute displacement between original and re-projected.
        5. Adjust ground points by the displacement and repeat.

        All N points are processed simultaneously — one R/Rdot forward
        call per iteration, no per-point loops.

        Parameters
        ----------
        lats, lons, heights : np.ndarray
            Target geodetic coordinates, each shape ``(N,)``.
        max_iter : int
            Maximum iterations.
        tol : float
            Ground plane displacement tolerance (meters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays.
        """
        from grdl.geolocation.projection import image_to_ground_plane

        # Convert target geodetic to ECF
        coords = _geodetic_to_ecef(
            np.column_stack([lats, lons, heights]))  # (N, 3) target ECF

        # Image plane parameters
        ref_point = self._srp                # (3,) ECF
        ref_pixel = np.array([self._r0, self._c0])
        u_row = self._row_vec                # (3,) unit vector
        u_col = self._col_vec                # (3,) unit vector
        row_ss = self._dr
        col_ss = self._dc

        # Image Plane Normal
        u_ipn = np.cross(u_row, u_col)
        u_ipn = u_ipn / np.linalg.norm(u_ipn)

        # Ground Plane Normal (WGS-84 ellipsoid normal at reference)
        from grdl.geolocation.projection import wgs84_norm
        u_gpn = wgs84_norm(ref_point)

        # Slant Plane Normal: for SIDD PGD the image plane IS the
        # slant plane, so uSPN = uIPN.  Scale factor for projection.
        u_spn = u_ipn
        sf = float(np.dot(u_spn, u_ipn))

        # Handle non-orthogonal row/col (general case)
        cos_theta = np.dot(u_row, u_col)
        sin_theta = np.sqrt(max(1.0 - cos_theta * cos_theta, 1e-30))
        ipp_transform = np.array(
            [[1, -cos_theta], [-cos_theta, 1]],
            dtype=np.float64) / (sin_theta * sin_theta)
        row_col_transform = np.column_stack([u_row, u_col])  # (3, 2)
        matrix_transform = row_col_transform @ ipp_transform  # (3, 2)

        n = coords.shape[0]
        g_n = coords.copy()  # current ground point estimates
        im_points = np.zeros((n, 2), dtype=np.float64)

        for _it in range(max_iter):
            # Step 1: Project ground points onto image plane
            dist_n = np.dot(ref_point - g_n, u_ipn) / sf  # (N,)
            i_n = g_n + np.outer(dist_n, u_spn)  # (N, 3)

            # Step 2: Convert image plane points to pixel coordinates
            delta_ipp = i_n - ref_point  # (N, 3)
            ip_iter = delta_ipp @ matrix_transform  # (N, 2)
            im_points[:, 0] = ip_iter[:, 0] / row_ss + ref_pixel[0]
            im_points[:, 1] = ip_iter[:, 1] / col_ss + ref_pixel[1]

            # Step 3: R/Rdot forward — project pixels to ground plane
            #         containing the scene points
            r, rdot, t_coa, arp_coa, varp_coa = \
                self._coa_proj.projection(im_points)
            p_n = image_to_ground_plane(
                r, rdot, arp_coa, varp_coa, g_n, u_gpn)

            # Step 4: Displacement between original and re-projected
            diff_n = coords - p_n
            delta_gpn = np.linalg.norm(diff_n, axis=1)

            # Step 5: Adjust ground point estimates
            g_n += diff_n

            if np.all(delta_gpn < tol):
                break

        return im_points[:, 0], im_points[:, 1]

    def _latlon_to_image_rdot_ppn(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        heights: np.ndarray,
        max_iter: int = 10,
        tol: float = 1e-2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """R/Rdot scene-to-image with per-point WGS-84 ellipsoid normals.

        Same algorithm as ``_latlon_to_image_rdot`` but recomputes
        the ground plane normal from each point's ECF position at every
        iteration.  Eliminates the flat-earth approximation error that
        grows with distance from the reference point.

        Parameters
        ----------
        lats, lons, heights : np.ndarray
            Target geodetic coordinates, each shape ``(N,)``.
        max_iter : int
            Maximum iterations.
        tol : float
            Ground plane displacement tolerance (meters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays.
        """
        from grdl.geolocation.projection import (
            image_to_ground_plane, wgs84_norm,
        )

        # Convert target geodetic to ECF
        coords = _geodetic_to_ecef(
            np.column_stack([lats, lons, heights]))

        # Image plane parameters (same setup as _latlon_to_image_rdot)
        ref_point = self._srp
        ref_pixel = np.array([self._r0, self._c0])
        u_row = self._row_vec
        u_col = self._col_vec
        row_ss = self._dr
        col_ss = self._dc

        u_ipn = np.cross(u_row, u_col)
        u_ipn = u_ipn / np.linalg.norm(u_ipn)
        u_spn = u_ipn
        sf = float(np.dot(u_spn, u_ipn))

        cos_theta = np.dot(u_row, u_col)
        sin_theta = np.sqrt(max(1.0 - cos_theta * cos_theta, 1e-30))
        ipp_transform = np.array(
            [[1, -cos_theta], [-cos_theta, 1]],
            dtype=np.float64) / (sin_theta * sin_theta)
        row_col_transform = np.column_stack([u_row, u_col])
        matrix_transform = row_col_transform @ ipp_transform

        n = coords.shape[0]
        g_n = coords.copy()
        im_points = np.zeros((n, 2), dtype=np.float64)

        for _it in range(max_iter):
            # Per-point ellipsoid normal from current ground estimate
            u_gpn = wgs84_norm(g_n)  # (N, 3)

            # Project ground points onto image plane
            dist_n = np.dot(ref_point - g_n, u_ipn) / sf
            i_n = g_n + np.outer(dist_n, u_spn)

            # Convert to pixel coordinates
            delta_ipp = i_n - ref_point
            ip_iter = delta_ipp @ matrix_transform
            im_points[:, 0] = ip_iter[:, 0] / row_ss + ref_pixel[0]
            im_points[:, 1] = ip_iter[:, 1] / col_ss + ref_pixel[1]

            # R/Rdot forward with per-point normals
            r, rdot, t_coa, arp_coa, varp_coa = \
                self._coa_proj.projection(im_points)
            p_n = image_to_ground_plane(
                r, rdot, arp_coa, varp_coa, g_n, u_gpn)

            diff_n = coords - p_n
            delta_gpn = np.linalg.norm(diff_n, axis=1)
            g_n += diff_n

            if np.all(delta_gpn < tol):
                break

        return im_points[:, 0], im_points[:, 1]

    # ------------------------------------------------------------------
    # R/Rdot refined projection
    # ------------------------------------------------------------------

    def _image_to_latlon_rdot_fwd(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """R/Rdot forward projection: pixel → geodetic.

        When an elevation model is configured on this geolocation
        object, the DEM is queried at each iteration to refine the
        projection surface.  Otherwise projects to constant HAE.
        """
        from grdl.geolocation.projection import image_to_ground_hae

        rows = np.atleast_1d(np.asarray(rows, dtype=np.float64))
        cols = np.atleast_1d(np.asarray(cols, dtype=np.float64))
        im_points = np.column_stack([rows, cols])
        gpp = image_to_ground_hae(
            self._coa_proj, im_points, hae=height,
            scp_ecf=self._scp_ecf,
            elevation_model=self.elevation,
        )
        geo = _ecef_to_geodetic(gpp)
        return geo[:, 0], geo[:, 1], geo[:, 2]

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

        geo = _ecef_to_geodetic(np.column_stack([x, y, z]))
        return geo[:, 0], geo[:, 1], geo[:, 2]

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

        ecef = _geodetic_to_ecef(np.column_stack([lats, lons, h]))

        # Offset from reference point
        dx = ecef[:, 0] - self._srp[0]
        dy = ecef[:, 1] - self._srp[1]
        dz = ecef[:, 2] - self._srp[2]

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

        geo = _ecef_to_geodetic(np.column_stack([x, y, z]))
        return geo[:, 0], geo[:, 1], geo[:, 2]

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

        ecef = _geodetic_to_ecef(np.column_stack([lats, lons, h]))

        dx = ecef[:, 0] - self._srp[0]
        dy = ecef[:, 1] - self._srp[1]
        dz = ecef[:, 2] - self._srp[2]

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
    def from_reader(
        cls,
        reader: object,
        refine: bool = True,
        dem_path: Optional[str] = None,
        geoid_path: Optional[str] = None,
    ) -> 'SIDDGeolocation':
        """Create SIDDGeolocation from a SIDDReader.

        Parameters
        ----------
        reader : SIDDReader
            An open SIDD reader with populated metadata.
        refine : bool, default=True
            Enable R/Rdot refinement when TimeCOAPoly and ARPPoly
            are available.
        dem_path : str or Path, optional
            Path to DEM/DTED data folder.
        geoid_path : str or Path, optional
            Path to geoid correction file (EGM96/EGM2008).

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
        return cls(
            metadata=reader.metadata,
            refine=refine,
            dem_path=dem_path,
            geoid_path=geoid_path,
        )
