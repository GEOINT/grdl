# -*- coding: utf-8 -*-
"""
SICD Geolocation - Coordinate transformations for SICD complex SAR imagery.

Implements the SICD Volume 3 projection model with three backends:

- **native** (preferred): Pure GRDL R/Rdot projection engine using
  ``COAProjection`` and ``image_to_ground_hae``.  No external dependencies
  beyond numpy.  Supports all formation types (PFA, INCA, RgAzComp, PLANE)
  and adjustable parameters (delta_arp, delta_varp, range_bias).
- **sarpy**: Delegates to ``sarpy.geometry.point_projection`` for projection.
- **sarkit**: Detection-time fallback (projection not yet implemented).

Dependencies
------------
sarpy (optional, for sarpy backend)
sarkit (optional, for sarkit backend)

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
2026-02-11

Modified
--------
2026-03-27  Add per-point ellipsoid normal method (_latlon_to_image_native_ppn)
            and per_point_normal constructor parameter.
2026-03-22  Update coordinate function calls to (N, M) stacked convention.
2026-03-16  Add native R/Rdot backend via COAProjection.
2026-02-17  height param broadened to Union[float, np.ndarray] for DEM support.
2026-02-11  Fixed from_reader() to prefer sarpy for projection even when
            reader used sarkit backend.
"""

# Standard library
from typing import Any, Optional, Tuple, Union, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import Geolocation
from grdl.geolocation.sar._backend import _HAS_SARPY, _HAS_SARKIT

if TYPE_CHECKING:
    from grdl.IO.models.sicd import SICDMetadata
    from grdl.IO.sar.sicd import SICDReader
    from grdl.geolocation.projection import COAProjection


def _select_backend(preferred: Optional[str] = None) -> str:
    """Select the best available projection backend.

    Priority: preferred > sarpy > native > sarkit.

    Parameters
    ----------
    preferred : str, optional
        Force a specific backend.

    Returns
    -------
    str
        One of ``'native'``, ``'sarpy'``, ``'sarkit'``.
    """
    if preferred is not None:
        return preferred
    if _HAS_SARPY:
        return 'sarpy'
    # Native is always available (pure numpy)
    return 'native'


class SICDGeolocation(Geolocation):
    """Geolocation for SICD (Sensor Independent Complex Data) imagery.

    Implements the SICD Volume 3 projection model to transform between
    image pixel coordinates (row, col) and geographic coordinates
    (lat, lon, HAE).

    Three backends are available:

    - **native** (default): Pure GRDL R/Rdot engine.  Uses
      :class:`~grdl.geolocation.projection.COAProjection` with
      formation-specific projectors (PFA, INCA, RgAzComp, PLANE).
      No external dependencies.  Supports adjustable parameters.
    - **sarpy**: Delegates to sarpy's ``point_projection`` module.
    - **sarkit**: Detection only; projection raises ``NotImplementedError``.

    Attributes
    ----------
    metadata : SICDMetadata
        Typed SICD metadata with all 17 sections.
    backend : str
        Active projection backend (``'native'``, ``'sarpy'``, or
        ``'sarkit'``).

    Parameters
    ----------
    metadata : SICDMetadata
        Typed SICD metadata.  Must have ``image_data`` with ``scp_pixel``
        and ``geo_data`` with ``scp``.
    raw_meta : Any, optional
        Raw backend metadata object (sarpy ``SICDType`` or sarkit XML).
        Required for ``sarpy`` and ``sarkit`` backends; ignored for
        ``native``.
    backend : str, optional
        Force a specific backend.  If ``None``, uses ``'native'``.
    delta_arp : np.ndarray, optional
        ARP position correction in ECF (meters), shape ``(3,)``.
    delta_varp : np.ndarray, optional
        ARP velocity correction in ECF (m/s), shape ``(3,)``.
    range_bias : float
        Range bias correction (meters).  Applied to all projections.
    dem_path : str or Path, optional
        Path to DEM/DTED data folder for terrain-corrected projection.
    geoid_path : str or Path, optional
        Path to geoid correction file (EGM96/EGM2008).

    Raises
    ------
    ValueError
        If required metadata sections are missing.

    Examples
    --------
    Native backend (no sarpy required):

    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.geolocation.sar.sicd import SICDGeolocation
    >>> with SICDReader('image.nitf') as reader:
    ...     geo = SICDGeolocation(reader.metadata)
    ...     lat, lon, h = geo.image_to_latlon(500, 1000)

    With adjustable parameters:

    >>> geo = SICDGeolocation(
    ...     reader.metadata,
    ...     range_bias=2.5,
    ...     delta_arp=np.array([0.1, -0.2, 0.05]),
    ... )

    From a reader (auto-selects backend):

    >>> geo = SICDGeolocation.from_reader(reader)
    """

    def __init__(
        self,
        metadata: 'SICDMetadata',
        raw_meta: Any = None,
        backend: Optional[str] = None,
        delta_arp: Optional[np.ndarray] = None,
        delta_varp: Optional[np.ndarray] = None,
        range_bias: float = 0.0,
        dem_path: Optional[str] = None,
        geoid_path: Optional[str] = None,
        per_point_normal: bool = True,
    ) -> None:
        backend = _select_backend(backend)
        self._use_ppn = per_point_normal

        # Validate required metadata sections
        if metadata.image_data is None:
            raise ValueError(
                "SICDMetadata.image_data is required for geolocation"
            )
        if metadata.image_data.scp_pixel is None:
            raise ValueError(
                "SICDMetadata.image_data.scp_pixel is required for "
                "geolocation (Scene Center Point pixel location)"
            )
        if metadata.geo_data is None:
            raise ValueError(
                "SICDMetadata.geo_data is required for geolocation"
            )
        if metadata.geo_data.scp is None:
            raise ValueError(
                "SICDMetadata.geo_data.scp is required for geolocation "
                "(Scene Center Point geographic location)"
            )

        self.metadata = metadata
        self.backend = backend
        self._raw_meta = raw_meta
        self._delta_arp = delta_arp
        self._delta_varp = delta_varp
        self._range_bias = range_bias

        shape = (metadata.rows, metadata.cols)

        super().__init__(
            shape, crs='WGS84', dem_path=dem_path, geoid_path=geoid_path
        )

        # Always try to build native COAProjection (needed for DEM
        # integration).  Falls back to sarpy/sarkit if metadata is
        # insufficient for native projection.
        self._coa_proj = None
        try:
            self._coa_proj = self._build_native_projection()
        except (ValueError, AttributeError):
            pass  # Missing Grid/Position metadata — native unavailable

        # When the native R/Rdot engine is available, it passes
        # self.elevation directly into image_to_ground_hae() which does
        # correct per-point DEM iteration.  Tell the base class not to
        # add a redundant outer DEM loop.
        self._handles_dem_internally = self._coa_proj is not None

        if backend == 'sarpy':
            self._sarpy_meta = raw_meta
        elif backend == 'sarkit':
            self._xmltree = raw_meta
            self._sarkit_xml_ref = raw_meta

    def _get_scp_hae(self) -> float:
        """Return the SCP geodetic height (HAE) from metadata."""
        if self.metadata.geo_data.scp.llh is not None:
            return float(self.metadata.geo_data.scp.llh.hae)
        scp_ecf = self.metadata.geo_data.scp.ecf.to_array()
        from grdl.geolocation.coordinates import ecef_to_geodetic
        geo = ecef_to_geodetic(scp_ecf)
        return float(geo[2])

    # ------------------------------------------------------------------
    # Standardized public properties
    # ------------------------------------------------------------------

    @property
    def default_hae(self) -> float:
        """Default height above ellipsoid from the Scene Center Point.

        Returns
        -------
        float
            SCP height above WGS-84 ellipsoid in meters.
        """
        return self._get_scp_hae()

    @property
    def projection_type(self) -> str:
        """Projection model type.

        SICD always uses the R/Rdot (Range-Doppler) sensor model.

        Returns
        -------
        str
            ``'R/Rdot'``
        """
        return 'R/Rdot'

    @property
    def has_rdot(self) -> bool:
        """Whether R/Rdot projection is active and functional.

        Returns
        -------
        bool
            True when the native COAProjection or sarpy backend is
            available for R/Rdot geolocation.
        """
        return self._coa_proj is not None or self.backend == 'sarpy'

    # ------------------------------------------------------------------
    # Native R/Rdot backend
    # ------------------------------------------------------------------

    def _build_native_projection(self) -> 'COAProjection':
        """Build the native COAProjection from GRDL SICDMetadata.

        Returns
        -------
        COAProjection
            Configured R/Rdot projection object.
        """
        from grdl.geolocation.projection import COAProjection

        return COAProjection.from_sicd(
            self.metadata,
            delta_arp=self._delta_arp,
            delta_varp=self._delta_varp,
            range_bias=self._range_bias,
        )

    def _image_to_latlon_native(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project image coordinates to ground via native R/Rdot engine.

        Parameters
        ----------
        rows : np.ndarray
            Row coordinates (1D array, float64).
        cols : np.ndarray
            Column coordinates (1D array, float64).
        height : float or np.ndarray, default=0.0
            Height above WGS84 ellipsoid (meters).  When
            ``self.elevation`` is set, this serves as the initial guess
            for the R/Rdot DEM iteration.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) arrays in WGS84 coordinates.
        """
        from grdl.geolocation.projection import image_to_ground_hae
        from grdl.geolocation.coordinates import ecef_to_geodetic

        im_points = np.column_stack([rows, cols])
        scp_ecf = self.metadata.geo_data.scp.ecf.to_array()

        # Default HAE: SCP height (never 0 for real scenes).
        # When height is an array, use the mean as the initial HAE
        # for the R/Rdot engine (it does its own per-point DEM
        # iteration internally when self.elevation is set).
        if np.ndim(height) > 0:
            h_arr = np.asarray(height, dtype=np.float64)
            hae = float(np.mean(h_arr)) if np.any(h_arr != 0.0) \
                else self._get_scp_hae()
        else:
            hae = float(height) if height != 0.0 else self._get_scp_hae()

        gpp = image_to_ground_hae(
            self._coa_proj,
            im_points,
            hae=hae,
            scp_ecf=scp_ecf,
            elevation_model=self.elevation,
        )

        geo = ecef_to_geodetic(gpp)
        lats, lons, heights = geo[:, 0], geo[:, 1], geo[:, 2]

        return lats, lons, heights

    def _latlon_to_image_native(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project ground coordinates to image via native R/Rdot engine.

        Fully vectorized scene-to-image per SICD Volume 3, Section 6.
        All N points processed simultaneously — one R/Rdot forward
        call per iteration.  Follows sarpy's ``_ground_to_image``
        algorithm.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North (1D array, float64).
        lons : np.ndarray
            Longitudes in degrees East (1D array, float64).
        height : float or np.ndarray, default=0.0
            Height above WGS84 ellipsoid (meters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays.
        """
        from grdl.geolocation.projection import (
            image_to_ground_plane, wgs84_norm)
        from grdl.geolocation.coordinates import geodetic_to_ecef

        # Determine per-point heights: DEM → explicit → SCP HAE
        if self.elevation is not None:
            heights_arr = self.elevation.get_elevation(lats, lons)
            if isinstance(heights_arr, (int, float)):
                heights_arr = np.full_like(lats, float(heights_arr))
            nan_mask = np.isnan(heights_arr)
            if np.any(nan_mask):
                # Fill gaps with explicit height or SCP HAE
                fill = float(height) if height != 0.0 else self._get_scp_hae()
                heights_arr[nan_mask] = fill
        elif np.ndim(height) > 0:
            heights_arr = np.asarray(height, dtype=np.float64)
        elif height != 0.0:
            heights_arr = np.full_like(lats, float(height))
        else:
            heights_arr = np.full_like(lats, self._get_scp_hae())

        # Convert to ECF (height is embedded)
        coords = geodetic_to_ecef(
            np.column_stack([lats, lons, heights_arr]))

        # Image plane parameters
        grid = self.metadata.grid
        scp_ecf = self.metadata.geo_data.scp.ecf.to_array()
        u_row = (grid.row.uvect_ecf.to_array()
                 if grid.row and grid.row.uvect_ecf else
                 np.array([1.0, 0.0, 0.0]))
        u_col = (grid.col.uvect_ecf.to_array()
                 if grid.col and grid.col.uvect_ecf else
                 np.array([0.0, 1.0, 0.0]))
        row_ss = grid.row.ss if grid.row and grid.row.ss else 1.0
        col_ss = grid.col.ss if grid.col and grid.col.ss else 1.0
        scp_pixel = np.array([self.metadata.image_data.scp_pixel.row,
                              self.metadata.image_data.scp_pixel.col])

        # Image Plane Normal and projection vectors
        u_ipn = np.cross(u_row, u_col)
        u_ipn = u_ipn / np.linalg.norm(u_ipn)
        u_gpn = wgs84_norm(scp_ecf)
        u_spn = u_ipn
        sf = float(np.dot(u_spn, u_ipn))

        # Handle non-orthogonal row/col
        cos_theta = np.dot(u_row, u_col)
        sin_theta = np.sqrt(max(1.0 - cos_theta * cos_theta, 1e-30))
        ipp_transform = np.array(
            [[1, -cos_theta], [-cos_theta, 1]],
            dtype=np.float64) / (sin_theta * sin_theta)
        matrix_transform = np.column_stack([u_row, u_col]) @ ipp_transform

        n = coords.shape[0]
        g_n = coords.copy()
        im_points = np.zeros((n, 2), dtype=np.float64)

        for _it in range(10):
            # Project ground points onto image plane
            dist_n = np.dot(scp_ecf - g_n, u_ipn) / sf
            i_n = g_n + np.outer(dist_n, u_spn)

            # Convert to pixel coordinates
            delta_ipp = i_n - scp_ecf
            ip_iter = delta_ipp @ matrix_transform
            im_points[:, 0] = ip_iter[:, 0] / row_ss + scp_pixel[0]
            im_points[:, 1] = ip_iter[:, 1] / col_ss + scp_pixel[1]

            # R/Rdot forward to ground plane at scene points
            r, rdot, _, arp_coa, varp_coa = \
                self._coa_proj.projection(im_points)
            p_n = image_to_ground_plane(
                r, rdot, arp_coa, varp_coa, g_n, u_gpn)

            # Displacement and adjustment
            diff_n = coords - p_n
            delta_gpn = np.linalg.norm(diff_n, axis=1)
            g_n += diff_n

            if np.all(delta_gpn < 1e-2):
                break

        return im_points[:, 0], im_points[:, 1]

    def _latlon_to_image_native_ppn(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ground-to-image with per-point WGS-84 ellipsoid normals.

        Same algorithm as ``_latlon_to_image_native`` but recomputes
        the ground plane normal from each point's ECF position at every
        iteration instead of using a single SCP-derived normal for the
        entire batch.  This eliminates the flat-earth approximation
        error that grows with distance from the SCP.

        For scenes spanning <10 km the improvement is sub-centimeter.
        For 50+ km scenes (e.g., Maritime Surveillance, spotlight mosaics)
        the improvement can reach 0.5-1 m.

        Parameters
        ----------
        lats, lons : np.ndarray
            Target geodetic coordinates, each shape ``(N,)``.
        height : float or np.ndarray, default=0.0
            Height above WGS-84 ellipsoid (meters).  Overridden by
            DEM when ``self.elevation`` is set.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays.
        """
        from grdl.geolocation.projection import (
            image_to_ground_plane, wgs84_norm,
        )

        # Height resolution — same as _latlon_to_image_native
        if self.elevation is not None:
            heights_arr = self.elevation.get_elevation(lats, lons)
            if isinstance(heights_arr, (int, float)):
                heights_arr = np.full_like(lats, float(heights_arr))
            nan_mask = np.isnan(heights_arr)
            if np.any(nan_mask):
                fill = float(height) if height != 0.0 else self._get_scp_hae()
                heights_arr[nan_mask] = fill
        elif np.ndim(height) > 0:
            heights_arr = np.asarray(height, dtype=np.float64)
        elif height != 0.0:
            heights_arr = np.full_like(lats, float(height))
        else:
            heights_arr = np.full_like(lats, self._get_scp_hae())

        # Convert to ECF
        coords = geodetic_to_ecef(
            np.column_stack([lats, lons, heights_arr]))

        # Image plane parameters (same setup as _latlon_to_image_native)
        grid = self.metadata.grid
        scp_ecf = self.metadata.geo_data.scp.ecf.to_array()
        u_row = (grid.row.uvect_ecf.to_array()
                 if grid.row and grid.row.uvect_ecf else
                 np.array([1.0, 0.0, 0.0]))
        u_col = (grid.col.uvect_ecf.to_array()
                 if grid.col and grid.col.uvect_ecf else
                 np.array([0.0, 1.0, 0.0]))
        row_ss = grid.row.ss if grid.row and grid.row.ss else 1.0
        col_ss = grid.col.ss if grid.col and grid.col.ss else 1.0
        scp_pixel = np.array([self.metadata.image_data.scp_pixel.row,
                              self.metadata.image_data.scp_pixel.col])

        u_ipn = np.cross(u_row, u_col)
        u_ipn = u_ipn / np.linalg.norm(u_ipn)
        u_spn = u_ipn
        sf = float(np.dot(u_spn, u_ipn))

        # Handle non-orthogonal row/col
        cos_theta = np.dot(u_row, u_col)
        sin_theta = np.sqrt(max(1.0 - cos_theta * cos_theta, 1e-30))
        ipp_transform = np.array(
            [[1, -cos_theta], [-cos_theta, 1]],
            dtype=np.float64) / (sin_theta * sin_theta)
        matrix_transform = np.column_stack([u_row, u_col]) @ ipp_transform

        n = coords.shape[0]
        g_n = coords.copy()
        im_points = np.zeros((n, 2), dtype=np.float64)

        for _it in range(10):
            # Per-point ellipsoid normal from current ground estimate
            u_gpn = wgs84_norm(g_n)  # (N, 3)

            # Project ground points onto image plane
            dist_n = np.dot(scp_ecf - g_n, u_ipn) / sf
            i_n = g_n + np.outer(dist_n, u_spn)

            # Convert to pixel coordinates
            delta_ipp = i_n - scp_ecf
            ip_iter = delta_ipp @ matrix_transform
            im_points[:, 0] = ip_iter[:, 0] / row_ss + scp_pixel[0]
            im_points[:, 1] = ip_iter[:, 1] / col_ss + scp_pixel[1]

            # R/Rdot forward to ground plane with per-point normals
            r, rdot, _, arp_coa, varp_coa = \
                self._coa_proj.projection(im_points)
            p_n = image_to_ground_plane(
                r, rdot, arp_coa, varp_coa, g_n, u_gpn)

            # Displacement and adjustment
            diff_n = coords - p_n
            delta_gpn = np.linalg.norm(diff_n, axis=1)
            g_n += diff_n

            if np.all(delta_gpn < 1e-2):
                break

        return im_points[:, 0], im_points[:, 1]

    # ------------------------------------------------------------------
    # Public interface dispatch
    # ------------------------------------------------------------------

    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform pixel coordinate arrays to geographic coordinate arrays.

        Always uses the native R/Rdot engine which supports DEM
        integration.  The native backend handles all formation types
        (PFA, INCA, RgAzComp, PLANE) and passes the elevation model
        into the R/Rdot iteration loop.

        Parameters
        ----------
        rows : np.ndarray
            Row coordinates (1D array, float64).
        cols : np.ndarray
            Column coordinates (1D array, float64).
        height : float or np.ndarray, default=0.0
            Height above WGS84 ellipsoid (meters).  Scalar applies a
            constant height to all points.  An array of shape ``(N,)``
            provides per-point heights.  When ``self.elevation`` is set,
            the R/Rdot engine uses the DEM directly and this parameter
            serves as the initial guess.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) arrays in WGS84 coordinates.
        """
        # Native backend first (handles DEM in R/Rdot iteration).
        # Falls back to sarpy if native projection is not available
        # (missing Grid/Position metadata for COAProjection).
        if self._coa_proj is not None:
            return self._image_to_latlon_native(rows, cols, height)
        elif self.backend == 'sarpy':
            return self._image_to_latlon_sarpy(rows, cols, height)
        else:
            return self._image_to_latlon_sarkit(rows, cols, height)

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform geographic coordinate arrays to pixel coordinate arrays.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North (1D array, float64).
        lons : np.ndarray
            Longitudes in degrees East (1D array, float64).
        height : float or np.ndarray, default=0.0
            Height above WGS84 ellipsoid (meters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays.
        """
        if self._coa_proj is not None:
            if self._use_ppn:
                return self._latlon_to_image_native_ppn(
                    lats, lons, height)
            return self._latlon_to_image_native(lats, lons, height)
        elif self.backend == 'sarpy':
            return self._latlon_to_image_sarpy(lats, lons, height)
        else:
            return self._latlon_to_image_sarkit(lats, lons, height)

    # ------------------------------------------------------------------
    # sarpy backend
    # ------------------------------------------------------------------

    def _image_to_latlon_sarpy(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project image coordinates to ground via sarpy."""
        from sarpy.geometry.point_projection import image_to_ground_geo

        im_points = np.column_stack([rows, cols])
        ground_geo = image_to_ground_geo(
            im_points,
            self._sarpy_meta,
            ordering='latlong',
            projection_type='HAE',
        )
        return ground_geo[:, 0], ground_geo[:, 1], ground_geo[:, 2]

    def _latlon_to_image_sarpy(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project ground coordinates to image via sarpy."""
        from sarpy.geometry.point_projection import ground_to_image_geo

        if np.ndim(height) > 0:
            heights_arr = np.asarray(height, dtype=np.float64)
        else:
            heights_arr = np.full_like(lats, height)
        coords = np.column_stack([lats, lons, heights_arr])

        image_points, _, _ = ground_to_image_geo(
            coords,
            self._sarpy_meta,
            ordering='latlong',
        )
        return image_points[:, 0], image_points[:, 1]

    # ------------------------------------------------------------------
    # sarkit backend (stub)
    # ------------------------------------------------------------------

    def _image_to_latlon_sarkit(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project image coordinates to ground via sarkit."""
        raise NotImplementedError(
            "SICD projection via sarkit is not yet implemented. "
            "Use backend='native' or install sarpy: pip install sarpy"
        )

    def _latlon_to_image_sarkit(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project ground coordinates to image via sarkit."""
        raise NotImplementedError(
            "SICD inverse projection via sarkit is not yet implemented. "
            "Use backend='native' or install sarpy: pip install sarpy"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_reader(
        cls,
        reader: 'SICDReader',
        backend: Optional[str] = None,
        delta_arp: Optional[np.ndarray] = None,
        delta_varp: Optional[np.ndarray] = None,
        range_bias: float = 0.0,
        per_point_normal: bool = False,
    ) -> 'SICDGeolocation':
        """Create SICDGeolocation from a SICDReader instance.

        Extracts the raw backend metadata from the reader and constructs
        the geolocation object with the best available projection backend.

        Parameters
        ----------
        reader : SICDReader
            An open SICD reader with loaded metadata.
        backend : str, optional
            Force a specific backend (``'native'``, ``'sarpy'``,
            ``'sarkit'``).  Default auto-selects ``'native'``.
        delta_arp : np.ndarray, optional
            ARP position correction in ECF (meters).
        delta_varp : np.ndarray, optional
            ARP velocity correction in ECF (m/s).
        range_bias : float
            Range bias correction (meters).

        Returns
        -------
        SICDGeolocation

        Examples
        --------
        >>> with SICDReader('image.nitf') as reader:
        ...     geo = SICDGeolocation.from_reader(reader)
        ...     lat, lon, h = geo.image_to_latlon(500, 1000)
        """
        proj_backend = _select_backend(backend)

        raw_meta = None
        if proj_backend == 'sarpy':
            if reader.backend == 'sarpy':
                raw_meta = reader._sarpy_meta
            elif _HAS_SARPY:
                from sarpy.io.complex.converter import open_complex
                sarpy_reader = open_complex(str(reader.filepath))
                raw_meta = sarpy_reader.sicd_meta
        elif proj_backend == 'sarkit':
            if reader.backend == 'sarkit':
                raw_meta = reader._xmltree
            elif reader.backend == 'sarpy':
                raw_meta = reader._sarpy_meta

        return cls(
            metadata=reader.metadata,
            raw_meta=raw_meta,
            backend=proj_backend,
            delta_arp=delta_arp,
            delta_varp=delta_varp,
            range_bias=range_bias,
            per_point_normal=per_point_normal,
        )
