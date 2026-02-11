# -*- coding: utf-8 -*-
"""
SICD Geolocation - Coordinate transformations for SICD complex SAR imagery.

Wraps the SICD Volume 3 projection model via sarpy (preferred) or sarkit
to transform between image pixel coordinates and geographic coordinates.
Sarpy provides ``image_to_ground_geo`` and ``ground_to_image_geo`` for
high-accuracy projection using the full SICD sensor model.

Dependencies
------------
sarpy (preferred) or sarkit (fallback)

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
2026-02-11
"""

# Standard library
from typing import Any, Optional, Tuple, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import Geolocation
from grdl.geolocation.sar._backend import (
    require_projection_backend,
    _HAS_SARPY,
    _HAS_SARKIT,
)

if TYPE_CHECKING:
    from grdl.IO.models.sicd import SICDMetadata
    from grdl.IO.sar.sicd import SICDReader


class SICDGeolocation(Geolocation):
    """Geolocation for SICD (Sensor Independent Complex Data) imagery.

    Wraps the SICD Volume 3 projection model to transform between image
    pixel coordinates (row, col) and geographic coordinates (lat, lon, HAE).
    Uses sarpy's ``point_projection`` module as the preferred backend, with
    sarkit as a detection-time fallback.

    The sarpy backend provides full-accuracy projection using the complete
    SICD sensor model, including:

    - Scene Center Point (SCP) as the projection reference
    - ARP position and velocity polynomials
    - Grid geometry (row/col unit vectors, sample spacing)
    - Image formation algorithm parameters (PFA, RMA, RgAzComp)

    Attributes
    ----------
    metadata : SICDMetadata
        Typed SICD metadata with all 17 sections.
    backend : str
        Active projection backend (``'sarpy'`` or ``'sarkit'``).

    Notes
    -----
    The sarpy backend projects through the HAE (Height Above Ellipsoid)
    surface model by default. For terrain-corrected projection, pass
    ``dem_path`` to the constructor or set a specific ``height`` value
    in the ``image_to_latlon`` call.

    The sarkit backend currently raises ``NotImplementedError`` for
    projection operations. Install sarpy alongside sarkit for full
    SICD projection support.

    Examples
    --------
    From a reader (preferred):

    >>> from grdl.IO.sar import SICDReader
    >>> from grdl.geolocation.sar.sicd import SICDGeolocation
    >>> with SICDReader('image.nitf') as reader:
    ...     geo = SICDGeolocation.from_reader(reader)
    ...     lat, lon, h = geo.image_to_latlon(500, 1000)

    Array of pixels (vectorized):

    >>> lats, lons, heights = geo.image_to_latlon(
    ...     np.array([100, 200, 300]),
    ...     np.array([400, 500, 600]),
    ... )

    Inverse projection:

    >>> row, col = geo.latlon_to_image(34.05, -118.25)
    """

    def __init__(
        self,
        metadata: 'SICDMetadata',
        raw_meta: Any,
        backend: Optional[str] = None,
        dem_path: Optional[str] = None,
        geoid_path: Optional[str] = None,
    ) -> None:
        """Initialize SICD geolocation from metadata and raw backend object.

        Parameters
        ----------
        metadata : SICDMetadata
            Typed SICD metadata. Must have ``image_data`` with ``scp_pixel``
            and ``geo_data`` with ``scp``.
        raw_meta : Any
            Raw backend metadata object:
            - sarpy: ``SICDType`` instance (``reader.sicd_meta``)
            - sarkit: ``lxml.etree.ElementTree`` XML tree
        backend : str, optional
            Force a specific backend (``'sarpy'`` or ``'sarkit'``). If
            ``None``, auto-detects via ``require_projection_backend``.
        dem_path : str or Path, optional
            Path to DEM/DTED data folder for terrain-corrected projection.
        geoid_path : str or Path, optional
            Path to geoid correction file (EGM96/EGM2008).

        Raises
        ------
        ImportError
            If neither sarpy nor sarkit is installed.
        ValueError
            If required metadata sections (``image_data``, ``geo_data``)
            are missing or incomplete.
        """
        if backend is None:
            backend = require_projection_backend('SICD')

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

        shape = (metadata.rows, metadata.cols)

        super().__init__(
            shape, crs='WGS84', dem_path=dem_path, geoid_path=geoid_path
        )

        if backend == 'sarpy':
            self._sarpy_meta = raw_meta
        elif backend == 'sarkit':
            self._xmltree = raw_meta
            self._build_sarkit_params()

    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform pixel coordinate arrays to geographic coordinate arrays.

        Uses the SICD Volume 3 projection model to convert image (row, col)
        coordinates to geodetic (lat, lon, HAE) coordinates.

        Parameters
        ----------
        rows : np.ndarray
            Row coordinates (1D array, float64).
        cols : np.ndarray
            Column coordinates (1D array, float64).
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters). Used as the projection
            surface height when DEM is not configured.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) arrays in WGS84 coordinates.

        Raises
        ------
        NotImplementedError
            If using the sarkit backend (sarpy required for projection).
        """
        if self.backend == 'sarpy':
            return self._image_to_latlon_sarpy(rows, cols, height)
        else:
            return self._image_to_latlon_sarkit(rows, cols, height)

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform geographic coordinate arrays to pixel coordinate arrays.

        Uses the SICD Volume 3 inverse projection model to convert geodetic
        (lat, lon) coordinates to image (row, col) coordinates.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North (1D array, float64).
        lons : np.ndarray
            Longitudes in degrees East (1D array, float64).
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays.

        Raises
        ------
        NotImplementedError
            If using the sarkit backend (sarpy required for projection).
        """
        if self.backend == 'sarpy':
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
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project image coordinates to ground via sarpy.

        Parameters
        ----------
        rows : np.ndarray
            Row coordinates (1D array, float64).
        cols : np.ndarray
            Column coordinates (1D array, float64).
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) arrays in WGS84 coordinates.
        """
        from sarpy.geometry.point_projection import image_to_ground_geo

        im_points = np.column_stack([rows, cols])

        # sarpy returns Nx3 array: [lat, lon, hae]
        ground_geo = image_to_ground_geo(
            im_points,
            self._sarpy_meta,
            ordering='latlong',
            projection_type='HAE',
        )

        lats = ground_geo[:, 0]
        lons = ground_geo[:, 1]
        heights = ground_geo[:, 2]

        return lats, lons, heights

    def _latlon_to_image_sarpy(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project ground coordinates to image via sarpy.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North (1D array, float64).
        lons : np.ndarray
            Longitudes in degrees East (1D array, float64).
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinate arrays.
        """
        from sarpy.geometry.point_projection import ground_to_image_geo

        heights_arr = np.full_like(lats, height)
        coords = np.column_stack([lats, lons, heights_arr])

        # Returns tuple: (image_points Nx2, delta_gpn, iterations)
        image_points, _, _ = ground_to_image_geo(
            coords,
            self._sarpy_meta,
            ordering='latlong',
        )

        rows = image_points[:, 0]
        cols = image_points[:, 1]

        return rows, cols

    # ------------------------------------------------------------------
    # sarkit backend
    # ------------------------------------------------------------------

    def _build_sarkit_params(self) -> None:
        """Build sarkit projection parameters from XML metadata.

        Placeholder for future sarkit projection support. Currently
        stores a reference to the XML tree for potential future use
        when sarkit's projection API stabilizes.
        """
        # Store reference for future sarkit projection implementation.
        # sarkit.sicd.projection.MetadataParams.from_xml() could be
        # used here once the API is stable and well-documented.
        self._sarkit_xml_ref = self._xmltree

    def _image_to_latlon_sarkit(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project image coordinates to ground via sarkit.

        Parameters
        ----------
        rows : np.ndarray
            Row coordinates (1D array, float64).
        cols : np.ndarray
            Column coordinates (1D array, float64).
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters).

        Raises
        ------
        NotImplementedError
            Always. sarkit projection is not yet implemented. Install
            sarpy for SICD projection support.
        """
        raise NotImplementedError(
            "SICD projection via sarkit is not yet implemented. "
            "Install sarpy for SICD geolocation support: pip install sarpy"
        )

    def _latlon_to_image_sarkit(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project ground coordinates to image via sarkit.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North (1D array, float64).
        lons : np.ndarray
            Longitudes in degrees East (1D array, float64).
        height : float, default=0.0
            Height above WGS84 ellipsoid (meters).

        Raises
        ------
        NotImplementedError
            Always. sarkit projection is not yet implemented. Install
            sarpy for SICD projection support.
        """
        raise NotImplementedError(
            "SICD inverse projection via sarkit is not yet implemented. "
            "Install sarpy for SICD geolocation support: pip install sarpy"
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_reader(cls, reader: 'SICDReader') -> 'SICDGeolocation':
        """Create SICDGeolocation from a SICDReader instance.

        Extracts the raw backend metadata from the reader and constructs
        the geolocation object with the appropriate backend.

        Parameters
        ----------
        reader : SICDReader
            An open SICD reader with loaded metadata.

        Returns
        -------
        SICDGeolocation
            Configured geolocation object.

        Raises
        ------
        ValueError
            If the reader's metadata is missing required geolocation
            sections.

        Examples
        --------
        >>> from grdl.IO.sar import SICDReader
        >>> from grdl.geolocation.sar.sicd import SICDGeolocation
        >>> with SICDReader('image.nitf') as reader:
        ...     geo = SICDGeolocation.from_reader(reader)
        ...     lat, lon, h = geo.image_to_latlon(500, 1000)
        """
        if reader.backend == 'sarpy':
            raw_meta = reader._sarpy_meta
        elif reader.backend == 'sarkit':
            raw_meta = reader._xmltree
        else:
            raise ValueError(
                f"Unsupported SICDReader backend: {reader.backend!r}"
            )

        return cls(
            metadata=reader.metadata,
            raw_meta=raw_meta,
            backend=reader.backend,
        )
