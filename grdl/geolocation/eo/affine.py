# -*- coding: utf-8 -*-
"""
Affine Geolocation - Coordinate transforms for geocoded rasters.

Provides ``AffineGeolocation``, a concrete ``Geolocation`` subclass for any
raster whose pixel-to-map relationship is described by a six-parameter affine
transform and a coordinate reference system. This covers the vast majority of
geocoded imagery: GeoTIFFs, COGs, orthorectified EO, geocoded SAR GRD, and
multispectral products.

Coordinate flow:

    pixel (row, col)  --affine-->  native CRS (x, y)  --pyproj-->  WGS84 (lat, lon)

When the native CRS is already geographic (e.g. EPSG:4326), the pyproj step
is skipped entirely for maximum performance.

Dependencies
------------
rasterio
pyproj

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
from typing import Optional, Tuple, Union, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import Geolocation
from grdl.geolocation.eo._backend import require_affine_backend

if TYPE_CHECKING:
    from rasterio.transform import Affine
    from grdl.IO.base import ImageReader


class AffineGeolocation(Geolocation):
    """Geolocation for any geocoded raster with an affine transform and CRS.

    Uses the six-parameter affine transform (as defined by rasterio /
    GDAL) to convert between pixel coordinates and native CRS coordinates,
    then uses pyproj to reproject to WGS84 when the native CRS is not
    already geographic.

    The affine transform maps pixel ``(col, row)`` to map ``(x, y)`` as::

        x = c + col * a + row * b
        y = f + col * d + row * e

    where ``(a, b, c, d, e, f)`` are the six affine parameters stored by
    rasterio as ``Affine(a, b, c, d, e, f)``.

    Parameters
    ----------
    transform : rasterio.transform.Affine
        Six-parameter affine transform mapping pixel to native CRS
        coordinates.
    shape : Tuple[int, int]
        Image shape ``(rows, cols)``.
    crs : str
        Coordinate reference system string (e.g. ``'EPSG:4326'``,
        ``'EPSG:32756'``).
    dem_path : str or Path, optional
        Path to DEM/DTED data folder for terrain-corrected heights.
    geoid_path : str or Path, optional
        Path to geoid correction file (EGM96/EGM2008).

    Attributes
    ----------
    native_crs : str
        The original CRS string as provided to the constructor.
    shape : Tuple[int, int]
        Image shape ``(rows, cols)``.
    crs : str
        Always ``'WGS84'`` (output CRS after reprojection).

    Raises
    ------
    ImportError
        If rasterio or pyproj is not installed.
    TypeError
        If *transform* is not a ``rasterio.transform.Affine`` instance.

    Examples
    --------
    From a rasterio Affine and CRS string:

    >>> from rasterio.transform import Affine
    >>> transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)
    >>> geo = AffineGeolocation(transform, (1024, 2048), 'EPSG:32756')
    >>> lat, lon, h = geo.image_to_latlon(512, 1024)

    From a GRDL reader:

    >>> from grdl.IO.geotiff import GeoTIFFReader
    >>> with GeoTIFFReader('image.tif') as reader:
    ...     geo = AffineGeolocation.from_reader(reader)
    ...     lat, lon, h = geo.image_to_latlon(0, 0)
    """

    def __init__(
        self,
        transform: 'Affine',
        shape: Tuple[int, int],
        crs: str,
        dem_path: Optional[Union[str, object]] = None,
        geoid_path: Optional[Union[str, object]] = None,
    ) -> None:
        # Fail fast if dependencies are missing
        require_affine_backend()

        # Import dependencies (guaranteed available after require_affine_backend)
        from rasterio.transform import Affine
        import pyproj

        # Validate transform type
        if not isinstance(transform, Affine):
            raise TypeError(
                f"transform must be a rasterio.transform.Affine instance, "
                f"got {type(transform).__name__}"
            )

        # Store the rasterio Affine object for inverse computation
        self._transform = transform

        # Extract the six affine parameters for vectorized forward math:
        #   x = c + col * a + row * b
        #   y = f + col * d + row * e
        self._a = float(transform.a)
        self._b = float(transform.b)
        self._c = float(transform.c)
        self._d = float(transform.d)
        self._e = float(transform.e)
        self._f = float(transform.f)

        # Determine if the native CRS is already geographic (lat/lon)
        native_crs_obj = pyproj.CRS(crs)
        self._is_geographic = native_crs_obj.is_geographic

        # Build pyproj Transformers once (only needed for projected CRS)
        self._to_wgs84 = None
        self._from_wgs84 = None
        if not self._is_geographic:
            wgs84 = pyproj.CRS('EPSG:4326')
            self._to_wgs84 = pyproj.Transformer.from_crs(
                native_crs_obj, wgs84, always_xy=True
            )
            self._from_wgs84 = pyproj.Transformer.from_crs(
                wgs84, native_crs_obj, always_xy=True
            )

        # Initialize base class (output CRS is always WGS84)
        super().__init__(
            shape, crs='WGS84', dem_path=dem_path, geoid_path=geoid_path
        )

        # Store the native CRS for reference
        self.native_crs = crs

    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform pixel coordinate arrays to WGS84 geographic coordinates.

        Applies the affine transform to get native CRS coordinates, then
        reprojects to WGS84 if needed. Fully vectorized -- no Python loops.

        Parameters
        ----------
        rows : np.ndarray
            Row coordinates (1D array, float64).
        cols : np.ndarray
            Column coordinates (1D array, float64).
        height : float, default=0.0
            Height above WGS84 ellipsoid in meters. Applied uniformly to
            all output points.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(lats, lons, heights)`` arrays in WGS84 coordinates.
        """
        # Vectorized affine: pixel (row, col) -> native CRS (x, y)
        xs = self._c + cols * self._a + rows * self._b
        ys = self._f + cols * self._d + rows * self._e

        if self._is_geographic:
            # Native CRS is geographic: x = lon, y = lat
            lons = xs
            lats = ys
        else:
            # Projected CRS: transform native (x, y) -> WGS84 (lon, lat)
            # pyproj with always_xy=True: input (x, y), output (lon, lat)
            lons, lats = self._to_wgs84.transform(xs, ys)

        heights = np.full_like(lats, height)
        return lats, lons, heights

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform WGS84 geographic coordinate arrays to pixel coordinates.

        Reprojects from WGS84 to native CRS if needed, then applies the
        inverse affine transform. Fully vectorized -- no Python loops.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North (1D array, float64).
        lons : np.ndarray
            Longitudes in degrees East (1D array, float64).
        height : float, default=0.0
            Height above WGS84 ellipsoid in meters (unused for 2D affine
            inverse, included for ABC compatibility).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(rows, cols)`` pixel coordinate arrays.
        """
        if self._is_geographic:
            # Geographic CRS: lon = x, lat = y
            xs = lons
            ys = lats
        else:
            # WGS84 (lon, lat) -> native CRS (x, y)
            # pyproj with always_xy=True: input (lon, lat), output (x, y)
            xs, ys = self._from_wgs84.transform(lons, lats)

        # Inverse affine: native CRS (x, y) -> pixel (row, col)
        inv = ~self._transform
        ia = float(inv.a)
        ib = float(inv.b)
        ic = float(inv.c)
        id_ = float(inv.d)
        ie = float(inv.e)
        if_ = float(inv.f)

        # Vectorized inverse affine math:
        #   col = ic + x * ia + y * ib
        #   row = if_ + x * id_ + y * ie
        cols = ic + xs * ia + ys * ib
        rows = if_ + xs * id_ + ys * ie

        return rows, cols

    @classmethod
    def from_reader(cls, reader: 'ImageReader') -> 'AffineGeolocation':
        """Create an AffineGeolocation from a GRDL imagery reader.

        Extracts the affine transform, CRS, and image shape from the
        reader's metadata. Works with any reader that stores a rasterio
        ``Affine`` transform in ``metadata['transform']`` and a CRS string
        in ``metadata['crs']`` (e.g. ``GeoTIFFReader``).

        Parameters
        ----------
        reader : ImageReader
            A GRDL imagery reader with populated metadata.

        Returns
        -------
        AffineGeolocation
            Configured geolocation object.

        Raises
        ------
        ValueError
            If the reader's metadata does not contain a valid ``transform``
            or ``crs``.

        Examples
        --------
        >>> from grdl.IO.geotiff import GeoTIFFReader
        >>> with GeoTIFFReader('image.tif') as reader:
        ...     geo = AffineGeolocation.from_reader(reader)
        ...     lat, lon, h = geo.image_to_latlon(0, 0)
        """
        transform = reader.metadata.get('transform')
        if transform is None:
            raise ValueError(
                "Reader metadata does not contain an affine transform. "
                "AffineGeolocation requires metadata['transform'] to be a "
                "rasterio.transform.Affine instance."
            )

        crs = reader.metadata.get('crs')
        if crs is None:
            raise ValueError(
                "Reader metadata does not contain a CRS. "
                "AffineGeolocation requires metadata['crs'] to be a valid "
                "coordinate reference system string (e.g. 'EPSG:4326')."
            )

        shape = (reader.metadata['rows'], reader.metadata['cols'])

        return cls(transform=transform, shape=shape, crs=crs)
