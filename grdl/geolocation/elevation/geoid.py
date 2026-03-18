# -*- coding: utf-8 -*-
"""
Geoid Correction - Undulation lookup from geoid model grids.

Loads a geoid model file (EGM96 15-arc-minute grid in PGM format) and
provides vectorized bilinear interpolation of geoid undulation values.
Undulation is the height difference between the geoid (MSL) and the
WGS84 ellipsoid: ``height_HAE = height_MSL + undulation``.

EGM96 Grid Specification
------------------------
- Format: PGM (Portable Gray Map), 16-bit unsigned
- Grid size: 1440 columns x 721 rows
- Resolution: 15 arc-minutes (0.25 degrees)
- Latitude range: 90N to 90S (north to south)
- Longitude range: 0 to 360 (eastward from Greenwich)
- Values: Geoid undulation in centimeters, offset by 32768

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
2026-03-18  Fix _interpolate_array to use actual grid dimensions instead of
            hardcoded EGM96 constants — was silently wrong for GeoTIFF geoids.
2026-02-11
"""

# Standard library
import logging
from pathlib import Path
from typing import Optional, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import _is_scalar, _to_array

logger = logging.getLogger(__name__)

# EGM96 15-arc-minute grid constants
_EGM96_NROWS = 721
_EGM96_NCOLS = 1440
_EGM96_LAT_STEP = 0.25  # degrees (15 arc-minutes)
_EGM96_LON_STEP = 0.25  # degrees (15 arc-minutes)
_EGM96_LAT_MAX = 90.0   # north
_EGM96_LON_MIN = 0.0    # Greenwich
_EGM96_OFFSET = 32768   # PGM offset for signed values


class GeoidCorrection:
    """Geoid undulation lookup from an EGM96 grid file.

    Loads the EGM96 15-arc-minute geoid undulation grid in PGM format
    and provides vectorized bilinear interpolation. The grid covers the
    entire globe at 0.25-degree resolution.

    Parameters
    ----------
    geoid_path : str or Path
        Path to the EGM96 geoid grid file in PGM format (e.g.,
        ``WW15MGH.GRD`` converted to PGM, or the standard
        ``egm96-15.pgm``).

    Raises
    ------
    FileNotFoundError
        If ``geoid_path`` does not exist.
    ValueError
        If the file cannot be parsed as a valid EGM96 PGM grid.

    Notes
    -----
    The EGM96 PGM file stores undulation values as unsigned 16-bit
    integers with an offset of 32768. The actual undulation in
    centimeters is: ``undulation_cm = raw_value - 32768``. Values are
    converted to meters on load.

    Examples
    --------
    >>> from grdl.geolocation.elevation.geoid import GeoidCorrection
    >>> geoid = GeoidCorrection('/data/egm96-15.pgm')
    >>> geoid.get_undulation(38.6, -90.2)
    -32.15
    >>> import numpy as np
    >>> geoid.get_undulation(
    ...     np.array([38.6, 40.7]), np.array([-90.2, -74.0])
    ... )
    array([-32.15, -32.68])
    """

    def __init__(self, geoid_path: str) -> None:
        """Initialize geoid correction model.

        Parameters
        ----------
        geoid_path : str or Path
            Path to a geoid undulation grid file.  Supported formats:

            - **PGM** (``*.pgm``): EGM96 15-arc-minute grid (P5 binary
              or P2 ASCII).  Fixed 721 x 1440 global grid.
            - **GeoTIFF** (``*.tif``, ``*.tiff``): Any geoid model
              (EGM96, EGM2008, etc.) stored as a single-band GeoTIFF
              with geographic CRS.  Grid dimensions and extent are read
              from the file.

        Raises
        ------
        FileNotFoundError
            If ``geoid_path`` does not exist.
        ValueError
            If the file format is not recognized.
        ImportError
            If rasterio is required but not installed (GeoTIFF path).
        """
        geoid_path = Path(geoid_path)
        if not geoid_path.exists():
            raise FileNotFoundError(
                f"Geoid file does not exist: {geoid_path}"
            )

        self._path = geoid_path

        suffix = geoid_path.suffix.lower()
        if suffix in ('.tif', '.tiff', '.geotiff'):
            self._load_geotiff(geoid_path)
        elif suffix in ('.pgm',):
            self._grid = self._load_pgm(geoid_path)
            # Pre-compute latitude and longitude vectors for interpolation
            # Latitude: 90 (north) to -90 (south), 721 points
            self._lats = np.linspace(
                _EGM96_LAT_MAX, -_EGM96_LAT_MAX, _EGM96_NROWS
            )
            # Longitude: 0 to 359.75, 1440 points
            self._lons = np.linspace(
                _EGM96_LON_MIN,
                _EGM96_LON_MIN + (_EGM96_NCOLS - 1) * _EGM96_LON_STEP,
                _EGM96_NCOLS,
            )
        else:
            raise ValueError(
                f"Unsupported geoid file format: {suffix!r}. "
                f"Expected .pgm, .tif, or .tiff."
            )
        logger.info("Loaded geoid grid %s", geoid_path.name)

    def _load_geotiff(self, filepath: Path) -> None:
        """Load a geoid GeoTIFF and set up interpolation arrays.

        Reads the raster band and extracts the geographic extent from
        the transform to build latitude/longitude vectors.

        Parameters
        ----------
        filepath : Path
            Path to the GeoTIFF file.
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "rasterio is required for GeoTIFF geoid files. "
                "Install with: pip install rasterio"
            )

        with rasterio.open(str(filepath)) as ds:
            self._grid = ds.read(1).astype(np.float64)
            nrows, ncols = self._grid.shape
            transform = ds.transform

            # Build lat/lon vectors from the affine transform
            # transform: col → lon, row → lat
            # For geographic CRS: transform.c = west lon,
            # transform.f = north lat
            lon_min = transform.c
            lat_max = transform.f
            lon_step = transform.a
            lat_step = transform.e  # negative (north → south)

            self._lons = np.linspace(
                lon_min + lon_step * 0.5,
                lon_min + lon_step * (ncols - 0.5),
                ncols,
            )
            self._lats = np.linspace(
                lat_max + lat_step * 0.5,
                lat_max + lat_step * (nrows - 0.5),
                nrows,
            )

    @staticmethod
    def _load_pgm(filepath: Path) -> np.ndarray:
        """Load an EGM96 PGM file and return undulation grid in meters.

        Reads the PGM file format (P5 binary or P2 ASCII), skips comment
        lines, and converts raw unsigned 16-bit values to undulation in
        meters using the EGM96 offset convention.

        Parameters
        ----------
        filepath : Path
            Path to the PGM file.

        Returns
        -------
        np.ndarray
            Geoid undulation grid in meters. Shape ``(721, 1440)``,
            dtype float64. Latitude runs from 90N (row 0) to 90S
            (row 720). Longitude runs from 0 (col 0) to 359.75
            (col 1439).

        Raises
        ------
        ValueError
            If the file format is invalid or grid dimensions do not
            match the expected EGM96 size.
        """
        with open(filepath, 'rb') as f:
            # Read magic number
            magic = f.readline().strip()

            if magic == b'P5':
                # Binary PGM
                return GeoidCorrection._load_pgm_binary(f)
            elif magic == b'P2':
                # ASCII PGM
                return GeoidCorrection._load_pgm_ascii(f)
            else:
                raise ValueError(
                    f"Invalid PGM magic number: {magic!r}. "
                    f"Expected 'P5' (binary) or 'P2' (ASCII)."
                )

    @staticmethod
    def _load_pgm_binary(f) -> np.ndarray:
        """Load binary (P5) PGM file.

        Parameters
        ----------
        f : file object
            Open file positioned after the magic number line.

        Returns
        -------
        np.ndarray
            Undulation grid in meters. Shape ``(721, 1440)``.
        """
        # Skip comment lines
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()

        # Parse dimensions
        parts = line.split()
        if len(parts) == 2:
            ncols, nrows = int(parts[0]), int(parts[1])
        else:
            # Dimensions may be on separate lines
            ncols = int(parts[0])
            nrows = int(f.readline().strip())

        # Read max value
        maxval = int(f.readline().strip())

        if nrows != _EGM96_NROWS or ncols != _EGM96_NCOLS:
            raise ValueError(
                f"Expected {_EGM96_NCOLS}x{_EGM96_NROWS} grid, "
                f"got {ncols}x{nrows}."
            )

        # Read binary data (16-bit big-endian unsigned integers)
        raw = np.frombuffer(
            f.read(nrows * ncols * 2), dtype=np.dtype('>u2')
        )
        raw = raw.reshape((nrows, ncols))

        # Convert to undulation in meters
        grid = (raw.astype(np.float64) - _EGM96_OFFSET) / 100.0
        return grid

    @staticmethod
    def _load_pgm_ascii(f) -> np.ndarray:
        """Load ASCII (P2) PGM file.

        Parameters
        ----------
        f : file object
            Open file positioned after the magic number line.

        Returns
        -------
        np.ndarray
            Undulation grid in meters. Shape ``(721, 1440)``.
        """
        # Read remaining content as text
        content = f.read().decode('ascii')
        lines = content.split('\n')

        # Skip comment lines and extract tokens
        tokens = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') or not stripped:
                continue
            tokens.extend(stripped.split())

        # Parse header: width height maxval
        if len(tokens) < 3:
            raise ValueError("Insufficient header data in ASCII PGM file.")

        ncols = int(tokens[0])
        nrows = int(tokens[1])
        _maxval = int(tokens[2])

        if nrows != _EGM96_NROWS or ncols != _EGM96_NCOLS:
            raise ValueError(
                f"Expected {_EGM96_NCOLS}x{_EGM96_NROWS} grid, "
                f"got {ncols}x{nrows}."
            )

        # Parse pixel values
        expected = nrows * ncols
        pixel_tokens = tokens[3:]
        if len(pixel_tokens) < expected:
            raise ValueError(
                f"Expected {expected} pixel values, got {len(pixel_tokens)}."
            )

        raw = np.array(
            [int(t) for t in pixel_tokens[:expected]], dtype=np.float64
        )
        raw = raw.reshape((nrows, ncols))

        # Convert to undulation in meters
        grid = (raw - _EGM96_OFFSET) / 100.0
        return grid

    def get_undulation(
        self,
        lat_or_points: Union[float, list, np.ndarray],
        lon: Optional[Union[float, list, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Query geoid undulation for one or more geographic locations.

        Accepts three input forms:

        - **Scalar:** ``get_undulation(lat, lon)`` returns a single float.
        - **Stacked (2, N) array:** ``get_undulation(points_2xN)`` returns
          an ``(N,)`` ndarray.
        - **Separate arrays:** ``get_undulation(lats_arr, lons_arr)`` returns
          an ndarray.

        Parameters
        ----------
        lat_or_points : float, list, or np.ndarray
            Latitude(s) when ``lon`` is provided, or a ``(2, N)`` ndarray
            of stacked ``[lats; lons]`` when ``lon`` is None.
        lon : float, list, or np.ndarray, optional
            Longitude(s). Omit to pass a ``(2, N)`` stacked array.

        Returns
        -------
        float
            When scalar inputs are given. Undulation in meters.
        np.ndarray
            When array inputs are given. Shape ``(N,)``. Undulation in
            meters.

        Raises
        ------
        ValueError
            If a ``(2, N)`` array is expected but the shape is wrong.

        Examples
        --------
        >>> geoid.get_undulation(38.6, -90.2)
        -32.15
        """
        if lon is None:
            pts = np.asarray(lat_or_points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] != 2:
                raise ValueError(
                    f"Expected (2, N) array, got shape {pts.shape}"
                )
            return self._interpolate_array(pts[0], pts[1])
        elif _is_scalar(lat_or_points) and _is_scalar(lon):
            lats_arr = _to_array(lat_or_points)
            lons_arr = _to_array(lon)
            result = self._interpolate_array(lats_arr, lons_arr)
            return float(result[0])
        else:
            lats_arr = _to_array(lat_or_points)
            lons_arr = _to_array(lon)
            return self._interpolate_array(lats_arr, lons_arr)

    def _interpolate_array(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """Bilinear interpolation of geoid undulation values.

        Vectorized implementation using numpy array operations. Works
        with any grid resolution by using the actual ``self._lats`` and
        ``self._lons`` vectors built at load time.  Handles longitude
        wrapping and latitude clamping.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes in degrees North. Shape ``(N,)``.
        lons : np.ndarray
            Longitudes in degrees East. Shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Interpolated undulation values in meters. Shape ``(N,)``.
        """
        nrows, ncols = self._grid.shape

        # Grid parameters from the actual loaded vectors
        lat_max = float(self._lats[0])
        lat_min = float(self._lats[-1])
        lon_min = float(self._lons[0])
        lon_max = float(self._lons[-1])

        lat_step = (lat_max - lat_min) / (nrows - 1)  # positive
        lon_step = (lon_max - lon_min) / (ncols - 1)

        # Determine if grid is global (wraps in longitude)
        lon_span = lon_max - lon_min + lon_step
        is_global = lon_span > 359.0

        # Clamp latitude to grid range
        lats_clamped = np.clip(lats, lat_min, lat_max)

        # Normalize longitude into the grid range
        if is_global:
            lons_normalized = lons % 360.0
            if lon_min < 0:
                # Grid uses [-180, 180) convention
                lons_normalized = np.where(
                    lons_normalized > 180.0,
                    lons_normalized - 360.0,
                    lons_normalized,
                )
        else:
            lons_normalized = np.clip(lons, lon_min, lon_max)

        # Convert to fractional grid indices
        # Latitude runs north-to-south (row 0 = lat_max)
        row_frac = (lat_max - lats_clamped) / lat_step
        col_frac = (lons_normalized - lon_min) / lon_step

        # Floor indices for bilinear interpolation
        row0 = np.floor(row_frac).astype(np.intp)
        col0 = np.floor(col_frac).astype(np.intp)

        # Clamp row indices to valid grid range
        row0 = np.clip(row0, 0, nrows - 2)
        row1 = row0 + 1

        # Handle longitude wrapping for column indices
        if is_global:
            col0 = col0 % ncols
            col1 = (col0 + 1) % ncols
        else:
            col0 = np.clip(col0, 0, ncols - 2)
            col1 = col0 + 1

        # Fractional parts for interpolation weights
        dr = np.clip(row_frac - row0.astype(np.float64), 0.0, 1.0)
        dc = np.clip(col_frac - np.floor(col_frac), 0.0, 1.0)

        # Sample four corners
        q00 = self._grid[row0, col0]
        q01 = self._grid[row0, col1]
        q10 = self._grid[row1, col0]
        q11 = self._grid[row1, col1]

        # Bilinear interpolation
        undulation = (
            q00 * (1.0 - dr) * (1.0 - dc)
            + q01 * (1.0 - dr) * dc
            + q10 * dr * (1.0 - dc)
            + q11 * dr * dc
        )

        return undulation
