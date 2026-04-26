# -*- coding: utf-8 -*-
"""
Geoid Correction - Undulation lookup from geoid model grids.

Loads a geoid model file and provides vectorized bilinear interpolation
of geoid undulation values.  Undulation is the height difference
between the geoid (MSL) and the WGS84 ellipsoid:
``height_HAE = height_MSL + undulation``.

Supported PGM Grids
-------------------
Any post-aligned global EGM-family grid in PGM format (P5 binary or P2
ASCII) is accepted; dimensions are read from the PGM header and the
latitude/longitude vectors are derived on load.  Known formats include:

- EGM96 15 arc-minute (721 x 1440)
- EGM2008 5 arc-minute (2161 x 4320)
- EGM2008 2.5 arc-minute (4321 x 8640)
- EGM2008 1 arc-minute (10801 x 21600)

PGM Storage Convention
----------------------
- Format: PGM (Portable Gray Map), 16-bit unsigned big-endian (P5) or
  ASCII (P2)
- Latitude range: 90N (row 0) to 90S (row ``nrows - 1``)
- Longitude range: 0 (col 0) to ``360 - lon_step`` (col ``ncols - 1``)
- Values: undulation in centimeters, offset by 32768 (EGM default);
  scale and offset may be overridden by PGM comment lines of the form
  ``# scale <value>`` and ``# offset <value>``

GeoTIFF geoids (single-band, geographic CRS) are also supported with
grid extent read entirely from the affine transform.

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
2026-02-11

Modified
--------
2026-04-21  Accept any post-aligned global EGM PGM grid; infer lat/lon
            vectors from header dimensions instead of enforcing the
            EGM96 15-arc-minute shape.
2026-03-27  Read scale/offset from file metadata (GeoTIFF tags, PGM comments)
            instead of always using hardcoded EGM96 defaults.
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

# EGM-family PGM default encoding (override via '# scale'/'# offset'
# comment lines in the PGM header).
_PGM_DEFAULT_OFFSET = 32768  # unsigned → signed centimeter offset
_PGM_DEFAULT_SCALE = 1.0 / 100.0  # centimeters → meters


class GeoidCorrection:
    """Geoid undulation lookup from an EGM-family grid file.

    Loads a post-aligned global geoid undulation grid (PGM or GeoTIFF)
    and provides vectorized bilinear interpolation. Grid dimensions and
    resolution are read from the file header — any standard
    EGM96/EGM2008 PGM grid is accepted, as well as any single-band
    geographic GeoTIFF.

    Parameters
    ----------
    geoid_path : str or Path
        Path to the geoid grid file. PGM grids must be post-aligned
        global (row 0 at 90N, last row at 90S; col 0 at 0° longitude).

    Raises
    ------
    FileNotFoundError
        If ``geoid_path`` does not exist.
    ValueError
        If the file cannot be parsed as a recognized geoid grid.

    Notes
    -----
    EGM-family PGM files store undulation values as unsigned 16-bit
    integers with a default offset of 32768 (raw centimeters →
    ``undulation_cm = raw_value - 32768``). Scale and offset may be
    overridden by ``# scale`` / ``# offset`` comment lines in the PGM
    header. Values are converted to meters on load.

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
            nrows, ncols = self._grid.shape
            # Assume a post-aligned global grid (row 0 = 90N, row
            # nrows-1 = 90S; col 0 = 0°, col ncols-1 = 360° - lon_step).
            # This covers every standard EGM96/EGM2008 PGM.
            lat_step = 180.0 / (nrows - 1)
            lon_step = 360.0 / ncols
            self._lats = np.linspace(90.0, -90.0, nrows)
            self._lons = np.linspace(0.0, 360.0 - lon_step, ncols)
            logger.debug(
                "PGM geoid grid %d x %d, lat_step=%.6g°, lon_step=%.6g°",
                nrows, ncols, lat_step, lon_step,
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
            raw = ds.read(1).astype(np.float64)
            nrows, ncols = raw.shape
            transform = ds.transform

            # Apply scale and offset if present in the file metadata.
            # rasterio exposes these as per-band tuples; default is
            # scale=1.0, offset=0.0 when not set.
            scale = ds.scales[0] if ds.scales else 1.0
            offset = ds.offsets[0] if ds.offsets else 0.0
            if scale != 1.0 or offset != 0.0:
                self._grid = raw * scale + offset
                logger.debug(
                    "Applied GeoTIFF scale=%.6g, offset=%.6g", scale, offset
                )
            else:
                self._grid = raw

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
        """Load an EGM-family PGM file and return undulation in meters.

        Reads the PGM file format (P5 binary or P2 ASCII), skips comment
        lines, and converts raw unsigned 16-bit values to undulation in
        meters using the default EGM offset convention (or the scale /
        offset declared in PGM comment lines, when present).

        Parameters
        ----------
        filepath : Path
            Path to the PGM file.

        Returns
        -------
        np.ndarray
            Geoid undulation grid in meters. Shape ``(nrows, ncols)``
            as declared in the PGM header, dtype float64. Latitude runs
            from 90N (row 0) to 90S (row ``nrows - 1``). Longitude runs
            from 0 (col 0) eastward to ``360 - lon_step``
            (col ``ncols - 1``).

        Raises
        ------
        ValueError
            If the PGM magic number is not ``P5`` or ``P2`` or the
            pixel stream is truncated.
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
    def _parse_pgm_comments(comment_lines: list) -> tuple:
        """Extract scale and offset from PGM comment lines if present.

        Parameters
        ----------
        comment_lines : list of bytes
            Comment lines (starting with ``#``) from the PGM header.

        Returns
        -------
        tuple of (float or None, float or None)
            ``(scale, offset)`` parsed from comments, or ``None`` for
            each if not found.
        """
        scale = None
        offset = None
        for line in comment_lines:
            text = line.decode('ascii', errors='ignore').strip().lstrip('#').strip()
            lower = text.lower()
            if lower.startswith('scale'):
                try:
                    scale = float(text.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif lower.startswith('offset'):
                try:
                    offset = float(text.split()[-1])
                except (ValueError, IndexError):
                    pass
        return scale, offset

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
            Undulation grid in meters. Shape ``(nrows, ncols)`` as
            declared in the PGM header.
        """
        # Collect comment lines for scale/offset parsing
        comment_lines = []
        line = f.readline()
        while line.startswith(b'#'):
            comment_lines.append(line)
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
        _maxval = int(f.readline().strip())

        # Read binary data (16-bit big-endian unsigned integers)
        raw = np.frombuffer(
            f.read(nrows * ncols * 2), dtype=np.dtype('>u2')
        )
        raw = raw.reshape((nrows, ncols))

        # Use file-embedded scale/offset if available, else EGM defaults
        file_scale, file_offset = GeoidCorrection._parse_pgm_comments(
            comment_lines
        )
        scale = file_scale if file_scale is not None else _PGM_DEFAULT_SCALE
        offset = (
            file_offset if file_offset is not None else _PGM_DEFAULT_OFFSET
        )
        if file_scale is not None or file_offset is not None:
            logger.debug(
                "PGM file-embedded scale=%.6g, offset=%.6g", scale, offset
            )

        grid = (raw.astype(np.float64) - offset) * scale
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
            Undulation grid in meters. Shape ``(nrows, ncols)`` as
            declared in the PGM header.
        """
        # Read remaining content as text
        content = f.read().decode('ascii')
        lines = content.split('\n')

        # Collect comment lines for scale/offset, extract data tokens
        comment_lines = []
        tokens = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                comment_lines.append(stripped.encode('ascii'))
            elif stripped:
                tokens.extend(stripped.split())

        # Parse header: width height maxval
        if len(tokens) < 3:
            raise ValueError("Insufficient header data in ASCII PGM file.")

        ncols = int(tokens[0])
        nrows = int(tokens[1])
        _maxval = int(tokens[2])

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

        # Use file-embedded scale/offset if available, else EGM defaults
        file_scale, file_offset = GeoidCorrection._parse_pgm_comments(
            comment_lines
        )
        scale = file_scale if file_scale is not None else _PGM_DEFAULT_SCALE
        offset = (
            file_offset if file_offset is not None else _PGM_DEFAULT_OFFSET
        )
        if file_scale is not None or file_offset is not None:
            logger.debug(
                "PGM file-embedded scale=%.6g, offset=%.6g", scale, offset
            )

        grid = (raw - offset) * scale
        return grid

    def get_undulation(
        self,
        lat_or_points: Union[float, list, np.ndarray],
        lon: Optional[Union[float, list, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """Query geoid undulation for one or more geographic locations.

        Accepts three input forms:

        - **Scalar:** ``get_undulation(lat, lon)`` returns a single float.
        - **Stacked (N, 2) array:** ``get_undulation(points_Nx2)`` returns
          an ``(N,)`` ndarray.
        - **Separate arrays:** ``get_undulation(lats_arr, lons_arr)`` returns
          an ndarray.

        Parameters
        ----------
        lat_or_points : float, list, or np.ndarray
            Latitude(s) when ``lon`` is provided, or an ``(N, 2)`` ndarray
            of stacked ``[lat, lon]`` rows when ``lon`` is None.
        lon : float, list, or np.ndarray, optional
            Longitude(s). Omit to pass an ``(N, 2)`` stacked array.

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
            If an ``(N, 2)`` array is expected but the shape is wrong.

        Examples
        --------
        >>> geoid.get_undulation(38.6, -90.2)
        -32.15
        """
        if lon is None:
            pts = np.asarray(lat_or_points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValueError(
                    f"Expected (N, 2) array, got shape {pts.shape}"
                )
            return self._interpolate_array(pts[:, 0], pts[:, 1])
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
