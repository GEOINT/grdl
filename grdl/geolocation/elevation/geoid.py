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
2026-02-11
"""

# Standard library
from pathlib import Path
from typing import Optional, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import _is_scalar, _to_array

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
            Path to the EGM96 geoid grid file in PGM format.

        Raises
        ------
        FileNotFoundError
            If ``geoid_path`` does not exist.
        ValueError
            If the file is not a valid EGM96 PGM grid.
        """
        geoid_path = Path(geoid_path)
        if not geoid_path.exists():
            raise FileNotFoundError(
                f"Geoid file does not exist: {geoid_path}"
            )

        self._path = geoid_path
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

        Vectorized implementation using numpy array operations. Handles
        longitude wrapping (input range [-180, 180] mapped to grid range
        [0, 360]) and latitude clamping to [-90, 90].

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
        # Clamp latitude to valid range
        lats_clamped = np.clip(lats, -90.0, 90.0)

        # Normalize longitude to [0, 360) range for grid lookup
        lons_normalized = lons % 360.0

        # Convert to fractional grid indices
        # Latitude: row 0 = 90N, row 720 = 90S
        row_frac = (_EGM96_LAT_MAX - lats_clamped) / _EGM96_LAT_STEP
        # Longitude: col 0 = 0E, col 1439 = 359.75E
        col_frac = lons_normalized / _EGM96_LON_STEP

        # Floor and ceil indices for bilinear interpolation
        row0 = np.floor(row_frac).astype(np.intp)
        col0 = np.floor(col_frac).astype(np.intp)

        # Clamp row indices to valid grid range
        row0 = np.clip(row0, 0, _EGM96_NROWS - 2)
        row1 = row0 + 1

        # Handle longitude wrapping for column indices
        col0 = col0 % _EGM96_NCOLS
        col1 = (col0 + 1) % _EGM96_NCOLS

        # Fractional parts for interpolation weights
        dr = row_frac - row0.astype(np.float64)
        dc = col_frac - np.floor(col_frac)

        # Clamp fractional parts
        dr = np.clip(dr, 0.0, 1.0)
        dc = np.clip(dc, 0.0, 1.0)

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
