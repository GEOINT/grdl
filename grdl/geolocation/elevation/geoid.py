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

PGM Storage Convention (GeographicLib)
--------------------------------------
GeographicLib distributes the EGM geoid grids as PGM files.  GRDL reads
that exact format:

- Format: PGM (Portable Gray Map), P5 (binary, big-endian) or P2
  (ASCII).  GeographicLib grids are P5 with a 16-bit ``maxval`` of 65535.
- Latitude range: 90N (row 0) to 90S (row ``nrows - 1``)
- Longitude range: 0 (col 0) to ``360 - 360/ncols`` (col ``ncols - 1``)
- Values: raw pixels are decoded to undulation in meters with the
  GeographicLib affine convention ``undulation = offset + scale * pixel``.
  ``offset`` and ``scale`` are read from the required ``# Offset`` and
  ``# Scale`` header comment lines (e.g. ``# Offset -108`` /
  ``# Scale 0.003``); a PGM lacking them is rejected.

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
2026-07-22  Read the GeographicLib geoid PGM format: decode undulation as
            ``offset + scale * pixel`` from the required ``# Offset`` /
            ``# Scale`` header lines, with a spec-correct NetPBM header
            tokenizer (comments/whitespace anywhere, 1- or 2-byte
            samples). Replaces the wrong ``(pixel - 32768) * 0.01``
            convention.
2026-05-26  Add ``to_shared`` / ``release_shared`` plus pickle hooks so
            the in-memory geoid grid can be hoisted into
            ``multiprocessing.shared_memory`` once and attached by
            every worker, instead of shipping a 285 MB float64 array
            through the IPC pipe per worker.
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
    GeographicLib geoid PGM files store undulation as unsigned integers
    decoded with the affine convention
    ``undulation_m = offset + scale * pixel``. ``offset`` and ``scale``
    are read from the required ``# Offset`` / ``# Scale`` header comment
    lines and applied on load; a PGM without them is rejected.

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

    # Shared-memory backing for the geoid grid. Populated by
    # :meth:`to_shared` so multiple worker processes can attach to a
    # single ~285 MB float64 raster instead of each unpickling its
    # own copy. Class-level defaults keep legacy instances picklable
    # without migration.
    _shm = None
    _shm_meta: Optional[dict] = None

    def __init__(self, geoid_path: str) -> None:
        """Initialize geoid correction model.

        Parameters
        ----------
        geoid_path : str or Path
            Path to a geoid undulation grid file.  Supported formats:

            - **PGM** (``*.pgm``): GeographicLib geoid grid (P5 binary
              or P2 ASCII) of any standard EGM96/EGM2008 resolution.
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

    # ────────────────────────────────────────────────────────
    # Shared-memory transport (cross-process zero-copy)
    # ────────────────────────────────────────────────────────

    def to_shared(self):
        """Move ``_grid`` into a shared-memory block. Idempotent.

        After this call, reads of ``_grid`` in this process use the
        shared block directly (no duplicate copy retained). Pickling
        this instance drops the grid bytes entirely and ships only
        ``(name, shape, dtype)`` metadata; the unpickling process
        attaches to the same block via
        :class:`multiprocessing.shared_memory.SharedMemory`.

        The caller owns the lifecycle. The block stays alive as long
        as this object holds it; call :meth:`release_shared` after
        every worker that attached has exited.

        Returns
        -------
        multiprocessing.shared_memory.SharedMemory
            The shared block. Already-shared instances return the
            existing handle.
        """
        if self._shm is not None:
            return self._shm
        from multiprocessing import shared_memory
        grid = np.ascontiguousarray(self._grid)
        shm = shared_memory.SharedMemory(create=True, size=grid.nbytes)
        view = np.ndarray(grid.shape, dtype=grid.dtype, buffer=shm.buf)
        view[...] = grid
        self._shm = shm
        self._shm_meta = {
            'name': shm.name,
            'shape': tuple(grid.shape),
            'dtype': str(grid.dtype),
        }
        # Rebind reads in this process to the shared view so the
        # parent doesn't keep a duplicate 285 MB array alive.
        self._grid = view
        return shm

    def release_shared(self):
        """Detach the grid into an owned copy, then close + unlink.

        Safe no-op when not shared. Idempotent. Call this only after
        every worker that attached to the shared block has exited —
        unlinking while a worker is mid-read may segfault that
        worker.
        """
        if self._shm is None:
            return
        # Detach so reads after release continue to work without a
        # dangling buffer reference.
        self._grid = np.array(self._grid)
        try:
            self._shm.close()
        finally:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
        self._shm = None
        self._shm_meta = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._shm_meta is not None:
            # Workers attach to the shared block instead of
            # unpickling 285 MB of float64 — drop the grid array
            # and the (unpicklable) SHM handle from the payload.
            state['_grid'] = None
            state['_shm'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._shm_meta is not None and self._grid is None:
            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(
                name=self._shm_meta['name'],
            )
            # Hold the handle for the worker's lifetime; the OS frees
            # the mapping on process exit.
            self._shm = shm
            self._grid = np.ndarray(
                self._shm_meta['shape'],
                dtype=np.dtype(self._shm_meta['dtype']),
                buffer=shm.buf,
            )

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
        """Load a GeographicLib geoid PGM and return undulation in meters.

        Parses the PGM header (``P5`` binary or ``P2`` ASCII) honoring the
        NetPBM whitespace/comment rules, then decodes raw pixel values to
        geoid undulation in meters using the GeographicLib convention::

            undulation_m = offset + scale * pixel

        ``offset`` and ``scale`` are read from the required ``# Offset``
        and ``# Scale`` header comment lines that GeographicLib writes
        into every geoid grid.  Without them the raw pixels are
        meaningless, so their absence is a hard error.

        Parameters
        ----------
        filepath : Path
            Path to the GeographicLib geoid PGM file.

        Returns
        -------
        np.ndarray
            Geoid undulation grid in meters. Shape ``(nrows, ncols)`` as
            declared in the PGM header, dtype float64. Row 0 is 90N, row
            ``nrows - 1`` is 90S; col 0 is 0 longitude, col ``ncols - 1``
            is ``360 - 360/ncols``.

        Raises
        ------
        ValueError
            If the magic number is not ``P5`` / ``P2``, the header is
            malformed, the pixel stream is truncated, or the required
            ``# Offset`` / ``# Scale`` comment lines are missing.
        """
        with open(filepath, 'rb') as f:
            magic, ncols, nrows, maxval, offset, scale = (
                GeoidCorrection._parse_pgm_header(f)
            )

            if offset is None or scale is None:
                raise ValueError(
                    f"{filepath.name} is missing the required '# Offset' / "
                    f"'# Scale' header lines. GeoidCorrection reads "
                    f"GeographicLib geoid PGM grids, which encode "
                    f"undulation as 'offset + scale * pixel'; a PGM without "
                    f"these values cannot be decoded to meters."
                )

            expected = nrows * ncols
            if magic == b'P5':
                # 1 byte/sample when maxval < 256, else 2-byte big-endian.
                dtype = np.dtype('>u1') if maxval < 256 else np.dtype('>u2')
                nbytes = expected * dtype.itemsize
                buf = f.read(nbytes)
                if len(buf) < nbytes:
                    raise ValueError(
                        f"Truncated PGM raster in {filepath.name}: expected "
                        f"{nbytes} bytes, got {len(buf)}."
                    )
                raw = np.frombuffer(buf, dtype=dtype).astype(np.float64)
            else:  # b'P2' — ASCII samples follow the header.
                tokens = f.read().decode('ascii').split()
                if len(tokens) < expected:
                    raise ValueError(
                        f"Truncated PGM raster in {filepath.name}: expected "
                        f"{expected} samples, got {len(tokens)}."
                    )
                raw = np.array(tokens[:expected], dtype=np.float64)

        raw = raw.reshape((nrows, ncols))
        # GeographicLib affine decode: undulation = offset + scale * pixel.
        grid = offset + scale * raw
        logger.debug(
            "GeographicLib PGM %d x %d, offset=%.6g, scale=%.6g",
            nrows, ncols, offset, scale,
        )
        return grid

    @staticmethod
    def _parse_pgm_header(f) -> tuple:
        """Parse a NetPBM PGM header from a binary stream.

        Reads the magic number, width, height, and maxval as
        whitespace-delimited tokens, skipping ``#`` comments (to end of
        line) wherever they appear, per the NetPBM specification. The
        GeographicLib ``# Offset`` and ``# Scale`` values are captured
        from the comment lines. On return the stream is positioned at the
        first raster byte: the single whitespace character terminating
        ``maxval`` has been consumed (exactly as the PGM format requires),
        so the P5 binary raster or the P2 ASCII samples follow
        immediately.

        Parameters
        ----------
        f : BinaryIO
            File opened in binary mode, positioned at the start.

        Returns
        -------
        tuple
            ``(magic, ncols, nrows, maxval, offset, scale)``. ``magic`` is
            ``b'P5'`` or ``b'P2'``; ``offset`` / ``scale`` are floats, or
            ``None`` when the corresponding comment line is absent.

        Raises
        ------
        ValueError
            If the magic number is not ``P5`` / ``P2`` or the header ends
            before width, height, and maxval are read.
        """
        comments = []

        def next_token() -> bytes:
            token = bytearray()
            while True:
                ch = f.read(1)
                if ch == b'':
                    break  # EOF
                if ch == b'#':
                    # Comment runs to end of line; capture the payload.
                    line = bytearray()
                    while True:
                        c = f.read(1)
                        if c == b'' or c == b'\n':
                            break
                        line += c
                    comments.append(bytes(line))
                    if token:
                        break
                    continue
                if ch.isspace():
                    if token:
                        break  # single whitespace terminator consumed
                    continue
                token += ch
            return bytes(token)

        magic = next_token()
        if magic not in (b'P5', b'P2'):
            raise ValueError(
                f"Invalid PGM magic number: {magic!r}. "
                f"Expected 'P5' (binary) or 'P2' (ASCII)."
            )
        try:
            ncols = int(next_token())
            nrows = int(next_token())
            maxval = int(next_token())
        except ValueError:
            raise ValueError(
                "Malformed PGM header: could not read width, height, and "
                "maxval."
            )

        offset, scale = GeoidCorrection._parse_pgm_comments(comments)
        return magic, ncols, nrows, maxval, offset, scale

    @staticmethod
    def _parse_pgm_comments(comment_lines) -> tuple:
        """Extract GeographicLib ``Offset`` and ``Scale`` from comments.

        GeographicLib geoid PGMs declare the affine decode parameters as
        header comment lines of the form ``# Offset -108`` and
        ``# Scale 0.003``. The leading ``#`` has already been stripped by
        :meth:`_parse_pgm_header`.

        Parameters
        ----------
        comment_lines : list of bytes
            Comment payloads (without the leading ``#``) from the header.

        Returns
        -------
        tuple of (float or None, float or None)
            ``(offset, scale)`` parsed from the comments, or ``None`` for
            each key not found.
        """
        offset = None
        scale = None
        for line in comment_lines:
            words = line.decode('ascii', errors='ignore').split()
            if len(words) < 2:
                continue
            key = words[0].lower()
            if key == 'offset':
                try:
                    offset = float(words[1])
                except ValueError:
                    pass
            elif key == 'scale':
                try:
                    scale = float(words[1])
                except ValueError:
                    pass
        return offset, scale

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
