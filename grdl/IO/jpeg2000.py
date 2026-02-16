# -*- coding: utf-8 -*-
"""
JPEG2000 Reader - Read JPEG2000 (JP2/J2K) imagery.

Base data format reader for JPEG2000 files (.jp2, .j2k) regardless of
producer. Unlocks Sentinel-2 native format (SAFE/JP2), Pleiades, SPOT,
and EnMAP imagery. Uses rasterio (GDAL JP2 driver) as the primary
backend, with glymur as a fallback for better handling of non-standard
encodings like Sentinel-2's 15-bit data.

Dependencies
------------
rasterio (primary) or glymur (fallback)

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
2026-02-10

Modified
--------
2026-02-10
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

# Try rasterio first (most common, handles geolocation)
try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

# Try glymur as fallback (better JP2 support, especially for Sentinel-2)
# Capture stderr during import: glymur can trigger a GDAL/NumPy ABI
# mismatch traceback when the system GDAL was compiled against NumPy 1.x
# but NumPy 2.x is installed.  The exception is caught below, but NumPy
# prints a noisy traceback to sys.stderr before raising.  We capture to a
# buffer and re-emit anything that isn't the known NumPy ABI noise.
import sys as _sys
_stderr_buf = type('_Buf', (), {
    '__init__': lambda self: setattr(self, '_parts', []),
    'write': lambda self, s: self._parts.append(s),
    'flush': lambda self: None,
    'getvalue': lambda self: ''.join(self._parts),
})()
_saved_stderr = _sys.stderr
_sys.stderr = _stderr_buf
try:
    import glymur
    _HAS_GLYMUR = True
except (ImportError, AttributeError):
    _HAS_GLYMUR = False
finally:
    _sys.stderr = _saved_stderr
    _captured = _stderr_buf.getvalue()
    if _captured and "_ARRAY_API not found" not in _captured:
        _sys.stderr.write(_captured)

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models import ImageMetadata


class JP2Reader(ImageReader):
    """Read JPEG2000 (JP2/J2K) imagery.

    Reads JPEG2000 files including Sentinel-2 native format, Pleiades,
    SPOT, and EnMAP imagery. Uses rasterio as the primary backend for
    compatibility with other geospatial tools. Falls back to glymur
    when rasterio lacks JP2 support or for better handling of
    non-standard encodings (e.g., Sentinel-2's 15-bit data).

    The reader targets single-band JP2 files. For multi-granule products
    like Sentinel-2 SAFE directories, open each JP2 band file separately
    or use a higher-level SAFE reader.

    Parameters
    ----------
    filepath : str or Path
        Path to the JPEG2000 file (.jp2, .j2k, .j2c).
    backend : str, optional
        Backend to use: 'rasterio', 'glymur', or 'auto' (default).
        'auto' tries rasterio first, then glymur if unavailable.

    Attributes
    ----------
    filepath : Path
        Path to the JPEG2000 file.
    metadata : Dict[str, Any]
        Standardized metadata dictionary.
    backend : str
        Active backend ('rasterio' or 'glymur').

    Raises
    ------
    ImportError
        If no supported backend is installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be opened or is not a valid JP2.

    Examples
    --------
    >>> from grdl.IO.jpeg2000 import JP2Reader
    >>> with JP2Reader('S2A_MSIL1C_B04.jp2') as reader:
    ...     chip = reader.read_chip(0, 512, 0, 512)
    ...     print(reader.metadata['format'])
    'JPEG2000'

    >>> # Use glymur explicitly for better 15-bit support
    >>> with JP2Reader('sentinel2_band.jp2', backend='glymur') as reader:
    ...     print(reader.get_shape())
    ...     full = reader.read_full()

    >>> # Check CRS (available with rasterio backend)
    >>> with JP2Reader('scene.jp2') as reader:
    ...     print(reader.metadata.get('crs'))
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        backend: str = 'auto',
    ) -> None:
        self._backend = backend
        self._dataset = None
        self._glymur_jp2 = None

        # Validate backend availability
        if backend == 'rasterio' or backend == 'auto':
            if not _HAS_RASTERIO and backend == 'rasterio':
                raise ImportError(
                    "rasterio is required for JP2 reading with backend='rasterio'. "
                    "Install with: pip install rasterio"
                )
        if backend == 'glymur' or backend == 'auto':
            if not _HAS_GLYMUR and backend == 'glymur':
                raise ImportError(
                    "glymur is required for JP2 reading with backend='glymur'. "
                    "Install with: pip install glymur"
                )
        if backend == 'auto' and not (_HAS_RASTERIO or _HAS_GLYMUR):
            raise ImportError(
                "Either rasterio or glymur is required for JP2 reading. "
                "Install with: pip install rasterio  OR  pip install glymur"
            )

        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Open the JP2 file and load metadata."""
        # Try rasterio first if auto or explicitly requested
        if self._backend in ('auto', 'rasterio') and _HAS_RASTERIO:
            try:
                self._dataset = rasterio.open(str(self.filepath))
                self.backend = 'rasterio'
                self._load_metadata_rasterio()
                return
            except Exception as e:
                if self._backend == 'rasterio':
                    raise ValueError(
                        f"Failed to open JP2 file with rasterio: {self.filepath}: {e}"
                    ) from e
                # If auto, continue to try glymur

        # Try glymur as fallback or if explicitly requested
        if self._backend in ('auto', 'glymur') and _HAS_GLYMUR:
            try:
                self._glymur_jp2 = glymur.Jp2k(str(self.filepath))
                self.backend = 'glymur'
                self._load_metadata_glymur()
                return
            except Exception as e:
                raise ValueError(
                    f"Failed to open JP2 file with glymur: {self.filepath}: {e}"
                ) from e

        raise ValueError(
            f"Could not open {self.filepath}. No suitable JP2 backend available."
        )

    def _load_metadata_rasterio(self) -> None:
        """Load metadata using rasterio backend."""
        ds = self._dataset

        # Rasterio stores bands as (band, row, col) with 1-based indexing
        bands = ds.count
        rows = ds.height
        cols = ds.width

        # Collect metadata
        extras: Dict[str, Any] = {}

        # Try to get compression info
        if ds.compression:
            extras['compression'] = ds.compression

        # JPEG2000 box metadata (if available)
        if hasattr(ds, 'tags') and callable(ds.tags):
            tags = ds.tags()
            if tags:
                extras.update(tags)

        extras['backend'] = 'rasterio'

        self.metadata = ImageMetadata(
            format='JPEG2000',
            rows=rows,
            cols=cols,
            dtype=str(ds.dtypes[0]),
            bands=bands,
            extras=extras,
        )

    def _load_metadata_glymur(self) -> None:
        """Load metadata using glymur backend."""
        jp2 = self._glymur_jp2

        shape = jp2.shape
        if len(shape) == 2:
            rows, cols = shape
            bands = 1
        elif len(shape) == 3:
            rows, cols, bands = shape
        else:
            raise ValueError(
                f"Unexpected JP2 shape: {shape}. Expected 2D or 3D array."
            )

        # Collect metadata from JP2 boxes
        extras: Dict[str, Any] = {}

        # Check for JP2 UUID boxes, XML boxes, etc.
        if hasattr(jp2, 'box') and jp2.box:
            # Number of resolution levels
            if hasattr(jp2, 'codestream'):
                try:
                    extras['num_resolutions'] = jp2.codestream.segment[2].num_res + 1
                except (IndexError, AttributeError):
                    pass

        extras['backend'] = 'glymur'

        self.metadata = ImageMetadata(
            format='JPEG2000',
            rows=rows,
            cols=cols,
            dtype=str(jp2.dtype),
            bands=bands,
            extras=extras,
        )

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the JP2 file.

        Uses efficient windowed reading when available (rasterio) or
        array slicing (glymur) to avoid loading the full image.

        Parameters
        ----------
        row_start : int
            Starting row index (inclusive).
        row_end : int
            Ending row index (exclusive).
        col_start : int
            Starting column index (inclusive).
        col_end : int
            Ending column index (exclusive).
        bands : Optional[List[int]]
            Band indices to read (0-based). If None, read all bands.
            Only applicable to multi-band JP2 files.

        Returns
        -------
        np.ndarray
            Image chip with shape ``(rows, cols)`` for single band or
            ``(bands, rows, cols)`` for multi-band.

        Raises
        ------
        ValueError
            If indices are out of bounds.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata['rows'] or col_end > self.metadata['cols']:
            raise ValueError("End indices exceed image dimensions")

        if self.backend == 'rasterio':
            return self._read_chip_rasterio(
                row_start, row_end, col_start, col_end, bands
            )
        else:
            return self._read_chip_glymur(
                row_start, row_end, col_start, col_end, bands
            )

    def _read_chip_rasterio(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]],
    ) -> np.ndarray:
        """Read chip using rasterio backend."""
        ds = self._dataset
        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)

        if bands is None:
            # Read all bands
            data = ds.read(window=window)
        else:
            # Read specific bands (convert to 1-based indexing)
            band_indices = [b + 1 for b in bands]
            data = ds.read(band_indices, window=window)

        # Rasterio returns (bands, rows, cols)
        if data.shape[0] == 1:
            return data[0]
        return data

    def _read_chip_glymur(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]],
    ) -> np.ndarray:
        """Read chip using glymur backend."""
        jp2 = self._glymur_jp2

        # Glymur uses [row_start:row_end, col_start:col_end] slicing
        # Returns (rows, cols) for 2D or (rows, cols, bands) for 3D
        data = jp2[row_start:row_end, col_start:col_end]

        if data.ndim == 2:
            return data

        # 3D: shape is (rows, cols, bands) - need to transpose to (bands, rows, cols)
        if bands is not None:
            # Select specific bands
            data = data[:, :, bands]

        # Transpose to match rasterio convention: (bands, rows, cols)
        data = np.transpose(data, (2, 0, 1))

        if data.shape[0] == 1:
            return data[0]
        return data

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire JP2 file.

        Parameters
        ----------
        bands : Optional[List[int]]
            Band indices to read (0-based). If None, read all bands.

        Returns
        -------
        np.ndarray
            Full image data.
        """
        return self.read_chip(
            0, self.metadata['rows'], 0, self.metadata['cols'], bands=bands
        )

    def get_shape(self) -> Tuple[int, ...]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, ...]
            ``(rows, cols)`` for single band or
            ``(rows, cols, bands)`` for multi-band.
        """
        if self.metadata['bands'] == 1:
            return (self.metadata['rows'], self.metadata['cols'])
        return (
            self.metadata['rows'],
            self.metadata['cols'],
            self.metadata['bands'],
        )

    def get_dtype(self) -> np.dtype:
        """Get the data type of the dataset.

        Returns
        -------
        np.dtype
        """
        return np.dtype(self.metadata['dtype'])

    def close(self) -> None:
        """Close the JP2 file handle."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
        # glymur Jp2k objects don't need explicit closing
        self._glymur_jp2 = None
