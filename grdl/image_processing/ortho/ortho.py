# -*- coding: utf-8 -*-
"""
Orthorectification - Reproject imagery to ground-referenced geographic grids.

Transforms imagery from its native acquisition geometry (slant range for SAR,
oblique for EO) to a regular geographic or ENU grid using inverse geolocation
and accelerated resampling. Works with any Geolocation subclass from
grdl.geolocation and accepts both ``GeographicGrid`` (WGS-84) and ``ENUGrid``
(local meters) as output specifications.

The coordinate mapping step (``compute_mapping``) parallelises across threads
for grids exceeding 1M pixels.  Resampling dispatches to the fastest available
backend via ``grdl.image_processing.ortho.accelerated.resample``:
numba (JIT parallel) > torch GPU > torch CPU > scipy parallel > scipy.

Dependencies
------------
scipy
numba (optional — JIT parallel acceleration)
torch (optional — GPU/CPU acceleration)

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
2026-01-30

Modified
--------
2026-03-19
"""

import logging
from typing import (
    Annotated, Any, Dict, List, Optional, Protocol, Tuple,
    TYPE_CHECKING, Union, runtime_checkable,
)

from grdl.exceptions import DependencyError
from grdl.image_processing.params import Desc, Options
from grdl.image_processing.versioning import processor_version

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.ndimage import map_coordinates as _map_coordinates
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from grdl.image_processing.base import ImageTransform

if TYPE_CHECKING:
    from grdl.geolocation.base import Geolocation
    from grdl.geolocation.elevation.base import ElevationModel
    from grdl.IO.base import ImageReader


# Mapping from interpolation name to scipy order parameter
_INTERPOLATION_ORDERS = {
    'nearest': 0,
    'bilinear': 1,
    'bicubic': 3,
}

# Pixel count above which compute_mapping parallelises across threads
_PARALLEL_THRESHOLD = 1_000_000


@runtime_checkable
class OutputGridProtocol(Protocol):
    """Contract for output grid objects used by ``Orthorectifier``.

    Both ``GeographicGrid`` (WGS-84 degrees) and ``ENUGrid`` (ENU meters)
    satisfy this protocol.  Any custom grid that implements these
    attributes and methods can be used as a drop-in replacement.

    Attributes
    ----------
    rows : int
        Number of output rows.
    cols : int
        Number of output columns.
    """

    rows: int
    cols: int

    def image_to_latlon(
        self,
        row: Union[float, np.ndarray],
        col: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert grid pixel coordinates to lat/lon."""
        ...

    def latlon_to_image(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert lat/lon to grid pixel coordinates."""
        ...

    def sub_grid(
        self,
        row_start: int,
        col_start: int,
        row_end: int,
        col_end: int,
    ) -> 'OutputGridProtocol':
        """Extract a sub-grid covering a rectangular tile."""
        ...


def validate_sub_grid_indices(
    rows: int,
    cols: int,
    row_start: int,
    col_start: int,
    row_end: int,
    col_end: int,
) -> None:
    """Validate sub-grid indices against parent grid dimensions.

    Parameters
    ----------
    rows : int
        Parent grid row count.
    cols : int
        Parent grid column count.
    row_start : int
        First row (inclusive).
    col_start : int
        First column (inclusive).
    row_end : int
        Last row (exclusive).
    col_end : int
        Last column (exclusive).

    Raises
    ------
    ValueError
        If indices are negative, exceed bounds, or produce an empty region.
    """
    if row_start < 0 or col_start < 0:
        raise ValueError(
            f"Indices must be non-negative, got "
            f"row_start={row_start}, col_start={col_start}"
        )
    if row_end > rows or col_end > cols:
        raise ValueError(
            f"Indices exceed grid dimensions ({rows}x{cols}), "
            f"got row_end={row_end}, col_end={col_end}"
        )
    if row_end <= row_start or col_end <= col_start:
        raise ValueError(
            f"Empty region: row [{row_start}, {row_end}), "
            f"col [{col_start}, {col_end})"
        )


class GeographicGrid:
    """
    Specification for an orthorectified output grid.

    Defines a regular geographic grid by its geographic bounds and pixel
    spacing. Provides simple affine transforms between grid pixel coordinates
    and geographic coordinates.

    The grid follows image conventions: row 0 is the northern (top) edge,
    row increases southward. Column 0 is the western (left) edge, column
    increases eastward.

    Attributes
    ----------
    min_lat : float
        Southern boundary (degrees North).
    max_lat : float
        Northern boundary (degrees North).
    min_lon : float
        Western boundary (degrees East).
    max_lon : float
        Eastern boundary (degrees East).
    pixel_size_lat : float
        Latitude spacing per pixel (degrees). Positive value.
    pixel_size_lon : float
        Longitude spacing per pixel (degrees). Positive value.
    rows : int
        Number of output rows.
    cols : int
        Number of output columns.

    Examples
    --------
    Create a grid covering a region at ~100m resolution:

    >>> grid = GeographicGrid(-31.5, -30.5, 115.5, 116.5, 0.001, 0.001)
    >>> print(f"{grid.rows} x {grid.cols}")
    1000 x 1000

    Create from a Geolocation object's footprint:

    >>> grid = GeographicGrid.from_geolocation(geo, 0.001, 0.001)
    """

    def __init__(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        pixel_size_lat: float,
        pixel_size_lon: float
    ) -> None:
        """
        Initialize output grid.

        Parameters
        ----------
        min_lat : float
            Southern boundary (degrees North).
        max_lat : float
            Northern boundary (degrees North).
        min_lon : float
            Western boundary (degrees East).
        max_lon : float
            Eastern boundary (degrees East).
        pixel_size_lat : float
            Latitude spacing per pixel (degrees). Must be positive.
        pixel_size_lon : float
            Longitude spacing per pixel (degrees). Must be positive.

        Raises
        ------
        ValueError
            If bounds are invalid or pixel sizes are not positive.
        """
        if max_lat <= min_lat:
            raise ValueError(
                f"max_lat ({max_lat}) must be greater than min_lat ({min_lat})"
            )
        if max_lon <= min_lon:
            raise ValueError(
                f"max_lon ({max_lon}) must be greater than min_lon ({min_lon})"
            )
        if pixel_size_lat <= 0:
            raise ValueError(
                f"pixel_size_lat must be positive, got {pixel_size_lat}"
            )
        if pixel_size_lon <= 0:
            raise ValueError(
                f"pixel_size_lon must be positive, got {pixel_size_lon}"
            )

        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.pixel_size_lat = pixel_size_lat
        self.pixel_size_lon = pixel_size_lon

        self.rows = int(np.ceil((max_lat - min_lat) / pixel_size_lat))
        self.cols = int(np.ceil((max_lon - min_lon) / pixel_size_lon))

    @classmethod
    def from_geolocation(
        cls,
        geolocation: 'Geolocation',
        pixel_size_lat: float,
        pixel_size_lon: float,
        margin: float = 0.0
    ) -> 'GeographicGrid':
        """
        Create grid from a Geolocation object's footprint bounds.

        Parameters
        ----------
        geolocation : Geolocation
            Source geolocation for computing bounds.
        pixel_size_lat : float
            Output latitude spacing (degrees).
        pixel_size_lon : float
            Output longitude spacing (degrees).
        margin : float, default=0.0
            Margin to add around bounds (degrees).

        Returns
        -------
        GeographicGrid
            Grid covering the geolocation footprint.

        Raises
        ------
        NotImplementedError
            If the geolocation has no bounds information.
        """
        min_lon, min_lat, max_lon, max_lat = geolocation.get_bounds()

        return cls(
            min_lat=min_lat - margin,
            max_lat=max_lat + margin,
            min_lon=min_lon - margin,
            max_lon=max_lon + margin,
            pixel_size_lat=pixel_size_lat,
            pixel_size_lon=pixel_size_lon,
        )

    def image_to_latlon(
        self,
        row: Union[float, np.ndarray],
        col: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert output grid pixel coordinates to lat/lon.

        Row 0 corresponds to max_lat (top/north edge). Row increases
        southward. Column 0 corresponds to min_lon (left/west edge).
        Column increases eastward.

        Parameters
        ----------
        row : float or np.ndarray
            Row coordinate(s) (0-based).
        col : float or np.ndarray
            Column coordinate(s) (0-based).

        Returns
        -------
        Tuple[float or np.ndarray, float or np.ndarray]
            (latitude, longitude) in degrees.
        """
        lat = self.max_lat - row * self.pixel_size_lat
        lon = self.min_lon + col * self.pixel_size_lon
        return lat, lon

    def latlon_to_image(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert lat/lon to output grid pixel coordinates.

        Parameters
        ----------
        lat : float or np.ndarray
            Latitude(s) in degrees North.
        lon : float or np.ndarray
            Longitude(s) in degrees East.

        Returns
        -------
        Tuple[float or np.ndarray, float or np.ndarray]
            (row, col) pixel coordinates.
        """
        row = (self.max_lat - lat) / self.pixel_size_lat
        col = (lon - self.min_lon) / self.pixel_size_lon
        return row, col

    def sub_grid(
        self,
        row_start: int,
        col_start: int,
        row_end: int,
        col_end: int,
    ) -> 'GeographicGrid':
        """Extract a sub-grid covering a rectangular tile of this grid.

        Creates a new ``GeographicGrid`` whose geographic bounds correspond to
        the pixel region ``[row_start:row_end, col_start:col_end]`` of this
        grid.  Pixel sizes are preserved.

        Parameters
        ----------
        row_start : int
            First row (inclusive).
        col_start : int
            First column (inclusive).
        row_end : int
            Last row (exclusive).
        col_end : int
            Last column (exclusive).

        Returns
        -------
        GeographicGrid
            Sub-grid with geographic bounds matching the tile region.

        Raises
        ------
        ValueError
            If indices are out of range or produce an empty region.
        """
        validate_sub_grid_indices(
            self.rows, self.cols, row_start, col_start, row_end, col_end,
        )

        tile_rows = row_end - row_start
        tile_cols = col_end - col_start

        # Compute bounds from the anchor corner and exact tile dimensions
        # to avoid floating-point drift from independent multiplications.
        tile_max_lat = self.max_lat - row_start * self.pixel_size_lat
        tile_min_lat = tile_max_lat - tile_rows * self.pixel_size_lat
        tile_min_lon = self.min_lon + col_start * self.pixel_size_lon
        tile_max_lon = tile_min_lon + tile_cols * self.pixel_size_lon

        sub = GeographicGrid(
            tile_min_lat, tile_max_lat,
            tile_min_lon, tile_max_lon,
            self.pixel_size_lat, self.pixel_size_lon,
        )
        # Force exact tile dimensions — ceil() in the constructor can
        # drift by ±1 from floating-point arithmetic on the bounds.
        sub.rows = tile_rows
        sub.cols = tile_cols
        return sub

    def __repr__(self) -> str:
        return (
            f"GeographicGrid(lat=[{self.min_lat:.4f}, {self.max_lat:.4f}], "
            f"lon=[{self.min_lon:.4f}, {self.max_lon:.4f}], "
            f"size={self.rows}x{self.cols}, "
            f"res=({self.pixel_size_lat:.6f}, {self.pixel_size_lon:.6f}))"
        )


# Backwards-compatible alias
OutputGrid = GeographicGrid


@processor_version('0.1.0')
class Orthorectifier(ImageTransform):
    """
    Orthorectify imagery from native geometry to a regular geographic grid.

    Transforms imagery from its acquisition geometry (slant range for SAR,
    oblique for EO) to a ground-projected, map-referenced regular grid
    using inverse geolocation and resampling.

    Algorithm
    ---------
    1. Define output grid (``GeographicGrid``) with geographic bounds and resolution.
    2. For each output pixel, compute the corresponding source pixel location
       via ``geolocation.latlon_to_image()`` (inverse transform).
    3. Resample source data at those fractional pixel locations using
       ``scipy.ndimage.map_coordinates``.

    All coordinate transforms and resampling are vectorized. No Python loops
    in the critical path.

    Attributes
    ----------
    geolocation : Geolocation
        Source image geolocation (provides latlon_to_image).
    output_grid : GeographicGrid
        Specification of the output geographic grid.
    interpolation : str
        Resampling method: 'nearest', 'bilinear', or 'bicubic'.

    Examples
    --------
    Orthorectify BIOMASS SAR data:

    >>> from grdl.IO import BIOMASSL1Reader
    >>> from grdl.geolocation.sar.gcp import GCPGeolocation
    >>> from grdl.image_processing import Orthorectifier, GeographicGrid
    >>>
    >>> with BIOMASSL1Reader('product_dir') as reader:
    ...     geo = GCPGeolocation(
    ...         reader.metadata['gcps'],
    ...         (reader.metadata['rows'], reader.metadata['cols']),
    ...     )
    ...     grid = GeographicGrid.from_geolocation(geo, 0.001, 0.001)
    ...     ortho = Orthorectifier(geo, grid)
    ...     result = ortho.apply_from_reader(reader, bands=[0])
    """

    # -- Annotated scalar field for GUI introspection (__param_specs__) --
    interpolation: Annotated[str, Options('nearest', 'bilinear', 'bicubic'), Desc('Resampling interpolation method')] = 'bilinear'

    def __init__(
        self,
        geolocation: 'Geolocation',
        output_grid: OutputGridProtocol,
        interpolation: str = 'bilinear',
        elevation: Optional['ElevationModel'] = None,
    ) -> None:
        """
        Initialize orthorectifier.

        Parameters
        ----------
        geolocation : Geolocation
            Geolocation for the source image. Must support
            ``latlon_to_image()`` for inverse mapping.
        output_grid : OutputGridProtocol
            Output grid specification defining bounds and resolution.
            Any object satisfying ``OutputGridProtocol`` (e.g.
            ``GeographicGrid``, ``ENUGrid``).
        interpolation : str, default='bilinear'
            Resampling method. One of 'nearest', 'bilinear', 'bicubic'.
        elevation : ElevationModel, optional
            Terrain elevation model for DEM-corrected projection. When
            provided, ``compute_mapping()`` looks up terrain heights at
            each output grid point and passes them to the inverse
            geolocation for terrain-corrected pixel mapping.

        Raises
        ------
        ValueError
            If interpolation method is not recognized.
        ImportError
            If scipy is not available.
        """
        if not SCIPY_AVAILABLE:
            raise DependencyError(
                "scipy is required for orthorectification. "
                "Install with: pip install scipy>=1.7.0"
            )

        if interpolation not in _INTERPOLATION_ORDERS:
            raise ValueError(
                f"Unknown interpolation method '{interpolation}'. "
                f"Must be one of: {list(_INTERPOLATION_ORDERS.keys())}"
            )

        self.geolocation = geolocation
        self.output_grid = output_grid
        self.interpolation = interpolation
        self.elevation = elevation
        self._order = _INTERPOLATION_ORDERS[interpolation]

        # Cached mapping (computed lazily)
        self._source_rows: Optional[np.ndarray] = None
        self._source_cols: Optional[np.ndarray] = None
        self._valid_mask: Optional[np.ndarray] = None

    def compute_mapping(
        self,
        num_workers: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the source pixel coordinates for every output pixel.

        For each output grid pixel, uses ``geolocation.latlon_to_image()``
        to find the corresponding source (row, col). The mapping is computed
        once and cached for reuse across bands.

        When the output grid exceeds ``_PARALLEL_THRESHOLD`` pixels,
        the mapping is computed in row-chunks across threads for speed.

        Parameters
        ----------
        num_workers : int, optional
            Thread count for parallel mapping.  Defaults to
            ``os.cpu_count() - 1``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (source_rows, source_cols, valid_mask) arrays, each with shape
            (output_grid.rows, output_grid.cols).
            - source_rows: Fractional row coordinates in the source image.
            - source_cols: Fractional column coordinates in the source image.
            - valid_mask: Boolean array. True where the output pixel maps
              to a valid source pixel (within bounds, not NaN).

        Notes
        -----
        Vectorized: generates full lat/lon grids with ``np.meshgrid``,
        calls ``latlon_to_image`` on flattened arrays, reshapes to 2D.
        For large grids (>1M pixels), computation is parallelized by
        row-chunks across a thread pool.
        """
        grid = self.output_grid
        total_pixels = grid.rows * grid.cols

        if self.elevation is not None:
            logger.debug("Using DEM for terrain-corrected mapping")
        else:
            logger.debug("No DEM provided, using flat-earth mapping")

        out_cols = np.arange(grid.cols, dtype=np.float64) + 0.5

        if total_pixels > _PARALLEL_THRESHOLD and grid.rows > 1:
            logger.info(
                "Computing %dx%d mapping (parallel)", grid.rows, grid.cols,
            )
            source_rows, source_cols = self._compute_mapping_parallel(
                out_cols, num_workers,
            )
        else:
            logger.info(
                "Computing %dx%d mapping (sequential)", grid.rows, grid.cols,
            )
            source_rows, source_cols = self._compute_strip(
                0, grid.rows, out_cols,
            )

        return self._finalize_mapping(source_rows, source_cols)

    def _compute_strip(
        self,
        row_start: int,
        row_end: int,
        out_cols: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute source pixel mapping for a contiguous row range.

        This is the single implementation of the inverse-geolocation
        core: grid pixel → lat/lon → (optional DEM) → source pixel.
        Both sequential and parallel paths call this method.

        Parameters
        ----------
        row_start : int
            First output row (inclusive).
        row_end : int
            Last output row (exclusive).
        out_cols : np.ndarray
            Pre-computed column centre coordinates (shared across strips).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (source_rows, source_cols) with shape
            ``(row_end - row_start, grid.cols)``.
        """
        grid = self.output_grid
        n_rows = row_end - row_start

        strip_rows = np.arange(row_start, row_end, dtype=np.float64) + 0.5
        col_g, row_g = np.meshgrid(out_cols, strip_rows)

        lats, lons = grid.image_to_latlon(row_g.ravel(), col_g.ravel())
        lats_flat = np.asarray(lats).ravel()
        lons_flat = np.asarray(lons).ravel()

        if self.elevation is not None:
            heights = self.elevation.get_elevation(lats_flat, lons_flat)
            heights = np.where(np.isfinite(heights), heights, 0.0)
            coords = np.column_stack([lats_flat, lons_flat, heights])
        else:
            # No ortho-level DEM — pass (N, 2) so the geolocation
            # object uses its own DEM / default HAE internally.
            coords = np.column_stack([lats_flat, lons_flat])
        src_px = self.geolocation.latlon_to_image(coords)
        return (
            src_px[:, 0].reshape(n_rows, grid.cols),
            src_px[:, 1].reshape(n_rows, grid.cols),
        )

    def _compute_mapping_parallel(
        self,
        out_cols: np.ndarray,
        num_workers: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mapping in parallel row-chunks using threads.

        Distributes row-strips across a thread pool, each calling
        ``_compute_strip``.  NumPy releases the GIL during array
        operations, enabling true multi-threading.

        Parameters
        ----------
        out_cols : np.ndarray
            Pre-computed column centre coordinates.
        num_workers : int, optional
            Thread count.  Defaults to ``os.cpu_count() - 1``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (source_rows, source_cols) with shape
            ``(grid.rows, grid.cols)``.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor

        grid = self.output_grid
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 1)

        source_rows = np.empty((grid.rows, grid.cols), dtype=np.float64)
        source_cols = np.empty((grid.rows, grid.cols), dtype=np.float64)

        chunk_h = max(1, grid.rows // num_workers)
        row_slices = []
        for s in range(0, grid.rows, chunk_h):
            row_slices.append(slice(s, min(s + chunk_h, grid.rows)))

        def _process_strip(rs: slice) -> None:
            sr, sc = self._compute_strip(rs.start, rs.stop, out_cols)
            source_rows[rs] = sr
            source_cols[rs] = sc

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            list(pool.map(_process_strip, row_slices))

        return source_rows, source_cols

    def _finalize_mapping(
        self,
        source_rows: np.ndarray,
        source_cols: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate mapping bounds, cache results, and return.

        Parameters
        ----------
        source_rows : np.ndarray
            Fractional source row coordinates, shape ``(OH, OW)``.
        source_cols : np.ndarray
            Fractional source column coordinates, shape ``(OH, OW)``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(source_rows, source_cols, valid_mask)``.
        """
        src_shape = self.geolocation.shape
        valid = (
            np.isfinite(source_rows) &
            np.isfinite(source_cols) &
            (source_rows >= 0) &
            (source_rows < src_shape[0]) &
            (source_cols >= 0) &
            (source_cols < src_shape[1])
        )

        self._source_rows = source_rows
        self._source_cols = source_cols
        self._valid_mask = valid

        valid_pct = 100.0 * np.count_nonzero(valid) / valid.size
        logger.debug("Valid pixel coverage: %.1f%%", valid_pct)

        return source_rows, source_cols, valid

    def apply(
        self,
        source: np.ndarray,
        nodata: float = 0.0,
        backend: str = 'auto',
        source_origin: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Orthorectify a source image array.

        Uses the accelerated resampling backend (GPU, numba, parallel
        scipy, or single-threaded scipy) selected automatically based
        on available libraries.

        Parameters
        ----------
        source : np.ndarray
            Source image. Shape (rows, cols) for single-band or
            (bands, rows, cols) for multi-band.
        nodata : float, default=0.0
            Fill value for output pixels with no source coverage.
        backend : str, default='auto'
            Resampling backend: ``'auto'``, ``'torch_gpu'``,
            ``'torch'``, ``'numba'``, ``'scipy_parallel'``,
            ``'scipy'``.
        source_origin : tuple[int, int], optional
            ``(row_start, col_start)`` pixel offset of *source* within
            the full image coordinate system used by the geolocation.
            When provided, the cached mapping coordinates are shifted
            to chip-relative indices and the valid mask is recomputed
            against the chip dimensions.  This allows orthorectifying a
            pre-read chip without re-reading from a reader.

        Returns
        -------
        np.ndarray
            Orthorectified image. Shape (output_rows, output_cols) for
            single-band or (bands, output_rows, output_cols) for
            multi-band. dtype matches source.

        Notes
        -----
        Complex-valued data is handled by resampling real and imaginary
        parts independently.  Multi-band images are processed in a
        single dispatch call when using torch or numba backends.
        """
        if self._source_rows is None:
            self.compute_mapping()

        source_rows = self._source_rows
        source_cols = self._source_cols
        valid_mask = self._valid_mask

        if source_origin is not None:
            row_off, col_off = source_origin
            source_rows = source_rows - row_off
            source_cols = source_cols - col_off

            chip_h = source.shape[-2] if source.ndim == 3 else source.shape[0]
            chip_w = source.shape[-1] if source.ndim == 3 else source.shape[1]
            valid_mask = (
                valid_mask
                & (source_rows >= 0)
                & (source_rows < chip_h)
                & (source_cols >= 0)
                & (source_cols < chip_w)
            )

        from grdl.image_processing.ortho.accelerated import resample

        return resample(
            source,
            source_rows,
            source_cols,
            valid_mask,
            order=self._order,
            nodata=nodata,
            backend=backend,
        )

    def apply_from_reader(
        self,
        reader: 'ImageReader',
        bands: Optional[List[int]] = None,
        nodata: float = 0.0
    ) -> np.ndarray:
        """
        Orthorectify imagery from a reader (handles large files).

        Reads the source data via the reader and orthorectifies it. For
        very large images, consider reading and processing in spatial
        subsets externally.

        Parameters
        ----------
        reader : ImageReader
            Source imagery reader.
        bands : Optional[List[int]], default=None
            Band indices to process. None means all bands.
        nodata : float, default=0.0
            Fill value for pixels without source coverage.

        Returns
        -------
        np.ndarray
            Orthorectified image. Shape (output_rows, output_cols) for
            single-band or (bands, output_rows, output_cols) for
            multi-band.
        """
        if self._source_rows is None:
            self.compute_mapping()

        source_rows = self._source_rows
        source_cols = self._source_cols
        valid = self._valid_mask

        # Determine source region needed from mapping bounds
        valid_src_rows = source_rows[valid]
        valid_src_cols = source_cols[valid]

        if valid_src_rows.size == 0:
            logger.warning("No valid source mapping found, returning nodata grid")
            # No valid mapping -- return all nodata
            shape = reader.get_shape()
            n_bands = len(bands) if bands else (shape[2] if len(shape) > 2 else 1)
            if n_bands == 1:
                return np.full(
                    (self.output_grid.rows, self.output_grid.cols),
                    nodata, dtype=np.float64
                )
            return np.full(
                (n_bands, self.output_grid.rows, self.output_grid.cols),
                nodata, dtype=np.float64
            )

        # Compute bounding box of needed source pixels (with padding for
        # interpolation kernel)
        pad = self._order + 1
        row_min = max(0, int(np.floor(valid_src_rows.min())) - pad)
        row_max = min(
            self.geolocation.shape[0],
            int(np.ceil(valid_src_rows.max())) + pad + 1
        )
        col_min = max(0, int(np.floor(valid_src_cols.min())) - pad)
        col_max = min(
            self.geolocation.shape[1],
            int(np.ceil(valid_src_cols.max())) + pad + 1
        )

        # Read the needed source region
        source_chip = reader.read_chip(row_min, row_max, col_min, col_max,
                                       bands=bands)
        is_complex = np.iscomplexobj(source_chip)

        # Adjust mapping coordinates to chip-relative
        chip_rows = source_rows - row_min
        chip_cols = source_cols - col_min

        # Update valid mask for chip bounds
        chip_h = source_chip.shape[-2] if source_chip.ndim == 3 else source_chip.shape[0]
        chip_w = source_chip.shape[-1] if source_chip.ndim == 3 else source_chip.shape[1]
        chip_valid = (
            valid &
            (chip_rows >= 0) &
            (chip_rows < chip_h) &
            (chip_cols >= 0) &
            (chip_cols < chip_w)
        )

        from grdl.image_processing.ortho.accelerated import resample

        return resample(
            source_chip, chip_rows, chip_cols, chip_valid,
            order=self._order, nodata=nodata,
        )

    def get_output_geolocation_metadata(self) -> Dict[str, Any]:
        """
        Return metadata describing the output grid's geolocation.

        Provides the affine transform parameters, CRS, bounds, and pixel
        size for the orthorectified output. Suitable for passing to an
        ImageWriter or for constructing a GeoTIFF header.

        Returns
        -------
        Dict[str, Any]
            - 'crs': str, coordinate reference system (WGS84)
            - 'bounds': Tuple, (min_lon, min_lat, max_lon, max_lat)
            - 'pixel_size_lat': float, latitude spacing (degrees)
            - 'pixel_size_lon': float, longitude spacing (degrees)
            - 'rows': int, number of output rows
            - 'cols': int, number of output columns
            - 'transform': Tuple, affine coefficients
              (origin_lon, pixel_size_lon, 0, origin_lat, 0, -pixel_size_lat)
        """
        from grdl.image_processing.ortho.enu_grid import ENUGrid

        grid = self.output_grid

        if isinstance(grid, ENUGrid):
            return {
                'crs': 'ENU',
                'reference_point': (
                    grid.ref_lat, grid.ref_lon, grid.ref_alt,
                ),
                'bounds_enu': (
                    grid.min_east, grid.min_north,
                    grid.max_east, grid.max_north,
                ),
                'pixel_size_east': grid.pixel_size_east,
                'pixel_size_north': grid.pixel_size_north,
                'rows': grid.rows,
                'cols': grid.cols,
                'transform': (
                    grid.min_east,
                    grid.pixel_size_east,
                    0.0,
                    grid.max_north,
                    0.0,
                    -grid.pixel_size_north,
                ),
            }

        return {
            'crs': 'WGS84',
            'bounds': (grid.min_lon, grid.min_lat, grid.max_lon, grid.max_lat),
            'pixel_size_lat': grid.pixel_size_lat,
            'pixel_size_lon': grid.pixel_size_lon,
            'rows': grid.rows,
            'cols': grid.cols,
            'transform': (
                grid.min_lon,           # origin longitude (top-left corner)
                grid.pixel_size_lon,    # pixel width (degrees)
                0.0,                    # rotation (0 for north-up)
                grid.max_lat,           # origin latitude (top-left corner)
                0.0,                    # rotation (0 for north-up)
                -grid.pixel_size_lat,   # pixel height (negative = south)
            ),
        }

    def __repr__(self) -> str:
        return (
            f"Orthorectifier(grid={self.output_grid}, "
            f"interpolation='{self.interpolation}')"
        )
