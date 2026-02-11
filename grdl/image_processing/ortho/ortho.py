# -*- coding: utf-8 -*-
"""
Orthorectification - Reproject imagery to ground-referenced geographic grids.

Transforms imagery from its native acquisition geometry (slant range for SAR,
oblique for EO) to a regular geographic grid using inverse geolocation and
resampling. Works with any Geolocation subclass from grdl.geolocation.

Dependencies
------------
scipy

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
2026-02-06
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from grdl.image_processing.versioning import processor_version

import numpy as np

try:
    from scipy.ndimage import map_coordinates
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    map_coordinates = None

from grdl.image_processing.base import ImageTransform

if TYPE_CHECKING:
    from grdl.geolocation.base import Geolocation
    from grdl.IO.base import ImageReader


# Mapping from interpolation name to scipy order parameter
_INTERPOLATION_ORDERS = {
    'nearest': 0,
    'bilinear': 1,
    'bicubic': 3,
}


class OutputGrid:
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

    >>> grid = OutputGrid(-31.5, -30.5, 115.5, 116.5, 0.001, 0.001)
    >>> print(f"{grid.rows} x {grid.cols}")
    1000 x 1000

    Create from a Geolocation object's footprint:

    >>> grid = OutputGrid.from_geolocation(geo, 0.001, 0.001)
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
    ) -> 'OutputGrid':
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
        OutputGrid
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

    def pixel_to_latlon(
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

    def latlon_to_pixel(
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

    def __repr__(self) -> str:
        return (
            f"OutputGrid(lat=[{self.min_lat:.4f}, {self.max_lat:.4f}], "
            f"lon=[{self.min_lon:.4f}, {self.max_lon:.4f}], "
            f"size={self.rows}x{self.cols}, "
            f"res=({self.pixel_size_lat:.6f}, {self.pixel_size_lon:.6f}))"
        )


@processor_version('0.1.0')
class Orthorectifier(ImageTransform):
    """
    Orthorectify imagery from native geometry to a regular geographic grid.

    Transforms imagery from its acquisition geometry (slant range for SAR,
    oblique for EO) to a ground-projected, map-referenced regular grid
    using inverse geolocation and resampling.

    Algorithm
    ---------
    1. Define output grid (``OutputGrid``) with geographic bounds and resolution.
    2. For each output pixel, compute the corresponding source pixel location
       via ``geolocation.latlon_to_pixel()`` (inverse transform).
    3. Resample source data at those fractional pixel locations using
       ``scipy.ndimage.map_coordinates``.

    All coordinate transforms and resampling are vectorized. No Python loops
    in the critical path.

    Attributes
    ----------
    geolocation : Geolocation
        Source image geolocation (provides latlon_to_pixel).
    output_grid : OutputGrid
        Specification of the output geographic grid.
    interpolation : str
        Resampling method: 'nearest', 'bilinear', or 'bicubic'.

    Examples
    --------
    Orthorectify BIOMASS SAR data:

    >>> from grdl.IO import BIOMASSL1Reader
    >>> from grdl.geolocation.sar.gcp import GCPGeolocation
    >>> from grdl.image_processing import Orthorectifier, OutputGrid
    >>>
    >>> with BIOMASSL1Reader('product_dir') as reader:
    ...     geo = GCPGeolocation(
    ...         reader.metadata['gcps'],
    ...         (reader.metadata['rows'], reader.metadata['cols']),
    ...     )
    ...     grid = OutputGrid.from_geolocation(geo, 0.001, 0.001)
    ...     ortho = Orthorectifier(geo, grid)
    ...     result = ortho.apply_from_reader(reader, bands=[0])
    """

    def __init__(
        self,
        geolocation: 'Geolocation',
        output_grid: OutputGrid,
        interpolation: str = 'bilinear'
    ) -> None:
        """
        Initialize orthorectifier.

        Parameters
        ----------
        geolocation : Geolocation
            Geolocation for the source image. Must support
            ``latlon_to_pixel()`` for inverse mapping.
        output_grid : OutputGrid
            Output grid specification defining bounds and resolution.
        interpolation : str, default='bilinear'
            Resampling method. One of 'nearest', 'bilinear', 'bicubic'.

        Raises
        ------
        ValueError
            If interpolation method is not recognized.
        ImportError
            If scipy is not available.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
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
        self._order = _INTERPOLATION_ORDERS[interpolation]

        # Cached mapping (computed lazily)
        self._source_rows: Optional[np.ndarray] = None
        self._source_cols: Optional[np.ndarray] = None
        self._valid_mask: Optional[np.ndarray] = None

    def compute_mapping(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the source pixel coordinates for every output pixel.

        For each output grid pixel, uses ``geolocation.latlon_to_pixel()``
        to find the corresponding source (row, col). The mapping is computed
        once and cached for reuse across bands.

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
        calls ``latlon_to_pixel`` on flattened arrays, reshapes to 2D.
        """
        grid = self.output_grid

        # Build 1D coordinate arrays for the output grid
        # Row 0 = max_lat (north), row N-1 = min_lat (south)
        out_lats = grid.max_lat - (np.arange(grid.rows) + 0.5) * grid.pixel_size_lat
        out_lons = grid.min_lon + (np.arange(grid.cols) + 0.5) * grid.pixel_size_lon

        # Create 2D meshgrid and flatten for vectorized transform
        lon_grid, lat_grid = np.meshgrid(out_lons, out_lats)
        lats_flat = lat_grid.ravel()
        lons_flat = lon_grid.ravel()

        # Inverse geolocation: output lat/lon -> source pixel (row, col)
        src_rows_flat, src_cols_flat = self.geolocation.latlon_to_pixel(
            lats_flat, lons_flat
        )

        # Reshape to output grid dimensions
        source_rows = src_rows_flat.reshape(grid.rows, grid.cols)
        source_cols = src_cols_flat.reshape(grid.rows, grid.cols)

        # Valid mask: not NaN and within source image bounds
        src_shape = self.geolocation.shape
        valid = (
            np.isfinite(source_rows) &
            np.isfinite(source_cols) &
            (source_rows >= 0) &
            (source_rows < src_shape[0]) &
            (source_cols >= 0) &
            (source_cols < src_shape[1])
        )

        # Cache for reuse
        self._source_rows = source_rows
        self._source_cols = source_cols
        self._valid_mask = valid

        return source_rows, source_cols, valid

    def apply(
        self,
        source: np.ndarray,
        nodata: float = 0.0
    ) -> np.ndarray:
        """
        Orthorectify a source image array.

        Parameters
        ----------
        source : np.ndarray
            Source image. Shape (rows, cols) for single-band or
            (bands, rows, cols) for multi-band.
        nodata : float, default=0.0
            Fill value for output pixels with no source coverage.

        Returns
        -------
        np.ndarray
            Orthorectified image. Shape (output_rows, output_cols) for
            single-band or (bands, output_rows, output_cols) for
            multi-band. dtype matches source.

        Notes
        -----
        Complex-valued data is handled by resampling real and imaginary
        parts independently.
        """
        if self._source_rows is None:
            self.compute_mapping()

        source_rows = self._source_rows
        source_cols = self._source_cols
        valid = self._valid_mask

        is_complex = np.iscomplexobj(source)
        is_multiband = source.ndim == 3

        if is_multiband:
            n_bands = source.shape[0]
            output = np.full(
                (n_bands, self.output_grid.rows, self.output_grid.cols),
                nodata,
                dtype=source.dtype
            )
            for b in range(n_bands):
                self._resample_band(
                    source[b], source_rows, source_cols, valid,
                    output[b], nodata, is_complex
                )
        else:
            output = np.full(
                (self.output_grid.rows, self.output_grid.cols),
                nodata,
                dtype=source.dtype
            )
            self._resample_band(
                source, source_rows, source_cols, valid,
                output, nodata, is_complex
            )

        return output

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
        chip_valid = (
            valid &
            (chip_rows >= 0) &
            (chip_rows < source_chip.shape[-2] if source_chip.ndim == 3
             else source_chip.shape[0]) &
            (chip_cols >= 0) &
            (chip_cols < source_chip.shape[-1] if source_chip.ndim == 3
             else source_chip.shape[1])
        )

        is_multiband = source_chip.ndim == 3

        if is_multiband:
            n_bands = source_chip.shape[0]
            output = np.full(
                (n_bands, self.output_grid.rows, self.output_grid.cols),
                nodata, dtype=source_chip.dtype
            )
            for b in range(n_bands):
                self._resample_band(
                    source_chip[b], chip_rows, chip_cols, chip_valid,
                    output[b], nodata, is_complex
                )
        else:
            output = np.full(
                (self.output_grid.rows, self.output_grid.cols),
                nodata, dtype=source_chip.dtype
            )
            self._resample_band(
                source_chip, chip_rows, chip_cols, chip_valid,
                output, nodata, is_complex
            )

        return output

    def _resample_band(
        self,
        source_band: np.ndarray,
        source_rows: np.ndarray,
        source_cols: np.ndarray,
        valid: np.ndarray,
        output_band: np.ndarray,
        nodata: float,
        is_complex: bool
    ) -> None:
        """
        Resample a single 2D band using map_coordinates.

        Parameters
        ----------
        source_band : np.ndarray
            Source band data, shape (rows, cols).
        source_rows : np.ndarray
            Fractional source row coordinates, shape (out_rows, out_cols).
        source_cols : np.ndarray
            Fractional source column coordinates, shape (out_rows, out_cols).
        valid : np.ndarray
            Boolean mask of valid output pixels, shape (out_rows, out_cols).
        output_band : np.ndarray
            Output array to fill in-place, shape (out_rows, out_cols).
        nodata : float
            Fill value for invalid pixels.
        is_complex : bool
            Whether the source data is complex-valued.
        """
        if not np.any(valid):
            return

        # Extract only valid coordinates for efficiency
        valid_rows = source_rows[valid]
        valid_cols = source_cols[valid]
        coords = np.array([valid_rows, valid_cols])

        if is_complex:
            # Resample real and imaginary parts independently
            real_vals = map_coordinates(
                source_band.real.astype(np.float64),
                coords, order=self._order, mode='constant', cval=nodata
            )
            imag_vals = map_coordinates(
                source_band.imag.astype(np.float64),
                coords, order=self._order, mode='constant', cval=nodata
            )
            output_band[valid] = (real_vals + 1j * imag_vals).astype(
                source_band.dtype
            )
        else:
            vals = map_coordinates(
                source_band.astype(np.float64),
                coords, order=self._order, mode='constant', cval=nodata
            )
            output_band[valid] = vals.astype(source_band.dtype)

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
        grid = self.output_grid
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
