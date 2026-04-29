# -*- coding: utf-8 -*-
"""
Orthorectification Builder - Universal builder for orthorectifying imagery.

Orchestrates reader, geolocation, resolution computation, output grid
construction, terrain-corrected orthorectification, and optional GeoTIFF
output.  Works with any ``ImageReader`` and ``Geolocation`` subclass.

Supports both WGS-84 geographic grids (``GeographicGrid``) and local ENU
(East-North-Up) grids in meters (``ENUGrid``) via ``with_enu_grid()``.
Tiled processing is available via ``with_tile_size()`` for memory-
efficient handling of large output grids.

Resampling uses the accelerated multi-backend dispatch chain from
``grdl.image_processing.ortho.accelerated``.

Dependencies
------------
scipy
numba (optional — JIT parallel acceleration)
torch (optional — GPU/CPU acceleration)

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
2026-02-17

Modified
--------
2026-03-27  DEM attached to geolocation instead of passed to Orthorectifier.
"""

# Standard library
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.exceptions import DependencyError
from grdl.image_processing.ortho.ortho import Orthorectifier, GeographicGrid

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from grdl.IO.base import ImageReader
    from grdl.geolocation.base import Geolocation


class OrthoResult:
    """Container for orthorectification results.

    Holds the orthorectified data along with grid and geolocation
    metadata.  Provides ``save_geotiff()`` for writing the result as
    a georeferenced GeoTIFF raster.

    Attributes
    ----------
    data : np.ndarray
        Orthorectified image.  Shape ``(rows, cols)`` or
        ``(bands, rows, cols)``.
    output_grid : GeographicGrid
        Grid specification for the output.
    geolocation_metadata : Dict[str, Any]
        CRS, affine transform, bounds, and pixel sizes.
    orthorectifier : Orthorectifier
        The configured orthorectifier (mapping cached for reuse).
    """

    def __init__(
        self,
        data: np.ndarray,
        output_grid: GeographicGrid,
        geolocation_metadata: Dict[str, Any],
        orthorectifier: Orthorectifier,
    ) -> None:
        self.data = data
        self.output_grid = output_grid
        self.geolocation_metadata = geolocation_metadata
        self.orthorectifier = orthorectifier

    @property
    def shape(self) -> tuple:
        """Shape of the orthorectified data."""
        return self.data.shape

    @property
    def is_enu(self) -> bool:
        """Whether the output is in ENU coordinates."""
        from grdl.image_processing.ortho.enu_grid import ENUGrid
        return isinstance(self.output_grid, ENUGrid)

    @property
    def enu_reference_point(
        self,
    ) -> Optional[Tuple[float, float, float]]:
        """Reference point (lat, lon, alt) if ENU, else None."""
        if self.is_enu:
            g = self.output_grid
            return (g.ref_lat, g.ref_lon, g.ref_alt)
        return None

    @property
    def pixel_size_meters(
        self,
    ) -> Optional[Tuple[float, float]]:
        """Pixel size (east_m, north_m) if ENU, else None."""
        if self.is_enu:
            g = self.output_grid
            return (g.pixel_size_east, g.pixel_size_north)
        return None

    @property
    def bounds_meters(
        self,
    ) -> Optional[Tuple[float, float, float, float]]:
        """ENU bounds (min_east, min_north, max_east, max_north) or None."""
        if self.is_enu:
            g = self.output_grid
            return (g.min_east, g.min_north, g.max_east, g.max_north)
        return None

    def save_geotiff(self, filepath: Union[str, Path]) -> None:
        """Save result as a georeferenced GeoTIFF.

        Parameters
        ----------
        filepath : str or Path
            Output file path.

        Raises
        ------
        ImportError
            If rasterio is not installed.
        """
        from grdl.IO.geotiff import GeoTIFFWriter

        try:
            from rasterio.transform import Affine
        except ImportError:
            raise DependencyError(
                "rasterio is required for GeoTIFF output. "
                "Install with: conda install -c conda-forge rasterio"
            )

        meta = self.geolocation_metadata
        t = meta['transform']
        transform = Affine(
            t[1],  # pixel_size_lon
            t[2],  # 0
            t[0],  # origin_lon
            t[4],  # 0
            t[5],  # -pixel_size_lat
            t[3],  # origin_lat
        )

        geolocation = {
            'crs': 'EPSG:4326',
            'transform': transform,
        }

        writer = GeoTIFFWriter(filepath)
        writer.write(self.data, geolocation=geolocation)


class OrthoBuilder:
    """Universal orthorectification pipeline.

    Wires together reader, geolocation, resolution, elevation, grid,
    and orthorectifier into a single configurable pipeline.  Each
    component can be auto-configured or explicitly overridden.

    Uses a builder pattern: call ``with_*()`` methods to configure,
    then ``run()`` to execute.

    Examples
    --------
    SICD orthorectification with terrain correction::

        from grdl.IO.sar import SICDReader
        from grdl.geolocation.sar.sicd import SICDGeolocation
        from grdl.geolocation.elevation import DTEDElevation
        from grdl.image_processing.ortho import OrthoBuilder

        with SICDReader('image.nitf') as reader:
            geo = SICDGeolocation.from_reader(reader)
            elev = DTEDElevation('/path/to/dted')
            result = (OrthoBuilder()
                      .with_reader(reader)
                      .with_geolocation(geo)
                      .with_elevation(elev)
                      .run())
            result.save_geotiff('ortho_output.tif')

    Pre-processed source array (SAR magnitude computed in slant range)::

        mag_db = 20.0 * np.log10(np.abs(reader.read_full()) + 1e-10)
        result = (OrthoBuilder()
                  .with_source_array(mag_db)
                  .with_metadata(reader.metadata)
                  .with_geolocation(geo)
                  .with_interpolation('nearest')
                  .run())
    """

    def __init__(self) -> None:
        self._reader: Optional['ImageReader'] = None
        self._metadata: Optional[Any] = None
        self._geolocation: Optional['Geolocation'] = None
        self._output_grid: Optional[GeographicGrid] = None
        self._pixel_size_lat: Optional[float] = None
        self._pixel_size_lon: Optional[float] = None
        self._interpolation: str = 'bilinear'
        self._bands: Optional[List[int]] = None
        self._nodata: float = 0.0
        self._margin: float = 0.0
        self._scale_factor: float = 1.0
        self._source_array: Optional[np.ndarray] = None
        self._roi_bounds: Optional[Tuple[float, float, float, float]] = None
        self._tile_size: Optional[Union[int, Tuple[int, int]]] = None
        self._enu_params: Optional[Dict[str, Any]] = None
        self._batch_size: int = 2_000_000

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def with_reader(self, reader: 'ImageReader') -> 'OrthoBuilder':
        """Set the source imagery reader.

        Parameters
        ----------
        reader : ImageReader
            An open imagery reader.

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._reader = reader
        return self

    def with_metadata(self, metadata: Any) -> 'OrthoBuilder':
        """Set metadata for auto-resolution computation.

        Use when working with a pre-loaded source array instead of a
        reader.  The metadata is only needed if resolution is not set
        explicitly via ``with_resolution()``.

        Parameters
        ----------
        metadata : Any
            Reader metadata (e.g., ``SICDMetadata``, BIOMASS dict).

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._metadata = metadata
        return self

    def with_geolocation(
        self, geolocation: 'Geolocation'
    ) -> 'OrthoBuilder':
        """Set the source geolocation.

        Parameters
        ----------
        geolocation : Geolocation
            Geolocation for the source imagery.

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._geolocation = geolocation
        return self

    def with_output_grid(self, grid: GeographicGrid) -> 'OrthoBuilder':
        """Set an explicit output grid (overrides auto-computation).

        Parameters
        ----------
        grid : GeographicGrid
            Pre-built output grid.

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._output_grid = grid
        return self

    def with_resolution(
        self, pixel_size_lat: float, pixel_size_lon: float
    ) -> 'OrthoBuilder':
        """Set explicit output resolution in degrees.

        Parameters
        ----------
        pixel_size_lat : float
            Latitude pixel size (degrees).
        pixel_size_lon : float
            Longitude pixel size (degrees).

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._pixel_size_lat = pixel_size_lat
        self._pixel_size_lon = pixel_size_lon
        return self

    def with_interpolation(self, method: str) -> 'OrthoBuilder':
        """Set resampling interpolation method.

        Parameters
        ----------
        method : str
            One of ``'nearest'``, ``'bilinear'``, ``'bicubic'``,
            ``'lanczos'`` (Lanczos-3, requires numba backend).

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._interpolation = method
        return self

    def with_batch_size(self, batch_size: int) -> 'OrthoBuilder':
        """Set inverse-mapping batch size (points per chunk).

        Caps peak memory during the Newton-Raphson inverse projection
        by chunking the flat output grid into slices of ``batch_size``
        points. Each chunk is projected independently and the DEM is
        pre-sampled once per chunk.

        Parameters
        ----------
        batch_size : int
            Points per chunk. Default 2,000,000.

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {batch_size}"
            )
        self._batch_size = batch_size
        return self

    def with_bands(self, bands: List[int]) -> 'OrthoBuilder':
        """Set which bands to orthorectify (reader mode only).

        Parameters
        ----------
        bands : List[int]
            Band indices.

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._bands = bands
        return self

    def with_nodata(self, nodata: float) -> 'OrthoBuilder':
        """Set nodata fill value for pixels without source coverage.

        Parameters
        ----------
        nodata : float
            Fill value.  Default is ``0.0``.

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._nodata = nodata
        return self

    def with_margin(self, margin: float) -> 'OrthoBuilder':
        """Set margin around the footprint bounds (degrees).

        Parameters
        ----------
        margin : float
            Margin in degrees added to each side of the footprint.

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._margin = margin
        return self

    def with_scale_factor(self, factor: float) -> 'OrthoBuilder':
        """Set resolution scale factor for auto-computed resolution.

        Parameters
        ----------
        factor : float
            Multiplier.  Values > 1.0 produce coarser output.

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._scale_factor = factor
        return self

    def with_source_array(self, array: np.ndarray) -> 'OrthoBuilder':
        """Use a pre-loaded source array instead of reading from reader.

        This is the recommended path for SAR: compute magnitude or other
        real-valued products in slant range, then orthorectify the result.

        Parameters
        ----------
        array : np.ndarray
            Source image data.  Shape ``(rows, cols)`` or
            ``(bands, rows, cols)``.

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._source_array = array
        return self

    def with_roi(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
    ) -> 'OrthoBuilder':
        """Restrict output to a geographic region of interest.

        Only the specified bounding box is orthorectified, rather than
        the full source image footprint.

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

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._roi_bounds = (min_lat, max_lat, min_lon, max_lon)
        return self

    def with_tile_size(
        self, tile_size: Union[int, Tuple[int, int]]
    ) -> 'OrthoBuilder':
        """Enable tiled processing with the given tile dimensions.

        When set, the output grid is partitioned into tiles of the
        specified size.  Each tile independently computes its mapping,
        reads the needed source chip, and resamples.  Peak memory for
        mapping arrays is proportional to tile size, not full output
        size.

        Parameters
        ----------
        tile_size : int or Tuple[int, int]
            Tile dimensions in output pixels.  If ``int``, square
            tiles.  ``(tile_rows, tile_cols)`` for rectangular tiles.

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._tile_size = tile_size
        return self

    def with_enu_grid(
        self,
        pixel_size_m: float,
        ref_lat: Optional[float] = None,
        ref_lon: Optional[float] = None,
        ref_alt: float = 0.0,
        margin_m: float = 0.0,
    ) -> 'OrthoBuilder':
        """Configure ENU (East-North-Up) output in meters.

        When set, the output grid is defined in local ENU meters
        centered on a reference point, instead of geographic lat/lon.
        If ``ref_lat`` / ``ref_lon`` are not provided, the image center
        is used.

        Parameters
        ----------
        pixel_size_m : float
            Pixel spacing in meters (east and north).
        ref_lat : float, optional
            Reference latitude (degrees). Defaults to image center.
        ref_lon : float, optional
            Reference longitude (degrees). Defaults to image center.
        ref_alt : float
            Reference altitude (meters HAE).
        margin_m : float
            Margin around footprint bounds in meters.

        Returns
        -------
        OrthoBuilder
            Self for chaining.
        """
        self._enu_params = {
            'pixel_size_m': pixel_size_m,
            'ref_lat': ref_lat,
            'ref_lon': ref_lon,
            'ref_alt': ref_alt,
            'margin_m': margin_m,
        }
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> OrthoResult:
        """Execute the orthorectification pipeline.

        Returns
        -------
        OrthoResult
            Container with orthorectified data, grid, and metadata.

        Raises
        ------
        ValueError
            If required components (geolocation, reader or source array)
            are missing or resolution cannot be determined.
        """
        if self._geolocation is None:
            raise ValueError(
                "Geolocation is required.  "
                "Call .with_geolocation() before .run()."
            )
        if self._reader is None and self._source_array is None:
            raise ValueError(
                "Either a reader (.with_reader()) or a source array "
                "(.with_source_array()) is required."
            )

        if self._tile_size is not None:
            return self._run_tiled()

        # 1. Resolve output grid
        grid = self._resolve_output_grid()

        # 2. Create orthorectifier
        logger.info(
            "Starting mapping for %dx%d grid, DEM %s",
            grid.rows, grid.cols,
            "attached" if getattr(self._geolocation, 'elevation', None) is not None
            else "not provided",
        )
        ortho = Orthorectifier(
            geolocation=self._geolocation,
            output_grid=grid,
            interpolation=self._interpolation,
            batch_size=self._batch_size,
        )

        # 3. Compute mapping (geolocation handles DEM internally)
        ortho.compute_mapping()

        # 4. Resample
        if self._source_array is not None:
            data = ortho.apply(self._source_array, nodata=self._nodata)
        else:
            data = ortho.apply_from_reader(
                self._reader, bands=self._bands, nodata=self._nodata
            )

        # 5. Package result
        geo_meta = ortho.get_output_geolocation_metadata()

        return OrthoResult(
            data=data,
            output_grid=grid,
            geolocation_metadata=geo_meta,
            orthorectifier=ortho,
        )

    def _resolve_output_grid(self) -> GeographicGrid:
        """Determine the output grid from explicit or auto-computed settings.

        Returns
        -------
        GeographicGrid
            Resolved output grid.

        Raises
        ------
        ValueError
            If resolution cannot be determined.
        """
        if self._output_grid is not None:
            return self._output_grid

        if self._enu_params is not None:
            from grdl.image_processing.ortho.enu_grid import ENUGrid
            return ENUGrid.from_geolocation(
                self._geolocation, **self._enu_params,
            )

        if (self._pixel_size_lat is not None
                and self._pixel_size_lon is not None):
            psl = self._pixel_size_lat
            psn = self._pixel_size_lon
        else:
            # Auto-compute from metadata
            from grdl.image_processing.ortho.resolution import (
                compute_output_resolution,
            )
            metadata = (
                self._metadata
                or (self._reader.metadata if self._reader else None)
            )
            if metadata is None:
                raise ValueError(
                    "Cannot auto-compute resolution without metadata.  "
                    "Call .with_metadata(), .with_reader(), "
                    ".with_resolution(), or .with_output_grid()."
                )
            psl, psn = compute_output_resolution(
                metadata,
                geolocation=self._geolocation,
                scale_factor=self._scale_factor,
            )
            logger.info(
                "Auto-computed resolution: %.8f deg lat, %.8f deg lon",
                psl, psn,
            )

        if self._roi_bounds is not None:
            min_lat, max_lat, min_lon, max_lon = self._roi_bounds
            return GeographicGrid(min_lat, max_lat, min_lon, max_lon, psl, psn)

        return GeographicGrid.from_geolocation(
            self._geolocation,
            pixel_size_lat=psl,
            pixel_size_lon=psn,
            margin=self._margin,
        )

    def _run_tiled(self) -> OrthoResult:
        """Execute orthorectification in spatial tiles.

        Partitions the output grid into tiles using
        ``grdl.data_prep.Tiler``, then processes each tile
        independently: compute mapping, read source chip, resample,
        and place into the pre-allocated output array.  Mapping memory
        is proportional to tile size, not full output size.

        Returns
        -------
        OrthoResult
            Container with the assembled orthorectified data.
        """
        from grdl.data_prep import Tiler

        # 1. Resolve full output grid
        grid = self._resolve_output_grid()

        # 2. Plan tiles
        tiler = Tiler(
            nrows=grid.rows, ncols=grid.cols,
            tile_size=self._tile_size,
        )
        tiles = tiler.tile_positions()
        logger.info(
            "Tiled processing: %dx%d grid, %d tiles",
            grid.rows, grid.cols, len(tiles),
        )

        # 3. Determine output dtype and shape
        if self._source_array is not None:
            dtype = self._source_array.dtype
            is_multiband = self._source_array.ndim == 3
            n_bands = self._source_array.shape[0] if is_multiband else 0
        else:
            dtype = np.float64
            is_multiband = False
            n_bands = 0

        if is_multiband:
            output = np.full(
                (n_bands, grid.rows, grid.cols),
                self._nodata, dtype=dtype,
            )
        else:
            output = np.full(
                (grid.rows, grid.cols),
                self._nodata, dtype=dtype,
            )

        # 4. Process each tile
        total_tiles = len(tiles)
        last_logged_pct = -1
        for tile_idx, tile in enumerate(tiles):
            sub = grid.sub_grid(
                tile.row_start, tile.col_start,
                tile.row_end, tile.col_end,
            )

            tile_ortho = Orthorectifier(
                geolocation=self._geolocation,
                output_grid=sub,
                interpolation=self._interpolation,
                batch_size=self._batch_size,
            )
            tile_ortho.compute_mapping()

            pct = (tile_idx + 1) * 100 // total_tiles
            if pct // 10 > last_logged_pct // 10:
                logger.debug(
                    "Tile progress: %d/%d (%d%%)",
                    tile_idx + 1, total_tiles, pct,
                )
                last_logged_pct = pct

            if self._source_array is not None:
                tile_data = tile_ortho.apply(
                    self._source_array, nodata=self._nodata,
                )
            else:
                tile_data = tile_ortho.apply_from_reader(
                    self._reader, bands=self._bands,
                    nodata=self._nodata,
                )

            # Place tile into output
            if is_multiband:
                output[
                    :,
                    tile.row_start:tile.row_end,
                    tile.col_start:tile.col_end,
                ] = tile_data
            else:
                output[
                    tile.row_start:tile.row_end,
                    tile.col_start:tile.col_end,
                ] = tile_data

        # 5. Build metadata from full grid
        meta_ortho = Orthorectifier(
            geolocation=self._geolocation,
            output_grid=grid,
            interpolation=self._interpolation,
            batch_size=self._batch_size,
        )
        geo_meta = meta_ortho.get_output_geolocation_metadata()

        return OrthoResult(
            data=output,
            output_grid=grid,
            geolocation_metadata=geo_meta,
            orthorectifier=meta_ortho,
        )


# ── Public function API ──────────────────────────────────────────────


def orthorectify(
    geolocation: 'Geolocation',
    *,
    reader: Optional['ImageReader'] = None,
    source_array: Optional[np.ndarray] = None,
    metadata: Optional[Any] = None,
    output_grid: Optional['GeographicGrid'] = None,
    resolution: Optional[Tuple[float, float]] = None,
    interpolation: str = 'bilinear',
    bands: Optional[List[int]] = None,
    nodata: float = 0.0,
    margin: float = 0.0,
    scale_factor: float = 1.0,
    roi: Optional[Tuple[float, float, float, float]] = None,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    enu_grid: Optional[Dict[str, Any]] = None,
    batch_size: int = 2_000_000,
) -> OrthoResult:
    """Orthorectify imagery to a geographic or ENU grid.

    Convenience function wrapping ``OrthoBuilder`` with keyword arguments.
    Provide either ``reader`` (reads pixels on demand) or ``source_array``
    (pre-loaded data).  Attach a DEM to ``geolocation.elevation`` before
    calling for terrain-corrected projection.

    Parameters
    ----------
    geolocation : Geolocation
        Source image geolocation (required).  Set
        ``geolocation.elevation`` for terrain correction.
    reader : ImageReader, optional
        Open imagery reader.  Mutually exclusive with ``source_array``.
    source_array : np.ndarray, optional
        Pre-loaded source array.  Mutually exclusive with ``reader``.
    metadata : Any, optional
        Reader metadata for auto-resolution (e.g. ``SICDMetadata``).
    output_grid : GeographicGrid or ENUGrid, optional
        Explicit output grid.  Overrides auto-computation.
    resolution : (float, float), optional
        ``(pixel_size_lat, pixel_size_lon)`` in degrees.
    interpolation : str, default='bilinear'
        Resampling method: ``'nearest'``, ``'bilinear'``, ``'bicubic'``,
        ``'lanczos'`` (Lanczos-3, requires numba backend).
    bands : list of int, optional
        Band indices to orthorectify (reader mode only).
    nodata : float, default=0.0
        Fill value for pixels without source coverage.
    margin : float, default=0.0
        Margin around footprint bounds in degrees.
    scale_factor : float, default=1.0
        Multiplier for auto-computed resolution.
    roi : (min_lat, max_lat, min_lon, max_lon), optional
        Restrict output to a geographic region of interest.
    tile_size : int or (int, int), optional
        Enable tiled processing with this tile dimension.
    enu_grid : dict, optional
        ENU grid parameters passed to ``with_enu_grid()``.  Keys:
        ``pixel_size_m``, ``ref_lat``, ``ref_lon``, ``ref_alt``,
        ``margin_m``.

    Returns
    -------
    OrthoResult
        Container with ``data``, ``output_grid``, and
        ``geolocation_metadata``.

    Raises
    ------
    ValueError
        If neither ``reader`` nor ``source_array`` is provided, or if
        resolution cannot be determined.

    Examples
    --------
    From a reader with DEM::

        geo.elevation = dem
        result = orthorectify(
            geolocation=geo,
            reader=reader,
            interpolation='bilinear',
        )

    From a pre-loaded array with explicit grid::

        result = orthorectify(
            geolocation=geo,
            source_array=mag,
            output_grid=enu_grid,
            interpolation='nearest',
            nodata=np.nan,
        )
    """
    builder = OrthoBuilder()

    if reader is not None:
        builder.with_reader(reader)
    if source_array is not None:
        builder.with_source_array(source_array)
    if metadata is not None:
        builder.with_metadata(metadata)

    builder.with_geolocation(geolocation)
    builder.with_interpolation(interpolation)
    builder.with_nodata(nodata)
    builder.with_batch_size(batch_size)

    if output_grid is not None:
        builder.with_output_grid(output_grid)
    if resolution is not None:
        builder.with_resolution(resolution[0], resolution[1])
    if bands is not None:
        builder.with_bands(bands)
    if margin != 0.0:
        builder.with_margin(margin)
    if scale_factor != 1.0:
        builder.with_scale_factor(scale_factor)
    if roi is not None:
        builder.with_roi(roi[0], roi[1], roi[2], roi[3])
    if tile_size is not None:
        builder.with_tile_size(tile_size)
    if enu_grid is not None:
        builder.with_enu_grid(**enu_grid)

    return builder.run()
