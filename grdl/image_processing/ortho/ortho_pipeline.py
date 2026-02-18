# -*- coding: utf-8 -*-
"""
Orthorectification Pipeline - Universal pipeline for orthorectifying imagery.

Orchestrates reader, geolocation, resolution computation, output grid
construction, terrain-corrected orthorectification, and optional GeoTIFF
output.  Works with any ``ImageReader`` and ``Geolocation`` subclass.

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
2026-02-17

Modified
--------
2026-02-17
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.image_processing.ortho.ortho import Orthorectifier, OutputGrid

if TYPE_CHECKING:
    from grdl.IO.base import ImageReader
    from grdl.geolocation.base import Geolocation
    from grdl.geolocation.elevation.base import ElevationModel


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
    output_grid : OutputGrid
        Grid specification for the output.
    geolocation_metadata : Dict[str, Any]
        CRS, affine transform, bounds, and pixel sizes.
    orthorectifier : Orthorectifier
        The configured orthorectifier (mapping cached for reuse).
    """

    def __init__(
        self,
        data: np.ndarray,
        output_grid: OutputGrid,
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
            raise ImportError(
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


class OrthoPipeline:
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
        from grdl.image_processing.ortho import OrthoPipeline

        with SICDReader('image.nitf') as reader:
            geo = SICDGeolocation.from_reader(reader)
            elev = DTEDElevation('/path/to/dted')
            result = (OrthoPipeline()
                      .with_reader(reader)
                      .with_geolocation(geo)
                      .with_elevation(elev)
                      .run())
            result.save_geotiff('ortho_output.tif')

    Pre-processed source array (SAR magnitude computed in slant range)::

        mag_db = 20.0 * np.log10(np.abs(reader.read_full()) + 1e-10)
        result = (OrthoPipeline()
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
        self._elevation: Optional['ElevationModel'] = None
        self._output_grid: Optional[OutputGrid] = None
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

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def with_reader(self, reader: 'ImageReader') -> 'OrthoPipeline':
        """Set the source imagery reader.

        Parameters
        ----------
        reader : ImageReader
            An open imagery reader.

        Returns
        -------
        OrthoPipeline
            Self for chaining.
        """
        self._reader = reader
        return self

    def with_metadata(self, metadata: Any) -> 'OrthoPipeline':
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
        OrthoPipeline
            Self for chaining.
        """
        self._metadata = metadata
        return self

    def with_geolocation(
        self, geolocation: 'Geolocation'
    ) -> 'OrthoPipeline':
        """Set the source geolocation.

        Parameters
        ----------
        geolocation : Geolocation
            Geolocation for the source imagery.

        Returns
        -------
        OrthoPipeline
            Self for chaining.
        """
        self._geolocation = geolocation
        return self

    def with_elevation(
        self, elevation: 'ElevationModel'
    ) -> 'OrthoPipeline':
        """Set the elevation model for terrain correction.

        Parameters
        ----------
        elevation : ElevationModel
            DEM / DTED elevation model.

        Returns
        -------
        OrthoPipeline
            Self for chaining.
        """
        self._elevation = elevation
        return self

    def with_output_grid(self, grid: OutputGrid) -> 'OrthoPipeline':
        """Set an explicit output grid (overrides auto-computation).

        Parameters
        ----------
        grid : OutputGrid
            Pre-built output grid.

        Returns
        -------
        OrthoPipeline
            Self for chaining.
        """
        self._output_grid = grid
        return self

    def with_resolution(
        self, pixel_size_lat: float, pixel_size_lon: float
    ) -> 'OrthoPipeline':
        """Set explicit output resolution in degrees.

        Parameters
        ----------
        pixel_size_lat : float
            Latitude pixel size (degrees).
        pixel_size_lon : float
            Longitude pixel size (degrees).

        Returns
        -------
        OrthoPipeline
            Self for chaining.
        """
        self._pixel_size_lat = pixel_size_lat
        self._pixel_size_lon = pixel_size_lon
        return self

    def with_interpolation(self, method: str) -> 'OrthoPipeline':
        """Set resampling interpolation method.

        Parameters
        ----------
        method : str
            One of ``'nearest'``, ``'bilinear'``, ``'bicubic'``.

        Returns
        -------
        OrthoPipeline
            Self for chaining.
        """
        self._interpolation = method
        return self

    def with_bands(self, bands: List[int]) -> 'OrthoPipeline':
        """Set which bands to orthorectify (reader mode only).

        Parameters
        ----------
        bands : List[int]
            Band indices.

        Returns
        -------
        OrthoPipeline
            Self for chaining.
        """
        self._bands = bands
        return self

    def with_nodata(self, nodata: float) -> 'OrthoPipeline':
        """Set nodata fill value for pixels without source coverage.

        Parameters
        ----------
        nodata : float
            Fill value.  Default is ``0.0``.

        Returns
        -------
        OrthoPipeline
            Self for chaining.
        """
        self._nodata = nodata
        return self

    def with_margin(self, margin: float) -> 'OrthoPipeline':
        """Set margin around the footprint bounds (degrees).

        Parameters
        ----------
        margin : float
            Margin in degrees added to each side of the footprint.

        Returns
        -------
        OrthoPipeline
            Self for chaining.
        """
        self._margin = margin
        return self

    def with_scale_factor(self, factor: float) -> 'OrthoPipeline':
        """Set resolution scale factor for auto-computed resolution.

        Parameters
        ----------
        factor : float
            Multiplier.  Values > 1.0 produce coarser output.

        Returns
        -------
        OrthoPipeline
            Self for chaining.
        """
        self._scale_factor = factor
        return self

    def with_source_array(self, array: np.ndarray) -> 'OrthoPipeline':
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
        OrthoPipeline
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
    ) -> 'OrthoPipeline':
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
        OrthoPipeline
            Self for chaining.
        """
        self._roi_bounds = (min_lat, max_lat, min_lon, max_lon)
        return self

    def with_tile_size(
        self, tile_size: Union[int, Tuple[int, int]]
    ) -> 'OrthoPipeline':
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
        OrthoPipeline
            Self for chaining.
        """
        self._tile_size = tile_size
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

        # 2. Create orthorectifier with optional DEM
        ortho = Orthorectifier(
            geolocation=self._geolocation,
            output_grid=grid,
            interpolation=self._interpolation,
            elevation=self._elevation,
        )

        # 3. Compute mapping (DEM heights injected here if configured)
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

    def _resolve_output_grid(self) -> OutputGrid:
        """Determine the output grid from explicit or auto-computed settings.

        Returns
        -------
        OutputGrid
            Resolved output grid.

        Raises
        ------
        ValueError
            If resolution cannot be determined.
        """
        if self._output_grid is not None:
            return self._output_grid

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

        if self._roi_bounds is not None:
            min_lat, max_lat, min_lon, max_lon = self._roi_bounds
            return OutputGrid(min_lat, max_lat, min_lon, max_lon, psl, psn)

        return OutputGrid.from_geolocation(
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
        for tile in tiles:
            sub = grid.sub_grid(
                tile.row_start, tile.col_start,
                tile.row_end, tile.col_end,
            )

            tile_ortho = Orthorectifier(
                geolocation=self._geolocation,
                output_grid=sub,
                interpolation=self._interpolation,
                elevation=self._elevation,
            )
            tile_ortho.compute_mapping()

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
            elevation=self._elevation,
        )
        geo_meta = meta_ortho.get_output_geolocation_metadata()

        return OrthoResult(
            data=output,
            output_grid=grid,
            geolocation_metadata=geo_meta,
            orthorectifier=meta_ortho,
        )
