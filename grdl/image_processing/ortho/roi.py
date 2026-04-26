# -*- coding: utf-8 -*-
"""
Point-ROI Orthorectification — flat-plane ENU ortho centered on a geographic point.

Provides :func:`orthorectify_point_roi`, a single-call helper that wraps the
full workflow for any GRDL-supported imagery format:

1. Auto-builds geolocation from the reader (SICD, SIDD, RPC, RSM, Affine, …).
2. Resolves a center point (lat/lon or row/col; defaults to image center).
3. Samples an optional DEM at the center and uses a flat constant-elevation
   plane — geographically exact at center, no per-pixel DEM warping.
4. Reads the minimal source chip covering the ROI.
5. Handles complex SAR data via ``complex_mode``.
6. Orthorectifies into a local ENU grid at a configurable pixel size.
7. Returns a :class:`PointRoiResult` with the ortho output, the source chip,
   and the center / chip-bounds metadata needed for display or downstream use.

Callable example
----------------
>>> from grdl.IO.generic import open_any
>>> from grdl.geolocation.elevation import open_elevation
>>> from grdl.geolocation.elevation.constant import ConstantElevation
>>> from grdl.image_processing.ortho import orthorectify_point_roi
>>>
>>> reader = open_any('/data/scene.nitf')
>>> elev   = open_elevation('/data/FABDEM/', location=(34.05, -118.25))
>>> result = orthorectify_point_roi(
...     reader, lat=34.05, lon=-118.25,
...     width_m=500, height_m=500, pixel_size_m=0.5,
...     elevation=elev,
... )
>>> reader.close()
>>> result.data.shape
(1000, 1000)

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-25

Modified
--------
2026-04-25
"""

# Standard library
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation import create_geolocation
from grdl.geolocation.chip import ChipGeolocation
from grdl.geolocation.coordinates import enu_to_geodetic
from grdl.geolocation.elevation.constant import ConstantElevation
from grdl.image_processing.ortho.enu_grid import ENUGrid
from grdl.image_processing.ortho.ortho_builder import OrthoResult, orthorectify
from grdl.image_processing.ortho.resolution import compute_output_resolution

if TYPE_CHECKING:
    from grdl.IO.base import ImageReader
    from grdl.geolocation.elevation.base import ElevationModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PointRoiResult:
    """Result of :func:`orthorectify_point_roi`.

    Parameters
    ----------
    ortho : OrthoResult
        Full orthorectification result: ``data``, ``output_grid``,
        ``save_geotiff()``, etc.
    source_chip : np.ndarray
        Display-ready source chip in image space.  Real-valued: magnitude
        when ``complex_mode='magnitude'``, otherwise the selected band
        as-is.  Shape ``(rows, cols)``.
    center_lat : float
        Scene-center latitude in degrees.
    center_lon : float
        Scene-center longitude in degrees.
    center_row : float
        Scene-center row in full-image pixel coordinates.
    center_col : float
        Scene-center column in full-image pixel coordinates.
    src_r0, src_r1 : int
        Row slice bounds of the source chip in full-image coordinates.
    src_c0, src_c1 : int
        Column slice bounds of the source chip in full-image coordinates.
    """

    ortho: OrthoResult
    source_chip: np.ndarray
    center_lat: float
    center_lon: float
    center_row: float
    center_col: float
    src_r0: int
    src_r1: int
    src_c0: int
    src_c1: int

    # ------------------------------------------------------------------
    # Convenience delegates so callers don't need to go through .ortho
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Orthorectified pixel array (same as ``ortho.data``)."""
        return self.ortho.data

    @property
    def grid(self) -> ENUGrid:
        """ENU output grid (same as ``ortho.output_grid``)."""
        return self.ortho.output_grid

    def save_geotiff(self, filepath: Union[str, Path]) -> None:
        """Write the orthorectified result to a GeoTIFF file."""
        self.ortho.save_geotiff(str(filepath))


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def orthorectify_point_roi(
    reader: 'ImageReader',
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    width_m: float = 500.0,
    height_m: float = 500.0,
    pixel_size_m: Optional[float] = None,
    interpolation: str = 'lanczos',
    band: int = 0,
    complex_mode: str = 'magnitude',
    elevation: Optional['ElevationModel'] = None,
    nodata: float = float('nan'),
) -> PointRoiResult:
    """Orthorectify an NxM meter area centered on a geographic or pixel point.

    Works with any imagery format supported by ``grdl.IO`` — SAR (SICD,
    SIDD, Sentinel-1 SLC, NISAR, …), EO NITF (RPC/RSM), GeoTIFF, and
    more.  The geolocation type is auto-detected from ``reader``.

    The output is a flat-plane ENU ortho: terrain height is sampled once
    at the center point (from ``elevation`` if supplied, otherwise the
    geolocation default HAE) and used as a constant-elevation plane for
    the entire ROI.  This preserves geographic accuracy at the center
    without per-pixel DEM warping.

    The reader is **not** closed by this function; the caller is
    responsible for opening and closing it.

    Parameters
    ----------
    reader : ImageReader
        Open GRDL image reader for the source scene.
    lat : float, optional
        Scene-center latitude in degrees.  Takes precedence over ``row``
        when both are supplied.
    lon : float, optional
        Scene-center longitude in degrees.  Required when ``lat`` is set.
    row : int, optional
        Scene-center row in full-image pixel coordinates.  Used when
        ``lat``/``lon`` are not supplied.
    col : int, optional
        Scene-center column in full-image pixel coordinates.
    width_m : float, default=500.0
        East-west ROI extent in meters.
    height_m : float, default=500.0
        North-south ROI extent in meters.
    pixel_size_m : float, optional
        Output pixel size in meters.  ``None`` → auto-detected from
        imagery metadata via ``compute_output_resolution``.
    interpolation : str, default='lanczos'
        Resampling method: ``'nearest'``, ``'bilinear'``, ``'bicubic'``,
        ``'lanczos'`` (Lanczos-3, requires numba backend).
    band : int, default=0
        0-based band index to extract for single-band output.
    complex_mode : str, default='magnitude'
        How to handle complex-valued SAR pixels:

        - ``'magnitude'`` — take ``|z|`` before interpolation (avoids
          phase-cancellation artefacts, recommended for display).
        - ``'complex'`` — orthorectify real and imaginary channels
          together, preserving phase.
    elevation : ElevationModel, optional
        Elevation model used to obtain terrain height at the center point.
        The height is applied as a constant plane (flat-terrain assumption)
        across the full ROI.  Any ``ElevationModel`` subclass is accepted
        (``GeoTIFFDEM``, ``TiledGeoTIFFDEM``, ``DTEDElevation``,
        ``ConstantElevation``, …).  Pass ``None`` to use the geolocation
        default height (SCP HAE for SICD, reference point HAE for SIDD,
        etc.).
    nodata : float, default=NaN
        Fill value for output pixels with no source coverage.

    Returns
    -------
    PointRoiResult
        Container with:

        - ``data`` / ``ortho.data`` — orthorectified pixel array
        - ``source_chip`` — display-ready source chip (image space)
        - ``center_lat``, ``center_lon``, ``center_row``, ``center_col``
        - ``src_r0``, ``src_r1``, ``src_c0``, ``src_c1`` — chip bounds
        - ``grid`` — ``ENUGrid`` (ENU bounds and pixel sizes)
        - ``save_geotiff(path)`` — write result to GeoTIFF

    Raises
    ------
    ValueError
        If the center point is outside the image bounds, or if the ROI
        has less than 1% overlap with the image.
    TypeError
        If ``geolocation`` cannot be auto-created from ``reader``.

    Examples
    --------
    >>> reader = open_any('/data/scene.nitf')
    >>> result = orthorectify_point_roi(
    ...     reader, lat=34.05, lon=-118.25,
    ...     width_m=500, height_m=500, pixel_size_m=0.5,
    ... )
    >>> reader.close()
    >>> result.data.shape
    (1000, 1000)
    """
    if complex_mode not in ('magnitude', 'complex'):
        raise ValueError(
            f"complex_mode must be 'magnitude' or 'complex', got {complex_mode!r}"
        )

    meta  = reader.metadata
    nrows = int(meta.rows)
    ncols = int(meta.cols)

    geo = create_geolocation(reader)

    # ── Resolve center point ─────────────────────────────────────────
    if lat is not None and lon is not None:
        center_row, center_col, center_lat, center_lon, center_h = (
            _resolve_latlon(geo, lat, lon, nrows, ncols)
        )
    elif row is not None or col is not None:
        r = int(row) if row is not None else nrows // 2
        c = int(col) if col is not None else ncols // 2
        center_row, center_col, center_lat, center_lon, center_h = (
            _resolve_pixel(geo, r, c, nrows, ncols)
        )
    else:
        center_row, center_col, center_lat, center_lon, center_h = (
            _resolve_pixel(geo, nrows // 2, ncols // 2, nrows, ncols)
        )

    # ── Flat-plane elevation ─────────────────────────────────────────
    if elevation is not None:
        queried   = elevation.get_elevation(center_lat, center_lon)
        terrain_h = float(np.asarray(queried).ravel()[0])
        if not np.isfinite(terrain_h):
            logger.warning(
                "DEM returned NaN at (%.5f, %.5f) — using geolocation "
                "default height %.1f m",
                center_lat, center_lon, center_h,
            )
            terrain_h = center_h
        else:
            logger.debug(
                "DEM height at center: %.1f m HAE", terrain_h,
            )
    else:
        terrain_h = center_h

    geo.elevation = ConstantElevation(height=terrain_h)
    center_h = terrain_h

    # ── Coverage check ───────────────────────────────────────────────
    coverage = _roi_coverage(
        geo, center_lat, center_lon, center_h,
        width_m, height_m, nrows, ncols,
    )
    if coverage < 0.01:
        raise ValueError(
            f"ROI has {100.0 * coverage:.1f}% overlap with the image "
            f"(< 1%).  Move the center point or reduce the ROI size."
        )
    if coverage < 1.0:
        logger.warning(
            "ROI has %.1f%% overlap — out-of-coverage pixels will be "
            "nodata-filled.",
            100.0 * coverage,
        )

    # ── Output pixel size ────────────────────────────────────────────
    if pixel_size_m is None:
        pixel_size_m = _auto_pixel_size_m(meta, geo)

    # ── Build ENU output grid ────────────────────────────────────────
    grid = ENUGrid(
        ref_lat=center_lat,
        ref_lon=center_lon,
        ref_alt=center_h,
        min_east=-width_m  / 2.0,
        max_east= width_m  / 2.0,
        min_north=-height_m / 2.0,
        max_north= height_m / 2.0,
        pixel_size_east=pixel_size_m,
        pixel_size_north=pixel_size_m,
    )

    # ── Minimal source chip ──────────────────────────────────────────
    corners_enu = np.array([
        [-width_m / 2, -height_m / 2, 0.0],
        [ width_m / 2, -height_m / 2, 0.0],
        [-width_m / 2,  height_m / 2, 0.0],
        [ width_m / 2,  height_m / 2, 0.0],
    ])
    corner_geo = enu_to_geodetic(
        corners_enu, np.array([center_lat, center_lon, center_h])
    )
    corner_px = geo.latlon_to_image(corner_geo)   # (4, 2)
    src_r0 = max(0,     int(np.floor(corner_px[:, 0].min())))
    src_r1 = min(nrows, int(np.ceil( corner_px[:, 0].max())))
    src_c0 = max(0,     int(np.floor(corner_px[:, 1].min())))
    src_c1 = min(ncols, int(np.ceil( corner_px[:, 1].max())))

    chip = reader.read_chip(src_r0, src_r1, src_c0, src_c1)

    # ── Complex preprocessing ────────────────────────────────────────
    is_complex = np.iscomplexobj(chip)

    if is_complex and complex_mode == 'magnitude':
        band_chip  = chip[band] if chip.ndim == 3 else chip
        source     = np.abs(band_chip).astype(np.float32)
        src_display = source
    elif is_complex:
        band_chip   = chip[band] if chip.ndim == 3 else chip
        source      = band_chip.astype(np.complex64)
        src_display = np.abs(source).astype(np.float32)
    else:
        source = chip
        if source.ndim == 3:
            src_display = source[min(band, source.shape[0] - 1)]
        else:
            src_display = source

    # ── Orthorectify ─────────────────────────────────────────────────
    chip_geo = ChipGeolocation(
        geo,
        row_offset=src_r0,
        col_offset=src_c0,
        shape=(src_r1 - src_r0, src_c1 - src_c0),
    )
    ortho_result = orthorectify(
        geolocation=chip_geo,
        source_array=source,
        output_grid=grid,
        interpolation=interpolation,
        nodata=nodata,
    )

    return PointRoiResult(
        ortho=ortho_result,
        source_chip=src_display,
        center_lat=center_lat,
        center_lon=center_lon,
        center_row=float(center_row),
        center_col=float(center_col),
        src_r0=src_r0,
        src_r1=src_r1,
        src_c0=src_c0,
        src_c1=src_c1,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_latlon(
    geo,
    lat: float,
    lon: float,
    nrows: int,
    ncols: int,
) -> tuple:
    """Map lat/lon → pixel; raise ValueError if outside image bounds."""
    rc = geo.latlon_to_image(lat, lon)
    center_row = float(rc[0])
    center_col = float(rc[1])

    if (center_row < 0 or center_row >= nrows
            or center_col < 0 or center_col >= ncols):
        raise ValueError(
            f"({lat:.6f}°, {lon:.6f}°) maps to pixel "
            f"({center_row:.1f}, {center_col:.1f}), outside "
            f"image bounds [0:{nrows}, 0:{ncols}]."
        )

    llh = geo.image_to_latlon(center_row, center_col)
    center_h = float(np.asarray(llh).ravel()[2])
    return center_row, center_col, lat, lon, center_h


def _resolve_pixel(
    geo,
    row: int,
    col: int,
    nrows: int,
    ncols: int,
) -> tuple:
    """Validate pixel center; resolve its geographic coordinates."""
    if row < 0 or row >= nrows or col < 0 or col >= ncols:
        raise ValueError(
            f"Pixel ({row}, {col}) is outside image bounds "
            f"[0:{nrows}, 0:{ncols}]."
        )

    llh = geo.image_to_latlon(float(row), float(col))
    arr = np.asarray(llh).ravel()
    return float(row), float(col), float(arr[0]), float(arr[1]), float(arr[2])


def _roi_coverage(
    geo,
    center_lat: float,
    center_lon: float,
    center_h: float,
    width_m: float,
    height_m: float,
    nrows: int,
    ncols: int,
    n_samples: int = 5,
) -> float:
    """Fraction of the NxM ROI that falls inside the image (0.0–1.0)."""
    ee, nn = np.meshgrid(
        np.linspace(-width_m  / 2.0, width_m  / 2.0, n_samples),
        np.linspace(-height_m / 2.0, height_m / 2.0, n_samples),
    )
    enu_pts = np.column_stack([ee.ravel(), nn.ravel(), np.zeros(ee.size)])
    geo_pts = enu_to_geodetic(enu_pts, np.array([center_lat, center_lon, center_h]))
    px      = geo.latlon_to_image(geo_pts[:, 0], geo_pts[:, 1], center_h)
    inside  = (
        (px[:, 0] >= 0) & (px[:, 0] < nrows) &
        (px[:, 1] >= 0) & (px[:, 1] < ncols)
    )
    return float(np.sum(inside)) / inside.size


def _auto_pixel_size_m(meta, geo) -> float:
    """Auto-detect native pixel size in meters; fall back to 1.0 m."""
    try:
        deg_lat, _ = compute_output_resolution(meta, geolocation=geo)
        return float(deg_lat) * 111_320.0
    except Exception:
        return 1.0
