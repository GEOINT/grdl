# -*- coding: utf-8 -*-
"""
Open Elevation - Auto-detect and load DEM/DTED from any path.

Factory function that accepts a file or directory path and returns
the appropriate ``ElevationModel`` subclass.  Handles:

- **Single GeoTIFF** (``.tif``, ``.tiff``): Opens directly as ``GeoTIFFDEM``.
- **DTED directory**: Opens as ``DTEDElevation`` if tiles exist.
- **FABDEM / tiled GeoTIFF directory**: Scans recursively for a tile
  matching the query location (e.g., ``S27E029_FABDEM_V1-2.tif``).
- **Constant height**: Falls back to ``ConstantElevation`` if nothing
  else works.

Dependencies
------------
rasterio (for GeoTIFF and DTED backends)

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
2026-03-18

Modified
--------
2026-03-27  Add interpolation parameter, pass through to all DEM backends.
2026-03-18
"""

# Standard library
import logging
from pathlib import Path
from typing import Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.elevation.base import ElevationModel
from grdl.geolocation.elevation.constant import ConstantElevation

logger = logging.getLogger(__name__)


def open_elevation(
    dem_path: str,
    geoid_path: Optional[str] = None,
    location: Optional[Tuple[float, float]] = None,
    fallback_height: float = 0.0,
    interpolation: int = 3,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> ElevationModel:
    """Auto-detect DEM format and return an appropriate ElevationModel.

    Parameters
    ----------
    dem_path : str or Path
        Path to a DEM file or directory.  Supported:

        - Single ``.tif`` / ``.tiff`` file → ``GeoTIFFDEM``
        - Directory of DTED tiles (``*.dt0``, ``*.dt1``, ``*.dt2``) →
          ``DTEDElevation``
        - Directory of tiled GeoTIFFs (e.g., FABDEM) → scans for a
          tile covering ``location``

    geoid_path : str or Path, optional
        Path to a geoid model file (``.pgm`` or ``.tif``) for
        MSL → HAE correction.
    location : tuple of (lat, lon), optional
        Approximate scene center in degrees.  Used to find the correct
        tile when ``dem_path`` is a directory of tiled GeoTIFFs.
    fallback_height : float
        Height (meters HAE) to use if no DEM can be loaded.  Returns
        a ``ConstantElevation`` in this case.
    interpolation : int, default=3
        Spline interpolation order for DEM sampling:

        - ``1`` — bilinear (C0, fast)
        - ``3`` — bicubic (C1, recommended for ortho)
        - ``5`` — quintic (C2, very smooth)

    Returns
    -------
    ElevationModel
        The loaded elevation model.  Never returns ``None``.

    Examples
    --------
    >>> from grdl.geolocation.elevation import open_elevation
    >>> elev = open_elevation('/data/srtm_30m.tif')
    >>> elev.get_elevation(34.05, -118.25)
    432.0

    >>> elev = open_elevation('/data/FABDEM/', location=(-26.09, 29.47))
    >>> elev.get_elevation(-26.09, 29.47)
    1605.3

    >>> elev = open_elevation('/nonexistent/', fallback_height=500.0)
    >>> elev.get_elevation(0, 0)
    500.0
    """
    dem_path = Path(dem_path)

    # ── Single GeoTIFF file ──────────────────────────────────────
    if dem_path.is_file():
        if dem_path.suffix.lower() in ('.tif', '.tiff', '.geotiff'):
            try:
                from grdl.geolocation.elevation.geotiff_dem import GeoTIFFDEM
                model = GeoTIFFDEM(
                    str(dem_path), geoid_path=geoid_path,
                    interpolation=interpolation,
                )
                logger.info("Loaded GeoTIFF DEM: %s", dem_path.name)
                return model
            except Exception as e:
                logger.warning("Failed to open GeoTIFF DEM %s: %s",
                               dem_path, e)
        else:
            logger.warning("Unsupported DEM file format: %s",
                           dem_path.suffix)

    # ── Directory ────────────────────────────────────────────────
    elif dem_path.is_dir():
        # Try DTED first. Pass bbox so the new TiledGeoDTED backend
        # can probe candidate tile paths directly instead of an
        # exhaustive rglob across the archive.
        model = _try_dted(
            dem_path, geoid_path, location, interpolation, bbox=bbox,
        )
        if model is not None:
            return model

        # Try tiled GeoTIFF (FABDEM, Copernicus, etc.)
        model = _try_tiled_geotiff(
            dem_path, geoid_path, location, interpolation,
            bbox=bbox,
        )
        if model is not None:
            return model

        logger.warning("No usable DEM found in %s", dem_path)

    else:
        raise FileNotFoundError(
            f"DEM path does not exist: {dem_path}"
        )

    # ── Fallback ─────────────────────────────────────────────────
    logger.info("Using constant elevation fallback: %.1f m",
                fallback_height)
    return ConstantElevation(height=fallback_height)


def _has_dted_tiles(
    dem_dir: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> bool:
    """Cheap probe: any DTED file anywhere under *dem_dir*?

    When ``bbox`` is supplied, only the cells overlapping the bbox
    (plus a 1-cell halo) are probed — bounded ``Path.exists()`` work
    instead of a recursive scan of the entire archive.

    Without a bbox, falls back to scanning one directory level
    (``e*``/``w*`` parent dirs, then their immediate DTED files).
    ``rglob`` is intentionally avoided: against a 19 000-tile archive
    the recursive walk is the dominant cost.
    """
    if bbox is not None:
        # Import inside to keep optional rasterio dependency at the
        # construction boundary, not the probe.
        from grdl.geolocation.elevation.tiled_geotiff_dted import (
            _candidate_paths,
        )
        min_lon, min_lat, max_lon, max_lat = bbox
        import math as _m
        lon_lo = int(_m.floor(min_lon)) - 1
        lon_hi = int(_m.floor(max_lon)) + 1
        lat_lo = int(_m.floor(min_lat)) - 1
        lat_hi = int(_m.floor(max_lat)) + 1
        for lon in range(lon_lo, lon_hi + 1):
            for lat in range(lat_lo, lat_hi + 1):
                for _ext, candidate in _candidate_paths(dem_dir, lon, lat):
                    if candidate.is_file():
                        return True
        return False

    # No bbox: one-level scan. Matches the layout we actually expect
    # (``<root>/e116/n34.dt2``) without descending into archive
    # subdirectories.
    import os as _os
    import re as _re
    lon_re = _re.compile(r'^[eEwW]\d{3}$')
    try:
        entries = list(_os.scandir(dem_dir))
    except OSError:
        return False
    for entry in entries:
        if not entry.is_dir() or not lon_re.match(entry.name):
            continue
        try:
            for lat_entry in _os.scandir(entry.path):
                if not lat_entry.is_file():
                    continue
                low = lat_entry.name.lower()
                if low.endswith(('.dt2', '.dt1', '.dt0')):
                    return True
        except OSError:
            continue
    return False


def _try_dted(
    dem_dir: Path,
    geoid_path: Optional[str],
    location: Optional[Tuple[float, float]],
    interpolation: int = 3,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[ElevationModel]:
    """Try to open a DTED directory and verify coverage at location.

    Prefers :class:`TiledGeoDTED` (bbox-aware, LRU file handles, no
    rglob).  Falls back to the legacy :class:`DTEDElevation` when the
    new backend is unavailable.  Returns ``None`` when no DTED files
    are present, so callers fall through to the GeoTIFF probe.
    """
    if not _has_dted_tiles(dem_dir, bbox=bbox):
        logger.debug(
            "No DTED tiles under %s — skipping DTED backend",
            dem_dir,
        )
        return None
    # New bbox-aware backend.
    try:
        from grdl.geolocation.elevation.tiled_geotiff_dted import (
            TiledGeoDTED,
        )
        dted: Optional[ElevationModel] = TiledGeoDTED(
            str(dem_dir),
            geoid_path=geoid_path,
            interpolation=interpolation,
            bbox=bbox,
        )
        if getattr(dted, "tile_count", 0) == 0:
            dted = None
    except Exception as exc:
        logger.debug(
            "TiledGeoDTED unavailable for %s (%s); falling back to "
            "DTEDElevation.", dem_dir, exc,
        )
        dted = None
    # Legacy fallback (no bbox support).
    if dted is None:
        try:
            from grdl.geolocation.elevation.dted import DTEDElevation
            legacy = DTEDElevation(
                str(dem_dir), geoid_path=geoid_path,
                interpolation=interpolation,
            )
            if legacy.tile_count == 0:
                logger.debug(
                    "DTED files found under %s but none parsed as "
                    "tiles — layout mismatch, skipping.", dem_dir,
                )
                return None
            dted = legacy
        except Exception:
            return None

    # Verify coverage if location is given.
    if location is not None:
        h = dted.get_elevation(location[0], location[1])
        if np.isnan(h):
            logger.debug(
                "DTED has no coverage at (%.2f, %.2f)",
                location[0], location[1],
            )
            return None

    logger.info("Loaded %s: %s", type(dted).__name__, dem_dir)
    return dted


def _try_tiled_geotiff(
    dem_dir: Path,
    geoid_path: Optional[str],
    location: Optional[Tuple[float, float]],
    interpolation: int = 3,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[ElevationModel]:
    """Scan a directory for GeoTIFF tiles and load as TiledGeoTIFFDEM.

    Uses ``TiledGeoTIFFDEM`` which indexes all tiles in the directory
    and seamlessly queries across tile boundaries with cross-tile
    interpolation.
    """
    # Quick check: any .tif files present?
    tif_files = list(dem_dir.rglob('*.tif'))
    if not tif_files:
        tif_files = list(dem_dir.rglob('*.tiff'))
    if not tif_files:
        return None

    try:
        from grdl.geolocation.elevation.tiled_geotiff_dem import (
            TiledGeoTIFFDEM,
        )
        model = TiledGeoTIFFDEM(
            str(dem_dir), geoid_path=geoid_path,
            interpolation=interpolation,
            bbox=bbox,
        )
        if model.tile_count == 0:
            return None

        # Verify coverage if location given
        if location is not None:
            h = model.get_elevation(location[0], location[1])
            if np.isnan(h):
                logger.debug(
                    "TiledGeoTIFFDEM has no coverage at (%.2f, %.2f)",
                    location[0], location[1])
                return None

        logger.info("Loaded TiledGeoTIFFDEM: %d tiles from %s",
                     model.tile_count, dem_dir)
        return model
    except Exception as e:
        logger.debug("Failed to load TiledGeoTIFFDEM from %s: %s",
                     dem_dir, e)
        return None
