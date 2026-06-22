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
2026-06-08  Layout-agnostic DTED probe (standard + nested archive) and
            bbox-aware DTEDElevation loader in _try_dted.
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
        # Try DTED first. Pass bbox so DTEDElevation can probe candidate
        # tile paths directly — covering standard and nested archive
        # layouts — instead of an exhaustive rglob across the archive.
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


def _has_dted_files(
    dem_dir: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> bool:
    """Layout-agnostic probe: any DTED file discoverable under *dem_dir*?

    Recognizes both the standard layout (``<root>/<lon>/<lat>.dt?``) and
    nested archive layouts (``<root>/dted/dted{2,1,0}/<lon>/<lat>.dt?``).

    When ``bbox`` is supplied, only the cells overlapping the bbox (plus
    a 1-cell halo) are probed via the same candidate-path generator the
    loader uses — bounded ``Path.is_file()`` work instead of a recursive
    scan of the entire archive.

    Without a bbox, a shallow scan of both layout roots is used (the
    longitude directories directly under the root, and those under
    ``<root>/dted/dted{2,1,0}``). ``rglob`` is intentionally avoided:
    against a 19 000-tile archive the recursive walk is the dominant
    cost.
    """
    # Import inside to keep the optional rasterio dependency at the
    # construction boundary, not the probe.
    from grdl.geolocation.elevation.dted import (
        _CaseInsensitiveDirCache,
        _find_dted_file,
    )

    if bbox is not None:
        # Shares the loader's OS-independent, cached resolution so the
        # probe agrees with discovery on case-sensitive filesystems.
        min_lon, min_lat, max_lon, max_lat = bbox
        import math as _m
        lon_lo = int(_m.floor(min_lon)) - 1
        lon_hi = int(_m.floor(max_lon)) + 1
        lat_lo = int(_m.floor(min_lat)) - 1
        lat_hi = int(_m.floor(max_lat)) + 1
        dir_cache = _CaseInsensitiveDirCache()
        for lon in range(lon_lo, lon_hi + 1):
            for lat in range(lat_lo, lat_hi + 1):
                if _find_dted_file(dem_dir, lon, lat, dir_cache) is not None:
                    return True
        return False

    # No bbox: shallow, case-insensitive scan of both layout roots.
    import os as _os
    import re as _re
    lon_re = _re.compile(r'^[ew]\d{3}$', _re.IGNORECASE)
    res_re = _re.compile(r'^dted$', _re.IGNORECASE)
    res_sub_re = _re.compile(r'^dted[012]$', _re.IGNORECASE)

    def _lon_dir_has_tile(lon_dir_path: str) -> bool:
        try:
            with _os.scandir(lon_dir_path) as it:
                for f in it:
                    if (
                        f.is_file()
                        and f.name.lower().endswith(('.dt2', '.dt1', '.dt0'))
                    ):
                        return True
        except OSError:
            return False
        return False

    def _has_lon_dir_with_tile(parent: str) -> bool:
        try:
            with _os.scandir(parent) as it:
                for entry in it:
                    if (
                        entry.is_dir()
                        and lon_re.match(entry.name)
                        and _lon_dir_has_tile(entry.path)
                    ):
                        return True
        except OSError:
            return False
        return False

    dem_dir_str = str(dem_dir)
    # Standard layout: <root>/<lon_dir>/...
    if _has_lon_dir_with_tile(dem_dir_str):
        return True
    # Nested archive layout: <root>/dted/dted{0,1,2}/<lon_dir>/... with
    # case-insensitive matching on the 'dted' and 'dted{N}' directories.
    try:
        with _os.scandir(dem_dir_str) as it:
            dted_dirs = [
                e.path for e in it if e.is_dir() and res_re.match(e.name)
            ]
    except OSError:
        dted_dirs = []
    for dted_dir in dted_dirs:
        try:
            with _os.scandir(dted_dir) as it:
                res_dirs = [
                    e.path for e in it
                    if e.is_dir() and res_sub_re.match(e.name)
                ]
        except OSError:
            continue
        for res_dir in res_dirs:
            if _has_lon_dir_with_tile(res_dir):
                return True
    return False


def _try_dted(
    dem_dir: Path,
    geoid_path: Optional[str],
    location: Optional[Tuple[float, float]],
    interpolation: int = 3,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[ElevationModel]:
    """Try to open a DTED directory and verify coverage at location.

    Prefers :class:`TiledGeoDTED` (bbox-aware, LRU file handles, and a
    pickle-light path-only index for multi-worker pipelines). Falls back
    to the bbox-aware :class:`DTEDElevation` if the tiled backend is
    unavailable. Both probe the standard layout and nested archive
    layouts (``<root>/dted/dted{2,1,0}/<lon>/<lat>.dt?``) via the shared,
    OS-independent discovery. Returns ``None`` when no DTED files are
    present (so callers fall through to the GeoTIFF probe), when no tiles
    index, or when a supplied ``location`` has no coverage.
    """
    if not _has_dted_files(dem_dir, bbox=bbox):
        logger.debug(
            "No DTED tiles under %s — skipping DTED backend",
            dem_dir,
        )
        return None

    dted: Optional[ElevationModel] = None

    # Preferred backend: TiledGeoDTED (fast, pickle-light, LRU handles).
    try:
        from grdl.geolocation.elevation.tiled_geotiff_dted import (
            TiledGeoDTED,
        )
        candidate = TiledGeoDTED(
            str(dem_dir),
            geoid_path=geoid_path,
            interpolation=interpolation,
            bbox=bbox,
        )
        if getattr(candidate, "tile_count", 0) > 0:
            dted = candidate
    except Exception as exc:
        logger.debug(
            "TiledGeoDTED unavailable for %s (%s); falling back to "
            "DTEDElevation.", dem_dir, exc,
        )

    # Fallback backend: bbox-aware DTEDElevation.
    if dted is None:
        try:
            from grdl.geolocation.elevation.dted import DTEDElevation
            legacy = DTEDElevation(
                str(dem_dir),
                geoid_path=geoid_path,
                interpolation=interpolation,
                bbox=bbox,
            )
        except Exception as exc:
            logger.debug(
                "Failed to construct DTEDElevation for %s: %s", dem_dir, exc,
            )
            return None
        if legacy.tile_count == 0:
            logger.debug(
                "DTED files found under %s but none indexed as tiles — "
                "layout mismatch, skipping.", dem_dir,
            )
            return None
        dted = legacy

    # Verify coverage if location is given.
    if location is not None:
        h = dted.get_elevation(location[0], location[1])
        if np.isnan(h):
            logger.debug(
                "DTED has no coverage at (%.2f, %.2f)",
                location[0], location[1],
            )
            return None

    logger.info(
        "Loaded %s: %d tile(s) from %s",
        type(dted).__name__, dted.tile_count, dem_dir,
    )
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
