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
duane.d.smalley@gmail.com

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
2026-03-18
"""

# Standard library
import glob
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
                model = GeoTIFFDEM(str(dem_path), geoid_path=geoid_path)
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
        # Try DTED first
        model = _try_dted(dem_path, geoid_path, location)
        if model is not None:
            return model

        # Try tiled GeoTIFF (FABDEM, Copernicus, etc.)
        model = _try_tiled_geotiff(dem_path, geoid_path, location)
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


def _try_dted(
    dem_dir: Path,
    geoid_path: Optional[str],
    location: Optional[Tuple[float, float]],
) -> Optional[ElevationModel]:
    """Try to open a DTED directory and verify coverage at location."""
    try:
        from grdl.geolocation.elevation.dted import DTEDElevation
        dted = DTEDElevation(str(dem_dir), geoid_path=geoid_path)

        # Verify coverage if location is given
        if location is not None:
            h = dted.get_elevation(location[0], location[1])
            if np.isnan(h):
                logger.debug("DTED has no coverage at (%.2f, %.2f)",
                             location[0], location[1])
                return None

        logger.info("Loaded DTED: %s", dem_dir)
        return dted
    except Exception:
        return None


def _try_tiled_geotiff(
    dem_dir: Path,
    geoid_path: Optional[str],
    location: Optional[Tuple[float, float]],
) -> Optional[ElevationModel]:
    """Scan a directory for a GeoTIFF tile covering the location."""
    if location is None:
        logger.debug("No location given — cannot find tiled GeoTIFF")
        return None

    lat, lon = location
    lat_i = int(np.floor(lat))
    lon_i = int(np.floor(lon))

    # Build search patterns for common tile naming conventions:
    # FABDEM: S27E029_FABDEM_V1-2.tif
    # SRTM:  S27E029.tif, S27E029.hgt
    # Copernicus: Copernicus_DSM_10_S27_00_E029_00.tif
    ns = 'S' if lat_i < 0 else 'N'
    ew = 'W' if lon_i < 0 else 'E'
    lat_abs = abs(lat_i)
    lon_abs = abs(lon_i)

    patterns = [
        f"**/{ns}{lat_abs:02d}{ew}{lon_abs:03d}*.tif",
        f"**/{ns}{lat_abs:02d}{ew}{lon_abs:03d}*.hgt",
        f"**/Copernicus*{ns}{lat_abs:02d}*{ew}{lon_abs:03d}*.tif",
    ]

    for pattern in patterns:
        matches = sorted(glob.glob(str(dem_dir / pattern), recursive=True))
        if matches:
            try:
                from grdl.geolocation.elevation.geotiff_dem import GeoTIFFDEM
                model = GeoTIFFDEM(matches[0], geoid_path=geoid_path)
                logger.info("Loaded tiled DEM: %s", Path(matches[0]).name)
                return model
            except Exception as e:
                logger.debug("Failed to open %s: %s", matches[0], e)

    logger.debug("No tiled GeoTIFF for (%d, %d) in %s",
                 lat_i, lon_i, dem_dir)
    return None
