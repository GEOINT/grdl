# -*- coding: utf-8 -*-
"""
Discovery Scanner - Fast metadata-only file and directory scanning.

Provides ``MetadataScanner`` for parallel metadata extraction from imagery
files using GRDL's ``open_any()`` reader cascade. Produces ``ScanResult``
dataclass instances containing full typed metadata objects (preserving
callable polynomials for downstream plotting) alongside JSON-serializable
summary dicts.

Dependencies
------------
grdl.IO
grdl.geolocation

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
2026-03-29

Modified
--------
2026-03-29
"""

# Standard library
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union,
)

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models.base import ImageMetadata

logger = logging.getLogger(__name__)

# Extensions recognized by GRDL readers
_IMAGERY_EXTENSIONS: Set[str] = {
    '.nitf', '.ntf', '.nsf',          # NITF (SICD, SIDD, EO NITF)
    '.tif', '.tiff', '.geotiff',      # GeoTIFF
    '.h5', '.he5', '.hdf5', '.hdf',   # HDF5 (BIOMASS, NISAR, ASTER, VIIRS)
    '.jp2', '.j2k',                   # JPEG2000 (Sentinel-2)
    '.cphd',                          # CPHD
}


@dataclass
class ScanResult:
    """Complete record from scanning a single imagery file.

    Attributes
    ----------
    filepath : Path
        Absolute path to the scanned file.
    format : str
        Format identifier (e.g., ``'SICD'``, ``'GeoTIFF'``).
    rows : int
        Number of image rows.
    cols : int
        Number of image columns.
    dtype : str
        NumPy dtype string (e.g., ``'complex64'``).
    bands : int or None
        Number of spectral bands, if applicable.
    crs : str or None
        Coordinate reference system identifier.
    modality : str or None
        Image modality (``'SAR'``, ``'EO'``, ``'MSI'``, etc.).
    sensor : str or None
        Sensor or platform name.
    datetime : datetime or None
        Acquisition date/time.
    footprint : dict or None
        GeoJSON polygon dict from ``Geolocation.get_footprint()``.
    bounds : tuple or None
        Bounding box as ``(west, south, east, north)`` in degrees.
    metadata_ref : ImageMetadata or None
        The actual typed metadata object (SICDMetadata, etc.) with
        callable polynomials preserved for downstream plotting.
    metadata_dict : dict
        JSON-serializable summary dict for table/tooltip display.
    scan_time_ms : float
        Time taken to scan this file in milliseconds.
    error : str or None
        Error message if scanning failed; ``None`` on success.
    """

    filepath: Path = field(default_factory=Path)
    format: str = ''
    rows: int = 0
    cols: int = 0
    dtype: str = ''
    bands: Optional[int] = None
    crs: Optional[str] = None
    modality: Optional[str] = None
    sensor: Optional[str] = None
    datetime: Optional[datetime] = None
    footprint: Optional[Dict[str, Any]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    metadata_ref: Optional[ImageMetadata] = None
    metadata_dict: Dict[str, Any] = field(default_factory=dict)
    geospatial: Dict[str, Any] = field(default_factory=dict)
    scan_time_ms: float = 0.0
    error: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Returns
        -------
        dict
            All fields except ``metadata_ref`` (not serializable).
        """
        return {
            'filepath': str(self.filepath),
            'format': self.format,
            'rows': self.rows,
            'cols': self.cols,
            'dtype': self.dtype,
            'bands': self.bands,
            'crs': self.crs,
            'modality': self.modality,
            'sensor': self.sensor,
            'datetime': self.datetime.isoformat() if self.datetime else None,
            'footprint': self.footprint,
            'bounds': list(self.bounds) if self.bounds else None,
            'metadata_dict': self.metadata_dict,
            'geospatial': self.geospatial,
            'scan_time_ms': round(self.scan_time_ms, 2),
            'error': self.error,
        }


def extract_modality(meta: 'ImageMetadata') -> Optional[str]:
    """Classify the image modality from a reader's metadata object.

    This is the canonical modality-from-metadata function in grdl.
    It is used internally by :class:`MetadataScanner` and should be
    used by any downstream layer (grdl-runtime, grdk) that needs to
    determine modality from a reader, rather than reimplementing the
    heuristics.

    Classification priority:

    1. Metadata class name — SAR formats (SICD, CPHD, NISAR, etc.)
       and EO/MSI/IR formats are identified by their metadata type.
    2. NumPy dtype — ``complex64`` / ``complex128`` indicates SAR.
    3. Band count — ``>=100`` → HSI, ``>=10`` → MSI, ``3 or 4`` → EO.
    4. ``extras['modality']`` — explicit override stored by readers such
       as :class:`~grdl.IO.generic.GDALFallbackReader`.

    Parameters
    ----------
    meta : ImageMetadata
        Metadata object from any grdl reader.

    Returns
    -------
    str or None
        One of ``'SAR'``, ``'EO'``, ``'IR'``, ``'MSI'``, ``'HSI'``,
        or ``None`` when classification is not possible.
    """
    type_name = type(meta).__name__

    _SAR_TYPES = {
        'SICDMetadata', 'SIDDMetadata', 'CPHDMetadata', 'CRSDMetadata',
        'Sentinel1SLCMetadata', 'BIOMASSMetadata', 'TerraSARMetadata',
        'NISARMetadata',
    }
    if type_name in _SAR_TYPES:
        return 'SAR'

    _EO_TYPES = {'EONITFMetadata', 'Sentinel2Metadata'}
    if type_name in _EO_TYPES:
        return 'EO'

    _IR_TYPES = {'ASTERMetadata'}
    if type_name in _IR_TYPES:
        return 'IR'

    _MSI_TYPES = {'VIIRSMetadata'}
    if type_name in _MSI_TYPES:
        return 'MSI'

    # Heuristic: complex dtype -> SAR
    dtype_str = str(getattr(meta, 'dtype', ''))
    if 'complex' in dtype_str:
        return 'SAR'

    # Band count heuristics
    bands = getattr(meta, 'bands', None)
    if bands is not None:
        if bands >= 100:
            return 'HSI'
        if bands >= 10:
            return 'MSI'
        if bands in (3, 4):
            return 'EO'

    # Explicit override stored in extras by readers like GDALFallbackReader
    extras = getattr(meta, 'extras', {})
    if isinstance(extras, dict):
        modality = extras.get('modality')
        if modality:
            return str(modality)

    return None


class MetadataScanner:
    """Fast metadata-only scanner for imagery files and directories.

    Uses GRDL's ``open_any()`` to detect formats and extract metadata
    without reading pixel data. Optionally calls ``create_geolocation()``
    to compute footprints.

    Parameters
    ----------
    compute_footprints : bool
        Whether to compute geographic footprints via geolocation.
        Default ``True``.
    """

    def __init__(self, compute_footprints: bool = True) -> None:
        self.compute_footprints = compute_footprints

    def scan_file(self, filepath: Union[str, Path]) -> ScanResult:
        """Scan a single file for metadata.

        Opens the file via ``open_any()``, extracts metadata and
        (optionally) computes the geographic footprint, then closes
        the reader immediately.

        Parameters
        ----------
        filepath : str or Path
            Path to the imagery file.

        Returns
        -------
        ScanResult
            Populated result.  If scanning fails, ``error`` is set and
            other fields may be empty.
        """
        filepath = Path(filepath).resolve()
        t0 = time.perf_counter()

        result = ScanResult(filepath=filepath)

        try:
            reader = self._open_reader(filepath)
        except Exception as exc:
            result.error = f"Open failed: {exc}"
            result.scan_time_ms = (time.perf_counter() - t0) * 1000
            return result

        try:
            meta = reader.metadata
            result.format = getattr(meta, 'format', type(meta).__name__)
            result.rows = getattr(meta, 'rows', 0)
            result.cols = getattr(meta, 'cols', 0)
            result.dtype = str(getattr(meta, 'dtype', ''))
            result.bands = getattr(meta, 'bands', None)
            result.crs = getattr(meta, 'crs', None)
            result.metadata_ref = meta
            result.modality = self._extract_modality(meta)
            result.sensor = self._extract_sensor(meta)
            result.datetime = self._extract_datetime(meta)
            result.metadata_dict = self._metadata_to_dict(meta)
            result.geospatial = self._extract_geospatial(meta)

            if self.compute_footprints:
                try:
                    from grdl.geolocation import create_geolocation
                    geo = create_geolocation(reader)
                    fp = geo.get_footprint()
                    if fp is not None:
                        result.footprint = fp
                    bounds = geo.get_bounds()
                    if bounds is not None:
                        result.bounds = tuple(bounds)
                except Exception as exc:
                    logger.debug(
                        "Footprint extraction failed for %s: %s",
                        filepath, exc,
                    )
        except Exception as exc:
            result.error = f"Metadata extraction failed: {exc}"
        finally:
            try:
                reader.close()
            except Exception:
                pass

        result.scan_time_ms = (time.perf_counter() - t0) * 1000
        return result

    def scan_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        extensions: Optional[Set[str]] = None,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int, ScanResult], None]] = None,
    ) -> List[ScanResult]:
        """Scan all imagery files in a directory.

        Parameters
        ----------
        directory : str or Path
            Root directory to scan.
        recursive : bool
            Whether to scan subdirectories.  Default ``True``.
        extensions : set of str, optional
            File extensions to include (lowercase, with dot).
            Defaults to GRDL's recognized imagery extensions.
        max_workers : int
            Number of parallel threads.  Default ``4``.
        progress_callback : callable, optional
            Called as ``callback(completed, total, result)`` after each
            file is scanned.

        Returns
        -------
        List[ScanResult]
            One result per scanned file (including failures).
        """
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Not a directory: {directory}")

        exts = extensions or _IMAGERY_EXTENSIONS
        exts = {e.lower() for e in exts}

        # Collect candidate files and product directories.
        # Product directories (.SAFE, BIOMASS, TerraSAR) are treated as
        # single scan targets — the reader handles internal structure.
        files: List[Path] = []
        skip_prefixes: List[Path] = []  # dirs we've claimed as products

        def _is_product_dir(d: Path) -> bool:
            """Check if a directory is a self-contained imagery product."""
            # Sentinel-1 SAFE
            if d.suffix.upper() == '.SAFE':
                return True
            if (d / 'manifest.safe').exists():
                return True
            # BIOMASS (has measurement/ and annotation/ subdirs)
            if (d / 'measurement').is_dir() and (d / 'annotation').is_dir():
                return True
            # TerraSAR-X (has imageData/ and ANNOTATION/ or annotation/)
            if ((d / 'imageData').is_dir()
                    and ((d / 'ANNOTATION').is_dir()
                         or (d / 'annotation').is_dir())):
                return True
            return False

        # Walk directory tree, collecting product dirs and loose files
        if recursive:
            for p in directory.rglob('*'):
                # Skip anything inside an already-claimed product dir
                if any(p == sd or sd in p.parents for sd in skip_prefixes):
                    continue
                if p.is_dir() and _is_product_dir(p):
                    files.append(p)
                    skip_prefixes.append(p)
                elif p.is_file() and p.suffix.lower() in exts:
                    files.append(p)
        else:
            for p in directory.iterdir():
                if p.is_dir() and _is_product_dir(p):
                    files.append(p)
                elif p.is_file() and p.suffix.lower() in exts:
                    files.append(p)

        total = len(files)
        results: List[ScanResult] = []

        if total == 0:
            return results

        completed = 0

        def _scan_and_report(fp: Path) -> ScanResult:
            return self.scan_file(fp)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_scan_and_report, f): f for f in files}
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total, r)

        return results

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _open_reader(filepath: Path):
        """Open file or product directory via GRDL readers.

        For product directories (.SAFE, BIOMASS, TerraSAR), routes
        directly to ``open_sar()``.  For files, uses ``open_any()``.

        Suppresses stderr noise from NITF parsing libraries (jbpy) that
        dump validation warnings when probing non-NITF files.
        """
        import os
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(2)
            try:
                os.dup2(devnull, 2)
                if filepath.is_dir():
                    # Product directories: try open_sar first (SAFE,
                    # TerraSAR), then open_biomass for BIOMASS dirs
                    reader = None
                    try:
                        from grdl.IO.sar import open_sar
                        reader = open_sar(str(filepath))
                    except Exception:
                        pass
                    if reader is None:
                        try:
                            from grdl.IO.sar import open_biomass
                            reader = open_biomass(str(filepath))
                        except Exception:
                            pass
                    if reader is None:
                        from grdl.IO.generic import open_any
                        reader = open_any(str(filepath))
                else:
                    from grdl.IO.generic import open_any
                    reader = open_any(str(filepath))
            finally:
                os.dup2(old_stderr, 2)
                os.close(devnull)
                os.close(old_stderr)
            return reader

    @staticmethod
    def _extract_modality(meta: ImageMetadata) -> Optional[str]:
        """Classify modality from metadata type.

        Delegates to the module-level :func:`extract_modality`.
        """
        return extract_modality(meta)

    @staticmethod
    def _extract_sensor(meta: ImageMetadata) -> Optional[str]:
        """Extract sensor/platform name from metadata."""
        type_name = type(meta).__name__

        # SICD
        if type_name == 'SICDMetadata':
            ci = getattr(meta, 'collection_info', None)
            if ci is not None:
                return getattr(ci, 'collector_name', None)

        # SIDD
        if type_name == 'SIDDMetadata':
            ef = getattr(meta, 'exploitation_features', None)
            if ef is not None:
                products = getattr(ef, 'products', None)
                if products and len(products) > 0:
                    ci = getattr(products[0], 'collection_info', None)
                    if ci:
                        return getattr(ci, 'sensor_name', None)

        # Sentinel-1
        if type_name == 'Sentinel1SLCMetadata':
            pi = getattr(meta, 'product_info', None)
            if pi is not None:
                return getattr(pi, 'mission_id', None)

        # BIOMASS
        if type_name == 'BIOMASSMetadata':
            return getattr(meta, 'mission', 'BIOMASS')

        # TerraSAR
        if type_name == 'TerraSARMetadata':
            pi = getattr(meta, 'product_info', None)
            if pi is not None:
                return getattr(pi, 'satellite', None)

        # NISAR
        if type_name == 'NISARMetadata':
            ident = getattr(meta, 'identification', None)
            if ident is not None:
                return getattr(ident, 'mission_name', 'NISAR')

        # CPHD
        if type_name == 'CPHDMetadata':
            ci = getattr(meta, 'collection_info', None)
            if ci is not None:
                return getattr(ci, 'collector_name', None)

        # Sentinel-2
        if type_name == 'Sentinel2Metadata':
            pi = getattr(meta, 'product_info', None)
            if pi is not None:
                return getattr(pi, 'spacecraft_name', None)

        return None

    @staticmethod
    def _extract_datetime(meta: ImageMetadata) -> Optional[datetime]:
        """Extract acquisition datetime from metadata."""
        type_name = type(meta).__name__

        dt_str = None

        # SICD
        if type_name == 'SICDMetadata':
            tl = getattr(meta, 'timeline', None)
            if tl is not None:
                dt_str = getattr(tl, 'collect_start', None)

        # SIDD
        elif type_name == 'SIDDMetadata':
            ef = getattr(meta, 'exploitation_features', None)
            if ef is not None:
                products = getattr(ef, 'products', None)
                if products and len(products) > 0:
                    ci = getattr(products[0], 'collection_info', None)
                    if ci:
                        dt_str = getattr(ci, 'collection_date_time', None)

        # Sentinel-1
        elif type_name == 'Sentinel1SLCMetadata':
            pi = getattr(meta, 'product_info', None)
            if pi is not None:
                dt_str = getattr(pi, 'start_time', None)

        # BIOMASS
        elif type_name == 'BIOMASSMetadata':
            dt_str = getattr(meta, 'start_time', None)

        # TerraSAR
        elif type_name == 'TerraSARMetadata':
            pi = getattr(meta, 'product_info', None)
            if pi is not None:
                dt_str = getattr(pi, 'start_time_utc', None)

        # NISAR
        elif type_name == 'NISARMetadata':
            ident = getattr(meta, 'identification', None)
            if ident is not None:
                dt_str = getattr(ident, 'zero_doppler_start_time', None)

        # CPHD
        elif type_name == 'CPHDMetadata':
            gl = getattr(meta, 'global_params', None)
            if gl is not None:
                dt_str = getattr(gl, 'collection_start', None)

        if dt_str is None:
            return None

        return _parse_datetime(dt_str)

    @staticmethod
    def _extract_geospatial(meta: ImageMetadata) -> Dict[str, Any]:
        """Extract all geospatial features from typed metadata.

        Returns a JSON-serializable dict of geospatial data organized
        by category.  Coordinates are in WGS-84 degrees where
        applicable; ECF positions are converted to geodetic.
        """
        geo = {}
        type_name = type(meta).__name__

        # ── SICD ──────────────────────────────────────────────────
        if type_name == 'SICDMetadata':
            # Scene Center Point
            gd = getattr(meta, 'geo_data', None)
            if gd:
                scp = getattr(gd, 'scp', None)
                if scp:
                    llh = getattr(scp, 'llh', None)
                    if llh:
                        geo['scp'] = {
                            'lat': llh.lat, 'lon': llh.lon,
                            'hae': getattr(llh, 'hae', 0.0),
                        }
                    ecf = getattr(scp, 'ecf', None)
                    if ecf:
                        geo['scp_ecef'] = {
                            'x': ecf.x, 'y': ecf.y, 'z': ecf.z,
                        }
                corners = getattr(gd, 'image_corners', None)
                if corners:
                    geo['image_corners'] = [
                        {'lat': c.lat, 'lon': c.lon} for c in corners
                    ]

            # SCPCOA geometry
            scpcoa = getattr(meta, 'scpcoa', None)
            if scpcoa:
                angles = {}
                for attr in [
                    'graze_ang', 'incidence_ang', 'azim_ang',
                    'twist_ang', 'slope_ang', 'layover_ang',
                    'doppler_cone_ang',
                ]:
                    val = getattr(scpcoa, attr, None)
                    if val is not None:
                        angles[attr] = val
                if angles:
                    geo['geometry_angles'] = angles

                sr = getattr(scpcoa, 'slant_range', None)
                gr = getattr(scpcoa, 'ground_range', None)
                if sr is not None:
                    geo['slant_range_m'] = sr
                if gr is not None:
                    geo['ground_range_m'] = gr

                side = getattr(scpcoa, 'side_of_track', None)
                if side:
                    geo['side_of_track'] = side

                # ARP position at SCP time (convert ECF → geodetic)
                arp_pos = getattr(scpcoa, 'arp_pos', None)
                if arp_pos:
                    geo['arp_at_scp'] = _ecef_to_latlon(
                        arp_pos.x, arp_pos.y, arp_pos.z,
                    )

            # ARP trajectory from position polynomial
            pos = getattr(meta, 'position', None)
            tl = getattr(meta, 'timeline', None)
            if pos and tl:
                arp_poly = getattr(pos, 'arp_poly', None)
                duration = getattr(tl, 'collect_duration', None)
                if arp_poly and duration:
                    geo['orbit_track'] = _eval_orbit_track(
                        arp_poly, duration,
                    )

            # Grid metadata
            grid = getattr(meta, 'grid', None)
            if grid:
                grid_info = {
                    'image_plane': getattr(grid, 'image_plane', None),
                    'type': getattr(grid, 'type', None),
                }
                for d in ('row', 'col'):
                    dp = getattr(grid, d, None)
                    if dp:
                        grid_info[f'{d}_ss'] = getattr(dp, 'ss', None)
                        grid_info[f'{d}_imp_resp_wid'] = getattr(
                            dp, 'imp_resp_wid', None,
                        )
                        uvect = getattr(dp, 'uvect_ecf', None)
                        if uvect:
                            grid_info[f'{d}_uvect'] = {
                                'x': uvect.x, 'y': uvect.y, 'z': uvect.z,
                            }
                geo['grid'] = grid_info

            # Beam ground footprint from antenna gain pattern
            # Uses a lightweight ScanResult stub to avoid circular ref
            _stub = ScanResult(
                filepath=Path(''), metadata_ref=meta,
            )
            beam_3db = compute_beam_footprint(_stub, threshold_db=-3.0)
            if beam_3db:
                geo['beam_footprint_3db'] = beam_3db
            beam_10db = compute_beam_footprint(_stub, threshold_db=-10.0)
            if beam_10db:
                geo['beam_footprint_10db'] = beam_10db

        # ── SIDD ──────────────────────────────────────────────────
        elif type_name == 'SIDDMetadata':
            gd = getattr(meta, 'geo_data', None)
            if gd:
                corners = getattr(gd, 'image_corners', None)
                if corners:
                    geo['image_corners'] = [
                        {'lat': c.lat, 'lon': c.lon} for c in corners
                    ]

            meas = getattr(meta, 'measurement', None)
            if meas:
                pp = getattr(meas, 'plane_projection', None)
                if pp:
                    rp = getattr(pp, 'reference_point', None)
                    if rp:
                        ecf = getattr(rp, 'ecef', None)
                        if ecf:
                            geo['reference_point'] = _ecef_to_latlon(
                                ecf.x, ecf.y, ecf.z,
                            )
                    ss = getattr(pp, 'sample_spacing', None)
                    if ss:
                        geo['sample_spacing'] = {
                            'row': getattr(ss, 'row', None),
                            'col': getattr(ss, 'col', None),
                        }
                    prod_plane = getattr(pp, 'product_plane', None)
                    if prod_plane:
                        ruv = getattr(prod_plane, 'row_unit_vector', None)
                        cuv = getattr(prod_plane, 'col_unit_vector', None)
                        if ruv and cuv:
                            geo['product_plane'] = {
                                'row_uvect': {
                                    'x': ruv.x, 'y': ruv.y, 'z': ruv.z,
                                },
                                'col_uvect': {
                                    'x': cuv.x, 'y': cuv.y, 'z': cuv.z,
                                },
                            }

                # ARP trajectory
                arp_poly = getattr(meas, 'arp_poly', None)
                if arp_poly:
                    # SIDD doesn't always have timeline; use 2s default
                    geo['orbit_track'] = _eval_orbit_track(arp_poly, 2.0)

            # Exploitation features — geometry angles
            ef = getattr(meta, 'exploitation_features', None)
            if ef:
                prods = getattr(ef, 'products', None)
                if prods:
                    for prod in prods:
                        cg = getattr(prod, 'collection_geometry', None)
                        if cg:
                            angles = {}
                            for attr in [
                                'azimuth', 'slope', 'squint', 'graze',
                                'tilt', 'doppler_cone_angle',
                            ]:
                                val = getattr(cg, attr, None)
                                if val is not None:
                                    angles[attr] = val
                            if angles:
                                geo['geometry_angles'] = angles
                        phenom = getattr(prod, 'phenomenology', None)
                        if phenom:
                            phen = {}
                            shadow = getattr(phenom, 'shadow', None)
                            layover = getattr(phenom, 'layover', None)
                            if shadow:
                                phen['shadow_angle'] = getattr(
                                    shadow, 'angle', None,
                                )
                                phen['shadow_magnitude'] = getattr(
                                    shadow, 'magnitude', None,
                                )
                            if layover:
                                phen['layover_angle'] = getattr(
                                    layover, 'angle', None,
                                )
                                phen['layover_magnitude'] = getattr(
                                    layover, 'magnitude', None,
                                )
                            mp = getattr(phenom, 'multi_path', None)
                            gt = getattr(phenom, 'ground_track', None)
                            if mp is not None:
                                phen['multi_path'] = mp
                            if gt is not None:
                                phen['ground_track'] = gt
                            if phen:
                                geo['phenomenology'] = phen
                        break  # first product only

        # ── Sentinel-1 SLC ────────────────────────────────────────
        elif type_name == 'Sentinel1SLCMetadata':
            # Dense geolocation grid
            gg = getattr(meta, 'geolocation_grid', None)
            if gg:
                geo['geolocation_grid'] = [
                    {
                        'lat': getattr(g, 'latitude', None),
                        'lon': getattr(g, 'longitude', None),
                        'height': getattr(g, 'height', None),
                        'line': getattr(g, 'line', None),
                        'pixel': getattr(g, 'pixel', None),
                        'incidence_angle': getattr(
                            g, 'incidence_angle', None,
                        ),
                        'elevation_angle': getattr(
                            g, 'elevation_angle', None,
                        ),
                    }
                    for g in gg
                ]

            # Orbit state vectors → ground track
            osv = getattr(meta, 'orbit_state_vectors', None)
            if osv:
                track = []
                for sv in osv:
                    p = sv.position
                    ll = _ecef_to_latlon(p.x, p.y, p.z)
                    ll['time'] = getattr(sv, 'time', None)
                    track.append(ll)
                geo['orbit_track'] = track

            # Burst boundaries
            bursts = getattr(meta, 'bursts', None)
            if bursts:
                geo['bursts'] = [
                    {
                        'index': getattr(b, 'index', i),
                        'azimuth_time': getattr(b, 'azimuth_time', None),
                        'first_line': getattr(b, 'first_line', None),
                        'last_line': getattr(b, 'last_line', None),
                    }
                    for i, b in enumerate(bursts)
                ]

            # Swath geometry
            si = getattr(meta, 'swath_info', None)
            if si:
                geo['geometry_angles'] = {
                    'incidence_angle_mid': getattr(
                        si, 'incidence_angle_mid', None,
                    ),
                }
                geo['swath'] = {
                    'name': getattr(si, 'swath', None),
                    'polarization': getattr(si, 'polarization', None),
                    'range_pixel_spacing': getattr(
                        si, 'range_pixel_spacing', None,
                    ),
                    'azimuth_pixel_spacing': getattr(
                        si, 'azimuth_pixel_spacing', None,
                    ),
                    'radar_frequency': getattr(
                        si, 'radar_frequency', None,
                    ),
                }

        # ── BIOMASS ───────────────────────────────────────────────
        elif type_name == 'BIOMASSMetadata':
            # corner_coords is {name: (lon, lat)}
            corners = getattr(meta, 'corner_coords', None)
            if corners:
                geo['image_corners'] = [
                    {'name': name, 'lon': coord[0], 'lat': coord[1]}
                    for name, coord in corners.items()
                ]

            # gcps are (lat, lon, height, row, col) tuples
            gcps = getattr(meta, 'gcps', None)
            if gcps:
                geo['gcps'] = [
                    {
                        'lat': float(g[0]), 'lon': float(g[1]),
                        'height': float(g[2]),
                        'row': float(g[3]), 'col': float(g[4]),
                    }
                    for g in gcps
                ]

        # ── TerraSAR ─────────────────────────────────────────────
        elif type_name == 'TerraSARMetadata':
            si = getattr(meta, 'scene_info', None)
            if si:
                clat = getattr(si, 'center_lat', None)
                clon = getattr(si, 'center_lon', None)
                if clat is not None and clon is not None:
                    geo['scene_center'] = {'lat': clat, 'lon': clon}
                ext = getattr(si, 'scene_extent', None)
                if ext:
                    geo['image_corners'] = [
                        {
                            'lat': getattr(c, 'lat', None),
                            'lon': getattr(c, 'lon', None),
                            'hae': getattr(c, 'hae', None),
                        }
                        for c in ext
                    ]

            gg = getattr(meta, 'geolocation_grid', None)
            if gg:
                geo['geolocation_grid'] = [
                    {
                        'lat': getattr(g, 'latitude', None),
                        'lon': getattr(g, 'longitude', None),
                        'height': getattr(g, 'height', None),
                        'incidence_angle': getattr(
                            g, 'incidence_angle', None,
                        ),
                    }
                    for g in gg
                ]

            osv = getattr(meta, 'orbit_state_vectors', None)
            if osv:
                track = []
                for sv in osv:
                    p = sv.position
                    ll = _ecef_to_latlon(p.x, p.y, p.z)
                    ll['time'] = getattr(sv, 'time', None)
                    track.append(ll)
                geo['orbit_track'] = track

        # ── NISAR ─────────────────────────────────────────────────
        elif type_name == 'NISARMetadata':
            glg = getattr(meta, 'geolocation_grid', None)
            if glg:
                cx = getattr(glg, 'coordinate_x', None)
                cy = getattr(glg, 'coordinate_y', None)
                if cx is not None and cy is not None:
                    # Subsample dense grids
                    step = max(1, len(cx.ravel()) // 200)
                    geo['geolocation_grid'] = [
                        {'lon': float(cx.ravel()[i]),
                         'lat': float(cy.ravel()[i])}
                        for i in range(0, len(cx.ravel()), step)
                    ]

            orb = getattr(meta, 'orbit', None)
            if orb:
                positions = getattr(orb, 'position', None)
                if positions is not None:
                    track = []
                    step = max(1, len(positions) // 50)
                    for i in range(0, len(positions), step):
                        p = positions[i]
                        track.append(_ecef_to_latlon(
                            float(p[0]), float(p[1]), float(p[2]),
                        ))
                    geo['orbit_track'] = track

        return geo

    @staticmethod
    def _metadata_to_dict(meta: ImageMetadata) -> Dict[str, Any]:
        """Recursively serialize metadata to a JSON-safe dict.

        Converts numpy arrays to lists, Poly objects to stubs,
        XYZ/LatLon to dicts, and datetime objects to ISO strings.
        """
        return _serialize(meta)


# ── Geospatial helpers ────────────────────────────────────────────────


def _ecef_to_latlon(x: float, y: float, z: float) -> Dict[str, float]:
    """Convert a single ECF position to geodetic lat/lon/alt.

    Uses GRDL's coordinate module if available, falls back to a simple
    closed-form approximation.
    """
    try:
        from grdl.geolocation.coordinates import ecef_to_geodetic
        result = ecef_to_geodetic(np.array([[x, y, z]]))
        return {
            'lat': float(result[0, 0]),
            'lon': float(result[0, 1]),
            'alt': float(result[0, 2]),
        }
    except Exception:
        # Simple fallback
        a = 6378137.0
        e2 = 6.6943799901377997e-3
        lon = float(np.degrees(np.arctan2(y, x)))
        p = np.sqrt(x**2 + y**2)
        lat = float(np.degrees(np.arctan2(z, p * (1 - e2))))
        return {'lat': lat, 'lon': lon, 'alt': 0.0}


def _eval_orbit_track(
    arp_poly: Any, duration: float, n_points: int = 30,
) -> List[Dict[str, float]]:
    """Evaluate an XYZPoly over a time interval and convert to geodetic.

    Returns a list of ``{'lat', 'lon', 'alt'}`` dicts for the orbit
    ground track.
    """
    t = np.linspace(0, duration, n_points)
    try:
        positions = arp_poly(t)  # (N, 3) or (3,)
        if positions.ndim == 1:
            positions = positions.reshape(1, 3)
    except Exception:
        return []

    try:
        from grdl.geolocation.coordinates import ecef_to_geodetic
        geo_pts = ecef_to_geodetic(positions)
        return [
            {
                'lat': float(geo_pts[i, 0]),
                'lon': float(geo_pts[i, 1]),
                'alt': float(geo_pts[i, 2]),
            }
            for i in range(len(geo_pts))
        ]
    except Exception:
        return [
            _ecef_to_latlon(
                float(positions[i, 0]),
                float(positions[i, 1]),
                float(positions[i, 2]),
            )
            for i in range(len(positions))
        ]


# ── Beam footprint ────────────────────────────────────────────────────


def compute_beam_footprint(
    metadata_or_result: Union[ScanResult, 'ImageMetadata'],
    threshold_db: float = -3.0,
    n_contour_points: int = 72,
    hae: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Compute the radar beam ground footprint from antenna metadata.

    Projects the antenna gain pattern contour at *threshold_db* through
    the collection geometry onto the WGS-84 surface, producing a GeoJSON
    polygon of the illuminated ground area.

    Requires SICD metadata with populated ``antenna`` and ``scpcoa``
    sections.  Uses the two-way pattern if available, otherwise
    combines tx and rcv gain polynomials.

    Parameters
    ----------
    metadata_or_result : ScanResult or ImageMetadata
        Either a ``ScanResult`` (uses ``metadata_ref``) or a typed
        metadata object directly (e.g., ``SICDMetadata``).  Accepting
        both means you can call this without the scanner::

            from grdl.IO.sar import SICDReader
            reader = SICDReader('scene.nitf')
            beam = compute_beam_footprint(reader.metadata)

    threshold_db : float
        Gain threshold relative to peak for the contour.
        Default ``-3.0`` (half-power beamwidth).
    n_contour_points : int
        Number of points around the contour.  Default 72 (5-degree
        steps).
    hae : float
        Height above ellipsoid for ground projection (meters).
        Default ``0.0``.

    Returns
    -------
    dict or None
        GeoJSON Polygon geometry of the beam footprint on the ground,
        or ``None`` if the required metadata is missing.  Also returns
        ``None`` for non-SICD formats.
    """
    if isinstance(metadata_or_result, ScanResult):
        meta = metadata_or_result.metadata_ref
    else:
        meta = metadata_or_result
    if meta is None or type(meta).__name__ != 'SICDMetadata':
        return None

    ant = getattr(meta, 'antenna', None)
    scpcoa = getattr(meta, 'scpcoa', None)
    geo_data = getattr(meta, 'geo_data', None)
    if ant is None or scpcoa is None or geo_data is None:
        return None

    scp_obj = getattr(geo_data, 'scp', None)
    if scp_obj is None or getattr(scp_obj, 'ecf', None) is None:
        return None

    arp_pos = getattr(scpcoa, 'arp_pos', None)
    arp_vel = getattr(scpcoa, 'arp_vel', None)
    scp_time = getattr(scpcoa, 'scp_time', None)
    if arp_pos is None or arp_vel is None:
        return None

    # Get the gain polynomial — prefer two_way, else combine tx * rcv
    gain_poly = _get_two_way_gain(ant)
    if gain_poly is None:
        return None

    # Get ACF (antenna coordinate frame) axes at SCP time
    acf_x, acf_y = _get_acf_axes(ant, scp_time)
    if acf_x is None or acf_y is None:
        return None

    # ── Step 1: Find -3dB contour in directional cosine space ─────
    contour_dcx, contour_dcy = _find_gain_contour(
        gain_poly, threshold_db, n_contour_points,
    )
    if contour_dcx is None:
        return None

    # ── Step 2: Convert (dcx, dcy) to ECF look vectors ───────────
    # ACF z-axis = x cross y (completes right-hand frame)
    acf_z = np.cross(acf_x, acf_y)
    acf_z /= np.linalg.norm(acf_z)

    # Boresight direction: from ARP to SCP
    arp = np.array([arp_pos.x, arp_pos.y, arp_pos.z])
    scp_ecf = np.array([scp_obj.ecf.x, scp_obj.ecf.y, scp_obj.ecf.z])
    boresight = scp_ecf - arp
    boresight /= np.linalg.norm(boresight)

    # Each contour point is a direction in ACF:
    #   look = dcx * acf_x + dcy * acf_y + sqrt(1 - dcx^2 - dcy^2) * acf_z
    dcz = np.sqrt(np.maximum(
        1.0 - contour_dcx**2 - contour_dcy**2, 0.0,
    ))
    look_ecf = (
        contour_dcx[:, None] * acf_x[None, :]
        + contour_dcy[:, None] * acf_y[None, :]
        + dcz[:, None] * acf_z[None, :]
    )

    # ── Step 3: Intersect look vectors with WGS-84 ellipsoid ─────
    ground_ecf = _intersect_ellipsoid(arp, look_ecf, hae)
    if ground_ecf is None:
        return None

    # ── Step 4: Convert ECF to geodetic ──────────────────────────
    from grdl.geolocation.coordinates import ecef_to_geodetic
    geo_pts = ecef_to_geodetic(ground_ecf)  # (N, 3) [lat, lon, alt]

    lats = geo_pts[:, 0]
    lons = geo_pts[:, 1]

    # Close the ring
    ring = [[float(lons[i]), float(lats[i])] for i in range(len(lats))]
    ring.append(ring[0])

    return {
        'type': 'Polygon',
        'coordinates': [ring],
    }


def _get_two_way_gain(antenna) -> Optional[Any]:
    """Get two-way gain Poly2D, combining tx+rcv if needed."""
    tw = getattr(antenna, 'two_way', None)
    if tw:
        arr = getattr(tw, 'array', None)
        if arr and getattr(arr, 'gain', None) is not None:
            return arr.gain

    # Combine tx and rcv (they're in dB, so add)
    tx = getattr(antenna, 'tx', None)
    rcv = getattr(antenna, 'rcv', None)
    if tx and rcv:
        tx_arr = getattr(tx, 'array', None)
        rcv_arr = getattr(rcv, 'array', None)
        if (tx_arr and getattr(tx_arr, 'gain', None) is not None
                and rcv_arr and getattr(rcv_arr, 'gain', None) is not None):
            return _CombinedGain(tx_arr.gain, rcv_arr.gain)

    # Single-path fallback
    for path in ['tx', 'rcv']:
        ap = getattr(antenna, path, None)
        if ap:
            arr = getattr(ap, 'array', None)
            if arr and getattr(arr, 'gain', None) is not None:
                return arr.gain

    return None


class _CombinedGain:
    """Sum of two gain Poly2D objects (dB addition for two-way)."""

    def __init__(self, tx_gain, rcv_gain):
        self._tx = tx_gain
        self._rcv = rcv_gain

    def __call__(self, x, y):
        return self._tx(x, y) + self._rcv(x, y)


def _get_acf_axes(
    antenna, scp_time: Optional[float],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Get ACF x and y unit vectors at SCP time."""
    t = scp_time if scp_time is not None else 0.0

    for path in ['two_way', 'tx', 'rcv']:
        ap = getattr(antenna, path, None)
        if ap is None:
            continue
        xp = getattr(ap, 'x_axis_poly', None)
        yp = getattr(ap, 'y_axis_poly', None)
        if xp is not None and yp is not None:
            x_vec = xp(t)
            y_vec = yp(t)
            x_vec = x_vec / np.linalg.norm(x_vec)
            y_vec = y_vec / np.linalg.norm(y_vec)
            return x_vec, y_vec

    return None, None


def _find_gain_contour(
    gain_poly,
    threshold_db: float,
    n_points: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Find the contour of a gain polynomial at a given dB level.

    Sweeps angles around boresight and finds the radius at each
    angle where gain drops to *threshold_db* below peak.

    Returns arrays of (dcx, dcy) coordinates on the contour.
    """
    # Evaluate on a fine grid to find peak and extent
    dc_max = 0.2  # search out to 0.2 rad (~11 degrees)
    dc = np.linspace(-dc_max, dc_max, 201)
    DCX, DCY = np.meshgrid(dc, dc)
    G = gain_poly(DCX, DCY)
    peak = float(G.max())
    level = peak + threshold_db  # threshold_db is negative

    # Sweep angles and find radial distance to contour
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radii = np.linspace(0.001, dc_max, 500)
    contour_dcx = np.zeros(n_points)
    contour_dcy = np.zeros(n_points)

    for i, theta in enumerate(angles):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dcx_ray = radii * cos_t
        dcy_ray = radii * sin_t
        g_ray = gain_poly(dcx_ray, dcy_ray)

        # Find where gain crosses the threshold
        below = g_ray < level
        if np.any(below):
            idx = np.argmax(below)
            if idx > 0:
                # Linear interpolation between last-above and first-below
                g0, g1 = float(g_ray[idx - 1]), float(g_ray[idx])
                r0, r1 = float(radii[idx - 1]), float(radii[idx])
                if g0 != g1:
                    frac = (level - g0) / (g1 - g0)
                    r_interp = r0 + frac * (r1 - r0)
                else:
                    r_interp = r0
                contour_dcx[i] = r_interp * cos_t
                contour_dcy[i] = r_interp * sin_t
            else:
                contour_dcx[i] = float(radii[0]) * cos_t
                contour_dcy[i] = float(radii[0]) * sin_t
        else:
            # Gain never drops below threshold in this range
            contour_dcx[i] = dc_max * cos_t
            contour_dcy[i] = dc_max * sin_t

    return contour_dcx, contour_dcy


def _intersect_ellipsoid(
    origin: np.ndarray,
    directions: np.ndarray,
    hae: float = 0.0,
) -> Optional[np.ndarray]:
    """Intersect rays from origin along directions with WGS-84 ellipsoid.

    Parameters
    ----------
    origin : np.ndarray
        Shape ``(3,)`` ECF origin of rays (meters).
    directions : np.ndarray
        Shape ``(N, 3)`` unit direction vectors in ECF.
    hae : float
        Height above ellipsoid (meters).

    Returns
    -------
    np.ndarray or None
        Shape ``(N, 3)`` ECF intersection points, or None on failure.
    """
    # WGS-84 semi-axes + hae
    a = 6378137.0 + hae
    b = 6378137.0 + hae
    c = 6356752.314245179 + hae

    # Normalize directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    d = directions / norms

    # Ray-ellipsoid intersection: solve ||P/abc||^2 = 1
    # P = origin + t * d
    # (ox + t*dx)^2/a^2 + (oy + t*dy)^2/b^2 + (oz + t*dz)^2/c^2 = 1
    ox, oy, oz = origin
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]

    A = dx**2 / a**2 + dy**2 / b**2 + dz**2 / c**2
    B = 2 * (ox * dx / a**2 + oy * dy / b**2 + oz * dz / c**2)
    C = ox**2 / a**2 + oy**2 / b**2 + oz**2 / c**2 - 1.0

    disc = B**2 - 4 * A * C
    valid = disc >= 0
    if not np.any(valid):
        return None

    # Take the nearer intersection (smaller positive t)
    sqrt_disc = np.sqrt(np.maximum(disc, 0))
    t1 = (-B - sqrt_disc) / (2 * A)
    t2 = (-B + sqrt_disc) / (2 * A)

    # Pick the closest positive t
    t = np.where(t1 > 0, t1, t2)

    points = np.column_stack([
        ox + t * dx,
        oy + t * dy,
        oz + t * dz,
    ])

    # Replace invalid intersections with NaN
    points[~valid] = np.nan

    return points


# ── Serialization helpers ─────────────────────────────────────────────


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Best-effort parse of a datetime string or object."""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    # Try common ISO formats
    for fmt in (
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _serialize(obj: Any, depth: int = 0) -> Any:
    """Recursively convert an object to JSON-safe types."""
    if depth > 20:
        return str(obj)

    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        if obj.size > 1000:
            return {
                '_type': 'ndarray',
                'shape': list(obj.shape),
                'dtype': str(obj.dtype),
                'summary': f'array({obj.shape}, {obj.dtype})',
            }
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)

    # GRDL common types
    type_name = type(obj).__name__
    if type_name == 'Poly1D':
        coefs = getattr(obj, 'coefs', None)
        return {
            '_type': 'Poly1D',
            'order': len(coefs) - 1 if coefs is not None else 0,
            'coefs': coefs.tolist() if isinstance(coefs, np.ndarray) else coefs,
        }
    if type_name == 'Poly2D':
        coefs = getattr(obj, 'coefs', None)
        shape = coefs.shape if isinstance(coefs, np.ndarray) else None
        return {
            '_type': 'Poly2D',
            'shape': list(shape) if shape else None,
            'coefs': coefs.tolist() if isinstance(coefs, np.ndarray) else coefs,
        }
    if type_name == 'XYZPoly':
        return {
            '_type': 'XYZPoly',
            'x': _serialize(getattr(obj, 'x', None), depth + 1),
            'y': _serialize(getattr(obj, 'y', None), depth + 1),
            'z': _serialize(getattr(obj, 'z', None), depth + 1),
        }
    if type_name == 'XYZ':
        return {
            'x': getattr(obj, 'x', 0.0),
            'y': getattr(obj, 'y', 0.0),
            'z': getattr(obj, 'z', 0.0),
        }
    if type_name in ('LatLon', 'LatLonHAE'):
        d = {'lat': getattr(obj, 'lat', 0.0), 'lon': getattr(obj, 'lon', 0.0)}
        if hasattr(obj, 'hae'):
            d['hae'] = getattr(obj, 'hae', 0.0)
        return d
    if type_name == 'RowCol':
        return {'row': getattr(obj, 'row', 0), 'col': getattr(obj, 'col', 0)}

    # Lists and tuples
    if isinstance(obj, (list, tuple)):
        return [_serialize(item, depth + 1) for item in obj]

    # Dicts
    if isinstance(obj, dict):
        return {
            str(k): _serialize(v, depth + 1) for k, v in obj.items()
        }

    # Dataclass-like objects (GRDL metadata classes)
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for fname in obj.__dataclass_fields__:
            val = getattr(obj, fname, None)
            if val is not None:
                result[fname] = _serialize(val, depth + 1)
        return result

    # Fallback
    return str(obj)
