# -*- coding: utf-8 -*-
"""
Sentinel-1 SLC Catalog - Discovery, indexing, remote query, and download
for Sentinel-1 IW/EW/SM SLC SAFE products.

Provides tools for discovering Sentinel-1 SLC products on disk, querying the
Copernicus Data Space Ecosystem (CDSE) STAC catalog, downloading products
with OAuth2 authentication, and tracking collections in a local SQLite
database. Supports offline filtering by time, orbit, polarization, and swath.

Dependencies
------------
rasterio (via Sentinel1SLCReader)
requests (for CDSE remote queries and downloads)

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-11

Modified
--------
2026-03-13
"""

# Standard library
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import re
import sqlite3
import warnings

# Third-party
try:
    import requests
except ImportError:
    pass

# GRDL internal
from grdl.IO.base import CatalogInterface
from grdl.IO.catalog.remote_utils import (
    REQUESTS_AVAILABLE,
    _normalize_stac_datetime,
    download_file,
    get_cdse_token,
    load_credentials,
)
from grdl.exceptions import DependencyError, ProcessorError, ValidationError

logger = logging.getLogger(__name__)


class Sentinel1SLCCatalog(CatalogInterface):
    """Catalog and download manager for Sentinel-1 IW/EW/SM SLC SAFE products.

    Provides integrated capabilities for:

    - Local file system discovery of Sentinel-1 SLC products
    - Remote querying via Copernicus Data Space Ecosystem (CDSE) STAC
    - Product download with CDSE OAuth2 authentication
    - SQLite database tracking of products

    Credentials are loaded from ``~/.config/geoint/credentials.json``
    (provider block ``esa_copernicus``).

    Attributes
    ----------
    search_path : Path
        Root directory for local product searches and downloads.
    db_path : Path
        Path to SQLite database file.
    conn : sqlite3.Connection
        Database connection.

    Examples
    --------
    >>> from grdl.IO.catalog import Sentinel1SLCCatalog
    >>> catalog = Sentinel1SLCCatalog('/data/sentinel1')
    >>> products = catalog.discover_local()
    >>> print(f"Found {len(products)} products")
    >>>
    >>> # Search CDSE for ascending products
    >>> remote = catalog.query_cdse(
    ...     bbox=(12.0, 41.0, 13.0, 42.0),
    ...     orbit_pass='ASCENDING',
    ...     max_results=10,
    ... )
    >>> catalog.download_product(remote[0]['id'])
    """

    # Copernicus Data Space Ecosystem OData API
    CDSE_ODATA_URL = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    )
    CDSE_DOWNLOAD_URL = (
        "https://zipper.dataspace.copernicus.eu/odata/v1/Products"
    )
    CDSE_COLLECTION = "SENTINEL-1"

    def __init__(
        self,
        search_path: Union[str, Path],
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize Sentinel-1 SLC catalog.

        Parameters
        ----------
        search_path : Union[str, Path]
            Root directory for local product searches.
        db_path : Optional[Union[str, Path]], default=None
            Path to SQLite database file. If None, uses
            ``search_path/sentinel1_catalog.db``.

        Raises
        ------
        NotADirectoryError
            If search_path is not a directory.
        """
        super().__init__(search_path)

        if db_path is None:
            db_path = Path.home() / ".config" / "geoint" / "catalogs" / "sentinel1_slc.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for product tracking."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                product_name TEXT UNIQUE NOT NULL,
                product_type TEXT,
                mission TEXT,
                mode TEXT,
                polarization TEXT,
                absolute_orbit INTEGER,
                relative_orbit INTEGER,
                orbit_pass TEXT,
                start_time TEXT,
                stop_time TEXT,
                swath TEXT,
                corner_coords TEXT,
                local_path TEXT,
                remote_url TEXT,
                file_size INTEGER,
                download_date TEXT,
                metadata_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_start_time "
            "ON products(start_time)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_absolute_orbit "
            "ON products(absolute_orbit)"
        )

        self.conn.commit()

        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add columns introduced after the initial schema."""
        cursor = self.conn.cursor()
        for col in ("remote_url TEXT",):
            try:
                cursor.execute(f"ALTER TABLE products ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass  # column already exists
        self.conn.commit()

    def discover_images(
        self,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Path]:
        """Discover Sentinel-1 SLC SAFE directories in the search path.

        Parameters
        ----------
        extensions : Optional[List[str]], default=None
            Not used for Sentinel-1 (SAFE directories, not files).
        recursive : bool, default=True
            Whether to search subdirectories recursively.

        Returns
        -------
        List[Path]
            List of discovered SAFE product directory paths.
        """
        return self.discover_local()

    def discover_local(self, update_db: bool = True) -> List[Path]:
        """Discover Sentinel-1 SLC products on local file system.

        Searches for directories matching Sentinel-1 SLC SAFE naming
        patterns (e.g., ``S1A_IW_SLC__1SDV_...SAFE``).

        Parameters
        ----------
        update_db : bool, default=True
            Whether to update the database with discovered products.

        Returns
        -------
        List[Path]
            List of discovered SAFE product paths.
        """
        products = []

        patterns = [
            "S1[ABC]_IW_SLC*.SAFE",
            "S1[ABC]_EW_SLC*.SAFE",
            "S1[ABC]_SM_SLC*.SAFE",
            # Uppercase and lowercase variations
            "S1?_IW_SLC*.SAFE",
            "S1?_EW_SLC*.SAFE",
        ]

        seen = set()
        for pattern in patterns:
            found = list(self.search_path.rglob(pattern))
            for p in found:
                if p.is_dir() and p not in seen:
                    products.append(p)
                    seen.add(p)

        if update_db:
            for product_path in products:
                self._index_local_product(product_path)

        logger.debug(
            "Sentinel-1 discover_local: found %d products in %s",
            len(products), self.search_path,
        )
        return products

    def _index_local_product(self, product_path: Path) -> None:
        """Index a local SAFE product in the database.

        Parameters
        ----------
        product_path : Path
            Path to Sentinel-1 SAFE product directory.
        """
        try:
            from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

            with Sentinel1SLCReader(product_path) as reader:
                metadata = reader.metadata
                pi = metadata.product_info
                si = metadata.swath_info

                product_id = product_path.name

                # Corner coords from geolocation grid (4 approx corners)
                corner_coords = None
                if metadata.geolocation_grid:
                    grid = metadata.geolocation_grid
                    lines = [p.line for p in grid]
                    pixels = [p.pixel for p in grid]
                    min_line = min(lines)
                    max_line = max(lines)
                    min_pixel = min(pixels)
                    max_pixel = max(pixels)

                    corners = []
                    for target_line, target_pixel in [
                        (min_line, min_pixel),
                        (min_line, max_pixel),
                        (max_line, max_pixel),
                        (max_line, min_pixel),
                    ]:
                        best = min(
                            grid,
                            key=lambda p, tl=target_line, tp=target_pixel: (
                                abs(p.line - tl) + abs(p.pixel - tp)
                            ),
                        )
                        corners.append([best.latitude, best.longitude])
                    corner_coords = json.dumps(corners)

                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT INTO products
                    (id, product_name, product_type, mission, mode,
                     polarization, absolute_orbit, relative_orbit, orbit_pass,
                     start_time, stop_time, swath, corner_coords,
                     local_path, file_size, download_date, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(product_name) DO UPDATE SET
                        product_type = excluded.product_type,
                        mission = excluded.mission,
                        mode = excluded.mode,
                        polarization = excluded.polarization,
                        absolute_orbit = excluded.absolute_orbit,
                        relative_orbit = excluded.relative_orbit,
                        orbit_pass = excluded.orbit_pass,
                        start_time = excluded.start_time,
                        stop_time = excluded.stop_time,
                        swath = excluded.swath,
                        corner_coords = excluded.corner_coords,
                        local_path = excluded.local_path,
                        file_size = excluded.file_size,
                        download_date = excluded.download_date,
                        metadata_json = excluded.metadata_json
                """, (
                    product_id,
                    product_path.name,
                    getattr(pi, 'product_type', None) if pi else None,
                    getattr(pi, 'mission', None) if pi else None,
                    getattr(pi, 'mode', None) if pi else None,
                    getattr(si, 'polarization', None) if si else None,
                    getattr(pi, 'absolute_orbit', None) if pi else None,
                    getattr(pi, 'relative_orbit', None) if pi else None,
                    getattr(pi, 'orbit_pass', None) if pi else None,
                    getattr(pi, 'start_time', None) if pi else None,
                    getattr(pi, 'stop_time', None) if pi else None,
                    getattr(si, 'swath', None) if si else None,
                    corner_coords,
                    str(product_path),
                    self._get_dir_size(product_path),
                    datetime.now().isoformat(),
                    json.dumps(metadata.to_dict(), default=str),
                ))
                self.conn.commit()

        except Exception as e:
            warnings.warn(
                f"Failed to index Sentinel-1 product {product_path}: {e}"
            )

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total

    def get_metadata_summary(
        self,
        image_paths: List[Path],
    ) -> List[Dict[str, Any]]:
        """Extract metadata summary from multiple Sentinel-1 SLC products.

        Parameters
        ----------
        image_paths : List[Path]
            List of Sentinel-1 SAFE product paths.

        Returns
        -------
        List[Dict[str, Any]]
            List of metadata dictionaries.
        """
        summaries = []

        for path in image_paths:
            try:
                from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

                with Sentinel1SLCReader(path) as reader:
                    metadata = reader.metadata
                    pi = metadata.product_info
                    si = metadata.swath_info

                    summaries.append({
                        'path': str(path),
                        'product_name': path.name,
                        'format': metadata.format,
                        'dimensions': (metadata.rows, metadata.cols),
                        'mission': getattr(pi, 'mission', None) if pi else None,
                        'mode': getattr(pi, 'mode', None) if pi else None,
                        'product_type': getattr(pi, 'product_type', None) if pi else None,
                        'polarization': getattr(si, 'polarization', None) if si else None,
                        'swath': getattr(si, 'swath', None) if si else None,
                        'absolute_orbit': getattr(pi, 'absolute_orbit', None) if pi else None,
                        'orbit_pass': getattr(pi, 'orbit_pass', None) if pi else None,
                        'start_time': getattr(pi, 'start_time', None) if pi else None,
                    })

            except Exception as e:
                warnings.warn(
                    f"Failed to extract metadata from {path}: {e}"
                )
                summaries.append({'path': str(path), 'error': str(e)})

        return summaries

    def find_overlapping(
        self,
        reference_bounds: Tuple[float, float, float, float],
        image_paths: List[Path],
    ) -> List[Path]:
        """Find Sentinel-1 products that overlap with a reference bounding box.

        Parameters
        ----------
        reference_bounds : Tuple[float, float, float, float]
            Reference bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
        image_paths : List[Path]
            List of Sentinel-1 SAFE product paths to check.

        Returns
        -------
        List[Path]
            List of overlapping product paths.
        """
        overlapping = []
        ref_min_lon, ref_min_lat, ref_max_lon, ref_max_lat = reference_bounds

        for path in image_paths:
            try:
                # Check database first
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT corner_coords FROM products WHERE id = ?",
                    (path.name,),
                )
                row = cursor.fetchone()

                if row and row['corner_coords']:
                    corners = json.loads(row['corner_coords'])
                    lats = [c[0] for c in corners]
                    lons = [c[1] for c in corners]
                else:
                    # Fall back to reading metadata
                    from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

                    with Sentinel1SLCReader(path) as reader:
                        grid = reader.metadata.geolocation_grid
                        if not grid:
                            continue
                        lats = [p.latitude for p in grid]
                        lons = [p.longitude for p in grid]

                prod_min_lat = min(lats)
                prod_max_lat = max(lats)
                prod_min_lon = min(lons)
                prod_max_lon = max(lons)

                if (prod_min_lon <= ref_max_lon
                        and prod_max_lon >= ref_min_lon
                        and prod_min_lat <= ref_max_lat
                        and prod_max_lat >= ref_min_lat):
                    overlapping.append(path)

            except Exception as e:
                warnings.warn(
                    f"Failed to check overlap for {path}: {e}"
                )

        return overlapping

    def query_database(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        orbit_pass: Optional[str] = None,
        absolute_orbit: Optional[int] = None,
        polarization: Optional[str] = None,
        swath: Optional[str] = None,
        has_local: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Query local database for Sentinel-1 products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date filter (ISO format).
        end_date : Optional[str], default=None
            End date filter (ISO format).
        orbit_pass : Optional[str], default=None
            Orbit pass filter (``'ASCENDING'`` or ``'DESCENDING'``).
        absolute_orbit : Optional[int], default=None
            Absolute orbit number filter.
        polarization : Optional[str], default=None
            Polarization filter (``'VV'``, ``'VH'``, etc.).
        swath : Optional[str], default=None
            Swath identifier filter (``'IW1'``, ``'IW2'``, ``'IW3'``).
        has_local : Optional[bool], default=None
            If True, return only products with local_path set.

        Returns
        -------
        List[Dict[str, Any]]
            List of products from database.
        """
        query = "SELECT * FROM products WHERE 1=1"
        params: List[Any] = []

        if start_date:
            query += " AND start_time >= ?"
            params.append(start_date)

        if end_date:
            query += " AND stop_time <= ?"
            params.append(end_date)

        if orbit_pass:
            query += " AND orbit_pass = ?"
            params.append(orbit_pass)

        if absolute_orbit is not None:
            query += " AND absolute_orbit = ?"
            params.append(absolute_orbit)

        if polarization:
            query += " AND polarization = ?"
            params.append(polarization)

        if swath:
            query += " AND swath = ?"
            params.append(swath)

        if has_local is not None:
            if has_local:
                query += " AND local_path IS NOT NULL"
            else:
                query += " AND local_path IS NULL"

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        return [dict(row) for row in cursor.fetchall()]

    # ── Remote (CDSE) ──────────────────────────────────────────────────

    def query_cdse(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        orbit_pass: Optional[str] = None,
        polarization: Optional[str] = None,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query Copernicus Data Space Ecosystem for Sentinel-1 SLC products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date in ISO format (YYYY-MM-DD).
        end_date : Optional[str], default=None
            End date in ISO format (YYYY-MM-DD).
        bbox : Optional[Tuple[float, float, float, float]], default=None
            Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
        orbit_pass : Optional[str], default=None
            Orbit pass filter (``'ASCENDING'`` or ``'DESCENDING'``).
        polarization : Optional[str], default=None
            Polarization filter (``'VV'``, ``'VH'``, ``'VV+VH'``).
        max_results : int, default=50
            Maximum number of results.

        Returns
        -------
        List[Dict[str, Any]]
            List of STAC feature dicts from CDSE.

        Raises
        ------
        DependencyError
            If ``requests`` is not installed.
        ProcessorError
            If STAC query fails.
        """
        if not REQUESTS_AVAILABLE:
            raise DependencyError(
                "Requests library required for CDSE queries. "
                "Install with: pip install requests"
            )

        # Validate inputs to prevent OData injection
        if bbox:
            west, south, east, north = bbox
            for coord_name, coord_val in [
                ("west", west), ("south", south),
                ("east", east), ("north", north),
            ]:
                if not isinstance(coord_val, (int, float)):
                    raise ValidationError(
                        f"bbox {coord_name} must be numeric, "
                        f"got {type(coord_val).__name__}"
                    )

        # Build OData $filter expression
        filter_parts = [
            f"Collection/Name eq '{self.CDSE_COLLECTION}'",
            "contains(Name,'SLC')",
        ]

        if start_date:
            ts = _normalize_stac_datetime(start_date)
            filter_parts.append(
                f"ContentDate/Start gt {ts.replace('Z', '.000Z')}"
            )
        if end_date:
            ts = _normalize_stac_datetime(end_date)
            filter_parts.append(
                f"ContentDate/Start lt {ts.replace('Z', '.000Z')}"
            )
        if bbox:
            wkt = (
                f"POLYGON(({west} {south},{east} {south},"
                f"{east} {north},{west} {north},{west} {south}))"
            )
            filter_parts.append(
                f"OData.CSC.Intersects(area=geography"
                f"'SRID=4326;{wkt}')"
            )

        params: Dict[str, str] = {
            "$filter": " and ".join(filter_parts),
            "$top": str(max_results),
            "$orderby": "ContentDate/Start desc",
        }

        try:
            response = requests.get(
                self.CDSE_ODATA_URL, params=params, timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            products = data.get("value", [])

            # Post-filter by orbit pass and polarization (OData has
            # limited attribute filtering for these fields)
            if orbit_pass:
                orbit_upper = orbit_pass.upper()
                products = [
                    p for p in products
                    if orbit_upper in p.get("Name", "").upper()
                    or orbit_upper[:3] in p.get("Name", "").upper()
                ]
            if polarization:
                pol = polarization.upper().replace("+", "")
                products = [
                    p for p in products
                    if pol in p.get("Name", "").upper()
                ]

            for product in products:
                self._index_remote_product(product)

            logger.info(
                "CDSE query returned %d Sentinel-1 SLC products",
                len(products),
            )
            return products

        except requests.RequestException as e:
            raise ProcessorError(
                f"CDSE OData query failed: {e}"
            ) from e

    def _index_remote_product(self, product: Dict[str, Any]) -> None:
        """Index a CDSE OData product in the local database.

        Parameters
        ----------
        product : Dict[str, Any]
            Product dict from CDSE OData ``/Products`` response.
        """
        try:
            product_id = product.get("Id", "")
            product_name = product.get("Name", "")
            remote_url = (
                f"{self.CDSE_DOWNLOAD_URL}({product_id})/$value"
            )

            # Parse name: S1A_IW_SLC__1SDV_20251202T060153_...
            parts = product_name.split("_")
            mission = parts[0] if len(parts) > 0 else ""
            mode = parts[1] if len(parts) > 1 else ""
            polarization = ""
            if len(parts) > 3 and len(parts[3]) >= 3:
                pol_code = parts[3][2:4]
                pol_map = {"DV": "VV+VH", "DH": "HH+HV",
                           "SV": "VV", "SH": "HH"}
                polarization = pol_map.get(pol_code, pol_code)

            start_time = product.get(
                "ContentDate", {}
            ).get("Start", "")
            stop_time = product.get(
                "ContentDate", {}
            ).get("End", "")

            footprint = product.get("GeoFootprint", {})
            corner_coords = json.dumps(
                footprint.get("coordinates", [])
            )

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO products
                (id, product_name, product_type, mission, mode,
                 polarization, absolute_orbit, orbit_pass,
                 start_time, stop_time, corner_coords,
                 remote_url, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_name) DO UPDATE SET
                    id = excluded.id,
                    remote_url = excluded.remote_url,
                    corner_coords = COALESCE(products.corner_coords,
                                             excluded.corner_coords),
                    metadata_json = excluded.metadata_json,
                    mission = COALESCE(products.mission, excluded.mission),
                    mode = COALESCE(products.mode, excluded.mode),
                    polarization = COALESCE(products.polarization,
                                            excluded.polarization)
            """, (
                product_id,
                product_name,
                "SLC",
                mission,
                mode,
                polarization,
                None,
                "",
                start_time,
                stop_time,
                corner_coords,
                remote_url,
                json.dumps(product),
            ))
            self.conn.commit()

        except Exception as e:
            warnings.warn(f"Failed to index remote Sentinel-1 product: {e}")

    def download_product(
        self,
        product_id: str,
        destination: Optional[Union[str, Path]] = None,
        extract: bool = True,
        force: bool = False,
    ) -> Path:
        """Download a Sentinel-1 SLC product from CDSE.

        Uses CDSE OAuth2 credentials from
        ``~/.config/geoint/credentials.json`` (``esa_copernicus`` block).

        If the product already exists locally (``local_path`` is set and
        the path exists on disk), a warning is issued and the existing
        path is returned unless ``force=True``.

        Parameters
        ----------
        product_id : str
            CDSE product ID. Must exist in the local database
            (run ``query_cdse`` first).
        destination : Optional[Union[str, Path]], default=None
            Directory to save the product. If None, uses ``search_path``.
        extract : bool, default=True
            If True, extract the ZIP and return the SAFE directory.
        force : bool, default=False
            If True, download even when the product already exists
            locally. The database ``local_path`` will be updated to
            the new download location.

        Returns
        -------
        Path
            Path to extracted SAFE directory or downloaded ZIP file.

        Raises
        ------
        ValueError
            If product not found or has no download URL.
        ProcessorError
            If download or authentication fails.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM products WHERE id = ?", (product_id,)
        )
        row = cursor.fetchone()

        if not row:
            raise ValueError(
                f"Product {product_id} not found in catalog. "
                f"Run query_cdse() first to populate the database."
            )

        remote_url = row["remote_url"]
        if not remote_url:
            raise ValueError(
                f"No download URL for product {product_id}"
            )

        # Check if the product already exists locally
        existing_path = row["local_path"]
        if existing_path and Path(existing_path).exists() and not force:
            warnings.warn(
                f"Product {row['product_name']} already exists locally "
                f"at {existing_path}. Downloading again will update the "
                f"database to point to the new location. Use "
                f"force=True to proceed."
            )
            return Path(existing_path)

        if destination is None:
            destination = self.search_path
        destination = Path(destination)

        access_token = get_cdse_token()

        product_path = download_file(
            url=remote_url,
            destination=destination,
            filename=f"{product_id}.zip",
            headers={"Authorization": f"Bearer {access_token}"},
            extract=extract,
        )

        cursor.execute("""
            UPDATE products
            SET local_path = ?, download_date = ?
            WHERE id = ?
        """, (
            str(product_path),
            datetime.now().isoformat(),
            product_id,
        ))
        self.conn.commit()

        return product_path

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()
