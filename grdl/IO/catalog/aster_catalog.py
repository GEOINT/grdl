# -*- coding: utf-8 -*-
"""
ASTER Catalog - Discovery, indexing, remote query, and download for ASTER
L1T and GDEM GeoTIFF products.

Provides tools for discovering ASTER products on disk, querying NASA
Earthdata CMR for available granules, downloading products via LP DAAC
with Earthdata Login authentication, and tracking collections in a local
SQLite database. Supports offline filtering by date, entity ID, WRS
path/row, and subsystem availability.

Dependencies
------------
rasterio (via ASTERReader)
requests (for Earthdata CMR queries and downloads)

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
2026-03-12
"""

# Standard library
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
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
    download_file,
    get_earthdata_token,
    load_credentials,
)
from grdl.exceptions import DependencyError, ProcessorError

logger = logging.getLogger(__name__)


class ASTERCatalog(CatalogInterface):
    """Catalog and download manager for ASTER L1T and GDEM products.

    Provides integrated capabilities for:

    - Local file system discovery of ASTER GeoTIFF products
    - Remote querying via NASA Earthdata CMR
    - Product download via LP DAAC with Earthdata Login authentication
    - SQLite database tracking of products

    Credentials are loaded from ``~/.config/geoint/credentials.json``
    (provider block ``nasa_earthdata``).

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
    >>> from grdl.IO.catalog import ASTERCatalog
    >>> catalog = ASTERCatalog('/data/aster')
    >>> products = catalog.discover_local()
    >>> print(f"Found {len(products)} products")
    >>>
    >>> # Search Earthdata for ASTER L1T products
    >>> remote = catalog.query_earthdata(
    ...     bbox=(-120.0, 34.0, -118.0, 36.0),
    ...     cloud_cover_max=20.0,
    ... )
    >>> catalog.download_product(remote[0]['id'])
    """

    # NASA CMR API for granule search
    CMR_SEARCH_URL = (
        "https://cmr.earthdata.nasa.gov/search/granules.json"
    )

    # CMR collection short name for ASTER L1T
    COLLECTION_L1T = "AST_L1T"

    def __init__(
        self,
        search_path: Union[str, Path],
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize ASTER catalog.

        Parameters
        ----------
        search_path : Union[str, Path]
            Root directory for local product searches.
        db_path : Optional[Union[str, Path]], default=None
            Path to SQLite database file. If None, uses
            ``search_path/aster_catalog.db``.

        Raises
        ------
        NotADirectoryError
            If search_path is not a directory.
        """
        super().__init__(search_path)

        if db_path is None:
            db_path = Path.home() / ".config" / "geoint" / "catalogs" / "aster.db"
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
                entity_id TEXT,
                local_granule_id TEXT,
                acquisition_date TEXT,
                acquisition_time TEXT,
                orbit_direction TEXT,
                wrs_path INTEGER,
                wrs_row INTEGER,
                scene_center_lat REAL,
                scene_center_lon REAL,
                sun_azimuth REAL,
                sun_elevation REAL,
                cloud_cover REAL,
                correction_level TEXT,
                vnir_available INTEGER,
                swir_available INTEGER,
                tir_available INTEGER,
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
            "CREATE INDEX IF NOT EXISTS idx_acquisition_date "
            "ON products(acquisition_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_id "
            "ON products(entity_id)"
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
        """Discover ASTER GeoTIFF files in the search path.

        Parameters
        ----------
        extensions : Optional[List[str]], default=None
            Not used; ASTER files are matched by filename pattern.
        recursive : bool, default=True
            Whether to search subdirectories recursively.

        Returns
        -------
        List[Path]
            List of discovered ASTER GeoTIFF file paths.
        """
        return self.discover_local()

    def discover_local(self, update_db: bool = True) -> List[Path]:
        """Discover ASTER products on local file system.

        Searches for GeoTIFF files matching ASTER naming patterns
        (e.g., ``AST_L1T_*.tif``, ``ASTGTM*.tif``).

        Parameters
        ----------
        update_db : bool, default=True
            Whether to update the database with discovered products.

        Returns
        -------
        List[Path]
            List of discovered ASTER GeoTIFF file paths.
        """
        products = []
        seen = set()

        patterns = [
            "AST_L1T_*.tif",
            "AST_L1T_*.TIF",
            "ASTGTM*.tif",
            "ASTGTM*.TIF",
            "AST_*.tif",
            "AST_*.TIF",
        ]

        for pattern in patterns:
            for p in self.search_path.rglob(pattern):
                if p.is_file() and p not in seen:
                    products.append(p)
                    seen.add(p)

        if update_db:
            for product_path in products:
                self._index_local_product(product_path)

        logger.debug(
            "ASTER discover_local: found %d products in %s",
            len(products), self.search_path,
        )
        return products

    def _index_local_product(self, product_path: Path) -> None:
        """Index a local ASTER GeoTIFF product in the database.

        Parameters
        ----------
        product_path : Path
            Path to ASTER GeoTIFF file.
        """
        try:
            from grdl.IO.ir.aster import ASTERReader

            with ASTERReader(product_path) as reader:
                metadata = reader.metadata

                product_id = product_path.name

                # Corner coords from metadata.corner_coords dict
                corner_coords = None
                if metadata.corner_coords:
                    corner_coords = json.dumps(
                        {k: list(v) for k, v in metadata.corner_coords.items()}
                    )

                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO products
                    (id, product_name, product_type, entity_id,
                     local_granule_id, acquisition_date, acquisition_time,
                     orbit_direction, wrs_path, wrs_row,
                     scene_center_lat, scene_center_lon,
                     sun_azimuth, sun_elevation, cloud_cover,
                     correction_level,
                     vnir_available, swir_available, tir_available,
                     corner_coords, local_path, file_size,
                     download_date, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    product_id,
                    product_path.name,
                    metadata.processing_level,
                    metadata.entity_id,
                    metadata.local_granule_id,
                    metadata.acquisition_date,
                    metadata.acquisition_time,
                    metadata.orbit_direction,
                    metadata.wrs_path,
                    metadata.wrs_row,
                    metadata.scene_center_lat,
                    metadata.scene_center_lon,
                    metadata.sun_azimuth,
                    metadata.sun_elevation,
                    metadata.cloud_cover,
                    metadata.correction_level,
                    int(metadata.vnir_available) if metadata.vnir_available is not None else None,
                    int(metadata.swir_available) if metadata.swir_available is not None else None,
                    int(metadata.tir_available) if metadata.tir_available is not None else None,
                    corner_coords,
                    str(product_path),
                    product_path.stat().st_size,
                    datetime.now().isoformat(),
                    json.dumps(metadata.to_dict(), default=str),
                ))
                self.conn.commit()

        except Exception as e:
            warnings.warn(
                f"Failed to index ASTER product {product_path}: {e}"
            )

    def get_metadata_summary(
        self,
        image_paths: List[Path],
    ) -> List[Dict[str, Any]]:
        """Extract metadata summary from multiple ASTER products.

        Parameters
        ----------
        image_paths : List[Path]
            List of ASTER GeoTIFF file paths.

        Returns
        -------
        List[Dict[str, Any]]
            List of metadata dictionaries.
        """
        summaries = []

        for path in image_paths:
            try:
                from grdl.IO.ir.aster import ASTERReader

                with ASTERReader(path) as reader:
                    metadata = reader.metadata

                    summaries.append({
                        'path': str(path),
                        'product_name': path.name,
                        'format': metadata.format,
                        'dimensions': (metadata.rows, metadata.cols),
                        'product_type': metadata.processing_level,
                        'entity_id': metadata.entity_id,
                        'acquisition_date': metadata.acquisition_date,
                        'acquisition_time': metadata.acquisition_time,
                        'orbit_direction': metadata.orbit_direction,
                        'wrs_path': metadata.wrs_path,
                        'wrs_row': metadata.wrs_row,
                        'scene_center_lat': metadata.scene_center_lat,
                        'scene_center_lon': metadata.scene_center_lon,
                        'cloud_cover': metadata.cloud_cover,
                        'vnir_available': metadata.vnir_available,
                        'swir_available': metadata.swir_available,
                        'tir_available': metadata.tir_available,
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
        """Find ASTER products that overlap with a reference bounding box.

        Parameters
        ----------
        reference_bounds : Tuple[float, float, float, float]
            Reference bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
        image_paths : List[Path]
            List of ASTER GeoTIFF file paths to check.

        Returns
        -------
        List[Path]
            List of overlapping product paths.
        """
        overlapping = []
        ref_min_lon, ref_min_lat, ref_max_lon, ref_max_lat = reference_bounds

        for path in image_paths:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT corner_coords, scene_center_lat, scene_center_lon "
                    "FROM products WHERE id = ?",
                    (path.name,),
                )
                row = cursor.fetchone()

                lats: List[float] = []
                lons: List[float] = []

                if row and row['corner_coords']:
                    corners = json.loads(row['corner_coords'])
                    # corners is {'UL': [lat, lon], 'UR': ..., ...}
                    for coord in corners.values():
                        lats.append(float(coord[0]))
                        lons.append(float(coord[1]))
                elif row and row['scene_center_lat'] is not None:
                    lats = [row['scene_center_lat']]
                    lons = [row['scene_center_lon']]
                else:
                    from grdl.IO.ir.aster import ASTERReader

                    with ASTERReader(path) as reader:
                        cc = reader.metadata.corner_coords
                        if cc:
                            for coord in cc.values():
                                lats.append(float(coord[0]))
                                lons.append(float(coord[1]))

                if not lats or not lons:
                    continue

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
        entity_id: Optional[str] = None,
        wrs_path: Optional[int] = None,
        wrs_row: Optional[int] = None,
        processing_level: Optional[str] = None,
        cloud_cover_max: Optional[float] = None,
        has_local: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Query local database for ASTER products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date filter (``YYYY/MM/DD`` format).
        end_date : Optional[str], default=None
            End date filter (``YYYY/MM/DD`` format).
        entity_id : Optional[str], default=None
            ASTER entity ID filter.
        wrs_path : Optional[int], default=None
            WRS path number filter.
        wrs_row : Optional[int], default=None
            WRS row number filter.
        processing_level : Optional[str], default=None
            Processing level filter (``'L1T'`` or ``'GDEM'``).
        cloud_cover_max : Optional[float], default=None
            Maximum cloud cover percentage filter.
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
            query += " AND acquisition_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND acquisition_date <= ?"
            params.append(end_date)

        if entity_id:
            query += " AND entity_id = ?"
            params.append(entity_id)

        if wrs_path is not None:
            query += " AND wrs_path = ?"
            params.append(wrs_path)

        if wrs_row is not None:
            query += " AND wrs_row = ?"
            params.append(wrs_row)

        if processing_level:
            query += " AND product_type = ?"
            params.append(processing_level)

        if cloud_cover_max is not None:
            query += " AND cloud_cover <= ?"
            params.append(cloud_cover_max)

        if has_local is not None:
            if has_local:
                query += " AND local_path IS NOT NULL"
            else:
                query += " AND local_path IS NULL"

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        return [dict(row) for row in cursor.fetchall()]

    # ── Remote (NASA Earthdata CMR) ────────────────────────────────────

    def query_earthdata(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        entity_id: Optional[str] = None,
        cloud_cover_max: Optional[float] = None,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query NASA Earthdata CMR for ASTER L1T granules.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date in ISO format (YYYY-MM-DD).
        end_date : Optional[str], default=None
            End date in ISO format (YYYY-MM-DD).
        bbox : Optional[Tuple[float, float, float, float]], default=None
            Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
        entity_id : Optional[str], default=None
            ASTER entity ID filter.
        cloud_cover_max : Optional[float], default=None
            Maximum cloud cover percentage.
        max_results : int, default=50
            Maximum number of results.

        Returns
        -------
        List[Dict[str, Any]]
            List of CMR granule dicts.

        Raises
        ------
        DependencyError
            If ``requests`` is not installed.
        ProcessorError
            If CMR query fails.
        """
        if not REQUESTS_AVAILABLE:
            raise DependencyError(
                "Requests library required for Earthdata queries. "
                "Install with: pip install requests"
            )

        params: Dict[str, Any] = {
            "short_name": self.COLLECTION_L1T,
            "page_size": max_results,
            "sort_key": "-start_date",
        }

        if bbox:
            params["bounding_box"] = (
                f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            )

        if start_date or end_date:
            start = start_date or ""
            end = end_date or ""
            params["temporal"] = f"{start},{end}"

        if entity_id:
            params["readable_granule_name"] = entity_id

        if cloud_cover_max is not None:
            params["cloud_cover"] = f",{cloud_cover_max}"

        try:
            response = requests.get(
                self.CMR_SEARCH_URL, params=params, timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            granules = data.get("feed", {}).get("entry", [])

            for granule in granules:
                self._index_remote_product(granule)

            logger.info(
                "Earthdata CMR returned %d ASTER granules",
                len(granules),
            )
            return granules

        except requests.RequestException as e:
            raise ProcessorError(
                f"Earthdata CMR query failed: {e}"
            ) from e

    def _index_remote_product(self, granule: Dict[str, Any]) -> None:
        """Index a CMR granule in the local database.

        Parameters
        ----------
        granule : Dict[str, Any]
            Granule entry from CMR JSON response.
        """
        try:
            granule_id = granule.get("id", granule.get("title", ""))
            title = granule.get("title", granule_id)

            # Extract download URL from links
            remote_url = ""
            for link in granule.get("links", []):
                href = link.get("href", "")
                if (href.endswith(".hdf") or href.endswith(".tif")
                        or "data#" in link.get("rel", "")):
                    remote_url = href
                    break

            time_start = granule.get("time_start", "")
            time_end = granule.get("time_end", "")

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO products
                (id, product_name, product_type,
                 acquisition_date, remote_url, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                granule_id,
                title,
                "L1T",
                time_start,
                remote_url,
                json.dumps(granule),
            ))
            self.conn.commit()

        except Exception as e:
            warnings.warn(f"Failed to index remote ASTER granule: {e}")

    def download_product(
        self,
        product_id: str,
        destination: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Download an ASTER product from LP DAAC via Earthdata.

        Uses Earthdata Login credentials from
        ``~/.config/geoint/credentials.json`` (``nasa_earthdata`` block).

        Parameters
        ----------
        product_id : str
            Granule ID. Must exist in the local database
            (run ``query_earthdata`` first).
        destination : Optional[Union[str, Path]], default=None
            Directory to save the product. If None, uses ``search_path``.

        Returns
        -------
        Path
            Path to downloaded file.

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
                f"Run query_earthdata() first to populate the database."
            )

        remote_url = row["remote_url"]
        if not remote_url:
            raise ValueError(
                f"No download URL for product {product_id}"
            )

        if destination is None:
            destination = self.search_path
        destination = Path(destination)

        access_token = get_earthdata_token()

        filename = remote_url.split("/")[-1] or f"{product_id}.hdf"
        product_path = download_file(
            url=remote_url,
            destination=destination,
            filename=filename,
            headers={"Authorization": f"Bearer {access_token}"},
            extract=False,
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
