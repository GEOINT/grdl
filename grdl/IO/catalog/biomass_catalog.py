# -*- coding: utf-8 -*-
"""
BIOMASS Catalog - Discovery, indexing, and download for BIOMASS products.

Provides tools for discovering BIOMASS imagery on disk, querying the ESA
MAAP STAC catalog, downloading products with OAuth2, and tracking
collections in a local SQLite database.

Dependencies
------------
requests

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
2026-01-30

Modified
--------
2026-03-13
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import sqlite3
import warnings
from datetime import datetime

# Third-party
try:
    import requests
except ImportError:
    pass

# GRDL internal
from grdl.exceptions import DependencyError, ProcessorError
from grdl.IO.base import CatalogInterface
from grdl.IO.catalog.remote_utils import (
    REQUESTS_AVAILABLE,
    _normalize_stac_datetime,
    download_file,
    load_credentials,
)

logger = logging.getLogger(__name__)


class BIOMASSCatalog(CatalogInterface):
    """Catalog and download manager for BIOMASS ESA satellite products.

    Provides integrated capabilities for:

    - Local file system discovery of BIOMASS products
    - Remote querying via ESA MAAP STAC catalog
    - Product download with OAuth2 authentication
    - SQLite database tracking of products

    Credentials are loaded from ``~/.config/geoint/credentials.json``.

    Attributes
    ----------
    search_path : Path
        Root directory for local product searches and downloads.
    db_path : Path
        Path to SQLite database file for product tracking.
    conn : sqlite3.Connection
        Database connection.

    Examples
    --------
    >>> from grdl.IO.sar import BIOMASSCatalog
    >>> catalog = BIOMASSCatalog('/data/biomass')
    >>>
    >>> # Search for products near New Norcia, Australia
    >>> products = catalog.query_esa(
    ...     bbox=(115.5, -31.5, 116.8, -30.5),
    ...     max_results=10,
    ... )
    >>> print(f"Found {len(products)} products")
    >>>
    >>> # Download the most recent product
    >>> catalog.download_product(products[0]['id'])
    """

    # ESA MAAP STAC API for BIOMASS product search and download
    MAAP_STAC_URL = "https://catalog.maap.eo.esa.int/catalogue"
    MAAP_IAM_URL = (
        "https://iam.maap.eo.esa.int/realms/esa-maap"
        "/protocol/openid-connect/token"
    )
    MAAP_TOKEN_CLIENT_ID = "offline-token"
    # Public client secret for the MAAP offline-token application.
    # Shared by all MAAP users; published in ESA documentation.
    MAAP_TOKEN_CLIENT_SECRET = "p1eL7uonXs6MDxtGbgKdPVRAmnGxHpVE"

    # STAC collection IDs (operational, open-access)
    COLLECTION_L1A = "BiomassLevel1a"
    COLLECTION_L1B = "BiomassLevel1b"
    COLLECTION_L2A = "BiomassLevel2a"

    def __init__(
        self,
        search_path: Union[str, Path],
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize BIOMASS catalog.

        Parameters
        ----------
        search_path : Union[str, Path]
            Root directory for local product searches and downloads.
        db_path : Optional[Union[str, Path]], default=None
            Path to SQLite database file. If None, uses
            ``search_path/biomass_catalog.db``.

        Raises
        ------
        NotADirectoryError
            If search_path is not a directory.
        """
        super().__init__(search_path)

        # Set up database
        if db_path is None:
            db_path = Path.home() / ".config" / "geoint" / "catalogs" / "biomass.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
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
                processing_level TEXT,
                swath TEXT,
                polarizations TEXT,
                orbit_number INTEGER,
                orbit_pass TEXT,
                start_time TEXT,
                stop_time TEXT,
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
            "CREATE INDEX IF NOT EXISTS idx_orbit "
            "ON products(orbit_number)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_processing_level "
            "ON products(processing_level)"
        )

        self.conn.commit()

    def discover_images(
        self,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Path]:
        """Discover BIOMASS product directories in the search path.

        BIOMASS products are directories with specific naming patterns.

        Parameters
        ----------
        extensions : Optional[List[str]], default=None
            Not used for BIOMASS (directories, not files).
        recursive : bool, default=True
            Whether to search subdirectories recursively.

        Returns
        -------
        List[Path]
            List of discovered BIOMASS product directory paths.
        """
        return self.discover_local()

    def discover_local(self, update_db: bool = True) -> List[Path]:
        """Discover BIOMASS products on local file system.

        Searches for directories matching BIOMASS product naming patterns
        (e.g., ``BIO_S1_SCS__1S_...``).

        Parameters
        ----------
        update_db : bool, default=True
            Whether to update the database with discovered products.

        Returns
        -------
        List[Path]
            List of discovered BIOMASS product paths.
        """
        products = []

        patterns = [
            "BIO_S*_SCS*",
            "BIO_S*_FBD*",
        ]

        for pattern in patterns:
            found = list(self.search_path.rglob(pattern))
            products.extend([p for p in found if p.is_dir()])

        if update_db:
            for product_path in products:
                self._index_local_product(product_path)

        return products

    def _index_local_product(self, product_path: Path) -> None:
        """Index a local product in the database.

        Parameters
        ----------
        product_path : Path
            Path to BIOMASS product directory.
        """
        try:
            from grdl.IO.sar.biomass import BIOMASSL1Reader

            with BIOMASSL1Reader(product_path) as reader:
                metadata = reader.metadata

                product_id = product_path.name
                cursor = self.conn.cursor()

                cursor.execute("""
                    INSERT INTO products
                    (id, product_name, product_type, processing_level,
                     swath, polarizations, orbit_number, orbit_pass,
                     start_time, stop_time, corner_coords, local_path,
                     file_size, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(product_name) DO UPDATE SET
                        product_type = excluded.product_type,
                        processing_level = excluded.processing_level,
                        swath = excluded.swath,
                        polarizations = excluded.polarizations,
                        orbit_number = excluded.orbit_number,
                        orbit_pass = excluded.orbit_pass,
                        start_time = excluded.start_time,
                        stop_time = excluded.stop_time,
                        corner_coords = excluded.corner_coords,
                        local_path = excluded.local_path,
                        file_size = excluded.file_size,
                        metadata_json = excluded.metadata_json
                """, (
                    product_id,
                    product_path.name,
                    metadata.get('product_type', 'SCS'),
                    'L1',
                    metadata.get('swath', ''),
                    json.dumps(metadata.get('polarizations', [])),
                    metadata.get('orbit_number', 0),
                    metadata.get('orbit_pass', ''),
                    metadata.get('start_time', ''),
                    metadata.get('stop_time', ''),
                    json.dumps(metadata.get('corner_coords', {})),
                    str(product_path),
                    self._get_dir_size(product_path),
                    json.dumps(metadata.to_dict(), default=str),
                ))

                self.conn.commit()

        except Exception as e:
            warnings.warn(f"Failed to index product {product_path}: {e}")

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total

    def _get_access_token(
        self,
        credentials_file: Optional[Union[str, Path]] = None,
    ) -> str:
        """Exchange an ESA MAAP offline token for a short-lived access token.

        Parameters
        ----------
        credentials_file : Optional[Union[str, Path]], default=None
            Path to credentials JSON file.

        Returns
        -------
        str
            Bearer access token for MAAP API requests.

        Raises
        ------
        RuntimeError
            If token exchange fails.
        """
        if not REQUESTS_AVAILABLE:
            raise DependencyError(
                "Requests library required. "
                "Install with: pip install requests"
            )

        creds = load_credentials("esa_maap", credentials_file)
        offline_token = creds["offline_token"]

        response = requests.post(
            self.MAAP_IAM_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": offline_token,
                "client_id": self.MAAP_TOKEN_CLIENT_ID,
                "client_secret": self.MAAP_TOKEN_CLIENT_SECRET,
                "scope": "offline_access openid",
            },
            timeout=30,
        )

        if response.status_code != 200:
            raise ProcessorError(
                f"MAAP token exchange failed ({response.status_code}): "
                f"{response.text[:300]}"
            )

        return response.json()["access_token"]

    def query_esa(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        orbit: Optional[int] = None,
        max_results: int = 50,
        collection: Optional[str] = None,
        product_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query ESA MAAP STAC catalog for BIOMASS products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date in ISO format (YYYY-MM-DD).
        end_date : Optional[str], default=None
            End date in ISO format (YYYY-MM-DD).
        bbox : Optional[Tuple[float, float, float, float]], default=None
            Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
        orbit : Optional[int], default=None
            Absolute orbit number filter.
        max_results : int, default=50
            Maximum number of results to return.
        collection : Optional[str], default=None
            STAC collection ID. If None, uses ``COLLECTION_L1A``.
        product_type : Optional[str], default=None
            BIOMASS product type filter. Common values:
            ``S3_SCS__1S`` (single-pol), ``S3_SCS__1M`` (multi-pol).
            If None, returns all product types.

        Returns
        -------
        List[Dict[str, Any]]
            List of STAC feature dicts, each with ``id``, ``properties``,
            ``geometry``, and ``assets`` keys.

        Raises
        ------
        ImportError
            If requests library is not installed.
        RuntimeError
            If STAC query fails.
        """
        if not REQUESTS_AVAILABLE:
            raise DependencyError(
                "Requests library required for ESA queries. "
                "Install with: pip install requests"
            )

        if collection is None:
            collection = self.COLLECTION_L1A

        search_url = f"{self.MAAP_STAC_URL}/search"

        payload: Dict[str, Any] = {
            "collections": [collection],
            "limit": max_results,
            "sortby": [{"field": "datetime", "direction": "desc"}],
        }

        if bbox:
            payload["bbox"] = list(bbox)

        if start_date or end_date:
            start = _normalize_stac_datetime(start_date or "..")
            end = _normalize_stac_datetime(end_date or "..")
            payload["datetime"] = f"{start}/{end}"

        filters = []
        if product_type:
            filters.append({
                "op": "=",
                "args": [{"property": "product:type"}, product_type],
            })
        if orbit is not None:
            filters.append({
                "op": "=",
                "args": [{"property": "sat:absolute_orbit"}, orbit],
            })

        if filters:
            if len(filters) == 1:
                payload["filter"] = filters[0]
            else:
                payload["filter"] = {"op": "and", "args": filters}
            payload["filter-lang"] = "cql2-json"

        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                search_url, json=payload, headers=headers, timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            products = data.get("features", [])

            for product in products:
                self._index_remote_product(product)

            return products

        except requests.RequestException as e:
            raise ProcessorError(f"MAAP STAC query failed: {e}") from e

    def _index_remote_product(self, feature: Dict[str, Any]) -> None:
        """Index a STAC feature in the local database.

        Parameters
        ----------
        feature : Dict[str, Any]
            GeoJSON Feature from MAAP STAC search response.
        """
        try:
            props = feature.get("properties", {})
            product_id = feature.get("id", "")

            assets = feature.get("assets", {})
            product_asset = assets.get("product", {})
            remote_url = product_asset.get("href", "")

            geom = feature.get("geometry", {})
            corner_coords = json.dumps(geom.get("coordinates", []))

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO products
                (id, product_name, product_type, processing_level,
                 orbit_number, orbit_pass, start_time, stop_time,
                 corner_coords, remote_url, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_name) DO UPDATE SET
                    id = excluded.id,
                    remote_url = excluded.remote_url,
                    corner_coords = COALESCE(products.corner_coords,
                                             excluded.corner_coords),
                    metadata_json = excluded.metadata_json,
                    product_type = COALESCE(products.product_type,
                                            excluded.product_type),
                    processing_level = COALESCE(products.processing_level,
                                                excluded.processing_level)
            """, (
                product_id,
                props.get("title", product_id),
                props.get("product:type", "SCS"),
                props.get("processing:level", "L1A"),
                props.get("sat:absolute_orbit", 0),
                props.get("sat:orbit_state", ""),
                props.get("start_datetime", ""),
                props.get("end_datetime", ""),
                corner_coords,
                remote_url,
                json.dumps(feature),
            ))
            self.conn.commit()

        except Exception as e:
            warnings.warn(f"Failed to index remote product: {e}")

    def download_product(
        self,
        product_id: str,
        destination: Optional[Union[str, Path]] = None,
        extract: bool = True,
        force: bool = False,
    ) -> Path:
        """Download a BIOMASS product ZIP from the ESA MAAP.

        Uses the MAAP offline token (from
        ``~/.config/geoint/credentials.json``) to authenticate via
        OAuth2 Bearer token.

        If the product already exists locally (``local_path`` is set and
        the path exists on disk), a warning is issued and the existing
        path is returned unless ``force=True``.

        Parameters
        ----------
        product_id : str
            STAC item ID (e.g. ``BIO_S1_SCS__1S_20251208T221918_...``).
            Must exist in the local database (run ``query_esa`` first).
        destination : Optional[Union[str, Path]], default=None
            Directory to save the product. If None, uses ``search_path``.
        extract : bool, default=True
            If True, extract the ZIP and return the extracted directory.
        force : bool, default=False
            If True, download even when the product already exists
            locally. The database ``local_path`` will be updated to
            the new download location.

        Returns
        -------
        Path
            Path to extracted product directory (if extract=True) or
            downloaded ZIP file.

        Raises
        ------
        ValueError
            If product not found in database or has no download URL.
        RuntimeError
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
                f"Run query_esa() first to populate the database."
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

        access_token = self._get_access_token()

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

    def get_metadata_summary(
        self,
        image_paths: List[Path],
    ) -> List[Dict[str, Any]]:
        """Extract metadata summary from multiple BIOMASS products.

        Parameters
        ----------
        image_paths : List[Path]
            List of BIOMASS product paths.

        Returns
        -------
        List[Dict[str, Any]]
            List of metadata dictionaries.
        """
        summaries = []

        for path in image_paths:
            try:
                from grdl.IO.sar.biomass import BIOMASSL1Reader

                with BIOMASSL1Reader(path) as reader:
                    summaries.append({
                        'path': str(path),
                        'product_name': path.name,
                        'format': reader.metadata['format'],
                        'dimensions': (
                            reader.metadata['rows'],
                            reader.metadata['cols'],
                        ),
                        'polarizations': reader.metadata.get(
                            'polarizations', []
                        ),
                        'orbit': reader.metadata.get('orbit_number'),
                        'start_time': reader.metadata.get('start_time'),
                    })

            except Exception as e:
                warnings.warn(
                    f"Failed to extract metadata from {path}: {e}"
                )
                summaries.append({
                    'path': str(path),
                    'error': str(e),
                })

        return summaries

    def find_overlapping(
        self,
        reference_bounds: Tuple[float, float, float, float],
        image_paths: List[Path],
    ) -> List[Path]:
        """Find BIOMASS products that overlap with a reference bounding box.

        Parameters
        ----------
        reference_bounds : Tuple[float, float, float, float]
            Reference bounding box as
            ``(min_lon, min_lat, max_lon, max_lat)``.
        image_paths : List[Path]
            List of BIOMASS product paths to check.

        Returns
        -------
        List[Path]
            List of overlapping product paths.
        """
        overlapping = []
        ref_min_lon, ref_min_lat, ref_max_lon, ref_max_lat = (
            reference_bounds
        )

        for path in image_paths:
            try:
                from grdl.IO.sar.biomass import BIOMASSL1Reader

                with BIOMASSL1Reader(path) as reader:
                    corners = reader.metadata.get('corner_coords')

                    if not corners:
                        continue

                    lats = [
                        corners[f'corner{i}'][0] for i in range(1, 5)
                    ]
                    lons = [
                        corners[f'corner{i}'][1] for i in range(1, 5)
                    ]

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
        orbit: Optional[int] = None,
        has_local: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Query local database for products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date filter (ISO format).
        end_date : Optional[str], default=None
            End date filter (ISO format).
        orbit : Optional[int], default=None
            Orbit number filter.
        has_local : Optional[bool], default=None
            Filter for products with/without local paths.

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

        if orbit is not None:
            query += " AND orbit_number = ?"
            params.append(orbit)

        if has_local is not None:
            if has_local:
                query += " AND local_path IS NOT NULL"
            else:
                query += " AND local_path IS NULL"

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()
