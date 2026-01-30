# -*- coding: utf-8 -*-
"""
Catalog Module - Image discovery, indexing, and download management.

Provides tools for discovering imagery on disk, querying remote data hubs,
downloading products, and tracking collections in a local database.

Dependencies
------------
requests

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
2026-01-30

Modified
--------
2026-01-30 - MAAP STAC API integration for search and download
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import sqlite3
import json
import warnings
from datetime import datetime

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn(
        "Requests not available for remote queries. Install with: pip install requests",
        ImportWarning
    )

from grdl.IO.base import CatalogInterface

# Default credentials file path (repo-agnostic, shared across projects)
_CREDENTIALS_PATH = Path.home() / ".config" / "geoint" / "credentials.json"


def load_credentials(
    provider: str = "esa_maap",
    credentials_file: Optional[Union[str, Path]] = None
) -> Dict[str, str]:
    """
    Load credentials from the shared geoint config file.

    For ``esa_maap``, returns a dict with ``offline_token``.
    For ``esa_copernicus``, returns a dict with ``username`` and ``password``.
    Falls back to environment variables if the file entry is empty.

    Parameters
    ----------
    provider : str, default="esa_maap"
        Key in the credentials JSON for the provider block.
    credentials_file : Optional[Union[str, Path]], default=None
        Path to credentials JSON. If None, uses
        ``~/.config/geoint/credentials.json``.

    Returns
    -------
    Dict[str, str]
        Credential fields for the requested provider.

    Raises
    ------
    ValueError
        If credentials are empty or missing for the requested provider.
    """
    cred_path = Path(credentials_file) if credentials_file else _CREDENTIALS_PATH

    if cred_path.exists():
        with open(cred_path, 'r') as f:
            creds = json.load(f)

        block = creds.get(provider, {})

        # Check for non-empty values
        if all(v for v in block.values()):
            return dict(block)

    # Fallback to environment variables
    if provider == "esa_maap":
        token = os.environ.get("ESA_MAAP_OFFLINE_TOKEN", "")
        if token:
            return {"offline_token": token}
    elif provider == "esa_copernicus":
        username = os.environ.get("ESA_BIOMASS_USER", "")
        password = os.environ.get("ESA_BIOMASS_PASSWORD", "")
        if username and password:
            return {"username": username, "password": password}

    raise ValueError(
        f"No credentials found for '{provider}'. "
        f"Set them in {_CREDENTIALS_PATH} or via environment variables."
    )


class BIOMASSCatalog(CatalogInterface):
    """
    Catalog and download manager for BIOMASS ESA satellite products.

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
    >>> catalog = BIOMASSCatalog('/data/biomass')
    >>>
    >>> # Search for products near New Norcia, Australia
    >>> products = catalog.query_esa(
    ...     bbox=(115.5, -31.5, 116.8, -30.5),
    ...     max_results=10
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
        db_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize BIOMASS catalog.

        Parameters
        ----------
        search_path : Union[str, Path]
            Root directory for local product searches and downloads
        db_path : Optional[Union[str, Path]], default=None
            Path to SQLite database file. If None, uses
            search_path/biomass_catalog.db

        Raises
        ------
        NotADirectoryError
            If search_path is not a directory
        """
        super().__init__(search_path)

        # Set up database
        if db_path is None:
            db_path = self.search_path / "biomass_catalog.db"
        self.db_path = Path(db_path)

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for product tracking."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name

        cursor = self.conn.cursor()

        # Create products table
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

        # Create index on common query fields
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON products(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orbit ON products(orbit_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_level ON products(processing_level)")

        self.conn.commit()

    def discover_images(
        self,
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Path]:
        """
        Discover BIOMASS product directories in the search path.

        BIOMASS products are directories with specific naming patterns.

        Parameters
        ----------
        extensions : Optional[List[str]], default=None
            Not used for BIOMASS (directories, not files)
        recursive : bool, default=True
            Whether to search subdirectories recursively

        Returns
        -------
        List[Path]
            List of discovered BIOMASS product directory paths
        """
        return self.discover_local()

    def discover_local(self, update_db: bool = True) -> List[Path]:
        """
        Discover BIOMASS products on local file system.

        Searches for directories matching BIOMASS product naming patterns
        (e.g., BIO_S1_SCS__1S_...).

        Parameters
        ----------
        update_db : bool, default=True
            Whether to update the database with discovered products

        Returns
        -------
        List[Path]
            List of discovered BIOMASS product paths
        """
        products = []

        # BIOMASS L1 SCS products start with BIO_S1_SCS
        patterns = [
            "BIO_S1_SCS*",  # L1 SCS products
            "BIO_S*_FBD*",  # Future: L2/L3 products
        ]

        for pattern in patterns:
            if recursive:
                found = list(self.search_path.rglob(pattern))
            else:
                found = list(self.search_path.glob(pattern))

            # Filter to directories only
            products.extend([p for p in found if p.is_dir()])

        # Update database if requested
        if update_db:
            for product_path in products:
                self._index_local_product(product_path)

        return products

    def _index_local_product(self, product_path: Path) -> None:
        """
        Index a local product in the database.

        Extracts metadata from the product and stores in database.

        Parameters
        ----------
        product_path : Path
            Path to BIOMASS product directory
        """
        try:
            # Open product to extract metadata
            from grdl.IO import BIOMASSL1Reader

            with BIOMASSL1Reader(product_path) as reader:
                metadata = reader.metadata

                # Prepare data for database
                product_id = product_path.name
                cursor = self.conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO products
                    (id, product_name, product_type, processing_level, swath,
                     polarizations, orbit_number, orbit_pass, start_time, stop_time,
                     corner_coords, local_path, file_size, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    product_id,
                    product_path.name,
                    metadata.get('product_type', 'SCS'),
                    'L1',  # L1 for SCS products
                    metadata.get('swath', ''),
                    json.dumps(metadata.get('polarizations', [])),
                    metadata.get('orbit_number', 0),
                    metadata.get('orbit_pass', ''),
                    metadata.get('start_time', ''),
                    metadata.get('stop_time', ''),
                    json.dumps(metadata.get('corner_coords', {})),
                    str(product_path),
                    self._get_dir_size(product_path),
                    json.dumps(metadata)
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
        credentials_file: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Exchange an ESA MAAP offline token for a short-lived access token.

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
            raise ImportError(
                "Requests library required. Install with: pip install requests"
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
            raise RuntimeError(
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
        product_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query ESA MAAP STAC catalog for BIOMASS products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date in ISO format (YYYY-MM-DD).
        end_date : Optional[str], default=None
            End date in ISO format (YYYY-MM-DD).
        bbox : Optional[Tuple[float, float, float, float]], default=None
            Bounding box as (min_lon, min_lat, max_lon, max_lat).
        orbit : Optional[int], default=None
            Absolute orbit number filter.
        max_results : int, default=50
            Maximum number of results to return.
        collection : Optional[str], default=None
            STAC collection ID. If None, uses ``COLLECTION_L1A``.
        product_type : Optional[str], default=None
            BIOMASS product type filter. Common values:
            ``S3_SCS__1S`` (single-pol processing),
            ``S3_SCS__1M`` (multi-pol processing).
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
            raise ImportError(
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
            start = start_date or ".."
            end = end_date or ".."
            payload["datetime"] = f"{start}/{end}"

        # STAC CQL2 filter for product type and orbit
        filters = []
        if product_type:
            filters.append({
                "op": "=",
                "args": [{"property": "product:type"}, product_type]
            })
        if orbit is not None:
            filters.append({
                "op": "=",
                "args": [{"property": "sat:absolute_orbit"}, orbit]
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
                search_url, json=payload, headers=headers, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            products = data.get("features", [])

            # Index results in local database
            for product in products:
                self._index_remote_product(product)

            return products

        except requests.RequestException as e:
            raise RuntimeError(f"MAAP STAC query failed: {e}") from e

    def _index_remote_product(self, feature: Dict[str, Any]) -> None:
        """
        Index a STAC feature in the local database.

        Parameters
        ----------
        feature : Dict[str, Any]
            GeoJSON Feature from MAAP STAC search response.
        """
        try:
            props = feature.get("properties", {})
            product_id = feature.get("id", "")

            # Extract download URL from the 'product' asset (ZIP)
            assets = feature.get("assets", {})
            product_asset = assets.get("product", {})
            remote_url = product_asset.get("href", "")

            # Corner coordinates from geometry
            geom = feature.get("geometry", {})
            corner_coords = json.dumps(geom.get("coordinates", []))

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO products
                (id, product_name, product_type, processing_level,
                 orbit_number, orbit_pass, start_time, stop_time,
                 corner_coords, remote_url, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        extract: bool = True
    ) -> Path:
        """
        Download a BIOMASS product ZIP from the ESA MAAP.

        Uses the MAAP offline token (from ``~/.config/geoint/credentials.json``)
        to authenticate via OAuth2 Bearer token.

        Parameters
        ----------
        product_id : str
            STAC item ID (e.g. ``BIO_S1_SCS__1S_20251208T221918_...``).
            Must exist in the local database (run ``query_esa`` first).
        destination : Optional[Union[str, Path]], default=None
            Directory to save the product. If None, uses ``search_path``.
        extract : bool, default=True
            If True, extract the ZIP and return the extracted directory path.

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
        import zipfile

        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "Requests library required for downloads. "
                "Install with: pip install requests"
            )

        # Get product info from database
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
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

        if destination is None:
            destination = self.search_path
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        # Get MAAP access token
        access_token = self._get_access_token()

        zip_path = destination / f"{product_id}.zip"

        try:
            print(f"Downloading {product_id}...")

            session = requests.Session()
            session.headers["Authorization"] = f"Bearer {access_token}"

            response = session.get(remote_url, stream=True, timeout=600)
            response.raise_for_status()

            # Stream to disk
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"\r  {downloaded / 1e6:.1f} / "
                            f"{total / 1e6:.1f} MB ({pct:.0f}%)",
                            end="", flush=True
                        )
            print()

            print(f"Downloaded to {zip_path}")

            # Extract if requested
            product_path = zip_path
            if extract and zipfile.is_zipfile(zip_path):
                extract_dir = destination / product_id
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
                zip_path.unlink()
                product_path = extract_dir
                print(f"Extracted to {product_path}")

            # Update database
            cursor.execute("""
                UPDATE products
                SET local_path = ?, download_date = ?
                WHERE id = ?
            """, (str(product_path), datetime.now().isoformat(), product_id))
            self.conn.commit()

            return product_path

        except requests.RequestException as e:
            if zip_path.exists():
                zip_path.unlink()
            raise RuntimeError(f"Download failed: {e}") from e

    def get_metadata_summary(
        self,
        image_paths: List[Path]
    ) -> List[Dict[str, Any]]:
        """
        Extract metadata summary from multiple BIOMASS products.

        Parameters
        ----------
        image_paths : List[Path]
            List of BIOMASS product paths

        Returns
        -------
        List[Dict[str, Any]]
            List of metadata dictionaries
        """
        summaries = []

        for path in image_paths:
            try:
                from grdl.IO import BIOMASSL1Reader

                with BIOMASSL1Reader(path) as reader:
                    summaries.append({
                        'path': str(path),
                        'product_name': path.name,
                        'format': reader.metadata['format'],
                        'dimensions': (reader.metadata['rows'], reader.metadata['cols']),
                        'polarizations': reader.metadata.get('polarizations', []),
                        'orbit': reader.metadata.get('orbit_number'),
                        'start_time': reader.metadata.get('start_time'),
                    })

            except Exception as e:
                warnings.warn(f"Failed to extract metadata from {path}: {e}")
                summaries.append({'path': str(path), 'error': str(e)})

        return summaries

    def find_overlapping(
        self,
        reference_bounds: Tuple[float, float, float, float],
        image_paths: List[Path]
    ) -> List[Path]:
        """
        Find BIOMASS products that overlap with a reference bounding box.

        Parameters
        ----------
        reference_bounds : Tuple[float, float, float, float]
            Reference bounding box as (min_lon, min_lat, max_lon, max_lat)
        image_paths : List[Path]
            List of BIOMASS product paths to check

        Returns
        -------
        List[Path]
            List of overlapping product paths

        Notes
        -----
        This is a basic implementation using corner coordinates.
        More sophisticated overlap detection can be added.
        """
        overlapping = []
        ref_min_lon, ref_min_lat, ref_max_lon, ref_max_lat = reference_bounds

        for path in image_paths:
            try:
                from grdl.IO import BIOMASSL1Reader

                with BIOMASSL1Reader(path) as reader:
                    corners = reader.metadata.get('corner_coords')

                    if not corners:
                        continue

                    # Extract lat/lon from corners
                    lats = [corners[f'corner{i}'][0] for i in range(1, 5)]
                    lons = [corners[f'corner{i}'][1] for i in range(1, 5)]

                    prod_min_lat, prod_max_lat = min(lats), max(lats)
                    prod_min_lon, prod_max_lon = min(lons), max(lons)

                    # Check for overlap
                    if (prod_min_lon <= ref_max_lon and prod_max_lon >= ref_min_lon and
                        prod_min_lat <= ref_max_lat and prod_max_lat >= ref_min_lat):
                        overlapping.append(path)

            except Exception as e:
                warnings.warn(f"Failed to check overlap for {path}: {e}")

        return overlapping

    def query_database(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        orbit: Optional[int] = None,
        has_local: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Query local database for products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date filter (ISO format)
        end_date : Optional[str], default=None
            End date filter (ISO format)
        orbit : Optional[int], default=None
            Orbit number filter
        has_local : Optional[bool], default=None
            Filter for products with/without local paths

        Returns
        -------
        List[Dict[str, Any]]
            List of products from database
        """
        query = "SELECT * FROM products WHERE 1=1"
        params = []

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

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
