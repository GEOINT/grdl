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
2026-01-30
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
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


class BIOMASSCatalog(CatalogInterface):
    """
    Catalog and download manager for BIOMASS ESA satellite products.

    Provides integrated capabilities for:
    - Local file system discovery of BIOMASS products
    - Remote querying of ESA data hubs
    - Product download from ESA
    - SQLite database tracking of products

    Attributes
    ----------
    search_path : Path
        Root directory for local product searches
    db_path : Path
        Path to SQLite database file for product tracking
    conn : sqlite3.Connection
        Database connection

    Examples
    --------
    >>> catalog = BIOMASSCatalog('/data/biomass', db_path='biomass_catalog.db')
    >>>
    >>> # Discover local products
    >>> local_products = catalog.discover_local()
    >>> print(f"Found {len(local_products)} local products")
    >>>
    >>> # Query ESA for products in date range
    >>> remote_products = catalog.query_esa(
    ...     start_date='2025-11-01',
    ...     end_date='2025-11-30'
    ... )
    >>>
    >>> # Download a product
    >>> catalog.download_product(remote_products[0]['id'])
    """

    # ESA Data Hub API endpoint (update with actual BIOMASS endpoint when available)
    ESA_API_BASE = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/BIOMASS/search.json"

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

    def query_esa(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        orbit: Optional[int] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query ESA data hub for available BIOMASS products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date in ISO format (YYYY-MM-DD)
        end_date : Optional[str], default=None
            End date in ISO format (YYYY-MM-DD)
        bbox : Optional[Tuple[float, float, float, float]], default=None
            Bounding box as (min_lon, min_lat, max_lon, max_lat)
        orbit : Optional[int], default=None
            Orbit number to search for
        max_results : int, default=100
            Maximum number of results to return

        Returns
        -------
        List[Dict[str, Any]]
            List of product metadata dictionaries from ESA

        Raises
        ------
        ImportError
            If requests library is not installed
        RuntimeError
            If API query fails

        Notes
        -----
        This is a placeholder implementation. The actual ESA API endpoint
        for BIOMASS may differ. Update ESA_API_BASE when official API is
        available.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests library required for ESA queries. Install with: pip install requests")

        # Build query parameters
        params = {
            'maxRecords': max_results,
            'productType': 'SCS',  # L1 SCS products
        }

        if start_date and end_date:
            params['startDate'] = start_date
            params['completionDate'] = end_date

        if bbox:
            params['box'] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        if orbit:
            params['orbitNumber'] = orbit

        try:
            response = requests.get(self.ESA_API_BASE, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Extract products from response
            # Note: Actual response format depends on ESA API
            products = data.get('features', [])

            # Store query results in database
            for product in products:
                self._index_remote_product(product)

            return products

        except requests.RequestException as e:
            raise RuntimeError(f"ESA API query failed: {e}") from e

    def _index_remote_product(self, product_data: Dict[str, Any]) -> None:
        """
        Index a remote product in the database.

        Parameters
        ----------
        product_data : Dict[str, Any]
            Product metadata from ESA API
        """
        try:
            # Extract fields from ESA response
            # Note: Field names depend on actual ESA API response format
            properties = product_data.get('properties', {})
            product_id = properties.get('id', product_data.get('id'))

            cursor = self.conn.cursor()

            cursor.execute("""
                INSERT OR IGNORE INTO products
                (id, product_name, product_type, processing_level,
                 orbit_number, start_time, stop_time, remote_url, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                product_id,
                properties.get('title', ''),
                properties.get('productType', 'SCS'),
                properties.get('processingLevel', 'L1'),
                properties.get('orbitNumber', 0),
                properties.get('startDate', ''),
                properties.get('completionDate', ''),
                product_data.get('downloadUrl', ''),
                json.dumps(product_data)
            ))

            self.conn.commit()

        except Exception as e:
            warnings.warn(f"Failed to index remote product: {e}")

    def download_product(
        self,
        product_id: str,
        destination: Optional[Path] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> Path:
        """
        Download a BIOMASS product from ESA.

        Parameters
        ----------
        product_id : str
            Product ID to download
        destination : Optional[Path], default=None
            Destination directory. If None, uses search_path
        username : Optional[str], default=None
            ESA account username for authentication
        password : Optional[str], default=None
            ESA account password for authentication

        Returns
        -------
        Path
            Path to downloaded product directory

        Raises
        ------
        ValueError
            If product not found in database
        ImportError
            If requests library not installed
        RuntimeError
            If download fails

        Notes
        -----
        This is a placeholder implementation. Actual download may require
        authentication and ESA-specific download protocols.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests library required for downloads. Install with: pip install requests")

        # Get product info from database
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
        row = cursor.fetchone()

        if not row:
            raise ValueError(f"Product {product_id} not found in catalog")

        remote_url = row['remote_url']
        if not remote_url:
            raise ValueError(f"No download URL available for product {product_id}")

        # Set destination
        if destination is None:
            destination = self.search_path

        # Download product
        # Note: Actual implementation depends on ESA download API
        try:
            print(f"Downloading {product_id}...")

            # Authenticate if credentials provided
            session = requests.Session()
            if username and password:
                session.auth = (username, password)

            # Download (simplified - actual implementation may need chunked download)
            response = session.get(remote_url, stream=True, timeout=300)
            response.raise_for_status()

            # Save to destination
            product_path = destination / f"{product_id}.zip"
            with open(product_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded to {product_path}")

            # Update database
            cursor.execute("""
                UPDATE products
                SET local_path = ?, download_date = ?
                WHERE id = ?
            """, (str(product_path), datetime.now().isoformat(), product_id))
            self.conn.commit()

            return product_path

        except requests.RequestException as e:
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
