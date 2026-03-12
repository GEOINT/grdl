# -*- coding: utf-8 -*-
"""
Sentinel-2 Catalog - Local discovery and SQLite indexing for Sentinel-2 MSI
products (SAFE archives and standalone JP2 band files).

Provides tools for discovering Sentinel-2 products on disk, extracting
metadata via Sentinel2Reader, and tracking collections in a local SQLite
database. Supports offline filtering by time, MGRS tile, band, and orbit.

Dependencies
------------
rasterio or glymur (via Sentinel2Reader / JP2Reader)

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
2026-03-11

Modified
--------
2026-03-11
"""

# Standard library
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import sqlite3
import warnings

# GRDL internal
from grdl.IO.base import CatalogInterface

logger = logging.getLogger(__name__)


class Sentinel2Catalog(CatalogInterface):
    """Catalog for Sentinel-2 MSI products on local disk.

    Discovers Sentinel-2 SAFE directories and standalone JP2 band files,
    extracts metadata via Sentinel2Reader, and maintains a local SQLite
    database for offline querying.

    Corner coordinates are derived from the rasterio transform and CRS
    stored in metadata.extras when available; otherwise stored as NULL.

    Attributes
    ----------
    search_path : Path
        Root directory for local product searches.
    db_path : Path
        Path to SQLite database file.
    conn : sqlite3.Connection
        Database connection.

    Examples
    --------
    >>> from grdl.IO.catalog import Sentinel2Catalog
    >>> catalog = Sentinel2Catalog('/data/sentinel2')
    >>> products = catalog.discover_local()
    >>> print(f"Found {len(products)} products")
    >>> results = catalog.query_database(mgrs_tile_id='T10SEG')
    """

    def __init__(
        self,
        search_path: Union[str, Path],
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize Sentinel-2 catalog.

        Parameters
        ----------
        search_path : Union[str, Path]
            Root directory for local product searches.
        db_path : Optional[Union[str, Path]], default=None
            Path to SQLite database file. If None, uses
            ``search_path/sentinel2_catalog.db``.

        Raises
        ------
        NotADirectoryError
            If search_path is not a directory.
        """
        super().__init__(search_path)

        if db_path is None:
            db_path = self.search_path / "sentinel2_catalog.db"
        self.db_path = Path(db_path)

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
                satellite TEXT,
                processing_level TEXT,
                sensing_datetime TEXT,
                product_discriminator TEXT,
                mgrs_tile_id TEXT,
                utm_zone INTEGER,
                latitude_band TEXT,
                relative_orbit INTEGER,
                orbit_direction TEXT,
                band_id TEXT,
                resolution_tier INTEGER,
                wavelength_center REAL,
                wavelength_range TEXT,
                baseline_processing TEXT,
                corner_coords TEXT,
                local_path TEXT,
                file_size INTEGER,
                download_date TEXT,
                metadata_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensing_datetime "
            "ON products(sensing_datetime)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_mgrs_tile_id "
            "ON products(mgrs_tile_id)"
        )

        self.conn.commit()

    def discover_images(
        self,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Path]:
        """Discover Sentinel-2 products in the search path.

        Parameters
        ----------
        extensions : Optional[List[str]], default=None
            Not used; Sentinel-2 products are matched by filename pattern.
        recursive : bool, default=True
            Whether to search subdirectories recursively.

        Returns
        -------
        List[Path]
            List of discovered product paths.
        """
        return self.discover_local()

    def discover_local(self, update_db: bool = True) -> List[Path]:
        """Discover Sentinel-2 products on local file system.

        Searches for SAFE directories and standalone JP2 band files matching
        Sentinel-2 naming patterns.

        Parameters
        ----------
        update_db : bool, default=True
            Whether to update the database with discovered products.

        Returns
        -------
        List[Path]
            List of discovered product paths (SAFE dirs and JP2 files).
        """
        products = []
        seen = set()

        # SAFE archive directories
        safe_patterns = [
            "S2[ABC]_MSI*.SAFE",
            "S2?_MSI*.SAFE",
        ]
        for pattern in safe_patterns:
            for p in self.search_path.rglob(pattern):
                if p.is_dir() and p not in seen:
                    products.append(p)
                    seen.add(p)

        # Standalone JP2 band files (not inside SAFE archives)
        for p in self.search_path.rglob("T*_*_B*.jp2"):
            if p.is_file() and p not in seen:
                # Skip if inside a SAFE directory (already captured above)
                if not any(part.endswith('.SAFE') for part in p.parts):
                    products.append(p)
                    seen.add(p)

        if update_db:
            for product_path in products:
                self._index_local_product(product_path)

        logger.debug(
            "Sentinel-2 discover_local: found %d products in %s",
            len(products), self.search_path,
        )
        return products

    def _compute_corner_coords(self, metadata: Any) -> Optional[str]:
        """Compute corner coordinates from rasterio transform in extras.

        Parameters
        ----------
        metadata : Sentinel2Metadata
            Metadata object with crs and extras containing transform.

        Returns
        -------
        Optional[str]
            JSON-encoded corner coordinates or None.
        """
        try:
            transform = metadata.extras.get('transform') if metadata.extras else None
            if transform is None or metadata.crs is None:
                return None

            rows = metadata.rows
            cols = metadata.cols

            # Four corner (row, col) → (x, y) via affine transform
            corners_xy = [
                transform * (0, 0),
                transform * (cols, 0),
                transform * (cols, rows),
                transform * (0, rows),
            ]

            crs_str = metadata.crs
            if 'EPSG:4326' in crs_str or 'epsg:4326' in crs_str.lower():
                # Already WGS-84: (x=lon, y=lat)
                corners = [
                    {'lon': float(x), 'lat': float(y)} for x, y in corners_xy
                ]
            else:
                # Reproject to WGS-84 using pyproj if available
                try:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(
                        crs_str, 'EPSG:4326', always_xy=True
                    )
                    corners = []
                    for x, y in corners_xy:
                        lon, lat = transformer.transform(x, y)
                        corners.append({'lon': float(lon), 'lat': float(lat)})
                except ImportError:
                    return None

            return json.dumps(corners)

        except Exception:
            return None

    def _index_local_product(self, product_path: Path) -> None:
        """Index a local Sentinel-2 product in the database.

        Parameters
        ----------
        product_path : Path
            Path to SAFE directory or JP2 band file.
        """
        try:
            from grdl.IO.eo.sentinel2 import Sentinel2Reader

            with Sentinel2Reader(product_path) as reader:
                metadata = reader.metadata

                product_id = product_path.name
                corner_coords = self._compute_corner_coords(metadata)

                wavelength_range = None
                if metadata.wavelength_range is not None:
                    wavelength_range = json.dumps(list(metadata.wavelength_range))

                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO products
                    (id, product_name, product_type, satellite,
                     processing_level, sensing_datetime, product_discriminator,
                     mgrs_tile_id, utm_zone, latitude_band,
                     relative_orbit, orbit_direction,
                     band_id, resolution_tier,
                     wavelength_center, wavelength_range,
                     baseline_processing,
                     corner_coords, local_path, file_size,
                     download_date, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?)
                """, (
                    product_id,
                    product_path.name,
                    metadata.product_type,
                    metadata.satellite,
                    metadata.processing_level,
                    metadata.sensing_datetime,
                    metadata.product_discriminator,
                    metadata.mgrs_tile_id,
                    metadata.utm_zone,
                    metadata.latitude_band,
                    metadata.relative_orbit,
                    metadata.orbit_direction,
                    metadata.band_id,
                    metadata.resolution_tier,
                    metadata.wavelength_center,
                    wavelength_range,
                    metadata.baseline_processing,
                    corner_coords,
                    str(product_path),
                    (product_path.stat().st_size if product_path.is_file()
                     else self._get_dir_size(product_path)),
                    datetime.now().isoformat(),
                    json.dumps(metadata.to_dict(), default=str),
                ))
                self.conn.commit()

        except Exception as e:
            warnings.warn(
                f"Failed to index Sentinel-2 product {product_path}: {e}"
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
        """Extract metadata summary from multiple Sentinel-2 products.

        Parameters
        ----------
        image_paths : List[Path]
            List of Sentinel-2 product paths.

        Returns
        -------
        List[Dict[str, Any]]
            List of metadata dictionaries.
        """
        summaries = []

        for path in image_paths:
            try:
                from grdl.IO.eo.sentinel2 import Sentinel2Reader

                with Sentinel2Reader(path) as reader:
                    metadata = reader.metadata

                    summaries.append({
                        'path': str(path),
                        'product_name': path.name,
                        'format': metadata.format,
                        'dimensions': (metadata.rows, metadata.cols),
                        'satellite': metadata.satellite,
                        'product_type': metadata.product_type,
                        'processing_level': metadata.processing_level,
                        'sensing_datetime': metadata.sensing_datetime,
                        'mgrs_tile_id': metadata.mgrs_tile_id,
                        'band_id': metadata.band_id,
                        'resolution_tier': metadata.resolution_tier,
                        'wavelength_center': metadata.wavelength_center,
                        'orbit_direction': metadata.orbit_direction,
                        'relative_orbit': metadata.relative_orbit,
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
        """Find Sentinel-2 products that overlap with a reference bounding box.

        Parameters
        ----------
        reference_bounds : Tuple[float, float, float, float]
            Reference bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
        image_paths : List[Path]
            List of product paths to check.

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
                    "SELECT corner_coords FROM products WHERE id = ?",
                    (path.name,),
                )
                row = cursor.fetchone()

                lats: List[float] = []
                lons: List[float] = []

                if row and row['corner_coords']:
                    corners = json.loads(row['corner_coords'])
                    lats = [c['lat'] for c in corners if 'lat' in c]
                    lons = [c['lon'] for c in corners if 'lon' in c]
                else:
                    from grdl.IO.eo.sentinel2 import Sentinel2Reader

                    with Sentinel2Reader(path) as reader:
                        corner_json = self._compute_corner_coords(reader.metadata)
                        if corner_json:
                            corners = json.loads(corner_json)
                            lats = [c['lat'] for c in corners if 'lat' in c]
                            lons = [c['lon'] for c in corners if 'lon' in c]

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
        mgrs_tile_id: Optional[str] = None,
        processing_level: Optional[str] = None,
        satellite: Optional[str] = None,
        band_id: Optional[str] = None,
        has_local: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Query local database for Sentinel-2 products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date filter (ISO format).
        end_date : Optional[str], default=None
            End date filter (ISO format).
        mgrs_tile_id : Optional[str], default=None
            MGRS tile identifier filter (e.g., ``'T10SEG'``).
        processing_level : Optional[str], default=None
            Processing level filter (``'L1C'`` or ``'L2A'``).
        satellite : Optional[str], default=None
            Satellite filter (``'S2A'``, ``'S2B'``, ``'S2C'``).
        band_id : Optional[str], default=None
            Band identifier filter (``'B04'``, ``'B08'``, etc.).
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
            query += " AND sensing_datetime >= ?"
            params.append(start_date)

        if end_date:
            query += " AND sensing_datetime <= ?"
            params.append(end_date)

        if mgrs_tile_id:
            query += " AND mgrs_tile_id = ?"
            params.append(mgrs_tile_id)

        if processing_level:
            query += " AND processing_level = ?"
            params.append(processing_level)

        if satellite:
            query += " AND satellite = ?"
            params.append(satellite)

        if band_id:
            query += " AND band_id = ?"
            params.append(band_id)

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
