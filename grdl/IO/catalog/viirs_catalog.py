# -*- coding: utf-8 -*-
"""
VIIRS Catalog - Local discovery and SQLite indexing for VIIRS HDF5 products
from Suomi NPP, NOAA-20, and NOAA-21.

Provides tools for discovering VIIRS products on disk, extracting metadata
via VIIRSReader, and tracking collections in a local SQLite database.
Supports offline filtering by time, product type, satellite, and day/night.

Dependencies
------------
h5py (via VIIRSReader)

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


class VIIRSCatalog(CatalogInterface):
    """Catalog for VIIRS HDF5 products on local disk.

    Discovers VIIRS HDF5 files (VNP, VJ1, VJ2 prefixes), extracts
    metadata via VIIRSReader, and maintains a local SQLite database for
    offline querying. Corner coordinates are derived from
    ``metadata.geospatial_bounds`` when available.

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
    >>> from grdl.IO.catalog import VIIRSCatalog
    >>> catalog = VIIRSCatalog('/data/viirs')
    >>> products = catalog.discover_local()
    >>> print(f"Found {len(products)} products")
    >>> results = catalog.query_database(product_short_name='VNP46A1',
    ...                                  day_night_flag='Night')
    """

    def __init__(
        self,
        search_path: Union[str, Path],
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize VIIRS catalog.

        Parameters
        ----------
        search_path : Union[str, Path]
            Root directory for local product searches.
        db_path : Optional[Union[str, Path]], default=None
            Path to SQLite database file. If None, uses
            ``search_path/viirs_catalog.db``.

        Raises
        ------
        NotADirectoryError
            If search_path is not a directory.
        """
        super().__init__(search_path)

        if db_path is None:
            db_path = self.search_path / "viirs_catalog.db"
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
                satellite_name TEXT,
                instrument_name TEXT,
                product_short_name TEXT,
                product_long_name TEXT,
                processing_level TEXT,
                collection_version TEXT,
                start_datetime TEXT,
                end_datetime TEXT,
                production_datetime TEXT,
                day_night_flag TEXT,
                granule_id TEXT,
                orbit_number INTEGER,
                horizontal_tile_number TEXT,
                vertical_tile_number TEXT,
                spatial_resolution INTEGER,
                corner_coords TEXT,
                local_path TEXT,
                file_size INTEGER,
                download_date TEXT,
                metadata_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_start_datetime "
            "ON products(start_datetime)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_product_short_name "
            "ON products(product_short_name)"
        )

        self.conn.commit()

    def discover_images(
        self,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Path]:
        """Discover VIIRS HDF5 files in the search path.

        Parameters
        ----------
        extensions : Optional[List[str]], default=None
            Not used; VIIRS files are matched by filename pattern.
        recursive : bool, default=True
            Whether to search subdirectories recursively.

        Returns
        -------
        List[Path]
            List of discovered VIIRS HDF5 file paths.
        """
        return self.discover_local()

    def discover_local(self, update_db: bool = True) -> List[Path]:
        """Discover VIIRS products on local file system.

        Searches for HDF5 files matching VIIRS naming patterns
        (e.g., ``VNP46A1.A2024001.h08v05.001.2024003012345.h5``).

        Parameters
        ----------
        update_db : bool, default=True
            Whether to update the database with discovered products.

        Returns
        -------
        List[Path]
            List of discovered VIIRS HDF5 file paths.
        """
        products = []
        seen = set()

        patterns = [
            "VNP*.h5",
            "VJ1*.h5",
            "VJ2*.h5",
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
            "VIIRS discover_local: found %d products in %s",
            len(products), self.search_path,
        )
        return products

    def _bounds_to_corner_coords(
        self,
        geospatial_bounds: Tuple[float, float, float, float],
    ) -> str:
        """Convert geospatial_bounds to JSON corner coordinates.

        Parameters
        ----------
        geospatial_bounds : Tuple[float, float, float, float]
            Bounding box as ``(lat_min, lat_max, lon_min, lon_max)``.

        Returns
        -------
        str
            JSON-encoded corner list.
        """
        lat_min, lat_max, lon_min, lon_max = geospatial_bounds
        corners = [
            {'lat': lat_max, 'lon': lon_min},  # UL
            {'lat': lat_max, 'lon': lon_max},  # UR
            {'lat': lat_min, 'lon': lon_max},  # LR
            {'lat': lat_min, 'lon': lon_min},  # LL
        ]
        return json.dumps(corners)

    def _index_local_product(self, product_path: Path) -> None:
        """Index a local VIIRS HDF5 product in the database.

        Parameters
        ----------
        product_path : Path
            Path to VIIRS HDF5 file.
        """
        try:
            from grdl.IO.multispectral.viirs import VIIRSReader

            with VIIRSReader(product_path) as reader:
                metadata = reader.metadata

                product_id = product_path.name

                corner_coords = None
                if metadata.geospatial_bounds is not None:
                    corner_coords = self._bounds_to_corner_coords(
                        metadata.geospatial_bounds
                    )

                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO products
                    (id, product_name, product_type, satellite_name,
                     instrument_name, product_short_name, product_long_name,
                     processing_level, collection_version,
                     start_datetime, end_datetime, production_datetime,
                     day_night_flag, granule_id, orbit_number,
                     horizontal_tile_number, vertical_tile_number,
                     spatial_resolution,
                     corner_coords, local_path, file_size,
                     download_date, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    product_id,
                    product_path.name,
                    metadata.product_short_name,
                    metadata.satellite_name,
                    metadata.instrument_name,
                    metadata.product_short_name,
                    metadata.product_long_name,
                    metadata.processing_level,
                    metadata.collection_version,
                    metadata.start_datetime,
                    metadata.end_datetime,
                    metadata.production_datetime,
                    metadata.day_night_flag,
                    metadata.granule_id,
                    metadata.orbit_number,
                    metadata.horizontal_tile_number,
                    metadata.vertical_tile_number,
                    metadata.spatial_resolution,
                    corner_coords,
                    str(product_path),
                    product_path.stat().st_size,
                    datetime.now().isoformat(),
                    json.dumps(metadata.to_dict(), default=str),
                ))
                self.conn.commit()

        except Exception as e:
            warnings.warn(
                f"Failed to index VIIRS product {product_path}: {e}"
            )

    def get_metadata_summary(
        self,
        image_paths: List[Path],
    ) -> List[Dict[str, Any]]:
        """Extract metadata summary from multiple VIIRS products.

        Parameters
        ----------
        image_paths : List[Path]
            List of VIIRS HDF5 file paths.

        Returns
        -------
        List[Dict[str, Any]]
            List of metadata dictionaries.
        """
        summaries = []

        for path in image_paths:
            try:
                from grdl.IO.multispectral.viirs import VIIRSReader

                with VIIRSReader(path) as reader:
                    metadata = reader.metadata

                    summaries.append({
                        'path': str(path),
                        'product_name': path.name,
                        'format': metadata.format,
                        'dimensions': (metadata.rows, metadata.cols),
                        'product_type': metadata.product_short_name,
                        'satellite_name': metadata.satellite_name,
                        'instrument_name': metadata.instrument_name,
                        'product_short_name': metadata.product_short_name,
                        'product_long_name': metadata.product_long_name,
                        'processing_level': metadata.processing_level,
                        'collection_version': metadata.collection_version,
                        'start_datetime': metadata.start_datetime,
                        'end_datetime': metadata.end_datetime,
                        'day_night_flag': metadata.day_night_flag,
                        'orbit_number': metadata.orbit_number,
                        'horizontal_tile_number': metadata.horizontal_tile_number,
                        'vertical_tile_number': metadata.vertical_tile_number,
                        'spatial_resolution': metadata.spatial_resolution,
                        'geospatial_bounds': metadata.geospatial_bounds,
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
        """Find VIIRS products that overlap with a reference bounding box.

        Parameters
        ----------
        reference_bounds : Tuple[float, float, float, float]
            Reference bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
        image_paths : List[Path]
            List of VIIRS HDF5 file paths to check.

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
                    from grdl.IO.multispectral.viirs import VIIRSReader

                    with VIIRSReader(path) as reader:
                        bounds = reader.metadata.geospatial_bounds
                        if bounds is not None:
                            lat_min, lat_max, lon_min, lon_max = bounds
                            lats = [lat_min, lat_max]
                            lons = [lon_min, lon_max]

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
        product_short_name: Optional[str] = None,
        satellite_name: Optional[str] = None,
        day_night_flag: Optional[str] = None,
        processing_level: Optional[str] = None,
        has_local: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Query local database for VIIRS products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date filter (ISO format).
        end_date : Optional[str], default=None
            End date filter (ISO format).
        product_short_name : Optional[str], default=None
            Product short name filter (e.g., ``'VNP46A1'``).
        satellite_name : Optional[str], default=None
            Satellite filter (``'Suomi NPP'``, ``'NOAA-20'``, ``'NOAA-21'``).
        day_night_flag : Optional[str], default=None
            Day/night flag filter (``'Day'``, ``'Night'``, ``'Both'``).
        processing_level : Optional[str], default=None
            Processing level filter (``'L1'``, ``'L2'``, ``'L3'``).
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
            query += " AND start_datetime >= ?"
            params.append(start_date)

        if end_date:
            query += " AND end_datetime <= ?"
            params.append(end_date)

        if product_short_name:
            query += " AND product_short_name = ?"
            params.append(product_short_name)

        if satellite_name:
            query += " AND satellite_name = ?"
            params.append(satellite_name)

        if day_night_flag:
            query += " AND day_night_flag = ?"
            params.append(day_night_flag)

        if processing_level:
            query += " AND processing_level = ?"
            params.append(processing_level)

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
