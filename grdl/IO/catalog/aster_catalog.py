# -*- coding: utf-8 -*-
"""
ASTER Catalog - Local discovery and SQLite indexing for ASTER L1T and GDEM
GeoTIFF products.

Provides tools for discovering ASTER products on disk, extracting metadata
via ASTERReader, and tracking collections in a local SQLite database.
Supports offline filtering by date, entity ID, WRS path/row, and subsystem
availability.

Dependencies
------------
rasterio (via ASTERReader)

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


class ASTERCatalog(CatalogInterface):
    """Catalog for ASTER L1T and GDEM GeoTIFF products on local disk.

    Discovers ASTER GeoTIFF files, extracts metadata via ASTERReader,
    and maintains a local SQLite database for offline querying.

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
    >>> from grdl.IO.catalog import ASTERCatalog
    >>> catalog = ASTERCatalog('/data/aster')
    >>> products = catalog.discover_local()
    >>> print(f"Found {len(products)} products")
    >>> results = catalog.query_database(cloud_cover_max=10.0)
    """

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
            db_path = self.search_path / "aster_catalog.db"
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

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()
