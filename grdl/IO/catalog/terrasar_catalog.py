# -*- coding: utf-8 -*-
"""
TerraSAR-X / TanDEM-X Catalog - Local discovery and SQLite indexing for
TerraSAR-X and TanDEM-X SAR products.

Provides tools for discovering TSX/TDX products on disk, extracting metadata
via TerraSARReader, and tracking collections in a local SQLite database.
Supports offline filtering by time, orbit, polarization, and imaging mode.

Dependencies
------------
rasterio (for MGD/GEC/EEC products via TerraSARReader)

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

# GRDL internal
from grdl.IO.base import CatalogInterface
from grdl.IO.sar.terrasar import _find_main_xml

logger = logging.getLogger(__name__)


class TerraSARCatalog(CatalogInterface):
    """Catalog for TerraSAR-X and TanDEM-X SAR products on local disk.

    Discovers TSX/TDX product directories, extracts metadata via
    TerraSARReader, and maintains a local SQLite database for offline
    querying.

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
    >>> from grdl.IO.catalog import TerraSARCatalog
    >>> catalog = TerraSARCatalog('/data/terrasar')
    >>> products = catalog.discover_local()
    >>> print(f"Found {len(products)} products")
    >>> results = catalog.query_database(imaging_mode='SM')
    """

    def __init__(
        self,
        search_path: Union[str, Path],
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize TerraSAR-X / TanDEM-X catalog.

        Parameters
        ----------
        search_path : Union[str, Path]
            Root directory for local product searches.
        db_path : Optional[Union[str, Path]], default=None
            Path to SQLite database file. If None, uses
            ``search_path/terrasar_catalog.db``.

        Raises
        ------
        NotADirectoryError
            If search_path is not a directory.
        """
        super().__init__(search_path)

        if db_path is None:
            db_path = self.search_path / "terrasar_catalog.db"
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
                mission TEXT,
                satellite TEXT,
                imaging_mode TEXT,
                look_direction TEXT,
                orbit_direction TEXT,
                polarization_mode TEXT,
                polarization_list TEXT,
                absolute_orbit INTEGER,
                start_time TEXT,
                stop_time TEXT,
                scene_center_lat REAL,
                scene_center_lon REAL,
                incidence_angle_near REAL,
                incidence_angle_far REAL,
                corner_coords TEXT,
                local_path TEXT,
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

    def discover_images(
        self,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Path]:
        """Discover TerraSAR-X / TanDEM-X product directories.

        Parameters
        ----------
        extensions : Optional[List[str]], default=None
            Not used for TerraSAR-X (directories, not files).
        recursive : bool, default=True
            Whether to search subdirectories recursively.

        Returns
        -------
        List[Path]
            List of discovered product directory paths.
        """
        return self.discover_local()

    def discover_local(self, update_db: bool = True) -> List[Path]:
        """Discover TerraSAR-X / TanDEM-X products on local file system.

        Searches for directories matching TSX/TDX product naming patterns
        (e.g., ``TSX1_SAR__SSC_...``, ``TDX1_SAR__SSC_...``).

        Parameters
        ----------
        update_db : bool, default=True
            Whether to update the database with discovered products.

        Returns
        -------
        List[Path]
            List of discovered product paths.
        """
        products = []

        patterns = [
            "TSX1_SAR*",
            "TDX1_SAR*",
        ]

        seen = set()
        for pattern in patterns:
            found = list(self.search_path.rglob(pattern))
            for p in found:
                if p.is_dir() and p not in seen:
                    # Only include directories that contain a valid
                    # annotation XML — avoids indexing auxiliary
                    # subdirectories (e.g. iif/) that match the name
                    # pattern but lack product metadata.
                    if _find_main_xml(p) is None:
                        continue
                    products.append(p)
                    seen.add(p)

        if update_db:
            for product_path in products:
                self._index_local_product(product_path)

        logger.debug(
            "TerraSAR discover_local: found %d products in %s",
            len(products), self.search_path,
        )
        return products

    def _index_local_product(self, product_path: Path) -> None:
        """Index a local TSX/TDX product in the database.

        Parameters
        ----------
        product_path : Path
            Path to TerraSAR-X / TanDEM-X product directory.
        """
        try:
            from grdl.IO.sar.terrasar import TerraSARReader

            with TerraSARReader(product_path) as reader:
                metadata = reader.metadata
                pi = metadata.product_info
                si = metadata.scene_info

                product_id = product_path.name

                # Corner coords from scene_extent (list of LatLonHAE)
                corner_coords = None
                if si and si.scene_extent:
                    corners = [
                        {
                            'lat': getattr(p, 'lat', None),
                            'lon': getattr(p, 'lon', None),
                            'hae': getattr(p, 'hae', None),
                        }
                        for p in si.scene_extent
                    ]
                    corner_coords = json.dumps(corners)

                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO products
                    (id, product_name, product_type, mission, satellite,
                     imaging_mode, look_direction, orbit_direction,
                     polarization_mode, polarization_list,
                     absolute_orbit, start_time, stop_time,
                     scene_center_lat, scene_center_lon,
                     incidence_angle_near, incidence_angle_far,
                     corner_coords, local_path, file_size,
                     download_date, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?)
                """, (
                    product_id,
                    product_path.name,
                    getattr(pi, 'product_type', None) if pi else None,
                    getattr(pi, 'mission', None) if pi else None,
                    getattr(pi, 'satellite', None) if pi else None,
                    getattr(pi, 'imaging_mode', None) if pi else None,
                    getattr(pi, 'look_direction', None) if pi else None,
                    getattr(pi, 'orbit_direction', None) if pi else None,
                    getattr(pi, 'polarization_mode', None) if pi else None,
                    json.dumps(getattr(pi, 'polarization_list', None) or []) if pi else None,
                    getattr(pi, 'absolute_orbit', None) if pi else None,
                    getattr(pi, 'start_time_utc', None) if pi else None,
                    getattr(pi, 'stop_time_utc', None) if pi else None,
                    getattr(si, 'center_lat', None) if si else None,
                    getattr(si, 'center_lon', None) if si else None,
                    getattr(si, 'incidence_angle_near', None) if si else None,
                    getattr(si, 'incidence_angle_far', None) if si else None,
                    corner_coords,
                    str(product_path),
                    self._get_dir_size(product_path),
                    datetime.now().isoformat(),
                    json.dumps(metadata.to_dict(), default=str),
                ))
                self.conn.commit()

        except Exception as e:
            warnings.warn(
                f"Failed to index TerraSAR product {product_path}: {e}"
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
        """Extract metadata summary from multiple TerraSAR products.

        Parameters
        ----------
        image_paths : List[Path]
            List of TerraSAR-X / TanDEM-X product paths.

        Returns
        -------
        List[Dict[str, Any]]
            List of metadata dictionaries.
        """
        summaries = []

        for path in image_paths:
            try:
                from grdl.IO.sar.terrasar import TerraSARReader

                with TerraSARReader(path) as reader:
                    metadata = reader.metadata
                    pi = metadata.product_info
                    si = metadata.scene_info

                    summaries.append({
                        'path': str(path),
                        'product_name': path.name,
                        'format': metadata.format,
                        'dimensions': (metadata.rows, metadata.cols),
                        'mission': getattr(pi, 'mission', None) if pi else None,
                        'satellite': getattr(pi, 'satellite', None) if pi else None,
                        'product_type': getattr(pi, 'product_type', None) if pi else None,
                        'imaging_mode': getattr(pi, 'imaging_mode', None) if pi else None,
                        'look_direction': getattr(pi, 'look_direction', None) if pi else None,
                        'orbit_direction': getattr(pi, 'orbit_direction', None) if pi else None,
                        'polarization_list': getattr(pi, 'polarization_list', None) if pi else None,
                        'absolute_orbit': getattr(pi, 'absolute_orbit', None) if pi else None,
                        'start_time': getattr(pi, 'start_time_utc', None) if pi else None,
                        'scene_center_lat': getattr(si, 'center_lat', None) if si else None,
                        'scene_center_lon': getattr(si, 'center_lon', None) if si else None,
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
        """Find TerraSAR products that overlap with a reference bounding box.

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
                # Check database first
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
                    lats = [c['lat'] for c in corners if c.get('lat') is not None]
                    lons = [c['lon'] for c in corners if c.get('lon') is not None]
                elif row and row['scene_center_lat'] is not None:
                    # Approximate with just center point
                    lats = [row['scene_center_lat']]
                    lons = [row['scene_center_lon']]
                else:
                    from grdl.IO.sar.terrasar import TerraSARReader

                    with TerraSARReader(path) as reader:
                        si = reader.metadata.scene_info
                        if si and si.scene_extent:
                            lats = [
                                getattr(p, 'lat', None)
                                for p in si.scene_extent
                                if getattr(p, 'lat', None) is not None
                            ]
                            lons = [
                                getattr(p, 'lon', None)
                                for p in si.scene_extent
                                if getattr(p, 'lon', None) is not None
                            ]

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
        orbit_direction: Optional[str] = None,
        absolute_orbit: Optional[int] = None,
        imaging_mode: Optional[str] = None,
        product_type: Optional[str] = None,
        has_local: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Query local database for TerraSAR products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date filter (ISO format).
        end_date : Optional[str], default=None
            End date filter (ISO format).
        orbit_direction : Optional[str], default=None
            Orbit direction filter (``'ASCENDING'`` or ``'DESCENDING'``).
        absolute_orbit : Optional[int], default=None
            Absolute orbit number filter.
        imaging_mode : Optional[str], default=None
            Imaging mode filter (``'SM'``, ``'HS'``, ``'SL'``, etc.).
        product_type : Optional[str], default=None
            Product type filter (``'SSC'``, ``'MGD'``, ``'GEC'``, ``'EEC'``).
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

        if orbit_direction:
            query += " AND orbit_direction = ?"
            params.append(orbit_direction)

        if absolute_orbit is not None:
            query += " AND absolute_orbit = ?"
            params.append(absolute_orbit)

        if imaging_mode:
            query += " AND imaging_mode = ?"
            params.append(imaging_mode)

        if product_type:
            query += " AND product_type = ?"
            params.append(product_type)

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
