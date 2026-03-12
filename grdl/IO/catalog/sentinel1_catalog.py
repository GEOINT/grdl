# -*- coding: utf-8 -*-
"""
Sentinel-1 SLC Catalog - Local discovery and SQLite indexing for Sentinel-1
IW/EW/SM SLC SAFE products.

Provides tools for discovering Sentinel-1 SLC products on disk, extracting
metadata via Sentinel1SLCReader, and tracking collections in a local SQLite
database. Supports offline filtering by time, orbit, polarization, and swath.

Dependencies
------------
rasterio (via Sentinel1SLCReader)

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


class Sentinel1SLCCatalog(CatalogInterface):
    """Catalog for Sentinel-1 IW/EW/SM SLC SAFE products on local disk.

    Discovers Sentinel-1 SLC SAFE directories, extracts metadata via
    Sentinel1SLCReader, and maintains a local SQLite database for
    offline querying.

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
    >>> from grdl.IO.catalog import Sentinel1SLCCatalog
    >>> catalog = Sentinel1SLCCatalog('/data/sentinel1')
    >>> products = catalog.discover_local()
    >>> print(f"Found {len(products)} products")
    >>> results = catalog.query_database(orbit_pass='ASCENDING')
    """

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
            db_path = self.search_path / "sentinel1_catalog.db"
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
                    samples = [p.sample for p in grid]
                    min_line = min(lines)
                    max_line = max(lines)
                    min_sample = min(samples)
                    max_sample = max(samples)

                    corners = []
                    for target_line, target_sample in [
                        (min_line, min_sample),
                        (min_line, max_sample),
                        (max_line, max_sample),
                        (max_line, min_sample),
                    ]:
                        best = min(
                            grid,
                            key=lambda p, tl=target_line, ts=target_sample: (
                                abs(p.line - tl) + abs(p.sample - ts)
                            ),
                        )
                        corners.append([best.latitude, best.longitude])
                    corner_coords = json.dumps(corners)

                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO products
                    (id, product_name, product_type, mission, mode,
                     polarization, absolute_orbit, relative_orbit, orbit_pass,
                     start_time, stop_time, swath, corner_coords,
                     local_path, file_size, download_date, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()
