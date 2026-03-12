# -*- coding: utf-8 -*-
"""
NISAR Catalog - Discovery, indexing, remote query, and download for NASA
NISAR RSLC and GSLC HDF5 products.

Provides tools for discovering NISAR products on disk, querying NASA
Earthdata CMR for available granules, downloading products via ASF DAAC
with Earthdata Login authentication, and tracking collections in a local
SQLite database. Supports offline filtering by time, orbit, radar band,
and polarization.

Dependencies
------------
h5py (via NISARReader)
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

logger = logging.getLogger(__name__)


class NISARCatalog(CatalogInterface):
    """Catalog and download manager for NASA NISAR RSLC and GSLC products.

    Provides integrated capabilities for:

    - Local file system discovery of NISAR HDF5 products
    - Remote querying via NASA Earthdata CMR
    - Product download via ASF DAAC with Earthdata Login authentication
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
    >>> from grdl.IO.catalog import NISARCatalog
    >>> catalog = NISARCatalog('/data/nisar')
    >>> products = catalog.discover_local()
    >>> print(f"Found {len(products)} products")
    >>>
    >>> # Search Earthdata for L-band RSLC products
    >>> remote = catalog.query_earthdata(
    ...     bbox=(-120.0, 34.0, -118.0, 36.0),
    ...     product_type='RSLC',
    ...     radar_band='LSAR',
    ... )
    >>> catalog.download_product(remote[0]['umm']['GranuleUR'])
    """

    # NASA CMR API for granule search
    CMR_SEARCH_URL = (
        "https://cmr.earthdata.nasa.gov/search/granules.json"
    )

    # CMR collection short names for NISAR products
    COLLECTION_RSLC = "NISAR_L1_RSLC"
    COLLECTION_GSLC = "NISAR_L1_GSLC"

    def __init__(
        self,
        search_path: Union[str, Path],
        db_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize NISAR catalog.

        Parameters
        ----------
        search_path : Union[str, Path]
            Root directory for local product searches.
        db_path : Optional[Union[str, Path]], default=None
            Path to SQLite database file. If None, uses
            ``search_path/nisar_catalog.db``.

        Raises
        ------
        NotADirectoryError
            If search_path is not a directory.
        """
        super().__init__(search_path)

        if db_path is None:
            db_path = self.search_path / "nisar_catalog.db"
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
                mission_id TEXT,
                radar_band TEXT,
                frequency TEXT,
                polarization TEXT,
                available_frequencies TEXT,
                available_polarizations TEXT,
                look_direction TEXT,
                orbit_pass_direction TEXT,
                absolute_orbit_number INTEGER,
                track_number INTEGER,
                frame_number INTEGER,
                granule_id TEXT,
                start_time TEXT,
                stop_time TEXT,
                bounding_polygon TEXT,
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
            "CREATE INDEX IF NOT EXISTS idx_absolute_orbit_number "
            "ON products(absolute_orbit_number)"
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
        """Discover NISAR HDF5 files in the search path.

        Parameters
        ----------
        extensions : Optional[List[str]], default=None
            Not used; NISAR files are matched by filename pattern.
        recursive : bool, default=True
            Whether to search subdirectories recursively.

        Returns
        -------
        List[Path]
            List of discovered NISAR HDF5 file paths.
        """
        return self.discover_local()

    def discover_local(self, update_db: bool = True) -> List[Path]:
        """Discover NISAR products on local file system.

        Searches for HDF5 files matching NISAR naming patterns
        (e.g., ``NISAR_L0_PR_RRSD_*.h5``).

        Parameters
        ----------
        update_db : bool, default=True
            Whether to update the database with discovered products.

        Returns
        -------
        List[Path]
            List of discovered NISAR HDF5 file paths.
        """
        products = []

        patterns = [
            "NISAR_L*.h5",
            "NISAR_S*.h5",
            "NISAR_*.h5",
        ]

        seen = set()
        for pattern in patterns:
            found = list(self.search_path.rglob(pattern))
            for p in found:
                if p.is_file() and p not in seen:
                    products.append(p)
                    seen.add(p)

        if update_db:
            for product_path in products:
                self._index_local_product(product_path)

        logger.debug(
            "NISAR discover_local: found %d products in %s",
            len(products), self.search_path,
        )
        return products

    def _index_local_product(self, product_path: Path) -> None:
        """Index a local NISAR HDF5 product in the database.

        Parameters
        ----------
        product_path : Path
            Path to NISAR HDF5 file.
        """
        try:
            from grdl.IO.sar.nisar import NISARReader

            with NISARReader(product_path) as reader:
                metadata = reader.metadata
                ident = metadata.identification

                product_id = product_path.name

                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO products
                    (id, product_name, product_type, mission_id, radar_band,
                     frequency, polarization, available_frequencies,
                     available_polarizations, look_direction,
                     orbit_pass_direction, absolute_orbit_number,
                     track_number, frame_number, granule_id,
                     start_time, stop_time, bounding_polygon,
                     corner_coords, local_path, file_size,
                     download_date, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    product_id,
                    product_path.name,
                    metadata.product_type,
                    getattr(ident, 'mission_id', None) if ident else None,
                    metadata.radar_band,
                    metadata.frequency,
                    metadata.polarization,
                    json.dumps(metadata.available_frequencies or []),
                    json.dumps(metadata.available_polarizations or []),
                    getattr(ident, 'look_direction', None) if ident else None,
                    getattr(ident, 'orbit_pass_direction', None) if ident else None,
                    getattr(ident, 'absolute_orbit_number', None) if ident else None,
                    getattr(ident, 'track_number', None) if ident else None,
                    getattr(ident, 'frame_number', None) if ident else None,
                    getattr(ident, 'granule_id', None) if ident else None,
                    getattr(ident, 'zero_doppler_start_time', None) if ident else None,
                    getattr(ident, 'zero_doppler_end_time', None) if ident else None,
                    getattr(ident, 'bounding_polygon', None) if ident else None,
                    None,  # corner_coords always NULL for NISAR
                    str(product_path),
                    product_path.stat().st_size,
                    datetime.now().isoformat(),
                    json.dumps(metadata.to_dict(), default=str),
                ))
                self.conn.commit()

        except Exception as e:
            warnings.warn(
                f"Failed to index NISAR product {product_path}: {e}"
            )

    def get_metadata_summary(
        self,
        image_paths: List[Path],
    ) -> List[Dict[str, Any]]:
        """Extract metadata summary from multiple NISAR products.

        Parameters
        ----------
        image_paths : List[Path]
            List of NISAR HDF5 file paths.

        Returns
        -------
        List[Dict[str, Any]]
            List of metadata dictionaries.
        """
        summaries = []

        for path in image_paths:
            try:
                from grdl.IO.sar.nisar import NISARReader

                with NISARReader(path) as reader:
                    metadata = reader.metadata
                    ident = metadata.identification

                    summaries.append({
                        'path': str(path),
                        'product_name': path.name,
                        'format': metadata.format,
                        'dimensions': (metadata.rows, metadata.cols),
                        'product_type': metadata.product_type,
                        'radar_band': metadata.radar_band,
                        'frequency': metadata.frequency,
                        'polarization': metadata.polarization,
                        'available_frequencies': metadata.available_frequencies,
                        'available_polarizations': metadata.available_polarizations,
                        'mission_id': getattr(ident, 'mission_id', None) if ident else None,
                        'look_direction': getattr(ident, 'look_direction', None) if ident else None,
                        'orbit_pass_direction': getattr(ident, 'orbit_pass_direction', None) if ident else None,
                        'absolute_orbit_number': getattr(ident, 'absolute_orbit_number', None) if ident else None,
                        'granule_id': getattr(ident, 'granule_id', None) if ident else None,
                        'start_time': getattr(ident, 'zero_doppler_start_time', None) if ident else None,
                        'bounding_polygon': getattr(ident, 'bounding_polygon', None) if ident else None,
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
        """Find NISAR products that overlap with a reference bounding box.

        Uses the bounding polygon from the database for spatial comparison.
        Falls back to reading the HDF5 file if not indexed.

        Parameters
        ----------
        reference_bounds : Tuple[float, float, float, float]
            Reference bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
        image_paths : List[Path]
            List of NISAR HDF5 file paths to check.

        Returns
        -------
        List[Path]
            List of overlapping product paths.
        """
        import re

        overlapping = []
        ref_min_lon, ref_min_lat, ref_max_lon, ref_max_lat = reference_bounds

        for path in image_paths:
            try:
                # Check database for bounding polygon
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT bounding_polygon FROM products WHERE id = ?",
                    (path.name,),
                )
                row = cursor.fetchone()

                wkt_polygon: Optional[str] = None
                if row and row['bounding_polygon']:
                    wkt_polygon = row['bounding_polygon']
                else:
                    from grdl.IO.sar.nisar import NISARReader

                    with NISARReader(path) as reader:
                        ident = reader.metadata.identification
                        if ident:
                            wkt_polygon = getattr(ident, 'bounding_polygon', None)

                if not wkt_polygon:
                    continue

                # Extract lon/lat pairs from WKT POLYGON string
                # e.g. POLYGON((lon lat, lon lat, ...))
                coords_str = re.findall(
                    r'[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+', wkt_polygon
                )
                if not coords_str:
                    continue

                lons: List[float] = []
                lats: List[float] = []
                for pair in coords_str:
                    parts = pair.split()
                    if len(parts) == 2:
                        lons.append(float(parts[0]))
                        lats.append(float(parts[1]))

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
        radar_band: Optional[str] = None,
        orbit_pass_direction: Optional[str] = None,
        absolute_orbit_number: Optional[int] = None,
        product_type: Optional[str] = None,
        polarization: Optional[str] = None,
        has_local: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Query local database for NISAR products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date filter (ISO format).
        end_date : Optional[str], default=None
            End date filter (ISO format).
        radar_band : Optional[str], default=None
            Radar band filter (``'LSAR'`` or ``'SSAR'``).
        orbit_pass_direction : Optional[str], default=None
            Orbit pass direction (``'ascending'`` or ``'descending'``).
        absolute_orbit_number : Optional[int], default=None
            Absolute orbit number filter.
        product_type : Optional[str], default=None
            Product type filter (``'RSLC'`` or ``'GSLC'``).
        polarization : Optional[str], default=None
            Polarization filter (``'HH'``, ``'HV'``, ``'VH'``, ``'VV'``).
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

        if radar_band:
            query += " AND radar_band = ?"
            params.append(radar_band)

        if orbit_pass_direction:
            query += " AND orbit_pass_direction = ?"
            params.append(orbit_pass_direction)

        if absolute_orbit_number is not None:
            query += " AND absolute_orbit_number = ?"
            params.append(absolute_orbit_number)

        if product_type:
            query += " AND product_type = ?"
            params.append(product_type)

        if polarization:
            query += " AND polarization = ?"
            params.append(polarization)

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
        radar_band: Optional[str] = None,
        product_type: Optional[str] = None,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query NASA Earthdata CMR for NISAR granules.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date in ISO format (YYYY-MM-DD).
        end_date : Optional[str], default=None
            End date in ISO format (YYYY-MM-DD).
        bbox : Optional[Tuple[float, float, float, float]], default=None
            Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
        radar_band : Optional[str], default=None
            Radar band filter (``'LSAR'`` or ``'SSAR'``).
        product_type : Optional[str], default=None
            Product type filter (``'RSLC'`` or ``'GSLC'``).
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

        # Select collection by product type
        if product_type and product_type.upper() == "GSLC":
            short_name = self.COLLECTION_GSLC
        else:
            short_name = self.COLLECTION_RSLC

        params: Dict[str, Any] = {
            "short_name": short_name,
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
                "Earthdata CMR returned %d NISAR granules",
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
                if href.endswith(".h5") or "data#" in link.get("rel", ""):
                    remote_url = href
                    break

            # Time range
            time_start = granule.get("time_start", "")
            time_end = granule.get("time_end", "")

            # Bounding box
            bounding_polygon = None
            if "boxes" in granule:
                bounding_polygon = granule["boxes"][0]

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO products
                (id, product_name, product_type,
                 start_time, stop_time, bounding_polygon,
                 remote_url, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                granule_id,
                title,
                granule.get("collection_concept_id", ""),
                time_start,
                time_end,
                bounding_polygon,
                remote_url,
                json.dumps(granule),
            ))
            self.conn.commit()

        except Exception as e:
            warnings.warn(f"Failed to index remote NISAR granule: {e}")

    def download_product(
        self,
        product_id: str,
        destination: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Download a NISAR HDF5 product from ASF DAAC via Earthdata.

        Uses Earthdata Login credentials from
        ``~/.config/geoint/credentials.json`` (``nasa_earthdata`` block).

        Parameters
        ----------
        product_id : str
            Granule ID or title. Must exist in the local database
            (run ``query_earthdata`` first).
        destination : Optional[Union[str, Path]], default=None
            Directory to save the product. If None, uses ``search_path``.

        Returns
        -------
        Path
            Path to downloaded HDF5 file.

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

        # NISAR products are HDF5 files — no extraction needed
        filename = remote_url.split("/")[-1] or f"{product_id}.h5"
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
