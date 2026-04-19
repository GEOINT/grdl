# -*- coding: utf-8 -*-
"""
Sentinel-2 Catalog - Discovery, indexing, remote query, and download for
Sentinel-2 MSI products (SAFE archives and standalone JP2 band files).

Provides tools for discovering Sentinel-2 products on disk, querying the
Copernicus Data Space Ecosystem (CDSE) STAC catalog, downloading products
with OAuth2 authentication, and tracking collections in a local SQLite
database. Supports offline filtering by time, MGRS tile, band, and orbit.

Dependencies
------------
rasterio or glymur (via Sentinel2Reader / JP2Reader)
requests (for CDSE remote queries and downloads)

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

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
2026-03-13
"""

# Standard library
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import re
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
    _normalize_stac_datetime,
    download_file,
    get_cdse_token,
    load_credentials,
)
from grdl.exceptions import DependencyError, ProcessorError, ValidationError

logger = logging.getLogger(__name__)


class Sentinel2Catalog(CatalogInterface):
    """Catalog and download manager for Sentinel-2 MSI products.

    Provides integrated capabilities for:

    - Local file system discovery of Sentinel-2 SAFE/JP2 products
    - Remote querying via Copernicus Data Space Ecosystem (CDSE) STAC
    - Product download with CDSE OAuth2 authentication
    - SQLite database tracking of products

    Credentials are loaded from ``~/.config/geoint/credentials.json``
    (provider block ``esa_copernicus``).

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
    >>> from grdl.IO.catalog import Sentinel2Catalog
    >>> catalog = Sentinel2Catalog('/data/sentinel2')
    >>> products = catalog.discover_local()
    >>> print(f"Found {len(products)} products")
    >>>
    >>> # Search CDSE for a specific MGRS tile
    >>> remote = catalog.query_cdse(
    ...     bbox=(12.0, 41.0, 13.0, 42.0),
    ...     processing_level='S2MSI1C',
    ...     max_results=10,
    ... )
    >>> catalog.download_product(remote[0]['id'])
    """

    # Copernicus Data Space Ecosystem OData API
    CDSE_ODATA_URL = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    )
    CDSE_DOWNLOAD_URL = (
        "https://zipper.dataspace.copernicus.eu/odata/v1/Products"
    )
    CDSE_COLLECTION = "SENTINEL-2"

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
            db_path = Path.home() / ".config" / "geoint" / "catalogs" / "sentinel2.db"
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
                remote_url TEXT,
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
            Metadata object with crs and transform attributes.

        Returns
        -------
        Optional[str]
            JSON-encoded corner coordinates or None.
        """
        try:
            transform = getattr(metadata, 'transform', None)
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
                    INSERT INTO products
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
                    ON CONFLICT(product_name) DO UPDATE SET
                        product_type = excluded.product_type,
                        satellite = excluded.satellite,
                        processing_level = excluded.processing_level,
                        sensing_datetime = excluded.sensing_datetime,
                        product_discriminator = excluded.product_discriminator,
                        mgrs_tile_id = excluded.mgrs_tile_id,
                        utm_zone = excluded.utm_zone,
                        latitude_band = excluded.latitude_band,
                        relative_orbit = excluded.relative_orbit,
                        orbit_direction = excluded.orbit_direction,
                        band_id = excluded.band_id,
                        resolution_tier = excluded.resolution_tier,
                        wavelength_center = excluded.wavelength_center,
                        wavelength_range = excluded.wavelength_range,
                        baseline_processing = excluded.baseline_processing,
                        corner_coords = excluded.corner_coords,
                        local_path = excluded.local_path,
                        file_size = excluded.file_size,
                        download_date = excluded.download_date,
                        metadata_json = excluded.metadata_json
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

    # ── Remote (CDSE) ──────────────────────────────────────────────────

    def query_cdse(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        mgrs_tile_id: Optional[str] = None,
        processing_level: Optional[str] = None,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query Copernicus Data Space Ecosystem for Sentinel-2 products.

        Parameters
        ----------
        start_date : Optional[str], default=None
            Start date in ISO format (YYYY-MM-DD).
        end_date : Optional[str], default=None
            End date in ISO format (YYYY-MM-DD).
        bbox : Optional[Tuple[float, float, float, float]], default=None
            Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
        mgrs_tile_id : Optional[str], default=None
            MGRS tile filter (e.g., ``'T10SEG'``).
        processing_level : Optional[str], default=None
            CDSE product type filter (``'S2MSI1C'``, ``'S2MSI2A'``).
        max_results : int, default=50
            Maximum number of results.

        Returns
        -------
        List[Dict[str, Any]]
            List of STAC feature dicts from CDSE.

        Raises
        ------
        DependencyError
            If ``requests`` is not installed.
        ProcessorError
            If STAC query fails.
        """
        if not REQUESTS_AVAILABLE:
            raise DependencyError(
                "Requests library required for CDSE queries. "
                "Install with: pip install requests"
            )

        # Validate string inputs to prevent OData injection
        if processing_level and not re.fullmatch(
            r"[A-Za-z0-9_]+", processing_level,
        ):
            raise ValidationError(
                f"Invalid processing_level: {processing_level!r}. "
                "Must contain only alphanumeric characters and underscores."
            )
        if mgrs_tile_id and not re.fullmatch(
            r"[0-9]{2}[A-Z]{3}", mgrs_tile_id,
        ):
            raise ValidationError(
                f"Invalid mgrs_tile_id: {mgrs_tile_id!r}. "
                "Expected format like 'T33UUP' (2 digits + 3 uppercase)."
            )
        if bbox:
            west, south, east, north = bbox
            for coord_name, coord_val in [
                ("west", west), ("south", south),
                ("east", east), ("north", north),
            ]:
                if not isinstance(coord_val, (int, float)):
                    raise ValidationError(
                        f"bbox {coord_name} must be numeric, "
                        f"got {type(coord_val).__name__}"
                    )

        # Build OData $filter expression
        filter_parts = [
            f"Collection/Name eq '{self.CDSE_COLLECTION}'",
        ]

        if processing_level:
            filter_parts.append(f"contains(Name,'{processing_level}')")

        if start_date:
            ts = _normalize_stac_datetime(start_date)
            filter_parts.append(
                f"ContentDate/Start gt {ts.replace('Z', '.000Z')}"
            )
        if end_date:
            ts = _normalize_stac_datetime(end_date)
            filter_parts.append(
                f"ContentDate/Start lt {ts.replace('Z', '.000Z')}"
            )
        if bbox:
            wkt = (
                f"POLYGON(({west} {south},{east} {south},"
                f"{east} {north},{west} {north},{west} {south}))"
            )
            filter_parts.append(
                f"OData.CSC.Intersects(area=geography"
                f"'SRID=4326;{wkt}')"
            )
        if mgrs_tile_id:
            # MGRS tile appears in the product name, e.g. _T33UUP_
            filter_parts.append(
                f"contains(Name,'_{mgrs_tile_id}_')"
            )

        params: Dict[str, str] = {
            "$filter": " and ".join(filter_parts),
            "$top": str(max_results),
            "$orderby": "ContentDate/Start desc",
        }

        try:
            response = requests.get(
                self.CDSE_ODATA_URL, params=params, timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            products = data.get("value", [])

            for product in products:
                self._index_remote_product(product)

            logger.info(
                "CDSE query returned %d Sentinel-2 products",
                len(products),
            )
            return products

        except requests.RequestException as e:
            raise ProcessorError(
                f"CDSE OData query failed: {e}"
            ) from e

    def _index_remote_product(self, product: Dict[str, Any]) -> None:
        """Index a CDSE OData product in the local database.

        Parameters
        ----------
        product : Dict[str, Any]
            Product dict from CDSE OData ``/Products`` response.
        """
        try:
            product_id = product.get("Id", "")
            product_name = product.get("Name", "")
            remote_url = (
                f"{self.CDSE_DOWNLOAD_URL}({product_id})/$value"
            )

            # Parse name: S2B_MSIL2A_20251201T105339_N0511_R051_T30TYN_...
            parts = product_name.split("_")
            satellite = parts[0] if len(parts) > 0 else ""
            processing_level = parts[1] if len(parts) > 1 else ""

            # Extract MGRS tile ID (e.g. T30TYN)
            mgrs_tile_id = ""
            for p in parts:
                if len(p) == 6 and p[0] == "T" and p[1:3].isdigit():
                    mgrs_tile_id = p
                    break

            start_time = product.get(
                "ContentDate", {}
            ).get("Start", "")

            footprint = product.get("GeoFootprint", {})
            corner_coords = json.dumps(
                footprint.get("coordinates", [])
            )

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO products
                (id, product_name, product_type, satellite,
                 processing_level, sensing_datetime,
                 mgrs_tile_id, relative_orbit, orbit_direction,
                 corner_coords, remote_url, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_name) DO UPDATE SET
                    id = excluded.id,
                    remote_url = excluded.remote_url,
                    corner_coords = COALESCE(products.corner_coords,
                                             excluded.corner_coords),
                    metadata_json = excluded.metadata_json,
                    satellite = COALESCE(products.satellite,
                                         excluded.satellite),
                    processing_level = COALESCE(products.processing_level,
                                                excluded.processing_level)
            """, (
                product_id,
                product_name,
                processing_level,
                satellite,
                processing_level,
                start_time,
                mgrs_tile_id,
                None,
                "",
                corner_coords,
                remote_url,
                json.dumps(product),
            ))
            self.conn.commit()

        except Exception as e:
            warnings.warn(f"Failed to index remote Sentinel-2 product: {e}")

    def download_product(
        self,
        product_id: str,
        destination: Optional[Union[str, Path]] = None,
        extract: bool = True,
        force: bool = False,
    ) -> Path:
        """Download a Sentinel-2 product from CDSE.

        Uses CDSE OAuth2 credentials from
        ``~/.config/geoint/credentials.json`` (``esa_copernicus`` block).

        If the product already exists locally (``local_path`` is set and
        the path exists on disk), a warning is issued and the existing
        path is returned unless ``force=True``.

        Parameters
        ----------
        product_id : str
            CDSE product ID. Must exist in the local database
            (run ``query_cdse`` first).
        destination : Optional[Union[str, Path]], default=None
            Directory to save the product. If None, uses ``search_path``.
        extract : bool, default=True
            If True, extract the ZIP and return the SAFE directory.
        force : bool, default=False
            If True, download even when the product already exists
            locally. The database ``local_path`` will be updated to
            the new download location.

        Returns
        -------
        Path
            Path to extracted SAFE directory or downloaded ZIP file.

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
                f"Run query_cdse() first to populate the database."
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

        access_token = get_cdse_token()

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

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()
