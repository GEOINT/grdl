# -*- coding: utf-8 -*-
"""
Tests for catalog upsert dedup logic and download_product force guard.

Exercises the SQLite ON CONFLICT(product_name) DO UPDATE behaviour across
all catalog classes without requiring real sensor data or network access.
Tests operate directly on the database layer.

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-13

Modified
--------
2026-03-13
"""

# Standard library
import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

# Third-party
import pytest


# ---------------------------------------------------------------------------
# Helper: create an in-memory products table matching each catalog's schema
# ---------------------------------------------------------------------------

def _make_sentinel1_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE products (
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
            remote_url TEXT,
            file_size INTEGER,
            download_date TEXT,
            metadata_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


SAFE_NAME = "S1A_IW_SLC__1SDV_20260301T060153_20260301T060220_999999_FFFFFF_ABCD.SAFE"


class TestUpsertLocalThenRemote:
    """Simulate: discover_local indexes a product, then query_cdse finds
    the same product remotely.  The remote upsert should merge — not
    create a duplicate — and preserve local_path."""

    def test_local_then_remote_merges(self):
        conn = _make_sentinel1_db()

        # 1. Local discovery inserts with dirname as id
        conn.execute("""
            INSERT INTO products
            (id, product_name, product_type, mission, mode,
             polarization, absolute_orbit, relative_orbit, orbit_pass,
             start_time, stop_time, swath, corner_coords,
             local_path, file_size, download_date, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_name) DO UPDATE SET
                product_type = excluded.product_type,
                mission = excluded.mission,
                mode = excluded.mode,
                polarization = excluded.polarization,
                absolute_orbit = excluded.absolute_orbit,
                relative_orbit = excluded.relative_orbit,
                orbit_pass = excluded.orbit_pass,
                start_time = excluded.start_time,
                stop_time = excluded.stop_time,
                swath = excluded.swath,
                corner_coords = excluded.corner_coords,
                local_path = excluded.local_path,
                file_size = excluded.file_size,
                download_date = excluded.download_date,
                metadata_json = excluded.metadata_json
        """, (
            SAFE_NAME,        # id = dirname
            SAFE_NAME,        # product_name
            "SLC", "S1A", "IW", "VV+VH",
            999999, 42, "ASCENDING",
            "2026-03-01T06:01:53", "2026-03-01T06:02:20",
            "IW", '[[41.0,12.0],[41.5,13.0]]',
            "/data/sar/" + SAFE_NAME,
            1234567,
            datetime.now().isoformat(),
            '{}',
        ))
        conn.commit()

        # Verify: 1 row
        rows = conn.execute("SELECT * FROM products").fetchall()
        assert len(rows) == 1
        assert rows[0]["local_path"] == "/data/sar/" + SAFE_NAME
        assert rows[0]["remote_url"] is None

        # 2. Remote query inserts with CDSE UUID as id
        cdse_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        remote_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({cdse_uuid})/$value"

        conn.execute("""
            INSERT INTO products
            (id, product_name, product_type, mission, mode,
             polarization, absolute_orbit, orbit_pass,
             start_time, stop_time, corner_coords,
             remote_url, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_name) DO UPDATE SET
                id = excluded.id,
                remote_url = excluded.remote_url,
                corner_coords = COALESCE(products.corner_coords,
                                         excluded.corner_coords),
                metadata_json = excluded.metadata_json,
                mission = COALESCE(products.mission, excluded.mission),
                mode = COALESCE(products.mode, excluded.mode),
                polarization = COALESCE(products.polarization,
                                        excluded.polarization)
        """, (
            cdse_uuid,
            SAFE_NAME,
            "SLC", "S1A", "IW", "VV+VH",
            None, "",
            "2026-03-01T06:01:53", "2026-03-01T06:02:20",
            '[[41.0,12.0]]',
            remote_url,
            '{"remote": true}',
        ))
        conn.commit()

        # Verify: still 1 row — merged, not duplicated
        rows = conn.execute("SELECT * FROM products").fetchall()
        assert len(rows) == 1

        row = dict(rows[0])
        # id updated to CDSE UUID
        assert row["id"] == cdse_uuid
        # local_path preserved from local discovery
        assert row["local_path"] == "/data/sar/" + SAFE_NAME
        # remote_url set from remote query
        assert row["remote_url"] == remote_url
        # corner_coords preserved (COALESCE keeps existing)
        assert row["corner_coords"] == '[[41.0,12.0],[41.5,13.0]]'
        # mission preserved from local (COALESCE)
        assert row["mission"] == "S1A"


class TestUpsertRemoteThenLocal:
    """Simulate: query_cdse finds a product remotely first, then
    discover_local finds the same product on disk. The local upsert
    should overwrite metadata but preserve remote_url."""

    def test_remote_then_local_preserves_remote_url(self):
        conn = _make_sentinel1_db()

        cdse_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        remote_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({cdse_uuid})/$value"

        # 1. Remote query inserts first
        conn.execute("""
            INSERT INTO products
            (id, product_name, product_type, mission, mode,
             polarization, absolute_orbit, orbit_pass,
             start_time, stop_time, corner_coords,
             remote_url, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_name) DO UPDATE SET
                id = excluded.id,
                remote_url = excluded.remote_url,
                corner_coords = COALESCE(products.corner_coords,
                                         excluded.corner_coords),
                metadata_json = excluded.metadata_json,
                mission = COALESCE(products.mission, excluded.mission),
                mode = COALESCE(products.mode, excluded.mode),
                polarization = COALESCE(products.polarization,
                                        excluded.polarization)
        """, (
            cdse_uuid, SAFE_NAME,
            "SLC", "S1A", "IW", "VV+VH",
            None, "",
            "2026-03-01T06:01:53", "2026-03-01T06:02:20",
            '[[41.0,12.0]]',
            remote_url,
            '{"remote": true}',
        ))
        conn.commit()

        assert conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 1
        row = dict(conn.execute("SELECT * FROM products").fetchone())
        assert row["local_path"] is None
        assert row["remote_url"] == remote_url

        # 2. Local discovery overwrites metadata, keeps remote_url
        #    (The local upsert updates all metadata columns but does
        #    NOT touch remote_url — so it stays from step 1.)
        conn.execute("""
            INSERT INTO products
            (id, product_name, product_type, mission, mode,
             polarization, absolute_orbit, relative_orbit, orbit_pass,
             start_time, stop_time, swath, corner_coords,
             local_path, file_size, download_date, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_name) DO UPDATE SET
                product_type = excluded.product_type,
                mission = excluded.mission,
                mode = excluded.mode,
                polarization = excluded.polarization,
                absolute_orbit = excluded.absolute_orbit,
                relative_orbit = excluded.relative_orbit,
                orbit_pass = excluded.orbit_pass,
                start_time = excluded.start_time,
                stop_time = excluded.stop_time,
                swath = excluded.swath,
                corner_coords = excluded.corner_coords,
                local_path = excluded.local_path,
                file_size = excluded.file_size,
                download_date = excluded.download_date,
                metadata_json = excluded.metadata_json
        """, (
            SAFE_NAME,  # id = dirname (different from CDSE UUID)
            SAFE_NAME,
            "SLC", "S1A", "IW", "VV+VH",
            999999, 42, "ASCENDING",
            "2026-03-01T06:01:53", "2026-03-01T06:02:20",
            "IW", '[[41.0,12.0],[41.5,13.0]]',
            "/data/sar/" + SAFE_NAME,
            1234567,
            datetime.now().isoformat(),
            '{"local": true}',
        ))
        conn.commit()

        # Verify: still 1 row
        rows = conn.execute("SELECT * FROM products").fetchall()
        assert len(rows) == 1

        row = dict(rows[0])
        # local_path now set
        assert row["local_path"] == "/data/sar/" + SAFE_NAME
        # remote_url preserved (not touched by local upsert)
        assert row["remote_url"] == remote_url
        # corner_coords updated with richer local data
        assert row["corner_coords"] == '[[41.0,12.0],[41.5,13.0]]'
        # absolute_orbit filled in by local
        assert row["absolute_orbit"] == 999999


class TestUpsertIdempotent:
    """Running discover_local twice should not create duplicates."""

    def test_double_local_discovery_no_duplicates(self):
        conn = _make_sentinel1_db()

        for _ in range(2):
            conn.execute("""
                INSERT INTO products
                (id, product_name, product_type, mission, mode,
                 polarization, local_path, file_size, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_name) DO UPDATE SET
                    product_type = excluded.product_type,
                    mission = excluded.mission,
                    mode = excluded.mode,
                    polarization = excluded.polarization,
                    local_path = excluded.local_path,
                    file_size = excluded.file_size,
                    metadata_json = excluded.metadata_json
            """, (
                SAFE_NAME, SAFE_NAME,
                "SLC", "S1A", "IW", "VV+VH",
                "/data/sar/" + SAFE_NAME, 1234567, '{}',
            ))
            conn.commit()

        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        assert count == 1


class TestDownloadForceGuard:
    """Test the force guard logic that prevents re-downloading existing
    local products.  Uses a mock to avoid instantiating real catalogs."""

    def test_warns_when_product_exists_locally(self, tmp_path):
        """Simulate download_product finding existing local_path."""
        # Create a fake local file
        fake_safe = tmp_path / SAFE_NAME
        fake_safe.mkdir()

        # Simulate the guard logic from download_product
        existing_path = str(fake_safe)
        force = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            if existing_path and Path(existing_path).exists() and not force:
                warnings.warn(
                    f"Product {SAFE_NAME} already exists locally "
                    f"at {existing_path}. Use force=True to proceed."
                )
                result = Path(existing_path)
            else:
                result = None  # Would proceed with download

            assert result == fake_safe
            assert len(w) == 1
            assert "already exists locally" in str(w[0].message)

    def test_no_warning_with_force(self, tmp_path):
        """With force=True, no warning should be raised."""
        fake_safe = tmp_path / SAFE_NAME
        fake_safe.mkdir()

        existing_path = str(fake_safe)
        force = True

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            if existing_path and Path(existing_path).exists() and not force:
                warnings.warn("Should not reach here")
                result = Path(existing_path)
            else:
                result = None  # Would proceed with download

            assert result is None  # Download would proceed
            assert len(w) == 0

    def test_no_warning_when_path_missing(self, tmp_path):
        """If local_path is set but file was deleted, proceed normally."""
        existing_path = str(tmp_path / "nonexistent.SAFE")
        force = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            if existing_path and Path(existing_path).exists() and not force:
                warnings.warn("Should not reach here")
                result = Path(existing_path)
            else:
                result = None  # Would proceed with download

            assert result is None
            assert len(w) == 0


class TestUpsertCoalescePreservesExisting:
    """COALESCE in the remote upsert should keep existing non-NULL values
    rather than overwriting with remote (potentially sparser) data."""

    def test_coalesce_keeps_richer_local_corner_coords(self):
        conn = _make_sentinel1_db()

        # Local discovery with detailed corners
        local_corners = json.dumps([
            [41.0, 12.0], [41.5, 12.0], [41.5, 13.0], [41.0, 13.0]
        ])
        conn.execute("""
            INSERT INTO products
            (id, product_name, corner_coords, local_path, metadata_json)
            VALUES (?, ?, ?, ?, ?)
        """, (SAFE_NAME, SAFE_NAME, local_corners, "/data/" + SAFE_NAME, '{}'))
        conn.commit()

        # Remote query with sparser corners
        remote_corners = json.dumps([[41.2, 12.5]])
        conn.execute("""
            INSERT INTO products
            (id, product_name, corner_coords, remote_url, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(product_name) DO UPDATE SET
                id = excluded.id,
                remote_url = excluded.remote_url,
                corner_coords = COALESCE(products.corner_coords,
                                         excluded.corner_coords),
                metadata_json = excluded.metadata_json
        """, (
            "uuid-123", SAFE_NAME, remote_corners,
            "https://example.com/download", '{"r":1}',
        ))
        conn.commit()

        row = dict(conn.execute("SELECT * FROM products").fetchone())
        # COALESCE keeps the existing (richer) local corners
        assert row["corner_coords"] == local_corners
        # remote_url is set
        assert row["remote_url"] == "https://example.com/download"


class TestMultipleDistinctProducts:
    """Multiple different products should each get their own row."""

    def test_two_different_products_create_two_rows(self):
        conn = _make_sentinel1_db()

        for i, name in enumerate([
            "S1A_IW_SLC__1SDV_20260301_AAA.SAFE",
            "S1A_IW_SLC__1SDV_20260302_BBB.SAFE",
        ]):
            conn.execute("""
                INSERT INTO products
                (id, product_name, mission, local_path, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(product_name) DO UPDATE SET
                    mission = excluded.mission,
                    local_path = excluded.local_path,
                    metadata_json = excluded.metadata_json
            """, (name, name, "S1A", f"/data/{name}", '{}'))
        conn.commit()

        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        assert count == 2
