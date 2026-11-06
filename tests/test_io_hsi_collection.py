# -*- coding: utf-8 -*-
"""
Unit tests for HSICollectionReader (Issue #78).
Validates multi-file single-band GeoTIFF collection aggregation, lazy reading,
and open_any integration without requiring external test datasets.
"""

from pathlib import Path
import numpy as np
import pytest
import rasterio

from grdl.IO import HSICollectionReader, open_any
from grdl.IO.hsi_collection import _is_hsi_collection_dir


@pytest.fixture
def hsi_collection_dir(tmp_path):
    """Create a synthetic 6-band HSI collection directory."""
    folder = tmp_path / "EO1_Hyperion_Test"
    folder.mkdir()

    # Create 6 single-band GeoTIFF files
    for b in range(1, 7):
        filepath = folder / f"EO1_TEST_B{b:03d}_L1T.TIF"
        data = np.full((30, 40), fill_value=float(b * 10), dtype=np.float32)
        with rasterio.open(
            str(filepath), 'w', driver='GTiff',
            height=30, width=40, count=1,
            dtype='float32'
        ) as ds:
            ds.write(data, 1)

    # Write synthetic MTL.TXT
    mtl_path = folder / "EO1_TEST_MTL_L1T.TXT"
    with open(mtl_path, 'w', encoding='utf-8') as f:
        f.write('SPACECRAFT_ID = "EO-1"\nSENSOR_ID = "HYPERION"\n')

    return folder


def test_is_hsi_collection_dir(hsi_collection_dir, tmp_path):
    assert _is_hsi_collection_dir(hsi_collection_dir) is True
    assert _is_hsi_collection_dir(tmp_path) is False


def test_hsi_collection_reader_metadata(hsi_collection_dir):
    with HSICollectionReader(hsi_collection_dir) as reader:
        meta = reader.metadata
        assert meta.format == "GeoTIFF_Collection"
        assert meta.rows == 30
        assert meta.cols == 40
        assert meta.bands == 6
        assert meta.dtype == "float32"
        assert meta.extras.get("SENSOR_ID") == "HYPERION"
        assert len(meta.channel_metadata) == 6
        assert meta.channel_metadata[0].name == "band_1"
        assert meta.channel_metadata[5].name == "band_6"


def test_hsi_collection_reader_chip(hsi_collection_dir):
    with HSICollectionReader(hsi_collection_dir) as reader:
        # Read full spatial chip across all bands
        chip = reader.read_chip(0, 10, 0, 10)
        assert chip.shape == (6, 10, 10)
        assert chip[0, 0, 0] == 10.0
        assert chip[5, 0, 0] == 60.0

        # Read specific band subset
        sub = reader.read_chip(0, 10, 0, 10, bands=[0, 2])
        assert sub.shape == (2, 10, 10)
        assert sub[0, 0, 0] == 10.0
        assert sub[1, 0, 0] == 30.0


def test_open_any_hsi_collection(hsi_collection_dir):
    with open_any(hsi_collection_dir) as reader:
        assert isinstance(reader, HSICollectionReader)
        assert reader.metadata.bands == 6

