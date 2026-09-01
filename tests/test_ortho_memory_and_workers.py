# -*- coding: utf-8 -*-
"""
Tests for ortho memory estimation and parallel tile processing.

Author
------
Duane Smalley, PhD
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-08-31

Modified
--------
2026-08-31
"""

# Standard library
from typing import List

# Third-party
import numpy as np
import pytest

# GRDL internal
from grdl.geolocation.eo.affine import AffineGeolocation
from grdl.image_processing.ortho import (
    GeographicGrid,
    MemoryEstimate,
    default_workers,
    estimate_ortho_memory,
    orthorectify,
)

rasterio = pytest.importorskip('rasterio')


@pytest.fixture
def geocoded(tmp_path):
    """A small single-band geocoded raster on disk."""
    from rasterio.transform import from_origin

    path = tmp_path / 'scene.tif'
    rows, cols = 240, 320
    data = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    with rasterio.open(
        path, 'w', driver='GTiff', height=rows, width=cols, count=1,
        dtype='float32', crs='EPSG:4326',
        transform=from_origin(-100.0, 40.0, 0.001, 0.001),
    ) as dataset:
        dataset.write(data, 1)
    return path


def _open(path):
    """Reader plus geolocation for a geocoded raster."""
    from grdl.IO.geotiff import GeoTIFFReader

    reader = GeoTIFFReader(str(path))
    return reader, AffineGeolocation.from_reader(reader)


def test_estimate_reports_output_as_the_floor() -> None:
    """The output array is counted and cannot be tiled away."""
    grid = GeographicGrid(39.0, 40.0, -100.0, -99.0, 0.001, 0.001)
    estimate = estimate_ortho_memory(grid, tile_size=256,
                                     dtype=np.float32)
    assert isinstance(estimate, MemoryEstimate)
    assert estimate.output_bytes == grid.rows * grid.cols * 4
    assert estimate.peak_bytes > estimate.output_bytes


def test_estimate_falls_with_tile_size() -> None:
    """Smaller tiles predict a smaller peak; that is the whole point."""
    grid = GeographicGrid(39.0, 40.0, -100.0, -99.0, 0.0005, 0.0005)
    big = estimate_ortho_memory(grid, tile_size=2048)
    small = estimate_ortho_memory(grid, tile_size=512)
    assert small.peak_bytes < big.peak_bytes


def test_estimate_scales_with_workers() -> None:
    """Per-tile terms are paid once per concurrent worker."""
    grid = GeographicGrid(39.0, 40.0, -100.0, -99.0, 0.0005, 0.0005)
    one = estimate_ortho_memory(grid, tile_size=512, workers=1)
    four = estimate_ortho_memory(grid, tile_size=512, workers=4)
    assert four.peak_bytes > one.peak_bytes


def test_untiled_is_dearer_than_tiled() -> None:
    """The untiled path is modelled as the expensive one it is."""
    grid = GeographicGrid(39.0, 40.0, -100.0, -99.0, 0.0005, 0.0005)
    untiled = estimate_ortho_memory(grid, tile_size=None)
    tiled = estimate_ortho_memory(grid, tile_size=512)
    assert untiled.peak_bytes > tiled.peak_bytes
    assert not untiled.tiled and tiled.tiled


def test_report_is_human_readable() -> None:
    """report() names the tiling and the predicted peak."""
    grid = GeographicGrid(39.0, 40.0, -100.0, -99.0, 0.001, 0.001)
    text = estimate_ortho_memory(grid, tile_size=256, workers=2).report()
    assert 'tiles' in text and 'peak' in text and 'workers' in text


def test_default_workers_is_sane() -> None:
    """A conservative positive count, not the whole core count."""
    assert 1 <= default_workers() <= 4


def test_estimate_only_skips_the_run(geocoded) -> None:
    """estimate_only returns an estimate instead of an OrthoResult."""
    reader, geo = _open(geocoded)
    try:
        estimate = orthorectify(
            geolocation=geo, reader=reader, metadata=reader.metadata,
            tile_size=64, estimate_only=True,
        )
        assert isinstance(estimate, MemoryEstimate)
    finally:
        reader.close()


def test_parallel_tiles_match_serial(geocoded) -> None:
    """Concurrency must not change a single pixel."""
    from grdl.IO.geotiff import GeoTIFFReader

    reader, geo = _open(geocoded)
    try:
        serial = orthorectify(
            geolocation=geo, reader=reader, metadata=reader.metadata,
            tile_size=64, nodata=np.nan,
        )
        parallel = orthorectify(
            geolocation=geo, reader=reader, metadata=reader.metadata,
            tile_size=64, nodata=np.nan, workers=4,
            reader_factory=lambda: GeoTIFFReader(str(geocoded)),
        )
        assert serial.data.shape == parallel.data.shape
        assert serial.data.dtype == parallel.data.dtype
        assert np.array_equal(
            np.nan_to_num(serial.data, nan=0.0),
            np.nan_to_num(parallel.data, nan=0.0),
        )
    finally:
        reader.close()


def test_workers_without_factory_falls_back(geocoded, caplog) -> None:
    """Sharing one file handle across threads is refused, not risked."""
    reader, geo = _open(geocoded)
    try:
        with caplog.at_level('WARNING'):
            result = orthorectify(
                geolocation=geo, reader=reader,
                metadata=reader.metadata, tile_size=64,
                nodata=np.nan, workers=4,
            )
        assert 'reader_factory' in caplog.text
        assert np.isfinite(result.data).any()
    finally:
        reader.close()


def test_worker_readers_are_closed(geocoded) -> None:
    """Per-thread readers do not leak; on Windows a leak also locks."""
    from grdl.IO.geotiff import GeoTIFFReader

    created: List[GeoTIFFReader] = []

    def factory() -> GeoTIFFReader:
        """Track every reader handed to a worker."""
        made = GeoTIFFReader(str(geocoded))
        created.append(made)
        return made

    reader, geo = _open(geocoded)
    try:
        orthorectify(
            geolocation=geo, reader=reader, metadata=reader.metadata,
            tile_size=64, nodata=np.nan, workers=3,
            reader_factory=factory,
        )
    finally:
        reader.close()

    assert created, 'the factory was never called'
    for made in created:
        assert made.dataset is None or made.dataset.closed


def test_rejects_non_positive_workers() -> None:
    """A zero or negative worker count fails loudly."""
    from grdl.image_processing.ortho import OrthoBuilder

    with pytest.raises(ValueError, match='workers'):
        OrthoBuilder().with_workers(0)
