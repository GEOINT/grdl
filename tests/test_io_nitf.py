# -*- coding: utf-8 -*-
"""
NITF Reader Tests - Unit tests for NITFReader.

Uses synthetic NITF files created with rasterio/GDAL for testing.

Dependencies
------------
pytest
rasterio

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
2026-02-09

Modified
--------
2026-02-09
"""

import pytest
import numpy as np

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

pytestmark = pytest.mark.skipif(
    not _HAS_RASTERIO, reason="rasterio not installed"
)


@pytest.fixture
def nitf_file(tmp_path):
    """Create a minimal NITF file for testing."""
    from rasterio.transform import from_bounds

    filepath = tmp_path / "test.ntf"
    data = np.random.randint(0, 255, (64, 128), dtype=np.uint8)

    transform = from_bounds(-180, -90, 180, 90, 128, 64)

    with rasterio.open(
        str(filepath), 'w', driver='NITF',
        height=64, width=128, count=1,
        dtype='uint8',
        transform=transform,
        crs='EPSG:4326',
        ICORDS='G',
    ) as ds:
        ds.write(data, 1)

    return filepath, data


@pytest.fixture
def nitf_multi(tmp_path):
    """Create a multi-band NITF file for testing."""
    from rasterio.transform import from_bounds

    filepath = tmp_path / "test_multi.ntf"
    data = np.random.randint(0, 255, (3, 50, 100), dtype=np.uint8)

    transform = from_bounds(-180, -90, 180, 90, 100, 50)

    with rasterio.open(
        str(filepath), 'w', driver='NITF',
        height=50, width=100, count=3,
        dtype='uint8',
        transform=transform,
        crs='EPSG:4326',
        ICORDS='G',
    ) as ds:
        ds.write(data)

    return filepath, data


def test_metadata(nitf_file):
    """Metadata extracted correctly from NITF."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        assert reader.metadata['format'] == 'NITF'
        assert reader.metadata['rows'] == 64
        assert reader.metadata['cols'] == 128
        assert reader.metadata['bands'] == 1


def test_get_shape_single(nitf_file):
    """get_shape returns (rows, cols) for single band NITF."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        assert reader.get_shape() == (64, 128)


def test_get_shape_multi(nitf_multi):
    """get_shape returns (rows, cols, bands) for multi-band NITF."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_multi
    with NITFReader(filepath) as reader:
        assert reader.get_shape() == (50, 100, 3)


def test_get_dtype(nitf_file):
    """get_dtype returns correct numpy dtype."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        assert reader.get_dtype() == np.dtype('uint8')


def test_read_chip(nitf_file):
    """read_chip returns correct data."""
    from grdl.IO.nitf import NITFReader

    filepath, original = nitf_file
    with NITFReader(filepath) as reader:
        chip = reader.read_chip(0, 32, 0, 64)
        assert chip.shape == (32, 64)
        assert chip.dtype == np.uint8


def test_read_full(nitf_file):
    """read_full returns entire image."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        full = reader.read_full()
        assert full.shape == (64, 128)


def test_read_chip_negative_start_raises(nitf_file):
    """Negative start indices raise ValueError."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        with pytest.raises(ValueError, match="non-negative"):
            reader.read_chip(-1, 10, 0, 10)


def test_read_chip_out_of_bounds_raises(nitf_file):
    """Out-of-bounds end indices raise ValueError."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        with pytest.raises(ValueError, match="exceed"):
            reader.read_chip(0, 100, 0, 10)


def test_file_not_found():
    """FileNotFoundError for non-existent file."""
    from grdl.IO.nitf import NITFReader

    with pytest.raises(FileNotFoundError):
        NITFReader('/nonexistent/file.ntf')


def test_context_manager(nitf_file):
    """Context manager opens and closes cleanly."""
    from grdl.IO.nitf import NITFReader

    filepath, _ = nitf_file
    with NITFReader(filepath) as reader:
        assert reader.metadata['format'] == 'NITF'


# ===================================================================
# Multi-image NITF (EONITFReader unification) tests
# ===================================================================


def _ichipb_xml(
    fi_row_off: float, fi_col_off: float,
    rows: int, cols: int,
    fi_row_scale: float = 1.0, fi_col_scale: float = 1.0,
    scale_factor: float = 1.0,
) -> str:
    """Build a minimal ICHIPB xml:TRE document for a fake segment.

    Encodes the eight OP/FI corners so the chip-to-full affine the
    parser reconstructs is::

        full = fi_*_off + fi_*_scale * chip

    The OP corners use 0.5 / (size - 0.5) chip-pixel-center convention.
    """
    op_r1, op_r2 = 0.5, rows - 0.5
    op_c1, op_c2 = 0.5, cols - 0.5
    fi_r1 = fi_row_off + fi_row_scale * op_r1
    fi_r2 = fi_row_off + fi_row_scale * op_r2
    fi_c1 = fi_col_off + fi_col_scale * op_c1
    fi_c2 = fi_col_off + fi_col_scale * op_c2

    def fld(name, value):
        return f'<field name="{name}" value="{value}"/>'

    parts = ['<tre name="ICHIPB">',
             fld('XFRM_FLAG', '2'),
             fld('SCALE_FACTOR', f'{scale_factor:.5f}'),
             fld('ANAMRPH_CORR', '0'),
             fld('SCANBLK_NUM', '0'),
             fld('OP_ROW_11', f'{op_r1:.6f}'),
             fld('OP_COL_11', f'{op_c1:.6f}'),
             fld('OP_ROW_12', f'{op_r1:.6f}'),
             fld('OP_COL_12', f'{op_c2:.6f}'),
             fld('OP_ROW_21', f'{op_r2:.6f}'),
             fld('OP_COL_21', f'{op_c1:.6f}'),
             fld('OP_ROW_22', f'{op_r2:.6f}'),
             fld('OP_COL_22', f'{op_c2:.6f}'),
             fld('FI_ROW_11', f'{fi_r1:.6f}'),
             fld('FI_COL_11', f'{fi_c1:.6f}'),
             fld('FI_ROW_12', f'{fi_r1:.6f}'),
             fld('FI_COL_12', f'{fi_c2:.6f}'),
             fld('FI_ROW_21', f'{fi_r2:.6f}'),
             fld('FI_COL_21', f'{fi_c1:.6f}'),
             fld('FI_ROW_22', f'{fi_r2:.6f}'),
             fld('FI_COL_22', f'{fi_c2:.6f}'),
             '</tre>']
    return ''.join(parts)


class _FakeDataset:
    """Stand-in for ``rasterio.DatasetReader`` for unit tests.

    Honors the slice of the API that ``EONITFReader`` consumes:
    ``height``, ``width``, ``count``, ``dtypes``, ``nodata``, ``crs``,
    ``rpcs``, ``subdatasets``, ``tags(ns=)``, ``tag_namespaces()``,
    ``read(indexes, window)``, and ``close()``.
    """

    def __init__(
        self,
        height: int = 0,
        width: int = 0,
        count: int = 0,
        dtype: str = 'uint8',
        nodata=None,
        ichipb_xml: str = None,
        data: np.ndarray = None,
        subdatasets: list = None,
    ):
        self.height = height
        self.width = width
        self.count = count
        self.dtypes = (dtype,) * max(count, 1)
        self.nodata = nodata
        self.crs = None
        self.rpcs = None
        self.subdatasets = subdatasets or []
        self._tre_xml = ichipb_xml
        self._data = data
        self.closed = False

    def tags(self, ns=None):
        if ns == 'xml:TRE':
            return {'doc': self._tre_xml} if self._tre_xml else {}
        if ns is None:
            return {}
        return {}

    def tag_namespaces(self):
        return ['xml:TRE'] if self._tre_xml else []

    def read(self, indexes=None, window=None):
        # indexes is None or a list of 1-based band indices; we always
        # return shape (bands, rows, cols) to match rasterio's contract.
        if self._data is None:
            raise RuntimeError("FakeDataset has no data buffer")
        d = self._data
        if d.ndim == 2:
            d = d[np.newaxis, ...]
        if window is not None:
            r0 = int(window.row_off)
            r1 = r0 + int(window.height)
            c0 = int(window.col_off)
            c1 = c0 + int(window.width)
            d = d[:, r0:r1, c0:c1]
        if indexes is not None:
            sel = [int(i) - 1 for i in indexes]
            d = d[sel]
        return d.copy()

    def close(self):
        self.closed = True


def _patch_rasterio_open(monkeypatch, parent_fake, segment_fakes):
    """Replace ``rasterio.open`` so the parent path returns ``parent_fake``
    and each ``NITF_IM:N:`` URI returns ``segment_fakes[N]``.
    """
    import grdl.IO.eo.nitf as nitf_module

    def fake_open(path, *args, **kwargs):
        s = str(path)
        if s.startswith('NITF_IM:'):
            idx = int(s.split(':')[1])
            return segment_fakes[idx]
        return parent_fake

    monkeypatch.setattr(nitf_module.rasterio, 'open', fake_open)


def _make_two_segment_fakes(
    parent_path: str,
    seg_rows: int = 64,
    seg_cols: int = 128,
    overlap_data: bool = False,
    nodata=None,
    second_scale_factor: float = 1.0,
    second_count: int = None,
    gap_rows: int = 0,
):
    """Build a parent fake + two horizontally-stacked segment fakes.

    Segment 0 covers full-image rows ``[0, seg_rows)``; segment 1 covers
    ``[seg_rows + gap_rows, 2*seg_rows + gap_rows)``.  Both span columns
    ``[0, seg_cols)``.
    """
    seg0_data = np.full(
        (seg_rows, seg_cols), 10, dtype=np.uint8)
    seg1_data = np.full(
        (seg_rows, seg_cols), 20, dtype=np.uint8)

    seg0 = _FakeDataset(
        height=seg_rows, width=seg_cols, count=1,
        nodata=nodata,
        ichipb_xml=_ichipb_xml(
            fi_row_off=0.0, fi_col_off=0.0,
            rows=seg_rows, cols=seg_cols),
        data=seg0_data,
    )
    seg1_count = second_count if second_count is not None else 1
    seg1 = _FakeDataset(
        height=seg_rows, width=seg_cols, count=seg1_count,
        nodata=nodata,
        ichipb_xml=_ichipb_xml(
            fi_row_off=float(seg_rows + gap_rows), fi_col_off=0.0,
            rows=seg_rows, cols=seg_cols,
            scale_factor=second_scale_factor),
        data=seg1_data if seg1_count == 1
        else np.broadcast_to(seg1_data, (seg1_count, seg_rows, seg_cols)).copy(),
    )

    parent = _FakeDataset(
        subdatasets=[
            f'NITF_IM:0:{parent_path}',
            f'NITF_IM:1:{parent_path}',
        ],
    )
    return parent, [seg0, seg1]


class TestMultiImageNITF:
    """Unified-reader behavior on multi-image NITF."""

    def test_single_segment_unchanged(self, nitf_file):
        """No subdatasets → existing single-image path verbatim."""
        from grdl.IO.eo.nitf import EONITFReader

        filepath, data = nitf_file
        with EONITFReader(filepath) as r:
            assert r.metadata.image_segments is None
            assert r.metadata.rows == 64
            assert r.metadata.cols == 128
            chip = r.read_chip(0, 32, 0, 64)
            assert chip.shape == (32, 64)

    def test_unified_shape_and_metadata(self, monkeypatch, tmp_path):
        """Two stacked segments expose full-image dims and segment list."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')   # only existence is checked
        parent, segs = _make_two_segment_fakes(str(path))
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            assert r.metadata.rows == 128         # 2 × 64
            assert r.metadata.cols == 128
            assert r.metadata.ichipb is None
            segs_info = r.metadata.image_segments
            assert segs_info is not None
            assert len(segs_info) == 2
            assert segs_info[0].fi_row_lo == 0
            assert segs_info[0].fi_row_hi == 64
            assert segs_info[1].fi_row_lo == 64
            assert segs_info[1].fi_row_hi == 128

    def test_read_chip_routes_to_correct_segment(
        self, monkeypatch, tmp_path,
    ):
        """Chip entirely inside one segment reads only that segment."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            chip0 = r.read_chip(10, 30, 0, 64)   # inside segment 0
            assert chip0.shape == (20, 64)
            assert np.all(chip0 == 10)

            chip1 = r.read_chip(80, 100, 0, 64)  # inside segment 1
            assert chip1.shape == (20, 64)
            assert np.all(chip1 == 20)

    def test_read_chip_spans_segments(self, monkeypatch, tmp_path):
        """Chip across boundary stitches both segments."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            # full-image rows 60..68, crossing the 64-row boundary
            chip = r.read_chip(60, 68, 0, 64)
            assert chip.shape == (8, 64)
            # First 4 rows from segment 0 (value 10), next 4 from
            # segment 1 (value 20).
            assert np.all(chip[:4] == 10)
            assert np.all(chip[4:] == 20)

    def test_chip_in_gap_filled_with_nodata(self, monkeypatch, tmp_path):
        """Pixels in a gap between segments fill with nodata."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(
            str(path), gap_rows=16, nodata=255)
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            # Total full-image rows: 64 + 16 (gap) + 64 = 144
            assert r.metadata.rows == 144
            # Read entirely inside the gap (rows 70..78).
            chip = r.read_chip(70, 78, 0, 64)
            assert chip.shape == (8, 64)
            assert np.all(chip == 255)

    def test_mixed_scale_factor_refused(self, monkeypatch, tmp_path):
        """Mixed SCALE_FACTOR raises ValueError."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(
            str(path), second_scale_factor=2.0)
        _patch_rasterio_open(monkeypatch, parent, segs)

        with pytest.raises(ValueError, match="Multi-resolution"):
            EONITFReader(path)

    def test_mixed_bands_refused(self, monkeypatch, tmp_path):
        """Mixed band counts raise ValueError."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(
            str(path), second_count=3)
        _patch_rasterio_open(monkeypatch, parent, segs)

        with pytest.raises(ValueError, match="bands"):
            EONITFReader(path)

    def test_image_index_pin(self, monkeypatch, tmp_path):
        """image_index=N pins to one segment, no unification."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path, image_index=1) as r:
            assert r.metadata.image_segments is None
            assert r.metadata.rows == 64
            assert r.metadata.cols == 128
            # Pinned reader reports the segment's own ICHIPB; geolocators
            # apply it.  The chip-local read returns segment 1's data.
            chip = r.read_chip(0, 64, 0, 128)
            assert np.all(chip == 20)

    def test_close_releases_all_segment_handles(
        self, monkeypatch, tmp_path,
    ):
        """close() closes every per-segment dataset."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        _patch_rasterio_open(monkeypatch, parent, segs)

        r = EONITFReader(path)
        r.close()
        assert all(s.closed for s in segs)

    def test_orthorectifier_consumes_unified_reader(
        self, monkeypatch, tmp_path,
    ):
        """End-to-end smoke: read_chip in full-image coords across segments.

        Verifies the regression target from the design plan: the
        orthorectifier path that calls ``reader.read_chip(r0, r1, c0,
        c1)`` in full-image coordinates works transparently across
        segment boundaries — no orthorectifier code change required.
        """
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            # Mimic the orthorectifier's call pattern: clamp to
            # (meta.rows, meta.cols) then call read_chip.
            r0 = max(0, 50)
            r1 = min(r.metadata.rows, 80)
            c0 = max(0, 0)
            c1 = min(r.metadata.cols, 64)
            chip = r.read_chip(r0, r1, c0, c1)
            assert chip.shape == (30, 64)
            # rows 50..63 → segment 0 (10); rows 64..79 → segment 1 (20)
            assert np.all(chip[:14] == 10)
            assert np.all(chip[14:] == 20)
