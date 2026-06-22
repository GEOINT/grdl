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
        tags_dict: dict = None,
        des_xml: str = None,
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
        self._des_xml = des_xml
        self._data = data
        self._tags_dict = tags_dict or {}
        self.closed = False

    def tags(self, ns=None):
        if ns == 'xml:TRE':
            return {'doc': self._tre_xml} if self._tre_xml else {}
        if ns == 'xml:DES':
            return {'doc': self._des_xml} if self._des_xml else {}
        if ns is None:
            return dict(self._tags_dict)
        return {}

    def tag_namespaces(self):
        out = []
        if self._tre_xml:
            out.append('xml:TRE')
        if self._des_xml:
            out.append('xml:DES')
        return out

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

    def test_mixed_scale_factor_grouped(self, monkeypatch, tmp_path):
        """Mixed SCALE_FACTOR no longer refuses: full-res group is primary."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(
            str(path), second_scale_factor=2.0)
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            # Primary = SCALE_FACTOR=1 group (segment 0 only).
            assert r.metadata.rows == 64
            assert r.metadata.cols == 128
            infos = r.metadata.image_segments
            assert len(infos) == 2
            by_idx = {i.segment_index: i for i in infos}
            assert by_idx[0].is_primary
            assert not by_idx[1].is_primary
            assert by_idx[0].group_id != by_idx[1].group_id
            chip = r.read_chip(0, 64, 0, 128)
            assert np.all(chip == 10)

    def test_mixed_bands_grouped(self, monkeypatch, tmp_path):
        """Mixed band counts no longer refuse: groups split, primary reads."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(
            str(path), second_count=3)
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            assert r.metadata.bands == 1
            assert r.metadata.rows == 64
            by_idx = {i.segment_index: i
                      for i in r.metadata.image_segments}
            assert by_idx[0].is_primary
            assert not by_idx[1].is_primary
            assert by_idx[1].bands == 3
            chip = r.read_chip(0, 64, 0, 128)
            assert np.all(chip == 10)

    def test_overview_segment_not_primary(self, monkeypatch, tmp_path):
        """IMAG='/2' overview segment lands in its own non-primary group."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        # Tag segment 1 as a half-resolution overview.  Same bands,
        # dtype, and SCALE_FACTOR — only IMAG distinguishes it.
        segs[1]._tags_dict = {'NITF_IMAG': '/2  '}
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            assert r.metadata.rows == 64       # primary = segment 0 only
            by_idx = {i.segment_index: i
                      for i in r.metadata.image_segments}
            assert by_idx[0].is_primary
            assert not by_idx[1].is_primary
            assert by_idx[1].imag == pytest.approx(0.5)
            chip = r.read_chip(0, 64, 0, 128)
            assert np.all(chip == 10)

    def test_iloc_placement_without_ichipb(self, monkeypatch, tmp_path):
        """ILOC/IALVL attachment chain places segments — no stack warning."""
        import warnings as _warnings
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        seg0 = _FakeDataset(
            height=64, width=128, count=1,
            data=np.full((64, 128), 10, dtype=np.uint8),
            tags_dict={
                'NITF_IDLVL': '1', 'NITF_IALVL': '0',
                'NITF_ILOC_ROW': '0', 'NITF_ILOC_COLUMN': '0',
            },
        )
        seg1 = _FakeDataset(
            height=64, width=128, count=1,
            data=np.full((64, 128), 20, dtype=np.uint8),
            tags_dict={
                'NITF_IDLVL': '2', 'NITF_IALVL': '1',
                'NITF_ILOC_ROW': '64', 'NITF_ILOC_COLUMN': '0',
            },
        )
        parent = _FakeDataset(subdatasets=[
            f'NITF_IM:0:{path}', f'NITF_IM:1:{path}'])
        _patch_rasterio_open(monkeypatch, parent, [seg0, seg1])

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            with EONITFReader(path) as r:
                assert r.metadata.rows == 128
                assert r.metadata.cols == 128
                infos = r.metadata.image_segments
                assert all(i.placement == 'iloc' for i in infos)
                chip = r.read_chip(60, 68, 0, 64)
                assert np.all(chip[:4] == 10)
                assert np.all(chip[4:] == 20)
        stack_warnings = [
            w for w in caught
            if 'sequential row stacking' in str(w.message)]
        assert not stack_warnings

    def test_heterogeneous_mask_segment(self, monkeypatch, tmp_path):
        """Primary mosaic + float32 mask: loads, stitches, mask pinnable."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        mask = _FakeDataset(
            height=32, width=32, count=1, dtype='float32',
            ichipb_xml=_ichipb_xml(
                fi_row_off=0.0, fi_col_off=0.0, rows=32, cols=32),
            data=np.full((32, 32), 7.0, dtype=np.float32),
        )
        segs = segs + [mask]
        parent.subdatasets.append(f'NITF_IM:2:{path}')
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            # Primary = the two-segment uint8 mosaic (larger area).
            assert r.metadata.rows == 128
            assert r.metadata.dtype == 'uint8'
            assert len(r.metadata.image_segments) == 3
            by_idx = {i.segment_index: i
                      for i in r.metadata.image_segments}
            assert by_idx[0].is_primary and by_idx[1].is_primary
            assert not by_idx[2].is_primary
            chip = r.read_chip(60, 68, 0, 64)
            assert np.all(chip[:4] == 10)
            assert np.all(chip[4:] == 20)

        # The mask stays readable by pinning its subdataset index.
        with EONITFReader(path, image_index=2) as r2:
            assert r2.metadata.dtype == 'float32'
            mchip = r2.read_chip(0, 32, 0, 32)
            assert np.all(mchip == 7.0)

    def test_image_groups_metadata(self, monkeypatch, tmp_path):
        """image_groups summarizes every group with a primary flag."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(
            str(path), second_count=3)
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            groups = r.image_groups
            assert groups is not None
            assert len(groups) == 2
            primary = [g for g in groups if g.is_primary]
            assert len(primary) == 1
            assert primary[0].bands == 1
            assert primary[0].segment_indices == [0]
            other = [g for g in groups if not g.is_primary][0]
            assert other.bands == 3
            assert other.segment_indices == [1]
            assert all(g.placement == 'ichipb' for g in groups)

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


# ===================================================================
# Reader extension tests: DES overflow, CSCRNA, extended TRE wiring,
# decimation, masks, ABPP, remote paths
# ===================================================================


def _cscrna_xml() -> str:
    """Build a minimal CSCRNA xml:TRE document."""
    def fld(name, value):
        return f'<field name="{name}" value="{value}"/>'

    corners = {
        'UL': (10.0, 20.0, 100.0),
        'UR': (10.0, 21.0, 110.0),
        'LR': (9.0, 21.0, 120.0),
        'LL': (9.0, 20.0, 130.0),
    }
    parts = ['<tre name="CSCRNA">', fld('PREDICT_CORNERS', 'Y')]
    for pfx, (lat, lon, ht) in corners.items():
        parts.append(fld(f'{pfx}CNR_LAT', f'{lat:.6f}'))
        parts.append(fld(f'{pfx}CNR_LONG', f'{lon:.6f}'))
        parts.append(fld(f'{pfx}CNR_HT', f'{ht:.1f}'))
    parts.append('</tre>')
    return ''.join(parts)


class TestReaderExtensions:
    """DES overflow, CSCRNA, extended TREs, decimation, mask, ABPP."""

    def test_des_overflow_tre_parsed(self, monkeypatch, tmp_path):
        """TREs inside a TRE_OVERFLOW DES (xml:DES) are recognized."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        ichipb = _ichipb_xml(
            fi_row_off=100.0, fi_col_off=0.0, rows=64, cols=128)
        des_doc = (
            '<des_list><des name="TRE_OVERFLOW">'
            + ichipb +
            '</des></des_list>'
        )
        parent = _FakeDataset(
            height=64, width=128, count=1,
            data=np.zeros((64, 128), dtype=np.uint8),
            des_xml=des_doc,        # NO xml:TRE namespace at all
        )
        _patch_rasterio_open(monkeypatch, parent, [])

        with EONITFReader(path) as r:
            assert r.metadata.ichipb is not None
            assert r.metadata.ichipb.fi_row_off == pytest.approx(100.0)
            assert r.tre_source == 'xml:TRE'

    def test_cscrna_parsed(self, monkeypatch, tmp_path):
        """CSCRNA corner footprint lands on metadata.cscrna."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent = _FakeDataset(
            height=64, width=128, count=1,
            data=np.zeros((64, 128), dtype=np.uint8),
            ichipb_xml=_cscrna_xml(),
        )
        _patch_rasterio_open(monkeypatch, parent, [])

        with EONITFReader(path) as r:
            cs = r.metadata.cscrna
            assert cs is not None
            assert cs.predicted is True
            assert cs.corners.shape == (4, 2)
            np.testing.assert_allclose(cs.corners[0], [10.0, 20.0])
            np.testing.assert_allclose(cs.corners[2], [9.0, 21.0])
            assert cs.heights is not None
            assert cs.mean_height == pytest.approx(115.0)

    def test_cscrna_cedata_parser(self):
        """Legacy byte-offset CSCRNA parser round-trips."""
        from grdl.IO.eo.nitf import _parse_cscrna_tre

        corners = [
            (10.0, 20.0, 100.0), (10.0, 21.0, 110.0),
            (9.0, 21.0, 120.0), (9.0, 20.0, 130.0),
        ]
        s = 'Y'
        for lat, lon, ht in corners:
            s += f'{lat:+09.6f}'[:9].rjust(9)
            s += f'{lon:+010.6f}'[:10].rjust(10)
            s += f'{ht:+08.1f}'[:8].rjust(8)
        parsed = _parse_cscrna_tre(s)
        assert parsed is not None
        assert parsed.predicted is True
        np.testing.assert_allclose(parsed.corners[1], [10.0, 21.0])
        np.testing.assert_allclose(parsed.heights[3], 130.0)
        assert _parse_cscrna_tre('garbage') is None

    def test_extended_tres_wired(self, monkeypatch, tmp_path):
        """BANDSB/RSMDCA stubs flow to metadata + accuracy."""
        from types import SimpleNamespace
        import grdl.IO.eo._tre_xml as tre_xml
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent = _FakeDataset(
            height=64, width=128, count=2,
            data=np.zeros((2, 64, 128), dtype=np.uint8),
            ichipb_xml='<tre name="IGNORED"/>',  # forces xml path probe
        )
        _patch_rasterio_open(monkeypatch, parent, [])

        bandsb = SimpleNamespace(
            band_names=['Blue', 'NIR'],
            wavelengths_um=[0.45, 0.86],
        )
        dca = SimpleNamespace(
            sigma_x=1.0, sigma_y=1.0, sigma_z=2.0, variant='A')
        sensrb = SimpleNamespace(sensor='FAKE_SENSOR')

        monkeypatch.setattr(
            tre_xml, 'parse_all_tres',
            lambda ds: {
                'BANDSB': bandsb, 'RSMDCA': dca, 'SENSRB': sensrb,
            },
        )

        with EONITFReader(path) as r:
            assert r.metadata.bandsb is bandsb
            assert r.metadata.sensrb is sensrb
            assert r.metadata.rsm_dca is dca
            assert r.metadata.band_names == ['Blue', 'NIR']
            assert r.metadata.wavelengths == [0.45, 0.86]
            acc = r.metadata.accuracy
            assert acc is not None
            assert acc.source == 'RSMDCA'
            # Circular case: CE90 = 2.1460 * sigma; LE90 = 1.6449 * sz
            assert acc.ce90 == pytest.approx(2.1460, abs=1e-4)
            assert acc.le90 == pytest.approx(3.2898, abs=1e-4)

    def test_decimation_served_by_overview_group(
        self, monkeypatch, tmp_path,
    ):
        """decimation=2 routes to the IMAG=/2 overview group."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        overview = _FakeDataset(
            height=64, width=64, count=1,
            ichipb_xml=_ichipb_xml(
                fi_row_off=0.0, fi_col_off=0.0, rows=64, cols=64),
            data=np.full((64, 64), 99, dtype=np.uint8),
            tags_dict={'NITF_IMAG': '/2  '},
        )
        segs = segs + [overview]
        parent.subdatasets.append(f'NITF_IM:2:{path}')
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            assert r.metadata.rows == 128
            chip = r.read_chip(0, 128, 0, 128, decimation=2)
            assert chip.shape == (64, 64)
            assert np.all(chip == 99)     # overview pixels, not 10/20

    def test_decimation_fallback_slicing(self, monkeypatch, tmp_path):
        """No overview group → full-res read sliced [::d]."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            chip = r.read_chip(0, 128, 0, 64, decimation=2)
            assert chip.shape == (64, 32)
            assert np.all(chip[:32] == 10)
            assert np.all(chip[32:] == 20)

    def test_decimation_validation(self, monkeypatch, tmp_path):
        """decimation < 1 raises ValueError."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            with pytest.raises(ValueError, match="decimation"):
                r.read_chip(0, 64, 0, 64, decimation=0)

    def test_read_mask_gap_invalid(self, monkeypatch, tmp_path):
        """read_mask: covered pixels 255, inter-segment gap 0."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(
            str(path), gap_rows=16, nodata=255)
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            mask = r.read_mask(56, 88, 0, 64)
            assert mask.shape == (32, 64)
            assert np.all(mask[:8] == 255)    # rows 56..63 → segment 0
            assert np.all(mask[8:24] == 0)    # rows 64..79 → gap
            assert np.all(mask[24:] == 255)   # rows 80..87 → segment 1

    def test_normalize_abpp(self, monkeypatch, tmp_path):
        """11-bit data in uint16 container scales by 2**11 - 1."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            r.metadata.abpp = 11
            chip = np.array([[2047, 0], [1024, 2047]], dtype=np.uint16)
            out = r.normalize_abpp(chip)
            assert out.dtype == np.float32
            assert out[0, 0] == pytest.approx(1.0)
            assert out[1, 0] == pytest.approx(1024.0 / 2047.0)

            r.metadata.abpp = None      # falls back to dtype width
            out2 = r.normalize_abpp(
                np.array([[255]], dtype=np.uint8))
            assert out2[0, 0] == pytest.approx(1.0)

    def test_get_lut(self, monkeypatch, tmp_path):
        """get_lut surfaces a GDAL colormap as an (N, 3) array."""
        from grdl.IO.eo.nitf import EONITFReader

        path = tmp_path / "fake.ntf"
        path.write_bytes(b'')
        parent, segs = _make_two_segment_fakes(str(path))
        _patch_rasterio_open(monkeypatch, parent, segs)

        with EONITFReader(path) as r:
            assert r.get_lut() is None      # fakes carry no colormap
            r.dataset.colormap = lambda b: {
                0: (10, 20, 30, 255), 1: (40, 50, 60, 255)}
            lut = r.get_lut()
            assert lut.shape == (2, 3)
            assert tuple(lut[1]) == (40, 50, 60)


class TestRemotePaths:
    """GDAL virtual-filesystem path normalization."""

    def test_normalize_remote_path(self):
        from grdl.IO.eo.nitf import _normalize_remote_path

        assert (_normalize_remote_path('https://x.com/a.ntf')
                == '/vsicurl/https://x.com/a.ntf')
        assert (_normalize_remote_path('http://x.com/a.ntf')
                == '/vsicurl/http://x.com/a.ntf')
        assert (_normalize_remote_path('s3://bucket/key.ntf')
                == '/vsis3/bucket/key.ntf')
        assert (_normalize_remote_path('/vsicurl/https://x/a.ntf')
                == '/vsicurl/https://x/a.ntf')
        assert _normalize_remote_path('/local/file.ntf') is None
        assert _normalize_remote_path('rel/file.ntf') is None

    def test_remote_open_bypasses_exists_check(
        self, monkeypatch,
    ):
        """https:// URI opens via /vsicurl/ without touching disk."""
        from grdl.IO.eo.nitf import EONITFReader

        parent = _FakeDataset(
            height=8, width=8, count=1,
            data=np.zeros((8, 8), dtype=np.uint8),
        )
        opened_paths = []

        def fake_open(path, *a, **k):
            opened_paths.append(str(path))
            return parent

        import grdl.IO.eo.nitf as nitf_module
        monkeypatch.setattr(nitf_module.rasterio, 'open', fake_open)

        with EONITFReader('https://example.com/img.ntf') as r:
            assert r.metadata.rows == 8
            assert r.filepath == '/vsicurl/https://example.com/img.ntf'
        assert opened_paths == ['/vsicurl/https://example.com/img.ntf']
