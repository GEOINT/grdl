# -*- coding: utf-8 -*-
"""
EO NITF Chip Writer Tests - write_chip, TRE serialization, open_any dispatch.

Builds small synthetic parent NITFs with rasterio (pixel data plus
RPC00B written from the RPC metadata domain), chips them with
``grdl.IO.eo.nitf_writer.write_chip``, and verifies pixels, ICHIPB
chip-to-full mapping, RPC geolocation equivalence between parent and
chip, RSM TRE serialization round-trips, and the NITF SICD/SIDD/EO
auto-dispatch in ``grdl.IO.generic.open_any``.

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
2026-06-09

Modified
--------
2026-06-09
"""

# Standard library
from datetime import datetime
from unittest import mock

# Third-party
import numpy as np
import pytest

try:
    import rasterio
    from rasterio.rpc import RPC
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

pytestmark = pytest.mark.skipif(
    not _HAS_RASTERIO, reason="rasterio not installed"
)


# ===================================================================
# Fixtures
# ===================================================================

PARENT_ROWS = 256
PARENT_COLS = 256


def _linear_rpc() -> 'RPC':
    """Build a simple linear (affine-like) RPC00B model.

    row = line_off + line_scale * P  with P = (lat - lat_off)/lat_scale
    col = samp_off + samp_scale * L  with L = (lon - long_off)/long_scale
    """
    line_num = [0.0] * 20
    line_num[2] = 1.0   # P term
    samp_num = [0.0] * 20
    samp_num[1] = 1.0   # L term
    den = [1.0] + [0.0] * 19
    return RPC(
        height_off=0.0, height_scale=500.0,
        lat_off=34.0, lat_scale=0.05,
        long_off=-117.0, long_scale=0.05,
        line_off=128.0, line_scale=128.0,
        samp_off=128.0, samp_scale=128.0,
        line_num_coeff=line_num, line_den_coeff=den,
        samp_num_coeff=samp_num, samp_den_coeff=den,
        err_bias=1.0, err_rand=0.5,
    )


@pytest.fixture
def parent_nitf(tmp_path):
    """Synthetic parent NITF with pixel data and an RPC00B TRE."""
    path = tmp_path / "parent.ntf"
    rng = np.random.default_rng(42)
    data = rng.integers(0, 255, (PARENT_ROWS, PARENT_COLS),
                        dtype=np.uint8)
    with rasterio.open(
        str(path), 'w', driver='NITF',
        height=PARENT_ROWS, width=PARENT_COLS,
        count=1, dtype='uint8', rpcs=_linear_rpc(),
    ) as ds:
        ds.write(data, 1)

    # Verify the env actually wrote a readable RPC (RPC00B from the
    # RPC metadata domain — GDAL NITF creation option RPC00B=YES).
    with rasterio.open(str(path)) as ds:
        assert ds.rpcs is not None, (
            "GDAL NITF driver did not round-trip RPC in this env")

    return path, data


# ===================================================================
# write_chip — pixels, shape, dtype
# ===================================================================

class TestWriteChipPixels:
    """write_chip output opens with EONITFReader with correct pixels."""

    def test_roundtrip_pixels_shape_dtype(self, parent_nitf, tmp_path):
        from grdl.IO.eo.nitf import EONITFReader
        from grdl.IO.eo.nitf_writer import write_chip

        path, data = parent_nitf
        out = tmp_path / "chip.ntf"
        with EONITFReader(path) as reader:
            result = write_chip(
                reader=reader,
                row_start=32, row_end=96,
                col_start=40, col_end=104,
                output_path=out,
            )
        assert result == out
        assert out.exists()

        with EONITFReader(out) as chip_reader:
            assert chip_reader.metadata.rows == 64
            assert chip_reader.metadata.cols == 64
            chip = chip_reader.read_chip(0, 64, 0, 64)
            assert chip.shape == (64, 64)
            assert chip.dtype == np.uint8
            np.testing.assert_array_equal(chip, data[32:96, 40:104])

    def test_exported_from_package(self):
        from grdl.IO.eo import write_chip as exported
        from grdl.IO.eo.nitf_writer import write_chip

        assert exported is write_chip

    def test_invalid_bounds_raise(self, parent_nitf, tmp_path):
        from grdl.IO.eo.nitf import EONITFReader
        from grdl.IO.eo.nitf_writer import write_chip

        path, _ = parent_nitf
        out = tmp_path / "bad.ntf"
        with EONITFReader(path) as reader:
            with pytest.raises(ValueError, match="empty"):
                write_chip(reader, 50, 50, 0, 64, out)
            with pytest.raises(ValueError, match="empty"):
                write_chip(reader, 64, 32, 0, 64, out)
            with pytest.raises(ValueError, match="non-negative"):
                write_chip(reader, -1, 32, 0, 64, out)
            with pytest.raises(ValueError, match="exceed"):
                write_chip(reader, 0, PARENT_ROWS + 1, 0, 64, out)


# ===================================================================
# write_chip — ICHIPB chip→full mapping
# ===================================================================

class TestChipICHIPB:
    """Chip metadata.ichipb maps chip (0,0) to full-image origin."""

    def test_ichipb_offsets(self, parent_nitf, tmp_path):
        from grdl.IO.eo.nitf import EONITFReader
        from grdl.IO.eo.nitf_writer import write_chip

        path, _ = parent_nitf
        out = tmp_path / "chip.ntf"
        with EONITFReader(path) as reader:
            write_chip(
                reader=reader,
                row_start=32, row_end=96,
                col_start=40, col_end=104,
                output_path=out,
            )

        with EONITFReader(out) as chip_reader:
            ichipb = chip_reader.metadata.ichipb
            assert ichipb is not None
            # full = fi_off + fi_scale * chip  →  chip (0,0) maps to
            # full (row_start, col_start).
            assert ichipb.fi_row_off == pytest.approx(32.0, abs=1e-6)
            assert ichipb.fi_col_off == pytest.approx(40.0, abs=1e-6)
            assert ichipb.fi_row_scale == pytest.approx(1.0, abs=1e-9)
            assert ichipb.fi_col_scale == pytest.approx(1.0, abs=1e-9)
            assert ichipb.full_image_rows == PARENT_ROWS
            assert ichipb.full_image_cols == PARENT_COLS

    def test_compose_with_parent_ichipb(self, parent_nitf, tmp_path):
        """Chipping a chip composes the two ICHIPB affines."""
        from grdl.IO.eo.nitf import EONITFReader
        from grdl.IO.eo.nitf_writer import write_chip

        path, _ = parent_nitf
        chip1 = tmp_path / "chip1.ntf"
        chip2 = tmp_path / "chip2.ntf"

        with EONITFReader(path) as reader:
            write_chip(reader, 32, 160, 40, 168, chip1)

        # chip1 has an ICHIPB with offsets (32, 40); chipping it again
        # at (10, 20) must yield full-image offsets (42, 60).
        with EONITFReader(chip1) as reader:
            assert reader.metadata.ichipb is not None
            write_chip(reader, 10, 74, 20, 84, chip2)

        with EONITFReader(chip2) as chip_reader:
            ichipb = chip_reader.metadata.ichipb
            assert ichipb is not None
            assert ichipb.fi_row_off == pytest.approx(42.0, abs=1e-6)
            assert ichipb.fi_col_off == pytest.approx(60.0, abs=1e-6)


# ===================================================================
# write_chip — RPC geolocation equivalence
# ===================================================================

class TestRPCGeolocationRoundTrip:
    """Parent and chip geolocate identically through RPCGeolocation."""

    def test_image_to_latlon_matches(self, parent_nitf, tmp_path):
        from grdl.IO.eo.nitf import EONITFReader
        from grdl.IO.eo.nitf_writer import write_chip
        from grdl.geolocation.eo.rpc import RPCGeolocation

        path, _ = parent_nitf
        out = tmp_path / "chip.ntf"
        row_start, col_start = 32, 40
        with EONITFReader(path) as reader:
            write_chip(
                reader=reader,
                row_start=row_start, row_end=96,
                col_start=col_start, col_end=104,
                output_path=out,
            )
            parent_geo = RPCGeolocation.from_reader(reader)

            with EONITFReader(out) as chip_reader:
                assert chip_reader.metadata.rpc is not None
                chip_geo = RPCGeolocation.from_reader(chip_reader)

                for r, c in [(50, 60), (32, 40), (95, 103)]:
                    p_lat, p_lon, _ = parent_geo.image_to_latlon(r, c)
                    c_lat, c_lon, _ = chip_geo.image_to_latlon(
                        r - row_start, c - col_start)
                    assert c_lat == pytest.approx(p_lat, abs=1e-6)
                    assert c_lon == pytest.approx(p_lon, abs=1e-6)

    def test_rpc_passed_through_full_image_space(
        self, parent_nitf, tmp_path,
    ):
        """The chip carries the parent's full-image RPC unchanged."""
        from grdl.IO.eo.nitf import EONITFReader
        from grdl.IO.eo.nitf_writer import write_chip

        path, _ = parent_nitf
        out = tmp_path / "chip.ntf"
        with EONITFReader(path) as reader:
            parent_rpc = reader.metadata.rpc
            write_chip(reader, 32, 96, 40, 104, out)

        with EONITFReader(out) as chip_reader:
            chip_rpc = chip_reader.metadata.rpc
            assert chip_rpc is not None
            assert chip_rpc.line_off == pytest.approx(
                parent_rpc.line_off)
            assert chip_rpc.samp_off == pytest.approx(
                parent_rpc.samp_off)
            np.testing.assert_allclose(
                chip_rpc.line_num_coef, parent_rpc.line_num_coef)
            np.testing.assert_allclose(
                chip_rpc.samp_num_coef, parent_rpc.samp_num_coef)


# ===================================================================
# RSM serialization round-trip (unit level)
# ===================================================================

def _synthetic_rsmpca():
    from grdl.IO.models.eo_nitf import RSMCoefficients

    rng = np.random.default_rng(7)
    return RSMCoefficients(
        row_off=1024.5, col_off=2048.25,
        row_norm_sf=1023.5, col_norm_sf=2047.75,
        x_off=-2.0412345, y_off=0.5931234, z_off=120.5,
        x_norm_sf=0.0123, y_norm_sf=0.0234, z_norm_sf=500.0,
        row_num_powers=np.array([3, 3, 3]),
        row_den_powers=np.array([3, 3, 3]),
        col_num_powers=np.array([3, 3, 3]),
        col_den_powers=np.array([3, 3, 3]),
        row_num_coefs=rng.normal(size=20),
        row_den_coefs=np.concatenate([[1.0], rng.normal(size=19) * 1e-3]),
        col_num_coefs=rng.normal(size=20),
        col_den_coefs=np.concatenate([[1.0], rng.normal(size=19) * 1e-3]),
        rsn=2, csn=3,
        row_fit_error=0.25, col_fit_error=0.5,
    )


class TestRSMSerializationRoundTrip:
    """Serialize synthetic RSM TREs, parse back with existing parsers."""

    def test_rsmpca_roundtrip(self):
        from grdl.IO.eo.nitf import _parse_rsmpca_tre
        from grdl.IO.eo.nitf_writer import _serialize_rsmpca

        original = _synthetic_rsmpca()
        cedata = _serialize_rsmpca(
            original, image_id='GRDL_TEST', edition='ED1')
        parsed = _parse_rsmpca_tre(cedata)
        assert parsed is not None

        assert parsed.rsn == original.rsn
        assert parsed.csn == original.csn
        assert parsed.row_fit_error == pytest.approx(
            original.row_fit_error)
        assert parsed.col_fit_error == pytest.approx(
            original.col_fit_error)
        for attr in ('row_off', 'col_off', 'x_off', 'y_off', 'z_off',
                     'row_norm_sf', 'col_norm_sf',
                     'x_norm_sf', 'y_norm_sf', 'z_norm_sf'):
            assert getattr(parsed, attr) == pytest.approx(
                getattr(original, attr), rel=1e-13), attr
        for attr in ('row_num_powers', 'row_den_powers',
                     'col_num_powers', 'col_den_powers'):
            np.testing.assert_array_equal(
                getattr(parsed, attr), getattr(original, attr))
        for attr in ('row_num_coefs', 'row_den_coefs',
                     'col_num_coefs', 'col_den_coefs'):
            np.testing.assert_allclose(
                getattr(parsed, attr), getattr(original, attr),
                rtol=1e-13, err_msg=attr)

    def test_rsmida_roundtrip(self):
        from grdl.IO.eo.nitf import _parse_rsmida_tre
        from grdl.IO.eo.nitf_writer import _serialize_rsmida
        from grdl.IO.models.eo_nitf import RSMIdentification
        from grdl.IO.models.common import XYZ

        rng = np.random.default_rng(11)
        original = RSMIdentification(
            image_id='GRDL_TEST_IMAGE',
            edition='EDITION_1',
            sensor_id='SENSOR_A',
            sensor_type_id='EO_FRAME',
            image_sensor_id='ISID_X',
            collection_datetime=datetime(2026, 6, 9, 12, 30, 15, 500000),
            ground_domain_type='G',
            ground_ref_point=XYZ(x=-2.0412345, y=0.5931234, z=120.5),
            num_row_sections=2,
            num_col_sections=3,
            time_ref_row=0.5,
            time_ref_col=0.25,
            ground_domain_vertices=rng.normal(size=(8, 3)),
            full_image_rows=2048,
            full_image_cols=4096,
            min_row=0, max_row=2047, min_col=0, max_col=4095,
        )

        cedata = _serialize_rsmida(original)
        assert len(cedata) == 1376  # GRNDD='G' fixed length
        parsed = _parse_rsmida_tre(cedata)
        assert parsed is not None

        assert parsed.image_id == original.image_id
        assert parsed.edition == original.edition
        assert parsed.sensor_id == original.sensor_id
        assert parsed.sensor_type_id == original.sensor_type_id
        assert parsed.image_sensor_id == original.image_sensor_id
        assert parsed.collection_datetime == original.collection_datetime
        assert parsed.ground_domain_type == 'G'
        assert parsed.num_row_sections == 2
        assert parsed.num_col_sections == 3
        assert parsed.time_ref_row == pytest.approx(0.5)
        assert parsed.time_ref_col == pytest.approx(0.25)
        assert parsed.ground_ref_point.x == pytest.approx(
            original.ground_ref_point.x, rel=1e-13)
        assert parsed.ground_ref_point.y == pytest.approx(
            original.ground_ref_point.y, rel=1e-13)
        assert parsed.ground_ref_point.z == pytest.approx(
            original.ground_ref_point.z, rel=1e-13)
        np.testing.assert_allclose(
            parsed.ground_domain_vertices,
            original.ground_domain_vertices, rtol=1e-13)
        assert parsed.full_image_rows == 2048
        assert parsed.full_image_cols == 4096
        assert parsed.min_row == 0
        assert parsed.max_row == 2047
        assert parsed.min_col == 0
        assert parsed.max_col == 4095


# ===================================================================
# RSM propagation through a written chip file
# ===================================================================

class _FakeRSMReader:
    """Minimal reader carrying RSM metadata for write_chip."""

    def __init__(self, metadata, data):
        self.metadata = metadata
        self._data = data

    def read_chip(self, row_start, row_end, col_start, col_end,
                  bands=None):
        return self._data[row_start:row_end, col_start:col_end].copy()


class TestRSMFilePropagation:
    """write_chip carries RSMIDA + RSMPCA into the output NITF."""

    def test_rsm_tres_written_and_parsed(self, tmp_path):
        from grdl.IO.eo.nitf import EONITFReader
        from grdl.IO.eo.nitf_writer import write_chip
        from grdl.IO.models.eo_nitf import (
            EONITFMetadata, RSMIdentification,
        )
        from grdl.IO.models.common import XYZ

        rsm = _synthetic_rsmpca()
        rsm_id = RSMIdentification(
            image_id='GRDL_RSM_FILE',
            edition='ED1',
            ground_domain_type='G',
            ground_ref_point=XYZ(x=-2.0, y=0.6, z=100.0),
            num_row_sections=1,
            num_col_sections=1,
            full_image_rows=128,
            full_image_cols=128,
        )
        meta = EONITFMetadata(
            format='NITF', rows=128, cols=128, bands=1, dtype='uint8',
            rsm=rsm, rsm_id=rsm_id,
        )
        data = np.arange(128 * 128, dtype=np.uint16).reshape(
            128, 128).astype(np.uint8)
        reader = _FakeRSMReader(meta, data)

        out = tmp_path / "rsm_chip.ntf"
        write_chip(reader, 16, 80, 24, 88, out)

        with EONITFReader(out) as chip_reader:
            assert chip_reader.metadata.rsm is not None
            assert chip_reader.metadata.rsm_id is not None
            got = chip_reader.metadata.rsm
            assert got.rsn == rsm.rsn
            assert got.csn == rsm.csn
            assert got.row_off == pytest.approx(rsm.row_off, rel=1e-12)
            np.testing.assert_allclose(
                got.row_num_coefs, rsm.row_num_coefs, rtol=1e-12)
            assert (chip_reader.metadata.rsm_id.image_id
                    == 'GRDL_RSM_FILE')
            ichipb = chip_reader.metadata.ichipb
            assert ichipb is not None
            assert ichipb.fi_row_off == pytest.approx(16.0, abs=1e-6)
            assert ichipb.fi_col_off == pytest.approx(24.0, abs=1e-6)


# ===================================================================
# open_any NITF dispatch
# ===================================================================

class TestOpenAnyNITFDispatch:
    """open_any routes NITF files by SICD/SIDD/EO sniff."""

    def test_plain_eo_nitf_routes_to_eonitfreader(self, parent_nitf):
        from grdl.IO.eo.nitf import EONITFReader
        from grdl.IO.generic import open_any

        path, _ = parent_nitf
        with open_any(path) as reader:
            assert isinstance(reader, EONITFReader)

    def test_sniff_detects_sicd_marker_bytes(self, tmp_path):
        from grdl.IO.generic import _sniff_nitf_kind

        path = tmp_path / "fake_sicd.ntf"
        path.write_bytes(
            b'NITF02.10' + b'\x00' * 256
            + b'<SICD xmlns="urn:SICD:1.3.0">' + b'\x00' * 64)
        assert _sniff_nitf_kind(path) == 'SICD'

    def test_sniff_detects_sidd_marker_bytes(self, tmp_path):
        from grdl.IO.generic import _sniff_nitf_kind

        path = tmp_path / "fake_sidd.ntf"
        path.write_bytes(
            b'NITF02.10' + b'\x00' * 256
            + b'<SIDD xmlns="urn:SIDD:2.0.0">' + b'\x00' * 64)
        assert _sniff_nitf_kind(path) == 'SIDD'

    def test_sniff_eo_nitf(self, parent_nitf):
        from grdl.IO.generic import _sniff_nitf_kind

        path, _ = parent_nitf
        assert _sniff_nitf_kind(path) == 'EO'

    def test_sicd_dispatch_decision(self, tmp_path, monkeypatch):
        """A SICD sniff routes to grdl.IO.sar.SICDReader."""
        import grdl.IO.generic as generic_mod
        from grdl.IO.generic import open_any

        path = tmp_path / "fake_sicd.ntf"
        path.write_bytes(b'NITF02.10' + b'<SICD ' + b'\x00' * 64)

        monkeypatch.setattr(
            generic_mod, '_sniff_nitf_kind', lambda p: 'SICD')

        sentinel = mock.MagicMock(name='sicd_reader')
        sar_mod = mock.MagicMock()
        sar_mod.SICDReader.return_value = sentinel

        real_import = generic_mod.importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == 'grdl.IO.sar':
                return sar_mod
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(
            generic_mod.importlib, 'import_module', fake_import)

        result = open_any(path)
        assert result is sentinel
        sar_mod.SICDReader.assert_called_once_with(path)

    def test_sidd_dispatch_decision(self, tmp_path, monkeypatch):
        """A SIDD sniff routes to grdl.IO.sar.SIDDReader."""
        import grdl.IO.generic as generic_mod
        from grdl.IO.generic import open_any

        path = tmp_path / "fake_sidd.ntf"
        path.write_bytes(b'NITF02.10' + b'<SIDD ' + b'\x00' * 64)

        monkeypatch.setattr(
            generic_mod, '_sniff_nitf_kind', lambda p: 'SIDD')

        sentinel = mock.MagicMock(name='sidd_reader')
        sar_mod = mock.MagicMock()
        sar_mod.SIDDReader.return_value = sentinel

        real_import = generic_mod.importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == 'grdl.IO.sar':
                return sar_mod
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(
            generic_mod.importlib, 'import_module', fake_import)

        result = open_any(path)
        assert result is sentinel
        sar_mod.SIDDReader.assert_called_once_with(path)

    def test_sicd_import_failure_falls_through(
        self, parent_nitf, monkeypatch,
    ):
        """SICD sniff + missing SAR deps warns and falls through."""
        import grdl.IO.generic as generic_mod
        from grdl.IO.generic import open_any

        path, _ = parent_nitf
        monkeypatch.setattr(
            generic_mod, '_sniff_nitf_kind', lambda p: 'SICD')

        real_import = generic_mod.importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == 'grdl.IO.sar':
                raise ImportError("sarpy not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(
            generic_mod.importlib, 'import_module', fake_import)

        with pytest.warns(RuntimeWarning, match="falling back"):
            reader = open_any(path)
        # The generic cascade still opens the (actually EO) file.
        assert reader is not None
        reader.close()
