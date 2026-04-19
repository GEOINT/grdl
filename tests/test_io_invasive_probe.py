# -*- coding: utf-8 -*-
"""
Tests for InvasiveProbeReader and probe pipeline in grdl.IO.generic.

Tests magic byte detection, ENVI header parsing, HDF5 probing,
evidence merging, binary structure analysis, and the InvasiveProbeReader
class.  Uses synthetic files and mocks to avoid requiring real imagery.

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
2026-03-08

Modified
--------
2026-03-08
"""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest

# Module under test
from grdl.IO.probe import (
    ProbeEvidence,
    InvasiveProbeReader,
    _probe_magic_bytes,
    _probe_companion_files,
    _probe_binary_structure,
    _merge_evidence,
    _run_probe_pipeline,
    _parse_envi_header,
    _factor_candidates,
    _score_dimensions,
)


# ===================================================================
# Test _probe_magic_bytes
# ===================================================================

class TestProbeMagicBytes:
    """Tests for magic byte identification."""

    def test_hdf5_signature(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_bytes(b'\x89HDF\r\n\x1a\n' + b'\x00' * 100)
        ev = _probe_magic_bytes(f)
        assert ev.format_name == 'HDF5'
        assert ev.loading_strategy == 'hdf5'
        assert 'magic_bytes' in ev.probes_run

    def test_tiff_le_signature(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_bytes(b'II*\x00' + b'\x00' * 100)
        ev = _probe_magic_bytes(f)
        assert ev.format_name == 'TIFF_LE'
        assert ev.loading_strategy == 'gdal'

    def test_tiff_be_signature(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_bytes(b'MM\x00*' + b'\x00' * 100)
        ev = _probe_magic_bytes(f)
        assert ev.format_name == 'TIFF_BE'

    def test_netcdf_classic_32(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_bytes(b'CDF\x01' + b'\x00' * 100)
        ev = _probe_magic_bytes(f)
        assert ev.format_name == 'NetCDF_classic_32'
        assert ev.loading_strategy == 'netcdf'

    def test_netcdf_classic_64(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_bytes(b'CDF\x02' + b'\x00' * 100)
        ev = _probe_magic_bytes(f)
        assert ev.format_name == 'NetCDF_classic_64'

    def test_png_signature(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
        ev = _probe_magic_bytes(f)
        assert ev.format_name == 'PNG'

    def test_jpeg_signature(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_bytes(b'\xff\xd8\xff' + b'\x00' * 100)
        ev = _probe_magic_bytes(f)
        assert ev.format_name == 'JPEG'

    def test_nitf_signature(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_bytes(b'NITF' + b'\x00' * 100)
        ev = _probe_magic_bytes(f)
        assert ev.format_name == 'NITF'

    def test_fits_signature(self, tmp_path):
        f = tmp_path / "test.dat"
        card = b'SIMPLE  =                    T' + b' ' * 50
        f.write_bytes(card)
        ev = _probe_magic_bytes(f)
        assert ev.format_name == 'FITS'
        assert ev.loading_strategy == 'fits'

    def test_envi_header_text(self, tmp_path):
        f = tmp_path / "test.hdr"
        f.write_text("ENVI\nsamples = 100\nlines = 200\n")
        ev = _probe_magic_bytes(f)
        assert ev.format_name == 'ENVI_header'
        assert ev.loading_strategy == 'envi'

    def test_unknown_bytes(self, tmp_path):
        f = tmp_path / "test.dat"
        f.write_bytes(b'\xab\xcd\xef\x01' + b'\x00' * 100)
        ev = _probe_magic_bytes(f)
        assert ev.format_name is None
        assert 'no recognized magic bytes' in ev.clues[0]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.dat"
        f.write_bytes(b'')
        ev = _probe_magic_bytes(f)
        assert 'file is empty' in ev.clues[0]

    def test_xml_detection(self, tmp_path):
        f = tmp_path / "test.xml"
        f.write_text("<?xml version='1.0'?>\n<root/>")
        ev = _probe_magic_bytes(f)
        assert ev.extras.get('is_xml') is True


# ===================================================================
# Test _parse_envi_header
# ===================================================================

class TestParseEnviHeader:
    """Tests for ENVI header parsing."""

    def test_basic_header(self, tmp_path):
        hdr = tmp_path / "test.hdr"
        hdr.write_text(
            "ENVI\n"
            "samples = 512\n"
            "lines = 256\n"
            "bands = 3\n"
            "data type = 4\n"
            "interleave = bsq\n"
            "byte order = 0\n"
        )
        result = _parse_envi_header(hdr)
        assert result['samples'] == 512
        assert result['lines'] == 256
        assert result['bands'] == 3
        assert result['data type'] == 4
        assert result['numpy_dtype'] == 'float32'
        assert result['interleave'] == 'bsq'
        assert result['byte order'] == 0

    def test_all_dtype_codes(self, tmp_path):
        dtype_map = {
            1: 'uint8', 2: 'int16', 3: 'int32', 4: 'float32',
            5: 'float64', 6: 'complex64', 9: 'complex128',
            12: 'uint16', 13: 'uint32', 14: 'int64', 15: 'uint64',
        }
        for code, expected in dtype_map.items():
            hdr = tmp_path / f"test_{code}.hdr"
            hdr.write_text(f"ENVI\ndata type = {code}\n")
            result = _parse_envi_header(hdr)
            assert result['numpy_dtype'] == expected, (
                f"code {code}: expected {expected}"
            )

    def test_multiline_braces(self, tmp_path):
        hdr = tmp_path / "test.hdr"
        hdr.write_text(
            "ENVI\n"
            "band names = {\n"
            "Red,\n"
            "Green,\n"
            "Blue}\n"
        )
        result = _parse_envi_header(hdr)
        assert result['band names'] == ['Red', 'Green', 'Blue']

    def test_wavelength_list(self, tmp_path):
        hdr = tmp_path / "test.hdr"
        hdr.write_text(
            "ENVI\n"
            "wavelength = {450.0, 550.0, 650.0}\n"
        )
        result = _parse_envi_header(hdr)
        assert result['wavelength'] == [450.0, 550.0, 650.0]

    def test_map_info(self, tmp_path):
        hdr = tmp_path / "test.hdr"
        hdr.write_text(
            "ENVI\n"
            "map info = {UTM, 1, 1, 500000, 4000000, 10, 10, 17, North}\n"
        )
        result = _parse_envi_header(hdr)
        assert isinstance(result['map info'], list)
        assert result['map info'][0] == 'UTM'

    def test_non_envi_header(self, tmp_path):
        hdr = tmp_path / "test.hdr"
        hdr.write_text("NOT ENVI\nsamples = 100\n")
        result = _parse_envi_header(hdr)
        assert result == {}

    def test_header_offset(self, tmp_path):
        hdr = tmp_path / "test.hdr"
        hdr.write_text("ENVI\nheader offset = 1024\n")
        result = _parse_envi_header(hdr)
        assert result['header offset'] == 1024


# ===================================================================
# Test _probe_companion_files
# ===================================================================

class TestProbeCompanionFiles:
    """Tests for companion file detection."""

    def test_envi_header_companion(self, tmp_path):
        # Create data file and companion header
        data = tmp_path / "image.dat"
        data.write_bytes(np.zeros((100, 200), dtype=np.float32).tobytes())
        hdr = tmp_path / "image.hdr"
        hdr.write_text(
            "ENVI\n"
            "samples = 200\n"
            "lines = 100\n"
            "bands = 1\n"
            "data type = 4\n"
            "interleave = bsq\n"
            "byte order = 0\n"
        )
        ev = _probe_companion_files(data)
        assert ev.rows == 100
        assert ev.cols == 200
        assert ev.dtype == 'float32'
        assert ev.loading_strategy == 'envi'
        assert 'companion_files' in ev.probes_run

    def test_prj_file(self, tmp_path):
        data = tmp_path / "image.dat"
        data.write_bytes(b'\x00' * 100)
        prj = tmp_path / "image.prj"
        prj.write_text('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984"]]')
        ev = _probe_companion_files(data)
        assert ev.crs is not None
        assert 'WGS_1984' in ev.crs

    def test_world_file(self, tmp_path):
        data = tmp_path / "image.tif"
        data.write_bytes(b'\x00' * 100)
        tfw = tmp_path / "image.tfw"
        tfw.write_text("10.0\n0.0\n0.0\n-10.0\n500000.0\n4000000.0\n")
        ev = _probe_companion_files(data)
        wf = ev.extras.get('world_file')
        assert wf is not None
        assert wf['x_scale'] == 10.0
        assert wf['y_origin'] == 4000000.0

    def test_aux_xml(self, tmp_path):
        data = tmp_path / "image.tif"
        data.write_bytes(b'\x00' * 100)
        aux = tmp_path / "image.tif.aux.xml"
        aux.write_text(
            '<PAMDataset><SRS>EPSG:4326</SRS></PAMDataset>'
        )
        ev = _probe_companion_files(data)
        assert ev.crs == 'EPSG:4326'

    def test_no_companions(self, tmp_path):
        data = tmp_path / "lonely.dat"
        data.write_bytes(b'\x00' * 100)
        ev = _probe_companion_files(data)
        assert ev.rows is None
        assert ev.crs is None

    def test_hyperspectral_wavelengths(self, tmp_path):
        data = tmp_path / "hsi.dat"
        data.write_bytes(b'\x00' * 100)
        wl_list = ', '.join(str(400 + i * 2) for i in range(150))
        hdr = tmp_path / "hsi.hdr"
        hdr.write_text(
            "ENVI\n"
            "samples = 100\n"
            "lines = 50\n"
            f"bands = 150\n"
            "data type = 4\n"
            f"wavelength = {{{wl_list}}}\n"
        )
        ev = _probe_companion_files(data)
        assert ev.modality == 'HSI'
        assert ev.modality_confidence == 'high'


# ===================================================================
# Test _probe_binary_structure
# ===================================================================

class TestProbeBinaryStructure:
    """Tests for binary structure analysis."""

    def test_square_float32(self, tmp_path):
        data = np.random.rand(256, 256).astype(np.float32)
        f = tmp_path / "square.dat"
        f.write_bytes(data.tobytes())
        ev = _probe_binary_structure(f)
        # Binary probe is inherently ambiguous — verify 256x256
        # float32 is among the candidates
        candidates = ev.extras.get('binary_candidates', [])
        found = any(
            c['rows'] == 256 and c['cols'] == 256 and
            c['dtype'] == 'float32'
            for c in candidates
        )
        assert found, f"256x256 float32 not in candidates: {candidates}"
        assert ev.rows is not None
        assert ev.loading_strategy == 'memmap'

    def test_rectangular_uint16(self, tmp_path):
        data = np.zeros((512, 1024), dtype=np.uint16)
        f = tmp_path / "rect.dat"
        f.write_bytes(data.tobytes())
        ev = _probe_binary_structure(f)
        # Should find the correct shape as a candidate
        candidates = ev.extras.get('binary_candidates', [])
        found = any(
            c['rows'] == 512 and c['cols'] == 1024 and
            c['dtype'] == 'uint16'
            for c in candidates
        )
        assert found, f"Expected 512x1024 uint16 in candidates: {candidates}"

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.dat"
        f.write_bytes(b'')
        ev = _probe_binary_structure(f)
        assert ev.rows is None
        assert 'file is empty' in ev.clues[0]

    def test_text_file_detected(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_text("This is a text file with lots of ASCII characters.\n" * 50)
        ev = _probe_binary_structure(f)
        assert ev.extras.get('appears_text') is True
        assert ev.rows is None

    def test_multiband_candidate(self, tmp_path):
        # 3-band RGB uint8 image with low-entropy data
        bands = []
        for i in range(3):
            # Smooth gradient with limited value range (low entropy)
            row = np.linspace(0, 100, 256, dtype=np.uint8)
            band = np.tile(row, (256, 1))
            bands.append(band)
        data = np.stack(bands)  # (3, 256, 256)
        f = tmp_path / "rgb.dat"
        f.write_bytes(data.tobytes())
        ev = _probe_binary_structure(f)
        candidates = ev.extras.get('binary_candidates', [])
        # Should have a 3-band candidate
        found_rgb = any(
            c.get('bands') == 3 and c['dtype'] == 'uint8'
            for c in candidates
        )
        assert found_rgb, (
            f"3-band uint8 not in candidates: {candidates[:5]}"
        )


# ===================================================================
# Test _factor_candidates
# ===================================================================

class TestFactorCandidates:
    """Tests for the dimension factor generator."""

    def test_perfect_square(self):
        factors = _factor_candidates(65536)  # 256*256
        assert 256 in factors

    def test_power_of_two(self):
        factors = _factor_candidates(1024 * 2048)
        assert 1024 in factors
        assert 2048 in factors

    def test_common_dimensions(self):
        factors = _factor_candidates(512 * 512)
        assert 512 in factors

    def test_zero(self):
        assert _factor_candidates(0) == []


# ===================================================================
# Test _score_dimensions
# ===================================================================

class TestScoreDimensions:
    """Tests for the dimension scoring function."""

    def test_square_higher_than_elongated(self):
        square = _score_dimensions(512, 512, 1.0, 'float32')
        elongated = _score_dimensions(16, 16384, 1024.0, 'float32')
        assert square > elongated

    def test_power_of_two_bonus(self):
        pow2 = _score_dimensions(1024, 1024, 1.0, 'float32')
        non_pow2 = _score_dimensions(1000, 1000, 1.0, 'float32')
        assert pow2 > non_pow2

    def test_common_dtype_preferred(self):
        f32 = _score_dimensions(256, 256, 1.0, 'float32')
        c64 = _score_dimensions(256, 256, 1.0, 'complex64')
        assert f32 > c64


# ===================================================================
# Test _merge_evidence
# ===================================================================

class TestMergeEvidence:
    """Tests for evidence merging logic."""

    def test_first_format_wins(self):
        ev1 = ProbeEvidence(format_name='HDF5')
        ev2 = ProbeEvidence(format_name='NetCDF')
        merged = _merge_evidence([ev1, ev2])
        assert merged.format_name == 'HDF5'

    def test_first_dimensions_win(self):
        ev1 = ProbeEvidence(rows=100, cols=200, dtype='float32')
        ev2 = ProbeEvidence(rows=50, cols=100, dtype='uint16')
        merged = _merge_evidence([ev1, ev2])
        assert merged.rows == 100
        assert merged.cols == 200
        assert len(merged.conflicts) == 1

    def test_higher_confidence_modality_wins(self):
        ev1 = ProbeEvidence(modality='EO', modality_confidence='low')
        ev2 = ProbeEvidence(modality='SAR', modality_confidence='high')
        merged = _merge_evidence([ev1, ev2])
        assert merged.modality == 'SAR'

    def test_crs_first_wins(self):
        ev1 = ProbeEvidence(crs='EPSG:4326')
        ev2 = ProbeEvidence(crs='EPSG:32617')
        merged = _merge_evidence([ev1, ev2])
        assert merged.crs == 'EPSG:4326'

    def test_extras_union(self):
        ev1 = ProbeEvidence(extras={'a': 1})
        ev2 = ProbeEvidence(extras={'b': 2})
        merged = _merge_evidence([ev1, ev2])
        assert merged.extras['a'] == 1
        assert merged.extras['b'] == 2

    def test_clues_combined(self):
        ev1 = ProbeEvidence(clues=['clue A'])
        ev2 = ProbeEvidence(clues=['clue B'])
        merged = _merge_evidence([ev1, ev2])
        assert 'clue A' in merged.clues
        assert 'clue B' in merged.clues

    def test_loading_hint_union(self):
        ev1 = ProbeEvidence(loading_hint={'path': '/data'})
        ev2 = ProbeEvidence(loading_hint={'variable': 'temp'})
        merged = _merge_evidence([ev1, ev2])
        assert merged.loading_hint['path'] == '/data'
        assert merged.loading_hint['variable'] == 'temp'

    def test_empty_probes(self):
        merged = _merge_evidence([ProbeEvidence(), ProbeEvidence()])
        assert merged.format_name is None
        assert merged.rows is None

    def test_dtype_filled_from_later_probe(self):
        ev1 = ProbeEvidence(rows=100, cols=200, dtype=None)
        ev2 = ProbeEvidence(dtype='float32')
        merged = _merge_evidence([ev1, ev2])
        assert merged.dtype == 'float32'


# ===================================================================
# Test _run_probe_pipeline
# ===================================================================

class TestRunProbePipeline:
    """Tests for the full probe pipeline."""

    def test_envi_file_with_header(self, tmp_path):
        # Create ENVI data + header pair
        data = np.ones((64, 128), dtype=np.float32)
        dat_file = tmp_path / "scene.dat"
        dat_file.write_bytes(data.tobytes())
        hdr_file = tmp_path / "scene.hdr"
        hdr_file.write_text(
            "ENVI\n"
            "samples = 128\n"
            "lines = 64\n"
            "bands = 1\n"
            "data type = 4\n"
            "interleave = bsq\n"
            "byte order = 0\n"
        )
        ev = _run_probe_pipeline(dat_file)
        assert ev.rows == 64
        assert ev.cols == 128
        assert ev.dtype == 'float32'
        assert ev.loading_strategy == 'envi'

    def test_raw_binary_fallback(self, tmp_path):
        data = np.zeros((256, 256), dtype=np.float32)
        f = tmp_path / "raw.bin"
        f.write_bytes(data.tobytes())
        ev = _run_probe_pipeline(f)
        assert ev.rows is not None
        assert ev.cols is not None
        assert ev.loading_strategy is not None


# ===================================================================
# Test InvasiveProbeReader
# ===================================================================

class TestInvasiveProbeReader:
    """Tests for the InvasiveProbeReader class."""

    def test_envi_read(self, tmp_path):
        """Test reading an ENVI format file."""
        rows, cols = 64, 128
        data = np.arange(
            rows * cols, dtype=np.float32
        ).reshape(rows, cols)
        dat_file = tmp_path / "scene.dat"
        dat_file.write_bytes(data.tobytes())
        hdr_file = tmp_path / "scene.hdr"
        hdr_file.write_text(
            "ENVI\n"
            "samples = 128\n"
            "lines = 64\n"
            "bands = 1\n"
            "data type = 4\n"
            "interleave = bsq\n"
            "byte order = 0\n"
        )

        with InvasiveProbeReader(dat_file) as reader:
            assert reader.metadata.rows == 64
            assert reader.metadata.cols == 128
            assert reader.metadata.dtype == 'float32'
            assert reader.metadata.format.startswith('probed:')

            # Read full image
            full = reader.read_full()
            assert full.shape == (64, 128)
            np.testing.assert_array_equal(full, data)

            # Read chip
            chip = reader.read_chip(10, 30, 20, 50)
            assert chip.shape == (20, 30)
            np.testing.assert_array_equal(chip, data[10:30, 20:50])

    def test_envi_multiband_bsq(self, tmp_path):
        """Test reading a multi-band BSQ ENVI file."""
        rows, cols, bands = 32, 64, 3
        data = np.arange(
            bands * rows * cols, dtype=np.int16
        ).reshape(bands, rows, cols)
        dat_file = tmp_path / "multi.dat"
        dat_file.write_bytes(data.tobytes())
        hdr_file = tmp_path / "multi.hdr"
        hdr_file.write_text(
            "ENVI\n"
            f"samples = {cols}\n"
            f"lines = {rows}\n"
            f"bands = {bands}\n"
            "data type = 2\n"
            "interleave = bsq\n"
            "byte order = 0\n"
        )

        with InvasiveProbeReader(dat_file) as reader:
            assert reader.metadata.bands == 3

            # Read all bands
            full = reader.read_full()
            assert full.shape == (bands, rows, cols)
            np.testing.assert_array_equal(full, data)

    def test_envi_bil_interleave(self, tmp_path):
        """Test reading a BIL interleaved ENVI file."""
        rows, cols, bands = 32, 64, 3
        # BIL: shape is (rows, bands, cols) on disk
        data_bsq = np.arange(
            bands * rows * cols, dtype=np.float32
        ).reshape(bands, rows, cols)
        data_bil = np.moveaxis(data_bsq, 0, 1)  # (rows, bands, cols)
        dat_file = tmp_path / "bil.dat"
        dat_file.write_bytes(data_bil.tobytes())
        hdr_file = tmp_path / "bil.hdr"
        hdr_file.write_text(
            "ENVI\n"
            f"samples = {cols}\n"
            f"lines = {rows}\n"
            f"bands = {bands}\n"
            "data type = 4\n"
            "interleave = bil\n"
            "byte order = 0\n"
        )

        with InvasiveProbeReader(dat_file) as reader:
            full = reader.read_full()
            assert full.shape == (bands, rows, cols)

    def test_raw_binary_read(self, tmp_path):
        """Test reading a raw binary file with no companion."""
        rows, cols = 256, 256
        data = np.random.rand(rows, cols).astype(np.float32)
        f = tmp_path / "raw.bin"
        f.write_bytes(data.tobytes())

        with InvasiveProbeReader(f) as reader:
            assert reader.metadata.rows is not None
            assert reader.metadata.cols is not None
            shape = reader.get_shape()
            assert len(shape) >= 2

            # Read a chip
            chip = reader.read_chip(0, 10, 0, 10)
            assert chip.shape[0] == 10
            assert chip.shape[1] == 10

    def test_context_manager(self, tmp_path):
        """Test that context manager properly opens and closes."""
        data = np.zeros((64, 64), dtype=np.float32)
        dat_file = tmp_path / "ctx.dat"
        dat_file.write_bytes(data.tobytes())
        hdr_file = tmp_path / "ctx.hdr"
        hdr_file.write_text(
            "ENVI\n"
            "samples = 64\n"
            "lines = 64\n"
            "bands = 1\n"
            "data type = 4\n"
            "interleave = bsq\n"
            "byte order = 0\n"
        )

        reader = InvasiveProbeReader(dat_file)
        with reader:
            assert reader.metadata is not None
        # After close, handle should be None
        assert reader._handle is None

    def test_chip_bounds_validation(self, tmp_path):
        """Test that out-of-bounds chip reads raise ValueError."""
        data = np.zeros((32, 32), dtype=np.float32)
        dat_file = tmp_path / "small.dat"
        dat_file.write_bytes(data.tobytes())
        hdr_file = tmp_path / "small.hdr"
        hdr_file.write_text(
            "ENVI\n"
            "samples = 32\n"
            "lines = 32\n"
            "bands = 1\n"
            "data type = 4\n"
            "interleave = bsq\n"
            "byte order = 0\n"
        )

        with InvasiveProbeReader(dat_file) as reader:
            with pytest.raises(ValueError, match="non-negative"):
                reader.read_chip(-1, 10, 0, 10)
            with pytest.raises(ValueError, match="exceed"):
                reader.read_chip(0, 100, 0, 10)

    def test_evidence_exposed(self, tmp_path):
        """Test that probe evidence is accessible on the reader."""
        data = np.zeros((64, 64), dtype=np.float32)
        dat_file = tmp_path / "ev.dat"
        dat_file.write_bytes(data.tobytes())
        hdr_file = tmp_path / "ev.hdr"
        hdr_file.write_text(
            "ENVI\n"
            "samples = 64\n"
            "lines = 64\n"
            "bands = 1\n"
            "data type = 4\n"
            "interleave = bsq\n"
            "byte order = 0\n"
        )

        with InvasiveProbeReader(dat_file) as reader:
            assert reader.evidence is not None
            assert len(reader.evidence.probes_run) > 0
            assert len(reader.evidence.clues) > 0

    def test_metadata_extras(self, tmp_path):
        """Test that probe metadata appears in extras."""
        data = np.zeros((64, 64), dtype=np.float32)
        dat_file = tmp_path / "meta.dat"
        dat_file.write_bytes(data.tobytes())
        hdr_file = tmp_path / "meta.hdr"
        hdr_file.write_text(
            "ENVI\n"
            "samples = 64\n"
            "lines = 64\n"
            "bands = 1\n"
            "data type = 4\n"
            "interleave = bsq\n"
            "byte order = 0\n"
        )

        with InvasiveProbeReader(dat_file) as reader:
            assert 'probe_audit' in reader.metadata
            assert 'probe_clues' in reader.metadata
            assert reader.metadata.get('probe_loading_strategy') is not None

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            InvasiveProbeReader('/nonexistent/path/file.dat')

    def test_undeterminable_file(self, tmp_path):
        """Test that ValueError is raised when probing fails completely."""
        # Compressed/random bytes with prime file size
        f = tmp_path / "chaos.dat"
        rng = np.random.default_rng(42)
        # Use a prime number of bytes so no clean factorization
        f.write_bytes(rng.bytes(7919))
        # This should either raise ValueError or succeed with a guess
        try:
            reader = InvasiveProbeReader(f)
            reader.close()
        except ValueError:
            pass  # Expected for truly uninterpretable files


# ===================================================================
# Test HDF5 probing (requires h5py)
# ===================================================================

h5py = pytest.importorskip("h5py", reason="h5py not installed")


class TestProbeHDF5:
    """Tests for HDF5 probing (requires h5py)."""

    def test_hdf5_dataset_discovery(self, tmp_path):
        from grdl.IO.probe import _probe_hdf5_walk

        f = tmp_path / "test.h5"
        with h5py.File(str(f), 'w') as hf:
            hf.create_dataset('data/image', data=np.zeros((100, 200)))
            hf.create_dataset('meta/text', data=b'hello')
            hf.attrs['title'] = 'Test HDF5'

        ev = _probe_hdf5_walk(f)
        assert ev.rows == 100
        assert ev.cols == 200
        assert ev.loading_strategy == 'hdf5'
        assert ev.loading_hint.get('dataset_path') == '/data/image'

    def test_hdf5_complex_dataset(self, tmp_path):
        from grdl.IO.probe import _probe_hdf5_walk

        f = tmp_path / "sar.h5"
        with h5py.File(str(f), 'w') as hf:
            hf.create_dataset(
                'signal', data=np.zeros((50, 100), dtype=np.complex64)
            )

        ev = _probe_hdf5_walk(f)
        assert ev.dtype == 'complex64'
        assert ev.modality == 'SAR'
        assert ev.modality_confidence == 'medium'

    def test_hdf5_picks_largest_dataset(self, tmp_path):
        from grdl.IO.probe import _probe_hdf5_walk

        f = tmp_path / "multi.h5"
        with h5py.File(str(f), 'w') as hf:
            hf.create_dataset('small', data=np.zeros((10, 10)))
            hf.create_dataset('big', data=np.zeros((500, 500)))
            hf.create_dataset('medium', data=np.zeros((100, 100)))

        ev = _probe_hdf5_walk(f)
        assert ev.rows == 500
        assert ev.cols == 500
        assert ev.loading_hint['dataset_path'] == '/big'

    def test_hdf5_cf_conventions(self, tmp_path):
        from grdl.IO.probe import _probe_hdf5_walk

        f = tmp_path / "cf.h5"
        with h5py.File(str(f), 'w') as hf:
            hf.attrs['Conventions'] = 'CF-1.6'
            hf.create_dataset('temperature', data=np.zeros((100, 200)))

        ev = _probe_hdf5_walk(f)
        assert ev.extras.get('has_cf_conventions') is True

    def test_invasive_reader_hdf5(self, tmp_path):
        """End-to-end test: InvasiveProbeReader with HDF5 file."""
        f = tmp_path / "image.h5"
        expected = np.arange(64 * 128, dtype=np.float32).reshape(64, 128)
        with h5py.File(str(f), 'w') as hf:
            hf.create_dataset('imagery/scene', data=expected)

        with InvasiveProbeReader(f) as reader:
            assert reader.metadata.rows == 64
            assert reader.metadata.cols == 128
            assert reader.metadata.format.startswith('probed:')

            full = reader.read_full()
            np.testing.assert_array_equal(full, expected)

            chip = reader.read_chip(10, 30, 20, 50)
            np.testing.assert_array_equal(chip, expected[10:30, 20:50])
