# -*- coding: utf-8 -*-
"""
Tests for grdl.IO.sar.sicd_collection — SICDCollectionReader and
open_sicd_collection factory.

Uses lightweight synthetic data (MagicMock sub-readers) so no real SICD
files are required.

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-26

Modified
--------
2026-05-26
"""

# Standard library
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List

# Third-party
import numpy as np
import pytest

# GRDL internal — reader and metadata
from grdl.IO.sar.sicd_collection import (
    SICDCollectionReader,
    open_sicd_collection,
    _infer_polarization,
)
from grdl.IO.models.sicd import (
    SICDMetadata,
    SICDRadarCollection,
    SICDRcvChannel,
    SICDCollectionMetadata,
)
from grdl.IO.models.base import ChannelMetadata
from grdl.vocabulary import PolarimetricMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sicd_meta(
    rows: int = 512,
    cols: int = 512,
    tx_rcv_polarization: str = 'H:H',
) -> SICDMetadata:
    """Return a minimal synthetic SICDMetadata."""
    rcv_channel = SICDRcvChannel(tx_rcv_polarization=tx_rcv_polarization)
    radar_collection = SICDRadarCollection(rcv_channels=[rcv_channel])
    return SICDMetadata(
        format='SICD',
        rows=rows,
        cols=cols,
        dtype='complex64',
        radar_collection=radar_collection,
    )


def _make_sicd_reader(
    rows: int = 512,
    cols: int = 512,
    tx_rcv_polarization: str = 'H:H',
) -> MagicMock:
    """Return a MagicMock that behaves like a SICDReader."""
    mock = MagicMock(spec=['metadata', 'read_chip', 'read_full', 'close'])
    mock.metadata = _make_sicd_meta(rows, cols, tx_rcv_polarization)
    mock.read_chip.return_value = np.zeros((rows, cols), dtype=np.complex64)
    return mock


def _make_collection(
    pols: List[str],
    rows: int = 512,
    cols: int = 512,
    filepaths: List[str] = None,
) -> SICDCollectionReader:
    """Build a SICDCollectionReader with mocked sub-readers."""
    if filepaths is None:
        filepaths = [f'/fake/{pol.lower()}.nitf' for pol in pols]

    pol_map = {
        'HH': 'H:H',
        'HV': 'H:V',
        'VH': 'V:H',
        'VV': 'V:V',
    }
    mock_readers = [
        _make_sicd_reader(rows, cols, pol_map.get(p, 'H:H'))
        for p in pols
    ]

    # Patch SICDReader construction and path existence
    with (
        patch('grdl.IO.sar.sicd_collection.SICDReader', side_effect=mock_readers),
        patch.object(Path, 'exists', return_value=True),
    ):
        reader = SICDCollectionReader(filepaths)

    # Stash mocks for assertion in tests
    reader._readers = mock_readers
    return reader


# ---------------------------------------------------------------------------
# _infer_polarization unit tests
# ---------------------------------------------------------------------------

class TestInferPolarization:

    def test_vv(self):
        meta = _make_sicd_meta(tx_rcv_polarization='V:V')
        assert _infer_polarization(meta) == 'VV'

    def test_hh(self):
        meta = _make_sicd_meta(tx_rcv_polarization='H:H')
        assert _infer_polarization(meta) == 'HH'

    def test_hv(self):
        meta = _make_sicd_meta(tx_rcv_polarization='H:V')
        assert _infer_polarization(meta) == 'HV'

    def test_vh(self):
        meta = _make_sicd_meta(tx_rcv_polarization='V:H')
        assert _infer_polarization(meta) == 'VH'

    def test_lowercase_input_normalised(self):
        meta = _make_sicd_meta(tx_rcv_polarization='h:v')
        assert _infer_polarization(meta) == 'HV'

    def test_no_colon_passthrough(self):
        meta = _make_sicd_meta(tx_rcv_polarization='HH')
        assert _infer_polarization(meta) == 'HH'

    def test_missing_rcv_channels_falls_back_to_tx_pol(self):
        meta = SICDMetadata(
            format='SICD',
            rows=512,
            cols=512,
            dtype='complex64',
            radar_collection=SICDRadarCollection(
                tx_polarization='V',
                rcv_channels=None,
            ),
        )
        assert _infer_polarization(meta) == 'V'

    def test_no_radar_collection_returns_unknown(self):
        meta = SICDMetadata(
            format='SICD',
            rows=512,
            cols=512,
            dtype='complex64',
            radar_collection=None,
        )
        result = _infer_polarization(meta)
        assert result == 'UNKNOWN'

    def test_multiple_rcv_channels_uses_index_zero(self):
        rcv_channels = [
            SICDRcvChannel(tx_rcv_polarization='H:H'),
            SICDRcvChannel(tx_rcv_polarization='V:V'),
        ]
        meta = SICDMetadata(
            format='SICD',
            rows=512,
            cols=512,
            dtype='complex64',
            radar_collection=SICDRadarCollection(rcv_channels=rcv_channels),
        )
        assert _infer_polarization(meta) == 'HH'


# ---------------------------------------------------------------------------
# SICDCollectionReader metadata tests
# ---------------------------------------------------------------------------

class TestSICDCollectionReaderMetadata:

    def test_axis_order_is_cyx(self):
        reader = _make_collection(['HH', 'VV'])
        assert reader.metadata.axis_order == 'CYX'

    def test_bands_count(self):
        reader = _make_collection(['HH', 'HV', 'VH', 'VV'])
        assert reader.metadata.bands == 4

    def test_channel_metadata_polarization_labels(self):
        reader = _make_collection(['HH', 'HV', 'VH', 'VV'])
        pols = [cm.polarization for cm in reader.metadata.channel_metadata]
        assert pols == ['HH', 'HV', 'VH', 'VV']

    def test_channel_metadata_indices(self):
        reader = _make_collection(['HH', 'VV'])
        assert reader.metadata.channel_metadata[0].index == 0
        assert reader.metadata.channel_metadata[1].index == 1

    def test_channel_metadata_role(self):
        reader = _make_collection(['HH', 'VV'])
        for cm in reader.metadata.channel_metadata:
            assert cm.role == 'measurement'

    def test_channel_metadata_source_indices(self):
        reader = _make_collection(['HH', 'VV'])
        assert reader.metadata.channel_metadata[0].source_indices == [0]
        assert reader.metadata.channel_metadata[1].source_indices == [1]

    def test_per_file_metadata_accessible(self):
        reader = _make_collection(['HH', 'VV'])
        assert len(reader.metadata.per_file_metadata) == 2
        assert isinstance(reader.metadata.per_file_metadata[0], SICDMetadata)

    def test_rows_cols_from_sub_readers(self):
        reader = _make_collection(['HH'], rows=256, cols=1024)
        assert reader.metadata.rows == 256
        assert reader.metadata.cols == 1024

    def test_format_string(self):
        reader = _make_collection(['HH', 'VV'])
        assert reader.metadata.format == 'SICD_COLLECTION'

    def test_dtype(self):
        reader = _make_collection(['HH', 'VV'])
        assert reader.metadata.dtype == 'complex64'

    def test_metadata_is_sicd_collection_metadata(self):
        reader = _make_collection(['HH', 'VV'])
        assert isinstance(reader.metadata, SICDCollectionMetadata)

    def test_polarization_override_used_verbatim(self):
        filepaths = ['/fake/hh.nitf', '/fake/vv.nitf']
        with (
            patch(
                'grdl.IO.sar.sicd_collection.SICDReader',
                side_effect=[
                    _make_sicd_reader(512, 512, 'H:H'),
                    _make_sicd_reader(512, 512, 'V:V'),
                ],
            ),
            patch.object(Path, 'exists', return_value=True),
        ):
            reader = SICDCollectionReader(
                filepaths, polarizations=['CUSTOM_HH', 'CUSTOM_VV']
            )
        pols = [cm.polarization for cm in reader.metadata.channel_metadata]
        assert pols == ['CUSTOM_HH', 'CUSTOM_VV']


# ---------------------------------------------------------------------------
# SICDCollectionReader read tests
# ---------------------------------------------------------------------------

class TestSICDCollectionReaderRead:

    def test_read_chip_returns_cyx_shape(self):
        reader = _make_collection(['HH', 'HV', 'VH', 'VV'])
        for mock in reader._readers:
            mock.read_chip.return_value = np.zeros((100, 200), dtype=np.complex64)
        result = reader.read_chip(0, 100, 0, 200)
        assert result.shape == (4, 100, 200)
        assert result.dtype == np.complex64

    def test_read_chip_two_channels(self):
        reader = _make_collection(['HH', 'VV'])
        for mock in reader._readers:
            mock.read_chip.return_value = np.ones((50, 50), dtype=np.complex64)
        result = reader.read_chip(0, 50, 0, 50)
        assert result.shape == (2, 50, 50)

    def test_read_chip_band_selection(self):
        reader = _make_collection(['HH', 'HV', 'VH', 'VV'])
        for mock in reader._readers:
            mock.read_chip.return_value = np.zeros((64, 64), dtype=np.complex64)
        result = reader.read_chip(0, 64, 0, 64, bands=[0, 2])
        assert result.shape == (2, 64, 64)

    def test_read_chip_band_selection_calls_correct_readers(self):
        reader = _make_collection(['HH', 'HV', 'VH', 'VV'])
        for mock in reader._readers:
            mock.read_chip.return_value = np.zeros((64, 64), dtype=np.complex64)
        reader.read_chip(0, 64, 0, 64, bands=[1, 3])
        reader._readers[0].read_chip.assert_not_called()
        reader._readers[1].read_chip.assert_called_once()
        reader._readers[2].read_chip.assert_not_called()
        reader._readers[3].read_chip.assert_called_once()

    def test_read_full_cyx_shape(self):
        rows, cols = 32, 64
        reader = _make_collection(['HH', 'VV'], rows=rows, cols=cols)
        for mock in reader._readers:
            mock.read_chip.return_value = np.zeros((rows, cols), dtype=np.complex64)
        result = reader.read_full()
        assert result.shape == (2, rows, cols)

    def test_read_chip_stacks_values_correctly(self):
        reader = _make_collection(['HH', 'VV'])
        reader._readers[0].read_chip.return_value = np.ones((4, 4), dtype=np.complex64)
        reader._readers[1].read_chip.return_value = np.full((4, 4), 2+0j, dtype=np.complex64)
        result = reader.read_chip(0, 4, 0, 4)
        np.testing.assert_array_equal(result[0], np.ones((4, 4), dtype=np.complex64))
        np.testing.assert_array_equal(result[1], np.full((4, 4), 2+0j, dtype=np.complex64))

    def test_read_band_returns_2d(self):
        reader = _make_collection(['HH', 'VV'])
        reader._readers[0].read_chip.return_value = np.ones((32, 32), dtype=np.complex64)
        reader._readers[1].read_chip.return_value = np.zeros((32, 32), dtype=np.complex64)
        result = reader.read_band(0)
        assert result.ndim == 2
        assert result.shape == (32, 32)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestSICDCollectionReaderErrors:

    def test_empty_filepaths_raises(self):
        with pytest.raises(ValueError, match='at least one path'):
            SICDCollectionReader([])

    def test_polarization_length_mismatch_raises(self):
        with (
            patch(
                'grdl.IO.sar.sicd_collection.SICDReader',
                side_effect=[_make_sicd_reader()],
            ),
            patch.object(Path, 'exists', return_value=True),
        ):
            with pytest.raises(ValueError, match='polarizations length'):
                SICDCollectionReader(
                    ['/fake/hh.nitf', '/fake/vv.nitf'],
                    polarizations=['HH'],   # wrong length
                )

    def test_dimension_mismatch_raises(self):
        readers = [
            _make_sicd_reader(512, 512),
            _make_sicd_reader(256, 512),  # different rows
        ]
        with (
            patch(
                'grdl.IO.sar.sicd_collection.SICDReader',
                side_effect=readers,
            ),
            patch.object(Path, 'exists', return_value=True),
        ):
            with pytest.raises(ValueError, match='dimensions'):
                SICDCollectionReader(['/fake/hh.nitf', '/fake/vv.nitf'])

    def test_invalid_band_index_raises(self):
        reader = _make_collection(['HH', 'VV'])
        with pytest.raises(ValueError, match='Band index'):
            reader.read_chip(0, 10, 0, 10, bands=[5])

    def test_negative_start_raises(self):
        reader = _make_collection(['HH'])
        with pytest.raises(ValueError, match='non-negative'):
            reader.read_chip(-1, 10, 0, 10)

    def test_out_of_bounds_end_raises(self):
        reader = _make_collection(['HH'], rows=512, cols=512)
        with pytest.raises(ValueError, match='exceed image dimensions'):
            reader.read_chip(0, 9999, 0, 512)


# ---------------------------------------------------------------------------
# close() / resource management tests
# ---------------------------------------------------------------------------

class TestSICDCollectionReaderClose:

    def test_close_releases_all_readers(self):
        reader = _make_collection(['HH', 'HV', 'VH', 'VV'])
        reader.close()
        for mock in reader._readers:
            mock.close.assert_called_once()

    def test_context_manager_calls_close(self):
        reader = _make_collection(['HH', 'VV'])
        with reader:
            pass
        for mock in reader._readers:
            mock.close.assert_called_once()


# ---------------------------------------------------------------------------
# Convenience method tests
# ---------------------------------------------------------------------------

class TestSICDCollectionReaderConvenience:

    def test_get_available_polarizations(self):
        reader = _make_collection(['HH', 'HV', 'VH', 'VV'])
        assert reader.get_available_polarizations() == ['HH', 'HV', 'VH', 'VV']

    def test_get_reader_for_returns_correct_sub_reader(self):
        reader = _make_collection(['HH', 'HV', 'VH', 'VV'])
        sub = reader.get_reader_for('VH')
        assert sub is reader._readers[2]

    def test_get_reader_for_case_insensitive(self):
        reader = _make_collection(['HH', 'VV'])
        assert reader.get_reader_for('hh') is reader._readers[0]

    def test_get_reader_for_missing_raises_key_error(self):
        reader = _make_collection(['HH', 'VV'])
        with pytest.raises(KeyError, match='HV'):
            reader.get_reader_for('HV')

    def test_get_shape_returns_rows_cols(self):
        reader = _make_collection(['HH', 'VV'], rows=256, cols=1024)
        assert reader.get_shape() == (256, 1024)

    def test_get_dtype_is_complex64(self):
        reader = _make_collection(['HH', 'VV'])
        assert reader.get_dtype() == np.dtype('complex64')


# ---------------------------------------------------------------------------
# PolarimetricMode integration
# ---------------------------------------------------------------------------

class TestPolarimetricModeIntegration:

    def test_quad_pol_mode(self):
        reader = _make_collection(['HH', 'HV', 'VH', 'VV'])
        mode = PolarimetricMode.from_reader(reader)
        assert mode == PolarimetricMode.QUAD_POL

    def test_dual_pol_mode_hh_hv(self):
        reader = _make_collection(['HH', 'HV'])
        mode = PolarimetricMode.from_reader(reader)
        assert mode == PolarimetricMode.DUAL_POL

    def test_single_pol_mode(self):
        reader = _make_collection(['HH'])
        mode = PolarimetricMode.from_reader(reader)
        assert mode == PolarimetricMode.SINGLE_POL


# ---------------------------------------------------------------------------
# open_sicd_collection factory tests
# ---------------------------------------------------------------------------

class TestOpenSICDCollection:

    def test_returns_sicd_collection_reader(self):
        filepaths = ['/fake/hh.nitf', '/fake/vv.nitf']
        with (
            patch(
                'grdl.IO.sar.sicd_collection.SICDReader',
                side_effect=[
                    _make_sicd_reader(512, 512, 'H:H'),
                    _make_sicd_reader(512, 512, 'V:V'),
                ],
            ),
            patch.object(Path, 'exists', return_value=True),
        ):
            reader = open_sicd_collection(filepaths)
        assert isinstance(reader, SICDCollectionReader)

    def test_empty_filepaths_raises(self):
        with pytest.raises(ValueError, match='at least one path'):
            open_sicd_collection([])

    def test_missing_file_raises(self):
        with (
            patch.object(Path, 'exists', return_value=False),
        ):
            with pytest.raises(FileNotFoundError):
                open_sicd_collection(['/nonexistent/hh.nitf'])

    def test_context_manager_usage(self):
        filepaths = ['/fake/hh.nitf', '/fake/vv.nitf']
        mock_readers = [_make_sicd_reader(), _make_sicd_reader(tx_rcv_polarization='V:V')]
        with (
            patch(
                'grdl.IO.sar.sicd_collection.SICDReader',
                side_effect=mock_readers,
            ),
            patch.object(Path, 'exists', return_value=True),
        ):
            with open_sicd_collection(filepaths) as reader:
                assert reader.metadata.bands == 2
