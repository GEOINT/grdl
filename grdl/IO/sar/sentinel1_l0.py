# -*- coding: utf-8 -*-
"""
Sentinel-1 Level-0 Reader - ESA Sentinel-1 IW Level-0 SAFE products.

Reads Sentinel-1 Level-0 raw data from the standard ESA SAFE directory
structure.  Each reader instance opens one SAFE product directory,
exposing all polarization channels.  ISP packet parsing and FDBAQ
decompression are handled by ``sentinel1decoder``.

Dependencies
------------
sentinel1decoder

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-02

Modified
--------
2026-03-02
"""

# Standard library
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

# Third-party
import numpy as np
import pandas as pd

try:
    import sentinel1decoder
    _HAS_S1DECODER = True
except ImportError:
    _HAS_S1DECODER = False

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models.sentinel1_l0 import (
    Sentinel1L0Metadata,
    S1L0ProductInfo,
    S1L0RadarParams,
    S1L0ChannelInfo,
    S1L0FootprintCoord,
)


# ===================================================================
# Constants
# ===================================================================

# Sentinel-1 C-band radar center frequency (Hz)
S1_CENTER_FREQUENCY = 5.405e9

# Polarization label from filename fragments
_POL_FROM_FILENAME = {
    '-vv-': ('VV', 'V', 'V'),
    '-vh-': ('VH', 'V', 'H'),
    '-hh-': ('HH', 'H', 'H'),
    '-hv-': ('HV', 'H', 'V'),
}

# Range decimation code to sampling rate (Hz)
# From ESA Sentinel-1 SAR Space Packet Protocol Data Unit Table 5.2-3
_RANGE_DECIMATION_TO_FS = {
    0: 3 * 64e6 / 1,
    1: 3 * 64e6 / (2 / 3),
    3: 2 * 64e6 / (3 / 4),
    4: 3 * 64e6 / 2,
    5: 4 * 64e6 / 3,
    6: 2 * 64e6 / (2 / 3),
    7: 64e6,
    8: 3 * 64e6 / 4,
    9: 2 * 64e6,
    10: 3 * 64e6 / (8 / 3),
    11: 3 * 64e6 / 4,
}


def _require_sentinel1decoder() -> None:
    """Raise ImportError if sentinel1decoder is not installed."""
    if not _HAS_S1DECODER:
        raise ImportError(
            "Reading Sentinel-1 Level-0 files requires sentinel1decoder. "
            "Install with: pip install sentinel1decoder"
        )


def _detect_pol_from_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """Detect polarization from measurement file name.

    Returns (label, tx_pol, rx_pol) or None if not detected.
    """
    lower = filename.lower()
    for fragment, pol_info in _POL_FROM_FILENAME.items():
        if fragment in lower:
            return pol_info
    return None


# ===================================================================
# Manifest parser
# ===================================================================

def _parse_manifest(safe_dir: Path) -> Dict[str, Any]:
    """Parse manifest.safe and return product-level metadata.

    Returns dict with keys: mission, mode, start_time, stop_time,
    orbit_pass, absolute_orbit, relative_orbit, footprint,
    measurement_files, annotation_files.
    """
    manifest_path = safe_dir / 'manifest.safe'
    if not manifest_path.exists():
        raise ValueError(f"No manifest.safe found in {safe_dir}")

    tree = ET.parse(str(manifest_path))
    root = tree.getroot()

    info: Dict[str, Any] = {
        'mission': None,
        'mode': None,
        'start_time': None,
        'stop_time': None,
        'orbit_pass': None,
        'absolute_orbit': None,
        'relative_orbit': None,
        'footprint': [],
        'measurement_files': [],
        'annotation_files': [],
    }

    # Extract scalar metadata by tag name (namespace-agnostic)
    mission_family = None
    mission_number = None
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        text = elem.text
        if text is None:
            continue
        if tag == 'familyName' and text == 'SENTINEL-1':
            mission_family = text
        elif tag == 'number' and mission_family:
            mission_number = text
            info['mission'] = f"S1{text}"
        elif tag == 'mode':
            info['mode'] = text
        elif tag == 'startTime':
            info['start_time'] = text
        elif tag == 'stopTime':
            info['stop_time'] = text
        elif tag == 'pass':
            info['orbit_pass'] = text
        elif tag == 'absoluteOrbitNumber' and info['absolute_orbit'] is None:
            info['absolute_orbit'] = int(text)
        elif tag == 'relativeOrbitNumber' and info['relative_orbit'] is None:
            info['relative_orbit'] = int(text)
        elif tag == 'coordinates':
            # Parse footprint: "lat,lon lat,lon ..."
            for pair in text.strip().split():
                parts = pair.split(',')
                if len(parts) == 2:
                    info['footprint'].append(
                        S1L0FootprintCoord(
                            lat=float(parts[0]),
                            lon=float(parts[1]),
                        )
                    )

    # Extract data object file references
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'dataObject':
            rep_id = elem.get('repID', '')
            for child in elem.iter():
                ctag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if ctag == 'fileLocation':
                    href = child.get('href', '')
                    # Remove leading ./
                    fname = href.lstrip('./')
                    if rep_id == 'measurementSchema':
                        info['measurement_files'].append(fname)
                    elif rep_id == 'measurementAnnotationSchema':
                        info['annotation_files'].append(fname)

    return info


# ===================================================================
# Sentinel-1 Level-0 Reader
# ===================================================================

class Sentinel1L0Reader(ImageReader):
    """Read Sentinel-1 Level-0 raw data from SAFE directories.

    Parses the SAFE manifest, decodes ISP packet headers, and provides
    FDBAQ decompression to recover complex I/Q samples.  Uses
    ``sentinel1decoder`` as the decoding backend.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.SAFE`` directory.

    Attributes
    ----------
    filepath : Path
        Path to the SAFE directory.
    metadata : Sentinel1L0Metadata
        Typed metadata including product info, radar params, and channels.
    channel_metadata : dict
        Per-channel ``pandas.DataFrame`` of decoded ISP metadata.
    channel_decoders : dict
        Per-channel ``sentinel1decoder.Level0Decoder`` instances.

    Raises
    ------
    ImportError
        If sentinel1decoder is not installed.
    FileNotFoundError
        If the SAFE directory or required files do not exist.
    ValueError
        If the directory is not a valid Sentinel-1 Level-0 product.

    Examples
    --------
    >>> from grdl.IO.sar import Sentinel1L0Reader
    >>> with Sentinel1L0Reader('S1C_IW_RAW__0SDV_....SAFE') as reader:
    ...     shape = reader.get_shape()
    ...     print(reader.metadata.channels.keys())
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        _require_sentinel1decoder()
        self._channel_decoders: Dict[str, sentinel1decoder.Level0Decoder] = {}
        self._channel_metadata: Dict[str, pd.DataFrame] = {}
        self._channel_echo_metadata: Dict[str, pd.DataFrame] = {}
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Parse SAFE directory and build metadata."""
        safe_dir = self.filepath
        if not safe_dir.is_dir():
            raise ValueError(
                f"Expected a SAFE directory, got: {safe_dir}"
            )

        # Parse manifest
        manifest = _parse_manifest(safe_dir)

        # Determine polarization mode from SAFE directory name
        safe_name = safe_dir.name.upper()
        pol_mode = None
        for code in ('0SDV', '0SDH', '0SSV', '0SSH'):
            if code in safe_name:
                pol_mode = code[-2:]  # DV, DH, SV, SH
                break

        product_info = S1L0ProductInfo(
            mission=manifest['mission'],
            mode=manifest['mode'],
            product_type='RAW',
            polarization_mode=pol_mode,
            start_time=manifest['start_time'],
            stop_time=manifest['stop_time'],
            absolute_orbit=manifest['absolute_orbit'],
            relative_orbit=manifest['relative_orbit'],
            orbit_pass=manifest['orbit_pass'],
        )

        # Build per-channel info from measurement files
        channels: Dict[str, S1L0ChannelInfo] = {}
        data_take_id = None

        for meas_file in sorted(manifest['measurement_files']):
            pol_info = _detect_pol_from_filename(meas_file)
            if pol_info is None:
                continue
            pol_label, tx_pol, rx_pol = pol_info

            meas_path = safe_dir / meas_file
            if not meas_path.exists():
                continue

            # Find matching annotation file
            annot_file = None
            for af in manifest['annotation_files']:
                if _detect_pol_from_filename(af) == pol_info:
                    annot_path = safe_dir / af
                    if annot_path.exists():
                        annot_file = str(annot_path)
                    break

            # Create decoder and parse metadata
            decoder = sentinel1decoder.Level0Decoder(str(meas_path))
            meta_df = decoder.decode_metadata()

            # Filter echo packets
            echo_mask = (
                meta_df['Signal Type'] == sentinel1decoder.SignalType.ECHO
            )
            echo_df = meta_df[echo_mask].copy()

            if len(echo_df) == 0:
                continue

            # Extract data take ID from first echo packet
            if data_take_id is None:
                data_take_id = int(echo_df['Data Take ID'].iloc[0])

            # Get swath numbers and quad counts
            swath_numbers = sorted(echo_df['Swath Number'].unique().tolist())
            max_quads = int(echo_df['Number of Quads'].max())

            channels[pol_label] = S1L0ChannelInfo(
                polarization=pol_label,
                tx_pol=tx_pol,
                rx_pol=rx_pol,
                measurement_file=str(meas_path),
                annotation_file=annot_file,
                num_echo_packets=len(echo_df),
                max_num_quads=max_quads,
                swath_numbers=swath_numbers,
            )

            self._channel_decoders[pol_label] = decoder
            self._channel_metadata[pol_label] = meta_df
            self._channel_echo_metadata[pol_label] = echo_df

        if not channels:
            raise ValueError(
                f"No valid polarization channels found in {safe_dir}"
            )

        # Derive radar parameters from first echo packet of first channel
        first_ch = next(iter(channels.values()))
        first_echo = self._channel_echo_metadata[first_ch.polarization]
        first_pkt = first_echo.iloc[0]

        # Sampling rate from range decimation
        range_dec_enum = first_pkt['Range Decimation']
        range_dec_val = range_dec_enum.value if hasattr(range_dec_enum, 'value') else int(range_dec_enum)
        sampling_rate = _RANGE_DECIMATION_TO_FS.get(range_dec_val, 64e6)

        # TX parameters are already converted by sentinel1decoder
        tx_ramp_rate = float(first_pkt['Tx Ramp Rate'])
        tx_pulse_length = float(first_pkt['Tx Pulse Length'])
        tx_bandwidth = abs(tx_ramp_rate * tx_pulse_length)
        chirp_rate = tx_ramp_rate
        pri = float(first_pkt['PRI'])

        radar_params = S1L0RadarParams(
            center_frequency=S1_CENTER_FREQUENCY,
            tx_bandwidth=tx_bandwidth,
            chirp_rate=chirp_rate,
            sampling_rate=sampling_rate,
            tx_pulse_length=tx_pulse_length,
            pri=pri,
        )

        # Build top-level metadata
        first_ch_info = next(iter(channels.values()))
        num_samples = 2 * first_ch_info.max_num_quads

        self.metadata = Sentinel1L0Metadata(
            format='S1_L0',
            rows=first_ch_info.num_echo_packets,
            cols=num_samples,
            dtype='complex64',
            extras={
                'num_channels': len(channels),
                'backend': 'sentinel1decoder',
            },
            product_info=product_info,
            radar_params=radar_params,
            channels=channels,
            footprint=manifest['footprint'],
            data_take_id=data_take_id,
        )

    def get_channel_echo_metadata(
        self, polarization: str,
    ) -> pd.DataFrame:
        """Get parsed ISP metadata for echo packets in a channel.

        Parameters
        ----------
        polarization : str
            Channel label (``'VV'``, ``'VH'``, ``'HH'``, ``'HV'``).

        Returns
        -------
        pd.DataFrame
            sentinel1decoder metadata for echo packets only.
        """
        if polarization not in self._channel_echo_metadata:
            raise ValueError(
                f"Channel '{polarization}' not found. "
                f"Available: {list(self._channel_echo_metadata.keys())}"
            )
        return self._channel_echo_metadata[polarization]

    def get_channel_all_metadata(
        self, polarization: str,
    ) -> pd.DataFrame:
        """Get parsed ISP metadata for all packets in a channel.

        Parameters
        ----------
        polarization : str
            Channel label.

        Returns
        -------
        pd.DataFrame
            sentinel1decoder metadata for all packets.
        """
        if polarization not in self._channel_metadata:
            raise ValueError(
                f"Channel '{polarization}' not found. "
                f"Available: {list(self._channel_metadata.keys())}"
            )
        return self._channel_metadata[polarization]

    def decode_channel(
        self,
        polarization: str,
        max_packets: Optional[int] = None,
    ) -> np.ndarray:
        """Decompress all echo packets for a channel.

        Parameters
        ----------
        polarization : str
            Channel label (``'VV'``, ``'VH'``, ``'HH'``, ``'HV'``).
        max_packets : int, optional
            If set, only decode the first N echo packets.

        Returns
        -------
        np.ndarray
            Complex signal data, shape ``(num_vectors, num_samples)``,
            dtype ``complex64``. Zero-padded to uniform sample count.
        """
        echo_df = self.get_channel_echo_metadata(polarization)
        decoder = self._channel_decoders[polarization]
        ch_info = self.metadata.channels[polarization]

        if max_packets is not None:
            echo_df = echo_df.head(max_packets)

        num_vectors = len(echo_df)
        num_samples_max = 2 * ch_info.max_num_quads

        # IW mode has multiple swaths with different num_quads.
        # sentinel1decoder requires uniform num_quads per batch,
        # so we decode per-group and reassemble.
        unique_nq = echo_df["Number of Quads"].unique()

        if len(unique_nq) == 1:
            raw = decoder.decode_packets(echo_df)
            if raw.shape[1] < num_samples_max:
                padded = np.zeros(
                    (num_vectors, num_samples_max), dtype=np.complex64,
                )
                padded[:, :raw.shape[1]] = raw
                return padded
            return raw.astype(np.complex64)

        # Multiple num_quads: decode per group, assemble
        result = np.zeros(
            (num_vectors, num_samples_max), dtype=np.complex64,
        )
        for nq in unique_nq:
            mask = echo_df["Number of Quads"] == nq
            group_df = echo_df[mask]
            group_indices = np.where(mask.values)[0]
            decoded = decoder.decode_packets(group_df)
            nsamp = decoded.shape[1]
            result[group_indices, :nsamp] = decoded.astype(np.complex64)

        return result

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial subset of signal data.

        Parameters
        ----------
        row_start, row_end : int
            Vector (pulse) range.
        col_start, col_end : int
            Sample range.
        bands : list of int, optional
            Channel indices. If None, reads the first channel.

        Returns
        -------
        np.ndarray
            Signal data chip, dtype complex64.
        """
        ch_idx = bands[0] if bands else 0
        pol = list(self.metadata.channels.keys())[ch_idx]
        full = self.decode_channel(pol)
        return full[row_start:row_end, col_start:col_end]

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read full signal data for a channel.

        Parameters
        ----------
        bands : list of int, optional
            Channel indices. If None, reads the first channel.

        Returns
        -------
        np.ndarray
            Signal data, dtype complex64.
        """
        ch_idx = bands[0] if bands else 0
        pol = list(self.metadata.channels.keys())[ch_idx]
        return self.decode_channel(pol)

    def get_shape(self) -> Tuple[int, int]:
        """Get signal dimensions for the first channel.

        Returns
        -------
        Tuple[int, int]
            ``(num_echo_packets, num_samples)`` for the first channel.
        """
        first_ch = next(iter(self.metadata.channels.values()))
        return (first_ch.num_echo_packets, 2 * first_ch.max_num_quads)

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            ``complex64``.
        """
        return np.dtype('complex64')

    def close(self) -> None:
        """Release resources."""
        self._channel_decoders.clear()
        self._channel_metadata.clear()
        self._channel_echo_metadata.clear()
