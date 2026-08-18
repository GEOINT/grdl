# -*- coding: utf-8 -*-
"""Compact-pol synthesis from quad-pol SAR data.

Reads any quad-pol product supported by GRDL (NISAR, BIOMASS, 4-channel SICD,
TerraSAR-X, etc.) and writes:

- Two SICD NITF files (``{stem}_rh.nitf``, ``{stem}_rv.nitf``) for the
  synthesized RH and RV receive channels, readable by :class:`SICDCollectionReader`.
- One HDF5 covariance matrix file (``{stem}_c2.h5``) in PolSARpro C2 format,
  readable by :class:`CompactPolC2H5Reader` and all GRDL compact-pol decomposers.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np

from grdl.IO.generic import open_any
from grdl.IO.sar.nisar import NISARReader
from grdl.IO.sar.sicd_writer import SICDWriter
from grdl.IO.models.common import RowCol
from grdl.IO.models.sicd import (
    SICDCollectionInfo,
    SICDCollectionMetadata,
    SICDImageData,
    SICDMetadata,
    SICDRadarCollection,
    SICDRcvChannel,
    SICDTimeline,
)

logger = logging.getLogger(__name__)
# Sarpy logs a spurious error when CollectionInfo lacks full geometry for NGA naming.
_sarpy_naming_logger = logging.getLogger('sarpy.io.complex.naming.utils')

# SICD polarization strings per NGA.STND.0024-2_1.0.0
_TX_POL = {'R': 'RHC', 'L': 'LHC'}
_TX_RCV_POL = {
    'R': {'H': 'RHC:H', 'V': 'RHC:V'},
    'L': {'H': 'LHC:H', 'V': 'LHC:V'},
}


# ---------------------------------------------------------------------------
# Source reading
# ---------------------------------------------------------------------------

def _is_nisar_h5(path: Path) -> bool:
    """Return True if path is a NISAR HDF5 product."""
    if path.suffix.lower() not in {'.h5', '.hdf5'}:
        return False
    try:
        with h5py.File(path, 'r') as f:
            science = f.get('science')
            return science is not None and ('LSAR' in science or 'SSAR' in science)
    except Exception:
        return False


def _read_source_cube(
    input_path: Path,
    frequency: Optional[str],
) -> Tuple[np.ndarray, object]:
    """Read a quad-pol data cube using the most specific available reader.

    Supports any quad-pol source readable by GRDL: NISAR, BIOMASS,
    4-channel SICD, Sentinel-1, TerraSAR-X, or GeoTIFF.  The returned cube
    must contain at least four co-registered complex channels (HH/HV/VH/VV).
    """
    reader = None

    if _is_nisar_h5(input_path):
        try:
            reader = NISARReader(input_path, frequency=frequency, polarizations='all')
        except Exception as exc:
            logger.warning('NISARReader failed (%s); falling back to open_any()', exc)

    if reader is None:
        reader = open_any(input_path)

    with reader:
        cube = np.asarray(reader.read_full())
        metadata = reader.metadata
    return cube, metadata


# ---------------------------------------------------------------------------
# Channel mapping and extraction
# ---------------------------------------------------------------------------

def _channel_map_from_metadata(metadata: object) -> Dict[str, int]:
    """Map HH/HV/VH/VV polarizations to channel indices from reader metadata."""
    channel_metadata = getattr(metadata, 'channel_metadata', None)
    if not channel_metadata:
        return {}
    mapping: Dict[str, int] = {}
    for i, ch in enumerate(channel_metadata):
        pol = getattr(ch, 'polarization', None)
        name = getattr(ch, 'name', None)
        token = (pol or name or '').upper().replace('_', '')
        for key in ('HH', 'HV', 'VH', 'VV'):
            if key in token and key not in mapping:
                mapping[key] = i
                break
    return mapping


_FALLBACK_MAP = {'HH': 0, 'HV': 1, 'VH': 2, 'VV': 3}


def _extract_quad_pol(
    cube: np.ndarray,
    metadata: object,
    hh_band: Optional[int],
    hv_band: Optional[int],
    vh_band: Optional[int],
    vv_band: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    if cube.ndim != 3:
        raise ValueError(f'Expected 3D image cube, got shape {cube.shape}')

    axis_order = getattr(metadata, 'axis_order', None)
    bands = int(getattr(metadata, 'bands', cube.shape[0]))
    if axis_order == 'CYX':
        byx = cube
    elif axis_order == 'YXC':
        byx = np.moveaxis(cube, -1, 0)
    else:
        byx = cube if cube.shape[0] == bands else np.moveaxis(cube, -1, 0)

    if byx.shape[0] < 4:
        raise ValueError(
            f'Quad-pol synthesis requires at least 4 channels (HH/HV/VH/VV), '
            f'got {byx.shape[0]}.  Ensure the input is a quad-pol product.'
        )

    mapping = _channel_map_from_metadata(metadata)
    for k, v in {'HH': hh_band, 'HV': hv_band, 'VH': vh_band, 'VV': vv_band}.items():
        if v is not None:
            idx = int(v) - 1
            if not (0 <= idx < byx.shape[0]):
                raise ValueError(f'{k} band override out of range: {v}')
            mapping[k] = idx
    for k, default in _FALLBACK_MAP.items():
        mapping.setdefault(k, default)

    if len({mapping[k] for k in _FALLBACK_MAP}) != 4:
        raise ValueError(f'Duplicate band indices in mapping: {mapping}')

    return (
        byx[mapping['HH']],
        byx[mapping['HV']],
        byx[mapping['VH']],
        byx[mapping['VV']],
        {k: mapping[k] for k in ('HH', 'HV', 'VH', 'VV')},
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize_compact_pol(
    s_hh: np.ndarray,
    s_hv: np.ndarray,
    s_vh: np.ndarray,
    s_vv: np.ndarray,
    transmit: str = 'R',
) -> Dict[str, np.ndarray]:
    """Synthesize compact-pol channels and C2 covariance from quad-pol S-matrix.

    Input must be quad-pol (HH/HV/VH/VV); all channels must share the same shape.

    Parameters
    ----------
    s_hh, s_hv, s_vh, s_vv : np.ndarray
        Complex quad-pol channels.
    transmit : {'R', 'L'}
        Circular transmit handedness.

    Returns
    -------
    dict with keys ``'RH'``, ``'RV'`` (complex64) and
    ``'C11'``, ``'C12_real'``, ``'C12_imag'``, ``'C22'`` (float32).
    """
    tx = transmit.upper()
    if tx not in {'R', 'L'}:
        raise ValueError(f"transmit must be 'R' or 'L', got {transmit!r}")
    for arr in (s_hv, s_vh, s_vv):
        if arr.shape != s_hh.shape:
            raise ValueError('All quad-pol channels must have the same shape')

    inv_sqrt2 = np.float32(1.0 / np.sqrt(2.0))
    if tx == 'R':
        s_rh = inv_sqrt2 * (s_hh - 1j * s_hv)
        s_rv = inv_sqrt2 * (s_vh - 1j * s_vv)
    else:
        s_rh = inv_sqrt2 * (s_hh + 1j * s_hv)
        s_rv = inv_sqrt2 * (s_vh + 1j * s_vv)

    c12 = s_rh * np.conj(s_rv)
    return {
        'RH': np.asarray(s_rh, dtype=np.complex64),
        'RV': np.asarray(s_rv, dtype=np.complex64),
        'C11': (np.abs(s_rh) ** 2).astype(np.float32),
        'C12_real': np.real(c12).astype(np.float32),
        'C12_imag': np.imag(c12).astype(np.float32),
        'C22': (np.abs(s_rv) ** 2).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# SICD NITF output
# ---------------------------------------------------------------------------

def _collect_start_from_source(source_meta: object) -> str:
    """Extract an ISO 8601 collect-start string from source metadata."""
    for attr in ('collect_start', 'zero_doppler_start_time', 'start_time', 'scene_start_time'):
        val = getattr(source_meta, attr, None)
        if val:
            return str(val)
    extras = getattr(source_meta, 'extras', {}) or {}
    for key in ('collect_start', 'start_time', 'zero_doppler_start_time'):
        if key in extras:
            return str(extras[key])
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def _stub_collection_info(source_meta: object) -> SICDCollectionInfo:
    """Build a minimal SICDCollectionInfo so sarpy can construct an NITF file title."""
    # Try to pull a sensor/platform name from the source metadata
    sensor = None
    for attr in ('sensor_name', 'platform', 'spacecraft_name', 'mission_id'):
        sensor = getattr(source_meta, attr, None)
        if sensor:
            break
    source_format = str(getattr(source_meta, 'format', '') or '')
    collector_name = sensor or (source_format.upper() if source_format else 'UNKNOWN')
    return SICDCollectionInfo(
        collector_name=collector_name,
        core_name='COMPACT_POL',
        collect_type='MONOSTATIC',
    )


def _build_compact_pol_sicd_metadata(
    source_meta: object,
    rcv_label: str,
    rows: int,
    cols: int,
    transmit: str,
) -> SICDMetadata:
    """Derive SICDMetadata for one synthesized receive channel.

    Inherits geometry from the SICD source when available; always overrides
    polarization to reflect the circular transmit convention.
    """
    tx = transmit.upper()
    radar_collection = SICDRadarCollection(
        tx_polarization=_TX_POL[tx],
        rcv_channels=[
            SICDRcvChannel(tx_rcv_polarization=_TX_RCV_POL[tx][rcv_label], index=0)
        ],
    )
    image_data = SICDImageData(
        pixel_type='RE32F_IM32F',
        num_rows=rows,
        num_cols=cols,
        scp_pixel=RowCol(row=rows // 2, col=cols // 2),
    )

    sicd_base: Optional[SICDMetadata] = None
    if isinstance(source_meta, SICDMetadata):
        sicd_base = source_meta
    elif (
        isinstance(source_meta, SICDCollectionMetadata)
        and source_meta.per_file_metadata
    ):
        sicd_base = source_meta.per_file_metadata[0]

    if sicd_base is not None:
        # Ensure Timeline and CollectionInfo are present; sarpy requires both for NITF naming.
        base_timeline = sicd_base.timeline or SICDTimeline(
            collect_start=_collect_start_from_source(source_meta)
        )
        base_cinfo = sicd_base.collection_info or _stub_collection_info(source_meta)
        return dataclasses.replace(
            sicd_base,
            format='SICD',
            rows=rows,
            cols=cols,
            dtype='complex64',
            bands=1,
            axis_order='YX',
            channel_metadata=None,
            image_data=image_data,
            radar_collection=radar_collection,
            timeline=base_timeline,
            collection_info=base_cinfo,
        )

    return SICDMetadata(
        format='SICD',
        rows=rows,
        cols=cols,
        dtype='complex64',
        bands=1,
        axis_order='YX',
        image_data=image_data,
        radar_collection=radar_collection,
        timeline=SICDTimeline(collect_start=_collect_start_from_source(source_meta)),
        collection_info=_stub_collection_info(source_meta),
    )


def write_compact_pol_sicd(
    rh: np.ndarray,
    rv: np.ndarray,
    output_stem: Union[str, Path],
    source_meta: object,
    transmit: str = 'R',
) -> Tuple[Path, Path]:
    """Write synthesized RH and RV channels as a SICD NITF pair.

    The pair is directly readable via::

        SICDCollectionReader([rh_path, rv_path], polarizations=['RH', 'RV'])

    Parameters
    ----------
    rh, rv : np.ndarray
        Complex64 2-D arrays, shape ``(rows, cols)``.
    output_stem : str or Path
        Path stem; files are written as ``{stem}_rh.nitf`` and ``{stem}_rv.nitf``.
    source_meta : ImageMetadata
        Source reader metadata; SICD geometry is inherited when available.
    transmit : {'R', 'L'}

    Returns
    -------
    (rh_path, rv_path)
    """
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = rh.shape

    _prev_level = _sarpy_naming_logger.level
    _sarpy_naming_logger.setLevel(logging.CRITICAL)
    try:
        for arr, label in ((rh, 'H'), (rv, 'V')):
            path = stem.parent / f'{stem.name}_r{label.lower()}.nitf'
            meta = _build_compact_pol_sicd_metadata(source_meta, label, rows, cols, transmit)
            SICDWriter(path, metadata=meta).write(arr)
            logger.info('Wrote %s', path)
    finally:
        _sarpy_naming_logger.setLevel(_prev_level)

    return (
        stem.parent / f'{stem.name}_rh.nitf',
        stem.parent / f'{stem.name}_rv.nitf',
    )


# ---------------------------------------------------------------------------
# PolSARpro-style C2 HDF5 output
# ---------------------------------------------------------------------------

def write_compact_pol_c2_h5(
    output_path: Union[str, Path],
    c11: np.ndarray,
    c12_real: np.ndarray,
    c12_imag: np.ndarray,
    c22: np.ndarray,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    """Write a PolSARpro-style compact-pol C2 covariance matrix HDF5 file.

    Layout mirrors the PolSARpro community naming convention:
    ``/C2/{C11, C12_real, C12_imag, C22}`` with PolSARpro config
    attributes (``Nrow``, ``Ncol``, ``PolarCase``, ``PolarType``).

    Parameters
    ----------
    output_path : str or Path
    c11, c12_real, c12_imag, c22 : np.ndarray
        Float32 covariance terms, shape ``(rows, cols)``.
    metadata : dict, optional
        Extra key/value pairs stored as string attributes under ``/metadata``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = c11.shape
    opts = {'compression': 'gzip', 'compression_opts': 4}

    with h5py.File(output_path, 'w') as f:
        grp = f.create_group('C2')
        grp.attrs['Nrow'] = rows
        grp.attrs['Ncol'] = cols
        grp.attrs['PolarCase'] = 'compact'
        grp.attrs['PolarType'] = 'C2'
        grp.attrs['grdl_schema_version'] = '1.0'

        grp.create_dataset('C11',      data=np.asarray(c11,      np.float32), **opts)
        grp.create_dataset('C12_real', data=np.asarray(c12_real, np.float32), **opts)
        grp.create_dataset('C12_imag', data=np.asarray(c12_imag, np.float32), **opts)
        grp.create_dataset('C22',      data=np.asarray(c22,      np.float32), **opts)

        if metadata:
            mg = f.create_group('metadata')
            for k, v in metadata.items():
                try:
                    mg.attrs[str(k)] = str(v)
                except Exception:
                    pass

    return output_path


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def convert_quad_pol_to_compact(
    input_path: Union[str, Path],
    output_stem: Union[str, Path],
    *,
    frequency: Optional[str] = None,
    transmit: str = 'R',
    hh_band: Optional[int] = None,
    hv_band: Optional[int] = None,
    vh_band: Optional[int] = None,
    vv_band: Optional[int] = None,
) -> Tuple[Path, Path, Path]:
    """Convert quad-pol SAR data to compact-pol SICD NITF pair and C2 HDF5.

    Reads any quad-pol product supported by GRDL and writes three output files.
    The SICD pair can be opened with::

        SICDCollectionReader([rh_path, rv_path], polarizations=['RH', 'RV'])

    The C2 file can be opened with ``open_sar(c2_path)`` or
    ``CompactPolC2H5Reader(c2_path)``; its 4-band CYX cube plugs directly into
    all GRDL compact-pol decomposers (``CompactPolMChi``, ``CompactPolMDelta``, etc.).

    Parameters
    ----------
    input_path : str or Path
        Quad-pol input file or directory.
    output_stem : str or Path
        Output path stem (no extension or channel suffix).
    frequency : str, optional
        Frequency band selector for NISAR (``'A'`` or ``'B'``).
    transmit : {'R', 'L'}
        Circular transmit handedness.
    hh_band, hv_band, vh_band, vv_band : int, optional
        1-based channel index overrides for source channel mapping.

    Returns
    -------
    (rh_path, rv_path, c2_path)
        Paths to the three output files.
    """
    input_path = Path(input_path)
    output_stem = Path(output_stem)

    cube, meta = _read_source_cube(input_path, frequency)
    s_hh, s_hv, s_vh, s_vv, mapping = _extract_quad_pol(
        cube, meta,
        hh_band=hh_band, hv_band=hv_band, vh_band=vh_band, vv_band=vv_band,
    )
    cpol = synthesize_compact_pol(s_hh, s_hv, s_vh, s_vv, transmit=transmit)

    rh_path, rv_path = write_compact_pol_sicd(
        cpol['RH'], cpol['RV'], output_stem, meta, transmit=transmit,
    )

    c2_path = output_stem.parent / f'{output_stem.name}_c2.h5'
    write_compact_pol_c2_h5(
        c2_path,
        cpol['C11'], cpol['C12_real'], cpol['C12_imag'], cpol['C22'],
        metadata={
            'source_path': str(input_path),
            'transmit': transmit.upper(),
            'source_format': str(getattr(meta, 'format', '')),
            'band_map': str(mapping),
        },
    )

    return rh_path, rv_path, c2_path
