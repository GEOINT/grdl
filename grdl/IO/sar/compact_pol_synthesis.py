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

import argparse
import dataclasses
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from grdl.IO.generic import open_any
from grdl.IO.sar.nisar import NISARReader
from grdl.IO.sar.sicd_writer import SICDWriter
from grdl.IO.models.common import (
    LatLon,
    LatLonHAE,
    Poly1D,
    Poly2D,
    RowCol,
    XYZ,
    XYZPoly,
)
from grdl.IO.models.nisar import NISARMetadata
from grdl.IO.models.sicd import (
    SICDCollectionInfo,
    SICDCollectionMetadata,
    SICDDirParam,
    SICDFullImage,
    SICDGeoData,
    SICDGrid,
    SICDImageData,
    SICDMetadata,
    SICDPosition,
    SICDSCP,
    SICDSCPCOA,
    SICDRadarCollection,
    SICDRcvChannel,
    SICDTimeline,
    SICDTxFrequency,
)
from grdl.geolocation.coordinates import geodetic_to_ecef

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
    identification = getattr(source_meta, 'identification', None)
    for attr in ('zero_doppler_start_time', 'start_time'):
        val = getattr(identification, attr, None)
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


def _parse_iso8601(value: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp, accepting trailing ``Z``."""
    token = value.strip()
    if not token:
        return None
    if token.endswith('Z'):
        token = f'{token[:-1]}+00:00'
    try:
        dt = datetime.fromisoformat(token)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _parse_time_reference(reference: Optional[str]) -> Optional[datetime]:
    """Parse a ``seconds since ...`` reference-epoch string."""
    if not reference:
        return None
    token = reference.strip()
    if token.lower().startswith('seconds since '):
        token = token[14:].strip()
    if 'T' not in token and ' ' in token:
        parts = token.split()
        if len(parts) >= 2 and ':' in parts[1]:
            token = f'{parts[0]}T{parts[1]}'
    return _parse_iso8601(token)


def _seconds_from_reference(
    timestamp: Optional[str],
    reference_epoch: Optional[str],
) -> Optional[float]:
    """Return seconds between an ISO timestamp and a reference epoch."""
    dt = _parse_iso8601(timestamp) if timestamp else None
    ref = _parse_time_reference(reference_epoch)
    if dt is None or ref is None:
        return None
    return float((dt - ref).total_seconds())


def _collect_duration_from_source(source_meta: object) -> Optional[float]:
    """Extract collection duration in seconds when available."""
    identification = getattr(source_meta, 'identification', None)
    start = getattr(identification, 'zero_doppler_start_time', None)
    end = getattr(identification, 'zero_doppler_end_time', None)
    start_dt = _parse_iso8601(start) if isinstance(start, str) else None
    end_dt = _parse_iso8601(end) if isinstance(end, str) else None
    if start_dt is not None and end_dt is not None:
        return float((end_dt - start_dt).total_seconds())

    geo = getattr(source_meta, 'geolocation_grid', None)
    times = getattr(geo, 'zero_doppler_time', None)
    if times is not None and np.size(times) > 1:
        arr = np.asarray(times, dtype=np.float64)
        return float(arr[-1] - arr[0])
    return None


def _time_reference_from_source(source_meta: object) -> Optional[str]:
    """Choose the best available time-reference string from source metadata."""
    swath = getattr(source_meta, 'swath_parameters', None)
    orbit = getattr(source_meta, 'orbit', None)
    for ref in (
        getattr(swath, 'zero_doppler_time_reference_epoch', None),
        getattr(orbit, 'reference_epoch', None),
    ):
        if ref:
            return str(ref)
    return None


def _select_nisar_height_index(heights: Optional[np.ndarray]) -> int:
    """Prefer the NISAR height layer closest to 0 m HAE."""
    if heights is None or np.size(heights) == 0:
        return 0
    arr = np.asarray(heights, dtype=np.float64).ravel()
    valid = np.isfinite(arr)
    if not np.any(valid):
        return int(arr.size // 2)
    valid_idx = np.flatnonzero(valid)
    return int(valid_idx[np.argmin(np.abs(arr[valid_idx]))])


def _build_nisar_row_axis(
    source_meta: NISARMetadata,
    rows: int,
) -> Optional[np.ndarray]:
    """Map geolocation-grid azimuth samples to image row coordinates."""
    geo = source_meta.geolocation_grid
    if geo is None or geo.zero_doppler_time is None:
        return None
    grid_time = np.asarray(geo.zero_doppler_time, dtype=np.float64)
    if grid_time.size == 0:
        return None

    swath = source_meta.swath_parameters
    img_time = getattr(swath, 'zero_doppler_time', None) if swath is not None else None
    if img_time is not None and np.size(img_time) > 1:
        img_arr = np.asarray(img_time, dtype=np.float64)
        return np.interp(
            grid_time, img_arr, np.arange(img_arr.size, dtype=np.float64),
        )

    ident = source_meta.identification
    ref = _time_reference_from_source(source_meta)
    start = _seconds_from_reference(
        getattr(ident, 'zero_doppler_start_time', None) if ident is not None else None,
        ref,
    )
    end = _seconds_from_reference(
        getattr(ident, 'zero_doppler_end_time', None) if ident is not None else None,
        ref,
    )
    if start is not None and end is not None and end != start and rows > 1:
        return (grid_time - start) * ((rows - 1) / (end - start))

    return np.linspace(0.0, max(rows - 1, 0), grid_time.size, dtype=np.float64)


def _build_nisar_col_axis(
    source_meta: NISARMetadata,
    cols: int,
) -> Optional[np.ndarray]:
    """Map geolocation-grid range samples to image column coordinates."""
    geo = source_meta.geolocation_grid
    if geo is None or geo.slant_range is None:
        return None
    grid_range = np.asarray(geo.slant_range, dtype=np.float64)
    if grid_range.size == 0:
        return None

    swath = source_meta.swath_parameters
    img_range = getattr(swath, 'slant_range', None) if swath is not None else None
    if img_range is not None and np.size(img_range) > 1:
        img_arr = np.asarray(img_range, dtype=np.float64)
        return np.interp(
            grid_range, img_arr, np.arange(img_arr.size, dtype=np.float64),
        )

    return np.linspace(0.0, max(cols - 1, 0), grid_range.size, dtype=np.float64)


def _build_nisar_geometry_context(
    source_meta: NISARMetadata,
    rows: int,
    cols: int,
) -> Optional[Dict[str, Any]]:
    """Build interpolation context from a NISAR RSLC geolocation grid."""
    geo = source_meta.geolocation_grid
    if (
        geo is None
        or geo.coordinate_x is None
        or geo.coordinate_y is None
        or geo.zero_doppler_time is None
        or geo.slant_range is None
    ):
        return None

    coord_x = np.asarray(geo.coordinate_x, dtype=np.float64)
    coord_y = np.asarray(geo.coordinate_y, dtype=np.float64)
    incidence = getattr(geo, 'incidence_angle', None)
    elevation = getattr(geo, 'elevation_angle', None)
    height_idx = _select_nisar_height_index(
        getattr(geo, 'height_above_ellipsoid', None),
    )

    if coord_x.ndim == 3:
        lon_grid = coord_x[height_idx]
        lat_grid = coord_y[height_idx]
        inc_grid = (
            np.asarray(incidence[height_idx], dtype=np.float64)
            if incidence is not None and np.asarray(incidence).ndim == 3
            else None
        )
        ele_grid = (
            np.asarray(elevation[height_idx], dtype=np.float64)
            if elevation is not None and np.asarray(elevation).ndim == 3
            else None
        )
    elif coord_x.ndim == 2:
        lon_grid = coord_x
        lat_grid = coord_y
        inc_grid = np.asarray(incidence, dtype=np.float64) if incidence is not None else None
        ele_grid = np.asarray(elevation, dtype=np.float64) if elevation is not None else None
    else:
        return None

    heights = getattr(geo, 'height_above_ellipsoid', None)
    if heights is not None and np.size(heights) > 0:
        height = float(np.asarray(heights, dtype=np.float64).ravel()[height_idx])
    else:
        height = 0.0

    row_axis = _build_nisar_row_axis(source_meta, rows)
    col_axis = _build_nisar_col_axis(source_meta, cols)
    if row_axis is None or col_axis is None:
        return None

    return {
        'lat_grid': lat_grid,
        'lon_grid': lon_grid,
        'inc_grid': inc_grid,
        'ele_grid': ele_grid,
        'height': height,
        'row_axis': np.asarray(row_axis, dtype=np.float64),
        'col_axis': np.asarray(col_axis, dtype=np.float64),
        'time_axis': np.asarray(geo.zero_doppler_time, dtype=np.float64),
        'range_axis': np.asarray(geo.slant_range, dtype=np.float64),
    }


def _bilinear_interpolate(
    grid: np.ndarray,
    row_axis: np.ndarray,
    col_axis: np.ndarray,
    row: float,
    col: float,
) -> float:
    """Bilinearly interpolate a 2-D grid on arbitrary row/col axes."""
    if grid.ndim != 2 or row_axis.size == 0 or col_axis.size == 0:
        raise ValueError('Expected a 2-D grid with non-empty axes')
    if row_axis.size == 1 and col_axis.size == 1:
        return float(grid[0, 0])

    row = float(np.clip(row, row_axis[0], row_axis[-1]))
    col = float(np.clip(col, col_axis[0], col_axis[-1]))

    r_hi = int(np.searchsorted(row_axis, row, side='right'))
    c_hi = int(np.searchsorted(col_axis, col, side='right'))
    r_hi = min(max(r_hi, 1), row_axis.size - 1)
    c_hi = min(max(c_hi, 1), col_axis.size - 1)
    r_lo = r_hi - 1
    c_lo = c_hi - 1

    r0 = float(row_axis[r_lo])
    r1 = float(row_axis[r_hi])
    c0 = float(col_axis[c_lo])
    c1 = float(col_axis[c_hi])
    fr = 0.0 if r1 == r0 else (row - r0) / (r1 - r0)
    fc = 0.0 if c1 == c0 else (col - c0) / (c1 - c0)

    g00 = float(grid[r_lo, c_lo])
    g01 = float(grid[r_lo, c_hi])
    g10 = float(grid[r_hi, c_lo])
    g11 = float(grid[r_hi, c_hi])
    return (
        (1.0 - fr) * (1.0 - fc) * g00
        + (1.0 - fr) * fc * g01
        + fr * (1.0 - fc) * g10
        + fr * fc * g11
    )


def _latlonhae_to_xyz(lat: float, lon: float, hae: float) -> XYZ:
    """Convert one WGS-84 point to ECF."""
    ecf = geodetic_to_ecef(np.array([lat, lon, hae], dtype=np.float64))
    return XYZ.from_array(ecf)


def _interpolate_nisar_llh(
    ctx: Dict[str, Any],
    row: float,
    col: float,
) -> LatLonHAE:
    """Interpolate a geodetic point from NISAR grid context."""
    lat = _bilinear_interpolate(
        ctx['lat_grid'], ctx['row_axis'], ctx['col_axis'], row, col,
    )
    lon = _bilinear_interpolate(
        ctx['lon_grid'], ctx['row_axis'], ctx['col_axis'], row, col,
    )
    return LatLonHAE(lat=float(lat), lon=float(lon), hae=float(ctx['height']))


def _interpolate_axis_value(
    samples: np.ndarray,
    sample_positions: np.ndarray,
    position: float,
) -> Optional[float]:
    """Linearly interpolate a 1-D sampled axis at a pixel position."""
    if samples.size == 0 or sample_positions.size == 0:
        return None
    if samples.size == 1 or sample_positions.size == 1:
        return float(np.asarray(samples, dtype=np.float64).ravel()[0])
    return float(
        np.interp(
            float(position),
            np.asarray(sample_positions, dtype=np.float64),
            np.asarray(samples, dtype=np.float64),
        )
    )


def _build_nisar_geo_data(
    rows: int,
    cols: int,
    ctx: Dict[str, Any],
) -> SICDGeoData:
    """Build SICD GeoData from the NISAR RSLC geolocation grid."""
    center_row = (rows - 1) / 2.0
    center_col = (cols - 1) / 2.0
    scp_llh = _interpolate_nisar_llh(ctx, center_row, center_col)
    scp = SICDSCP(
        ecf=_latlonhae_to_xyz(scp_llh.lat, scp_llh.lon, scp_llh.hae),
        llh=scp_llh,
    )
    image_corners = [
        _interpolate_nisar_llh(ctx, 0.0, 0.0),
        _interpolate_nisar_llh(ctx, 0.0, cols - 1),
        _interpolate_nisar_llh(ctx, rows - 1, cols - 1),
        _interpolate_nisar_llh(ctx, rows - 1, 0.0),
    ]
    return SICDGeoData(
        earth_model='WGS_84',
        scp=scp,
        image_corners=[LatLon(lat=pt.lat, lon=pt.lon) for pt in image_corners],
    )


def _build_nisar_grid(
    source_meta: NISARMetadata,
    rows: int,
    cols: int,
    ctx: Dict[str, Any],
    scp_time: float,
) -> SICDGrid:
    """Build a locally consistent SICD plane grid from NISAR metadata."""
    center_row = (rows - 1) / 2.0
    center_col = (cols - 1) / 2.0
    row_span = min(8.0, max(min(center_row, (rows - 1) - center_row), 1.0))
    col_span = min(8.0, max(min(center_col, (cols - 1) - center_col), 1.0))

    r0 = max(center_row - row_span, 0.0)
    r1 = min(center_row + row_span, rows - 1)
    c0 = max(center_col - col_span, 0.0)
    c1 = min(center_col + col_span, cols - 1)

    row_p0 = _interpolate_nisar_llh(ctx, r0, center_col)
    row_p1 = _interpolate_nisar_llh(ctx, r1, center_col)
    col_p0 = _interpolate_nisar_llh(ctx, center_row, c0)
    col_p1 = _interpolate_nisar_llh(ctx, center_row, c1)

    row_v = (
        _latlonhae_to_xyz(row_p1.lat, row_p1.lon, row_p1.hae).to_array()
        - _latlonhae_to_xyz(row_p0.lat, row_p0.lon, row_p0.hae).to_array()
    )
    col_v = (
        _latlonhae_to_xyz(col_p1.lat, col_p1.lon, col_p1.hae).to_array()
        - _latlonhae_to_xyz(col_p0.lat, col_p0.lon, col_p0.hae).to_array()
    )

    row_norm = float(np.linalg.norm(row_v))
    col_norm = float(np.linalg.norm(col_v))
    row_ss = row_norm / max(r1 - r0, 1.0)
    col_ss = col_norm / max(c1 - c0, 1.0)

    swath = source_meta.swath_parameters
    if not np.isfinite(row_ss) or row_ss == 0.0:
        row_ss = float(
            getattr(swath, 'scene_center_along_track_spacing', None) or 1.0,
        )
        row_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        row_norm = 1.0
    if not np.isfinite(col_ss) or col_ss == 0.0:
        col_ss = float(
            getattr(swath, 'slant_range_spacing', None)
            or getattr(swath, 'scene_center_ground_range_spacing', None)
            or 1.0,
        )
        col_v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        col_norm = 1.0

    return SICDGrid(
        image_plane='SLANT',
        type='PLANE',
        row=SICDDirParam(
            uvect_ecf=XYZ.from_array(row_v / row_norm),
            ss=float(row_ss),
            sgn=1,
        ),
        col=SICDDirParam(
            uvect_ecf=XYZ.from_array(col_v / col_norm),
            ss=float(col_ss),
            sgn=1,
        ),
        time_coa_poly=Poly2D(coefs=np.array([[float(scp_time)]], dtype=np.float64)),
    )


def _build_nisar_position(
    source_meta: NISARMetadata,
    collect_start: str,
) -> Optional[SICDPosition]:
    """Fit orbit state vectors into SICD Position.ARPPoly."""
    orbit = source_meta.orbit
    if orbit is None or orbit.time is None or orbit.position is None:
        return None
    time = np.asarray(orbit.time, dtype=np.float64).ravel()
    pos = np.asarray(orbit.position, dtype=np.float64)
    if time.size < 2 or pos.shape != (time.size, 3):
        return None

    t0 = _seconds_from_reference(collect_start, orbit.reference_epoch)
    if t0 is None:
        t0 = float(time[0])
    rel_time = time - t0
    deg = min(5, rel_time.size - 1)
    px = np.polynomial.Polynomial.fit(rel_time, pos[:, 0], deg=deg).convert()
    py = np.polynomial.Polynomial.fit(rel_time, pos[:, 1], deg=deg).convert()
    pz = np.polynomial.Polynomial.fit(rel_time, pos[:, 2], deg=deg).convert()
    return SICDPosition(
        arp_poly=XYZPoly(
            x=Poly1D(coefs=np.asarray(px.coef, dtype=np.float64)),
            y=Poly1D(coefs=np.asarray(py.coef, dtype=np.float64)),
            z=Poly1D(coefs=np.asarray(pz.coef, dtype=np.float64)),
        )
    )


def _interpolate_orbit_vector(
    orbit: Any,
    seconds_since_reference: float,
    field: str,
) -> Optional[np.ndarray]:
    """Interpolate an orbit position or velocity vector at the requested time."""
    time = np.asarray(getattr(orbit, 'time', None), dtype=np.float64)
    values = np.asarray(getattr(orbit, field, None), dtype=np.float64)
    if time.ndim != 1 or values.ndim != 2 or values.shape != (time.size, 3):
        return None
    return np.array([
        np.interp(seconds_since_reference, time, values[:, i]) for i in range(3)
    ], dtype=np.float64)


def _interpolate_orbit_acceleration(
    orbit: Any,
    seconds_since_reference: float,
) -> Optional[np.ndarray]:
    """Differentiate velocity samples to estimate ARP acceleration."""
    time = np.asarray(getattr(orbit, 'time', None), dtype=np.float64)
    vel = np.asarray(getattr(orbit, 'velocity', None), dtype=np.float64)
    if time.ndim != 1 or vel.ndim != 2 or vel.shape != (time.size, 3) or time.size < 2:
        return None
    idx = int(np.clip(np.searchsorted(time, seconds_since_reference), 1, time.size - 1))
    t0 = float(time[idx - 1])
    t1 = float(time[idx])
    if t1 == t0:
        return None
    return (vel[idx] - vel[idx - 1]) / (t1 - t0)


def _build_nisar_scpcoa(
    source_meta: NISARMetadata,
    rows: int,
    cols: int,
    ctx: Dict[str, Any],
    side_of_track: Optional[str],
    collect_start: str,
) -> SICDSCPCOA:
    """Build SICD SCPCOA from NISAR orbit and geolocation-grid metadata."""
    center_row = (rows - 1) / 2.0
    center_col = (cols - 1) / 2.0

    scp_time_abs = _interpolate_axis_value(
        ctx['time_axis'], ctx['row_axis'], center_row,
    )
    slant_range = _interpolate_axis_value(
        ctx['range_axis'], ctx['col_axis'], center_col,
    )
    time_ref = _time_reference_from_source(source_meta)
    collect_start_seconds = _seconds_from_reference(collect_start, time_ref)

    if scp_time_abs is not None and collect_start_seconds is not None:
        scp_time = float(scp_time_abs - collect_start_seconds)
        orbit_time = float(scp_time_abs)
    else:
        duration = _collect_duration_from_source(source_meta)
        if duration is not None and rows > 1:
            scp_time = float(duration * (center_row / (rows - 1)))
        else:
            scp_time = 0.0
        orbit_time = (
            float(collect_start_seconds + scp_time)
            if collect_start_seconds is not None else scp_time
        )

    orbit = source_meta.orbit
    arp_pos = (
        _interpolate_orbit_vector(orbit, orbit_time, 'position')
        if orbit is not None else None
    )
    arp_vel = (
        _interpolate_orbit_vector(orbit, orbit_time, 'velocity')
        if orbit is not None else None
    )
    arp_acc = (
        _interpolate_orbit_acceleration(orbit, orbit_time)
        if orbit is not None else None
    )

    incidence = None
    if ctx['inc_grid'] is not None:
        incidence = _bilinear_interpolate(
            ctx['inc_grid'], ctx['row_axis'], ctx['col_axis'],
            center_row, center_col,
        )
    elevation = None
    if ctx['ele_grid'] is not None:
        elevation = _bilinear_interpolate(
            ctx['ele_grid'], ctx['row_axis'], ctx['col_axis'],
            center_row, center_col,
        )

    graze = None
    if incidence is not None and np.isfinite(incidence):
        graze = float(90.0 - incidence)
    elif elevation is not None and np.isfinite(elevation):
        graze = float(elevation)

    return SICDSCPCOA(
        scp_time=float(scp_time),
        arp_pos=XYZ.from_array(arp_pos) if arp_pos is not None else None,
        arp_vel=XYZ.from_array(arp_vel) if arp_vel is not None else None,
        arp_acc=XYZ.from_array(arp_acc) if arp_acc is not None else None,
        side_of_track=side_of_track,
        slant_range=float(slant_range) if slant_range is not None else None,
        incidence_ang=float(incidence) if incidence is not None else None,
        graze_ang=graze,
    )


def _build_nisar_sicd_sections(
    source_meta: NISARMetadata,
    rows: int,
    cols: int,
    side_of_track: Optional[str],
    collect_start: str,
) -> Dict[str, Any]:
    """Build SICD geometry sections from NISAR RSLC metadata."""
    ctx = _build_nisar_geometry_context(source_meta, rows, cols)
    if ctx is None:
        return {}

    scpcoa = _build_nisar_scpcoa(
        source_meta, rows, cols, ctx, side_of_track, collect_start,
    )
    return {
        'geo_data': _build_nisar_geo_data(rows, cols, ctx),
        'grid': _build_nisar_grid(
            source_meta, rows, cols, ctx, scpcoa.scp_time or 0.0,
        ),
        'position': _build_nisar_position(source_meta, collect_start),
        'timeline': SICDTimeline(
            collect_start=collect_start,
            collect_duration=_collect_duration_from_source(source_meta),
        ),
        'scpcoa': scpcoa,
    }


def _side_of_track_from_source(source_meta: object) -> Optional[str]:
    """Infer SICD SCPCOA.SideOfTrack from source metadata when available."""
    scpcoa = getattr(source_meta, 'scpcoa', None)
    side = getattr(scpcoa, 'side_of_track', None)
    if isinstance(side, str):
        side = side.strip().upper()
        if side in {'L', 'R'}:
            return side

    identification = getattr(source_meta, 'identification', None)
    look_direction = getattr(identification, 'look_direction', None)
    if isinstance(look_direction, str):
        token = look_direction.strip().upper()
        if token.startswith('L'):
            return 'L'
        if token.startswith('R'):
            return 'R'

    for attr in ('look_direction', 'side_of_track'):
        val = getattr(source_meta, attr, None)
        if isinstance(val, str):
            token = val.strip().upper()
            if token.startswith('L'):
                return 'L'
            if token.startswith('R'):
                return 'R'

    extras = getattr(source_meta, 'extras', {}) or {}
    for key in ('look_direction', 'side_of_track'):
        val = extras.get(key)
        if isinstance(val, str):
            token = val.strip().upper()
            if token.startswith('L'):
                return 'L'
            if token.startswith('R'):
                return 'R'
    return None


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
    swath = getattr(source_meta, 'swath_parameters', None)
    tx_freq = None
    fc = getattr(swath, 'processed_center_frequency', None) if swath is not None else None
    bw = getattr(swath, 'processed_range_bandwidth', None) if swath is not None else None
    if fc is not None and bw is not None:
        half_bw = 0.5 * float(bw)
        tx_freq = SICDTxFrequency(
            min=float(fc) - half_bw,
            max=float(fc) + half_bw,
        )
    radar_collection = SICDRadarCollection(
        tx_frequency=tx_freq,
        tx_polarization=_TX_POL[tx],
        rcv_channels=[
            SICDRcvChannel(tx_rcv_polarization=_TX_RCV_POL[tx][rcv_label], index=1)
        ]
    )
    image_data = SICDImageData(
        pixel_type='RE32F_IM32F',
        num_rows=rows,
        num_cols=cols,
        first_row=0,
        first_col=0,
        full_image=SICDFullImage(num_rows=rows, num_cols=cols),
        scp_pixel=RowCol(row=rows // 2, col=cols // 2),
    )

    sicd_base: Optional[SICDMetadata] = None
    side_of_track = _side_of_track_from_source(source_meta)
    collect_start = _collect_start_from_source(source_meta)
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
            collect_start=collect_start,
            collect_duration=_collect_duration_from_source(source_meta),
        )
        base_cinfo = sicd_base.collection_info or _stub_collection_info(source_meta)
        base_scpcoa = sicd_base.scpcoa
        if side_of_track is not None:
            if base_scpcoa is None:
                base_scpcoa = SICDSCPCOA(side_of_track=side_of_track)
            elif base_scpcoa.side_of_track != side_of_track:
                base_scpcoa = dataclasses.replace(base_scpcoa, side_of_track=side_of_track)
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
            scpcoa=base_scpcoa,
        )

    nisar_sections: Dict[str, Any] = {}
    if isinstance(source_meta, NISARMetadata):
        nisar_sections = _build_nisar_sicd_sections(
            source_meta, rows, cols, side_of_track, collect_start,
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
        geo_data=nisar_sections.get('geo_data'),
        grid=nisar_sections.get('grid'),
        position=nisar_sections.get('position'),
        timeline=nisar_sections.get(
            'timeline',
            SICDTimeline(
                collect_start=collect_start,
                collect_duration=_collect_duration_from_source(source_meta),
            ),
        ),
        collection_info=_stub_collection_info(source_meta),
        scpcoa=nisar_sections.get('scpcoa')
        if 'scpcoa' in nisar_sections
        else (
            SICDSCPCOA(side_of_track=side_of_track)
            if side_of_track is not None else None
        ),
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


def main(argv: Optional[List[str]] = None) -> int:
    """Run compact-pol synthesis from the command line."""
    from grdl.IO.sar import CompactPolC2H5Reader, SICDCollectionReader

    parser = argparse.ArgumentParser(
        description=(
            'Synthesize compact-pol outputs from quad-pol SAR data.\n\n'
            'Writes:\n'
            '  {stem}_rh.nitf  synthesized RH channel (SICD NITF)\n'
            '  {stem}_rv.nitf  synthesized RV channel (SICD NITF)\n'
            '  {stem}_c2.h5    PolSARpro-style compact-pol C2 covariance matrix'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input quad-pol product path (NISAR, BIOMASS, SICD, etc.)',
    )
    parser.add_argument(
        '--output-stem',
        type=Path,
        required=True,
        help='Output path stem; produces {stem}_rh.nitf, {stem}_rv.nitf, {stem}_c2.h5',
    )
    parser.add_argument(
        '--frequency',
        type=str,
        default=None,
        help='NISAR frequency band (A or B)',
    )
    parser.add_argument(
        '--transmit',
        type=str,
        default='R',
        choices=['R', 'L'],
        help='Circular transmit handedness (default: R)',
    )
    parser.add_argument(
        '--hh-band', type=int, default=None,
        help='1-based band index override for HH',
    )
    parser.add_argument(
        '--hv-band', type=int, default=None,
        help='1-based band index override for HV',
    )
    parser.add_argument(
        '--vh-band', type=int, default=None,
        help='1-based band index override for VH',
    )
    parser.add_argument(
        '--vv-band', type=int, default=None,
        help='1-based band index override for VV',
    )
    args = parser.parse_args(argv)

    rh_path, rv_path, c2_path = convert_quad_pol_to_compact(
        input_path=args.input,
        output_stem=args.output_stem,
        frequency=args.frequency,
        transmit=args.transmit,
        hh_band=args.hh_band,
        hv_band=args.hv_band,
        vh_band=args.vh_band,
        vv_band=args.vv_band,
    )
    print(f'[OK] {rh_path.name}')
    print(f'[OK] {rv_path.name}')
    print(f'[OK] {c2_path.name}')

    with SICDCollectionReader([rh_path, rv_path], polarizations=['RH', 'RV']) as r:
        meta = r.metadata
        print(f'[OK] SICD pair: bands={meta.bands}  shape=({meta.rows}, {meta.cols})')

    with CompactPolC2H5Reader(c2_path) as r:
        cube = r.read_full()
        print(f'[OK] C2 cube: shape={cube.shape}  dtype={cube.dtype}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
