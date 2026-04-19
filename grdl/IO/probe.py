# -*- coding: utf-8 -*-
"""
Invasive Probe Reader - Non-cooperative file probing for unknown formats.

Provides ``InvasiveProbeReader``, a reader that probes unknown files using
magic bytes, HDF5 tree walking, NetCDF introspection, companion file
scanning (ENVI headers, world files, XML sidecars), and binary structure
analysis to extract metadata and load data from files that no specialized
reader can handle.

Also provides ``ProbeEvidence``, the evidence accumulation dataclass, and
all probe functions used by the pipeline.

Dependencies
------------
h5py (optional)
xarray (optional)

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

from __future__ import annotations

# Standard library
import logging
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

try:
    import xarray as xr
    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models import ImageMetadata

logger = logging.getLogger(__name__)


# ===================================================================
# ProbeEvidence dataclass
# ===================================================================

@dataclass
class ProbeEvidence:
    """Accumulated evidence from non-cooperative file probing.

    Each probe function contributes partial knowledge about an unknown
    file.  Fields are all optional because no single probe is guaranteed
    to fill everything.  The evidence merger resolves the best consensus.

    Attributes
    ----------
    format_name : str or None
        Detected format (e.g., ``'HDF5'``, ``'NetCDF'``, ``'ENVI'``).
    magic_signature : str or None
        Human-readable name of recognized magic bytes.
    rows : int or None
        Image rows (may be guessed).
    cols : int or None
        Image columns (may be guessed).
    bands : int or None
        Number of bands.
    dtype : str or None
        NumPy dtype string.
    crs : str or None
        Coordinate reference system.
    modality : str or None
        Classified modality (SAR, EO, IR, MSI, HSI).
    modality_confidence : str
        Confidence level for modality classification.
    loading_strategy : str or None
        Backend to use: ``'hdf5'``, ``'netcdf'``, ``'envi'``,
        ``'memmap'``, ``'fits'``, ``'gdal'``.
    loading_hint : Dict[str, Any]
        Backend-specific params (e.g., dataset path for HDF5).
    extras : Dict[str, Any]
        All discovered metadata scraps.
    probes_run : List[str]
        Audit trail of probes executed.
    clues : List[str]
        Evidence supporting classifications.
    conflicts : List[str]
        Disagreements between probes.
    """

    format_name: Optional[str] = None
    magic_signature: Optional[str] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    bands: Optional[int] = None
    dtype: Optional[str] = None
    crs: Optional[str] = None
    modality: Optional[str] = None
    modality_confidence: str = 'low'
    loading_strategy: Optional[str] = None
    loading_hint: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    probes_run: List[str] = field(default_factory=list)
    clues: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)


# ===================================================================
# Magic byte signatures
# ===================================================================

_MAGIC_SIGNATURES: List[Tuple[bytes, str, Optional[str]]] = [
    # (signature_bytes, format_name, loading_strategy)
    (b'\x89HDF\r\n\x1a\n', 'HDF5', 'hdf5'),
    (b'CDF\x01', 'NetCDF_classic_32', 'netcdf'),
    (b'CDF\x02', 'NetCDF_classic_64', 'netcdf'),
    (b'II*\x00', 'TIFF_LE', 'gdal'),
    (b'MM\x00*', 'TIFF_BE', 'gdal'),
    (b'II+\x00', 'BigTIFF_LE', 'gdal'),
    (b'MM\x00+', 'BigTIFF_BE', 'gdal'),
    (b'\x89PNG\r\n\x1a\n', 'PNG', 'gdal'),
    (b'\xff\xd8\xff', 'JPEG', 'gdal'),
    (b'\x00\x00\x00\x0cjP', 'JPEG2000', 'gdal'),
    (b'BM', 'BMP', 'gdal'),
    (b'NITF', 'NITF', 'gdal'),
    (b'NSIF', 'NITF', 'gdal'),
]

# FITS uses an 80-byte card starting with SIMPLE
_FITS_MAGIC = b'SIMPLE  ='


# ===================================================================
# Probe functions
# ===================================================================

def _probe_magic_bytes(filepath: Path) -> ProbeEvidence:
    """Identify file format from the first 32 bytes.

    Parameters
    ----------
    filepath : Path
        Path to the file.

    Returns
    -------
    ProbeEvidence
        Evidence with format_name, magic_signature, and preliminary
        loading_strategy populated if a signature matches.
    """
    ev = ProbeEvidence()
    ev.probes_run.append('magic_bytes')

    try:
        with open(filepath, 'rb') as f:
            header = f.read(80)  # 80 bytes covers FITS card
    except (OSError, PermissionError) as e:
        ev.clues.append(f"cannot read file header: {e}")
        return ev

    if len(header) == 0:
        ev.clues.append("file is empty")
        return ev

    # Check FITS first (longer signature)
    if header[:len(_FITS_MAGIC)] == _FITS_MAGIC:
        ev.format_name = 'FITS'
        ev.magic_signature = 'FITS (SIMPLE = T)'
        ev.loading_strategy = 'fits'
        ev.clues.append("FITS magic bytes detected")
        return ev

    # Check other signatures
    for sig, fmt, strategy in _MAGIC_SIGNATURES:
        if header[:len(sig)] == sig:
            ev.format_name = fmt
            ev.magic_signature = fmt
            ev.loading_strategy = strategy
            ev.clues.append(f"{fmt} magic bytes detected")
            return ev

    # Check for text-based formats
    try:
        text_start = header[:64].decode('ascii', errors='strict')
        if text_start.strip().startswith('ENVI'):
            ev.format_name = 'ENVI_header'
            ev.magic_signature = 'ENVI header file'
            ev.loading_strategy = 'envi'
            ev.clues.append("ENVI header text detected")
            return ev
        if text_start.strip().startswith(('<?xml', '<xml')):
            ev.extras['is_xml'] = True
            ev.clues.append("XML text detected (companion or metadata)")
    except (UnicodeDecodeError, ValueError):
        pass

    ev.clues.append("no recognized magic bytes")
    return ev


def _probe_hdf5_walk(filepath: Path) -> ProbeEvidence:
    """Walk HDF5 structure to find datasets and attributes.

    Parameters
    ----------
    filepath : Path
        Path to an HDF5 file.

    Returns
    -------
    ProbeEvidence
        Evidence with dimensions, dtype, and dataset path from the
        largest 2D+ numeric dataset found.
    """
    ev = ProbeEvidence()

    if not _HAS_H5PY:
        ev.probes_run.append('hdf5_walk (skipped: h5py not installed)')
        return ev

    ev.probes_run.append('hdf5_walk')

    try:
        f = h5py.File(str(filepath), 'r')
    except Exception as e:
        ev.clues.append(f"h5py cannot open file: {e}")
        return ev

    try:
        # Collect all datasets
        datasets: List[Tuple[str, Any]] = []

        def _visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, obj))

        f.visititems(_visitor)

        if not datasets:
            ev.clues.append("HDF5 file contains no datasets")
            return ev

        ev.extras['hdf5_dataset_count'] = len(datasets)

        # Catalog all datasets
        dataset_info = []
        for ds_name, ds in datasets:
            info: Dict[str, Any] = {
                'path': '/' + ds_name,
                'shape': ds.shape,
                'dtype': str(ds.dtype),
                'ndim': ds.ndim,
            }
            # Collect dataset attributes
            if ds.attrs:
                info['attrs'] = {
                    k: _safe_attr_value(v) for k, v in ds.attrs.items()
                }
            dataset_info.append(info)

        ev.extras['hdf5_datasets'] = dataset_info

        # Pick the best dataset: largest 2D+ numeric array
        best_path: Optional[str] = None
        best_size = 0
        best_ds = None

        for ds_name, ds in datasets:
            if ds.ndim < 2:
                continue
            if not np.issubdtype(ds.dtype, np.number) and \
               not np.issubdtype(ds.dtype, np.complexfloating):
                continue
            size = 1
            for dim in ds.shape:
                size *= dim
            if size > best_size:
                best_size = size
                best_path = '/' + ds_name
                best_ds = ds

        if best_ds is not None:
            ev.rows = best_ds.shape[0]
            ev.cols = best_ds.shape[1]
            ev.bands = best_ds.shape[2] if best_ds.ndim >= 3 else 1
            ev.dtype = str(best_ds.dtype)
            ev.loading_strategy = 'hdf5'
            ev.loading_hint = {'dataset_path': best_path}
            ev.clues.append(
                f"best dataset: {best_path} "
                f"shape={best_ds.shape} dtype={best_ds.dtype}"
            )

            # Classify modality from dtype
            if np.issubdtype(best_ds.dtype, np.complexfloating):
                ev.modality = 'SAR'
                ev.modality_confidence = 'medium'
                ev.clues.append("complex dtype suggests SAR")
        else:
            ev.clues.append("no 2D+ numeric datasets found")

        # Collect root attributes
        root_attrs: Dict[str, Any] = {}
        for k, v in f.attrs.items():
            root_attrs[k] = _safe_attr_value(v)
        if root_attrs:
            ev.extras['hdf5_root_attrs'] = root_attrs

        # Check for CF conventions
        conventions = root_attrs.get('Conventions', '')
        if isinstance(conventions, str) and 'CF' in conventions.upper():
            ev.extras['has_cf_conventions'] = True
            ev.clues.append(f"CF Conventions: {conventions}")

        # Check for known SAR group structures
        _sar_group_markers = {
            'science/LSAR', 'science/SSAR', 'metadata/orbit',
            'swaths', 'calibration',
        }
        group_names = set()
        f.visit(lambda name: group_names.add(name.lower()))
        sar_matches = _sar_group_markers & group_names
        if sar_matches:
            ev.modality = 'SAR'
            ev.modality_confidence = 'high'
            ev.clues.append(
                f"SAR group structure: {', '.join(sar_matches)}"
            )

    finally:
        # Store handle for potential reuse by reader, but track it
        ev.extras['_hdf5_file'] = f

    return ev


def _safe_attr_value(val: Any) -> Any:
    """Convert HDF5 attribute values to JSON-safe Python types.

    Parameters
    ----------
    val : Any
        Raw HDF5 attribute value.

    Returns
    -------
    Any
        Python-native value.
    """
    if isinstance(val, bytes):
        try:
            return val.decode('utf-8')
        except UnicodeDecodeError:
            return repr(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val


# ---- ENVI header parsing ----

_ENVI_DTYPE_MAP: Dict[int, str] = {
    1: 'uint8',
    2: 'int16',
    3: 'int32',
    4: 'float32',
    5: 'float64',
    6: 'complex64',
    9: 'complex128',
    12: 'uint16',
    13: 'uint32',
    14: 'int64',
    15: 'uint64',
}


def _parse_envi_header(hdr_path: Path) -> Dict[str, Any]:
    """Parse an ENVI .hdr file into a metadata dictionary.

    Parameters
    ----------
    hdr_path : Path
        Path to the ENVI header file.

    Returns
    -------
    Dict[str, Any]
        Parsed key-value pairs with ENVI-specific conversions applied.
    """
    result: Dict[str, Any] = {}

    try:
        text = hdr_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return result

    lines = text.splitlines()
    if not lines or not lines[0].strip().upper().startswith('ENVI'):
        return result

    # Parse key = value pairs, handling multi-line {} values
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if not line or '=' not in line:
            i += 1
            continue

        key, _, value = line.partition('=')
        key = key.strip().lower()
        value = value.strip()

        # Handle multi-line brace-enclosed values
        if '{' in value and '}' not in value:
            while i + 1 < len(lines) and '}' not in value:
                i += 1
                value += ' ' + lines[i].strip()

        # Clean up brace-enclosed values
        if value.startswith('{') and value.endswith('}'):
            inner = value[1:-1].strip()
            # Try to parse as a list
            items = [item.strip() for item in inner.split(',')]
            try:
                value = [float(x) for x in items if x]
            except ValueError:
                value = [x for x in items if x]

        result[key] = value
        i += 1

    # Convert known numeric fields
    for int_key in ('lines', 'samples', 'bands', 'data type',
                     'byte order', 'header offset'):
        if int_key in result:
            try:
                result[int_key] = int(result[int_key])
            except (ValueError, TypeError):
                pass

    # Map ENVI data type code to numpy dtype
    if 'data type' in result and isinstance(result['data type'], int):
        result['numpy_dtype'] = _ENVI_DTYPE_MAP.get(
            result['data type'], None
        )

    return result


def _probe_companion_files(filepath: Path) -> ProbeEvidence:
    """Scan for sidecar/companion files near the target file.

    Looks for ENVI headers, XML metadata, projection files, and
    world files in the same directory.

    Parameters
    ----------
    filepath : Path
        Path to the data file.

    Returns
    -------
    ProbeEvidence
        Evidence populated from any discovered companion files.
    """
    ev = ProbeEvidence()
    ev.probes_run.append('companion_files')

    stem = filepath.stem
    parent = filepath.parent

    # ---- ENVI header (.hdr) ----
    hdr_path = parent / (stem + '.hdr')
    if not hdr_path.exists():
        hdr_path = parent / (stem + '.HDR')
    if hdr_path.exists() and hdr_path != filepath:
        hdr = _parse_envi_header(hdr_path)
        if hdr:
            ev.extras['envi_header'] = hdr
            ev.extras['envi_header_path'] = str(hdr_path)
            ev.clues.append(f"ENVI header found: {hdr_path.name}")

            if 'lines' in hdr and isinstance(hdr['lines'], int):
                ev.rows = hdr['lines']
            if 'samples' in hdr and isinstance(hdr['samples'], int):
                ev.cols = hdr['samples']
            if 'bands' in hdr and isinstance(hdr['bands'], int):
                ev.bands = hdr['bands']
            if 'numpy_dtype' in hdr and hdr['numpy_dtype'] is not None:
                ev.dtype = hdr['numpy_dtype']

            interleave = hdr.get('interleave', 'bsq')
            if isinstance(interleave, str):
                interleave = interleave.strip().lower()
            ev.loading_hint['interleave'] = interleave
            ev.loading_hint['header_offset'] = hdr.get(
                'header offset', 0
            )
            ev.loading_hint['byte_order'] = hdr.get('byte order', 0)
            ev.loading_hint['header_file'] = str(hdr_path)
            ev.loading_strategy = 'envi'

            # Map info for CRS
            map_info = hdr.get('map info')
            if isinstance(map_info, list) and len(map_info) >= 7:
                ev.extras['envi_map_info'] = map_info
                proj_name = map_info[0] if isinstance(
                    map_info[0], str
                ) else str(map_info[0])
                ev.clues.append(f"ENVI projection: {proj_name}")

            # Wavelength info for modality
            wavelengths = hdr.get('wavelength')
            if isinstance(wavelengths, list) and len(wavelengths) > 0:
                ev.extras['wavelengths'] = wavelengths
                n_wl = len(wavelengths)
                if n_wl > 100:
                    ev.modality = 'HSI'
                    ev.modality_confidence = 'high'
                    ev.clues.append(
                        f"{n_wl} wavelengths (hyperspectral)"
                    )
                elif n_wl > 10:
                    ev.modality = 'MSI'
                    ev.modality_confidence = 'medium'
                    ev.clues.append(
                        f"{n_wl} wavelengths (multispectral)"
                    )

    # ---- Projection file (.prj) ----
    prj_path = parent / (stem + '.prj')
    if not prj_path.exists():
        prj_path = parent / (stem + '.PRJ')
    if prj_path.exists():
        try:
            wkt = prj_path.read_text(encoding='utf-8', errors='replace')
            wkt = wkt.strip()
            if wkt:
                ev.crs = wkt
                ev.extras['prj_file'] = str(prj_path)
                ev.clues.append(f"CRS from .prj: {wkt[:60]}...")
        except OSError:
            pass

    # ---- World files (.tfw, .jgw, .pgw, .wld) ----
    _world_exts = ['.tfw', '.jgw', '.pgw', '.wld',
                   '.TFW', '.JGW', '.PGW', '.WLD']
    for wext in _world_exts:
        wf_path = parent / (stem + wext)
        if wf_path.exists():
            try:
                wf_lines = wf_path.read_text(
                    encoding='utf-8', errors='replace'
                ).strip().splitlines()
                if len(wf_lines) >= 6:
                    params = [float(x.strip()) for x in wf_lines[:6]]
                    ev.extras['world_file'] = {
                        'x_scale': params[0],
                        'y_rotation': params[1],
                        'x_rotation': params[2],
                        'y_scale': params[3],
                        'x_origin': params[4],
                        'y_origin': params[5],
                    }
                    ev.extras['world_file_path'] = str(wf_path)
                    ev.clues.append(
                        f"world file: {wf_path.name}"
                    )
            except (OSError, ValueError):
                pass
            break

    # ---- Auxiliary XML (.aux.xml) ----
    aux_path = filepath.with_suffix(filepath.suffix + '.aux.xml')
    if aux_path.exists():
        try:
            aux_text = aux_path.read_text(
                encoding='utf-8', errors='replace'
            )
            ev.extras['aux_xml'] = aux_text[:4096]
            ev.extras['aux_xml_path'] = str(aux_path)
            ev.clues.append(f"GDAL aux.xml found: {aux_path.name}")

            # Extract SRS if present
            if '<SRS>' in aux_text:
                start = aux_text.index('<SRS>') + 5
                end = aux_text.index('</SRS>', start)
                srs = aux_text[start:end].strip()
                if srs and ev.crs is None:
                    ev.crs = srs
                    ev.clues.append("CRS from aux.xml SRS")
        except (OSError, ValueError):
            pass

    return ev


def _probe_netcdf(filepath: Path) -> ProbeEvidence:
    """Probe NetCDF file structure using xarray.

    Parameters
    ----------
    filepath : Path
        Path to a NetCDF file.

    Returns
    -------
    ProbeEvidence
        Evidence with variable info, dimensions, and attributes.
    """
    ev = ProbeEvidence()

    if not _HAS_XARRAY:
        ev.probes_run.append('netcdf (skipped: xarray not installed)')
        return ev

    ev.probes_run.append('netcdf')

    try:
        ds = xr.open_dataset(str(filepath))
    except Exception as e:
        ev.clues.append(f"xarray cannot open file: {e}")
        return ev

    try:
        # Collect global attributes
        global_attrs = dict(ds.attrs)
        if global_attrs:
            ev.extras['netcdf_global_attrs'] = {
                k: _safe_attr_value(v) for k, v in global_attrs.items()
            }

        # Collect coordinate info
        coord_names = set(ds.coords.keys())
        ev.extras['netcdf_coordinates'] = list(coord_names)

        # Collect all variables
        var_info = []
        best_var: Optional[str] = None
        best_size = 0

        for var_name in ds.data_vars:
            var = ds[var_name]
            info: Dict[str, Any] = {
                'name': var_name,
                'shape': var.shape,
                'dtype': str(var.dtype),
                'dims': list(var.dims),
                'ndim': var.ndim,
            }
            if var.attrs:
                info['attrs'] = {
                    k: _safe_attr_value(v)
                    for k, v in var.attrs.items()
                }
            var_info.append(info)

            # Pick largest 2D+ numeric variable
            if var.ndim >= 2 and np.issubdtype(var.dtype, np.number):
                size = int(np.prod(var.shape))
                if size > best_size:
                    best_size = size
                    best_var = var_name

        ev.extras['netcdf_variables'] = var_info

        if best_var is not None:
            var = ds[best_var]
            # Handle 2D vs 3D+ shapes
            if var.ndim == 2:
                ev.rows = var.shape[0]
                ev.cols = var.shape[1]
                ev.bands = 1
            else:
                # Assume last two dims are spatial
                ev.rows = var.shape[-2]
                ev.cols = var.shape[-1]
                ev.bands = int(np.prod(var.shape[:-2]))
            ev.dtype = str(var.dtype)
            ev.loading_strategy = 'netcdf'
            ev.loading_hint = {
                'variable': best_var,
                'dims': list(var.dims),
            }
            ev.clues.append(
                f"best variable: {best_var} "
                f"shape={var.shape} dtype={var.dtype}"
            )

        # Check CRS from CF conventions
        for var_name in ds.data_vars:
            var = ds[var_name]
            if 'grid_mapping' in var.attrs:
                gm_name = var.attrs['grid_mapping']
                if gm_name in ds:
                    gm = ds[gm_name]
                    if 'crs_wkt' in gm.attrs:
                        ev.crs = str(gm.attrs['crs_wkt'])
                        ev.clues.append(
                            f"CRS from CF grid_mapping: {gm_name}"
                        )
                break

    finally:
        ds.close()

    return ev


def _probe_fits(filepath: Path) -> ProbeEvidence:
    """Probe FITS file for image data using astropy.

    Parameters
    ----------
    filepath : Path
        Path to a FITS file.

    Returns
    -------
    ProbeEvidence
        Evidence with dimensions and header metadata.
    """
    ev = ProbeEvidence()

    try:
        from astropy.io import fits as astro_fits
    except ImportError:
        ev.probes_run.append('fits (skipped: astropy not installed)')
        return ev

    ev.probes_run.append('fits')

    try:
        hdul = astro_fits.open(str(filepath), mode='readonly')
    except Exception as e:
        ev.clues.append(f"astropy cannot open file: {e}")
        return ev

    try:
        # Find the first image HDU
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                ev.rows = hdu.data.shape[-2]
                ev.cols = hdu.data.shape[-1]
                if hdu.data.ndim >= 3:
                    ev.bands = int(np.prod(hdu.data.shape[:-2]))
                else:
                    ev.bands = 1
                ev.dtype = str(hdu.data.dtype)
                ev.loading_strategy = 'fits'
                ev.loading_hint = {'hdu_index': hdul.index_of(hdu)}
                ev.clues.append(
                    f"FITS image HDU: shape={hdu.data.shape} "
                    f"dtype={hdu.data.dtype}"
                )
                break

        # Extract header metadata from primary HDU
        primary_hdr = hdul[0].header
        fits_meta: Dict[str, Any] = {}
        for card in primary_hdr.cards:
            if card.keyword and card.keyword not in (
                'SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'END', '',
            ) and not card.keyword.startswith('NAXIS'):
                fits_meta[card.keyword] = card.value

        if fits_meta:
            ev.extras['fits_header'] = fits_meta

    finally:
        hdul.close()

    return ev


def _probe_binary_structure(filepath: Path) -> ProbeEvidence:
    """Infer image dimensions and dtype from raw file structure.

    Last-resort probe that analyzes file size and byte statistics to
    guess the data format.  Produces candidates ranked by plausibility.

    Parameters
    ----------
    filepath : Path
        Path to the binary file.

    Returns
    -------
    ProbeEvidence
        Evidence with best-guess rows, cols, dtype, and a list of
        all candidates in ``extras['binary_candidates']``.
    """
    ev = ProbeEvidence()
    ev.probes_run.append('binary_structure')

    try:
        file_size = filepath.stat().st_size
    except OSError as e:
        ev.clues.append(f"cannot stat file: {e}")
        return ev

    if file_size == 0:
        ev.clues.append("file is empty")
        return ev

    ev.extras['file_size_bytes'] = file_size

    # Read a sample for byte statistics
    try:
        with open(filepath, 'rb') as f:
            sample = f.read(min(4096, file_size))
    except (OSError, PermissionError):
        sample = b''

    # Check if file is mostly text
    if sample:
        text_chars = sum(
            1 for b in sample
            if 32 <= b <= 126 or b in (9, 10, 13)
        )
        text_ratio = text_chars / len(sample)
        if text_ratio > 0.9:
            ev.extras['appears_text'] = True
            ev.clues.append(
                f"file appears to be text ({text_ratio:.0%} printable)"
            )
            return ev

    # Compute byte entropy
    if sample:
        byte_counts = np.zeros(256, dtype=np.float64)
        for b in sample:
            byte_counts[b] += 1
        byte_counts /= len(sample)
        nonzero = byte_counts[byte_counts > 0]
        entropy = -np.sum(nonzero * np.log2(nonzero))
        ev.extras['byte_entropy'] = float(entropy)

        if entropy > 7.9:
            ev.clues.append(
                f"high entropy ({entropy:.2f}) — likely compressed"
            )
            return ev

    # Try candidate dtypes
    _candidate_dtypes = [
        'float32', 'uint16', 'int16', 'uint8',
        'float64', 'complex64',
    ]

    candidates: List[Dict[str, Any]] = []

    for dtype_name in _candidate_dtypes:
        item_size = np.dtype(dtype_name).itemsize
        if file_size % item_size != 0:
            continue

        total_pixels = file_size // item_size

        # Find factor pairs that could be (rows, cols)
        for rows_cand in _factor_candidates(total_pixels):
            cols_cand = total_pixels // rows_cand
            if cols_cand < 1:
                continue

            aspect = max(rows_cand, cols_cand) / max(
                min(rows_cand, cols_cand), 1
            )
            score = _score_dimensions(
                rows_cand, cols_cand, aspect, dtype_name
            )
            candidates.append({
                'rows': rows_cand,
                'cols': cols_cand,
                'dtype': dtype_name,
                'aspect_ratio': round(aspect, 2),
                'score': score,
            })

    # Also try multi-band interpretations for common dtypes
    for dtype_name in ('uint8', 'uint16', 'float32'):
        item_size = np.dtype(dtype_name).itemsize
        for n_bands in (3, 4):
            band_size = file_size // (item_size * n_bands)
            if file_size != item_size * n_bands * band_size:
                continue
            for rows_cand in _factor_candidates(band_size):
                cols_cand = band_size // rows_cand
                if cols_cand < 1:
                    continue
                aspect = max(rows_cand, cols_cand) / max(
                    min(rows_cand, cols_cand), 1
                )
                score = _score_dimensions(
                    rows_cand, cols_cand, aspect, dtype_name
                )
                score += 5  # Bonus for RGB/RGBA interpretation
                candidates.append({
                    'rows': rows_cand,
                    'cols': cols_cand,
                    'bands': n_bands,
                    'dtype': dtype_name,
                    'aspect_ratio': round(aspect, 2),
                    'score': score,
                })

    if not candidates:
        ev.clues.append("no valid dimension candidates found")
        return ev

    # Sort by score descending
    candidates.sort(key=lambda c: c['score'], reverse=True)
    ev.extras['binary_candidates'] = candidates[:20]

    best = candidates[0]
    ev.rows = best['rows']
    ev.cols = best['cols']
    ev.bands = best.get('bands', 1)
    ev.dtype = best['dtype']
    ev.loading_strategy = 'memmap'
    ev.loading_hint = {
        'offset': 0,
        'order': 'C',
    }
    ev.clues.append(
        f"best guess: {best['rows']}x{best['cols']} "
        f"{best['dtype']} (score={best['score']:.1f}, "
        f"aspect={best['aspect_ratio']})"
    )
    if len(candidates) > 1:
        ev.clues.append(
            f"{len(candidates)} total candidates — "
            f"dimensions may be ambiguous"
        )

    return ev


def _factor_candidates(n: int) -> List[int]:
    """Generate plausible image row counts for a total pixel count.

    Produces factors of ``n`` that are likely image dimensions,
    prioritizing power-of-two multiples and common tile sizes.

    Parameters
    ----------
    n : int
        Total number of pixels.

    Returns
    -------
    List[int]
        Candidate row counts sorted by plausibility.
    """
    if n <= 0:
        return []

    factors = set()
    sqrt_n = int(math.isqrt(n))

    # Add exact sqrt if perfect square
    if sqrt_n * sqrt_n == n:
        factors.add(sqrt_n)

    # Common image dimensions
    _common_dims = [
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
        100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500,
        2000, 3000, 4000, 5000, 6000, 8000, 10000, 10980,
    ]

    for dim in _common_dims:
        if dim > 0 and n % dim == 0:
            quotient = n // dim
            if quotient >= 1:
                factors.add(dim)
                factors.add(quotient)

    # Factors near the square root (within 4x aspect ratio)
    lower = max(1, sqrt_n // 4)
    upper = min(n, sqrt_n * 4)
    for f in range(max(1, lower), min(upper + 1, 100001)):
        if n % f == 0:
            factors.add(f)
            factors.add(n // f)

    return sorted(factors)


def _score_dimensions(
    rows: int,
    cols: int,
    aspect: float,
    dtype_name: str,
) -> float:
    """Score a candidate (rows, cols, dtype) interpretation.

    Parameters
    ----------
    rows : int
        Candidate row count.
    cols : int
        Candidate column count.
    aspect : float
        Aspect ratio (larger/smaller).
    dtype_name : str
        NumPy dtype string.

    Returns
    -------
    float
        Plausibility score (higher is better).
    """
    score = 0.0

    # Prefer square-ish images
    if aspect <= 1.5:
        score += 20
    elif aspect <= 2.0:
        score += 15
    elif aspect <= 4.0:
        score += 10
    elif aspect <= 8.0:
        score += 5
    else:
        score -= 10

    # Prefer dimensions that are multiples of common tile sizes
    for tile in (256, 512, 1024):
        if rows % tile == 0:
            score += 3
        if cols % tile == 0:
            score += 3

    # Prefer power-of-2 dimensions
    if rows > 0 and (rows & (rows - 1)) == 0:
        score += 5
    if cols > 0 and (cols & (cols - 1)) == 0:
        score += 5

    # Prefer common geospatial dtypes
    _dtype_scores = {
        'float32': 8,
        'uint16': 7,
        'int16': 6,
        'uint8': 5,
        'float64': 4,
        'complex64': 3,
    }
    score += _dtype_scores.get(dtype_name, 0)

    # Penalize tiny dimensions
    if rows < 8 or cols < 8:
        score -= 20

    return score


# ===================================================================
# Evidence merger and probe pipeline
# ===================================================================

def _merge_evidence(probes: List[ProbeEvidence]) -> ProbeEvidence:
    """Merge multiple probe results into a single consensus.

    Parameters
    ----------
    probes : List[ProbeEvidence]
        Ordered list of probe results (earlier = higher priority
        for identity fields).

    Returns
    -------
    ProbeEvidence
        Merged evidence with conflicts recorded.
    """
    merged = ProbeEvidence()

    for ev in probes:
        merged.probes_run.extend(ev.probes_run)
        merged.clues.extend(ev.clues)

        # Identity: first non-None wins
        if merged.format_name is None and ev.format_name is not None:
            merged.format_name = ev.format_name
        if merged.magic_signature is None and ev.magic_signature is not None:
            merged.magic_signature = ev.magic_signature

        # Dimensions: first non-None wins (probes ordered by priority)
        if ev.rows is not None and ev.cols is not None:
            if merged.rows is None:
                merged.rows = ev.rows
                merged.cols = ev.cols
                merged.dtype = ev.dtype or merged.dtype
                merged.bands = ev.bands or merged.bands
                merged.loading_strategy = (
                    ev.loading_strategy or merged.loading_strategy
                )
                merged.loading_hint.update(ev.loading_hint)
            elif (merged.rows != ev.rows or merged.cols != ev.cols):
                merged.conflicts.append(
                    f"dimension conflict: "
                    f"({merged.rows}x{merged.cols}) vs "
                    f"({ev.rows}x{ev.cols}) from "
                    f"{ev.probes_run}"
                )

        # dtype: fill if still missing
        if merged.dtype is None and ev.dtype is not None:
            merged.dtype = ev.dtype

        # CRS: first non-None wins
        if merged.crs is None and ev.crs is not None:
            merged.crs = ev.crs

        # Modality: higher confidence wins
        _conf_rank = {'high': 3, 'medium': 2, 'low': 1}
        if ev.modality is not None:
            ev_rank = _conf_rank.get(ev.modality_confidence, 0)
            merged_rank = _conf_rank.get(
                merged.modality_confidence, 0
            )
            if merged.modality is None or ev_rank > merged_rank:
                merged.modality = ev.modality
                merged.modality_confidence = ev.modality_confidence

        # Loading strategy: first concrete one wins
        if merged.loading_strategy is None and \
                ev.loading_strategy is not None:
            merged.loading_strategy = ev.loading_strategy

        # Loading hints: union
        for k, v in ev.loading_hint.items():
            if k not in merged.loading_hint:
                merged.loading_hint[k] = v

        # Extras: union (later probes don't overwrite earlier)
        for k, v in ev.extras.items():
            if k not in merged.extras:
                merged.extras[k] = v

        # Conflicts from individual probes
        merged.conflicts.extend(ev.conflicts)

    return merged


def _run_probe_pipeline(filepath: Path) -> ProbeEvidence:
    """Execute the full probe pipeline on an unknown file.

    Parameters
    ----------
    filepath : Path
        Path to the file.

    Returns
    -------
    ProbeEvidence
        Merged evidence from all applicable probes.
    """
    probes: List[ProbeEvidence] = []

    # Probe 1: Magic bytes (always, < 1ms)
    magic_ev = _probe_magic_bytes(filepath)
    probes.append(magic_ev)

    fmt = magic_ev.format_name

    # Probe 2: Format-specific deep probes
    if fmt == 'HDF5':
        hdf5_ev = _probe_hdf5_walk(filepath)
        probes.append(hdf5_ev)
        if hdf5_ev.extras.get('has_cf_conventions'):
            probes.append(_probe_netcdf(filepath))
    elif fmt in ('NetCDF_classic_32', 'NetCDF_classic_64'):
        probes.append(_probe_netcdf(filepath))
    elif fmt == 'FITS':
        probes.append(_probe_fits(filepath))

    # Probe 3: Companion files (always)
    probes.append(_probe_companion_files(filepath))

    # Merge what we have so far
    merged = _merge_evidence(probes)

    # Probe 4: Binary structure (only if dimensions still unknown)
    if merged.rows is None or merged.cols is None:
        logger.debug("Dimensions unknown after initial probes, trying binary structure analysis")
        binary_ev = _probe_binary_structure(filepath)
        probes.append(binary_ev)
        merged = _merge_evidence(probes)

    logger.info(
        "Probe result for %s: format=%s, strategy=%s",
        filepath.name, merged.format_name, merged.loading_strategy,
    )
    logger.debug(
        "Probe evidence for %s: probes=%s, clues=%s",
        filepath.name, merged.probes_run, merged.clues,
    )

    return merged


# ===================================================================
# InvasiveProbeReader
# ===================================================================

class InvasiveProbeReader(ImageReader):
    """Non-cooperative reader that probes unknown files for metadata.

    Uses magic byte identification, HDF5/NetCDF introspection,
    companion file scanning, and binary structure analysis to extract
    the best possible metadata and provide ``read_chip()`` /
    ``read_full()`` for files that no specialized reader can handle.

    Parameters
    ----------
    filepath : str or Path
        Path to the file.

    Attributes
    ----------
    evidence : ProbeEvidence
        Full probe results, including audit trail and candidates.
    metadata : ImageMetadata
        Best-effort metadata constructed from probe evidence.

    Raises
    ------
    ValueError
        If probing cannot determine image dimensions.

    Examples
    --------
    >>> from grdl.IO.probe import InvasiveProbeReader
    >>> with InvasiveProbeReader('mystery.dat') as reader:
    ...     print(reader.evidence.clues)
    ...     print(reader.metadata.format)
    ...     chip = reader.read_chip(0, 256, 0, 256)
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.evidence: ProbeEvidence = ProbeEvidence()
        self._handle: Any = None  # h5py.File, mmap, xr.Dataset, etc.
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Run probe pipeline and construct metadata."""
        self.evidence = _run_probe_pipeline(self.filepath)

        if self.evidence.rows is None or self.evidence.cols is None:
            raise ValueError(
                f"Cannot determine image dimensions for "
                f"{self.filepath}. "
                f"Probes run: {self.evidence.probes_run}. "
                f"Clues: {self.evidence.clues}"
            )

        if self.evidence.dtype is None:
            self.evidence.dtype = 'uint8'
            self.evidence.clues.append(
                "dtype unknown, defaulting to uint8"
            )
            logger.warning(
                "dtype unknown for %s, defaulting to uint8",
                self.filepath.name,
            )

        # Build format name
        fmt = self.evidence.format_name or 'unknown'
        format_name = f"probed:{fmt}"

        # Build extras with probe namespace
        extras: Dict[str, Any] = {
            'probe_format': self.evidence.format_name,
            'probe_magic': self.evidence.magic_signature,
            'probe_modality': self.evidence.modality,
            'probe_modality_confidence': (
                self.evidence.modality_confidence
            ),
            'probe_loading_strategy': self.evidence.loading_strategy,
            'probe_loading_hint': self.evidence.loading_hint,
            'probe_audit': self.evidence.probes_run,
            'probe_clues': self.evidence.clues,
        }
        if self.evidence.conflicts:
            extras['probe_conflicts'] = self.evidence.conflicts

        # Merge in raw extras (skip internal handles)
        for k, v in self.evidence.extras.items():
            if not k.startswith('_'):
                extras[k] = v

        self.metadata = ImageMetadata(
            format=format_name,
            rows=self.evidence.rows,
            cols=self.evidence.cols,
            dtype=self.evidence.dtype,
            bands=self.evidence.bands,
            crs=self.evidence.crs,
            extras=extras,
        )

        # Open the loading backend
        self._open_backend()

    def _open_backend(self) -> None:
        """Open the appropriate loading backend based on evidence."""
        strategy = self.evidence.loading_strategy
        hint = self.evidence.loading_hint

        if strategy == 'hdf5' and _HAS_H5PY:
            # Reuse handle from probe if available
            hf = self.evidence.extras.get('_hdf5_file')
            if hf is not None and hf.id.valid:
                self._handle = hf
            else:
                self._handle = h5py.File(str(self.filepath), 'r')

        elif strategy == 'netcdf' and _HAS_XARRAY:
            self._handle = xr.open_dataset(str(self.filepath))

        elif strategy == 'envi':
            offset = hint.get('header_offset', 0)
            dtype = np.dtype(self.evidence.dtype)
            interleave = hint.get('interleave', 'bsq')
            byte_order = hint.get('byte_order', 0)

            if byte_order == 1:
                dtype = dtype.newbyteorder('>')
            else:
                dtype = dtype.newbyteorder('<')

            rows = self.evidence.rows
            cols = self.evidence.cols
            bands = self.evidence.bands or 1

            if interleave == 'bsq':
                shape = (bands, rows, cols)
            elif interleave == 'bil':
                shape = (rows, bands, cols)
            elif interleave == 'bip':
                shape = (rows, cols, bands)
            else:
                shape = (bands, rows, cols)

            self._handle = np.memmap(
                str(self.filepath), dtype=dtype, mode='r',
                offset=offset, shape=shape,
            )
            hint['_memmap_interleave'] = interleave

        elif strategy == 'memmap':
            offset = hint.get('offset', 0)
            dtype = np.dtype(self.evidence.dtype)
            rows = self.evidence.rows
            cols = self.evidence.cols
            bands = self.evidence.bands or 1

            if bands > 1:
                shape = (bands, rows, cols)
            else:
                shape = (rows, cols)

            self._handle = np.memmap(
                str(self.filepath), dtype=dtype, mode='r',
                offset=offset, shape=shape,
            )

        elif strategy == 'fits':
            try:
                from astropy.io import fits as astro_fits
                self._handle = astro_fits.open(
                    str(self.filepath), mode='readonly'
                )
            except ImportError:
                pass

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip using the probed loading strategy.

        Parameters
        ----------
        row_start : int
            Starting row index (inclusive).
        row_end : int
            Ending row index (exclusive).
        col_start : int
            Starting column index (inclusive).
        col_end : int
            Ending column index (exclusive).
        bands : Optional[List[int]]
            Band indices to read (0-based). If None, read all bands.

        Returns
        -------
        np.ndarray
            Image chip.

        Raises
        ------
        ValueError
            If indices are out of bounds or loading fails.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata.rows or \
                col_end > self.metadata.cols:
            raise ValueError("End indices exceed image dimensions")

        strategy = self.evidence.loading_strategy
        hint = self.evidence.loading_hint

        if strategy == 'hdf5' and self._handle is not None:
            return self._read_chip_hdf5(
                row_start, row_end, col_start, col_end, bands, hint,
            )

        if strategy == 'netcdf' and self._handle is not None:
            return self._read_chip_netcdf(
                row_start, row_end, col_start, col_end, bands, hint,
            )

        if strategy == 'envi' and self._handle is not None:
            return self._read_chip_envi(
                row_start, row_end, col_start, col_end, bands, hint,
            )

        if strategy == 'memmap' and self._handle is not None:
            return self._read_chip_memmap(
                row_start, row_end, col_start, col_end, bands,
            )

        if strategy == 'fits' and self._handle is not None:
            return self._read_chip_fits(
                row_start, row_end, col_start, col_end, bands, hint,
            )

        raise ValueError(
            f"No loading backend available for strategy "
            f"'{strategy}'. Probes: {self.evidence.probes_run}"
        )

    def _read_chip_hdf5(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]],
        hint: Dict[str, Any],
    ) -> np.ndarray:
        """Read chip from HDF5 dataset."""
        ds_path = hint.get('dataset_path')
        if ds_path is None:
            raise ValueError("No HDF5 dataset path in probe hints")

        ds = self._handle[ds_path]

        if ds.ndim == 2:
            data = ds[row_start:row_end, col_start:col_end]
        elif ds.ndim >= 3:
            if bands is not None:
                data = ds[
                    row_start:row_end, col_start:col_end, bands
                ]
            else:
                data = ds[
                    row_start:row_end, col_start:col_end, ...
                ]
        else:
            raise ValueError(
                f"Dataset {ds_path} has {ds.ndim} dimensions"
            )

        return np.asarray(data)

    def _read_chip_netcdf(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]],
        hint: Dict[str, Any],
    ) -> np.ndarray:
        """Read chip from NetCDF variable."""
        var_name = hint.get('variable')
        if var_name is None:
            raise ValueError("No NetCDF variable in probe hints")

        var = self._handle[var_name]
        dims = hint.get('dims', list(var.dims))

        # Build index dict for last two spatial dims
        sel: Dict[str, Any] = {}
        sel[dims[-2]] = slice(row_start, row_end)
        sel[dims[-1]] = slice(col_start, col_end)

        if bands is not None and len(dims) > 2:
            sel[dims[0]] = bands

        data = var.isel(**sel).values
        return data

    def _read_chip_envi(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]],
        hint: Dict[str, Any],
    ) -> np.ndarray:
        """Read chip from ENVI memmap with interleave handling."""
        interleave = hint.get('_memmap_interleave', 'bsq')
        mmap = self._handle

        if bands is None:
            band_sel = slice(None)
        else:
            band_sel = bands

        if interleave == 'bsq':
            # shape: (bands, rows, cols)
            data = mmap[band_sel, row_start:row_end, col_start:col_end]
        elif interleave == 'bil':
            # shape: (rows, bands, cols)
            data = mmap[row_start:row_end, band_sel, col_start:col_end]
            if data.ndim == 3:
                data = np.moveaxis(data, 1, 0)
        elif interleave == 'bip':
            # shape: (rows, cols, bands)
            data = mmap[
                row_start:row_end, col_start:col_end, band_sel
            ]
            if data.ndim == 3:
                data = np.moveaxis(data, 2, 0)
        else:
            data = mmap[band_sel, row_start:row_end, col_start:col_end]

        data = np.array(data)

        # Squeeze single band
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]

        return data

    def _read_chip_memmap(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]],
    ) -> np.ndarray:
        """Read chip from raw binary memmap."""
        mmap = self._handle

        if mmap.ndim == 2:
            data = np.array(
                mmap[row_start:row_end, col_start:col_end]
            )
        elif mmap.ndim >= 3:
            if bands is not None:
                data = np.array(
                    mmap[bands, row_start:row_end, col_start:col_end]
                )
            else:
                data = np.array(
                    mmap[:, row_start:row_end, col_start:col_end]
                )
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]
        else:
            raise ValueError(f"Memmap has {mmap.ndim} dimensions")

        return data

    def _read_chip_fits(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]],
        hint: Dict[str, Any],
    ) -> np.ndarray:
        """Read chip from FITS HDU."""
        hdu_idx = hint.get('hdu_index', 0)
        data = self._handle[hdu_idx].data

        if data.ndim == 2:
            return np.array(
                data[row_start:row_end, col_start:col_end]
            )

        if bands is not None:
            return np.array(
                data[..., row_start:row_end, col_start:col_end][bands]
            )

        result = np.array(
            data[..., row_start:row_end, col_start:col_end]
        )
        if result.ndim >= 3 and result.shape[0] == 1:
            result = result[0]
        return result

    def get_shape(self) -> Tuple[int, ...]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, ...]
            ``(rows, cols)`` for single band or
            ``(rows, cols, bands)`` for multi-band.
        """
        b = self.metadata.bands
        if b is None or b <= 1:
            return (self.metadata.rows, self.metadata.cols)
        return (self.metadata.rows, self.metadata.cols, b)

    def get_dtype(self) -> np.dtype:
        """Get the data type of the image.

        Returns
        -------
        np.dtype
        """
        return np.dtype(self.metadata.dtype)

    def close(self) -> None:
        """Close any open file handles."""
        if self._handle is None:
            return

        if _HAS_H5PY and isinstance(self._handle, h5py.File):
            try:
                self._handle.close()
            except Exception:
                pass
        elif _HAS_XARRAY and isinstance(self._handle, xr.Dataset):
            try:
                self._handle.close()
            except Exception:
                pass
        elif isinstance(self._handle, np.memmap):
            del self._handle
        elif hasattr(self._handle, 'close'):
            try:
                self._handle.close()
            except Exception:
                pass

        self._handle = None
