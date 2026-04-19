# -*- coding: utf-8 -*-
"""
CRSD Reader - Compensated Radar Signal Data format.

NGA standard for compensated radar signal data. Uses sarkit as the
backend (no sarpy fallback — sarpy does not fully support CRSD).

Dependencies
------------
sarkit

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
2026-04-18  Add typed CRSDMetadata with DwellPolynomials + Doppler rate fields.
2026-02-10
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models import ImageMetadata
from grdl.IO.models.common import XYZ, Poly2D
from grdl.IO.models.crsd import (
    CRSDChannelParameters,
    CRSDDwellPolynomialSet,
    CRSDMetadata,
    CRSDReferenceGeometry,
)
from grdl.IO.sar._backend import require_sarkit


# ===================================================================
# CRSD XML helpers
# ===================================================================

def _findtext_float(elem, path: str) -> Optional[float]:
    """Extract a float from an XML element; None if absent or blank."""
    if elem is None:
        return None
    raw = elem.findtext(path)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_ecf(elem) -> Optional[XYZ]:
    """Parse an ECF XYZ element with X/Y/Z children."""
    if elem is None:
        return None
    x = _findtext_float(elem, '{*}X')
    y = _findtext_float(elem, '{*}Y')
    z = _findtext_float(elem, '{*}Z')
    if x is None or y is None or z is None:
        return None
    return XYZ(x=x, y=y, z=z)


def _parse_xy(elem) -> Optional[Tuple[float, float]]:
    """Parse an image-area coordinate pair (X/Y children)."""
    if elem is None:
        return None
    x = _findtext_float(elem, '{*}X')
    y = _findtext_float(elem, '{*}Y')
    if x is None or y is None:
        return None
    return (x, y)


def _parse_crsd_poly2d(elem) -> Optional[Poly2D]:
    """Parse a 2-D polynomial element into :class:`Poly2D`.

    CRSD 2-D polynomials carry ``order1``/``order2`` attributes and a
    sequence of ``<Coef exponent1="i" exponent2="j">value</Coef>``
    children.  Coefficients are placed into a dense
    ``(order1+1, order2+1)`` array matching :class:`Poly2D`'s
    convention (``coefs[i, j]`` is the coefficient of ``x**i y**j``).
    """
    if elem is None:
        return None
    try:
        order1 = int(elem.get('order1'))
        order2 = int(elem.get('order2'))
    except (TypeError, ValueError):
        return None
    coefs = np.zeros((order1 + 1, order2 + 1), dtype=np.float64)
    for coef in elem.findall('{*}Coef'):
        try:
            i = int(coef.get('exponent1'))
            j = int(coef.get('exponent2'))
            value = float(coef.text) if coef.text else 0.0
        except (TypeError, ValueError):
            continue
        if 0 <= i <= order1 and 0 <= j <= order2:
            coefs[i, j] = value
    return Poly2D(coefs=coefs)


class CRSDReader(ImageReader):
    """Read CRSD (Compensated Radar Signal Data) format.

    CRSD is the NGA standard for compensated radar signal data,
    providing a standardized format for bistatic and multistatic SAR
    collections. Requires sarkit (no sarpy fallback).

    Parameters
    ----------
    filepath : str or Path
        Path to the CRSD file.

    Attributes
    ----------
    filepath : Path
        Path to the CRSD file.
    metadata : Dict[str, Any]
        Standardized metadata dictionary.

    Raises
    ------
    ImportError
        If sarkit is not installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid CRSD.

    Examples
    --------
    >>> from grdl.IO.sar import CRSDReader
    >>> with CRSDReader('data.crsd') as reader:
    ...     shape = reader.get_shape()
    ...     signal = reader.read_chip(0, 100, 0, 200)
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        require_sarkit('CRSD')
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load CRSD metadata using sarkit."""
        import sarkit.crsd

        try:
            self._file_handle = open(str(self.filepath), 'rb')
            self._reader = sarkit.crsd.Reader(self._file_handle)

            xml = self._reader.metadata.xmltree
            product_type = xml.getroot().tag.split('}')[-1] \
                if xml.getroot().tag.startswith('{') else xml.getroot().tag

            # Per-channel vector/sample counts from Data/Receive
            channel_shapes: Dict[str, Dict[str, int]] = {}
            for ch_elem in xml.findall(
                '{*}Data/{*}Receive/{*}Channel'
            ):
                ch_id = ch_elem.findtext('{*}ChId')
                num_vectors = int(ch_elem.findtext('{*}NumVectors'))
                num_samples = int(ch_elem.findtext('{*}NumSamples'))
                channel_shapes[ch_id] = {
                    'num_vectors': num_vectors,
                    'num_samples': num_samples,
                }

            # Top-level DwellPolynomials: CODTime and DwellTime (SAR only)
            cod_polys: Dict[str, CRSDDwellPolynomialSet] = {}
            dwell_polys: Dict[str, CRSDDwellPolynomialSet] = {}
            for cod_elem in xml.findall(
                '{*}DwellPolynomials/{*}CODTime'
            ):
                ident = cod_elem.findtext('{*}Identifier')
                poly_elem = cod_elem.find('{*}CODTimePoly')
                poly = _parse_crsd_poly2d(poly_elem)
                if ident and poly is not None:
                    cod_polys[ident] = CRSDDwellPolynomialSet(
                        identifier=ident, poly=poly)
            for dt_elem in xml.findall(
                '{*}DwellPolynomials/{*}DwellTime'
            ):
                ident = dt_elem.findtext('{*}Identifier')
                poly_elem = dt_elem.find('{*}DwellTimePoly')
                poly = _parse_crsd_poly2d(poly_elem)
                if ident and poly is not None:
                    dwell_polys[ident] = CRSDDwellPolynomialSet(
                        identifier=ident, poly=poly)

            # Per-channel typed parameters
            channel_params: Dict[str, CRSDChannelParameters] = {}
            for ch_elem in xml.findall(
                '{*}Channel/{*}Parameters'
            ):
                ch_id = ch_elem.findtext('{*}Identifier')
                if ch_id is None:
                    continue

                shape = channel_shapes.get(ch_id, {})
                f0_ref = _findtext_float(ch_elem, '{*}F0Ref')
                ref_fixed_raw = ch_elem.findtext('{*}RefFreqFixed')
                ref_freq_fixed = None
                if ref_fixed_raw is not None:
                    ref_freq_fixed = ref_fixed_raw.strip().lower() in (
                        'true', '1')

                # Receive reference point
                rcv_ecf = _parse_ecf(ch_elem.find('{*}RcvRefPoint/{*}ECF'))
                rcv_iac = _parse_xy(ch_elem.find('{*}RcvRefPoint/{*}IAC'))

                # SAR-only: Dwell polynomial cross-references
                cod_id = ch_elem.findtext(
                    '{*}SARImage/{*}DwellTimes/{*}Polynomials/{*}CODId')
                dwell_id = ch_elem.findtext(
                    '{*}SARImage/{*}DwellTimes/{*}Polynomials/{*}DwellId')

                cod_poly = cod_polys[cod_id].poly if cod_id and cod_id in cod_polys else None
                dwell_poly = dwell_polys[dwell_id].poly if dwell_id and dwell_id in dwell_polys else None

                channel_params[ch_id] = CRSDChannelParameters(
                    channel_id=ch_id,
                    num_vectors=shape.get('num_vectors'),
                    num_samples=shape.get('num_samples'),
                    f0_ref=f0_ref,
                    ref_freq_fixed=ref_freq_fixed,
                    rcv_ref_point_ecf=rcv_ecf,
                    rcv_ref_point_iac=rcv_iac,
                    cod_poly_id=cod_id,
                    dwell_poly_id=dwell_id,
                    cod_poly=cod_poly,
                    dwell_poly=dwell_poly,
                )

            # Scalar reference geometry
            rg_elem = xml.find('{*}ReferenceGeometry')
            reference_geometry = None
            if rg_elem is not None:
                ref_ecf = _parse_ecf(rg_elem.find('{*}RefPoint/{*}ECF'))
                ref_iac = _parse_xy(rg_elem.find('{*}RefPoint/{*}IAC'))
                cod_t = _findtext_float(rg_elem, '{*}SARImage/{*}CODTime')
                dwell_t = _findtext_float(rg_elem, '{*}SARImage/{*}DwellTime')
                reference_geometry = CRSDReferenceGeometry(
                    ref_point_ecf=ref_ecf,
                    ref_point_iac=ref_iac,
                    cod_time=cod_t,
                    dwell_time=dwell_t,
                )

            extras: Dict[str, Any] = {
                'backend': 'sarkit',
                'num_channels': len(channel_shapes),
                # Preserve legacy dict for any caller still relying on it
                'channels_legacy': channel_shapes,
            }

            collector = xml.findtext(
                '{*}CollectionInfo/{*}CollectorName'
            )
            classification = xml.findtext(
                '{*}CollectionInfo/{*}Classification'
            )
            if collector:
                extras['collector_name'] = collector
            if classification:
                extras['classification'] = classification

            # Use first channel dimensions as rows/cols
            first_ch = next(iter(channel_shapes.values()))
            self.metadata = CRSDMetadata(
                format='CRSD',
                rows=first_ch['num_vectors'],
                cols=first_ch['num_samples'],
                dtype='complex64',
                extras=extras,
                product_type=product_type,
                channels=channel_params,
                cod_polys=cod_polys,
                dwell_polys=dwell_polys,
                reference_geometry=reference_geometry,
            )

            self._xmltree = xml

        except Exception as e:
            raise ValueError(f"Failed to load CRSD metadata: {e}") from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read radar signal data.

        Parameters
        ----------
        row_start : int
            Starting vector index.
        row_end : int
            Ending vector index.
        col_start : int
            Starting sample index.
        col_end : int
            Ending sample index.
        bands : Optional[List[int]]
            Channel indices to read. If None, reads the first channel.

        Returns
        -------
        np.ndarray
            Signal data.
        """
        channel = bands[0] if bands else 0
        ch_id = list(self.metadata['channels'].keys())[channel]
        data = self._reader.read_signal(ch_id)
        return data[row_start:row_end, col_start:col_end]

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read full signal data for a channel.

        Parameters
        ----------
        bands : Optional[List[int]]
            Channel indices to read. If None, reads the first channel.

        Returns
        -------
        np.ndarray
            Signal data for the specified channel.
        """
        channel = bands[0] if bands else 0
        ch_id = list(self.metadata['channels'].keys())[channel]
        return self._reader.read_signal(ch_id)

    def get_shape(self) -> Tuple[int, ...]:
        """Get signal dimensions for first channel.

        Returns
        -------
        Tuple[int, ...]
            ``(num_vectors, num_samples)`` for the first channel.
        """
        first_channel = next(iter(self.metadata.channels.values()))
        return (first_channel.num_vectors, first_channel.num_samples)

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            ``complex64`` for CRSD signal data.
        """
        return np.dtype('complex64')

    def close(self) -> None:
        """Close the reader and release resources."""
        if hasattr(self, '_reader') and self._reader is not None:
            self._reader.done()
        if hasattr(self, '_file_handle') and self._file_handle is not None:
            self._file_handle.close()
