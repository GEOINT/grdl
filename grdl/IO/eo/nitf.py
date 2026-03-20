# -*- coding: utf-8 -*-
"""
EO NITF Reader - Read electro-optical NITF imagery with RPC/RSM metadata.

Extends the base NITF reading capability with extraction of EO-specific
geolocation models: RPC (Rational Polynomial Coefficients) from the
RPC00B TRE, and RSM (Replacement Sensor Model) from RSMPCA/RSMIDA TREs.

Uses rasterio (GDAL NITF driver) for pixel access and metadata
extraction.  No sarpy or sarkit dependency.

Dependencies
------------
rasterio

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
2026-03-17

Modified
--------
2026-03-19
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models.eo_nitf import (
    EONITFMetadata,
    RPCCoefficients,
    RSMCoefficients,
    RSMIdentification,
)
from grdl.IO.models.common import XYZ


def _parse_rsmpca_tre(tre_value: str) -> Optional[RSMCoefficients]:
    """Parse an RSMPCA TRE CEDATA string into RSMCoefficients.

    Field layout per STDI-0002 Vol 1 Appendix U, Table 4 (RSMPCA)::

        IID(80) EDITION(40) RSN(3) CSN(3) RFEP(21) CFEP(21)
        RNRMO(21) CNRMO(21) XNRMO(21) YNRMO(21) ZNRMO(21)
        RNRMSF(21) CNRMSF(21) XNRMSF(21) YNRMSF(21) ZNRMSF(21)
        RNPWRX(1) RNPWRY(1) RNPWRZ(1) RNTRMS(3) RNPCF(21)×RNTRMS
        RDPWRX(1) RDPWRY(1) RDPWRZ(1) RDTRMS(3) RDPCF(21)×RDTRMS
        CNPWRX(1) CNPWRY(1) CNPWRZ(1) CNTRMS(3) CNPCF(21)×CNTRMS
        CDPWRX(1) CDPWRY(1) CDPWRZ(1) CDTRMS(3) CDPCF(21)×CDTRMS

    Parameters
    ----------
    tre_value : str
        Raw CEDATA string from GDAL TRE metadata.

    Returns
    -------
    RSMCoefficients or None
        Parsed coefficients, or None if parsing fails.
    """
    try:
        v = tre_value.strip()
        # Minimum RSMPCA size is 486 bytes (per spec CEL range)
        if len(v) < 486:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_float(n: int = 21) -> float:
            nonlocal pos
            val = float(v[pos:pos + n].strip())
            pos += n
            return val

        def read_int(n: int = 1) -> int:
            nonlocal pos
            val = int(v[pos:pos + n].strip())
            pos += n
            return val

        # --- Image Information (skipped, not stored) ---
        _iid = read_str(80)       # IID
        _edition = read_str(40)   # EDITION

        # --- Section identification ---
        _rsn = read_int(3)        # RSN (row section number)
        _csn = read_int(3)        # CSN (column section number)

        # --- Fitting errors (optional, skipped) ---
        _rfep = read_str(21)      # RFEP
        _cfep = read_str(21)      # CFEP

        # --- Normalization offsets (5 × 21 bytes) ---
        row_off = read_float()     # RNRMO
        col_off = read_float()     # CNRMO
        x_off = read_float()       # XNRMO
        y_off = read_float()       # YNRMO
        z_off = read_float()       # ZNRMO

        # --- Normalization scale factors (5 × 21 bytes) ---
        row_norm_sf = read_float()  # RNRMSF
        col_norm_sf = read_float()  # CNRMSF
        x_norm_sf = read_float()    # XNRMSF
        y_norm_sf = read_float()    # YNRMSF
        z_norm_sf = read_float()    # ZNRMSF

        # --- Row numerator polynomial ---
        rnpwrx = read_int()         # RNPWRX
        rnpwry = read_int()         # RNPWRY
        rnpwrz = read_int()         # RNPWRZ
        rntms = read_int(3)         # RNTRMS
        row_num_coefs = np.array(
            [read_float() for _ in range(rntms)])

        # --- Row denominator polynomial ---
        rdpwrx = read_int()
        rdpwry = read_int()
        rdpwrz = read_int()
        rdtms = read_int(3)
        row_den_coefs = np.array(
            [read_float() for _ in range(rdtms)])

        # --- Column numerator polynomial ---
        cnpwrx = read_int()
        cnpwry = read_int()
        cnpwrz = read_int()
        cntms = read_int(3)
        col_num_coefs = np.array(
            [read_float() for _ in range(cntms)])

        # --- Column denominator polynomial ---
        cdpwrx = read_int()
        cdpwry = read_int()
        cdpwrz = read_int()
        cdtms = read_int(3)
        col_den_coefs = np.array(
            [read_float() for _ in range(cdtms)])

        return RSMCoefficients(
            row_off=row_off,
            col_off=col_off,
            row_norm_sf=row_norm_sf,
            col_norm_sf=col_norm_sf,
            x_off=x_off,
            y_off=y_off,
            z_off=z_off,
            x_norm_sf=x_norm_sf,
            y_norm_sf=y_norm_sf,
            z_norm_sf=z_norm_sf,
            row_num_powers=np.array([rnpwrx, rnpwry, rnpwrz]),
            row_den_powers=np.array([rdpwrx, rdpwry, rdpwrz]),
            col_num_powers=np.array([cnpwrx, cnpwry, cnpwrz]),
            col_den_powers=np.array([cdpwrx, cdpwry, cdpwrz]),
            row_num_coefs=row_num_coefs,
            row_den_coefs=row_den_coefs,
            col_num_coefs=col_num_coefs,
            col_den_coefs=col_den_coefs,
        )
    except (ValueError, IndexError):
        return None


def _parse_rsmida_tre(tre_value: str) -> Optional[RSMIdentification]:
    """Parse an RSMIDA TRE CEDATA string into RSMIdentification.

    Field layout per STDI-0002 Vol 1 Appendix U, Table 2 (RSMIDA)::

        IID(80) EDITION(40) ISID(40) SID(40) STID(40)
        YEAR(4) MONTH(2) DAY(2) HOUR(2) MINUTE(2) SECOND(9)
        NRG(8) NCG(8) TRG(21) TCG(21) GRNDD(1)
        XUOR(21) YUOR(21) ZUOR(21)
        XUXR(21)..ZUZR(21)  [9 unit-vector components]
        V1X(21)..V8Z(21)    [24 ground domain vertices]
        GRPX(21) GRPY(21) GRPZ(21)
        FULLR(8) FULLC(8) MINR(8) MAXR(8) MINC(8) MAXC(8)
        [illumination and trajectory fields follow]

    Total CEDATA length: 1628 bytes.

    Parameters
    ----------
    tre_value : str
        Raw CEDATA string from GDAL TRE metadata.

    Returns
    -------
    RSMIdentification or None
        Parsed identification, or None if parsing fails.
    """
    try:
        v = tre_value.strip()
        # RSMIDA CEDATA is fixed at 1628 bytes
        if len(v) < 500:
            return None

        pos = 0

        def read_str(n: int) -> str:
            nonlocal pos
            s = v[pos:pos + n].strip()
            pos += n
            return s

        def read_float(n: int = 21) -> float:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            if not raw:
                return 0.0
            return float(raw)

        def read_int(n: int) -> int:
            nonlocal pos
            raw = v[pos:pos + n].strip()
            pos += n
            if not raw:
                return 0
            return int(raw)

        # --- Image Information ---
        image_id = read_str(80)     # IID
        edition = read_str(40)      # EDITION

        # --- Sensor identification ---
        _isid = read_str(40)        # ISID (40 bytes, not 80)
        sensor_id = read_str(40)    # SID
        sensor_type_id = read_str(40)  # STID

        # --- Date/time ---
        _year = read_str(4)         # YEAR
        _month = read_str(2)        # MONTH
        _day = read_str(2)          # DAY
        _hour = read_str(2)         # HOUR
        _minute = read_str(2)       # MINUTE
        _second = read_str(9)       # SECOND

        # --- Time-of-image model ---
        nrg = read_int(8)           # NRG (8 bytes)
        ncg = read_int(8)           # NCG (8 bytes)
        _trg = read_str(21)         # TRG (21 bytes)
        _tcg = read_str(21)         # TCG (21 bytes)

        # --- Ground coordinate system ---
        grndd = read_str(1)         # GRNDD

        # --- Rectangular coordinate origin (3 × 21) ---
        _xuor = read_str(21)        # XUOR
        _yuor = read_str(21)        # YUOR
        _zuor = read_str(21)        # ZUOR

        # --- Rectangular unit vectors (9 × 21) ---
        for _ in range(9):
            read_str(21)            # XUXR..ZUZR

        # --- Ground domain vertices (24 × 21 = 504 bytes) ---
        for _ in range(24):
            read_str(21)            # V1X..V8Z

        # --- Ground reference point (3 × 21) ---
        grpx = read_float(21)       # GRPX
        grpy = read_float(21)       # GRPY
        grpz = read_float(21)       # GRPZ

        return RSMIdentification(
            image_id=image_id,
            edition=edition,
            sensor_id=sensor_id,
            sensor_type_id=sensor_type_id,
            ground_domain_type=grndd,
            ground_ref_point=XYZ(x=grpx, y=grpy, z=grpz),
            num_row_sections=nrg,
            num_col_sections=ncg,
        )
    except (ValueError, IndexError):
        return None


class EONITFReader(ImageReader):
    """Read electro-optical NITF imagery with RPC/RSM geolocation.

    Opens EO NITF files via rasterio (GDAL NITF driver) and extracts
    RPC00B and RSM TRE metadata in addition to standard imagery.

    Parameters
    ----------
    filepath : str or Path
        Path to the EO NITF file.

    Attributes
    ----------
    filepath : Path
        Path to the NITF file.
    metadata : EONITFMetadata
        Typed metadata including RPC/RSM geolocation models.
    dataset : rasterio.DatasetReader
        Rasterio dataset object.
    has_rpc : bool
        Whether RPC coefficients are available.
    has_rsm : bool
        Whether RSM coefficients are available.

    Raises
    ------
    ImportError
        If rasterio is not installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be opened as NITF.

    Examples
    --------
    >>> from grdl.IO.eo.nitf import EONITFReader
    >>> with EONITFReader('worldview.ntf') as reader:
    ...     print(f"RPC: {reader.has_rpc}, RSM: {reader.has_rsm}")
    ...     chip = reader.read_chip(0, 512, 0, 512)
    ...     if reader.has_rpc:
    ...         from grdl.geolocation.eo.rpc import RPCGeolocation
    ...         geo = RPCGeolocation.from_reader(reader)
    ...         lat, lon, h = geo.image_to_latlon(256, 256)
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        if not _HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for EO NITF reading. "
                "Install with: pip install rasterio  "
                "or: conda install -c conda-forge rasterio"
            )
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load EO NITF metadata including RPC and RSM TREs."""
        try:
            self.dataset = rasterio.open(str(self.filepath))
        except Exception as e:
            raise ValueError(
                f"Failed to open EO NITF file: {e}") from e

        # Extract RPC coefficients
        rpc = None
        try:
            rpcs_obj = self.dataset.rpcs
            if rpcs_obj is not None:
                rpc = RPCCoefficients.from_rasterio(rpcs_obj)
        except (AttributeError, TypeError):
            pass

        # Extract RSM from TRE namespace
        rsm = None
        rsm_id = None
        try:
            namespaces = self.dataset.tag_namespaces()
            for ns in namespaces:
                tags = self.dataset.tags(ns=ns)
                if tags:
                    for key, val in tags.items():
                        if 'RSMPCA' in key.upper() and rsm is None:
                            rsm = _parse_rsmpca_tre(val)
                        if 'RSMIDA' in key.upper() and rsm_id is None:
                            rsm_id = _parse_rsmida_tre(val)
        except (AttributeError, TypeError):
            pass

        # Extract NITF header fields from tags
        tags = self.dataset.tags() or {}
        iid1 = tags.get('NITF_IID1', tags.get('IID1'))
        iid2 = tags.get('NITF_IID2', tags.get('IID2'))
        icords = tags.get('NITF_ICORDS', tags.get('ICORDS'))
        icat = tags.get('NITF_ICAT', tags.get('ICAT'))
        abpp_str = tags.get('NITF_ABPP', tags.get('ABPP'))
        abpp = int(abpp_str) if abpp_str else None

        self.metadata = EONITFMetadata(
            format='NITF',
            rows=self.dataset.height,
            cols=self.dataset.width,
            bands=self.dataset.count,
            dtype=str(self.dataset.dtypes[0]),
            crs=str(self.dataset.crs) if self.dataset.crs else None,
            nodata=self.dataset.nodata,
            rpc=rpc,
            rsm=rsm,
            rsm_id=rsm_id,
            iid1=iid1,
            iid2=iid2,
            icords=icords,
            icat=icat,
            abpp=abpp,
        )

    @property
    def has_rpc(self) -> bool:
        """Whether RPC coefficients are available."""
        return self.metadata.rpc is not None

    @property
    def has_rsm(self) -> bool:
        """Whether RSM coefficients are available."""
        return self.metadata.rsm is not None

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the EO NITF file.

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
            Image chip with shape ``(rows, cols)`` for single band or
            ``(bands, rows, cols)`` for multi-band.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata.rows or col_end > self.metadata.cols:
            raise ValueError("End indices exceed image dimensions")

        window = Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start,
        )

        if bands is None:
            data = self.dataset.read(window=window)
        else:
            data = self.dataset.read(
                [b + 1 for b in bands], window=window)

        if data.shape[0] == 1:
            return data[0]
        return data

    def get_shape(self) -> Tuple[int, int]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, int]
            ``(rows, cols)``.
        """
        return (self.metadata.rows, self.metadata.cols)

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            Pixel data type.
        """
        return np.dtype(self.metadata.dtype)

    def close(self) -> None:
        """Close the rasterio dataset."""
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.dataset.close()
            self.dataset = None
