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
2026-03-17
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
    """Parse an RSMPCA TRE string into RSMCoefficients.

    The RSMPCA TRE is a fixed-format ASCII string with normalization
    parameters followed by polynomial coefficients.

    Parameters
    ----------
    tre_value : str
        Raw TRE value string from GDAL metadata.

    Returns
    -------
    RSMCoefficients or None
        Parsed coefficients, or None if parsing fails.
    """
    try:
        v = tre_value.strip()
        if len(v) < 200:
            return None

        pos = 0

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

        def read_int3() -> int:
            nonlocal pos
            val = int(v[pos:pos + 3].strip())
            pos += 3
            return val

        # Normalization parameters (5 × 21 bytes)
        row_norm_sf = read_float()
        col_norm_sf = read_float()
        x_norm_sf = read_float()
        y_norm_sf = read_float()
        z_norm_sf = read_float()

        # Row/col and ground offsets (5 × 21 bytes)
        row_off = read_float()
        col_off = read_float()
        x_off = read_float()
        y_off = read_float()
        z_off = read_float()

        # Row numerator polynomial structure
        rnpwrx = read_int()
        rnpwry = read_int()
        rnpwrz = read_int()
        rntms = read_int3()
        row_num_coefs = np.array(
            [read_float() for _ in range(rntms)])

        # Row denominator polynomial structure
        rdpwrx = read_int()
        rdpwry = read_int()
        rdpwrz = read_int()
        rdtms = read_int3()
        row_den_coefs = np.array(
            [read_float() for _ in range(rdtms)])

        # Column numerator polynomial structure
        cnpwrx = read_int()
        cnpwry = read_int()
        cnpwrz = read_int()
        cntms = read_int3()
        col_num_coefs = np.array(
            [read_float() for _ in range(cntms)])

        # Column denominator polynomial structure
        cdpwrx = read_int()
        cdpwry = read_int()
        cdpwrz = read_int()
        cdtms = read_int3()
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
    """Parse an RSMIDA TRE string into RSMIdentification.

    Parameters
    ----------
    tre_value : str
        Raw TRE value string from GDAL metadata.

    Returns
    -------
    RSMIdentification or None
        Parsed identification, or None if parsing fails.
    """
    try:
        v = tre_value.strip()
        if len(v) < 200:
            return None

        image_id = v[0:80].strip()
        edition = v[80:120].strip()

        # Skip to sensor fields
        pos = 120
        # ISID (80), SID (40), STID (40)
        isid = v[pos:pos + 80].strip()
        pos += 80
        sensor_id = v[pos:pos + 40].strip()
        pos += 40
        sensor_type_id = v[pos:pos + 40].strip()
        pos += 40

        # Skip date/time fields to reach ground domain
        # YEAR(4) MONTH(2) DAY(2) HOUR(2) MINUTE(2) SECOND(9) = 21 bytes
        pos += 21

        # NRG(3), NCG(3)
        nrg = int(v[pos:pos + 3].strip() or '0')
        pos += 3
        ncg = int(v[pos:pos + 3].strip() or '0')
        pos += 3

        # TRG(5), TCG(5)
        pos += 10

        # GRNDD(1) — ground domain type
        grndd = v[pos:pos + 1].strip()
        pos += 1

        # Ground reference point (3 × 21 bytes)
        grpx = float(v[pos:pos + 21].strip() or '0')
        pos += 21
        grpy = float(v[pos:pos + 21].strip() or '0')
        pos += 21
        grpz = float(v[pos:pos + 21].strip() or '0')
        pos += 21

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
