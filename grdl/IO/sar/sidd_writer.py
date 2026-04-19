# -*- coding: utf-8 -*-
"""
SIDD Writer - Write derived SAR products in SIDD NITF format.

Converts GRDL's typed ``SIDDMetadata`` to sarpy's ``SIDDType`` and
writes the product image data via sarpy's NITF writer backend.

Dependencies
------------
sarpy

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
2026-03-07

Modified
--------
2026-03-10
"""

from __future__ import annotations

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.exceptions import DependencyError
from grdl.IO.base import ImageWriter
from grdl.IO.models import SIDDMetadata
from grdl.IO.models.common import XYZ, LatLon, Poly2D

try:
    from sarpy.io.product.sidd3_elements.SIDD import SIDDType
    from sarpy.io.product.sidd3_elements.ProductCreation import (
        ProductCreationType, ProcessorInformationType,
    )
    from sarpy.io.product.sidd3_elements.Display import ProductDisplayType
    from sarpy.io.product.sidd3_elements.GeoData import GeoDataType
    from sarpy.io.product.sidd3_elements.Measurement import (
        MeasurementType, PlaneProjectionType,
    )
    from sarpy.io.product.sidd3_elements.ExploitationFeatures import (
        ExploitationFeaturesType,
    )
    from sarpy.io.product.sidd import SIDDWriter as _SarpySIDDWriter
    _HAS_SARPY_SIDD = True
except ImportError:
    _HAS_SARPY_SIDD = False


# ===================================================================
# Conversion helpers: GRDL model → sarpy SIDDType v3
# ===================================================================

def _sidd_metadata_to_sarpy(meta: SIDDMetadata) -> 'SIDDType':
    """Convert GRDL SIDDMetadata to sarpy SIDDType v3.

    Builds a minimal but valid SIDDType from available metadata.
    Missing sections are left as None.

    Parameters
    ----------
    meta : SIDDMetadata
        GRDL typed SIDD metadata.

    Returns
    -------
    SIDDType
        Sarpy SIDD v3 metadata object ready for writing.
    """
    kwargs: Dict[str, Any] = {}

    # ProductCreation
    pc = meta.product_creation
    if pc is not None:
        proc_info = None
        pi = pc.processor_information
        if pi is not None:
            proc_info = ProcessorInformationType(
                Application=pi.application,
                ProcessingDateTime=pi.processing_date_time,
                Site=pi.site,
                Profile=pi.profile,
            )
        kwargs['ProductCreation'] = ProductCreationType(
            ProcessorInformation=proc_info,
            ProductName=pc.product_name,
            ProductClass=pc.product_class,
            ProductType=pc.product_type,
        )

    # Display — construct a minimal ProductDisplayType
    disp = meta.display
    if disp is not None:
        kwargs['Display'] = ProductDisplayType(
            PixelType=disp.pixel_type or 'MONO8I',
            NumBands=disp.num_bands or 1,
        )

    return SIDDType(**kwargs)


# ===================================================================
# SIDDWriter
# ===================================================================

class SIDDWriter(ImageWriter):
    """Write derived SAR imagery in SIDD NITF format.

    Accepts GRDL's ``SIDDMetadata`` and converts it to sarpy's internal
    SIDDType for NITF writing.  The product image data is written via
    sarpy's SIDD writer.

    Parameters
    ----------
    filepath : str or Path
        Output path for the SIDD NITF file.
    metadata : SIDDMetadata
        Typed SIDD metadata to populate the NITF header.

    Raises
    ------
    ImportError
        If sarpy is not installed.

    Examples
    --------
    >>> from grdl.IO.sar import SIDDWriter
    >>> from grdl.IO.models import SIDDMetadata, SIDDDisplay
    >>> meta = SIDDMetadata(
    ...     format='SIDD', rows=512, cols=512, dtype='uint8',
    ...     display=SIDDDisplay(pixel_type='MONO8I', num_bands=1),
    ... )
    >>> writer = SIDDWriter('output.nitf', metadata=meta)
    >>> writer.write(image_array)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        metadata: Optional[SIDDMetadata] = None,
    ) -> None:
        if not _HAS_SARPY_SIDD:
            raise DependencyError(
                "sarpy is required for SIDDWriter. "
                "Install with: pip install sarpy"
            )
        super().__init__(filepath, metadata)

        if metadata is not None:
            self._sarpy_meta = _sidd_metadata_to_sarpy(metadata)
        else:
            self._sarpy_meta = SIDDType()

    def set_sarpy_metadata(self, sidd_type: 'SIDDType') -> None:
        """Override with a raw sarpy SIDDType for advanced use.

        Parameters
        ----------
        sidd_type : SIDDType
            Fully populated sarpy SIDD v3 metadata object.
        """
        self._sarpy_meta = sidd_type

    def write(
        self,
        data: np.ndarray,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write product image to SIDD NITF file.

        Parameters
        ----------
        data : np.ndarray
            2D or 3D array. Shape ``(rows, cols)`` for monochrome or
            ``(rows, cols, bands)`` for multi-band (e.g. RGB).
        geolocation : dict, optional
            Ignored for SIDD (geolocation is in the metadata).

        Raises
        ------
        ValueError
            If data has unsupported dimensions.
        """
        if data.ndim not in (2, 3):
            raise ValueError(
                f"SIDD data must be 2D or 3D, got shape {data.shape}"
            )

        writer = _SarpySIDDWriter(
            str(self.filepath),
            sidd_meta=self._sarpy_meta,
            check_existence=False,
        )
        writer.write_chip(data, start_indices=(0, 0), index=0)
        writer.close()

    def write_chip(
        self,
        data: np.ndarray,
        row_start: int,
        col_start: int,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a chip to an existing SIDD file.

        Not supported for SIDD format — SIDD files must be written
        as complete images via ``write()``.

        Raises
        ------
        NotImplementedError
            Always raised; SIDD does not support incremental chip writes
            through this interface.
        """
        raise NotImplementedError(
            "SIDD format does not support chip-level writes. "
            "Use write() with the full image."
        )
