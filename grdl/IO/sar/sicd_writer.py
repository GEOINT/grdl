# -*- coding: utf-8 -*-
"""
SICD Writer - Write complex SAR imagery in SICD NITF format.

Converts GRDL's typed ``SICDMetadata`` to sarpy's ``SICDType`` and
writes the complex image data via sarpy's NITF writer backend.

Dependencies
------------
sarpy

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
2026-02-12

Modified
--------
2026-02-12
"""

# Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.base import ImageWriter
from grdl.IO.models import SICDMetadata
from grdl.IO.models.common import XYZ, LatLon, LatLonHAE, Poly1D, Poly2D, XYZPoly

try:
    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.complex.sicd_elements.CollectionInfo import (
        CollectionInfoType, RadarModeType,
    )
    from sarpy.io.complex.sicd_elements.GeoData import GeoDataType, SCPType
    from sarpy.io.complex.sicd_elements.ImageData import (
        ImageDataType, FullImageType,
    )
    from sarpy.io.complex.sicd_elements.Grid import GridType, DirParamType
    from sarpy.io.complex.sicd_elements.Timeline import TimelineType
    from sarpy.io.complex.sicd_elements.Position import (
        PositionType, XYZPolyType,
    )
    from sarpy.io.complex.sicd_elements.SCPCOA import SCPCOAType
    from sarpy.io.complex.sicd_elements.ImageFormation import (
        ImageFormationType, RcvChanProcType, TxFrequencyProcType,
        ProcessingType,
    )
    from sarpy.io.complex.sicd import SICDWriter as _SarpySICDWriter
    _HAS_SARPY = True
except ImportError:
    _HAS_SARPY = False


# ===================================================================
# Conversion helpers: GRDL model → sarpy types
# ===================================================================

def _to_sarpy_xyz(v: Optional[XYZ]) -> Optional[list]:
    """Convert GRDL XYZ to sarpy-compatible list."""
    if v is None:
        return None
    return [v.x, v.y, v.z]


def _to_sarpy_latlonhae(v: Optional[LatLonHAE]) -> Optional[list]:
    """Convert GRDL LatLonHAE to sarpy-compatible list."""
    if v is None:
        return None
    return [v.lat, v.lon, v.hae]


def _to_sarpy_xyzpoly(v: Optional[XYZPoly]) -> Optional[XYZPolyType]:
    """Convert GRDL XYZPoly to sarpy XYZPolyType."""
    if v is None or not _HAS_SARPY:
        return None
    x_coefs = v.x.coefs if v.x and v.x.coefs is not None else None
    y_coefs = v.y.coefs if v.y and v.y.coefs is not None else None
    z_coefs = v.z.coefs if v.z and v.z.coefs is not None else None
    return XYZPolyType(x_coefs, y_coefs, z_coefs)


def _to_sarpy_poly2d(v: Optional[Poly2D]) -> Optional[list]:
    """Convert GRDL Poly2D to sarpy-compatible nested list."""
    if v is None or v.coefs is None:
        return None
    return v.coefs.tolist()


def _sicd_metadata_to_sarpy(meta: SICDMetadata) -> 'SICDType':
    """Convert GRDL SICDMetadata to sarpy SICDType.

    Populates the SICD sections that are available in the metadata.
    Missing sections are left as None.

    Parameters
    ----------
    meta : SICDMetadata
        GRDL typed SICD metadata.

    Returns
    -------
    SICDType
        Sarpy SICD metadata object ready for writing.
    """
    sicd = SICDType()

    # CollectionInfo
    ci = meta.collection_info
    if ci is not None:
        radar_mode = None
        if ci.radar_mode is not None:
            radar_mode = RadarModeType(
                ModeType=ci.radar_mode.mode_type,
                ModeID=ci.radar_mode.mode_id,
            )
        sicd.CollectionInfo = CollectionInfoType(
            CollectorName=ci.collector_name,
            CoreName=ci.core_name,
            RadarMode=radar_mode,
            Classification=ci.classification,
        )

    # GeoData
    gd = meta.geo_data
    if gd is not None:
        scp = None
        if gd.scp is not None:
            scp = SCPType(
                _to_sarpy_xyz(gd.scp.ecf),
                _to_sarpy_latlonhae(gd.scp.llh),
            )
        corners = None
        if gd.image_corners:
            corners = [
                [c.lat, c.lon] for c in gd.image_corners
            ]
        sicd.GeoData = GeoDataType(
            gd.earth_model or 'WGS_84',
            scp,
            corners,
        )

    # ImageData
    idata = meta.image_data
    if idata is not None:
        pixel_type = idata.pixel_type or 'RE32F_IM32F'
        nr = idata.num_rows
        nc = idata.num_cols
        scp_pixel = None
        if idata.scp_pixel is not None:
            scp_pixel = [
                int(idata.scp_pixel.row), int(idata.scp_pixel.col),
            ]
        valid_data = [
            [0, 0], [0, nc - 1], [nr - 1, nc - 1], [nr - 1, 0],
        ]
        full_image = FullImageType(nr, nc)
        sicd.ImageData = ImageDataType(
            pixel_type, None, nr, nc,
            idata.first_row, idata.first_col,
            full_image, scp_pixel, valid_data,
        )

    # Grid
    grid = meta.grid
    if grid is not None:
        row_dp = _build_dir_param(grid.row)
        col_dp = _build_dir_param(grid.col)
        sicd.Grid = GridType(
            grid.image_plane,
            grid.type,
            _to_sarpy_poly2d(grid.time_coa_poly),
            row_dp,
            col_dp,
        )

    # Timeline
    tl = meta.timeline
    if tl is not None:
        sicd.Timeline = TimelineType(
            tl.collect_start,
            tl.collect_duration,
            None,
        )

    # Position
    pos = meta.position
    if pos is not None:
        sicd.Position = PositionType(
            _to_sarpy_xyzpoly(pos.arp_poly),
            _to_sarpy_xyzpoly(pos.grp_poly),
            _to_sarpy_xyzpoly(pos.tx_apc_poly),
            None,
        )

    # SCPCOA
    sc = meta.scpcoa
    if sc is not None:
        sicd.SCPCOA = SCPCOAType(
            sc.scp_time,
            _to_sarpy_xyz(sc.arp_pos),
            _to_sarpy_xyz(sc.arp_vel),
            _to_sarpy_xyz(sc.arp_acc),
            sc.side_of_track,
            sc.slant_range,
            sc.ground_range,
            sc.doppler_cone_ang,
            sc.graze_ang,
            sc.incidence_ang,
            sc.twist_ang,
            sc.slope_ang,
            sc.azim_ang,
            sc.layover_ang,
        )

    # ImageFormation
    imf = meta.image_formation
    if imf is not None:
        rcv_chan_proc = None
        rcp = imf.rcv_chan_proc
        if rcp is not None:
            rcv_chan_proc = RcvChanProcType(
                rcp.num_chan_proc,
                rcp.prf_scale_factor,
                rcp.chan_indices,
            )
        tx_freq_proc = None
        tfp = imf.tx_frequency_proc
        if tfp is not None:
            tx_freq_proc = TxFrequencyProcType(
                tfp.min_proc, tfp.max_proc,
            )
        sicd.ImageFormation = ImageFormationType(
            rcv_chan_proc,
            None,  # TxRcvPolarizationProc
            imf.t_start_proc,
            imf.t_end_proc,
            tx_freq_proc,
            imf.seg_id,
            imf.image_form_algo,
            None,  # STBeamComp
            imf.image_beam_comp,
            imf.az_autofocus,
            imf.rg_autofocus,
            None,  # Processing
            None,  # PolarizationCalibration
        )

    return sicd


def _build_dir_param(dp) -> Optional['DirParamType']:
    """Convert GRDL SICDDirParam to sarpy DirParamType."""
    if dp is None or not _HAS_SARPY:
        return None
    return DirParamType(
        _to_sarpy_xyz(dp.uvect_ecf),
        dp.ss,
        dp.imp_resp_wid,
        dp.sgn,
        dp.imp_resp_bw,
        dp.k_ctr,
        dp.delta_k1,
        dp.delta_k2,
        _to_sarpy_poly2d(dp.delta_k_coa_poly),
        None,  # WgtType
        None,  # WgtFunct
    )


# ===================================================================
# SICDWriter
# ===================================================================

class SICDWriter(ImageWriter):
    """Write complex SAR imagery in SICD NITF format.

    Accepts GRDL's ``SICDMetadata`` and converts it to sarpy's internal
    types for NITF writing. The complex image data is written via
    sarpy's SICD writer.

    Parameters
    ----------
    filepath : str or Path
        Output path for the SICD NITF file.
    metadata : SICDMetadata
        Typed SICD metadata to populate the NITF header.

    Raises
    ------
    ImportError
        If sarpy is not installed.

    Examples
    --------
    >>> from grdl.IO.sar import SICDWriter
    >>> from grdl.IO.models import SICDMetadata
    >>> writer = SICDWriter('output.nitf', metadata=sicd_meta)
    >>> writer.write(complex_image_array)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        metadata: Optional[SICDMetadata] = None,
    ) -> None:
        if not _HAS_SARPY:
            raise ImportError(
                "sarpy is required for SICDWriter. "
                "Install with: pip install sarpy"
            )
        super().__init__(filepath, metadata)

        if metadata is not None:
            self._sarpy_meta = _sicd_metadata_to_sarpy(metadata)
        else:
            self._sarpy_meta = SICDType()

    def set_sarpy_metadata(self, sicd_type: 'SICDType') -> None:
        """Override with a raw sarpy SICDType for advanced use.

        Parameters
        ----------
        sicd_type : SICDType
            Fully populated sarpy SICD metadata object.
        """
        self._sarpy_meta = sicd_type

    def write(
        self,
        data: np.ndarray,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write complex SAR image to SICD NITF file.

        Parameters
        ----------
        data : np.ndarray
            Complex-valued 2D array, shape ``(rows, cols)``.
        geolocation : dict, optional
            Ignored for SICD (geolocation is in the metadata).

        Raises
        ------
        ValueError
            If data is not complex-valued or not 2D.
        """
        if data.ndim != 2:
            raise ValueError(
                f"SICD data must be 2D, got shape {data.shape}"
            )
        if not np.iscomplexobj(data):
            raise ValueError(
                f"SICD data must be complex-valued, got dtype {data.dtype}"
            )

        writer = _SarpySICDWriter(
            str(self.filepath),
            self._sarpy_meta,
            check_existence=False,
        )
        writer.write(data)

    def write_chip(
        self,
        data: np.ndarray,
        row_start: int,
        col_start: int,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a chip to an existing SICD file.

        Not supported for SICD format — SICD files must be written
        as complete images.

        Raises
        ------
        NotImplementedError
            Always raised; SICD does not support partial writes.
        """
        raise NotImplementedError(
            "SICD format does not support chip-level writes. "
            "Use write() with the full image."
        )
