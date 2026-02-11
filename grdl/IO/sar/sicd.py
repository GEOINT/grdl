# -*- coding: utf-8 -*-
"""
SICD Reader - Sensor Independent Complex Data format.

NGA standard for complex SAR imagery. Uses sarkit as the primary backend
with sarpy as fallback. Populates all 17 SICD metadata sections as
nested dataclasses via ``SICDMetadata``.

Dependencies
------------
sarkit (primary) or sarpy (fallback)

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
2026-02-09

Modified
--------
2026-02-10
"""

# Standard library
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models import (
    SICDMetadata,
    SICDCollectionInfo,
    SICDRadarMode,
    SICDImageCreation,
    SICDImageData,
    SICDFullImage,
    SICDGeoData,
    SICDSCP,
    SICDGrid,
    SICDDirParam,
    SICDWgtType,
    SICDTimeline,
    SICDIPPSet,
    SICDPosition,
    SICDRadarCollection,
    SICDTxFrequency,
    SICDWaveformParams,
    SICDRcvChannel,
    SICDArea,
    SICDAreaCorner,
    SICDImageFormation,
    SICDRcvChanProc,
    SICDTxFrequencyProc,
    SICDProcessingStep,
    SICDSCPCOA,
    SICDRadiometric,
    SICDNoiseLevel,
    SICDAntenna,
    SICDAntParam,
    SICDEBType,
    SICDGainPhasePoly,
    SICDErrorStatistics,
    SICDCompositeSCPError,
    SICDPosVelErr,
    SICDCorrCoefs,
    SICDErrorDecorrFunc,
    SICDRadarSensorError,
    SICDMatchInfo,
    SICDMatchType,
    SICDMatchCollection,
    SICDRgAzComp,
    SICDPFA,
    SICDSTDeskew,
    SICDRMA,
    SICDRMRef,
    SICDINCA,
)
from grdl.IO.models.common import (
    XYZ,
    LatLon,
    LatLonHAE,
    RowCol,
    Poly1D,
    Poly2D,
    XYZPoly,
)
from grdl.IO.sar._backend import require_sar_backend


# ===================================================================
# XML extraction helpers (sarkit backend)
# ===================================================================

def _xml_float(elem: Optional[ET.Element], path: str) -> Optional[float]:
    """Extract float from XML path, returning None if absent."""
    val = elem.findtext(path) if elem is not None else None
    return float(val) if val is not None else None


def _xml_int(elem: Optional[ET.Element], path: str) -> Optional[int]:
    """Extract int from XML path, returning None if absent."""
    val = elem.findtext(path) if elem is not None else None
    return int(val) if val is not None else None


def _xml_str(elem: Optional[ET.Element], path: str) -> Optional[str]:
    """Extract string from XML path, returning None if absent."""
    return elem.findtext(path) if elem is not None else None


def _xml_bool(elem: Optional[ET.Element], path: str) -> Optional[bool]:
    """Extract boolean from XML path, returning None if absent."""
    val = elem.findtext(path) if elem is not None else None
    if val is None:
        return None
    return val.lower() in ('true', '1', 'yes')


def _xml_xyz(elem: Optional[ET.Element], path: str) -> Optional[XYZ]:
    """Extract XYZ from XML sub-element."""
    if elem is None:
        return None
    sub = elem.find(path)
    if sub is None:
        return None
    x = _xml_float(sub, '{*}X')
    y = _xml_float(sub, '{*}Y')
    z = _xml_float(sub, '{*}Z')
    if x is None and y is None and z is None:
        return None
    return XYZ(x=x or 0.0, y=y or 0.0, z=z or 0.0)


def _xml_latlonhae(
    elem: Optional[ET.Element], path: str,
) -> Optional[LatLonHAE]:
    """Extract LatLonHAE from XML sub-element."""
    if elem is None:
        return None
    sub = elem.find(path)
    if sub is None:
        return None
    lat = _xml_float(sub, '{*}Lat')
    lon = _xml_float(sub, '{*}Lon')
    hae = _xml_float(sub, '{*}HAE')
    if lat is None:
        return None
    return LatLonHAE(lat=lat, lon=lon or 0.0, hae=hae or 0.0)


def _xml_rowcol(elem: Optional[ET.Element], path: str) -> Optional[RowCol]:
    """Extract RowCol from XML sub-element."""
    if elem is None:
        return None
    sub = elem.find(path)
    if sub is None:
        return None
    row = _xml_float(sub, '{*}Row')
    col = _xml_float(sub, '{*}Col')
    if row is None:
        return None
    return RowCol(row=row, col=col or 0.0)


def _xml_poly1d(elem: Optional[ET.Element]) -> Optional[Poly1D]:
    """Extract Poly1D from XML element with Coef children."""
    if elem is None:
        return None
    order = int(elem.get('order1', '0'))
    coefs = np.zeros(order + 1)
    for coef_el in elem.findall('{*}Coef'):
        exp = int(coef_el.get('exponent1', '0'))
        coefs[exp] = float(coef_el.text)
    return Poly1D(coefs=coefs)


def _xml_poly1d_at(
    elem: Optional[ET.Element], path: str,
) -> Optional[Poly1D]:
    """Extract Poly1D from XML path."""
    if elem is None:
        return None
    sub = elem.find(path)
    return _xml_poly1d(sub)


def _xml_poly2d(elem: Optional[ET.Element]) -> Optional[Poly2D]:
    """Extract Poly2D from XML element with Coef children."""
    if elem is None:
        return None
    order1 = int(elem.get('order1', '0'))
    order2 = int(elem.get('order2', '0'))
    coefs = np.zeros((order1 + 1, order2 + 1))
    for coef_el in elem.findall('{*}Coef'):
        exp1 = int(coef_el.get('exponent1', '0'))
        exp2 = int(coef_el.get('exponent2', '0'))
        coefs[exp1, exp2] = float(coef_el.text)
    return Poly2D(coefs=coefs)


def _xml_poly2d_at(
    elem: Optional[ET.Element], path: str,
) -> Optional[Poly2D]:
    """Extract Poly2D from XML path."""
    if elem is None:
        return None
    sub = elem.find(path)
    return _xml_poly2d(sub)


def _xml_xyzpoly(
    elem: Optional[ET.Element], path: str,
) -> Optional[XYZPoly]:
    """Extract XYZPoly from XML sub-element."""
    if elem is None:
        return None
    sub = elem.find(path)
    if sub is None:
        return None
    x = _xml_poly1d(sub.find('{*}X'))
    y = _xml_poly1d(sub.find('{*}Y'))
    z = _xml_poly1d(sub.find('{*}Z'))
    return XYZPoly(x=x, y=y, z=z)


# ===================================================================
# Sarpy extraction helpers
# ===================================================================

def _safe_get(obj: Any, *attrs: str) -> Any:
    """Safely traverse nested attributes, returning None if any is None."""
    for attr in attrs:
        if obj is None:
            return None
        obj = getattr(obj, attr, None)
    return obj


def _sarpy_xyz(obj: Any) -> Optional[XYZ]:
    """Convert sarpy XYZType to XYZ."""
    if obj is None:
        return None
    return XYZ(
        x=float(obj.X) if obj.X is not None else 0.0,
        y=float(obj.Y) if obj.Y is not None else 0.0,
        z=float(obj.Z) if obj.Z is not None else 0.0,
    )


def _sarpy_latlon(obj: Any) -> Optional[LatLon]:
    """Convert sarpy LatLonType to LatLon."""
    if obj is None:
        return None
    return LatLon(
        lat=float(obj.Lat) if obj.Lat is not None else 0.0,
        lon=float(obj.Lon) if obj.Lon is not None else 0.0,
    )


def _sarpy_latlonhae(obj: Any) -> Optional[LatLonHAE]:
    """Convert sarpy LatLonHAEType to LatLonHAE."""
    if obj is None:
        return None
    return LatLonHAE(
        lat=float(obj.Lat) if obj.Lat is not None else 0.0,
        lon=float(obj.Lon) if obj.Lon is not None else 0.0,
        hae=float(obj.HAE) if obj.HAE is not None else 0.0,
    )


def _sarpy_rowcol(obj: Any) -> Optional[RowCol]:
    """Convert sarpy RowColType to RowCol."""
    if obj is None:
        return None
    return RowCol(
        row=float(obj.Row) if obj.Row is not None else 0.0,
        col=float(obj.Col) if obj.Col is not None else 0.0,
    )


def _sarpy_poly1d(obj: Any) -> Optional[Poly1D]:
    """Convert sarpy Poly1DType to Poly1D."""
    if obj is None:
        return None
    coefs = _safe_get(obj, 'Coefs')
    if coefs is None:
        return None
    return Poly1D(coefs=np.array(coefs))


def _sarpy_poly2d(obj: Any) -> Optional[Poly2D]:
    """Convert sarpy Poly2DType to Poly2D."""
    if obj is None:
        return None
    coefs = _safe_get(obj, 'Coefs')
    if coefs is None:
        return None
    return Poly2D(coefs=np.array(coefs))


def _sarpy_xyzpoly(obj: Any) -> Optional[XYZPoly]:
    """Convert sarpy XYZPolyType to XYZPoly."""
    if obj is None:
        return None
    return XYZPoly(
        x=_sarpy_poly1d(_safe_get(obj, 'X')),
        y=_sarpy_poly1d(_safe_get(obj, 'Y')),
        z=_sarpy_poly1d(_safe_get(obj, 'Z')),
    )


# ===================================================================
# Section extractors — sarkit (XML)
# ===================================================================

def _extract_collection_info_xml(
    xml: ET.Element,
) -> Optional[SICDCollectionInfo]:
    """Extract CollectionInfo from SICD XML."""
    ci = xml.find('{*}CollectionInfo')
    if ci is None:
        return None

    radar_mode = None
    rm = ci.find('{*}RadarMode')
    if rm is not None:
        radar_mode = SICDRadarMode(
            mode_type=_xml_str(rm, '{*}ModeType'),
            mode_id=_xml_str(rm, '{*}ModeID'),
        )

    country_codes = None
    cc_elems = ci.findall('{*}CountryCode')
    if cc_elems:
        country_codes = [cc.text for cc in cc_elems if cc.text]

    return SICDCollectionInfo(
        collector_name=_xml_str(ci, '{*}CollectorName'),
        illuminator_name=_xml_str(ci, '{*}IlluminatorName'),
        core_name=_xml_str(ci, '{*}CoreName'),
        collect_type=_xml_str(ci, '{*}CollectType'),
        radar_mode=radar_mode,
        classification=_xml_str(ci, '{*}Classification'),
        country_codes=country_codes,
    )


def _extract_image_creation_xml(
    xml: ET.Element,
) -> Optional[SICDImageCreation]:
    """Extract ImageCreation from SICD XML."""
    ic = xml.find('{*}ImageCreation')
    if ic is None:
        return None
    return SICDImageCreation(
        application=_xml_str(ic, '{*}Application'),
        date_time=_xml_str(ic, '{*}DateTime'),
        site=_xml_str(ic, '{*}Site'),
        profile=_xml_str(ic, '{*}Profile'),
    )


def _extract_image_data_xml(xml: ET.Element) -> Optional[SICDImageData]:
    """Extract ImageData from SICD XML."""
    idata = xml.find('{*}ImageData')
    if idata is None:
        return None

    full_image = None
    fi = idata.find('{*}FullImage')
    if fi is not None:
        full_image = SICDFullImage(
            num_rows=_xml_int(fi, '{*}NumRows') or 0,
            num_cols=_xml_int(fi, '{*}NumCols') or 0,
        )

    return SICDImageData(
        pixel_type=_xml_str(idata, '{*}PixelType'),
        num_rows=_xml_int(idata, '{*}NumRows') or 0,
        num_cols=_xml_int(idata, '{*}NumCols') or 0,
        first_row=_xml_int(idata, '{*}FirstRow') or 0,
        first_col=_xml_int(idata, '{*}FirstCol') or 0,
        full_image=full_image,
        scp_pixel=_xml_rowcol(idata, '{*}SCPPixel'),
    )


def _extract_geo_data_xml(xml: ET.Element) -> Optional[SICDGeoData]:
    """Extract GeoData from SICD XML."""
    geo = xml.find('{*}GeoData')
    if geo is None:
        return None

    scp = None
    scp_elem = geo.find('{*}SCP')
    if scp_elem is not None:
        scp = SICDSCP(
            ecf=_xml_xyz(scp_elem, '{*}ECF'),
            llh=_xml_latlonhae(scp_elem, '{*}LLH'),
        )

    corners = None
    ic = geo.find('{*}ImageCorners')
    if ic is not None:
        corners = []
        for label in ('FRFC', 'FRLC', 'LRLC', 'LRFC'):
            c = ic.find('{*}' + label)
            if c is not None:
                lat = _xml_float(c, '{*}Lat')
                lon = _xml_float(c, '{*}Lon')
                if lat is not None:
                    corners.append(LatLon(lat=lat, lon=lon or 0.0))

    return SICDGeoData(
        earth_model=_xml_str(geo, '{*}EarthModel') or 'WGS_84',
        scp=scp,
        image_corners=corners if corners else None,
    )


def _extract_dir_param_xml(
    elem: Optional[ET.Element],
) -> Optional[SICDDirParam]:
    """Extract Grid DirParam (Row or Col) from XML element."""
    if elem is None:
        return None

    wgt_type = None
    wt = elem.find('{*}WgtType')
    if wt is not None:
        params = None
        wt_params = wt.findall('{*}Parameter')
        if wt_params:
            params = {
                p.get('name', ''): (p.text or '')
                for p in wt_params
            }
        wgt_type = SICDWgtType(
            window_name=_xml_str(wt, '{*}WindowName'),
            parameters=params,
        )

    return SICDDirParam(
        uvect_ecf=_xml_xyz(elem, '{*}UVectECF'),
        ss=_xml_float(elem, '{*}SS'),
        imp_resp_wid=_xml_float(elem, '{*}ImpRespWid'),
        sgn=_xml_int(elem, '{*}Sgn'),
        imp_resp_bw=_xml_float(elem, '{*}ImpRespBW'),
        k_ctr=_xml_float(elem, '{*}KCtr'),
        delta_k1=_xml_float(elem, '{*}DeltaK1'),
        delta_k2=_xml_float(elem, '{*}DeltaK2'),
        delta_k_coa_poly=_xml_poly2d_at(elem, '{*}DeltaKCOAPoly'),
        wgt_type=wgt_type,
    )


def _extract_grid_xml(xml: ET.Element) -> Optional[SICDGrid]:
    """Extract Grid from SICD XML."""
    grid = xml.find('{*}Grid')
    if grid is None:
        return None
    return SICDGrid(
        image_plane=_xml_str(grid, '{*}ImagePlane'),
        type=_xml_str(grid, '{*}Type'),
        row=_extract_dir_param_xml(grid.find('{*}Row')),
        col=_extract_dir_param_xml(grid.find('{*}Col')),
        time_coa_poly=_xml_poly2d_at(grid, '{*}TimeCOAPoly'),
    )


def _extract_timeline_xml(xml: ET.Element) -> Optional[SICDTimeline]:
    """Extract Timeline from SICD XML."""
    tl = xml.find('{*}Timeline')
    if tl is None:
        return None

    ipp_sets = None
    ipp_elem = tl.find('{*}IPP')
    if ipp_elem is not None:
        ipp_sets = []
        for s in ipp_elem.findall('{*}Set'):
            ipp_sets.append(SICDIPPSet(
                t_start=_xml_float(s, '{*}TStart') or 0.0,
                t_end=_xml_float(s, '{*}TEnd') or 0.0,
                ipp_start=_xml_int(s, '{*}IPPStart') or 0,
                ipp_end=_xml_int(s, '{*}IPPEnd') or 0,
                ipp_poly=_xml_poly1d_at(s, '{*}IPPPoly'),
                index=int(s.get('index', '0')),
            ))

    return SICDTimeline(
        collect_start=_xml_str(tl, '{*}CollectStart'),
        collect_duration=_xml_float(tl, '{*}CollectDuration'),
        ipp=ipp_sets if ipp_sets else None,
    )


def _extract_position_xml(xml: ET.Element) -> Optional[SICDPosition]:
    """Extract Position from SICD XML."""
    pos = xml.find('{*}Position')
    if pos is None:
        return None
    return SICDPosition(
        arp_poly=_xml_xyzpoly(pos, '{*}ARPPoly'),
        grp_poly=_xml_xyzpoly(pos, '{*}GRPPoly'),
        tx_apc_poly=_xml_xyzpoly(pos, '{*}TxAPCPoly'),
    )


def _extract_radar_collection_xml(
    xml: ET.Element,
) -> Optional[SICDRadarCollection]:
    """Extract RadarCollection from SICD XML."""
    rc = xml.find('{*}RadarCollection')
    if rc is None:
        return None

    tx_freq = None
    tf = rc.find('{*}TxFrequency')
    if tf is not None:
        tx_freq = SICDTxFrequency(
            min=_xml_float(tf, '{*}Min'),
            max=_xml_float(tf, '{*}Max'),
        )

    waveforms = None
    wf_list = rc.findall('{*}Waveform/{*}WFParameters')
    if wf_list:
        waveforms = []
        for wf in wf_list:
            waveforms.append(SICDWaveformParams(
                tx_pulse_length=_xml_float(wf, '{*}TxPulseLength'),
                tx_rf_bandwidth=_xml_float(wf, '{*}TxRFBandwidth'),
                tx_freq_start=_xml_float(wf, '{*}TxFreqStart'),
                tx_fm_rate=_xml_float(wf, '{*}TxFMRate'),
                rcv_window_length=_xml_float(wf, '{*}RcvWindowLength'),
                adc_sample_rate=_xml_float(wf, '{*}ADCSampleRate'),
                rcv_if_bandwidth=_xml_float(wf, '{*}RcvIFBandwidth'),
                rcv_freq_start=_xml_float(wf, '{*}RcvFreqStart'),
                rcv_demod_type=_xml_str(wf, '{*}RcvDemodType'),
                rcv_fm_rate=_xml_float(wf, '{*}RcvFMRate'),
                index=int(wf.get('index', '0')),
            ))

    rcv_channels = None
    ch_list = rc.findall('{*}RcvChannels/{*}ChanParameters')
    if ch_list:
        rcv_channels = []
        for ch in ch_list:
            rcv_channels.append(SICDRcvChannel(
                tx_rcv_polarization=_xml_str(ch, '{*}TxRcvPolarization'),
                rcv_ape_index=_xml_int(ch, '{*}RcvAPCIndex'),
                index=int(ch.get('index', '0')),
            ))

    area = None
    area_elem = rc.find('{*}Area')
    if area_elem is not None:
        corner_pts = None
        acp_elems = area_elem.findall('{*}Corner/{*}ACP')
        if acp_elems:
            corner_pts = []
            for acp in acp_elems:
                corner_pts.append(SICDAreaCorner(
                    lat=_xml_float(acp, '{*}Lat') or 0.0,
                    lon=_xml_float(acp, '{*}Lon') or 0.0,
                    hae=_xml_float(acp, '{*}HAE') or 0.0,
                    index=int(acp.get('index', '0')),
                ))
        area = SICDArea(corner_points=corner_pts)

    return SICDRadarCollection(
        tx_frequency=tx_freq,
        ref_freq_index=_xml_int(rc, '{*}RefFreqIndex'),
        waveform=waveforms,
        tx_polarization=_xml_str(rc, '{*}TxPolarization'),
        rcv_channels=rcv_channels,
        area=area,
    )


def _extract_image_formation_xml(
    xml: ET.Element,
) -> Optional[SICDImageFormation]:
    """Extract ImageFormation from SICD XML."""
    imf = xml.find('{*}ImageFormation')
    if imf is None:
        return None

    rcv_chan_proc = None
    rcp = imf.find('{*}RcvChanProc')
    if rcp is not None:
        chan_indices = None
        ci_elems = rcp.findall('{*}ChanIndex')
        if ci_elems:
            chan_indices = [int(c.text) for c in ci_elems if c.text]
        rcv_chan_proc = SICDRcvChanProc(
            num_chan_proc=_xml_int(rcp, '{*}NumChanProc'),
            prf_scale_factor=_xml_float(rcp, '{*}PRFScaleFactor'),
            chan_indices=chan_indices,
        )

    tx_freq_proc = None
    tfp = imf.find('{*}TxFrequencyProc')
    if tfp is not None:
        tx_freq_proc = SICDTxFrequencyProc(
            min_proc=_xml_float(tfp, '{*}MinProc'),
            max_proc=_xml_float(tfp, '{*}MaxProc'),
        )

    processing = None
    proc_elems = imf.findall('{*}Processing')
    if proc_elems:
        processing = []
        for p in proc_elems:
            params = None
            p_params = p.findall('{*}Parameter')
            if p_params:
                params = {
                    pp.get('name', ''): (pp.text or '')
                    for pp in p_params
                }
            processing.append(SICDProcessingStep(
                type=_xml_str(p, '{*}Type'),
                applied=_xml_bool(p, '{*}Applied'),
                parameters=params,
            ))

    return SICDImageFormation(
        rcv_chan_proc=rcv_chan_proc,
        image_form_algo=_xml_str(imf, '{*}ImageFormAlgo'),
        t_start_proc=_xml_float(imf, '{*}TStartProc'),
        t_end_proc=_xml_float(imf, '{*}TEndProc'),
        tx_frequency_proc=tx_freq_proc,
        seg_id=_xml_str(imf, '{*}SegmentIdentifier'),
        image_beam_comp=_xml_str(imf, '{*}ImageBeamComp'),
        az_autofocus=_xml_str(imf, '{*}AzAutofocus'),
        rg_autofocus=_xml_str(imf, '{*}RgAutofocus'),
        processing=processing,
        polarization_hv_angle_poly=_xml_poly1d_at(
            imf, '{*}PolarizationHVAnglePoly'
        ),
    )


def _extract_scpcoa_xml(xml: ET.Element) -> Optional[SICDSCPCOA]:
    """Extract SCPCOA from SICD XML."""
    sc = xml.find('{*}SCPCOA')
    if sc is None:
        return None
    return SICDSCPCOA(
        scp_time=_xml_float(sc, '{*}SCPTime'),
        arp_pos=_xml_xyz(sc, '{*}ARPPos'),
        arp_vel=_xml_xyz(sc, '{*}ARPVel'),
        arp_acc=_xml_xyz(sc, '{*}ARPAcc'),
        side_of_track=_xml_str(sc, '{*}SideOfTrack'),
        slant_range=_xml_float(sc, '{*}SlantRange'),
        ground_range=_xml_float(sc, '{*}GroundRange'),
        doppler_cone_ang=_xml_float(sc, '{*}DopplerConeAng'),
        graze_ang=_xml_float(sc, '{*}GrazeAng'),
        incidence_ang=_xml_float(sc, '{*}IncidenceAng'),
        twist_ang=_xml_float(sc, '{*}TwistAng'),
        slope_ang=_xml_float(sc, '{*}SlopeAng'),
        azim_ang=_xml_float(sc, '{*}AzimAng'),
        layover_ang=_xml_float(sc, '{*}LayoverAng'),
    )


def _extract_radiometric_xml(
    xml: ET.Element,
) -> Optional[SICDRadiometric]:
    """Extract Radiometric from SICD XML."""
    rad = xml.find('{*}Radiometric')
    if rad is None:
        return None

    noise_level = None
    nl = rad.find('{*}NoiseLevel')
    if nl is not None:
        noise_level = SICDNoiseLevel(
            noise_level_type=_xml_str(nl, '{*}NoiseLevelType'),
            noise_poly=_xml_poly2d_at(nl, '{*}NoisePoly'),
        )

    return SICDRadiometric(
        noise_level=noise_level,
        rcs_sf_poly=_xml_poly2d_at(rad, '{*}RCSSFPoly'),
        sigma_zero_sf_poly=_xml_poly2d_at(rad, '{*}SigmaZeroSFPoly'),
        beta_zero_sf_poly=_xml_poly2d_at(rad, '{*}BetaZeroSFPoly'),
        gamma_zero_sf_poly=_xml_poly2d_at(rad, '{*}GammaZeroSFPoly'),
    )


def _extract_ant_param_xml(
    elem: Optional[ET.Element],
) -> Optional[SICDAntParam]:
    """Extract single antenna parameter set from XML."""
    if elem is None:
        return None

    eb = None
    eb_elem = elem.find('{*}EB')
    if eb_elem is not None:
        eb = SICDEBType(
            dcx_poly=_xml_poly1d_at(eb_elem, '{*}DCXPoly'),
            dcy_poly=_xml_poly1d_at(eb_elem, '{*}DCYPoly'),
        )

    array_gp = None
    arr = elem.find('{*}Array')
    if arr is not None:
        array_gp = SICDGainPhasePoly(
            gain=_xml_poly2d_at(arr, '{*}GainPoly'),
            phase=_xml_poly2d_at(arr, '{*}PhasePoly'),
        )

    elem_gp = None
    el = elem.find('{*}Elem')
    if el is not None:
        elem_gp = SICDGainPhasePoly(
            gain=_xml_poly2d_at(el, '{*}GainPoly'),
            phase=_xml_poly2d_at(el, '{*}PhasePoly'),
        )

    return SICDAntParam(
        x_axis_poly=_xml_xyzpoly(elem, '{*}XAxisPoly'),
        y_axis_poly=_xml_xyzpoly(elem, '{*}YAxisPoly'),
        freq_zero=_xml_float(elem, '{*}FreqZero'),
        eb=eb,
        array=array_gp,
        elem=elem_gp,
        gain_bs_poly=_xml_poly1d_at(elem, '{*}GainBSPoly'),
        eb_freq_shift=_xml_bool(elem, '{*}EBFreqShift'),
        ml_freq_dilation=_xml_bool(elem, '{*}MLFreqDilation'),
    )


def _extract_antenna_xml(xml: ET.Element) -> Optional[SICDAntenna]:
    """Extract Antenna from SICD XML."""
    ant = xml.find('{*}Antenna')
    if ant is None:
        return None
    return SICDAntenna(
        tx=_extract_ant_param_xml(ant.find('{*}Tx')),
        rcv=_extract_ant_param_xml(ant.find('{*}Rcv')),
        two_way=_extract_ant_param_xml(ant.find('{*}TwoWay')),
    )


def _extract_error_decorr_xml(
    elem: Optional[ET.Element],
) -> Optional[SICDErrorDecorrFunc]:
    """Extract ErrorDecorrFunc from XML element."""
    if elem is None:
        return None
    return SICDErrorDecorrFunc(
        corr_coef_zero=_xml_float(elem, '{*}CorrCoefZero'),
        decorr_rate=_xml_float(elem, '{*}DecorrRate'),
    )


def _extract_error_statistics_xml(
    xml: ET.Element,
) -> Optional[SICDErrorStatistics]:
    """Extract ErrorStatistics from SICD XML."""
    err = xml.find('{*}ErrorStatistics')
    if err is None:
        return None

    composite_scp = None
    cscp = err.find('{*}CompositeSCP')
    if cscp is not None:
        composite_scp = SICDCompositeSCPError(
            rg=_xml_float(cscp, '{*}Rg'),
            az=_xml_float(cscp, '{*}Az'),
            rg_az=_xml_float(cscp, '{*}RgAz'),
        )

    monostatic = None
    mono = err.find('{*}Components/{*}PosVelErr')
    if mono is None:
        mono = err.find('{*}Monostatic')
    if mono is not None:
        corr_coefs = None
        cc = mono.find('{*}CorrCoefs')
        if cc is not None:
            corr_coefs = SICDCorrCoefs(
                p1p2=_xml_float(cc, '{*}P1P2'),
                p1p3=_xml_float(cc, '{*}P1P3'),
                p1v1=_xml_float(cc, '{*}P1V1'),
                p1v2=_xml_float(cc, '{*}P1V2'),
                p1v3=_xml_float(cc, '{*}P1V3'),
                p2p3=_xml_float(cc, '{*}P2P3'),
                p2v1=_xml_float(cc, '{*}P2V1'),
                p2v2=_xml_float(cc, '{*}P2V2'),
                p2v3=_xml_float(cc, '{*}P2V3'),
                p3v1=_xml_float(cc, '{*}P3V1'),
                p3v2=_xml_float(cc, '{*}P3V2'),
                p3v3=_xml_float(cc, '{*}P3V3'),
                v1v2=_xml_float(cc, '{*}V1V2'),
                v1v3=_xml_float(cc, '{*}V1V3'),
                v2v3=_xml_float(cc, '{*}V2V3'),
            )
        monostatic = SICDPosVelErr(
            frame=_xml_str(mono, '{*}Frame'),
            p1=_xml_float(mono, '{*}P1'),
            p2=_xml_float(mono, '{*}P2'),
            p3=_xml_float(mono, '{*}P3'),
            v1=_xml_float(mono, '{*}V1'),
            v2=_xml_float(mono, '{*}V2'),
            v3=_xml_float(mono, '{*}V3'),
            corr_coefs=corr_coefs,
            position_decorr=_extract_error_decorr_xml(
                mono.find('{*}PositionDecorr')
            ),
        )

    radar_sensor = None
    rs = err.find('{*}RadarSensor')
    if rs is not None:
        radar_sensor = SICDRadarSensorError(
            range_bias=_xml_float(rs, '{*}RangeBias'),
            clock_freq_sf=_xml_float(rs, '{*}ClockFreqSF'),
            transmit_freq_sf=_xml_float(rs, '{*}TransmitFreqSF'),
            range_bias_decorr=_extract_error_decorr_xml(
                rs.find('{*}RangeBiasDecorr')
            ),
        )

    return SICDErrorStatistics(
        composite_scp=composite_scp,
        monostatic=monostatic,
        radar_sensor=radar_sensor,
    )


def _extract_match_info_xml(xml: ET.Element) -> Optional[SICDMatchInfo]:
    """Extract MatchInfo from SICD XML."""
    mi = xml.find('{*}MatchInfo')
    if mi is None:
        return None

    match_types = []
    for mt_elem in mi.findall('{*}MatchType'):
        collections = None
        mc_elems = mt_elem.findall('{*}MatchCollection')
        if mc_elems:
            collections = []
            for mc in mc_elems:
                params = None
                p_elems = mc.findall('{*}Parameter')
                if p_elems:
                    params = {
                        p.get('name', ''): (p.text or '')
                        for p in p_elems
                    }
                collections.append(SICDMatchCollection(
                    core_name=_xml_str(mc, '{*}CoreName'),
                    match_index=_xml_int(mc, '{*}MatchIndex'),
                    parameters=params,
                ))
        match_types.append(SICDMatchType(
            type_id=_xml_str(mt_elem, '{*}TypeID'),
            current_index=_xml_int(mt_elem, '{*}CurrentIndex'),
            match_collections=collections,
        ))

    return SICDMatchInfo(
        match_types=match_types if match_types else None,
    )


def _extract_rg_az_comp_xml(xml: ET.Element) -> Optional[SICDRgAzComp]:
    """Extract RgAzComp from SICD XML."""
    rac = xml.find('{*}RgAzComp')
    if rac is None:
        return None
    return SICDRgAzComp(
        az_sf=_xml_float(rac, '{*}AzSF'),
        kaz_poly=_xml_poly1d_at(rac, '{*}KazPoly'),
    )


def _extract_pfa_xml(xml: ET.Element) -> Optional[SICDPFA]:
    """Extract PFA from SICD XML."""
    pfa = xml.find('{*}PFA')
    if pfa is None:
        return None

    st_deskew = None
    std = pfa.find('{*}STDeskew')
    if std is not None:
        st_deskew = SICDSTDeskew(
            applied=_xml_bool(std, '{*}Applied'),
            st_ds_phase_poly=_xml_poly2d_at(std, '{*}STDSPhasePoly'),
        )

    return SICDPFA(
        fpn=_xml_xyz(pfa, '{*}FPN'),
        ipn=_xml_xyz(pfa, '{*}IPN'),
        polar_ang_ref_time=_xml_float(pfa, '{*}PolarAngRefTime'),
        polar_ang_poly=_xml_poly1d_at(pfa, '{*}PolarAngPoly'),
        spatial_freq_sf_poly=_xml_poly1d_at(pfa, '{*}SpatialFreqSFPoly'),
        krg1=_xml_float(pfa, '{*}Krg1'),
        krg2=_xml_float(pfa, '{*}Krg2'),
        kaz1=_xml_float(pfa, '{*}Kaz1'),
        kaz2=_xml_float(pfa, '{*}Kaz2'),
        st_deskew=st_deskew,
    )


def _extract_rma_xml(xml: ET.Element) -> Optional[SICDRMA]:
    """Extract RMA from SICD XML."""
    rma = xml.find('{*}RMA')
    if rma is None:
        return None

    rm_ref = None
    rmr = rma.find('{*}RMRefTime') or rma.find('{*}RMCR')
    if rmr is not None:
        rm_ref = SICDRMRef(
            pos_ref=_xml_xyz(rmr, '{*}PosRef'),
            vel_ref=_xml_xyz(rmr, '{*}VelRef'),
            dop_cone_ang_ref=_xml_float(rmr, '{*}DopConeAngRef'),
        )

    inca = None
    inca_elem = rma.find('{*}INCA')
    if inca_elem is not None:
        inca = SICDINCA(
            time_ca_poly=_xml_poly1d_at(inca_elem, '{*}TimeCAPoly'),
            r_ca_scp=_xml_float(inca_elem, '{*}R_CA_SCP'),
            freq_zero=_xml_float(inca_elem, '{*}FreqZero'),
            d_rate_sf_poly=_xml_poly2d_at(inca_elem, '{*}DRateSFPoly'),
            dop_centroid_poly=_xml_poly2d_at(
                inca_elem, '{*}DopCentroidPoly'
            ),
            dop_centroid_coa=_xml_bool(inca_elem, '{*}DopCentroidCOA'),
        )

    return SICDRMA(
        rm_ref=rm_ref,
        inca=inca,
        image_type=_xml_str(rma, '{*}ImageType'),
    )


# ===================================================================
# Section extractors — sarpy
# ===================================================================

def _extract_collection_info_sarpy(
    sm: Any,
) -> Optional[SICDCollectionInfo]:
    """Extract CollectionInfo from sarpy SICDType."""
    ci = _safe_get(sm, 'CollectionInfo')
    if ci is None:
        return None

    radar_mode = None
    rm = _safe_get(ci, 'RadarMode')
    if rm is not None:
        radar_mode = SICDRadarMode(
            mode_type=_safe_get(rm, 'ModeType'),
            mode_id=_safe_get(rm, 'ModeID'),
        )

    return SICDCollectionInfo(
        collector_name=_safe_get(ci, 'CollectorName'),
        illuminator_name=_safe_get(ci, 'IlluminatorName'),
        core_name=_safe_get(ci, 'CoreName'),
        collect_type=_safe_get(ci, 'CollectType'),
        radar_mode=radar_mode,
        classification=_safe_get(ci, 'Classification'),
        country_codes=_safe_get(ci, 'CountryCodes'),
    )


def _extract_image_creation_sarpy(
    sm: Any,
) -> Optional[SICDImageCreation]:
    """Extract ImageCreation from sarpy SICDType."""
    ic = _safe_get(sm, 'ImageCreation')
    if ic is None:
        return None
    dt = _safe_get(ic, 'DateTime')
    return SICDImageCreation(
        application=_safe_get(ic, 'Application'),
        date_time=str(dt) if dt is not None else None,
        site=_safe_get(ic, 'Site'),
        profile=_safe_get(ic, 'Profile'),
    )


def _extract_image_data_sarpy(sm: Any) -> Optional[SICDImageData]:
    """Extract ImageData from sarpy SICDType."""
    idata = _safe_get(sm, 'ImageData')
    if idata is None:
        return None

    full_image = None
    fi = _safe_get(idata, 'FullImage')
    if fi is not None:
        full_image = SICDFullImage(
            num_rows=_safe_get(fi, 'NumRows') or 0,
            num_cols=_safe_get(fi, 'NumCols') or 0,
        )

    return SICDImageData(
        pixel_type=_safe_get(idata, 'PixelType'),
        num_rows=_safe_get(idata, 'NumRows') or 0,
        num_cols=_safe_get(idata, 'NumCols') or 0,
        first_row=_safe_get(idata, 'FirstRow') or 0,
        first_col=_safe_get(idata, 'FirstCol') or 0,
        full_image=full_image,
        scp_pixel=_sarpy_rowcol(_safe_get(idata, 'SCPPixel')),
        amp_table=(
            np.array(_safe_get(idata, 'AmpTable'))
            if _safe_get(idata, 'AmpTable') is not None
            else None
        ),
    )


def _extract_geo_data_sarpy(sm: Any) -> Optional[SICDGeoData]:
    """Extract GeoData from sarpy SICDType."""
    geo = _safe_get(sm, 'GeoData')
    if geo is None:
        return None

    scp = None
    scp_obj = _safe_get(geo, 'SCP')
    if scp_obj is not None:
        scp = SICDSCP(
            ecf=_sarpy_xyz(_safe_get(scp_obj, 'ECF')),
            llh=_sarpy_latlonhae(_safe_get(scp_obj, 'LLH')),
        )

    corners = None
    ic = _safe_get(geo, 'ImageCorners')
    if ic is not None:
        corners = []
        for label in ('FRFC', 'FRLC', 'LRLC', 'LRFC'):
            c = getattr(ic, label, None)
            if c is not None:
                corners.append(_sarpy_latlon(c))

    return SICDGeoData(
        earth_model=_safe_get(geo, 'EarthModel') or 'WGS_84',
        scp=scp,
        image_corners=corners if corners else None,
    )


def _extract_dir_param_sarpy(dp: Any) -> Optional[SICDDirParam]:
    """Extract DirParam from sarpy DirParamType."""
    if dp is None:
        return None

    wgt_type = None
    wt = _safe_get(dp, 'WgtType')
    if wt is not None:
        params = None
        wt_params = _safe_get(wt, 'Parameters')
        if wt_params:
            params = {
                str(k): str(v) for k, v in wt_params.items()
            } if hasattr(wt_params, 'items') else None
        wgt_type = SICDWgtType(
            window_name=_safe_get(wt, 'WindowName'),
            parameters=params,
        )

    return SICDDirParam(
        uvect_ecf=_sarpy_xyz(_safe_get(dp, 'UVectECF')),
        ss=_safe_get(dp, 'SS'),
        imp_resp_wid=_safe_get(dp, 'ImpRespWid'),
        sgn=_safe_get(dp, 'Sgn'),
        imp_resp_bw=_safe_get(dp, 'ImpRespBW'),
        k_ctr=_safe_get(dp, 'KCtr'),
        delta_k1=_safe_get(dp, 'DeltaK1'),
        delta_k2=_safe_get(dp, 'DeltaK2'),
        delta_k_coa_poly=_sarpy_poly2d(_safe_get(dp, 'DeltaKCOAPoly')),
        wgt_type=wgt_type,
    )


def _extract_grid_sarpy(sm: Any) -> Optional[SICDGrid]:
    """Extract Grid from sarpy SICDType."""
    grid = _safe_get(sm, 'Grid')
    if grid is None:
        return None
    return SICDGrid(
        image_plane=_safe_get(grid, 'ImagePlane'),
        type=_safe_get(grid, 'Type'),
        row=_extract_dir_param_sarpy(_safe_get(grid, 'Row')),
        col=_extract_dir_param_sarpy(_safe_get(grid, 'Col')),
        time_coa_poly=_sarpy_poly2d(_safe_get(grid, 'TimeCOAPoly')),
    )


def _extract_timeline_sarpy(sm: Any) -> Optional[SICDTimeline]:
    """Extract Timeline from sarpy SICDType."""
    tl = _safe_get(sm, 'Timeline')
    if tl is None:
        return None

    cs = _safe_get(tl, 'CollectStart')

    ipp_sets = None
    ipp = _safe_get(tl, 'IPP')
    if ipp is not None:
        sets = _safe_get(ipp, 'Sets') or []
        if sets:
            ipp_sets = []
            for i, s in enumerate(sets):
                ipp_sets.append(SICDIPPSet(
                    t_start=_safe_get(s, 'TStart') or 0.0,
                    t_end=_safe_get(s, 'TEnd') or 0.0,
                    ipp_start=_safe_get(s, 'IPPStart') or 0,
                    ipp_end=_safe_get(s, 'IPPEnd') or 0,
                    ipp_poly=_sarpy_poly1d(_safe_get(s, 'IPPPoly')),
                    index=_safe_get(s, 'index') or i,
                ))

    return SICDTimeline(
        collect_start=str(cs) if cs is not None else None,
        collect_duration=_safe_get(tl, 'CollectDuration'),
        ipp=ipp_sets,
    )


def _extract_position_sarpy(sm: Any) -> Optional[SICDPosition]:
    """Extract Position from sarpy SICDType."""
    pos = _safe_get(sm, 'Position')
    if pos is None:
        return None
    return SICDPosition(
        arp_poly=_sarpy_xyzpoly(_safe_get(pos, 'ARPPoly')),
        grp_poly=_sarpy_xyzpoly(_safe_get(pos, 'GRPPoly')),
        tx_apc_poly=_sarpy_xyzpoly(_safe_get(pos, 'TxAPCPoly')),
    )


def _extract_radar_collection_sarpy(
    sm: Any,
) -> Optional[SICDRadarCollection]:
    """Extract RadarCollection from sarpy SICDType."""
    rc = _safe_get(sm, 'RadarCollection')
    if rc is None:
        return None

    tx_freq = None
    tf = _safe_get(rc, 'TxFrequency')
    if tf is not None:
        tx_freq = SICDTxFrequency(
            min=_safe_get(tf, 'Min'),
            max=_safe_get(tf, 'Max'),
        )

    waveforms = None
    wf_list = _safe_get(rc, 'Waveform')
    if wf_list:
        waveforms = []
        for i, wf in enumerate(wf_list):
            waveforms.append(SICDWaveformParams(
                tx_pulse_length=_safe_get(wf, 'TxPulseLength'),
                tx_rf_bandwidth=_safe_get(wf, 'TxRFBandwidth'),
                tx_freq_start=_safe_get(wf, 'TxFreqStart'),
                tx_fm_rate=_safe_get(wf, 'TxFMRate'),
                rcv_window_length=_safe_get(wf, 'RcvWindowLength'),
                adc_sample_rate=_safe_get(wf, 'ADCSampleRate'),
                rcv_if_bandwidth=_safe_get(wf, 'RcvIFBandwidth'),
                rcv_freq_start=_safe_get(wf, 'RcvFreqStart'),
                rcv_demod_type=_safe_get(wf, 'RcvDemodType'),
                rcv_fm_rate=_safe_get(wf, 'RcvFMRate'),
                index=_safe_get(wf, 'index') or i,
            ))

    rcv_channels = None
    ch_list = _safe_get(rc, 'RcvChannels')
    if ch_list:
        rcv_channels = []
        for i, ch in enumerate(ch_list):
            rcv_channels.append(SICDRcvChannel(
                tx_rcv_polarization=_safe_get(ch, 'TxRcvPolarization'),
                rcv_ape_index=_safe_get(ch, 'RcvAPCIndex'),
                index=_safe_get(ch, 'index') or i,
            ))

    return SICDRadarCollection(
        tx_frequency=tx_freq,
        ref_freq_index=_safe_get(rc, 'RefFreqIndex'),
        waveform=waveforms,
        tx_polarization=_safe_get(rc, 'TxPolarization'),
        rcv_channels=rcv_channels,
    )


def _extract_image_formation_sarpy(
    sm: Any,
) -> Optional[SICDImageFormation]:
    """Extract ImageFormation from sarpy SICDType."""
    imf = _safe_get(sm, 'ImageFormation')
    if imf is None:
        return None

    rcv_chan_proc = None
    rcp = _safe_get(imf, 'RcvChanProc')
    if rcp is not None:
        rcv_chan_proc = SICDRcvChanProc(
            num_chan_proc=_safe_get(rcp, 'NumChanProc'),
            prf_scale_factor=_safe_get(rcp, 'PRFScaleFactor'),
            chan_indices=_safe_get(rcp, 'ChanIndices'),
        )

    tx_freq_proc = None
    tfp = _safe_get(imf, 'TxFrequencyProc')
    if tfp is not None:
        tx_freq_proc = SICDTxFrequencyProc(
            min_proc=_safe_get(tfp, 'MinProc'),
            max_proc=_safe_get(tfp, 'MaxProc'),
        )

    return SICDImageFormation(
        rcv_chan_proc=rcv_chan_proc,
        image_form_algo=_safe_get(imf, 'ImageFormAlgo'),
        t_start_proc=_safe_get(imf, 'TStartProc'),
        t_end_proc=_safe_get(imf, 'TEndProc'),
        tx_frequency_proc=tx_freq_proc,
        seg_id=_safe_get(imf, 'SegmentIdentifier'),
        image_beam_comp=_safe_get(imf, 'ImageBeamComp'),
        az_autofocus=_safe_get(imf, 'AzAutofocus'),
        rg_autofocus=_safe_get(imf, 'RgAutofocus'),
    )


def _extract_scpcoa_sarpy(sm: Any) -> Optional[SICDSCPCOA]:
    """Extract SCPCOA from sarpy SICDType."""
    sc = _safe_get(sm, 'SCPCOA')
    if sc is None:
        return None
    return SICDSCPCOA(
        scp_time=_safe_get(sc, 'SCPTime'),
        arp_pos=_sarpy_xyz(_safe_get(sc, 'ARPPos')),
        arp_vel=_sarpy_xyz(_safe_get(sc, 'ARPVel')),
        arp_acc=_sarpy_xyz(_safe_get(sc, 'ARPAcc')),
        side_of_track=_safe_get(sc, 'SideOfTrack'),
        slant_range=_safe_get(sc, 'SlantRange'),
        ground_range=_safe_get(sc, 'GroundRange'),
        doppler_cone_ang=_safe_get(sc, 'DopplerConeAng'),
        graze_ang=_safe_get(sc, 'GrazeAng'),
        incidence_ang=_safe_get(sc, 'IncidenceAng'),
        twist_ang=_safe_get(sc, 'TwistAng'),
        slope_ang=_safe_get(sc, 'SlopeAng'),
        azim_ang=_safe_get(sc, 'AzimAng'),
        layover_ang=_safe_get(sc, 'LayoverAng'),
    )


def _extract_radiometric_sarpy(sm: Any) -> Optional[SICDRadiometric]:
    """Extract Radiometric from sarpy SICDType."""
    rad = _safe_get(sm, 'Radiometric')
    if rad is None:
        return None

    noise_level = None
    nl = _safe_get(rad, 'NoiseLevel')
    if nl is not None:
        noise_level = SICDNoiseLevel(
            noise_level_type=_safe_get(nl, 'NoiseLevelType'),
            noise_poly=_sarpy_poly2d(_safe_get(nl, 'NoisePoly')),
        )

    return SICDRadiometric(
        noise_level=noise_level,
        rcs_sf_poly=_sarpy_poly2d(_safe_get(rad, 'RCSSFPoly')),
        sigma_zero_sf_poly=_sarpy_poly2d(
            _safe_get(rad, 'SigmaZeroSFPoly')
        ),
        beta_zero_sf_poly=_sarpy_poly2d(
            _safe_get(rad, 'BetaZeroSFPoly')
        ),
        gamma_zero_sf_poly=_sarpy_poly2d(
            _safe_get(rad, 'GammaZeroSFPoly')
        ),
    )


def _extract_rma_sarpy(sm: Any) -> Optional[SICDRMA]:
    """Extract RMA from sarpy SICDType."""
    rma = _safe_get(sm, 'RMA')
    if rma is None:
        return None

    inca = None
    inca_obj = _safe_get(rma, 'INCA')
    if inca_obj is not None:
        inca = SICDINCA(
            time_ca_poly=_sarpy_poly1d(_safe_get(inca_obj, 'TimeCAPoly')),
            r_ca_scp=_safe_get(inca_obj, 'R_CA_SCP'),
            freq_zero=_safe_get(inca_obj, 'FreqZero'),
            d_rate_sf_poly=_sarpy_poly2d(
                _safe_get(inca_obj, 'DRateSFPoly')
            ),
            dop_centroid_poly=_sarpy_poly2d(
                _safe_get(inca_obj, 'DopCentroidPoly')
            ),
            dop_centroid_coa=_safe_get(inca_obj, 'DopCentroidCOA'),
        )

    return SICDRMA(
        rm_ref=None,
        inca=inca,
        image_type=_safe_get(rma, 'ImageType'),
    )


# ===================================================================
# SICDReader
# ===================================================================

class SICDReader(ImageReader):
    """Read SICD (Sensor Independent Complex Data) format.

    SICD is the NGA standard for complex SAR imagery in NITF containers.
    This reader uses sarkit as the primary backend with sarpy as fallback.

    Parameters
    ----------
    filepath : str or Path
        Path to the SICD file (NITF or other SICD container).

    Attributes
    ----------
    filepath : Path
        Path to the SICD file.
    metadata : SICDMetadata
        Complete typed metadata with all 17 SICD sections.
    backend : str
        Active backend (``'sarkit'`` or ``'sarpy'``).

    Raises
    ------
    ImportError
        If neither sarkit nor sarpy is installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a valid SICD file.

    Examples
    --------
    >>> from grdl.IO.sar import SICDReader
    >>> with SICDReader('image.nitf') as reader:
    ...     chip = reader.read_chip(0, 1000, 0, 1000)
    ...     magnitude = np.abs(chip)

    Notes
    -----
    SICD data is complex-valued (I/Q). Use ``abs()`` for magnitude images.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.backend = require_sar_backend('SICD')
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load SICD metadata using the active backend."""
        if self.backend == 'sarkit':
            self._load_metadata_sarkit()
        else:
            self._load_metadata_sarpy()

    def _load_metadata_sarkit(self) -> None:
        """Load metadata via sarkit — all 17 sections."""
        import sarkit.sicd

        try:
            self._file_handle = open(str(self.filepath), 'rb')
            self._reader = sarkit.sicd.NitfReader(self._file_handle)

            xml = self._reader.metadata.xmltree
            num_rows = int(xml.findtext('{*}ImageData/{*}NumRows'))
            num_cols = int(xml.findtext('{*}ImageData/{*}NumCols'))

            self.metadata = SICDMetadata(
                format='SICD',
                rows=num_rows,
                cols=num_cols,
                dtype='complex64',
                backend='sarkit',
                collection_info=_extract_collection_info_xml(xml),
                image_creation=_extract_image_creation_xml(xml),
                image_data=_extract_image_data_xml(xml),
                geo_data=_extract_geo_data_xml(xml),
                grid=_extract_grid_xml(xml),
                timeline=_extract_timeline_xml(xml),
                position=_extract_position_xml(xml),
                radar_collection=_extract_radar_collection_xml(xml),
                image_formation=_extract_image_formation_xml(xml),
                scpcoa=_extract_scpcoa_xml(xml),
                radiometric=_extract_radiometric_xml(xml),
                antenna=_extract_antenna_xml(xml),
                error_statistics=_extract_error_statistics_xml(xml),
                match_info=_extract_match_info_xml(xml),
                rg_az_comp=_extract_rg_az_comp_xml(xml),
                pfa=_extract_pfa_xml(xml),
                rma=_extract_rma_xml(xml),
            )

            # Store raw XML tree for advanced users
            self._xmltree = xml

        except Exception as e:
            raise ValueError(f"Failed to load SICD metadata: {e}") from e

    def _load_metadata_sarpy(self) -> None:
        """Load metadata via sarpy (fallback) — all 17 sections."""
        from sarpy.io.complex.converter import open_complex

        try:
            self._reader = open_complex(str(self.filepath))
            sm = self._reader.sicd_meta

            self.metadata = SICDMetadata(
                format='SICD',
                rows=sm.ImageData.NumRows,
                cols=sm.ImageData.NumCols,
                dtype='complex64',
                backend='sarpy',
                collection_info=_extract_collection_info_sarpy(sm),
                image_creation=_extract_image_creation_sarpy(sm),
                image_data=_extract_image_data_sarpy(sm),
                geo_data=_extract_geo_data_sarpy(sm),
                grid=_extract_grid_sarpy(sm),
                timeline=_extract_timeline_sarpy(sm),
                position=_extract_position_sarpy(sm),
                radar_collection=_extract_radar_collection_sarpy(sm),
                image_formation=_extract_image_formation_sarpy(sm),
                scpcoa=_extract_scpcoa_sarpy(sm),
                radiometric=_extract_radiometric_sarpy(sm),
                rma=_extract_rma_sarpy(sm),
            )

            self._sarpy_meta = sm

        except Exception as e:
            raise ValueError(f"Failed to load SICD metadata: {e}") from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the SICD image.

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
            Ignored for SICD (single complex band).

        Returns
        -------
        np.ndarray
            Complex-valued image chip with shape ``(rows, cols)``.

        Raises
        ------
        ValueError
            If indices are out of bounds.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata['rows'] or col_end > self.metadata['cols']:
            raise ValueError("End indices exceed image dimensions")

        if self.backend == 'sarkit':
            data, _ = self._reader.read_sub_image(
                row_start, col_start, row_end, col_end,
            )
            return data
        else:
            return self._reader[row_start:row_end, col_start:col_end]

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire SICD image.

        Parameters
        ----------
        bands : Optional[List[int]]
            Ignored for SICD (single complex band).

        Returns
        -------
        np.ndarray
            Full complex-valued image.

        Warnings
        --------
        SICD files can be very large. Consider using ``read_chip()``
        instead.
        """
        if self.backend == 'sarkit':
            return self._reader.read_image()
        else:
            return self._reader[:, :]

    def get_shape(self) -> Tuple[int, int]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, int]
            ``(rows, cols)``.
        """
        return (self.metadata['rows'], self.metadata['cols'])

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            ``complex64`` for SICD data.
        """
        return np.dtype('complex64')

    def close(self) -> None:
        """Close the reader and release resources."""
        if self.backend == 'sarkit':
            if hasattr(self, '_reader') and self._reader is not None:
                self._reader.done()
            if hasattr(self, '_file_handle') and self._file_handle is not None:
                self._file_handle.close()
        else:
            if hasattr(self, '_reader') and self._reader is not None:
                self._reader.close()
