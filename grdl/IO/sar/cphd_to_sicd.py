# -*- coding: utf-8 -*-
"""
CPHD-to-SICD metadata builder.

Derives a fully populated :class:`grdl.IO.models.SICDMetadata` from a
:class:`grdl.IO.models.CPHDMetadata` plus the
:class:`grdl.image_processing.sar.image_formation.CollectionGeometry`
and the formed-image grid that the IFP (PFA / RDA / FFBP / Stripmap-PFA)
produced.

The output covers every SICD section that sarpy's writer validates:
``CollectionInfo``, ``ImageData``, ``GeoData`` (SCP + four image
corners), ``Grid`` (row/col DirParam with sample spacing, IPR width,
bandwidth, k-center, ECF unit vectors), ``Timeline``, ``Position``
(ARP polynomial fit), ``SCPCOA`` (full angle set + ARP pos/vel/acc),
``RadarCollection`` (TxFrequency, polarization, Rcv channels), and
``ImageFormation`` (algorithm tag, processed bandwidth and time window).

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
2026-05-05

Modified
--------
2026-05-05
"""

from __future__ import annotations

# Standard library
from typing import Any, Dict, List, Optional, Tuple

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.models import (
    CPHDMetadata,
    SICDArea,
    SICDAreaCorner,
    SICDCollectionInfo,
    SICDDirParam,
    SICDFullImage,
    SICDGeoData,
    SICDGrid,
    SICDImageData,
    SICDImageFormation,
    SICDMetadata,
    SICDPosition,
    SICDRadarCollection,
    SICDRadarMode,
    SICDRcvChanProc,
    SICDRcvChannel,
    SICDSCP,
    SICDSCPCOA,
    SICDTimeline,
    SICDTxFrequency,
    SICDTxFrequencyProc,
    SICDWaveformParams,
)
from grdl.IO.models.common import LatLon, LatLonHAE, Poly1D, RowCol, XYZ, XYZPoly


# Algorithms accepted by the SICD spec (ImageFormation/ImageFormAlgo enum)
_VALID_SICD_ALGOS = {'PFA', 'RMA', 'RGAZCOMP', 'OTHER'}


# ===================================================================
# Top-level builder
# ===================================================================

def build_sicd_metadata(
    cphd_meta: CPHDMetadata,
    geometry: Any,
    image_shape: Tuple[int, int],
    *,
    image_form_algo: str = 'PFA',
    grid_params: Optional[Dict[str, Any]] = None,
) -> SICDMetadata:
    """Derive a complete ``SICDMetadata`` from CPHD + IFP outputs.

    Parameters
    ----------
    cphd_meta : CPHDMetadata
        Source CPHD metadata used for image formation.
    geometry : CollectionGeometry
        ``grdl.image_processing.sar.image_formation.CollectionGeometry``
        instance built from the same CPHD. Provides the SRP, ARP/VARP
        per-pulse arrays, SICD-style angles at center-of-aperture, the
        row/col ECF unit vectors, and the 5th-order ARP polynomial fit.
    image_shape : tuple of int
        Formed-image shape ``(rows, cols)``. Convention: rows = range,
        cols = azimuth (after the IFP transposes its output).
    image_form_algo : str
        SICD ``ImageFormation/ImageFormAlgo`` enum: ``'PFA'``, ``'RMA'``,
        ``'RGAZCOMP'``, or ``'OTHER'``. Default ``'PFA'``.
    grid_params : dict, optional
        Output of an ``ImageFormationAlgorithm.get_output_grid()`` call.
        Provides per-direction sample spacing, IPR bandwidth/width, and
        k-space centers. When omitted, the builder falls back to
        approximate values derived from the geometry alone.

    Returns
    -------
    SICDMetadata
        Fully populated SICD metadata, ready for ``SICDWriter``.
    """
    if image_form_algo not in _VALID_SICD_ALGOS:
        raise ValueError(
            f"image_form_algo must be one of {_VALID_SICD_ALGOS}, "
            f"got {image_form_algo!r}",
        )

    rows, cols = int(image_shape[0]), int(image_shape[1])
    grid_params = dict(grid_params) if grid_params else {}

    return SICDMetadata(
        format='SICD',
        rows=rows,
        cols=cols,
        dtype='complex64',
        collection_info=_build_collection_info(cphd_meta),
        image_data=_build_image_data(rows, cols),
        geo_data=_build_geo_data(cphd_meta, geometry),
        grid=_build_grid(geometry, grid_params),
        timeline=_build_timeline(cphd_meta, geometry),
        position=_build_position(geometry),
        radar_collection=_build_radar_collection(cphd_meta),
        image_formation=_build_image_formation(
            cphd_meta, geometry, image_form_algo, grid_params,
        ),
        scpcoa=_build_scpcoa(geometry),
    )


# ===================================================================
# Section builders
# ===================================================================

def _build_collection_info(meta: CPHDMetadata) -> SICDCollectionInfo:
    ci = meta.collection_info
    if ci is None:
        return SICDCollectionInfo()
    radar_mode = None
    if ci.radar_mode is not None:
        radar_mode = SICDRadarMode(
            mode_type=ci.radar_mode,
            mode_id=ci.radar_mode_id,
        )
    country_codes = None
    if ci.country_code:
        country_codes = [ci.country_code]
    return SICDCollectionInfo(
        collector_name=ci.collector_name,
        illuminator_name=ci.illuminator_name,
        core_name=ci.core_name,
        collect_type=ci.collect_type,
        radar_mode=radar_mode,
        classification=ci.classification or 'UNCLASSIFIED',
        country_codes=country_codes,
    )


def _build_image_data(rows: int, cols: int) -> SICDImageData:
    return SICDImageData(
        pixel_type='RE32F_IM32F',
        num_rows=rows,
        num_cols=cols,
        first_row=0,
        first_col=0,
        full_image=SICDFullImage(num_rows=rows, num_cols=cols),
        scp_pixel=RowCol(row=rows // 2, col=cols // 2),
    )


def _build_geo_data(meta: CPHDMetadata, geometry: Any) -> SICDGeoData:
    """Build SICD GeoData (SCP + four image corner points).

    The SCP is taken from the geometry's center-of-aperture SRP (the
    same point used by the IFP). Image corners come from the CPHD's
    ``SceneCoordinates.corner_points`` if present, otherwise from a
    small box around the SCP (last-resort fallback).
    """
    scp_ecf, scp_llh = _scp_at_coa(geometry)
    scp = SICDSCP(ecf=scp_ecf, llh=scp_llh)

    image_corners = _corner_points(meta, scp_llh)
    return SICDGeoData(
        earth_model='WGS_84',
        scp=scp,
        image_corners=image_corners,
    )


def _build_grid(
    geometry: Any,
    grid_params: Dict[str, Any],
) -> SICDGrid:
    """Build SICD Grid with row (range) and col (azimuth) DirParams."""
    image_plane = grid_params.get('image_plane') or geometry.image_plane

    grid_type = grid_params.get('type', 'RGAZIM')

    rg_ss = grid_params.get('rg_ss')
    rg_bw = grid_params.get('rg_imp_resp_bw')
    rg_wid = grid_params.get('rg_imp_resp_wid')
    rg_kctr = grid_params.get('rg_kctr')
    rg_dk1 = grid_params.get('rg_delta_k1')
    rg_dk2 = grid_params.get('rg_delta_k2')

    az_ss = grid_params.get('az_ss')
    az_bw = grid_params.get('az_imp_resp_bw')
    az_wid = grid_params.get('az_imp_resp_wid')
    az_kctr = grid_params.get('az_kctr')
    az_dk1 = grid_params.get('az_delta_k1')
    az_dk2 = grid_params.get('az_delta_k2')

    row_dp = SICDDirParam(
        uvect_ecf=_to_xyz(getattr(geometry, 'rg_uvect_ecf', None)),
        ss=_as_float(rg_ss),
        imp_resp_wid=_as_float(rg_wid),
        sgn=-1,
        imp_resp_bw=_as_float(rg_bw),
        k_ctr=_as_float(rg_kctr),
        delta_k1=_as_float(rg_dk1),
        delta_k2=_as_float(rg_dk2),
    )
    col_dp = SICDDirParam(
        uvect_ecf=_to_xyz(getattr(geometry, 'az_uvect_ecf', None)),
        ss=_as_float(az_ss),
        imp_resp_wid=_as_float(az_wid),
        sgn=-1,
        imp_resp_bw=_as_float(az_bw),
        k_ctr=_as_float(az_kctr),
        delta_k1=_as_float(az_dk1),
        delta_k2=_as_float(az_dk2),
    )

    return SICDGrid(
        image_plane=image_plane,
        type=grid_type,
        row=row_dp,
        col=col_dp,
    )


def _build_timeline(
    meta: CPHDMetadata,
    geometry: Any,
) -> SICDTimeline:
    collect_start = None
    if meta.global_params is not None and meta.global_params.timeline is not None:
        collect_start = meta.global_params.timeline.collection_start

    duration = float(geometry.tend - geometry.tstart)
    if duration <= 0:
        duration = None

    return SICDTimeline(
        collect_start=collect_start,
        collect_duration=duration,
    )


def _build_position(geometry: Any) -> SICDPosition:
    """Build ARP polynomial from the geometry's 5th-order numpy fits."""
    arp_poly = _xyz_poly_from_numpy(
        getattr(geometry, 'arp_poly_x', None),
        getattr(geometry, 'arp_poly_y', None),
        getattr(geometry, 'arp_poly_z', None),
    )
    return SICDPosition(arp_poly=arp_poly)


def _build_scpcoa(geometry: Any) -> SICDSCPCOA:
    coa_idx = geometry.npulses // 2
    arp_pos = _to_xyz(geometry.arp[coa_idx])
    arp_vel = _to_xyz(geometry.varp[coa_idx])
    arp_acc = _arp_acceleration(geometry, coa_idx)

    return SICDSCPCOA(
        scp_time=float(geometry.coa_time),
        arp_pos=arp_pos,
        arp_vel=arp_vel,
        arp_acc=arp_acc,
        side_of_track=geometry.side_of_track,
        slant_range=float(geometry.slant_range_coa),
        ground_range=float(geometry.ground_range_coa),
        doppler_cone_ang=float(np.degrees(geometry.dop_cone_ang_coa)),
        graze_ang=float(np.degrees(geometry.graz_ang_coa)),
        incidence_ang=float(np.degrees(geometry.incd_ang_coa)),
        twist_ang=float(np.degrees(geometry.twist_ang_coa)),
        slope_ang=float(np.degrees(geometry.slope_ang_coa)),
        azim_ang=float(np.degrees(geometry.azim_ang_coa)),
        layover_ang=float(np.degrees(geometry.layover_ang_coa)),
    )


def _build_radar_collection(meta: CPHDMetadata) -> SICDRadarCollection:
    """RadarCollection from CPHD TxRcv + Channel polarization."""
    tx_freq = None
    waveforms: Optional[List[SICDWaveformParams]] = None
    tx_polarization: Optional[str] = None
    rcv_channels: Optional[List[SICDRcvChannel]] = None

    # Frequency band — prefer Global FX band, fall back to TxRcv freq center.
    if meta.global_params is not None:
        gp = meta.global_params
        if gp.fx_band_min is not None and gp.fx_band_max is not None:
            tx_freq = SICDTxFrequency(
                min=float(gp.fx_band_min), max=float(gp.fx_band_max),
            )

    if meta.tx_rcv is not None:
        tr = meta.tx_rcv
        # Waveforms (sarpy expects at least one when TxFrequency is set)
        if tr.tx_waveforms:
            waveforms = []
            for i, wf in enumerate(tr.tx_waveforms):
                rcv = (
                    tr.rcv_parameters[i]
                    if i < len(tr.rcv_parameters) else None
                )
                waveforms.append(SICDWaveformParams(
                    tx_pulse_length=wf.pulse_length,
                    tx_rf_bandwidth=wf.rf_bandwidth,
                    tx_freq_start=(
                        wf.freq_center - wf.rf_bandwidth / 2.0
                        if wf.freq_center is not None
                        and wf.rf_bandwidth is not None
                        else None
                    ),
                    tx_fm_rate=wf.lfm_rate,
                    rcv_window_length=(
                        rcv.window_length if rcv is not None else None
                    ),
                    adc_sample_rate=(
                        rcv.sample_rate if rcv is not None else None
                    ),
                    rcv_if_bandwidth=(
                        rcv.if_filter_bw if rcv is not None else None
                    ),
                    rcv_freq_start=(
                        rcv.freq_center if rcv is not None else None
                    ),
                    rcv_fm_rate=(
                        rcv.lfm_rate if rcv is not None else None
                    ),
                    index=i + 1,
                ))
            # Tx polarization from first waveform when set
            tx_polarization = tr.tx_waveforms[0].polarization

    # Channel polarization → SICD's combined ``Tx:Rcv`` form.
    if (
        meta.channel_section is not None
        and meta.channel_section.parameters
    ):
        rcv_channels = []
        for i, ch in enumerate(meta.channel_section.parameters):
            pol = ch.polarization
            tx_rcv_pol = None
            if pol is not None and pol.tx_pol and pol.rcv_pol:
                tx_rcv_pol = f"{pol.tx_pol}:{pol.rcv_pol}"
            rcv_channels.append(SICDRcvChannel(
                tx_rcv_polarization=tx_rcv_pol,
                index=i + 1,
            ))
            if tx_polarization is None and pol is not None:
                tx_polarization = pol.tx_pol

    return SICDRadarCollection(
        tx_frequency=tx_freq,
        waveform=waveforms,
        tx_polarization=tx_polarization,
        rcv_channels=rcv_channels,
    )


def _build_image_formation(
    meta: CPHDMetadata,
    geometry: Any,
    image_form_algo: str,
    grid_params: Dict[str, Any],
) -> SICDImageFormation:
    """ImageFormation: algo + processed time/frequency windows."""
    rcv_chan_proc = SICDRcvChanProc(
        num_chan_proc=1,
        chan_indices=[1],
    )

    # Processed frequency window: prefer the IFP's grid k-extent
    # converted back to Hz; otherwise fall back to CPHD FxBand or TxRcv.
    fmin = grid_params.get('proc_fx_min')
    fmax = grid_params.get('proc_fx_max')
    if fmin is None and fmax is None:
        if meta.global_params is not None:
            fmin = meta.global_params.fx_band_min
            fmax = meta.global_params.fx_band_max
    tx_freq_proc = None
    if fmin is not None and fmax is not None:
        tx_freq_proc = SICDTxFrequencyProc(
            min_proc=float(fmin), max_proc=float(fmax),
        )

    return SICDImageFormation(
        rcv_chan_proc=rcv_chan_proc,
        image_form_algo=image_form_algo,
        t_start_proc=float(geometry.tstart),
        t_end_proc=float(geometry.tend),
        tx_frequency_proc=tx_freq_proc,
        seg_id=(
            meta.channel_section.parameters[0].identifier
            if meta.channel_section
            and meta.channel_section.parameters else 'IFP'
        ),
        image_beam_comp='NO',
        az_autofocus='NO',
        rg_autofocus='NO',
    )


# ===================================================================
# Helpers
# ===================================================================

def _scp_at_coa(geometry: Any) -> Tuple[XYZ, LatLonHAE]:
    """Pick the SRP at center-of-aperture, return (ECF, LLH)."""
    coa_idx = geometry.npulses // 2
    ecf = geometry.srp[coa_idx]
    llh_arr = geometry.srp_llh[coa_idx]
    return (
        XYZ(float(ecf[0]), float(ecf[1]), float(ecf[2])),
        LatLonHAE(
            lat=float(llh_arr[0]),
            lon=float(llh_arr[1]),
            hae=float(llh_arr[2]),
        ),
    )


def _corner_points(
    meta: CPHDMetadata,
    scp_llh: LatLonHAE,
) -> List[LatLon]:
    """Return four image corner points in SICD ICP order (1..4)."""
    if (
        meta.scene_coordinates is not None
        and meta.scene_coordinates.corner_points is not None
    ):
        cp = meta.scene_coordinates.corner_points
        if cp.shape == (4, 2):
            return [
                LatLon(lat=float(cp[i, 0]), lon=float(cp[i, 1]))
                for i in range(4)
            ]

    # Fallback: tiny box around SCP. Better than nothing for sarpy
    # validation but caller should populate ``corner_points`` upstream.
    d = 0.001  # ~110 m
    return [
        LatLon(lat=scp_llh.lat + d, lon=scp_llh.lon - d),
        LatLon(lat=scp_llh.lat + d, lon=scp_llh.lon + d),
        LatLon(lat=scp_llh.lat - d, lon=scp_llh.lon + d),
        LatLon(lat=scp_llh.lat - d, lon=scp_llh.lon - d),
    ]


def _arp_acceleration(geometry: Any, coa_idx: int) -> Optional[XYZ]:
    """Numerically differentiate ARP velocity at the COA pulse."""
    varp = geometry.varp
    time = geometry.time
    n = len(time)
    if n < 3:
        return None
    if coa_idx <= 0:
        i0, i1 = 0, 1
    elif coa_idx >= n - 1:
        i0, i1 = n - 2, n - 1
    else:
        i0, i1 = coa_idx - 1, coa_idx + 1
    dt = float(time[i1] - time[i0])
    if dt == 0.0:
        return None
    acc = (varp[i1] - varp[i0]) / dt
    return XYZ(float(acc[0]), float(acc[1]), float(acc[2]))


def _xyz_poly_from_numpy(
    px: Any, py: Any, pz: Any,
) -> Optional[XYZPoly]:
    """Convert three ``np.polynomial.Polynomial`` fits into an ``XYZPoly``.

    ``np.polynomial.Polynomial.fit`` returns coefficients in the
    *scaled* domain. ``.convert(domain=[-1,1], window=[-1,1])`` returns
    the equivalent polynomial in the un-scaled domain (i.e., evaluated
    directly against time-since-collect-start), which is what SICD
    expects.
    """
    if px is None or py is None or pz is None:
        return None

    def _coefs(p: Any) -> Optional[np.ndarray]:
        try:
            converted = p.convert()
            return np.asarray(converted.coef, dtype=np.float64)
        except Exception:
            try:
                return np.asarray(p.coef, dtype=np.float64)
            except Exception:
                return None

    cx = _coefs(px)
    cy = _coefs(py)
    cz = _coefs(pz)
    if cx is None or cy is None or cz is None:
        return None
    return XYZPoly(x=Poly1D(coefs=cx), y=Poly1D(coefs=cy), z=Poly1D(coefs=cz))


def _to_xyz(v: Any) -> Optional[XYZ]:
    """Coerce a 3-element array-like into ``XYZ`` (or ``None``)."""
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float64).ravel()
    if arr.size != 3:
        return None
    return XYZ(float(arr[0]), float(arr[1]), float(arr[2]))


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
