# -*- coding: utf-8 -*-
"""
CRSD XML Metadata Builder for Sentinel-1 Level 0 Data.

Constructs an ``lxml.etree.ElementTree`` conforming to the
NGA.STND.0080-1 CRSD schema from Sentinel-1 L0 burst metadata,
orbit state vectors, and waveform parameters.

The output XML is consumed by ``sarkit.crsd.Writer`` to produce
a NITF CRSD file.

Dependencies
------------
lxml

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-22

Modified
--------
2026-05-22
"""

# Standard library
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np
from lxml import etree

# GRDL internal
from grdl.IO.sar.sentinel1_l0.constants import (
    SENTINEL1_CENTER_FREQUENCY_HZ,
    SPEED_OF_LIGHT,
    WGS84_ECCENTRICITY_SQ,
    WGS84_SEMI_MAJOR_AXIS,
)

logger = logging.getLogger(__name__)

CRSD_NAMESPACE = "http://api.nsgreg.nga.mil/schema/crsd/1.0"

# NGA naming conventions for antenna elements
_APC_PREFIX = "AntennaPhaseCenter_"
_APAT_PREFIX = "AntennaPattern_"
_ACF_PREFIX = "AntCoordFrame_"


# =====================================================================
# Data containers
# =====================================================================


@dataclass
class BurstChannelInfo:
    """Per-burst/channel metadata collected during L0 parsing.

    Parameters
    ----------
    identifier : str
        Channel identifier, e.g. ``"043_090466_IW1"``.
    swath_name : str
        Sub-swath name, e.g. ``"IW1"``.
    polarization : str
        Polarization code, e.g. ``"VV"``, ``"VH"``.
    num_vectors : int
        Number of receive vectors (azimuth lines) in the burst.
    num_samples : int
        Number of range samples per vector.
    tx_time_first : float
        First transmit time relative to ``CollectionRefTime`` (s).
    tx_time_last : float
        Last transmit time relative to ``CollectionRefTime`` (s).
    rcv_start_first : float
        First receive start time relative to ``CollectionRefTime`` (s).
    rcv_start_last : float
        Last receive start time relative to ``CollectionRefTime`` (s).
    f0_ref : float
        Reference frequency (Hz).
    fs : float
        Range sampling rate (Hz).
    bw_inst : float
        Instantaneous bandwidth (Hz).
    fx_freq0 : float
        Chirp start frequency (Hz).
    fx_rate : float
        Chirp rate (Hz/s).
    fx_bw : float
        Chirp bandwidth (Hz).
    tx_pulse_duration : float
        Transmit pulse duration (s).
    ref_vector_index : int
        Reference vector index (typically mid-burst).
    tx_ref_pos : np.ndarray
        Transmit reference position ECEF (3,).
    tx_ref_vel : np.ndarray
        Transmit reference velocity ECEF (3,).
    rcv_ref_pos_iac : Tuple[float, float]
        Receive reference point in image area coords.
    """

    identifier: str
    swath_name: str
    polarization: str
    num_vectors: int
    num_samples: int
    tx_time_first: float
    tx_time_last: float
    rcv_start_first: float
    rcv_start_last: float
    f0_ref: float
    fs: float
    bw_inst: float
    fx_freq0: float
    fx_rate: float
    fx_bw: float
    tx_pulse_duration: float
    ref_vector_index: int
    tx_ref_pos: np.ndarray
    tx_ref_vel: np.ndarray
    rcv_ref_pos_iac: Tuple[float, float] = (0.0, 0.0)


@dataclass
class CRSDSceneInfo:
    """Scene-level metadata for the CRSD product.

    Parameters
    ----------
    iarp_ecf : np.ndarray
        Image Area Reference Point in ECEF (3,).
    iarp_llh : Tuple[float, float, float]
        IARP as (lat_deg, lon_deg, hae_m).
    uiax : np.ndarray
        Unit vector along image-area X axis in ECEF (3,).
    uiay : np.ndarray
        Unit vector along image-area Y axis in ECEF (3,).
    image_area_x1y1 : Tuple[float, float]
        Image area bounding box corner in IAC.
    image_area_x2y2 : Tuple[float, float]
        Image area bounding box corner in IAC.
    corner_coords : List[Tuple[float, float]]
        Four IACP corner coords as (lat, lon).
    """

    iarp_ecf: np.ndarray
    iarp_llh: Tuple[float, float, float]
    uiax: np.ndarray
    uiay: np.ndarray
    image_area_x1y1: Tuple[float, float] = (0.0, 0.0)
    image_area_x2y2: Tuple[float, float] = (0.0, 0.0)
    corner_coords: List[Tuple[float, float]] = field(
        default_factory=list,
    )


@dataclass
class CRSDReferenceGeometryInfo:
    """Reference geometry at center of dwell.

    Parameters
    ----------
    ref_pos_ecf : np.ndarray
        Reference point ECEF (3,) — typically IARP.
    ref_pos_iac : Tuple[float, float]
        Reference point in image-area coords.
    cod_time : float
        Center-of-dwell time relative to CollectionRefTime (s).
    dwell_time : float
        Total dwell time (s).
    platform_pos : np.ndarray
        Platform ECEF position at COD time (3,).
    platform_vel : np.ndarray
        Platform ECEF velocity at COD time (3,).
    side_of_track : str
        ``"R"`` or ``"L"``.
    """

    ref_pos_ecf: np.ndarray
    ref_pos_iac: Tuple[float, float]
    cod_time: float
    dwell_time: float
    platform_pos: np.ndarray
    platform_vel: np.ndarray
    side_of_track: str = "R"


# =====================================================================
# Helper functions
# =====================================================================


def _sub(parent: etree._Element, tag: str, text: str = None,
         **attribs) -> etree._Element:
    """Add a child element with optional text."""
    el = etree.SubElement(parent, tag, **attribs)
    if text is not None:
        el.text = str(text)
    return el


def _sub_xyz(parent: etree._Element, tag: str,
             xyz: np.ndarray) -> etree._Element:
    """Add an XYZ sub-element."""
    el = etree.SubElement(parent, tag)
    _sub(el, "X", f"{xyz[0]:.15g}")
    _sub(el, "Y", f"{xyz[1]:.15g}")
    _sub(el, "Z", f"{xyz[2]:.15g}")
    return el


def _sub_latlon(parent: etree._Element, tag: str,
                lat: float, lon: float) -> etree._Element:
    """Add a LatLon sub-element."""
    el = etree.SubElement(parent, tag)
    _sub(el, "Lat", f"{lat:.15g}")
    _sub(el, "Lon", f"{lon:.15g}")
    return el


def _sub_llh(parent: etree._Element, tag: str,
             lat: float, lon: float,
             hae: float) -> etree._Element:
    """Add a LatLonHAE sub-element."""
    el = etree.SubElement(parent, tag)
    _sub(el, "Lat", f"{lat:.15g}")
    _sub(el, "Lon", f"{lon:.15g}")
    _sub(el, "HAE", f"{hae:.15g}")
    return el


def _sub_intfrac(parent: etree._Element, tag: str,
                 offset: int, size: int,
                 fmt: str = "Int=I8;Frac=F8;") -> etree._Element:
    """Add a PerParameterIntFrac element."""
    el = etree.SubElement(parent, tag)
    _sub(el, "Offset", str(offset))
    _sub(el, "Size", str(size))
    _sub(el, "Format", fmt)
    return el


def _sub_f8(parent: etree._Element, tag: str,
            offset: int) -> etree._Element:
    """Add a PerParameterF8 element."""
    el = etree.SubElement(parent, tag)
    _sub(el, "Offset", str(offset))
    _sub(el, "Size", "1")
    _sub(el, "Format", "F8")
    return el


def _sub_xyz_pp(parent: etree._Element, tag: str,
                offset: int) -> etree._Element:
    """Add a PerParameterXYZ element."""
    el = etree.SubElement(parent, tag)
    _sub(el, "Offset", str(offset))
    _sub(el, "Size", "3")
    _sub(el, "Format", "X=F8;Y=F8;Z=F8;")
    return el


def _sub_eb(parent: etree._Element, tag: str,
            offset: int) -> etree._Element:
    """Add a PerParameterEB element."""
    el = etree.SubElement(parent, tag)
    _sub(el, "Offset", str(offset))
    _sub(el, "Size", "2")
    _sub(el, "Format", "DCX=F8;DCY=F8;")
    return el


def _sub_i8(parent: etree._Element, tag: str,
            offset: int) -> etree._Element:
    """Add a PerParameterI8 element."""
    el = etree.SubElement(parent, tag)
    _sub(el, "Offset", str(offset))
    _sub(el, "Size", "1")
    _sub(el, "Format", "I8")
    return el


def _sub_polarization(parent: etree._Element, tag: str,
                      pol_id: str) -> etree._Element:
    """Add a HVPolarizationDescription element."""
    el = etree.SubElement(parent, tag)
    _sub(el, "PolarizationID", pol_id)
    if pol_id == "V":
        _sub(el, "AmpH", "0.0")
        _sub(el, "AmpV", "1.0")
        _sub(el, "PhaseH", "0.0")
        _sub(el, "PhaseV", "0.0")
    else:
        _sub(el, "AmpH", "1.0")
        _sub(el, "AmpV", "0.0")
        _sub(el, "PhaseH", "0.0")
        _sub(el, "PhaseV", "0.0")
    return el


def _datetime_to_crsd(dt: datetime) -> str:
    """Format a datetime as CRSD XDTType string (UTC with Z)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def ecef_to_geodetic(
    x: float, y: float, z: float,
) -> Tuple[float, float, float]:
    """Convert ECEF to geodetic (lat_deg, lon_deg, hae_m).

    Uses Bowring's iterative method.
    """
    a = WGS84_SEMI_MAJOR_AXIS
    e2 = WGS84_ECCENTRICITY_SQ
    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(5):
        sin_lat = np.sin(lat)
        n_val = a / np.sqrt(1.0 - e2 * sin_lat ** 2)
        lat = np.arctan2(z + e2 * n_val * sin_lat, p)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n_val = a / np.sqrt(1.0 - e2 * sin_lat ** 2)
    if abs(cos_lat) > 1e-10:
        hae = p / cos_lat - n_val
    else:
        # At the poles, use the z-based formula to avoid 0/0
        hae = abs(z) - n_val * (1.0 - e2)
    lat_deg = np.degrees(lat)
    return float(lat_deg), float(lon), float(hae)


def _compute_reference_geometry(
    ref_pos: np.ndarray,
    platform_pos: np.ndarray,
    platform_vel: np.ndarray,
) -> Dict[str, float]:
    """Compute geometric angles from platform to reference point.

    Returns dict with SlantRange, GroundRange, GrazeAngle,
    IncidenceAngle, DopplerConeAngle, AzimuthAngle, etc.
    """
    r_vec = ref_pos - platform_pos
    slant_range = float(np.linalg.norm(r_vec))
    r_hat = r_vec / slant_range

    # Doppler cone angle
    v_hat = platform_vel / np.linalg.norm(platform_vel)
    cos_dca = float(np.dot(r_hat, v_hat))
    dca_deg = float(np.degrees(np.arccos(np.clip(cos_dca, -1, 1))))

    # Grazing / incidence from Earth center
    ref_norm = ref_pos / np.linalg.norm(ref_pos)
    cos_graze = float(np.dot(-r_hat, ref_norm))
    graze_deg = float(np.degrees(np.arcsin(
        np.clip(cos_graze, -1, 1),
    )))
    inc_deg = 90.0 - graze_deg

    # Ground range (arc distance on sphere)
    plat_norm = platform_pos / np.linalg.norm(platform_pos)
    cos_angle = float(np.dot(plat_norm, ref_norm))
    # Ground range = Earth_radius * central angle
    earth_r = np.linalg.norm(ref_pos)
    ground_range = float(
        earth_r * np.arccos(np.clip(cos_angle, -1, 1)),
    )

    return {
        "SlantRange": slant_range,
        "GroundRange": ground_range,
        "GrazeAngle": graze_deg,
        "IncidenceAngle": inc_deg,
        "DopplerConeAngle": dca_deg,
    }


# =====================================================================
# XML Builder
# =====================================================================


class CRSDMetadataBuilder:
    """Build a CRSD XML tree from Sentinel-1 L0 metadata.

    Parameters
    ----------
    product_name : str
        SAFE product name (e.g. ``"S1A_IW_RAW__0SDV_..."``).
    collection_ref_time : datetime
        UTC reference epoch for all relative times.
    sensor_name : str
        Sensor identifier (e.g. ``"S1A"``).
    channels : list of BurstChannelInfo
        Per-burst channel metadata.
    scene : CRSDSceneInfo
        Scene-level geometry.
    ref_geometry : CRSDReferenceGeometryInfo
        Reference geometry at center of collection.
    tropo_n0 : float, optional
        Tropospheric refractivity at surface.
    """

    def __init__(
        self,
        product_name: str,
        collection_ref_time: datetime,
        sensor_name: str,
        channels: List[BurstChannelInfo],
        scene: CRSDSceneInfo,
        ref_geometry: CRSDReferenceGeometryInfo,
        tropo_n0: Optional[float] = None,
    ) -> None:
        self.product_name = product_name
        self.collection_ref_time = collection_ref_time
        self.sensor_name = sensor_name
        self.channels = channels
        self.scene = scene
        self.ref_geometry = ref_geometry
        self.tropo_n0 = tropo_n0

    def build(self) -> etree._ElementTree:
        """Construct the complete CRSD XML tree.

        Returns
        -------
        lxml.etree.ElementTree
            Validated CRSD metadata tree with ``CRSDsar`` root.
        """
        nsmap = {None: CRSD_NAMESPACE}
        root = etree.Element(
            f"{{{CRSD_NAMESPACE}}}CRSDsar", nsmap=nsmap,
        )

        self._build_product_info(root)
        self._build_sar_info(root)
        self._build_transmit_info(root)
        self._build_receive_info(root)
        self._build_global(root)
        self._build_scene_coordinates(root)
        self._build_data(root)
        self._build_tx_sequence(root)
        self._build_channel(root)
        self._build_reference_geometry(root)
        self._build_dwell_polynomials(root)
        self._build_support_array(root)
        self._build_ppp(root)
        self._build_pvp(root)
        self._build_antenna(root)

        tree = etree.ElementTree(root)
        return tree

    # -----------------------------------------------------------------
    # Block builders
    # -----------------------------------------------------------------

    def _build_product_info(self, root: etree._Element) -> None:
        pi = _sub(root, "ProductInfo")
        _sub(pi, "ProductName", self.product_name)
        _sub(pi, "Classification", "UNCLASSIFIED")
        _sub(pi, "ReleaseInfo", "UNRESTRICTED")
        ci = _sub(pi, "CreationInfo")
        _sub(ci, "Application", "GRDL Sentinel-1 L0 to CRSD")
        _sub(ci, "DateTime", _datetime_to_crsd(datetime.utcnow()))

    def _build_sar_info(self, root: etree._Element) -> None:
        si = _sub(root, "SARInfo")
        _sub(si, "CollectType", "MONOSTATIC")
        rm = _sub(si, "RadarMode")
        _sub(rm, "ModeType", "STRIPMAP")
        # Optional aux data parameter references
        if hasattr(self, 'aux_parameters'):
            for name, value in self.aux_parameters.items():
                _sub(si, "Parameter", value, name=name)

    def _build_transmit_info(self, root: etree._Element) -> None:
        ti = _sub(root, "TransmitInfo")
        _sub(ti, "SensorName", self.sensor_name)
        _sub(ti, "EventName", self.product_name)

    def _build_receive_info(self, root: etree._Element) -> None:
        ri = _sub(root, "ReceiveInfo")
        _sub(ri, "SensorName", self.sensor_name)
        _sub(ri, "EventName", self.product_name)

    def _build_global(self, root: etree._Element) -> None:
        gl = _sub(root, "Global")
        _sub(
            gl, "CollectionRefTime",
            _datetime_to_crsd(self.collection_ref_time),
        )

        if self.tropo_n0 is not None:
            tp = _sub(gl, "TropoParameters")
            _sub(tp, "N0", f"{self.tropo_n0:.15g}")
            _sub(tp, "RefHeight", "ZERO")

        # Transmit time extents
        tx_times_first = [c.tx_time_first for c in self.channels]
        tx_times_last = [c.tx_time_last for c in self.channels]
        tx = _sub(gl, "Transmit")
        _sub(tx, "TxTime1", f"{min(tx_times_first):.15g}")
        _sub(tx, "TxTime2", f"{max(tx_times_last):.15g}")

        # Frequency extents across all channels
        fx_min = min(c.fx_freq0 for c in self.channels)
        fx_max = max(
            c.fx_freq0 + c.fx_rate * c.tx_pulse_duration
            for c in self.channels
        )
        _sub(tx, "FxMin", f"{fx_min:.15g}")
        _sub(tx, "FxMax", f"{fx_max:.15g}")

        # Receive time extents
        rcv = _sub(gl, "Receive")
        rcv_first = [c.rcv_start_first for c in self.channels]
        rcv_last = [c.rcv_start_last for c in self.channels]
        _sub(rcv, "RcvStartTime1", f"{min(rcv_first):.15g}")
        _sub(rcv, "RcvStartTime2", f"{max(rcv_last):.15g}")

        # Receive frequency extents
        frcv_min = min(
            c.f0_ref - c.bw_inst / 2.0 for c in self.channels
        )
        frcv_max = max(
            c.f0_ref + c.bw_inst / 2.0 for c in self.channels
        )
        _sub(rcv, "FrcvMin", f"{frcv_min:.15g}")
        _sub(rcv, "FrcvMax", f"{frcv_max:.15g}")

    def _build_scene_coordinates(
        self, root: etree._Element,
    ) -> None:
        sc = _sub(root, "SceneCoordinates")
        _sub(sc, "EarthModel", "WGS_84")

        iarp = _sub(sc, "IARP")
        _sub_xyz(iarp, "ECF", self.scene.iarp_ecf)
        lat, lon, hae = self.scene.iarp_llh
        _sub_llh(iarp, "LLH", lat, lon, hae)

        rs = _sub(sc, "ReferenceSurface")
        planar = _sub(rs, "Planar")
        _sub_xyz(planar, "uIAX", self.scene.uiax)
        _sub_xyz(planar, "uIAY", self.scene.uiay)

        # Image area
        ia = _sub(sc, "ImageArea")
        x1, y1 = self.scene.image_area_x1y1
        x2, y2 = self.scene.image_area_x2y2
        xy1 = _sub(ia, "X1Y1")
        _sub(xy1, "X", f"{x1:.15g}")
        _sub(xy1, "Y", f"{y1:.15g}")
        xy2 = _sub(ia, "X2Y2")
        _sub(xy2, "X", f"{x2:.15g}")
        _sub(xy2, "Y", f"{y2:.15g}")

        # Polygon — simple bounding rectangle
        poly = _sub(ia, "Polygon", size="4")
        corners_iac = [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2),
        ]
        for idx, (cx, cy) in enumerate(corners_iac, 1):
            v = _sub(poly, "Vertex", index=str(idx))
            _sub(v, "X", f"{cx:.15g}")
            _sub(v, "Y", f"{cy:.15g}")

        # Corner points
        iacp_block = _sub(sc, "ImageAreaCornerPoints")
        for idx, (clat, clon) in enumerate(
            self.scene.corner_coords[:4], 1,
        ):
            cp = _sub(iacp_block, "IACP", index=str(idx))
            _sub(cp, "Lat", f"{clat:.15g}")
            _sub(cp, "Lon", f"{clon:.15g}")

    def _build_data(self, root: etree._Element) -> None:
        """Build BinaryDataParameters block.

        Byte offsets are placeholders — ``sarkit.crsd.Writer``
        recomputes them internally.
        """
        data = _sub(root, "Data")

        # Support arrays — minimal: flat frequency response
        support = _sub(data, "Support")
        _sub(support, "NumSupportArrays", "1")
        sa = _sub(support, "SupportArray")
        _sub(sa, "SAId", "flat_fx_response")
        _sub(sa, "NumRows", "1")
        _sub(sa, "NumCols", "30")
        _sub(sa, "BytesPerElement", "8")
        _sub(sa, "ArrayByteOffset", "0")

        # Transmit
        tx = _sub(data, "Transmit")
        _sub(tx, "NumBytesPPP", "200")
        _sub(tx, "NumTxSequences", str(len(self.channels)))
        ppp_offset = 0
        for ch in self.channels:
            ts = _sub(tx, "TxSequence")
            _sub(ts, "TxId", ch.identifier)
            _sub(ts, "NumPulses", str(ch.num_vectors))
            _sub(ts, "PPPArrayByteOffset", str(ppp_offset))
            ppp_offset += ch.num_vectors * 200

        # Receive
        rcv = _sub(data, "Receive")
        _sub(rcv, "SignalArrayFormat", "CI2")
        _sub(rcv, "NumBytesPVP", "216")
        _sub(rcv, "NumCRSDChannels", str(len(self.channels)))

        sig_offset = 0
        pvp_offset = 0
        for ch in self.channels:
            c = _sub(rcv, "Channel")
            _sub(c, "ChId", ch.identifier)
            _sub(c, "NumVectors", str(ch.num_vectors))
            _sub(c, "NumSamples", str(ch.num_samples))
            _sub(c, "SignalArrayByteOffset", str(sig_offset))
            _sub(c, "PVPArrayByteOffset", str(pvp_offset))
            # CI2 = 2 bytes per sample
            sig_offset += ch.num_vectors * ch.num_samples * 2
            pvp_offset += ch.num_vectors * 216

    def _build_tx_sequence(self, root: etree._Element) -> None:
        txseq = _sub(root, "TxSequence")
        mid_idx = len(self.channels) // 2
        _sub(txseq, "RefTxId", self.channels[mid_idx].identifier)
        _sub(txseq, "TxWFType", "LFM")

        for ch in self.channels:
            params = _sub(txseq, "Parameters")
            _sub(params, "Identifier", ch.identifier)
            _sub(params, "RefPulseIndex", str(ch.ref_vector_index))
            _sub(params, "FxResponseId", "flat_fx_response")
            _sub(params, "FxBWFixed", "true")
            _sub(params, "FxC", f"{ch.fx_freq0 + ch.fx_bw / 2.0:.15g}")
            _sub(params, "FxBW", f"{ch.fx_bw:.15g}")
            _sub(params, "TXmtMin", f"{ch.tx_pulse_duration:.15g}")
            _sub(params, "TXmtMax", f"{ch.tx_pulse_duration:.15g}")
            _sub(params, "TxTime1", f"{ch.tx_time_first:.15g}")
            _sub(params, "TxTime2", f"{ch.tx_time_last:.15g}")
            tx_pol = ch.polarization[0]
            _sub(params, "TxAPCId",
                 f"{_APC_PREFIX}{ch.identifier}")
            _sub(params, "TxAPATId",
                 f"{_APAT_PREFIX}{ch.identifier}_{tx_pol}")

            tp = _sub(params, "TxRefPoint")
            _sub_xyz(tp, "ECF", ch.tx_ref_pos)
            iac = _sub(tp, "IAC")
            _sub(iac, "X", f"{ch.rcv_ref_pos_iac[0]:.15g}")
            _sub(iac, "Y", f"{ch.rcv_ref_pos_iac[1]:.15g}")

            _sub_polarization(params, "TxPolarization", tx_pol)
            _sub(params, "TxRefRadIntensity", "0.0")
            _sub(params, "TxRadIntErrorStdDev", "1.0")
            _sub(params, "TxRefLAtm", "1.0")

    def _build_channel(self, root: etree._Element) -> None:
        ch_block = _sub(root, "Channel")

        # RefChId — pick the middle channel
        mid_idx = len(self.channels) // 2
        _sub(ch_block, "RefChId", self.channels[mid_idx].identifier)

        for ch in self.channels:
            params = _sub(ch_block, "Parameters")
            _sub(params, "Identifier", ch.identifier)
            _sub(params, "RefVectorIndex",
                 str(ch.ref_vector_index))
            _sub(params, "RefFreqFixed", "true")
            _sub(params, "FrcvFixed", "true")
            _sub(params, "SignalNormal", "true")
            _sub(params, "F0Ref", f"{ch.f0_ref:.15g}")
            _sub(params, "Fs", f"{ch.fs:.15g}")
            _sub(params, "BWInst", f"{ch.bw_inst:.15g}")
            _sub(params, "RcvStartTime1",
                 f"{ch.rcv_start_first:.15g}")
            _sub(params, "RcvStartTime2",
                 f"{ch.rcv_start_last:.15g}")

            frcv_min = ch.f0_ref - ch.bw_inst / 2.0
            frcv_max = ch.f0_ref + ch.bw_inst / 2.0
            _sub(params, "FrcvMin", f"{frcv_min:.15g}")
            _sub(params, "FrcvMax", f"{frcv_max:.15g}")

            rcv_pol = ch.polarization[-1]
            _sub(params, "RcvAPCId",
                 f"{_APC_PREFIX}{ch.identifier}")
            _sub(params, "RcvAPATId",
                 f"{_APAT_PREFIX}{ch.identifier}_{rcv_pol}")

            rp = _sub(params, "RcvRefPoint")
            _sub_xyz(rp, "ECF", ch.tx_ref_pos)
            iac = _sub(rp, "IAC")
            _sub(iac, "X", f"{ch.rcv_ref_pos_iac[0]:.15g}")
            _sub(iac, "Y", f"{ch.rcv_ref_pos_iac[1]:.15g}")

            _sub_polarization(
                params, "RcvPolarization", rcv_pol,
            )
            _sub(params, "RcvRefIrradiance", "0.0")
            _sub(params, "RcvIrradianceErrorStdDev", "1.0")
            _sub(params, "RcvRefLAtm", "1.0")
            _sub(params, "PNCRSD", "0.0")
            _sub(params, "BNCRSD", "1.0")

            # SARImage sub-block
            sar = _sub(params, "SARImage")
            _sub(sar, "TxId", ch.identifier)
            _sub(sar, "RefVectorPulseIndex",
                 str(ch.ref_vector_index))
            _sub_polarization(sar, "TxPolarization",
                              ch.polarization[0])
            dw = _sub(sar, "DwellTimes")
            polys = _sub(dw, "Polynomials")
            _sub(polys, "CODId", ch.identifier)
            _sub(polys, "DwellId", ch.identifier)

            img_area = _sub(sar, "ImageArea")
            x1y1 = _sub(img_area, "X1Y1")
            _sub(x1y1, "X", f"{self.scene.image_area_x1y1[0]:.15g}")
            _sub(x1y1, "Y", f"{self.scene.image_area_x1y1[1]:.15g}")
            x2y2 = _sub(img_area, "X2Y2")
            _sub(x2y2, "X", f"{self.scene.image_area_x2y2[0]:.15g}")
            _sub(x2y2, "Y", f"{self.scene.image_area_x2y2[1]:.15g}")

    def _build_dwell_polynomials(
        self, root: etree._Element,
    ) -> None:
        """Build DwellPolynomials — CODTime and DwellTime per channel."""
        dp = _sub(root, "DwellPolynomials")
        _sub(dp, "NumCODTimes", str(len(self.channels)))

        for ch in self.channels:
            cod = _sub(dp, "CODTime")
            _sub(cod, "Identifier", ch.identifier)
            # Constant polynomial: CODTime = mid-burst time
            mid_time = (ch.tx_time_first + ch.tx_time_last) / 2.0
            poly = _sub(cod, "CODTimePoly", order1="0", order2="0")
            _sub(poly, "Coef", f"{mid_time:.15g}",
                 exponent1="0", exponent2="0")

        _sub(dp, "NumDwellTimes", str(len(self.channels)))

        for ch in self.channels:
            dw = _sub(dp, "DwellTime")
            _sub(dw, "Identifier", ch.identifier)
            # Constant polynomial: DwellTime = burst duration
            burst_dur = ch.tx_time_last - ch.tx_time_first
            poly = _sub(dw, "DwellTimePoly", order1="0", order2="0")
            _sub(poly, "Coef", f"{burst_dur:.15g}",
                 exponent1="0", exponent2="0")

    def _build_reference_geometry(
        self, root: etree._Element,
    ) -> None:
        rg = _sub(root, "ReferenceGeometry")
        rg_info = self.ref_geometry

        # RefPoint (NGA uses RefPoint, not IARP)
        rp = _sub(rg, "RefPoint")
        _sub_xyz(rp, "ECF", rg_info.ref_pos_ecf)
        iac = _sub(rp, "IAC")
        _sub(iac, "X", f"{rg_info.ref_pos_iac[0]:.15g}")
        _sub(iac, "Y", f"{rg_info.ref_pos_iac[1]:.15g}")

        geom = _compute_reference_geometry(
            rg_info.ref_pos_ecf,
            rg_info.platform_pos,
            rg_info.platform_vel,
        )

        # SARImage block
        sar = _sub(rg, "SARImage")
        _sub(sar, "CODTime", f"{rg_info.cod_time:.15g}")
        _sub(sar, "DwellTime", f"{rg_info.dwell_time:.15g}")
        _sub(sar, "ReferenceTime", f"{rg_info.cod_time:.15g}")
        _sub_xyz(sar, "ARPPos", rg_info.platform_pos)
        _sub_xyz(sar, "ARPVel", rg_info.platform_vel)
        _sub(sar, "BistaticAngle", "0.0")
        _sub(sar, "BistaticAngleRate", "0.0")
        _sub(sar, "SideOfTrack", rg_info.side_of_track)
        _sub(sar, "SlantRange",
             f"{geom['SlantRange']:.15g}")
        _sub(sar, "GroundRange",
             f"{geom['GroundRange']:.15g}")
        _sub(sar, "DopplerConeAngle",
             f"{geom['DopplerConeAngle']:.15g}")
        _sub(sar, "GrazeAngle",
             f"{geom['GrazeAngle']:.15g}")
        _sub(sar, "IncidenceAngle",
             f"{geom['IncidenceAngle']:.15g}")

        # TxParameters (monostatic: same as Rcv)
        tx = _sub(rg, "TxParameters")
        _sub(tx, "Time", f"{rg_info.cod_time:.15g}")
        _sub_xyz(tx, "APCPos", rg_info.platform_pos)
        _sub_xyz(tx, "APCVel", rg_info.platform_vel)
        _sub(tx, "SideOfTrack", rg_info.side_of_track)
        _sub(tx, "SlantRange",
             f"{geom['SlantRange']:.15g}")
        _sub(tx, "GroundRange",
             f"{geom['GroundRange']:.15g}")
        _sub(tx, "DopplerConeAngle",
             f"{geom['DopplerConeAngle']:.15g}")
        _sub(tx, "GrazeAngle",
             f"{geom['GrazeAngle']:.15g}")
        _sub(tx, "IncidenceAngle",
             f"{geom['IncidenceAngle']:.15g}")

        # RcvParameters
        rcv = _sub(rg, "RcvParameters")
        _sub(rcv, "Time", f"{rg_info.cod_time:.15g}")
        _sub_xyz(rcv, "APCPos", rg_info.platform_pos)
        _sub_xyz(rcv, "APCVel", rg_info.platform_vel)
        _sub(rcv, "SideOfTrack", rg_info.side_of_track)
        _sub(rcv, "SlantRange",
             f"{geom['SlantRange']:.15g}")
        _sub(rcv, "GroundRange",
             f"{geom['GroundRange']:.15g}")
        _sub(rcv, "DopplerConeAngle",
             f"{geom['DopplerConeAngle']:.15g}")
        _sub(rcv, "GrazeAngle",
             f"{geom['GrazeAngle']:.15g}")
        _sub(rcv, "IncidenceAngle",
             f"{geom['IncidenceAngle']:.15g}")

    def _build_support_array(
        self, root: etree._Element,
    ) -> None:
        sa = _sub(root, "SupportArray")

        # Frequency response array
        fxr = _sub(sa, "FxResponseArray")
        _sub(fxr, "Identifier", "flat_fx_response")
        _sub(fxr, "ElementFormat", "Amp=F4;Phase=F4;")
        _sub(fxr, "Fx0FXR", f"{self.channels[0].fx_freq0:.15g}")
        _sub(fxr, "FxSSFXR", f"{self.channels[0].fx_bw / 29.0:.15g}")

    def _build_ppp(self, root: etree._Element) -> None:
        """Build PerPulseParameters layout definition."""
        ppp = _sub(root, "PPP")
        # Field layout: matches NGA reference at 200 bytes total
        _sub_intfrac(ppp, "TxTime", offset=0, size=2)
        _sub_xyz_pp(ppp, "TxPos", offset=2)
        _sub_xyz_pp(ppp, "TxVel", offset=5)
        _sub_f8(ppp, "FX1", offset=8)
        _sub_f8(ppp, "FX2", offset=9)
        _sub_f8(ppp, "TXmt", offset=10)
        _sub_intfrac(ppp, "PhiX0", offset=11, size=2)
        _sub_f8(ppp, "FxFreq0", offset=13)
        _sub_f8(ppp, "FxRate", offset=14)
        _sub_f8(ppp, "TxRadInt", offset=15)
        _sub_xyz_pp(ppp, "TxACX", offset=16)
        _sub_xyz_pp(ppp, "TxACY", offset=19)
        _sub_eb(ppp, "TxEB", offset=22)
        _sub_i8(ppp, "FxResponseIndex", offset=24)

    def _build_pvp(self, root: etree._Element) -> None:
        """Build PerVectorParameters layout definition."""
        pvp = _sub(root, "PVP")
        # Matches NGA reference layout at 216 bytes total
        _sub_intfrac(pvp, "RcvStart", offset=0, size=2)
        _sub_xyz_pp(pvp, "RcvPos", offset=2)
        _sub_xyz_pp(pvp, "RcvVel", offset=5)
        _sub_f8(pvp, "FRCV1", offset=8)
        _sub_f8(pvp, "FRCV2", offset=9)
        _sub_intfrac(pvp, "RefPhi0", offset=10, size=2)
        _sub_f8(pvp, "RefFreq", offset=12)
        _sub_f8(pvp, "DFIC0", offset=13)
        _sub_f8(pvp, "FICRate", offset=14)
        _sub_xyz_pp(pvp, "RcvACX", offset=15)
        _sub_xyz_pp(pvp, "RcvACY", offset=18)
        _sub_eb(pvp, "RcvEB", offset=24)
        _sub_i8(pvp, "SIGNAL", offset=26)
        _sub_f8(pvp, "AmpSF", offset=22)
        _sub_f8(pvp, "DGRGC", offset=23)
        _sub_i8(pvp, "TxPulseIndex", offset=21)

    def _build_antenna(self, root: etree._Element) -> None:
        ant = _sub(root, "Antenna")
        _sub(ant, "NumACFs", str(len(self.channels)))
        _sub(ant, "NumAPCs", str(len(self.channels)))
        _sub(ant, "NumAntPats", str(len(self.channels)))

        for ch in self.channels:
            acf = _sub(ant, "AntCoordFrame")
            _sub(acf, "Identifier",
                 f"{_ACF_PREFIX}{ch.identifier}")

        for ch in self.channels:
            apc = _sub(ant, "AntPhaseCenter")
            _sub(apc, "Identifier",
                 f"{_APC_PREFIX}{ch.identifier}")
            _sub(apc, "ACFId",
                 f"{_ACF_PREFIX}{ch.identifier}")
            xyz = _sub(apc, "APCXYZ")
            _sub(xyz, "X", "0.0")
            _sub(xyz, "Y", "0.0")
            _sub(xyz, "Z", "0.0")

        for ch in self.channels:
            rcv_pol = ch.polarization[-1]
            pat = _sub(ant, "AntPattern")
            _sub(pat, "Identifier",
                 f"{_APAT_PREFIX}{ch.identifier}_{rcv_pol}")
            _sub(pat, "FreqZero", f"{ch.f0_ref:.15g}")
            _sub(pat, "ArrayGPId",
                 f"AGP_{ch.identifier}")
            _sub(pat, "ElementGPId",
                 f"AGP_{ch.identifier}")
            ebfs = _sub(pat, "EBFreqShift")
            _sub(ebfs, "DCXSF", "0.0")
            _sub(ebfs, "DCYSF", "0.0")
            mlfd = _sub(pat, "MLFreqDilation")
            _sub(mlfd, "DCXSF", "0.0")
            _sub(mlfd, "DCYSF", "0.0")
            gbp = _sub(pat, "GainBSPoly", order1="0")
            _sub(gbp, "Coef", "0.0", exponent1="0")
            apr = _sub(pat, "AntPolRef")
            rcv_pol = ch.polarization[-1]
            if rcv_pol == "V":
                _sub(apr, "AmpX", "0.0")
                _sub(apr, "AmpY", "1.0")
            else:
                _sub(apr, "AmpX", "1.0")
                _sub(apr, "AmpY", "0.0")
            _sub(apr, "PhaseX", "0.0")
            _sub(apr, "PhaseY", "0.0")
