# -*- coding: utf-8 -*-
"""
CRSD to CPHD Conversion Pipeline.

Converts NGA CRSD (Compensated Radar Signal Data) files to CPHD
(Compensated Phase History Data) 1.0.1 format.  Maps the CRSD
separated transmit (PPP) and receive (PVP) parameters into CPHD's
unified per-vector parameters, and applies a range FFT to convert
from time-domain to FX-domain.

Dependencies
------------
sarkit, lxml, numpy

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
2026-03-02

Modified
--------
2026-03-02
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.linalg import norm
from lxml import etree

import sarkit.crsd
import sarkit.cphd

logger = logging.getLogger(__name__)

# Speed of light (m/s)
_C = 299792458.0

# CPHD 1.0.1 namespace
CPHD_NS = "http://api.nsgreg.nga.mil/schema/cphd/1.0.1"
_NSMAP = {None: CPHD_NS}


# ===================================================================
# XML builder helpers
# ===================================================================


def _sub(parent: etree._Element, tag: str, text: str = None) -> etree._Element:
    """Add a sub-element in the CPHD namespace."""
    elem = etree.SubElement(parent, f"{{{CPHD_NS}}}{tag}")
    if text is not None:
        elem.text = str(text)
    return elem


def _sub_xyz(
    parent: etree._Element, tag: str, x: float, y: float, z: float,
) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "X", f"{x:.12g}")
    _sub(elem, "Y", f"{y:.12g}")
    _sub(elem, "Z", f"{z:.12g}")
    return elem


def _sub_xy(parent: etree._Element, tag: str, x: float, y: float) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "X", f"{x:.12g}")
    _sub(elem, "Y", f"{y:.12g}")
    return elem


def _sub_latlon(
    parent: etree._Element, tag: str, lat: float, lon: float,
) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "Lat", f"{lat:.10g}")
    _sub(elem, "Lon", f"{lon:.10g}")
    return elem


# ===================================================================
# IntFrac / XYZ helpers for reading sarkit structured arrays
# ===================================================================


def _read_intfrac(arr: np.ndarray, field: str) -> np.ndarray:
    """Read an IntFrac field from a sarkit structured array.

    When sarkit reads IntFrac fields (format ``Int=I8;Frac=F8;``),
    the returned structured sub-dtype has named subfields 'Int' and
    'Frac'.  The float value is ``int_part + frac_part``.

    Returns
    -------
    np.ndarray
        Float64 array: int_part + frac_part.
    """
    sub = arr[field]
    if sub.dtype.names and "Int" in sub.dtype.names:
        int_part = sub["Int"].astype(np.float64)
        frac_part = sub["Frac"].astype(np.float64)
        return int_part + frac_part
    else:
        int_part = sub.astype(np.float64)
        frac_offset = arr.dtype.fields[field][1] + 8
        frac_view = np.ndarray(
            len(arr),
            dtype=np.float64,
            buffer=arr.data,
            offset=frac_offset,
            strides=(arr.dtype.itemsize,),
        )
        return int_part + frac_view


def _read_xyz(arr: np.ndarray, field: str) -> np.ndarray:
    """Read an XYZ field from a sarkit structured array.

    Handles both structured (X/Y/Z subfields) and flat (3,) array
    representations.

    Returns
    -------
    np.ndarray
        Shape (N, 3) float64 array.
    """
    sub = arr[field]
    if sub.dtype.names and "X" in sub.dtype.names:
        x = sub["X"].astype(np.float64)
        y = sub["Y"].astype(np.float64)
        z = sub["Z"].astype(np.float64)
        return np.column_stack([x, y, z])
    return np.asarray(sub, dtype=np.float64).reshape(-1, 3)


# ===================================================================
# CPHD PVP field layout
# ===================================================================

# (field_name, CPHD_format_string, size_in_8byte_words)
CPHD_PVP_FIELDS = [
    ("TxTime", "F8", 1),
    ("TxPos", "X=F8;Y=F8;Z=F8;", 3),
    ("TxVel", "X=F8;Y=F8;Z=F8;", 3),
    ("RcvTime", "F8", 1),
    ("RcvPos", "X=F8;Y=F8;Z=F8;", 3),
    ("RcvVel", "X=F8;Y=F8;Z=F8;", 3),
    ("SRPPos", "X=F8;Y=F8;Z=F8;", 3),
    ("aFDOP", "F8", 1),
    ("aFRR1", "F8", 1),
    ("aFRR2", "F8", 1),
    ("FX1", "F8", 1),
    ("FX2", "F8", 1),
    ("TOA1", "F8", 1),
    ("TOA2", "F8", 1),
    ("TDTropoSRP", "F8", 1),
    ("SC0", "F8", 1),
    ("SCSS", "F8", 1),
    ("SIGNAL", "I8", 1),
    ("AmpSF", "F8", 1),
]


# ===================================================================
# CRSDToCPHD converter
# ===================================================================


class CRSDToCPHD:
    """Convert a CRSD file to CPHD 1.0.1 format.

    Reads the CRSD file via sarkit, converts time-domain signal to
    FX-domain via range FFT, maps CRSD PPP+PVP to CPHD unified PVP,
    and writes a valid CPHD 1.0.1 file.

    Parameters
    ----------
    crsd_path : str or Path
        Input CRSD file.
    output_path : str or Path
        Output CPHD file.
    channel : str, optional
        Channel ID to convert.  If *None*, converts the first channel.
    max_pulses : int, optional
        Limit processing to this many pulses (for memory / speed).

    Examples
    --------
    >>> converter = CRSDToCPHD("input.crsd", "output.cphd")
    >>> converter.convert()

    >>> CRSDToCPHD("input.crsd", "output.cphd", channel="VV", max_pulses=512).convert()
    """

    def __init__(
        self,
        crsd_path: Union[str, Path],
        output_path: Union[str, Path],
        channel: Optional[str] = None,
        max_pulses: Optional[int] = None,
    ):
        self.crsd_path = Path(crsd_path)
        self.output_path = Path(output_path)
        self.channel = channel
        self.max_pulses = max_pulses

    # ── public API ──────────────────────────────────────────────

    def convert(self) -> Path:
        """Run the CRSD → CPHD conversion.

        Returns
        -------
        Path
            Path to the written CPHD file.
        """
        t0 = time.perf_counter()
        logger.info("CRSD → CPHD: reading %s", self.crsd_path.name)

        # ── Read CRSD ──
        fh = open(str(self.crsd_path), "rb")
        reader = sarkit.crsd.Reader(fh)
        crsd_xml = reader.metadata.xmltree

        ch_id, n_vec, n_samp = self._select_channel(crsd_xml)
        tx_id = self._first_tx_id(crsd_xml)

        ppp = reader.read_ppps(tx_id)
        pvp_full = reader.read_pvps(ch_id)
        pvp = pvp_full[:n_vec]
        del pvp_full

        signal_full = reader.read_signal(ch_id)
        signal = signal_full[:n_vec]
        del signal_full

        reader.done()
        fh.close()

        logger.info(
            "  channel=%s  vectors=%d  samples=%d  ppp=%d",
            ch_id, n_vec, n_samp, len(ppp),
        )

        # ── Extract CRSD metadata ──
        crsd_meta = self._extract_crsd_metadata(crsd_xml)

        # ── Extract timing from PPP / PVP ──
        tx_times = _read_intfrac(ppp, "TxTime")
        tx_pos = _read_xyz(ppp, "TxPos")
        tx_vel = _read_xyz(ppp, "TxVel")
        fx1_ppp = ppp["FX1"].astype(np.float64)
        fx2_ppp = ppp["FX2"].astype(np.float64)

        rcv_start = _read_intfrac(pvp, "RcvStart")
        rcv_pos = _read_xyz(pvp, "RcvPos")
        rcv_vel = _read_xyz(pvp, "RcvVel")

        tx_pulse_idx = pvp["TxPulseIndex"].astype(np.int64)
        signal_flags = (
            pvp["SIGNAL"].astype(np.int64)
            if "SIGNAL" in pvp.dtype.names
            else np.ones(n_vec, dtype=np.int64)
        )
        amp_sf = (
            pvp["AmpSF"].astype(np.float64)
            if "AmpSF" in pvp.dtype.names
            else np.ones(n_vec)
        )

        # ── Range FFT: time-domain → FX-domain ──
        logger.info("  Applying range FFT (%d × %d)…", n_vec, n_samp)
        t_fft = time.perf_counter()
        signal_fx = np.fft.fftshift(np.fft.fft(signal, axis=1), axes=1)
        signal_fx = signal_fx.astype(np.complex64)
        del signal
        logger.info("  FFT done in %.2fs", time.perf_counter() - t_fft)

        # FX-domain parameters
        f0 = crsd_meta["f0"]
        fs = crsd_meta["fs"]
        fx1_abs = f0 - fs / 2.0
        fx2_abs = f0 + fs / 2.0
        scss = fs / (n_samp - 1) if n_samp > 1 else fs

        # ── Merge PPP + PVP → CPHD PVP ──
        ppp_n = len(ppp)
        tx_idx = np.clip(tx_pulse_idx, 0, ppp_n - 1)

        cphd_vectors = _CPHDVectors(
            n_vec=n_vec,
            tx_time=tx_times[tx_idx],
            tx_pos=tx_pos[tx_idx],
            tx_vel=tx_vel[tx_idx],
            rcv_time=rcv_start,
            rcv_pos=rcv_pos,
            rcv_vel=rcv_vel,
            iarp_ecf=crsd_meta["iarp_ecf"],
            fx1_abs=fx1_abs,
            fx2_abs=fx2_abs,
            f0=f0,
            fs=fs,
            n_samp=n_samp,
            scss=scss,
            signal_flags=signal_flags,
            amp_sf=amp_sf,
        )

        # ── Build CPHD XML ──
        xmltree = self._build_cphd_xml(
            crsd_meta, ch_id, n_vec, n_samp, cphd_vectors,
        )

        # ── Build PVP structured array ──
        pvp_dtype = sarkit.cphd.get_pvp_dtype(xmltree)
        cphd_pvp = cphd_vectors.to_structured(pvp_dtype)

        # ── Write CPHD ──
        logger.info("  Writing CPHD: %s", self.output_path.name)
        cphd_meta_obj = sarkit.cphd.Metadata(xmltree=xmltree)
        with open(str(self.output_path), "wb") as f_out:
            writer = sarkit.cphd.Writer(f_out, cphd_meta_obj)
            writer.write_pvp(ch_id, cphd_pvp)
            writer.write_signal(ch_id, signal_fx)

        del signal_fx
        dt = time.perf_counter() - t0
        size_gb = self.output_path.stat().st_size / (1024**3)
        logger.info("  CPHD written: %.2f GB in %.1fs", size_gb, dt)

        return self.output_path

    # ── private helpers ─────────────────────────────────────────

    def _select_channel(
        self, crsd_xml: etree._ElementTree,
    ) -> tuple[str, int, int]:
        """Discover channels and select one."""
        channels = []
        for ch_elem in crsd_xml.findall("{*}Data/{*}Receive/{*}Channel"):
            ch_id = ch_elem.findtext("{*}ChId")
            n_vec = int(ch_elem.findtext("{*}NumVectors"))
            n_samp = int(ch_elem.findtext("{*}NumSamples"))
            channels.append((ch_id, n_vec, n_samp))

        if self.channel is not None:
            sel = [c for c in channels if c[0] == self.channel]
            if not sel:
                avail = [c[0] for c in channels]
                raise ValueError(
                    f"Channel '{self.channel}' not found. Available: {avail}"
                )
            ch_id, n_vec, n_samp = sel[0]
        else:
            ch_id, n_vec, n_samp = channels[0]

        if self.max_pulses is not None and self.max_pulses < n_vec:
            n_vec = self.max_pulses

        return ch_id, n_vec, n_samp

    @staticmethod
    def _first_tx_id(crsd_xml: etree._ElementTree) -> str:
        for tx_elem in crsd_xml.findall("{*}Data/{*}Transmit/{*}TxSequence"):
            return tx_elem.findtext("{*}TxId") or "TX1"
        return "TX1"

    @staticmethod
    def _extract_crsd_metadata(crsd_xml: etree._ElementTree) -> dict:
        """Extract the metadata needed for CPHD construction."""
        meta: dict = {}

        # Collection reference time
        meta["ref_time_str"] = (
            crsd_xml.findtext("{*}Global/{*}CollectionRefTime") or ""
        )

        # Scene coordinates — IARP
        iarp_ecf = np.zeros(3)
        iarp_llh = np.zeros(3)
        iarp_elem = crsd_xml.find("{*}SceneCoordinates/{*}IARP/{*}ECF")
        if iarp_elem is not None:
            iarp_ecf[0] = float(iarp_elem.findtext("{*}X") or 0)
            iarp_ecf[1] = float(iarp_elem.findtext("{*}Y") or 0)
            iarp_ecf[2] = float(iarp_elem.findtext("{*}Z") or 0)
        llh_elem = crsd_xml.find("{*}SceneCoordinates/{*}IARP/{*}LLH")
        if llh_elem is not None:
            iarp_llh[0] = float(llh_elem.findtext("{*}Lat") or 0)
            iarp_llh[1] = float(llh_elem.findtext("{*}Lon") or 0)
            iarp_llh[2] = float(llh_elem.findtext("{*}HAE") or 0)
        meta["iarp_ecf"] = iarp_ecf
        meta["iarp_llh"] = iarp_llh

        # Corner points
        corners = []
        for cp in crsd_xml.findall(
            "{*}SceneCoordinates/{*}ImageAreaCornerPoints/{*}IACP"
        ):
            lat = float(cp.findtext("{*}Lat") or 0)
            lon = float(cp.findtext("{*}Lon") or 0)
            corners.append((lat, lon))
        meta["corners"] = corners

        # Reference surface (planar unit vectors)
        u_iax = np.array([1.0, 0.0, 0.0])
        u_iay = np.array([0.0, 1.0, 0.0])
        planar = crsd_xml.find(
            "{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar"
        )
        if planar is not None:
            iax = planar.find("{*}uIAX")
            if iax is not None:
                u_iax = np.array([
                    float(iax.findtext("{*}X") or 0),
                    float(iax.findtext("{*}Y") or 0),
                    float(iax.findtext("{*}Z") or 0),
                ])
            iay = planar.find("{*}uIAY")
            if iay is not None:
                u_iay = np.array([
                    float(iay.findtext("{*}X") or 0),
                    float(iay.findtext("{*}Y") or 0),
                    float(iay.findtext("{*}Z") or 0),
                ])
        meta["u_iax"] = u_iax
        meta["u_iay"] = u_iay

        # Radar parameters
        tx_param = crsd_xml.find("{*}TxSequence/{*}Parameters")
        meta["f0"] = float(tx_param.findtext("{*}FxC") or 5.405e9)
        meta["bw"] = float(tx_param.findtext("{*}FxBW") or 48.3e6)

        ch_param = crsd_xml.find("{*}Channel/{*}Parameters")
        meta["fs"] = float(ch_param.findtext("{*}Fs") or meta["bw"])

        # Polarization
        rx_pol_elem = ch_param.find("{*}RcvPolarization/{*}PolarizationID")
        meta["rx_pol"] = rx_pol_elem.text if rx_pol_elem is not None else "V"
        tx_pol_elem = crsd_xml.find(
            "{*}TxSequence/{*}Parameters/{*}TxPolarization/{*}PolarizationID"
        )
        meta["tx_pol"] = tx_pol_elem.text if tx_pol_elem is not None else "V"

        # SAR info
        meta["mode_type"] = (
            crsd_xml.findtext("{*}SARInfo/{*}RadarMode/{*}ModeType")
            or "DYNAMIC STRIPMAP"
        )
        meta["mode_id"] = (
            crsd_xml.findtext("{*}SARInfo/{*}RadarMode/{*}ModeID") or "IW"
        )
        meta["sensor_name"] = (
            crsd_xml.findtext("{*}TransmitInfo/{*}SensorName") or "SENTINEL-1"
        )

        # Transmit pulse length
        meta["tx_pulse_len"] = float(
            crsd_xml.findtext("{*}TxSequence/{*}Parameters/{*}TXmtMin") or 62e-6
        )

        return meta

    def _build_cphd_xml(
        self,
        meta: dict,
        ch_id: str,
        n_vec: int,
        n_samp: int,
        vectors: _CPHDVectors,
    ) -> etree._ElementTree:
        """Construct the full CPHD 1.0.1 XML tree."""
        root = etree.Element(f"{{{CPHD_NS}}}CPHD", nsmap=_NSMAP)

        num_bytes_pvp = sum(s for _, _, s in CPHD_PVP_FIELDS) * 8
        f0 = meta["f0"]
        bw = meta["bw"]
        fs = meta["fs"]
        iarp_ecf = meta["iarp_ecf"]
        iarp_llh = meta["iarp_llh"]

        fx1_abs = f0 - fs / 2.0
        fx2_abs = f0 + fs / 2.0
        toa_saved = (n_samp - 1) / fs if n_samp > 1 else 1.0 / fs
        dwell_time = float(vectors.tx_time.max() - vectors.tx_time.min())
        tx_pulse_len = meta["tx_pulse_len"]
        lfm_rate = bw / tx_pulse_len if tx_pulse_len > 0 else 0.0

        # -- CollectionID --
        cid = _sub(root, "CollectionID")
        _sub(cid, "CollectorName", meta["sensor_name"])
        _sub(cid, "CoreName", self.crsd_path.stem)
        _sub(cid, "CollectType", "MONOSTATIC")
        rm = _sub(cid, "RadarMode")
        _sub(rm, "ModeType", meta["mode_type"])
        _sub(rm, "ModeID", meta["mode_id"])
        _sub(cid, "Classification", "UNCLASSIFIED")
        _sub(cid, "ReleaseInfo", "PUBLIC RELEASE")

        # -- Global --
        gl = _sub(root, "Global")
        _sub(gl, "DomainType", "FX")
        _sub(gl, "SGN", "-1")
        tl = _sub(gl, "Timeline")
        _sub(tl, "CollectionStart", meta["ref_time_str"])
        _sub(tl, "TxTime1", f"{float(vectors.tx_time.min()):.12g}")
        _sub(tl, "TxTime2", f"{float(vectors.tx_time.max()):.12g}")
        fxb = _sub(gl, "FxBand")
        _sub(fxb, "FxMin", f"{fx1_abs:.12g}")
        _sub(fxb, "FxMax", f"{fx2_abs:.12g}")
        toas = _sub(gl, "TOASwath")
        _sub(toas, "TOAMin", f"{-toa_saved / 2.0:.12g}")
        _sub(toas, "TOAMax", f"{toa_saved / 2.0:.12g}")

        # -- SceneCoordinates --
        sc = _sub(root, "SceneCoordinates")
        _sub(sc, "EarthModel", "WGS_84")
        iarp_e = _sub(sc, "IARP")
        _sub_xyz(iarp_e, "ECF", *iarp_ecf)
        llh_e = _sub(iarp_e, "LLH")
        _sub(llh_e, "Lat", f"{iarp_llh[0]:.10g}")
        _sub(llh_e, "Lon", f"{iarp_llh[1]:.10g}")
        _sub(llh_e, "HAE", f"{iarp_llh[2]:.6g}")
        rs = _sub(sc, "ReferenceSurface")
        pl = _sub(rs, "Planar")
        _sub_xyz(pl, "uIAX", *meta["u_iax"])
        _sub_xyz(pl, "uIAY", *meta["u_iay"])
        ia = _sub(sc, "ImageArea")
        _sub_xy(ia, "X1Y1", -1.0, -1.0)
        _sub_xy(ia, "X2Y2", 1.0, 1.0)
        if meta["corners"]:
            iacp = _sub(sc, "ImageAreaCornerPoints")
            for idx, (lat, lon) in enumerate(meta["corners"][:4], start=1):
                cp = _sub_latlon(iacp, "IACP", lat, lon)
                cp.set("index", str(idx))

        # -- Data --
        data = _sub(root, "Data")
        _sub(data, "SignalArrayFormat", "CF8")
        _sub(data, "NumBytesPVP", str(num_bytes_pvp))
        _sub(data, "NumCPHDChannels", "1")
        ch_data = _sub(data, "Channel")
        _sub(ch_data, "Identifier", ch_id)
        _sub(ch_data, "NumVectors", str(n_vec))
        _sub(ch_data, "NumSamples", str(n_samp))
        _sub(ch_data, "SignalArrayByteOffset", "0")
        _sub(ch_data, "PVPArrayByteOffset", "0")
        _sub(data, "NumSupportArrays", "0")

        # -- Channel --
        ch_sec = _sub(root, "Channel")
        _sub(ch_sec, "RefChId", ch_id)
        _sub(ch_sec, "FXFixedCPHD", "true")
        _sub(ch_sec, "TOAFixedCPHD", "true")
        _sub(ch_sec, "SRPFixedCPHD", "true")
        ch_params = _sub(ch_sec, "Parameters")
        _sub(ch_params, "Identifier", ch_id)
        _sub(ch_params, "RefVectorIndex", "0")
        _sub(ch_params, "FXFixed", "true")
        _sub(ch_params, "TOAFixed", "true")
        _sub(ch_params, "SRPFixed", "true")
        pol_e = _sub(ch_params, "Polarization")
        _sub(pol_e, "TxPol", meta["tx_pol"])
        _sub(pol_e, "RcvPol", meta["rx_pol"])
        _sub(ch_params, "FxC", f"{f0:.12g}")
        _sub(ch_params, "FxBW", f"{fs:.12g}")
        _sub(ch_params, "TOASaved", f"{toa_saved:.12g}")
        dw = _sub(ch_params, "DwellTimes")
        _sub(dw, "CODId", "COD1")
        _sub(dw, "DwellId", "DW1")

        # -- PVP --
        pvp_sec = _sub(root, "PVP")
        offset = 0
        for name, fmt, size in CPHD_PVP_FIELDS:
            elem = _sub(pvp_sec, name)
            _sub(elem, "Offset", str(offset))
            _sub(elem, "Size", str(size))
            _sub(elem, "Format", fmt)
            offset += size

        # -- Dwell --
        dwell_sec = _sub(root, "Dwell")
        _sub(dwell_sec, "NumCODTimes", "1")
        cod = _sub(dwell_sec, "CODTime")
        _sub(cod, "Identifier", "COD1")
        cod_poly = _sub(cod, "CODTimePoly")
        cod_poly.set("order1", "0")
        cod_poly.set("order2", "0")
        cod_coef = _sub(cod_poly, "Coef", "0.0")
        cod_coef.set("exponent1", "0")
        cod_coef.set("exponent2", "0")
        _sub(dwell_sec, "NumDwellTimes", "1")
        dtw = _sub(dwell_sec, "DwellTime")
        _sub(dtw, "Identifier", "DW1")
        dw_poly = _sub(dtw, "DwellTimePoly")
        dw_poly.set("order1", "0")
        dw_poly.set("order2", "0")
        dw_coef = _sub(dw_poly, "Coef", f"{dwell_time:.12g}")
        dw_coef.set("exponent1", "0")
        dw_coef.set("exponent2", "0")

        # -- ReferenceGeometry --
        ref_idx = n_vec // 2
        rg = _sub(root, "ReferenceGeometry")
        _sub_xyz(_sub(rg, "SRP"), "ECF", *iarp_ecf)
        _sub_xy(_sub(rg, "SRP"), "IAC", 0.0, 0.0)
        _sub(rg, "ReferenceTime", "0.0")
        _sub(rg, "SRPCODTime", "0.0")
        _sub(rg, "SRPDwellTime", f"{dwell_time:.12g}")

        mono = _sub(rg, "Monostatic")
        arp_mid = vectors.tx_pos[ref_idx]
        vel_mid = vectors.tx_vel[ref_idx]
        _sub_xyz(mono, "ARPPos", *arp_mid)
        _sub_xyz(mono, "ARPVel", *vel_mid)

        # Compute geometry angles
        srp_vec = iarp_ecf - arp_mid
        slant_range = norm(srp_vec)
        r_hat_ref = (
            srp_vec / slant_range if slant_range > 0
            else np.array([1.0, 0.0, 0.0])
        )
        earth_normal = (
            iarp_ecf / norm(iarp_ecf) if norm(iarp_ecf) > 0
            else np.array([0.0, 0.0, 1.0])
        )
        cos_graze = np.dot(r_hat_ref, earth_normal)
        graze_ang = np.degrees(np.arcsin(np.clip(abs(cos_graze), 0, 1)))
        incidence_ang = 90.0 - graze_ang
        ground_range = slant_range * np.cos(np.radians(graze_ang))
        v_hat = (
            vel_mid / norm(vel_mid) if norm(vel_mid) > 0
            else np.array([1.0, 0.0, 0.0])
        )
        dca = np.degrees(
            np.arccos(np.clip(np.dot(r_hat_ref, v_hat), -1, 1))
        )
        cross = np.cross(vel_mid, srp_vec)
        side = "L" if np.dot(cross, earth_normal) > 0 else "R"

        _sub(mono, "SideOfTrack", side)
        _sub(mono, "SlantRange", f"{slant_range:.6g}")
        _sub(mono, "GroundRange", f"{ground_range:.6g}")
        _sub(mono, "DopplerConeAngle", f"{dca:.6g}")
        _sub(mono, "GrazeAngle", f"{graze_ang:.6g}")
        _sub(mono, "IncidenceAngle", f"{incidence_ang:.6g}")
        _sub(mono, "AzimuthAngle", "0.0")
        _sub(mono, "TwistAngle", "0.0")
        _sub(mono, "SlopeAngle", "0.0")
        _sub(mono, "LayoverAngle", "0.0")

        # -- TxRcv --
        txrcv = _sub(root, "TxRcv")
        _sub(txrcv, "NumTxWFs", "1")
        txwf = _sub(txrcv, "TxWFParameters")
        _sub(txwf, "Identifier", "TXW1")
        _sub(txwf, "PulseLength", f"{tx_pulse_len:.12g}")
        _sub(txwf, "RFBandwidth", f"{bw:.12g}")
        _sub(txwf, "FreqCenter", f"{f0:.12g}")
        _sub(txwf, "LFMRate", f"{lfm_rate:.12g}")
        _sub(txwf, "Polarization", meta["tx_pol"])
        _sub(txrcv, "NumRcvs", "1")
        rcvp = _sub(txrcv, "RcvParameters")
        _sub(rcvp, "Identifier", "RCV1")
        _sub(rcvp, "WindowLength", f"{float(n_samp) / fs:.12g}")
        _sub(rcvp, "SampleRate", f"{fs:.12g}")
        _sub(rcvp, "IFFilterBW", f"{bw:.12g}")
        _sub(rcvp, "FreqCenter", f"{f0:.12g}")
        _sub(rcvp, "LFMRate", f"{lfm_rate:.12g}")
        _sub(rcvp, "Polarization", meta["rx_pol"])

        # TxRcv references in Channel/Parameters
        txrcv_ref = _sub(ch_params, "TxRcv")
        _sub(txrcv_ref, "TxWFId", "TXW1")
        _sub(txrcv_ref, "RcvId", "RCV1")

        return etree.ElementTree(root)


# ===================================================================
# Internal: merged CPHD vectors
# ===================================================================


class _CPHDVectors:
    """Holds the merged CPHD per-vector parameter arrays."""

    def __init__(
        self,
        *,
        n_vec: int,
        tx_time: np.ndarray,
        tx_pos: np.ndarray,
        tx_vel: np.ndarray,
        rcv_time: np.ndarray,
        rcv_pos: np.ndarray,
        rcv_vel: np.ndarray,
        iarp_ecf: np.ndarray,
        fx1_abs: float,
        fx2_abs: float,
        f0: float,
        fs: float,
        n_samp: int,
        scss: float,
        signal_flags: np.ndarray,
        amp_sf: np.ndarray,
    ):
        self.n_vec = n_vec
        self.tx_time = tx_time
        self.tx_pos = tx_pos
        self.tx_vel = tx_vel
        self.rcv_time = rcv_time
        self.rcv_pos = rcv_pos
        self.rcv_vel = rcv_vel

        # SRP = IARP (constant)
        self.srp_pos = np.tile(iarp_ecf, (n_vec, 1))

        # Frequency parameters
        self.fx1 = np.full(n_vec, fx1_abs)
        self.fx2 = np.full(n_vec, fx2_abs)
        self.sc0 = np.full(n_vec, fx1_abs)
        self.scss = np.full(n_vec, scss)

        # TOA parameters
        toa_saved = (n_samp - 1) / fs if n_samp > 1 else 1.0 / fs
        self.toa1 = np.full(n_vec, -toa_saved / 2.0)
        self.toa2 = np.full(n_vec, toa_saved / 2.0)

        # Tropo delay (no correction applied)
        self.td_tropo = np.zeros(n_vec)

        self.signal_flags = signal_flags
        self.amp_sf = amp_sf

        # Geometry-derived fields
        mid_pos = 0.5 * (tx_pos + rcv_pos)
        mid_vel = 0.5 * (tx_vel + rcv_vel)
        srp_to_arp = mid_pos - self.srp_pos
        r_srp = norm(srp_to_arp, axis=1)
        r_srp = np.where(r_srp == 0, 1.0, r_srp)
        r_hat = srp_to_arp / r_srp[:, np.newaxis]
        v_r = np.sum(mid_vel * r_hat, axis=1)
        wavelength = _C / f0

        self.a_fdop = -2.0 * v_r / wavelength
        self.a_frr1 = _C / (2.0 * fx1_abs) * np.ones(n_vec)
        self.a_frr2 = _C / (2.0 * fx2_abs) * np.ones(n_vec)

    def to_structured(self, dtype: np.dtype) -> np.ndarray:
        """Pack into a sarkit PVP structured array."""
        arr = np.zeros(self.n_vec, dtype=dtype)
        arr["TxTime"] = self.tx_time
        arr["TxPos"] = self.tx_pos
        arr["TxVel"] = self.tx_vel
        arr["RcvTime"] = self.rcv_time
        arr["RcvPos"] = self.rcv_pos
        arr["RcvVel"] = self.rcv_vel
        arr["SRPPos"] = self.srp_pos
        arr["aFDOP"] = self.a_fdop
        arr["aFRR1"] = self.a_frr1
        arr["aFRR2"] = self.a_frr2
        arr["FX1"] = self.fx1
        arr["FX2"] = self.fx2
        arr["TOA1"] = self.toa1
        arr["TOA2"] = self.toa2
        arr["TDTropoSRP"] = self.td_tropo
        arr["SC0"] = self.sc0
        arr["SCSS"] = self.scss
        arr["SIGNAL"] = self.signal_flags
        arr["AmpSF"] = self.amp_sf
        return arr
