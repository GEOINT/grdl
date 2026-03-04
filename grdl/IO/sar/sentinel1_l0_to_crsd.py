# -*- coding: utf-8 -*-
"""
Sentinel-1 Level-0 to CRSD Conversion Pipeline.

Converts Sentinel-1 Level-0 IW SAFE products to NGA CRSD
(Compensated Radar Signal Data) format.  Handles ISP decoding,
FDBAQ decompression, orbit interpolation, and CRSD XML/binary
construction via sarkit.

Dependencies
------------
sentinel1decoder, sarkit, scipy, lxml

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
2026-03-03
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lxml import etree
from scipy.interpolate import CubicSpline

import sarkit.crsd
import sarkit.wgs84
import sentinel1decoder

from grdl.IO.sar.sentinel1_l0 import (
    Sentinel1L0Reader,
    S1_CENTER_FREQUENCY,
)

logger = logging.getLogger(__name__)

# CRSD XML namespace
CRSD_NS = "http://api.nsgreg.nga.mil/schema/crsd/1.0"
NSMAP = {None: CRSD_NS}

# GPS epoch: 6 Jan 1980 00:00:00 UTC
# S1 coarse time reference: 1 Jan 2000 00:00:00 UTC (GPS time)
_GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
_S1_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
_S1_EPOCH_OFFSET = (_S1_EPOCH - _GPS_EPOCH).total_seconds()


# ===================================================================
# XML builder helpers
# ===================================================================

def _sub(parent: etree._Element, tag: str, text: str = None) -> etree._Element:
    """Add a sub-element with optional text."""
    elem = etree.SubElement(parent, f"{{{CRSD_NS}}}{tag}")
    if text is not None:
        elem.text = str(text)
    return elem


def _sub_xyz(parent: etree._Element, tag: str, x: float, y: float, z: float) -> etree._Element:
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


def _sub_latlon(parent: etree._Element, tag: str, lat: float, lon: float) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "Lat", f"{lat:.10g}")
    _sub(elem, "Lon", f"{lon:.10g}")
    return elem


def _sub_latlonhae(parent: etree._Element, tag: str, lat: float, lon: float, hae: float) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "Lat", f"{lat:.10g}")
    _sub(elem, "Lon", f"{lon:.10g}")
    _sub(elem, "HAE", f"{hae:.6g}")
    return elem


def _sub_pxp_f8(parent: etree._Element, name: str, offset: int) -> int:
    """Add an F8 per-parameter field. Returns new offset."""
    elem = _sub(parent, name)
    _sub(elem, "Offset", str(offset))
    _sub(elem, "Size", "1")
    _sub(elem, "Format", "F8")
    return offset + 1


def _sub_pxp_i8(parent: etree._Element, name: str, offset: int) -> int:
    """Add an I8 per-parameter field. Returns new offset."""
    elem = _sub(parent, name)
    _sub(elem, "Offset", str(offset))
    _sub(elem, "Size", "1")
    _sub(elem, "Format", "I8")
    return offset + 1


def _sub_pxp_intfrac(parent: etree._Element, name: str, offset: int) -> int:
    """Add IntFrac (I8+F8) per-parameter field. Returns new offset."""
    elem = _sub(parent, name)
    _sub(elem, "Offset", str(offset))
    _sub(elem, "Size", "2")
    _sub(elem, "Format", "Int=I8;Frac=F8;")
    return offset + 2


def _sub_pxp_xyz(parent: etree._Element, name: str, offset: int) -> int:
    """Add XYZ per-parameter field (X=F8;Y=F8;Z=F8). Returns new offset."""
    elem = _sub(parent, name)
    _sub(elem, "Offset", str(offset))
    _sub(elem, "Size", "3")
    _sub(elem, "Format", "X=F8;Y=F8;Z=F8;")
    return offset + 3


def _sub_pxp_eb(parent: etree._Element, name: str, offset: int) -> int:
    """Add EB per-parameter field (DCX=F8;DCY=F8). Returns new offset."""
    elem = _sub(parent, name)
    _sub(elem, "Offset", str(offset))
    _sub(elem, "Size", "2")
    _sub(elem, "Format", "DCX=F8;DCY=F8;")
    return offset + 2


# ===================================================================
# Orbit interpolator
# ===================================================================

class _OrbitInterpolator:
    """Cubic spline interpolation of satellite orbit from ephemeris.

    Parameters
    ----------
    ephemeris : pd.DataFrame
        sentinel1decoder ``Level0File.ephemeris`` DataFrame.
    """

    def __init__(self, ephemeris) -> None:
        # Deduplicate by timestamp
        eph = ephemeris.drop_duplicates(
            subset=["POD Solution Data Timestamp"]
        ).sort_values("POD Solution Data Timestamp").reset_index(drop=True)

        if len(eph) < 4:
            raise ValueError(
                f"Need at least 4 unique orbit state vectors, got {len(eph)}"
            )

        self._times = eph["POD Solution Data Timestamp"].values.astype(np.float64)
        self._pos = np.column_stack([
            eph["X-axis position ECEF"].values.astype(np.float64),
            eph["Y-axis position ECEF"].values.astype(np.float64),
            eph["Z-axis position ECEF"].values.astype(np.float64),
        ])
        self._vel = np.column_stack([
            eph["X-axis velocity ECEF"].values.astype(np.float64),
            eph["Y-axis velocity ECEF"].values.astype(np.float64),
            eph["Z-axis velocity ECEF"].values.astype(np.float64),
        ])

        self._pos_spline = CubicSpline(self._times, self._pos)
        self._vel_spline = CubicSpline(self._times, self._vel)

    def interpolate(
        self, times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate position and velocity at given times.

        Parameters
        ----------
        times : array_like
            GPS seconds since 1 Jan 2000.

        Returns
        -------
        positions : ndarray, shape (N, 3)
            ECF positions in meters.
        velocities : ndarray, shape (N, 3)
            ECF velocities in m/s.
        """
        times = np.asarray(times, dtype=np.float64)
        pos = self._pos_spline(times)
        vel = self._vel_spline(times)
        return pos, vel

    @property
    def time_range(self) -> Tuple[float, float]:
        """Time range covered by the ephemeris data."""
        return float(self._times[0]), float(self._times[-1])


# ===================================================================
# Antenna frame computation
# ===================================================================

def _compute_antenna_frame(
    pos: np.ndarray, vel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute antenna coordinate frame unit vectors (ACX, ACY).

    For Sentinel-1, ACX is along-track (velocity direction) and
    ACY is cross-track (completing the right-handed frame with
    the nadir-pointing Z axis).

    Parameters
    ----------
    pos : (..., 3) ndarray
        ECF position in meters.
    vel : (..., 3) ndarray
        ECF velocity in m/s.

    Returns
    -------
    acx : (..., 3) ndarray
        Along-track unit vector.
    acy : (..., 3) ndarray
        Cross-track unit vector.
    """
    # Velocity direction (along-track)
    v_norm = np.linalg.norm(vel, axis=-1, keepdims=True)
    v_norm = np.where(v_norm == 0, 1.0, v_norm)
    acx = vel / v_norm

    # Nadir direction (down)
    p_norm = np.linalg.norm(pos, axis=-1, keepdims=True)
    p_norm = np.where(p_norm == 0, 1.0, p_norm)
    nadir = -pos / p_norm

    # Cross-track = nadir x along-track
    acy = np.cross(nadir, acx)
    acy_norm = np.linalg.norm(acy, axis=-1, keepdims=True)
    acy_norm = np.where(acy_norm == 0, 1.0, acy_norm)
    acy = acy / acy_norm

    return acx, acy


# ===================================================================
# Internal: per-swath-pol channel descriptor
# ===================================================================

@dataclass
class _SwathChannel:
    """Metadata for one swath+polarization combination during conversion."""

    key: str                    # e.g. "IW1_VV"
    swath_number: int           # 1, 2, or 3
    polarization: str           # "VV", "VH", etc.
    tx_pol: str
    rx_pol: str
    num_vectors: int            # echo packets in this swath
    num_samples: int            # 2 * max_num_quads for this swath
    echo_df: pd.DataFrame       # filtered to this swath
    global_indices: np.ndarray  # row indices into full echo_df → TxPulseIndex


# ===================================================================
# Conversion pipeline
# ===================================================================

class Sentinel1L0ToCRSD:
    """Convert Sentinel-1 Level-0 IW SAFE to NGA CRSD format.

    Each IW sub-swath + polarization combination becomes a distinct
    CRSD receive channel (e.g. ``IW1_VV``, ``IW2_VH``), so that each
    channel has uniform signal dimensions (no cross-swath zero-padding).

    Parameters
    ----------
    safe_path : str or Path
        Path to the ``.SAFE`` directory.
    output_path : str or Path
        Output path for the ``.crsd`` file.
    orbit_file : str or Path, optional
        Path to an ESA precise orbit file (EOF XML). If not provided,
        orbit is extracted from ISP sub-commutated telemetry.
    channel : str, optional
        Polarization to convert (e.g. ``"VV"``, ``"VH"``).
        If *None* (default), all polarizations are included.
    swath : int, optional
        Sub-swath number to convert (1, 2, or 3 for IW mode).
        If *None* (default), all sub-swaths are included.

    Examples
    --------
    >>> converter = Sentinel1L0ToCRSD(
    ...     safe_path='S1C_IW_RAW__0SDV_....SAFE',
    ...     output_path='output.crsd',
    ... )
    >>> converter.convert()

    >>> # Single swath + polarization
    >>> Sentinel1L0ToCRSD(
    ...     safe_path='S1C_IW_RAW__0SDV_....SAFE',
    ...     output_path='output.IW1_VV.crsd',
    ...     channel='VV',
    ...     swath=1,
    ... ).convert()
    """

    def __init__(
        self,
        safe_path: Union[str, Path],
        output_path: Union[str, Path],
        orbit_file: Optional[Union[str, Path]] = None,
        channel: Optional[str] = None,
        swath: Optional[int] = None,
    ) -> None:
        self.safe_path = Path(safe_path)
        self.output_path = Path(output_path)
        self.orbit_file = Path(orbit_file) if orbit_file else None
        self.channel = channel
        self.swath = swath

    def convert(self) -> Path:
        """Run the full conversion pipeline.

        Returns
        -------
        Path
            Path to the output CRSD file.
        """
        logger.info("Opening Sentinel-1 Level-0: %s", self.safe_path)

        # Step 1: Open L0 reader
        reader = Sentinel1L0Reader(self.safe_path)
        meta = reader.metadata
        all_channels = meta.channels
        radar = meta.radar_params
        product = meta.product_info

        # Filter to requested polarization (default: all)
        if self.channel is not None:
            if self.channel not in all_channels:
                avail = list(all_channels.keys())
                raise ValueError(
                    f"Channel '{self.channel}' not found. Available: {avail}"
                )
            pol_channels = {self.channel: all_channels[self.channel]}
        else:
            pol_channels = all_channels

        logger.info(
            "Product: %s %s, polarizations: %s",
            product.mission, product.mode,
            list(pol_channels.keys()),
        )

        # Step 2: Build orbit interpolator
        orbit = self._build_orbit_interpolator(pol_channels)

        # Step 3: Compute scene geometry
        iarp_llh, iarp_ecf = self._compute_iarp(meta.footprint)
        ref_time_s1, collection_ref_dt = self._compute_ref_time(reader)

        # Step 4: Build swath channel descriptors
        # Get full echo metadata per polarization (all swaths)
        full_echo_meta = {}
        for pol in pol_channels:
            full_echo_meta[pol] = reader.get_channel_echo_metadata(pol)

        # Build per-swath+pol channel descriptors
        swath_channels = {}  # key -> _SwathChannel
        for pol, ch_info in pol_channels.items():
            swath_info = reader.get_swath_info(pol)
            full_df = full_echo_meta[pol]

            for sn in sorted(swath_info.keys()):
                if self.swath is not None and sn != self.swath:
                    continue
                si = swath_info[sn]
                key = si.channel_key  # e.g. "IW1_VV"

                # Filter echo_df using raw ISP swath number
                raw_sn = si.raw_swath_number
                mask = full_df["Swath Number"].values == raw_sn
                swath_df = full_df[mask].reset_index(drop=True)
                global_indices = np.where(mask)[0]

                swath_channels[key] = _SwathChannel(
                    key=key,
                    swath_number=sn,
                    polarization=pol,
                    tx_pol=ch_info.tx_pol,
                    rx_pol=ch_info.rx_pol,
                    num_vectors=len(swath_df),
                    num_samples=2 * si.max_num_quads,
                    echo_df=swath_df,
                    global_indices=global_indices,
                )

        if not swath_channels:
            all_swaths = set()
            for si_dict in (reader.get_swath_info(p) for p in pol_channels):
                all_swaths.update(si_dict.keys())
            raise ValueError(
                f"No swath channels to convert. "
                f"Requested swath={self.swath}, "
                f"available logical swaths: {sorted(all_swaths)}"
            )

        logger.info(
            "Swath channels: %s", list(swath_channels.keys()),
        )

        # Step 5: Derive timing bounds
        timing = self._compute_timing_bounds(
            swath_channels, full_echo_meta, ref_time_s1, radar,
        )

        # Step 6: Build CRSD XML
        xmltree = self._build_crsd_xml(
            meta, swath_channels, full_echo_meta, radar, orbit,
            iarp_llh, iarp_ecf, collection_ref_dt, ref_time_s1, timing,
        )

        # Step 7-11: Build arrays and write CRSD
        self._write_crsd(
            xmltree, reader, swath_channels, full_echo_meta,
            radar, orbit, ref_time_s1,
        )

        reader.close()
        logger.info("CRSD written to: %s", self.output_path)
        return self.output_path

    # ---------------------------------------------------------------
    # Step 2: Orbit
    # ---------------------------------------------------------------

    def _build_orbit_interpolator(
        self, channels: Dict,
    ) -> _OrbitInterpolator:
        """Build orbit interpolator from ISP sub-commutated telemetry."""
        if self.orbit_file is not None:
            return self._load_orbit_from_eof(self.orbit_file)

        # Use sentinel1decoder Level0File ephemeris
        first_ch = next(iter(channels.values()))
        l0file = sentinel1decoder.Level0File(first_ch.measurement_file)
        eph = l0file.ephemeris

        if eph is None or len(eph) == 0:
            raise ValueError(
                "No orbit data found in ISP sub-commutated telemetry"
            )

        logger.info(
            "Extracted %d orbit state vectors from ISP telemetry",
            len(eph),
        )
        return _OrbitInterpolator(eph)

    def _load_orbit_from_eof(self, eof_path: Path) -> _OrbitInterpolator:
        """Load orbit from ESA EOF XML file."""
        import pandas as pd

        tree = etree.parse(str(eof_path))
        root = tree.getroot()

        records = []
        for osv in root.iter():
            tag = osv.tag.split("}")[-1] if "}" in osv.tag else osv.tag
            if tag == "OSV":
                utc_text = None
                x = y = z = vx = vy = vz = 0.0
                for child in osv:
                    ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if ctag == "UTC":
                        utc_text = child.text
                    elif ctag == "X":
                        x = float(child.text)
                    elif ctag == "Y":
                        y = float(child.text)
                    elif ctag == "Z":
                        z = float(child.text)
                    elif ctag == "VX":
                        vx = float(child.text)
                    elif ctag == "VY":
                        vy = float(child.text)
                    elif ctag == "VZ":
                        vz = float(child.text)
                if utc_text:
                    # Parse UTC string: "UTC=2026-02-18T15:27:03.121269"
                    utc_str = utc_text.replace("UTC=", "").strip()
                    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
                    gps_s = (dt - _S1_EPOCH).total_seconds()
                    records.append({
                        "POD Solution Data Timestamp": gps_s,
                        "X-axis position ECEF": x,
                        "Y-axis position ECEF": y,
                        "Z-axis position ECEF": z,
                        "X-axis velocity ECEF": vx,
                        "Y-axis velocity ECEF": vy,
                        "Z-axis velocity ECEF": vz,
                    })

        if not records:
            raise ValueError(f"No OSV records found in {eof_path}")

        eph = pd.DataFrame(records)
        logger.info("Loaded %d orbit state vectors from %s", len(eph), eof_path)
        return _OrbitInterpolator(eph)

    # ---------------------------------------------------------------
    # Step 3: Scene geometry
    # ---------------------------------------------------------------

    def _compute_iarp(
        self, footprint: List,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute IARP as footprint centroid.

        Returns (lat_lon_hae, ecf) arrays.
        """
        if not footprint:
            raise ValueError("No footprint coordinates in manifest")

        # Use unique corners (remove closing duplicate)
        coords = [(c.lat, c.lon) for c in footprint]
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]

        mean_lat = np.mean([c[0] for c in coords])
        mean_lon = np.mean([c[1] for c in coords])
        hae = 0.0  # Assume sea level

        iarp_llh = np.array([mean_lat, mean_lon, hae])
        iarp_ecf = sarkit.wgs84.geodetic_to_cartesian(iarp_llh)

        return iarp_llh, iarp_ecf

    def _compute_ref_time(
        self, reader: Sentinel1L0Reader,
    ) -> Tuple[float, datetime]:
        """Compute collection reference time (mid-collection).

        Returns (gps_seconds_since_s1_epoch, datetime_utc).
        """
        first_ch = next(iter(reader.metadata.channels.keys()))
        echo_meta = reader.get_channel_echo_metadata(first_ch)

        coarse = echo_meta["Coarse Time"].values.astype(np.float64)
        fine = echo_meta["Fine Time"].values.astype(np.float64)
        sensing_times = coarse + fine

        t_start = sensing_times.min()
        t_end = sensing_times.max()
        t_mid = (t_start + t_end) / 2.0

        # Convert to datetime (coarse time is GPS seconds since GPS epoch)
        ref_dt = _GPS_EPOCH + __import__("datetime").timedelta(seconds=float(t_mid))

        return float(t_mid), ref_dt

    # ---------------------------------------------------------------
    # Step 5: Timing bounds
    # ---------------------------------------------------------------

    def _compute_timing_bounds(
        self,
        swath_channels: Dict[str, _SwathChannel],
        full_echo_meta: Dict[str, pd.DataFrame],
        ref_time: float,
        radar,
    ) -> Dict:
        """Compute global and per-channel timing and frequency bounds."""
        # Global TX timing from full (all-swaths) echo metadata
        all_coarse = []
        all_fine = []
        all_swst = []

        for pol, echo_df in full_echo_meta.items():
            all_coarse.append(echo_df["Coarse Time"].values.astype(np.float64))
            all_fine.append(echo_df["Fine Time"].values.astype(np.float64))
            all_swst.append(echo_df["SWST"].values.astype(np.float64))

        coarse = np.concatenate(all_coarse)
        fine = np.concatenate(all_fine)
        swst = np.concatenate(all_swst)

        sensing = coarse + fine
        bw = radar.tx_bandwidth
        f0 = radar.center_frequency

        # Per-channel receive timing
        per_channel = {}
        for key, sc in swath_channels.items():
            ch_coarse = sc.echo_df["Coarse Time"].values.astype(np.float64)
            ch_fine = sc.echo_df["Fine Time"].values.astype(np.float64)
            ch_swst = sc.echo_df["SWST"].values.astype(np.float64)
            ch_sensing = ch_coarse + ch_fine
            per_channel[key] = {
                "rcv_start_time1": float((ch_sensing + ch_swst).min() - ref_time),
                "rcv_start_time2": float((ch_sensing + ch_swst).max() - ref_time),
            }

        return {
            "tx_time1": float(sensing.min() - ref_time),
            "tx_time2": float(sensing.max() - ref_time),
            "fx_min": f0 - bw / 2,
            "fx_max": f0 + bw / 2,
            "rcv_start_time1": float((sensing + swst).min() - ref_time),
            "rcv_start_time2": float((sensing + swst).max() - ref_time),
            "frcv_min": f0 - bw / 2,
            "frcv_max": f0 + bw / 2,
            "per_channel": per_channel,
        }

    # ---------------------------------------------------------------
    # Step 6: Build CRSD XML
    # ---------------------------------------------------------------

    def _build_crsd_xml(
        self, meta, swath_channels, full_echo_meta, radar, orbit,
        iarp_llh, iarp_ecf, collection_ref_dt, ref_time, timing,
    ) -> etree.ElementTree:
        """Build the complete CRSDsar XML tree."""
        product = meta.product_info
        root = etree.Element(f"{{{CRSD_NS}}}CRSDsar", nsmap=NSMAP)

        # Determine TX polarization from first swath channel
        first_sc = next(iter(swath_channels.values()))
        tx_pol = first_sc.tx_pol  # 'V' or 'H'

        # --- ProductInfo ---
        pi = _sub(root, "ProductInfo")
        _sub(pi, "ProductName", self.safe_path.name)
        _sub(pi, "Classification", "UNCLASSIFIED")
        _sub(pi, "ReleaseInfo", "PUBLIC RELEASE")
        ci = _sub(pi, "CreationInfo")
        _sub(ci, "Application", "grdl-sentinel1-l0-to-crsd")
        _sub(ci, "DateTime", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

        # --- SARInfo ---
        si = _sub(root, "SARInfo")
        _sub(si, "CollectType", "MONOSTATIC")
        rm = _sub(si, "RadarMode")
        _sub(rm, "ModeType", "DYNAMIC STRIPMAP")
        _sub(rm, "ModeID", product.mode or "IW")

        # --- TransmitInfo ---
        ti = _sub(root, "TransmitInfo")
        _sub(ti, "SensorName", product.mission or "SENTINEL-1")
        _sub(ti, "EventName", f"DT-{meta.data_take_id or 0:08X}")

        # --- ReceiveInfo ---
        ri = _sub(root, "ReceiveInfo")
        _sub(ri, "SensorName", product.mission or "SENTINEL-1")
        _sub(ri, "EventName", f"DT-{meta.data_take_id or 0:08X}")

        # --- Global ---
        gl = _sub(root, "Global")
        _sub(gl, "CollectionRefTime", collection_ref_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        gl_tx = _sub(gl, "Transmit")
        _sub(gl_tx, "TxTime1", f"{timing['tx_time1']:.12g}")
        _sub(gl_tx, "TxTime2", f"{timing['tx_time2']:.12g}")
        _sub(gl_tx, "FxMin", f"{timing['fx_min']:.12g}")
        _sub(gl_tx, "FxMax", f"{timing['fx_max']:.12g}")
        gl_rcv = _sub(gl, "Receive")
        _sub(gl_rcv, "RcvStartTime1", f"{timing['rcv_start_time1']:.12g}")
        _sub(gl_rcv, "RcvStartTime2", f"{timing['rcv_start_time2']:.12g}")
        _sub(gl_rcv, "FrcvMin", f"{timing['frcv_min']:.12g}")
        _sub(gl_rcv, "FrcvMax", f"{timing['frcv_max']:.12g}")

        # --- SceneCoordinates ---
        sc = _sub(root, "SceneCoordinates")
        _sub(sc, "EarthModel", "WGS_84")
        iarp_elem = _sub(sc, "IARP")
        _sub_xyz(iarp_elem, "ECF", iarp_ecf[0], iarp_ecf[1], iarp_ecf[2])
        _sub_latlonhae(iarp_elem, "LLH", iarp_llh[0], iarp_llh[1], iarp_llh[2])
        ref_surf = _sub(sc, "ReferenceSurface")
        _sub(ref_surf, "Planar")
        u_east = sarkit.wgs84.east(iarp_llh)
        u_north = sarkit.wgs84.north(iarp_llh)
        planar = ref_surf.find(f"{{{CRSD_NS}}}Planar")
        _sub_xyz(planar, "uIAX", u_east[0], u_east[1], u_east[2])
        _sub_xyz(planar, "uIAY", u_north[0], u_north[1], u_north[2])

        # ImageArea (required) — use largest swath channel
        num_lines = max(sch.num_vectors for sch in swath_channels.values())
        num_samples = max(sch.num_samples for sch in swath_channels.values())
        ia = _sub(sc, "ImageArea")
        _sub_xy(ia, "X1Y1", 0.0, 0.0)
        _sub_xy(ia, "X2Y2", float(num_lines - 1), float(num_samples - 1))
        polygon = _sub(ia, "Polygon")
        polygon.set("size", "4")
        corners = [
            (0.0, 0.0),
            (float(num_lines - 1), 0.0),
            (float(num_lines - 1), float(num_samples - 1)),
            (0.0, float(num_samples - 1)),
        ]
        for vi, (cx, cy) in enumerate(corners, start=1):
            v = _sub_xy(polygon, "Vertex", cx, cy)
            v.set("index", str(vi))

        # ImageAreaCornerPoints
        if meta.footprint:
            fp = meta.footprint
            unique_fp = fp[:-1] if len(fp) > 1 and fp[0].lat == fp[-1].lat and fp[0].lon == fp[-1].lon else fp
            iacp = _sub(sc, "ImageAreaCornerPoints")
            for idx, coord in enumerate(unique_fp[:4], start=1):
                cp = _sub_latlon(iacp, "IACP", coord.lat, coord.lon)
                cp.set("index", str(idx))

        # --- Data ---
        self._build_data_section(root, swath_channels, full_echo_meta, radar)

        # --- TxSequence ---
        self._build_tx_sequence(root, radar, timing, tx_pol, iarp_ecf)

        # --- Channel ---
        self._build_channel_section(root, swath_channels, radar, timing, iarp_ecf)

        # --- ReferenceGeometry (placeholder, computed after PVP/PPP) ---
        rg = _sub(root, "ReferenceGeometry")
        rp = _sub(rg, "RefPoint")
        _sub_xyz(rp, "ECF", iarp_ecf[0], iarp_ecf[1], iarp_ecf[2])
        _sub_xy(rp, "IAC", 0.0, 0.0)
        # SARImage, TxParameters, RcvParameters will be filled by sarkit

        # --- SupportArray ---
        self._build_support_array_section(root, radar)

        # --- PPP ---
        self._build_ppp_section(root)

        # --- PVP ---
        self._build_pvp_section(root)

        # --- Antenna ---
        self._build_antenna_section(root, swath_channels, tx_pol)

        tree = etree.ElementTree(root)
        return tree

    def _build_data_section(self, root, swath_channels, full_echo_meta, radar):
        """Build the Data section (binary data parameters)."""
        data = _sub(root, "Data")

        # Support arrays: GP1 (GainPhase), FXR1 (FxResponse), DTA1 (DwellTime)
        support = _sub(data, "Support")
        _sub(support, "NumSupportArrays", "3")

        sa1 = _sub(support, "SupportArray")
        _sub(sa1, "SAId", "GP1")
        _sub(sa1, "NumRows", "1")
        _sub(sa1, "NumCols", "1")
        _sub(sa1, "BytesPerElement", "8")
        _sub(sa1, "ArrayByteOffset", "0")

        sa2 = _sub(support, "SupportArray")
        _sub(sa2, "SAId", "FXR1")
        _sub(sa2, "NumRows", "2")
        _sub(sa2, "NumCols", "1")
        _sub(sa2, "BytesPerElement", "8")
        _sub(sa2, "ArrayByteOffset", "8")

        sa3 = _sub(support, "SupportArray")
        _sub(sa3, "SAId", "DTA1")
        _sub(sa3, "NumRows", "1")
        _sub(sa3, "NumCols", "1")
        _sub(sa3, "BytesPerElement", "8")
        _sub(sa3, "ArrayByteOffset", "24")

        # Transmit — NumPulses = total echo packets for first pol (all swaths)
        first_pol = next(iter(full_echo_meta.keys()))
        total_pulses = len(full_echo_meta[first_pol])

        tx = _sub(data, "Transmit")
        _sub(tx, "NumBytesPPP", str(self._ppp_num_bytes()))
        _sub(tx, "NumTxSequences", "1")
        txs = _sub(tx, "TxSequence")
        _sub(txs, "TxId", "TX1")
        _sub(txs, "NumPulses", str(total_pulses))
        _sub(txs, "PPPArrayByteOffset", "0")

        # Receive — one channel per swath+pol
        rcv = _sub(data, "Receive")
        _sub(rcv, "SignalArrayFormat", "CF8")
        _sub(rcv, "NumBytesPVP", str(self._pvp_num_bytes()))
        _sub(rcv, "NumCRSDChannels", str(len(swath_channels)))
        sig_offset = 0
        pvp_offset = 0
        for key, sc in swath_channels.items():
            ch_elem = _sub(rcv, "Channel")
            _sub(ch_elem, "ChId", key)
            _sub(ch_elem, "NumVectors", str(sc.num_vectors))
            _sub(ch_elem, "NumSamples", str(sc.num_samples))
            _sub(ch_elem, "SignalArrayByteOffset", str(sig_offset))
            _sub(ch_elem, "PVPArrayByteOffset", str(pvp_offset))
            sig_offset += sc.num_vectors * sc.num_samples * 8  # CF8 = 8 bytes
            pvp_offset += sc.num_vectors * self._pvp_num_bytes()

    def _build_tx_sequence(self, root, radar, timing, tx_pol, iarp_ecf):
        """Build the TxSequence section."""
        txseq = _sub(root, "TxSequence")
        _sub(txseq, "RefTxId", "TX1")
        _sub(txseq, "TxWFType", "LFM")
        params = _sub(txseq, "Parameters")
        _sub(params, "Identifier", "TX1")
        _sub(params, "RefPulseIndex", "0")
        _sub(params, "FxResponseId", "FXR1")
        _sub(params, "FxBWFixed", "true")
        _sub(params, "FxC", f"{radar.center_frequency:.12g}")
        _sub(params, "FxBW", f"{radar.tx_bandwidth:.12g}")
        _sub(params, "TXmtMin", f"{radar.tx_pulse_length:.12g}")
        _sub(params, "TXmtMax", f"{radar.tx_pulse_length:.12g}")
        _sub(params, "TxTime1", f"{timing['tx_time1']:.12g}")
        _sub(params, "TxTime2", f"{timing['tx_time2']:.12g}")
        _sub(params, "TxAPCId", "APC_TX")
        _sub(params, "TxAPATId", "APAT_TX")

        rp = _sub(params, "TxRefPoint")
        _sub_xyz(rp, "ECF", iarp_ecf[0], iarp_ecf[1], iarp_ecf[2])
        _sub_xy(rp, "IAC", 0.0, 0.0)

        pol_elem = _sub(params, "TxPolarization")
        _sub(pol_elem, "PolarizationID", tx_pol)
        _sub(pol_elem, "AmpH", "0" if tx_pol == "V" else "1")
        _sub(pol_elem, "AmpV", "1" if tx_pol == "V" else "0")
        _sub(pol_elem, "PhaseH", "0")
        _sub(pol_elem, "PhaseV", "0")

        _sub(params, "TxRefRadIntensity", "1.0")
        _sub(params, "TxRadIntErrorStdDev", "0.0")
        _sub(params, "TxRefLAtm", "0.0")

    def _build_channel_section(self, root, swath_channels, radar, timing, iarp_ecf):
        """Build the Channel section (receive channel parameters)."""
        ch_root = _sub(root, "Channel")
        first_key = next(iter(swath_channels.keys()))
        _sub(ch_root, "RefChId", first_key)

        per_ch_timing = timing.get("per_channel", {})

        for key, sc in swath_channels.items():
            ch_timing = per_ch_timing.get(key, timing)

            params = _sub(ch_root, "Parameters")
            _sub(params, "Identifier", key)
            _sub(params, "RefVectorIndex", "0")
            _sub(params, "RefFreqFixed", "true")
            _sub(params, "FrcvFixed", "true")
            _sub(params, "SignalNormal", "true")
            _sub(params, "F0Ref", f"{radar.center_frequency:.12g}")
            _sub(params, "Fs", f"{radar.sampling_rate:.12g}")
            _sub(params, "BWInst", f"{radar.tx_bandwidth:.12g}")
            _sub(params, "RcvStartTime1", f"{ch_timing['rcv_start_time1']:.12g}")
            _sub(params, "RcvStartTime2", f"{ch_timing['rcv_start_time2']:.12g}")
            _sub(params, "FrcvMin", f"{timing['frcv_min']:.12g}")
            _sub(params, "FrcvMax", f"{timing['frcv_max']:.12g}")
            _sub(params, "RcvAPCId", "APC_RX")
            _sub(params, "RcvAPATId", f"APAT_RX_{key}")

            rp = _sub(params, "RcvRefPoint")
            _sub_xyz(rp, "ECF", iarp_ecf[0], iarp_ecf[1], iarp_ecf[2])
            _sub_xy(rp, "IAC", 0.0, 0.0)

            rpol = _sub(params, "RcvPolarization")
            _sub(rpol, "PolarizationID", sc.rx_pol)
            _sub(rpol, "AmpH", "0" if sc.rx_pol == "V" else "1")
            _sub(rpol, "AmpV", "1" if sc.rx_pol == "V" else "0")
            _sub(rpol, "PhaseH", "0")
            _sub(rpol, "PhaseV", "0")

            _sub(params, "RcvRefIrradiance", "1.0")
            _sub(params, "RcvIrradianceErrorStdDev", "0.0")
            _sub(params, "RcvRefLAtm", "0.0")
            _sub(params, "PNCRSD", "0.0")
            _sub(params, "BNCRSD", "1.0")

            # SARImage
            sar_img = _sub(params, "SARImage")
            _sub(sar_img, "TxId", "TX1")
            _sub(sar_img, "RefVectorPulseIndex", "0")

            sar_tx_pol = _sub(sar_img, "TxPolarization")
            _sub(sar_tx_pol, "PolarizationID", sc.tx_pol)
            _sub(sar_tx_pol, "AmpH", "0" if sc.tx_pol == "V" else "1")
            _sub(sar_tx_pol, "AmpV", "1" if sc.tx_pol == "V" else "0")
            _sub(sar_tx_pol, "PhaseH", "0")
            _sub(sar_tx_pol, "PhaseV", "0")

            dwell = _sub(sar_img, "DwellTimes")
            dta = _sub(dwell, "Array")
            _sub(dta, "DTAId", "DTA1")

            ia = _sub(sar_img, "ImageArea")
            _sub_xy(ia, "X1Y1", -1.0, -1.0)
            _sub_xy(ia, "X2Y2", 1.0, 1.0)
            polygon = _sub(ia, "Polygon")
            polygon.set("size", "4")
            for vi, (vx, vy) in enumerate(
                [(-1, -1), (1, -1), (1, 1), (-1, 1)], start=1,
            ):
                v = _sub_xy(polygon, "Vertex", float(vx), float(vy))
                v.set("index", str(vi))

    def _build_support_array_section(self, root, radar):
        """Build the SupportArray section."""
        sa = _sub(root, "SupportArray")

        f0 = radar.center_frequency
        bw = radar.tx_bandwidth

        # GainPhaseArray (must come first per XSD sequence)
        gpa = _sub(sa, "GainPhaseArray")
        _sub(gpa, "Identifier", "GP1")
        _sub(gpa, "ElementFormat", "Gain=F4;Phase=F4;")
        _sub(gpa, "X0", "0.0")
        _sub(gpa, "Y0", "-1.0")
        _sub(gpa, "XSS", "1.0")
        _sub(gpa, "YSS", "2.0")

        # FxResponseArray (second)
        fxr = _sub(sa, "FxResponseArray")
        _sub(fxr, "Identifier", "FXR1")
        _sub(fxr, "ElementFormat", "Amp=F4;Phase=F4;")
        _sub(fxr, "Fx0FXR", f"{f0 - bw/2:.12g}")
        _sub(fxr, "FxSSFXR", f"{bw:.12g}")

        # DwellTimeArray (optional, needed for SARImage DTA reference)
        dta = _sub(sa, "DwellTimeArray")
        _sub(dta, "Identifier", "DTA1")
        _sub(dta, "ElementFormat", "COD=F4;DT=F4;")
        _sub(dta, "X0", "0.0")
        _sub(dta, "Y0", "0.0")
        _sub(dta, "XSS", "1.0")
        _sub(dta, "YSS", "1.0")

    def _build_ppp_section(self, root):
        """Build PPP (Per-Pulse Parameters) field definitions."""
        ppp = _sub(root, "PPP")
        off = 0
        off = _sub_pxp_intfrac(ppp, "TxTime", off)
        off = _sub_pxp_xyz(ppp, "TxPos", off)
        off = _sub_pxp_xyz(ppp, "TxVel", off)
        off = _sub_pxp_f8(ppp, "FX1", off)
        off = _sub_pxp_f8(ppp, "FX2", off)
        off = _sub_pxp_f8(ppp, "TXmt", off)
        off = _sub_pxp_intfrac(ppp, "PhiX0", off)
        off = _sub_pxp_f8(ppp, "FxFreq0", off)
        off = _sub_pxp_f8(ppp, "FxRate", off)
        off = _sub_pxp_f8(ppp, "TxRadInt", off)
        off = _sub_pxp_xyz(ppp, "TxACX", off)
        off = _sub_pxp_xyz(ppp, "TxACY", off)
        off = _sub_pxp_eb(ppp, "TxEB", off)
        off = _sub_pxp_i8(ppp, "FxResponseIndex", off)
        # Total: 2+3+3+1+1+1+2+1+1+1+3+3+2+1 = 25 doubles

    def _build_pvp_section(self, root):
        """Build PVP (Per-Vector Parameters) field definitions."""
        pvp = _sub(root, "PVP")
        off = 0
        off = _sub_pxp_intfrac(pvp, "RcvStart", off)
        off = _sub_pxp_xyz(pvp, "RcvPos", off)
        off = _sub_pxp_xyz(pvp, "RcvVel", off)
        off = _sub_pxp_f8(pvp, "FRCV1", off)
        off = _sub_pxp_f8(pvp, "FRCV2", off)
        off = _sub_pxp_intfrac(pvp, "RefPhi0", off)
        off = _sub_pxp_f8(pvp, "RefFreq", off)
        off = _sub_pxp_f8(pvp, "DFIC0", off)
        off = _sub_pxp_f8(pvp, "FICRate", off)
        off = _sub_pxp_xyz(pvp, "RcvACX", off)
        off = _sub_pxp_xyz(pvp, "RcvACY", off)
        off = _sub_pxp_eb(pvp, "RcvEB", off)
        off = _sub_pxp_i8(pvp, "SIGNAL", off)
        off = _sub_pxp_f8(pvp, "AmpSF", off)
        off = _sub_pxp_f8(pvp, "DGRGC", off)
        off = _sub_pxp_i8(pvp, "TxPulseIndex", off)
        # Total: 2+3+3+1+1+2+1+1+1+3+3+2+1+1+1+1 = 27 doubles

    def _build_antenna_section(self, root, swath_channels, tx_pol):
        """Build Antenna section with placeholder patterns."""
        ant = _sub(root, "Antenna")

        # Coordinate frame
        _sub(ant, "NumACFs", "1")
        _sub(ant, "NumAPCs", "2")
        num_apats = 1 + len(swath_channels)
        _sub(ant, "NumAPATs", str(num_apats))

        acf = _sub(ant, "AntCoordFrame")
        _sub(acf, "Identifier", "ACF1")

        # Phase centers (co-located for monostatic)
        apc_tx = _sub(ant, "AntPhaseCenter")
        _sub(apc_tx, "Identifier", "APC_TX")
        _sub(apc_tx, "ACFId", "ACF1")
        _sub_xyz(apc_tx, "APCXYZ", 0.0, 0.0, 0.0)

        apc_rx = _sub(ant, "AntPhaseCenter")
        _sub(apc_rx, "Identifier", "APC_RX")
        _sub(apc_rx, "ACFId", "ACF1")
        _sub_xyz(apc_rx, "APCXYZ", 0.0, 0.0, 0.0)

        # Antenna patterns (placeholder flat)
        def _add_pattern(ident, gp_id):
            ap = _sub(ant, "AntPattern")
            _sub(ap, "Identifier", ident)
            _sub(ap, "FreqZero", f"{S1_CENTER_FREQUENCY:.12g}")
            _sub(ap, "ArrayGPId", gp_id)
            _sub(ap, "ElemGPId", gp_id)
            ebfs = _sub(ap, "EBFreqShift")
            _sub(ebfs, "DCXSF", "0.0")
            _sub(ebfs, "DCYSF", "0.0")
            mlfd = _sub(ap, "MLFreqDilation")
            _sub(mlfd, "DCXSF", "0.0")
            _sub(mlfd, "DCYSF", "0.0")
            gbe = _sub(ap, "GainBSPoly")
            gbe.set("order1", "0")
            coef = _sub(gbe, "Coef", "0.0")
            coef.set("exponent1", "0")
            # AntPolRef (required)
            apr = _sub(ap, "AntPolRef")
            _sub(apr, "AmpX", "1.0")
            _sub(apr, "AmpY", "0.0")
            _sub(apr, "PhaseX", "0.0")
            _sub(apr, "PhaseY", "0.0")

        _add_pattern("APAT_TX", "GP1")
        for key in swath_channels:
            _add_pattern(f"APAT_RX_{key}", "GP1")

    # ---------------------------------------------------------------
    # PPP/PVP byte size helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _ppp_num_bytes() -> int:
        """Number of bytes per PPP record (25 doubles * 8 bytes)."""
        return 25 * 8

    @staticmethod
    def _pvp_num_bytes() -> int:
        """Number of bytes per PVP record (27 doubles * 8 bytes)."""
        return 27 * 8

    # ---------------------------------------------------------------
    # Step 7-11: Write arrays and CRSD file
    # ---------------------------------------------------------------

    def _write_crsd(
        self, xmltree, reader, swath_channels, full_echo_meta,
        radar, orbit, ref_time,
    ):
        """Build all binary arrays and write CRSD file."""
        import sarkit.crsd

        # PPP from full echo_df of first polarization (all swaths)
        first_pol = next(iter(full_echo_meta.keys()))
        full_df = full_echo_meta[first_pol]

        # Try to compute reference geometry
        try:
            first_sc = next(iter(swath_channels.values()))
            ppp_array = self._build_ppp_array(
                xmltree, full_df, orbit, ref_time, radar,
            )
            pvp_array = self._build_pvp_array(
                xmltree, first_sc.echo_df, orbit, ref_time, radar,
                global_indices=first_sc.global_indices,
            )

            rg_elem = sarkit.crsd.compute_reference_geometry(
                xmltree, pvps=pvp_array, ppps=ppp_array,
            )
            # Replace placeholder ReferenceGeometry
            old_rg = xmltree.getroot().find(f"{{{CRSD_NS}}}ReferenceGeometry")
            if old_rg is not None:
                parent = xmltree.getroot()
                idx = list(parent).index(old_rg)
                parent.remove(old_rg)
                parent.insert(idx, rg_elem)
        except Exception as e:
            logger.warning("Could not compute ReferenceGeometry: %s", e)
            self._fill_placeholder_ref_geometry(
                xmltree, orbit, ref_time, radar,
            )

        # Create writer
        crsd_metadata = sarkit.crsd.Metadata(xmltree=xmltree)
        fh = open(str(self.output_path), "wb")
        writer = sarkit.crsd.Writer(fh, crsd_metadata)

        try:
            # Write support arrays
            gp_dtype = np.dtype([("Gain", "<f4"), ("Phase", "<f4")])
            gp_data = np.array([(1.0, 0.0)], dtype=gp_dtype).reshape(1, 1)
            writer.write_support_array("GP1", gp_data)

            fxr_dtype = np.dtype([("Amp", "<f4"), ("Phase", "<f4")])
            fxr_data = np.array([(1.0, 0.0), (1.0, 0.0)], dtype=fxr_dtype).reshape(2, 1)
            writer.write_support_array("FXR1", fxr_data)

            dta_dtype = np.dtype([("COD", "<f4"), ("DT", "<f4")])
            dta_data = np.array([(0.0, 0.0)], dtype=dta_dtype).reshape(1, 1)
            writer.write_support_array("DTA1", dta_data)

            # Write PPP once (all swaths, first polarization)
            ppp = self._build_ppp_array(
                xmltree, full_df, orbit, ref_time, radar,
            )
            writer.write_ppp("TX1", ppp)

            # Write signal + PVP per swath channel
            for key, sc in swath_channels.items():
                logger.info(
                    "Decompressing %d echo packets for %s (swath %d, %s)...",
                    sc.num_vectors, key, sc.swath_number, sc.polarization,
                )
                signal = reader.decode_channel(
                    sc.polarization, swath=sc.swath_number,
                )
                logger.info(
                    "Signal shape: %s, writing to CRSD...",
                    signal.shape,
                )

                pvp = self._build_pvp_array(
                    xmltree, sc.echo_df, orbit, ref_time, radar,
                    global_indices=sc.global_indices,
                )

                writer.write_signal(key, signal)
                writer.write_pvp(key, pvp)

                logger.info("Channel %s written.", key)

            writer.done()
        finally:
            fh.close()

    @staticmethod
    def _set_intfrac(
        arr: np.ndarray, field: str, int_vals: np.ndarray, frac_vals: np.ndarray,
    ) -> None:
        """Set both integer and fractional parts of an IntFrac field.

        sarkit's PPP/PVP dtype exposes only the I8 (integer) part as a
        named field.  The F8 (fractional) part occupies the next 8 bytes
        in the record, which we access via a buffer view.
        """
        arr[field] = int_vals
        frac_offset = arr.dtype.fields[field][1] + 8
        frac_view = np.ndarray(
            len(arr), dtype=np.float64,
            buffer=arr.data, offset=frac_offset,
            strides=(arr.dtype.itemsize,),
        )
        frac_view[:] = frac_vals

    def _build_ppp_array(
        self, xmltree, echo_df, orbit, ref_time, radar,
    ) -> np.ndarray:
        """Build Per-Pulse Parameter array."""
        ppp_dtype = sarkit.crsd.get_ppp_dtype(xmltree)
        n = len(echo_df)
        ppp = np.zeros(n, dtype=ppp_dtype)

        # Compute sensing times relative to ref_time
        coarse = echo_df["Coarse Time"].values.astype(np.float64)
        fine = echo_df["Fine Time"].values.astype(np.float64)
        abs_times = coarse + fine
        rel_times = abs_times - ref_time

        # Interpolate orbit
        positions, velocities = orbit.interpolate(abs_times)

        # Antenna frame
        acx, acy = _compute_antenna_frame(positions, velocities)

        # Fill PPP fields
        t_int = np.floor(rel_times).astype(np.int64)
        t_frac = rel_times - t_int

        self._set_intfrac(ppp, "TxTime", t_int, t_frac)

        ppp["TxPos"] = positions
        ppp["TxVel"] = velocities

        f0 = radar.center_frequency
        bw = radar.tx_bandwidth
        ppp["FX1"] = f0 - bw / 2
        ppp["FX2"] = f0 + bw / 2
        ppp["TXmt"] = radar.tx_pulse_length
        self._set_intfrac(ppp, "PhiX0", np.zeros(n, np.int64), np.zeros(n))
        ppp["FxFreq0"] = f0 - bw / 2
        ppp["FxRate"] = radar.chirp_rate
        ppp["TxRadInt"] = 1.0

        ppp["TxACX"] = acx
        ppp["TxACY"] = acy
        ppp["TxEB"] = np.zeros((n, 2))
        ppp["FxResponseIndex"] = 0

        return ppp

    def _build_pvp_array(
        self, xmltree, echo_df, orbit, ref_time, radar,
        global_indices=None,
    ) -> np.ndarray:
        """Build Per-Vector Parameter array.

        Parameters
        ----------
        global_indices : ndarray, optional
            Row indices into the full (all-swath) echo DataFrame, used
            as ``TxPulseIndex`` to map each receive vector to the
            correct PPP transmit pulse.  When *None*, sequential
            indices ``[0, 1, ..., n-1]`` are used (legacy behaviour).
        """
        pvp_dtype = sarkit.crsd.get_pvp_dtype(xmltree)
        n = len(echo_df)
        pvp = np.zeros(n, dtype=pvp_dtype)

        coarse = echo_df["Coarse Time"].values.astype(np.float64)
        fine = echo_df["Fine Time"].values.astype(np.float64)
        swst = echo_df["SWST"].values.astype(np.float64)

        abs_times = coarse + fine
        rcv_abs = abs_times + swst
        rcv_rel = rcv_abs - ref_time

        positions, velocities = orbit.interpolate(rcv_abs)
        acx, acy = _compute_antenna_frame(positions, velocities)

        r_int = np.floor(rcv_rel).astype(np.int64)
        r_frac = rcv_rel - r_int

        self._set_intfrac(pvp, "RcvStart", r_int, r_frac)

        pvp["RcvPos"] = positions
        pvp["RcvVel"] = velocities

        f0 = radar.center_frequency
        bw = radar.tx_bandwidth
        pvp["FRCV1"] = f0 - bw / 2
        pvp["FRCV2"] = f0 + bw / 2
        self._set_intfrac(pvp, "RefPhi0", np.zeros(n, np.int64), np.zeros(n))
        pvp["RefFreq"] = f0
        pvp["DFIC0"] = 0.0
        pvp["FICRate"] = 0.0

        pvp["RcvACX"] = acx
        pvp["RcvACY"] = acy
        pvp["RcvEB"] = np.zeros((n, 2))

        pvp["SIGNAL"] = 1
        pvp["AmpSF"] = 1.0

        # DGRGC from RX gain
        rx_gain_db = echo_df["Rx Gain"].values.astype(np.float64)
        pvp["DGRGC"] = np.power(10.0, rx_gain_db / 20.0)

        # TxPulseIndex maps receive vectors to PPP transmit pulses
        if global_indices is not None:
            pvp["TxPulseIndex"] = global_indices.astype(np.int64)
        else:
            pvp["TxPulseIndex"] = np.arange(n, dtype=np.int64)

        return pvp

    def _fill_placeholder_ref_geometry(
        self, xmltree, orbit, ref_time, radar,
    ):
        """Fill ReferenceGeometry with approximate values if sarkit
        compute_reference_geometry fails."""
        root = xmltree.getroot()
        rg = root.find(f"{{{CRSD_NS}}}ReferenceGeometry")
        if rg is None:
            return

        pos, vel = orbit.interpolate(np.array([ref_time]))
        pos = pos[0]
        vel = vel[0]

        iarp_elem = root.find(f".//{{{CRSD_NS}}}IARP/{{{CRSD_NS}}}ECF")
        if iarp_elem is not None:
            iarp = np.array([
                float(iarp_elem.findtext(f"{{{CRSD_NS}}}X")),
                float(iarp_elem.findtext(f"{{{CRSD_NS}}}Y")),
                float(iarp_elem.findtext(f"{{{CRSD_NS}}}Z")),
            ])
        else:
            iarp = np.zeros(3)

        look = iarp - pos
        slant_range = np.linalg.norm(look)
        look_unit = look / slant_range if slant_range > 0 else look
        vel_unit = vel / np.linalg.norm(vel) if np.linalg.norm(vel) > 0 else vel
        dop_cone = np.degrees(np.arccos(np.clip(np.dot(look_unit, vel_unit), -1, 1)))

        # Side of track
        cross = np.cross(vel, look)
        pos_unit = pos / np.linalg.norm(pos)
        side = "L" if np.dot(cross, pos_unit) > 0 else "R"

        # Ground range (approximate)
        earth_r = 6371000.0
        ground_range = earth_r * np.arcsin(
            np.clip(slant_range / (np.linalg.norm(pos)), 0, 1)
        )

        incidence = np.degrees(np.arcsin(
            np.clip(ground_range / slant_range, 0, 1)
        )) if slant_range > 0 else 0.0
        graze = 90.0 - incidence

        sar_img = _sub(rg, "SARImage")
        _sub(sar_img, "CODTime", "0.0")
        _sub(sar_img, "DwellTime", "0.0")
        _sub(sar_img, "ReferenceTime", "0.0")
        _sub_xyz(sar_img, "ARPPos", pos[0], pos[1], pos[2])
        _sub_xyz(sar_img, "ARPVel", vel[0], vel[1], vel[2])
        _sub(sar_img, "BistaticAngle", "0.0")
        _sub(sar_img, "BistaticAngleRate", "0.0")
        _sub(sar_img, "SideOfTrack", side)
        _sub(sar_img, "SlantRange", f"{slant_range:.6g}")
        _sub(sar_img, "GroundRange", f"{ground_range:.6g}")
        _sub(sar_img, "DopplerConeAngle", f"{dop_cone:.6g}")
        _sub(sar_img, "SquintAngle", "0.0")
        _sub(sar_img, "AzimuthAngle", "0.0")
        _sub(sar_img, "GrazeAngle", f"{graze:.6g}")
        _sub(sar_img, "IncidenceAngle", f"{incidence:.6g}")
        _sub(sar_img, "TwistAngle", "0.0")
        _sub(sar_img, "SlopeAngle", "0.0")
        _sub(sar_img, "LayoverAngle", "0.0")

        for param_name in ("TxParameters", "RcvParameters"):
            p = _sub(rg, param_name)
            _sub(p, "Time", "0.0")
            _sub_xyz(p, "APCPos", pos[0], pos[1], pos[2])
            _sub_xyz(p, "APCVel", vel[0], vel[1], vel[2])
            _sub(p, "SideOfTrack", side)
            _sub(p, "SlantRange", f"{slant_range:.6g}")
            _sub(p, "GroundRange", f"{ground_range:.6g}")
            _sub(p, "DopplerConeAngle", f"{dop_cone:.6g}")
            _sub(p, "SquintAngle", "0.0")
            _sub(p, "AzimuthAngle", "0.0")
            _sub(p, "GrazeAngle", f"{graze:.6g}")
            _sub(p, "IncidenceAngle", f"{incidence:.6g}")
