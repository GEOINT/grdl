# -*- coding: utf-8 -*-
"""
Sentinel-1 Level-0 to CRSD Conversion Pipeline.

Converts Sentinel-1 Level-0 IW SAFE products to NGA CRSD
(Compensated Radar Signal Data) format.  Handles ISP metadata
decoding, FDBAQ decompression, orbit interpolation, and CRSD
XML/binary construction via sarkit.

Each IW sub-swath + polarization combination becomes a distinct
CRSD receive channel (e.g. ``IW1_VV``, ``IW2_VH``) so that each
channel has uniform signal dimensions (no cross-swath zero-padding).

Dependencies
------------
sarkit, sentinel1decoder, scipy, lxml

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
2026-04-16

Modified
--------
2026-04-16
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from lxml import etree

import sarkit.crsd
import sarkit.wgs84

from grdl.geolocation.coordinates import geodetic_to_ecef
from grdl.IO.sar.sentinel1_l0.constants import (
    FDBAQ_AMPLITUDE_SCALE,
    GPS_LEAP_SECONDS,
    IW_LOGICAL_TO_RAW,
    IW_RAW_TO_LOGICAL,
    MEAN_EARTH_RADIUS,
    POL_TO_TXRX,
    SENTINEL1_CENTER_FREQUENCY_HZ,
    SPEED_OF_LIGHT,
)
from grdl.IO.sar.sentinel1_l0.decoder import Sentinel1Decoder
from grdl.IO.sar.sentinel1_l0.orbit import OrbitInterpolator, OrbitLoader
from grdl.IO.sar.sentinel1_l0.reader import Sentinel1L0Reader

logger = logging.getLogger(__name__)

# CRSD XML namespace
CRSD_NS = "http://api.nsgreg.nga.mil/schema/crsd/1.0"
NSMAP = {None: CRSD_NS}

# GPS epoch: 6 Jan 1980 00:00:00 UTC
_GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


# =============================================================================
# XML builder helpers
# =============================================================================

def _sub(
    parent: etree._Element, tag: str, text: str = None,
) -> etree._Element:
    """Add a namespaced sub-element with optional text."""
    elem = etree.SubElement(parent, f"{{{CRSD_NS}}}{tag}")
    if text is not None:
        elem.text = str(text)
    return elem


def _sub_xyz(
    parent: etree._Element,
    tag: str,
    x: float,
    y: float,
    z: float,
) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "X", f"{x:.12g}")
    _sub(elem, "Y", f"{y:.12g}")
    _sub(elem, "Z", f"{z:.12g}")
    return elem


def _sub_xy(
    parent: etree._Element,
    tag: str,
    x: float,
    y: float,
) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "X", f"{x:.12g}")
    _sub(elem, "Y", f"{y:.12g}")
    return elem


def _sub_latlon(
    parent: etree._Element,
    tag: str,
    lat: float,
    lon: float,
) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "Lat", f"{lat:.10g}")
    _sub(elem, "Lon", f"{lon:.10g}")
    return elem


def _sub_latlonhae(
    parent: etree._Element,
    tag: str,
    lat: float,
    lon: float,
    hae: float,
) -> etree._Element:
    elem = _sub(parent, tag)
    _sub(elem, "Lat", f"{lat:.10g}")
    _sub(elem, "Lon", f"{lon:.10g}")
    _sub(elem, "HAE", f"{hae:.6g}")
    return elem


def _sub_pxp_f8(
    parent: etree._Element, name: str, offset: int,
) -> int:
    """Add an F8 per-parameter field. Returns new offset."""
    elem = _sub(parent, name)
    _sub(elem, "Offset", str(offset))
    _sub(elem, "Size", "1")
    _sub(elem, "Format", "F8")
    return offset + 1


def _sub_pxp_i8(
    parent: etree._Element, name: str, offset: int,
) -> int:
    """Add an I8 per-parameter field. Returns new offset."""
    elem = _sub(parent, name)
    _sub(elem, "Offset", str(offset))
    _sub(elem, "Size", "1")
    _sub(elem, "Format", "I8")
    return offset + 1


def _sub_pxp_intfrac(
    parent: etree._Element, name: str, offset: int,
) -> int:
    """Add IntFrac (I8+F8) per-parameter field. Returns new offset."""
    elem = _sub(parent, name)
    _sub(elem, "Offset", str(offset))
    _sub(elem, "Size", "2")
    _sub(elem, "Format", "Int=I8;Frac=F8;")
    return offset + 2


def _sub_pxp_xyz(
    parent: etree._Element, name: str, offset: int,
) -> int:
    """Add XYZ per-parameter field. Returns new offset."""
    elem = _sub(parent, name)
    _sub(elem, "Offset", str(offset))
    _sub(elem, "Size", "3")
    _sub(elem, "Format", "X=F8;Y=F8;Z=F8;")
    return offset + 3


def _sub_pxp_eb(
    parent: etree._Element, name: str, offset: int,
) -> int:
    """Add EB per-parameter field (DCX+DCY). Returns new offset."""
    elem = _sub(parent, name)
    _sub(elem, "Offset", str(offset))
    _sub(elem, "Size", "2")
    _sub(elem, "Format", "DCX=F8;DCY=F8;")
    return offset + 2


# =============================================================================
# Antenna frame computation
# =============================================================================

def _compute_antenna_frame(
    pos: np.ndarray,
    vel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute antenna coordinate frame unit vectors (ACX, ACY).

    Parameters
    ----------
    pos : ndarray, shape (..., 3)
        ECF positions in metres.
    vel : ndarray, shape (..., 3)
        ECF velocities in m/s.

    Returns
    -------
    acx : ndarray, shape (..., 3)
        Along-track unit vector.
    acy : ndarray, shape (..., 3)
        Cross-track unit vector.
    """
    v_norm = np.linalg.norm(vel, axis=-1, keepdims=True)
    v_norm = np.where(v_norm == 0, 1.0, v_norm)
    acx = vel / v_norm

    p_norm = np.linalg.norm(pos, axis=-1, keepdims=True)
    p_norm = np.where(p_norm == 0, 1.0, p_norm)
    nadir = -pos / p_norm

    acy = np.cross(nadir, acx)
    acy_norm = np.linalg.norm(acy, axis=-1, keepdims=True)
    acy_norm = np.where(acy_norm == 0, 1.0, acy_norm)
    acy = acy / acy_norm
    return acx, acy


# =============================================================================
# Per-swath+pol channel descriptor
# =============================================================================

@dataclass
class _SwathChannel:
    """Metadata for one swath+polarization during conversion."""

    key: str             # e.g. "IW1_VV"
    swath_number: int    # logical swath number (1, 2, or 3)
    raw_swath: int       # raw ISP swath number (10, 11, or 12 for IW)
    polarization: str    # polarization string e.g. "VV"
    tx_pol: str          # transmit polarization "V" or "H"
    rx_pol: str          # receive polarization "V" or "H"
    num_vectors: int     # number of echo packets in this swath
    num_samples: int     # number of range samples (2 * max_num_quads)
    echo_df: Any         # pandas DataFrame filtered to this swath
    global_indices: np.ndarray   # row indices into full echo_df → TxPulseIndex


# =============================================================================
# Conversion pipeline
# =============================================================================

class Sentinel1L0ToCRSD:
    """Convert Sentinel-1 Level-0 IW SAFE to NGA CRSD format.

    Each IW sub-swath + polarization combination becomes a distinct
    CRSD receive channel (e.g. ``IW1_VV``, ``IW2_VH``), so that each
    channel has uniform signal dimensions.

    Parameters
    ----------
    safe_path : str or Path
        Path to the ``.SAFE`` directory.
    output_path : str or Path
        Output path for the ``.crsd`` file.
    orbit_file : str or Path, optional
        Path to an ESA Precise Orbit Ephemeris ``.EOF`` file.
        When provided, POE orbit data is used in preference to the
        sub-commutated ISP annotation orbit vectors.
    channel : str, optional
        Polarization to convert (e.g. ``"VV"``).
        If *None* (default), all polarizations are converted.
    swath : int, optional
        Logical sub-swath number to convert (1, 2, or 3 for IW).
        If *None* (default), all sub-swaths are converted.

    Examples
    --------
    >>> converter = Sentinel1L0ToCRSD(
    ...     safe_path='S1A_IW_RAW__0SDV_....SAFE',
    ...     output_path='output.crsd',
    ... )
    >>> converter.convert()
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
            Path to the written CRSD file.
        """
        if self.orbit_file is None:
            raise ValueError(
                "A precise orbit ephemeris (POE .EOF) file is required. "
                "Pass orbit_file= to Sentinel1L0ToCRSD. "
                "Annotation-embedded orbit vectors are insufficient for "
                "CRSD-quality timing and position accuracy."
            )

        logger.info(
            "Opening Sentinel-1 Level-0: %s", self.safe_path,
        )
        reader = Sentinel1L0Reader(self.safe_path)
        meta = reader.metadata

        # Determine which polarizations to convert
        all_pols = meta.polarizations
        if self.channel is not None:
            if self.channel not in all_pols:
                raise ValueError(
                    f"Polarization {self.channel!r} not in product. "
                    f"Available: {all_pols}"
                )
            pols = [self.channel]
        else:
            pols = list(all_pols)

        radar = meta.radar_parameters
        product_id = meta

        logger.info(
            "Product polarizations: %s", pols,
        )

        # Step 1: Build orbit interpolator
        orbit = self._build_orbit_interpolator(reader, meta)

        # Step 2: Decode per-pulse metadata and build swath channels
        full_echo_dfs: Dict[str, Any] = {}
        cal_noise_dfs: Dict[str, Any] = {}
        for pol in pols:
            echo_df, cal_df = self._decode_echo_df(reader, pol)
            full_echo_dfs[pol] = echo_df
            cal_noise_dfs[pol] = cal_df

        swath_channels = self._build_swath_channels(
            reader, pols, full_echo_dfs,
        )
        if not swath_channels:
            avail = sorted({
                sw
                for pol in pols
                for sw in reader.get_swath_info(pol).keys()
            })
            raise ValueError(
                f"No swath channels to convert. "
                f"Requested swath={self.swath}, "
                f"available logical swaths: {avail}"
            )
        logger.info(
            "Swath channels: %s", list(swath_channels.keys()),
        )

        # Step 3: Compute collection reference time
        ref_time_gps, collection_ref_dt = self._compute_ref_time(
            full_echo_dfs,
        )

        # Step 4: Compute scene geometry
        iarp_llh, iarp_ecf = self._compute_iarp(meta)

        # Step 5: Derive timing/frequency bounds
        timing = self._compute_timing_bounds(
            swath_channels, full_echo_dfs, ref_time_gps, radar,
        )

        # Step 6: Build CRSD XML
        xmltree = self._build_crsd_xml(
            meta,
            swath_channels,
            full_echo_dfs,
            cal_noise_dfs,
            radar,
            orbit,
            iarp_llh,
            iarp_ecf,
            collection_ref_dt,
            ref_time_gps,
            timing,
        )

        # Step 7: Write binary arrays and CRSD file
        self._write_crsd(
            xmltree,
            reader,
            swath_channels,
            full_echo_dfs,
            radar,
            orbit,
            ref_time_gps,
        )

        reader.close()
        logger.info("CRSD written: %s", self.output_path)
        return self.output_path

    # ---------------------------------------------------------------
    # Step 1: Orbit
    # ---------------------------------------------------------------

    def _build_orbit_interpolator(
        self,
        reader: Sentinel1L0Reader,
        meta: Any,
    ) -> OrbitInterpolator:
        """Build orbit interpolator from annotation data or POE file.

        Uses :class:`OrbitLoader` to merge annotation and optional POE
        orbit vectors, then returns an :class:`OrbitInterpolator`
        referenced to the product start time.

        Parameters
        ----------
        reader : Sentinel1L0Reader
        meta : Sentinel1L0Metadata

        Returns
        -------
        OrbitInterpolator
        """
        loader = OrbitLoader()

        # Load annotation orbit vectors (always available)
        if meta.orbit_state_vectors:
            loader.load_annotation_vectors(meta.orbit_state_vectors)
            logger.debug(
                "Loaded %d annotation orbit vectors",
                len(meta.orbit_state_vectors),
            )

        # Optionally load POE file
        if self.orbit_file is not None:
            loader.load_poe_file(self.orbit_file)
            logger.info(
                "Loaded POE orbit from %s", self.orbit_file,
            )

        reference_time = meta.start_time
        interp = loader.create_interpolator(
            reference_time=reference_time, prefer_poe=True,
        )
        logger.info(
            "Orbit interpolator ready, reference_time=%s",
            reference_time.isoformat(),
        )
        return interp

    # ---------------------------------------------------------------
    # Step 2: Decode echo metadata per polarization
    # ---------------------------------------------------------------

    def _decode_echo_df(
        self,
        reader: Sentinel1L0Reader,
        polarization: str,
    ) -> Any:
        """Decode packet metadata for a polarization.

        Uses the measurement file from the SAFE product structure and
        :class:`~grdl.IO.sar.sentinel1_l0.decoder.Sentinel1Decoder`
        to obtain a per-pulse metadata DataFrame.

        Returned DataFrame has columns including (but not limited to):
        ``Coarse Time``, ``Fine Time``, ``SWST``, ``PRI``, ``Rank``,
        ``Swath Number``, ``Rx Gain``, ``Number of Quads``.

        .. note::
           ``Fine Time`` in the parsed (non-raw) DataFrame is already
           a fractional second in ``[0, 1)`` — the decoder converts the
           raw uint16 as ``fine / 65536`` automatically.  No extra
           conversion is required here.

        Parameters
        ----------
        reader : Sentinel1L0Reader
        polarization : str

        Returns
        -------
        tuple of (pandas.DataFrame, pandas.DataFrame)
            ``(echo_df, cal_noise_df)`` — ECHO pulses only (Signal Type
            == 0) for PPP/PVP construction, and all non-ECHO packets
            (calibration types, noise) for metadata preservation in the
            CRSD output.  Calibration/noise packets are never written to
            the CRSD signal arrays but their timing and type are stored
            in ``ProductInfo/ProcessingParameters``.
        """
        product = reader.product
        if product is None:
            raise RuntimeError("Reader product not opened")

        mfs = product.get_measurement_file(polarization)
        if mfs is None:
            raise ValueError(
                f"No measurement file for polarization {polarization!r}"
            )

        decoder = Sentinel1Decoder(mfs.measurement_file)
        df = decoder.decode_metadata()

        # Split into ECHO-only and non-ECHO (calibration/noise).
        # PPP and PVP must contain only ECHO pulses (Signal Type == 0)
        # so that timing and orbit interpolation are valid.  Calibration
        # and noise packets are returned separately so the caller can
        # preserve their metadata in the CRSD ProcessingParameters.
        if "Signal Type" in df.columns:
            # sentinel1decoder returns "Signal Type" as a custom class
            # whose __eq__ and __int__ do not work with plain int.
            # Use getattr(x, 'value', x) which returns .value for enum-
            # like types and x itself for plain numpy integers.
            echo_mask = df["Signal Type"].apply(
                lambda x: getattr(x, "value", x) == 0
            )
            echo_df = df[echo_mask].reset_index(drop=True)
            cal_noise_df = df[~echo_mask].reset_index(drop=True)
            logger.info(
                "Polarization %s: %d ECHO, %d calibration/noise packets.",
                polarization, len(echo_df), len(cal_noise_df),
            )
        else:
            echo_df = df
            cal_noise_df = df.iloc[0:0].copy()

        logger.debug(
            "Decoded %d ECHO packets for polarization %s",
            len(echo_df),
            polarization,
        )
        return echo_df, cal_noise_df

    # ---------------------------------------------------------------
    # Step 3: Build swath channel descriptors
    # ---------------------------------------------------------------

    def _build_swath_channels(
        self,
        reader: Sentinel1L0Reader,
        pols: List[str],
        full_echo_dfs: Dict[str, Any],
    ) -> Dict[str, _SwathChannel]:
        """Build per-swath+polarization channel descriptors.

        Parameters
        ----------
        reader : Sentinel1L0Reader
        pols : list of str
        full_echo_dfs : dict
            Polarization → full echo DataFrame (all swaths).

        Returns
        -------
        dict
            Channel key (e.g. ``"IW1_VV"``) → :class:`_SwathChannel`.
        """
        channels: Dict[str, _SwathChannel] = {}

        for pol in pols:
            tx_pol, rx_pol = POL_TO_TXRX.get(pol.upper(), ("V", "V"))
            swath_info = reader.get_swath_info(pol)
            full_df = full_echo_dfs[pol]

            for raw_sw_key, sw_info in sorted(swath_info.items()):
                # reader.get_swath_info() keys are raw swath numbers
                # (10, 11, 12 for IW).  Convert to 1-based logical index
                # before building the channel key and applying the swath
                # filter the caller passed in.
                logical_sw = IW_RAW_TO_LOGICAL.get(
                    raw_sw_key, raw_sw_key
                )
                raw_sw = raw_sw_key

                if self.swath is not None and logical_sw != self.swath:
                    continue

                key = f"IW{logical_sw}_{pol}"

                # Filter echo_df to this raw swath.  sentinel1decoder
                # returns "Swath Number" as a custom class whose __eq__
                # does not compare equal to plain int.  Use int() instead.
                if "Swath Number" in full_df.columns:
                    mask = full_df["Swath Number"].apply(
                        lambda x: getattr(x, "value", int(x)) == raw_sw
                    ).values
                    swath_df = full_df[mask].reset_index(drop=True)
                    global_indices = np.where(mask)[0]
                else:
                    swath_df = full_df.copy()
                    global_indices = np.arange(len(full_df))

                if len(swath_df) == 0:
                    logger.warning(
                        "No echo packets found for %s (raw swath %d)",
                        key, raw_sw,
                    )
                    continue

                # Number of range samples: 2 × max number of quads
                if "Number of Quads" in swath_df.columns:
                    num_samples = int(
                        2 * swath_df["Number of Quads"].max()
                    )
                else:
                    num_samples = sw_info.bursts[0].num_samples if sw_info.bursts else 0

                channels[key] = _SwathChannel(
                    key=key,
                    swath_number=logical_sw,
                    raw_swath=raw_sw,
                    polarization=pol,
                    tx_pol=tx_pol,
                    rx_pol=rx_pol,
                    num_vectors=len(swath_df),
                    num_samples=num_samples,
                    echo_df=swath_df,
                    global_indices=global_indices,
                )

        return channels

    # ---------------------------------------------------------------
    # Step 4: Collection reference time
    # ---------------------------------------------------------------

    def _compute_ref_time(
        self,
        full_echo_dfs: Dict[str, Any],
    ) -> Tuple[float, datetime]:
        """Compute collection reference time at mid-scene.

        GPS seconds (absolute) and UTC datetime at the mid-point of
        the collection are returned.  GPS−UTC offset of 18 s is applied
        to the UTC conversion.

        .. note::
           ``Fine Time`` is already fractional seconds in the decoded
           DataFrame.  No ``/ 65536`` conversion is needed here.

        Parameters
        ----------
        full_echo_dfs : dict
            Polarization → full echo DataFrame.

        Returns
        -------
        (float, datetime)
            ``(ref_time_gps, collection_ref_utc)``
        """
        first_df = next(iter(full_echo_dfs.values()))
        coarse = first_df["Coarse Time"].values.astype(np.float64)
        fine = first_df["Fine Time"].values.astype(np.float64)
        sensing_gps = coarse + fine   # GPS seconds since GPS epoch

        ref_time_gps = (sensing_gps.min() + sensing_gps.max()) / 2.0

        # GPS → UTC: subtract leap seconds
        utc_seconds = ref_time_gps - GPS_LEAP_SECONDS
        from datetime import timedelta
        collection_ref_utc = (
            _GPS_EPOCH + timedelta(seconds=float(utc_seconds))
        ).replace(tzinfo=None)

        return float(ref_time_gps), collection_ref_utc

    # ---------------------------------------------------------------
    # Step 5: Scene geometry
    # ---------------------------------------------------------------

    def _compute_iarp(
        self,
        meta: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute IARP from manifest footprint centroid.

        Parameters
        ----------
        meta : Sentinel1L0Metadata

        Returns
        -------
        (iarp_llh, iarp_ecf) : tuple of ndarray
        """
        footprint = getattr(meta, "footprint", None) or []
        if not footprint:
            logger.warning(
                "No footprint in metadata; using (0, 0, 0) IARP"
            )
            iarp_llh = np.array([0.0, 0.0, 0.0])
            iarp_ecf = geodetic_to_ecef(iarp_llh)
            return iarp_llh, iarp_ecf

        coords = [(c.lat, c.lon) for c in footprint]
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]

        mean_lat = float(np.mean([c[0] for c in coords]))
        mean_lon = float(np.mean([c[1] for c in coords]))
        hae = 0.0

        iarp_llh = np.array([mean_lat, mean_lon, hae])
        iarp_ecf = geodetic_to_ecef(iarp_llh)
        return iarp_llh, iarp_ecf

    # ---------------------------------------------------------------
    # Step 6: Timing and frequency bounds
    # ---------------------------------------------------------------

    def _compute_timing_bounds(
        self,
        swath_channels: Dict[str, _SwathChannel],
        full_echo_dfs: Dict[str, Any],
        ref_time_gps: float,
        radar: Any,
    ) -> Dict:
        """Compute global and per-channel timing/frequency bounds.

        Parameters
        ----------
        swath_channels : dict
        full_echo_dfs : dict
        ref_time_gps : float
        radar : S1L0RadarParameters

        Returns
        -------
        dict
        """
        import pandas as pd

        # Global bounds from all-swath, all-pol echo DataFrames
        all_sensing: List[np.ndarray] = []
        all_rcv_start: List[np.ndarray] = []
        for df in full_echo_dfs.values():
            coarse = df["Coarse Time"].values.astype(np.float64)
            fine = df["Fine Time"].values.astype(np.float64)
            # ISP Coarse+Fine = GPS seconds at first receive sample
            # (RcvStart).  TX time = RcvStart - SWST - Rank*PRI
            # per [ISP] S1-IF-ASD-PL-0007 Table 3-2.
            sensing = coarse + fine
            if (
                "SWST" not in df.columns
                or "Rank" not in df.columns
                or "PRI" not in df.columns
            ):
                raise ValueError(
                    "Echo DataFrame missing SWST/Rank/PRI columns — "
                    "cannot compute TX pulse origin times."
                )
            swst_b = df["SWST"].values.astype(np.float64)
            rank_b = df["Rank"].values.astype(np.float64)
            pri_b = df["PRI"].values.astype(np.float64)
            rf_times = sensing - swst_b - rank_b * pri_b
            all_sensing.append(rf_times)
            all_rcv_start.append(sensing)  # ISP time IS the receive start

        sensing_all = np.concatenate(all_sensing) - ref_time_gps
        rcv_all = np.concatenate(all_rcv_start) - ref_time_gps

        f0 = float(getattr(radar, "center_frequency_hz", SENTINEL1_CENTER_FREQUENCY_HZ))
        bw = float(getattr(radar, "tx_bandwidth_hz", 42.79e6))

        per_channel: Dict[str, Dict] = {}
        for key, sc in swath_channels.items():
            ch_coarse = sc.echo_df["Coarse Time"].values.astype(np.float64)
            ch_fine = sc.echo_df["Fine Time"].values.astype(np.float64)
            # ISP time is the receive start directly.
            ch_rcv = ch_coarse + ch_fine
            per_channel[key] = {
                "rcv_start_time1": float(ch_rcv.min() - ref_time_gps),
                "rcv_start_time2": float(ch_rcv.max() - ref_time_gps),
            }

        return {
            "tx_time1": float(sensing_all.min()),
            "tx_time2": float(sensing_all.max()),
            "fx_min": f0 - bw / 2.0,
            "fx_max": f0 + bw / 2.0,
            "rcv_start_time1": float(rcv_all.min()),
            "rcv_start_time2": float(rcv_all.max()),
            "frcv_min": f0 - bw / 2.0,
            "frcv_max": f0 + bw / 2.0,
            "per_channel": per_channel,
            "f0": f0,
            "bw": bw,
        }

    # ---------------------------------------------------------------
    # Step 7: Build CRSD XML
    # ---------------------------------------------------------------

    def _build_crsd_xml(
        self,
        meta: Any,
        swath_channels: Dict[str, _SwathChannel],
        full_echo_dfs: Dict[str, Any],
        cal_noise_dfs: Dict[str, Any],
        radar: Any,
        orbit: OrbitInterpolator,
        iarp_llh: np.ndarray,
        iarp_ecf: np.ndarray,
        collection_ref_dt: datetime,
        ref_time_gps: float,
        timing: Dict,
    ) -> etree.ElementTree:
        """Build the complete CRSDsar XML tree.

        Parameters
        ----------
        meta : Sentinel1L0Metadata
        swath_channels : dict
        full_echo_dfs : dict
        cal_noise_dfs : dict
            Per-polarization DataFrames of non-ECHO (calibration/noise)
            packets.  Not written to signal arrays; stored as metadata.
        radar : S1L0RadarParameters
        orbit : OrbitInterpolator
        iarp_llh : ndarray
        iarp_ecf : ndarray
        collection_ref_dt : datetime
        ref_time_gps : float
        timing : dict

        Returns
        -------
        lxml.etree.ElementTree
        """
        root = etree.Element(f"{{{CRSD_NS}}}CRSDsar", nsmap=NSMAP)
        first_sc = next(iter(swath_channels.values()))
        tx_pol = first_sc.tx_pol

        f0 = timing["f0"]
        bw = timing["bw"]

        # --- ProductInfo ---
        pi = _sub(root, "ProductInfo")
        _sub(pi, "ProductName", self.safe_path.name)
        _sub(pi, "Classification", "UNCLASSIFIED")
        _sub(pi, "ReleaseInfo", "PUBLIC RELEASE")
        ci = _sub(pi, "CreationInfo")
        _sub(ci, "Application", "grdl-sentinel1-l0-to-crsd")
        _sub(
            ci,
            "DateTime",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        # FDBAQ calibration note (metadata only — I/Q data is not scaled)
        proc = _sub(pi, "ProcessingParameters")
        _sub(proc, "FDBaqAmplitudeScale", f"{FDBAQ_AMPLITUDE_SCALE:.6g}")
        _sub(
            proc,
            "FDBaqScaleNote",
            "sentinel1decoder FDBAQ reconstruction returns amplitudes "
            "approximately 1.607x smaller than ESA NRL x sigma reference. "
            "Apply this scale factor before absolute radiometric calibration.",
        )

        # --- Per-packet calibration/compression statistics ---
        # Aggregate BAQ Mode and ECC Number across all echo packets so that
        # downstream processors know the compression scheme used.
        _first_df = next(iter(full_echo_dfs.values()))
        if "BAQ Mode" in _first_df.columns:
            _baq_vals = np.concatenate([
                df["BAQ Mode"].values for df in full_echo_dfs.values()
            ])
            _baq_modes, _baq_counts = np.unique(_baq_vals, return_counts=True)
            _baq_summary = "; ".join(
                f"mode={int(m)} ({int(c)} pulses)"
                for m, c in zip(_baq_modes, _baq_counts)
            )
            _sub(proc, "BaqModes", _baq_summary)
            # Primary BAQ mode: mode with the most pulses
            _sub(proc, "PrimaryBaqMode", str(int(_baq_modes[np.argmax(_baq_counts)])))
        if "ECC Number" in _first_df.columns:
            _ecc_vals = np.concatenate([
                df["ECC Number"].values for df in full_echo_dfs.values()
            ])
            _ecc_unique = np.unique(_ecc_vals)
            _sub(proc, "EccNumbers", ";".join(str(int(e)) for e in _ecc_unique))
        if "Data Take ID" in _first_df.columns:
            _dtid = int(_first_df["Data Take ID"].iloc[0])
            _sub(proc, "DataTakeID", f"0x{_dtid:08X}")

        # Calibration and noise packet statistics, per polarization.
        # ECHO pulses are written to the CRSD signal arrays; cal/noise
        # packets are NOT in the signal data.  Their timing and type are
        # stored here so users can cross-reference with the original SAFE
        # product (e.g. to apply ESA's noise floor or calibration LUTs).
        for _pol, _cn_df in cal_noise_dfs.items():
            if len(_cn_df) == 0:
                continue
            cn_elem = _sub(proc, "CalibrationNoiseStats")
            cn_elem.set("polarization", _pol)
            _sub(cn_elem, "TotalPackets", str(len(_cn_df)))
            if "Signal Type" in _cn_df.columns and "Coarse Time" in _cn_df.columns:
                for _stype in sorted(_cn_df["Signal Type"].unique()):
                    _grp = _cn_df[_cn_df["Signal Type"] == _stype]
                    _t_coarse = _grp["Coarse Time"].values.astype(np.float64)
                    _t_fine = (
                        _grp["Fine Time"].values.astype(np.float64)
                        if "Fine Time" in _grp.columns
                        else np.zeros(len(_grp))
                    )
                    _t_gps = _t_coarse + _t_fine
                    sg = _sub(cn_elem, "SignalTypeGroup")
                    sg.set("signalType", str(int(_stype)))
                    sg.set("count", str(len(_grp)))
                    sg.set("tGpsStart", f"{float(_t_gps.min()):.3f}")
                    sg.set("tGpsStop", f"{float(_t_gps.max()):.3f}")

        # --- SARInfo ---
        si = _sub(root, "SARInfo")
        _sub(si, "CollectType", "MONOSTATIC")
        rm = _sub(si, "RadarMode")
        _sub(rm, "ModeType", "DYNAMIC STRIPMAP")
        mode = getattr(meta, "mode", None)
        _sub(rm, "ModeID", str(mode) if mode else "IW")

        # --- TransmitInfo ---
        mission = getattr(meta, "mission", None)
        mission_str = str(mission) if mission else "SENTINEL-1"
        data_take_id = getattr(meta, "data_take_id", 0) or 0
        event_name = f"DT-{int(data_take_id):08X}"
        ti = _sub(root, "TransmitInfo")
        _sub(ti, "SensorName", mission_str)
        _sub(ti, "EventName", event_name)

        # --- ReceiveInfo ---
        ri = _sub(root, "ReceiveInfo")
        _sub(ri, "SensorName", mission_str)
        _sub(ri, "EventName", event_name)

        # --- Global ---
        gl = _sub(root, "Global")
        _sub(
            gl,
            "CollectionRefTime",
            collection_ref_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        )
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
        sc_elem = _sub(root, "SceneCoordinates")
        _sub(sc_elem, "EarthModel", "WGS_84")
        iarp_elem = _sub(sc_elem, "IARP")
        _sub_xyz(iarp_elem, "ECF", iarp_ecf[0], iarp_ecf[1], iarp_ecf[2])
        _sub_latlonhae(iarp_elem, "LLH", iarp_llh[0], iarp_llh[1], iarp_llh[2])
        ref_surf = _sub(sc_elem, "ReferenceSurface")
        planar = _sub(ref_surf, "Planar")
        u_east = sarkit.wgs84.east(iarp_llh)
        u_north = sarkit.wgs84.north(iarp_llh)
        _sub_xyz(planar, "uIAX", u_east[0], u_east[1], u_east[2])
        _sub_xyz(planar, "uIAY", u_north[0], u_north[1], u_north[2])

        # ImageArea
        max_vecs = max(sc.num_vectors for sc in swath_channels.values())
        max_samps = max(sc.num_samples for sc in swath_channels.values())
        ia_elem = _sub(sc_elem, "ImageArea")
        _sub_xy(ia_elem, "X1Y1", 0.0, 0.0)
        _sub_xy(ia_elem, "X2Y2", float(max_vecs - 1), float(max_samps - 1))
        poly = _sub(ia_elem, "Polygon")
        poly.set("size", "4")
        for vi, (cx, cy) in enumerate(
            [
                (0.0, 0.0),
                (float(max_vecs - 1), 0.0),
                (float(max_vecs - 1), float(max_samps - 1)),
                (0.0, float(max_samps - 1)),
            ],
            start=1,
        ):
            v = _sub_xy(poly, "Vertex", cx, cy)
            v.set("index", str(vi))

        # ImageAreaCornerPoints from footprint
        footprint = getattr(meta, "footprint", None) or []
        if footprint:
            fps = list(footprint)
            if len(fps) > 1 and fps[0].lat == fps[-1].lat and fps[0].lon == fps[-1].lon:
                fps = fps[:-1]
            iacp = _sub(sc_elem, "ImageAreaCornerPoints")
            for idx, coord in enumerate(fps[:4], start=1):
                cp = _sub_latlon(iacp, "IACP", coord.lat, coord.lon)
                cp.set("index", str(idx))

        # --- Data ---
        self._build_data_section(
            root, swath_channels, full_echo_dfs,
        )

        # --- TxSequence ---
        self._build_tx_sequence(root, f0, bw, timing, tx_pol, iarp_ecf, radar)

        # --- Channel ---
        self._build_channel_section(root, swath_channels, f0, bw, timing, iarp_ecf, radar)

        # --- ReferenceGeometry placeholder ---
        _sub(root, "ReferenceGeometry")

        # --- SupportArray ---
        self._build_support_array_section(root, f0, bw)

        # --- PPP ---
        ppp_sec = _sub(root, "PPP")
        off = 0
        off = _sub_pxp_intfrac(ppp_sec, "TxTime", off)
        off = _sub_pxp_xyz(ppp_sec, "TxPos", off)
        off = _sub_pxp_xyz(ppp_sec, "TxVel", off)
        off = _sub_pxp_f8(ppp_sec, "FX1", off)
        off = _sub_pxp_f8(ppp_sec, "FX2", off)
        off = _sub_pxp_f8(ppp_sec, "TXmt", off)
        off = _sub_pxp_intfrac(ppp_sec, "PhiX0", off)
        off = _sub_pxp_f8(ppp_sec, "FxFreq0", off)
        off = _sub_pxp_f8(ppp_sec, "FxRate", off)
        off = _sub_pxp_f8(ppp_sec, "TxRadInt", off)
        off = _sub_pxp_xyz(ppp_sec, "TxACX", off)
        off = _sub_pxp_xyz(ppp_sec, "TxACY", off)
        off = _sub_pxp_eb(ppp_sec, "TxEB", off)
        off = _sub_pxp_i8(ppp_sec, "FxResponseIndex", off)
        # 25 doubles total

        # --- PVP ---
        pvp_sec = _sub(root, "PVP")
        off = 0
        off = _sub_pxp_intfrac(pvp_sec, "RcvStart", off)
        off = _sub_pxp_xyz(pvp_sec, "RcvPos", off)
        off = _sub_pxp_xyz(pvp_sec, "RcvVel", off)
        off = _sub_pxp_f8(pvp_sec, "FRCV1", off)
        off = _sub_pxp_f8(pvp_sec, "FRCV2", off)
        off = _sub_pxp_intfrac(pvp_sec, "RefPhi0", off)
        off = _sub_pxp_f8(pvp_sec, "RefFreq", off)
        off = _sub_pxp_f8(pvp_sec, "DFIC0", off)
        off = _sub_pxp_f8(pvp_sec, "FICRate", off)
        off = _sub_pxp_xyz(pvp_sec, "RcvACX", off)
        off = _sub_pxp_xyz(pvp_sec, "RcvACY", off)
        off = _sub_pxp_eb(pvp_sec, "RcvEB", off)
        off = _sub_pxp_i8(pvp_sec, "SIGNAL", off)
        off = _sub_pxp_f8(pvp_sec, "AmpSF", off)
        off = _sub_pxp_f8(pvp_sec, "DGRGC", off)
        off = _sub_pxp_i8(pvp_sec, "TxPulseIndex", off)
        # 27 doubles total

        # --- Antenna ---
        self._build_antenna_section(root, swath_channels, tx_pol, f0)

        return etree.ElementTree(root)

    def _build_data_section(
        self,
        root: etree._Element,
        swath_channels: Dict[str, _SwathChannel],
        full_echo_dfs: Dict[str, Any],
    ) -> None:
        """Build the Data section."""
        data = _sub(root, "Data")

        # Support arrays
        support = _sub(data, "Support")
        _sub(support, "NumSupportArrays", "3")

        for sa_id, nrows, ncols, bpe, byte_off in [
            ("GP1", "1", "1", "8", "0"),
            ("FXR1", "2", "1", "8", "8"),
            ("DTA1", "1", "1", "8", "24"),
        ]:
            sa = _sub(support, "SupportArray")
            _sub(sa, "SAId", sa_id)
            _sub(sa, "NumRows", nrows)
            _sub(sa, "NumCols", ncols)
            _sub(sa, "BytesPerElement", bpe)
            _sub(sa, "ArrayByteOffset", byte_off)

        # Transmit (PPP)
        first_pol = next(iter(full_echo_dfs.keys()))
        total_pulses = len(full_echo_dfs[first_pol])
        ppp_bytes = 25 * 8

        tx = _sub(data, "Transmit")
        _sub(tx, "NumBytesPPP", str(ppp_bytes))
        _sub(tx, "NumTxSequences", "1")
        txs = _sub(tx, "TxSequence")
        _sub(txs, "TxId", "TX1")
        _sub(txs, "NumPulses", str(total_pulses))
        _sub(txs, "PPPArrayByteOffset", "0")

        # Receive (PVP + signal per channel)
        pvp_bytes = 27 * 8
        rcv = _sub(data, "Receive")
        _sub(rcv, "SignalArrayFormat", "CF8")
        _sub(rcv, "NumBytesPVP", str(pvp_bytes))
        _sub(rcv, "NumCRSDChannels", str(len(swath_channels)))

        sig_off = 0
        pvp_off = 0
        for key, sc in swath_channels.items():
            ch = _sub(rcv, "Channel")
            _sub(ch, "ChId", key)
            _sub(ch, "NumVectors", str(sc.num_vectors))
            _sub(ch, "NumSamples", str(sc.num_samples))
            _sub(ch, "SignalArrayByteOffset", str(sig_off))
            _sub(ch, "PVPArrayByteOffset", str(pvp_off))
            sig_off += sc.num_vectors * sc.num_samples * 8   # CF8
            pvp_off += sc.num_vectors * pvp_bytes

    def _build_tx_sequence(
        self,
        root: etree._Element,
        f0: float,
        bw: float,
        timing: Dict,
        tx_pol: str,
        iarp_ecf: np.ndarray,
        radar: Any,
    ) -> None:
        """Build the TxSequence section."""
        tx_pulse_length = float(
            getattr(radar, "tx_pulse_length_s", 52.0e-6)
        )
        chirp_rate = float(
            getattr(radar, "chirp_rate_hz_per_s", 779.038e9)
        )

        txseq = _sub(root, "TxSequence")
        _sub(txseq, "RefTxId", "TX1")
        _sub(txseq, "TxWFType", "LFM")
        params = _sub(txseq, "Parameters")
        _sub(params, "Identifier", "TX1")
        _sub(params, "RefPulseIndex", "0")
        _sub(params, "FxResponseId", "FXR1")
        _sub(params, "FxBWFixed", "true")
        _sub(params, "FxC", f"{f0:.12g}")
        _sub(params, "FxBW", f"{bw:.12g}")
        _sub(params, "TXmtMin", f"{tx_pulse_length:.12g}")
        _sub(params, "TXmtMax", f"{tx_pulse_length:.12g}")
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

    def _build_channel_section(
        self,
        root: etree._Element,
        swath_channels: Dict[str, _SwathChannel],
        f0: float,
        bw: float,
        timing: Dict,
        iarp_ecf: np.ndarray,
        radar: Any,
    ) -> None:
        """Build the Channel section."""
        sampling_rate = float(
            getattr(radar, "sampling_rate_hz", 64.345238e6)
        )
        per_ch = timing.get("per_channel", {})
        ch_root = _sub(root, "Channel")
        first_key = next(iter(swath_channels.keys()))
        _sub(ch_root, "RefChId", first_key)

        for key, sc in swath_channels.items():
            ch_t = per_ch.get(key, timing)
            params = _sub(ch_root, "Parameters")
            _sub(params, "Identifier", key)
            _sub(params, "RefVectorIndex", "0")
            _sub(params, "RefFreqFixed", "true")
            _sub(params, "FrcvFixed", "true")
            _sub(params, "SignalNormal", "true")
            _sub(params, "F0Ref", f"{f0:.12g}")
            _sub(params, "Fs", f"{sampling_rate:.12g}")
            _sub(params, "BWInst", f"{bw:.12g}")
            _sub(params, "RcvStartTime1", f"{ch_t['rcv_start_time1']:.12g}")
            _sub(params, "RcvStartTime2", f"{ch_t['rcv_start_time2']:.12g}")
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
            p2 = _sub(ia, "Polygon")
            p2.set("size", "4")
            for vi, (vx2, vy2) in enumerate(
                [(-1, -1), (1, -1), (1, 1), (-1, 1)], start=1,
            ):
                v2 = _sub_xy(p2, "Vertex", float(vx2), float(vy2))
                v2.set("index", str(vi))

    def _build_support_array_section(
        self,
        root: etree._Element,
        f0: float,
        bw: float,
    ) -> None:
        """Build the SupportArray section."""
        sa_root = _sub(root, "SupportArray")

        gpa = _sub(sa_root, "GainPhaseArray")
        _sub(gpa, "Identifier", "GP1")
        _sub(gpa, "ElementFormat", "Gain=F4;Phase=F4;")
        _sub(gpa, "X0", "0.0")
        _sub(gpa, "Y0", "-1.0")
        _sub(gpa, "XSS", "1.0")
        _sub(gpa, "YSS", "2.0")

        fxr = _sub(sa_root, "FxResponseArray")
        _sub(fxr, "Identifier", "FXR1")
        _sub(fxr, "ElementFormat", "Amp=F4;Phase=F4;")
        _sub(fxr, "Fx0FXR", f"{f0 - bw / 2.0:.12g}")
        _sub(fxr, "FxSSFXR", f"{bw:.12g}")

        dta = _sub(sa_root, "DwellTimeArray")
        _sub(dta, "Identifier", "DTA1")
        _sub(dta, "ElementFormat", "COD=F4;DT=F4;")
        _sub(dta, "X0", "0.0")
        _sub(dta, "Y0", "0.0")
        _sub(dta, "XSS", "1.0")
        _sub(dta, "YSS", "1.0")

    def _build_antenna_section(
        self,
        root: etree._Element,
        swath_channels: Dict[str, _SwathChannel],
        tx_pol: str,
        f0: float,
    ) -> None:
        """Build the Antenna section with placeholder flat patterns."""
        num_apats = 1 + len(swath_channels)
        ant = _sub(root, "Antenna")
        _sub(ant, "NumACFs", "1")
        _sub(ant, "NumAPCs", "2")
        _sub(ant, "NumAPATs", str(num_apats))

        acf = _sub(ant, "AntCoordFrame")
        _sub(acf, "Identifier", "ACF1")

        apc_tx = _sub(ant, "AntPhaseCenter")
        _sub(apc_tx, "Identifier", "APC_TX")
        _sub(apc_tx, "ACFId", "ACF1")
        _sub_xyz(apc_tx, "APCXYZ", 0.0, 0.0, 0.0)

        apc_rx = _sub(ant, "AntPhaseCenter")
        _sub(apc_rx, "Identifier", "APC_RX")
        _sub(apc_rx, "ACFId", "ACF1")
        _sub_xyz(apc_rx, "APCXYZ", 0.0, 0.0, 0.0)

        def _add_pattern(ident: str, gp_id: str) -> None:
            ap = _sub(ant, "AntPattern")
            _sub(ap, "Identifier", ident)
            _sub(ap, "FreqZero", f"{f0:.12g}")
            _sub(ap, "ArrayGPId", gp_id)
            _sub(ap, "ElemGPId", gp_id)
            ebfs = _sub(ap, "EBFreqShift")
            _sub(ebfs, "DCXSF", "0.0")
            _sub(ebfs, "DCYSF", "0.0")
            mlfd = _sub(ap, "MLFreqDilation")
            _sub(mlfd, "DCXSF", "0.0")
            _sub(mlfd, "DCYSF", "0.0")
            gbp = _sub(ap, "GainBSPoly")
            gbp.set("order1", "0")
            coef = _sub(gbp, "Coef", "0.0")
            coef.set("exponent1", "0")
            apr = _sub(ap, "AntPolRef")
            _sub(apr, "AmpX", "1.0")
            _sub(apr, "AmpY", "0.0")
            _sub(apr, "PhaseX", "0.0")
            _sub(apr, "PhaseY", "0.0")

        _add_pattern("APAT_TX", "GP1")
        for key in swath_channels:
            _add_pattern(f"APAT_RX_{key}", "GP1")

    # ---------------------------------------------------------------
    # Step 8: Write CRSD binary
    # ---------------------------------------------------------------

    def _write_crsd(
        self,
        xmltree: etree.ElementTree,
        reader: Sentinel1L0Reader,
        swath_channels: Dict[str, _SwathChannel],
        full_echo_dfs: Dict[str, Any],
        radar: Any,
        orbit: OrbitInterpolator,
        ref_time_gps: float,
    ) -> None:
        """Build all binary arrays and write the CRSD file.

        Parameters
        ----------
        xmltree : ElementTree
        reader : Sentinel1L0Reader
        swath_channels : dict
        full_echo_dfs : dict
        radar : S1L0RadarParameters
        orbit : OrbitInterpolator
        ref_time_gps : float
            GPS seconds at collection reference time.
        """
        first_pol = next(iter(full_echo_dfs.keys()))
        full_df = full_echo_dfs[first_pol]

        f0 = float(
            getattr(radar, "center_frequency_hz", SENTINEL1_CENTER_FREQUENCY_HZ)
        )
        bw = float(getattr(radar, "tx_bandwidth_hz", 42.79e6))
        tx_pulse_len = float(getattr(radar, "tx_pulse_length_s", 52.0e-6))
        chirp_rate = float(getattr(radar, "chirp_rate_hz_per_s", 779.038e9))

        # Try to compute reference geometry using sarkit
        ppp_for_rg = self._build_ppp_array(
            xmltree, full_df, orbit, ref_time_gps,
            f0, bw, tx_pulse_len, chirp_rate,
        )
        first_sc = next(iter(swath_channels.values()))
        pvp_for_rg = self._build_pvp_array(
            xmltree, first_sc.echo_df, orbit, ref_time_gps,
            f0, bw, global_indices=first_sc.global_indices,
        )

        try:
            rg_elem = sarkit.crsd.compute_reference_geometry(
                xmltree, pvps=pvp_for_rg, ppps=ppp_for_rg,
            )
            # Replace placeholder ReferenceGeometry element
            root = xmltree.getroot()
            old_rg = root.find(f"{{{CRSD_NS}}}ReferenceGeometry")
            if old_rg is not None:
                idx = list(root).index(old_rg)
                root.remove(old_rg)
                root.insert(idx, rg_elem)
        except Exception as exc:
            logger.warning(
                "Could not compute ReferenceGeometry via sarkit: %s; "
                "using law-of-cosines approximation",
                exc,
            )
            self._fill_placeholder_ref_geometry(
                xmltree, orbit, ref_time_gps,
            )

        crsd_meta = sarkit.crsd.Metadata(xmltree=xmltree)
        fh = open(str(self.output_path), "wb")
        writer = sarkit.crsd.Writer(fh, crsd_meta)

        try:
            # Support arrays
            gp_dtype = np.dtype([("Gain", "<f4"), ("Phase", "<f4")])
            writer.write_support_array(
                "GP1",
                np.array([(1.0, 0.0)], dtype=gp_dtype).reshape(1, 1),
            )
            fxr_dtype = np.dtype([("Amp", "<f4"), ("Phase", "<f4")])
            writer.write_support_array(
                "FXR1",
                np.array(
                    [(1.0, 0.0), (1.0, 0.0)], dtype=fxr_dtype,
                ).reshape(2, 1),
            )
            dta_dtype = np.dtype([("COD", "<f4"), ("DT", "<f4")])
            writer.write_support_array(
                "DTA1",
                np.array([(0.0, 0.0)], dtype=dta_dtype).reshape(1, 1),
            )

            # PPP (all pulses, first polarization — all swaths)
            ppp = self._build_ppp_array(
                xmltree, full_df, orbit, ref_time_gps,
                f0, bw, tx_pulse_len, chirp_rate,
            )
            writer.write_ppp("TX1", ppp)
            del ppp  # release PPP memory before signal reads

            # Memory-safe strategy: allocate the signal array as a
            # numpy.memmap backed by a sidecar temp file so only one
            # burst's decoded I/Q data is resident in RAM at a time.

            for key, sc in swath_channels.items():
                logger.info(
                    "Writing channel %s (swath %d, %s), "
                    "%d vectors × %d samples ...",
                    key, sc.swath_number, sc.polarization,
                    sc.num_vectors, sc.num_samples,
                )
                burst_reader = reader._burst_readers.get(sc.polarization)
                if burst_reader is None:
                    raise RuntimeError(
                        f"No burst reader for polarization {sc.polarization!r}"
                    )

                # Allocate signal as big-endian CF8 memmap backed by a
                # sidecar temp file alongside the output.  Only one burst's
                # decoded I/Q is resident in RAM at a time.
                _tmp_path = self.output_path.with_name(
                    f".{key}.sig.tmp"
                )
                try:
                    # Shape: (num_vectors, num_samples), dtype '>c8'
                    signal = np.memmap(
                        _tmp_path,
                        dtype=">c8",
                        mode="w+",
                        shape=(sc.num_vectors, sc.num_samples),
                    )
                    # Fill burst by burst — only one burst in RAM
                    row = 0
                    for burst_info, burst_iq in burst_reader.iter_bursts(
                        swath=sc.raw_swath
                    ):
                        if burst_iq is None or len(burst_iq) == 0:
                            continue
                        n_rows = burst_iq.shape[0]
                        n_cols = min(burst_iq.shape[1], sc.num_samples)
                        end = min(row + n_rows, sc.num_vectors)
                        actual = end - row
                        if actual <= 0:
                            break
                        # Convert to big-endian complex64 in-place slice
                        signal[row:end, :n_cols] = burst_iq[:actual, :n_cols].astype(">c8")
                        row = end
                        del burst_iq  # free burst memory immediately

                    if row != sc.num_vectors:
                        logger.warning(
                            "Channel %s: filled %d/%d vectors from bursts",
                            key, row, sc.num_vectors,
                        )

                    # Flush memmap to disk, then pass to sarkit writer
                    # (sarkit reads shape/dtype via array interface; since
                    # signal is already '>c8' the .astype call is a no-op)
                    signal.flush()
                    writer.write_signal(key, signal)
                    del signal  # unmap before removing temp file
                finally:
                    _tmp_path.unlink(missing_ok=True)

                pvp = self._build_pvp_array(
                    xmltree, sc.echo_df, orbit, ref_time_gps,
                    f0, bw, global_indices=sc.global_indices,
                )
                writer.write_pvp(key, pvp)
                logger.info("Channel %s written.", key)

            writer.done()

        finally:
            fh.close()

    # ---------------------------------------------------------------
    # PPP / PVP array builders
    # ---------------------------------------------------------------

    @staticmethod
    def _set_intfrac(
        arr: np.ndarray,
        field: str,
        int_vals: np.ndarray,
        frac_vals: np.ndarray,
    ) -> None:
        """Set integer and fractional parts of an IntFrac PPP/PVP field.

        sarkit exposes only the I8 part as a named dtype field; the F8
        part occupies the 8 bytes immediately following, accessed via a
        buffer view.

        Parameters
        ----------
        arr : ndarray
            Structured PPP or PVP array.
        field : str
            Name of the IntFrac field (I8 part).
        int_vals : ndarray of int64
        frac_vals : ndarray of float64
        """
        arr[field] = int_vals
        frac_offset = arr.dtype.fields[field][1] + 8
        frac_view = np.ndarray(
            len(arr),
            dtype=np.float64,
            buffer=arr.data,
            offset=frac_offset,
            strides=(arr.dtype.itemsize,),
        )
        frac_view[:] = frac_vals

    def _build_ppp_array(
        self,
        xmltree: etree.ElementTree,
        echo_df: Any,
        orbit: OrbitInterpolator,
        ref_time_gps: float,
        f0: float,
        bw: float,
        tx_pulse_len: float,
        chirp_rate: float,
    ) -> np.ndarray:
        """Build Per-Pulse Parameter array.

        Timing bias is applied so that ``TxTime`` records the true RF
        transmit event time, not the ISP datation timestamp.

        Parameters
        ----------
        xmltree : ElementTree
        echo_df : DataFrame
            Full-polarization echo DataFrame (all swaths).
        orbit : OrbitInterpolator
            Referenced to ``meta.start_time`` (UTC datetime).
        ref_time_gps : float
            Collection reference time in GPS seconds.
        f0, bw : float
            Center frequency and bandwidth in Hz.
        tx_pulse_len : float
            TX pulse length in seconds.
        chirp_rate : float
            Chirp rate in Hz/s.

        Returns
        -------
        ndarray
        """
        ppp_dtype = sarkit.crsd.get_ppp_dtype(xmltree)
        n = len(echo_df)
        ppp = np.zeros(n, dtype=ppp_dtype)

        coarse = echo_df["Coarse Time"].values.astype(np.float64)
        fine = echo_df["Fine Time"].values.astype(np.float64)
        # Fine Time is already fractional (decoded as raw/65536).
        # ISP Coarse+Fine = GPS seconds at the receive window start.
        # TX time = RcvStart - SWST - Rank*PRI  ([ISP] Table 3-2).
        abs_isp = coarse + fine
        if (
            "SWST" not in echo_df.columns
            or "Rank" not in echo_df.columns
            or "PRI" not in echo_df.columns
        ):
            raise ValueError(
                "echo_df missing SWST/Rank/PRI columns — "
                "cannot compute TX pulse origin times."
            )
        swst_tx = echo_df["SWST"].values.astype(np.float64)
        rank_tx = echo_df["Rank"].values.astype(np.float64)
        pri_tx = echo_df["PRI"].values.astype(np.float64)
        abs_rf = abs_isp - swst_tx - rank_tx * pri_tx

        # Relative to collection reference time (GPS seconds)
        rel_rf = abs_rf - ref_time_gps

        # Orbit: OrbitInterpolator uses times relative to its reference_time
        # (UTC datetime).  ISP Coarse Time is GPS time; GPS is ahead of UTC
        # by GPS_LEAP_SECONDS.  Subtract leap seconds to convert GPS → UTC
        # before computing relative times, otherwise all orbit queries are
        # shifted by ~18 s (~135 km positional error).
        ref_dt = orbit.reference_time
        orbit_ref_gps = _datetime_to_gps(ref_dt)
        orbit_rel_times = (abs_rf - GPS_LEAP_SECONDS) - orbit_ref_gps

        positions, velocities = orbit.interpolate(orbit_rel_times)
        acx, acy = _compute_antenna_frame(positions, velocities)

        t_int = np.floor(rel_rf).astype(np.int64)
        t_frac = rel_rf - t_int.astype(np.float64)

        self._set_intfrac(ppp, "TxTime", t_int, t_frac)
        ppp["TxPos"] = positions
        ppp["TxVel"] = velocities

        # Per-packet chirp parameters — sentinel1decoder provides these
        # per-pulse (Tx Ramp Rate = chirp rate in Hz/s; Tx Pulse Start
        # Frequency offsets the start of the chirp from the carrier;
        # Tx Pulse Length is the transmit duration in seconds).
        # Fall back to the scalar metadata values when columns are absent.
        if "Tx Pulse Length" in echo_df.columns:
            tx_pulse_len_arr = echo_df["Tx Pulse Length"].values.astype(np.float64)
        else:
            tx_pulse_len_arr = np.full(n, tx_pulse_len)

        if "Tx Ramp Rate" in echo_df.columns:
            chirp_rate_arr = echo_df["Tx Ramp Rate"].values.astype(np.float64)
        else:
            chirp_rate_arr = np.full(n, chirp_rate)

        if "Tx Pulse Start Frequency" in echo_df.columns:
            # TXPSF is a frequency offset from the carrier; FX1 = f0 + TXPSF
            txpsf = echo_df["Tx Pulse Start Frequency"].values.astype(np.float64)
            fx1_arr = f0 + txpsf
            fx2_arr = fx1_arr + np.abs(chirp_rate_arr) * tx_pulse_len_arr
        else:
            fx1_arr = np.full(n, f0 - bw / 2.0)
            fx2_arr = np.full(n, f0 + bw / 2.0)

        ppp["FX1"] = fx1_arr
        ppp["FX2"] = fx2_arr
        ppp["TXmt"] = tx_pulse_len_arr
        self._set_intfrac(
            ppp, "PhiX0", np.zeros(n, np.int64), np.zeros(n),
        )
        ppp["FxFreq0"] = fx1_arr
        ppp["FxRate"] = chirp_rate_arr
        ppp["TxRadInt"] = 1.0
        ppp["TxACX"] = acx
        ppp["TxACY"] = acy
        ppp["TxEB"] = np.zeros((n, 2))
        ppp["FxResponseIndex"] = 0

        return ppp

    def _build_pvp_array(
        self,
        xmltree: etree.ElementTree,
        echo_df: Any,
        orbit: OrbitInterpolator,
        ref_time_gps: float,
        f0: float,
        bw: float,
        global_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build Per-Vector Parameter array.

        Parameters
        ----------
        xmltree : ElementTree
        echo_df : DataFrame
            Swath-filtered echo DataFrame.
        orbit : OrbitInterpolator
        ref_time_gps : float
        f0, bw : float
        global_indices : ndarray, optional
            Row indices into the full (all-swath) PPP array used as
            ``TxPulseIndex``.  When *None*, sequential indices are used.

        Returns
        -------
        ndarray
        """
        pvp_dtype = sarkit.crsd.get_pvp_dtype(xmltree)
        n = len(echo_df)
        pvp = np.zeros(n, dtype=pvp_dtype)

        coarse = echo_df["Coarse Time"].values.astype(np.float64)
        fine = echo_df["Fine Time"].values.astype(np.float64)
        # ISP Coarse+Fine IS the GPS time of the first received echo
        # sample (receive window start).  RcvStart = abs_isp directly;
        # no subtraction or addition of SWST is needed here.
        abs_isp = coarse + fine
        abs_rcv = abs_isp
        rel_rcv = abs_rcv - ref_time_gps

        ref_dt = orbit.reference_time
        orbit_ref_gps = _datetime_to_gps(ref_dt)
        # GPS → UTC: subtract leap seconds before computing orbit-relative time
        orbit_rel_times = (abs_rcv - GPS_LEAP_SECONDS) - orbit_ref_gps

        positions, velocities = orbit.interpolate(orbit_rel_times)
        acx, acy = _compute_antenna_frame(positions, velocities)

        r_int = np.floor(rel_rcv).astype(np.int64)
        r_frac = rel_rcv - r_int.astype(np.float64)

        self._set_intfrac(pvp, "RcvStart", r_int, r_frac)
        pvp["RcvPos"] = positions
        pvp["RcvVel"] = velocities
        pvp["FRCV1"] = f0 - bw / 2.0
        pvp["FRCV2"] = f0 + bw / 2.0
        self._set_intfrac(
            pvp, "RefPhi0", np.zeros(n, np.int64), np.zeros(n),
        )
        pvp["RefFreq"] = f0
        pvp["DFIC0"] = 0.0
        pvp["FICRate"] = 0.0
        pvp["RcvACX"] = acx
        pvp["RcvACY"] = acy
        pvp["RcvEB"] = np.zeros((n, 2))
        pvp["SIGNAL"] = 1
        pvp["AmpSF"] = 1.0

        if "Rx Gain" in echo_df.columns:
            rx_gain_db = echo_df["Rx Gain"].values.astype(np.float64)
            pvp["DGRGC"] = np.power(10.0, rx_gain_db / 20.0)
        else:
            pvp["DGRGC"] = 1.0

        if global_indices is not None:
            pvp["TxPulseIndex"] = global_indices.astype(np.int64)
        else:
            pvp["TxPulseIndex"] = np.arange(n, dtype=np.int64)

        return pvp

    def _fill_placeholder_ref_geometry(
        self,
        xmltree: etree.ElementTree,
        orbit: OrbitInterpolator,
        ref_time_gps: float,
    ) -> None:
        """Fill ReferenceGeometry with law-of-cosines approximation.

        Used as fallback when sarkit's ``compute_reference_geometry``
        fails.  Geometry is computed using the spherical-Earth
        law-of-cosines formula for accurate incidence/graze angles.

        Parameters
        ----------
        xmltree : ElementTree
        orbit : OrbitInterpolator
        ref_time_gps : float
        """
        root = xmltree.getroot()
        rg = root.find(f"{{{CRSD_NS}}}ReferenceGeometry")
        if rg is None:
            rg = _sub(root, "ReferenceGeometry")

        ref_dt = orbit.reference_time
        orbit_ref_gps = _datetime_to_gps(ref_dt)
        # ref_time_gps is GPS; orbit reference is UTC → subtract leap seconds
        orbit_rel = (ref_time_gps - GPS_LEAP_SECONDS) - orbit_ref_gps
        pos, vel = orbit.interpolate_single(float(orbit_rel))

        # Get IARP ECF from XML
        iarp_ecf_elem = root.find(
            f".//{{{CRSD_NS}}}IARP/{{{CRSD_NS}}}ECF"
        )
        if iarp_ecf_elem is not None:
            iarp = np.array([
                float(iarp_ecf_elem.findtext(f"{{{CRSD_NS}}}X") or 0),
                float(iarp_ecf_elem.findtext(f"{{{CRSD_NS}}}Y") or 0),
                float(iarp_ecf_elem.findtext(f"{{{CRSD_NS}}}Z") or 0),
            ])
        else:
            iarp = np.zeros(3)

        look = iarp - pos
        slant_range = float(np.linalg.norm(look))
        look_unit = look / slant_range if slant_range > 0.0 else look
        vel_norm = float(np.linalg.norm(vel))
        vel_unit = vel / vel_norm if vel_norm > 0.0 else vel
        dop_cone = float(
            np.degrees(
                np.arccos(np.clip(np.dot(look_unit, vel_unit), -1.0, 1.0))
            )
        )

        # Side of track
        cross = np.cross(vel, look)
        pos_unit = pos / float(np.linalg.norm(pos))
        side = "L" if np.dot(cross, pos_unit) > 0 else "R"

        # Law-of-cosines incidence/graze geometry
        R_E = MEAN_EARTH_RADIUS
        R_sat = float(np.linalg.norm(pos))
        R_sl = slant_range
        cos_ea = np.clip(
            (R_sat ** 2 + R_E ** 2 - R_sl ** 2)
            / (2.0 * R_sat * R_E),
            -1.0,
            1.0,
        )
        earth_angle = float(np.arccos(cos_ea))
        ground_range = R_E * earth_angle
        sin_inc = np.clip(R_sat * np.sin(earth_angle) / R_sl, -1.0, 1.0)
        incidence = float(np.degrees(np.arcsin(sin_inc)))
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


# =============================================================================
# Helpers
# =============================================================================

def _datetime_to_gps(dt: datetime) -> float:
    """Convert a timezone-aware or naive UTC datetime to GPS seconds.

    GPS epoch: 6 January 1980 00:00:00 UTC.  No leap-second correction
    is applied (GPS seconds are continuous, UTC is not).

    Parameters
    ----------
    dt : datetime
        UTC datetime (aware or naive).

    Returns
    -------
    float
        GPS seconds since the GPS epoch.
    """
    from datetime import timedelta
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (_GPS_EPOCH - _GPS_EPOCH).total_seconds() + (
        dt - _GPS_EPOCH
    ).total_seconds()
