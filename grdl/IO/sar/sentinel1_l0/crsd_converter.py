# -*- coding: utf-8 -*-
"""
Sentinel-1 Level 0 to CRSD Converter.

Orchestrates the conversion of a Sentinel-1 IW Level 0 SAFE product
into a NITF CRSD (Compensated Received Signal Data) file. The output
is a per-polarization CRSD file containing one channel per burst,
with raw (uncompressed) I/Q signal data in CI2 format.

The pipeline:
1. Open the SAFE product and decode L0 packets
2. Load precise orbit data (POEORB)
3. For each burst: extract timing, interpolate orbit, compute geometry
4. Build CRSD XML metadata tree
5. Write CRSD via ``sarkit.crsd.Writer``

Dependencies
------------
lxml
sarkit
sentinel1decoder

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
2026-05-27
"""

# Standard library
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np
from lxml import etree

# GRDL internal
from grdl.IO.sar._backend import require_sarkit
from grdl.IO.sar.sentinel1_l0.constants import (
    BURST_GAP_THRESHOLD_US,
    GPS_EPOCH_DAY,
    GPS_EPOCH_MONTH,
    GPS_EPOCH_YEAR,
    GPS_LEAP_SECONDS,
    IW_MODE_PARAMS,
    SENTINEL1_CENTER_FREQUENCY_HZ,
    WGS84_SEMI_MAJOR_AXIS,
)
from grdl.IO.sar.sentinel1_l0.decoder import Sentinel1Decoder
from grdl.IO.sar.sentinel1_l0.crsd_metadata_builder import (
    BurstChannelInfo,
    CRSDMetadataBuilder,
    CRSDReferenceGeometryInfo,
    CRSDSceneInfo,
    ecef_to_geodetic,
)
from grdl.IO.sar.crsd_pvp_builder import (
    build_ppp_array,
    build_pvp_array,
    get_ppp_dtype,
    get_pvp_dtype,
    quantize_to_ci2,
)
from grdl.IO.sar.sentinel1_l0.geometry import GeometryCalculator
from grdl.IO.sar.sentinel1_l0.orbit import (
    OrbitLoader,
    ensure_poe_available,
)
from grdl.IO.sar.sentinel1_l0.safe_product import SAFEProduct
from grdl.IO.sar.sentinel1_l0.timing import (
    TimingCalculator,
    determine_reference_time,
)

require_sarkit('CRSD')
import sarkit.crsd

logger = logging.getLogger(__name__)

# GPS epoch for time conversion (naive UTC — consistent with
# POEParser and TimingCalculator which use naive datetimes)
_GPS_EPOCH = datetime(
    GPS_EPOCH_YEAR, GPS_EPOCH_MONTH, GPS_EPOCH_DAY,
)


# =====================================================================
# Swath mapping
# =====================================================================

# Raw swath numbers → human-readable IW names
# Sentinel-1 IW mode uses swath numbers 10/11/12 for echo data;
# 43 and 93 are calibration packets (Rx Cal, Tx Cal).
_SWATH_NAME_MAP = {
    1: "IW1", 2: "IW2", 3: "IW3",
    10: "IW1", 11: "IW2", 12: "IW3",
}
_CALIBRATION_SWATHS = {43, 93}


def _swath_name(swath_num: int) -> str:
    """Convert raw swath number to IW name."""
    return _SWATH_NAME_MAP.get(swath_num, f"SW{swath_num}")


# =====================================================================
# Channel identifier
# =====================================================================


def _make_channel_id(
    relative_orbit: int,
    burst_cycle: int,
    swath_name: str,
) -> str:
    """Build NGA-style channel identifier.

    Format: ``{relative_orbit:03d}_{burst_cycle:06d}_{swath_name}``
    e.g. ``"043_090466_IW1"``.
    """
    return f"{relative_orbit:03d}_{burst_cycle:06d}_{swath_name}"


# =====================================================================
# GPS / UTC time helpers
# =====================================================================


def _gps_seconds_to_utc(gps_seconds: float) -> datetime:
    """Convert GPS seconds-since-epoch to UTC datetime."""
    gps_dt = _GPS_EPOCH + timedelta(seconds=gps_seconds)
    return gps_dt - timedelta(seconds=GPS_LEAP_SECONDS)


def _utc_to_relative(
    utc_dt: datetime, ref_time: datetime,
) -> float:
    """Seconds from ref_time to utc_dt."""
    return (utc_dt - ref_time).total_seconds()


def _gps_to_relative(
    gps_seconds: float, ref_time: datetime,
) -> float:
    """GPS seconds-since-epoch to seconds relative to ref_time."""
    utc = _gps_seconds_to_utc(gps_seconds)
    return _utc_to_relative(utc, ref_time)


# =====================================================================
# Echo burst descriptor
# =====================================================================


@dataclass
class _EchoBurst:
    """Internal burst descriptor with correct packet indices.

    Unlike ``BurstInfo`` from the burst reader, this stores the
    *original* DataFrame index positions so that ``iloc`` slicing
    on the full packet DataFrame returns only the correct packets
    for this burst.
    """

    swath_num: int
    swath_name: str
    packet_iloc: np.ndarray  # original iloc positions in full DF
    num_lines: int
    num_samples: int  # 2 × Number of Quads
    reference_time_gps: float  # GPS seconds of first packet
    pri: float
    rank: int
    swst: float


@dataclass
class _ChannelState:
    """Per-channel arrays needed for polarization and PVP/PPP.

    Kept separate from :class:`BurstChannelInfo` so channel metadata
    remains a stable schema and does not rely on dynamic attributes.
    """

    tx_times: np.ndarray
    tx_pos: np.ndarray
    tx_vel: np.ndarray
    rcv_pos: np.ndarray
    rcv_vel: np.ndarray
    tx_acx: np.ndarray
    tx_acy: np.ndarray
    rcv_acx: np.ndarray
    rcv_acy: np.ndarray


# =====================================================================
# Converter
# =====================================================================


class Sentinel1L0ToCRSD:
    """Convert Sentinel-1 IW Level 0 SAFE to CRSD.

    Parameters
    ----------
    safe_path : str or Path
        Path to the ``.SAFE`` directory.
    poe_path : str or Path, optional
        Path to a POEORB file or directory containing POEORB files.
        If a directory, the correct file is auto-selected.
    polarization : str
        Polarization to convert (``"VV"``, ``"VH"``, etc.).
    relative_orbit : int, optional
        Relative orbit number. If ``None``, extracted from product
        name or defaults to 0.
    output_dir : str or Path, optional
        Output directory. Defaults to current directory.
    swaths : list of str, optional
        Sub-swaths to include (e.g. ``["IW1", "IW2", "IW3"]``).
        Defaults to all detected sub-swaths.
    """

    def __init__(
        self,
        safe_path: Union[str, Path],
        poe_path: Optional[Union[str, Path]] = None,
        polarization: str = "VV",
        relative_orbit: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        swaths: Optional[List[str]] = None,
    ) -> None:
        self.safe_path = Path(safe_path)
        self.poe_path = (
            Path(poe_path) if poe_path is not None else None
        )
        self.polarization = polarization.upper()
        self.relative_orbit = relative_orbit or 0
        self.output_dir = (
            Path(output_dir) if output_dir else Path(".")
        )
        self.swaths = swaths

        # Populated during convert()
        self._safe: Optional[SAFEProduct] = None
        self._decoder: Optional[Sentinel1Decoder] = None
        self._geometry: Optional[GeometryCalculator] = None
        self._timing: Optional[TimingCalculator] = None
        self._ref_time: Optional[datetime] = None
        self._product_name: str = ""
        self._sensor_name: str = "S1A"

    def convert(self) -> Path:
        """Run the full conversion pipeline.

        Returns
        -------
        Path
            Path to the generated CRSD file.
        """
        logger.info(
            "Starting S1 L0 → CRSD conversion: %s [%s]",
            self.safe_path.name, self.polarization,
        )

        # Step 1: Open SAFE product
        self._open_safe()

        # Step 2: Load orbit
        self._load_orbit()

        # Step 3: Detect bursts and compute per-burst metadata
        burst_channels, burst_data, channel_states = (
            self._process_bursts()
        )

        if not burst_channels:
            raise ValueError(
                f"No bursts found for polarization "
                f"{self.polarization}"
            )

        # Step 4: Compute scene geometry
        scene = self._compute_scene(burst_channels)
        self._compute_channel_polarization_params(
            burst_channels, scene, channel_states,
        )
        ref_geom = self._compute_reference_geometry(
            burst_channels, scene,
        )

        # Step 5: Build XML metadata
        builder = CRSDMetadataBuilder(
            product_name=self._product_name,
            collection_ref_time=self._ref_time,
            sensor_name=self._sensor_name,
            channels=burst_channels,
            scene=scene,
            ref_geometry=ref_geom,
        )
        xmltree = builder.build()

        # Step 6: Write CRSD file
        output_path = self._write_crsd(
            xmltree,
            burst_channels,
            burst_data,
            channel_states,
        )

        logger.info("CRSD written: %s", output_path)
        return output_path

    # -----------------------------------------------------------------
    # Step 1: Open SAFE
    # -----------------------------------------------------------------

    def _open_safe(self) -> None:
        """Open and validate the SAFE product."""
        self._safe = SAFEProduct(
            self.safe_path, validate=True,
        )
        pi = self._safe.product_info
        if pi is not None:
            self._product_name = pi.full_name
            self._sensor_name = f"S1{pi.mission}"
            if self.relative_orbit == 0 and pi.orbit_number:
                # Compute relative orbit from absolute
                abs_orbit = pi.orbit_number
                self.relative_orbit = (
                    (abs_orbit - 73) % 175 + 1
                )
        else:
            self._product_name = self.safe_path.stem

        # Find measurement file for the requested polarization
        meas_files = [
            mf for mf in self._safe.measurement_files
            if mf.polarization.upper() == self.polarization
        ]
        if not meas_files:
            raise FileNotFoundError(
                f"No measurement file for polarization "
                f"{self.polarization} in {self.safe_path}"
            )
        mf = meas_files[0]
        self._decoder = Sentinel1Decoder(mf.measurement_file)
        self._decoder.decode_metadata()
        logger.info(
            "Opened SAFE: %s, pol=%s, file=%s (%d packets)",
            self._product_name, self.polarization,
            mf.measurement_file.name,
            len(self._decoder._metadata_df),
        )

    # -----------------------------------------------------------------
    # Step 2: Load orbit
    # -----------------------------------------------------------------

    def _load_orbit(self) -> None:
        """Load POE orbit data and create geometry calculator.

        Searches for a local POE file first.  If none is found,
        attempts to download from the ASF orbit API (backed by
        the Copernicus Data Space mirror on AWS) or ESA's orbit
        archive.  If download also fails, raises with manual
        download instructions.
        """
        df = self._decoder._metadata_df
        echo_mask = ~df["Swath Number"].isin(
            _CALIBRATION_SWATHS,
        )
        echo_df = df[echo_mask]
        if echo_df.empty:
            raise ValueError(
                "No echo packets in measurement file"
            )

        # First echo packet time (GPS seconds)
        first_coarse = float(echo_df["Coarse Time"].iloc[0])
        first_fine = float(echo_df["Fine Time"].iloc[0])
        first_gps = first_coarse + first_fine
        first_utc = _gps_seconds_to_utc(first_gps)

        # Scene name for ASF API lookup (SAFE dir minus .SAFE)
        scene_name = self.safe_path.stem
        if scene_name.endswith(".SAFE"):
            scene_name = scene_name[:-5]

        # Determine search directory
        if self.poe_path is not None:
            p = Path(self.poe_path)
            if p.is_file() and p.suffix.upper() == ".EOF":
                # Explicit POE file — load directly
                orbit_loader = OrbitLoader()
                orbit_loader.load_poe_file(p)
                orbit_vectors = orbit_loader.get_vectors(
                    prefer_poe=True,
                )
                self._ref_time = determine_reference_time(
                    first_utc,
                )
                self._geometry = GeometryCalculator(
                    orbit_vectors=orbit_vectors,
                    reference_time=self._ref_time,
                )
                self._timing = TimingCalculator(
                    t_ref=self._ref_time,
                    prf_hz=IW_MODE_PARAMS[
                        "pulse_repetition_frequency_hz"
                    ],
                    pulse_duration_s=IW_MODE_PARAMS[
                        "tx_pulse_length_s"
                    ],
                    range_sampling_rate_hz=IW_MODE_PARAMS[
                        "range_sampling_rate_hz"
                    ],
                )
                logger.info(
                    "Orbit loaded from file: %s (%d vectors)",
                    p.name, len(orbit_vectors),
                )
                return
            search_dir = p if p.is_dir() else p.parent
        else:
            # Default: search alongside the SAFE product
            search_dir = self.safe_path.parent

        # ensure_poe_available: local search → ASF API → ESA →
        # raises FileNotFoundError with download instructions
        poe_file = ensure_poe_available(
            target_time=first_utc,
            mission=self._sensor_name,
            scene_name=scene_name,
            search_dir=search_dir,
            download=True,
        )

        orbit_loader = OrbitLoader()
        orbit_loader.load_poe_file(poe_file)

        orbit_vectors = orbit_loader.get_vectors(prefer_poe=True)

        # Reference time = first burst time truncated to whole second
        self._ref_time = determine_reference_time(first_utc)

        # Create geometry calculator
        self._geometry = GeometryCalculator(
            orbit_vectors=orbit_vectors,
            reference_time=self._ref_time,
        )

        # Create timing calculator
        self._timing = TimingCalculator(
            t_ref=self._ref_time,
            prf_hz=IW_MODE_PARAMS[
                "pulse_repetition_frequency_hz"
            ],
            pulse_duration_s=IW_MODE_PARAMS[
                "tx_pulse_length_s"
            ],
            range_sampling_rate_hz=IW_MODE_PARAMS[
                "range_sampling_rate_hz"
            ],
        )
        logger.info(
            "Orbit loaded: %d vectors, ref_time=%s",
            len(orbit_vectors), self._ref_time.isoformat(),
        )

    # -----------------------------------------------------------------
    # Step 3: Process bursts
    # -----------------------------------------------------------------

    def _detect_echo_bursts(self) -> List[_EchoBurst]:
        """Detect echo bursts directly from packet metadata.

        Groups packets by swath, filters out calibration packets,
        and detects burst boundaries by inter-packet time gaps.
        Returns bursts with correct original-DataFrame iloc
        positions so that signal decoding retrieves only the
        intended echo packets.

        Returns
        -------
        list of _EchoBurst
            Echo bursts sorted by swath then time.
        """
        df = self._decoder._metadata_df
        echo_mask = ~df["Swath Number"].isin(
            _CALIBRATION_SWATHS,
        )

        # Positional indices (for iloc) of echo packets
        echo_iloc = np.where(echo_mask.values)[0]
        echo_df = df.iloc[echo_iloc].copy()
        echo_df["_iloc"] = echo_iloc

        if echo_df.empty:
            return []

        gap_threshold_us = BURST_GAP_THRESHOLD_US
        bursts: List[_EchoBurst] = []

        for swath_num, swath_df in echo_df.groupby(
            "Swath Number",
        ):
            sname = _swath_name(int(swath_num))
            if self.swaths and sname not in self.swaths:
                continue

            # Original iloc positions in the full DataFrame
            orig_positions = swath_df["_iloc"].values

            coarse = swath_df["Coarse Time"].values
            fine = swath_df["Fine Time"].values
            times_us = (coarse + fine) * 1e6

            # Burst boundaries at large time gaps
            diffs = np.diff(times_us)
            gap_idx = np.where(
                diffs > gap_threshold_us,
            )[0]

            # Build segments: [0, gap1+1), [gap1+1, gap2+1), ...
            seg_starts = np.concatenate(
                [[0], gap_idx + 1],
            )
            seg_ends = np.concatenate(
                [gap_idx + 1, [len(swath_df)]],
            )

            for seg_start, seg_end in zip(
                seg_starts, seg_ends,
            ):
                n_lines = int(seg_end - seg_start)
                if n_lines < 10:
                    continue  # skip tiny fragments

                pkt_iloc = orig_positions[seg_start:seg_end]
                seg_df = swath_df.iloc[seg_start:seg_end]

                # Number of quads → use mode (most common)
                quad_vals = seg_df[
                    "Number of Quads"
                ].values
                vals, counts = np.unique(
                    quad_vals, return_counts=True,
                )
                mode_quads = int(vals[np.argmax(counts)])
                n_samples = mode_quads * 2

                # Timing from first few packets
                ref_gps = float(
                    coarse[seg_start] + fine[seg_start]
                )
                pri_vals = seg_df["PRI"].values[:10]
                pri = float(np.mean(pri_vals))
                rank = int(seg_df["Rank"].iloc[0])
                swst_vals = seg_df["SWST"].values[:10]
                swst = float(np.mean(swst_vals))

                bursts.append(_EchoBurst(
                    swath_num=int(swath_num),
                    swath_name=sname,
                    packet_iloc=pkt_iloc,
                    num_lines=n_lines,
                    num_samples=n_samples,
                    reference_time_gps=ref_gps,
                    pri=pri,
                    rank=rank,
                    swst=swst,
                ))

        # Filter partial edge bursts (< 90% of median per swath)
        if bursts:
            swath_bursts: Dict[str, List[_EchoBurst]] = {}
            for b in bursts:
                swath_bursts.setdefault(
                    b.swath_name, [],
                ).append(b)
            filtered = []
            for sname, slist in swath_bursts.items():
                if len(slist) >= 3:
                    median_lines = float(np.median(
                        [b.num_lines for b in slist],
                    ))
                    min_lines = int(median_lines * 0.9)
                    slist = [
                        b for b in slist
                        if b.num_lines >= min_lines
                    ]
                filtered.extend(slist)
            bursts = filtered

        # Sort by swath name then time
        bursts.sort(
            key=lambda b: (b.swath_name, b.reference_time_gps),
        )
        return bursts

    def _process_bursts(
        self,
    ) -> Tuple[
        List[BurstChannelInfo],
        Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        Dict[str, _ChannelState],
    ]:
        """Process all bursts: decode, compute timing, geometry.

        Returns
        -------
        channels : list of BurstChannelInfo
            Metadata per burst/channel.
        burst_data : dict
            ``{channel_id: (ci2_signal, amp_sf, pvp_times)}``
        channel_states : dict
            ``{channel_id: _ChannelState}`` for per-channel arrays.
        """
        echo_bursts = self._detect_echo_bursts()

        if not echo_bursts:
            raise ValueError(
                f"No echo bursts found for polarization "
                f"{self.polarization}",
            )

        channels: List[BurstChannelInfo] = []
        burst_data: Dict[
            str, Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        channel_states: Dict[str, _ChannelState] = {}

        # Burst cycle base from GPS time
        cycle_duration = IW_MODE_PARAMS.get(
            "burst_cycle_duration_s", 2.722,
        )
        first_gps = echo_bursts[0].reference_time_gps
        burst_cycle_base = int(first_gps / cycle_duration)

        # Group by swath for logging
        swath_counts: Dict[str, int] = {}
        for b in echo_bursts:
            swath_counts[b.swath_name] = (
                swath_counts.get(b.swath_name, 0) + 1
            )
        for sname, cnt in sorted(swath_counts.items()):
            logger.info(
                "Detected %s: %d echo bursts", sname, cnt,
            )

        # Per-swath burst index for cycle numbering
        swath_idx: Dict[str, int] = {}

        for burst in echo_bursts:
            sname = burst.swath_name
            idx = swath_idx.get(sname, 0)
            swath_idx[sname] = idx + 1

            burst_cycle = burst_cycle_base + idx
            chan_id = _make_channel_id(
                self.relative_orbit, burst_cycle, sname,
            )

            channel_info, data, state = self._process_single_burst(
                burst, chan_id,
            )
            channels.append(channel_info)
            burst_data[chan_id] = data
            channel_states[chan_id] = state

            logger.debug(
                "  Burst %s: %d vectors × %d samples",
                chan_id, burst.num_lines, burst.num_samples,
            )

        return channels, burst_data, channel_states

    def _process_single_burst(
        self,
        burst: _EchoBurst,
        channel_id: str,
    ) -> Tuple[
        BurstChannelInfo,
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        _ChannelState,
    ]:
        """Process a single burst → channel info + binary data.

        Returns
        -------
        channel_info : BurstChannelInfo
        data : (ci2_signal, amp_sf, rcv_times)
        state : _ChannelState
            Per-channel arrays for polarization and PVP/PPP building.
        """
        n_lines = burst.num_lines
        n_samples = burst.num_samples

        # -- Timing --
        burst_start_utc = _gps_seconds_to_utc(
            burst.reference_time_gps,
        )
        pri_s = burst.pri if burst.pri > 0 else (
            1.0 / IW_MODE_PARAMS[
                "pulse_repetition_frequency_hz"
            ]
        )

        # TX times relative to CollectionRefTime
        t0_rel = _utc_to_relative(
            burst_start_utc, self._ref_time,
        )
        tx_times = t0_rel + np.arange(n_lines) * pri_s

        # RCV times: tx_time + rank * PRI + SWST
        rank = burst.rank if burst.rank > 0 else 0
        swst_s = burst.swst if burst.swst > 0 else 0.0
        rcv_delay = rank * pri_s + swst_s
        rcv_times = tx_times + rcv_delay
        # CRSD expects per-vector receive start times to lie on the
        # sample clock grid relative to the first vector in channel.
        sample_period = 1.0 / IW_MODE_PARAMS[
            "range_sampling_rate_hz"
        ]
        rcv_times = rcv_times[0] + np.round(
            (rcv_times - rcv_times[0]) / sample_period,
        ) * sample_period

        # -- Orbit interpolation --
        tx_pos, tx_vel = (
            self._geometry.interpolate_position_velocity(
                tx_times,
            )
        )
        rcv_pos, rcv_vel = (
            self._geometry.interpolate_position_velocity(
                rcv_times,
            )
        )

        # ACF vectors (antenna coordinate frame)
        _, _, tx_acx, tx_acy = (
            self._geometry.get_state_vectors_at_times(
                tx_times,
            )
        )
        _, _, rcv_acx, rcv_acy = (
            self._geometry.get_state_vectors_at_times(
                rcv_times,
            )
        )

        # -- Signal data --
        # Decode only this burst's echo packets using their
        # original iloc positions in the full DataFrame.
        full_df = self._decoder._metadata_df
        burst_df = full_df.iloc[burst.packet_iloc]
        signal = self._decoder.decode_packets(
            packet_df=burst_df,
        )

        # Handle shape: decoder returns (N, 2*quads) when all
        # packets have the same quad count.
        if signal.ndim == 1:
            # Fallback: reshape if total elements match
            expected = n_lines * n_samples
            if signal.size >= expected:
                signal = signal[:expected].reshape(
                    n_lines, n_samples,
                )
            else:
                raise ValueError(
                    f"Signal size {signal.size} < expected "
                    f"{n_lines}×{n_samples} for {channel_id}"
                )
        else:
            # Trim to expected dimensions
            if signal.shape[0] > n_lines:
                signal = signal[:n_lines]
            if signal.shape[1] > n_samples:
                signal = signal[:, :n_samples]

        # Quantize to CI2
        ci2, amp_sf = quantize_to_ci2(signal)

        # -- Waveform parameters --
        fs = IW_MODE_PARAMS["range_sampling_rate_hz"]
        bw = IW_MODE_PARAMS["chirp_bandwidth_hz"]
        tx_dur = IW_MODE_PARAMS["tx_pulse_length_s"]
        ramp_rate = IW_MODE_PARAMS[
            "tx_pulse_ramp_rate_hz_per_s"
        ]

        # F0Ref = center frequency of digitized band
        f0_ref = SENTINEL1_CENTER_FREQUENCY_HZ
        # BWInst = instantaneous signal bandwidth (chirp sweep).
        # Using Fs here drives OSR to ~1 and triggers consistency
        # failures; CRSD checks expect receive sampling to over-sample
        # the occupied signal bandwidth.
        bw_inst = bw

        # Chirp start frequency (baseband offset)
        # fx_freq0 = f0_ref - ramp_rate * tx_dur / 2
        fx_freq0 = f0_ref - ramp_rate * tx_dur / 2.0
        fx_bw = ramp_rate * tx_dur

        # Reference vector at mid-burst
        ref_idx = n_lines // 2

        # Build channel info
        channel_info = BurstChannelInfo(
            identifier=channel_id,
            swath_name=burst.swath_name,
            polarization=self.polarization,
            num_vectors=n_lines,
            num_samples=n_samples,
            tx_time_first=float(tx_times[0]),
            tx_time_last=float(tx_times[-1]),
            rcv_start_first=float(rcv_times[0]),
            rcv_start_last=float(rcv_times[-1]),
            f0_ref=f0_ref,
            fs=fs,
            bw_inst=bw_inst,
            fx_freq0=fx_freq0,
            fx_rate=ramp_rate,
            fx_bw=fx_bw,
            tx_pulse_duration=tx_dur,
            ref_vector_index=ref_idx,
            tx_ref_pos=tx_pos[ref_idx],
            tx_ref_vel=tx_vel[ref_idx],
        )

        # Pack data for later writing
        data = (ci2, amp_sf, rcv_times)
        state = _ChannelState(
            tx_times=tx_times,
            tx_pos=tx_pos,
            tx_vel=tx_vel,
            rcv_pos=rcv_pos,
            rcv_vel=rcv_vel,
            tx_acx=tx_acx,
            tx_acy=tx_acy,
            rcv_acx=rcv_acx,
            rcv_acy=rcv_acy,
        )

        return channel_info, data, state

    # -----------------------------------------------------------------
    # Step 4: Scene geometry
    # -----------------------------------------------------------------

    def _compute_channel_polarization_params(
        self,
        channels: List[BurstChannelInfo],
        scene: CRSDSceneInfo,
        channel_states: Dict[str, _ChannelState],
    ) -> None:
        """Populate per-channel Tx/Rcv HV polarization parameters.

        These values are derived from APC geometry and antenna
        X/Y polarization references using the same Sarkit utility
        used by CrsdConsistency checks.
        """
        for ch in channels:
            state = channel_states[ch.identifier]
            idx = int(np.clip(
                ch.ref_vector_index, 0, ch.num_vectors - 1,
            ))

            tx_pol = ch.polarization[0]
            rcv_pol = ch.polarization[-1]

            if tx_pol == "V":
                tx_amp_x, tx_amp_y = 0.0, 1.0
            else:
                tx_amp_x, tx_amp_y = 1.0, 0.0

            if rcv_pol == "V":
                rcv_amp_x, rcv_amp_y = 0.0, 1.0
            else:
                rcv_amp_x, rcv_amp_y = 1.0, 0.0

            tx_params = sarkit.crsd.compute_h_v_pol_parameters(
                state.tx_pos[idx],
                state.tx_acx[idx],
                state.tx_acy[idx],
                scene.iarp_ecf,
                1,
                tx_amp_x,
                tx_amp_y,
                0.0,
                0.0,
            )
            rcv_params = sarkit.crsd.compute_h_v_pol_parameters(
                state.rcv_pos[idx],
                state.rcv_acx[idx],
                state.rcv_acy[idx],
                scene.iarp_ecf,
                -1,
                rcv_amp_x,
                rcv_amp_y,
                0.0,
                0.0,
            )

            ch.tx_pol_params = (
                float(tx_params[0]),
                float(tx_params[1]),
                float(tx_params[2]),
                float(tx_params[3]),
            )
            ch.rcv_pol_params = (
                float(rcv_params[0]),
                float(rcv_params[1]),
                float(rcv_params[2]),
                float(rcv_params[3]),
            )

    def _compute_scene(
        self, channels: List[BurstChannelInfo],
    ) -> CRSDSceneInfo:
        """Compute scene-level geometry from all channels."""
        # IARP = ground point at mid-collection
        mid_ch = channels[len(channels) // 2]
        ref_pos = mid_ch.tx_ref_pos
        ref_vel = mid_ch.tx_ref_vel

        # Project to ground: nadir point approximation
        # Use platform position → Earth intersection
        ref_ecf_norm = ref_pos / np.linalg.norm(ref_pos)
        earth_surface = ref_ecf_norm * WGS84_SEMI_MAJOR_AXIS
        iarp_ecf = earth_surface

        lat, lon, hae = ecef_to_geodetic(
            iarp_ecf[0], iarp_ecf[1], iarp_ecf[2],
        )

        # Image area axes from local tangent frame at IARP.
        # uIAX is the platform velocity projected onto tangent plane;
        # uIAY completes a right-handed frame with local up.
        up = iarp_ecf / np.linalg.norm(iarp_ecf)
        v_tan = ref_vel - np.dot(ref_vel, up) * up
        if np.linalg.norm(v_tan) < 1e-9:
            # Robust fallback if velocity projection is degenerate.
            trial = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(trial, up)) > 0.9:
                trial = np.array([1.0, 0.0, 0.0])
            v_tan = np.cross(up, trial)
        uiax = v_tan / np.linalg.norm(v_tan)
        uiay = np.cross(up, uiax)
        uiay /= np.linalg.norm(uiay)

        # Image area extents in IAC (metres)
        # Compute from all channel nadir projections (ground points),
        # not from platform positions.
        all_ground_pos = np.array([
            (ch.tx_ref_pos / np.linalg.norm(ch.tx_ref_pos))
            * WGS84_SEMI_MAJOR_AXIS
            for ch in channels
        ])
        offsets = all_ground_pos - iarp_ecf
        x_coords = offsets @ uiax
        y_coords = offsets @ uiay

        margin = 5000.0
        x1 = float(np.min(x_coords)) - margin
        y1 = float(np.min(y_coords)) - margin
        x2 = float(np.max(x_coords)) + margin
        y2 = float(np.max(y_coords)) + margin

        # Corner coordinates (approximate geodetic)
        corners = []
        for cx, cy in [
            (x1, y1), (x1, y2), (x2, y2), (x2, y1),
        ]:
            corner_ecf = iarp_ecf + cx * uiax + cy * uiay
            corner_ecf *= (
                WGS84_SEMI_MAJOR_AXIS
                / np.linalg.norm(corner_ecf)
            )
            clat, clon, _ = ecef_to_geodetic(
                corner_ecf[0], corner_ecf[1], corner_ecf[2],
            )
            corners.append((clat, clon))

        return CRSDSceneInfo(
            iarp_ecf=iarp_ecf,
            iarp_llh=(lat, lon, hae),
            uiax=uiax,
            uiay=uiay,
            image_area_x1y1=(x1, y1),
            image_area_x2y2=(x2, y2),
            corner_coords=corners,
        )

    def _compute_reference_geometry(
        self,
        channels: List[BurstChannelInfo],
        scene: CRSDSceneInfo,
    ) -> CRSDReferenceGeometryInfo:
        """Compute reference geometry at collection center."""
        mid_ch = channels[len(channels) // 2]
        cod_time = (
            mid_ch.tx_time_first + mid_ch.tx_time_last
        ) / 2.0
        dwell_time = (
            mid_ch.tx_time_last - mid_ch.tx_time_first
        )

        return CRSDReferenceGeometryInfo(
            ref_pos_ecf=scene.iarp_ecf,
            ref_pos_iac=(0.0, 0.0),
            cod_time=cod_time,
            dwell_time=dwell_time,
            platform_pos=mid_ch.tx_ref_pos,
            platform_vel=mid_ch.tx_ref_vel,
            side_of_track="R",
        )

    # -----------------------------------------------------------------
    # Step 6: Write CRSD
    # -----------------------------------------------------------------

    def _write_crsd(
        self,
        xmltree: etree._ElementTree,
        channels: List[BurstChannelInfo],
        burst_data: Dict[
            str, Tuple[np.ndarray, np.ndarray, np.ndarray]
        ],
        channel_states: Dict[str, _ChannelState],
    ) -> Path:
        """Write the CRSD file using sarkit.

        Parameters
        ----------
        xmltree : lxml.etree.ElementTree
            CRSD metadata XML tree.
        channels : list of BurstChannelInfo
            Per-channel metadata.
        burst_data : dict
            ``{channel_id: (ci2_signal, amp_sf, rcv_times)}``
        channel_states : dict
            ``{channel_id: _ChannelState}`` for per-channel arrays.

        Returns
        -------
        Path
            Output file path.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self._product_name
        out_name = f"{stem}_{self.polarization}.crsd"
        output_path = self.output_dir / out_name

        pvp_dtype = get_pvp_dtype(xmltree)
        ppp_dtype = get_ppp_dtype(xmltree)

        pvp_by_channel: Dict[str, np.ndarray] = {}
        ppp_by_channel: Dict[str, np.ndarray] = {}
        for ch in channels:
            cid = ch.identifier
            ci2, amp_sf, rcv_times = burst_data[cid]
            state = channel_states[cid]

            pvp_by_channel[cid] = build_pvp_array(
                channel=ch,
                pvp_dtype=pvp_dtype,
                rcv_times=rcv_times,
                rcv_positions=state.rcv_pos,
                rcv_velocities=state.rcv_vel,
                rcv_acx=state.rcv_acx,
                rcv_acy=state.rcv_acy,
                amp_sf=amp_sf,
            )
            ppp_by_channel[cid] = build_ppp_array(
                channel=ch,
                ppp_dtype=ppp_dtype,
                tx_times=state.tx_times,
                tx_positions=state.tx_pos,
                tx_velocities=state.tx_vel,
                tx_acx=state.tx_acx,
                tx_acy=state.tx_acy,
                tx_rad_int=np.full(ch.num_vectors, 0.0),
            )

        ref_ch_id = xmltree.findtext(".//Channel/RefChId")
        if not ref_ch_id or ref_ch_id not in pvp_by_channel:
            ref_ch_id = channels[0].identifier

        computed_refgeom = sarkit.crsd.compute_reference_geometry(
            xmltree,
            pvps=pvp_by_channel[ref_ch_id],
            ppps=ppp_by_channel[ref_ch_id],
        )
        root = xmltree.getroot()
        old_refgeom = root.find("ReferenceGeometry")
        if old_refgeom is not None:
            idx = list(root).index(old_refgeom)
            root.remove(old_refgeom)
            root.insert(idx, computed_refgeom)
        else:
            root.append(computed_refgeom)

        metadata = sarkit.crsd.Metadata(xmltree=xmltree)

        with open(output_path, "wb") as f:
            writer = sarkit.crsd.Writer(f, metadata)

            # Support array dtype: Amp(F4) + Phase(F4)
            sa_dtype = np.dtype([
                ("Amp", "<f4"), ("Phase", "<f4"),
            ])

            # Write flat frequency response array
            fx_response = np.ones((1, 30), dtype=sa_dtype)
            fx_response["Phase"] = 0.0
            writer.write_support_array(
                "flat_fx_response", fx_response,
            )

            for ch in channels:
                cid = ch.identifier
                ci2, _, _ = burst_data[cid]
                pvp = pvp_by_channel[cid]
                ppp = ppp_by_channel[cid]

                # Write arrays
                writer.write_signal(cid, ci2)
                writer.write_pvp(cid, pvp)
                writer.write_ppp(cid, ppp)

                logger.debug("Wrote channel %s", cid)

            writer.done()

        return output_path


# =====================================================================
# Convenience function
# =====================================================================


def convert_s1_l0_to_crsd(
    safe_path: Union[str, Path],
    poe_path: Optional[Union[str, Path]] = None,
    polarization: str = "VV",
    relative_orbit: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    swaths: Optional[List[str]] = None,
) -> Path:
    """Convert a Sentinel-1 Level 0 SAFE product to CRSD.

    Parameters
    ----------
    safe_path : str or Path
        Path to the ``.SAFE`` directory.
    poe_path : str or Path, optional
        Path to a POEORB file or directory.
    polarization : str
        Polarization channel to convert (default ``"VV"``).
    relative_orbit : int, optional
        Relative orbit number. Auto-detected if ``None``.
    output_dir : str or Path, optional
        Output directory for the CRSD file.
    swaths : list of str, optional
        Sub-swaths to include (e.g. ``["IW1"]``).

    Returns
    -------
    Path
        Path to the generated CRSD file.

    Examples
    --------
    >>> from grdl.IO.sar.sentinel1_l0 import convert_s1_l0_to_crsd
    >>> crsd_path = convert_s1_l0_to_crsd(
    ...     '/data/S1A_IW_RAW__0SDV_....SAFE/',
    ...     poe_path='/data/poe/',
    ...     polarization='VV',
    ...     output_dir='/data/output/',
    ... )
    """
    converter = Sentinel1L0ToCRSD(
        safe_path=safe_path,
        poe_path=poe_path,
        polarization=polarization,
        relative_orbit=relative_orbit,
        output_dir=output_dir,
        swaths=swaths,
    )
    return converter.convert()
