# -*- coding: utf-8 -*-
"""
CPHD to SICD Conversion Pipeline.

Forms a SAR image from CPHD (Compensated Phase History Data) and
writes it as SICD (Sensor Independent Complex Data) NITF.  Supports
multiple image formation algorithms with auto-detection from CPHD
metadata: spotlight collections default to PFA, stripmap/TOPSAR
collections default to RDA.

Dependencies
------------
sarpy, numpy, scipy

Author
------
Jason Fritz, PhD
fritz-jason@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-03

Modified
--------
2026-03-03
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm
from numpy.polynomial import polynomial as nppoly

from grdl.IO.sar.cphd import CPHDReader
from grdl.IO.sar.sicd_writer import SICDWriter
from grdl.IO.models.cphd import CPHDMetadata

logger = logging.getLogger(__name__)

# Speed of light (m/s)
_C = 299792458.0

# Valid algorithm names
_ALGORITHMS = {"auto", "rda", "pfa", "stripmap_pfa", "ffbp"}


# ===================================================================
# Burst detection helper
# ===================================================================

def _detect_bursts(tx_time: np.ndarray) -> List[Tuple[int, int]]:
    """Detect burst boundaries from PVP transmit times.

    Returns a list of ``(start, end)`` index tuples (end exclusive) for
    each burst.  A burst boundary is a gap > 10x the median PRI.

    Parameters
    ----------
    tx_time : np.ndarray
        1-D array of transmit times (seconds), shape ``(N,)``.

    Returns
    -------
    list of (int, int)
        Per-burst ``(start_index, end_index)`` pairs.
    """
    dt = np.diff(tx_time)
    median_pri = np.median(dt)
    gap_mask = dt > 10 * median_pri
    gap_indices = np.where(gap_mask)[0]

    starts = [0] + [int(g + 1) for g in gap_indices]
    ends = [int(g + 1) for g in gap_indices] + [len(tx_time)]
    return list(zip(starts, ends))


# ===================================================================
# Polarization helper
# ===================================================================

def _extract_polarization(meta: CPHDMetadata) -> Tuple[str, str]:
    """Extract polarization from CPHD metadata.

    Returns ``(pol_str, tx_pol)`` — e.g. ``("V:V", "V")``.
    Tries the channel identifier first (e.g. ``"IW1_VV"`` → ``"VV"``),
    then falls back to ``"V:V"``.
    """
    if meta.channels:
        ch_id = meta.channels[0].identifier
        # Channel IDs may contain polarization as last 2 chars or
        # after underscore, e.g. "IW1_VV", "CH1_HH", or just "VV"
        for token in reversed(ch_id.split("_")):
            if len(token) == 2 and all(c in "VH" for c in token):
                return f"{token[0]}:{token[1]}", token[0]
        # Try the full identifier if it's 2 chars
        if len(ch_id) == 2 and all(c in "VH" for c in ch_id):
            return f"{ch_id[0]}:{ch_id[1]}", ch_id[0]
    return "V:V", "V"


# ===================================================================
# CPHDToSICD converter
# ===================================================================

class CPHDToSICD:
    """Form a SAR image from CPHD and write as SICD NITF.

    Reads a CPHD file, optionally selects a burst, runs an image
    formation algorithm, constructs SICD metadata from the CPHD
    metadata, and writes the complex image as SICD NITF.

    Parameters
    ----------
    cphd_path : str or Path
        Input CPHD file.
    output_path : str or Path
        Output SICD NITF file.
    algorithm : str
        Image formation algorithm.  ``'auto'`` (default) selects
        PFA for spotlight and RDA for stripmap/TOPSAR.  Explicit
        choices: ``'rda'``, ``'pfa'``, ``'stripmap_pfa'``,
        ``'ffbp'``.
    block_size : int, str, or None
        Subaperture block size for RDA/StripmapPFA:
        ``'auto'``, integer, ``0`` or ``None`` for full aperture.
    burst : int, optional
        1-based burst index for TOPSAR/burst-mode data.  Burst
        boundaries are detected from PVP timing gaps.  Default
        ``None`` processes all vectors.
    max_pulses : int, optional
        Maximum number of pulses to process.
    verbose : bool
        Pass through to the image formation algorithm for detailed
        console output.

    Attributes
    ----------
    image : np.ndarray or None
        Complex SAR image after ``convert()`` completes.
    metadata : CPHDMetadata or None
        CPHD metadata (possibly burst-sliced) after ``convert()``.

    Examples
    --------
    >>> from grdl.IO.sar import CPHDToSICD
    >>> converter = CPHDToSICD('input.cphd', 'output.sicd', burst=2)
    >>> converter.convert()
    >>> print(converter.image.shape)
    """

    def __init__(
        self,
        cphd_path: Union[str, Path],
        output_path: Union[str, Path],
        algorithm: str = "auto",
        block_size: Union[int, str, None] = "auto",
        burst: Optional[int] = None,
        max_pulses: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        algorithm = algorithm.lower()
        if algorithm not in _ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm {algorithm!r}. "
                f"Choose from: {sorted(_ALGORITHMS)}"
            )
        self.cphd_path = Path(cphd_path)
        self.output_path = Path(output_path)
        self.algorithm = algorithm
        self.block_size = block_size
        self.burst = burst
        self.max_pulses = max_pulses
        self.verbose = verbose

        self.image: Optional[np.ndarray] = None
        self.metadata: Optional[CPHDMetadata] = None

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def convert(self) -> Path:
        """Run the CPHD → image → SICD conversion.

        Returns
        -------
        Path
            Path to the written SICD file.
        """
        t0 = time.perf_counter()

        # Read CPHD
        meta, signal = self._read_cphd()

        # Burst selection
        if self.burst is not None:
            signal = self._select_burst(signal, meta)

        # Create processor and form image
        processor = self._create_processor(meta)
        grid_info = processor.get_output_grid()
        logger.info(
            "Output grid: range_res=%.3f m, az_res=%.3f m",
            grid_info.get("range_resolution", 0),
            grid_info.get("azimuth_resolution", 0),
        )

        logger.info("Forming SAR image...")
        t_form = time.perf_counter()
        image = processor.form_image(signal, geometry=None)
        t_done = time.perf_counter()
        logger.info(
            "Image formed: %s in %.2fs", image.shape, t_done - t_form,
        )

        del signal  # free memory

        # Build SICD metadata and write
        sicd_type = self._build_sicd_type(meta, image, grid_info)

        logger.info("Writing SICD: %s", self.output_path.name)
        self._write_sicd(image, sicd_type)

        size_mb = self.output_path.stat().st_size / (1024**2)
        logger.info("SICD written: %.1f MB", size_mb)
        logger.info(
            "CPHD-to-SICD complete in %.2fs", time.perf_counter() - t0,
        )

        self.image = image
        self.metadata = meta
        return self.output_path

    # ---------------------------------------------------------------
    # Private: read
    # ---------------------------------------------------------------

    def _read_cphd(self) -> Tuple[CPHDMetadata, np.ndarray]:
        """Read CPHD metadata and signal data."""
        logger.info("Reading CPHD: %s", self.cphd_path.name)
        with CPHDReader(self.cphd_path) as reader:
            meta = reader.metadata

            logger.info(
                "CPHD: %d vectors x %d samples", meta.rows, meta.cols,
            )
            if meta.global_params:
                gp = meta.global_params
                logger.info(
                    "Domain: %s, PhaseSGN: %s",
                    gp.domain_type, gp.phase_sgn,
                )
                if gp.center_frequency:
                    logger.info(
                        "Center freq: %.4f GHz, BW: %.2f MHz",
                        gp.center_frequency / 1e9,
                        gp.bandwidth / 1e6,
                    )
            if meta.pvp and meta.pvp.srp_pos is not None:
                r = norm(
                    meta.pvp.srp_pos[0]
                    - 0.5 * (meta.pvp.tx_pos[0] + meta.pvp.rcv_pos[0])
                )
                logger.info("Slant range: %.1f km", r / 1000)

            signal = reader.read_full()
            logger.info("Signal: %s, dtype: %s", signal.shape, signal.dtype)

        return meta, signal

    # ---------------------------------------------------------------
    # Private: burst selection
    # ---------------------------------------------------------------

    def _select_burst(
        self,
        signal: np.ndarray,
        meta: CPHDMetadata,
    ) -> np.ndarray:
        """Select a single burst from TOPSAR/burst-mode data."""
        if meta.pvp is None or meta.pvp.tx_time is None:
            logger.warning(
                "Burst %d requested but no PVP tx_time; skipping",
                self.burst,
            )
            return signal

        bursts = _detect_bursts(meta.pvp.tx_time)
        logger.info("Burst detection: %d bursts found", len(bursts))
        for bi, (bs, be) in enumerate(bursts):
            n = be - bs
            dur = meta.pvp.tx_time[be - 1] - meta.pvp.tx_time[bs]
            pri = dur / max(n - 1, 1)
            logger.debug(
                "  Burst %d: pulses [%d:%d]  n=%d  dur=%.3fs  PRF=%.0fHz",
                bi + 1, bs, be, n, dur, 1 / pri,
            )

        if self.burst < 1 or self.burst > len(bursts):
            raise ValueError(
                f"Burst {self.burst} out of range; "
                f"available: 1\u2013{len(bursts)}"
            )
        bs, be = bursts[self.burst - 1]
        logger.info(
            "Selected burst %d: pulses [%d:%d] (%d vectors)",
            self.burst, bs, be, be - bs,
        )

        # Slice signal
        signal = signal[bs:be]
        meta.rows = signal.shape[0]

        # Slice all PVP arrays
        pvp = meta.pvp
        for field_name in [
            "tx_time", "tx_pos", "rcv_time", "rcv_pos", "srp_pos",
            "fx1", "fx2", "tx_vel", "rcv_vel", "sc0", "scss",
            "signal", "a_fdop", "a_frr1", "a_frr2", "amp_sf",
            "toa1", "toa2",
        ]:
            arr = getattr(pvp, field_name, None)
            if arr is not None:
                setattr(pvp, field_name, arr[bs:be])

        return signal

    # ---------------------------------------------------------------
    # Private: processor creation
    # ---------------------------------------------------------------

    def _resolve_algorithm(self, meta: CPHDMetadata) -> str:
        """Resolve 'auto' to a concrete algorithm name."""
        if self.algorithm != "auto":
            return self.algorithm

        mode = None
        if meta.collection_info:
            mode = meta.collection_info.radar_mode
        if mode and "SPOTLIGHT" in mode.upper():
            logger.info(
                "Auto-selected PFA for radar_mode=%r", mode,
            )
            return "pfa"
        logger.info(
            "Auto-selected RDA for radar_mode=%r", mode,
        )
        return "rda"

    def _create_processor(self, meta: CPHDMetadata) -> Any:
        """Create the image formation processor."""
        algo = self._resolve_algorithm(meta)

        from grdl.interpolation import PolyphaseInterpolator
        interpolator = PolyphaseInterpolator(
            kernel_length=8, num_phases=128, prototype="kaiser",
        )

        # Resolve block_size string to the value expected by constructors
        resolved_block = self.block_size
        if isinstance(resolved_block, str):
            if resolved_block == "auto":
                resolved_block = "auto"
            elif resolved_block == "0":
                resolved_block = None
            else:
                resolved_block = int(resolved_block)

        if algo == "rda":
            from grdl.image_processing.sar import RangeDopplerAlgorithm
            logger.info("Creating RangeDopplerAlgorithm")
            return RangeDopplerAlgorithm(
                metadata=meta,
                interpolator=interpolator,
                range_weighting="taylor",
                block_size=resolved_block,
                overlap=0.5,
                apply_amp_sf=True,
                trim_invalid=True,
                verbose=self.verbose,
            )

        if algo == "pfa":
            from grdl.image_processing.sar import (
                CollectionGeometry,
                PolarGrid,
                PolarFormatAlgorithm,
            )
            logger.info("Creating PolarFormatAlgorithm")
            phase_sgn = -1
            if meta.global_params and meta.global_params.phase_sgn:
                phase_sgn = int(meta.global_params.phase_sgn)
            geometry = CollectionGeometry(meta)
            grid = PolarGrid(geometry)
            return PolarFormatAlgorithm(
                grid=grid,
                interpolator=interpolator,
                weighting="taylor",
                phase_sgn=phase_sgn,
            )

        if algo == "stripmap_pfa":
            from grdl.image_processing.sar import StripmapPFA
            logger.info("Creating StripmapPFA")
            return StripmapPFA(
                metadata=meta,
                interpolator=interpolator,
                weighting="taylor",
                overlap_fraction=0.5,
                verbose=self.verbose,
            )

        if algo == "ffbp":
            from grdl.image_processing.sar import FastBackProjection
            logger.info("Creating FastBackProjection")
            return FastBackProjection(
                metadata=meta,
                interpolator=interpolator,
                range_weighting="taylor",
                apply_amp_sf=True,
                trim_invalid=True,
                verbose=self.verbose,
            )

        raise ValueError(f"Unknown algorithm: {algo!r}")

    # ---------------------------------------------------------------
    # Private: SICD metadata
    # ---------------------------------------------------------------

    def _build_sicd_type(
        self,
        meta: CPHDMetadata,
        image: np.ndarray,
        grid_info: Dict[str, Any],
    ) -> Any:
        """Build a sarpy SICDType from CPHD metadata and formed image."""
        from sarpy.io.complex.sicd_elements.SICD import SICDType
        from sarpy.io.complex.sicd_elements.ImageData import (
            ImageDataType, FullImageType,
        )
        from sarpy.io.complex.sicd_elements.CollectionInfo import (
            CollectionInfoType, RadarModeType,
        )
        from sarpy.io.complex.sicd_elements.ImageFormation import (
            ImageFormationType, RcvChanProcType, TxFrequencyProcType,
        )
        from sarpy.io.complex.sicd_elements.Timeline import TimelineType
        from sarpy.io.complex.sicd_elements.GeoData import (
            GeoDataType, SCPType,
        )
        from sarpy.io.complex.sicd_elements.Position import (
            PositionType, XYZPolyType,
        )
        from sarpy.io.complex.sicd_elements.Grid import (
            GridType, DirParamType,
        )
        from sarpy.io.complex.sicd_elements.RadarCollection import (
            RadarCollectionType, TxFrequencyType, ChanParametersType,
        )
        from sarpy.io.complex.sicd_elements.blocks import (
            LatLonCornerStringType, XYZType, Poly2DType,
        )
        from sarpy.geometry.geocoords import ecf_to_geodetic

        nr, nc = image.shape
        sicd = SICDType()

        # ── ImageData ──
        sicd.ImageData = ImageDataType(
            PixelType="RE32F_IM32F",
            NumRows=nr,
            NumCols=nc,
            FirstRow=0,
            FirstCol=0,
            FullImage=FullImageType(NumRows=nr, NumCols=nc),
            SCPPixel=[nr // 2, nc // 2],
            ValidData=[
                [0, 0], [0, nc - 1], [nr - 1, nc - 1], [nr - 1, 0],
            ],
        )

        # ── CollectionInfo ──
        ci = meta.collection_info
        radar_mode = None
        if ci and ci.radar_mode:
            radar_mode = RadarModeType(
                ModeType=ci.radar_mode,
                ModeID=ci.radar_mode_id,
            )
        core_name = self.cphd_path.stem
        if ci and ci.core_name:
            core_name = ci.core_name
        classification = "UNCLASSIFIED"
        if ci and ci.classification:
            classification = ci.classification
        sicd.CollectionInfo = CollectionInfoType(
            Classification=classification,
            CoreName=core_name,
            RadarMode=radar_mode,
        )

        # ── Timeline ──
        collect_start = datetime.now(timezone.utc)
        collect_duration = 1.0
        if meta.pvp and meta.pvp.tx_time is not None:
            collect_duration = float(
                meta.pvp.tx_time[-1] - meta.pvp.tx_time[0]
            )
        sicd.Timeline = TimelineType(
            CollectStart=collect_start,
            CollectDuration=max(collect_duration, 0.001),
        )

        # ── GeoData ──
        scp_ecf = None
        if (meta.scene_coordinates
                and meta.scene_coordinates.iarp_ecf is not None):
            scp_ecf = meta.scene_coordinates.iarp_ecf
        elif (meta.reference_geometry
                and meta.reference_geometry.srp_ecf is not None):
            scp_ecf = meta.reference_geometry.srp_ecf
        elif meta.pvp and meta.pvp.srp_pos is not None:
            scp_ecf = np.mean(meta.pvp.srp_pos, axis=0)

        if scp_ecf is not None:
            scp = SCPType(ECF=scp_ecf)
            corner_labels = [
                "1:FRFC", "2:FRLC", "3:LRLC", "4:LRFC",
            ]
            corners = None
            sc = meta.scene_coordinates
            if (sc and sc.corner_points is not None
                    and len(sc.corner_points) >= 4):
                corners = [
                    LatLonCornerStringType(
                        Lat=float(sc.corner_points[i, 0]),
                        Lon=float(sc.corner_points[i, 1]),
                        index=corner_labels[i],
                    )
                    for i in range(4)
                ]
            else:
                llh = ecf_to_geodetic(scp_ecf)
                lat, lon = float(llh[0]), float(llh[1])
                dlat, dlon = 0.5, 0.5
                corners = [
                    LatLonCornerStringType(
                        Lat=lat + dlat, Lon=lon - dlon, index="1:FRFC",
                    ),
                    LatLonCornerStringType(
                        Lat=lat + dlat, Lon=lon + dlon, index="2:FRLC",
                    ),
                    LatLonCornerStringType(
                        Lat=lat - dlat, Lon=lon + dlon, index="3:LRLC",
                    ),
                    LatLonCornerStringType(
                        Lat=lat - dlat, Lon=lon - dlon, index="4:LRFC",
                    ),
                ]
            sicd.GeoData = GeoDataType(
                EarthModel="WGS_84", SCP=scp, ImageCorners=corners,
            )

        # ── Position (ARP polynomial from PVP) ──
        if (meta.pvp and meta.pvp.tx_pos is not None
                and meta.pvp.tx_time is not None):
            arp_pos = meta.pvp.tx_pos
            if meta.pvp.rcv_pos is not None:
                arp_pos = 0.5 * (meta.pvp.tx_pos + meta.pvp.rcv_pos)

            t = meta.pvp.tx_time
            t_ref = 0.5 * (t[0] + t[-1])
            t_rel = t - t_ref
            poly_order = min(5, len(t_rel) - 1)
            px = nppoly.polyfit(t_rel, arp_pos[:, 0], poly_order)
            py = nppoly.polyfit(t_rel, arp_pos[:, 1], poly_order)
            pz = nppoly.polyfit(t_rel, arp_pos[:, 2], poly_order)
            sicd.Position = PositionType(
                ARPPoly=XYZPolyType(X=px, Y=py, Z=pz),
            )

        # ── Grid ──
        if (meta.pvp and meta.pvp.tx_pos is not None
                and meta.global_params):
            gp = meta.global_params
            pvp = meta.pvp
            c = _C

            mid = len(pvp.tx_time) // 2
            arp_mid = pvp.tx_pos[mid]
            if pvp.rcv_pos is not None:
                arp_mid = 0.5 * (pvp.tx_pos[mid] + pvp.rcv_pos[mid])
            srp_mid = (
                pvp.srp_pos[mid]
                if pvp.srp_pos is not None
                else scp_ecf
            )
            row_uvec = srp_mid - arp_mid
            row_uvec = row_uvec / norm(row_uvec)

            if pvp.tx_vel is not None:
                vel_mid = pvp.tx_vel[mid]
            else:
                vel_mid = (
                    (pvp.tx_pos[mid + 1] - pvp.tx_pos[mid - 1])
                    / (pvp.tx_time[mid + 1] - pvp.tx_time[mid - 1])
                )
            col_uvec = vel_mid / norm(vel_mid)

            bw = float(gp.bandwidth)
            fc = float(gp.center_frequency)
            row_ss = c / (2.0 * bw)
            row_imp_bw = 1.0 / row_ss
            row_imp_wid = grid_info.get("range_resolution", row_ss)
            row_kctr = 2.0 * fc / c

            dt_median = float(np.median(np.diff(pvp.tx_time)))
            v_platform = float(norm(vel_mid))
            col_ss = v_platform * dt_median
            col_imp_wid = grid_info.get("azimuth_resolution", col_ss)
            col_imp_bw = 1.0 / col_ss if col_ss > 0 else 1.0
            col_kctr = 0.0

            phase_sgn = int(gp.phase_sgn) if gp.phase_sgn else -1

            row_dir = DirParamType(
                UVectECF=XYZType(
                    X=float(row_uvec[0]),
                    Y=float(row_uvec[1]),
                    Z=float(row_uvec[2]),
                ),
                SS=row_ss,
                ImpRespWid=row_imp_wid,
                Sgn=phase_sgn,
                ImpRespBW=row_imp_bw,
                KCtr=row_kctr,
                DeltaK1=-0.5 / row_ss,
                DeltaK2=0.5 / row_ss,
            )
            col_dir = DirParamType(
                UVectECF=XYZType(
                    X=float(col_uvec[0]),
                    Y=float(col_uvec[1]),
                    Z=float(col_uvec[2]),
                ),
                SS=col_ss,
                ImpRespWid=col_imp_wid,
                Sgn=phase_sgn,
                ImpRespBW=col_imp_bw,
                KCtr=col_kctr,
                DeltaK1=-0.5 / col_ss,
                DeltaK2=0.5 / col_ss,
            )
            sicd.Grid = GridType(
                ImagePlane="SLANT",
                Type="RGAZIM",
                TimeCOAPoly=Poly2DType(Coefs=np.array([[0.0]])),
                Row=row_dir,
                Col=col_dir,
            )

            # ── RadarCollection + ImageFormation ──
            f_min = fc - bw / 2.0
            f_max = fc + bw / 2.0
            pol_str, tx_pol = _extract_polarization(meta)

            sicd.RadarCollection = RadarCollectionType(
                TxFrequency=TxFrequencyType(Min=f_min, Max=f_max),
                TxPolarization=tx_pol,
                RcvChannels=[
                    ChanParametersType(
                        TxRcvPolarization=pol_str, index=1,
                    ),
                ],
            )

            algo = self._resolve_algorithm(meta)
            if algo == "pfa":
                sicd_algo = "PFA"
            else:
                sicd_algo = "RGAZCOMP"

            sicd.ImageFormation = ImageFormationType(
                RcvChanProc=RcvChanProcType(
                    NumChanProc=1, ChanIndices=[1],
                ),
                TxRcvPolarizationProc=pol_str,
                TStartProc=0.0,
                TEndProc=max(collect_duration, 0.001),
                TxFrequencyProc=TxFrequencyProcType(
                    MinProc=f_min, MaxProc=f_max,
                ),
                ImageFormAlgo=sicd_algo,
                STBeamComp="NO",
                ImageBeamComp="NO",
                AzAutofocus="NO",
                RgAutofocus="NO",
            )

        # Auto-derive SCPCOA, RgAzComp, and other dependent fields
        sicd.derive()

        return sicd

    # ---------------------------------------------------------------
    # Private: write
    # ---------------------------------------------------------------

    def _write_sicd(self, image: np.ndarray, sicd_type: Any) -> None:
        """Write the formed image as SICD NITF."""
        writer = SICDWriter(self.output_path)
        writer.set_sarpy_metadata(sicd_type)
        writer.write(image.astype(np.complex64))


# ===================================================================
# CLI
# ===================================================================

def _parse_args():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Form a SAR image from CPHD and write as SICD NITF.",
    )
    parser.add_argument(
        "cphd_path",
        type=Path,
        help="Path to the input CPHD file.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        default=None,
        help="Output SICD NITF path. Defaults to <cphd_stem>.sicd "
             "in the same directory.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="auto",
        choices=sorted(_ALGORITHMS),
        help="Image formation algorithm (default: auto). "
             "'auto' selects PFA for spotlight and RDA for stripmap.",
    )
    parser.add_argument(
        "--block-size",
        type=str,
        default="auto",
        help="RDA subaperture block size: 'auto', integer, or '0' "
             "for full aperture.",
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=None,
        help="1-based burst index for TOPSAR/burst-mode data.",
    )
    parser.add_argument(
        "--max-pulses",
        type=int,
        default=None,
        help="Limit processing to this many pulses.",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        default=False,
        help="Suppress verbose output from the image formation processor.",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level (default: INFO).",
    )
    parser.add_argument(
        "--logfile",
        type=Path,
        default=None,
        help="Write log messages to this file in addition to stderr.",
    )
    return parser.parse_args()


def _setup_logging(loglevel: str, logfile: Optional[Path] = None) -> None:
    """Configure the root logger with console and optional file handlers."""
    level = getattr(logging, loglevel.upper(), logging.WARNING)
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(logfile)))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


def main() -> None:
    """CLI entry point for CPHD → SICD conversion."""
    args = _parse_args()
    _setup_logging(args.loglevel, args.logfile)

    cphd_path = args.cphd_path
    output_path = args.output_path
    if output_path is None:
        output_path = cphd_path.with_suffix(".sicd")

    converter = CPHDToSICD(
        cphd_path=cphd_path,
        output_path=output_path,
        algorithm=args.algorithm,
        block_size=args.block_size,
        burst=args.burst,
        max_pulses=args.max_pulses,
        verbose=not args.no_verbose,
    )
    result = converter.convert()
    logger.info("Done: %s", result)


if __name__ == "__main__":
    main()
