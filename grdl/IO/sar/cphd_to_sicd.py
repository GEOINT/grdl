# -*- coding: utf-8 -*-
"""
CPHD to SICD Conversion Pipeline.

Forms a SAR image from CPHD (Compensated Phase History Data) and
writes it as SICD (Sensor Independent Complex Data) NITF.  Supports
multiple image formation algorithms with auto-detection from CPHD
metadata: spotlight collections default to PFA, stripmap/TOPSAR
collections default to RDA.

SICD metadata is derived via
:func:`grdl.IO.sar.cphd_metadata.build_sicd_metadata`, which produces
a fully populated :class:`grdl.IO.models.SICDMetadata` from the CPHD
metadata and the IFP's ``CollectionGeometry`` output -- ensuring rich
SCPCOA, accurate ARP polynomial fits, and waveform provenance.

Dependencies
------------
sarpy, numpy, scipy

Author
------
Jason Fritz, PhD
fritz-jason@zai.com

Ava Courtney
courtney-ava@zai.com

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
2026-05-06
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm

from grdl.IO.sar.cphd import CPHDReader
from grdl.IO.sar.cphd_metadata import build_sicd_metadata
from grdl.IO.sar.sicd_writer import SICDWriter
from grdl.IO.models.cphd import CPHDMetadata

logger = logging.getLogger(__name__)

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
# CPHDToSICD converter
# ===================================================================

class CPHDToSICD:
    """Form a SAR image from CPHD and write as SICD NITF.

    Reads a CPHD file, optionally selects a burst, runs an image
    formation algorithm, constructs SICD metadata from the CPHD
    metadata and the IFP\'s CollectionGeometry via
    :func:`~grdl.IO.sar.cphd_metadata.build_sicd_metadata`, and writes
    the complex image as SICD NITF.

    Parameters
    ----------
    cphd_path : str or Path
        Input CPHD file.
    output_path : str or Path
        Output SICD NITF file.
    algorithm : str
        Image formation algorithm.  ``\'auto\'`` (default) selects
        PFA for spotlight and RDA for stripmap/TOPSAR.  Explicit
        choices: ``\'rda\'``, ``\'pfa\'``, ``\'stripmap_pfa\'``,
        ``\'ffbp\'``.
    block_size : int, str, or None
        Subaperture block size for RDA/StripmapPFA:
        ``\'auto\'``, integer, ``0`` or ``None`` for full aperture.
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
    >>> converter = CPHDToSICD(\'input.cphd\', \'output.sicd\', burst=2)
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
        """Run the CPHD -> image -> SICD conversion.

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
        processor, geometry = self._create_processor(meta)
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

        # Build SICD metadata using the dedicated metadata builder
        algo = self._resolve_algorithm(meta)
        sicd_algo = "PFA" if algo == "pfa" else "RGAZCOMP"
        sicd_meta = build_sicd_metadata(
            cphd_meta=meta,
            geometry=geometry,
            image_shape=image.shape,
            image_form_algo=sicd_algo,
            grid_params=grid_info,
        )

        logger.info("Writing SICD: %s", self.output_path.name)
        writer = SICDWriter(self.output_path, metadata=sicd_meta)
        writer.write(image.astype(np.complex64))

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
        """Resolve \'auto\' to a concrete algorithm name."""
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

    def _create_processor(self, meta: CPHDMetadata) -> Tuple[Any, Any]:
        """Create the image formation processor and CollectionGeometry.

        Returns
        -------
        tuple of (processor, geometry)
            ``processor`` is the IFP instance (RDA, PFA, etc.).
            ``geometry`` is the ``CollectionGeometry`` used to build it,
            which is later passed to ``build_sicd_metadata``.
        """
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
            from grdl.image_processing.sar.image_formation import (
                CollectionGeometry,
            )
            logger.info("Creating RangeDopplerAlgorithm")
            geometry = CollectionGeometry(meta)
            processor = RangeDopplerAlgorithm(
                metadata=meta,
                interpolator=interpolator,
                range_weighting="taylor",
                block_size=resolved_block,
                overlap=0.5,
                apply_amp_sf=True,
                trim_invalid=True,
                verbose=self.verbose,
            )
            return processor, geometry

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
            processor = PolarFormatAlgorithm(
                grid=grid,
                interpolator=interpolator,
                weighting="taylor",
                phase_sgn=phase_sgn,
            )
            return processor, geometry

        if algo == "stripmap_pfa":
            from grdl.image_processing.sar import StripmapPFA
            from grdl.image_processing.sar.image_formation import (
                CollectionGeometry,
            )
            logger.info("Creating StripmapPFA")
            geometry = CollectionGeometry(meta)
            processor = StripmapPFA(
                metadata=meta,
                interpolator=interpolator,
                weighting="taylor",
                overlap_fraction=0.5,
                verbose=self.verbose,
            )
            return processor, geometry

        if algo == "ffbp":
            from grdl.image_processing.sar import FastBackProjection
            from grdl.image_processing.sar.image_formation import (
                CollectionGeometry,
            )
            logger.info("Creating FastBackProjection")
            geometry = CollectionGeometry(meta)
            processor = FastBackProjection(
                metadata=meta,
                interpolator=interpolator,
                range_weighting="taylor",
                apply_amp_sf=True,
                trim_invalid=True,
                verbose=self.verbose,
            )
            return processor, geometry

        raise ValueError(f"Unknown algorithm: {algo!r}")


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
             "\'auto\' selects PFA for spotlight and RDA for stripmap.",
    )
    parser.add_argument(
        "--block-size",
        type=str,
        default="auto",
        help="RDA subaperture block size: \'auto\', integer, or \'0\' "
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
    import logging as _logging
    level = getattr(_logging, loglevel.upper(), _logging.WARNING)
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    handlers: list[_logging.Handler] = [_logging.StreamHandler()]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(_logging.FileHandler(str(logfile)))

    _logging.basicConfig(
        level=level, format=fmt, datefmt=datefmt, handlers=handlers,
    )


def main() -> None:
    """CLI entry point for CPHD -> SICD conversion."""
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
