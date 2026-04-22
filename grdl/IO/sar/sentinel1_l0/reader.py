# -*- coding: utf-8 -*-
"""
Sentinel-1 Level 0 Reader - Public API.

High-level reader for Sentinel-1 L0 SAFE products.  Inherits
:class:`grdl.IO.base.ImageReader`; provides:

- SAFE product validation and file discovery
- Annotation XML parsing
- Orbit and attitude loading (annotation + optional POE file)
- Burst-level I/Q data access (via the optional ``sentinel1decoder``
  package)
- Typed :class:`Sentinel1L0Metadata` exposed through
  ``self.metadata``
- ``read_chip()`` mapping onto the currently selected burst's
  decoded pulses × range-samples

Examples
--------
>>> from grdl.IO.sar import Sentinel1L0Reader
>>>
>>> with Sentinel1L0Reader('S1A_IW_RAW__0SDV_...SAFE') as r:
...     print(r.metadata.summary())
...     for info, iq in r.iter_bursts(polarization='VV'):
...         print(info.swath, info.num_lines, iq.shape)

Dependencies
------------
rasterio is **not** required.  ``sentinel1decoder`` is optional —
metadata parsing works without it, but ``read_burst()`` /
``read_chip()`` require ``pip install grdl[s1_l0]``.

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

# Standard library
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

# Third-party
import numpy as np

# GRDL internal
from grdl.exceptions import DependencyError
from grdl.IO.base import ImageReader
from grdl.IO.models.sentinel1_l0 import (
    S1L0AttitudeRecord,
    S1L0DownlinkInfo,
    S1L0OrbitStateVector,
    S1L0RadarParameters,
    S1L0SwathParameters,
    Sentinel1L0Metadata,
    Sentinel1Mission,
    Sentinel1Mode,
)
from grdl.IO.sar.sentinel1_l0.annotation_parser import (
    AnnotationData,
    AnnotationParser,
    merge_annotation_data,
)
from grdl.IO.sar.sentinel1_l0.burst_reader import (
    BurstInfo,
    BurstReader,
    SwathInfo,
)
from grdl.IO.sar.sentinel1_l0.constants import (
    IW_MODE_PARAMS,
    IW_RAW_TO_LOGICAL,
    MODE_PARAMS,
    SENTINEL1_CENTER_FREQUENCY_HZ,
    raw_swath_to_name,
)
from grdl.IO.sar.sentinel1_l0.decoder import (
    check_decoder_available,
)
from grdl.IO.sar.sentinel1_l0.geometry import GeometryCalculator
from grdl.IO.sar.sentinel1_l0.orbit import OrbitLoader
from grdl.IO.sar.sentinel1_l0.safe_product import (
    SAFEProduct,
    ProductIdentifier,
)
from grdl.IO.sar.sentinel1_l0.timing import TimingCalculator

logger = logging.getLogger(__name__)


# =============================================================================
# Reader configuration
# =============================================================================


@dataclass
class ReaderConfig:
    """Configuration options for :class:`Sentinel1L0Reader`.

    Parameters
    ----------
    validate_safe : bool, optional
        Validate SAFE directory structure during load.
    parse_annotations : bool, optional
        Parse annotation XML files.
    load_poe : bool, optional
        Attempt to load a POE orbit file.
    poe_directory : Path, optional
        Directory to search for POE files.
    enable_gpu : bool, optional
        Enable GPU acceleration for decoding (reserved).
    burst_gap_threshold_us : int, optional
        Packet-time gap threshold for burst boundary detection.
    burst_line_filter_ratio : float, optional
        Minimum fraction of median burst-line count to retain a
        burst (filters partial edge bursts).
    """

    validate_safe: bool = True
    parse_annotations: bool = True
    load_poe: bool = True
    poe_directory: Optional[Path] = None
    enable_gpu: bool = False
    burst_gap_threshold_us: int = 1_000_000
    burst_line_filter_ratio: float = 0.9


# =============================================================================
# Sentinel-1 L0 Reader
# =============================================================================


class Sentinel1L0Reader(ImageReader):
    """Read a Sentinel-1 Level 0 SAFE product.

    Inherits :class:`grdl.IO.base.ImageReader`.  Metadata is
    populated on construction; ``read_chip()`` operates on the
    currently selected burst.  Use the L0-native API
    (:meth:`read_burst`, :meth:`iter_bursts`, :meth:`read_swath`)
    for burst-aware access.

    Parameters
    ----------
    filepath : str or Path
        Path to SAFE product directory.
    config : ReaderConfig, optional
        Configuration overrides.
    current_burst : int, optional
        Initial burst index for :meth:`read_chip` (default 0).
    current_polarization : str, optional
        Initial polarization for :meth:`read_chip` (default
        first available).

    Raises
    ------
    FileNotFoundError
        If the SAFE directory does not exist.
    grdl.IO.sar.sentinel1_l0.safe_product.InvalidSAFEProductError
        If structural validation fails.

    Attributes
    ----------
    filepath : Path
        Path to the SAFE product directory.
    metadata : Sentinel1L0Metadata
        Typed metadata populated from manifest, annotation XML,
        and packet-derived parameters.

    Examples
    --------
    >>> with Sentinel1L0Reader('product.SAFE') as r:
    ...     print(r.metadata.mission)
    ...     bursts = r.bursts
    ...     iq = r.read_burst(bursts[0])
    """

    FORMAT_NAME = "Sentinel-1 L0"

    def __init__(
        self,
        filepath: Union[str, Path],
        config: Optional[ReaderConfig] = None,
        current_burst: int = 0,
        current_polarization: Optional[str] = None,
    ) -> None:
        # State must exist before super().__init__ runs
        # ``_load_metadata``.
        self._config = config or ReaderConfig()
        self._product: Optional[SAFEProduct] = None
        self._annotation_data: Dict[str, AnnotationData] = {}
        self._orbit_loader: Optional[OrbitLoader] = None
        self._burst_readers: Dict[str, BurstReader] = {}
        self._geometry_calculator: Optional[
            GeometryCalculator
        ] = None
        self._timing_calculator: Optional[
            TimingCalculator
        ] = None
        self._current_burst_index: int = current_burst
        self._current_polarization: Optional[str] = (
            current_polarization.upper()
            if current_polarization else None
        )

        super().__init__(filepath)

    # -----------------------------------------------------------------
    # ImageReader contract
    # -----------------------------------------------------------------

    def _load_metadata(self) -> None:
        """Parse SAFE / annotations / POE and populate ``self.metadata``."""
        self._product = SAFEProduct(
            self.filepath,
            validate=self._config.validate_safe,
        )

        if self._config.parse_annotations:
            self._parse_annotations()

        if self._config.load_poe:
            self._load_poe()

        self._init_burst_readers()

        self.metadata = self._build_metadata()

        if self._current_polarization is None:
            # Default to first measurement-file polarization.
            pols = list(self._burst_readers.keys())
            if pols:
                self._current_polarization = pols[0]
            elif self.metadata.polarizations:
                self._current_polarization = (
                    self.metadata.polarizations[0]
                )

        logger.info(
            f"Opened Sentinel-1 L0 product: "
            f"{self.filepath.name}"
        )

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read decoded I/Q from the currently selected burst.

        ``rows`` correspond to azimuth pulses within the burst;
        ``cols`` correspond to range samples.  Burst selection is
        controlled by :meth:`set_current_burst` /
        :attr:`current_burst_index` and
        :attr:`current_polarization` (defaults to burst 0 of the
        first available polarization).

        Parameters
        ----------
        row_start, row_end : int
            Pulse index range (half-open).
        col_start, col_end : int
            Range-sample index range (half-open).
        bands : list of int, optional
            Ignored — Sentinel-1 L0 is single-band complex I/Q.

        Returns
        -------
        numpy.ndarray
            Complex I/Q samples, shape ``(row_end - row_start,
            col_end - col_start)``.

        Raises
        ------
        grdl.exceptions.DependencyError
            If ``sentinel1decoder`` is not installed.
        RuntimeError
            If no burst reader is available for the current
            polarization.
        IndexError
            If slice extents exceed the burst dimensions.
        """
        _ = bands  # L0 is single-band; argument accepted per ABC.

        if not check_decoder_available():
            raise DependencyError(
                "Sentinel-1 L0 chip reads require "
                "sentinel1decoder. Install with: "
                "pip install grdl[s1_l0]"
            )

        reader = self._get_burst_reader(
            self._current_polarization
        )
        if reader is None:
            raise RuntimeError(
                "No burst reader available for polarization "
                f"{self._current_polarization!r}"
            )

        bursts = reader.get_burst_info()
        if (
            self._current_burst_index < 0
            or self._current_burst_index >= len(bursts)
        ):
            raise IndexError(
                f"Current burst index "
                f"{self._current_burst_index} out of range "
                f"[0, {len(bursts)})"
            )

        iq = reader.read_burst(
            bursts[self._current_burst_index]
        )
        # iq is (num_lines, num_samples).
        if row_end > iq.shape[0] or col_end > iq.shape[1]:
            raise IndexError(
                f"Chip ({row_start}:{row_end}, "
                f"{col_start}:{col_end}) exceeds burst "
                f"shape {iq.shape}"
            )
        return iq[row_start:row_end, col_start:col_end]

    def get_shape(self) -> Tuple[int, ...]:
        """Return image shape ``(rows, cols)``.

        ``rows`` is the sum of azimuth lines across all bursts
        of the currently selected polarization; ``cols`` is the
        maximum range-sample count.
        """
        if self.metadata is None:
            return (0, 0)
        return (self.metadata.rows, self.metadata.cols)

    def get_dtype(self) -> np.dtype:
        """Sentinel-1 L0 returns complex64 I/Q samples."""
        return np.dtype("complex64")

    # -----------------------------------------------------------------
    # L0-native public API
    # -----------------------------------------------------------------

    @property
    def product(self) -> Optional[SAFEProduct]:
        """Parsed :class:`SAFEProduct` for the opened directory."""
        return self._product

    @property
    def product_info(self) -> Optional[ProductIdentifier]:
        """Product identifier parsed from the directory name."""
        if self._product:
            return self._product.product_info
        return None

    @property
    def config(self) -> ReaderConfig:
        """Reader configuration in effect."""
        return self._config

    @property
    def bursts(self) -> List[BurstInfo]:
        """All bursts from the currently selected polarization."""
        return self.get_burst_info(self._current_polarization)

    @property
    def current_burst_index(self) -> int:
        """Burst index used by :meth:`read_chip`."""
        return self._current_burst_index

    @property
    def current_polarization(self) -> Optional[str]:
        """Polarization used by :meth:`read_chip`."""
        return self._current_polarization

    @property
    def orbit_state_vectors(self) -> List[S1L0OrbitStateVector]:
        """Orbit state vectors from annotation (or POE if loaded)."""
        if self.metadata is None:
            return []
        return self.metadata.orbit_state_vectors

    @property
    def attitude_records(self) -> List[S1L0AttitudeRecord]:
        """Attitude records from annotation."""
        if self.metadata is None:
            return []
        return self.metadata.attitude_records

    @property
    def radar_parameters(
        self,
    ) -> Optional[S1L0RadarParameters]:
        """Radar signal parameters."""
        if self.metadata is None:
            return None
        return self.metadata.radar_parameters

    @property
    def swath_parameters(
        self,
    ) -> List[S1L0SwathParameters]:
        """Per-swath parameters."""
        if self.metadata is None:
            return []
        return self.metadata.swath_parameters

    def set_current_burst(
        self,
        burst_index: int,
        polarization: Optional[str] = None,
    ) -> None:
        """Set the burst targeted by :meth:`read_chip`.

        Parameters
        ----------
        burst_index : int
            Burst index within the selected polarization.
        polarization : str, optional
            Polarization (``"VV"``, ``"VH"``, ``"HH"``,
            ``"HV"``).  If ``None``, the current polarization is
            kept.
        """
        self._current_burst_index = burst_index
        if polarization is not None:
            self._current_polarization = polarization.upper()

    def get_burst_info(
        self,
        polarization: Optional[str] = None,
    ) -> List[BurstInfo]:
        """Get bursts for a polarization (or all if ``None``).

        Parameters
        ----------
        polarization : str, optional
            Polarization filter.  If ``None``, bursts from every
            available polarization are concatenated.

        Returns
        -------
        list of BurstInfo
        """
        if polarization is not None:
            reader = self._get_burst_reader(polarization)
            if reader is None:
                return []
            return reader.get_burst_info()

        all_bursts: List[BurstInfo] = []
        for reader in self._burst_readers.values():
            all_bursts.extend(reader.get_burst_info())
        return all_bursts

    def get_swath_info(
        self,
        polarization: Optional[str] = None,
    ) -> Dict[int, SwathInfo]:
        """Get swath summaries for a polarization.

        Parameters
        ----------
        polarization : str, optional
            Polarization filter.  If ``None``, uses the current
            polarization (default is the first loaded).

        Returns
        -------
        dict
            Swath number → :class:`SwathInfo`.
        """
        reader = self._get_burst_reader(
            polarization
            if polarization is not None
            else self._current_polarization
        )
        if reader is None:
            return {}
        return reader.get_swath_info()

    def read_burst(
        self,
        burst: Union[BurstInfo, int],
        polarization: Optional[str] = None,
    ) -> np.ndarray:
        """Read and decode a single burst.

        Parameters
        ----------
        burst : BurstInfo or int
            Burst descriptor or burst index within the selected
            polarization.
        polarization : str, optional
            Polarization to read.  Defaults to
            :attr:`current_polarization`.

        Returns
        -------
        numpy.ndarray
            Complex I/Q samples, shape ``(num_lines,
            num_samples)``.
        """
        reader = self._get_burst_reader(
            polarization
            if polarization is not None
            else self._current_polarization
        )
        if reader is None:
            raise RuntimeError(
                "No burst reader available for polarization "
                f"{polarization or self._current_polarization!r}"
            )
        return reader.read_burst(burst)

    def read_burst_parallel(
        self,
        swath: int,
        burst: int,
        polarization: Optional[str] = None,
        num_workers: Optional[int] = None,
        chunk_size: int = 1024,
    ) -> np.ndarray:
        """Read a burst by ``(swath, burst_index)`` in parallel.

        Parameters
        ----------
        swath : int
            Raw swath number (e.g., ``10`` for IW1).
        burst : int
            Burst index within the swath.
        polarization : str, optional
            Polarization to read.  Defaults to
            :attr:`current_polarization`.
        num_workers : int, optional
            Number of worker processes.
        chunk_size : int, optional
            Packets per chunk.

        Returns
        -------
        numpy.ndarray
            Complex I/Q samples.

        Raises
        ------
        ValueError
            If no burst matches ``(swath, burst)``.
        """
        reader = self._get_burst_reader(
            polarization
            if polarization is not None
            else self._current_polarization
        )
        if reader is None:
            raise RuntimeError(
                "No burst reader available"
            )

        all_bursts = reader.get_burst_info()
        swath_bursts = [
            b for b in all_bursts if b.swath == swath
        ]
        if not swath_bursts:
            raise ValueError(
                f"No bursts found for swath {swath}"
            )

        target = None
        for b in swath_bursts:
            if b.burst_index == burst:
                target = b
                break
        if target is None:
            available = sorted(
                b.burst_index for b in swath_bursts
            )
            raise ValueError(
                f"Burst index {burst} not found for swath "
                f"{swath}. Available: {available}"
            )

        return reader.read_burst_parallel(
            burst=target,
            num_workers=num_workers,
            chunk_size=chunk_size,
        )

    def read_swath(
        self,
        swath: int,
        polarization: Optional[str] = None,
    ) -> np.ndarray:
        """Read and decode an entire swath.

        Parameters
        ----------
        swath : int
            Raw swath number.
        polarization : str, optional
            Polarization to read.

        Returns
        -------
        numpy.ndarray
            Complex I/Q samples.
        """
        reader = self._get_burst_reader(
            polarization
            if polarization is not None
            else self._current_polarization
        )
        if reader is None:
            raise RuntimeError(
                "No burst reader available"
            )
        return reader.read_swath(swath)

    def iter_bursts(
        self,
        polarization: Optional[str] = None,
        swath: Optional[int] = None,
    ) -> Iterator[Tuple[BurstInfo, np.ndarray]]:
        """Iterate over bursts, yielding ``(info, iq_data)``.

        Parameters
        ----------
        polarization : str, optional
            Polarization filter.  Defaults to current.
        swath : int, optional
            Additional swath filter.

        Yields
        ------
        (BurstInfo, numpy.ndarray)
        """
        reader = self._get_burst_reader(
            polarization
            if polarization is not None
            else self._current_polarization
        )
        if reader is None:
            return
        yield from reader.iter_bursts(swath=swath)

    def get_burst_swst_array(
        self,
        swath: int,
        burst_index: int,
        polarization: Optional[str] = None,
    ) -> np.ndarray:
        """Per-packet SWST values for a burst.

        Parameters
        ----------
        swath : int
            Raw swath number.
        burst_index : int
            Burst index within the swath.
        polarization : str, optional
            Polarization to read.

        Returns
        -------
        numpy.ndarray
            SWST values in seconds (one per packet).
        """
        reader = self._get_burst_reader(
            polarization
            if polarization is not None
            else self._current_polarization
        )
        if reader is None:
            raise RuntimeError(
                "No burst reader available"
            )

        bursts = reader.get_burst_info()
        swath_bursts = [
            b for b in bursts if b.swath == swath
        ]
        if burst_index >= len(swath_bursts):
            raise IndexError(
                f"Burst {burst_index} >= "
                f"{len(swath_bursts)}"
            )
        return reader.get_burst_swst_array(
            swath_bursts[burst_index]
        )

    def get_timing_calculator(
        self,
    ) -> Optional[TimingCalculator]:
        """Build or return a cached :class:`TimingCalculator`."""
        if (
            self._timing_calculator is None
            and self.metadata is not None
        ):
            t_ref = self.metadata.start_time
            rp = self.metadata.radar_parameters
            if t_ref and rp:
                self._timing_calculator = TimingCalculator(
                    t_ref=t_ref,
                    prf_hz=(
                        rp.pulse_repetition_frequency_hz
                    ),
                    pulse_duration_s=rp.tx_pulse_length_s,
                    range_sampling_rate_hz=(
                        rp.range_sampling_rate_hz
                    ),
                )
        return self._timing_calculator

    def get_geometry_calculator(
        self,
    ) -> Optional[GeometryCalculator]:
        """Build or return a cached :class:`GeometryCalculator`."""
        if (
            self._geometry_calculator is None
            and self.metadata is not None
            and len(self.metadata.orbit_state_vectors) >= 4
            and self.metadata.start_time is not None
        ):
            self._geometry_calculator = GeometryCalculator(
                self.metadata.orbit_state_vectors,
                self.metadata.start_time,
                self.metadata.attitude_records or None,
            )
        return self._geometry_calculator

    def summary(self) -> str:
        """Return a human-readable summary of the product."""
        lines = [
            f"Sentinel1L0Reader: {self.filepath.name}",
        ]
        if self._product and self._product.product_info:
            info = self._product.product_info
            lines.extend([
                f"  Mission: {info.platform_name}",
                f"  Mode: {info.mode}",
                f"  Polarizations: {info.polarizations}",
            ])
        if self.metadata is not None:
            lines.extend([
                f"  Start: {self.metadata.start_time}",
                f"  Stop: {self.metadata.stop_time}",
                f"  Orbit vectors: "
                f"{len(self.metadata.orbit_state_vectors)}",
                f"  Attitude records: "
                f"{len(self.metadata.attitude_records)}",
                f"  Swaths: "
                f"{len(self.metadata.swath_parameters)}",
            ])
        lines.append(
            f"  Burst readers: {len(self._burst_readers)}"
        )
        return "\n".join(lines)

    def close(self) -> None:
        """Release underlying resources."""
        for reader in self._burst_readers.values():
            reader.close()
        self._burst_readers.clear()
        self._product = None
        self._orbit_loader = None
        self._geometry_calculator = None
        self._timing_calculator = None

    def __enter__(self) -> "Sentinel1L0Reader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _parse_annotations(self) -> None:
        """Parse all annotation XML files in the SAFE product."""
        parser = AnnotationParser()
        for annot_file in self._product.annotation_files:
            try:
                data = parser.parse(annot_file)
                self._annotation_data[data.polarization] = data
            except Exception as e:
                warnings.warn(
                    f"Failed to parse {annot_file.name}: {e}"
                )

    def _load_poe(self) -> None:
        """Load a POE orbit file.

        Search order:

        1. Explicit ``ReaderConfig.poe_directory`` (if set).
        2. A ``.EOF`` file directly inside the SAFE directory
           (ESA sometimes ships one alongside ``manifest.safe``).
        3. A sibling ``POEORB/`` directory next to the SAFE
           directory.

        The first POE file whose validity period covers the
        product start time wins.  Annotation-provided orbit
        vectors (L1 products) are used as a seed so that
        :attr:`OrbitLoader.has_annotation_orbit` reflects the
        L1 path.
        """
        self._orbit_loader = OrbitLoader()

        # Seed with annotation vectors (only present for L1).
        for data in self._annotation_data.values():
            if data.orbit_state_vectors:
                self._orbit_loader.load_annotation_vectors(
                    data.orbit_state_vectors
                )
                break

        # Derive the search time.
        start_time = None
        for data in self._annotation_data.values():
            if data.orbit_state_vectors:
                start_time = (
                    data.orbit_state_vectors[0].time
                )
                break
        if start_time is None and self._product.product_info:
            start_time = (
                self._product.product_info.start_time
            )
        if start_time is None:
            return

        mission_letter = (
            self._product.product_info.mission
            if self._product.product_info else "A"
        )
        mission = f"S1{mission_letter}"

        # Candidate search locations in priority order.
        search_dirs = []
        if self._config.poe_directory:
            search_dirs.append(
                Path(self._config.poe_directory)
            )
        search_dirs.append(self.filepath)
        search_dirs.append(self.filepath.parent / "POEORB")

        for directory in search_dirs:
            if not directory.exists():
                continue
            try:
                found = self._orbit_loader.find_and_load_poe(
                    directory, start_time, mission,
                )
                if found:
                    logger.info(
                        "Loaded POE orbit from %s",
                        directory,
                    )
                    return
            except Exception as e:
                logger.warning(
                    "POE load from %s failed: %s",
                    directory, e,
                )
                continue

        if not self._orbit_loader.has_poe_orbit:
            logger.debug(
                "No POE file found for %s in %s",
                mission, [str(d) for d in search_dirs],
            )

    def _init_burst_readers(self) -> None:
        """Initialize a :class:`BurstReader` per measurement file."""
        if not check_decoder_available():
            logger.debug(
                "sentinel1decoder not available — burst "
                "readers not initialized"
            )
            return

        for meas in self._product.measurement_files:
            try:
                reader = BurstReader(
                    meas.measurement_file,
                    index_file=meas.index_file,
                    burst_gap_threshold_us=(
                        self._config.burst_gap_threshold_us
                    ),
                    burst_line_filter_ratio=(
                        self._config.burst_line_filter_ratio
                    ),
                )
                key = (
                    meas.polarization.upper()
                    if meas.polarization
                    else meas.stem
                )
                self._burst_readers[key] = reader
            except Exception as e:
                warnings.warn(
                    f"Failed to init burst reader for "
                    f"{meas.stem}: {e}"
                )

    def _get_burst_reader(
        self, polarization: Optional[str],
    ) -> Optional[BurstReader]:
        """Return the burst reader for a given polarization."""
        if not self._burst_readers:
            return None
        if polarization is not None:
            return self._burst_readers.get(
                polarization.upper()
            )
        return next(
            iter(self._burst_readers.values()), None
        )

    def _build_metadata(self) -> Sentinel1L0Metadata:
        """Assemble :class:`Sentinel1L0Metadata` from product,
        annotation, POE, and packet data."""
        meta = Sentinel1L0Metadata(
            format=self.FORMAT_NAME,
            rows=0,
            cols=0,
            dtype="complex64",
        )

        info = (
            self._product.product_info
            if self._product else None
        )
        if info:
            meta.product_id = info.full_name
            meta.polarizations = list(info.polarizations)
            if info.start_time:
                meta.start_time = info.start_time
            if info.stop_time:
                meta.stop_time = info.stop_time
            meta.mission = _mission_from_letter(
                info.mission
            )
            meta.mode = _mode_from_string(info.mode)

        # Manifest overrides directory-name times with
        # sub-second precision when available.
        if self._product is not None:
            try:
                mi = self._product.manifest_info
                if mi and mi.acquisition_period_start:
                    meta.start_time = (
                        mi.acquisition_period_start
                        .replace(tzinfo=None)
                    )
                if mi and mi.acquisition_period_stop:
                    meta.stop_time = (
                        mi.acquisition_period_stop
                        .replace(tzinfo=None)
                    )
            except Exception as e:
                logger.debug(
                    f"Could not read manifest for times: {e}"
                )

        # Merge annotation data (only present for L1-flavor
        # SAFE products; L0 has no annotation XML).
        if self._annotation_data:
            merged = merge_annotation_data(
                self._annotation_data
            )
            meta.orbit_state_vectors = (
                merged.orbit_state_vectors
            )
            meta.attitude_records = merged.attitude_records
            meta.radar_parameters = merged.radar_parameters
            meta.swath_parameters = merged.swath_parameters
            meta.geolocation_grid = merged.geolocation_grid
            meta.downlink_info = merged.downlink_info
        else:
            mode_str = (
                meta.mode.value if meta.mode else "IW"
            )
            meta.radar_parameters = _default_radar_params(
                mode_str
            )

        # Pull orbit vectors from POE when they are missing
        # from the annotation data (the common L0 case).
        if (
            not meta.orbit_state_vectors
            and self._orbit_loader is not None
            and self._orbit_loader.has_poe_orbit
        ):
            meta.orbit_state_vectors = (
                self._orbit_loader.get_vectors(
                    prefer_poe=True,
                )
            )

        # Fallback: derive swath parameters from packet-based
        # burst detection when annotations are missing or empty.
        if (
            not meta.swath_parameters
            and self._burst_readers
            and meta.mode is not None
        ):
            meta.swath_parameters = (
                self._swath_params_from_bursts(meta)
            )

        # Populate per-swath downlink info from ISP packets if
        # not already present.
        if not meta.downlink_info and self._burst_readers:
            meta.downlink_info = (
                self._downlink_from_packets(meta)
            )

        # Overall dimensions from packet-based burst detection.
        rows, cols = self._rows_cols_from_bursts()
        meta.rows = rows
        meta.cols = cols
        return meta

    def _swath_params_from_bursts(
        self, meta: Sentinel1L0Metadata,
    ) -> List[S1L0SwathParameters]:
        """Derive swath parameters from packet-based bursts."""
        mode_str = (
            meta.mode.value if meta.mode else "IW"
        )
        rp = meta.radar_parameters
        rs_rate = (
            rp.range_sampling_rate_hz
            if rp
            else IW_MODE_PARAMS["range_sampling_rate_hz"]
        )

        swath_map: Dict[str, Dict] = {}
        for reader in self._burst_readers.values():
            for b in reader.get_burst_info():
                if b.swath not in IW_RAW_TO_LOGICAL:
                    continue
                sw_name = raw_swath_to_name(
                    b.swath, mode_str
                )
                swath_map.setdefault(
                    sw_name, {"bursts": []}
                )["bursts"].append(b)

        pol = (
            meta.polarizations[0]
            if meta.polarizations else ""
        )
        swath_params: List[S1L0SwathParameters] = []
        for sw_name in sorted(swath_map.keys()):
            burst_list = swath_map[sw_name]["bursts"]
            lines = [b.num_lines for b in burst_list]
            samples = [
                b.num_samples for b in burst_list
                if b.num_samples > 0
            ]
            pris = [
                b.pri for b in burst_list if b.pri > 0
            ]
            swsts = [
                b.swst for b in burst_list if b.swst > 0
            ]
            swath_params.append(S1L0SwathParameters(
                swath_id=sw_name,
                polarization=pol,
                num_bursts=len(burst_list),
                lines_per_burst=(
                    int(sum(lines) / len(lines))
                    if lines else 0
                ),
                samples_per_burst=(
                    int(sum(samples) / len(samples))
                    if samples else 0
                ),
                azimuth_time_interval=(
                    sum(pris) / len(pris) if pris else 0.0
                ),
                range_sampling_rate=rs_rate,
                slant_range_time=(
                    sum(swsts) / len(swsts)
                    if swsts else 0.0
                ),
            ))
        return swath_params

    def _downlink_from_packets(
        self, meta: Sentinel1L0Metadata,
    ) -> List[S1L0DownlinkInfo]:
        """Derive per-swath downlink info from ISP packets."""
        mode_str = (
            meta.mode.value if meta.mode else "IW"
        )
        downlinks: List[S1L0DownlinkInfo] = []
        seen: set = set()

        for br in self._burst_readers.values():
            decoder = getattr(br, "_decoder", None)
            if decoder is None:
                continue
            mdf = getattr(decoder, "_metadata_df", None)
            if mdf is None:
                continue

            for sw in sorted(mdf["Swath Number"].unique()):
                if sw not in IW_RAW_TO_LOGICAL:
                    continue
                sw_name = raw_swath_to_name(sw, mode_str)
                if sw_name in seen:
                    continue
                seen.add(sw_name)

                sub = mdf[mdf["Swath Number"] == sw]

                try:
                    from sentinel1decoder.utilities import (
                        range_dec_to_sample_rate,
                    )
                    rgdec = int(
                        sub["Range Decimation"].median()
                    )
                    rgdec_fs = range_dec_to_sample_rate(
                        rgdec
                    )
                except Exception:
                    rgdec = -1
                    rgdec_fs = 0.0

                pri_median = float(sub["PRI"].median())
                downlinks.append(S1L0DownlinkInfo(
                    prf=(
                        1.0 / pri_median
                        if pri_median > 0 else 0.0
                    ),
                    pri=pri_median,
                    rank=int(sub["Rank"].median()),
                    swst=float(sub["SWST"].median()),
                    swl=(
                        int(sub["SWL"].median())
                        if "SWL" in sub.columns else 0
                    ),
                    swath=sw_name,
                    tx_pulse_ramp_rate=float(
                        sub["Tx Ramp Rate"].median()
                    ),
                    tx_pulse_length=float(
                        sub["Tx Pulse Length"].median()
                    ),
                    tx_pulse_start_frequency=float(
                        sub["Tx Pulse Start Frequency"]
                        .median()
                    ),
                    range_decimation_code=rgdec,
                    range_sampling_rate_hz=rgdec_fs,
                ))
            break  # One burst reader is enough.
        return downlinks

    def _rows_cols_from_bursts(self) -> Tuple[int, int]:
        """Aggregate rows/cols across bursts in the first reader."""
        reader = self._get_burst_reader(
            self._current_polarization
        )
        if reader is None:
            return (0, 0)
        try:
            bursts = reader.get_burst_info()
        except Exception:
            return (0, 0)
        if not bursts:
            return (0, 0)
        rows = sum(b.num_lines for b in bursts)
        cols = max(
            (b.num_samples for b in bursts), default=0,
        )
        return (rows, cols)


# =============================================================================
# Mapping helpers
# =============================================================================


def _mission_from_letter(
    letter: Optional[str],
) -> Optional[Sentinel1Mission]:
    """Map a mission letter to :class:`Sentinel1Mission`."""
    if not letter:
        return None
    mapping = {
        "A": Sentinel1Mission.S1A,
        "B": Sentinel1Mission.S1B,
        "C": Sentinel1Mission.S1C,
        "D": Sentinel1Mission.S1D,
    }
    return mapping.get(letter.upper())


def _mode_from_string(
    mode: Optional[str],
) -> Optional[Sentinel1Mode]:
    """Map a mode string to :class:`Sentinel1Mode`."""
    if not mode:
        return None
    mapping = {
        "IW": Sentinel1Mode.IW,
        "EW": Sentinel1Mode.EW,
        "SM": Sentinel1Mode.SM,
        "WV": Sentinel1Mode.WV,
    }
    return mapping.get(mode.upper())


def _default_radar_params(
    mode_str: str,
) -> S1L0RadarParameters:
    """Build default radar parameters for a mode."""
    params = MODE_PARAMS.get(mode_str, IW_MODE_PARAMS)
    return S1L0RadarParameters(
        center_frequency_hz=SENTINEL1_CENTER_FREQUENCY_HZ,
        range_sampling_rate_hz=(
            params["range_sampling_rate_hz"]
        ),
        pulse_repetition_frequency_hz=(
            params["pulse_repetition_frequency_hz"]
        ),
        tx_pulse_length_s=params["tx_pulse_length_s"],
        tx_pulse_ramp_rate_hz_per_s=(
            params["tx_pulse_ramp_rate_hz_per_s"]
        ),
        azimuth_steering_rate_deg_per_s=(
            params.get(
                "azimuth_steering_rate_deg_per_s", 0.0
            )
        ),
    )


# =============================================================================
# Convenience function
# =============================================================================


def open_safe_product(
    path: Union[str, Path],
    **kwargs,
) -> Sentinel1L0Reader:
    """Open a Sentinel-1 L0 SAFE product.

    Parameters
    ----------
    path : str or Path
        Path to SAFE product directory.
    **kwargs
        Forwarded to :class:`ReaderConfig`.

    Returns
    -------
    Sentinel1L0Reader
    """
    config = ReaderConfig(**kwargs)
    return Sentinel1L0Reader(path, config=config)
