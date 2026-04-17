# -*- coding: utf-8 -*-
"""
Sentinel-1 L0 SAFE Product Handling.

SAFE (Standard Archive Format for Europe) product structure handling
for Sentinel-1 Level 0 products.  Provides validation, file discovery,
and manifest parsing.

SAFE Directory Structure
------------------------
::

    S1x_IW_RAW__0SDV_...SAFE/
    ├── manifest.safe                    # Product manifest XML
    ├── annotation/                      # Annotation XML files
    │   ├── s1x-iw-raw-vv-...xml
    │   └── s1x-iw-raw-vh-...xml
    ├── measurement/                     # Measurement data files
    │   ├── s1x-iw-raw-vv-....dat       # Raw ISP packets
    │   ├── s1x-iw-raw-vv-...-index.dat # Burst index
    │   └── s1x-iw-raw-vv-...-annot.dat # Packet annotations
    └── support/                         # Support files (optional)

References
----------
- Sentinel-1 Level-0 Product Format Specifications (S1PD.SP.00110)

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
2026-04-16

Modified
--------
2026-04-16
"""

# Standard library
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class InvalidSAFEProductError(Exception):
    """Raised when SAFE product structure is invalid."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(
            f"Invalid SAFE product '{path}': {reason}"
        )


# =============================================================================
# File discovery results
# =============================================================================


@dataclass
class MeasurementFileSet:
    """Related files for a single measurement channel.

    Parameters
    ----------
    measurement_file : Path
        Path to measurement ``.dat`` file.
    index_file : Path, optional
        Path to burst index file.
    annot_file : Path, optional
        Path to packet annotation file.
    polarization : str, optional
        Polarization derived from filename.
    swath : str, optional
        Swath identifier derived from filename.
    """

    measurement_file: Path
    index_file: Optional[Path] = None
    annot_file: Optional[Path] = None
    polarization: str = ""
    swath: str = ""

    @property
    def has_index(self) -> bool:
        """Whether a burst index file is available."""
        return (
            self.index_file is not None
            and self.index_file.exists()
        )

    @property
    def has_annotations(self) -> bool:
        """Whether a packet annotation file is available."""
        return (
            self.annot_file is not None
            and self.annot_file.exists()
        )

    @property
    def stem(self) -> str:
        """Base filename without extension."""
        return self.measurement_file.stem


# =============================================================================
# Product info from filename
# =============================================================================


@dataclass(frozen=True)
class ProductIdentifier:
    """Product information extracted from a SAFE directory name.

    Sentinel-1 naming convention::

        S1A_IW_RAW__0SDV_20230501T060001_20230501T060101_\
048123_05C7E2_F3B9.SAFE

    Parameters
    ----------
    mission : str
        Mission letter (``A``, ``B``, ``C``, ``D``).
    mode : str
        Acquisition mode (``IW``, ``EW``, ``SM``, ``WV``).
    product_type : str
        Product type (``RAW``).
    resolution : str
        Resolution class (``_``, ``F``, ``H``, ``M``).
    processing_level : str
        Processing level (``0``, ``1``, ``2``).
    product_class : str
        Product class (``S``, ``A``).
    polarization_code : str
        Polarization (``DV``, ``DH``, ``SV``, ``SH``).
    start_time, stop_time : datetime or None
        Acquisition start and stop.
    orbit_number : int
        Absolute orbit number.
    mission_datatake_id : str
        Datatake identifier.
    product_id : str
        Unique product identifier.
    full_name : str
        Complete product name without ``.SAFE``.
    """

    mission: str
    mode: str
    product_type: str
    resolution: str
    processing_level: str
    product_class: str
    polarization_code: str
    start_time: Optional[datetime]
    stop_time: Optional[datetime]
    orbit_number: int
    mission_datatake_id: str
    product_id: str
    full_name: str

    @property
    def platform_name(self) -> str:
        """Full platform name (e.g., ``'Sentinel-1A'``)."""
        return f"Sentinel-1{self.mission}"

    @property
    def is_dual_pol(self) -> bool:
        """Whether the product is dual polarization."""
        return self.polarization_code.startswith("D")

    @property
    def polarizations(self) -> List[str]:
        """List of polarization channels present.

        Returns
        -------
        list of str
            ``['VV', 'VH']`` or ``['HH', 'HV']`` for dual,
            ``['VV']`` or ``['HH']`` for single.
        """
        pol_map = {
            "DV": ["VV", "VH"],
            "DH": ["HH", "HV"],
            "SV": ["VV"],
            "SH": ["HH"],
        }
        return pol_map.get(self.polarization_code, [])


# Product name pattern.
PRODUCT_PATTERN = re.compile(
    r"S1([ABCD])_"                     # Mission
    r"(IW|EW|SM|WV)_"                  # Mode
    r"RAW(_+)(\d)([SA])([A-Z]{2})_"    # Type/resolution/level/class/pol
    r"(\d{8}T\d{6})_"                  # Start time
    r"(\d{8}T\d{6})_"                  # Stop time
    r"(\d{6})_"                        # Orbit number
    r"([0-9A-F]{6})_"                  # Mission datatake ID
    r"([0-9A-F]{4})"                   # Product ID
    r"\.SAFE$",
    re.IGNORECASE,
)


def parse_product_name(
    product_name: str,
) -> Optional[ProductIdentifier]:
    """Parse product information from a SAFE directory name.

    Parameters
    ----------
    product_name : str
        SAFE product directory name (with ``.SAFE`` extension).

    Returns
    -------
    ProductIdentifier or None
        Extracted info, or ``None`` if the name does not match the
        Sentinel-1 naming pattern.
    """
    match = PRODUCT_PATTERN.match(product_name)
    if not match:
        return None

    try:
        start_time = datetime.strptime(
            match.group(7), "%Y%m%dT%H%M%S"
        )
        stop_time = datetime.strptime(
            match.group(8), "%Y%m%dT%H%M%S"
        )
    except ValueError:
        start_time = None
        stop_time = None

    return ProductIdentifier(
        mission=match.group(1),
        mode=match.group(2),
        product_type="RAW",
        resolution=match.group(3).strip("_") or "_",
        processing_level=match.group(4),
        product_class=match.group(5),
        polarization_code=match.group(6),
        start_time=start_time,
        stop_time=stop_time,
        orbit_number=int(match.group(9)),
        mission_datatake_id=match.group(10),
        product_id=match.group(11),
        full_name=product_name.replace(".SAFE", ""),
    )


# =============================================================================
# Manifest parser
# =============================================================================

# XML namespaces for manifest.safe.
SAFE_NAMESPACES = {
    "safe": "http://www.esa.int/safe/sentinel-1.0",
    "s1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1",
    "s1sarl0": (
        "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/"
        "level-0"
    ),
    "gml": "http://www.opengis.net/gml",
}


@dataclass
class ManifestInfo:
    """Information extracted from ``manifest.safe``.

    Covers all fields from S1PD.SP.00110.ATSR Table 3-6 Metadata
    Section.

    Ref: [L0FMT] S1PD.SP.00110.ATSR Issue 3.1, Table 3-6.
    """

    # --- Orbit reference (measurementOrbitReference) ---
    orbit_direction: str = ""
    orbit_number_start: int = 0
    orbit_number_stop: int = 0
    relative_orbit: int = 0
    relative_orbit_stop: int = 0
    cycle_number: Optional[int] = None
    phase_identifier: str = ""
    ascending_node_time: Optional[datetime] = None
    frame_number: int = 0

    # --- Footprint (measurementFrameSet) ---
    footprint_coords: List[Tuple[float, float]] = field(
        default_factory=list
    )

    # --- Processing (processing) ---
    processing_center: str = ""
    processing_time: Optional[datetime] = None
    processing_start: Optional[datetime] = None
    processing_name: str = ""
    processing_facility: Optional[Dict[str, str]] = None
    processing_resources: Optional[List[Dict[str, str]]] = None
    software_name: str = ""
    software_version: str = ""

    # --- Acquisition period (acquisitionPeriod) ---
    acquisition_period_start: Optional[datetime] = None
    acquisition_period_stop: Optional[datetime] = None

    # --- Platform (platform) ---
    nssdc_identifier: str = ""
    family_name: str = ""
    platform_number: str = ""
    instrument_family: str = ""
    instrument_abbreviation: str = ""

    # --- Instrument mode (s1sarl0:instrumentMode) ---
    instrument_mode: str = ""
    instrument_swath: str = ""

    # --- General product info (generalProductInformation) ---
    product_class: str = ""
    product_class_description: str = ""
    product_consolidation: str = ""
    product_sensing_consolidation: str = ""
    instrument_configuration_id: Optional[int] = None
    mission_data_take_id: str = ""
    slice_product_flag: Optional[bool] = None
    slice_number: Optional[int] = None
    total_slices: Optional[int] = None
    echo_compression_type: str = ""
    noise_compression_type: str = ""
    cal_compression_type: str = ""
    transmitter_receiver_pol: List[str] = field(
        default_factory=list
    )

    # --- ANX timing ---
    start_time_anx: Optional[float] = None
    stop_time_anx: Optional[float] = None

    # --- Extended product info ---
    cal_isp_present: Optional[bool] = None
    noise_isp_present: Optional[bool] = None
    circulation_flag: Optional[int] = None
    theoretical_slice_length: Optional[float] = None
    slice_overlap: Optional[float] = None
    data_take_start_time: Optional[datetime] = None
    packet_store_id: List[int] = field(default_factory=list)
    byte_order: str = ""
    average_bit_rate: Optional[int] = None
    first_burst_cycle_number: Optional[int] = None

    # --- Quality (qualityInformation) ---
    quality_entries: List[Dict[str, Any]] = field(
        default_factory=list
    )

    @property
    def center_lat(self) -> Optional[float]:
        """Center latitude of footprint."""
        if self.footprint_coords:
            lats = [c[0] for c in self.footprint_coords]
            return sum(lats) / len(lats)
        return None

    @property
    def center_lon(self) -> Optional[float]:
        """Center longitude of footprint."""
        if self.footprint_coords:
            lons = [c[1] for c in self.footprint_coords]
            return sum(lons) / len(lons)
        return None

    @property
    def is_ascending(self) -> bool:
        """Whether the orbit pass is ascending."""
        return self.orbit_direction.upper() == "ASCENDING"


# =============================================================================
# XML element helpers
# =============================================================================


def _safe_int(elem: Optional[ET.Element]) -> Optional[int]:
    """Extract integer text from an XML element."""
    if elem is not None and elem.text:
        try:
            return int(elem.text.strip())
        except ValueError:
            return None
    return None


def _safe_float(elem: Optional[ET.Element]) -> Optional[float]:
    """Extract float text from an XML element."""
    if elem is not None and elem.text:
        try:
            return float(elem.text.strip())
        except ValueError:
            return None
    return None


def _safe_text(elem: Optional[ET.Element]) -> str:
    """Extract text from an XML element, default ``''``."""
    if elem is not None and elem.text:
        return elem.text.strip()
    return ""


def _safe_datetime(
    text: Optional[str],
) -> Optional[datetime]:
    """Parse ISO-8601 datetime, with ``Z`` suffix support."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(
            text.strip().replace("Z", "+00:00")
        )
    except ValueError:
        return None


def _safe_bool(elem: Optional[ET.Element]) -> Optional[bool]:
    """Extract boolean (``'true'``/``'false'``) text from element."""
    if elem is not None and elem.text:
        return elem.text.strip().lower() == "true"
    return None


def parse_manifest(manifest_path: Path) -> ManifestInfo:
    """Parse a ``manifest.safe`` XML file.

    Extracts all Table 3-6 metadata elements including platform,
    orbit reference, general product information, processing,
    quality, and footprint.

    Parameters
    ----------
    manifest_path : Path
        Path to ``manifest.safe``.

    Returns
    -------
    ManifestInfo
        All extracted manifest information.

    Raises
    ------
    InvalidSAFEProductError
        If the manifest cannot be parsed.
    """
    info = ManifestInfo()

    try:
        tree = ET.parse(manifest_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise InvalidSAFEProductError(
            str(manifest_path.parent),
            f"Failed to parse manifest.safe: {e}",
        )

    ns = SAFE_NAMESPACES

    # ---------------------------------------------------------
    # Platform (Table 3-6: platform section)
    # ---------------------------------------------------------
    platform = root.find(".//safe:platform", ns)
    if platform is not None:
        info.nssdc_identifier = _safe_text(
            platform.find("safe:nssdcIdentifier", ns)
        )
        info.family_name = _safe_text(
            platform.find("safe:familyName", ns)
        )
        info.platform_number = _safe_text(
            platform.find("safe:number", ns)
        )

        instrument = platform.find("safe:instrument", ns)
        if instrument is not None:
            info.instrument_family = _safe_text(
                instrument.find("safe:familyName", ns)
            )
            info.instrument_abbreviation = _safe_text(
                instrument.find("safe:abbreviation", ns)
            )

            inst_mode = instrument.find(
                ".//s1sarl0:instrumentMode", ns
            )
            if inst_mode is not None:
                info.instrument_mode = _safe_text(
                    inst_mode.find("s1sarl0:mode", ns)
                )
                info.instrument_swath = _safe_text(
                    inst_mode.find("s1sarl0:swath", ns)
                )

    # ---------------------------------------------------------
    # Orbit reference (measurementOrbitReference)
    # ---------------------------------------------------------
    orbit_elem = root.find(".//s1:pass", ns)
    if orbit_elem is not None and orbit_elem.text:
        info.orbit_direction = orbit_elem.text.strip().upper()

    orbit_start = root.find(
        ".//safe:orbitNumber[@type='start']", ns
    )
    info.orbit_number_start = _safe_int(orbit_start) or 0

    orbit_stop = root.find(
        ".//safe:orbitNumber[@type='stop']", ns
    )
    info.orbit_number_stop = _safe_int(orbit_stop) or 0

    rel_start = root.find(
        ".//safe:relativeOrbitNumber[@type='start']", ns
    )
    if rel_start is None:
        rel_start = root.find(
            ".//s1:relativeOrbitNumber", ns
        )
    info.relative_orbit = _safe_int(rel_start) or 0

    rel_stop = root.find(
        ".//safe:relativeOrbitNumber[@type='stop']", ns
    )
    info.relative_orbit_stop = _safe_int(rel_stop) or 0

    info.cycle_number = _safe_int(
        root.find(".//safe:cycleNumber", ns)
    )
    info.phase_identifier = _safe_text(
        root.find(".//safe:phaseIdentifier", ns)
    )

    anx_elem = root.find(".//s1:ascendingNodeTime", ns)
    info.ascending_node_time = _safe_datetime(
        _safe_text(anx_elem) if anx_elem is not None else None
    )

    frame = root.find(".//safe:frame", ns)
    if frame is not None:
        info.frame_number = (
            _safe_int(frame.find("safe:frameNumber", ns)) or 0
        )

    # ---------------------------------------------------------
    # Footprint (measurementFrameSet/frame/footPrint)
    # ---------------------------------------------------------
    coords_elem = root.find(
        ".//safe:frameSet/safe:frame"
        "/safe:footPrint/gml:coordinates",
        ns,
    )
    if coords_elem is not None and coords_elem.text:
        try:
            coords_text = coords_elem.text.strip()
            for point in coords_text.split():
                lat_str, lon_str = point.split(",")
                info.footprint_coords.append(
                    (float(lat_str), float(lon_str))
                )
        except (ValueError, AttributeError):
            pass

    # ---------------------------------------------------------
    # ANX timing (s1:timeANX)
    # ---------------------------------------------------------
    info.start_time_anx = _safe_float(
        root.find(".//s1:startTimeANX", ns)
    )
    info.stop_time_anx = _safe_float(
        root.find(".//s1:stopTimeANX", ns)
    )

    # ---------------------------------------------------------
    # Processing (processing section)
    # ---------------------------------------------------------
    proc_elem = root.find(".//safe:processing", ns)
    if proc_elem is not None:
        info.processing_name = proc_elem.get("name", "")
        info.processing_center = proc_elem.get("site", "")
        info.processing_start = _safe_datetime(
            proc_elem.get("start")
        )
        info.processing_time = _safe_datetime(
            proc_elem.get("stop")
        )

        facility = proc_elem.find(".//safe:facility", ns)
        if facility is not None:
            info.processing_facility = {
                "country": facility.get("country", ""),
                "name": facility.get("name", ""),
                "organisation": facility.get("organisation", ""),
                "site": facility.get("site", ""),
            }

        software = proc_elem.find(".//safe:software", ns)
        if software is not None:
            info.software_name = software.get("name", "")
            info.software_version = software.get("version", "")

        resources = proc_elem.findall(
            ".//safe:resource", ns
        )
        if resources:
            info.processing_resources = []
            for res in resources:
                info.processing_resources.append({
                    "name": res.get("name", ""),
                    "role": res.get("role", ""),
                    "href": res.get("href", ""),
                    "version": res.get("version", ""),
                })

    # ---------------------------------------------------------
    # Acquisition period
    # ---------------------------------------------------------
    acq_period = root.find(".//safe:acquisitionPeriod", ns)
    if acq_period is not None:
        start_el = acq_period.find("safe:startTime", ns)
        stop_el = acq_period.find("safe:stopTime", ns)
        info.acquisition_period_start = _safe_datetime(
            _safe_text(start_el)
        )
        info.acquisition_period_stop = _safe_datetime(
            _safe_text(stop_el)
        )

    # ---------------------------------------------------------
    # General product information
    # ---------------------------------------------------------
    _parse_product_info(root, ns, info)

    # ---------------------------------------------------------
    # Quality information
    # ---------------------------------------------------------
    _parse_quality_info(root, ns, info)

    return info


def _parse_product_info(
    root: ET.Element,
    ns: Dict[str, str],
    info: ManifestInfo,
) -> None:
    """Extract ``generalProductInformation`` fields into ``info``."""
    gpi = root.find(
        ".//s1sarl0:standAloneProductInformation", ns
    )
    if gpi is None:
        gpi = root.find(
            ".//s1sarl0:generalProductInformation", ns
        )
    if gpi is None:
        return

    info.product_class = _safe_text(
        gpi.find("s1sarl0:productClass", ns)
    )
    info.product_class_description = _safe_text(
        gpi.find("s1sarl0:productClassDescription", ns)
    )
    info.product_consolidation = _safe_text(
        gpi.find("s1sarl0:productConsolidation", ns)
    )
    info.product_sensing_consolidation = _safe_text(
        gpi.find("s1sarl0:productSensingConsolidation", ns)
    )
    info.instrument_configuration_id = _safe_int(
        gpi.find("s1sarl0:instrumentConfigurationID", ns)
    )
    info.mission_data_take_id = _safe_text(
        gpi.find("s1sarl0:missionDataTakeID", ns)
    )
    info.slice_product_flag = _safe_bool(
        gpi.find("s1sarl0:sliceProductFlag", ns)
    )
    info.slice_number = _safe_int(
        gpi.find("s1sarl0:sliceNumber", ns)
    )
    info.total_slices = _safe_int(
        gpi.find("s1sarl0:totalSlices", ns)
    )
    info.echo_compression_type = _safe_text(
        gpi.find("s1sarl0:echoCompressionType", ns)
    )
    info.noise_compression_type = _safe_text(
        gpi.find("s1sarl0:noiseCompressionType", ns)
    )
    info.cal_compression_type = _safe_text(
        gpi.find("s1sarl0:calCompressionType", ns)
    )

    pol_elems = gpi.findall(
        "s1sarl0:transmitterReceiverPolarisation", ns
    )
    info.transmitter_receiver_pol = [
        _safe_text(p) for p in pol_elems if _safe_text(p)
    ]

    info.cal_isp_present = _safe_bool(
        gpi.find("s1sarl0:calISPPresent", ns)
    )
    info.noise_isp_present = _safe_bool(
        gpi.find("s1sarl0:noiseISPPresent", ns)
    )
    info.circulation_flag = _safe_int(
        gpi.find("s1sarl0:circulationFlag", ns)
    )
    info.theoretical_slice_length = _safe_float(
        gpi.find("s1sarl0:theoreticalSliceLength", ns)
    )
    info.slice_overlap = _safe_float(
        gpi.find("s1sarl0:sliceOverlap", ns)
    )
    dts = gpi.find("s1sarl0:dataTakeStartTime", ns)
    info.data_take_start_time = _safe_datetime(
        _safe_text(dts) if dts is not None else None
    )

    ps_elems = gpi.findall("s1sarl0:packetStoreID", ns)
    info.packet_store_id = [
        int(p.text.strip())
        for p in ps_elems
        if p.text and p.text.strip().isdigit()
    ]

    info.byte_order = _safe_text(
        gpi.find("s1sarl0:byteOrder", ns)
    )
    info.average_bit_rate = _safe_int(
        gpi.find("s1sarl0:averageBitRate", ns)
    )
    info.first_burst_cycle_number = _safe_int(
        gpi.find("s1sarl0:firstBurstCycleNumber", ns)
    )


def _parse_quality_info(
    root: ET.Element,
    ns: Dict[str, str],
    info: ManifestInfo,
) -> None:
    """Extract quality information entries into ``info``."""
    qual_elems = root.findall(".//s1:qualityProperties", ns)
    for qe in qual_elems:
        entry: Dict[str, Any] = {}
        parent = qe.find("..")
        if parent is not None:
            obj_id = parent.get("dataObjectID", "")
            if obj_id:
                entry["data_object_id"] = obj_id

        entry["num_of_elements"] = _safe_int(
            qe.find("s1:numOfElements", ns)
        )
        entry["num_of_missing_elements"] = _safe_int(
            qe.find("s1:numOfMissingElements", ns)
        )
        entry["num_of_corrupted_elements"] = _safe_int(
            qe.find("s1:numOfCorruptedElements", ns)
        )
        entry["num_of_rs_incorrigible"] = _safe_int(
            qe.find("s1:numOfRSIncorrigibleElements", ns)
        )
        entry["num_of_rs_corrected"] = _safe_int(
            qe.find("s1:numOfRSCorrectedElements", ns)
        )
        entry["num_of_rs_corrected_symbols"] = _safe_int(
            qe.find("s1:numOfRSCorrectedSymbols", ns)
        )
        info.quality_entries.append(entry)

    if not qual_elems:
        qual_data = root.findall(
            ".//safe:qualityDataObjectID", ns
        )
        for qd in qual_data:
            entry = {
                "data_object_id": qd.text.strip()
                if qd.text else "",
            }
            info.quality_entries.append(entry)


# =============================================================================
# SAFE Product class
# =============================================================================


@dataclass
class SAFEProduct:
    """Sentinel-1 SAFE product representation.

    Provides access to SAFE product structure, file discovery, and
    validation.  Manifest information is lazy-loaded on first access.

    Parameters
    ----------
    path : str or Path
        Path to SAFE product directory.
    validate : bool, optional
        Whether to validate structure on initialization.

    Examples
    --------
    >>> product = SAFEProduct('/path/to/S1A_IW_RAW__0SDV...SAFE')
    >>> product.validate()
    >>> for meas in product.measurement_files:
    ...     print(meas.polarization, meas.measurement_file)
    """

    path: Path
    _product_info: Optional[ProductIdentifier] = field(
        default=None, repr=False
    )
    _manifest_info: Optional[ManifestInfo] = field(
        default=None, repr=False
    )
    _manifest_tree: Optional[ET.ElementTree] = field(
        default=None, repr=False
    )
    _measurement_files: List[MeasurementFileSet] = field(
        default_factory=list, repr=False
    )
    _annotation_files: List[Path] = field(
        default_factory=list, repr=False
    )
    _validated: bool = field(default=False, repr=False)

    def __init__(
        self,
        path: Union[str, Path],
        validate: bool = True,
    ) -> None:
        self.path = Path(path)
        self._product_info = None
        self._manifest_info = None
        self._manifest_tree = None
        self._measurement_files = []
        self._annotation_files = []
        self._validated = False

        if validate:
            self.validate()

        self._discover_files()

    @property
    def product_info(self) -> Optional[ProductIdentifier]:
        """Product identifier parsed from directory name."""
        if self._product_info is None:
            self._product_info = parse_product_name(
                self.path.name
            )
        return self._product_info

    @property
    def manifest_info(self) -> ManifestInfo:
        """Manifest information (lazy-loaded)."""
        if self._manifest_info is None:
            manifest_path = self.path / "manifest.safe"
            self._manifest_info = parse_manifest(manifest_path)
        return self._manifest_info

    @property
    def manifest(self) -> Optional[ET.Element]:
        """Root element of ``manifest.safe``."""
        if self._manifest_tree is None:
            manifest_path = self.path / "manifest.safe"
            if manifest_path.exists():
                self._manifest_tree = ET.parse(manifest_path)
        return (
            self._manifest_tree.getroot()
            if self._manifest_tree
            else None
        )

    @property
    def measurement_files(self) -> List[MeasurementFileSet]:
        """Discovered measurement file sets."""
        return self._measurement_files

    @property
    def annotation_files(self) -> List[Path]:
        """Discovered annotation XML files."""
        return self._annotation_files

    @property
    def product_name(self) -> str:
        """Product name without ``.SAFE`` extension."""
        return self.path.name.replace(".SAFE", "")

    @property
    def polarizations(self) -> List[str]:
        """Polarizations present in the product."""
        if self.product_info:
            return self.product_info.polarizations
        # Derive from measurement files.
        pols = set()
        for meas in self._measurement_files:
            if meas.polarization:
                pols.add(meas.polarization.upper())
        return sorted(pols)

    def validate(self) -> None:
        """Validate SAFE product structure.

        Raises
        ------
        InvalidSAFEProductError
            If the structure is invalid.
        """
        if self._validated:
            return

        if not self.path.exists():
            raise InvalidSAFEProductError(
                str(self.path), "Path does not exist"
            )

        if not self.path.is_dir():
            raise InvalidSAFEProductError(
                str(self.path), "Path is not a directory"
            )

        if not self.path.suffix.upper() == ".SAFE":
            raise InvalidSAFEProductError(
                str(self.path),
                "Directory must have .SAFE extension",
            )

        manifest_path = self.path / "manifest.safe"
        if not manifest_path.exists():
            raise InvalidSAFEProductError(
                str(self.path), "Missing manifest.safe"
            )

        measurement_dir = self.path / "measurement"
        if measurement_dir.exists():
            dat_files = list(measurement_dir.glob("*.dat"))
            dat_files = [
                f for f in dat_files
                if not f.name.endswith(
                    ("-annot.dat", "-index.dat")
                )
            ]
        else:
            dat_files = [
                f for f in self.path.glob("s1?-*-raw-*.dat")
                if not f.name.endswith(
                    ("-annot.dat", "-index.dat")
                )
            ]

        if not dat_files:
            raise InvalidSAFEProductError(
                str(self.path),
                "No measurement data files (.dat) found",
            )

        self._validated = True
        logger.debug(
            f"Validated SAFE product: {self.path.name}"
        )

    def _discover_files(self) -> None:
        """Discover measurement and annotation files."""
        self._discover_measurement_files()
        self._discover_annotation_files()

    def _discover_measurement_files(self) -> None:
        """Discover ``.dat`` files and associated index/annot files."""
        self._measurement_files = []

        measurement_dir = self.path / "measurement"
        if measurement_dir.exists():
            dat_files = sorted(measurement_dir.glob("*.dat"))
        else:
            dat_files = sorted(
                self.path.glob("s1?-*-raw-*.dat")
            )

        dat_files = [
            f for f in dat_files
            if not f.name.endswith(
                ("-annot.dat", "-index.dat")
            )
        ]

        for dat_file in dat_files:
            index_path = (
                dat_file.parent
                / f"{dat_file.stem}-index.dat"
            )
            annot_path = (
                dat_file.parent
                / f"{dat_file.stem}-annot.dat"
            )

            pol, swath = self._parse_measurement_filename(
                dat_file.name
            )

            self._measurement_files.append(MeasurementFileSet(
                measurement_file=dat_file,
                index_file=(
                    index_path if index_path.exists() else None
                ),
                annot_file=(
                    annot_path if annot_path.exists() else None
                ),
                polarization=pol,
                swath=swath,
            ))

        logger.debug(
            f"Discovered {len(self._measurement_files)} "
            "measurement files"
        )

    def _discover_annotation_files(self) -> None:
        """Discover annotation XML files."""
        self._annotation_files = []

        annotation_dir = self.path / "annotation"
        if annotation_dir.exists():
            xml_files = sorted(annotation_dir.glob("*.xml"))
        else:
            xml_files = sorted(
                self.path.glob("*-annot.xml")
            )

        self._annotation_files = xml_files
        logger.debug(
            f"Discovered {len(self._annotation_files)} "
            "annotation files"
        )

    @staticmethod
    def _parse_measurement_filename(
        filename: str,
    ) -> Tuple[str, str]:
        """Parse ``(polarization, swath)`` from a measurement filename.

        Parameters
        ----------
        filename : str
            Measurement file name.

        Returns
        -------
        (str, str)
            ``(polarization, swath)`` extracted from the name.
        """
        filename_lower = filename.lower()

        polarization = ""
        for pol in ["vv", "vh", "hh", "hv"]:
            if f"-{pol}-" in filename_lower:
                polarization = pol.upper()
                break

        swath = ""
        for mode in ["iw", "ew", "sm", "wv"]:
            if f"-{mode}-" in filename_lower:
                swath = mode.upper()
                break

        return polarization, swath

    def get_measurement_file(
        self, polarization: str,
    ) -> Optional[MeasurementFileSet]:
        """Get measurement file set for a specific polarization.

        Parameters
        ----------
        polarization : str
            Polarization (``VV``, ``VH``, ``HH``, ``HV``).

        Returns
        -------
        MeasurementFileSet or None
            Matching file set, or ``None`` if not found.
        """
        pol_upper = polarization.upper()
        for meas in self._measurement_files:
            if meas.polarization == pol_upper:
                return meas
        return None

    def get_annotation_file(
        self, polarization: str,
    ) -> Optional[Path]:
        """Get annotation file for a specific polarization.

        Parameters
        ----------
        polarization : str
            Polarization (``VV``, ``VH``, ``HH``, ``HV``).

        Returns
        -------
        Path or None
            Path to the annotation file, or ``None`` if not found.
        """
        pol_lower = polarization.lower()
        for annot in self._annotation_files:
            if pol_lower in annot.name.lower():
                return annot
        return None

    def summary(self) -> str:
        """Return a human-readable summary of the product."""
        lines = [
            f"SAFEProduct: {self.path.name}",
            f"  Path: {self.path}",
        ]

        if self.product_info:
            info = self.product_info
            lines.extend([
                f"  Mission: {info.platform_name}",
                f"  Mode: {info.mode}",
                f"  Polarization: {info.polarization_code}",
                f"  Orbit: {info.orbit_number}",
            ])
            if info.start_time:
                lines.append(
                    f"  Start: {info.start_time.isoformat()}"
                )
            if info.stop_time:
                lines.append(
                    f"  Stop: {info.stop_time.isoformat()}"
                )

        lines.extend([
            f"  Measurement files: "
            f"{len(self._measurement_files)}",
            f"  Annotation files: "
            f"{len(self._annotation_files)}",
        ])

        for meas in self._measurement_files:
            extras = []
            if meas.has_index:
                extras.append("index")
            if meas.has_annotations:
                extras.append("annot")
            extra_str = (
                f" [{', '.join(extras)}]" if extras else ""
            )
            lines.append(
                f"    {meas.polarization}: "
                f"{meas.stem}{extra_str}"
            )

        return "\n".join(lines)
