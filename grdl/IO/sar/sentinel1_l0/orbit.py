# -*- coding: utf-8 -*-
"""
Sentinel-1 L0 Orbit and Attitude Handling.

Consolidated orbit and attitude processing for Sentinel-1 data:

- Three-tier orbit source resolution (download → file → annotation)
- Auto-download of POEORB / RESORB from ESA step.esa.int
- Loading orbit data from annotation XML
- Loading POE (Precise Orbit Ephemerides) files
- Merging annotation and POE orbit data
- Orbit state vector interpolation (``CubicHermiteSpline``)
- Attitude interpolation

Orbit Source Tiers
------------------
Use the :data:`ORBIT_SOURCE_AUTO`, :data:`ORBIT_SOURCE_DOWNLOAD`,
:data:`ORBIT_SOURCE_FILE`, and :data:`ORBIT_SOURCE_ANNOTATION`
constants to select the strategy.  Pass the chosen constant
(or the equivalent string) as the ``orbit_source`` argument to
:class:`~grdl.IO.sar.sentinel1_l0.crsd_writer.Sentinel1L0ToCRSD`.

``"auto"``
    Try download first.  Fall back to annotation vectors with a
    precision warning if download fails.
``"download"``
    Require a successful ESA download.  POEORB is tried first
    (precise, < 5 cm; available ~21 days post-acquisition);
    RESORB is tried if POEORB is not yet available (~3 h post-
    acquisition).
``"file"``
    Load the caller-supplied ``orbit_file`` ``.EOF`` path.
    Raises if the file is ``None`` or does not exist.
``"annotation"``
    Use the sub-commutated orbit vectors embedded in the ISP
    annotation XML (~25 vectors at ~1 Hz spacing).  Always
    available but position accuracy is ~10 m — sufficient for
    burst-level phase coherence, inadequate for absolute
    geolocation to < 1 m.

Download
--------
Files are fetched from the ESA auxiliary data server::

    https://step.esa.int/auxdata/orbits/Sentinel-1/
        {POEORB|RESORB}/{S1A|S1B|S1C|S1D}/{year}/{month:02d}/

No authentication is required.  Downloaded files are cached to
``orbit_cache_dir`` (default ``~/.grdl/orbits``) and reused on
subsequent calls.

EOF File Naming
---------------
POEORB files follow the naming convention::

    S1A_OPER_AUX_POEORB_OPOD_YYYYMMDDTHHMMSS_VYYYYMMDDTHHMMSS_YYYYMMDDTHHMMSS.EOF

The ``V`` prefix encodes the validity window.  Filenames alone
are sufficient to determine coverage — no pre-download needed.

Dependencies
------------
requests (optional — required only for ``"download"`` and
``"auto"`` orbit sources)

References
----------
- Sentinel-1 POD Service File Format Specification
- https://step.esa.int/auxdata/orbits/Sentinel-1/

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
2026-04-27
"""

# Standard library
import logging
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Third-party
import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline

# GRDL internal
from grdl.IO.models.sentinel1_l0 import (
    S1L0AttitudeRecord,
    S1L0OrbitStateVector,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Orbit source constants
# =============================================================================

#: Try online download first, fall back to annotation vectors.
ORBIT_SOURCE_AUTO: str = "auto"
#: Download POEORB → RESORB from ESA; raise if both fail.
ORBIT_SOURCE_DOWNLOAD: str = "download"
#: Load a caller-supplied local .EOF file.
ORBIT_SOURCE_FILE: str = "file"
#: Use annotation-embedded sub-commutated state vectors (~1 Hz).
ORBIT_SOURCE_ANNOTATION: str = "annotation"

_VALID_ORBIT_SOURCES = frozenset({
    ORBIT_SOURCE_AUTO,
    ORBIT_SOURCE_DOWNLOAD,
    ORBIT_SOURCE_FILE,
    ORBIT_SOURCE_ANNOTATION,
})

# Base URL for ESA auxiliary orbit data (no auth required)
_ORBIT_BASE_URL = "https://step.esa.int/auxdata/orbits/Sentinel-1"

# EOF filename validity-window pattern:
#   ...._V20251207T225942_20251209T005942.EOF
#             ^^^^^^^^^^^^^^ start  ^^^^^^^^^^^^^^ stop
_EOF_VALIDITY_RE = re.compile(
    r"_V(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF$",
    re.IGNORECASE,
)

# Href pattern in an Apache directory listing
_HREF_RE = re.compile(r'href="([^"]+\.EOF)"', re.IGNORECASE)


# =============================================================================
# POE file parser
# =============================================================================


@dataclass
class POEFileInfo:
    """Information about a POE file.

    Parameters
    ----------
    filepath : Path
        Path to the ``.EOF`` file.
    mission : str, optional
        Mission name (e.g., ``'Sentinel-1A'``).
    validity_start, validity_stop : datetime, optional
        Start and end of validity period (UTC).
    num_vectors : int, optional
        Number of orbit state vectors parsed.
    """

    filepath: Path
    mission: str = ""
    validity_start: Optional[datetime] = None
    validity_stop: Optional[datetime] = None
    num_vectors: int = 0

    def covers_time(self, target_time: datetime) -> bool:
        """Whether this file covers a specific UTC time."""
        if (
            self.validity_start is None
            or self.validity_stop is None
        ):
            return False
        return (
            self.validity_start
            <= target_time
            <= self.validity_stop
        )


class POEParser:
    """Parser for ESA Precise Orbit Ephemerides (POE) files.

    POE files are XML format containing orbit state vectors at
    10-second intervals with position accuracy < 5 cm.

    Parameters
    ----------
    filepath : str or Path
        Path to ``.EOF`` file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Examples
    --------
    >>> parser = POEParser('/path/to/S1A_OPER_AUX_POEORB_*.EOF')
    >>> vectors = parser.parse()
    >>> print(f'Found {len(vectors)} orbit state vectors')
    >>> print(f'Validity: {parser.info.validity_start}')
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)
        self.info = POEFileInfo(filepath=self.filepath)
        self._orbit_vectors: List[S1L0OrbitStateVector] = []

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"POE file not found: {filepath}"
            )
        if not self.filepath.suffix.upper() == ".EOF":
            warnings.warn(
                "POE files typically have .EOF extension, "
                f"got: {self.filepath.suffix}"
            )

    @property
    def orbit_vectors(self) -> List[S1L0OrbitStateVector]:
        """Parsed orbit state vectors."""
        return self._orbit_vectors

    def parse(self) -> List[S1L0OrbitStateVector]:
        """Parse the POE file and extract orbit state vectors.

        Returns
        -------
        list of S1L0OrbitStateVector
            Parsed vectors sorted by time.

        Raises
        ------
        ValueError
            If the file format is invalid.
        """
        try:
            tree = ET.parse(self.filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse POE XML: {e}")

        self._parse_header(root)
        self._parse_osvs(root)

        self._orbit_vectors.sort(key=lambda x: x.time)
        self.info.num_vectors = len(self._orbit_vectors)
        return self._orbit_vectors

    def _parse_header(self, root: ET.Element) -> None:
        """Parse the POE file header."""
        header = root.find(".//Fixed_Header")
        if header is None:
            return

        mission_elem = header.find("Mission")
        if mission_elem is not None and mission_elem.text:
            self.info.mission = mission_elem.text

        validity = header.find("Validity_Period")
        if validity is not None:
            start_elem = validity.find("Validity_Start")
            stop_elem = validity.find("Validity_Stop")
            if start_elem is not None and start_elem.text:
                self.info.validity_start = self._parse_datetime(
                    start_elem.text
                )
            if stop_elem is not None and stop_elem.text:
                self.info.validity_stop = self._parse_datetime(
                    stop_elem.text
                )

    def _parse_osvs(self, root: ET.Element) -> None:
        """Parse orbit state vectors from ``Data_Block``."""
        data_block = root.find(".//Data_Block")
        if data_block is None:
            raise ValueError("No Data_Block found in POE file")

        osv_list = data_block.find(".//List_of_OSVs")
        if osv_list is None:
            raise ValueError(
                "No List_of_OSVs found in POE file"
            )

        for osv in osv_list.findall("OSV"):
            try:
                osv_obj = self._parse_osv(osv)
                self._orbit_vectors.append(osv_obj)
            except (ValueError, AttributeError) as e:
                warnings.warn(f"Failed to parse OSV: {e}")
                continue

    def _parse_osv(
        self, osv: ET.Element,
    ) -> S1L0OrbitStateVector:
        """Parse a single orbit state vector."""
        utc_elem = osv.find("UTC")
        if utc_elem is None or not utc_elem.text:
            raise ValueError("Missing UTC time in OSV")

        time = self._parse_datetime(utc_elem.text)

        x = float(osv.find("X").text)
        y = float(osv.find("Y").text)
        z = float(osv.find("Z").text)
        vx = float(osv.find("VX").text)
        vy = float(osv.find("VY").text)
        vz = float(osv.find("VZ").text)

        return S1L0OrbitStateVector(
            time=time,
            x=x, y=y, z=z,
            vx=vx, vy=vy, vz=vz,
        )

    @staticmethod
    def _parse_datetime(dt_string: str) -> datetime:
        """Parse a datetime string from a POE file."""
        # Remove prefix if present (e.g., ``"UTC=2025-..."``).
        if "=" in dt_string:
            dt_string = dt_string.split("=", 1)[1]
        dt_string = dt_string.strip()

        formats = [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_string, fmt)
            except ValueError:
                continue
        raise ValueError(
            f"Unable to parse datetime: {dt_string}"
        )


# =============================================================================
# Orbit interpolator
# =============================================================================


class OrbitInterpolator:
    """Cubic-spline interpolator for orbit state vectors.

    Provides high-precision interpolation of position and velocity
    at arbitrary times within the valid range.

    Parameters
    ----------
    orbit_vectors : list of S1L0OrbitStateVector
        Orbit state vectors to interpolate.
    reference_time : datetime
        Reference time for relative-seconds math.

    Raises
    ------
    ValueError
        If fewer than 4 vectors are provided (needed for cubic
        interpolation).

    Examples
    --------
    >>> interp = OrbitInterpolator(orbit_vectors, reference_time)
    >>> pos, vel = interp.interpolate(times_array)
    """

    def __init__(
        self,
        orbit_vectors: List[S1L0OrbitStateVector],
        reference_time: datetime,
    ) -> None:
        if len(orbit_vectors) < 4:
            raise ValueError(
                "Need at least 4 orbit vectors for cubic "
                f"interpolation, got {len(orbit_vectors)}"
            )
        self.reference_time = reference_time
        self._vectors = orbit_vectors
        self._build_interpolators()

    def _build_interpolators(self) -> None:
        times = np.array([
            (osv.time - self.reference_time).total_seconds()
            for osv in self._vectors
        ])

        px = np.array([osv.x for osv in self._vectors])
        py = np.array([osv.y for osv in self._vectors])
        pz = np.array([osv.z for osv in self._vectors])
        vx = np.array([osv.vx for osv in self._vectors])
        vy = np.array([osv.vy for osv in self._vectors])
        vz = np.array([osv.vz for osv in self._vectors])

        # Deduplicate on timestamp — sentinel1decoder v2.0.0 broadcasts the
        # same state vector in every sub-commutation slot, producing hundreds
        # of identical timestamps.  Keep the first occurrence of each unique
        # time (they carry identical values so the choice is arbitrary).
        _, unique_idx = np.unique(times, return_index=True)
        if len(unique_idx) < len(times):
            n_dup = len(times) - len(unique_idx)
            logger.debug(
                "Orbit: removed %d duplicate timestamp(s) "
                "(%d unique vectors remain)",
                n_dup, len(unique_idx),
            )
            times = times[unique_idx]
            px, py, pz = px[unique_idx], py[unique_idx], pz[unique_idx]
            vx, vy, vz = vx[unique_idx], vy[unique_idx], vz[unique_idx]

        if len(times) < 4:
            raise ValueError(
                f"Need at least 4 unique orbit timestamps for cubic "
                f"interpolation, got {len(times)} after deduplication."
            )

        self._px = CubicHermiteSpline(times, px, vx)
        self._py = CubicHermiteSpline(times, py, vy)
        self._pz = CubicHermiteSpline(times, pz, vz)
        # Velocity is the analytical derivative of the position spline.
        # This guarantees kinematic consistency (RcvVel = d/dt RcvPos)
        # without fitting an independent spline to the velocity nodes.
        self._pvx = self._px.derivative()
        self._pvy = self._py.derivative()
        self._pvz = self._pz.derivative()

        self.valid_start = float(times[0])
        self.valid_end = float(times[-1])

    def interpolate(
        self, times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate position and velocity at given times.

        Parameters
        ----------
        times : numpy.ndarray
            Times in seconds relative to ``reference_time``.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            ``(positions, velocities)``, each shape ``(N, 3)``.
        """
        px = self._px(times)
        py = self._py(times)
        pz = self._pz(times)
        vx = self._pvx(times)
        vy = self._pvy(times)
        vz = self._pvz(times)

        positions = np.column_stack([px, py, pz])
        velocities = np.column_stack([vx, vy, vz])
        return positions, velocities

    def interpolate_single(
        self, time_seconds: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate at a single time.

        Parameters
        ----------
        time_seconds : float
            Time in seconds relative to ``reference_time``.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            ``(position, velocity)`` as 1D arrays.
        """
        times = np.array([time_seconds])
        pos, vel = self.interpolate(times)
        return pos[0], vel[0]


# =============================================================================
# Attitude interpolator
# =============================================================================


class AttitudeInterpolator:
    """Cubic-spline interpolator for attitude records.

    Interpolates roll, pitch, yaw angles at arbitrary times.

    Parameters
    ----------
    attitude_records : list of S1L0AttitudeRecord
        Attitude records to interpolate.
    reference_time : datetime
        Reference time for relative-seconds math.

    Raises
    ------
    ValueError
        If fewer than 4 records are provided.

    Examples
    --------
    >>> interp = AttitudeInterpolator(records, reference_time)
    >>> roll, pitch, yaw = interp.interpolate(times_array)
    """

    def __init__(
        self,
        attitude_records: List[S1L0AttitudeRecord],
        reference_time: datetime,
    ) -> None:
        if len(attitude_records) < 4:
            raise ValueError(
                "Need at least 4 attitude records for "
                "interpolation, got "
                f"{len(attitude_records)}"
            )
        self.reference_time = reference_time
        self._records = attitude_records
        self._build_interpolators()

    def _build_interpolators(self) -> None:
        times = np.array([
            (att.time - self.reference_time).total_seconds()
            for att in self._records
        ])
        roll = np.array([att.roll for att in self._records])
        pitch = np.array([att.pitch for att in self._records])
        yaw = np.array([att.yaw for att in self._records])

        self._roll = CubicSpline(times, roll)
        self._pitch = CubicSpline(times, pitch)
        self._yaw = CubicSpline(times, yaw)

        self.valid_start = float(times[0])
        self.valid_end = float(times[-1])

    def interpolate(
        self, times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate attitude at given times.

        Parameters
        ----------
        times : numpy.ndarray
            Times in seconds relative to ``reference_time``.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            ``(roll, pitch, yaw)`` arrays in degrees.
        """
        roll = self._roll(times)
        pitch = self._pitch(times)
        yaw = self._yaw(times)
        return roll, pitch, yaw


# =============================================================================
# Orbit data loader
# =============================================================================


class OrbitLoader:
    """Unified loader for orbit data.

    Handles loading from annotation XML and/or POE files, with
    automatic preference for POE data when available.

    Examples
    --------
    >>> loader = OrbitLoader()
    >>> loader.load_annotation_vectors(annotation_vectors)
    >>> loader.load_poe_file('/path/to/poe.EOF')
    >>> vectors = loader.get_merged_vectors()
    """

    def __init__(self) -> None:
        self._annotation_vectors: List[
            S1L0OrbitStateVector
        ] = []
        self._poe_vectors: List[S1L0OrbitStateVector] = []
        self._poe_info: Optional[POEFileInfo] = None

    @property
    def has_annotation_orbit(self) -> bool:
        """Whether annotation orbit data is loaded."""
        return len(self._annotation_vectors) > 0

    @property
    def has_poe_orbit(self) -> bool:
        """Whether POE orbit data is loaded."""
        return len(self._poe_vectors) > 0

    def load_annotation_vectors(
        self,
        vectors: List[S1L0OrbitStateVector],
    ) -> None:
        """Load orbit vectors parsed from annotation XML.

        Parameters
        ----------
        vectors : list of S1L0OrbitStateVector
            Orbit state vectors.
        """
        self._annotation_vectors = vectors
        logger.debug(
            f"Loaded {len(vectors)} annotation orbit vectors"
        )

    def load_poe_file(
        self, poe_path: Union[str, Path],
    ) -> None:
        """Load orbit vectors from a POE ``.EOF`` file.

        Parameters
        ----------
        poe_path : str or Path
            Path to the ``.EOF`` file.
        """
        parser = POEParser(poe_path)
        self._poe_vectors = parser.parse()
        self._poe_info = parser.info
        logger.debug(
            f"Loaded {len(self._poe_vectors)} POE orbit "
            "vectors"
        )

    def find_and_load_poe(
        self,
        directory: Union[str, Path],
        target_time: datetime,
        mission: str = "S1A",
    ) -> bool:
        """Find and load a POE file covering a specific time.

        Parameters
        ----------
        directory : str or Path
            Directory to search.
        target_time : datetime
            Target UTC time.
        mission : str, optional
            Mission identifier (``'S1A'``, ``'S1B'``).

        Returns
        -------
        bool
            ``True`` if a covering POE file was loaded.
        """
        poe_file = find_poe_file_for_time(
            directory, target_time, mission
        )
        if poe_file:
            self.load_poe_file(poe_file)
            return True
        return False

    def get_vectors(
        self, prefer_poe: bool = True,
    ) -> List[S1L0OrbitStateVector]:
        """Return orbit vectors, preferring POE if available.

        Parameters
        ----------
        prefer_poe : bool, optional
            Return POE vectors when available.

        Returns
        -------
        list of S1L0OrbitStateVector
        """
        if prefer_poe and self.has_poe_orbit:
            return self._poe_vectors
        return self._annotation_vectors

    def get_merged_vectors(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[S1L0OrbitStateVector]:
        """Get merged orbit vectors covering a time range.

        If POE data covers the entire range it is preferred;
        otherwise annotation data is used.

        Parameters
        ----------
        start_time, end_time : datetime, optional
            Required coverage range.

        Returns
        -------
        list of S1L0OrbitStateVector
        """
        if (
            self.has_poe_orbit and start_time and end_time
        ):
            if (
                self._poe_info
                and self._poe_info.covers_time(start_time)
                and self._poe_info.covers_time(end_time)
            ):
                return self._poe_vectors

        if self.has_annotation_orbit:
            return self._annotation_vectors
        if self.has_poe_orbit:
            return self._poe_vectors
        return []

    def create_interpolator(
        self,
        reference_time: datetime,
        prefer_poe: bool = True,
    ) -> OrbitInterpolator:
        """Create an :class:`OrbitInterpolator` from loaded data.

        Parameters
        ----------
        reference_time : datetime
            Reference time for interpolation.
        prefer_poe : bool, optional
            Prefer POE data when available.

        Returns
        -------
        OrbitInterpolator
        """
        vectors = self.get_vectors(prefer_poe=prefer_poe)
        return OrbitInterpolator(vectors, reference_time)


# =============================================================================
# Convenience functions
# =============================================================================



# =============================================================================
# Orbit file download helpers
# =============================================================================


def _parse_eof_validity(filename: str) -> Optional[Tuple[datetime, datetime]]:
    """Parse validity start/stop from an EOF filename.

    Parameters
    ----------
    filename : str
        EOF filename, e.g.
        ``S1A_OPER_AUX_POEORB_OPOD_..._V20251207T225942_20251209T005942.EOF``

    Returns
    -------
    tuple of (datetime, datetime) or None
        ``(validity_start, validity_stop)`` in UTC, or ``None`` if the
        filename does not match the expected pattern.
    """
    m = _EOF_VALIDITY_RE.search(filename)
    if m is None:
        return None
    fmt = "%Y%m%dT%H%M%S"
    try:
        start = datetime.strptime(m.group(1), fmt)
        stop = datetime.strptime(m.group(2), fmt)
        return start, stop
    except ValueError:
        return None


def _list_remote_eof_files(
    product_type: str,
    mission: str,
    year: int,
    month: int,
) -> List[Tuple[str, str]]:
    """List EOF filenames and download URLs for a given year/month.

    Fetches the Apache directory listing from ESA's step.esa.int
    auxiliary data server.  No authentication is required.

    Parameters
    ----------
    product_type : str
        ``"POEORB"`` or ``"RESORB"``.
    mission : str
        Mission identifier (``"S1A"``, ``"S1B"``, ``"S1C"``, ``"S1D"``).
    year, month : int
        Year and month of the directory to list.

    Returns
    -------
    list of (str, str)
        ``[(filename, download_url), ...]``

    Raises
    ------
    RuntimeError
        If ``requests`` is not installed.
    OSError
        If the HTTP request fails (non-200 status).
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "The 'requests' package is required for orbit download. "
            "Install it with: pip install requests"
        )

    url = (
        f"{_ORBIT_BASE_URL}/{product_type}/{mission}/"
        f"{year}/{month:02d}/"
    )
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise OSError(
            f"Failed to list {product_type} directory for "
            f"{mission}/{year}/{month:02d}: "
            f"HTTP {resp.status_code} from {url}"
        )

    filenames = _HREF_RE.findall(resp.text)
    results = []
    for name in filenames:
        # href may be just the filename or a full URL path
        if name.startswith("http"):
            full_url = name
            fname = name.split("/")[-1]
        else:
            # strip any leading path components
            fname = name.split("/")[-1]
            full_url = url + fname
        if fname.upper().endswith(".EOF"):
            results.append((fname, full_url))
    return results


def _find_covering_eof(
    file_list: List[Tuple[str, str]],
    target_time: datetime,
) -> Optional[Tuple[str, str]]:
    """Find the EOF entry whose validity window covers *target_time*.

    Parameters
    ----------
    file_list : list of (str, str)
        Filename / URL pairs from :func:`_list_remote_eof_files`.
    target_time : datetime
        UTC time to cover.

    Returns
    -------
    tuple of (str, str) or None
        ``(filename, url)`` of the first matching entry, or ``None``.
    """
    for fname, url in file_list:
        window = _parse_eof_validity(fname)
        if window is None:
            continue
        vstart, vstop = window
        if vstart <= target_time <= vstop:
            return fname, url
    return None


def download_orbit_file(
    mission: str,
    sensing_start: datetime,
    cache_dir: Union[str, Path],
    product_type: str = "POEORB",
) -> Path:
    """Download an EOF orbit file covering *sensing_start*.

    Checks the local *cache_dir* first; downloads only when no
    matching cached file exists.

    POEORB files span ~2 days and are published ~21 days after
    the acquisition.  RESORB files span ~6 hours and are available
    ~3 hours post-acquisition.  The validity window is encoded in
    the EOF filename, so only the directory listing needs to be
    fetched to locate the right file.

    Both the current month's directory and, when the sensing date
    falls in the first half of the month, the previous month's
    directory are searched.

    Parameters
    ----------
    mission : str
        ``"S1A"``, ``"S1B"``, ``"S1C"``, or ``"S1D"``.
    sensing_start : datetime
        UTC acquisition start time.
    cache_dir : str or Path
        Directory to store downloaded files.
    product_type : str, optional
        ``"POEORB"`` (default) or ``"RESORB"``.

    Returns
    -------
    Path
        Path to the downloaded (or cached) ``.EOF`` file.

    Raises
    ------
    RuntimeError
        If no covering file is found or the download fails.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache first — scan all cached .EOF files for coverage
    for cached in sorted(cache_dir.glob(f"{mission}*_{product_type}*.EOF")):
        window = _parse_eof_validity(cached.name)
        if window and window[0] <= sensing_start <= window[1]:
            logger.info(
                "Using cached %s file: %s", product_type, cached.name,
            )
            return cached
    # Also accept files with legacy naming (no product_type in name)
    for cached in sorted(cache_dir.glob(f"{mission}*.EOF")):
        window = _parse_eof_validity(cached.name)
        if window and window[0] <= sensing_start <= window[1]:
            logger.info(
                "Using cached orbit file: %s", cached.name,
            )
            return cached

    # Build list of (year, month) directories to search.
    # POEORB files span ~2 days so a Dec-09 acquisition could be in
    # the Dec directory.  Search the same month and (if day <= 5)
    # the previous month as well.
    year, month, day = sensing_start.year, sensing_start.month, sensing_start.day
    month_dirs = [(year, month)]
    if day <= 5:
        prev = sensing_start - timedelta(days=5)
        month_dirs.append((prev.year, prev.month))

    mission_upper = mission.upper()

    for yr, mo in month_dirs:
        try:
            file_list = _list_remote_eof_files(
                product_type, mission_upper, yr, mo,
            )
        except OSError as exc:
            logger.debug(
                "Could not list %s/%d/%02d: %s", product_type, yr, mo, exc,
            )
            continue

        entry = _find_covering_eof(file_list, sensing_start)
        if entry is None:
            continue

        fname, url = entry
        dest = cache_dir / fname
        if dest.exists():
            logger.info(
                "Using cached orbit file: %s", fname,
            )
            return dest

        logger.info(
            "Downloading %s orbit: %s", product_type, fname,
        )
        try:
            import requests  # noqa: PLC0415
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
        except Exception as exc:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download {fname}: {exc}"
            ) from exc

        logger.info(
            "Downloaded %s (%d bytes) to %s",
            fname, dest.stat().st_size, cache_dir,
        )
        return dest

    raise RuntimeError(
        f"No {product_type} file found covering {sensing_start.isoformat()} "
        f"for mission {mission_upper}. "
        f"Try product_type='RESORB' if the acquisition is recent "
        f"(< 21 days ago), or pass orbit_source='file' with a manual "
        f"orbit_file= path."
    )


# =============================================================================
# Orbit resolver — three-tier strategy
# =============================================================================


class OrbitResolver:
    """Resolve an orbit for a Sentinel-1 acquisition from one of three tiers.

    The resolution strategy is controlled by *orbit_source*:

    ``"auto"``
        Download POEORB → RESORB; if both fail (e.g. network down or
        acquisition too recent), fall back to annotation vectors with a
        precision warning.
    ``"download"``
        Same download order as ``"auto"`` but **raises** if both ESA
        product types fail — no annotation fallback.
    ``"file"``
        Load *orbit_file*; raises ``ValueError`` if it is ``None`` or
        does not exist.
    ``"annotation"``
        Use the sub-commutated state vectors embedded in the annotation
        XML.  Always available; position accuracy ~10 m.

    Parameters
    ----------
    orbit_source : str, optional
        One of ``"auto"``, ``"download"``, ``"file"``, ``"annotation"``.
        Default ``"auto"``.
    orbit_file : str or Path, optional
        Path to a local ``.EOF`` file.  Required when
        ``orbit_source="file"``; ignored otherwise.
    orbit_cache_dir : str or Path, optional
        Directory for caching downloaded EOF files.  Defaults to
        ``~/.grdl/orbits``.

    Examples
    --------
    >>> resolver = OrbitResolver(orbit_source='auto')
    >>> interp = resolver.resolve(meta)

    >>> resolver = OrbitResolver(orbit_source='file',
    ...                          orbit_file='/data/orbits/S1A_OPER_AUX_POEORB_*.EOF')
    >>> interp = resolver.resolve(meta)
    """

    def __init__(
        self,
        orbit_source: str = ORBIT_SOURCE_AUTO,
        orbit_file: Optional[Union[str, Path]] = None,
        orbit_cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if orbit_source not in _VALID_ORBIT_SOURCES:
            raise ValueError(
                f"Invalid orbit_source {orbit_source!r}. "
                f"Choose from: {sorted(_VALID_ORBIT_SOURCES)}"
            )
        self.orbit_source = orbit_source
        self.orbit_file = Path(orbit_file) if orbit_file else None
        self.orbit_cache_dir = (
            Path(orbit_cache_dir)
            if orbit_cache_dir
            else Path.home() / ".grdl" / "orbits"
        )
        # Set by resolve() — records which tier was actually used.
        # One of: "poeorb", "resorb", "file", "annotation".
        self.orbit_source_used: Optional[str] = None

    def resolve(
        self,
        annotation_vectors: List[S1L0OrbitStateVector],
        sensing_start: datetime,
        reference_time: datetime,
        mission: str = "S1A",
    ) -> "OrbitInterpolator":
        """Resolve orbit vectors and return an interpolator.

        Parameters
        ----------
        annotation_vectors : list of S1L0OrbitStateVector
            Orbit vectors from annotation XML.  Used when
            ``orbit_source`` is ``"annotation"`` or as the fallback
            in ``"auto"`` mode.
        sensing_start : datetime
            UTC acquisition start time (used to locate the correct
            EOF file by its validity window).
        reference_time : datetime
            Reference epoch for the returned
            :class:`OrbitInterpolator`.
        mission : str, optional
            Satellite identifier: ``"S1A"``, ``"S1B"``, ``"S1C"``,
            or ``"S1D"``.  Default ``"S1A"``.

        Returns
        -------
        OrbitInterpolator
        """
        loader = OrbitLoader()
        if annotation_vectors:
            loader.load_annotation_vectors(annotation_vectors)

        if self.orbit_source == ORBIT_SOURCE_ANNOTATION:
            if not loader.has_annotation_orbit:
                raise ValueError(
                    "orbit_source='annotation' was requested but no "
                    "annotation orbit vectors are available."
                )
            logger.info(
                "Using annotation orbit vectors (%d vectors, ~10 m accuracy).",
                len(annotation_vectors),
            )
            warnings.warn(
                "Annotation orbit vectors have ~10 m position accuracy. "
                "For absolute geolocation or image formation to < 1 m, "
                "use orbit_source='download' or provide an EOF file "
                "via orbit_source='file'.",
                UserWarning,
                stacklevel=2,
            )
            self.orbit_source_used = "annotation"
            return loader.create_interpolator(
                reference_time=reference_time, prefer_poe=False,
            )

        if self.orbit_source == ORBIT_SOURCE_FILE:
            if self.orbit_file is None:
                raise ValueError(
                    "orbit_source='file' requires orbit_file= to be set."
                )
            if not self.orbit_file.exists():
                raise FileNotFoundError(
                    f"Orbit file not found: {self.orbit_file}"
                )
            loader.load_poe_file(self.orbit_file)
            logger.info(
                "Loaded orbit from local file: %s", self.orbit_file.name,
            )
            self.orbit_source_used = "file"
            return loader.create_interpolator(
                reference_time=reference_time, prefer_poe=True,
            )

        # "download" or "auto" — attempt POEORB then RESORB
        eof_path: Optional[Path] = None
        last_exc: Optional[Exception] = None
        used_product_type: Optional[str] = None
        for product_type in ("POEORB", "RESORB"):
            try:
                eof_path = download_orbit_file(
                    mission=mission,
                    sensing_start=sensing_start,
                    cache_dir=self.orbit_cache_dir,
                    product_type=product_type,
                )
                used_product_type = product_type.lower()
                logger.info(
                    "Using %s orbit: %s", product_type, eof_path.name,
                )
                break
            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "%s download failed: %s", product_type, exc,
                )

        if eof_path is not None:
            loader.load_poe_file(eof_path)
            self.orbit_source_used = used_product_type
            return loader.create_interpolator(
                reference_time=reference_time, prefer_poe=True,
            )

        # Download failed
        if self.orbit_source == ORBIT_SOURCE_DOWNLOAD:
            raise RuntimeError(
                "orbit_source='download' failed for both POEORB and "
                f"RESORB: {last_exc}. "
                "Check network connectivity or use orbit_source='file' "
                "with a manually downloaded .EOF."
            ) from last_exc

        # "auto" — fall back to annotation vectors
        if not loader.has_annotation_orbit:
            raise RuntimeError(
                "Orbit download failed and no annotation vectors are "
                f"available: {last_exc}"
            ) from last_exc

        warnings.warn(
            f"Orbit download failed ({last_exc}). "
            "Falling back to annotation orbit vectors "
            f"({len(annotation_vectors)} vectors, ~10 m accuracy). "
            "For phase-coherent image formation you should provide a "
            "precise orbit — set orbit_source='download' or pass a "
            "local orbit_file= with orbit_source='file'.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "Orbit download failed; using annotation fallback: %s", last_exc,
        )
        self.orbit_source_used = "annotation"
        return loader.create_interpolator(
            reference_time=reference_time, prefer_poe=False,
        )


# =============================================================================
# Convenience functions
# =============================================================================


def parse_poe_file(    poe_path: Union[str, Path],
) -> List[S1L0OrbitStateVector]:
    """Parse a POE file and return orbit state vectors.

    Parameters
    ----------
    poe_path : str or Path
        Path to ``.EOF`` file.

    Returns
    -------
    list of S1L0OrbitStateVector
    """
    parser = POEParser(poe_path)
    return parser.parse()


def find_poe_file_for_time(
    directory: Union[str, Path],
    target_time: datetime,
    mission: str = "S1A",
) -> Optional[Path]:
    """Find a POE file covering a specific time.

    Parameters
    ----------
    directory : str or Path
        Directory to search.
    target_time : datetime
        Target UTC time.
    mission : str, optional
        Mission identifier (``'S1A'``, ``'S1B'``).

    Returns
    -------
    Path or None
        Path to matching POE file, or ``None`` if not found.
    """
    directory = Path(directory)
    if not directory.exists():
        return None

    eof_files = list(directory.glob(f"{mission}*.EOF"))
    if not eof_files:
        return None

    for eof_file in eof_files:
        try:
            parser = POEParser(eof_file)
            parser.parse()
            if parser.info.covers_time(target_time):
                return eof_file
        except Exception as e:
            logger.warning(
                f"Failed to parse {eof_file.name}: {e}"
            )
            continue
    return None


def interpolate_orbit(
    vectors: List[S1L0OrbitStateVector],
    reference_time: datetime,
    times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate orbit at given times (convenience wrapper).

    Parameters
    ----------
    vectors : list of S1L0OrbitStateVector
        Orbit state vectors.
    reference_time : datetime
        Reference time.
    times : numpy.ndarray
        Times in seconds relative to ``reference_time``.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(positions, velocities)``.
    """
    interp = OrbitInterpolator(vectors, reference_time)
    return interp.interpolate(times)
