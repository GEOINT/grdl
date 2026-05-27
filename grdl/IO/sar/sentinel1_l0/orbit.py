# -*- coding: utf-8 -*-
"""
Sentinel-1 L0 Orbit and Attitude Handling.

Consolidated orbit and attitude processing for Sentinel-1 data:

- Loading orbit data from annotation XML
- Loading POE (Precise Orbit Ephemerides) files
- Merging annotation and POE orbit data
- Orbit state vector interpolation (scipy ``CubicSpline``)
- Attitude interpolation

POE Files
---------
ESA provides POE files with orbit state vectors at 10-second
intervals and position accuracy < 5 cm.  These are more precise
than the annotation orbit data and should be preferred when
available.

File naming: ``S1A_OPER_AUX_POEORB_OPOD_\
YYYYMMDD_VYYYYMMDD_YYYYMMDD.EOF``

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
2026-05-22
"""

# Standard library
import logging
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Third-party
import numpy as np
from scipy.interpolate import CubicSpline

# GRDL internal
from grdl.IO.models.sentinel1_l0 import (
    S1L0AttitudeRecord,
    S1L0OrbitStateVector,
)

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _requests = None  # type: ignore[assignment]
    _HAS_REQUESTS = False

logger = logging.getLogger(__name__)

# Remote POE sources (no authentication required)
ASF_ORBIT_API_URL = "https://s1-orbits.asf.alaska.edu/scene"
ESA_POE_BASE_URL = (
    "https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB"
)
POE_DOWNLOAD_TIMEOUT = 60  # seconds


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

        self._px = CubicSpline(times, px)
        self._py = CubicSpline(times, py)
        self._pz = CubicSpline(times, pz)
        self._vx = CubicSpline(times, vx)
        self._vy = CubicSpline(times, vy)
        self._vz = CubicSpline(times, vz)

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
        vx = self._vx(times)
        vy = self._vy(times)
        vz = self._vz(times)

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
        scene_name: Optional[str] = None,
        download: bool = True,
    ) -> bool:
        """Find and load a POE file covering a specific time.

        Searches ``directory`` for a local ``.EOF`` file first.
        If none is found and ``download`` is ``True``, attempts
        to download from the ASF orbit API (backed by the
        Copernicus Data Space mirror on AWS).

        Parameters
        ----------
        directory : str or Path
            Directory to search (and download into).
        target_time : datetime
            Target UTC time.
        mission : str, optional
            Mission identifier (``'S1A'``, ``'S1B'``, ``'S1C'``).
        scene_name : str, optional
            Sentinel-1 scene name (e.g., SAFE directory name
            without ``.SAFE``).  Enables download via the ASF
            orbit API.
        download : bool, optional
            Attempt download if no local file is found.

        Returns
        -------
        bool
            ``True`` if a covering POE file was loaded.
        """
        # 1. Local search
        poe_file = find_poe_file_for_time(
            directory, target_time, mission
        )
        if poe_file:
            self.load_poe_file(poe_file)
            return True

        if not download:
            return False

        # 2. Download from ASF / ESA
        downloaded = download_poe(
            target_time=target_time,
            mission=mission,
            scene_name=scene_name,
            output_dir=directory,
        )
        if downloaded:
            self.load_poe_file(downloaded)
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


def parse_poe_file(
    poe_path: Union[str, Path],
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


# =============================================================================
# POE download
# =============================================================================

_POE_FILENAME_RE = re.compile(
    r"(S1[A-D])_OPER_AUX_POEORB_\S+?_V"
    r"(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF"
)


def _parse_poe_validity_from_filename(
    filename: str,
) -> Optional[Tuple[str, datetime, datetime]]:
    """Extract mission and validity window from a POE filename.

    Parameters
    ----------
    filename : str
        POE filename (e.g.,
        ``S1A_OPER_AUX_POEORB_..._V<start>_<end>.EOF``).

    Returns
    -------
    (str, datetime, datetime) or None
        ``(mission, validity_start, validity_stop)``, or ``None``
        if the filename does not match.
    """
    m = _POE_FILENAME_RE.search(filename)
    if m is None:
        return None
    try:
        start = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S")
        end = datetime.strptime(m.group(3), "%Y%m%dT%H%M%S")
        return m.group(1), start, end
    except ValueError:
        return None


def _download_poe_asf(
    scene_name: str,
    output_dir: Path,
    timeout: int = POE_DOWNLOAD_TIMEOUT,
) -> Optional[Path]:
    """Download a POE file via the ASF orbit API.

    The ASF API accepts a Sentinel-1 scene name and redirects to
    the best available orbit file (POEORB preferred, RESORB
    fallback) hosted on S3.

    Parameters
    ----------
    scene_name : str
        Scene name (SAFE directory name without ``.SAFE``).
    output_dir : Path
        Directory to save the downloaded file.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    Path or None
        Path to the downloaded ``.EOF`` file, or ``None`` on
        failure.
    """
    url = f"{ASF_ORBIT_API_URL}/{scene_name}"
    logger.info("Querying ASF orbit API: %s", url)
    try:
        resp = _requests.get(
            url, timeout=timeout, allow_redirects=True,
        )
        resp.raise_for_status()
    except _requests.RequestException as e:
        logger.warning("ASF orbit API request failed: %s", e)
        return None

    # Derive filename from Content-Disposition or URL.
    cd = resp.headers.get("Content-Disposition", "")
    fname_match = re.search(r'filename="?([^";]+)"?', cd)
    if fname_match:
        fname = fname_match.group(1)
    else:
        fname = resp.url.rsplit("/", 1)[-1].split("?")[0]

    if not fname.upper().endswith(".EOF"):
        logger.warning(
            "Unexpected orbit filename: %s", fname,
        )
        return None

    local_path = output_dir / fname
    if local_path.exists():
        logger.info("POE file already cached: %s", local_path)
        return local_path

    local_path.write_bytes(resp.content)
    logger.info("Downloaded POE file: %s", local_path)
    return local_path


def _download_poe_esa(
    mission: str,
    target_time: datetime,
    output_dir: Path,
    timeout: int = POE_DOWNLOAD_TIMEOUT,
) -> Optional[Path]:
    """Download a POE file from the ESA step.esa.int archive.

    Falls back to directory-listing scrape of the ESA HTTP archive.

    Parameters
    ----------
    mission : str
        Mission prefix (``'S1A'``, ``'S1B'``, ``'S1C'``).
    target_time : datetime
        Target UTC time that must fall within the file's validity.
    output_dir : Path
        Directory to save the downloaded file.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    Path or None
        Path to the downloaded ``.EOF`` file, or ``None``.
    """
    prefix = mission[:3].upper()
    year = target_time.strftime("%Y")
    month = target_time.strftime("%m")
    dir_url = f"{ESA_POE_BASE_URL}/{prefix}/{year}/{month}/"

    logger.info("Listing ESA orbit directory: %s", dir_url)
    try:
        resp = _requests.get(dir_url, timeout=timeout)
        resp.raise_for_status()
    except _requests.RequestException as e:
        logger.warning(
            "ESA orbit directory request failed: %s", e,
        )
        return None

    # Scrape .EOF filenames from Apache directory listing.
    filenames = re.findall(
        rf"{prefix}_OPER_AUX_POEORB_\S+?\.EOF", resp.text,
    )
    if not filenames:
        logger.warning("No POE files found at %s", dir_url)
        return None

    for fname in sorted(set(filenames)):
        parsed = _parse_poe_validity_from_filename(fname)
        if parsed is None:
            continue
        _, v_start, v_end = parsed
        if v_start <= target_time <= v_end:
            local_path = output_dir / fname
            if local_path.exists():
                logger.info(
                    "POE file already cached: %s", local_path,
                )
                return local_path

            file_url = f"{dir_url}{fname}"
            logger.info("Downloading POE file: %s", fname)
            try:
                file_resp = _requests.get(
                    file_url, timeout=timeout * 2,
                )
                file_resp.raise_for_status()
            except _requests.RequestException as e:
                logger.warning(
                    "Failed to download %s: %s", fname, e,
                )
                return None
            local_path.write_bytes(file_resp.content)
            logger.info(
                "Downloaded POE file: %s", local_path,
            )
            return local_path

    logger.warning(
        "No POE file covering %s at %s",
        target_time.isoformat(), dir_url,
    )
    return None


def download_poe(
    target_time: Optional[datetime] = None,
    mission: str = "S1A",
    scene_name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    timeout: int = POE_DOWNLOAD_TIMEOUT,
) -> Optional[Path]:
    """Download a POE orbit file from a remote archive.

    Tries the ASF orbit API first (fast, single redirect), then
    falls back to ESA step.esa.int (directory listing scrape).
    Both sources are public and require no authentication.

    Parameters
    ----------
    target_time : datetime, optional
        UTC time that must fall within the orbit file's validity.
        Required for the ESA fallback; not needed when
        ``scene_name`` is provided (ASF path).
    mission : str, optional
        Mission prefix (``'S1A'``, ``'S1B'``, ``'S1C'``).
    scene_name : str, optional
        Sentinel-1 scene name (enables ASF API lookup).  This is
        the SAFE directory name without the ``.SAFE`` suffix.
    output_dir : str or Path, optional
        Directory to save the downloaded file.  Defaults to the
        current working directory.
    timeout : int, optional
        HTTP timeout in seconds.

    Returns
    -------
    Path or None
        Path to the downloaded ``.EOF`` file, or ``None`` if no
        file could be obtained.

    Raises
    ------
    grdl.exceptions.DependencyError
        If ``requests`` is not installed.
    """
    if not _HAS_REQUESTS:
        from grdl.exceptions import DependencyError
        raise DependencyError(
            "The 'requests' package is required to download "
            "POE orbit files.  Install with: "
            "pip install requests"
        )

    out = Path(output_dir) if output_dir else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)

    # 1. ASF orbit API (preferred — single redirect)
    if scene_name:
        result = _download_poe_asf(
            scene_name, out, timeout=timeout,
        )
        if result:
            return result
        logger.info(
            "ASF lookup failed; falling back to ESA archive",
        )

    # 2. ESA step.esa.int (requires target_time)
    if target_time is not None:
        result = _download_poe_esa(
            mission, target_time, out, timeout=timeout,
        )
        if result:
            return result

    target_time_str = (
        target_time.isoformat()
        if target_time is not None
        else "unknown time"
    )
    logger.warning(
        "Could not download POE orbit file for %s (%s)",
        target_time_str, mission,
    )
    return None


def ensure_poe_available(
    target_time: datetime,
    mission: str = "S1A",
    scene_name: Optional[str] = None,
    search_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
) -> Path:
    """Ensure a POE file is available, downloading if needed.

    Parameters
    ----------
    target_time : datetime
        UTC time that must fall within the file's validity.
    mission : str, optional
        Mission prefix.
    scene_name : str, optional
        Sentinel-1 scene name (enables ASF API lookup).
    search_dir : str or Path, optional
        Directory to search for local files (and save downloads).
    download : bool, optional
        Whether to attempt remote download if no local file
        exists.

    Returns
    -------
    Path
        Path to the ``.EOF`` file.

    Raises
    ------
    FileNotFoundError
        If no file could be found or downloaded.  The error
        message includes manual-download instructions.
    """
    directory = Path(search_dir) if search_dir else Path.cwd()

    # Check local
    local = find_poe_file_for_time(
        directory, target_time, mission,
    )
    if local:
        return local

    # Try download
    if download and _HAS_REQUESTS:
        downloaded = download_poe(
            target_time=target_time,
            mission=mission,
            scene_name=scene_name,
            output_dir=directory,
        )
        if downloaded:
            return downloaded

    # Build helpful error with alternative download sources.
    prefix = mission[:3].upper()
    year = target_time.strftime("%Y")
    month = target_time.strftime("%m")
    esa_url = (
        f"{ESA_POE_BASE_URL}/{prefix}/{year}/{month}/"
    )
    aws_url = (
        "https://s1-orbits.s3.us-west-2.amazonaws.com/"
        f"index.html#{prefix}/"
    )
    raise FileNotFoundError(
        f"No POE orbit file found covering "
        f"{target_time.isoformat()}.\n"
        f"Searched: {directory}\n\n"
        f"Download manually from one of:\n"
        f"  ESA:  {esa_url}\n"
        f"  AWS:  {aws_url}\n\n"
        f"Then either:\n"
        f"  • Place the .EOF file in: {directory}\n"
        f"  • Pass the file path via poe_path= parameter"
    )
