# -*- coding: utf-8 -*-
"""
NISAR Reader - NASA NISAR L-band/S-band SAR product reader.

Reads NISAR RSLC (Range Doppler SLC) and GSLC (Geocoded SLC)
products from HDF5 files.  Each reader instance opens one
frequency + polarization combination, exposing the complex-valued
imagery via efficient h5py hyperslab reads.

Product type (RSLC/GSLC) and radar band (LSAR/SSAR) are
auto-detected from the HDF5 structure.  The user selects which
frequency sub-band (A/B) and polarization (HH/HV/VH/VV) to read.

Dependencies
------------
h5py

Author
------
Jason Fritz
jpfritz@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-25

Modified
--------
2026-03-10
"""

# Standard library
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

# GRDL internal
from grdl.exceptions import DependencyError
from grdl.IO.base import ImageReader
from grdl.IO.models.nisar import (
    NISARMetadata,
    NISARIdentification,
    NISAROrbit,
    NISARAttitude,
    NISARSwathParameters,
    NISARGridParameters,
    NISARGeolocationGrid,
    NISARCalibration,
    NISARProcessingInfo,
)

logger = logging.getLogger(__name__)


# ===================================================================
# HDF5 scalar read helpers
# ===================================================================

def _read_scalar_str(group: "h5py.Group", name: str) -> Optional[str]:
    """Read a scalar string dataset, decoding bytes to str."""
    if name not in group:
        return None
    val = group[name][()]
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode('utf-8').strip()
    return str(val).strip()


def _read_scalar_int(group: "h5py.Group", name: str) -> Optional[int]:
    """Read a scalar integer dataset."""
    if name not in group:
        return None
    return int(group[name][()])


def _read_scalar_float(group: "h5py.Group", name: str) -> Optional[float]:
    """Read a scalar float dataset."""
    if name not in group:
        return None
    return float(group[name][()])


def _read_scalar_bool(group: "h5py.Group", name: str) -> Optional[bool]:
    """Read a scalar boolean dataset (handles byte-string 'True'/'False')."""
    if name not in group:
        return None
    val = group[name][()]
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode('utf-8').strip().lower() in ('true', '1', 'yes')
    return bool(val)


def _read_array(group: "h5py.Group", name: str) -> Optional[np.ndarray]:
    """Read a dataset as a numpy array, returning None if absent."""
    if name not in group:
        return None
    return group[name][()]


def _read_string_list(
    group: "h5py.Group", name: str,
) -> Optional[List[str]]:
    """Read a 1-D byte-string dataset as a list of Python strings."""
    if name not in group:
        return None
    raw = group[name][()]
    if isinstance(raw, (bytes, np.bytes_)):
        return [raw.decode('utf-8').strip()]
    return [
        v.decode('utf-8').strip() if isinstance(v, (bytes, np.bytes_))
        else str(v).strip()
        for v in raw
    ]


def _read_time_reference(group: "h5py.Group", name: str) -> Optional[str]:
    """Read the 'units' attribute from a time dataset as a reference epoch."""
    if name not in group:
        return None
    ds = group[name]
    if 'units' in ds.attrs:
        units = ds.attrs['units']
        if isinstance(units, (bytes, np.bytes_)):
            return units.decode('utf-8')
        return str(units)
    return None


# ===================================================================
# NISARReader
# ===================================================================

class NISARReader(ImageReader):
    """Read NISAR RSLC or GSLC products from HDF5 files.

    Each instance opens one frequency + polarization combination.
    Product type (RSLC/GSLC) and radar band (LSAR/SSAR) are
    auto-detected from the HDF5 group hierarchy.

    Parameters
    ----------
    filepath : str or Path
        Path to a NISAR HDF5 file (``.h5``).
    frequency : str, optional
        Frequency sub-band to read (``'A'`` or ``'B'``).  If ``None``,
        the first available frequency is selected.
    polarization : str, optional
        Polarization channel to read (``'HH'``, ``'HV'``, ``'VH'``,
        ``'VV'``).  If ``None``, the first available polarization is
        selected.

    Attributes
    ----------
    metadata : NISARMetadata
        Complete typed metadata with identification, orbit, attitude,
        swath/grid parameters, geolocation grid, and calibration.

    Raises
    ------
    ImportError
        If h5py is not installed.
    FileNotFoundError
        If the HDF5 file does not exist.
    ValueError
        If the file is not a valid NISAR product, or the requested
        frequency/polarization is not available.

    Examples
    --------
    >>> from grdl.IO.sar.nisar import NISARReader
    >>> with NISARReader('NISAR_L1_RSLC_example.h5') as reader:
    ...     print(reader.metadata.product_type)  # 'RSLC'
    ...     chip = reader.read_chip(0, 1000, 0, 1000)
    ...     print(chip.shape, chip.dtype)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        frequency: Optional[str] = None,
        polarization: Optional[str] = None,
    ) -> None:
        if not _HAS_H5PY:
            raise DependencyError(
                "Reading NISAR products requires h5py. "
                "Install with: pip install h5py"
            )

        self._requested_frequency = frequency
        self._requested_polarization = polarization

        # Will be set in _load_metadata
        self._file: Optional["h5py.File"] = None
        self._radar_band: str = ''
        self._product_type: str = ''
        self._base_path: str = ''
        self._frequency: str = ''
        self._polarization: str = ''
        self._available_frequencies: List[str] = []
        self._available_polarizations: List[str] = []
        self._imagery_path: str = ''

        super().__init__(filepath)

    # ---------------------------------------------------------------
    # Auto-detection
    # ---------------------------------------------------------------

    def _detect_radar_band(self) -> str:
        """Detect LSAR or SSAR from the ``science/`` group."""
        science = self._file['science']
        for band in ('LSAR', 'SSAR'):
            if band in science:
                return band
        raise ValueError(
            f"No LSAR or SSAR group found in {self.filepath}. "
            "File may not be a valid NISAR product."
        )

    def _detect_product_type(self) -> str:
        """Detect RSLC or GSLC from the radar band group."""
        band_group = self._file[f'science/{self._radar_band}']
        for ptype in ('RSLC', 'GSLC'):
            if ptype in band_group:
                return ptype
        raise ValueError(
            f"No RSLC or GSLC group found under "
            f"science/{self._radar_band}/. "
            "File may not be a valid NISAR product."
        )

    def _discover_frequencies(self) -> List[str]:
        """Enumerate available frequency sub-bands (A, B, ...)."""
        container = 'swaths' if self._product_type == 'RSLC' else 'grids'
        container_path = f'{self._base_path}/{container}'
        group = self._file[container_path]
        freqs = []
        for key in group.keys():
            if key.startswith('frequency'):
                freqs.append(key.replace('frequency', ''))
        if not freqs:
            raise ValueError(
                f"No frequency groups found in {container_path}"
            )
        return sorted(freqs)

    def _resolve_frequency(self) -> str:
        """Validate and return the active frequency."""
        if self._requested_frequency is not None:
            freq = self._requested_frequency.upper()
            if freq not in self._available_frequencies:
                raise ValueError(
                    f"Frequency '{self._requested_frequency}' not available. "
                    f"Available: {self._available_frequencies}"
                )
            return freq
        return self._available_frequencies[0]

    def _discover_polarizations(self) -> List[str]:
        """Read available polarizations for the active frequency."""
        container = 'swaths' if self._product_type == 'RSLC' else 'grids'
        freq_path = (
            f'{self._base_path}/{container}/frequency{self._frequency}'
        )
        freq_group = self._file[freq_path]

        # Prefer the explicit list dataset
        pols = _read_string_list(freq_group, 'listOfPolarizations')
        if pols:
            return pols

        # Fallback: enumerate dataset names matching polarization codes
        logger.warning(
            "NISAR listOfPolarizations not found; falling back to "
            "dataset enumeration for frequency %s",
            self._frequency,
        )
        pol_names = {'HH', 'HV', 'VH', 'VV'}
        pols = sorted(
            k for k in freq_group.keys()
            if k in pol_names and isinstance(freq_group[k], h5py.Dataset)
        )
        if not pols:
            raise ValueError(
                f"No polarization datasets found in {freq_path}"
            )
        return pols

    def _resolve_polarization(self) -> str:
        """Validate and return the active polarization."""
        if self._requested_polarization is not None:
            pol = self._requested_polarization.upper()
            if pol not in self._available_polarizations:
                raise ValueError(
                    f"Polarization '{self._requested_polarization}' not "
                    f"available. Available: {self._available_polarizations}"
                )
            return pol
        return self._available_polarizations[0]

    def _resolve_imagery_path(self) -> str:
        """Build the full HDF5 path to the imagery dataset."""
        container = 'swaths' if self._product_type == 'RSLC' else 'grids'
        return (
            f'{self._base_path}/{container}/'
            f'frequency{self._frequency}/{self._polarization}'
        )

    # ---------------------------------------------------------------
    # Metadata extraction
    # ---------------------------------------------------------------

    def _extract_identification(self) -> Optional[NISARIdentification]:
        """Extract identification metadata."""
        id_path = f'science/{self._radar_band}/identification'
        if id_path not in self._file:
            return None
        grp = self._file[id_path]
        return NISARIdentification(
            mission_id=_read_scalar_str(grp, 'missionId'),
            product_type=_read_scalar_str(grp, 'productType'),
            radar_band=_read_scalar_str(grp, 'radarBand'),
            look_direction=_read_scalar_str(grp, 'lookDirection'),
            orbit_pass_direction=_read_scalar_str(grp, 'orbitPassDirection'),
            absolute_orbit_number=_read_scalar_int(
                grp, 'absoluteOrbitNumber'
            ),
            track_number=_read_scalar_int(grp, 'trackNumber'),
            frame_number=_read_scalar_int(grp, 'frameNumber'),
            is_geocoded=_read_scalar_bool(grp, 'isGeocoded'),
            processing_type=_read_scalar_str(grp, 'processingType'),
            zero_doppler_start_time=_read_scalar_str(
                grp, 'zeroDopplerStartTime'
            ),
            zero_doppler_end_time=_read_scalar_str(
                grp, 'zeroDopplerEndTime'
            ),
            bounding_polygon=_read_scalar_str(grp, 'boundingPolygon'),
            granule_id=_read_scalar_str(grp, 'granuleId'),
            instrument_name=_read_scalar_str(grp, 'instrumentName'),
            processing_date_time=_read_scalar_str(
                grp, 'processingDateTime'
            ),
            product_version=_read_scalar_str(grp, 'productVersion'),
        )

    def _extract_orbit(self) -> Optional[NISAROrbit]:
        """Extract orbit state vectors."""
        orbit_path = f'{self._base_path}/metadata/orbit'
        if orbit_path not in self._file:
            return None
        grp = self._file[orbit_path]
        return NISAROrbit(
            time=_read_array(grp, 'time'),
            position=_read_array(grp, 'position'),
            velocity=_read_array(grp, 'velocity'),
            reference_epoch=_read_time_reference(grp, 'time'),
            orbit_type=_read_scalar_str(grp, 'orbitType'),
            interp_method=_read_scalar_str(grp, 'interpMethod'),
        )

    def _extract_attitude(self) -> Optional[NISARAttitude]:
        """Extract spacecraft attitude data."""
        att_path = f'{self._base_path}/metadata/attitude'
        if att_path not in self._file:
            return None
        grp = self._file[att_path]
        return NISARAttitude(
            time=_read_array(grp, 'time'),
            quaternions=_read_array(grp, 'quaternions'),
            euler_angles=_read_array(grp, 'eulerAngles'),
            reference_epoch=_read_time_reference(grp, 'time'),
            attitude_type=_read_scalar_str(grp, 'attitudeType'),
        )

    def _extract_swath_parameters(self) -> Optional[NISARSwathParameters]:
        """Extract RSLC swath parameters for the active frequency."""
        freq_path = (
            f'{self._base_path}/swaths/frequency{self._frequency}'
        )
        if freq_path not in self._file:
            return None
        grp = self._file[freq_path]
        return NISARSwathParameters(
            acquired_center_frequency=_read_scalar_float(
                grp, 'acquiredCenterFrequency'
            ),
            acquired_range_bandwidth=_read_scalar_float(
                grp, 'acquiredRangeBandwidth'
            ),
            processed_center_frequency=_read_scalar_float(
                grp, 'processedCenterFrequency'
            ),
            processed_range_bandwidth=_read_scalar_float(
                grp, 'processedRangeBandwidth'
            ),
            processed_azimuth_bandwidth=_read_scalar_float(
                grp, 'processedAzimuthBandwidth'
            ),
            slant_range_spacing=_read_scalar_float(
                grp, 'slantRangeSpacing'
            ),
            nominal_acquisition_prf=_read_scalar_float(
                grp, 'nominalAcquisitionPRF'
            ),
            scene_center_along_track_spacing=_read_scalar_float(
                grp, 'sceneCenterAlongTrackSpacing'
            ),
            scene_center_ground_range_spacing=_read_scalar_float(
                grp, 'sceneCenterGroundRangeSpacing'
            ),
            polarizations=_read_string_list(grp, 'listOfPolarizations'),
            number_of_sub_swaths=_read_scalar_int(
                grp, 'numberOfSubSwaths'
            ),
            zero_doppler_time_spacing=_read_scalar_float(
                grp, 'zeroDopplerTimeSpacing'
            ),
            slant_range=_read_array(grp, 'slantRange'),
            zero_doppler_time=_read_array(grp, 'zeroDopplerTime'),
            zero_doppler_time_reference_epoch=_read_time_reference(
                grp, 'zeroDopplerTime'
            ),
        )

    def _extract_grid_parameters(self) -> Optional[NISARGridParameters]:
        """Extract GSLC grid parameters for the active frequency."""
        freq_path = (
            f'{self._base_path}/grids/frequency{self._frequency}'
        )
        if freq_path not in self._file:
            return None
        grp = self._file[freq_path]
        return NISARGridParameters(
            x_coordinates=_read_array(grp, 'xCoordinates'),
            y_coordinates=_read_array(grp, 'yCoordinates'),
            x_coordinate_spacing=_read_scalar_float(
                grp, 'xCoordinateSpacing'
            ),
            y_coordinate_spacing=_read_scalar_float(
                grp, 'yCoordinateSpacing'
            ),
            epsg=_read_scalar_int(grp, 'projection'),
            center_frequency=_read_scalar_float(grp, 'centerFrequency'),
            range_bandwidth=_read_scalar_float(grp, 'rangeBandwidth'),
            azimuth_bandwidth=_read_scalar_float(grp, 'azimuthBandwidth'),
            slant_range_spacing=_read_scalar_float(
                grp, 'slantRangeSpacing'
            ),
            polarizations=_read_string_list(grp, 'listOfPolarizations'),
        )

    def _extract_geolocation_grid(self) -> Optional[NISARGeolocationGrid]:
        """Extract RSLC geolocation grid."""
        geo_path = f'{self._base_path}/metadata/geolocationGrid'
        if geo_path not in self._file:
            return None
        grp = self._file[geo_path]
        return NISARGeolocationGrid(
            coordinate_x=_read_array(grp, 'coordinateX'),
            coordinate_y=_read_array(grp, 'coordinateY'),
            epsg=_read_scalar_int(grp, 'epsg'),
            slant_range=_read_array(grp, 'slantRange'),
            zero_doppler_time=_read_array(grp, 'zeroDopplerTime'),
            height_above_ellipsoid=_read_array(
                grp, 'heightAboveEllipsoid'
            ),
            incidence_angle=_read_array(grp, 'incidenceAngle'),
            elevation_angle=_read_array(grp, 'elevationAngle'),
        )

    def _extract_calibration(self) -> Optional[NISARCalibration]:
        """Extract radiometric calibration grids."""
        cal_path = (
            f'{self._base_path}/metadata/calibrationInformation/geometry'
        )
        if cal_path not in self._file:
            return None
        grp = self._file[cal_path]
        return NISARCalibration(
            sigma0=_read_array(grp, 'sigma0'),
            beta0=_read_array(grp, 'beta0'),
            gamma0=_read_array(grp, 'gamma0'),
        )

    def _extract_processing_info(self) -> Optional[NISARProcessingInfo]:
        """Extract processing information."""
        proc_path = f'{self._base_path}/metadata/processingInformation'
        if proc_path not in self._file:
            return None

        algo_path = f'{proc_path}/algorithms'
        software_version = None
        algorithms: Dict[str, Any] = {}
        if algo_path in self._file:
            algo_grp = self._file[algo_path]
            software_version = _read_scalar_str(
                algo_grp, 'softwareVersion'
            )
            for key in algo_grp.keys():
                if key != 'softwareVersion':
                    val = _read_scalar_str(algo_grp, key)
                    if val is not None:
                        algorithms[key] = val

        return NISARProcessingInfo(
            software_version=software_version,
            algorithms=algorithms if algorithms else None,
        )

    # ---------------------------------------------------------------
    # ImageReader interface
    # ---------------------------------------------------------------

    def _load_metadata(self) -> None:
        """Open HDF5 and extract all metadata."""
        self._file = h5py.File(str(self.filepath), 'r')

        # Auto-detect structure
        self._radar_band = self._detect_radar_band()
        self._product_type = self._detect_product_type()
        self._base_path = (
            f'science/{self._radar_band}/{self._product_type}'
        )

        # Discover and select frequency / polarization
        self._available_frequencies = self._discover_frequencies()
        self._frequency = self._resolve_frequency()
        self._available_polarizations = self._discover_polarizations()
        self._polarization = self._resolve_polarization()

        logger.debug(
            "NISAR band=%s, product=%s, frequencies=%s, polarizations=%s",
            self._radar_band,
            self._product_type,
            self._available_frequencies,
            self._available_polarizations,
        )

        # Resolve imagery dataset path
        self._imagery_path = self._resolve_imagery_path()
        ds = self._file[self._imagery_path]
        rows, cols = ds.shape
        dtype_str = str(ds.dtype)

        # Extract metadata sections
        identification = self._extract_identification()
        orbit = self._extract_orbit()
        attitude = self._extract_attitude()
        swath_params = (
            self._extract_swath_parameters()
            if self._product_type == 'RSLC' else None
        )
        grid_params = (
            self._extract_grid_parameters()
            if self._product_type == 'GSLC' else None
        )
        geolocation = (
            self._extract_geolocation_grid()
            if self._product_type == 'RSLC' else None
        )
        calibration = self._extract_calibration()
        processing = self._extract_processing_info()

        # Determine CRS
        crs = None
        if self._product_type == 'GSLC' and grid_params is not None:
            if grid_params.epsg is not None:
                crs = f'EPSG:{grid_params.epsg}'

        self.metadata = NISARMetadata(
            format=f'NISAR_{self._product_type}',
            rows=rows,
            cols=cols,
            dtype=dtype_str,
            crs=crs,
            product_type=self._product_type,
            radar_band=self._radar_band,
            frequency=self._frequency,
            polarization=self._polarization,
            available_frequencies=self._available_frequencies,
            available_polarizations=self._available_polarizations,
            identification=identification,
            orbit=orbit,
            attitude=attitude,
            swath_parameters=swath_params,
            grid_parameters=grid_params,
            geolocation_grid=geolocation,
            calibration=calibration,
            processing_info=processing,
        )

        logger.info(
            "Loaded NISAR %s %s (%d x %d) freq=%s pol=%s",
            self._product_type,
            self.filepath.name,
            rows,
            cols,
            self._frequency,
            self._polarization,
        )

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip via h5py hyperslab selection.

        Parameters
        ----------
        row_start : int
            Starting row index (inclusive).
        row_end : int
            Ending row index (exclusive).
        col_start : int
            Starting column index (inclusive).
        col_end : int
            Ending column index (exclusive).
        bands : Optional[List[int]]
            Ignored (single polarization per reader instance).

        Returns
        -------
        np.ndarray
            Image chip with shape ``(rows, cols)`` and the native dtype
            of the dataset (typically ``complex64``).

        Raises
        ------
        ValueError
            If indices are out of bounds.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata.rows or col_end > self.metadata.cols:
            raise ValueError(
                f"End indices ({row_end}, {col_end}) exceed image "
                f"dimensions ({self.metadata.rows}, {self.metadata.cols})"
            )

        ds = self._file[self._imagery_path]
        return ds[row_start:row_end, col_start:col_end]

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire image in a single h5py call.

        Parameters
        ----------
        bands : Optional[List[int]]
            Ignored.

        Returns
        -------
        np.ndarray
            Full image array.
        """
        ds = self._file[self._imagery_path]
        return ds[()]

    def get_shape(self) -> Tuple[int, ...]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, ...]
            ``(rows, cols)``.
        """
        return (self.metadata.rows, self.metadata.cols)

    def get_dtype(self) -> np.dtype:
        """Get the data type of the imagery dataset.

        Returns
        -------
        np.dtype
            Typically ``complex64`` for SLC data.
        """
        return np.dtype(self.metadata.dtype)

    # ---------------------------------------------------------------
    # Convenience methods
    # ---------------------------------------------------------------

    def get_product_type(self) -> str:
        """Return the detected product type.

        Returns
        -------
        str
            ``'RSLC'`` or ``'GSLC'``.
        """
        return self._product_type

    def get_radar_band(self) -> str:
        """Return the detected radar band group name.

        Returns
        -------
        str
            ``'LSAR'`` or ``'SSAR'``.
        """
        return self._radar_band

    def get_available_frequencies(self) -> List[str]:
        """Return available frequency identifiers in the file.

        Returns
        -------
        List[str]
            E.g., ``['A']`` or ``['A', 'B']``.
        """
        return list(self._available_frequencies)

    def get_available_polarizations(self) -> List[str]:
        """Return available polarizations for the active frequency.

        Returns
        -------
        List[str]
            E.g., ``['HH']`` or ``['HH', 'HV', 'VH', 'VV']``.
        """
        return list(self._available_polarizations)

    def read_mask(self) -> Optional[np.ndarray]:
        """Read the GSLC validity mask.

        Only available for GSLC products.  Returns ``None`` for RSLC.

        Returns
        -------
        np.ndarray or None
            Mask array with shape ``(rows, cols)`` and dtype ``uint8``,
            or ``None`` if not available.
        """
        if self._product_type != 'GSLC':
            return None
        mask_path = (
            f'{self._base_path}/grids/'
            f'frequency{self._frequency}/mask'
        )
        if mask_path in self._file:
            return self._file[mask_path][()]
        return None

    def close(self) -> None:
        """Close the HDF5 file handle."""
        if hasattr(self, '_file') and self._file is not None:
            self._file.close()
            self._file = None


def open_nisar(
    filepath: Union[str, Path],
    frequency: Optional[str] = None,
    polarization: Optional[str] = None,
) -> NISARReader:
    """Open a NISAR RSLC or GSLC product.

    Convenience factory that auto-detects the product type (RSLC/GSLC)
    and radar band (LSAR/SSAR) from the HDF5 structure.

    Parameters
    ----------
    filepath : str or Path
        Path to the NISAR HDF5 file.
    frequency : str, optional
        Frequency sub-band (``'A'`` or ``'B'``).  Auto-detects if None.
    polarization : str, optional
        Polarization channel (``'HH'``, ``'HV'``, ``'VH'``, ``'VV'``).
        Auto-detects if None.

    Returns
    -------
    NISARReader
        Reader instance for the selected frequency/polarization.

    Examples
    --------
    >>> from grdl.IO.sar.nisar import open_nisar
    >>> reader = open_nisar('NISAR_L1_RSLC_example.h5')
    >>> print(reader.metadata.product_type)
    'RSLC'
    >>> reader.close()
    """
    return NISARReader(
        filepath, frequency=frequency, polarization=polarization
    )
