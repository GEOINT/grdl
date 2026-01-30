# -*- coding: utf-8 -*-
"""
BIOMASS Readers - ESA BIOMASS P-band SAR satellite data readers.

Provides readers for BIOMASS L1 SCS (Single-look Complex Slant) products.
BIOMASS data is stored in SAFE-like directory structure with magnitude and
phase GeoTIFF files plus XML annotation metadata.

Dependencies
------------
rasterio

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
2026-01-30

Modified
--------
2026-01-30
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import xml.etree.ElementTree as ET
import warnings

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn(
        "Rasterio not available. Install with: pip install rasterio",
        ImportWarning
    )

from grdl.IO.base import ImageReader


class BIOMASSL1Reader(ImageReader):
    """
    Reader for BIOMASS L1 SCS (Single-look Complex Slant) products.

    BIOMASS L1 products contain complex-valued P-band SAR data in slant range
    geometry. Data is stored as separate magnitude and phase GeoTIFF files
    with XML annotation metadata.

    Attributes
    ----------
    filepath : Path
        Path to BIOMASS product directory
    metadata : Dict[str, Any]
        BIOMASS-specific metadata including mission parameters
    polarizations : List[str]
        Available polarizations (HH, HV, VH, VV)
    magnitude_dataset : rasterio.DatasetReader
        Rasterio dataset for magnitude TIFF
    phase_dataset : rasterio.DatasetReader
        Rasterio dataset for phase TIFF

    Notes
    -----
    BIOMASS L1 data is complex-valued. Each polarization is stored as a
    separate band in the magnitude and phase TIFFs.

    The reader reconstructs complex values as: complex = magnitude * exp(1j * phase)

    Examples
    --------
    >>> with BIOMASSL1Reader('BIO_S1_SCS__1S_...') as reader:
    ...     print(f"Polarizations: {reader.metadata['polarizations']}")
    ...     # Read HH polarization chip (band 0)
    ...     hh_chip = reader.read_chip(0, 1000, 0, 1000, bands=[0])
    ...     # Convert to dB magnitude
    ...     hh_db = 20 * np.log10(np.abs(hh_chip) + 1e-10)
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        """
        Initialize BIOMASS L1 SCS reader.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to BIOMASS product directory

        Raises
        ------
        FileNotFoundError
            If directory does not exist
        ImportError
            If rasterio is not installed
        ValueError
            If directory is not a valid BIOMASS L1 product
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError(
                "Rasterio is required for BIOMASS reading. "
                "Install with: pip install rasterio"
            )

        # BIOMASS products are directories
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Product directory not found: {filepath}")
        if not filepath.is_dir():
            raise ValueError(f"BIOMASS products must be directories: {filepath}")

        self.filepath = filepath
        self.metadata: Dict[str, Any] = {}

        # Initialize datasets (will be set in _load_metadata)
        self.magnitude_dataset = None
        self.phase_dataset = None
        self.polarizations: List[str] = []

        # Load metadata and open datasets
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load BIOMASS L1 metadata from XML annotation and TIFF files."""
        # Find annotation XML
        annot_dir = self.filepath / "annotation"
        if not annot_dir.exists():
            raise ValueError(f"No annotation directory found in {self.filepath}")

        annot_files = list(annot_dir.glob("*_annot.xml"))
        if not annot_files:
            raise ValueError(f"No annotation XML found in {annot_dir}")

        annot_file = annot_files[0]  # Use first annotation file

        # Parse annotation XML
        try:
            tree = ET.parse(annot_file)
            root = tree.getroot()

            # Extract key metadata
            acq_info = root.find('acquisitionInformation')
            sar_image = root.find('sarImage')
            inst_params = root.find('instrumentParameters')

            self.metadata = {
                'format': 'BIOMASS_L1_SCS',
                'mission': acq_info.findtext('mission', 'BIOMASS'),
                'swath': acq_info.findtext('swath', ''),
                'product_type': acq_info.findtext('productType', 'SCS'),
                'start_time': acq_info.findtext('startTime', ''),
                'stop_time': acq_info.findtext('stopTime', ''),
                'orbit_number': int(acq_info.findtext('absoluteOrbitNumber', '0')),
                'orbit_pass': acq_info.findtext('orbitPass', ''),
                'rows': int(sar_image.findtext('numberOfLines', '0')),
                'cols': int(sar_image.findtext('numberOfSamples', '0')),
                'range_pixel_spacing': float(sar_image.findtext('rangePixelSpacing', '0')),
                'azimuth_pixel_spacing': float(sar_image.findtext('azimuthPixelSpacing', '0')),
                'pixel_type': sar_image.findtext('pixelType', '32 bit Float'),
                'pixel_representation': sar_image.findtext('pixelRepresentation', 'Abs Phase'),
                'projection': sar_image.findtext('projection', 'Slant Range'),
                'nodata_value': float(sar_image.findtext('noDataValue', '-9999.0')),
            }

            # Extract polarizations
            pol_list = acq_info.find('polarisationList')
            if pol_list is not None:
                self.polarizations = [pol.text for pol in pol_list.findall('polarisation')]
                self.metadata['polarizations'] = self.polarizations
                self.metadata['num_polarizations'] = len(self.polarizations)

            # Extract PRF if available
            prf_list = inst_params.find('prfList') if inst_params is not None else None
            if prf_list is not None and len(prf_list) > 0:
                prf_elem = prf_list.find('prf/value')
                if prf_elem is not None:
                    self.metadata['prf'] = float(prf_elem.text)

            # Extract footprint (corner coordinates)
            footprint = sar_image.findtext('footprint', '')
            if footprint:
                coords = [float(x) for x in footprint.split()]
                # Footprint is lat1 lon1 lat2 lon2 lat3 lon3 lat4 lon4
                self.metadata['corner_coords'] = {
                    'corner1': (coords[0], coords[1]),
                    'corner2': (coords[2], coords[3]),
                    'corner3': (coords[4], coords[5]),
                    'corner4': (coords[6], coords[7]),
                }

        except Exception as e:
            raise ValueError(f"Failed to parse annotation XML: {e}") from e

        # Open magnitude and phase TIFFs
        measurement_dir = self.filepath / "measurement"
        if not measurement_dir.exists():
            raise ValueError(f"No measurement directory found in {self.filepath}")

        # Find magnitude and phase TIFFs
        mag_files = list(measurement_dir.glob("*_abs.tiff"))
        phase_files = list(measurement_dir.glob("*_phase.tiff"))

        if not mag_files or not phase_files:
            raise ValueError(f"No magnitude/phase TIFFs found in {measurement_dir}")

        try:
            self.magnitude_dataset = rasterio.open(mag_files[0])
            self.phase_dataset = rasterio.open(phase_files[0])

            # Verify dimensions match
            if (self.magnitude_dataset.height != self.phase_dataset.height or
                self.magnitude_dataset.width != self.phase_dataset.width or
                self.magnitude_dataset.count != self.phase_dataset.count):
                raise ValueError("Magnitude and phase TIFFs have mismatched dimensions")

            # Update metadata with TIFF info
            self.metadata['dtype'] = 'complex64'
            self.metadata['bands'] = self.magnitude_dataset.count

            # Extract geolocation from GCPs
            if self.magnitude_dataset.gcps[0]:
                gcps, crs = self.magnitude_dataset.gcps
                self.metadata['gcps'] = [(gcp.x, gcp.y, gcp.z, gcp.row, gcp.col)
                                          for gcp in gcps]
                self.metadata['crs'] = str(crs)

        except Exception as e:
            raise ValueError(f"Failed to open magnitude/phase TIFFs: {e}") from e

    def _reconstruct_complex(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct complex values from magnitude and phase.

        Parameters
        ----------
        magnitude : np.ndarray
            Magnitude array
        phase : np.ndarray
            Phase array (in radians)

        Returns
        -------
        np.ndarray
            Complex-valued array (complex64)
        """
        # Handle nodata values
        nodata = self.metadata.get('nodata_value', -9999.0)
        valid_mask = (magnitude != nodata) & (phase != nodata)

        # Reconstruct complex: magnitude * exp(1j * phase)
        complex_data = np.zeros(magnitude.shape, dtype=np.complex64)
        complex_data[valid_mask] = magnitude[valid_mask] * np.exp(1j * phase[valid_mask])

        return complex_data

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Read a spatial chip from the BIOMASS L1 SCS product.

        Parameters
        ----------
        row_start : int
            Starting row index (inclusive)
        row_end : int
            Ending row index (exclusive)
        col_start : int
            Starting column index (inclusive)
        col_end : int
            Ending column index (exclusive)
        bands : Optional[List[int]], default=None
            Polarization indices to read (0-based: 0=HH, 1=HV, 2=VH, 3=VV).
            If None, read all polarizations.

        Returns
        -------
        np.ndarray
            Complex-valued image chip with shape (rows, cols) for single
            polarization or (pols, rows, cols) for multiple polarizations

        Raises
        ------
        ValueError
            If indices are out of bounds
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata['rows'] or col_end > self.metadata['cols']:
            raise ValueError("End indices exceed image dimensions")

        # Create rasterio window
        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)

        # Determine bands to read
        if bands is None:
            bands_to_read = list(range(1, self.metadata['bands'] + 1))  # 1-based
        else:
            bands_to_read = [b + 1 for b in bands]  # Convert to 1-based

        # Read magnitude and phase
        mag_data = self.magnitude_dataset.read(bands_to_read, window=window)
        phase_data = self.phase_dataset.read(bands_to_read, window=window)

        # Reconstruct complex data
        if mag_data.ndim == 2:
            # Single band (rasterio read without band list)
            complex_chip = self._reconstruct_complex(mag_data, phase_data)
        else:
            # Multiple bands (or single band from list) - reconstruct each
            complex_chip = np.zeros(mag_data.shape, dtype=np.complex64)
            for i in range(mag_data.shape[0]):
                complex_chip[i] = self._reconstruct_complex(mag_data[i], phase_data[i])

            # Squeeze if only one band was requested
            if len(bands_to_read) == 1:
                complex_chip = complex_chip.squeeze(axis=0)

        return complex_chip

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """
        Read the entire BIOMASS L1 SCS image.

        Parameters
        ----------
        bands : Optional[List[int]], default=None
            Polarization indices to read (0-based). If None, read all.

        Returns
        -------
        np.ndarray
            Full complex-valued image

        Warnings
        --------
        BIOMASS L1 files can be very large. Consider using read_chip() instead.
        """
        return self.read_chip(
            0, self.metadata['rows'],
            0, self.metadata['cols'],
            bands=bands
        )

    def get_shape(self) -> Tuple[int, ...]:
        """
        Get image dimensions.

        Returns
        -------
        Tuple[int, ...]
            (rows, cols) for single polarization or
            (rows, cols, pols) for multi-polarization
        """
        if self.metadata['bands'] == 1:
            return (self.metadata['rows'], self.metadata['cols'])
        return (self.metadata['rows'], self.metadata['cols'], self.metadata['bands'])

    def get_dtype(self) -> np.dtype:
        """
        Get data type.

        Returns
        -------
        np.dtype
            Complex64 for BIOMASS L1 SCS data
        """
        return np.dtype('complex64')

    def get_geolocation(self) -> Optional[Dict[str, Any]]:
        """
        Get geolocation information.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with GCPs, corner coordinates, and CRS info.
            BIOMASS L1 is in slant range geometry (not geocoded).
        """
        return {
            'crs': self.metadata.get('crs', 'WGS84'),
            'projection': self.metadata.get('projection', 'Slant Range'),
            'corner_coords': self.metadata.get('corner_coords'),
            'gcps': self.metadata.get('gcps'),  # Sample of GCPs
            'range_pixel_spacing': self.metadata.get('range_pixel_spacing'),
            'azimuth_pixel_spacing': self.metadata.get('azimuth_pixel_spacing'),
        }

    def get_polarization_name(self, band_index: int) -> str:
        """
        Get polarization name for a band index.

        Parameters
        ----------
        band_index : int
            Band index (0-based)

        Returns
        -------
        str
            Polarization name (e.g., 'HH', 'HV', 'VH', 'VV')
        """
        if band_index < 0 or band_index >= len(self.polarizations):
            raise ValueError(f"Invalid band index: {band_index}")
        return self.polarizations[band_index]

    def close(self) -> None:
        """Close the rasterio datasets."""
        if self.magnitude_dataset is not None:
            self.magnitude_dataset.close()
        if self.phase_dataset is not None:
            self.phase_dataset.close()


def open_biomass(filepath: Union[str, Path]) -> ImageReader:
    """
    Auto-detect BIOMASS product level and return appropriate reader.

    Currently only supports L1 SCS products.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to BIOMASS product directory

    Returns
    -------
    ImageReader
        BIOMASS reader instance (currently only BIOMASSL1Reader)

    Raises
    ------
    ValueError
        If format cannot be determined or is unsupported

    Examples
    --------
    >>> reader = open_biomass('BIO_S1_SCS__1S_...')
    >>> print(f"Format: {reader.metadata['format']}")
    >>> chip = reader.read_chip(0, 1000, 0, 1000, bands=[0])  # HH pol
    >>> reader.close()
    """
    filepath = Path(filepath)

    # Check if directory name contains L1 SCS product identifier
    if 'SCS' in filepath.name.upper():
        try:
            return BIOMASSL1Reader(filepath)
        except Exception as e:
            raise ValueError(f"Failed to open as BIOMASS L1 SCS: {e}") from e

    raise ValueError(
        f"Could not determine BIOMASS product type for {filepath}. "
        "Currently only L1 SCS products are supported."
    )