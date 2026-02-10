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
2026-02-10
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

from grdl.IO.base import ImageReader
from grdl.IO.models import BIOMASSMetadata


class BIOMASSL1Reader(ImageReader):
    """Reader for BIOMASS L1 SCS (Single-look Complex Slant) products.

    BIOMASS L1 products contain complex-valued P-band SAR data in slant
    range geometry. Data is stored as separate magnitude and phase
    GeoTIFF files with XML annotation metadata.

    Parameters
    ----------
    filepath : str or Path
        Path to BIOMASS product directory.

    Attributes
    ----------
    filepath : Path
        Path to the product directory.
    metadata : BIOMASSMetadata
        Typed BIOMASS metadata with all annotation fields.
    polarizations : List[str]
        Available polarizations (HH, HV, VH, VV).
    magnitude_dataset : rasterio.DatasetReader
        Rasterio dataset for magnitude TIFF.
    phase_dataset : rasterio.DatasetReader
        Rasterio dataset for phase TIFF.

    Raises
    ------
    ImportError
        If rasterio is not installed.
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If the directory is not a valid BIOMASS L1 product.

    Notes
    -----
    Complex values are reconstructed as:
    ``complex = magnitude * exp(1j * phase)``

    Examples
    --------
    >>> from grdl.IO.sar import BIOMASSL1Reader
    >>> with BIOMASSL1Reader('BIO_S1_SCS__1S_...') as reader:
    ...     hh_chip = reader.read_chip(0, 1000, 0, 1000, bands=[0])
    ...     hh_db = 20 * np.log10(np.abs(hh_chip) + 1e-10)
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        if not _HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for BIOMASS reading. "
                "Install with: pip install rasterio"
            )

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"Product directory not found: {filepath}"
            )
        if not filepath.is_dir():
            raise ValueError(
                f"BIOMASS products must be directories: {filepath}"
            )

        self.filepath = filepath
        self.magnitude_dataset = None
        self.phase_dataset = None
        self.polarizations: List[str] = []

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load BIOMASS L1 metadata from XML annotation and TIFF files."""
        annot_dir = self.filepath / "annotation"
        if not annot_dir.exists():
            raise ValueError(
                f"No annotation directory found in {self.filepath}"
            )

        annot_files = list(annot_dir.glob("*_annot.xml"))
        if not annot_files:
            raise ValueError(f"No annotation XML found in {annot_dir}")

        annot_file = annot_files[0]

        try:
            tree = ET.parse(annot_file)
            root = tree.getroot()

            acq_info = root.find('acquisitionInformation')
            sar_image = root.find('sarImage')
            inst_params = root.find('instrumentParameters')

            rows = int(sar_image.findtext('numberOfLines', '0'))
            cols = int(sar_image.findtext('numberOfSamples', '0'))

            mission = acq_info.findtext('mission', 'BIOMASS')
            swath = acq_info.findtext('swath', '')
            product_type = acq_info.findtext('productType', 'SCS')
            start_time = acq_info.findtext('startTime', '')
            stop_time = acq_info.findtext('stopTime', '')
            orbit_number = int(
                acq_info.findtext('absoluteOrbitNumber', '0')
            )
            orbit_pass = acq_info.findtext('orbitPass', '')
            range_pixel_spacing = float(
                sar_image.findtext('rangePixelSpacing', '0')
            )
            azimuth_pixel_spacing = float(
                sar_image.findtext('azimuthPixelSpacing', '0')
            )
            pixel_type = sar_image.findtext(
                'pixelType', '32 bit Float'
            )
            pixel_representation = sar_image.findtext(
                'pixelRepresentation', 'Abs Phase'
            )
            projection = sar_image.findtext(
                'projection', 'Slant Range'
            )
            nodata_value = float(
                sar_image.findtext('noDataValue', '-9999.0')
            )

            polarizations: Optional[List[str]] = None
            num_polarizations: Optional[int] = None
            pol_list = acq_info.find('polarisationList')
            if pol_list is not None:
                self.polarizations = [
                    pol.text for pol in pol_list.findall('polarisation')
                ]
                polarizations = self.polarizations
                num_polarizations = len(self.polarizations)

            prf: Optional[float] = None
            prf_list = (
                inst_params.find('prfList')
                if inst_params is not None
                else None
            )
            if prf_list is not None and len(prf_list) > 0:
                prf_elem = prf_list.find('prf/value')
                if prf_elem is not None:
                    prf = float(prf_elem.text)

            corner_coords: Optional[
                Dict[str, Tuple[float, float]]
            ] = None
            footprint = sar_image.findtext('footprint', '')
            if footprint:
                coords = [float(x) for x in footprint.split()]
                corner_coords = {
                    'corner1': (coords[0], coords[1]),
                    'corner2': (coords[2], coords[3]),
                    'corner3': (coords[4], coords[5]),
                    'corner4': (coords[6], coords[7]),
                }

        except Exception as e:
            raise ValueError(
                f"Failed to parse annotation XML: {e}"
            ) from e

        measurement_dir = self.filepath / "measurement"
        if not measurement_dir.exists():
            raise ValueError(
                f"No measurement directory found in {self.filepath}"
            )

        mag_files = list(measurement_dir.glob("*_abs.tiff"))
        phase_files = list(measurement_dir.glob("*_phase.tiff"))

        if not mag_files or not phase_files:
            raise ValueError(
                f"No magnitude/phase TIFFs found in {measurement_dir}"
            )

        try:
            self.magnitude_dataset = rasterio.open(mag_files[0])
            self.phase_dataset = rasterio.open(phase_files[0])

            if (self.magnitude_dataset.height != self.phase_dataset.height
                    or self.magnitude_dataset.width != self.phase_dataset.width
                    or self.magnitude_dataset.count
                    != self.phase_dataset.count):
                raise ValueError(
                    "Magnitude and phase TIFFs have mismatched dimensions"
                )

            bands = self.magnitude_dataset.count

            gcps: Optional[
                List[Tuple[float, float, float, float, float]]
            ] = None
            crs_str: Optional[str] = None
            if self.magnitude_dataset.gcps[0]:
                gcps_raw, crs_val = self.magnitude_dataset.gcps
                gcps = [
                    (gcp.x, gcp.y, gcp.z, gcp.row, gcp.col)
                    for gcp in gcps_raw
                ]
                crs_str = str(crs_val)

            self.metadata = BIOMASSMetadata(
                format='BIOMASS_L1_SCS',
                rows=rows,
                cols=cols,
                dtype='complex64',
                bands=bands,
                crs=crs_str,
                mission=mission,
                swath=swath,
                product_type=product_type,
                start_time=start_time,
                stop_time=stop_time,
                orbit_number=orbit_number,
                orbit_pass=orbit_pass,
                polarizations=polarizations,
                num_polarizations=num_polarizations,
                range_pixel_spacing=range_pixel_spacing,
                azimuth_pixel_spacing=azimuth_pixel_spacing,
                pixel_type=pixel_type,
                pixel_representation=pixel_representation,
                projection=projection,
                nodata_value=nodata_value,
                corner_coords=corner_coords,
                prf=prf,
                gcps=gcps,
            )

        except Exception as e:
            raise ValueError(
                f"Failed to open magnitude/phase TIFFs: {e}"
            ) from e

    def _reconstruct_complex(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct complex values from magnitude and phase.

        Parameters
        ----------
        magnitude : np.ndarray
            Magnitude array.
        phase : np.ndarray
            Phase array (radians).

        Returns
        -------
        np.ndarray
            Complex-valued array (complex64).
        """
        nodata = (
            self.metadata.nodata_value
            if self.metadata.nodata_value is not None
            else -9999.0
        )
        valid_mask = (magnitude != nodata) & (phase != nodata)

        complex_data = np.zeros(magnitude.shape, dtype=np.complex64)
        complex_data[valid_mask] = (
            magnitude[valid_mask] * np.exp(1j * phase[valid_mask])
        )
        return complex_data

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the BIOMASS L1 SCS product.

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
            Polarization indices (0-based: 0=HH, 1=HV, 2=VH, 3=VV).
            If None, read all polarizations.

        Returns
        -------
        np.ndarray
            Complex-valued chip. Shape ``(rows, cols)`` for single
            polarization or ``(pols, rows, cols)`` for multiple.

        Raises
        ------
        ValueError
            If indices are out of bounds.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if (row_end > self.metadata['rows']
                or col_end > self.metadata['cols']):
            raise ValueError("End indices exceed image dimensions")

        window = Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start,
        )

        if bands is None:
            bands_to_read = list(
                range(1, self.metadata['bands'] + 1)
            )
        else:
            bands_to_read = [b + 1 for b in bands]

        mag_data = self.magnitude_dataset.read(
            bands_to_read, window=window
        )
        phase_data = self.phase_dataset.read(
            bands_to_read, window=window
        )

        if mag_data.ndim == 2:
            complex_chip = self._reconstruct_complex(mag_data, phase_data)
        else:
            complex_chip = np.zeros(mag_data.shape, dtype=np.complex64)
            for i in range(mag_data.shape[0]):
                complex_chip[i] = self._reconstruct_complex(
                    mag_data[i], phase_data[i]
                )
            if len(bands_to_read) == 1:
                complex_chip = complex_chip.squeeze(axis=0)

        return complex_chip

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire BIOMASS L1 SCS image.

        Parameters
        ----------
        bands : Optional[List[int]]
            Polarization indices (0-based). If None, read all.

        Returns
        -------
        np.ndarray
            Full complex-valued image.
        """
        return self.read_chip(
            0, self.metadata['rows'],
            0, self.metadata['cols'],
            bands=bands,
        )

    def get_shape(self) -> Tuple[int, ...]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, ...]
            ``(rows, cols)`` or ``(rows, cols, pols)``.
        """
        if self.metadata['bands'] == 1:
            return (self.metadata['rows'], self.metadata['cols'])
        return (
            self.metadata['rows'],
            self.metadata['cols'],
            self.metadata['bands'],
        )

    def get_dtype(self) -> np.dtype:
        """Get data type.

        Returns
        -------
        np.dtype
            ``complex64`` for BIOMASS L1 SCS data.
        """
        return np.dtype('complex64')

    def get_polarization_name(self, band_index: int) -> str:
        """Get polarization name for a band index.

        Parameters
        ----------
        band_index : int
            Band index (0-based).

        Returns
        -------
        str
            Polarization name (e.g., ``'HH'``, ``'HV'``).

        Raises
        ------
        ValueError
            If band_index is out of range.
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
    """Auto-detect BIOMASS product level and return appropriate reader.

    Currently only supports L1 SCS products.

    Parameters
    ----------
    filepath : str or Path
        Path to BIOMASS product directory.

    Returns
    -------
    ImageReader
        BIOMASS reader instance.

    Raises
    ------
    ValueError
        If format cannot be determined or is unsupported.

    Examples
    --------
    >>> reader = open_biomass('BIO_S1_SCS__1S_...')
    >>> chip = reader.read_chip(0, 1000, 0, 1000, bands=[0])
    >>> reader.close()
    """
    filepath = Path(filepath)

    if 'SCS' in filepath.name.upper():
        try:
            return BIOMASSL1Reader(filepath)
        except Exception as e:
            raise ValueError(
                f"Failed to open as BIOMASS L1 SCS: {e}"
            ) from e

    raise ValueError(
        f"Could not determine BIOMASS product type for {filepath}. "
        "Currently only L1 SCS products are supported."
    )
