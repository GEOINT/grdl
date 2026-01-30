# -*- coding: utf-8 -*-
"""
SAR Readers - Synthetic Aperture Radar imagery readers.

Provides readers for various SAR formats including NGA standards (SICD, CPHD, CRSD)
and common SAR products (GRD). Built on SARPY for NGA format compliance.

Dependencies
------------
sarpy
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
import warnings

import numpy as np

try:
    from sarpy.io.complex.converter import open_complex
    from sarpy.io.phase_history.converter import open_phase_history
    SARPY_AVAILABLE = True
except ImportError:
    SARPY_AVAILABLE = False
    warnings.warn(
        "SARPY not available. Install with: pip install sarpy",
        ImportWarning
    )

try:
    import rasterio
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from grdl.IO.base import ImageReader


class SICDReader(ImageReader):
    """
    Reader for SICD (Sensor Independent Complex Data) format.

    SICD is the NGA standard for complex SAR imagery. This reader uses
    SARPY to handle SICD files in NITF or other container formats.

    Attributes
    ----------
    filepath : Path
        Path to the SICD file
    metadata : Dict[str, Any]
        SICD metadata including sensor parameters, collection geometry
    sicd_meta : object
        Native SARPY SICD metadata object
    reader : object
        SARPY reader object

    Notes
    -----
    SICD data is complex-valued (I/Q). Use abs() for magnitude images.

    Examples
    --------
    >>> with SICDReader('image.nitf') as reader:
    ...     shape = reader.get_shape()
    ...     chip = reader.read_chip(0, 1000, 0, 1000)
    ...     magnitude = np.abs(chip)
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        """
        Initialize SICD reader.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to SICD file (NITF or other SICD container)

        Raises
        ------
        FileNotFoundError
            If file does not exist
        ImportError
            If SARPY is not installed
        ValueError
            If file is not a valid SICD file
        """
        if not SARPY_AVAILABLE:
            raise ImportError(
                "SARPY is required for SICD reading. "
                "Install with: pip install sarpy"
            )

        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load SICD metadata using SARPY."""
        try:
            self.reader = open_complex(str(self.filepath))
            self.sicd_meta = self.reader.sicd_meta

            # Extract key metadata into standardized dict
            self.metadata = {
                'format': 'SICD',
                'rows': self.sicd_meta.ImageData.NumRows,
                'cols': self.sicd_meta.ImageData.NumCols,
                'dtype': 'complex64',
                'collector_name': self.sicd_meta.CollectionInfo.CollectorName,
                'core_name': self.sicd_meta.CollectionInfo.CoreName,
                'classification': self.sicd_meta.CollectionInfo.Classification,
                'collect_start': str(self.sicd_meta.Timeline.CollectStart),
                'collect_duration': self.sicd_meta.Timeline.CollectDuration,
                'pixel_type': self.sicd_meta.ImageData.PixelType,
                'image_plane': self.sicd_meta.Grid.ImagePlane,
                'center_frequency': self.sicd_meta.RadarCollection.TxFrequency.get_center_frequency(),
                'bandwidth': self.sicd_meta.RadarCollection.Waveform[0].get_transmit_bandwidth(),
            }

            # Add geolocation if available
            if hasattr(self.sicd_meta, 'GeoData') and self.sicd_meta.GeoData is not None:
                self.metadata['scp_llh'] = self.sicd_meta.GeoData.SCP.LLH.get_array()
                if self.sicd_meta.GeoData.ImageCorners is not None:
                    corners = self.sicd_meta.GeoData.ImageCorners
                    self.metadata['corner_coords'] = {
                        'icp1': corners.ICP1.get_array(),
                        'icp2': corners.ICP2.get_array(),
                        'icp3': corners.ICP3.get_array(),
                        'icp4': corners.ICP4.get_array(),
                    }

        except Exception as e:
            raise ValueError(f"Failed to load SICD metadata: {e}") from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Read a spatial chip from the SICD image.

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
            Ignored for SICD (single complex band)

        Returns
        -------
        np.ndarray
            Complex-valued image chip with shape (rows, cols)
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata['rows'] or col_end > self.metadata['cols']:
            raise ValueError("End indices exceed image dimensions")

        # SARPY uses (row_start, row_end, col_start, col_end) indexing
        chip = self.reader[row_start:row_end, col_start:col_end]
        return chip

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """
        Read the entire SICD image.

        Parameters
        ----------
        bands : Optional[List[int]], default=None
            Ignored for SICD (single complex band)

        Returns
        -------
        np.ndarray
            Full complex-valued image

        Warnings
        --------
        SICD files can be very large. Consider using read_chip() instead.
        """
        return self.reader[:, :]

    def get_shape(self) -> Tuple[int, int]:
        """
        Get image dimensions.

        Returns
        -------
        Tuple[int, int]
            (rows, cols)
        """
        return (self.metadata['rows'], self.metadata['cols'])

    def get_dtype(self) -> np.dtype:
        """
        Get data type.

        Returns
        -------
        np.dtype
            Complex64 for SICD data
        """
        return np.dtype('complex64')

    def get_geolocation(self) -> Optional[Dict[str, Any]]:
        """
        Get geolocation information.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with Scene Center Point (SCP) coordinates and
            image corner coordinates in lat/lon/height
        """
        if 'scp_llh' not in self.metadata:
            return None

        geo_info = {
            'scp_llh': self.metadata['scp_llh'],  # [lat, lon, height]
            'corner_coords': self.metadata.get('corner_coords'),
            'projection': 'SICD native geometry',
        }
        return geo_info

    def close(self) -> None:
        """Close the SARPY reader."""
        if hasattr(self, 'reader') and self.reader is not None:
            self.reader.close()


class CPHDReader(ImageReader):
    """
    Reader for CPHD (Compensated Phase History Data) format.

    CPHD is the NGA standard for phase history data. This reader uses
    SARPY to handle CPHD files.

    Attributes
    ----------
    filepath : Path
        Path to the CPHD file
    metadata : Dict[str, Any]
        CPHD metadata including collection parameters
    cphd_meta : object
        Native SARPY CPHD metadata object
    reader : object
        SARPY reader object

    Notes
    -----
    CPHD contains phase history data, not formed imagery. Use sarpy's
    focusing algorithms to generate images from CPHD.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        """
        Initialize CPHD reader.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to CPHD file

        Raises
        ------
        ImportError
            If SARPY is not installed
        ValueError
            If file is not valid CPHD
        """
        if not SARPY_AVAILABLE:
            raise ImportError(
                "SARPY is required for CPHD reading. "
                "Install with: pip install sarpy"
            )

        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load CPHD metadata using SARPY."""
        try:
            self.reader = open_phase_history(str(self.filepath))
            self.cphd_meta = self.reader.cphd_meta

            # Extract key metadata
            self.metadata = {
                'format': 'CPHD',
                'num_channels': self.cphd_meta.Data.NumCPHDChannels,
                'classification': self.cphd_meta.CollectionInfo.Classification,
                'collector_name': self.cphd_meta.CollectionInfo.CollectorName,
                'core_name': self.cphd_meta.CollectionInfo.CoreName,
                'collect_start': str(self.cphd_meta.Global.CollectStart),
                'ref_frequency': self.cphd_meta.Global.RefFreqIndex,
            }

            # CPHD doesn't have fixed rows/cols like imagery
            # Store channel info instead
            self.metadata['channels'] = {}
            for i, channel in enumerate(self.cphd_meta.Data.Channels):
                self.metadata['channels'][channel.Identifier] = {
                    'num_vectors': channel.NumVectors,
                    'num_samples': channel.NumSamples,
                }

        except Exception as e:
            raise ValueError(f"Failed to load CPHD metadata: {e}") from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Read phase history vectors.

        Parameters
        ----------
        row_start : int
            Starting vector index
        row_end : int
            Ending vector index
        col_start : int
            Starting sample index
        col_start : int
            Ending sample index
        bands : Optional[List[int]], default=None
            Channel indices to read

        Returns
        -------
        np.ndarray
            Phase history data
        """
        # CPHD reading is channel-based, delegate to SARPY
        channel = bands[0] if bands else 0
        return self.reader.read_chip(
            index=channel,
            dim1_range=(row_start, row_end),
            dim2_range=(col_start, col_end)
        )

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """
        Read full phase history data.

        Parameters
        ----------
        bands : Optional[List[int]], default=None
            Channel indices to read

        Returns
        -------
        np.ndarray
            Phase history data for specified channels
        """
        channel = bands[0] if bands else 0
        return self.reader.read(index=channel)

    def get_shape(self) -> Tuple[int, int]:
        """
        Get phase history dimensions for first channel.

        Returns
        -------
        Tuple[int, int]
            (num_vectors, num_samples) for first channel
        """
        first_channel = list(self.metadata['channels'].values())[0]
        return (first_channel['num_vectors'], first_channel['num_samples'])

    def get_dtype(self) -> np.dtype:
        """
        Get data type.

        Returns
        -------
        np.dtype
            Complex64 for CPHD data
        """
        return np.dtype('complex64')

    def get_geolocation(self) -> Optional[Dict[str, Any]]:
        """
        Get geolocation information.

        Returns
        -------
        Optional[Dict[str, Any]]
            Reference point and coordinate system info
        """
        if not hasattr(self.cphd_meta, 'ReferenceGeometry'):
            return None

        return {
            'reference_point': 'See CPHD ReferenceGeometry',
            'projection': 'Phase history space',
        }

    def close(self) -> None:
        """Close the SARPY reader."""
        if hasattr(self, 'reader') and self.reader is not None:
            self.reader.close()


class GRDReader(ImageReader):
    """
    Reader for GRD (Ground Range Detected) SAR products.

    GRD products are geocoded, detected SAR imagery typically in GeoTIFF
    format. Common for Sentinel-1, RADARSAT, etc.

    Attributes
    ----------
    filepath : Path
        Path to GRD file (typically GeoTIFF)
    metadata : Dict[str, Any]
        GRD metadata including geolocation
    dataset : rasterio.DatasetReader
        Rasterio dataset object

    Notes
    -----
    GRD data is real-valued (magnitude/intensity), not complex.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        """
        Initialize GRD reader.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to GRD file (GeoTIFF)

        Raises
        ------
        ImportError
            If rasterio is not installed
        ValueError
            If file cannot be opened
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError(
                "Rasterio is required for GRD reading. "
                "Install with: pip install rasterio"
            )

        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Load GRD metadata using rasterio."""
        try:
            self.dataset = rasterio.open(str(self.filepath))

            self.metadata = {
                'format': 'GRD',
                'rows': self.dataset.height,
                'cols': self.dataset.width,
                'bands': self.dataset.count,
                'dtype': str(self.dataset.dtypes[0]),
                'crs': str(self.dataset.crs),
                'transform': self.dataset.transform,
                'bounds': self.dataset.bounds,
                'resolution': self.dataset.res,
                'nodata': self.dataset.nodata,
            }

            # Extract any SAR-specific metadata from tags
            if 'TIFFTAG_IMAGEDESCRIPTION' in self.dataset.tags():
                self.metadata['description'] = self.dataset.tags()['TIFFTAG_IMAGEDESCRIPTION']

        except Exception as e:
            raise ValueError(f"Failed to load GRD metadata: {e}") from e

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Read a spatial chip from GRD.

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
            Band indices to read (1-based). If None, read all bands.

        Returns
        -------
        np.ndarray
            Image chip with shape (rows, cols) for single band or
            (bands, rows, cols) for multi-band
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata['rows'] or col_end > self.metadata['cols']:
            raise ValueError("End indices exceed image dimensions")

        # Rasterio uses window-based reading
        from rasterio.windows import Window
        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)

        if bands is None:
            data = self.dataset.read(window=window)
        else:
            # Rasterio uses 1-based band indexing
            data = self.dataset.read([b + 1 for b in bands], window=window)

        # Return shape (rows, cols) for single band
        if data.shape[0] == 1:
            return data[0]

        # Return shape (bands, rows, cols) for multi-band
        return data

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """
        Read the entire GRD image.

        Parameters
        ----------
        bands : Optional[List[int]], default=None
            Band indices to read (0-based). If None, read all bands.

        Returns
        -------
        np.ndarray
            Full image data
        """
        if bands is None:
            data = self.dataset.read()
        else:
            data = self.dataset.read([b + 1 for b in bands])

        if data.shape[0] == 1:
            return data[0]
        return data

    def get_shape(self) -> Tuple[int, ...]:
        """
        Get image dimensions.

        Returns
        -------
        Tuple[int, ...]
            (rows, cols) for single band or (rows, cols, bands) for multi-band
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
            NumPy data type of the image
        """
        return np.dtype(self.metadata['dtype'])

    def get_geolocation(self) -> Optional[Dict[str, Any]]:
        """
        Get geolocation information.

        Returns
        -------
        Dict[str, Any]
            CRS, affine transform, and geographic bounds
        """
        return {
            'crs': self.metadata['crs'],
            'transform': self.metadata['transform'],
            'bounds': self.metadata['bounds'],
            'resolution': self.metadata['resolution'],
        }

    def close(self) -> None:
        """Close the rasterio dataset."""
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.dataset.close()


def open_sar(filepath: Union[str, Path]) -> ImageReader:
    """
    Auto-detect SAR format and return appropriate reader.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to SAR file

    Returns
    -------
    ImageReader
        Appropriate SAR reader instance (SICDReader, CPHDReader, or GRDReader)

    Raises
    ------
    ValueError
        If format cannot be determined or is unsupported

    Examples
    --------
    >>> reader = open_sar('image.nitf')
    >>> print(f"Format: {reader.metadata['format']}")
    >>> chip = reader.read_chip(0, 1000, 0, 1000)
    >>> reader.close()
    """
    filepath = Path(filepath)

    # Try SICD first (NITF container or .sicd extension)
    if SARPY_AVAILABLE:
        try:
            reader = SICDReader(filepath)
            return reader
        except (ValueError, Exception):
            pass

        # Try CPHD
        try:
            reader = CPHDReader(filepath)
            return reader
        except (ValueError, Exception):
            pass

    # Try GRD (GeoTIFF)
    if RASTERIO_AVAILABLE and filepath.suffix.lower() in ['.tif', '.tiff']:
        try:
            reader = GRDReader(filepath)
            return reader
        except (ValueError, Exception):
            pass

    raise ValueError(
        f"Could not determine SAR format for {filepath}. "
        "Ensure file is valid SICD, CPHD, or GRD format and "
        "required libraries (sarpy, rasterio) are installed."
    )
