# -*- coding: utf-8 -*-
"""
HDF5 Reader - Read HDF5 and HDF-EOS5 imagery.

Base data format reader for HDF5 files regardless of producer
(NASA, ASI, JAXA, etc.). Supports explicit dataset path selection
or auto-detection of the first suitable numeric array. Lives at the
IO level so modality submodules can use it without cross-submodule
dependencies.

Dependencies
------------
h5py

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
2026-02-10

Modified
--------
2026-02-10
"""

# Standard library
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
from grdl.IO.base import ImageReader


def _find_datasets(
    group: "h5py.Group",
    min_ndim: int = 2,
) -> List[Tuple[str, Tuple[int, ...], str]]:
    """Walk an HDF5 group and collect numeric datasets.

    Parameters
    ----------
    group : h5py.Group
        Root group or subgroup to search.
    min_ndim : int
        Minimum number of dimensions to include. Default is 2.

    Returns
    -------
    List[Tuple[str, Tuple[int, ...], str]]
        List of ``(path, shape, dtype)`` for each matching dataset.
    """
    results: List[Tuple[str, Tuple[int, ...], str]] = []

    def _visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            if obj.ndim >= min_ndim and np.issubdtype(obj.dtype, np.number):
                results.append((f"/{name}", obj.shape, str(obj.dtype)))

    group.visititems(_visitor)
    return results


class HDF5Reader(ImageReader):
    """Read HDF5 and HDF-EOS5 imagery.

    Reads any HDF5 file containing 2D or 3D numeric arrays, including
    NASA Earthdata products (MODIS, VIIRS, ASTER, ICESat-2, GEDI,
    EMIT), ASI PRISMA, and JAXA missions. Uses h5py as the backend.

    The reader targets a single dataset within the HDF5 hierarchy.
    Provide ``dataset_path`` to select a specific dataset, or omit it
    to auto-detect the first 2D+ numeric array.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file (.h5, .he5, .hdf5).
    dataset_path : str, optional
        Internal HDF5 path to the target dataset
        (e.g., ``'/HDFEOS/GRIDS/VNP_Grid_16Day/Data Fields/NDVI'``).
        If None, auto-detects the first suitable dataset.

    Attributes
    ----------
    filepath : Path
        Path to the HDF5 file.
    metadata : Dict[str, Any]
        Standardized metadata dictionary.
    dataset_path : str
        Resolved internal path to the active dataset.

    Raises
    ------
    ImportError
        If h5py is not installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If no suitable dataset is found or ``dataset_path`` is invalid.

    Examples
    --------
    >>> from grdl.IO.hdf5 import HDF5Reader
    >>> with HDF5Reader('MOD09GA.h5', dataset_path='/MODIS_Grid/sur_refl_b01') as reader:
    ...     chip = reader.read_chip(0, 512, 0, 512)
    ...     print(reader.metadata['format'])
    'HDF5'

    >>> # Auto-detect first dataset
    >>> with HDF5Reader('product.h5') as reader:
    ...     print(reader.dataset_path)
    ...     print(reader.get_shape())

    >>> # Browse available datasets
    >>> datasets = HDF5Reader.list_datasets('product.h5')
    >>> for path, shape, dtype in datasets:
    ...     print(f"{path}: {shape} ({dtype})")
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        dataset_path: Optional[str] = None,
    ) -> None:
        if not _HAS_H5PY:
            raise ImportError(
                "h5py is required for HDF5 reading. "
                "Install with: pip install h5py"
            )
        self._requested_path = dataset_path
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Open the HDF5 file and load metadata for the target dataset."""
        try:
            self._file = h5py.File(str(self.filepath), "r")
        except Exception as e:
            raise ValueError(
                f"Failed to open HDF5 file: {self.filepath}: {e}"
            ) from e

        # Resolve dataset path
        if self._requested_path is not None:
            if self._requested_path not in self._file:
                available = _find_datasets(self._file)
                paths = [p for p, _, _ in available]
                self._file.close()
                raise ValueError(
                    f"Dataset path '{self._requested_path}' not found in "
                    f"{self.filepath}. Available datasets: {paths}"
                )
            obj = self._file[self._requested_path]
            if not isinstance(obj, h5py.Dataset):
                self._file.close()
                raise ValueError(
                    f"Path '{self._requested_path}' is a group, not a dataset."
                )
            self.dataset_path = self._requested_path
        else:
            # Auto-detect first suitable numeric dataset (>= 2D)
            candidates = _find_datasets(self._file, min_ndim=2)
            if not candidates:
                self._file.close()
                raise ValueError(
                    f"No 2D+ numeric datasets found in {self.filepath}. "
                    "Provide an explicit dataset_path."
                )
            self.dataset_path = candidates[0][0]

        ds = self._file[self.dataset_path]
        ndim = ds.ndim

        if ndim == 2:
            rows, cols = ds.shape
            bands = 1
        elif ndim == 3:
            bands, rows, cols = ds.shape
        else:
            self._file.close()
            raise ValueError(
                f"Dataset '{self.dataset_path}' has {ndim} dimensions. "
                "Only 2D and 3D datasets are supported."
            )

        # Collect HDF5 attributes as extras
        extras: Dict[str, Any] = {
            'dataset_path': self.dataset_path,
        }
        for key, val in ds.attrs.items():
            try:
                if isinstance(val, bytes):
                    extras[key] = val.decode("utf-8", errors="replace")
                elif isinstance(val, np.ndarray):
                    extras[key] = val.tolist()
                elif isinstance(val, (np.integer, np.floating)):
                    extras[key] = val.item()
                else:
                    extras[key] = val
            except Exception:
                pass

        # Root-level attributes (file-level metadata)
        for key, val in self._file.attrs.items():
            extra_key = f"file_{key}"
            try:
                if isinstance(val, bytes):
                    extras[extra_key] = val.decode("utf-8", errors="replace")
                elif isinstance(val, np.ndarray):
                    extras[extra_key] = val.tolist()
                elif isinstance(val, (np.integer, np.floating)):
                    extras[extra_key] = val.item()
                else:
                    extras[extra_key] = val
            except Exception:
                pass

        self.metadata = {
            'format': 'HDF5',
            'rows': rows,
            'cols': cols,
            'bands': bands,
            'dtype': str(ds.dtype),
            **extras,
        }

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the HDF5 dataset.

        Uses HDF5 hyperslab selection for efficient partial reads
        without loading the full dataset into memory.

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
            Band indices to read (0-based). If None, read all bands.
            Only applicable to 3D datasets.

        Returns
        -------
        np.ndarray
            Image chip with shape ``(rows, cols)`` for single band or
            ``(bands, rows, cols)`` for multi-band.

        Raises
        ------
        ValueError
            If indices are out of bounds.
        """
        if row_start < 0 or col_start < 0:
            raise ValueError("Start indices must be non-negative")
        if row_end > self.metadata['rows'] or col_end > self.metadata['cols']:
            raise ValueError("End indices exceed image dimensions")

        ds = self._file[self.dataset_path]

        if ds.ndim == 2:
            return ds[row_start:row_end, col_start:col_end]

        # 3D: shape is (bands, rows, cols)
        if bands is None:
            data = ds[:, row_start:row_end, col_start:col_end]
        else:
            data = ds[bands, row_start:row_end, col_start:col_end]

        if data.shape[0] == 1:
            return data[0]
        return data

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """Read the entire dataset.

        Parameters
        ----------
        bands : Optional[List[int]]
            Band indices to read (0-based). If None, read all bands.

        Returns
        -------
        np.ndarray
            Full dataset.
        """
        ds = self._file[self.dataset_path]

        if ds.ndim == 2:
            return ds[()]

        if bands is None:
            data = ds[()]
        else:
            data = ds[bands]

        if data.shape[0] == 1:
            return data[0]
        return data

    def get_shape(self) -> Tuple[int, ...]:
        """Get image dimensions.

        Returns
        -------
        Tuple[int, ...]
            ``(rows, cols)`` for single band or
            ``(rows, cols, bands)`` for multi-band.
        """
        if self.metadata['bands'] == 1:
            return (self.metadata['rows'], self.metadata['cols'])
        return (
            self.metadata['rows'],
            self.metadata['cols'],
            self.metadata['bands'],
        )

    def get_dtype(self) -> np.dtype:
        """Get the data type of the dataset.

        Returns
        -------
        np.dtype
        """
        return np.dtype(self.metadata['dtype'])

    def get_geolocation(self) -> Optional[Dict[str, Any]]:
        """Get geolocation information from HDF5 attributes.

        Attempts to extract CRS and spatial extent from dataset or
        file-level attributes. Returns None if no geolocation
        metadata is found.

        Returns
        -------
        Optional[Dict[str, Any]]
            Geolocation dict if available, else None.
        """
        geo: Dict[str, Any] = {}

        # Common HDF-EOS / CF attribute names for CRS
        for key in ('crs', 'spatial_ref', 'Projection', 'file_crs',
                     'file_spatial_ref'):
            if key in self.metadata:
                geo['crs'] = self.metadata[key]
                break

        # Common attribute names for bounds
        for key in ('bounds', 'geospatial_bounds', 'file_geospatial_bounds'):
            if key in self.metadata:
                geo['bounds'] = self.metadata[key]
                break

        # Lat/lon extent attributes (NASA Earthdata convention)
        lat_keys = ('geospatial_lat_min', 'file_geospatial_lat_min',
                     'SouthBoundingCoordinate')
        lon_keys = ('geospatial_lon_min', 'file_geospatial_lon_min',
                     'WestBoundingCoordinate')
        for lat_key in lat_keys:
            if lat_key in self.metadata:
                prefix = lat_key.rsplit('min', 1)[0] if 'min' in lat_key else ''
                max_lat_key = f"{prefix}max" if prefix else lat_key.replace(
                    'South', 'North')
                geo.setdefault('lat_min', self.metadata.get(lat_key))
                geo.setdefault('lat_max', self.metadata.get(max_lat_key))
                break
        for lon_key in lon_keys:
            if lon_key in self.metadata:
                prefix = lon_key.rsplit('min', 1)[0] if 'min' in lon_key else ''
                max_lon_key = f"{prefix}max" if prefix else lon_key.replace(
                    'West', 'East')
                geo.setdefault('lon_min', self.metadata.get(lon_key))
                geo.setdefault('lon_max', self.metadata.get(max_lon_key))
                break

        return geo if geo else None

    def close(self) -> None:
        """Close the HDF5 file handle."""
        if hasattr(self, '_file') and self._file is not None:
            self._file.close()
            self._file = None

    @staticmethod
    def list_datasets(
        filepath: Union[str, Path],
        min_ndim: int = 2,
    ) -> List[Tuple[str, Tuple[int, ...], str]]:
        """List numeric datasets in an HDF5 file.

        Utility method for exploring an HDF5 file's contents without
        opening a full reader instance.

        Parameters
        ----------
        filepath : str or Path
            Path to the HDF5 file.
        min_ndim : int
            Minimum number of dimensions. Default is 2.

        Returns
        -------
        List[Tuple[str, Tuple[int, ...], str]]
            List of ``(path, shape, dtype)`` tuples.

        Raises
        ------
        ImportError
            If h5py is not installed.

        Examples
        --------
        >>> datasets = HDF5Reader.list_datasets('product.h5')
        >>> for path, shape, dtype in datasets:
        ...     print(f"{path}: {shape} ({dtype})")
        /HDFEOS/GRIDS/NDVI: (2400, 2400) float32
        """
        if not _HAS_H5PY:
            raise ImportError(
                "h5py is required for HDF5 reading. "
                "Install with: pip install h5py"
            )
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with h5py.File(str(filepath), "r") as f:
            return _find_datasets(f, min_ndim=min_ndim)
