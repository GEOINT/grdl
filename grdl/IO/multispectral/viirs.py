# -*- coding: utf-8 -*-
"""
VIIRS Reader - Read VIIRS HDF5 satellite products.

Sensor-specific reader for VIIRS (Visible Infrared Imaging Radiometer
Suite) products from Suomi NPP, NOAA-20, and NOAA-21. Wraps
``HDF5Reader`` for pixel access and extracts VIIRS-specific metadata
from HDF5 file-level and dataset-level attributes into a typed
``VIIRSMetadata`` dataclass.

Supports VNP46A1 (nighttime lights), VNP13A1 (vegetation indices),
surface reflectance, and other gridded/swath products in HDF5 format.

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
from grdl.IO.models import VIIRSMetadata


def _extract_h5_attr(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Safely extract an HDF5 attribute, decoding bytes.

    Parameters
    ----------
    obj : h5py.File, h5py.Group, or h5py.Dataset
        HDF5 object with ``.attrs``.
    key : str
        Attribute name.
    default : Any
        Value to return if key is missing.

    Returns
    -------
    Any
        Attribute value (decoded if bytes), or default.
    """
    if key not in obj.attrs:
        return default
    val = obj.attrs[key]
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.item()
        return tuple(val.tolist())
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val


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


class VIIRSReader(ImageReader):
    """Read VIIRS HDF5 satellite products.

    Wraps ``h5py`` for pixel access and extracts VIIRS-specific metadata
    from HDF5 file-level and dataset-level attributes into a typed
    ``VIIRSMetadata`` dataclass.

    Parameters
    ----------
    filepath : str or Path
        Path to the VIIRS HDF5 file (.h5).
    dataset_path : str, optional
        Internal HDF5 path to the target dataset. If None,
        auto-detects the first suitable 2D+ numeric dataset.

    Attributes
    ----------
    filepath : Path
        Path to the HDF5 file.
    metadata : VIIRSMetadata
        Typed VIIRS metadata with satellite, temporal, and calibration fields.
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
    >>> from grdl.IO.multispectral import VIIRSReader
    >>> with VIIRSReader('VNP46A1.h5') as reader:
    ...     print(reader.metadata.product_short_name)
    ...     print(reader.metadata.day_night_flag)
    ...     chip = reader.read_chip(0, 256, 0, 256)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        dataset_path: Optional[str] = None,
    ) -> None:
        if not _HAS_H5PY:
            raise ImportError(
                "h5py is required for VIIRS reading. "
                "Install with: pip install h5py"
            )
        self._requested_path = dataset_path
        super().__init__(filepath)

    def _load_metadata(self) -> None:
        """Open the HDF5 file and load VIIRS-specific metadata."""
        try:
            self._file = h5py.File(str(self.filepath), "r")
        except Exception as e:
            raise ValueError(
                f"Failed to open VIIRS HDF5 file: {self.filepath}: {e}"
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

        # Extract file-level attributes
        f = self._file
        satellite_name = _extract_h5_attr(f, 'SatelliteName')
        instrument_name = _extract_h5_attr(f, 'InstrumentName')
        processing_level = _extract_h5_attr(f, 'ProcessingLevel')
        product_short_name = (
            _extract_h5_attr(f, 'ShortName')
            or _extract_h5_attr(f, 'ProductShortName')
        )
        product_long_name = (
            _extract_h5_attr(f, 'LongName')
            or _extract_h5_attr(f, 'ProductLongName')
        )
        collection_version = _extract_h5_attr(f, 'VersionId')
        start_datetime = (
            _extract_h5_attr(f, 'RangeBeginningDate')
            or _extract_h5_attr(f, 'StartTime')
        )
        start_time = _extract_h5_attr(f, 'RangeBeginningTime')
        if start_datetime and start_time and 'T' not in str(start_datetime):
            start_datetime = f"{start_datetime}T{start_time}"
        end_datetime = (
            _extract_h5_attr(f, 'RangeEndingDate')
            or _extract_h5_attr(f, 'EndTime')
        )
        end_time = _extract_h5_attr(f, 'RangeEndingTime')
        if end_datetime and end_time and 'T' not in str(end_datetime):
            end_datetime = f"{end_datetime}T{end_time}"
        production_datetime = _extract_h5_attr(f, 'ProductionDateTime')
        day_night_flag = _extract_h5_attr(f, 'DayNightFlag')
        granule_id = _extract_h5_attr(f, 'GranuleID')
        orbit_number = _extract_h5_attr(f, 'OrbitNumber')
        if orbit_number is not None:
            try:
                orbit_number = int(orbit_number)
            except (ValueError, TypeError):
                orbit_number = None

        h_tile = _extract_h5_attr(f, 'HorizontalTileNumber')
        v_tile = _extract_h5_attr(f, 'VerticalTileNumber')

        # Geospatial bounds
        lat_min = _extract_h5_attr(f, 'SouthBoundingCoordinate')
        lat_max = _extract_h5_attr(f, 'NorthBoundingCoordinate')
        lon_min = _extract_h5_attr(f, 'WestBoundingCoordinate')
        lon_max = _extract_h5_attr(f, 'EastBoundingCoordinate')
        geospatial_bounds = None
        if all(v is not None for v in (lat_min, lat_max, lon_min, lon_max)):
            try:
                geospatial_bounds = (
                    float(lat_min), float(lat_max),
                    float(lon_min), float(lon_max),
                )
            except (ValueError, TypeError):
                pass

        # Dataset-level attributes
        ds_long_name = _extract_h5_attr(ds, 'long_name')
        ds_units = _extract_h5_attr(ds, 'units')
        valid_range = _extract_h5_attr(ds, 'valid_range')
        if isinstance(valid_range, (list, tuple)) and len(valid_range) == 2:
            valid_range = (float(valid_range[0]), float(valid_range[1]))
        else:
            valid_range = None
        fill_value = _extract_h5_attr(ds, '_FillValue')
        if fill_value is not None:
            try:
                fill_value = float(fill_value)
            except (ValueError, TypeError):
                fill_value = None
        scale_factor = _extract_h5_attr(ds, 'scale_factor')
        if scale_factor is not None:
            try:
                scale_factor = float(scale_factor)
            except (ValueError, TypeError):
                scale_factor = None
        add_offset = _extract_h5_attr(ds, 'add_offset')
        if add_offset is not None:
            try:
                add_offset = float(add_offset)
            except (ValueError, TypeError):
                add_offset = None

        # Collect remaining HDF5 attrs as extras
        extras: Dict[str, Any] = {}
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
        for key, val in f.attrs.items():
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

        self.metadata = VIIRSMetadata(
            format='VIIRS',
            rows=rows,
            cols=cols,
            dtype=str(ds.dtype),
            bands=bands,
            extras=extras,
            satellite_name=satellite_name,
            instrument_name=instrument_name,
            processing_level=processing_level,
            product_short_name=product_short_name,
            product_long_name=product_long_name,
            collection_version=collection_version,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            production_datetime=production_datetime,
            day_night_flag=day_night_flag,
            granule_id=granule_id,
            orbit_number=orbit_number,
            horizontal_tile_number=str(h_tile) if h_tile is not None else None,
            vertical_tile_number=str(v_tile) if v_tile is not None else None,
            geospatial_bounds=geospatial_bounds,
            dataset_long_name=ds_long_name,
            dataset_units=ds_units,
            valid_range=valid_range,
            fill_value=fill_value,
            scale_factor=scale_factor,
            add_offset=add_offset,
            dataset_path=self.dataset_path,
        )

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Read a spatial chip from the VIIRS dataset.

        Uses HDF5 hyperslab selection for efficient partial reads.

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
        if row_end > self.metadata.rows or col_end > self.metadata.cols:
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
        """Read the entire VIIRS dataset.

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
        if self.metadata.bands == 1:
            return (self.metadata.rows, self.metadata.cols)
        return (self.metadata.rows, self.metadata.cols, self.metadata.bands)

    def get_dtype(self) -> np.dtype:
        """Get the data type of the dataset.

        Returns
        -------
        np.dtype
        """
        return np.dtype(self.metadata.dtype)

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
        """List numeric datasets in a VIIRS HDF5 file.

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
        """
        if not _HAS_H5PY:
            raise ImportError(
                "h5py is required for VIIRS reading. "
                "Install with: pip install h5py"
            )
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with h5py.File(str(filepath), "r") as f:
            return _find_datasets(f, min_ndim=min_ndim)
