# -*- coding: utf-8 -*-
"""
IO Base Classes - Abstract interfaces for imagery readers and writers.

Defines abstract base classes for reading and writing geospatial imagery data.
All concrete implementations (SAR, EO, etc.) must inherit from these classes.

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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np


class ImageReader(ABC):
    """
    Abstract base class for all imagery readers.

    Defines the interface for reading geospatial imagery data from various
    formats. Concrete implementations must provide format-specific logic
    for metadata extraction, data loading, and geolocation.

    Attributes
    ----------
    filepath : Path
        Path to the image file or directory
    metadata : Dict[str, Any]
        Image metadata extracted from the file

    Notes
    -----
    Implementations should use lazy loading where possible to avoid loading
    large datasets into memory until explicitly requested.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        """
        Initialize the image reader.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the image file or directory

        Raises
        ------
        FileNotFoundError
            If the specified filepath does not exist
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self.metadata: Dict[str, Any] = {}
        self._load_metadata()

    @abstractmethod
    def _load_metadata(self) -> None:
        """
        Load metadata from the image file.

        This method should populate self.metadata with format-specific
        metadata including image dimensions, data type, geolocation info,
        sensor parameters, etc.
        """
        pass

    @abstractmethod
    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Read a spatial subset (chip) of the image.

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
            List of band indices to read. If None, read all bands.

        Returns
        -------
        np.ndarray
            Image data with shape (rows, cols) for single-band or
            (rows, cols, bands) for multi-band imagery

        Raises
        ------
        ValueError
            If indices are out of bounds or invalid
        """
        pass

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        """
        Read the entire image.

        Parameters
        ----------
        bands : Optional[List[int]], default=None
            List of band indices to read. If None, read all bands.

        Returns
        -------
        np.ndarray
            Full image data

        Notes
        -----
        Use with caution for large images as this loads the entire
        dataset into memory. Subclasses may override for more efficient
        full-image reads.
        """
        shape = self.get_shape()
        rows, cols = shape[0], shape[1]
        return self.read_chip(0, rows, 0, cols, bands=bands)

    @abstractmethod
    def get_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the image.

        Returns
        -------
        Tuple[int, ...]
            Shape tuple (rows, cols) for single-band or
            (rows, cols, bands) for multi-band imagery
        """
        pass

    @abstractmethod
    def get_dtype(self) -> np.dtype:
        """
        Get the data type of the image.

        Returns
        -------
        np.dtype
            NumPy data type of the image pixels
        """
        pass

    @abstractmethod
    def get_geolocation(self) -> Optional[Dict[str, Any]]:
        """
        Get geolocation information for the image.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing geolocation metadata such as:
            - 'crs': Coordinate reference system
            - 'transform': Affine transformation matrix
            - 'bounds': Geographic bounds (min_x, min_y, max_x, max_y)
            - 'corner_coords': Corner coordinates
            Returns None if no geolocation info available
        """
        pass

    def close(self) -> None:
        """
        Close the reader and release resources.

        Default implementation does nothing. Override if the reader
        maintains open file handles or other resources.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


class ImageWriter(ABC):
    """
    Abstract base class for all imagery writers.

    Defines the interface for writing geospatial imagery data to various
    formats. Concrete implementations must provide format-specific logic
    for encoding data and metadata.

    Attributes
    ----------
    filepath : Path
        Path where the image will be written
    metadata : Dict[str, Any]
        Image metadata to be written
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the image writer.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path where the image will be written
        metadata : Optional[Dict[str, Any]], default=None
            Metadata to include in the output file
        """
        self.filepath = Path(filepath)
        self.metadata = metadata or {}

    @abstractmethod
    def write(
        self,
        data: np.ndarray,
        geolocation: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Write image data to file.

        Parameters
        ----------
        data : np.ndarray
            Image data to write
        geolocation : Optional[Dict[str, Any]], default=None
            Geolocation information to include in the output

        Raises
        ------
        ValueError
            If data format is incompatible with the output format
        IOError
            If writing fails
        """
        pass

    @abstractmethod
    def write_chip(
        self,
        data: np.ndarray,
        row_start: int,
        col_start: int,
        geolocation: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Write a spatial subset to an existing file.

        Parameters
        ----------
        data : np.ndarray
            Image chip data to write
        row_start : int
            Starting row index in the output file
        col_start : int
            Starting column index in the output file
        geolocation : Optional[Dict[str, Any]], default=None
            Geolocation information for the chip

        Raises
        ------
        ValueError
            If chip location is out of bounds
        IOError
            If writing fails
        """
        pass

    def close(self) -> None:
        """
        Close the writer and release resources.

        Default implementation does nothing. Override if the writer
        maintains open file handles or other resources.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


class CatalogInterface(ABC):
    """
    Abstract base class for image cataloging and discovery.

    Defines the interface for discovering, indexing, and querying
    collections of imagery files.

    Attributes
    ----------
    search_path : Path
        Root directory to search for imagery
    """

    def __init__(self, search_path: Union[str, Path]) -> None:
        """
        Initialize the catalog interface.

        Parameters
        ----------
        search_path : Union[str, Path]
            Root directory to search for imagery

        Raises
        ------
        NotADirectoryError
            If search_path is not a directory
        """
        self.search_path = Path(search_path)
        if not self.search_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.search_path}")

    @abstractmethod
    def discover_images(
        self,
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Path]:
        """
        Discover image files in the search path.

        Parameters
        ----------
        extensions : Optional[List[str]], default=None
            File extensions to search for (e.g., ['.tif', '.ntf']).
            If None, search for all known image formats.
        recursive : bool, default=True
            Whether to search subdirectories recursively

        Returns
        -------
        List[Path]
            List of discovered image file paths
        """
        pass

    @abstractmethod
    def get_metadata_summary(
        self,
        image_paths: List[Path]
    ) -> List[Dict[str, Any]]:
        """
        Extract metadata summary from multiple images.

        Parameters
        ----------
        image_paths : List[Path]
            List of image file paths

        Returns
        -------
        List[Dict[str, Any]]
            List of metadata dictionaries, one per image
        """
        pass

    @abstractmethod
    def find_overlapping(
        self,
        reference_bounds: Tuple[float, float, float, float],
        image_paths: List[Path]
    ) -> List[Path]:
        """
        Find images that overlap with a reference bounding box.

        Parameters
        ----------
        reference_bounds : Tuple[float, float, float, float]
            Reference bounding box as (min_x, min_y, max_x, max_y)
        image_paths : List[Path]
            List of image file paths to check

        Returns
        -------
        List[Path]
            List of image paths that overlap with the reference bounds
        """
        pass
