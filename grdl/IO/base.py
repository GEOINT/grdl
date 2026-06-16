# -*- coding: utf-8 -*-
"""
IO Base Classes - Abstract interfaces for imagery readers and writers.

Defines abstract base classes for reading and writing geospatial imagery data.
All concrete implementations (SAR, EO, etc.) must inherit from these classes.

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
2026-01-30

Modified
--------
2026-06-16  Add _ensure_2d() and _validate_single_pol() shape helpers.
2026-06-16  Add _enforce_2d flag and _assert_2d() strict 2D shape helper.
2026-04-18  Add read_band(index) convenience method to ImageReader.
2026-02-10
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from grdl.IO.models import ImageMetadata


class ImageReader(ABC):
    """
    Abstract base class for all imagery readers.

    Defines the interface for reading geospatial imagery data from various
    formats. Concrete implementations must provide format-specific logic
    for metadata extraction and data loading.

    Attributes
    ----------
    filepath : Path
        Path to the image file or directory
    metadata : ImageMetadata
        Typed image metadata extracted from the file

    Notes
    -----
    Implementations should use lazy loading where possible to avoid loading
    large datasets into memory until explicitly requested.
    """

    #: Set to True on single-channel SAR readers to guarantee that
    #: read_chip / read_full always return a 2-D (rows, cols) array.
    _enforce_2d: bool = False

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

        self.metadata: ImageMetadata
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

    def read_band(self, band: int) -> np.ndarray:
        """Read a single band as a 2-D array.

        Convenience wrapper around :meth:`read_full` that returns a
        2-D ``(rows, cols)`` array regardless of whether the underlying
        reader returns ``(bands, rows, cols)`` or ``(rows, cols, bands)``.

        Parameters
        ----------
        band : int
            0-based band index.

        Returns
        -------
        np.ndarray
            2-D band data, shape ``(rows, cols)``.

        Raises
        ------
        IndexError
            If ``band`` is out of range for the image.
        """
        n_bands = getattr(self.metadata, 'bands', 1) or 1
        if band < 0 or band >= n_bands:
            raise IndexError(
                f"Band index {band} out of range for image with "
                f"{n_bands} band(s)"
            )

        data = self.read_full(bands=[band])

        # read_full may return (1, rows, cols), (rows, cols, 1), or
        # (rows, cols) depending on the concrete reader.  Collapse any
        # singleton band axis to guarantee a 2-D result.
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            # Detect which axis is the band by matching the expected
            # (rows, cols) from get_shape().
            shape = self.get_shape()
            rows, cols = shape[0], shape[1]
            if data.shape == (1, rows, cols):
                return data[0]
            if data.shape == (rows, cols, 1):
                return data[..., 0]
            # Fall back: squeeze any singleton dim
            return np.squeeze(data)
        raise ValueError(
            f"Unexpected data shape {data.shape} returned by "
            f"{type(self).__name__}.read_full"
        )

    @staticmethod
    def _assert_2d(
        data: np.ndarray,
        context: str = '',
        strict: bool = False,
    ) -> np.ndarray:
        """Validate or coerce a single-channel array to 2-D.

        Called by concrete single-pol SAR readers (SICD, CPHD, CRSD) to
        guarantee a ``(rows, cols)`` return shape.

        Parameters
        ----------
        data : np.ndarray
            Array returned by the reader's ``read_chip`` or ``read_full``.
        context : str
            Human-readable label used in error messages, e.g.
            ``'SICDReader.read_chip'``.
        strict : bool
            If ``True``, raise :class:`ValueError` when a singleton band
            axis is present rather than squeezing silently.  Strict mode
            is used when ``_enforce_2d = True`` to catch reader bugs at
            the earliest possible point.

        Returns
        -------
        np.ndarray
            2-D ``(rows, cols)`` array.

        Raises
        ------
        ValueError
            In strict mode when the array has a singleton band axis that
            indicates a reader implementation defect.
        ValueError
            When ``data`` has more than one non-trivial band axis and
            cannot be reduced to 2-D regardless of mode.
        """
        if data.ndim == 2:
            return data

        if data.ndim == 3:
            if data.shape[0] == 1:          # (1, rows, cols)
                if strict:
                    raise ValueError(
                        f"[{context}] Strict 2D policy violation: "
                        f"single-channel array has shape {data.shape}. "
                        "Reader must return (rows, cols), not (1, rows, cols). "
                        "This is a reader implementation defect."
                    )
                return data[0]
            if data.shape[2] == 1:          # (rows, cols, 1)
                if strict:
                    raise ValueError(
                        f"[{context}] Strict 2D policy violation: "
                        f"single-channel array has shape {data.shape}. "
                        "Reader must return (rows, cols), not (rows, cols, 1). "
                        "This is a reader implementation defect."
                    )
                return data[..., 0]
            raise ValueError(
                f"[{context}] Cannot apply 2D policy to a multi-band array "
                f"with shape {data.shape}. Use bands= to select a single channel."
            )

        raise ValueError(
            f"[{context}] Unexpected array dimensionality {data.ndim} "
            f"(shape {data.shape}). Expected 2-D or 3-D."
        )

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

    # -----------------------------------------------------------------
    # Shape-contract helpers (static — callable from concrete readers)
    # -----------------------------------------------------------------

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Squeeze singleton band axes to enforce the 2D output contract.

        All single-band reads must return shape ``(rows, cols)``, not
        ``(1, rows, cols)`` or ``(rows, cols, 1)``.  This is the
        canonical GRDL output contract documented in PATTERNS.md §1.

        Concrete readers should call this at the end of
        ``read_chip()`` / ``read_full()`` before returning single-band
        data.

        Parameters
        ----------
        arr : np.ndarray
            Raw array returned by a backend read call.

        Returns
        -------
        np.ndarray
            Array squeezed to ``(rows, cols)`` when the band dimension
            is a singleton; multi-band arrays are returned unchanged.

        Raises
        ------
        ValueError
            If ``arr`` has fewer than 2 dimensions.
        """
        if arr.ndim < 2:
            raise ValueError(
                f"Read data must be at least 2D, got shape {arr.shape}"
            )
        if arr.ndim == 3:
            if arr.shape[0] == 1:        # (1, rows, cols) → (rows, cols)
                return arr[0]
            if arr.shape[2] == 1:        # (rows, cols, 1) → (rows, cols)
                return arr[..., 0]
        return arr

    @staticmethod
    def _validate_single_pol(arr: np.ndarray, context: str = "") -> None:
        """Raise if a single-polarization read is not strictly 2D.

        Intended for SAR readers that carry complex single-pol data
        where a lingering extra axis would violate the shape contract
        and cause silent broadcast errors downstream.

        Parameters
        ----------
        arr : np.ndarray
            Array to validate.
        context : str, optional
            Reader name or operation description included in the error
            message for easier diagnosis.

        Raises
        ------
        ValueError
            If ``arr.ndim != 2``.
        """
        if arr.ndim != 2:
            prefix = f"[{context}] " if context else ""
            raise ValueError(
                f"{prefix}Single-polarization read must return a 2D "
                f"array (rows, cols), got shape {arr.shape}. "
                "For multi-band reads supply an explicit bands=[…] list."
            )

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
    metadata : Optional[ImageMetadata]
        Typed image metadata to be written
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        metadata: Optional[ImageMetadata] = None
    ) -> None:
        """
        Initialize the image writer.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path where the image will be written
        metadata : Optional[ImageMetadata], default=None
            Typed metadata to include in the output file
        """
        self.filepath = Path(filepath)
        self.metadata = metadata

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
