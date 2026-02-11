# -*- coding: utf-8 -*-
"""
PNG Writer - Write uint8 grayscale and RGB arrays to PNG format.

Writes 2D grayscale or 3D RGB arrays to PNG files using Pillow.
Float inputs are auto-normalized to 0-255 with a warning.

Dependencies
------------
Pillow

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Third-party
import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# GRDL internal
from grdl.IO.base import ImageWriter
from grdl.IO.models import ImageMetadata


class PngWriter(ImageWriter):
    """Write uint8 grayscale or RGB arrays to PNG files.

    Accepts 2D ``(rows, cols)`` grayscale or 3D ``(rows, cols, 3)`` RGB
    arrays. Float arrays are auto-normalized to the ``[0, 255]`` uint8
    range with a warning.

    Parameters
    ----------
    filepath : str or Path
        Output PNG file path.
    metadata : ImageMetadata, optional
        Typed metadata (not embedded in PNG; for writer bookkeeping).

    Raises
    ------
    ImportError
        If Pillow is not installed.

    Examples
    --------
    >>> from grdl.IO.png import PngWriter
    >>> import numpy as np
    >>> data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    >>> with PngWriter('output.png') as writer:
    ...     writer.write(data)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        metadata: Optional[ImageMetadata] = None,
    ) -> None:
        if not _HAS_PIL:
            raise ImportError(
                "Pillow is required for PNG writing. "
                "Install with: pip install Pillow"
            )
        super().__init__(filepath, metadata)

    def write(
        self,
        data: np.ndarray,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write image data to a PNG file.

        Parameters
        ----------
        data : np.ndarray
            Image data with shape ``(rows, cols)`` for grayscale or
            ``(rows, cols, 3)`` for RGB. Must be uint8, or a float
            array (auto-normalized to 0-255).
        geolocation : Dict[str, Any], optional
            Not used for PNG format (no geolocation support).

        Raises
        ------
        ValueError
            If array shape is not 2D grayscale or 3D RGB.
        """
        if data.ndim == 2:
            mode = 'L'
        elif data.ndim == 3 and data.shape[2] == 3:
            mode = 'RGB'
        else:
            raise ValueError(
                f"Expected 2D grayscale (rows, cols) or 3D RGB "
                f"(rows, cols, 3), got shape {data.shape}"
            )

        if np.issubdtype(data.dtype, np.floating):
            warnings.warn(
                f"Float array (dtype={data.dtype}) auto-normalized to "
                f"uint8 [0, 255] for PNG output.",
                UserWarning,
                stacklevel=2,
            )
            dmin = data.min()
            dmax = data.max()
            if dmax - dmin > 0:
                data = ((data - dmin) / (dmax - dmin) * 255.0)
            else:
                data = np.zeros_like(data)
            data = data.astype(np.uint8)

        if data.dtype != np.uint8:
            data = data.astype(np.uint8)

        img = Image.fromarray(data, mode=mode)
        img.save(str(self.filepath))

    def write_chip(
        self,
        data: np.ndarray,
        row_start: int,
        col_start: int,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Not supported for PNG format.

        Raises
        ------
        NotImplementedError
            Always raised; PNG files do not support partial writes.
        """
        raise NotImplementedError(
            "PNG format does not support partial (chip) writes."
        )
