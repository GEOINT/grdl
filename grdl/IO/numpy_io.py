# -*- coding: utf-8 -*-
"""
NumPy Writer - Write arrays to NumPy .npy and .npz formats.

Writes single arrays to ``.npy`` files and multiple named arrays to
``.npz`` archives. Optionally writes a JSON sidecar file with metadata
(shape, dtype, source file, processing parameters).

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
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.IO.base import ImageWriter
from grdl.IO.models import ImageMetadata


class NumpyWriter(ImageWriter):
    """Write arrays to NumPy .npy and .npz formats.

    Writes a single array to ``.npy`` format via ``write()``, or
    multiple named arrays to ``.npz`` format via ``write_npz()``.
    When ``write_sidecar`` metadata key is truthy (default), a JSON
    sidecar file is written alongside the array file containing shape,
    dtype, and any additional metadata.

    Parameters
    ----------
    filepath : str or Path
        Output file path (``.npy`` or ``.npz``).
    metadata : ImageMetadata, optional
        Typed metadata to include in the JSON sidecar.  The keys
        ``'shape'`` and ``'dtype'`` are populated automatically.

    Examples
    --------
    >>> from grdl.IO.numpy_io import NumpyWriter
    >>> import numpy as np
    >>> data = np.random.rand(32, 32).astype(np.float32)
    >>> with NumpyWriter('output.npy') as writer:
    ...     writer.write(data)
    """

    def write(
        self,
        data: np.ndarray,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a single array to a .npy file.

        Parameters
        ----------
        data : np.ndarray
            Array data to write.
        geolocation : Dict[str, Any], optional
            Geolocation information included in the sidecar metadata.
        """
        np.save(str(self.filepath), data)
        self._write_sidecar(data, geolocation)

    def write_npz(
        self,
        arrays: Dict[str, np.ndarray],
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write multiple named arrays to a .npz archive.

        Parameters
        ----------
        arrays : Dict[str, np.ndarray]
            Dictionary mapping array names to array data.
        geolocation : Dict[str, Any], optional
            Geolocation information included in the sidecar metadata.
        """
        np.savez(str(self.filepath), **arrays)
        # Build sidecar from first array for shape/dtype
        first_key = next(iter(arrays))
        first_array = arrays[first_key]
        sidecar = {
            'shape': list(first_array.shape),
            'dtype': str(first_array.dtype),
            'array_names': list(arrays.keys()),
        }
        if self.metadata:
            sidecar.update(self.metadata.to_dict())
        if geolocation:
            sidecar['geolocation'] = geolocation
        sidecar_path = Path(str(self.filepath) + '.json')
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar, f, indent=2, default=str)

    def write_chip(
        self,
        data: np.ndarray,
        row_start: int,
        col_start: int,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Not supported for NumPy format.

        Raises
        ------
        NotImplementedError
            Always raised; NumPy files do not support partial writes.
        """
        raise NotImplementedError(
            "NumPy .npy format does not support partial (chip) writes."
        )

    def _write_sidecar(
        self,
        data: np.ndarray,
        geolocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a JSON sidecar file with metadata.

        Parameters
        ----------
        data : np.ndarray
            The array that was written, used to extract shape and dtype.
        geolocation : Dict[str, Any], optional
            Geolocation information to include.
        """
        sidecar: Dict[str, Any] = {
            'shape': list(data.shape),
            'dtype': str(data.dtype),
        }
        if self.metadata:
            sidecar.update(self.metadata.to_dict())
        if geolocation:
            sidecar['geolocation'] = geolocation

        sidecar_path = self.filepath.with_suffix(
            self.filepath.suffix + '.json'
        )
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar, f, indent=2, default=str)
