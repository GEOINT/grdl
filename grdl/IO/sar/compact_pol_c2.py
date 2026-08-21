# -*- coding: utf-8 -*-
"""
Compact-Pol C2 Reader - PolSARpro-style covariance matrix HDF5.

Provides format detection and a reader that loads compact-pol C2 datasets
from HDF5 files into a 4-band CYX float32 cube:
``[C11, C12_real, C12_imag, C22]``.

Dependencies
------------
h5py

Author
------
GRDL Development Team

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-06-20

Modified
--------
2026-08-21
"""

from __future__ import annotations

# Standard library
import logging
from pathlib import Path
from typing import List, Optional, Union

# Third-party
import numpy as np

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

# GRDL internal
from grdl.IO.base import ImageReader
from grdl.IO.models.base import ChannelMetadata, ImageMetadata

logger = logging.getLogger(__name__)

# PolSARpro C2 dataset order: matches CompactPolDecompositionBase.execute() contract
_C2_NAMES = ('C11', 'C12_real', 'C12_imag', 'C22')


def is_compact_pol_c2_h5(path: Union[str, Path]) -> bool:
    """Return True if path is a PolSARpro-style compact-pol C2 HDF5 file."""
    path = Path(path)
    if path.suffix.lower() not in {'.h5', '.hdf5'}:
        return False
    if not _HAS_H5PY:
        return False
    try:
        with h5py.File(path, 'r') as f:
            c2 = f.get('C2')
            return c2 is not None and all(k in c2 for k in _C2_NAMES)
    except Exception:
        return False


class CompactPolC2H5Reader(ImageReader):
    """Reader for PolSARpro-style compact-pol C2 covariance matrix HDF5 files.

    Returns a 4-band CYX float32 cube ``[C11, C12_real, C12_imag, C22]``
    that plugs directly into all GRDL compact-pol decomposers via
    ``CompactPolDecompositionBase.execute()``.

    Auto-detected by ``open_sar()`` when the file contains ``/C2/C11``.
    """

    def _load_metadata(self) -> None:
        if not _HAS_H5PY:
            raise ImportError('h5py is required for CompactPolC2H5Reader')
        with h5py.File(self.filepath, 'r') as f:
            grp = f['C2']
            rows = int(grp.attrs.get('Nrow', grp['C11'].shape[0]))
            cols = int(grp.attrs.get('Ncol', grp['C11'].shape[1]))
            transmit = str(grp.attrs.get('transmit', ''))

        extras: dict = {'PolarType': 'C2', 'PolarCase': 'compact'}
        if transmit:
            extras['transmit'] = transmit

        self.metadata = ImageMetadata(
            format='COMPACT_POL_C2',
            rows=rows,
            cols=cols,
            dtype='float32',
            bands=4,
            axis_order='CYX',
            channel_metadata=[
                ChannelMetadata(index=i, name=name, role='covariance', source_indices=[i])
                for i, name in enumerate(_C2_NAMES)
            ],
            extras=extras,
        )

    def get_shape(self):
        return (self.metadata.rows, self.metadata.cols)

    def get_dtype(self):
        return np.float32

    def read_chip(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        bands: Optional[List[int]] = None,
    ) -> np.ndarray:
        if not _HAS_H5PY:
            raise ImportError('h5py is required for CompactPolC2H5Reader')
        indices = bands if bands is not None else list(range(4))
        selected = [_C2_NAMES[i] for i in indices]
        with h5py.File(self.filepath, 'r') as f:
            grp = f['C2']
            slices = [grp[name][row_start:row_end, col_start:col_end] for name in selected]
        return np.stack(slices, axis=0).astype(np.float32)

    def read_full(self, bands: Optional[List[int]] = None) -> np.ndarray:
        if not _HAS_H5PY:
            raise ImportError('h5py is required for CompactPolC2H5Reader')
        indices = bands if bands is not None else list(range(4))
        selected = [_C2_NAMES[i] for i in indices]
        with h5py.File(self.filepath, 'r') as f:
            grp = f['C2']
            slices = [grp[name][:] for name in selected]
        return np.stack(slices, axis=0).astype(np.float32)


def open_compact_pol_c2(path: Union[str, Path]) -> CompactPolC2H5Reader:
    """Open a PolSARpro-style compact-pol C2 HDF5 file."""
    return CompactPolC2H5Reader(path)
