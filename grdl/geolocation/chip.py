# -*- coding: utf-8 -*-
"""
Chip Geolocation - Coordinate adapter for image sub-regions.

Wraps any ``Geolocation`` object to translate between chip-local pixel
coordinates and the full-image coordinate system.  This eliminates the
need for manual ``_ChipGeolocationWrapper`` boilerplate in every script
that orthorectifies a chip.

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
2026-03-27

Modified
--------
2026-03-27
"""

# Standard library
from typing import Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import Geolocation


class ChipGeolocation(Geolocation):
    """Geolocation adapter that offsets coordinates for an image chip.

    Wraps a full-image ``Geolocation`` and applies a constant pixel
    offset so that chip-local coordinates (starting at 0, 0) are
    correctly mapped to the full-image coordinate system used by the
    underlying geolocation model.

    Parameters
    ----------
    geolocation : Geolocation
        Full-image geolocation object.
    row_offset : int or float
        Row offset of chip origin in the full image (``row_start``).
    col_offset : int or float
        Column offset of chip origin in the full image (``col_start``).
    shape : tuple of (int, int)
        Chip dimensions ``(nrows, ncols)``.

    Examples
    --------
    Extract a chip and create a chip-local geolocation::

        region = extractor.chip_at_point(center_row, center_col, 500, 500)
        chip = reader.read_chip(
            region.row_start, region.row_end,
            region.col_start, region.col_end,
        )
        chip_geo = ChipGeolocation(
            geo,
            row_offset=region.row_start,
            col_offset=region.col_start,
            shape=(region.nrows, region.ncols),
        )
        result = orthorectify(geolocation=chip_geo, source_array=chip, ...)
    """

    def __init__(
        self,
        geolocation: Geolocation,
        row_offset: Union[int, float],
        col_offset: Union[int, float],
        shape: Tuple[int, int],
    ) -> None:
        # Initialize base with chip shape, inheriting CRS from parent.
        # Do NOT pass dem_path — we inherit the parent's elevation directly.
        super().__init__(shape=shape, crs=geolocation.crs)
        self._geo = geolocation
        self._row_offset = float(row_offset)
        self._col_offset = float(col_offset)
        # Inherit elevation and DEM handling from parent geolocation
        self.elevation = geolocation.elevation
        self._handles_dem_internally = geolocation._handles_dem_internally

    @property
    def default_hae(self) -> float:
        """Delegate to parent geolocation's default HAE."""
        return self._geo.default_hae

    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Offset chip rows/cols to full-image and delegate."""
        return self._geo._image_to_latlon_array(
            rows + self._row_offset,
            cols + self._col_offset,
            height,
        )

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Delegate to parent and offset result to chip-local coords."""
        full_rows, full_cols = self._geo._latlon_to_image_array(
            lats, lons, height,
        )
        return full_rows - self._row_offset, full_cols - self._col_offset
