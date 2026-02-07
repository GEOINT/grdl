# -*- coding: utf-8 -*-
"""
Tiler - Split images into overlapping tiles with configurable stride.

Provides tile extraction and reassembly for 2D and 3D imagery arrays.
Tiles can overlap (stride < tile_size) and edge tiles are padded as needed.
Reassembly via ``untile()`` averages overlapping regions for seamless
reconstruction.

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
2026-02-06

Modified
--------
2026-02-06
"""

# Standard library
from typing import Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.data_prep.base import _normalize_pair, _validate_image


class Tiler:
    """Split images into overlapping tiles with configurable stride.

    Tiles are extracted row-major from top-left to bottom-right. When the
    image dimensions are not evenly divisible by the stride, edge tiles
    are padded using the specified pad mode.

    Parameters
    ----------
    tile_size : int or Tuple[int, int]
        (tile_rows, tile_cols). If int, square tiles.
    stride : int or Tuple[int, int], optional
        (stride_rows, stride_cols). Defaults to tile_size (no overlap).
    pad_mode : str
        numpy pad mode for edge tiles. Default ``'constant'``.
    pad_value : float
        Value for constant padding. Default ``0.0``.

    Raises
    ------
    TypeError
        If tile_size or stride is not int or Tuple[int, int].
    ValueError
        If tile_size or stride has non-positive elements, or if stride
        exceeds tile_size in any dimension.

    Examples
    --------
    >>> import numpy as np
    >>> from grdl.data_prep import Tiler
    >>> image = np.random.rand(100, 100)
    >>> tiler = Tiler(tile_size=32, stride=16)
    >>> tiles = tiler.tile(image)
    >>> tiles.shape
    (49, 32, 32)
    >>> restored = tiler.untile(tiles, image.shape)
    >>> np.allclose(image, restored)
    True
    """

    def __init__(
        self,
        tile_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        pad_mode: str = 'constant',
        pad_value: float = 0.0,
    ) -> None:
        self._tile_size = _normalize_pair(tile_size, 'tile_size')
        if stride is None:
            self._stride = self._tile_size
        else:
            self._stride = _normalize_pair(stride, 'stride')
        if self._stride[0] > self._tile_size[0] or \
                self._stride[1] > self._tile_size[1]:
            raise ValueError(
                f"stride {self._stride} must not exceed "
                f"tile_size {self._tile_size}"
            )
        self._pad_mode = pad_mode
        self._pad_value = pad_value

    @property
    def tile_size(self) -> Tuple[int, int]:
        """The (tile_rows, tile_cols) dimensions.

        Returns
        -------
        Tuple[int, int]
            Tile dimensions.
        """
        return self._tile_size

    @property
    def stride(self) -> Tuple[int, int]:
        """The (stride_rows, stride_cols) step sizes.

        Returns
        -------
        Tuple[int, int]
            Stride dimensions.
        """
        return self._stride

    def _spatial_shape(self, image_shape: Tuple[int, ...]) -> Tuple[int, int]:
        """Extract the spatial (rows, cols) from a 2D or 3D shape.

        Parameters
        ----------
        image_shape : Tuple[int, ...]
            Shape of a 2D ``(rows, cols)`` or 3D ``(bands, rows, cols)``
            array.

        Returns
        -------
        Tuple[int, int]
            (rows, cols) spatial dimensions.

        Raises
        ------
        ValueError
            If shape is not 2D or 3D.
        """
        if len(image_shape) == 2:
            return (image_shape[0], image_shape[1])
        if len(image_shape) == 3:
            return (image_shape[1], image_shape[2])
        raise ValueError(
            f"image_shape must be 2D or 3D, got {len(image_shape)}D"
        )

    def _compute_grid(
        self, rows: int, cols: int
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Compute the tile origin grid.

        Parameters
        ----------
        rows : int
            Number of image rows.
        cols : int
            Number of image columns.

        Returns
        -------
        row_starts : np.ndarray
            1D array of row starting positions.
        col_starts : np.ndarray
            1D array of column starting positions.
        n_row_tiles : int
            Number of tile rows.
        n_col_tiles : int
            Number of tile columns.
        """
        tr, tc = self._tile_size
        sr, sc = self._stride

        # Compute number of tiles needed to cover each dimension.
        # At least 1 tile; then one more per stride step needed beyond the
        # first tile.
        if rows <= tr:
            n_row_tiles = 1
        else:
            n_row_tiles = 1 + int(np.ceil((rows - tr) / sr))

        if cols <= tc:
            n_col_tiles = 1
        else:
            n_col_tiles = 1 + int(np.ceil((cols - tc) / sc))

        row_starts = np.arange(n_row_tiles) * sr
        col_starts = np.arange(n_col_tiles) * sc

        return row_starts, col_starts, n_row_tiles, n_col_tiles

    def tile_positions(self, image_shape: Tuple[int, ...]) -> np.ndarray:
        """Compute tile origin positions without allocating tile data.

        Parameters
        ----------
        image_shape : Tuple[int, ...]
            Shape of a 2D ``(rows, cols)`` or 3D ``(bands, rows, cols)``
            array.

        Returns
        -------
        np.ndarray
            ``(n_tiles, 2)`` array of ``(row_start, col_start)`` positions,
            ordered row-major.

        Raises
        ------
        ValueError
            If image_shape is not 2D or 3D.
        """
        rows, cols = self._spatial_shape(image_shape)
        row_starts, col_starts, n_row, n_col = self._compute_grid(rows, cols)

        # Build (n_tiles, 2) grid of origins using meshgrid
        rr, cc = np.meshgrid(row_starts, col_starts, indexing='ij')
        positions = np.column_stack([rr.ravel(), cc.ravel()])
        return positions

    def tile(self, image: np.ndarray) -> np.ndarray:
        """Split image into tiles.

        Parameters
        ----------
        image : np.ndarray
            2D ``(rows, cols)`` or 3D ``(bands, rows, cols)`` array.

        Returns
        -------
        np.ndarray
            For 2D input: ``(n_tiles, tile_rows, tile_cols)`` array.
            For 3D input: ``(n_tiles, bands, tile_rows, tile_cols)`` array.

        Raises
        ------
        TypeError
            If image is not a numpy ndarray.
        ValueError
            If image is not 2D or 3D.
        """
        _validate_image(image)

        is_3d = image.ndim == 3
        if is_3d:
            bands = image.shape[0]
            rows, cols = image.shape[1], image.shape[2]
        else:
            rows, cols = image.shape[0], image.shape[1]

        tr, tc = self._tile_size
        row_starts, col_starts, n_row, n_col = self._compute_grid(rows, cols)
        n_tiles = n_row * n_col

        # Determine how much padding is needed
        max_row_end = int(row_starts[-1]) + tr
        max_col_end = int(col_starts[-1]) + tc
        pad_rows = max(0, max_row_end - rows)
        pad_cols = max(0, max_col_end - cols)

        # Pad image if necessary
        if pad_rows > 0 or pad_cols > 0:
            if is_3d:
                pad_width = ((0, 0), (0, pad_rows), (0, pad_cols))
            else:
                pad_width = ((0, pad_rows), (0, pad_cols))
            if self._pad_mode == 'constant':
                padded = np.pad(
                    image, pad_width, mode='constant',
                    constant_values=self._pad_value
                )
            else:
                padded = np.pad(image, pad_width, mode=self._pad_mode)
        else:
            padded = image

        # Extract tiles using vectorized slicing
        positions = self.tile_positions(image.shape)

        if is_3d:
            tiles = np.empty((n_tiles, bands, tr, tc), dtype=image.dtype)
            for i, (rs, cs) in enumerate(positions):
                tiles[i] = padded[:, rs:rs + tr, cs:cs + tc]
        else:
            tiles = np.empty((n_tiles, tr, tc), dtype=image.dtype)
            for i, (rs, cs) in enumerate(positions):
                tiles[i] = padded[rs:rs + tr, cs:cs + tc]

        return tiles

    def untile(
        self, tiles: np.ndarray, image_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Reassemble tiles back into a full image.

        Overlapping regions are averaged. The returned array matches the
        original ``image_shape`` exactly (padding is removed).

        Parameters
        ----------
        tiles : np.ndarray
            For 2D images: ``(n_tiles, tile_rows, tile_cols)`` array.
            For 3D images: ``(n_tiles, bands, tile_rows, tile_cols)`` array.
        image_shape : Tuple[int, ...]
            Original image shape: ``(rows, cols)`` or
            ``(bands, rows, cols)``.

        Returns
        -------
        np.ndarray
            Reconstructed image with shape ``image_shape``.

        Raises
        ------
        ValueError
            If image_shape is not 2D or 3D, or if tile count does not
            match expected grid.
        """
        is_3d = len(image_shape) == 3
        if is_3d:
            bands = image_shape[0]
            rows, cols = image_shape[1], image_shape[2]
        elif len(image_shape) == 2:
            rows, cols = image_shape[0], image_shape[1]
        else:
            raise ValueError(
                f"image_shape must be 2D or 3D, got {len(image_shape)}D"
            )

        tr, tc = self._tile_size
        row_starts, col_starts, n_row, n_col = self._compute_grid(rows, cols)
        n_tiles = n_row * n_col

        if tiles.shape[0] != n_tiles:
            raise ValueError(
                f"Expected {n_tiles} tiles for image_shape {image_shape}, "
                f"got {tiles.shape[0]}"
            )

        # Compute padded canvas dimensions
        max_row_end = int(row_starts[-1]) + tr
        max_col_end = int(col_starts[-1]) + tc
        canvas_rows = max(rows, max_row_end)
        canvas_cols = max(cols, max_col_end)

        # Accumulate tile values and counts for averaging overlaps
        if is_3d:
            accumulator = np.zeros(
                (bands, canvas_rows, canvas_cols), dtype=np.float64
            )
            counts = np.zeros(
                (1, canvas_rows, canvas_cols), dtype=np.float64
            )
        else:
            accumulator = np.zeros(
                (canvas_rows, canvas_cols), dtype=np.float64
            )
            counts = np.zeros(
                (canvas_rows, canvas_cols), dtype=np.float64
            )

        positions = self.tile_positions(image_shape)

        for i, (rs, cs) in enumerate(positions):
            if is_3d:
                accumulator[:, rs:rs + tr, cs:cs + tc] += tiles[i].astype(
                    np.float64
                )
                counts[0, rs:rs + tr, cs:cs + tc] += 1.0
            else:
                accumulator[rs:rs + tr, cs:cs + tc] += tiles[i].astype(
                    np.float64
                )
                counts[rs:rs + tr, cs:cs + tc] += 1.0

        # Average overlapping regions (counts is always >= 1 in tiled area)
        if is_3d:
            # Broadcast counts across bands
            result = accumulator / counts
            result = result[:, :rows, :cols]
        else:
            result = accumulator / counts
            result = result[:rows, :cols]

        return result.astype(tiles.dtype)

    def __repr__(self) -> str:
        return (
            f"Tiler(tile_size={self._tile_size}, stride={self._stride}, "
            f"pad_mode='{self._pad_mode}', pad_value={self._pad_value})"
        )
