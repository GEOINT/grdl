# -*- coding: utf-8 -*-
"""
RSM Geolocation - Replacement Sensor Model projection for EO NITF.

Implements the RSM ground-to-image and image-to-ground projection
models per NGA STDI-0002 Appendix U (RSM TRE specification).  The RSM
model uses rational polynomials with variable-order terms, offering
higher fidelity than RPC for complex sensor geometries.

The **inverse** (ground → image) is a direct polynomial evaluation.
The **forward** (image → ground) is iterative since the polynomial
maps ground→image; Newton-Raphson iteration inverts it at a given HAE.

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
2026-03-17

Modified
--------
2026-04-01  Add multi-segment RSM support and ICHIPB integration.
2026-03-31  Add interpolation parameter for DEM sampling order.
2026-03-22  Update coordinate function calls to (N, M) stacked convention.
"""

# Standard library
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import Geolocation
from grdl.geolocation.coordinates import (
    geodetic_to_ecef,
    ecef_to_geodetic,
)

if TYPE_CHECKING:
    from grdl.IO.models.eo_nitf import (
        ICHIPBMetadata,
        RSMCoefficients,
        RSMIdentification,
        RSMSegmentGrid,
    )


def _build_monomial_exponents(
    max_powers: np.ndarray,
) -> np.ndarray:
    """Build exponent matrix for RSM polynomial terms.

    Generates all ``(i, j, k)`` triplets where
    ``i <= max_powers[0]``, ``j <= max_powers[1]``,
    ``k <= max_powers[2]``.

    Parameters
    ----------
    max_powers : np.ndarray
        Maximum powers ``[max_x, max_y, max_z]``, shape ``(3,)``.

    Returns
    -------
    np.ndarray
        Exponent matrix, shape ``(M, 3)`` where M is the number of
        monomial terms.
    """
    px, py, pz = int(max_powers[0]), int(max_powers[1]), int(max_powers[2])
    exponents = []
    # Per RSMPCA spec (STDI-0002 Vol 1 App U, Section 7.2, Table 4):
    # coefficients are ordered with x varying fastest, then y, then z.
    # Summation: Σ_{k=0}^{pz} Σ_{j=0}^{py} Σ_{i=0}^{px} a_{ijk} x^i y^j z^k
    # First entry = a_{000}, second = a_{100}, etc.
    for k in range(pz + 1):
        for j in range(py + 1):
            for i in range(px + 1):
                exponents.append([i, j, k])
    return np.array(exponents, dtype=np.int32)


def _rsm_monomials(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    exponents: np.ndarray,
) -> np.ndarray:
    """Evaluate RSM monomial terms at normalized ground coordinates.

    Parameters
    ----------
    x, y, z : np.ndarray
        Normalized ground coordinates, each shape ``(N,)``.
    exponents : np.ndarray
        Exponent matrix, shape ``(M, 3)``.

    Returns
    -------
    np.ndarray
        Monomial matrix, shape ``(N, M)``.
    """
    n = len(x)
    m = len(exponents)
    result = np.ones((n, m), dtype=np.float64)
    for idx in range(m):
        ei, ej, ek = exponents[idx]
        if ei > 0:
            result[:, idx] *= x ** ei
        if ej > 0:
            result[:, idx] *= y ** ej
        if ek > 0:
            result[:, idx] *= z ** ek
    return result


def _rsm_evaluate(
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    rsm: 'RSMCoefficients',
    ground_domain_type: str = 'G',
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate RSM model: geodetic → image pixel coordinates.

    Parameters
    ----------
    lats : np.ndarray
        Latitudes (degrees), shape ``(N,)``.
    lons : np.ndarray
        Longitudes (degrees), shape ``(N,)``.
    heights : np.ndarray
        Heights (meters HAE), shape ``(N,)``.
    rsm : RSMCoefficients
        RSM coefficient set.
    ground_domain_type : str
        ``'G'`` for geodetic (lat/lon/hae), ``'C'`` for cartographic,
        ``'R'`` for relative ECEF.

    Returns
    -------
    rows, cols : np.ndarray
        Image pixel coordinates, each shape ``(N,)``.
    """
    if ground_domain_type == 'R':
        # Rectangular: convert to ECEF first
        ecef = geodetic_to_ecef(np.column_stack([lats, lons, heights]))
        x_raw, y_raw, z_raw = ecef[:, 0], ecef[:, 1], ecef[:, 2]
    else:
        # Geodetic (G or H): per RSMIDA spec (STDI-0002 Vol 1 App U,
        # Section 5.3), x=longitude, y=latitude, z=height.
        # Units: radians for x and y, meters for z.
        x_raw = np.deg2rad(lons)
        y_raw = np.deg2rad(lats)
        z_raw = heights

    # Normalize ground coordinates
    x = (x_raw - rsm.x_off) / rsm.x_norm_sf
    y = (y_raw - rsm.y_off) / rsm.y_norm_sf
    z = (z_raw - rsm.z_off) / rsm.z_norm_sf

    # Row polynomial evaluation
    row_num_exp = _build_monomial_exponents(rsm.row_num_powers)
    row_den_exp = _build_monomial_exponents(rsm.row_den_powers)
    row_num_mono = _rsm_monomials(x, y, z, row_num_exp)
    row_den_mono = _rsm_monomials(x, y, z, row_den_exp)

    # Trim coefficients to match number of terms
    rn_coefs = rsm.row_num_coefs[:row_num_mono.shape[1]]
    rd_coefs = rsm.row_den_coefs[:row_den_mono.shape[1]]

    row_num = row_num_mono @ rn_coefs
    row_den = row_den_mono @ rd_coefs

    # Column polynomial evaluation
    col_num_exp = _build_monomial_exponents(rsm.col_num_powers)
    col_den_exp = _build_monomial_exponents(rsm.col_den_powers)
    col_num_mono = _rsm_monomials(x, y, z, col_num_exp)
    col_den_mono = _rsm_monomials(x, y, z, col_den_exp)

    cn_coefs = rsm.col_num_coefs[:col_num_mono.shape[1]]
    cd_coefs = rsm.col_den_coefs[:col_den_mono.shape[1]]

    col_num = col_num_mono @ cn_coefs
    col_den = col_den_mono @ cd_coefs

    # De-normalize to pixel coordinates
    rows = rsm.row_off + rsm.row_norm_sf * (row_num / row_den)
    cols = rsm.col_off + rsm.col_norm_sf * (col_num / col_den)

    return rows, cols


def _apply_ichipb_forward(
    rows: np.ndarray,
    cols: np.ndarray,
    ichipb: 'ICHIPBMetadata',
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform chip pixel coordinates to full-image coordinates.

    Parameters
    ----------
    rows, cols : np.ndarray
        Chip pixel coordinates, each shape ``(N,)``.
    ichipb : ICHIPBMetadata
        ICHIPB transform parameters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Full-image (rows, cols).
    """
    fi_row_off = ichipb.fi_row_off if ichipb.fi_row_off is not None else 0.0
    fi_col_off = ichipb.fi_col_off if ichipb.fi_col_off is not None else 0.0
    fi_row_scale = ichipb.fi_row_scale if ichipb.fi_row_scale is not None else 1.0
    fi_col_scale = ichipb.fi_col_scale if ichipb.fi_col_scale is not None else 1.0
    full_rows = fi_row_off + fi_row_scale * rows
    full_cols = fi_col_off + fi_col_scale * cols
    return full_rows, full_cols


def _apply_ichipb_inverse(
    rows: np.ndarray,
    cols: np.ndarray,
    ichipb: 'ICHIPBMetadata',
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform full-image coordinates back to chip pixel coordinates.

    Parameters
    ----------
    rows, cols : np.ndarray
        Full-image pixel coordinates, each shape ``(N,)``.
    ichipb : ICHIPBMetadata
        ICHIPB transform parameters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Chip (rows, cols).
    """
    fi_row_off = ichipb.fi_row_off if ichipb.fi_row_off is not None else 0.0
    fi_col_off = ichipb.fi_col_off if ichipb.fi_col_off is not None else 0.0
    fi_row_scale = ichipb.fi_row_scale if ichipb.fi_row_scale is not None else 1.0
    fi_col_scale = ichipb.fi_col_scale if ichipb.fi_col_scale is not None else 1.0
    chip_rows = (rows - fi_row_off) / fi_row_scale
    chip_cols = (cols - fi_col_off) / fi_col_scale
    return chip_rows, chip_cols


class RSMGeolocation(Geolocation):
    """Geolocation for EO NITF imagery using RSM polynomial coefficients.

    The RSM model provides a high-fidelity sensor-model-independent
    mapping using rational polynomials with variable-order terms.

    The **inverse** (ground → image) is a direct polynomial evaluation.
    The **forward** (image → ground) is iterative via Newton-Raphson.

    Supports multi-segment RSM where the image is partitioned into a
    grid of sections, each with its own polynomial coefficients.

    Parameters
    ----------
    rsm : RSMCoefficients
        RSM polynomial coefficients (from RSMPCA TRE).  For
        single-segment files this is the only segment.  For
        multi-segment files this is used as the default/fallback.
    rsm_id : RSMIdentification, optional
        RSM identification (from RSMIDA TRE).  Provides
        ``ground_domain_type`` for coordinate interpretation.
    rsm_segments : RSMSegmentGrid, optional
        Full grid of RSM segments for multi-section imagery.
        When provided, per-pixel segment selection is used.
    ichipb : ICHIPBMetadata, optional
        Image chip transform.  When provided, chip pixel
        coordinates are transformed to full-image coordinates
        before polynomial evaluation.
    shape : tuple of int
        Image dimensions ``(rows, cols)``.
    dem_path : str or Path, optional
        Path to DEM for terrain-corrected projection.
    geoid_path : str or Path, optional
        Path to geoid correction file.
    interpolation : int
        DEM interpolation order (default 3, bicubic).
    """

    def __init__(
        self,
        rsm: 'RSMCoefficients',
        rsm_id: Optional['RSMIdentification'] = None,
        rsm_segments: Optional['RSMSegmentGrid'] = None,
        ichipb: Optional['ICHIPBMetadata'] = None,
        shape: Tuple[int, int] = (1, 1),
        dem_path: Optional[str] = None,
        geoid_path: Optional[str] = None,
        interpolation: int = 3,
    ) -> None:
        if rsm is None:
            raise ValueError("RSMCoefficients is required")
        self.rsm = rsm
        self.rsm_id = rsm_id
        self.ichipb = ichipb
        self._ground_domain = (
            rsm_id.ground_domain_type if rsm_id and
            rsm_id.ground_domain_type else 'G')

        # Multi-segment support
        self._segments = None
        self._segment_bounds = None
        if (rsm_segments is not None
                and len(rsm_segments.segments) > 1):
            self._segments = rsm_segments
            self._segment_bounds = self._precompute_segment_bounds(
                rsm_segments)

        super().__init__(
            shape, crs='WGS84', dem_path=dem_path, geoid_path=geoid_path,
            interpolation=interpolation)

    @staticmethod
    def _precompute_segment_bounds(
        grid: 'RSMSegmentGrid',
    ) -> Dict[Tuple[int, int], Tuple[float, float, float, float]]:
        """Compute pixel bounding boxes for each RSM segment.

        Each segment's valid region is defined by its normalization
        center and scale: ``[off - sf, off + sf]`` in both row and col.

        Returns
        -------
        dict
            Mapping ``(rsn, csn)`` → ``(row_min, row_max, col_min, col_max)``.
        """
        bounds = {}
        for key, seg in grid.segments.items():
            bounds[key] = (
                seg.row_off - abs(seg.row_norm_sf),
                seg.row_off + abs(seg.row_norm_sf),
                seg.col_off - abs(seg.col_norm_sf),
                seg.col_off + abs(seg.col_norm_sf),
            )
        return bounds

    def _select_segments_for_pixels(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        """Select the best RSM segment for each pixel location.

        Returns an integer index array mapping each point to a segment.
        Points outside all segments fall back to the nearest segment.

        Parameters
        ----------
        rows, cols : np.ndarray
            Pixel coordinates, each shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Segment keys as list, length N.
        """
        keys = list(self._segment_bounds.keys())
        n = len(rows)
        assignments = [keys[0]] * n

        for i in range(n):
            r, c = rows[i], cols[i]
            best_key = keys[0]
            best_dist = float('inf')
            for key in keys:
                rmin, rmax, cmin, cmax = self._segment_bounds[key]
                if rmin <= r <= rmax and cmin <= c <= cmax:
                    # Inside this segment — compute distance to center
                    seg = self._segments.segments[key]
                    dist = ((r - seg.row_off) ** 2
                            + (c - seg.col_off) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_key = key
            assignments[i] = best_key

        return assignments

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ground → image: direct RSM polynomial evaluation.

        For multi-segment RSM, evaluates all segments and selects the
        one whose output pixel falls within its valid normalization
        window.

        Parameters
        ----------
        lats : np.ndarray
            Latitudes (degrees), shape ``(N,)``.
        lons : np.ndarray
            Longitudes (degrees), shape ``(N,)``.
        height : float or np.ndarray
            Heights above WGS-84 (meters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (rows, cols) pixel coordinates.
        """
        if np.ndim(height) == 0:
            h = np.full_like(lats, float(height))
        else:
            h = np.asarray(height, dtype=np.float64)

        if self._segments is None:
            # Single-segment: evaluate directly
            rows, cols = _rsm_evaluate(
                lats, lons, h, self.rsm, self._ground_domain)
        else:
            # Multi-segment: evaluate all, pick best per point
            n = len(lats)
            rows = np.empty(n, dtype=np.float64)
            cols = np.empty(n, dtype=np.float64)
            keys = list(self._segments.segments.keys())

            # Evaluate all segments
            seg_results = {}
            for key in keys:
                seg = self._segments.segments[key]
                r, c = _rsm_evaluate(
                    lats, lons, h, seg, self._ground_domain)
                seg_results[key] = (r, c)

            # Per-point: pick segment whose output is in-bounds
            for i in range(n):
                best_key = keys[0]
                best_dist = float('inf')
                for key in keys:
                    r, c = seg_results[key]
                    rmin, rmax, cmin, cmax = self._segment_bounds[key]
                    if rmin <= r[i] <= rmax and cmin <= c[i] <= cmax:
                        seg = self._segments.segments[key]
                        dist = ((r[i] - seg.row_off) ** 2
                                + (c[i] - seg.col_off) ** 2)
                        if dist < best_dist:
                            best_dist = dist
                            best_key = key
                r_best, c_best = seg_results[best_key]
                rows[i] = r_best[i]
                cols[i] = c_best[i]

        # Apply inverse ICHIPB: full-image → chip coordinates
        if self.ichipb is not None:
            rows, cols = _apply_ichipb_inverse(rows, cols, self.ichipb)

        return rows, cols

    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Image → ground: iterative Newton-Raphson inversion.

        Finds (lat, lon) such that ``RSM(lat, lon, h) ≈ (row, col)``.
        For multi-segment RSM, selects the segment per pixel based on
        input coordinates.

        Parameters
        ----------
        rows : np.ndarray
            Row pixel coordinates, shape ``(N,)``.
        cols : np.ndarray
            Column pixel coordinates, shape ``(N,)``.
        height : float or np.ndarray
            Height above WGS-84 (meters).  Scalar applies a constant
            height to all points.  An array of shape ``(N,)`` provides
            per-point heights for terrain-corrected projection.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) in WGS-84.
        """
        # Apply ICHIPB forward: chip → full-image coordinates
        if self.ichipb is not None:
            rows, cols = _apply_ichipb_forward(rows, cols, self.ichipb)

        n = len(rows)

        if np.ndim(height) > 0:
            h_arr = np.asarray(height, dtype=np.float64)
        else:
            h_arr = np.full(n, float(height))

        if self._segments is None:
            # Single-segment path
            return self._newton_raphson_inverse(
                rows, cols, h_arr, self.rsm)
        else:
            # Multi-segment: group points by segment, solve each group
            assignments = self._select_segments_for_pixels(rows, cols)
            lats_out = np.empty(n, dtype=np.float64)
            lons_out = np.empty(n, dtype=np.float64)

            # Group indices by segment key
            groups: Dict[Tuple[int, int], List[int]] = {}
            for i, key in enumerate(assignments):
                groups.setdefault(key, []).append(i)

            for key, indices in groups.items():
                idx = np.array(indices)
                seg = self._segments.segments[key]
                la, lo, _ = self._newton_raphson_inverse(
                    rows[idx], cols[idx], h_arr[idx], seg)
                lats_out[idx] = la
                lons_out[idx] = lo

            return lats_out, lons_out, h_arr.copy()

    def _newton_raphson_inverse(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        h_arr: np.ndarray,
        rsm: 'RSMCoefficients',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Newton-Raphson inversion for a single RSM segment.

        Parameters
        ----------
        rows, cols : np.ndarray
            Target pixel coordinates, shape ``(N,)``.
        h_arr : np.ndarray
            Heights (meters HAE), shape ``(N,)``.
        rsm : RSMCoefficients
            Segment polynomial coefficients.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) in WGS-84.
        """
        n = len(rows)

        # Initial guess from RSM normalization center
        if self._ground_domain == 'R':
            init_geo = ecef_to_geodetic(
                np.array([rsm.x_off, rsm.y_off, rsm.z_off]))
            lats = np.full(n, float(init_geo[0]))
            lons = np.full(n, float(init_geo[1]))
        else:
            lats = np.full(n, np.rad2deg(rsm.y_off))
            lons = np.full(n, np.rad2deg(rsm.x_off))

        # Finite difference steps (degrees)
        if self._ground_domain == 'R':
            dlat = max(abs(rsm.y_norm_sf) * 1e-6, 1e-8)
            dlon = max(abs(rsm.x_norm_sf) * 1e-6, 1e-8)
        else:
            dlat = max(np.rad2deg(abs(rsm.y_norm_sf)) * 1e-6, 1e-8)
            dlon = max(np.rad2deg(abs(rsm.x_norm_sf)) * 1e-6, 1e-8)

        max_iter = 20
        tol = 1e-8  # pixels

        for _ in range(max_iter):
            r0, c0 = _rsm_evaluate(
                lats, lons, h_arr, rsm, self._ground_domain)

            dr = rows - r0
            dc = cols - c0

            err = np.sqrt(dr ** 2 + dc ** 2)
            if np.max(err) < tol:
                break

            # Jacobian via finite differences
            r_dlat, c_dlat = _rsm_evaluate(
                lats + dlat, lons, h_arr, rsm, self._ground_domain)
            r_dlon, c_dlon = _rsm_evaluate(
                lats, lons + dlon, h_arr, rsm, self._ground_domain)

            dr_dlat = (r_dlat - r0) / dlat
            dc_dlat = (c_dlat - c0) / dlat
            dr_dlon = (r_dlon - r0) / dlon
            dc_dlon = (c_dlon - c0) / dlon

            det = dr_dlat * dc_dlon - dr_dlon * dc_dlat
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)

            d_lat = (dc_dlon * dr - dr_dlon * dc) / det
            d_lon = (-dc_dlat * dr + dr_dlat * dc) / det

            lats += d_lat
            lons += d_lon

        return lats, lons, h_arr.copy()

    @classmethod
    def from_reader(cls, reader: object, **kwargs) -> 'RSMGeolocation':
        """Create from an EONITFReader.

        Parameters
        ----------
        reader : EONITFReader
            An open EO NITF reader with populated metadata.
        **kwargs
            Passed to the constructor (e.g., ``dem_path``,
            ``geoid_path``, ``interpolation``).

        Returns
        -------
        RSMGeolocation

        Raises
        ------
        ValueError
            If the reader's metadata has no RSM coefficients.
        """
        meta = reader.metadata
        if meta.rsm is None:
            raise ValueError(
                "Reader metadata has no RSM coefficients. "
                "Ensure the NITF file contains RSMPCA TRE."
            )
        shape = reader.get_shape()
        return cls(
            rsm=meta.rsm,
            rsm_id=getattr(meta, 'rsm_id', None),
            rsm_segments=getattr(meta, 'rsm_segments', None),
            ichipb=getattr(meta, 'ichipb', None),
            shape=shape,
            **kwargs,
        )
