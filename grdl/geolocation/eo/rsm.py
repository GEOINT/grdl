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
170194430+DDSmalls@users.noreply.github.com

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
2026-04-17  Replace FD Jacobian with closed-form analytic Jacobian.
2026-04-01  Add multi-segment RSM support and ICHIPB integration.
2026-03-31  Add interpolation parameter for DEM sampling order.
2026-03-22  Update coordinate function calls to (N, M) stacked convention.
"""

# Standard library
import warnings
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


def _assert_term_counts(
    rsm: 'RSMCoefficients',
    n_num_terms: int,
    n_den_terms: int,
    kind: str,
) -> None:
    """Verify polynomial coefficient arrays match the declared powers.

    Per STDI-0002 App U, RNTRMS/CNTRMS/RDTRMS/CDTRMS declare the number
    of coefficients for each polynomial.  Mismatches between the
    declared powers and the coefficient-array length silently turn into
    truncated polynomials when ``coefs[:n_terms]`` is taken — preferable
    to fail loudly so callers can investigate the source file rather
    than receive a numerically wrong projection.
    """
    if kind == 'row':
        num_arr = rsm.row_num_coefs
        den_arr = rsm.row_den_coefs
    else:
        num_arr = rsm.col_num_coefs
        den_arr = rsm.col_den_coefs
    if num_arr.size != n_num_terms or den_arr.size != n_den_terms:
        raise ValueError(
            f"RSM {kind} polynomial term-count mismatch: "
            f"declared powers expect num={n_num_terms}, den={n_den_terms} "
            f"terms but coefficient arrays have num={num_arr.size}, "
            f"den={den_arr.size}.  Source TRE is likely malformed."
        )


def _rsm_monomials_with_partials(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    exponents: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate RSM monomials and their partials w.r.t. x and y.

    The Jacobian of an RSM polynomial in normalized coordinates is
    computed analytically — ``∂(x^i y^j z^k)/∂x = i·x^(i-1) y^j z^k`` —
    avoiding the step-size tuning and cancellation of finite differences.

    Returns
    -------
    mono, dmono_dx, dmono_dy : np.ndarray
        Shape ``(N, M)`` each.
    """
    n = len(x)
    m = len(exponents)
    mono = np.ones((n, m), dtype=np.float64)
    dmono_dx = np.zeros((n, m), dtype=np.float64)
    dmono_dy = np.zeros((n, m), dtype=np.float64)

    for idx in range(m):
        ei, ej, ek = int(exponents[idx, 0]), int(exponents[idx, 1]), int(exponents[idx, 2])

        x_pow = x ** ei if ei > 0 else np.ones(n)
        y_pow = y ** ej if ej > 0 else np.ones(n)
        z_pow = z ** ek if ek > 0 else np.ones(n)

        mono[:, idx] = x_pow * y_pow * z_pow

        if ei > 0:
            x_pow_m1 = x ** (ei - 1) if ei > 1 else np.ones(n)
            dmono_dx[:, idx] = ei * x_pow_m1 * y_pow * z_pow
        if ej > 0:
            y_pow_m1 = y ** (ej - 1) if ej > 1 else np.ones(n)
            dmono_dy[:, idx] = ej * x_pow * y_pow_m1 * z_pow

    return mono, dmono_dx, dmono_dy


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

    _assert_term_counts(rsm, row_num_mono.shape[1], row_den_mono.shape[1],
                        kind='row')

    rn_coefs = rsm.row_num_coefs
    rd_coefs = rsm.row_den_coefs

    row_num = row_num_mono @ rn_coefs
    row_den = row_den_mono @ rd_coefs

    # Column polynomial evaluation
    col_num_exp = _build_monomial_exponents(rsm.col_num_powers)
    col_den_exp = _build_monomial_exponents(rsm.col_den_powers)
    col_num_mono = _rsm_monomials(x, y, z, col_num_exp)
    col_den_mono = _rsm_monomials(x, y, z, col_den_exp)

    _assert_term_counts(rsm, col_num_mono.shape[1], col_den_mono.shape[1],
                        kind='col')

    cn_coefs = rsm.col_num_coefs
    cd_coefs = rsm.col_den_coefs

    col_num = col_num_mono @ cn_coefs
    col_den = col_den_mono @ cd_coefs

    # De-normalize to pixel coordinates
    rows = rsm.row_off + rsm.row_norm_sf * (row_num / row_den)
    cols = rsm.col_off + rsm.col_norm_sf * (col_num / col_den)

    return rows, cols


def _rsm_evaluate_with_jacobian(
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    rsm: 'RSMCoefficients',
    ground_domain_type: str = 'G',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate RSM and its analytic Jacobian w.r.t. (lat, lon).

    For ``ground_domain_type`` 'G' or 'H' the chain rule converts
    normalized→radian→degree; for 'R' (ECEF) the Jacobian is returned
    w.r.t. the two horizontal ECEF axes (callers using 'R' should
    compose with the geodetic→ECEF Jacobian before inversion).

    Returns
    -------
    rows, cols, dr_dlat, dr_dlon, dc_dlat, dc_dlon : np.ndarray
        Shape ``(N,)`` each.  Derivatives are w.r.t. degrees.
    """
    if ground_domain_type == 'R':
        ecef = geodetic_to_ecef(np.column_stack([lats, lons, heights]))
        x_raw, y_raw, z_raw = ecef[:, 0], ecef[:, 1], ecef[:, 2]
    else:
        x_raw = np.deg2rad(lons)
        y_raw = np.deg2rad(lats)
        z_raw = heights

    x = (x_raw - rsm.x_off) / rsm.x_norm_sf
    y = (y_raw - rsm.y_off) / rsm.y_norm_sf
    z = (z_raw - rsm.z_off) / rsm.z_norm_sf

    row_num_exp = _build_monomial_exponents(rsm.row_num_powers)
    row_den_exp = _build_monomial_exponents(rsm.row_den_powers)
    col_num_exp = _build_monomial_exponents(rsm.col_num_powers)
    col_den_exp = _build_monomial_exponents(rsm.col_den_powers)

    rn_m, rn_dx, rn_dy = _rsm_monomials_with_partials(x, y, z, row_num_exp)
    rd_m, rd_dx, rd_dy = _rsm_monomials_with_partials(x, y, z, row_den_exp)
    cn_m, cn_dx, cn_dy = _rsm_monomials_with_partials(x, y, z, col_num_exp)
    cd_m, cd_dx, cd_dy = _rsm_monomials_with_partials(x, y, z, col_den_exp)

    _assert_term_counts(rsm, rn_m.shape[1], rd_m.shape[1], kind='row')
    _assert_term_counts(rsm, cn_m.shape[1], cd_m.shape[1], kind='col')

    rn_coefs = rsm.row_num_coefs
    rd_coefs = rsm.row_den_coefs
    cn_coefs = rsm.col_num_coefs
    cd_coefs = rsm.col_den_coefs

    row_num = rn_m @ rn_coefs
    row_den = rd_m @ rd_coefs
    col_num = cn_m @ cn_coefs
    col_den = cd_m @ cd_coefs

    row_num_dx = rn_dx @ rn_coefs
    row_den_dx = rd_dx @ rd_coefs
    col_num_dx = cn_dx @ cn_coefs
    col_den_dx = cd_dx @ cd_coefs

    row_num_dy = rn_dy @ rn_coefs
    row_den_dy = rd_dy @ rd_coefs
    col_num_dy = cn_dy @ cn_coefs
    col_den_dy = cd_dy @ cd_coefs

    # Quotient rule in normalized coordinates
    inv_row_den_sq = 1.0 / (row_den * row_den)
    inv_col_den_sq = 1.0 / (col_den * col_den)

    d_row_norm_dx = (row_num_dx * row_den - row_num * row_den_dx) * inv_row_den_sq
    d_row_norm_dy = (row_num_dy * row_den - row_num * row_den_dy) * inv_row_den_sq
    d_col_norm_dx = (col_num_dx * col_den - col_num * col_den_dx) * inv_col_den_sq
    d_col_norm_dy = (col_num_dy * col_den - col_num * col_den_dy) * inv_col_den_sq

    rows = rsm.row_off + rsm.row_norm_sf * (row_num / row_den)
    cols = rsm.col_off + rsm.col_norm_sf * (col_num / col_den)

    # Chain rule: d/dlat = d/dy_norm · (dy_norm/dy_raw) · (dy_raw/dlat)
    # For G/H domains: y_raw = deg2rad(lat), x_raw = deg2rad(lon).
    # dy_norm/dy_raw = 1/y_norm_sf; dy_raw/dlat = π/180.
    if ground_domain_type == 'R':
        # x_raw, y_raw are ECEF components — return normalized-coord
        # derivatives; callers compose with geodetic→ECEF outside.
        lat_factor = 1.0 / rsm.y_norm_sf
        lon_factor = 1.0 / rsm.x_norm_sf
    else:
        lat_factor = np.deg2rad(1.0) / rsm.y_norm_sf
        lon_factor = np.deg2rad(1.0) / rsm.x_norm_sf

    dr_dlat = rsm.row_norm_sf * d_row_norm_dy * lat_factor
    dr_dlon = rsm.row_norm_sf * d_row_norm_dx * lon_factor
    dc_dlat = rsm.col_norm_sf * d_col_norm_dy * lat_factor
    dc_dlon = rsm.col_norm_sf * d_col_norm_dx * lon_factor

    return rows, cols, dr_dlat, dr_dlon, dc_dlat, dc_dlon


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
    ) -> List[Tuple[int, int]]:
        """Select the best RSM segment for each pixel location.

        Used by the inverse path (image → ground) where the input image
        pixel directly determines which section's polynomial is valid.
        For each pixel, prefers segments whose image-side normalization
        window contains the pixel; falls back to the nearest segment
        center when no segment matches.

        Parameters
        ----------
        rows, cols : np.ndarray
            Pixel coordinates, each shape ``(N,)``.

        Returns
        -------
        list of (int, int)
            Segment keys ``(rsn, csn)``, length N.
        """
        keys = list(self._segment_bounds.keys())
        n = len(rows)
        assignments: List[Tuple[int, int]] = [keys[0]] * n

        for i in range(n):
            r, c = rows[i], cols[i]
            best_inside_key: Optional[Tuple[int, int]] = None
            best_inside_dist = float('inf')
            best_fallback_key = keys[0]
            best_fallback_dist = float('inf')
            for key in keys:
                rmin, rmax, cmin, cmax = self._segment_bounds[key]
                seg = self._segments.segments[key]
                dist = ((r - seg.row_off) ** 2
                        + (c - seg.col_off) ** 2)
                if rmin <= r <= rmax and cmin <= c <= cmax:
                    if dist < best_inside_dist:
                        best_inside_dist = dist
                        best_inside_key = key
                if dist < best_fallback_dist:
                    best_fallback_dist = dist
                    best_fallback_key = key
            assignments[i] = (best_inside_key
                              if best_inside_key is not None
                              else best_fallback_key)

        return assignments

    def _select_segments_for_ground(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        heights: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Select the best RSM segment for each ground point.

        Used by the forward path (ground → image).  Per STDI-0002 App U,
        each segment's polynomial is fitted within its declared ground
        normalization window ``[x_off ± x_norm_sf, y_off ± y_norm_sf]``.
        The principled selector picks whichever segment's ground window
        contains the input lat/lon (or x/y/z under domain ``'R'``).
        Ties are broken by smallest normalized-coordinate magnitude.
        Points outside every window fall back to the nearest segment
        center in normalized coordinates.

        Parameters
        ----------
        lats, lons : np.ndarray
            Latitude/longitude in degrees, shape ``(N,)``.
        heights : np.ndarray
            Heights HAE in meters, shape ``(N,)``.

        Returns
        -------
        list of (int, int)
            Segment keys ``(rsn, csn)``, length N.
        """
        keys = list(self._segments.segments.keys())
        n = len(lats)
        assignments: List[Tuple[int, int]] = [keys[0]] * n

        # Per-segment normalized coordinates of every input point.
        norm_xy: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for key in keys:
            seg = self._segments.segments[key]
            if self._ground_domain == 'R':
                ecef = geodetic_to_ecef(np.column_stack([lats, lons, heights]))
                x_raw = ecef[:, 0]
                y_raw = ecef[:, 1]
            else:
                x_raw = np.deg2rad(lons)
                y_raw = np.deg2rad(lats)
            xn = (x_raw - seg.x_off) / seg.x_norm_sf
            yn = (y_raw - seg.y_off) / seg.y_norm_sf
            norm_xy[key] = (xn, yn)

        for i in range(n):
            best_inside_key: Optional[Tuple[int, int]] = None
            best_inside_d2 = float('inf')
            best_fallback_key = keys[0]
            best_fallback_d2 = float('inf')
            for key in keys:
                xn, yn = norm_xy[key]
                d2 = xn[i] * xn[i] + yn[i] * yn[i]
                if abs(xn[i]) <= 1.0 and abs(yn[i]) <= 1.0:
                    if d2 < best_inside_d2:
                        best_inside_d2 = d2
                        best_inside_key = key
                if d2 < best_fallback_d2:
                    best_fallback_d2 = d2
                    best_fallback_key = key
            assignments[i] = (best_inside_key
                              if best_inside_key is not None
                              else best_fallback_key)

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
            # Multi-segment: pick segment by ground-domain normalization
            # window, then evaluate that segment per group.  Ground-side
            # selection matches STDI-0002 App U §U.7 — each section is
            # the polynomial fit for a specific ground sub-region, so
            # the input lat/lon directly determines which polynomial is
            # the right one to evaluate.
            n = len(lats)
            rows = np.empty(n, dtype=np.float64)
            cols = np.empty(n, dtype=np.float64)
            assignments = self._select_segments_for_ground(lats, lons, h)
            groups: Dict[Tuple[int, int], List[int]] = {}
            for i, key in enumerate(assignments):
                groups.setdefault(key, []).append(i)
            for key, indices in groups.items():
                idx = np.array(indices)
                seg = self._segments.segments[key]
                r, c = _rsm_evaluate(
                    lats[idx], lons[idx], h[idx], seg, self._ground_domain)
                rows[idx] = r
                cols[idx] = c

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

        # Newton-Raphson with closed-form analytic Jacobian. The RSM
        # polynomials are ratios of monomials in normalized coordinates,
        # so partial derivatives are exact (quotient rule on ∂x^i/∂x).
        # This is both faster and numerically cleaner than finite
        # differences — no step-size tuning, no cancellation.
        max_iter = 20
        tol = 1e-3  # pixels — converged when max residual < this
        err_max = float('inf')

        for _ in range(max_iter):
            r0, c0, dr_dlat, dr_dlon, dc_dlat, dc_dlon = \
                _rsm_evaluate_with_jacobian(
                    lats, lons, h_arr, rsm, self._ground_domain)

            dr = rows - r0
            dc = cols - c0

            err = np.sqrt(dr ** 2 + dc ** 2)
            err_max = float(np.max(err))
            if err_max < tol:
                break

            det = dr_dlat * dc_dlon - dr_dlon * dc_dlat
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)

            d_lat = (dc_dlon * dr - dr_dlon * dc) / det
            d_lon = (-dc_dlat * dr + dr_dlat * dc) / det

            lats += d_lat
            lons += d_lon

        if err_max >= tol:
            warnings.warn(
                f"RSM Newton-Raphson did not converge after {max_iter} "
                f"iterations: max residual {err_max:.3g} pixels (tol "
                f"{tol:g}).  Result may be inaccurate; check that the "
                f"input pixel is within the polynomial validity window.",
                RuntimeWarning, stacklevel=3)

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
