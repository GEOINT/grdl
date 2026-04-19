# -*- coding: utf-8 -*-
"""
RPC Geolocation - Rational Polynomial Coefficient projection for EO NITF.

Implements the RPC00B ground-to-image and image-to-ground projection
models per NGA STDI-0002 Appendix E.  The RPC model maps normalized
geodetic coordinates (latitude, longitude, height) to image pixels via
cubic rational polynomials with 20 monomial terms.

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
2026-04-01  Add ICHIPB chip transform integration.
2026-03-31  Add interpolation parameter for DEM sampling order.
2026-03-17
"""

# Standard library
from typing import Optional, Tuple, Union, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import Geolocation

if TYPE_CHECKING:
    from grdl.IO.models.eo_nitf import ICHIPBMetadata, RPCCoefficients


def _rpc_monomials(p: np.ndarray, l: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Compute the 20-term RPC monomial vector.

    The monomial ordering follows the NGA RPC00B standard::

        [1, L, P, H, L·P, L·H, P·H, L², P², H²,
         P·L·H, L³, L·P², L·H², L²·P, P³, P·H², L²·H, P²·H, H³]

    Parameters
    ----------
    p : np.ndarray
        Normalized latitude, shape ``(N,)``.
    l : np.ndarray
        Normalized longitude, shape ``(N,)``.
    h : np.ndarray
        Normalized height, shape ``(N,)``.

    Returns
    -------
    np.ndarray
        Monomial matrix, shape ``(N, 20)``.
    """
    ones = np.ones_like(p)
    return np.column_stack([
        ones,       # 0:  1
        l,          # 1:  L
        p,          # 2:  P
        h,          # 3:  H
        l * p,      # 4:  L·P
        l * h,      # 5:  L·H
        p * h,      # 6:  P·H
        l * l,      # 7:  L²
        p * p,      # 8:  P²
        h * h,      # 9:  H²
        p * l * h,  # 10: P·L·H
        l ** 3,     # 11: L³
        l * p * p,  # 12: L·P²
        l * h * h,  # 13: L·H²
        l * l * p,  # 14: L²·P
        p ** 3,     # 15: P³
        p * h * h,  # 16: P·H²
        l * l * h,  # 17: L²·H
        p * p * h,  # 18: P²·H
        h ** 3,     # 19: H³
    ])


def _rpc_monomials_dp(p: np.ndarray, l: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Partial derivatives of the RPC monomial vector w.r.t. P (latitude).

    Derivatives follow the same NGA RPC00B ordering as
    :func:`_rpc_monomials`.
    """
    z = np.zeros_like(p)
    o = np.ones_like(p)
    return np.column_stack([
        z,              # 0:  1
        z,              # 1:  L
        o,              # 2:  P         → 1
        z,              # 3:  H
        l,              # 4:  L·P       → L
        z,              # 5:  L·H
        h,              # 6:  P·H       → H
        z,              # 7:  L²
        2.0 * p,        # 8:  P²        → 2P
        z,              # 9:  H²
        l * h,          # 10: P·L·H     → L·H
        z,              # 11: L³
        2.0 * l * p,    # 12: L·P²      → 2·L·P
        z,              # 13: L·H²
        l * l,          # 14: L²·P      → L²
        3.0 * p * p,    # 15: P³        → 3P²
        h * h,          # 16: P·H²      → H²
        z,              # 17: L²·H
        2.0 * p * h,    # 18: P²·H      → 2·P·H
        z,              # 19: H³
    ])


def _rpc_monomials_dl(p: np.ndarray, l: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Partial derivatives of the RPC monomial vector w.r.t. L (longitude)."""
    z = np.zeros_like(p)
    o = np.ones_like(p)
    return np.column_stack([
        z,              # 0:  1
        o,              # 1:  L         → 1
        z,              # 2:  P
        z,              # 3:  H
        p,              # 4:  L·P       → P
        h,              # 5:  L·H       → H
        z,              # 6:  P·H
        2.0 * l,        # 7:  L²        → 2L
        z,              # 8:  P²
        z,              # 9:  H²
        p * h,          # 10: P·L·H     → P·H
        3.0 * l * l,    # 11: L³        → 3L²
        p * p,          # 12: L·P²      → P²
        h * h,          # 13: L·H²      → H²
        2.0 * l * p,    # 14: L²·P      → 2·L·P
        z,              # 15: P³
        z,              # 16: P·H²
        2.0 * l * h,    # 17: L²·H      → 2·L·H
        z,              # 18: P²·H
        z,              # 19: H³
    ])


def _rpc_evaluate(
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    rpc: 'RPCCoefficients',
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate RPC model: geodetic → image pixel coordinates.

    Parameters
    ----------
    lats : np.ndarray
        Latitudes (degrees), shape ``(N,)``.
    lons : np.ndarray
        Longitudes (degrees), shape ``(N,)``.
    heights : np.ndarray
        Heights above WGS-84 ellipsoid (meters), shape ``(N,)``.
    rpc : RPCCoefficients
        RPC coefficient set.

    Returns
    -------
    rows : np.ndarray
        Row (line) pixel coordinates, shape ``(N,)``.
    cols : np.ndarray
        Column (sample) pixel coordinates, shape ``(N,)``.
    """
    # Normalize ground coordinates
    p = (lats - rpc.lat_off) / rpc.lat_scale
    l_n = (lons - rpc.long_off) / rpc.long_scale
    h = (heights - rpc.height_off) / rpc.height_scale

    # Compute monomial vector
    mono = _rpc_monomials(p, l_n, h)

    # Evaluate rational polynomials
    line_num = mono @ rpc.line_num_coef
    line_den = mono @ rpc.line_den_coef
    samp_num = mono @ rpc.samp_num_coef
    samp_den = mono @ rpc.samp_den_coef

    # De-normalize to pixel coordinates
    rows = rpc.line_off + rpc.line_scale * (line_num / line_den)
    cols = rpc.samp_off + rpc.samp_scale * (samp_num / samp_den)

    return rows, cols


def _rpc_evaluate_with_jacobian(
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    rpc: 'RPCCoefficients',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate RPC and its analytic Jacobian w.r.t. (lat, lon).

    Returns the image pixel coordinates and the 2×2 Jacobian entries
    ``d(row,col)/d(lat,lon)`` in degrees, computed from the closed-form
    quotient-rule derivatives of the rational polynomials — avoiding the
    step-size tuning and numerical cancellation of finite differences.

    Returns
    -------
    rows : np.ndarray, shape (N,)
    cols : np.ndarray, shape (N,)
    dr_dlat : np.ndarray, shape (N,)
    dr_dlon : np.ndarray, shape (N,)
    dc_dlat : np.ndarray, shape (N,)
    dc_dlon : np.ndarray, shape (N,)
    """
    # Normalize ground coordinates
    p = (lats - rpc.lat_off) / rpc.lat_scale
    l_n = (lons - rpc.long_off) / rpc.long_scale
    h = (heights - rpc.height_off) / rpc.height_scale

    mono = _rpc_monomials(p, l_n, h)
    mono_dp = _rpc_monomials_dp(p, l_n, h)
    mono_dl = _rpc_monomials_dl(p, l_n, h)

    # Polynomials
    line_num = mono @ rpc.line_num_coef
    line_den = mono @ rpc.line_den_coef
    samp_num = mono @ rpc.samp_num_coef
    samp_den = mono @ rpc.samp_den_coef

    # Polynomial derivatives in normalized coordinates
    line_num_dp = mono_dp @ rpc.line_num_coef
    line_den_dp = mono_dp @ rpc.line_den_coef
    samp_num_dp = mono_dp @ rpc.samp_num_coef
    samp_den_dp = mono_dp @ rpc.samp_den_coef

    line_num_dl = mono_dl @ rpc.line_num_coef
    line_den_dl = mono_dl @ rpc.line_den_coef
    samp_num_dl = mono_dl @ rpc.samp_num_coef
    samp_den_dl = mono_dl @ rpc.samp_den_coef

    # Quotient rule: d(N/D)/dx = (N' * D - N * D') / D²
    inv_line_den_sq = 1.0 / (line_den * line_den)
    inv_samp_den_sq = 1.0 / (samp_den * samp_den)

    d_line_norm_dp = (line_num_dp * line_den - line_num * line_den_dp) * inv_line_den_sq
    d_line_norm_dl = (line_num_dl * line_den - line_num * line_den_dl) * inv_line_den_sq
    d_samp_norm_dp = (samp_num_dp * samp_den - samp_num * samp_den_dp) * inv_samp_den_sq
    d_samp_norm_dl = (samp_num_dl * samp_den - samp_num * samp_den_dl) * inv_samp_den_sq

    # De-normalize to pixel coordinates and physical-degree derivatives.
    # Chain rule: d/dlat = (1/lat_scale) * d/dP; d/dlon = (1/lon_scale) * d/dL.
    rows = rpc.line_off + rpc.line_scale * (line_num / line_den)
    cols = rpc.samp_off + rpc.samp_scale * (samp_num / samp_den)

    lat_scale_inv = 1.0 / rpc.lat_scale
    lon_scale_inv = 1.0 / rpc.long_scale

    dr_dlat = rpc.line_scale * d_line_norm_dp * lat_scale_inv
    dr_dlon = rpc.line_scale * d_line_norm_dl * lon_scale_inv
    dc_dlat = rpc.samp_scale * d_samp_norm_dp * lat_scale_inv
    dc_dlon = rpc.samp_scale * d_samp_norm_dl * lon_scale_inv

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
    return fi_row_off + fi_row_scale * rows, fi_col_off + fi_col_scale * cols


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
    return (rows - fi_row_off) / fi_row_scale, (cols - fi_col_off) / fi_col_scale


class RPCGeolocation(Geolocation):
    """Geolocation for EO NITF imagery using RPC00B coefficients.

    The RPC model provides a sensor-model-independent mapping between
    geodetic coordinates and image pixels via cubic rational polynomials.
    This class implements both forward (image → ground) and inverse
    (ground → image) projections.

    The **inverse** (ground → image) is a direct polynomial evaluation
    and is exact to the precision of the RPC fit.

    The **forward** (image → ground) is iterative: given a target HAE,
    Newton-Raphson iteration finds the (lat, lon) whose RPC evaluation
    matches the input (row, col).

    Parameters
    ----------
    rpc : RPCCoefficients
        RPC00B coefficient set.
    ichipb : ICHIPBMetadata, optional
        Image chip transform.  When provided, chip pixel
        coordinates are transformed to full-image coordinates
        before polynomial evaluation.
    shape : tuple of int
        Image dimensions ``(rows, cols)``.
    dem_path : str or Path, optional
        Path to DEM for terrain-corrected forward projection.
    geoid_path : str or Path, optional
        Path to geoid correction file.
    interpolation : int
        DEM interpolation order (default 3, bicubic).

    Examples
    --------
    >>> from grdl.IO.models.eo_nitf import RPCCoefficients
    >>> from grdl.geolocation.eo.rpc import RPCGeolocation
    >>> rpc = RPCCoefficients(...)  # from reader
    >>> geo = RPCGeolocation(rpc, shape=(4096, 4096))
    >>> lat, lon, h = geo.image_to_latlon(2048, 2048)
    >>> row, col = geo.latlon_to_image(lat, lon, h)
    """

    def __init__(
        self,
        rpc: 'RPCCoefficients',
        ichipb: Optional['ICHIPBMetadata'] = None,
        shape: Tuple[int, int] = (1, 1),
        dem_path: Optional[str] = None,
        geoid_path: Optional[str] = None,
        interpolation: int = 3,
    ) -> None:
        if rpc is None:
            raise ValueError("RPCCoefficients is required")
        self.rpc = rpc
        self.ichipb = ichipb
        super().__init__(
            shape, crs='WGS84', dem_path=dem_path, geoid_path=geoid_path,
            interpolation=interpolation)

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ground → image: direct RPC polynomial evaluation.

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

        rows, cols = _rpc_evaluate(lats, lons, h, self.rpc)

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

        Finds (lat, lon) such that ``RPC(lat, lon, h) ≈ (row, col)``
        using 2D Newton-Raphson with finite-difference Jacobian.

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
        rpc = self.rpc

        # Initial guess: RPC normalization center
        lats = np.full(n, rpc.lat_off)
        lons = np.full(n, rpc.long_off)
        if np.ndim(height) > 0:
            h_arr = np.asarray(height, dtype=np.float64)
        else:
            h_arr = np.full(n, float(height))

        # Newton-Raphson with closed-form analytic Jacobian derived from
        # the quotient rule on the RPC rational polynomials. This avoids
        # the step-size tuning and numerical cancellation of finite
        # differences and matches the reference implementation used by
        # OSSIM/GDAL/Orfeo.
        max_iter = 20
        tol = 1e-8  # pixels

        for _ in range(max_iter):
            r0, c0, dr_dlat, dr_dlon, dc_dlat, dc_dlon = \
                _rpc_evaluate_with_jacobian(lats, lons, h_arr, rpc)

            # Residual
            dr = rows - r0
            dc = cols - c0

            err = np.sqrt(dr ** 2 + dc ** 2)
            if np.max(err) < tol:
                break

            # 2×2 inverse Jacobian per point
            det = dr_dlat * dc_dlon - dr_dlon * dc_dlat
            # Guard against singular Jacobian near coefficient boundaries
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)

            d_lat = (dc_dlon * dr - dr_dlon * dc) / det
            d_lon = (-dc_dlat * dr + dr_dlat * dc) / det

            lats += d_lat
            lons += d_lon

        return lats, lons, h_arr.copy()

    @classmethod
    def from_coefficients(
        cls,
        rpc: 'RPCCoefficients',
        shape: Tuple[int, int] = (1, 1),
    ) -> 'RPCGeolocation':
        """Create from an RPCCoefficients object.

        Parameters
        ----------
        rpc : RPCCoefficients
            RPC coefficient set.
        shape : tuple of int
            Image dimensions ``(rows, cols)``.

        Returns
        -------
        RPCGeolocation
        """
        return cls(rpc=rpc, shape=shape)

    @classmethod
    def from_reader(cls, reader: object, **kwargs) -> 'RPCGeolocation':
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
        RPCGeolocation

        Raises
        ------
        ValueError
            If the reader's metadata has no RPC coefficients.
        """
        meta = reader.metadata
        if meta.rpc is None:
            raise ValueError(
                "Reader metadata has no RPC coefficients. "
                "Ensure the NITF file contains an RPC00B TRE."
            )
        shape = reader.get_shape()
        return cls(
            rpc=meta.rpc,
            ichipb=getattr(meta, 'ichipb', None),
            shape=shape,
            **kwargs,
        )
