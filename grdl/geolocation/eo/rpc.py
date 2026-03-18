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
2026-03-17
"""

# Standard library
from typing import Optional, Tuple, Union, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import Geolocation

if TYPE_CHECKING:
    from grdl.IO.models.eo_nitf import RPCCoefficients


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
    shape : tuple of int
        Image dimensions ``(rows, cols)``.
    dem_path : str or Path, optional
        Path to DEM for terrain-corrected forward projection.
    geoid_path : str or Path, optional
        Path to geoid correction file.

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
        shape: Tuple[int, int] = (1, 1),
        dem_path: Optional[str] = None,
        geoid_path: Optional[str] = None,
    ) -> None:
        if rpc is None:
            raise ValueError("RPCCoefficients is required")
        self.rpc = rpc
        super().__init__(
            shape, crs='WGS84', dem_path=dem_path, geoid_path=geoid_path)

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
        return rows, cols

    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Image → ground: iterative Newton-Raphson inversion.

        At a fixed HAE, finds (lat, lon) such that
        ``RPC(lat, lon, hae) ≈ (row, col)`` using 2D Newton-Raphson
        with analytical Jacobian via finite differences.

        Parameters
        ----------
        rows : np.ndarray
            Row pixel coordinates, shape ``(N,)``.
        cols : np.ndarray
            Column pixel coordinates, shape ``(N,)``.
        height : float
            Target height above WGS-84 (meters).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (lats, lons, heights) in WGS-84.
        """
        n = len(rows)
        rpc = self.rpc
        hae = float(height)

        # Initial guess: RPC normalization center
        lats = np.full(n, rpc.lat_off)
        lons = np.full(n, rpc.long_off)
        h_arr = np.full(n, hae)

        # Finite difference step for Jacobian (degrees)
        dlat = rpc.lat_scale * 1e-6
        dlon = rpc.long_scale * 1e-6

        max_iter = 20
        tol = 1e-8  # pixels

        for _ in range(max_iter):
            # Evaluate RPC at current (lat, lon)
            r0, c0 = _rpc_evaluate(lats, lons, h_arr, rpc)

            # Residual
            dr = rows - r0
            dc = cols - c0

            # Check convergence
            err = np.sqrt(dr ** 2 + dc ** 2)
            if np.max(err) < tol:
                break

            # Jacobian via finite differences
            r_dlat, c_dlat = _rpc_evaluate(
                lats + dlat, lons, h_arr, rpc)
            r_dlon, c_dlon = _rpc_evaluate(
                lats, lons + dlon, h_arr, rpc)

            dr_dlat = (r_dlat - r0) / dlat
            dc_dlat = (c_dlat - c0) / dlat
            dr_dlon = (r_dlon - r0) / dlon
            dc_dlon = (c_dlon - c0) / dlon

            # 2x2 inverse Jacobian per point
            det = dr_dlat * dc_dlon - dr_dlon * dc_dlat
            # Guard against singular Jacobian
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)

            d_lat = (dc_dlon * dr - dr_dlon * dc) / det
            d_lon = (-dc_dlat * dr + dr_dlat * dc) / det

            lats += d_lat
            lons += d_lon

        heights = np.full(n, hae)
        return lats, lons, heights

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
    def from_reader(cls, reader: object) -> 'RPCGeolocation':
        """Create from an EONITFReader.

        Parameters
        ----------
        reader : EONITFReader
            An open EO NITF reader with populated metadata.

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
        return cls(rpc=meta.rpc, shape=shape)
