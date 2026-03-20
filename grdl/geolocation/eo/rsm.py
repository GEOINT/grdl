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
2026-03-19
"""

# Standard library
from typing import Optional, Tuple, Union, TYPE_CHECKING

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import Geolocation
from grdl.geolocation.coordinates import (
    geodetic_to_ecef,
    ecef_to_geodetic,
)

if TYPE_CHECKING:
    from grdl.IO.models.eo_nitf import RSMCoefficients, RSMIdentification


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
        ex, ey, ez = geodetic_to_ecef(lats, lons, heights)
        x_raw, y_raw, z_raw = ex, ey, ez
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


class RSMGeolocation(Geolocation):
    """Geolocation for EO NITF imagery using RSM polynomial coefficients.

    The RSM model provides a high-fidelity sensor-model-independent
    mapping using rational polynomials with variable-order terms.

    The **inverse** (ground → image) is a direct polynomial evaluation.
    The **forward** (image → ground) is iterative via Newton-Raphson.

    Parameters
    ----------
    rsm : RSMCoefficients
        RSM polynomial coefficients (from RSMPCA TRE).
    rsm_id : RSMIdentification, optional
        RSM identification (from RSMIDA TRE).  Provides
        ``ground_domain_type`` for coordinate interpretation.
    shape : tuple of int
        Image dimensions ``(rows, cols)``.
    dem_path : str or Path, optional
        Path to DEM for terrain-corrected projection.
    geoid_path : str or Path, optional
        Path to geoid correction file.
    """

    def __init__(
        self,
        rsm: 'RSMCoefficients',
        rsm_id: Optional['RSMIdentification'] = None,
        shape: Tuple[int, int] = (1, 1),
        dem_path: Optional[str] = None,
        geoid_path: Optional[str] = None,
    ) -> None:
        if rsm is None:
            raise ValueError("RSMCoefficients is required")
        self.rsm = rsm
        self.rsm_id = rsm_id
        self._ground_domain = (
            rsm_id.ground_domain_type if rsm_id and
            rsm_id.ground_domain_type else 'G')
        super().__init__(
            shape, crs='WGS84', dem_path=dem_path, geoid_path=geoid_path)

    def _latlon_to_image_array(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        height: Union[float, np.ndarray] = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ground → image: direct RSM polynomial evaluation.

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

        rows, cols = _rsm_evaluate(
            lats, lons, h, self.rsm, self._ground_domain)
        return rows, cols

    def _image_to_latlon_array(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        height: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Image → ground: iterative Newton-Raphson inversion.

        At a fixed HAE, finds (lat, lon) such that
        ``RSM(lat, lon, hae) ≈ (row, col)``.

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
        rsm = self.rsm
        hae = float(height)

        # Initial guess from RSM normalization center
        if self._ground_domain == 'R':
            # ECEF center — convert to geodetic for initial guess
            init_lats, init_lons, init_h = ecef_to_geodetic(
                np.array([rsm.x_off]),
                np.array([rsm.y_off]),
                np.array([rsm.z_off]),
            )
            lats = np.full(n, float(init_lats[0]))
            lons = np.full(n, float(init_lons[0]))
        else:
            # Geodetic (G or H): x_off = longitude (radians),
            # y_off = latitude (radians). Convert to degrees.
            lats = np.full(n, np.rad2deg(rsm.y_off))
            lons = np.full(n, np.rad2deg(rsm.x_off))

        h_arr = np.full(n, hae)

        # Finite difference steps (degrees)
        if self._ground_domain == 'R':
            dlat = max(abs(rsm.y_norm_sf) * 1e-6, 1e-8)
            dlon = max(abs(rsm.x_norm_sf) * 1e-6, 1e-8)
        else:
            # Geodetic: norm_sf is in radians, convert to degree scale
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

        heights = np.full(n, hae)
        return lats, lons, heights

    @classmethod
    def from_reader(cls, reader: object) -> 'RSMGeolocation':
        """Create from an EONITFReader.

        Parameters
        ----------
        reader : EONITFReader
            An open EO NITF reader with populated metadata.

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
            shape=shape,
        )
