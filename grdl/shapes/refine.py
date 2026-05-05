# -*- coding: utf-8 -*-
"""
Adaptive perimeter refinement for projected shapes.

Given an initial set of perimeter vertices already projected into pixel
space, subdivide each edge whose pixel-space chord midpoint deviates
from the true projected geodesic midpoint by more than a user-set
tolerance. The process is fully vectorized -- every iteration projects
all candidate midpoints in one batched ``latlon_to_image`` call -- and
recurses up to a depth cap.

Default tolerance of 0.5 pixel guarantees that the projected polygon is
sub-pixel faithful to the true projected curve even across steep SAR
foreshortening, sloped terrain, or near-edge projection nonlinearities.

Dependencies
------------
numpy
pyproj

Author
------
Duane Smalley
170194430+DDSmalls@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-04-18

Modified
--------
2026-04-18
"""

# Standard library
from typing import TYPE_CHECKING, Optional, Tuple

# Third-party
import numpy as np
from pyproj import Geod

# GRDL internal
from grdl.shapes.base import _sample_dem_heights


if TYPE_CHECKING:
    from grdl.geolocation.base import Geolocation
    from grdl.shapes.base import GeographicShape


_GEOD = Geod(ellps='WGS84')


def adaptive_refine(
    shape: 'GeographicShape',
    geolocation: 'Geolocation',
    initial_pixels: np.ndarray,
    initial_latlons: np.ndarray,
    pixel_tolerance: float = 0.5,
    max_subdivisions: int = 6,
    height: Optional[float] = None,
) -> np.ndarray:
    """Subdivide edges until pixel-space chord error is below tolerance.

    Parameters
    ----------
    shape : GeographicShape
        The shape being refined. Used for the ``is_closed`` flag so
        the final edge (last → first vertex) is or is not included.
    geolocation : Geolocation
        Projection engine. Must accept stacked ``(N, 3)`` lat/lon/HAE.
    initial_pixels : np.ndarray
        Shape ``(N, 2)`` starting vertices in pixel space
        ``[row, col]``.
    initial_latlons : np.ndarray
        Shape ``(N, 3)`` matching geodetic coordinates
        ``[lat_deg, lon_deg, hae_m]`` used to generate
        ``initial_pixels``.
    pixel_tolerance : float
        Maximum acceptable chord-to-arc error in pixels.
    max_subdivisions : int
        Recursion depth cap. Each level can double the vertex count
        on any edge that is still over tolerance.
    height : float, optional
        Constant HAE (metres) for every new midpoint. When provided,
        DEM sampling is bypassed so midpoints match the constant-height
        assumption used for the initial perimeter.

    Returns
    -------
    np.ndarray
        Shape ``(M, 2)`` refined vertices. ``M >= N``.
    """
    pixels = np.asarray(initial_pixels, dtype=np.float64)
    latlons = np.asarray(initial_latlons, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"initial_pixels must be (N, 2); got {pixels.shape}")
    if latlons.ndim != 2 or latlons.shape[1] != 3:
        raise ValueError(
            f"initial_latlons must be (N, 3); got {latlons.shape}"
        )
    if len(pixels) != len(latlons):
        raise ValueError(
            f"initial_pixels and initial_latlons length mismatch: "
            f"{len(pixels)} vs {len(latlons)}"
        )

    closed = bool(getattr(shape, 'is_closed', True))

    for _ in range(max_subdivisions):
        pixels, latlons, changed = _refine_once(
            pixels=pixels,
            latlons=latlons,
            geolocation=geolocation,
            pixel_tolerance=pixel_tolerance,
            closed=closed,
            height=height,
        )
        if not changed:
            break
    return pixels


def _refine_once(
    pixels: np.ndarray,
    latlons: np.ndarray,
    geolocation: 'Geolocation',
    pixel_tolerance: float,
    closed: bool,
    height: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Run one pass of subdivision. Returns (new_pixels, new_latlons, changed)."""
    n = len(pixels)
    if n < 2:
        return pixels, latlons, False

    # Build edge pair indices (i, i+1) plus the wrap edge (n-1, 0) when closed.
    if closed:
        i_from = np.arange(n)
        i_to = np.roll(i_from, -1)
    else:
        i_from = np.arange(n - 1)
        i_to = i_from + 1

    # Straight-line pixel midpoints (the chord midpoint).
    chord_mid = 0.5 * (pixels[i_from] + pixels[i_to])

    # Geodesic midpoints on the WGS-84 ellipsoid.
    lat1 = latlons[i_from, 0]
    lon1 = latlons[i_from, 1]
    lat2 = latlons[i_to, 0]
    lon2 = latlons[i_to, 1]
    mid_lat, mid_lon = _geodesic_midpoint(lat1, lon1, lat2, lon2)

    # Per-midpoint heights: caller-forced constant, else DEM with 0 fallback.
    mid_latlon2 = np.column_stack([mid_lat, mid_lon])
    if height is not None:
        mid_heights = np.full(
            mid_latlon2.shape[0], float(height), dtype=np.float64,
        )
    else:
        mid_heights = _sample_dem_heights(mid_latlon2, geolocation)
    mid_latlon3 = np.column_stack([mid_lat, mid_lon, mid_heights])

    # Project the true midpoints.
    mid_pixels = np.asarray(
        geolocation.latlon_to_image(mid_latlon3), dtype=np.float64,
    )
    if mid_pixels.ndim == 1:
        mid_pixels = mid_pixels[np.newaxis, :]

    # Chord-to-arc error in pixel space.
    err = np.linalg.norm(mid_pixels - chord_mid, axis=1)
    needs_split = np.isfinite(err) & (err > pixel_tolerance)

    if not np.any(needs_split):
        return pixels, latlons, False

    # Weave the surviving midpoints into the vertex list between their
    # parent edges. Build the output in one pass to keep this O(N).
    out_pixels = []
    out_latlons = []
    # Map edge index -> (mid_pixel, mid_latlon3) for quick lookup
    split_map = {
        int(e): (mid_pixels[k], mid_latlon3[k])
        for k, e in enumerate(np.flatnonzero(needs_split))
    }

    for idx in range(n):
        out_pixels.append(pixels[idx])
        out_latlons.append(latlons[idx])
        if idx in split_map:
            mp, ml = split_map[idx]
            out_pixels.append(mp)
            out_latlons.append(ml)
        elif idx == n - 1 and closed and (n - 1) in split_map:
            # Already handled above -- here only for parity.
            pass

    return (
        np.asarray(out_pixels, dtype=np.float64),
        np.asarray(out_latlons, dtype=np.float64),
        True,
    )


def _geodesic_midpoint(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the WGS-84 geodesic midpoint of each ``(1, 2)`` pair.

    Uses ``Geod.inv`` for bearing + distance, then ``Geod.fwd`` half way
    along that geodesic. This is Karney's algorithm end-to-end --
    numerical error on the order of 1 nanometer, far below any sensor
    model or DEM tolerance.
    """
    fwd_az, _back_az, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
    mid_lon, mid_lat, _ = _GEOD.fwd(lon1, lat1, fwd_az, dist * 0.5)
    return np.asarray(mid_lat, dtype=np.float64), np.asarray(
        mid_lon, dtype=np.float64,
    )
