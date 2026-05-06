# -*- coding: utf-8 -*-
"""
Geographic polygon defined by a vertex list.

Edges connecting the user-supplied vertices are interpolated according
to ``edge_mode``:

- ``'geodesic'`` (default): follow the WGS-84 geodesic between
  successive vertices. Uses :meth:`pyproj.Geod.npts` so intermediate
  points are Karney-accurate.
- ``'rhumb'`` : constant bearing line (straightforward for navigation
  context, diverges from geodesic at high latitudes).
- ``'straight'`` : linear interpolation in lat/lon. Fast, only valid
  for very small shapes -- documented as such in the parameter.

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
from typing import Tuple

# Third-party
import numpy as np
from pyproj import Geod

# GRDL internal
from grdl.shapes.base import GeographicShape


_GEOD = Geod(ellps='WGS84')
_EDGE_MODES = ('geodesic', 'rhumb', 'straight')


class GeoPolygon(GeographicShape):
    """A geographic polygon with configurable edge interpolation.

    Parameters
    ----------
    vertices_latlon : array-like
        ``(K, 2)`` lat/lon vertices, degrees. If the first and last
        vertex coincide they are treated as duplicate; uniqueness is
        not enforced otherwise.
    edge_mode : str
        ``'geodesic'`` | ``'rhumb'`` | ``'straight'``.
    closed : bool
        When True (default), the polygon wraps from the last vertex
        back to the first. When False, the shape is an open polyline
        (``is_closed=False`` -> rasterization draws an outline only).
    """

    def __init__(
        self,
        vertices_latlon: np.ndarray,
        edge_mode: str = 'geodesic',
        closed: bool = True,
    ) -> None:
        verts = np.asarray(vertices_latlon, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError(
                f"vertices_latlon must be (K, 2); got {verts.shape}"
            )
        if len(verts) < 2:
            raise ValueError("polygon requires at least 2 vertices")
        if edge_mode not in _EDGE_MODES:
            raise ValueError(
                f"edge_mode must be one of {_EDGE_MODES}; got {edge_mode!r}"
            )

        # Drop explicit duplicate closing vertex, if present.
        if closed and len(verts) >= 3 and np.allclose(verts[0], verts[-1]):
            verts = verts[:-1]

        self.vertices = verts
        self.edge_mode = edge_mode
        self.is_closed = bool(closed)

    # ----- GeographicShape contract -------------------------------------

    @property
    def center_latlon(self) -> Tuple[float, float]:
        # Simple centroid of the input vertex set. Good enough for use
        # as the ENU reference in containment tests.
        return (
            float(self.vertices[:, 0].mean()),
            float(self.vertices[:, 1].mean()),
        )

    def _perimeter_latlon(self, n: int) -> np.ndarray:
        # Distribute ``n`` intermediate samples across edges proportional
        # to each edge's length so no single edge dominates.
        k = len(self.vertices)
        edge_iter = range(k if self.is_closed else k - 1)

        if self.edge_mode == 'straight':
            edge_lengths = np.array([
                np.linalg.norm(self._edge_vector(i)) for i in edge_iter
            ])
        else:
            edge_lengths = np.array([
                self._edge_length_m(i) for i in edge_iter
            ])

        total = float(np.sum(edge_lengths))
        if total <= 0:
            raise ValueError("polygon has zero total edge length")

        # Minimum 2 samples per edge (endpoints) plus extras proportional
        # to length.
        n_edges = len(edge_lengths)
        extras = max(0, n - n_edges)
        extra_alloc = np.floor(extras * edge_lengths / total).astype(int)
        per_edge = 1 + extra_alloc  # endpoints + interior samples

        lats_out = []
        lons_out = []
        for idx, (edge_idx, samples) in enumerate(zip(edge_iter, per_edge)):
            start = self.vertices[edge_idx]
            end = self.vertices[(edge_idx + 1) % k]
            n_samples = max(1, int(samples))
            lats, lons = self._sample_edge(start, end, n_samples)
            # Append without duplicating the next edge's start vertex.
            lats_out.append(lats[:-1] if idx < n_edges - 1 else lats)
            lons_out.append(lons[:-1] if idx < n_edges - 1 else lons)

        lats_all = np.concatenate(lats_out)
        lons_all = np.concatenate(lons_out)

        # If we have an open polyline, include the final vertex; if
        # closed, drop it (it's redundant with the first).
        if self.is_closed and np.allclose(
            [lats_all[0], lons_all[0]],
            [lats_all[-1], lons_all[-1]],
        ):
            lats_all = lats_all[:-1]
            lons_all = lons_all[:-1]

        return np.column_stack([lats_all, lons_all])

    # ----- Edge helpers -------------------------------------------------

    def _edge_vector(self, i: int) -> np.ndarray:
        k = len(self.vertices)
        start = self.vertices[i]
        end = self.vertices[(i + 1) % k]
        return end - start

    def _edge_length_m(self, i: int) -> float:
        k = len(self.vertices)
        lat1, lon1 = self.vertices[i]
        lat2, lon2 = self.vertices[(i + 1) % k]
        _, _, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
        return float(dist)

    def _sample_edge(
        self,
        start: np.ndarray,
        end: np.ndarray,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample ``n_samples`` points along the edge, including endpoints."""
        lat1, lon1 = float(start[0]), float(start[1])
        lat2, lon2 = float(end[0]), float(end[1])
        if n_samples <= 1:
            return (
                np.array([lat1, lat2], dtype=np.float64),
                np.array([lon1, lon2], dtype=np.float64),
            )

        if self.edge_mode == 'geodesic':
            pts = _GEOD.npts(
                lon1, lat1, lon2, lat2, n_samples - 1, initial_idx=0,
                terminus_idx=0,
            )
            # npts returns interior points only: prepend start, append end.
            lats = np.concatenate(
                [[lat1], [p[1] for p in pts], [lat2]]
            )
            lons = np.concatenate(
                [[lon1], [p[0] for p in pts], [lon2]]
            )
            return lats, lons

        if self.edge_mode == 'rhumb':
            # Rhumb line via constant-bearing interpolation along a
            # Mercator-projected straight line. pyproj has no direct
            # rhumb sampler, so interpolate in Mercator latitude.
            ys = _mercator_y(np.array([lat1, lat2], dtype=np.float64))
            t = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
            lons = lon1 + t * (lon2 - lon1)
            y = ys[0] + t * (ys[1] - ys[0])
            lats = _inverse_mercator_y(y)
            return lats, lons

        # 'straight'
        t = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
        lats = lat1 + t * (lat2 - lat1)
        lons = lon1 + t * (lon2 - lon1)
        return lats, lons

    def __repr__(self) -> str:
        return (
            f"GeoPolygon(n={len(self.vertices)}, "
            f"edge_mode={self.edge_mode!r}, closed={self.is_closed})"
        )


def _mercator_y(lat_deg: np.ndarray) -> np.ndarray:
    """Spherical Mercator y from latitude (radians-of-arc units)."""
    lat_rad = np.radians(np.clip(lat_deg, -89.5, 89.5))
    return np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))


def _inverse_mercator_y(y: np.ndarray) -> np.ndarray:
    lat_rad = 2.0 * (np.arctan(np.exp(y)) - np.pi / 4.0)
    return np.degrees(lat_rad)
