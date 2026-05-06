# -*- coding: utf-8 -*-
"""
GeographicShape abstract base class.

Defines the contract every shape in grdl.shapes implements:
generate a dense perimeter in lat/lon, project it through an
existing :class:`~grdl.geolocation.base.Geolocation` (optionally
sampling DEM heights at every vertex), and rasterize the projected
polygon into a boolean pixel mask.

The ABC is thin on purpose; subclasses implement ``_perimeter_latlon``
and the base class provides DEM sampling, projection, adaptive
refinement, and rasterization. All coordinate conventions follow the
library standard: pixel-space ``(row, col)``, geographic ``[lat, lon]``
degrees, heights in metres above the WGS-84 ellipsoid (HAE).

Dependencies
------------
numpy (core)

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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

# Third-party
import numpy as np


if TYPE_CHECKING:
    from grdl.geolocation.base import Geolocation


class GeographicShape(ABC):
    """Abstract base for geographic shapes that project into imagery.

    Subclasses implement :meth:`_perimeter_latlon` to produce ``N``
    geodetic perimeter vertices. The base class adds:

    - :meth:`perimeter_latlon` — adds per-vertex DEM heights.
    - :meth:`to_pixels` — projects through a ``Geolocation`` with
      adaptive refinement against a pixel tolerance.
    - :meth:`rasterize` — produces a boolean mask in pixel space.
    - :meth:`contains` — point-in-shape test in geographic coordinates.

    Attributes
    ----------
    center_latlon : Tuple[float, float]
        The geographic center used as the reference for ENU-frame
        operations (adaptive refinement anchor, ellipse rotation
        reference, etc.).
    is_closed : bool
        True for filled shapes (circles, ellipses, polygons), False
        for open polylines (arcs). Controls whether rasterize permits
        a fill.
    """

    is_closed: bool = True

    # ----- Subclass contract --------------------------------------------

    @property
    @abstractmethod
    def center_latlon(self) -> Tuple[float, float]:
        """Return the shape's geographic center ``(lat_deg, lon_deg)``."""

    @abstractmethod
    def _perimeter_latlon(self, n: int) -> np.ndarray:
        """Return ``n`` perimeter vertices as an ``(n, 2)`` ``[lat, lon]`` array.

        Subclasses own the geometry. This method returns degrees only;
        DEM heights are attached later by the base class.
        """

    # ----- Perimeter with DEM heights -----------------------------------

    def perimeter_latlon(
        self,
        n: int = 256,
        geolocation: Optional['Geolocation'] = None,
        height: Optional[float] = None,
    ) -> np.ndarray:
        """Return ``(N, 3)`` geodetic perimeter with per-vertex heights.

        Parameters
        ----------
        n : int
            Number of perimeter samples to generate.
        geolocation : Geolocation, optional
            When provided and the geolocation has an attached elevation
            model, sample DEM height at every vertex. When absent or no
            DEM, heights default to 0.0 m (ellipsoid).
        height : float, optional
            When provided, every vertex is assigned this constant HAE
            (metres) and the DEM lookup is skipped. Use to render a
            ground shape as if it sat on a flat reference surface —
            e.g. to avoid terrain-induced self-intersection in slant
            geometry.

        Returns
        -------
        np.ndarray
            Shape ``(N, 3)`` with columns ``[lat_deg, lon_deg, hae_m]``.
        """
        latlon = np.asarray(self._perimeter_latlon(n), dtype=np.float64)
        if latlon.ndim != 2 or latlon.shape[1] != 2:
            raise ValueError(
                f"_perimeter_latlon must return (N, 2); got {latlon.shape}"
            )
        if height is not None:
            heights = np.full(latlon.shape[0], float(height), dtype=np.float64)
        else:
            heights = _sample_dem_heights(latlon, geolocation)
        return np.column_stack([latlon[:, 0], latlon[:, 1], heights])

    # ----- Projection into image coordinates ----------------------------

    def to_pixels(
        self,
        geolocation: 'Geolocation',
        n_initial: int = 128,
        pixel_tolerance: float = 0.5,
        max_subdivisions: int = 6,
        refine: bool = True,
        height: Optional[float] = None,
    ) -> np.ndarray:
        """Project the shape perimeter into pixel space with adaptive refinement.

        Parameters
        ----------
        geolocation : Geolocation
            Any GRDL geolocation (SICD, RPC, RSM, Affine, ...). DEM
            heights, if attached to the geolocation, are used during
            projection.
        n_initial : int
            Initial perimeter sample count before refinement.
        pixel_tolerance : float
            Maximum allowed chord-to-arc error in pixels. Edges whose
            midpoint lies further than this from the projected geodesic
            midpoint are subdivided.
        max_subdivisions : int
            Recursion depth cap. With the default 6, each initial edge
            can yield up to 2^6 = 64 additional samples.
        refine : bool
            When False, skips adaptive refinement and returns the
            projected initial perimeter only. Useful for speed when
            sub-pixel fidelity is not required.
        height : float, optional
            Constant HAE (metres) to project every perimeter vertex at.
            When provided, DEM sampling is skipped for both the initial
            perimeter and adaptive-refinement midpoints. Use this when
            rendering a ground shape on slant-range imagery to avoid
            terrain-induced self-intersection from layover.

        Returns
        -------
        np.ndarray
            Shape ``(M, 2)`` with columns ``[row, col]``. For closed
            shapes the polygon is not explicitly closed (no duplicated
            first vertex); callers that need a closed ring should
            append ``pixels[0]``.
        """
        latlons = self.perimeter_latlon(
            n=n_initial, geolocation=geolocation, height=height,
        )
        pixels = np.asarray(
            geolocation.latlon_to_image(latlons), dtype=np.float64,
        )
        if not refine:
            return pixels

        # Import here to avoid circular import (refine imports base for typing).
        from grdl.shapes.refine import adaptive_refine
        return adaptive_refine(
            shape=self,
            geolocation=geolocation,
            initial_pixels=pixels,
            initial_latlons=latlons,
            pixel_tolerance=pixel_tolerance,
            max_subdivisions=max_subdivisions,
            height=height,
        )

    # ----- Boolean mask in image space ----------------------------------

    def rasterize(
        self,
        geolocation: 'Geolocation',
        image_shape: Tuple[int, int],
        fill: bool = True,
        outline: bool = False,
        outline_thickness: int = 1,
        n_initial: int = 128,
        pixel_tolerance: float = 0.5,
    ) -> np.ndarray:
        """Rasterize the shape into a boolean mask of ``image_shape``.

        Parameters
        ----------
        geolocation : Geolocation
            Target image geolocation.
        image_shape : Tuple[int, int]
            ``(rows, cols)`` of the output mask.
        fill : bool
            Whether to include the interior of the polygon in the mask.
            Defaults to True. Forced to False for open shapes (arcs).
        outline : bool
            Whether to include the polygon perimeter in the mask.
        outline_thickness : int
            Outline thickness in pixels (1 = native skimage perimeter,
            >1 = binary dilation).
        n_initial, pixel_tolerance : see :meth:`to_pixels`.

        Returns
        -------
        np.ndarray[bool]
            Boolean mask of shape ``image_shape``. ``True`` where the
            shape lies.
        """
        pixels = self.to_pixels(
            geolocation=geolocation,
            n_initial=n_initial,
            pixel_tolerance=pixel_tolerance,
        )
        from grdl.shapes.rasterize import rasterize_polygon
        effective_fill = fill and self.is_closed
        return rasterize_polygon(
            pixels=pixels,
            image_shape=image_shape,
            fill=effective_fill,
            outline=outline or not self.is_closed,
            outline_thickness=outline_thickness,
            closed=self.is_closed,
        )

    # ----- Geographic point-in-shape (default: projected mask test) -----

    def contains(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        n: int = 512,
    ) -> np.ndarray:
        """Test whether ``(lat, lon)`` point(s) lie inside the shape.

        The default implementation projects the shape perimeter into
        a local ENU frame at :attr:`center_latlon` and tests containment
        with a matplotlib ``Path``. Subclasses with closed-form
        containment (e.g. :class:`~grdl.shapes.circle.Circle`) should
        override for speed and exactness.

        Parameters
        ----------
        lat, lon : array-like
            Geodetic coordinates in degrees. Must broadcast to the
            same shape.
        n : int
            Perimeter sample count for the fallback polygon test.

        Returns
        -------
        np.ndarray[bool]
            Shape matching the broadcasted ``(lat, lon)``.
        """
        from matplotlib.path import Path  # lazy
        from grdl.geolocation.coordinates import geodetic_to_enu

        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if lat.shape != lon.shape:
            lat, lon = np.broadcast_arrays(lat, lon)
        flat = np.column_stack([
            np.zeros_like(lat).ravel() + lat.ravel(),
            lon.ravel(),
            np.zeros_like(lat).ravel(),
        ])
        ref = np.array([self.center_latlon[0], self.center_latlon[1], 0.0])
        enu_pts = geodetic_to_enu(flat, ref)

        perim = self._perimeter_latlon(n)
        perim3 = np.column_stack([
            perim[:, 0], perim[:, 1], np.zeros(len(perim)),
        ])
        perim_enu = geodetic_to_enu(perim3, ref)

        path = Path(perim_enu[:, :2], closed=self.is_closed)
        mask = path.contains_points(enu_pts[:, :2])
        return mask.reshape(lat.shape)


# ----- Module-level helpers -------------------------------------------

def _sample_dem_heights(
    latlon: np.ndarray,
    geolocation: Optional['Geolocation'],
) -> np.ndarray:
    """Look up DEM heights at ``latlon`` when a DEM is available.

    Returns an ``(N,)`` array of HAE meters. Falls back to zeros when
    the geolocation is None or has no elevation model. Values that the
    DEM reports as NaN are replaced with 0.0 so downstream projection
    does not crash.
    """
    n = latlon.shape[0]
    heights = np.zeros(n, dtype=np.float64)
    if geolocation is None:
        return heights
    elevation = getattr(geolocation, 'elevation', None)
    if elevation is None:
        return heights
    try:
        dem_h = np.asarray(
            elevation.get_elevation(latlon[:, 0], latlon[:, 1]),
            dtype=np.float64,
        )
    except Exception:
        return heights
    if dem_h.shape != (n,):
        dem_h = np.broadcast_to(dem_h, (n,)).astype(np.float64)
    finite = np.isfinite(dem_h)
    heights[finite] = dem_h[finite]
    return heights


# ----- Batch operations (exposed through backend.py re-bind) -----------

def to_pixels_batch(
    shapes: Sequence[GeographicShape],
    geolocation: 'Geolocation',
    n_initial: int = 128,
    pixel_tolerance: float = 0.5,
    max_subdivisions: int = 6,
    parallel: Optional[str] = None,
) -> list:
    """Project many shapes in parallel. Returns a list of ``(M_i, 2)`` arrays."""
    from grdl.shapes.backend import batch_map

    def _one(shape: GeographicShape) -> np.ndarray:
        return shape.to_pixels(
            geolocation=geolocation,
            n_initial=n_initial,
            pixel_tolerance=pixel_tolerance,
            max_subdivisions=max_subdivisions,
        )

    return batch_map(_one, shapes, parallel=parallel)


def rasterize_batch(
    shapes: Sequence[GeographicShape],
    geolocation: 'Geolocation',
    image_shape: Tuple[int, int],
    fill: bool = True,
    outline: bool = False,
    outline_thickness: int = 1,
    n_initial: int = 128,
    pixel_tolerance: float = 0.5,
    parallel: Optional[str] = None,
) -> list:
    """Rasterize many shapes in parallel. Returns a list of boolean masks."""
    from grdl.shapes.backend import batch_map

    def _one(shape: GeographicShape) -> np.ndarray:
        return shape.rasterize(
            geolocation=geolocation,
            image_shape=image_shape,
            fill=fill,
            outline=outline,
            outline_thickness=outline_thickness,
            n_initial=n_initial,
            pixel_tolerance=pixel_tolerance,
        )

    return batch_map(_one, shapes, parallel=parallel)
