# -*- coding: utf-8 -*-
"""
SIDD Builder - Construct SIDD product metadata from an output grid.

Bridges the orthorectification output grids in
``grdl.image_processing.ortho`` to the typed ``SIDDMetadata`` container
that :class:`~grdl.IO.sar.sidd_writer.SIDDWriter` writes.  Given any
grid satisfying ``OutputGridProtocol`` and the shape of the product
raster, this module produces a complete, schema-valid SIDD metadata
structure.

The grid determines which SIDD projection is emitted:

===========================  ==============================  ==========
Grid                         SIDD projection                 Exactness
===========================  ==============================  ==========
``ENUGrid``                  ``PlaneProjection`` (PGD)       exact
``RotatedENUGrid``           ``PlaneProjection`` (PGD)       exact
``GeographicGrid``           ``GeographicProjection`` (GGD)  exact
anything else                ``PolynomialProjection``        fitted
===========================  ==============================  ==========

ENU grids are planar by construction -- ``image_to_latlon`` evaluates on
the up=0 tangent plane at the grid reference point -- so the ECEF
reference point, row/column unit vectors and sample spacing describe the
grid without approximation.  ``GeographicGrid`` is a constant angular
grid, which is exactly what GGD encodes (arc-seconds per pixel, latitude
decreasing with row).  Any other grid -- UTM, Web Mercator, or a custom
implementation -- is captured by least-squares polynomial fits of its
``image_to_latlon`` / ``latlon_to_image`` mappings.

Sections populated
------------------
``ProductCreation``, ``Display``, ``GeoData`` and ``Measurement`` are
always built.  Given the source ``SICDMetadata`` and ``Geolocation``,
the following are populated as well:

- ``Measurement.ValidData`` and ``GeoData.ValidData`` -- the source
  image's valid-data polygon projected into product pixel coordinates,
  clipped to the product extent, hulled and ordered clockwise from the
  minimum-row vertex.  Falls back to the full product rectangle when
  the source carries no valid-data polygon.
- ``Measurement.ARPPoly`` / ``ARPFlag`` -- carried from SICD Position.
- ``Measurement.*.TimeCOAPoly`` -- refit into product pixel coordinates.
- ``ExploitationFeatures`` -- collection geometry, phenomenology,
  resolution, polarization, input ROI, and product ground-plane
  resolution / ellipticity / north angle.
- ``Radiometric`` -- the SICD scale-factor and noise polynomials refit
  into product pixel coordinates.
- ``ErrorStatistics`` and ``MatchInfo`` -- carried across unchanged;
  both are expressed in collection geometry, not image coordinates.
- ``ProductProcessing`` -- a record of the orthorectification.
- ``DigitalElevationData`` -- when ``dem_posts_per_degree`` is supplied.

Dependencies
------------
scipy

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
2026-09-03

Modified
--------
2026-09-03
"""

from __future__ import annotations

# Standard library
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Tuple

# Third-party
import numpy as np
from scipy.spatial import ConvexHull, QhullError

# GRDL internal
from grdl.exceptions import ValidationError
from grdl.geolocation.coordinates import ecef_to_geodetic, geodetic_to_ecef
from grdl.IO.models.common import LatLon, Poly2D, RowCol, XYZ
from grdl.IO.models.sicd import SICDMetadata, SICDNoiseLevel, SICDRadiometric
from grdl.IO.models.sidd import (
    SIDDAngleMagnitude,
    SIDDClassification,
    SIDDCollectionGeometry,
    SIDDCollectionInfo,
    SIDDCollectionPhenomenology,
    SIDDDigitalElevationData,
    SIDDDisplay,
    SIDDExploitationFeatures,
    SIDDExploitationFeaturesProduct,
    SIDDGeoData,
    SIDDGeographicCoordinates,
    SIDDGeographicProjection,
    SIDDGeopositioning,
    SIDDInputROI,
    SIDDMeasurement,
    SIDDMetadata,
    SIDDPlaneProjection,
    SIDDPolynomialProjection,
    SIDDPositionalAccuracy,
    SIDDProcessingModule,
    SIDDProductCreation,
    SIDDProductPlane,
    SIDDProductProcessing,
    SIDDProductResolution,
    SIDDProcessorInformation,
    SIDDRadarMode,
    SIDDReferencePoint,
    SIDDTxRcvPolarization,
)

logger = logging.getLogger(__name__)


# Number of bands and numpy dtype implied by each SIDD pixel type.
PIXEL_TYPES = {
    'MONO8I': (1, np.uint8),
    'MONO16I': (1, np.uint16),
    'RGB24I': (3, np.uint8),
}

# Half-power beamwidth of a rectangular (uniform) window, in cycles.
# Uniformly-weighted resolution is this constant divided by the impulse
# response bandwidth.
_UNIFORM_HPBW = 0.8859

# Degrees per arc-second.
_ARCSEC = 1.0 / 3600.0

# Samples added along each source polygon edge before projection, so a
# curved ground track is captured rather than chorded.
_EDGE_SAMPLES = 24


# ===================================================================
# Pixel type helpers
# ===================================================================

def infer_pixel_type(data: np.ndarray) -> str:
    """Choose the SIDD pixel type matching a product array.

    Parameters
    ----------
    data : np.ndarray
        Product raster.  Shape ``(rows, cols)`` for single band, or
        ``(3, rows, cols)`` / ``(rows, cols, 3)`` for three-band.

    Returns
    -------
    str
        ``'MONO8I'``, ``'MONO16I'``, or ``'RGB24I'``.

    Raises
    ------
    ValidationError
        If the array is neither single-band nor three-band, or the
        dtype has no SIDD equivalent.

    Examples
    --------
    >>> infer_pixel_type(np.zeros((10, 10), dtype=np.uint8))
    'MONO8I'
    >>> infer_pixel_type(np.zeros((3, 10, 10), dtype=np.uint8))
    'RGB24I'
    """
    arr = np.asarray(data)
    if arr.ndim == 2:
        bands = 1
    elif arr.ndim == 3:
        if arr.shape[0] == 3 or arr.shape[-1] == 3:
            bands = 3
        else:
            raise ValidationError(
                f"3D product must have three bands on the first or last "
                f"axis, got shape {arr.shape}"
            )
    else:
        raise ValidationError(
            f"Product must be 2D or 3D, got shape {arr.shape}"
        )

    if bands == 3:
        return 'RGB24I'
    if arr.dtype == np.uint16:
        return 'MONO16I'
    return 'MONO8I'


def to_display_samples(
    data: np.ndarray,
    pixel_type: str = 'MONO8I',
    nodata: int = 0,
) -> np.ndarray:
    """Convert a normalized stretch result to SIDD display samples.

    ``grdl.contrast`` stretches -- and
    ``grdl.image_processing.intensity.PercentileStretch`` -- return
    ``float32`` in ``[0, 1]`` with ``NaN`` wherever there was no
    coverage.  SIDD carries integer display samples, so this scales the
    normalized range onto the pixel type's full range and replaces
    non-finite samples with ``nodata``.

    Integer input is returned unchanged after a dtype check, so this is
    safe to apply to either.

    Parameters
    ----------
    data : np.ndarray
        Normalized floating point samples in ``[0, 1]``, or an array
        already in the pixel type's integer dtype.
    pixel_type : str
        Target SIDD pixel type.
    nodata : int
        Value written where the input is not finite.  Default 0.

    Returns
    -------
    np.ndarray
        Array in the pixel type's dtype, same shape as the input.

    Raises
    ------
    ValidationError
        If ``pixel_type`` is unsupported, or an integer array is given
        in the wrong dtype.

    Examples
    --------
    >>> stretched = PercentileStretch().apply(np.abs(ortho))
    >>> samples = to_display_samples(stretched, 'MONO8I')
    >>> samples.dtype
    dtype('uint8')
    """
    if pixel_type not in PIXEL_TYPES:
        raise ValidationError(
            f"Unsupported pixel_type '{pixel_type}'. "
            f"Must be one of {sorted(PIXEL_TYPES)}"
        )
    dtype = np.dtype(PIXEL_TYPES[pixel_type][1])
    arr = np.asarray(data)

    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype != dtype:
            raise ValidationError(
                f"Pixel type '{pixel_type}' requires dtype "
                f"{dtype.name}, got {arr.dtype}"
            )
        return arr

    if not np.issubdtype(arr.dtype, np.floating):
        raise ValidationError(
            f"Expected floating point or {dtype.name} samples, got "
            f"{arr.dtype}"
        )

    # Imported here rather than at module scope: grdl.contrast pulls in
    # grdl.image_processing, whose ortho subpackage imports back into
    # grdl.geolocation, and grdl.IO is on that cycle.
    from grdl.contrast.base import clip_cast

    scale = float(np.iinfo(dtype).max)
    finite = np.isfinite(arr)
    scaled = np.where(finite, arr, 0.0) * scale
    out = clip_cast(scaled, dtype)
    if not finite.all():
        out[~finite] = nodata
    return out


# ===================================================================
# Local ENU basis
# ===================================================================

def _enu_basis_ecef(
    lat_deg: float,
    lon_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF unit vectors of the local East/North/Up frame.

    Parameters
    ----------
    lat_deg : float
        Geodetic latitude in degrees.
    lon_deg : float
        Longitude in degrees.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(east, north, up)``, each a shape ``(3,)`` unit vector in
        ECEF.  ``up`` is the WGS-84 ellipsoid normal at the point.
    """
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
    sin_lon, cos_lon = np.sin(lon_r), np.cos(lon_r)

    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([
        -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat,
    ])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return east, north, up


def _ecef_to_latlon(ecef: np.ndarray) -> Tuple[float, float, float]:
    """Geodetic coordinates of a single ECEF point."""
    llh = ecef_to_geodetic(np.asarray(ecef, dtype=np.float64).reshape(1, 3))
    return float(llh[0, 0]), float(llh[0, 1]), float(llh[0, 2])


# ===================================================================
# 2D polynomial least-squares fit
# ===================================================================

def _fit_poly2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    order_x: int,
    order_y: int,
) -> np.ndarray:
    """Least-squares fit of ``z`` as a 2D polynomial in ``x`` and ``y``.

    The fit is performed on variables scaled to roughly unit magnitude
    for conditioning, then the coefficients are rescaled so that the
    returned array evaluates directly against the unscaled inputs.

    Parameters
    ----------
    x, y : np.ndarray
        Independent variables, any broadcast-compatible shape.
    z : np.ndarray
        Dependent variable, same shape as ``x`` and ``y``.
    order_x, order_y : int
        Polynomial order in each variable.

    Returns
    -------
    np.ndarray
        Coefficient array of shape ``(order_x + 1, order_y + 1)`` where
        ``coefs[i, j]`` multiplies ``x**i * y**j``.

    Raises
    ------
    ValidationError
        If there are fewer samples than coefficients.
    """
    xf = np.asarray(x, dtype=np.float64).ravel()
    yf = np.asarray(y, dtype=np.float64).ravel()
    zf = np.asarray(z, dtype=np.float64).ravel()

    n_coefs = (order_x + 1) * (order_y + 1)
    if xf.size < n_coefs:
        raise ValidationError(
            f"Need at least {n_coefs} samples to fit a "
            f"({order_x}, {order_y}) polynomial, got {xf.size}"
        )

    # Scale so the Vandermonde columns stay comparable in magnitude.
    sx = 1.0 / max(np.max(np.abs(xf)), 1.0)
    sy = 1.0 / max(np.max(np.abs(yf)), 1.0)
    xs, ys = xf * sx, yf * sy

    x_pows = np.stack([xs ** i for i in range(order_x + 1)], axis=1)
    y_pows = np.stack([ys ** j for j in range(order_y + 1)], axis=1)
    design = (x_pows[:, :, None] * y_pows[:, None, :]).reshape(
        xf.size, n_coefs,
    )

    coefs, *_ = np.linalg.lstsq(design, zf, rcond=None)
    scaled = coefs.reshape(order_x + 1, order_y + 1)

    # Undo the input scaling: c_ij * sx**i * sy**j multiplies x**i y**j.
    i_pow = (sx ** np.arange(order_x + 1))[:, None]
    j_pow = (sy ** np.arange(order_y + 1))[None, :]
    return scaled * i_pow * j_pow


# ===================================================================
# Product pixel <-> source image sampling
# ===================================================================

class _SourceSampleMap:
    """Lattice of product pixels paired with source image coordinates.

    SICD polynomials -- ``TimeCOAPoly`` and the radiometric scale
    factors -- are functions of slant-plane image coordinates in meters.
    SIDD requires the same quantities as functions of product row and
    column pixel indices.  This class samples a regular lattice over the
    product grid once, carries each sample back through the grid and the
    source geolocation to source physical coordinates, and then refits
    any number of polynomials against that single lattice.

    Parameters
    ----------
    grid : OutputGridProtocol
        Product grid.
    source_metadata : SICDMetadata
        Source metadata supplying the SCP pixel and sample spacing.
    geolocation : Geolocation
        Source geolocation, used to map ground positions back to source
        pixels.
    samples : int
        Lattice size per axis.

    Attributes
    ----------
    usable : bool
        False when the source metadata lacks the fields needed to map
        into slant-plane coordinates, or too few samples projected.
    """

    def __init__(
        self,
        grid: Any,
        source_metadata: SICDMetadata,
        geolocation: Any,
        samples: int = 12,
    ) -> None:
        self.usable = False
        self.prod_rows = np.empty(0)
        self.prod_cols = np.empty(0)
        self.src_row_m = np.empty(0)
        self.src_col_m = np.empty(0)

        image_data = source_metadata.image_data
        src_grid = source_metadata.grid
        if (image_data is None or image_data.scp_pixel is None
                or src_grid is None or src_grid.row is None
                or src_grid.col is None
                or src_grid.row.ss is None or src_grid.col.ss is None):
            return

        rows = np.linspace(0.0, grid.rows - 1.0, samples)
        cols = np.linspace(0.0, grid.cols - 1.0, samples)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')

        lat, lon = grid.image_to_latlon(rr.ravel(), cc.ravel())
        latlon = np.column_stack([
            np.asarray(lat, dtype=np.float64),
            np.asarray(lon, dtype=np.float64),
        ])
        src_pixels = np.asarray(geolocation.latlon_to_image(latlon))

        first_row = image_data.first_row or 0
        first_col = image_data.first_col or 0
        row_m = (
            src_pixels[:, 0] - image_data.scp_pixel.row + first_row
        ) * float(src_grid.row.ss)
        col_m = (
            src_pixels[:, 1] - image_data.scp_pixel.col + first_col
        ) * float(src_grid.col.ss)

        good = np.isfinite(row_m) & np.isfinite(col_m)
        if good.sum() < 16:
            return

        self.prod_rows = rr.ravel()[good]
        self.prod_cols = cc.ravel()[good]
        self.src_row_m = row_m[good]
        self.src_col_m = col_m[good]
        self.usable = True

    def refit(self, poly: Optional[Poly2D]) -> Optional[Poly2D]:
        """Refit a SICD image-coordinate polynomial into product pixels.

        Parameters
        ----------
        poly : Poly2D or None
            Polynomial in slant-plane image coordinates (meters).

        Returns
        -------
        Poly2D or None
            Equivalent polynomial in product pixel coordinates, or None
            when the input is None or the lattice is unusable.  A
            constant polynomial is carried across unchanged, since no
            coordinate transform applies to it.
        """
        if poly is None or poly.coefs is None:
            return None
        order = max(poly.order)
        if order <= 0:
            return Poly2D(coefs=np.array(poly.coefs, dtype=np.float64))
        if not self.usable:
            return None

        values = np.asarray(
            poly(self.src_row_m, self.src_col_m), dtype=np.float64,
        )
        good = np.isfinite(values)
        if good.sum() < (order + 1) ** 2:
            return None
        return Poly2D(coefs=_fit_poly2d(
            self.prod_rows[good], self.prod_cols[good], values[good],
            order, order,
        ))


# ===================================================================
# Valid data polygon
# ===================================================================

def _densify(vertices: np.ndarray, per_edge: int) -> np.ndarray:
    """Insert intermediate points along each edge of a closed polygon.

    Parameters
    ----------
    vertices : np.ndarray
        Shape ``(N, 2)`` polygon vertices.
    per_edge : int
        Samples per edge, including the starting vertex.

    Returns
    -------
    np.ndarray
        Shape ``(N * per_edge, 2)`` densified polygon.
    """
    n = vertices.shape[0]
    t = np.linspace(0.0, 1.0, per_edge, endpoint=False)[:, None]
    starts = vertices
    ends = np.roll(vertices, -1, axis=0)
    segments = starts[:, None, :] + t[None, :, :] * (
        ends - starts
    )[:, None, :]
    return segments.reshape(n * per_edge, 2)


def _clip_to_rect(
    polygon: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Clip a convex polygon to the product pixel rectangle.

    Sutherland-Hodgman against the four axis-aligned edges of
    ``[0, rows - 1] x [0, cols - 1]``.  The subject polygon must be
    convex, which the caller guarantees by hulling first.

    Parameters
    ----------
    polygon : np.ndarray
        Shape ``(N, 2)`` vertices as ``(row, col)``.
    rows, cols : int
        Product raster dimensions.

    Returns
    -------
    np.ndarray
        Clipped polygon, shape ``(M, 2)``.  Empty when the polygon does
        not intersect the rectangle.
    """
    # Each boundary is (axis, limit, keep_greater).
    boundaries = (
        (0, 0.0, True),
        (0, float(rows - 1), False),
        (1, 0.0, True),
        (1, float(cols - 1), False),
    )

    output = np.asarray(polygon, dtype=np.float64)
    for axis, limit, keep_greater in boundaries:
        if output.shape[0] == 0:
            return output

        # Inside-ness for the whole ring at once, so the edge walk below
        # only indexes precomputed flags.
        coord = output[:, axis]
        inside = coord >= limit if keep_greater else coord <= limit

        clipped = []
        n = output.shape[0]
        for i in range(n):
            prev_i = i - 1
            prev, curr = output[prev_i], output[i]
            if inside[i] != inside[prev_i]:
                denom = curr[axis] - prev[axis]
                if abs(denom) > 1e-12:
                    t = (limit - prev[axis]) / denom
                    clipped.append(prev + t * (curr - prev))
            if inside[i]:
                clipped.append(curr)
        output = (
            np.array(clipped, dtype=np.float64)
            if clipped else np.empty((0, 2))
        )
    return output


def _order_clockwise(polygon: np.ndarray) -> np.ndarray:
    """Order polygon vertices clockwise from the minimum-row vertex.

    "Clockwise" is as seen in the image, with row increasing downward
    and column increasing to the right -- the convention the SIDD
    schema requires for ``ValidData``.

    Parameters
    ----------
    polygon : np.ndarray
        Shape ``(N, 2)`` vertices as ``(row, col)``.

    Returns
    -------
    np.ndarray
        Reordered vertices, shape ``(N, 2)``.
    """
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.shape[0] < 3:
        return pts

    # Shoelace with x = col, y = row.  Row points down, so a positive
    # sum means clockwise on screen.
    x, y = pts[:, 1], pts[:, 0]
    area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area < 0:
        pts = pts[::-1]

    # Rotate so the minimum-row vertex leads, breaking ties on column.
    start = int(np.lexsort((pts[:, 1], pts[:, 0]))[0])
    return np.roll(pts, -start, axis=0)


def _dedupe(polygon: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Drop consecutive duplicate vertices from a closed polygon."""
    if polygon.shape[0] < 2:
        return polygon
    keep = [0]
    for i in range(1, polygon.shape[0]):
        if np.linalg.norm(polygon[i] - polygon[keep[-1]]) > tol:
            keep.append(i)
    if (len(keep) > 2
            and np.linalg.norm(polygon[keep[-1]] - polygon[keep[0]]) <= tol):
        keep.pop()
    return polygon[keep]


def _simplify_convex(polygon: np.ndarray, tol: float) -> np.ndarray:
    """Drop vertices that bow less than ``tol`` from their neighbours.

    Projecting a densified source polygon produces a hull that traces a
    curved edge with dozens of near-collinear vertices.  The SIDD
    schema wants a simple polygon, and exploitation tools expect a
    handful of vertices, so vertices whose perpendicular offset from
    the chord joining their neighbours is under ``tol`` pixels are
    removed.  Removal shrinks the polygon by at most ``tol``, well
    inside the schema's allowance for approximating the valid region.

    Parameters
    ----------
    polygon : np.ndarray
        Shape ``(N, 2)`` convex polygon as ``(row, col)``.
    tol : float
        Maximum perpendicular deviation to discard, in pixels.

    Returns
    -------
    np.ndarray
        Simplified polygon, never fewer than three vertices.
    """
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.shape[0] <= 3 or tol <= 0:
        return pts

    changed = True
    while changed and pts.shape[0] > 3:
        changed = False
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
        chord = nxt - prev
        chord_len = np.linalg.norm(chord, axis=1)
        offset = pts - prev
        # 2-D cross product magnitude over chord length.
        cross = np.abs(
            chord[:, 0] * offset[:, 1] - chord[:, 1] * offset[:, 0]
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation = np.where(chord_len > 1e-12, cross / chord_len, 0.0)

        worst = int(np.argmin(deviation))
        if deviation[worst] < tol:
            pts = np.delete(pts, worst, axis=0)
            changed = True
    return pts


def _product_rectangle(rows: int, cols: int) -> np.ndarray:
    """The full product extent as a clockwise ``(row, col)`` polygon."""
    return np.array([
        [0.0, 0.0],
        [0.0, cols - 1.0],
        [rows - 1.0, cols - 1.0],
        [rows - 1.0, 0.0],
    ], dtype=np.float64)


def _valid_data_polygon(
    grid: Any,
    rows: int,
    cols: int,
    source_metadata: Optional[SICDMetadata],
    geolocation: Optional[Any],
    tolerance: float = 1.0,
) -> np.ndarray:
    """Project the source valid-data polygon into product pixels.

    Takes the SICD ``ImageData.ValidData`` polygon (or the full source
    image rectangle when absent), densifies its edges, carries it
    through the source geolocation and the product grid, hulls the
    result, clips it to the product extent and orders it clockwise.

    Parameters
    ----------
    grid : OutputGridProtocol
        Product grid.
    rows, cols : int
        Product raster dimensions.
    source_metadata : SICDMetadata or None
        Source metadata.
    geolocation : Geolocation or None
        Source geolocation.
    tolerance : float
        Vertex simplification tolerance in pixels.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)`` polygon as ``(row, col)`` in product pixel
        coordinates.  The full product rectangle when the source
        polygon cannot be projected.
    """
    rect = _product_rectangle(rows, cols)
    if source_metadata is None or geolocation is None:
        return rect

    image_data = source_metadata.image_data
    if image_data is None:
        return rect

    src_verts = image_data.valid_data
    if src_verts:
        source = np.array(
            [[v.row, v.col] for v in src_verts], dtype=np.float64,
        )
    elif image_data.num_rows and image_data.num_cols:
        source = _product_rectangle(
            int(image_data.num_rows), int(image_data.num_cols),
        )
    else:
        return rect

    if source.shape[0] < 3:
        return rect

    dense = _densify(source, _EDGE_SAMPLES)
    ground = np.asarray(geolocation.image_to_latlon(dense))
    prod_rows, prod_cols = grid.latlon_to_image(
        ground[:, 0], ground[:, 1],
    )
    pts = np.column_stack([
        np.asarray(prod_rows, dtype=np.float64),
        np.asarray(prod_cols, dtype=np.float64),
    ])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        logger.warning(
            "Source valid-data polygon did not project into the product "
            "grid; using the full product rectangle."
        )
        return rect

    try:
        hull = pts[ConvexHull(pts).vertices]
    except (QhullError, ValueError):
        logger.warning(
            "Could not hull the projected valid-data polygon; using the "
            "full product rectangle."
        )
        return rect

    clipped = _clip_to_rect(hull, rows, cols)
    clipped = _dedupe(clipped)
    if clipped.shape[0] < 3:
        logger.warning(
            "Projected valid-data polygon does not overlap the product "
            "extent; using the full product rectangle."
        )
        return rect

    return _order_clockwise(_simplify_convex(clipped, tolerance))


def _input_roi(
    source_metadata: Optional[SICDMetadata],
    geolocation: Optional[Any],
    grid: Any,
    rows: int,
    cols: int,
) -> Optional[SIDDInputROI]:
    """Bounding box of the source pixels the product was drawn from.

    Parameters
    ----------
    source_metadata : SICDMetadata or None
        Source metadata, for clamping to the source image extent.
    geolocation : Geolocation or None
        Source geolocation.
    grid : OutputGridProtocol
        Product grid.
    rows, cols : int
        Product raster dimensions.

    Returns
    -------
    SIDDInputROI or None
        None when the source pixels cannot be determined.
    """
    if geolocation is None:
        return None

    perimeter = _densify(_product_rectangle(rows, cols), _EDGE_SAMPLES)
    lat, lon = grid.image_to_latlon(perimeter[:, 0], perimeter[:, 1])
    src = np.asarray(geolocation.latlon_to_image(np.column_stack([
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
    ])))
    src = src[np.isfinite(src).all(axis=1)]
    if src.shape[0] == 0:
        return None

    r0, c0 = np.floor(src.min(axis=0))
    r1, c1 = np.ceil(src.max(axis=0))

    if source_metadata is not None and source_metadata.image_data:
        image_data = source_metadata.image_data
        if image_data.num_rows and image_data.num_cols:
            r0 = max(r0, 0.0)
            c0 = max(c0, 0.0)
            r1 = min(r1, float(image_data.num_rows - 1))
            c1 = min(c1, float(image_data.num_cols - 1))

    if r1 < r0 or c1 < c0:
        return None

    return SIDDInputROI(
        size=RowCol(row=float(r1 - r0 + 1), col=float(c1 - c0 + 1)),
        upper_left=RowCol(row=float(r0), col=float(c0)),
    )


# ===================================================================
# Exploitation geometry (per the SIDD Design & Exploitation document)
# ===================================================================

class _ExploitationGeometry:
    """Collection geometry and phenomenology in the product plane.

    Computes the SIDD exploitation angles from the SICD collection
    geometry at the scene centre point and the product's row/column
    unit vectors.

    Parameters
    ----------
    scp_ecf : np.ndarray
        Scene centre point in ECEF meters, shape ``(3,)``.
    arp_pos : np.ndarray
        Aperture reference point position at COA, ECEF meters.
    arp_vel : np.ndarray
        Aperture reference point velocity at COA, ECEF m/s.
    row_vector : np.ndarray
        Product row-increasing unit vector in ECEF.
    col_vector : np.ndarray
        Product column-increasing unit vector in ECEF.
    etp : np.ndarray
        Earth tangent plane normal at the scene centre point.
    """

    def __init__(
        self,
        scp_ecf: np.ndarray,
        arp_pos: np.ndarray,
        arp_vel: np.ndarray,
        row_vector: np.ndarray,
        col_vector: np.ndarray,
        etp: np.ndarray,
    ) -> None:
        self.scp = np.asarray(scp_ecf, dtype=np.float64)
        self.arp_pos = np.asarray(arp_pos, dtype=np.float64)
        self.arp_vel = np.asarray(arp_vel, dtype=np.float64)

        self.slant_x = self._unit(self.arp_pos - self.scp)
        slant_z = self._unit(np.cross(self.slant_x, self.arp_vel))
        if slant_z.dot(self.arp_pos) < 0:
            slant_z = -slant_z
        self.slant_z = slant_z
        self.slant_y = self._unit(np.cross(self.slant_z, self.slant_x))

        self.etp = self._unit(etp)
        self.row_vector = self._unit(row_vector)
        self.col_vector = self._unit(col_vector)
        self.normal = self._unit(np.cross(self.row_vector, self.col_vector))

    @staticmethod
    def _unit(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm < 1e-9:
            raise ValidationError(
                "Cannot normalize a near-zero vector for exploitation "
                "geometry"
            )
        return arr / norm

    @property
    def azimuth(self) -> float:
        """Line-of-sight azimuth, degrees clockwise from north."""
        ground_los = -(self.slant_x - self.slant_x.dot(self.etp) * self.etp)
        lat, lon, _ = _ecef_to_latlon(self.scp)
        _, north, _ = _enu_basis_ecef(lat, lon)
        east = np.cross(north, self.etp)
        ang = np.degrees(np.arctan2(
            -ground_los.dot(east), ground_los.dot(north),
        ))
        return float(ang % 360.0)

    @property
    def slope(self) -> float:
        """Angle between the earth tangent plane and the slant plane."""
        return float(np.degrees(np.arccos(
            np.clip(self.slant_z.dot(self.etp), -1.0, 1.0),
        )))

    @property
    def graze(self) -> float:
        """Angle between the ground plane and the line of sight."""
        return float(np.degrees(np.arcsin(
            np.clip(self.slant_x.dot(self.etp), -1.0, 1.0),
        )))

    @property
    def tilt(self) -> float:
        """Angle between the ground plane and the cross-range vector."""
        return float(np.degrees(np.arctan(
            self.etp.dot(self.slant_y) / self.etp.dot(self.slant_z),
        )))

    @property
    def doppler_cone(self) -> float:
        """Angle between the velocity vector and the line of sight."""
        vel_unit = self._unit(self.arp_vel)
        return float(np.degrees(np.arccos(
            np.clip(vel_unit.dot(-self.slant_x), -1.0, 1.0),
        )))

    @property
    def squint(self) -> float:
        """Ground-projected angle from ground track to line of sight."""
        lat, lon, _ = _ecef_to_latlon(self.scp)
        _, north, _ = _enu_basis_ecef(lat, lon)
        east = np.cross(north, self.etp)

        los_g = -(self.slant_x - self.slant_x.dot(self.etp) * self.etp)
        vel_g = self.arp_vel - self.arp_vel.dot(self.etp) * self.etp
        if np.linalg.norm(los_g) < 1e-9 or np.linalg.norm(vel_g) < 1e-9:
            return 0.0

        los_ang = np.arctan2(los_g.dot(east), los_g.dot(north))
        vel_ang = np.arctan2(vel_g.dot(east), vel_g.dot(north))
        diff = np.degrees(los_ang - vel_ang)
        return float((diff + 180.0) % 360.0 - 180.0)

    @property
    def shadow(self) -> Tuple[float, float]:
        """Shadow ``(angle_deg, magnitude)`` in the product plane."""
        shadow = self.etp - self.slant_x / self.slant_x.dot(self.etp)
        prime = shadow - (
            shadow.dot(self.normal) / self.slant_z.dot(self.normal)
        ) * self.slant_z
        angle = np.degrees(np.arctan2(
            self.row_vector.dot(prime), self.col_vector.dot(prime),
        ))
        return float(angle % 360.0), float(np.linalg.norm(prime))

    @property
    def layover(self) -> Tuple[float, float]:
        """Layover ``(angle_deg, magnitude)`` in the product plane."""
        lay = self.normal - self.slant_z / self.slant_z.dot(self.normal)
        angle = np.degrees(np.arctan2(
            self.row_vector.dot(lay), self.col_vector.dot(lay),
        ))
        return float(angle % 360.0), float(np.linalg.norm(lay))

    @property
    def multipath(self) -> float:
        """Multipath angle in the product plane, degrees."""
        mp = self.slant_x - self.slant_z * (
            self.slant_x.dot(self.normal) / self.slant_z.dot(self.normal)
        )
        angle = np.degrees(np.arctan2(
            self.col_vector.dot(mp), self.row_vector.dot(mp),
        ))
        return float(angle % 360.0)

    @property
    def ground_track(self) -> float:
        """Ground track angle in the product plane, degrees."""
        track = self.arp_vel - self.arp_vel.dot(self.normal) * self.normal
        angle = np.degrees(np.arctan2(
            self.col_vector.dot(track), self.row_vector.dot(track),
        ))
        return float(angle % 360.0)

    @property
    def north(self) -> float:
        """Clockwise angle from the increasing-column direction to north."""
        lat, lon, _ = _ecef_to_latlon(self.scp)
        _, north_vec, _ = _enu_basis_ecef(lat, lon)
        prime = north_vec - self.slant_z * (
            north_vec.dot(self.normal) / self.slant_z.dot(self.normal)
        )
        angle = np.degrees(np.arctan2(
            self.row_vector.dot(prime), self.col_vector.dot(prime),
        ))
        return float(angle % 360.0)

    def ground_plane_resolution(
        self,
        rho_range: float,
        rho_azimuth: float,
    ) -> Tuple[float, float]:
        """Project slant-plane resolution into the product plane.

        Parameters
        ----------
        rho_range : float
            Slant-plane range resolution in meters.
        rho_azimuth : float
            Slant-plane azimuth resolution in meters.

        Returns
        -------
        Tuple[float, float]
            ``(row_resolution, col_resolution)`` in meters.
        """
        x_g = self.slant_x - self.slant_x.dot(self.normal) * self.normal
        theta_r = -np.arctan2(
            self.col_vector.dot(x_g), self.row_vector.dot(x_g),
        )
        graze = np.radians(self.graze)
        tilt = np.radians(self.tilt)

        k_r1 = (np.cos(theta_r) / np.cos(graze)) ** 2 + (
            np.sin(theta_r) ** 2 * np.tan(graze) * np.tan(tilt)
            - np.sin(2 * theta_r) / np.cos(graze)
        ) * np.tan(graze) * np.tan(tilt)
        k_r2 = (np.sin(theta_r) / np.cos(tilt)) ** 2

        k_c1 = (
            np.sin(theta_r) ** 2 / np.cos(graze)
            - np.sin(2 * theta_r) * np.tan(graze) * np.tan(tilt)
        ) / np.cos(graze) + (
            np.cos(theta_r) * np.tan(graze) * np.tan(tilt)
        ) ** 2
        k_c2 = (np.cos(theta_r) / np.cos(tilt)) ** 2

        r2 = rho_range * rho_range
        c2 = rho_azimuth * rho_azimuth
        return (
            float(np.sqrt(abs(k_r1 * r2 + k_r2 * c2))),
            float(np.sqrt(abs(k_c1 * r2 + k_c2 * c2))),
        )


# ===================================================================
# Grid -> SIDD projection
# ===================================================================

def _plane_projection_from_enu(
    grid: Any,
    time_coa: Poly2D,
) -> SIDDPlaneProjection:
    """Build an exact PlaneProjection from an ENU or rotated ENU grid.

    Parameters
    ----------
    grid : ENUGrid or RotatedENUGrid
        Source grid.  Both place their pixels on the up=0 tangent plane
        at ``(ref_lat, ref_lon, ref_alt)``, which is exactly the plane
        SIDD's PGD projection describes.
    time_coa : Poly2D
        Center-of-aperture time polynomial in product pixel coordinates.

    Returns
    -------
    SIDDPlaneProjection
        Reference point, sample spacing, product plane and TimeCOAPoly.
    """
    east, north, _ = _enu_basis_ecef(grid.ref_lat, grid.ref_lon)
    ref_ecef = geodetic_to_ecef(
        np.array([grid.ref_lat, grid.ref_lon, grid.ref_alt]),
    )

    if hasattr(grid, 'angle'):
        # RotatedENUGrid: u along +col, v along -row, rotated by angle.
        cos_a, sin_a = np.cos(grid.angle), np.sin(grid.angle)
        col_enu = np.array([cos_a, sin_a])
        row_enu = np.array([sin_a, -cos_a])
        ref_row = grid.max_v / grid.pixel_size
        ref_col = -grid.min_u / grid.pixel_size
        ss_row = ss_col = float(grid.pixel_size)
    else:
        # ENUGrid: col along +east, row along -north.
        col_enu = np.array([1.0, 0.0])
        row_enu = np.array([0.0, -1.0])
        ref_row = grid.max_north / grid.pixel_size_north
        ref_col = -grid.min_east / grid.pixel_size_east
        ss_row = float(grid.pixel_size_north)
        ss_col = float(grid.pixel_size_east)

    row_ecef = row_enu[0] * east + row_enu[1] * north
    col_ecef = col_enu[0] * east + col_enu[1] * north

    return SIDDPlaneProjection(
        reference_point=SIDDReferencePoint(
            ecef=XYZ.from_array(ref_ecef),
            point=RowCol(row=float(ref_row), col=float(ref_col)),
            name='ORP',
        ),
        sample_spacing=RowCol(row=ss_row, col=ss_col),
        time_coa_poly=time_coa,
        product_plane=SIDDProductPlane(
            row_unit_vector=XYZ.from_array(row_ecef),
            col_unit_vector=XYZ.from_array(col_ecef),
        ),
    )


def _geographic_projection(
    grid: Any,
    time_coa: Poly2D,
) -> SIDDGeographicProjection:
    """Build an exact GeographicProjection from a GeographicGrid.

    SIDD's GGD maps ``lat = lat_0 - delta_row * (row - row_0) / 3600``
    and ``lon = lon_0 + delta_col * (col - col_0) / 3600``, which is
    exactly ``GeographicGrid``'s north-up constant angular mapping with
    the sample spacing expressed in arc-seconds.

    Parameters
    ----------
    grid : GeographicGrid
        Source grid.
    time_coa : Poly2D
        Center-of-aperture time polynomial in product pixel coordinates.

    Returns
    -------
    SIDDGeographicProjection
        Reference point and arc-second sample spacing.
    """
    ref_row = (grid.rows - 1) / 2.0
    ref_col = (grid.cols - 1) / 2.0
    ref_lat, ref_lon = grid.image_to_latlon(ref_row, ref_col)
    ref_ecef = geodetic_to_ecef(
        np.array([float(ref_lat), float(ref_lon), 0.0]),
    )

    return SIDDGeographicProjection(
        reference_point=SIDDReferencePoint(
            ecef=XYZ.from_array(ref_ecef),
            point=RowCol(row=ref_row, col=ref_col),
            name='ORP',
        ),
        sample_spacing=RowCol(
            row=float(grid.pixel_size_lat) / _ARCSEC,
            col=float(grid.pixel_size_lon) / _ARCSEC,
        ),
        time_coa_poly=time_coa,
    )


def _polynomial_projection(
    grid: Any,
    order: int = 3,
) -> SIDDPolynomialProjection:
    """Fit a PolynomialProjection to an arbitrary grid.

    Samples the grid's ``image_to_latlon`` and ``latlon_to_image``
    mappings on a regular lattice and fits forward and inverse
    polynomials.  Used for grids with no closed-form SIDD equivalent
    (UTM, Web Mercator, custom projections).

    Parameters
    ----------
    grid : OutputGridProtocol
        Source grid.
    order : int
        Polynomial order in each variable.  Default 3.

    Returns
    -------
    SIDDPolynomialProjection
        Forward (row/col to lat/lon) and inverse polynomials.
    """
    samples = max(order + 4, 8)
    rows = np.linspace(0.0, grid.rows - 1.0, samples)
    cols = np.linspace(0.0, grid.cols - 1.0, samples)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')

    lat, lon = grid.image_to_latlon(rr.ravel(), cc.ravel())
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    row_back, col_back = grid.latlon_to_image(lat, lon)

    ref_row = (grid.rows - 1) / 2.0
    ref_col = (grid.cols - 1) / 2.0
    ref_lat, ref_lon = grid.image_to_latlon(ref_row, ref_col)
    ref_ecef = geodetic_to_ecef(
        np.array([float(ref_lat), float(ref_lon), 0.0]),
    )

    return SIDDPolynomialProjection(
        reference_point=SIDDReferencePoint(
            ecef=XYZ.from_array(ref_ecef),
            point=RowCol(row=ref_row, col=ref_col),
            name='ORP',
        ),
        row_col_to_lat=Poly2D(
            coefs=_fit_poly2d(rr, cc, lat, order, order),
        ),
        row_col_to_lon=Poly2D(
            coefs=_fit_poly2d(rr, cc, lon, order, order),
        ),
        lat_lon_to_row=Poly2D(
            coefs=_fit_poly2d(lat, lon, np.asarray(row_back), order, order),
        ),
        lat_lon_to_col=Poly2D(
            coefs=_fit_poly2d(lat, lon, np.asarray(col_back), order, order),
        ),
    )


def _product_plane_vectors(
    grid: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """ECEF row/column unit vectors at the grid centre.

    Derived from finite differences of ``image_to_latlon`` so it works
    for any grid, not just the planar ones.

    Parameters
    ----------
    grid : OutputGridProtocol
        Source grid.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] or None
        ``(row_vector, col_vector)`` unit vectors in ECEF, or None if
        the grid is degenerate (fewer than two rows or columns).
    """
    if grid.rows < 2 or grid.cols < 2:
        return None

    r0 = (grid.rows - 1) / 2.0
    c0 = (grid.cols - 1) / 2.0
    step = 0.5

    probe_rows = np.array([r0 - step, r0 + step, r0, r0])
    probe_cols = np.array([c0, c0, c0 - step, c0 + step])
    lat, lon = grid.image_to_latlon(probe_rows, probe_cols)
    llh = np.column_stack([
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
        np.zeros(4),
    ])
    ecef = geodetic_to_ecef(llh)

    row_vec = ecef[1] - ecef[0]
    col_vec = ecef[3] - ecef[2]
    row_norm = np.linalg.norm(row_vec)
    col_norm = np.linalg.norm(col_vec)
    if row_norm < 1e-9 or col_norm < 1e-9:
        return None
    return row_vec / row_norm, col_vec / col_norm


# ===================================================================
# TimeCOAPoly and Radiometric
# ===================================================================

def _time_coa_poly(
    source_metadata: Optional[SICDMetadata],
    sample_map: Optional[_SourceSampleMap],
) -> Poly2D:
    """Refit the source TimeCOAPoly into product pixel coordinates.

    Parameters
    ----------
    source_metadata : SICDMetadata or None
        Source SICD metadata carrying ``grid.time_coa_poly``.
    sample_map : _SourceSampleMap or None
        Product-to-source sampling lattice.

    Returns
    -------
    Poly2D
        TimeCOAPoly in product pixel coordinates.  Falls back to a
        constant polynomial (the SCP COA time, or zero) when the refit
        is not possible.
    """
    fallback = 0.0
    if (source_metadata is not None
            and source_metadata.scpcoa is not None
            and source_metadata.scpcoa.scp_time is not None):
        fallback = float(source_metadata.scpcoa.scp_time)

    if source_metadata is None or source_metadata.grid is None:
        return Poly2D(coefs=np.array([[fallback]]))

    src_poly = source_metadata.grid.time_coa_poly
    if src_poly is None or src_poly.coefs is None:
        return Poly2D(coefs=np.array([[fallback]]))

    if max(src_poly.order) <= 0:
        # Constant COA time -- typical of a spotlight collect.  No
        # coordinate transform applies, so carry it across unchanged.
        return Poly2D(coefs=np.array(src_poly.coefs, dtype=np.float64))

    if sample_map is None or not sample_map.usable:
        logger.warning(
            "Cannot map product pixels back to the source image; "
            "TimeCOAPoly reduced to the constant %.6f s. Pass "
            "geolocation= to refit the full polynomial.", fallback,
        )
        return Poly2D(coefs=np.array([[fallback]]))

    refit = sample_map.refit(src_poly)
    if refit is None:
        logger.warning(
            "TimeCOAPoly refit failed; using the constant %.6f s.",
            fallback,
        )
        return Poly2D(coefs=np.array([[fallback]]))
    return refit


def _build_radiometric(
    source_metadata: Optional[SICDMetadata],
    sample_map: Optional[_SourceSampleMap],
) -> Optional[SICDRadiometric]:
    """Refit the SICD radiometric polynomials into product coordinates.

    The SICD scale-factor and noise polynomials are functions of
    slant-plane image coordinates; SIDD requires the same quantities as
    functions of product pixel indices.  Orthorectification resamples
    geometry but not radiometry, so refitting through the pixel mapping
    is exact up to the polynomial order.

    Parameters
    ----------
    source_metadata : SICDMetadata or None
        Source metadata carrying the ``radiometric`` section.
    sample_map : _SourceSampleMap or None
        Product-to-source sampling lattice.

    Returns
    -------
    SICDRadiometric or None
        None when the source has no radiometric section, or nothing
        could be refit.
    """
    if source_metadata is None or source_metadata.radiometric is None:
        return None
    rad = source_metadata.radiometric

    if sample_map is None or not sample_map.usable:
        logger.warning(
            "Cannot map product pixels back to the source image; the "
            "Radiometric section is omitted rather than carried across "
            "in slant-plane coordinates, where it would be wrong."
        )
        return None

    noise_level = None
    if rad.noise_level is not None:
        noise_poly = sample_map.refit(rad.noise_level.noise_poly)
        if noise_poly is not None:
            noise_level = SICDNoiseLevel(
                noise_level_type=rad.noise_level.noise_level_type,
                noise_poly=noise_poly,
            )

    result = SICDRadiometric(
        noise_level=noise_level,
        rcs_sf_poly=sample_map.refit(rad.rcs_sf_poly),
        sigma_zero_sf_poly=sample_map.refit(rad.sigma_zero_sf_poly),
        beta_zero_sf_poly=sample_map.refit(rad.beta_zero_sf_poly),
        gamma_zero_sf_poly=sample_map.refit(rad.gamma_zero_sf_poly),
    )
    if all(getattr(result, f) is None for f in (
            'noise_level', 'rcs_sf_poly', 'sigma_zero_sf_poly',
            'beta_zero_sf_poly', 'gamma_zero_sf_poly')):
        return None
    return result


# ===================================================================
# ExploitationFeatures
# ===================================================================

def _split_polarization(pol: Optional[str]) -> Optional[SIDDTxRcvPolarization]:
    """Split a SICD ``'TX:RCV'`` polarization string.

    Parameters
    ----------
    pol : str or None
        SICD-style polarization, e.g. ``'V:V'`` or ``'H:V'``.  SICD
        also allows the bare tokens ``'OTHER'`` and ``'UNKNOWN'`` for a
        collect whose polarization is not one of the standard pairs.

    Returns
    -------
    SIDDTxRcvPolarization or None
        Split pair, or None if the string is absent or malformed.

    Notes
    -----
    A bare token maps to itself on both transmit and receive.  Dropping
    it instead is not a safe default: ``ExploitationFeatures/Product``
    requires a ``Polarization`` before ``North``, so returning None
    there emits ``North`` into the slot the schema reserves for
    ``Polarization`` and the whole product fails to validate.
    """
    if not pol:
        return None
    text = pol.upper().strip()
    if text in ('OTHER', 'UNKNOWN'):
        return SIDDTxRcvPolarization(
            tx_polarization=text, rcv_polarization=text,
        )
    parts = text.split(':')
    if len(parts) != 2:
        return None
    tx, rcv = parts[0].strip(), parts[1].strip()
    if not tx or not rcv:
        return None
    return SIDDTxRcvPolarization(tx_polarization=tx, rcv_polarization=rcv)


def _uniform_resolution(dir_param: Any) -> Optional[float]:
    """Uniformly-weighted resolution from a SICD direction parameter.

    Parameters
    ----------
    dir_param : SICDDirParam or None
        SICD ``Grid.Row`` or ``Grid.Col``.

    Returns
    -------
    float or None
        Resolution in meters: ``0.8859 / ImpRespBW`` when the bandwidth
        is available, else ``ImpRespWid``.  This is the grid-plane
        value; the image-to-slant sensitivity is not applied, so for
        strongly squinted collects it differs slightly from the true
        slant-plane resolution.
    """
    if dir_param is None:
        return None
    bw = getattr(dir_param, 'imp_resp_bw', None)
    if bw:
        return _UNIFORM_HPBW / float(bw)
    wid = getattr(dir_param, 'imp_resp_wid', None)
    return float(wid) if wid else None


def _build_exploitation_features(
    source_metadata: Optional[SICDMetadata],
    geometry: Optional[_ExploitationGeometry],
    sensor_name: Optional[str],
    collection_date_time: Optional[str],
    input_roi: Optional[SIDDInputROI],
) -> SIDDExploitationFeatures:
    """Assemble the ExploitationFeatures section.

    Parameters
    ----------
    source_metadata : SICDMetadata or None
        Source collection metadata.
    geometry : _ExploitationGeometry or None
        Product-plane geometry calculator, when the SICD carries enough
        information to construct one.
    sensor_name : str or None
        Explicit sensor name override.
    collection_date_time : str or None
        Explicit collection timestamp override (ISO 8601).
    input_roi : SIDDInputROI or None
        Source image region the product was built from.

    Returns
    -------
    SIDDExploitationFeatures
        One collection entry and one product entry.
    """
    radar_mode = None
    duration = None
    rho_range = None
    rho_azimuth = None
    polarizations = None
    identifier = 'collection'

    if source_metadata is not None:
        ci = source_metadata.collection_info
        if ci is not None:
            sensor_name = sensor_name or ci.collector_name
            identifier = ci.core_name or identifier
            if ci.radar_mode is not None:
                radar_mode = SIDDRadarMode(
                    mode_type=ci.radar_mode.mode_type,
                    mode_id=ci.radar_mode.mode_id,
                )
        if source_metadata.timeline is not None:
            collection_date_time = (
                collection_date_time
                or source_metadata.timeline.collect_start
            )
            duration = source_metadata.timeline.collect_duration
        if source_metadata.grid is not None:
            rho_range = _uniform_resolution(source_metadata.grid.row)
            rho_azimuth = _uniform_resolution(source_metadata.grid.col)
        if source_metadata.image_formation is not None:
            pol = _split_polarization(
                source_metadata.image_formation.tx_rcv_polarization_proc,
            )
            if pol is not None:
                polarizations = [pol]

    if collection_date_time is None:
        collection_date_time = datetime.now(timezone.utc).isoformat(
            timespec='seconds',
        ).replace('+00:00', 'Z')
        logger.warning(
            "No collection date/time available; stamping the current time. "
            "Pass collection_date_time= to set it explicitly."
        )

    phenomenology = None
    geometry_block = None
    if geometry is not None:
        shadow_ang, shadow_mag = geometry.shadow
        layover_ang, layover_mag = geometry.layover
        phenomenology = SIDDCollectionPhenomenology(
            shadow=SIDDAngleMagnitude(
                angle=shadow_ang, magnitude=shadow_mag,
            ),
            layover=SIDDAngleMagnitude(
                angle=layover_ang, magnitude=layover_mag,
            ),
            multi_path=geometry.multipath,
            ground_track=geometry.ground_track,
        )
        geometry_block = SIDDCollectionGeometry(
            azimuth=geometry.azimuth,
            slope=geometry.slope,
            squint=geometry.squint,
            graze=geometry.graze,
            tilt=geometry.tilt,
            doppler_cone_angle=geometry.doppler_cone,
        )

    collection = SIDDCollectionInfo(
        sensor_name=sensor_name or 'UNKNOWN',
        radar_mode=radar_mode or SIDDRadarMode(mode_type='SPOTLIGHT'),
        collection_date_time=collection_date_time,
        collection_duration=(
            float(duration) if duration is not None else 0.0
        ),
        resolution_range=rho_range,
        resolution_azimuth=rho_azimuth,
        polarizations=polarizations,
        geometry=geometry_block,
        phenomenology=phenomenology,
        input_roi=input_roi,
        identifier=identifier,
    )

    resolution = None
    ellipticity = None
    north = None
    if geometry is not None and rho_range and rho_azimuth:
        rho_row, rho_col = geometry.ground_plane_resolution(
            rho_range, rho_azimuth,
        )
        resolution = SIDDProductResolution(row=rho_row, col=rho_col)
        ellipticity = (
            rho_row / rho_col if rho_row >= rho_col else rho_col / rho_row
        )
        north = geometry.north

    product = SIDDExploitationFeaturesProduct(
        resolution=resolution,
        ellipticity=ellipticity,
        north=north,
        polarizations=polarizations,
    )

    return SIDDExploitationFeatures(
        collections=[collection], products=[product],
    )


# ===================================================================
# Optional sections
# ===================================================================

def _build_product_processing(
    grid: Any,
    projection: str,
    interpolation: Optional[str],
    extra: Optional[Dict[str, str]],
) -> SIDDProductProcessing:
    """Record how the product was made as a processing module.

    Parameters
    ----------
    grid : OutputGridProtocol
        Product grid.
    projection : str
        SIDD projection type emitted.
    interpolation : str or None
        Resampling kernel used by the orthorectifier.
    extra : dict of str to str, optional
        Additional free-form parameters to record.

    Returns
    -------
    SIDDProductProcessing
        A single ``Orthorectification`` processing module.
    """
    params: Dict[str, str] = {
        'GridType': type(grid).__name__,
        'Projection': projection,
        'OutputRows': str(int(grid.rows)),
        'OutputCols': str(int(grid.cols)),
    }
    if interpolation:
        params['Interpolation'] = str(interpolation)
    for attr, key in (
        ('pixel_size', 'PixelSizeMeters'),
        ('pixel_size_east', 'PixelSizeEastMeters'),
        ('pixel_size_north', 'PixelSizeNorthMeters'),
        ('pixel_size_lat', 'PixelSizeLatDegrees'),
        ('pixel_size_lon', 'PixelSizeLonDegrees'),
    ):
        value = getattr(grid, attr, None)
        if value is not None:
            params[key] = f'{float(value):.10g}'
    if extra:
        params.update({str(k): str(v) for k, v in extra.items()})

    return SIDDProductProcessing(
        processing_modules=[SIDDProcessingModule(
            module_name='Orthorectification',
            name='GRDL',
            parameters=params,
        )],
    )


def _build_digital_elevation_data(
    geolocation: Optional[Any],
    posts_per_degree: Optional[Tuple[float, float]],
    grid: Any,
    rows: int,
    cols: int,
) -> Optional[SIDDDigitalElevationData]:
    """Describe the terrain model used, when its posting is known.

    The SIDD section requires the DEM's angular post density and a
    reference origin.  ``ElevationModel`` does not expose a posting, so
    the caller supplies it; without it the section is omitted rather
    than filled with invented numbers.

    Parameters
    ----------
    geolocation : Geolocation or None
        Source geolocation, checked for an attached elevation model.
    posts_per_degree : tuple of float, optional
        ``(latitude_density, longitude_density)`` posts per degree.
    grid : OutputGridProtocol
        Product grid, used for the reference origin.
    rows, cols : int
        Product raster dimensions.

    Returns
    -------
    SIDDDigitalElevationData or None
        None when no DEM is attached or no posting was supplied.
    """
    if geolocation is None:
        return None
    elevation = getattr(geolocation, 'elevation', None)
    if elevation is None:
        return None
    if posts_per_degree is None:
        logger.debug(
            "A DEM is attached but dem_posts_per_degree was not supplied; "
            "the DigitalElevationData section is omitted."
        )
        return None

    lat_density, lon_density = posts_per_degree
    # South-west corner of the product, the SIDD reference origin.
    corner_rows = np.array([0.0, 0.0, rows - 1.0, rows - 1.0])
    corner_cols = np.array([0.0, cols - 1.0, cols - 1.0, 0.0])
    lat, lon = grid.image_to_latlon(corner_rows, corner_cols)
    origin = LatLon(
        lat=float(np.min(np.asarray(lat))),
        lon=float(np.min(np.asarray(lon))),
    )

    vertical_datum = (
        'EGM96' if getattr(elevation, 'geoid_path', None) else 'HAE'
    )
    return SIDDDigitalElevationData(
        geographic_coordinates=SIDDGeographicCoordinates(
            longitude_density=float(lon_density),
            latitude_density=float(lat_density),
            reference_origin=origin,
        ),
        geopositioning=SIDDGeopositioning(
            coordinate_system_type='GGS',
            geodetic_datum='World Geodetic System 1984',
            reference_ellipsoid='World Geodetic System 1984',
            vertical_datum=vertical_datum,
            sounding_datum='MSL',
            false_origin=0,
        ),
        positional_accuracy=SIDDPositionalAccuracy(num_regions=1),
    )


# ===================================================================
# Public entry point
# ===================================================================

def build_sidd_metadata(
    grid: Any,
    shape: Sequence[int],
    *,
    pixel_type: str = 'MONO8I',
    source_metadata: Optional[SICDMetadata] = None,
    geolocation: Optional[Any] = None,
    product_class: str = 'Detected Image',
    product_name: Optional[str] = None,
    product_type: Optional[str] = None,
    classification: str = 'U',
    owner_producer: str = 'USA',
    application: str = 'GRDL',
    site: str = 'UNKNOWN',
    sensor_name: Optional[str] = None,
    collection_date_time: Optional[str] = None,
    valid_data: Optional[Sequence[Tuple[float, float]]] = None,
    valid_data_tolerance: float = 1.0,
    projection: Optional[str] = None,
    interpolation: Optional[str] = None,
    dem_posts_per_degree: Optional[Tuple[float, float]] = None,
    processing_parameters: Optional[Dict[str, str]] = None,
) -> SIDDMetadata:
    """Build complete SIDD metadata for a product on an output grid.

    Populates every section the SIDD 3.0 schema requires, plus every
    optional section that can be derived from the source metadata and
    the grid.  See the module docstring for the full list.

    Parameters
    ----------
    grid : OutputGridProtocol
        The grid the product was sampled on.  ``ENUGrid`` and
        ``RotatedENUGrid`` yield an exact ``PlaneProjection``,
        ``GeographicGrid`` an exact ``GeographicProjection``, and any
        other grid a fitted ``PolynomialProjection``.
    shape : Sequence[int]
        Product raster shape, ``(rows, cols)``.  A three-element shape
        from a multi-band product is accepted; the band axis is
        identified by matching against the grid dimensions.
    pixel_type : str
        SIDD pixel type: ``'MONO8I'``, ``'MONO16I'`` or ``'RGB24I'``.
        Use :func:`infer_pixel_type` to derive it from the array.
    source_metadata : SICDMetadata, optional
        Source SICD metadata.  Supplies the ARP polynomial, collection
        geometry, resolution, polarization, TimeCOAPoly, radiometric
        calibration, error statistics and match info.  Without it the
        product still writes but omits the SAR-specific sections and
        will not validate against the SIDD schema.
    geolocation : Geolocation, optional
        Source geolocation.  Required to project the valid-data
        polygon, determine the input ROI, and refit the TimeCOAPoly and
        radiometric polynomials into product coordinates.
    product_class : str
        Descriptive product class, e.g. ``'Detected Image'``.
    product_name : str, optional
        Product name.  Defaults to the source core name plus the
        product class.
    product_type : str, optional
        Product type string.
    classification : str
        Security classification: ``'U'``, ``'C'``, ``'R'``, ``'S'``,
        ``'TS'``.  Default ``'U'``.
    owner_producer : str
        Owner/producer country code.  Default ``'USA'``.
    application : str
        Processing application name recorded in ProductCreation.
    site : str
        Processing site recorded in ProductCreation.
    sensor_name : str, optional
        Sensor name override for ExploitationFeatures.
    collection_date_time : str, optional
        Collection timestamp override (ISO 8601).
    valid_data : Sequence of (row, col), optional
        Explicit valid-data polygon in product pixel coordinates,
        clockwise from the minimum-row vertex.  When omitted the
        source polygon is projected into the product grid, or the full
        product rectangle is used if that is not possible.
    valid_data_tolerance : float
        Simplification tolerance in pixels for the projected valid-data
        polygon.  Vertices bowing less than this from their neighbours
        are dropped, so a curved footprint edge becomes a handful of
        vertices instead of dozens.  Default 1.0; set 0 to keep every
        hull vertex.
    projection : str, optional
        Force a projection type: ``'PlaneProjection'``,
        ``'GeographicProjection'`` or ``'PolynomialProjection'``.
        Defaults to the exact match for the grid type.
    interpolation : str, optional
        Resampling kernel name, recorded in ProductProcessing.
    dem_posts_per_degree : tuple of float, optional
        ``(latitude_density, longitude_density)`` of the terrain model.
        Supplying it populates the DigitalElevationData section.
    processing_parameters : dict of str to str, optional
        Extra free-form parameters recorded in ProductProcessing.

    Returns
    -------
    SIDDMetadata
        Complete typed metadata ready for ``SIDDWriter``.

    Raises
    ------
    ValidationError
        If ``pixel_type`` is unsupported, ``projection`` is unknown, or
        ``shape`` does not match the grid dimensions.

    Examples
    --------
    >>> result = orthorectify(reader=reader, geolocation=geo,
    ...                       output_grid=grid)
    >>> meta = build_sidd_metadata(
    ...     result.output_grid, result.data.shape,
    ...     pixel_type='MONO8I',
    ...     source_metadata=reader.metadata,
    ...     geolocation=geo,
    ... )
    >>> SIDDWriter('product.nitf', metadata=meta).write(stretched)
    """
    if pixel_type not in PIXEL_TYPES:
        raise ValidationError(
            f"Unsupported pixel_type '{pixel_type}'. "
            f"Must be one of {sorted(PIXEL_TYPES)}"
        )
    num_bands = PIXEL_TYPES[pixel_type][0]

    rows, cols = _resolve_shape(shape, grid)

    # One lattice, reused by every polynomial refit below.
    sample_map = None
    if source_metadata is not None and geolocation is not None:
        sample_map = _SourceSampleMap(grid, source_metadata, geolocation)

    # --- Measurement -------------------------------------------------
    time_coa = _time_coa_poly(source_metadata, sample_map)

    if projection is None:
        if hasattr(grid, 'ref_lat') and (
            hasattr(grid, 'pixel_size_east') or hasattr(grid, 'angle')
        ):
            projection = 'PlaneProjection'
        elif hasattr(grid, 'pixel_size_lat'):
            projection = 'GeographicProjection'
        else:
            projection = 'PolynomialProjection'

    plane_proj = None
    geo_proj = None
    poly_proj = None
    if projection == 'PlaneProjection':
        plane_proj = _plane_projection_from_enu(grid, time_coa)
    elif projection == 'GeographicProjection':
        geo_proj = _geographic_projection(grid, time_coa)
    elif projection == 'PolynomialProjection':
        poly_proj = _polynomial_projection(grid)
    else:
        raise ValidationError(
            f"Unknown projection '{projection}'. Must be one of "
            f"'PlaneProjection', 'GeographicProjection', "
            f"'PolynomialProjection'"
        )

    if valid_data is not None:
        valid_poly = _order_clockwise(np.array(
            [[float(r), float(c)] for r, c in valid_data],
            dtype=np.float64,
        ))
    else:
        valid_poly = _valid_data_polygon(
            grid, rows, cols, source_metadata, geolocation,
            valid_data_tolerance,
        )
    valid_pixels = [
        RowCol(row=float(r), col=float(c)) for r, c in valid_poly
    ]

    arp_poly = None
    arp_flag = None
    if source_metadata is not None and source_metadata.position is not None:
        arp_poly = source_metadata.position.arp_poly
        arp_flag = 'REALTIME'
    if arp_poly is None:
        logger.warning(
            "No ARP polynomial available; Measurement.ARPPoly will be "
            "omitted and the product will not validate against the SIDD "
            "schema. Supply source_metadata from a SICD to populate it."
        )

    measurement = SIDDMeasurement(
        projection_type=projection,
        plane_projection=plane_proj,
        geographic_projection=geo_proj,
        polynomial_projection=poly_proj,
        pixel_footprint=RowCol(row=rows, col=cols),
        arp_flag=arp_flag,
        arp_poly=arp_poly,
        valid_data=valid_pixels,
    )

    # --- GeoData -----------------------------------------------------
    corner_rows = np.array([0.0, 0.0, rows - 1.0, rows - 1.0])
    corner_cols = np.array([0.0, cols - 1.0, cols - 1.0, 0.0])
    corner_lat, corner_lon = grid.image_to_latlon(corner_rows, corner_cols)
    image_corners = [
        LatLon(lat=float(la), lon=float(lo))
        for la, lo in zip(np.asarray(corner_lat),
                          np.asarray(corner_lon), strict=True)
    ]

    valid_lat, valid_lon = grid.image_to_latlon(
        valid_poly[:, 0], valid_poly[:, 1],
    )
    valid_geo = [
        LatLon(lat=float(la), lon=float(lo))
        for la, lo in zip(np.asarray(valid_lat),
                          np.asarray(valid_lon), strict=True)
    ]

    geo_data = SIDDGeoData(
        earth_model='WGS_84',
        image_corners=image_corners,
        valid_data=valid_geo,
    )

    # --- ExploitationFeatures ----------------------------------------
    geometry = _make_geometry(grid, source_metadata)
    roi = _input_roi(source_metadata, geolocation, grid, rows, cols)
    exploitation = _build_exploitation_features(
        source_metadata, geometry, sensor_name, collection_date_time, roi,
    )

    # --- ProductCreation ---------------------------------------------
    now = datetime.now(timezone.utc).isoformat(
        timespec='seconds',
    ).replace('+00:00', 'Z')
    if product_name is None:
        core = None
        if source_metadata is not None and source_metadata.collection_info:
            core = source_metadata.collection_info.core_name
        product_name = f"{core} {product_class}" if core else product_class

    product_creation = SIDDProductCreation(
        processor_information=SIDDProcessorInformation(
            application=application,
            processing_date_time=now,
            site=site,
        ),
        classification=SIDDClassification(
            classification=classification,
            owner_producer=owner_producer,
            create_date=now[:10],
        ),
        product_name=product_name,
        product_class=product_class,
        product_type=product_type,
    )

    # --- Optional sections carried from the source -------------------
    radiometric = _build_radiometric(source_metadata, sample_map)
    error_statistics = (
        source_metadata.error_statistics
        if source_metadata is not None else None
    )
    match_info = (
        source_metadata.match_info if source_metadata is not None else None
    )

    return SIDDMetadata(
        format='SIDD',
        rows=rows,
        cols=cols,
        dtype=np.dtype(PIXEL_TYPES[pixel_type][1]).name,
        product_creation=product_creation,
        display=SIDDDisplay(
            pixel_type=pixel_type,
            num_bands=num_bands,
            default_band_display=1,
        ),
        geo_data=geo_data,
        measurement=measurement,
        exploitation_features=exploitation,
        radiometric=radiometric,
        error_statistics=error_statistics,
        match_info=match_info,
        product_processing=_build_product_processing(
            grid, projection, interpolation, processing_parameters,
        ),
        digital_elevation_data=_build_digital_elevation_data(
            geolocation, dem_posts_per_degree, grid, rows, cols,
        ),
    )


def _resolve_shape(
    shape: Sequence[int],
    grid: Any,
) -> Tuple[int, int]:
    """Extract ``(rows, cols)`` from a product shape, ignoring bands.

    Parameters
    ----------
    shape : Sequence[int]
        Product raster shape with two or three entries.
    grid : OutputGridProtocol
        Grid the product was sampled on.

    Returns
    -------
    Tuple[int, int]
        ``(rows, cols)``.

    Raises
    ------
    ValidationError
        If the shape does not match the grid dimensions.
    """
    dims = tuple(int(d) for d in shape)
    expected = (int(grid.rows), int(grid.cols))

    if len(dims) == 2:
        found = dims
    elif len(dims) == 3:
        if dims[1:] == expected:
            found = dims[1:]
        elif dims[:2] == expected:
            found = dims[:2]
        else:
            raise ValidationError(
                f"Product shape {dims} does not contain the grid "
                f"dimensions {expected} on either the leading or "
                f"trailing axes"
            )
    else:
        raise ValidationError(
            f"Product shape must have 2 or 3 entries, got {dims}"
        )

    if found != expected:
        raise ValidationError(
            f"Product shape {dims} disagrees with the grid dimensions "
            f"{expected}"
        )
    return found


def _make_geometry(
    grid: Any,
    source_metadata: Optional[SICDMetadata],
) -> Optional[_ExploitationGeometry]:
    """Build the exploitation geometry calculator, if possible.

    Parameters
    ----------
    grid : OutputGridProtocol
        Product grid.
    source_metadata : SICDMetadata or None
        Source metadata carrying the SCP and ARP state at COA.

    Returns
    -------
    _ExploitationGeometry or None
        None when the SICD lacks the SCP or ARP state, or when the grid
        is too small to derive product plane vectors.
    """
    if source_metadata is None:
        return None
    geo_data = source_metadata.geo_data
    scpcoa = source_metadata.scpcoa
    if geo_data is None or geo_data.scp is None or scpcoa is None:
        return None
    if geo_data.scp.ecf is None or scpcoa.arp_pos is None:
        return None
    if scpcoa.arp_vel is None:
        return None

    vectors = _product_plane_vectors(grid)
    if vectors is None:
        return None
    row_vec, col_vec = vectors

    scp_ecf = geo_data.scp.ecf.to_array()
    lat, lon, _ = _ecef_to_latlon(scp_ecf)
    _, _, etp = _enu_basis_ecef(lat, lon)

    try:
        return _ExploitationGeometry(
            scp_ecf,
            scpcoa.arp_pos.to_array(),
            scpcoa.arp_vel.to_array(),
            row_vec,
            col_vec,
            etp,
        )
    except ValidationError:
        logger.warning(
            "Could not derive product exploitation geometry from the "
            "source SICD; ExploitationFeatures geometry omitted."
        )
        return None
