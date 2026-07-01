# -*- coding: utf-8 -*-
"""
Coordinate-only fast path for GeoJSON.

Parses a GeoJSON file (or already-loaded dict) into flat numpy coordinate
arrays without constructing ``Feature`` objects, shapely geometries, or
per-feature UUIDs.  Designed for the common case where only the vertex
coordinates are needed -- e.g. loading lat/lon points and polygon rings,
projecting them into image space via a ``Geolocation``, and overlaying
them on imagery to examine detections.

All vertices are concatenated into a single ``(N, 2)`` array so a whole
file can be transformed with one vectorized ``Geolocation.latlon_to_image``
call.  A part-offset index lets callers slice back out individual points,
lines, and polygon rings after transformation.

Supported geometry types: ``Point``, ``MultiPoint``, ``LineString``,
``MultiLineString``, ``Polygon``, ``MultiPolygon``, ``GeometryCollection``.

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
2026-06-04

Modified
--------
2026-06-04
"""

# Standard library
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.base import Geolocation

# Part-type tags stored per part in ``CoordSet.part_types``.
PART_POINT = 'point'
PART_LINE = 'line'
PART_EXTERIOR = 'exterior'
PART_HOLE = 'hole'


@dataclass
class CoordSet:
    """
    Flat coordinate arrays extracted from a GeoJSON source.

    A ``CoordSet`` holds every vertex of every geometry stacked into one
    ``(N, 2)`` array, plus a part-offset index describing how those
    vertices group into points, lines, and polygon rings.  The same
    container is reused for both geographic and pixel coordinates; the
    ``space`` field records which.

    Parameters
    ----------
    xy : np.ndarray
        ``(N, 2)`` float64 array of all vertices.  For ``space='lonlat'``
        columns are ``[lon, lat]`` (GeoJSON axis order).  For
        ``space='pixel'`` columns are ``[row, col]``.
    offsets : np.ndarray
        ``(P + 1,)`` int64 array.  Part ``p`` occupies
        ``xy[offsets[p]:offsets[p + 1]]``.
    part_types : np.ndarray
        ``(P,)`` array of part-type tags (``'point'``, ``'line'``,
        ``'exterior'``, ``'hole'``).
    feature_index : np.ndarray
        ``(P,)`` int64 array giving the source feature index of each part.
    n_features : int
        Number of features in the source collection (including any with
        null or empty geometry, which contribute no parts).
    space : str
        ``'lonlat'`` or ``'pixel'``.  Default ``'lonlat'``.
    """

    xy: np.ndarray
    offsets: np.ndarray
    part_types: np.ndarray
    feature_index: np.ndarray
    n_features: int
    space: str = 'lonlat'

    # -----------------------------------------------------------------
    # Sizes
    # -----------------------------------------------------------------
    @property
    def n_parts(self) -> int:
        """Number of parts (points, lines, and polygon rings)."""
        return int(self.offsets.shape[0] - 1)

    @property
    def n_vertices(self) -> int:
        """Total number of vertices across all parts."""
        return int(self.xy.shape[0])

    def __len__(self) -> int:
        return self.n_features

    # -----------------------------------------------------------------
    # Coordinate views
    # -----------------------------------------------------------------
    @property
    def latlon(self) -> np.ndarray:
        """
        Vertices as an ``(N, 2)`` ``[lat, lon]`` array.

        This is the axis order expected by ``Geolocation.latlon_to_image``.
        Only valid when ``space == 'lonlat'``.

        Returns
        -------
        np.ndarray
            ``(N, 2)`` array with columns ``[lat, lon]``.

        Raises
        ------
        ValueError
            If the coordinates are not in geographic space.
        """
        if self.space != 'lonlat':
            raise ValueError(
                f"latlon requires space='lonlat', got {self.space!r}")
        return self.xy[:, ::-1]

    # -----------------------------------------------------------------
    # Part access
    # -----------------------------------------------------------------
    def part(self, index: int) -> np.ndarray:
        """
        Return the vertices of a single part.

        Parameters
        ----------
        index : int
            Part index in ``[0, n_parts)``.

        Returns
        -------
        np.ndarray
            ``(M, 2)`` view into ``xy`` for that part.
        """
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.xy[start:stop]

    def iter_parts(self) -> Iterator[Tuple[int, str, np.ndarray]]:
        """
        Iterate over parts.

        Yields
        ------
        Tuple[int, str, np.ndarray]
            ``(feature_index, part_type, vertices)`` for each part, where
            ``vertices`` is an ``(M, 2)`` view into ``xy``.
        """
        for p in range(self.n_parts):
            yield (
                int(self.feature_index[p]),
                str(self.part_types[p]),
                self.part(p),
            )

    # -----------------------------------------------------------------
    # Image-space projection
    # -----------------------------------------------------------------
    def to_image(
        self,
        geolocation: Geolocation,
        height: float = 0.0,
    ) -> 'CoordSet':
        """
        Project all vertices into image pixel space in one batched call.

        Parameters
        ----------
        geolocation : Geolocation
            Geolocation object for the target image.  If a DEM is attached
            to ``geolocation.elevation`` it is used internally for terrain
            correction.
        height : float
            Fallback height (HAE, meters) for vertices, used when no DEM is
            attached.  Default 0.0.

        Returns
        -------
        CoordSet
            New ``CoordSet`` with ``space='pixel'`` and ``xy`` columns
            ``[row, col]``.  Part structure is preserved.

        Raises
        ------
        ValueError
            If this ``CoordSet`` is not in geographic space.
        """
        if self.space != 'lonlat':
            raise ValueError(
                f"to_image requires space='lonlat', got {self.space!r}")
        if self.n_vertices == 0:
            rowcol = np.empty((0, 2), dtype=np.float64)
        else:
            rowcol = geolocation.latlon_to_image(self.latlon, height=height)
        return CoordSet(
            xy=np.asarray(rowcol, dtype=np.float64),
            offsets=self.offsets.copy(),
            part_types=self.part_types.copy(),
            feature_index=self.feature_index.copy(),
            n_features=self.n_features,
            space='pixel',
        )


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------
def _emit_geometry(
    geom: Optional[Dict[str, Any]],
    feature: int,
    verts: List[List[float]],
    offsets: List[int],
    types: List[str],
    feat_idx: List[int],
    holes: bool,
) -> None:
    """Flatten one GeoJSON geometry into part buffers."""
    if geom is None:
        return
    gtype = geom.get('type')
    c = geom.get('coordinates')

    if gtype == 'Point':
        verts.append(c)
        offsets.append(len(verts))
        types.append(PART_POINT)
        feat_idx.append(feature)
    elif gtype == 'MultiPoint':
        for pt in c:
            verts.append(pt)
            offsets.append(len(verts))
            types.append(PART_POINT)
            feat_idx.append(feature)
    elif gtype == 'LineString':
        verts.extend(c)
        offsets.append(len(verts))
        types.append(PART_LINE)
        feat_idx.append(feature)
    elif gtype == 'MultiLineString':
        for line in c:
            verts.extend(line)
            offsets.append(len(verts))
            types.append(PART_LINE)
            feat_idx.append(feature)
    elif gtype == 'Polygon':
        _emit_polygon(c, feature, verts, offsets, types, feat_idx, holes)
    elif gtype == 'MultiPolygon':
        for poly in c:
            _emit_polygon(poly, feature, verts, offsets, types,
                          feat_idx, holes)
    elif gtype == 'GeometryCollection':
        for sub in geom.get('geometries', []):
            _emit_geometry(sub, feature, verts, offsets, types,
                           feat_idx, holes)
    else:
        raise ValueError(f"Unsupported geometry type: {gtype!r}")


def _emit_polygon(
    rings: List[List[List[float]]],
    feature: int,
    verts: List[List[float]],
    offsets: List[int],
    types: List[str],
    feat_idx: List[int],
    holes: bool,
) -> None:
    """Flatten a single polygon's rings (exterior first, then holes)."""
    for ri, ring in enumerate(rings):
        if ri > 0 and not holes:
            break
        verts.extend(ring)
        offsets.append(len(verts))
        types.append(PART_EXTERIOR if ri == 0 else PART_HOLE)
        feat_idx.append(feature)


def coords_from_geojson(
    geojson: Dict[str, Any],
    holes: bool = True,
) -> CoordSet:
    """
    Extract coordinates from an already-loaded GeoJSON dict.

    Parameters
    ----------
    geojson : Dict[str, Any]
        A GeoJSON ``FeatureCollection``, ``Feature``, or bare geometry.
    holes : bool
        If True, polygon interior rings (holes) are included as ``'hole'``
        parts.  If False, only exterior rings are kept.  Default True.

    Returns
    -------
    CoordSet
        Geographic-space coordinate set (``space='lonlat'``).
    """
    gtype = geojson.get('type')
    if gtype == 'FeatureCollection':
        features = geojson.get('features', [])
        geoms = [f.get('geometry') for f in features]
    elif gtype == 'Feature':
        geoms = [geojson.get('geometry')]
    else:
        # Bare geometry object.
        geoms = [geojson]

    verts: List[List[float]] = []
    offsets: List[int] = [0]
    types: List[str] = []
    feat_idx: List[int] = []

    for fi, geom in enumerate(geoms):
        _emit_geometry(geom, fi, verts, offsets, types, feat_idx, holes)

    xy = (np.asarray(verts, dtype=np.float64)
          if verts else np.empty((0, 2), dtype=np.float64))
    return CoordSet(
        xy=xy,
        offsets=np.asarray(offsets, dtype=np.int64),
        part_types=np.asarray(types, dtype='<U8'),
        feature_index=np.asarray(feat_idx, dtype=np.int64),
        n_features=len(geoms),
        space='lonlat',
    )


def read_coords(
    path: Union[str, Path],
    holes: bool = True,
) -> CoordSet:
    """
    Read GeoJSON coordinates from a file without building Feature objects.

    This is a fast, low-overhead alternative to ``VectorReader.read`` for
    when only vertex coordinates are needed.  It skips shapely geometry
    construction, per-feature ``Feature`` objects, property copies, and
    UUID generation, building flat numpy arrays directly.

    Parameters
    ----------
    path : str or Path
        Path to a ``.geojson`` / ``.json`` file.
    holes : bool
        If True, include polygon interior rings.  Default True.

    Returns
    -------
    CoordSet
        Geographic-space coordinate set (``space='lonlat'``).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return coords_from_geojson(data, holes=holes)
