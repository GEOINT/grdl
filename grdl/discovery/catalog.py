# -*- coding: utf-8 -*-
"""
Discovery Catalog - In-memory metadata catalog with spatial/temporal queries.

Provides ``LocalCatalog`` for storing, filtering, and exporting scan
results with spatial overlap detection, proximity search, temporal
neighbor finding, co-location analysis, pair discovery, and region
clustering.  Optional SQLite persistence for cross-session retention.

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
2026-03-29

Modified
--------
2026-03-29
"""

# Standard library
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import numpy as np

# GRDL internal
from grdl.geolocation.utils import geographic_distance, geographic_distance_batch

from grdl.discovery.scanner import ScanResult


class LocalCatalog:
    """In-memory catalog of scan results with filtering and GeoJSON export.

    Parameters
    ----------
    db_path : str or Path, optional
        Path to a SQLite database for persistence.  If ``None``, the
        catalog is purely in-memory and lost on exit.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self._items: Dict[str, ScanResult] = {}
        self._db_path = Path(db_path) if db_path else None
        if self._db_path is not None:
            self._init_db()
            self._load_from_db()

    # ── Public API ────────────────────────────────────────────────────

    def add(self, result: ScanResult) -> None:
        """Add a single scan result to the catalog.

        Parameters
        ----------
        result : ScanResult
            Scan result to add.  Keyed by ``filepath``.
        """
        key = str(result.filepath)
        self._items[key] = result
        if self._db_path is not None:
            self._persist_one(result)

    def add_batch(self, results: List[ScanResult]) -> None:
        """Add multiple scan results.

        Parameters
        ----------
        results : list of ScanResult
            Results to add.
        """
        for r in results:
            self._items[str(r.filepath)] = r
        if self._db_path is not None:
            self._persist_batch(results)

    def get(self, filepath: Union[str, Path]) -> Optional[ScanResult]:
        """Retrieve a single result by filepath.

        Parameters
        ----------
        filepath : str or Path
            File path to look up.

        Returns
        -------
        ScanResult or None
        """
        return self._items.get(str(filepath))

    def filter(
        self,
        modality: Optional[str] = None,
        sensor: Optional[str] = None,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        format: Optional[str] = None,
        text: Optional[str] = None,
    ) -> List[ScanResult]:
        """Filter catalog items.

        Parameters
        ----------
        modality : str, optional
            Filter by modality (e.g., ``'SAR'``, ``'EO'``).
        sensor : str, optional
            Filter by sensor/platform name (case-insensitive substring).
        date_start : datetime, optional
            Minimum acquisition date (inclusive).
        date_end : datetime, optional
            Maximum acquisition date (inclusive).
        bbox : tuple, optional
            Bounding box ``(west, south, east, north)`` in degrees.
            Items whose bounds intersect are included.
        format : str, optional
            Filter by format string (case-insensitive substring).
        text : str, optional
            Free-text search across filepath and sensor name.

        Returns
        -------
        list of ScanResult
        """
        results = list(self._items.values())

        if modality is not None:
            mod_upper = modality.upper()
            results = [
                r for r in results
                if r.modality and r.modality.upper() == mod_upper
            ]

        if sensor is not None:
            sensor_lower = sensor.lower()
            results = [
                r for r in results
                if r.sensor and sensor_lower in r.sensor.lower()
            ]

        if date_start is not None:
            results = [
                r for r in results
                if r.datetime is not None and r.datetime >= date_start
            ]

        if date_end is not None:
            results = [
                r for r in results
                if r.datetime is not None and r.datetime <= date_end
            ]

        if bbox is not None:
            west, south, east, north = bbox
            results = [
                r for r in results
                if r.bounds is not None and _bbox_intersects(
                    r.bounds, (west, south, east, north)
                )
            ]

        if format is not None:
            fmt_lower = format.lower()
            results = [
                r for r in results
                if fmt_lower in r.format.lower()
            ]

        if text is not None:
            text_lower = text.lower()
            results = [
                r for r in results
                if text_lower in str(r.filepath).lower()
                or (r.sensor and text_lower in r.sensor.lower())
                or text_lower in r.format.lower()
            ]

        return results

    # ── Spatial / temporal query helpers ────────────────────────────

    def _build_spatial_index(self) -> Tuple[
        List[ScanResult], np.ndarray, np.ndarray,
    ]:
        """Build vectorized arrays of bbox centers for fast queries.

        Returns
        -------
        items : list of ScanResult
            Items with valid bounds (no errors).
        centers : np.ndarray
            Shape ``(N, 2)`` array of ``[lat, lon]`` centers.
        bounds : np.ndarray
            Shape ``(N, 4)`` array of ``[west, south, east, north]``.
        """
        items = []
        center_list = []
        bounds_list = []
        for r in self._items.values():
            if r.bounds is None or r.error is not None:
                continue
            w, s, e, n = r.bounds
            items.append(r)
            center_list.append([(s + n) / 2, (w + e) / 2])
            bounds_list.append([w, s, e, n])
        if not items:
            return [], np.empty((0, 2)), np.empty((0, 4))
        return (
            items,
            np.array(center_list, dtype=np.float64),
            np.array(bounds_list, dtype=np.float64),
        )

    def find_overlapping(
        self,
        target: Union[str, Path, ScanResult],
        min_overlap_fraction: float = 0.0,
    ) -> List[Tuple[ScanResult, float]]:
        """Find catalog items whose footprint overlaps a target.

        Parameters
        ----------
        target : str, Path, or ScanResult
            A filepath (looked up in catalog) or a ScanResult.
        min_overlap_fraction : float
            Minimum overlap as a fraction of the smaller bounding box
            area.  ``0.0`` returns any intersection; ``0.5`` requires
            at least 50 % overlap.  Range ``[0, 1]``.

        Returns
        -------
        list of (ScanResult, float)
            Matching items with their overlap fraction, sorted from
            highest to lowest overlap.  The target itself is excluded.
        """
        ref = self._resolve_target(target)
        if ref is None or ref.bounds is None:
            return []

        items, _, bounds_arr = self._build_spatial_index()
        if len(items) == 0:
            return []

        ref_key = str(ref.filepath)
        rw, rs, re, rn = ref.bounds
        ref_area = (re - rw) * (rn - rs)

        # Vectorized intersection
        i_w = np.maximum(bounds_arr[:, 0], rw)
        i_s = np.maximum(bounds_arr[:, 1], rs)
        i_e = np.minimum(bounds_arr[:, 2], re)
        i_n = np.minimum(bounds_arr[:, 3], rn)

        i_width = np.maximum(i_e - i_w, 0)
        i_height = np.maximum(i_n - i_s, 0)
        i_area = i_width * i_height

        item_area = (bounds_arr[:, 2] - bounds_arr[:, 0]) * (bounds_arr[:, 3] - bounds_arr[:, 1])
        min_area = np.minimum(item_area, ref_area)
        min_area = np.where(min_area > 0, min_area, 1.0)
        fracs = np.minimum(i_area / min_area, 1.0)

        results = []
        for idx in np.where(fracs > min_overlap_fraction)[0]:
            r = items[idx]
            if str(r.filepath) == ref_key:
                continue
            results.append((r, float(fracs[idx])))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_nearby(
        self,
        lat: float,
        lon: float,
        radius_km: float = 100.0,
    ) -> List[Tuple[ScanResult, float]]:
        """Find catalog items near a geographic point.

        Uses vectorized Haversine via ``geographic_distance_batch``.

        Parameters
        ----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        radius_km : float
            Search radius in kilometers.  Default 100 km.

        Returns
        -------
        list of (ScanResult, float)
            Matching items with their distance in km, sorted nearest
            first.
        """
        items, centers, _ = self._build_spatial_index()
        if len(items) == 0:
            return []

        dists_m = geographic_distance_batch(
            np.full(len(items), lat),
            np.full(len(items), lon),
            centers[:, 0],
            centers[:, 1],
        )
        dists_km = dists_m / 1000

        mask = dists_km <= radius_km
        indices = np.where(mask)[0]
        order = indices[np.argsort(dists_km[indices])]

        return [(items[i], float(dists_km[i])) for i in order]

    def find_containing(
        self,
        lat: float,
        lon: float,
    ) -> List[ScanResult]:
        """Find catalog items whose bounding box contains a point.

        Parameters
        ----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.

        Returns
        -------
        list of ScanResult
            Items whose bounds contain the point.
        """
        items, _, bounds_arr = self._build_spatial_index()
        if len(items) == 0:
            return []

        mask = (
            (bounds_arr[:, 0] <= lon) & (lon <= bounds_arr[:, 2])
            & (bounds_arr[:, 1] <= lat) & (lat <= bounds_arr[:, 3])
        )
        return [items[i] for i in np.where(mask)[0]]

    def find_same_region(
        self,
        target: Union[str, Path, ScanResult],
        radius_km: float = 50.0,
    ) -> List[Tuple[ScanResult, float]]:
        """Find items covering roughly the same geographic region.

        Parameters
        ----------
        target : str, Path, or ScanResult
            Reference item (filepath or ScanResult).
        radius_km : float
            Maximum center-to-center distance in km.  Default 50 km.

        Returns
        -------
        list of (ScanResult, float)
            Matching items with distance in km, nearest first.
            The target itself is excluded.
        """
        ref = self._resolve_target(target)
        if ref is None or ref.bounds is None:
            return []

        items, centers, _ = self._build_spatial_index()
        if len(items) == 0:
            return []

        w, s, e, n = ref.bounds
        clat, clon = (s + n) / 2, (w + e) / 2
        ref_key = str(ref.filepath)

        dists_m = geographic_distance_batch(
            np.full(len(items), clat),
            np.full(len(items), clon),
            centers[:, 0],
            centers[:, 1],
        )
        dists_km = dists_m / 1000

        mask = dists_km <= radius_km
        results = []
        for idx in np.where(mask)[0]:
            if str(items[idx].filepath) == ref_key:
                continue
            results.append((items[idx], float(dists_km[idx])))

        results.sort(key=lambda x: x[1])
        return results

    def find_temporal_neighbors(
        self,
        target: Union[str, Path, ScanResult],
        window_days: float = 30.0,
    ) -> List[Tuple[ScanResult, float]]:
        """Find items acquired within a time window of a target.

        Parameters
        ----------
        target : str, Path, or ScanResult
            Reference item.
        window_days : float
            Maximum time separation in days.  Default 30.

        Returns
        -------
        list of (ScanResult, float)
            Matching items with their time offset in days (absolute),
            sorted nearest in time first.  The target itself is
            excluded.
        """
        ref = self._resolve_target(target)
        if ref is None or ref.datetime is None:
            return []

        ref_key = str(ref.filepath)
        ref_ts = ref.datetime.timestamp()
        window_s = window_days * 86400

        # Build timestamp array for items that have datetime
        timed_items = []
        ts_list = []
        for r in self._items.values():
            if r.datetime is not None and str(r.filepath) != ref_key:
                timed_items.append(r)
                ts_list.append(r.datetime.timestamp())

        if not timed_items:
            return []

        ts_arr = np.array(ts_list, dtype=np.float64)
        deltas_s = np.abs(ts_arr - ref_ts)
        mask = deltas_s <= window_s
        indices = np.where(mask)[0]
        order = indices[np.argsort(deltas_s[indices])]

        return [
            (timed_items[i], float(deltas_s[i] / 86400))
            for i in order
        ]

    def find_colocated(
        self,
        target: Union[str, Path, ScanResult],
        radius_km: float = 50.0,
        window_days: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """Find items in the same region AND similar time window.

        Combines vectorized spatial proximity with temporal proximity.

        Parameters
        ----------
        target : str, Path, or ScanResult
            Reference item.
        radius_km : float
            Maximum center-to-center distance in km.  Default 50 km.
        window_days : float
            Maximum time separation in days.  Default 30.

        Returns
        -------
        list of dict
            Each dict contains:

            - ``'result'``: the matching ``ScanResult``
            - ``'distance_km'``: spatial distance
            - ``'time_offset_days'``: temporal offset (absolute)
            - ``'same_sensor'``: bool
            - ``'same_modality'``: bool
        """
        ref = self._resolve_target(target)
        if ref is None or ref.bounds is None:
            return []

        items, centers, _ = self._build_spatial_index()
        if len(items) == 0:
            return []

        w, s, e, n = ref.bounds
        clat, clon = (s + n) / 2, (w + e) / 2
        ref_key = str(ref.filepath)
        ref_dt = ref.datetime
        window_s = window_days * 86400

        # Vectorized spatial filter
        dists_m = geographic_distance_batch(
            np.full(len(items), clat),
            np.full(len(items), clon),
            centers[:, 0],
            centers[:, 1],
        )
        dists_km = dists_m / 1000
        spatial_mask = dists_km <= radius_km

        results = []
        for idx in np.where(spatial_mask)[0]:
            r = items[idx]
            if str(r.filepath) == ref_key:
                continue

            time_offset = None
            if ref_dt is not None and r.datetime is not None:
                delta_s = abs(r.datetime.timestamp() - ref_dt.timestamp())
                if delta_s > window_s:
                    continue
                time_offset = delta_s / 86400

            results.append({
                'result': r,
                'distance_km': round(float(dists_km[idx]), 2),
                'time_offset_days': round(time_offset, 2) if time_offset is not None else None,
                'same_sensor': (
                    ref.sensor is not None
                    and r.sensor is not None
                    and ref.sensor.lower() == r.sensor.lower()
                ),
                'same_modality': (
                    ref.modality is not None
                    and r.modality is not None
                    and ref.modality.upper() == r.modality.upper()
                ),
            })

        results.sort(key=lambda x: (x['distance_km'], x['time_offset_days'] or 0))
        return results

    def find_pairs(
        self,
        max_distance_km: float = 10.0,
        max_time_delta_days: float = 365.0,
        same_modality: bool = False,
        cross_sensor: bool = False,
    ) -> List[Dict[str, Any]]:
        """Find all pairs of items that are spatially and temporally close.

        Uses a vectorized distance matrix for O(N^2) pair comparison
        without Python inner loops.

        Parameters
        ----------
        max_distance_km : float
            Maximum center-to-center distance.  Default 10 km.
        max_time_delta_days : float
            Maximum time separation.  Default 365 days.
        same_modality : bool
            If ``True``, only return pairs with matching modality.
        cross_sensor : bool
            If ``True``, only return pairs from *different* sensors.

        Returns
        -------
        list of dict
            Each dict contains ``'a'``, ``'b'``, ``'distance_km'``,
            ``'time_delta_days'``, ``'modality_a'``, ``'modality_b'``,
            ``'sensor_a'``, ``'sensor_b'``.
        """
        items, centers, _ = self._build_spatial_index()
        n = len(items)
        if n < 2:
            return []

        # Vectorized pairwise distance matrix (upper triangle)
        # Build all (i, j) pairs with i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        dists_m = geographic_distance_batch(
            centers[i_idx, 0], centers[i_idx, 1],
            centers[j_idx, 0], centers[j_idx, 1],
        )
        dists_km = dists_m / 1000

        # Spatial filter
        spatial_mask = dists_km <= max_distance_km
        pair_i = i_idx[spatial_mask]
        pair_j = j_idx[spatial_mask]
        pair_dists = dists_km[spatial_mask]

        pairs = []
        for p in range(len(pair_i)):
            a = items[pair_i[p]]
            b = items[pair_j[p]]

            if same_modality:
                if not (a.modality and b.modality
                        and a.modality.upper() == b.modality.upper()):
                    continue

            if cross_sensor:
                if (a.sensor and b.sensor
                        and a.sensor.lower() == b.sensor.lower()):
                    continue

            time_delta = None
            if a.datetime is not None and b.datetime is not None:
                td = abs(a.datetime.timestamp() - b.datetime.timestamp()) / 86400
                if td > max_time_delta_days:
                    continue
                time_delta = td

            pairs.append({
                'a': a,
                'b': b,
                'distance_km': round(float(pair_dists[p]), 2),
                'time_delta_days': round(time_delta, 2) if time_delta is not None else None,
                'modality_a': a.modality,
                'modality_b': b.modality,
                'sensor_a': a.sensor,
                'sensor_b': b.sensor,
            })

        pairs.sort(key=lambda x: (x['distance_km'], x['time_delta_days'] or 0))
        return pairs

    def group_by_region(
        self,
        cluster_radius_km: float = 25.0,
    ) -> List[List[ScanResult]]:
        """Group catalog items into spatial clusters.

        Computes a full pairwise distance matrix and uses it for
        greedy seed-and-expand clustering.

        Parameters
        ----------
        cluster_radius_km : float
            Maximum distance from any cluster member to join.
            Default 25 km.

        Returns
        -------
        list of list of ScanResult
            Each inner list is a spatial cluster, sorted largest first.
        """
        items, centers, _ = self._build_spatial_index()
        n = len(items)
        if n == 0:
            return []

        # Full pairwise distance matrix
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        if n > 1:
            i_idx, j_idx = np.triu_indices(n, k=1)
            dists_m = geographic_distance_batch(
                centers[i_idx, 0], centers[i_idx, 1],
                centers[j_idx, 0], centers[j_idx, 1],
            )
            dists_km = dists_m / 1000
            dist_matrix[i_idx, j_idx] = dists_km
            dist_matrix[j_idx, i_idx] = dists_km

        assigned = np.zeros(n, dtype=bool)
        clusters: List[List[ScanResult]] = []

        for seed in range(n):
            if assigned[seed]:
                continue

            # Expand cluster from seed using distance matrix
            members = {seed}
            frontier = {seed}
            assigned[seed] = True

            while frontier:
                new_frontier = set()
                for m in frontier:
                    neighbors = np.where(
                        (~assigned) & (dist_matrix[m] <= cluster_radius_km)
                    )[0]
                    for nb in neighbors:
                        if nb not in members:
                            members.add(nb)
                            assigned[nb] = True
                            new_frontier.add(nb)
                frontier = new_frontier

            clusters.append([items[i] for i in sorted(members)])

        clusters.sort(key=len, reverse=True)
        return clusters

    # ── Internal ──────────────────────────────────────────────────

    def _resolve_target(
        self, target: Union[str, Path, ScanResult],
    ) -> Optional[ScanResult]:
        """Resolve a target to a ScanResult."""
        if isinstance(target, ScanResult):
            return target
        return self.get(target)

    def get_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics over the catalog.

        Returns
        -------
        dict
            Keys: ``total``, ``by_modality``, ``by_sensor``,
            ``by_format``, ``date_range``, ``errors``.
        """
        items = list(self._items.values())
        by_modality: Dict[str, int] = {}
        by_sensor: Dict[str, int] = {}
        by_format: Dict[str, int] = {}
        dates: List[datetime] = []
        errors = 0

        for r in items:
            if r.error:
                errors += 1
            mod = r.modality or 'Unknown'
            by_modality[mod] = by_modality.get(mod, 0) + 1
            sen = r.sensor or 'Unknown'
            by_sensor[sen] = by_sensor.get(sen, 0) + 1
            fmt = r.format or 'Unknown'
            by_format[fmt] = by_format.get(fmt, 0) + 1
            if r.datetime is not None:
                dates.append(r.datetime)

        date_range = None
        if dates:
            date_range = {
                'earliest': min(dates).isoformat(),
                'latest': max(dates).isoformat(),
            }

        return {
            'total': len(items),
            'by_modality': by_modality,
            'by_sensor': by_sensor,
            'by_format': by_format,
            'date_range': date_range,
            'errors': errors,
        }

    def to_geojson(
        self,
        items: Optional[List[ScanResult]] = None,
    ) -> Dict[str, Any]:
        """Convert catalog items to a GeoJSON FeatureCollection.

        Parameters
        ----------
        items : list of ScanResult, optional
            Subset to convert.  Defaults to all items.

        Returns
        -------
        dict
            GeoJSON FeatureCollection.  Items without a footprint are
            included with a ``null`` geometry (so they still appear in
            metadata tables).
        """
        if items is None:
            items = list(self._items.values())

        features = []
        for r in items:
            geometry = _normalize_geojson_geometry(r.footprint)
            fname = r.filepath.name if isinstance(r.filepath, Path) else Path(r.filepath).name
            base_props = {
                'filepath': str(r.filepath),
                'filename': fname,
                'format': r.format,
                'modality': r.modality,
                'sensor': r.sensor,
                'rows': r.rows,
                'cols': r.cols,
                'dtype': r.dtype,
                'bands': r.bands,
                'datetime': r.datetime.isoformat() if r.datetime else None,
                'bounds': list(r.bounds) if r.bounds else None,
                'error': r.error,
                'feature_type': 'footprint',
            }
            features.append({
                'type': 'Feature',
                'geometry': geometry,
                'properties': base_props,
            })

            # Add geospatial point features
            gs = r.geospatial

            # SCP marker (SICD)
            scp = gs.get('scp')
            if scp:
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [scp['lon'], scp['lat']],
                    },
                    'properties': {
                        'feature_type': 'scp',
                        'filepath': str(r.filepath),
                        'filename': fname,
                        'hae': scp.get('hae', 0),
                        'label': f"SCP {fname}",
                    },
                })

            # Reference point (SIDD)
            rp = gs.get('reference_point')
            if rp:
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [rp['lon'], rp['lat']],
                    },
                    'properties': {
                        'feature_type': 'reference_point',
                        'filepath': str(r.filepath),
                        'filename': fname,
                        'alt': rp.get('alt', 0),
                        'label': f"Ref {fname}",
                    },
                })

            # Scene center (TerraSAR)
            sc = gs.get('scene_center')
            if sc:
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [sc['lon'], sc['lat']],
                    },
                    'properties': {
                        'feature_type': 'scene_center',
                        'filepath': str(r.filepath),
                        'filename': fname,
                        'label': f"Center {fname}",
                    },
                })

            # Orbit ground track (all formats with orbit data)
            orbit = gs.get('orbit_track')
            if orbit and len(orbit) >= 2:
                coords = [[p['lon'], p['lat']] for p in orbit]
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': coords,
                    },
                    'properties': {
                        'feature_type': 'orbit_track',
                        'filepath': str(r.filepath),
                        'filename': fname,
                        'modality': r.modality,
                        'points': len(orbit),
                    },
                })

            # GCPs (BIOMASS)
            gcps = gs.get('gcps')
            if gcps:
                gcp_coords = [
                    [g['lon'], g['lat']] for g in gcps
                ]
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'MultiPoint',
                        'coordinates': gcp_coords,
                    },
                    'properties': {
                        'feature_type': 'gcps',
                        'filepath': str(r.filepath),
                        'filename': fname,
                        'count': len(gcps),
                        'label': f"GCPs {fname}",
                    },
                })

            # Geolocation grid (Sentinel-1, TerraSAR, NISAR)
            gg = gs.get('geolocation_grid')
            if gg and len(gg) > 0:
                gg_coords = [
                    [g['lon'], g['lat']]
                    for g in gg
                    if g.get('lat') is not None and g.get('lon') is not None
                ]
                if gg_coords:
                    features.append({
                        'type': 'Feature',
                        'geometry': {
                            'type': 'MultiPoint',
                            'coordinates': gg_coords,
                        },
                        'properties': {
                            'feature_type': 'geolocation_grid',
                            'filepath': str(r.filepath),
                            'filename': fname,
                            'count': len(gg_coords),
                            'label': f"GeoGrid {fname}",
                        },
                    })

            # Beam footprint polygons (SICD)
            for beam_key, label in [
                ('beam_footprint_3db', 'Beam -3dB'),
                ('beam_footprint_10db', 'Beam -10dB'),
            ]:
                beam = gs.get(beam_key)
                if beam and beam.get('coordinates'):
                    features.append({
                        'type': 'Feature',
                        'geometry': beam,
                        'properties': {
                            'feature_type': beam_key,
                            'filepath': str(r.filepath),
                            'filename': fname,
                            'label': f"{label} {fname}",
                        },
                    })

        return {
            'type': 'FeatureCollection',
            'features': features,
        }

    def remove(self, filepath: Union[str, Path]) -> bool:
        """Remove an item by filepath.

        Returns
        -------
        bool
            ``True`` if the item was found and removed.
        """
        key = str(filepath)
        if key in self._items:
            del self._items[key]
            if self._db_path is not None:
                self._delete_from_db(key)
            return True
        return False

    def clear(self) -> None:
        """Remove all items from the catalog."""
        self._items.clear()
        if self._db_path is not None:
            self._clear_db()

    @property
    def count(self) -> int:
        """Number of items in the catalog."""
        return len(self._items)

    def all_items(self) -> List[ScanResult]:
        """Return all items as a list.

        Returns
        -------
        list of ScanResult
        """
        return list(self._items.values())

    # ── SQLite persistence ────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the catalog table if it doesn't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_results (
                    filepath TEXT PRIMARY KEY,
                    data TEXT NOT NULL
                )
            """)

    def _persist_one(self, result: ScanResult) -> None:
        """Write a single result to the database."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scan_results (filepath, data) VALUES (?, ?)",
                (str(result.filepath), json.dumps(result.to_json())),
            )

    def _persist_batch(self, results: List[ScanResult]) -> None:
        """Write multiple results to the database."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO scan_results (filepath, data) VALUES (?, ?)",
                [(str(r.filepath), json.dumps(r.to_json())) for r in results],
            )

    def _load_from_db(self) -> None:
        """Load results from the database into memory.

        Note: This restores JSON-serializable fields only.
        ``metadata_ref`` (typed metadata objects) are not persisted.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute("SELECT filepath, data FROM scan_results")
            for filepath, data_json in cursor:
                try:
                    d = json.loads(data_json)
                    result = ScanResult(
                        filepath=Path(d.get('filepath', filepath)),
                        format=d.get('format', ''),
                        rows=d.get('rows', 0),
                        cols=d.get('cols', 0),
                        dtype=d.get('dtype', ''),
                        bands=d.get('bands'),
                        crs=d.get('crs'),
                        modality=d.get('modality'),
                        sensor=d.get('sensor'),
                        footprint=d.get('footprint'),
                        bounds=tuple(d['bounds']) if d.get('bounds') else None,
                        metadata_dict=d.get('metadata_dict', {}),
                        scan_time_ms=d.get('scan_time_ms', 0),
                        error=d.get('error'),
                    )
                    dt_str = d.get('datetime')
                    if dt_str:
                        try:
                            result.datetime = datetime.fromisoformat(dt_str)
                        except (ValueError, TypeError):
                            pass
                    self._items[filepath] = result
                except (json.JSONDecodeError, KeyError):
                    continue

    def _delete_from_db(self, filepath: str) -> None:
        """Delete a single record from the database."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "DELETE FROM scan_results WHERE filepath = ?",
                (filepath,),
            )

    def _clear_db(self) -> None:
        """Delete all records from the database."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM scan_results")


# ── Utility ───────────────────────────────────────────────────────────


def _normalize_geojson_geometry(
    footprint: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize a GRDL footprint dict to valid GeoJSON Polygon geometry.

    GRDL's ``get_footprint()`` returns coordinates as a flat list of
    ``(lon, lat)`` tuples.  GeoJSON Polygon requires
    ``[[[lon, lat], ...]]`` (list of rings).
    """
    if footprint is None:
        return None
    if footprint.get('type') in ('None', None):
        return None

    coords = footprint.get('coordinates')
    if coords is None:
        return None

    # Already proper GeoJSON: [[[lon, lat], ...]]
    if (isinstance(coords, list) and len(coords) > 0
            and isinstance(coords[0], list)
            and len(coords[0]) > 0
            and isinstance(coords[0][0], list)):
        return {'type': 'Polygon', 'coordinates': coords}

    # Flat list of (lon, lat) tuples from GRDL
    ring = [list(c) for c in coords]
    # Close the ring if not already closed
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])

    return {'type': 'Polygon', 'coordinates': [ring]}


def _bbox_intersects(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    """Test if two (west, south, east, north) bounding boxes intersect."""
    a_w, a_s, a_e, a_n = a
    b_w, b_s, b_e, b_n = b
    if a_e < b_w or b_e < a_w:
        return False
    if a_n < b_s or b_n < a_s:
        return False
    return True


def _bbox_overlap_fraction(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Compute overlap fraction of two bboxes relative to the smaller.

    Returns a value in [0, 1].  1.0 means the smaller box is fully
    contained in the larger.
    """
    a_w, a_s, a_e, a_n = a
    b_w, b_s, b_e, b_n = b

    # Intersection
    i_w = max(a_w, b_w)
    i_s = max(a_s, b_s)
    i_e = min(a_e, b_e)
    i_n = min(a_n, b_n)

    if i_e <= i_w or i_n <= i_s:
        return 0.0

    i_area = (i_e - i_w) * (i_n - i_s)
    a_area = (a_e - a_w) * (a_n - a_s)
    b_area = (b_e - b_w) * (b_n - b_s)

    min_area = min(a_area, b_area)
    if min_area <= 0:
        return 0.0

    return min(i_area / min_area, 1.0)


