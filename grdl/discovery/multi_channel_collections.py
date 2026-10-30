"""Discover polarization collections without reading imagery pixels.

Loose SICD files are associated by metadata, never by filenames. Native
multi-polarization products are kept intact, with separate records for each
NISAR frequency or Sentinel-1 swath.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from grdl.discovery.scanner import MetadataScanner, ScanResult
from grdl.IO import get_reader
from grdl.IO.models.biomass import BIOMASSMetadata
from grdl.IO.models.nisar import NISARMetadata
from grdl.IO.models.sentinel1_slc import Sentinel1SLCMetadata
from grdl.IO.models.sicd import SICDMetadata
from grdl.IO.models.terrasar import TerraSARMetadata
from grdl.IO.sar.collection_validation import (
    CollectionIssue,
    CollectionReport,
    validate_collection,
    validate_sicd_collection,
)

logger = logging.getLogger(__name__)

CollectionStatus = Literal["ready", "partial", "single", "ambiguous", "rejected"]


@dataclass
class PolarizationCollection:
    """A proposed collection and the settings needed to open it.

    ``ready`` means a validated dual-, quad- or compact-pol candidate;
    ``partial`` means three linear channels; ``single`` has no partner.
    ``ambiguous`` candidates must not be selected automatically. Metadata
    checks do not establish pixel co-registration or phase coherence.
    """

    filepaths: list[Path]
    reader: str
    reader_kwargs: dict[str, str | list[str]]
    validation: CollectionReport
    status: CollectionStatus

    @property
    def polarizations(self) -> list[str]:
        """Channel labels, in the order to use when opening the reader."""
        return self.validation.polarizations

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe manifest entry (no typed metadata or pixels)."""
        paths = [str(p) for p in self.filepaths]
        return {
            "source": paths if self.reader == "grdl.IO.sar.SICDCollectionReader" else paths[0],
            "reader": self.reader,
            "reader_kwargs": self.reader_kwargs,
            "polarizations": self.polarizations,
            "mode": self.validation.mode,
            "status": self.status,
            "validation": {
                "ok": self.validation.ok,
                "issues": [asdict(issue) for issue in self.validation.issues],
            },
        }


@dataclass
class DiscoveryProblem:
    """A scan failure or an explicitly unsupported input."""

    filepath: Path
    code: str
    message: str
    level: Literal["error", "info"] = "error"


@dataclass
class CollectionDiscoveryReport:
    """Collections plus inputs that could not be evaluated.

    Unsupported formats are informational problems. Other problems, rejected
    collections and ambiguous collections make ``has_errors`` true.
    """

    collections: list[PolarizationCollection] = field(default_factory=list)
    problems: list[DiscoveryProblem] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(p.level == "error" for p in self.problems) or any(
            c.status in ("ambiguous", "rejected") for c in self.collections
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete report, including unsuccessful inputs."""
        return {
            "schema_version": 1,
            "collections": [c.to_dict() for c in self.collections],
            "problems": [{**asdict(p), "filepath": str(p.filepath)} for p in self.problems],
        }


def _status(report: CollectionReport) -> CollectionStatus:
    if not report.ok:
        return "rejected"
    if report.mode == "single":
        return "single"
    if report.mode == "triple":
        return "partial"
    if report.mode in ("dual", "quad", "compact"):
        return "ready" if len(report.polarizations) > 1 else "single"
    return "rejected"


def _identity_issues(meta: SICDMetadata) -> list[CollectionIssue]:
    """Require enough finite metadata to associate files conservatively."""
    missing = []
    if not meta.collection_info or not meta.collection_info.collector_name:
        missing.append("collector_name")
    start = meta.timeline.collect_start if meta.timeline else None
    try:
        if not start:
            raise ValueError("missing time")
        datetime.fromisoformat(start.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        missing.append("parseable collect_start")
    scp = meta.geo_data.scp if meta.geo_data else None
    coords: tuple[float, ...]
    if scp and scp.ecf:
        coords = (scp.ecf.x, scp.ecf.y, scp.ecf.z)
    elif scp and scp.llh:
        coords = (scp.llh.lat, scp.llh.lon, scp.llh.hae)
        if not (-90 <= scp.llh.lat <= 90 and -180 <= scp.llh.lon <= 180):
            missing.append("valid scene-center latitude/longitude")
    else:
        coords = ()
    if not coords or not all(v is not None and math.isfinite(v) for v in coords):
        missing.append("finite scene center")
    if meta.rows <= 0 or meta.cols <= 0:
        missing.append("positive image dimensions")
    for axis in ("row", "col"):
        grid_axis = getattr(meta.grid, axis, None)
        spacing = grid_axis.ss if grid_axis else None
        if spacing is None or not math.isfinite(spacing) or spacing <= 0:
            missing.append(f"positive finite {axis} spacing")
    if not meta.scpcoa or not meta.scpcoa.side_of_track:
        missing.append("side_of_track")
    if meta.scpcoa:
        for name in ("graze_ang", "doppler_cone_ang"):
            angle = getattr(meta.scpcoa, name)
            if angle is not None and not math.isfinite(angle):
                missing.append(f"finite {name}")
    return (
        [
            CollectionIssue(
                "error",
                "insufficient_identity",
                "Cannot associate this file automatically; missing " + ", ".join(missing),
            )
        ]
        if missing
        else []
    )


def group_sicd_collections(
    results: Iterable[ScanResult],
    *,
    max_time_delta_s: float = 1.0,
    max_scp_distance_m: float = 25.0,
    max_spacing_rel_diff: float = 1e-3,
    max_angle_diff_deg: float = 0.5,
) -> CollectionDiscoveryReport:
    """Group freshly scanned SICD files using collection validation.

    Pairwise compatibility forms a graph. A connected component is accepted
    only when every pair is compatible and each polarization is unique.
    Duplicate channels and overlapping/non-transitive matches are reported as
    ambiguous instead of greedily choosing a file. Output order is stable.

    Required association metadata: collector, acquisition time, scene center,
    positive dimensions and sample spacing, and side of track. Other missing
    fields remain validator warnings. Filenames and core names are not keys.
    Defaults match ``validate_sicd_collection``; all thresholds must be finite
    and nonnegative. Comparisons are quadratic within a collector.

    Non-SICD records are reported as unsupported. Persisted LocalCatalog
    records must be rescanned because typed ``metadata_ref`` is not persisted.
    """
    thresholds = {
        "max_time_delta_s": max_time_delta_s,
        "max_scp_distance_m": max_scp_distance_m,
        "max_spacing_rel_diff": max_spacing_rel_diff,
        "max_angle_diff_deg": max_angle_diff_deg,
    }
    if any(not math.isfinite(v) or v < 0 for v in thresholds.values()):
        raise ValueError("Collection thresholds must be finite and nonnegative.")
    output = CollectionDiscoveryReport()
    candidates: list[tuple[ScanResult, SICDMetadata, CollectionReport]] = []
    seen: set[Path] = set()
    for result in sorted(results, key=lambda r: str(r.filepath)):
        path = result.filepath.resolve()
        if path in seen:
            raise ValueError(f"Duplicate scan result for {path}")
        seen.add(path)
        if result.error:
            output.problems.append(DiscoveryProblem(path, "scan_failed", result.error))
            continue
        if not isinstance(result.metadata_ref, SICDMetadata):
            code = "missing_metadata" if result.metadata_ref is None else "unsupported_format"
            output.problems.append(
                DiscoveryProblem(
                    path,
                    code,
                    f"Fresh SICD metadata required; found {result.format!r}.",
                    "error" if code == "missing_metadata" else "info",
                )
            )
            continue
        meta = result.metadata_ref
        report = validate_sicd_collection([meta], None, **thresholds)
        report.issues.extend(_identity_issues(meta))
        if report.mode == "other":
            report.issues.append(
                CollectionIssue(
                    "error",
                    "unresolved_polarization",
                    "A recognized, unambiguous polarization is required for automatic grouping.",
                )
            )
        if not report.ok:
            output.collections.append(_sicd_candidate([result], report, "rejected"))
        else:
            candidates.append((result, meta, report))

    neighbors: list[set[int]] = [set() for _ in candidates]
    for i, j in combinations(range(len(candidates)), 2):
        left, right = candidates[i][1], candidates[j][1]
        if left.collection_info is None or right.collection_info is None:
            continue
        if left.collection_info.collector_name != right.collection_info.collector_name:
            continue
        report = validate_sicd_collection(
            [left, right],
            [candidates[i][2].polarizations[0], candidates[j][2].polarizations[0]],
            **thresholds,
        )
        # Duplicate labels must connect too, so alternatives cannot be silently
        # resolved by path order.
        if all(issue.code == "duplicate_polarization" for issue in report.errors) and not any(
            issue.code == "missing_scp" for issue in report.warnings
        ):
            neighbors[i].add(j)
            neighbors[j].add(i)

    remaining = set(range(len(candidates)))
    while remaining:
        component: set[int] = set()
        pending = [min(remaining)]
        while pending:
            i = pending.pop()
            if i not in component:
                component.add(i)
                pending.extend(neighbors[i] - component)
        remaining -= component
        indices = sorted(
            component,
            key=lambda i: (
                candidates[i][2].polarizations[0],
                str(candidates[i][0].filepath),
            ),
        )
        report = validate_sicd_collection(
            [candidates[i][1] for i in indices],
            [candidates[i][2].polarizations[0] for i in indices],
            **thresholds,
        )
        ambiguous = any(not component - {i} <= neighbors[i] for i in component) or len(
            set(report.polarizations)
        ) != len(indices)
        if ambiguous:
            report.issues.append(
                CollectionIssue(
                    "error",
                    "ambiguous_membership",
                    "Duplicate polarizations or overlapping matches; select a collection manually.",
                )
            )
        status = "ambiguous" if ambiguous else _status(report)
        output.collections.append(
            _sicd_candidate(
                [candidates[i][0] for i in indices],
                report,
                status,
            )
        )
    output.collections.sort(key=lambda c: tuple(str(p) for p in c.filepaths))
    return output


def _sicd_candidate(
    results: list[ScanResult],
    report: CollectionReport,
    status: CollectionStatus,
) -> PolarizationCollection:
    return PolarizationCollection(
        [r.filepath.resolve() for r in results],
        "grdl.IO.sar.SICDCollectionReader",
        {"polarizations": report.polarizations},
        report,
        status,
    )


def _native_collections(result: ScanResult) -> CollectionDiscoveryReport:
    meta = result.metadata_ref
    selectors: list[dict[str, str | list[str]]] = [{}]
    if isinstance(meta, NISARMetadata):
        key, reader_name = "nisar", "NISARReader"
        selectors = [
            {"frequency": f, "polarizations": "all"}
            for f in sorted(meta.available_frequencies or [])
        ]
        if not selectors:
            raise ValueError("NISAR metadata has no available frequencies.")
    elif isinstance(meta, Sentinel1SLCMetadata):
        from grdl.IO.sar.sentinel1_slc import Sentinel1SLCReader

        key, reader_name = "sentinel1-slc", "Sentinel1SLCReader"
        with Sentinel1SLCReader(result.filepath) as reader:
            selectors = [
                {"swath": s, "polarizations": "all"} for s in sorted(reader.get_available_swaths())
            ]
        if not selectors:
            raise ValueError("Sentinel-1 product has no available swaths.")
    elif isinstance(meta, TerraSARMetadata):
        key, reader_name = "terrasar", "TerraSARReader"
        selectors = [{"polarizations": "all"}]
    elif isinstance(meta, BIOMASSMetadata):
        key, reader_name = "biomass", "BIOMASSL1Reader"
    else:
        raise ValueError(f"Unsupported native product {result.format!r}")

    output = CollectionDiscoveryReport()
    for kwargs in selectors:
        try:
            with get_reader(key, result.filepath, **kwargs) as native_reader:
                report = validate_collection(native_reader)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            message = f"{kwargs}: {exc}"
            logger.warning("Cannot enumerate polarizations in %s: %s", result.filepath, message)
            output.problems.append(
                DiscoveryProblem(
                    result.filepath.resolve(),
                    "product_open_failed",
                    message,
                )
            )
            continue
        if report.mode == "other":
            report.issues.append(
                CollectionIssue(
                    "error",
                    "unresolved_polarization",
                    "Product does not expose a recognized polarization set.",
                )
            )
        output.collections.append(
            PolarizationCollection(
                [result.filepath.resolve()],
                f"grdl.IO.sar.{reader_name}",
                kwargs,
                report,
                _status(report),
            )
        )
    return output


def discover_polarization_collections(
    directory: str | Path,
    *,
    recursive: bool = True,
    max_workers: int = 4,
    max_time_delta_s: float = 1.0,
    max_scp_distance_m: float = 25.0,
    max_spacing_rel_diff: float = 1e-3,
    max_angle_diff_deg: float = 0.5,
) -> CollectionDiscoveryReport:
    """Scan a directory and report SICD and native polarization collections.

    Native NISAR, Sentinel-1 SLC, TerraSAR and BIOMASS products are reopened
    metadata-only to enumerate their channels. No pixels or reciprocity
    samples are read. Unsupported formats and scan/open failures are retained
    in ``problems``. Native products are never merged with one another.
    """
    thresholds = {
        "max_time_delta_s": max_time_delta_s,
        "max_scp_distance_m": max_scp_distance_m,
        "max_spacing_rel_diff": max_spacing_rel_diff,
        "max_angle_diff_deg": max_angle_diff_deg,
    }
    if any(not math.isfinite(v) or v < 0 for v in thresholds.values()):
        raise ValueError("Collection thresholds must be finite and nonnegative.")
    if max_workers < 1:
        raise ValueError("max_workers must be positive.")
    results = MetadataScanner(compute_footprints=False).scan_directory(
        directory,
        recursive=recursive,
        max_workers=max_workers,
    )
    native_types = (NISARMetadata, Sentinel1SLCMetadata, TerraSARMetadata, BIOMASSMetadata)
    native = [r for r in results if not r.error and isinstance(r.metadata_ref, native_types)]
    output = group_sicd_collections(
        [r for r in results if r.error or not isinstance(r.metadata_ref, native_types)],
        **thresholds,
    )
    for result in sorted(native, key=lambda r: str(r.filepath)):
        try:
            native_report = _native_collections(result)
            output.collections.extend(native_report.collections)
            output.problems.extend(native_report.problems)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.warning("Cannot enumerate polarizations in %s: %s", result.filepath, exc)
            output.problems.append(
                DiscoveryProblem(
                    result.filepath.resolve(),
                    "product_open_failed",
                    str(exc),
                )
            )
    output.collections.sort(
        key=lambda c: (
            tuple(str(p) for p in c.filepaths),
            str(c.reader_kwargs),
        )
    )
    return output
