"""Metadata-only polarization discovery, ambiguity and manifest tests."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from grdl.discovery import (
    CollectionDiscoveryReport,
    MetadataScanner,
    ScanResult,
    discover_polarization_collections,
    group_sicd_collections,
)
from grdl.example.discover_polarization_collections import main
from grdl.IO.models.base import ChannelMetadata, ImageMetadata
from grdl.IO.models.biomass import BIOMASSMetadata
from grdl.IO.models.common import XYZ, LatLonHAE
from grdl.IO.models.nisar import NISARMetadata
from grdl.IO.models.sentinel1_slc import Sentinel1SLCMetadata
from grdl.IO.models.sicd import (
    SICDSCP,
    SICDSCPCOA,
    SICDCollectionInfo,
    SICDDirParam,
    SICDGeoData,
    SICDGrid,
    SICDImageFormation,
    SICDMetadata,
    SICDRadarCollection,
    SICDRcvChannel,
    SICDTimeline,
)
from grdl.IO.models.terrasar import TerraSARMetadata
from grdl.IO.sar import validate_sicd_collection


def _scan(tmp_path: Path, pol: str, name: str | None = None, **kwargs) -> ScanResult:
    tx = {"R": "RHC", "L": "LHC"}.get(pol[0], pol[0])
    meta = SICDMetadata(
        format="SICD",
        rows=100,
        cols=200,
        dtype="complex64",
        collection_info=SICDCollectionInfo(collector_name="SAT", core_name=pol),
        timeline=SICDTimeline(collect_start="2026-01-01T00:00:00Z"),
        geo_data=SICDGeoData(scp=SICDSCP(ecf=XYZ(1e6, 2e6, 3e6))),
        grid=SICDGrid(row=SICDDirParam(ss=1.0), col=SICDDirParam(ss=1.0)),
        scpcoa=SICDSCPCOA(side_of_track="R", graze_ang=30.0, doppler_cone_ang=90.0),
        radar_collection=SICDRadarCollection(
            rcv_channels=[SICDRcvChannel(tx_rcv_polarization=f"{tx}:{pol[1]}")],
        ),
    )
    meta = replace(meta, **kwargs)
    return ScanResult(filepath=tmp_path / (name or f"{pol}.nitf"), format="SICD", metadata_ref=meta)


def test_quad_order_is_stable_and_manifest_is_json_safe(tmp_path):
    results = [_scan(tmp_path, p) for p in ("VV", "HV", "HH", "VH")]
    report = group_sicd_collections(results)
    assert not report.has_errors
    (collection,) = report.collections
    assert collection.status == "ready"
    assert collection.validation.mode == "quad"
    assert collection.polarizations == ["HH", "HV", "VH", "VV"]
    assert [p.name for p in collection.filepaths] == ["HH.nitf", "HV.nitf", "VH.nitf", "VV.nitf"]
    manifest = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    (entry,) = manifest["collections"]
    assert manifest["schema_version"] == 1
    assert entry["source"] == [str(p) for p in collection.filepaths]
    assert entry["reader_kwargs"] == {"polarizations": collection.polarizations}
    assert entry["validation"]["issues"][0]["code"] == "core_name_differs"
    assert group_sicd_collections(reversed(results)).to_dict() == report.to_dict()


@pytest.mark.parametrize(
    ("pols", "status", "mode"),
    [
        (["HH"], "single", "single"),
        (["HH", "HV"], "ready", "dual"),
        (["HH", "HV", "VV"], "partial", "triple"),
        (["RH", "RV"], "ready", "compact"),
    ],
)
def test_modes(tmp_path, pols, status, mode):
    (collection,) = group_sicd_collections([_scan(tmp_path, p) for p in pols]).collections
    assert (collection.status, collection.validation.mode) == (status, mode)


def test_processed_polarization_overrides_first_receive_channel(tmp_path):
    first = _scan(tmp_path, "HH")
    second = _scan(
        tmp_path,
        "HH",
        "misleading_hh.nitf",
        image_formation=SICDImageFormation(tx_rcv_polarization_proc="V:V"),
    )
    (collection,) = group_sicd_collections([first, second]).collections
    assert collection.status == "ready"
    assert collection.polarizations == ["HH", "VV"]


def test_duplicate_polarizations_are_not_arbitrarily_selected(tmp_path):
    results = [
        _scan(tmp_path, "HH", "a.nitf"),
        _scan(tmp_path, "HH", "b.nitf"),
        _scan(tmp_path, "VV"),
    ]
    report = group_sicd_collections(results)
    (collection,) = report.collections
    assert collection.status == "ambiguous"
    assert len(collection.filepaths) == 3
    assert report.has_errors
    assert {"duplicate_polarization", "ambiguous_membership"} <= {
        i.code for i in collection.validation.errors
    }


def test_nontransitive_time_matches_are_ambiguous(tmp_path):
    results = [
        _scan(tmp_path, p, timeline=SICDTimeline(collect_start=f"2026-01-01T00:00:0{s}Z"))
        for p, s in (("HH", "0.0"), ("HV", "0.8"), ("VV", "1.6"))
    ]
    (collection,) = group_sicd_collections(results).collections
    assert collection.status == "ambiguous"
    assert len(collection.filepaths) == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"rows": 101},
        {"collection_info": SICDCollectionInfo(collector_name="OTHER")},
        {"timeline": SICDTimeline(collect_start="2026-01-02T00:00:00Z")},
        {"geo_data": SICDGeoData(scp=SICDSCP(ecf=XYZ(1e6 + 26, 2e6, 3e6)))},
        {"grid": SICDGrid(row=SICDDirParam(ss=1.002), col=SICDDirParam(ss=1.0))},
        {"scpcoa": SICDSCPCOA(side_of_track="L")},
        {"scpcoa": SICDSCPCOA(side_of_track="R", graze_ang=30.6)},
    ],
)
def test_different_collects_are_not_merged(tmp_path, changes):
    report = group_sicd_collections([_scan(tmp_path, "HH"), _scan(tmp_path, "VV", **changes)])
    assert [c.status for c in report.collections] == ["single", "single"]
    assert not report.has_errors


@pytest.mark.parametrize(
    ("keyword", "changes", "limit"),
    [
        ("max_time_delta_s", {"timeline": SICDTimeline(collect_start="2026-01-01T00:00:02Z")}, 2),
        (
            "max_scp_distance_m",
            {"geo_data": SICDGeoData(scp=SICDSCP(ecf=XYZ(1e6 + 30, 2e6, 3e6)))},
            30,
        ),
        (
            "max_spacing_rel_diff",
            {"grid": SICDGrid(row=SICDDirParam(ss=1.01), col=SICDDirParam(ss=1.0))},
            0.01,
        ),
        ("max_angle_diff_deg", {"scpcoa": SICDSCPCOA(side_of_track="R", graze_ang=31)}, 1),
    ],
)
def test_thresholds_are_forwarded(tmp_path, keyword, changes, limit):
    results = [_scan(tmp_path, "HH"), _scan(tmp_path, "VV", **changes)]
    assert len(group_sicd_collections(results).collections) == 2
    (collection,) = group_sicd_collections(results, **{keyword: limit}).collections
    assert collection.status == "ready"


@pytest.mark.parametrize(
    "changes",
    [
        {"collection_info": None},
        {"timeline": None},
        {"geo_data": None},
        {"grid": None},
        {"timeline": SICDTimeline(collect_start="not-a-time")},
        {"geo_data": SICDGeoData(scp=SICDSCP(ecf=XYZ(float("nan"), 2, 3)))},
        {"grid": SICDGrid(row=SICDDirParam(ss=0), col=SICDDirParam(ss=1))},
        {"scpcoa": SICDSCPCOA(side_of_track="R", graze_ang=float("inf"))},
    ],
)
def test_missing_or_nonfinite_identity_is_rejected(tmp_path, changes):
    report = group_sicd_collections([_scan(tmp_path, "HH", **changes)])
    (collection,) = report.collections
    assert collection.status == "rejected"
    assert "insufficient_identity" in {i.code for i in collection.validation.errors}
    assert report.has_errors


def test_unknown_polarization_is_rejected(tmp_path):
    (collection,) = group_sicd_collections(
        [
            _scan(tmp_path, "HH", radar_collection=None),
        ]
    ).collections
    assert collection.status == "rejected"
    assert "unresolved_polarization" in {i.code for i in collection.validation.errors}


def test_scene_centers_must_be_comparable_not_merely_present(tmp_path):
    results = [
        _scan(tmp_path, "HH"),
        _scan(tmp_path, "VV", geo_data=SICDGeoData(scp=SICDSCP(llh=LatLonHAE(10, 20, 0)))),
    ]
    assert [c.status for c in group_sicd_collections(results).collections] == ["single", "single"]


def test_latlon_scene_centers_can_be_matched(tmp_path):
    geo = SICDGeoData(scp=SICDSCP(llh=LatLonHAE(10, 20, 0)))
    (collection,) = group_sicd_collections(
        [
            _scan(tmp_path, "HH", geo_data=geo),
            _scan(tmp_path, "VV", geo_data=geo),
        ]
    ).collections
    assert collection.status == "ready"


def test_scanner_failures_unsupported_and_persisted_records_remain_visible(tmp_path):
    results = [
        ScanResult(filepath=tmp_path / "bad.nitf", error="Cannot open file"),
        ScanResult(filepath=tmp_path / "saved.nitf", format="SICD"),
        ScanResult(
            filepath=tmp_path / "eo.tif",
            format="GeoTIFF",
            metadata_ref=ImageMetadata(format="GeoTIFF", rows=10, cols=10, dtype="float32"),
        ),
    ]
    report = group_sicd_collections(results)
    assert not report.collections
    assert {p.code for p in report.problems} == {
        "scan_failed",
        "missing_metadata",
        "unsupported_format",
    }
    assert report.has_errors
    assert len(report.to_dict()["problems"]) == 3


def test_duplicate_paths_raise(tmp_path):
    result = _scan(tmp_path, "HH")
    with pytest.raises(ValueError, match="Duplicate scan result"):
        group_sicd_collections([result, result])


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_thresholds_raise_before_scanning(tmp_path, value):
    with patch.object(MetadataScanner, "scan_directory") as scan:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            discover_polarization_collections(tmp_path, max_time_delta_s=value)
        scan.assert_not_called()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        group_sicd_collections([], max_scp_distance_m=value)


def test_utc_offsets_and_naive_sicd_times_compare_correctly(tmp_path):
    results = [
        _scan(tmp_path, "HH", timeline=SICDTimeline(collect_start="2026-01-01T00:00:00Z")),
        _scan(tmp_path, "HV", timeline=SICDTimeline(collect_start="2026-01-01T01:00:00+01:00")),
        _scan(tmp_path, "VV", timeline=SICDTimeline(collect_start="2026-01-01T00:00:00")),
    ]
    report = validate_sicd_collection([r.metadata_ref for r in results])
    assert report.ok
    (collection,) = group_sicd_collections(results).collections
    assert collection.status == "partial"


@contextmanager
def _native_reader(pols):
    yield SimpleNamespace(
        metadata=ImageMetadata(
            format="SAR",
            rows=10,
            cols=10,
            dtype="complex64",
            bands=len(pols),
            axis_order="CYX",
            channel_metadata=[
                ChannelMetadata(index=i, polarization=p, name=p) for i, p in enumerate(pols)
            ],
        )
    )


def test_native_nisar_frequencies_are_independent_and_failures_retained(tmp_path):
    result = ScanResult(
        filepath=tmp_path / "NISAR.h5",
        format="NISAR",
        metadata_ref=NISARMetadata(
            format="NISAR",
            rows=10,
            cols=10,
            dtype="complex64",
            available_frequencies=["C", "B", "A"],
        ),
    )
    calls = []

    def factory(key, path, **kwargs):
        calls.append((key, path, kwargs))
        if kwargs["frequency"] == "B":
            raise OSError("broken frequency")
        return _native_reader(
            ["HH", "HV", "VH", "VV"] if kwargs["frequency"] == "A" else ["VV", "VH"]
        )

    with (
        patch.object(MetadataScanner, "scan_directory", return_value=[result]) as scan,
        patch("grdl.discovery.multi_channel_collections.get_reader", side_effect=factory),
    ):
        report = discover_polarization_collections(tmp_path, recursive=False, max_workers=1)
    scan.assert_called_once_with(tmp_path, recursive=False, max_workers=1)
    assert [c.reader_kwargs["frequency"] for c in report.collections] == ["A", "C"]
    assert [c.validation.mode for c in report.collections] == ["quad", "dual"]
    assert all(c.status == "ready" for c in report.collections)
    assert len(calls) == 3
    assert all(kwargs["polarizations"] == "all" for _, _, kwargs in calls)
    assert report.has_errors
    (problem,) = report.problems
    assert problem.code == "product_open_failed" and "broken frequency" in problem.message
    assert isinstance(report.to_dict()["collections"][0]["source"], str)


@pytest.mark.parametrize(
    ("metadata_cls", "key", "kwargs"),
    [
        (BIOMASSMetadata, "biomass", {}),
        (TerraSARMetadata, "terrasar", {"polarizations": "all"}),
    ],
)
def test_native_product_channels(tmp_path, metadata_cls, key, kwargs):
    result = ScanResult(
        filepath=tmp_path / "product",
        metadata_ref=metadata_cls(format="SAR", rows=10, cols=10, dtype="complex64"),
    )
    with (
        patch.object(MetadataScanner, "scan_directory", return_value=[result]),
        patch(
            "grdl.discovery.multi_channel_collections.get_reader",
            side_effect=lambda *a, **k: _native_reader(["HH", "HV", "VH", "VV"]),
        ) as reader,
    ):
        report = discover_polarization_collections(tmp_path)
    reader.assert_called_once_with(key, result.filepath, **kwargs)
    (collection,) = report.collections
    assert collection.status == "ready" and not report.has_errors


def test_sentinel1_swaths_are_separate(tmp_path):
    result = ScanResult(
        filepath=tmp_path / "product.SAFE",
        metadata_ref=Sentinel1SLCMetadata(
            format="Sentinel-1_IW_SLC", rows=10, cols=10, dtype="complex64"
        ),
    )
    with (
        patch.object(MetadataScanner, "scan_directory", return_value=[result]),
        patch("grdl.IO.sar.sentinel1_slc.Sentinel1SLCReader") as probe,
        patch(
            "grdl.discovery.multi_channel_collections.get_reader",
            side_effect=lambda *a, **k: _native_reader(["VV", "VH"]),
        ) as reader,
    ):
        probe.return_value.__enter__.return_value.get_available_swaths.return_value = ["IW2", "IW1"]
        report = discover_polarization_collections(tmp_path)
    assert [c.reader_kwargs["swath"] for c in report.collections] == ["IW1", "IW2"]
    assert reader.call_count == 2
    assert all(c.validation.mode == "dual" for c in report.collections)
    assert not report.has_errors


def test_cli_json_output_and_no_overwrite(tmp_path, capsys):
    report = group_sicd_collections([_scan(tmp_path, "HH"), _scan(tmp_path, "VV")])
    target = tmp_path / "collections.json"
    with patch(
        "grdl.example.discover_polarization_collections.discover_polarization_collections",
        return_value=report,
    ):
        assert main([str(tmp_path), "--format", "json"]) == 0
        assert json.loads(capsys.readouterr().out) == report.to_dict()
        assert main([str(tmp_path), "--format", "json", "--output", str(target)]) == 0
        assert json.loads(target.read_text()) == report.to_dict()
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path), "--format", "json", "--output", str(target)])
        assert exc.value.code == 2
        assert json.loads(target.read_text()) == report.to_dict()


def test_cli_reports_errors_and_returns_nonzero(tmp_path, capsys):
    report = group_sicd_collections([ScanResult(filepath=tmp_path / "bad", error="unreadable")])
    with patch(
        "grdl.example.discover_polarization_collections.discover_polarization_collections",
        return_value=report,
    ):
        assert main([str(tmp_path)]) == 1
    assert "[scan_failed]" in capsys.readouterr().out


def test_empty_directory_and_invalid_workers(tmp_path):
    assert discover_polarization_collections(tmp_path).to_dict() == {
        "schema_version": 1,
        "collections": [],
        "problems": [],
    }
    assert not CollectionDiscoveryReport().has_errors
    with pytest.raises(ValueError, match="positive"):
        discover_polarization_collections(tmp_path, max_workers=0)


def test_directory_scanning_preserves_case_insensitive_extensions(tmp_path):
    for name in ("one.NITF", "two.ntf", "ignore.txt"):
        (tmp_path / name).touch()
    results = {
        name: _scan(tmp_path, pol, name) for name, pol in (("one.NITF", "HH"), ("two.ntf", "VV"))
    }
    with patch.object(MetadataScanner, "scan_file", side_effect=lambda path: results[path.name]):
        report = discover_polarization_collections(tmp_path, max_workers=1)
    (collection,) = report.collections
    assert collection.status == "ready"
    assert [p.name for p in collection.filepaths] == ["one.NITF", "two.ntf"]
