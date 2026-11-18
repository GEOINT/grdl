# -*- coding: utf-8 -*-
"""Unit tests for grdl.discovery.catalog_builder."""

from grdl.discovery.catalog_builder import build_catalog_data, export_catalog_yaml


def test_grdl_catalog_excludes_modality_extensions():
    import grdl.synthetic.injectors  # ensure synthetic injectors are loaded

    data = build_catalog_data("grdl")
    artifacts = data["artifacts"]
    names = {a["name"] for a in artifacts}
    classes = {a["processor_class"] for a in artifacts}

    assert "MeanFilter" in names
    assert "GaussianFilter" in names
    assert "Orthorectifier" in names
    assert "Pipeline" in names

    # Non-processor @processor_tags classes (synthetic data injectors) must be excluded
    assert "GeometricEOInjector" not in names
    assert "GeometricSARInjector" not in names

    # Base ABCs must not be cataloged as concrete processors
    assert "ImageProcessor" not in names
    assert "ImageTransform" not in names

    # Compatibility shims must not leak grdl_sar / grdl_eo classes into grdl's catalog
    assert "LeeFilter" not in names
    assert "PauliDecomposition" not in names
    assert "BroveyPanSharpenTransform" not in names
    assert all(c.startswith("grdl.") for c in classes)


def test_export_catalog_yaml_roundtrip(tmp_path):
    out = export_catalog_yaml("grdl", tmp_path / "catalog.yaml")
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "MeanFilter" in content
    assert "grdl.image_processing.filters.linear.MeanFilter" in content

