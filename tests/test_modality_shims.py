# -*- coding: utf-8 -*-
"""Tests for deprecated compatibility shims forwarding to grdl-sar and grdl-eo."""

import importlib
import sys
import warnings

import pytest


def _clear_shim(name: str) -> None:
    sys.modules.pop(name, None)


def test_sar_submodule_shim_warns_and_forwards_private_symbols():
    pytest.importorskip("grdl_sar")
    import grdl.image_processing  # ensures finder is installed

    _clear_shim("grdl.image_processing.filters.standard_lee")
    _clear_shim("grdl.image_processing.decomposition.pauli")

    with pytest.warns(DeprecationWarning, match="grdl-sar"):
        mod = importlib.import_module("grdl.image_processing.filters.standard_lee")
    from grdl_sar.filters.standard_lee import LeeFilter

    assert mod.LeeFilter is LeeFilter

    with pytest.warns(DeprecationWarning, match="grdl-sar"):
        pauli_mod = importlib.import_module("grdl.image_processing.decomposition.pauli")
    from grdl_sar.decomposition.pauli import PauliDecomposition

    assert pauli_mod.PauliDecomposition is PauliDecomposition


def test_eo_submodule_shim_warns_and_forwards_all_symbols():
    pytest.importorskip("grdl_eo")
    import grdl.image_processing

    _clear_shim("grdl.image_processing.fusion")
    _clear_shim("grdl.image_processing.fusion.pansharpen")
    _clear_shim("grdl.image_processing.spectral.indices")
    _clear_shim("grdl.image_processing.filters.edge")

    with pytest.warns(DeprecationWarning, match="grdl-eo"):
        fusion_mod = importlib.import_module("grdl.image_processing.fusion.pansharpen")
    from grdl_eo.fusion.pansharpen import BroveyPanSharpenTransform

    assert fusion_mod.BroveyPanSharpenTransform is BroveyPanSharpenTransform

    with pytest.warns(DeprecationWarning, match="grdl-eo"):
        spec_mod = importlib.import_module("grdl.image_processing.spectral.indices")
    from grdl_eo.spectral.indices import compute_spectral_index

    assert spec_mod.compute_spectral_index is compute_spectral_index


def test_package_level_getattr_warns():
    pytest.importorskip("grdl_sar")
    import grdl.image_processing as ip

    with pytest.warns(DeprecationWarning, match="grdl-sar"):
        cls = ip.LeeFilter
    from grdl_sar import LeeFilter

    assert cls is LeeFilter

