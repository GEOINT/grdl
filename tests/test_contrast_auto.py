# -*- coding: utf-8 -*-
"""Tests for grdl.contrast.auto_select metadata-driven dispatch."""

import pytest

from grdl.contrast import auto_select


class _Stub:
    """Minimal stand-in so we can probe class-name-based dispatch
    without instantiating the real (heavily-nested) metadata classes."""


def _stub_named(name: str):
    cls = type(name, (_Stub,), {})
    return cls()


class TestAutoSelect:
    def test_none_returns_universal_default(self):
        assert auto_select(None) == 'percentile'

    @pytest.mark.parametrize('name', [
        'SICDMetadata', 'SIDDMetadata', 'CPHDMetadata', 'CRSDMetadata',
        'BIOMASSMetadata', 'Sentinel1SLCMetadata', 'TerraSARMetadata',
        'NISARMetadata',
    ])
    def test_sar_metadata_picks_brighter(self, name):
        assert auto_select(_stub_named(name)) == 'brighter'

    @pytest.mark.parametrize('name', [
        'Sentinel2Metadata', 'VIIRSMetadata', 'ASTERMetadata',
    ])
    def test_msi_ir_metadata_picks_percentile(self, name):
        assert auto_select(_stub_named(name)) == 'percentile'

    def test_eo_nitf_picks_gamma(self):
        assert auto_select(_stub_named('EONITFMetadata')) == 'gamma'

    def test_unknown_metadata_falls_back_to_percentile(self):
        assert auto_select(_stub_named('UnknownMetadata')) == 'percentile'
        assert auto_select(_stub_named('SomeOtherType')) == 'percentile'

    def test_returned_name_is_dispatchable(self):
        # Every name we return should be a known operator the example's
        # dispatcher (and downstream tooling) can handle.  Sanity-check
        # against the canonical set of grdl.contrast operator slugs.
        valid = {
            'percentile', 'linear', 'log', 'mangis', 'brighter', 'darker',
            'high_contrast', 'nrl', 'gamma', 'histogram', 'clahe',
        }
        for name in (
            None,
            _stub_named('SICDMetadata'),
            _stub_named('EONITFMetadata'),
            _stub_named('Sentinel2Metadata'),
            _stub_named('UnrecognizedMetadata'),
        ):
            assert auto_select(name) in valid
