# -*- coding: utf-8 -*-
"""
SAR Backend Detection Tests - Unit tests for _backend.py.

Tests for sarkit/sarpy availability detection and the require_*
helper functions.

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-09

Modified
--------
2026-02-09
"""

import pytest
from unittest import mock


def test_flags_are_bools():
    """Detection flags are boolean."""
    from grdl.IO.sar._backend import _HAS_SARKIT, _HAS_SARPY
    assert isinstance(_HAS_SARKIT, bool)
    assert isinstance(_HAS_SARPY, bool)


def test_require_sar_backend_returns_string():
    """require_sar_backend returns 'sarkit' or 'sarpy'."""
    from grdl.IO.sar._backend import require_sar_backend
    result = require_sar_backend('TEST')
    assert result in ('sarkit', 'sarpy')


def test_require_sar_backend_prefers_sarkit():
    """require_sar_backend prefers sarkit when both are available."""
    from grdl.IO.sar import _backend
    with mock.patch.object(_backend, '_HAS_SARKIT', True):
        with mock.patch.object(_backend, '_HAS_SARPY', True):
            assert _backend.require_sar_backend('TEST') == 'sarkit'


def test_require_sar_backend_falls_back_to_sarpy():
    """require_sar_backend falls back to sarpy when sarkit is absent."""
    from grdl.IO.sar import _backend
    with mock.patch.object(_backend, '_HAS_SARKIT', False):
        with mock.patch.object(_backend, '_HAS_SARPY', True):
            assert _backend.require_sar_backend('TEST') == 'sarpy'


def test_require_sar_backend_raises_when_none():
    """require_sar_backend raises ImportError when neither available."""
    from grdl.IO.sar import _backend
    with mock.patch.object(_backend, '_HAS_SARKIT', False):
        with mock.patch.object(_backend, '_HAS_SARPY', False):
            with pytest.raises(ImportError, match='sarkit'):
                _backend.require_sar_backend('SICD')


def test_require_sarkit_passes():
    """require_sarkit passes when sarkit is available."""
    from grdl.IO.sar import _backend
    with mock.patch.object(_backend, '_HAS_SARKIT', True):
        _backend.require_sarkit('CRSD')  # should not raise


def test_require_sarkit_raises_when_missing():
    """require_sarkit raises ImportError when sarkit is absent."""
    from grdl.IO.sar import _backend
    with mock.patch.object(_backend, '_HAS_SARKIT', False):
        with pytest.raises(ImportError, match='sarkit'):
            _backend.require_sarkit('CRSD')


def test_error_message_includes_format():
    """Error messages include the format name."""
    from grdl.IO.sar import _backend
    with mock.patch.object(_backend, '_HAS_SARKIT', False):
        with mock.patch.object(_backend, '_HAS_SARPY', False):
            with pytest.raises(ImportError, match='SICD'):
                _backend.require_sar_backend('SICD')

    with mock.patch.object(_backend, '_HAS_SARKIT', False):
        with pytest.raises(ImportError, match='CRSD'):
            _backend.require_sarkit('CRSD')
