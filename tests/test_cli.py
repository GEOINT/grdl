"""Tests for the GRDL version CLI."""

from __future__ import annotations

from grdl.cli import get_version, main


def test_get_version_returns_string():
    version = get_version()
    assert isinstance(version, str)
    assert version


def test_main_quiet_prints_version(capsys):
    exit_code = main(["--quiet"])
    assert exit_code == 0
    out = capsys.readouterr().out.strip()
    assert out == get_version()
