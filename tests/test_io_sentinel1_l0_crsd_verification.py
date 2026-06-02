# -*- coding: utf-8 -*-
"""
Tests for split-gate CRSD verification wrapper.

Verifies that CRSD consistency checks can run with schema check
excluded while explicit schema validation is reported separately.

Author
------
Jason Fritz
43161141+stryder-vtx@users.noreply.github.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-05-27

Modified
--------
2026-05-27
"""

# Standard library
from pathlib import Path

# Third-party
import pytest

# GRDL internal
import grdl.IO.sar.sentinel1_l0.crsd_verification as cv


class _FakeSchema:
    def __init__(self, ok: bool, error_log: str = "") -> None:
        self._ok = ok
        self.error_log = error_log

    def validate(self, _root) -> bool:
        return self._ok


class _FakeChecker:
    def __init__(self, *, schema: str, failures: dict) -> None:
        self.schema = schema
        self.crsdroot = type("_Root", (), {"tag": "{ns}CRSDsar"})()
        self._failures = failures
        self.ignore_patterns = None

    def check(self, ignore_patterns=None) -> None:
        self.ignore_patterns = tuple(ignore_patterns or ())

    def failures(self):
        return self._failures


def test_verify_crsd_split_gates_success(monkeypatch, tmp_path):
    crsd_path = tmp_path / "x.crsd"
    crsd_path.write_bytes(b"dummy")

    fake_checker = _FakeChecker(
        schema="/tmp/schema.xsd",
        failures={},
    )

    monkeypatch.setattr(
        cv.CrsdConsistency,
        "from_file",
        staticmethod(lambda _fh: fake_checker),
    )
    monkeypatch.setattr(
        cv.etree,
        "XMLSchema",
        lambda file: _FakeSchema(ok=True),
    )

    result = cv.verify_crsd_split_gates(crsd_path)

    assert result.path == crsd_path
    assert result.consistency_valid
    assert result.schema_valid
    assert result.passed
    assert "check_against_schema" in result.ignored_checks
    assert fake_checker.ignore_patterns is not None
    assert "check_against_schema" in fake_checker.ignore_patterns


def test_verify_crsd_split_gates_schema_failure(monkeypatch, tmp_path):
    crsd_path = tmp_path / "x.crsd"
    crsd_path.write_bytes(b"dummy")

    fake_checker = _FakeChecker(
        schema="/tmp/schema.xsd",
        failures={},
    )

    monkeypatch.setattr(
        cv.CrsdConsistency,
        "from_file",
        staticmethod(lambda _fh: fake_checker),
    )
    monkeypatch.setattr(
        cv.etree,
        "XMLSchema",
        lambda file: _FakeSchema(ok=False, error_log="line 1 bad"),
    )

    result = cv.verify_crsd_split_gates(crsd_path)

    assert result.consistency_valid
    assert not result.schema_valid
    assert not result.passed
    assert result.schema_error is not None
    assert "line 1 bad" in result.schema_error


def test_verify_crsd_split_gates_consistency_failure(monkeypatch, tmp_path):
    crsd_path = tmp_path / "x.crsd"
    crsd_path.write_bytes(b"dummy")

    fake_failures = {
        "check_inst_osr": {
            "details": [
                {
                    "severity": "Error",
                    "message": "Need: sufficient oversample",
                }
            ]
        }
    }
    fake_checker = _FakeChecker(
        schema="/tmp/schema.xsd",
        failures=fake_failures,
    )

    monkeypatch.setattr(
        cv.CrsdConsistency,
        "from_file",
        staticmethod(lambda _fh: fake_checker),
    )
    monkeypatch.setattr(
        cv.etree,
        "XMLSchema",
        lambda file: _FakeSchema(ok=True),
    )

    result = cv.verify_crsd_split_gates(crsd_path)

    assert not result.consistency_valid
    assert result.schema_valid
    assert not result.passed
    assert "check_inst_osr" in result.consistency_failures


def test_verify_crsd_split_gates_schema_unavailable(monkeypatch, tmp_path):
    crsd_path = tmp_path / "x.crsd"
    crsd_path.write_bytes(b"dummy")

    fake_checker = _FakeChecker(schema=None, failures={})

    monkeypatch.setattr(
        cv.CrsdConsistency,
        "from_file",
        staticmethod(lambda _fh: fake_checker),
    )

    result = cv.verify_crsd_split_gates(crsd_path)

    assert not result.schema_valid
    assert not result.passed
    assert result.schema_error is not None
    assert "Schema unavailable" in result.schema_error
