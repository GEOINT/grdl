# -*- coding: utf-8 -*-
"""
CRSD verification wrapper with schema/consistency split gates.

Runs Sarkit CRSD consistency checks while excluding the schema check,
then performs explicit XML schema validation as a second, separate
step. This isolates current verifier/schema mismatches without hiding
real CRSD metadata and binary-structure failures.

Dependencies
------------
lxml
sarkit

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# Third-party
from lxml import etree
from sarkit.verification import CrsdConsistency

DEFAULT_IGNORED_CHECKS: Tuple[str, ...] = ("check_against_schema",)


@dataclass(frozen=True)
class CrsdVerificationResult:
    """Result for split-gate CRSD validation.

    Parameters
    ----------
    path : Path
        Checked CRSD file path.
    consistency_failures : Dict[str, Any]
        Failures returned by CrsdConsistency after excluded checks.
    schema_valid : bool
        True if XML validates against the resolved schema.
    schema_error : str, optional
        Schema validation error log or load error text.
    schema_path : Path, optional
        Schema used for explicit validation.
    ignored_checks : tuple of str
        Check names excluded from CrsdConsistency execution.
    """

    path: Path
    consistency_failures: Dict[str, Any]
    schema_valid: bool
    schema_error: Optional[str]
    schema_path: Optional[Path]
    ignored_checks: Tuple[str, ...]

    @property
    def consistency_valid(self) -> bool:
        """Return True when non-excluded consistency checks pass."""
        return len(self.consistency_failures) == 0

    @property
    def passed(self) -> bool:
        """Return True only when consistency and schema both pass."""
        return self.consistency_valid and self.schema_valid


def verify_crsd_split_gates(
    path: Union[str, Path],
    ignored_checks: Optional[Sequence[str]] = None,
) -> CrsdVerificationResult:
    """Validate CRSD with split consistency and schema gates.

    Parameters
    ----------
    path : str or Path
        CRSD file path.
    ignored_checks : sequence of str, optional
        Additional CrsdConsistency check names to ignore.

    Returns
    -------
    CrsdVerificationResult
        Structured verification result.
    """
    crsd_path = Path(path)
    ignore_set = set(DEFAULT_IGNORED_CHECKS)
    if ignored_checks is not None:
        ignore_set.update(ignored_checks)
    ignore_tuple = tuple(sorted(ignore_set))

    with open(crsd_path, "rb") as fh:
        checker = CrsdConsistency.from_file(fh)
        schema_path = (
            Path(str(checker.schema))
            if checker.schema is not None
            else None
        )

        checker.check(ignore_patterns=ignore_tuple)
        failures = checker.failures()

        schema_valid = False
        schema_error: Optional[str] = None

        if schema_path is None:
            schema_error = (
                "Schema unavailable for CRSD namespace "
                f"{checker.crsdroot.tag}."
            )
        else:
            try:
                schema = etree.XMLSchema(file=str(schema_path))
                schema_valid = schema.validate(checker.crsdroot)
                if not schema_valid:
                    schema_error = str(schema.error_log)
            except Exception as exc:
                schema_error = f"{type(exc).__name__}: {exc}"

    return CrsdVerificationResult(
        path=crsd_path,
        consistency_failures=failures,
        schema_valid=schema_valid,
        schema_error=schema_error,
        schema_path=schema_path,
        ignored_checks=ignore_tuple,
    )
