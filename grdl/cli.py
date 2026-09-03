"""Command-line helpers for GRDL."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys


def get_version() -> str:
    """Return the installed GRDL version."""
    try:
        from grdl import __version__
    except (ImportError, AttributeError):
        try:
            return importlib.metadata.version("grdl")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"
    return __version__


def main(argv: list[str] | None = None) -> int:
    """Print the installed GRDL version."""
    parser = argparse.ArgumentParser(prog="grdl-version", add_help=True)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the version string.",
    )
    args = parser.parse_args(argv)
    version = get_version()
    if args.quiet:
        print(version)
    else:
        print(f"grdl {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
