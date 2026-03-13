# -*- coding: utf-8 -*-
"""
Catalog Discovery Example - Scan directories for all supported sensor data.

Demonstrates how to dynamically discover every cataloger registered in the
GRDL IO catalog module, search a primary data path for compatible imagery,
and fall back to a secondary path when the primary yields no results.

This script is a self-contained reference for anyone wanting to understand
how the GRDL cataloging system works end-to-end.

Usage
-----
    python catalog_discovery.py [--primary /data/sar] [--fallback /code/grdl-te/data]

Dependencies
------------
(none beyond grdl core -- catalogs may require rasterio, h5py, etc.)

Author
------
Ava Courtney
courtney-ava@zai.com

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-03-12

Modified
--------
2026-03-12
"""

# Standard library
import argparse
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

# GRDL internal -- the catalog module re-exports every CatalogInterface
# subclass registered in the library.  Importing the module gives us access
# to all of them without hard-coding names.
import grdl.IO.catalog as catalog_module
from grdl.IO.base import CatalogInterface

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default search paths
# ---------------------------------------------------------------------------
DEFAULT_PRIMARY_PATH = Path("/data/sar")
DEFAULT_FALLBACK_PATH = Path("/code/grdl-te/data")


# ============================================================================
# Core helpers
# ============================================================================

def discover_catalogers() -> List[Tuple[str, Type[CatalogInterface]]]:
    """Dynamically discover all CatalogInterface subclasses in grdl.IO.catalog.

    The GRDL catalog module (`grdl.IO.catalog`) re-exports every concrete
    catalog class via its ``__all__`` list.  This function inspects the
    module at runtime and collects every exported class that inherits from
    ``CatalogInterface``, so newly-added catalogs are picked up
    automatically with zero code changes here.

    Returns
    -------
    List[Tuple[str, Type[CatalogInterface]]]
        Sorted list of ``(class_name, class_object)`` pairs.

    Examples
    --------
    >>> catalogers = discover_catalogers()
    >>> for name, cls in catalogers:
    ...     print(name)
    ASTERCatalog
    BIOMASSCatalog
    ...
    """
    catalogers: List[Tuple[str, Type[CatalogInterface]]] = []

    # Walk every public name exported by grdl.IO.catalog.  Using __all__
    # ensures we only consider names the module explicitly exports (not
    # helper functions or private symbols).
    for name in getattr(catalog_module, "__all__", dir(catalog_module)):
        obj = getattr(catalog_module, name, None)

        # Keep only concrete classes that inherit from CatalogInterface.
        if (
            obj is not None
            and inspect.isclass(obj)
            and issubclass(obj, CatalogInterface)
            and obj is not CatalogInterface  # skip the ABC itself
        ):
            catalogers.append((name, obj))

    # Sort alphabetically for deterministic output.
    catalogers.sort(key=lambda pair: pair[0])
    return catalogers


def search_with_catalog(
    catalog_cls: Type[CatalogInterface],
    search_path: Path,
) -> List[Path]:
    """Instantiate a catalog on *search_path* and return discovered images.

    Parameters
    ----------
    catalog_cls : Type[CatalogInterface]
        Concrete catalog class (e.g. ``BIOMASSCatalog``).
    search_path : Path
        Directory to search.

    Returns
    -------
    List[Path]
        Image paths (or SAFE directories) found by the catalog.

    Notes
    -----
    If the search path does not exist or is not a directory, an empty list
    is returned instead of raising, so the caller can proceed to the
    fallback path.
    """
    if not search_path.is_dir():
        logger.debug(
            "%s: directory does not exist: %s", catalog_cls.__name__, search_path
        )
        return []

    try:
        # CatalogInterface.__init__ validates the directory and each
        # concrete subclass sets up its SQLite database (if applicable).
        catalog = catalog_cls(search_path)
    except Exception as exc:
        logger.warning(
            "%s: failed to initialize on %s — %s", catalog_cls.__name__, search_path, exc
        )
        return []

    try:
        # discover_images() is the standard ABC method every catalog
        # implements.  It returns paths to files or SAFE directories that
        # match the sensor-specific naming patterns.
        results = catalog.discover_images()
    except Exception as exc:
        logger.warning(
            "%s: discover_images raised %s — %s",
            catalog_cls.__name__,
            type(exc).__name__,
            exc,
        )
        results = []
    finally:
        # Always clean up the catalog (closes SQLite connections, etc.).
        if hasattr(catalog, "close"):
            catalog.close()

    return results


def run_discovery(
    primary_path: Path,
    fallback_path: Path,
) -> List[Dict[str, Any]]:
    """Run every registered cataloger against the primary and fallback paths.

    For each cataloger:

    1. Search the **primary** path.
    2. If zero results are found *and* a fallback path was provided,
       search the **fallback** path instead.

    Parameters
    ----------
    primary_path : Path
        Primary data directory.
    fallback_path : Path
        Fallback data directory used when the primary yields no results.

    Returns
    -------
    List[Dict[str, Any]]
        One dict per cataloger with keys:
        ``"cataloger"``, ``"path_searched"``, ``"files"``.
    """
    catalogers = discover_catalogers()

    if not catalogers:
        logger.error("No CatalogInterface subclasses found in grdl.IO.catalog")
        return []

    logger.info(
        "Discovered %d cataloger(s): %s",
        len(catalogers),
        ", ".join(name for name, _ in catalogers),
    )

    summary: List[Dict[str, Any]] = []

    for name, cls in catalogers:
        logger.info("--- %s ---", name)

        # Step 1: try the primary path.
        results = search_with_catalog(cls, primary_path)
        path_used = primary_path

        # Step 2: fall back if primary returned nothing.
        if not results and fallback_path.is_dir():
            logger.info(
                "%s: no results in primary path, trying fallback: %s",
                name,
                fallback_path,
            )
            results = search_with_catalog(cls, fallback_path)
            path_used = fallback_path

        summary.append(
            {
                "cataloger": name,
                "path_searched": str(path_used),
                "files": [str(p) for p in results],
            }
        )

    return summary


# ============================================================================
# Pretty-printing
# ============================================================================

def print_summary(summary: List[Dict[str, Any]]) -> None:
    """Print a human-readable summary table to stdout.

    Parameters
    ----------
    summary : List[Dict[str, Any]]
        Output of :func:`run_discovery`.
    """
    separator = "=" * 72
    print(f"\n{separator}")
    print("  GRDL Catalog Discovery Summary")
    print(separator)

    for entry in summary:
        name = entry["cataloger"]
        path = entry["path_searched"]
        files = entry["files"]

        print(f"\n  Cataloger : {name}")
        print(f"  Path      : {path}")

        if files:
            print(f"  Found     : {len(files)} item(s)")
            for f in files:
                print(f"              - {f}")
        else:
            print("  Found     : No data found")

    print(f"\n{separator}\n")


# ============================================================================
# CLI entry-point
# ============================================================================

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : List[str] | None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``primary`` and ``fallback`` paths.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Discover all GRDL catalogers and search for compatible data "
            "in a primary directory, falling back to a secondary directory "
            "when no results are found."
        ),
    )
    parser.add_argument(
        "--primary",
        type=Path,
        default=DEFAULT_PRIMARY_PATH,
        help="Primary data directory (default: %(default)s)",
    )
    parser.add_argument(
        "--fallback",
        type=Path,
        default=DEFAULT_FALLBACK_PATH,
        help="Fallback data directory (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry-point: discover catalogers, search paths, and print results.

    Parameters
    ----------
    argv : List[str] | None
        CLI arguments (defaults to ``sys.argv[1:]``).
    """
    args = parse_args(argv)

    logger.info("Primary path  : %s", args.primary)
    logger.info("Fallback path : %s", args.fallback)

    summary = run_discovery(args.primary, args.fallback)
    print_summary(summary)


if __name__ == "__main__":
    main()
