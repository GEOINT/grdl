# -*- coding: utf-8 -*-
"""
Catalog Builder - Dynamic decorator introspection & catalog.yaml generation.

Discovers concrete processor classes inside any GRDL-compatible package
(``grdl``, ``grdl_sar``, ``grdl_eo``, or third-party extensions) by
introspecting ``@processor_version``, ``@processor_tags``, and
``__param_specs__``, and exports a ``catalog.yaml`` artifact compatible with
``grdl_rt.catalog.YamlArtifactCatalog``.

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import Any

from grdl.image_processing.base import ImageProcessor

logger = logging.getLogger(__name__)

_TYPE_MAP: dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}


def extract_param_schema(processor_class: type) -> dict[str, Any]:
    """Produce a JSON Schema (draft 2020-12) from ``__param_specs__``."""
    param_specs: tuple | None = getattr(processor_class, "__param_specs__", None)

    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    if not param_specs:
        return schema

    required: list[str] = []
    for spec in param_specs:
        prop: dict[str, Any] = {}
        json_type = _TYPE_MAP.get(spec.param_type)
        if json_type:
            prop["type"] = json_type
        if spec.description:
            prop["description"] = spec.description
        if spec.min_value is not None:
            prop["minimum"] = spec.min_value
        if spec.max_value is not None:
            prop["maximum"] = spec.max_value
        if spec.choices is not None:
            prop["enum"] = list(spec.choices)
        if getattr(spec, "_has_default", False) and not isinstance(spec.default, property):
            prop["default"] = spec.default

        schema["properties"][spec.name] = prop
        if spec.required:
            required.append(spec.name)

    if required:
        schema["required"] = required

    return schema


def _serialize_tags(raw_tags: dict[str, Any]) -> dict[str, list[str]]:
    """Convert ``__processor_tags__`` enum/tuple values to ``Dict[str, List[str]]``."""
    serialized: dict[str, list[str]] = {}
    for key, val in raw_tags.items():
        if val is None or key == "description":
            continue
        if isinstance(val, (tuple, list, set)):
            items = [item.value if hasattr(item, "value") else str(item) for item in val]
            if items:
                serialized[key] = items
        else:
            serialized[key] = [val.value if hasattr(val, "value") else str(val)]
    return serialized


def _infer_processor_type(cls: type) -> str:
    """Infer catalog ``processor_type`` from class hierarchy and tags."""
    mro_names = {base.__name__ for base in cls.__mro__}
    if "ImageDetector" in mro_names or "CFARDetector" in mro_names:
        return "detector"
    if "PolarimetricDecomposition" in mro_names:
        return "decomposition"
    if "ImageFormationAlgorithm" in mro_names:
        return "image_formation"
    tags = getattr(cls, "__processor_tags__", {}) or {}
    cat = tags.get("category")
    cat_val = cat.value if hasattr(cat, "value") else cat
    if cat_val == "decomposition":
        return "decomposition"
    if cat_val == "image_formation":
        return "image_formation"
    return "transform"


def _first_doc_paragraph(cls: type) -> str:
    """Return the first non-empty summary line from a class docstring."""
    doc = inspect.getdoc(cls) or ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _resolve_package_version(package_name: str, dist_name: str | None = None) -> str:
    """Resolve installed distribution version, falling back to ``0.1.0``."""
    candidates = [c for c in (dist_name, package_name.replace("_", "-"), package_name) if c]
    for name in candidates:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "0.1.0"


_BASE_ABCS = frozenset({
    "ImageProcessor",
    "ImageTransform",
    "BandwiseTransformMixin",
    "ImageDetector",
    "PolarimetricDecomposition",
    "ImageFormationAlgorithm",
})


def discover_package_processors(
    package_name: str,
    *,
    ignore_errors: bool = True,
) -> list[type]:
    """Walk *package_name* and return concrete tagged processor classes.

    Only classes whose ``__module__`` belongs to *package_name* are included,
    ensuring compatibility re-exports from other packages are never duplicated.
    """
    pkg = importlib.import_module(package_name)
    # Guard against CWD PEP 420 namespace shadowing (e.g., running from C:\dev)
    if getattr(pkg, "__file__", None) is None and hasattr(pkg, "__path__"):
        cwd = os.path.abspath(os.getcwd())
        saved_sys_path = list(sys.path)
        try:
            sys.path = [p for p in sys.path if p and os.path.abspath(p) != cwd]
            sys.modules.pop(package_name, None)
            pkg = importlib.import_module(package_name)
        finally:
            sys.path = saved_sys_path

    discovered: dict[str, type] = {}

    module_names = [package_name]
    if hasattr(pkg, "__path__"):
        for mod_info in pkgutil.walk_packages(pkg.__path__, prefix=f"{package_name}."):
            # Skip test modules or editable repo root namespace mirror (<pkg>.<pkg>.*)
            if (
                ".tests" in mod_info.name
                or mod_info.name.endswith(".__main__")
                or mod_info.name == f"{package_name}.{package_name}"
                or mod_info.name.startswith(f"{package_name}.{package_name}.")
            ):
                continue
            module_names.append(mod_info.name)

    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            if ignore_errors:
                logger.debug("Skipping module %s during catalog scan: %s", mod_name, exc)
                continue
            raise

        for _attr_name, obj in vars(mod).items():
            if not inspect.isclass(obj):
                continue
            cls_mod = getattr(obj, "__module__", "")
            if cls_mod != package_name and not cls_mod.startswith(f"{package_name}."):
                continue
            if cls_mod.startswith(f"{package_name}.{package_name}."):
                continue
            if inspect.isabstract(obj) or obj.__name__ in _BASE_ABCS:
                continue

            mro_names = {base.__name__ for base in obj.__mro__}
            is_proc = issubclass(obj, ImageProcessor) or "ImageFormationAlgorithm" in mro_names
            if not is_proc:
                continue

            fqn = f"{cls_mod}.{obj.__name__}"
            discovered[fqn] = obj

    return [discovered[k] for k in sorted(discovered)]


def build_catalog_data(
    package_name: str,
    *,
    dist_name: str | None = None,
    author: str = "geoint.org",
    license_name: str = "MIT",
    ignore_errors: bool = True,
) -> dict[str, Any]:
    """Build a ``YamlArtifactCatalog``-compatible dict for *package_name*."""
    pkg_dist = dist_name or package_name.replace("_", "-")
    pkg_version = _resolve_package_version(package_name, pkg_dist)
    classes = discover_package_processors(package_name, ignore_errors=ignore_errors)

    artifacts: list[dict[str, Any]] = []
    for idx, cls in enumerate(classes, start=1):
        fqn = f"{cls.__module__}.{cls.__name__}"
        raw_tags = getattr(cls, "__processor_tags__", {}) or {}
        proc_version = getattr(cls, "__processor_version__", pkg_version)
        description = raw_tags.get("description") or _first_doc_paragraph(cls)

        artifacts.append(
            {
                "id": idx,
                "name": cls.__name__,
                "version": str(proc_version),
                "artifact_type": "grdl_processor",
                "description": description,
                "author": author,
                "license": license_name,
                "pypi_package": pkg_dist,
                "conda_package": pkg_dist,
                "conda_channel": "conda-forge",
                "processor_class": fqn,
                "processor_version": str(proc_version),
                "processor_type": _infer_processor_type(cls),
                "yaml_definition": None,
                "python_dsl": None,
                "tags": _serialize_tags(raw_tags),
                "requires_global_pass": bool(getattr(cls, "__has_global_pass__", False)),
                "alternatives": [],
                "param_schema": extract_param_schema(cls),
            }
        )

    return {
        "catalog_meta": {
            "format_version": 1,
            "package": pkg_dist,
            "package_version": pkg_version,
            "next_id": len(artifacts) + 1,
        },
        "artifacts": artifacts,
        "remote_versions": [],
    }


def export_catalog_yaml(
    package_name: str,
    output_path: str | Path,
    *,
    dist_name: str | None = None,
    author: str = "geoint.org",
    license_name: str = "MIT",
    ignore_errors: bool = True,
) -> Path:
    """Introspect *package_name* and write ``catalog.yaml`` to *output_path*."""
    data = build_catalog_data(
        package_name,
        dist_name=dist_name,
        author=author,
        license_name=license_name,
        ignore_errors=ignore_errors,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yaml

        text = yaml.safe_dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    except ImportError:
        text = json.dumps(data, indent=2)

    out.write_text(text, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m grdl.discovery.catalog_builder``."""
    parser = argparse.ArgumentParser(
        description="Generate catalog.yaml by introspecting GRDL processor decorators."
    )
    parser.add_argument("package", help="Python package name to introspect (e.g. grdl_sar)")
    parser.add_argument("-o", "--output", default="catalog.yaml", help="Output YAML path")
    parser.add_argument("--dist-name", default=None, help="Distribution package name")
    parser.add_argument("--strict", action="store_true", help="Fail on module import errors")
    args = parser.parse_args(argv)

    out = export_catalog_yaml(
        args.package,
        args.output,
        dist_name=args.dist_name,
        ignore_errors=not args.strict,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
