"""Report directory polarization collections using the GRDL discovery API.

Run with ``python -m grdl.discovery.polarization_collections_cli --help``.
This script inventories metadata; it never runs image processing.

The thin reporting script uses the same API:

```bash
python -m grdl.discovery.polarization_collections_cli /data/imagery
python -m grdl.discovery.polarization_collections_cli /data/imagery \
    --format json --output collections.json
```

`--help` lists recursion, worker count and matching-threshold options. Reports
include unsuccessful inputs; output files are never overwritten. Exit status is
0 for a report without errors (including an empty scan, partial collections or
single channels), 1 for scan/open failures, rejected or ambiguous candidates,
and 2 for invalid arguments or report-writing errors. Logs go to stderr; JSON
on stdout is machine-readable.

"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from grdl.discovery import CollectionDiscoveryReport, discover_polarization_collections


def _text_report(report: CollectionDiscoveryReport) -> str:
    lines = [f"Found {len(report.collections)} candidate collection(s)."]
    for collection in report.collections:
        lines.append(
            f"{collection.status.upper()}: {collection.validation.mode} "
            f"[{', '.join(collection.polarizations)}] "
            f"{collection.reader_kwargs}"
        )
        lines.extend(f"  {path}" for path in collection.filepaths)
        lines.extend(
            f"  {issue.level}: [{issue.code}] {issue.message}"
            for issue in collection.validation.issues
        )
    lines.extend(
        f"{problem.level}: [{problem.code}] {problem.filepath}: {problem.message}"
        for problem in report.problems
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write a report; exit 1 for failures/ambiguity, 2 for invalid CLI input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing imagery/products.")
    parser.add_argument("--no-recursive", action="store_true", help="Scan direct children only.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-time-delta-s", type=float, default=1.0)
    parser.add_argument("--max-scp-distance-m", type=float, default=25.0)
    parser.add_argument("--max-spacing-rel-diff", type=float, default=1e-3)
    parser.add_argument("--max-angle-diff-deg", type=float, default=0.5)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write to a new file instead of stdout. Existing files are never overwritten.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        report = discover_polarization_collections(
            args.directory,
            recursive=not args.no_recursive,
            max_workers=args.max_workers,
            max_time_delta_s=args.max_time_delta_s,
            max_scp_distance_m=args.max_scp_distance_m,
            max_spacing_rel_diff=args.max_spacing_rel_diff,
            max_angle_diff_deg=args.max_angle_diff_deg,
        )
        text = (
            json.dumps(report.to_dict(), indent=2, allow_nan=False)
            if args.format == "json"
            else _text_report(report)
        )
        if args.output is None:
            print(text)
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
