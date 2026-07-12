#!/usr/bin/env python3
"""Generate the canonical Telluride Polycam manifest and calibration audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gluemap.datasets.polycam import write_polycam_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("polycam_root", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--reset-jump-threshold-m",
        type=float,
        default=2.0,
        help="Camera-center step treated as a tracking reset (default: 2 m).",
    )
    args = parser.parse_args()
    manifest_path, report_path = write_polycam_audit(
        args.polycam_root,
        args.output_directory,
        reset_jump_threshold_m=args.reset_jump_threshold_m,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {report_path}")
    print(
        f"Validated {report['frame_count']} frames; resets after "
        f"{report['reset_after_sequence_indices']}"
    )


if __name__ == "__main__":
    main()
