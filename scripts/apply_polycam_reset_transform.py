#!/usr/bin/env python3
"""Apply an accepted rigid reset transform to a canonical Polycam manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gluemap.datasets.polycam import write_corrected_polycam_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("transform", type=Path)
    parser.add_argument("output_manifest", type=Path)
    args = parser.parse_args()
    output = write_corrected_polycam_manifest(
        args.manifest, args.transform, args.output_manifest
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    correction = manifest["pose_correction"]
    print(f"Wrote {output}")
    print(
        "Corrected boundary step: "
        f"{correction['corrected_boundary_step_m']:.6f} m"
    )


if __name__ == "__main__":
    main()
