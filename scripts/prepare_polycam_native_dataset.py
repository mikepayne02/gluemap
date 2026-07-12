#!/usr/bin/env python3
"""Materialize an upright, sequence-ordered image set for native GLUEMAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from gluemap.datasets.polycam import load_polycam_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--exclude-frame", type=int, action="append", default=[]
    )
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    manifest = load_polycam_manifest(args.manifest)
    excluded = set(args.exclude_frame)
    images_directory = args.output_directory / "images"
    images_directory.mkdir(parents=True, exist_ok=True)
    mapping = []
    for manifest_index, frame in enumerate(manifest["frames"]):
        if manifest_index in excluded:
            continue
        native_index = len(mapping)
        image_name = f"{native_index:06d}.jpg"
        output_path = images_directory / image_name
        if not output_path.is_file():
            source_path = args.source_root / frame["paths"]["rgb"]
            with Image.open(source_path) as image:
                upright = image.convert("RGB").transpose(
                    Image.Transpose.ROTATE_270
                )
                upright.save(output_path, quality=args.quality, subsampling=0)
        mapping.append(
            {
                "native_index": native_index,
                "manifest_index": manifest_index,
                "image_name": image_name,
                "timestamp": int(frame["timestamp"]),
            }
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "index_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output_directory),
                "image_count": len(mapping),
                "excluded_frames": sorted(excluded),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
