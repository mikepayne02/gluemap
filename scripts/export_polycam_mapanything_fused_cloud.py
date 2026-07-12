#!/usr/bin/env python3
"""Export a fused PLY from saved MapAnything groups and solved transforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from align_polycam_mapanything_groups import _export_cloud
from gluemap.datasets.polycam import load_polycam_manifest
from run_polycam_mapanything_group import _group_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("group_config", type=Path)
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("group_transforms", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pixel-step", type=int, default=4)
    args = parser.parse_args()

    manifest = load_polycam_manifest(args.manifest)
    config = json.loads(args.group_config.read_text(encoding="utf-8"))
    saved_transforms = json.loads(args.group_transforms.read_text(encoding="utf-8"))
    groups = []
    transforms = []
    for group_spec in config["groups"]:
        name = group_spec["name"]
        saved = torch.load(
            args.results_directory / name / "predictions.pt",
            map_location="cpu",
            weights_only=False,
        )
        _, frame_indices = _group_indices(config, name)
        predicted_poses = np.stack(
            [
                prediction["camera_poses"].double().squeeze().numpy()
                for prediction in saved["predictions"]
            ]
        )
        groups.append(
            {
                "name": name,
                "frame_indices": frame_indices,
                "predicted_poses": predicted_poses,
                "predictions": saved["predictions"],
            }
        )
        transform = saved_transforms[name]
        transforms.append(
            (
                np.asarray(transform["rotation"], dtype=np.float64),
                np.asarray(transform["translation"], dtype=np.float64),
                float(transform["scale"]),
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vertices = _export_cloud(
        groups,
        transforms,
        manifest,
        args.source_root,
        args.output,
        pixel_step=args.pixel_step,
    )
    print(json.dumps({"output": str(args.output), "vertices": vertices}, indent=2))


if __name__ == "__main__":
    main()
