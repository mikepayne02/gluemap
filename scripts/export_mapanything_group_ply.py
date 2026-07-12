#!/usr/bin/env python3
"""Export a saved Telluride MapAnything group prediction as a colored PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gluemap.datasets.polycam import load_polycam_manifest
from project_polycam_diagnostics import StreamingPlyWriter
from run_polycam_mapanything_group import _group_indices, _load_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("group_config", type=Path)
    parser.add_argument("group_name")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pixel-step", type=int, default=2)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    args = parser.parse_args()

    if args.pixel_step < 1:
        raise ValueError("pixel-step must be positive")

    manifest = load_polycam_manifest(args.manifest)
    config = json.loads(args.group_config.read_text(encoding="utf-8"))
    _, indices = _group_indices(config, args.group_name)
    saved = torch.load(args.predictions, map_location="cpu", weights_only=False)
    predictions = saved["predictions"]
    if len(predictions) != len(indices):
        raise ValueError("Prediction and configured view counts differ")

    writer = StreamingPlyWriter(args.output)
    for frame_index, prediction in zip(indices, predictions):
        depth = prediction["depth_z"].float().squeeze().numpy()
        height, width = depth.shape
        intrinsics = prediction["intrinsics"].float().squeeze().numpy()
        camera_pose = prediction["camera_poses"].float().squeeze().numpy()
        mask = prediction["non_ambiguous_mask"].bool().squeeze().numpy()
        valid = (
            mask
            & np.isfinite(depth)
            & (depth > 0.0)
            & (depth <= args.max_depth_m)
        )

        view = _load_view(
            manifest["frames"][frame_index],
            args.source_root,
            min_confidence=255,
            include_pose=False,
            orientation="upright_cw",
        )
        rgb = np.asarray(
            Image.fromarray(view["img"]).resize(
                (width, height), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )

        vv, uu = np.mgrid[0:height:args.pixel_step, 0:width:args.pixel_step]
        sampled_depth = depth[:: args.pixel_step, :: args.pixel_step]
        sampled_valid = valid[:: args.pixel_step, :: args.pixel_step]
        z = sampled_depth[sampled_valid]
        x = (uu[sampled_valid] - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (vv[sampled_valid] - intrinsics[1, 2]) * z / intrinsics[1, 1]
        camera_points = np.stack([x, y, z], axis=1)
        world_points = (
            camera_points @ camera_pose[:3, :3].T + camera_pose[:3, 3]
        )
        colors = rgb[:: args.pixel_step, :: args.pixel_step][sampled_valid]
        writer.write(world_points.astype(np.float32), colors)

    writer.close()
    print(json.dumps({"output": str(args.output), "vertices": writer.count}, indent=2))


if __name__ == "__main__":
    main()
