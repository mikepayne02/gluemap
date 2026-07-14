#!/usr/bin/env python3
"""Validate revisited geometry using measured depth and solved cameras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image
from scipy.spatial import cKDTree

from gluemap.datasets.polycam import rotate_intrinsics_cw


def _voxelize(points: np.ndarray, size: float) -> np.ndarray:
    cells = np.floor(points / size).astype(np.int32)
    _, keep = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(keep)]


def _depth_cloud(
    manifest_indices: list[int],
    mapping_by_manifest: dict[int, dict],
    manifest: dict,
    source_root: Path,
    images: dict[str, pycolmap.Image],
    pixel_step: int,
    voxel_size: float,
) -> np.ndarray:
    clouds = []
    for manifest_index in manifest_indices:
        row = mapping_by_manifest[manifest_index]
        frame = manifest["frames"][manifest_index]
        image = images[row["image_name"]]
        with Image.open(source_root / frame["paths"]["depth"]) as source:
            raw_depth = np.asarray(source, dtype=np.float32)
        with Image.open(source_root / frame["paths"]["confidence"]) as source:
            raw_confidence = np.asarray(source, dtype=np.uint8)
        depth = np.rot90(raw_depth, k=3) * float(
            manifest.get("depth_unit_m", 0.001)
        )
        confidence = np.rot90(raw_confidence, k=3)
        y, x = np.mgrid[
            0 : depth.shape[0] : pixel_step,
            0 : depth.shape[1] : pixel_step,
        ]
        z = depth[y, x]
        valid = (
            np.isfinite(z)
            & (z >= 0.08)
            & (z <= 5.0)
            & (confidence[y, x] == 255)
        )
        if not np.any(valid):
            continue
        intrinsics = rotate_intrinsics_cw(
            np.asarray(frame["intrinsics_depth"], dtype=np.float64),
            (raw_depth.shape[1], raw_depth.shape[0]),
        )
        z = z[valid].astype(np.float64)
        camera_points = np.column_stack(
            [
                (x[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (y[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ]
        )
        world_from_camera = np.asarray(
            image.cam_from_world().rotation.matrix(), dtype=np.float64
        ).T
        center = np.asarray(image.projection_center(), dtype=np.float64)
        clouds.append(camera_points @ world_from_camera.T + center)
    if not clouds:
        raise RuntimeError(
            f"No valid depth for manifest frames {manifest_indices}"
        )
    return _voxelize(np.concatenate(clouds), voxel_size)


def _split_contiguous(indices: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in sorted(indices):
        if not runs or index > runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(index)
    return runs


def _overlap(first: np.ndarray, second: np.ndarray) -> dict:
    first_to_second = cKDTree(second).query(first, workers=-1)[0]
    second_to_first = cKDTree(first).query(second, workers=-1)[0]
    distances = np.concatenate([first_to_second, second_to_first])
    return {
        "point_counts": [len(first), len(second)],
        "symmetric_distance_m": {
            "median": float(np.median(distances)),
            "p90": float(np.percentile(distances, 90)),
            "p95": float(np.percentile(distances, 95)),
        },
        "symmetric_overlap_fraction": {
            f"within_{threshold:.2f}m": float(np.mean(distances <= threshold))
            for threshold in (0.05, 0.10, 0.20, 0.30)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("index_mapping", type=Path)
    parser.add_argument("group_config", type=Path)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pixel-step", type=int, default=4)
    parser.add_argument("--voxel-size-m", type=float, default=0.03)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    mapping = json.loads(args.index_mapping.read_text(encoding="utf-8"))
    mapping_by_manifest = {int(row["manifest_index"]): row for row in mapping}
    groups = json.loads(args.group_config.read_text(encoding="utf-8"))["groups"]
    reconstruction = pycolmap.Reconstruction(str(args.reconstruction))
    images = {image.name: image for image in reconstruction.images.values()}
    missing = [
        row["image_name"] for row in mapping if row["image_name"] not in images
    ]
    if missing:
        raise RuntimeError(f"Reconstruction is missing {len(missing)} images")

    bridge_results = []
    pair_distances = {}
    for group in groups:
        if group.get("kind") != "verified_bridge":
            continue
        runs = _split_contiguous(group["frame_indices"])
        if len(runs) != 2:
            raise RuntimeError(f"{group['name']} does not contain two visits")
        clouds = [
            _depth_cloud(
                run,
                mapping_by_manifest,
                manifest,
                args.source_root,
                images,
                args.pixel_step,
                args.voxel_size_m,
            )
            for run in runs
        ]
        result = {
            "name": group["name"],
            "frame_ranges": [[run[0], run[-1]] for run in runs],
            **_overlap(*clouds),
        }
        bridge_results.append(result)
        for first, second in group["verified_correspondences"]:
            first_center = np.asarray(
                images[
                    mapping_by_manifest[first]["image_name"]
                ].projection_center()
            )
            second_center = np.asarray(
                images[
                    mapping_by_manifest[second]["image_name"]
                ].projection_center()
            )
            pair_distances[f"{first}-{second}"] = float(
                np.linalg.norm(first_center - second_center)
            )

    centers = np.stack(
        [
            np.asarray(images[row["image_name"]].projection_center())
            for row in mapping
        ]
    )
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    points = list(reconstruction.points3D.values())
    result = {
        "image_count": reconstruction.num_images(),
        "point_count": reconstruction.num_points3D(),
        "verified_camera_pair_distance_m": pair_distances,
        "consecutive_camera_step_m": {
            "median": float(np.median(steps)),
            "p99": float(np.percentile(steps, 99)),
            "maximum": float(np.max(steps)),
        },
        "bridge_depth_consistency": bridge_results,
    }
    if points:
        xyz = np.stack([np.asarray(point.xyz) for point in points])
        center = np.median(xyz, axis=0)
        radii = np.linalg.norm(xyz - center, axis=1)
        result["sparse_points"] = {
            "finite": int(np.isfinite(xyz).all(axis=1).sum()),
            "radius_m": {
                "p99": float(np.percentile(radii, 99)),
                "maximum": float(np.max(radii)),
            },
            "track_length": {
                "median": float(
                    np.median([len(point.track.elements) for point in points])
                ),
                "two_view_fraction": float(
                    np.mean(
                        [len(point.track.elements) == 2 for point in points]
                    )
                ),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
