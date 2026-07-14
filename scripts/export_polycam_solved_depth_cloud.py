#!/usr/bin/env python3
"""Fuse Polycam LiDAR depth using a solved COLMAP camera trajectory.

The COLMAP reconstruction remains unchanged. A gravity-constrained diagnostic
similarity puts the cloud into the familiar upright, metric house frame without
using ARKit positions as optimization targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image


def _rotate_intrinsics_cw(
    intrinsics: np.ndarray, source_size_wh: tuple[int, int]
) -> np.ndarray:
    width, height = source_size_wh
    matrix = np.asarray(intrinsics, dtype=np.float64)
    return np.array(
        [
            [matrix[1, 1], 0.0, (height - 1.0) - matrix[1, 2]],
            [0.0, matrix[0, 0], matrix[0, 2]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _umeyama(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate ``target = scale * rotation @ source + translation``."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_t
    source_variance = np.mean(np.sum(source_centered**2, axis=1))
    scale = float(np.sum(singular_values * signs) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _robust_similarity(
    source: np.ndarray,
    target: np.ndarray,
    retain_fraction: float = 0.85,
    iterations: int = 6,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    retain = max(3, int(round(len(source) * retain_fraction)))
    mask = np.ones(len(source), dtype=bool)
    for _ in range(iterations):
        scale, rotation, translation = _umeyama(source[mask], target[mask])
        fitted = scale * (source @ rotation.T) + translation
        residuals = np.linalg.norm(fitted - target, axis=1)
        indices = np.argpartition(residuals, retain - 1)[:retain]
        next_mask = np.zeros(len(source), dtype=bool)
        next_mask[indices] = True
        if np.array_equal(mask, next_mask):
            break
        mask = next_mask
    scale, rotation, translation = _umeyama(source[mask], target[mask])
    fitted = scale * (source @ rotation.T) + translation
    residuals = np.linalg.norm(fitted - target, axis=1)
    return scale, rotation, translation, mask, residuals


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest rotation mapping one unit direction to another."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-12:
        if cosine > 0.0:
            return np.eye(3)
        basis = np.eye(3)[np.argmin(np.abs(source))]
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def _gravity_constrained_similarity(
    source: np.ndarray,
    target: np.ndarray,
    source_gravity: np.ndarray,
    retain_fraction: float = 0.85,
    iterations: int = 6,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit scale, yaw, and translation after mapping gravity exactly to +Y."""
    tilt = _rotation_between(source_gravity, np.array([0.0, 1.0, 0.0]))
    tilted = source @ tilt.T
    retain = max(3, int(round(len(source) * retain_fraction)))
    mask = np.ones(len(source), dtype=bool)
    for _ in range(iterations):
        source_xz = tilted[mask][:, [0, 2]]
        target_xz = target[mask][:, [0, 2]]
        source_xz -= source_xz.mean(axis=0)
        target_xz -= target_xz.mean(axis=0)
        left, _, right_t = np.linalg.svd(target_xz.T @ source_xz)
        signs = np.ones(2)
        if np.linalg.det(left @ right_t) < 0:
            signs[-1] = -1.0
        yaw_xz = left @ np.diag(signs) @ right_t
        yaw = np.array(
            [
                [yaw_xz[0, 0], 0.0, yaw_xz[0, 1]],
                [0.0, 1.0, 0.0],
                [yaw_xz[1, 0], 0.0, yaw_xz[1, 1]],
            ]
        )
        rotation = yaw @ tilt
        rotated = source @ rotation.T
        source_centered = rotated[mask] - rotated[mask].mean(axis=0)
        target_centered = target[mask] - target[mask].mean(axis=0)
        scale = float(
            np.sum(source_centered * target_centered)
            / np.sum(source_centered**2)
        )
        translation = target[mask].mean(axis=0) - scale * rotated[mask].mean(
            axis=0
        )
        fitted = scale * rotated + translation
        residuals = np.linalg.norm(fitted - target, axis=1)
        indices = np.argpartition(residuals, retain - 1)[:retain]
        next_mask = np.zeros(len(source), dtype=bool)
        next_mask[indices] = True
        if np.array_equal(mask, next_mask):
            break
        mask = next_mask
    return scale, rotation, translation, mask, residuals


def _rotate_c2w_opencv_cw(c2w_opencv: np.ndarray) -> np.ndarray:
    old_camera_from_new = np.eye(4, dtype=np.float64)
    old_camera_from_new[:3, :3] = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return np.asarray(c2w_opencv, dtype=np.float64) @ old_camera_from_new


def _voxelize(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points.astype(np.float32), colors.astype(np.uint8)
    coordinates = np.floor(points / voxel_size).astype(np.int32)
    keys = np.empty(
        len(coordinates),
        dtype=np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4")]),
    )
    keys["x"], keys["y"], keys["z"] = coordinates.T
    _, first = np.unique(keys, return_index=True)
    first.sort()
    return points[first].astype(np.float32), colors[first].astype(np.uint8)


def _write_binary_ply(
    path: Path, points: np.ndarray, colors: np.ndarray
) -> None:
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(points), dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def _camera_from_world_rotation(image: pycolmap.Image) -> np.ndarray:
    return np.asarray(
        image.cam_from_world().rotation.matrix(), dtype=np.float64
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("index_mapping", type=Path)
    parser.add_argument("images_root", type=Path)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--pixel-step", type=int, default=2)
    parser.add_argument("--minimum-confidence", type=int, default=255)
    parser.add_argument("--minimum-depth-m", type=float, default=0.08)
    parser.add_argument("--maximum-depth-m", type=float, default=5.0)
    parser.add_argument("--voxel-size-m", type=float, default=0.025)
    parser.add_argument("--chunk-frames", type=int, default=128)
    parser.add_argument("--alignment-retain-fraction", type=float, default=0.85)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    mapping = json.loads(args.index_mapping.read_text(encoding="utf-8"))
    reconstruction = pycolmap.Reconstruction(str(args.reconstruction))
    images_by_name = {
        image.name: image for image in reconstruction.images.values()
    }

    solved_centers = []
    arkit_centers = []
    gravity_estimates = []
    for row in mapping:
        name = row["image_name"]
        if name not in images_by_name:
            raise RuntimeError(f"COLMAP model does not register {name}")
        solved_centers.append(images_by_name[name].projection_center())
        frame = manifest["frames"][int(row["manifest_index"])]
        arkit_centers.append(np.asarray(frame["corrected_c2w_opencv"])[:3, 3])
        prior_c2w = _rotate_c2w_opencv_cw(frame["corrected_c2w_opencv"])
        camera_gravity = prior_c2w[:3, :3].T @ np.array([0.0, 1.0, 0.0])
        gravity_estimates.append(
            _camera_from_world_rotation(images_by_name[name]).T @ camera_gravity
        )
    solved_centers = np.asarray(solved_centers, dtype=np.float64)
    arkit_centers = np.asarray(arkit_centers, dtype=np.float64)
    solved_gravity = np.sum(gravity_estimates, axis=0)
    solved_gravity /= np.linalg.norm(solved_gravity)
    scale, alignment_rotation, alignment_translation, inliers, residuals = (
        _gravity_constrained_similarity(
            solved_centers,
            arkit_centers,
            solved_gravity,
            retain_fraction=args.alignment_retain_fraction,
        )
    )
    aligned_gravity = alignment_rotation @ solved_gravity

    output_points: list[np.ndarray] = []
    output_colors: list[np.ndarray] = []
    chunk_points: list[np.ndarray] = []
    chunk_colors: list[np.ndarray] = []
    input_points = 0

    camera_poses: dict[str, list[list[float]]] = {}
    for sequence_index, row in enumerate(mapping):
        image_name = row["image_name"]
        image = images_by_name[image_name]
        frame = manifest["frames"][int(row["manifest_index"])]

        raw_depth_path = args.source_root / frame["paths"]["depth"]
        raw_confidence_path = args.source_root / frame["paths"]["confidence"]
        with Image.open(raw_depth_path) as depth_image:
            raw_depth = np.asarray(depth_image, dtype=np.float32)
        with Image.open(raw_confidence_path) as confidence_image:
            raw_confidence = np.asarray(confidence_image, dtype=np.uint8)
        depth = np.rot90(raw_depth, k=3).copy() * float(
            manifest.get("depth_unit_m", 0.001)
        )
        confidence = np.rot90(raw_confidence, k=3).copy()

        y, x = np.mgrid[
            0 : depth.shape[0] : args.pixel_step,
            0 : depth.shape[1] : args.pixel_step,
        ]
        z = depth[y, x]
        valid = (
            np.isfinite(z)
            & (z >= args.minimum_depth_m)
            & (z <= args.maximum_depth_m)
            & (confidence[y, x] >= args.minimum_confidence)
        )
        if not np.any(valid):
            continue

        raw_width, raw_height = raw_depth.shape[1], raw_depth.shape[0]
        intrinsics = _rotate_intrinsics_cw(
            np.asarray(frame["intrinsics_depth"], dtype=np.float64),
            (raw_width, raw_height),
        )
        sampled_x = x[valid].astype(np.float64)
        sampled_y = y[valid].astype(np.float64)
        sampled_z = z[valid].astype(np.float64)
        points_camera = np.column_stack(
            [
                (sampled_x - intrinsics[0, 2]) * sampled_z / intrinsics[0, 0],
                (sampled_y - intrinsics[1, 2]) * sampled_z / intrinsics[1, 1],
                sampled_z,
            ]
        )

        world_from_camera_rotation = _camera_from_world_rotation(image).T
        solved_center = np.asarray(image.projection_center(), dtype=np.float64)
        points_solved = (
            points_camera @ world_from_camera_rotation.T + solved_center
        )
        points_aligned = (
            scale * (points_solved @ alignment_rotation.T)
            + alignment_translation
        ).astype(np.float32)

        with Image.open(args.images_root / image_name) as rgb_image:
            rgb = np.asarray(rgb_image.convert("RGB"), dtype=np.uint8)
        rgb_x = np.clip(
            np.round(
                (x[valid] + 0.5) * rgb.shape[1] / depth.shape[1] - 0.5
            ).astype(int),
            0,
            rgb.shape[1] - 1,
        )
        rgb_y = np.clip(
            np.round(
                (y[valid] + 0.5) * rgb.shape[0] / depth.shape[0] - 0.5
            ).astype(int),
            0,
            rgb.shape[0] - 1,
        )
        colors = rgb[rgb_y, rgb_x]
        input_points += len(points_aligned)
        chunk_points.append(points_aligned)
        chunk_colors.append(colors)

        aligned_rotation = alignment_rotation @ world_from_camera_rotation
        aligned_center = (
            scale * (alignment_rotation @ solved_center) + alignment_translation
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = aligned_rotation
        pose[:3, 3] = aligned_center
        camera_poses[image_name] = pose.tolist()

        if (
            len(chunk_points) >= args.chunk_frames
            or sequence_index == len(mapping) - 1
        ):
            points, colors = _voxelize(
                np.concatenate(chunk_points),
                np.concatenate(chunk_colors),
                args.voxel_size_m,
            )
            output_points.append(points)
            output_colors.append(colors)
            chunk_points.clear()
            chunk_colors.clear()
            print(
                json.dumps(
                    {
                        "processed_frames": sequence_index + 1,
                        "input_points": input_points,
                        "chunk_voxels": len(points),
                    }
                ),
                flush=True,
            )

    points, colors = _voxelize(
        np.concatenate(output_points),
        np.concatenate(output_colors),
        args.voxel_size_m,
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    cloud_path = args.output_directory / "final_solved_lidar_dense.ply"
    cameras_path = (
        args.output_directory / "final_solved_cameras_arkit_axes.json"
    )
    summary_path = args.output_directory / "dense_fusion_summary.json"
    _write_binary_ply(cloud_path, points, colors)
    cameras_path.write_text(json.dumps(camera_poses) + "\n", encoding="utf-8")
    summary = {
        "purpose": "diagnostic_only",
        "source": (
            "measured Polycam LiDAR depth back-projected through "
            f"{args.reconstruction.name} cameras"
        ),
        "arkit_used_for_optimization": False,
        "frames": len(mapping),
        "input_depth_points": input_points,
        "output_voxels": len(points),
        "pixel_step": args.pixel_step,
        "minimum_confidence": args.minimum_confidence,
        "depth_range_m": [args.minimum_depth_m, args.maximum_depth_m],
        "voxel_size_m": args.voxel_size_m,
        "diagnostic_alignment": {
            "scale": scale,
            "rotation": alignment_rotation.tolist(),
            "translation": alignment_translation.tolist(),
            "inlier_cameras": int(inliers.sum()),
            "median_camera_disagreement_m": float(np.median(residuals)),
            "p90_camera_disagreement_m": float(np.percentile(residuals, 90)),
            "max_camera_disagreement_m": float(residuals.max()),
            "solved_gravity": solved_gravity.tolist(),
            "aligned_gravity": aligned_gravity.tolist(),
            "gravity_alignment_error_deg": float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            aligned_gravity @ np.array([0.0, 1.0, 0.0]),
                            -1.0,
                            1.0,
                        )
                    )
                )
            ),
        },
        "cloud": str(cloud_path),
        "cameras": str(cameras_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
