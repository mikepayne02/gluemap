#!/usr/bin/env python3
"""Align pose-free MapAnything groups through shared predicted cameras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from gluemap.datasets.polycam import (
    load_polycam_manifest,
    rotate_c2w_opencv_cw,
)
from project_polycam_diagnostics import StreamingPlyWriter
from run_polycam_mapanything_group import _group_indices, _load_view


def _initial_transform(predicted: np.ndarray, arkit: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rotations = arkit[:, :3, :3] @ np.swapaxes(predicted[:, :3, :3], 1, 2)
    rotation = Rotation.from_matrix(rotations).mean().as_matrix()
    translations = arkit[:, :3, 3] - predicted[:, :3, 3] @ rotation.T
    return rotation, np.median(translations, axis=0), 1.0


def _pack(transforms: list[tuple[np.ndarray, np.ndarray, float]], anchor: int) -> np.ndarray:
    values = []
    for index, (rotation, translation, scale) in enumerate(transforms):
        if index == anchor:
            continue
        values.extend(Rotation.from_matrix(rotation).as_rotvec())
        values.extend(translation)
        values.append(np.log(scale))
    return np.asarray(values, dtype=np.float64)


def _unpack(
    parameters: np.ndarray,
    initial: list[tuple[np.ndarray, np.ndarray, float]],
    anchor: int,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    transforms = []
    offset = 0
    for index, anchored in enumerate(initial):
        if index == anchor:
            transforms.append(anchored)
            continue
        rotation = Rotation.from_rotvec(parameters[offset : offset + 3]).as_matrix()
        translation = parameters[offset + 3 : offset + 6]
        scale = float(np.exp(parameters[offset + 6]))
        transforms.append((rotation, translation, scale))
        offset += 7
    return transforms


def _transform_pose(
    pose: np.ndarray, transform: tuple[np.ndarray, np.ndarray, float]
) -> np.ndarray:
    rotation, translation, scale = transform
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation @ pose[:3, :3]
    result[:3, 3] = scale * (rotation @ pose[:3, 3]) + translation
    return result


def _shared_edges(groups: list[dict]) -> list[tuple[int, int, list[int]]]:
    memberships = [set(group["frame_indices"]) for group in groups]
    edges = []
    for first in range(len(groups)):
        for second in range(first + 1, len(groups)):
            shared = sorted(memberships[first] & memberships[second])
            if len(shared) >= 2:
                edges.append((first, second, shared))
    return edges


def _check_connected(group_count: int, edges: list[tuple[int, int, list[int]]]) -> None:
    adjacency = {index: set() for index in range(group_count)}
    for first, second, _ in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    reached = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    if len(reached) != group_count:
        missing = sorted(set(range(group_count)) - reached)
        raise RuntimeError(f"Group graph is disconnected; missing {missing}")


def _global_cameras(groups: list[dict], transforms: list[tuple[np.ndarray, np.ndarray, float]]) -> dict[int, np.ndarray]:
    candidates: dict[int, list[np.ndarray]] = {}
    for group, transform in zip(groups, transforms):
        for frame_index, pose in zip(group["frame_indices"], group["predicted_poses"]):
            candidates.setdefault(frame_index, []).append(_transform_pose(pose, transform))
    cameras = {}
    for frame_index, poses in candidates.items():
        rotations = Rotation.from_matrix(np.stack([pose[:3, :3] for pose in poses])).mean().as_matrix()
        translations = np.median(np.stack([pose[:3, 3] for pose in poses]), axis=0)
        camera = np.eye(4, dtype=np.float64)
        camera[:3, :3] = rotations
        camera[:3, 3] = translations
        cameras[frame_index] = camera
    return cameras


def _shared_camera_stats(
    groups: list[dict],
    transforms: list[tuple[np.ndarray, np.ndarray, float]],
    edges: list[tuple[int, int, list[int]]],
) -> dict[str, float]:
    position_errors = []
    rotation_errors = []
    for first, second, shared in edges:
        for frame_index in shared:
            first_pose = _transform_pose(
                groups[first]["pose_by_frame"][frame_index], transforms[first]
            )
            second_pose = _transform_pose(
                groups[second]["pose_by_frame"][frame_index], transforms[second]
            )
            position_errors.append(
                np.linalg.norm(first_pose[:3, 3] - second_pose[:3, 3])
            )
            rotation_errors.append(
                np.rad2deg(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            first_pose[:3, :3].T @ second_pose[:3, :3]
                        ).as_rotvec()
                    )
                )
            )
    return {
        "position_median_m": float(np.median(position_errors)),
        "position_p95_m": float(np.percentile(position_errors, 95)),
        "rotation_median_deg": float(np.median(rotation_errors)),
        "rotation_p95_deg": float(np.percentile(rotation_errors, 95)),
    }


def _edge_initial_stats(
    groups: list[dict],
    transforms: list[tuple[np.ndarray, np.ndarray, float]],
    edge: tuple[int, int, list[int]],
) -> tuple[float, float]:
    first, second, shared = edge
    position_errors = []
    rotation_errors = []
    for frame_index in shared:
        first_pose = _transform_pose(
            groups[first]["pose_by_frame"][frame_index], transforms[first]
        )
        second_pose = _transform_pose(
            groups[second]["pose_by_frame"][frame_index], transforms[second]
        )
        position_errors.append(
            np.linalg.norm(first_pose[:3, 3] - second_pose[:3, 3])
        )
        rotation_errors.append(
            np.rad2deg(
                np.linalg.norm(
                    Rotation.from_matrix(
                        first_pose[:3, :3].T @ second_pose[:3, :3]
                    ).as_rotvec()
                )
            )
        )
    return float(np.median(position_errors)), float(np.median(rotation_errors))


def _export_cloud(
    groups: list[dict],
    transforms: list[tuple[np.ndarray, np.ndarray, float]],
    manifest: dict,
    source_root: Path,
    output_path: Path,
    *,
    pixel_step: int,
) -> int:
    choices: dict[int, tuple[float, int, int]] = {}
    for group_index, group in enumerate(groups):
        for local_index, (frame_index, prediction) in enumerate(
            zip(group["frame_indices"], group["predictions"])
        ):
            confidence = float(prediction["conf"].float().mean())
            candidate = (confidence, group_index, local_index)
            if frame_index not in choices or candidate[0] > choices[frame_index][0]:
                choices[frame_index] = candidate

    writer = StreamingPlyWriter(output_path)
    for frame_index in sorted(choices):
        _, group_index, local_index = choices[frame_index]
        group = groups[group_index]
        prediction = group["predictions"][local_index]
        depth = prediction["depth_z"].float().squeeze().numpy()
        intrinsics = prediction["intrinsics"].float().squeeze().numpy()
        local_pose = group["predicted_poses"][local_index]
        global_pose = _transform_pose(local_pose, transforms[group_index])
        mask = prediction["non_ambiguous_mask"].bool().squeeze().numpy()
        valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth <= 8.0)
        height, width = depth.shape
        view = _load_view(
            manifest["frames"][frame_index],
            source_root,
            min_confidence=255,
            include_pose=False,
            orientation="upright_cw",
        )
        rgb = np.asarray(
            Image.fromarray(view["img"]).resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        vv, uu = np.mgrid[0:height:pixel_step, 0:width:pixel_step]
        sampled_depth = depth[::pixel_step, ::pixel_step]
        sampled_valid = valid[::pixel_step, ::pixel_step]
        z = sampled_depth[sampled_valid]
        x = (uu[sampled_valid] - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (vv[sampled_valid] - intrinsics[1, 2]) * z / intrinsics[1, 1]
        camera_points = np.stack([x, y, z], axis=1)
        world_points = camera_points @ global_pose[:3, :3].T + global_pose[:3, 3]
        colors = rgb[::pixel_step, ::pixel_step][sampled_valid]
        writer.write(world_points.astype(np.float32), colors)
    writer.close()
    return writer.count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("group_config", type=Path)
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pixel-step", type=int, default=4)
    parser.add_argument("--skip-cloud", action="store_true")
    parser.add_argument("--anchor-group", default=None)
    parser.add_argument("--shared-position-sigma-m", type=float, default=0.10)
    parser.add_argument("--shared-rotation-sigma-deg", type=float, default=5.0)
    parser.add_argument("--arkit-translation-sigma-m", type=float, default=3.0)
    parser.add_argument("--arkit-rotation-sigma-deg", type=float, default=30.0)
    parser.add_argument("--scale-prior-sigma", type=float, default=0.10)
    parser.add_argument("--max-nfev", type=int, default=150)
    parser.add_argument("--max-edge-position-median-m", type=float, default=None)
    parser.add_argument("--max-edge-rotation-median-deg", type=float, default=None)
    args = parser.parse_args()

    manifest = load_polycam_manifest(args.manifest)
    config = json.loads(args.group_config.read_text(encoding="utf-8"))
    groups = []
    for group_spec in config["groups"]:
        name = group_spec["name"]
        prediction_path = args.results_directory / name / "predictions.pt"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        saved = torch.load(prediction_path, map_location="cpu", weights_only=False)
        _, frame_indices = _group_indices(config, name)
        predicted_poses = np.stack(
            [prediction["camera_poses"].double().squeeze().numpy() for prediction in saved["predictions"]]
        )
        arkit_poses = np.stack(
            [
                rotate_c2w_opencv_cw(
                    np.asarray(manifest["frames"][index]["corrected_c2w_opencv"])
                )
                for index in frame_indices
            ]
        )
        groups.append(
            {
                "name": name,
                "kind": group_spec.get("kind"),
                "frame_indices": frame_indices,
                "predicted_poses": predicted_poses,
                "arkit_poses": arkit_poses,
                "predictions": saved["predictions"],
                "pose_by_frame": dict(zip(frame_indices, predicted_poses)),
            }
        )

    edges = _shared_edges(groups)
    unfiltered_edge_count = len(edges)
    initial = [
        _initial_transform(group["predicted_poses"], group["arkit_poses"])
        for group in groups
    ]
    if (
        args.max_edge_position_median_m is not None
        or args.max_edge_rotation_median_deg is not None
    ):
        filtered_edges = []
        rejected_edges = []
        for edge in edges:
            position_median, rotation_median = _edge_initial_stats(
                groups, initial, edge
            )
            score = max(
                position_median / args.max_edge_position_median_m
                if args.max_edge_position_median_m is not None
                else 0.0,
                rotation_median / args.max_edge_rotation_median_deg
                if args.max_edge_rotation_median_deg is not None
                else 0.0,
            )
            if (
                args.max_edge_position_median_m is not None
                and position_median > args.max_edge_position_median_m
            ):
                rejected_edges.append((score, edge))
                continue
            if (
                args.max_edge_rotation_median_deg is not None
                and rotation_median > args.max_edge_rotation_median_deg
            ):
                rejected_edges.append((score, edge))
                continue
            filtered_edges.append(edge)
        parent = list(range(len(groups)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> bool:
            first_root = find(first)
            second_root = find(second)
            if first_root == second_root:
                return False
            parent[second_root] = first_root
            return True

        for first, second, _ in filtered_edges:
            union(first, second)
        connectivity_edges_added = 0
        for _, edge in sorted(rejected_edges, key=lambda item: item[0]):
            if union(edge[0], edge[1]):
                filtered_edges.append(edge)
                connectivity_edges_added += 1
        edges = filtered_edges
    else:
        connectivity_edges_added = 0
    print(
        json.dumps(
            {
                "status": "edge_filter",
                "shared_group_edges_before": unfiltered_edge_count,
                "shared_group_edges_after": len(edges),
                "connectivity_edges_added": connectivity_edges_added,
            }
        ),
        flush=True,
    )
    _check_connected(len(groups), edges)
    shared_first = []
    shared_second = []
    shared_first_centers = []
    shared_second_centers = []
    shared_first_rotations = []
    shared_second_rotations = []
    for first, second, shared in edges:
        for frame_index in shared:
            first_pose = groups[first]["pose_by_frame"][frame_index]
            second_pose = groups[second]["pose_by_frame"][frame_index]
            shared_first.append(first)
            shared_second.append(second)
            shared_first_centers.append(first_pose[:3, 3])
            shared_second_centers.append(second_pose[:3, 3])
            shared_first_rotations.append(first_pose[:3, :3])
            shared_second_rotations.append(second_pose[:3, :3])
    shared_first = np.asarray(shared_first, dtype=np.int64)
    shared_second = np.asarray(shared_second, dtype=np.int64)
    shared_first_centers = np.asarray(shared_first_centers)
    shared_second_centers = np.asarray(shared_second_centers)
    shared_first_rotations = np.asarray(shared_first_rotations)
    shared_second_rotations = np.asarray(shared_second_rotations)
    group_names = [group["name"] for group in groups]
    anchor_name = args.anchor_group
    if anchor_name is None and "basement_lower_landing" in group_names:
        anchor_name = "basement_lower_landing"
    if anchor_name is not None:
        if anchor_name not in group_names:
            raise ValueError(f"Unknown anchor group {anchor_name!r}")
        anchor = group_names.index(anchor_name)
    else:
        segment_start = int(config["segment"]["start"])
        anchor = min(
            range(len(groups)),
            key=lambda index: (
                segment_start not in groups[index]["frame_indices"],
                groups[index]["kind"] == "verified_bridge",
                len(groups[index]["frame_indices"]),
            ),
        )

    def residuals(parameters: np.ndarray) -> np.ndarray:
        transforms = _unpack(parameters, initial, anchor)
        transform_rotations = np.stack([transform[0] for transform in transforms])
        transform_translations = np.stack([transform[1] for transform in transforms])
        transform_scales = np.asarray([transform[2] for transform in transforms])
        first_centers = (
            transform_scales[shared_first, None]
            * np.einsum(
                "nij,nj->ni",
                transform_rotations[shared_first],
                shared_first_centers,
            )
            + transform_translations[shared_first]
        )
        second_centers = (
            transform_scales[shared_second, None]
            * np.einsum(
                "nij,nj->ni",
                transform_rotations[shared_second],
                shared_second_centers,
            )
            + transform_translations[shared_second]
        )
        first_rotations = np.einsum(
            "nij,njk->nik",
            transform_rotations[shared_first],
            shared_first_rotations,
        )
        second_rotations = np.einsum(
            "nij,njk->nik",
            transform_rotations[shared_second],
            shared_second_rotations,
        )
        rotation_errors = Rotation.from_matrix(
            np.einsum("nji,njk->nik", first_rotations, second_rotations)
        ).as_rotvec()
        shared_residual = np.concatenate(
            [
                (first_centers - second_centers) / args.shared_position_sigma_m,
                rotation_errors / np.deg2rad(args.shared_rotation_sigma_deg),
            ],
            axis=1,
        ).ravel()
        prior_residual = []
        for index, (current, prior) in enumerate(zip(transforms, initial)):
            if index == anchor:
                continue
            rotation_error = Rotation.from_matrix(prior[0].T @ current[0]).as_rotvec()
            prior_residual.extend(
                rotation_error / np.deg2rad(args.arkit_rotation_sigma_deg)
            )
            prior_residual.extend(
                (current[1] - prior[1]) / args.arkit_translation_sigma_m
            )
            prior_residual.append(np.log(current[2]) / args.scale_prior_sigma)
        return np.concatenate([shared_residual, np.asarray(prior_residual)])

    initial_parameters = _pack(initial, anchor)
    before = residuals(initial_parameters)
    shared_stats_before = _shared_camera_stats(groups, initial, edges)
    print(
        json.dumps(
            {
                "status": "optimizing",
                "groups": len(groups),
                "shared_group_edges": len(edges),
                "shared_camera_stats_before": shared_stats_before,
            }
        ),
        flush=True,
    )
    parameter_columns = {}
    column = 0
    for index in range(len(groups)):
        if index == anchor:
            continue
        parameter_columns[index] = slice(column, column + 7)
        column += 7
    residual_count = 6 * sum(len(shared) for _, _, shared in edges)
    residual_count += 7 * (len(groups) - 1)
    sparsity = lil_matrix((residual_count, len(initial_parameters)), dtype=np.int8)
    row = 0
    for first, second, shared in edges:
        for _ in shared:
            for index in (first, second):
                if index != anchor:
                    sparsity[row : row + 6, parameter_columns[index]] = 1
            row += 6
    for index in range(len(groups)):
        if index == anchor:
            continue
        sparsity[row : row + 7, parameter_columns[index]] = 1
        row += 7
    solution = least_squares(
        residuals,
        initial_parameters,
        jac_sparsity=sparsity.tocsr(),
        loss="huber",
        f_scale=1.0,
        max_nfev=args.max_nfev,
        tr_solver="lsmr",
        verbose=1,
    )
    transforms = _unpack(solution.x, initial, anchor)
    after = residuals(solution.x)
    shared_stats_after = _shared_camera_stats(groups, transforms, edges)
    cameras = _global_cameras(groups, transforms)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    transforms_json = {
        group["name"]: {
            "rotation": transform[0].tolist(),
            "translation": transform[1].tolist(),
            "scale": transform[2],
        }
        for group, transform in zip(groups, transforms)
    }
    (args.output_directory / "group_transforms.json").write_text(
        json.dumps(transforms_json, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_directory / "optimized_cameras.json").write_text(
        json.dumps({str(index): pose.tolist() for index, pose in cameras.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    center_writer = StreamingPlyWriter(
        args.output_directory / "optimized_camera_centers.ply"
    )
    ordered_camera_indices = sorted(cameras)
    camera_centers = np.stack(
        [cameras[index][:3, 3] for index in ordered_camera_indices]
    ).astype(np.float32)
    center_writer.write(
        camera_centers,
        np.tile(np.array([[0, 255, 255]], dtype=np.uint8), (len(camera_centers), 1)),
    )
    center_writer.close()
    vertex_count = 0
    if not args.skip_cloud:
        vertex_count = _export_cloud(
            groups,
            transforms,
            manifest,
            args.source_root,
            args.output_directory / "fused_cloud.ply",
            pixel_step=args.pixel_step,
        )
    arkit_centers = np.stack(
        [
            np.asarray(manifest["frames"][index]["corrected_c2w_opencv"])[:3, 3]
            for index in ordered_camera_indices
        ]
    )
    camera_displacements = np.linalg.norm(camera_centers - arkit_centers, axis=1)

    def camera_distance(first: int, second: int) -> float | None:
        if first not in cameras or second not in cameras:
            return None
        return float(
            np.linalg.norm(cameras[first][:3, 3] - cameras[second][:3, 3])
        )

    summary = {
        "groups": len(groups),
        "shared_group_edges": len(edges),
        "shared_group_edges_before_filter": unfiltered_edge_count,
        "connectivity_edges_added": connectivity_edges_added,
        "anchor_group": groups[anchor]["name"],
        "weights": {
            "shared_position_sigma_m": args.shared_position_sigma_m,
            "shared_rotation_sigma_deg": args.shared_rotation_sigma_deg,
            "arkit_translation_sigma_m": args.arkit_translation_sigma_m,
            "arkit_rotation_sigma_deg": args.arkit_rotation_sigma_deg,
            "scale_prior_sigma": args.scale_prior_sigma,
        },
        "solver_success": bool(solution.success),
        "solver_message": solution.message,
        "residual_rms_before": float(np.sqrt(np.mean(before**2))),
        "residual_rms_after": float(np.sqrt(np.mean(after**2))),
        "shared_camera_stats_before": shared_stats_before,
        "shared_camera_stats_after": shared_stats_after,
        "scale_min": float(min(transform[2] for transform in transforms)),
        "scale_max": float(max(transform[2] for transform in transforms)),
        "camera_count": len(cameras),
        "camera_displacement_from_arkit_m": {
            "median": float(np.median(camera_displacements)),
            "p95": float(np.percentile(camera_displacements, 95)),
            "maximum": float(np.max(camera_displacements)),
        },
        "verified_camera_distances_m": {
            "2379-3034": camera_distance(2379, 3034),
            "2380-3034": camera_distance(2380, 3034),
            "2232-3088": camera_distance(2232, 3088),
        },
        "vertices": vertex_count,
    }
    (args.output_directory / "alignment_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
