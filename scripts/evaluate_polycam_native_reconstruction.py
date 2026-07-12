#!/usr/bin/env python3
"""Align a native GLUEMAP reconstruction to ARKit and export diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pycolmap
from project_polycam_diagnostics import StreamingPlyWriter
from scipy.spatial.transform import Rotation

from gluemap.datasets.polycam import load_polycam_manifest, rotate_c2w_opencv_cw


def _similarity(source: np.ndarray, target: np.ndarray):
    source_mean, target_mean = source.mean(0), target.mean(0)
    centered_source, centered_target = (
        source - source_mean,
        target - target_mean,
    )
    left, singular_values, right_t = np.linalg.svd(
        centered_source.T @ centered_target / len(source)
    )
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    scale = float(
        singular_values.sum()
        / np.mean(np.sum(centered_source * centered_source, axis=1))
    )
    translation = target_mean - scale * (rotation @ source_mean)
    return rotation, translation, scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("native_summary", type=Path)
    parser.add_argument("reconstruction", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    manifest = load_polycam_manifest(args.manifest)
    native_summary = json.loads(args.native_summary.read_text(encoding="utf-8"))
    manifest_indices = [
        int(index) for index in native_summary["native_to_manifest"]
    ]
    reconstruction = pycolmap.Reconstruction(str(args.reconstruction))
    images = sorted(
        reconstruction.images.values(), key=lambda image: image.name
    )
    if len(images) != len(manifest_indices):
        raise ValueError(
            f"Reconstruction has {len(images)} images, expected "
            f"{len(manifest_indices)}"
        )

    predicted_centers = np.stack(
        [image.projection_center() for image in images]
    )
    arkit_poses = np.stack(
        [
            rotate_c2w_opencv_cw(
                np.asarray(manifest["frames"][index]["corrected_c2w_opencv"])
            )
            for index in manifest_indices
        ]
    )
    rotation, translation, scale = _similarity(
        predicted_centers, arkit_poses[:, :3, 3]
    )
    aligned_centers = scale * (predicted_centers @ rotation.T) + translation
    position_errors = np.linalg.norm(
        aligned_centers - arkit_poses[:, :3, 3], axis=1
    )
    position_residuals = aligned_centers - arkit_poses[:, :3, 3]
    rotation_errors = []
    aligned_poses = {}
    for image, manifest_index, center, arkit_pose in zip(
        images, manifest_indices, aligned_centers, arkit_poses, strict=True
    ):
        camera_to_world_rotation = image.cam_from_world().rotation.matrix().T
        aligned_rotation = rotation @ camera_to_world_rotation
        rotation_errors.append(
            np.degrees(
                np.linalg.norm(
                    Rotation.from_matrix(
                        aligned_rotation.T @ arkit_pose[:3, :3]
                    ).as_rotvec()
                )
            )
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = aligned_rotation
        pose[:3, 3] = center
        aligned_poses[str(manifest_index)] = pose.tolist()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "aligned_cameras.json").write_text(
        json.dumps(aligned_poses, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_directory / "arkit_disagreement.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "manifest_index",
                "image_name",
                "arkit_position_disagreement_m",
                "arkit_rotation_disagreement_deg",
                "disagreement_x_m",
                "disagreement_y_m",
                "disagreement_z_m",
                "arkit_x_m",
                "arkit_y_m",
                "arkit_z_m",
            ]
        )
        for (
            image,
            manifest_index,
            position_error,
            rotation_error,
            residual,
            pose,
        ) in zip(
            images,
            manifest_indices,
            position_errors,
            rotation_errors,
            position_residuals,
            arkit_poses,
            strict=True,
        ):
            writer.writerow(
                [
                    manifest_index,
                    image.name,
                    float(position_error),
                    float(rotation_error),
                    *[float(value) for value in residual],
                    *[float(value) for value in pose[:3, 3]],
                ]
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(
        manifest_indices, position_errors, color="black", linewidth=1.2
    )
    axes[0].plot(
        manifest_indices,
        position_residuals[:, 0],
        label="x",
        linewidth=0.8,
        alpha=0.8,
    )
    axes[0].plot(
        manifest_indices,
        position_residuals[:, 1],
        label="y",
        linewidth=0.8,
        alpha=0.8,
    )
    axes[0].plot(
        manifest_indices,
        position_residuals[:, 2],
        label="z",
        linewidth=0.8,
        alpha=0.8,
    )
    axes[0].set_ylabel("Position disagreement (m)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=4)
    axes[1].plot(manifest_indices, rotation_errors, color="tab:orange")
    axes[1].set_ylabel("Rotation disagreement (deg)")
    axes[1].set_xlabel("Manifest frame index")
    axes[1].grid(alpha=0.2)
    figure.suptitle(
        "Native GLUEMAP disagreement with ARKit after one diagnostic Sim3"
    )
    figure.tight_layout()
    figure.savefig(args.output_directory / "arkit_disagreement.png", dpi=180)
    plt.close(figure)
    points_output = None
    if reconstruction.num_points3D() > 0:
        points = np.stack(
            [point.xyz for point in reconstruction.points3D.values()]
        )
        colors = np.stack(
            [point.color for point in reconstruction.points3D.values()]
        )
        aligned_points = scale * (points @ rotation.T) + translation
        points_output = args.output_directory / "aligned_sparse_points.ply"
        writer = StreamingPlyWriter(points_output)
        writer.write(aligned_points.astype(np.float32), colors.astype(np.uint8))
        writer.close()

    summary = {
        "image_count": len(images),
        "point_count": reconstruction.num_points3D(),
        "similarity_scale": scale,
        "arkit_position_disagreement_m": {
            "median": float(np.median(position_errors)),
            "p95": float(np.percentile(position_errors, 95)),
            "maximum": float(np.max(position_errors)),
        },
        "arkit_rotation_disagreement_deg": {
            "median": float(np.median(rotation_errors)),
            "p95": float(np.percentile(rotation_errors, 95)),
            "maximum": float(np.max(rotation_errors)),
        },
        "aligned_points": None if points_output is None else str(points_output),
    }
    (args.output_directory / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
