#!/usr/bin/env python3
"""Export GlueMap COLMAP results into the Polycam prior frame for inspection."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pycolmap


def frame_index(name: str) -> int | None:
    match = re.search(r"(\d+)", Path(name).name)
    return int(match.group(1)) if match else None


def load_prior_centers(path: Path) -> dict[int, np.ndarray]:
    data = json.loads(path.read_text())
    centers: dict[int, np.ndarray] = {}
    for frame in data["frames"]:
        idx = frame_index(frame["file_path"])
        if idx is None:
            continue
        transform = np.asarray(frame["transform_matrix"], dtype=np.float64)
        centers[idx] = transform[:3, 3]
    return centers


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Return s, R, t mapping src to dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 3:
        raise ValueError("Need at least 3 points for Sim3 alignment")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    x = src - mu_src
    y = dst - mu_dst
    cov = (y.T @ x) / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rot = u @ sign @ vt
    if with_scale:
        var = np.mean(np.sum(x * x, axis=1))
        scale = float(np.trace(np.diag(d) @ sign) / var)
    else:
        scale = 1.0
    trans = mu_dst - scale * rot @ mu_src
    return scale, rot, trans


def transform_points(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (scale * (rot @ points.T)).T + trans


def residual_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def write_vertex_ply(path: Path, vertices: list[tuple[float, float, float, int, int, int]]) -> None:
    with path.open("w") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(vertices)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for vertex in vertices:
            file.write("%.8f %.8f %.8f %d %d %d\n" % vertex)


def write_edge_ply(
    path: Path,
    vertices: list[tuple[float, float, float, int, int, int]],
    edges: list[tuple[int, int, int, int, int]],
) -> None:
    with path.open("w") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(vertices)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write(f"element edge {len(edges)}\n")
        file.write("property int vertex1\nproperty int vertex2\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for vertex in vertices:
            file.write("%.8f %.8f %.8f %d %d %d\n" % vertex)
        for edge in edges:
            file.write("%d %d %d %d %d\n" % edge)


def progress_color(position: int, count: int) -> tuple[int, int, int]:
    if count <= 1:
        return 255, 255, 255
    t = position / (count - 1)
    if t < 0.25:
        u = t / 0.25
        return int(40 * (1 - u)), int(120 + 135 * u), 255
    if t < 0.50:
        u = (t - 0.25) / 0.25
        return int(40 + 180 * u), 255, int(255 * (1 - u))
    if t < 0.75:
        u = (t - 0.50) / 0.25
        return 255, int(255 - 120 * u), 0
    u = (t - 0.75) / 0.25
    return 255, int(135 * (1 - u)), int(180 * u)


def camera_ticks(center: np.ndarray, cam_from_world: pycolmap.Rigid3d, length: float) -> tuple[np.ndarray, np.ndarray]:
    rot = cam_from_world.rotation.matrix()
    forward = rot.T @ np.array([0.0, 0.0, 1.0])
    up = rot.T @ np.array([0.0, -1.0, 0.0])
    return center + length * forward, center + length * up


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--prior-transforms", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default="gluemap")
    parser.add_argument("--tick-length", type=float, default=0.15)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(str(args.reconstruction))
    priors = load_prior_centers(args.prior_transforms)

    image_rows = []
    for image_id, image in rec.images.items():
        if not image.has_pose:
            continue
        idx = frame_index(image.name)
        if idx is None or idx not in priors:
            continue
        center = np.asarray(image.projection_center(), dtype=np.float64)
        image_rows.append((idx, image.name, image_id, center, image.cam_from_world()))

    image_rows.sort(key=lambda row: row[0])
    if len(image_rows) < 3:
        raise RuntimeError("Too few registered images matched to prior frames")

    glue_centers = np.asarray([row[3] for row in image_rows])
    prior_centers = np.asarray([priors[row[0]] for row in image_rows])

    scale, rot, trans = umeyama(glue_centers, prior_centers, with_scale=True)
    aligned_centers = transform_points(glue_centers, scale, rot, trans)
    residuals = np.linalg.norm(aligned_centers - prior_centers, axis=1)

    se3_scale, se3_rot, se3_trans = umeyama(glue_centers, prior_centers, with_scale=False)
    se3_centers = transform_points(glue_centers, se3_scale, se3_rot, se3_trans)
    se3_residuals = np.linalg.norm(se3_centers - prior_centers, axis=1)

    points = []
    for point in rec.points3D.values():
        color = np.asarray(point.color, dtype=np.uint8)
        points.append((np.asarray(point.xyz, dtype=np.float64), color))
    point_xyz = np.asarray([p[0] for p in points]) if points else np.empty((0, 3))
    point_rgb = np.asarray([p[1] for p in points], dtype=np.uint8) if points else np.empty((0, 3), dtype=np.uint8)
    aligned_points = transform_points(point_xyz, scale, rot, trans) if len(points) else point_xyz

    native_point_vertices = [
        (float(x), float(y), float(z), int(r), int(g), int(b))
        for (x, y, z), (r, g, b) in zip(point_xyz, point_rgb)
    ]
    aligned_point_vertices = [
        (float(x), float(y), float(z), int(r), int(g), int(b))
        for (x, y, z), (r, g, b) in zip(aligned_points, point_rgb)
    ]

    write_vertex_ply(args.out_dir / f"{args.name}_sparse_points_native.ply", native_point_vertices)
    write_vertex_ply(args.out_dir / f"{args.name}_sparse_points_v31d_frame.ply", aligned_point_vertices)

    native_camera_vertices = []
    aligned_camera_vertices = []
    path_vertices = []
    path_edges = []
    for pos, (idx, _name, _image_id, center, cam_from_world) in enumerate(image_rows):
        color = progress_color(pos, len(image_rows))
        fwd, up = camera_ticks(center, cam_from_world, args.tick_length / max(scale, 1e-8))
        aligned = transform_points(np.vstack([center, fwd, up]), scale, rot, trans)
        native_camera_vertices.extend(
            [
                (*center, *color),
                (*fwd, 255, 255, 255),
                (*up, 80, 220, 255),
            ]
        )
        aligned_camera_vertices.extend(
            [
                (*aligned[0], *color),
                (*aligned[1], 255, 255, 255),
                (*aligned[2], 80, 220, 255),
            ]
        )
        path_vertices.append((*aligned[0], *color))
        if pos:
            path_edges.append((pos - 1, pos, 210, 210, 210))

    write_vertex_ply(args.out_dir / f"{args.name}_camera_ticks_native.ply", native_camera_vertices)
    write_vertex_ply(args.out_dir / f"{args.name}_camera_ticks_v31d_frame.ply", aligned_camera_vertices)
    write_edge_ply(args.out_dir / f"{args.name}_sampled_camera_path_v31d_frame.ply", path_vertices, path_edges)

    combined_vertices = aligned_point_vertices + aligned_camera_vertices
    write_vertex_ply(args.out_dir / f"{args.name}_points_and_camera_ticks_v31d_frame.ply", combined_vertices)

    overlay_vertices = []
    overlay_edges = []
    for prior in prior_centers:
        overlay_vertices.append((*prior, 170, 170, 170))
    base = len(overlay_vertices)
    for pos, center in enumerate(aligned_centers):
        rr = min(1.0, residuals[pos] / max(float(np.quantile(residuals, 0.95)), 1e-6))
        overlay_vertices.append((*center, int(255 * rr), int(255 * (1.0 - rr)), 0))
    for pos in range(len(image_rows) - 1):
        overlay_edges.append((pos, pos + 1, 120, 120, 120))
        overlay_edges.append((base + pos, base + pos + 1, 255, 100, 0))
    p90 = float(np.quantile(residuals, 0.90))
    for pos, residual in enumerate(residuals):
        if residual > p90:
            overlay_edges.append((pos, base + pos, 255, 0, 0))
        else:
            overlay_edges.append((pos, base + pos, 80, 80, 255))
    write_edge_ply(args.out_dir / f"{args.name}_vs_v31d_prior_overlay_sim3fit.ply", overlay_vertices, overlay_edges)

    csv_path = args.out_dir / f"{args.name}_vs_v31d_prior_residuals.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "image", "prior_x", "prior_y", "prior_z", "glue_x", "glue_y", "glue_z", "sim3_residual_m", "se3_residual_m"])
        for row, prior, glue, sim3_res, se3_res in zip(image_rows, prior_centers, aligned_centers, residuals, se3_residuals):
            writer.writerow([row[0], row[1], *prior.tolist(), *glue.tolist(), sim3_res, se3_res])

    worst = sorted(
        [(row[1], int(row[0]), float(res)) for row, res in zip(image_rows, residuals)],
        key=lambda item: item[2],
        reverse=True,
    )[:20]
    summary = {
        "reconstruction": str(args.reconstruction),
        "prior_transforms": str(args.prior_transforms),
        "registered_images": len(rec.images),
        "matched_prior_keyframes": len(image_rows),
        "points3D": len(points),
        "sim3_scale_gluemap_to_v31d": float(scale),
        "sim3_residual_m": residual_stats(residuals),
        "se3_residual_m": residual_stats(se3_residuals),
        "worst_sim3_frames": worst,
        "outputs": {
            "sparse_points_v31d_frame": f"{args.name}_sparse_points_v31d_frame.ply",
            "camera_ticks_v31d_frame": f"{args.name}_camera_ticks_v31d_frame.ply",
            "sampled_camera_path_v31d_frame": f"{args.name}_sampled_camera_path_v31d_frame.ply",
            "points_and_camera_ticks_v31d_frame": f"{args.name}_points_and_camera_ticks_v31d_frame.ply",
            "vs_v31d_prior_overlay": f"{args.name}_vs_v31d_prior_overlay_sim3fit.ply",
            "residuals_csv": csv_path.name,
        },
    }
    (args.out_dir / f"{args.name}_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = [
        f"Reconstruction: {args.reconstruction}",
        f"Registered images: {len(rec.images)}",
        f"Matched v31d keyframes: {len(image_rows)}",
        f"Points3D: {len(points)}",
        f"Sim3 scale GlueMap->v31d: {scale:.8f}",
        f"Sim3 residual meters: {json.dumps(summary['sim3_residual_m'])}",
        f"SE3 residual meters: {json.dumps(summary['se3_residual_m'])}",
        "Worst Sim3 frames:",
    ]
    text.extend([f"  {name}: {res:.4f} m" for name, _idx, res in worst])
    (args.out_dir / f"{args.name}_analysis_summary.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
