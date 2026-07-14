#!/usr/bin/env python3
"""Render dense plan and section diagnostics from a binary RGB PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def _read_ply(path: Path, maximum_points: int) -> np.ndarray:
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        offset = stream.tell()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("Only binary little-endian PLY is supported")
    vertex_line = next(
        line for line in header if line.startswith("element vertex ")
    )
    vertex_count = int(vertex_line.split()[-1])
    points = np.memmap(
        path,
        mode="r",
        dtype=POINT_DTYPE,
        offset=offset,
        shape=(vertex_count,),
    )
    stride = max(1, int(np.ceil(vertex_count / maximum_points)))
    return np.asarray(points[::stride])


def _load_cameras(path: Path) -> np.ndarray:
    poses = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray([np.asarray(pose)[:3, 3] for pose in poses.values()])


def _rgb(points: np.ndarray) -> np.ndarray:
    colors = np.column_stack(
        [points["red"], points["green"], points["blue"]]
    ).astype(np.float32)
    colors /= 255.0
    return np.clip(0.2 + 0.8 * colors, 0.0, 1.0)


def _limits(values: np.ndarray, cameras: np.ndarray | None = None):
    finite = values[np.isfinite(values)]
    lower, upper = np.percentile(finite, [0.25, 99.75])
    if cameras is not None and len(cameras):
        lower = min(lower, float(np.min(cameras)))
        upper = max(upper, float(np.max(cameras)))
    span = max(float(upper - lower), 1.0)
    margin = span * 0.04
    return float(lower - margin), float(upper + margin)


def _style_axis(axis) -> None:
    axis.set_facecolor("#03070b")
    axis.tick_params(colors="#a8bac7")
    for spine in axis.spines.values():
        spine.set_color("#304552")


def _scatter_rgb(axis, points: np.ndarray, first: str, second: str) -> None:
    axis.scatter(
        points[first],
        points[second],
        c=_rgb(points),
        s=0.10,
        alpha=0.38,
        linewidths=0,
        rasterized=True,
    )


def _save_whole_house(
    points: np.ndarray, cameras: np.ndarray, output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(16, 16), constrained_layout=True)
    figure.patch.set_facecolor("#03070b")
    height = points["y"]
    lower, upper = np.percentile(height, [1, 99])
    normalized = np.clip((height - lower) / max(upper - lower, 1e-6), 0, 1)
    colors = plt.get_cmap("turbo")(normalized)
    colors[:, 3] = 0.34
    axis.scatter(
        points["x"],
        points["z"],
        c=colors,
        s=0.10,
        linewidths=0,
        rasterized=True,
    )
    axis.plot(
        cameras[:, 0],
        cameras[:, 2],
        color="#38e8f4",
        linewidth=0.35,
        alpha=0.28,
    )
    axis.scatter(cameras[:, 0], cameras[:, 2], c="#38e8f4", s=1.0, alpha=0.45)
    axis.set_xlim(*_limits(points["x"], cameras[:, 0]))
    axis.set_ylim(*_limits(points["z"], cameras[:, 2]))
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Z (m), top-down")
    axis.set_title("Solved cameras + measured LiDAR — height-colored x-ray")
    _style_axis(axis)
    figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _save_floor_slices(
    points: np.ndarray,
    cameras: np.ndarray,
    output: Path,
    lower_boundary: float,
    upper_boundary: float,
) -> dict[str, int]:
    slices = [
        ("Top floor", upper_boundary, 5.2),
        ("Main floor", lower_boundary, upper_boundary),
        ("Basement", -4.8, lower_boundary),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(24, 9), constrained_layout=True)
    figure.patch.set_facecolor("#03070b")
    counts = {}
    for axis, (label, lower, upper) in zip(axes, slices, strict=True):
        mask = (points["y"] >= lower) & (points["y"] < upper)
        selected = points[mask]
        camera_mask = (cameras[:, 1] >= lower) & (cameras[:, 1] < upper)
        selected_cameras = cameras[camera_mask]
        camera_path = cameras[:, [0, 2]].copy()
        camera_path[~camera_mask] = np.nan
        counts[label] = int(mask.sum())
        _scatter_rgb(axis, selected, "x", "z")
        axis.plot(
            camera_path[:, 0],
            camera_path[:, 1],
            color="#38e8f4",
            linewidth=0.4,
            alpha=0.35,
        )
        axis.scatter(
            selected_cameras[:, 0],
            selected_cameras[:, 2],
            c="#38e8f4",
            s=1.2,
            alpha=0.5,
        )
        axis.set_xlim(*_limits(selected["x"], selected_cameras[:, 0]))
        axis.set_ylim(*_limits(selected["z"], selected_cameras[:, 2]))
        axis.invert_yaxis()
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{label}  ({lower:g} ≤ Y < {upper:g} m)")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Z (m), top-down")
        _style_axis(axis)
    figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    return counts


def _save_sections(
    points: np.ndarray,
    cameras: np.ndarray,
    output: Path,
    lower_boundary: float,
    upper_boundary: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(22, 10), constrained_layout=True)
    figure.patch.set_facecolor("#03070b")
    for axis, horizontal, label in [
        (axes[0], "x", "South/north-facing elevation"),
        (axes[1], "z", "East/west-facing elevation"),
    ]:
        _scatter_rgb(axis, points, horizontal, "y")
        camera_horizontal = (
            cameras[:, 0] if horizontal == "x" else cameras[:, 2]
        )
        axis.scatter(
            camera_horizontal, cameras[:, 1], c="#38e8f4", s=1.0, alpha=0.4
        )
        axis.axhline(lower_boundary, color="#38e8f4", linewidth=0.8)
        axis.axhline(upper_boundary, color="#38e8f4", linewidth=0.8)
        axis.set_xlim(*_limits(points[horizontal], camera_horizontal))
        axis.set_ylim(*_limits(points["y"], cameras[:, 1]))
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(label)
        axis.set_xlabel(f"{horizontal.upper()} (m)")
        axis.set_ylabel("Y / elevation (m)")
        _style_axis(axis)
    figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point_cloud", type=Path)
    parser.add_argument("cameras", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--maximum-points", type=int, default=3_000_000)
    parser.add_argument("--lower-boundary", type=float, default=-1.4)
    parser.add_argument("--upper-boundary", type=float, default=2.5)
    args = parser.parse_args()

    points = _read_ply(args.point_cloud, args.maximum_points)
    cameras = _load_cameras(args.cameras)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    _save_whole_house(
        points, cameras, args.output_directory / "dense_whole_house_topdown.png"
    )
    counts = _save_floor_slices(
        points,
        cameras,
        args.output_directory / "dense_floor_slices_topdown.png",
        args.lower_boundary,
        args.upper_boundary,
    )
    _save_sections(
        points,
        cameras,
        args.output_directory / "dense_orthographic_sections.png",
        args.lower_boundary,
        args.upper_boundary,
    )
    print(
        json.dumps(
            {
                "sampled_points": len(points),
                "floor_sample_counts": counts,
                "output_directory": str(args.output_directory),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
