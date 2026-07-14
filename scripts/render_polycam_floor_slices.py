#!/usr/bin/env python3
"""Render plan-view floor slices and an elevation from a binary RGB PLY."""

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


def _read_binary_rgb_ply(path: Path, maximum_points: int) -> np.ndarray:
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        offset = stream.tell()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("Only binary little-endian PLY files are supported")
    vertex_line = next(
        line for line in header if line.startswith("element vertex ")
    )
    vertex_count = int(vertex_line.split()[-1])
    points = np.memmap(
        path, mode="r", dtype=POINT_DTYPE, offset=offset, shape=(vertex_count,)
    )
    stride = max(1, int(np.ceil(vertex_count / maximum_points)))
    return np.asarray(points[::stride])


def _load_camera_centers(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    poses = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray([np.asarray(pose)[:3, 3] for pose in poses.values()])


def _scatter(
    ax, points: np.ndarray, mask: np.ndarray, horizontal: str, vertical: str
) -> None:
    selected = points[mask]
    colors = (
        np.column_stack([selected["red"], selected["green"], selected["blue"]])
        / 255.0
    )
    colors = np.clip(0.22 + 0.78 * colors, 0.0, 1.0)
    ax.scatter(
        selected[horizontal],
        selected[vertical],
        c=colors,
        s=0.12,
        alpha=0.42,
        linewidths=0,
        rasterized=True,
    )


def _robust_limits(
    values: np.ndarray,
    extra_values: np.ndarray | None = None,
) -> tuple[float, float] | None:
    values = values[np.isfinite(values)]
    if extra_values is not None:
        extra_values = extra_values[np.isfinite(extra_values)]
    if values.size == 0 and (extra_values is None or extra_values.size == 0):
        return None
    if values.size:
        lower, upper = np.percentile(values, [0.5, 99.5])
    else:
        lower, upper = float(extra_values.min()), float(extra_values.max())
    if extra_values is not None and extra_values.size:
        lower = min(lower, float(extra_values.min()))
        upper = max(upper, float(extra_values.max()))
    span = max(float(upper - lower), 0.5)
    margin = 0.06 * span
    return float(lower - margin), float(upper + margin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point_cloud", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cameras", type=Path)
    parser.add_argument("--lower-boundary", type=float, default=-1.4)
    parser.add_argument("--upper-boundary", type=float, default=2.5)
    parser.add_argument("--minimum-y", type=float, default=-4.8)
    parser.add_argument("--maximum-y", type=float, default=5.2)
    parser.add_argument("--maximum-points", type=int, default=2_500_000)
    parser.add_argument(
        "--viewpoint",
        choices=["top-down", "bottom-up"],
        default="top-down",
        help="Plan-view handedness; top-down mirrors Z relative to below.",
    )
    parser.add_argument(
        "--title",
        default="Telluride full-house fusion — floor-separated preview",
    )
    args = parser.parse_args()

    points = _read_binary_rgb_ply(args.point_cloud, args.maximum_points)
    finite = (
        np.isfinite(points["x"])
        & np.isfinite(points["y"])
        & np.isfinite(points["z"])
    )
    cameras = _load_camera_centers(args.cameras)
    slices = [
        ("Top floor", args.upper_boundary, args.maximum_y),
        ("Main floor", args.lower_boundary, args.upper_boundary),
        ("Basement", args.minimum_y, args.lower_boundary),
    ]

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(20, 18), constrained_layout=True)
    figure.patch.set_facecolor("#05090d")
    figure.suptitle(args.title, fontsize=22)
    for ax, (label, lower, upper) in zip(axes.flat[:3], slices, strict=True):
        mask = finite & (points["y"] >= lower) & (points["y"] < upper)
        _scatter(ax, points, mask, "x", "z")
        if cameras is not None:
            camera_mask = (cameras[:, 1] >= lower) & (cameras[:, 1] < upper)
            ax.scatter(
                cameras[camera_mask, 0],
                cameras[camera_mask, 2],
                s=2,
                c="#28d7e5",
                alpha=0.45,
            )
        else:
            camera_mask = None
        selected = points[mask]
        selected_cameras = cameras[camera_mask] if cameras is not None else None
        x_limits = _robust_limits(
            selected["x"],
            None if selected_cameras is None else selected_cameras[:, 0],
        )
        z_limits = _robust_limits(
            selected["z"],
            None if selected_cameras is None else selected_cameras[:, 2],
        )
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if z_limits is not None:
            ax.set_ylim(*z_limits)
        ax.set_title(f"{label}  ({lower:g} ≤ Y < {upper:g} m)", fontsize=16)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        if args.viewpoint == "top-down":
            ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")

    elevation = axes.flat[3]
    _scatter(elevation, points, finite, "x", "y")
    elevation.axhline(args.lower_boundary, color="#28d7e5", linewidth=1)
    elevation.axhline(args.upper_boundary, color="#28d7e5", linewidth=1)
    if cameras is not None:
        elevation.scatter(
            cameras[:, 0], cameras[:, 1], s=2, c="#28d7e5", alpha=0.45
        )
    elevation_x_limits = _robust_limits(
        points["x"][finite], None if cameras is None else cameras[:, 0]
    )
    elevation_y_limits = _robust_limits(
        points["y"][finite], None if cameras is None else cameras[:, 1]
    )
    if elevation_x_limits is not None:
        elevation.set_xlim(*elevation_x_limits)
    if elevation_y_limits is not None:
        elevation.set_ylim(*elevation_y_limits)
    elevation.set_title("Elevation and slice boundaries", fontsize=16)
    elevation.set_xlabel("X (m)")
    elevation.set_ylabel("Y (m)")
    for ax in axes.flat:
        ax.set_facecolor("#05090d")
        ax.tick_params(colors="#9fb0bd")
        for spine in ax.spines.values():
            spine.set_color("#344653")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170, facecolor=figure.get_facecolor())
    print(
        json.dumps({"output": str(args.output), "sampled_points": len(points)})
    )


if __name__ == "__main__":
    main()
