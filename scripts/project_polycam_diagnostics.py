#!/usr/bin/env python3
"""Project audited Polycam RGB-D into reset-segment clouds and overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gluemap.datasets.polycam import frame_world_points, load_polycam_manifest

_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


class StreamingPlyWriter:
    """Write a binary PLY without retaining all vertices in memory."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("wb+")
        prefix = b"element vertex "
        header_before_count = b"ply\nformat binary_little_endian 1.0\n" + prefix
        self.handle.write(header_before_count)
        self.count_offset = len(header_before_count)
        self.handle.write(b"0" * 20)
        self.handle.write(
            b"\nproperty float x\nproperty float y\nproperty float z\n"
            b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            b"end_header\n"
        )
        self.count = 0

    def write(self, points: np.ndarray, colors: np.ndarray) -> None:
        vertices = np.empty(len(points), dtype=_VERTEX_DTYPE)
        vertices["x"] = points[:, 0]
        vertices["y"] = points[:, 1]
        vertices["z"] = points[:, 2]
        vertices["red"] = colors[:, 0]
        vertices["green"] = colors[:, 1]
        vertices["blue"] = colors[:, 2]
        vertices.tofile(self.handle)
        self.count += len(vertices)

    def close(self) -> None:
        self.handle.seek(self.count_offset)
        self.handle.write(f"{self.count:020d}".encode("ascii"))
        self.handle.close()


def _depth_colors(depth_m: np.ndarray, maximum_m: float) -> np.ndarray:
    normalized = np.clip(depth_m / maximum_m, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def write_overlay(
    frame: dict,
    source_root: Path,
    output_path: Path,
    *,
    min_confidence: int,
    max_depth_m: float,
) -> None:
    paths = frame["paths"]
    with Image.open(source_root / paths["rgb"]) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(source_root / paths["depth"]) as image:
        depth_image = image.resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
        )
        depth = np.asarray(depth_image, dtype=np.float32) * 0.001
    with Image.open(source_root / paths["confidence"]) as image:
        confidence_image = image.resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
        )
        confidence = np.asarray(confidence_image, dtype=np.uint8)
    valid = (
        (confidence >= min_confidence) & (depth > 0.08) & (depth <= max_depth_m)
    )
    colors = _depth_colors(depth, max_depth_m)
    overlay = rgb.copy()
    overlay[valid] = (
        0.45 * rgb[valid].astype(np.float32)
        + 0.55 * colors[valid].astype(np.float32)
    ).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path, quality=92)


def _overlay_indices(
    frame_count: int, reset_after: list[int], count: int
) -> list[int]:
    indices = set(np.linspace(0, frame_count - 1, count, dtype=int).tolist())
    for index in reset_after:
        indices.update(range(max(0, index - 2), min(frame_count, index + 4)))
    return sorted(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--pixel-step", type=int, default=8)
    parser.add_argument("--min-confidence", type=int, default=255)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    parser.add_argument("--overlay-count", type=int, default=24)
    parser.add_argument(
        "--pose-prefix",
        choices=["raw", "corrected"],
        default="raw",
        help="Manifest pose fields used for projection (default: raw).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Output filename prefix (default: the pose prefix).",
    )
    args = parser.parse_args()

    manifest = load_polycam_manifest(args.manifest)
    source_root = Path(manifest["source_root"])
    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    label = args.label or args.pose_prefix
    pose_key_opencv = f"{args.pose_prefix}_c2w_opencv"
    pose_key_arkit = f"{args.pose_prefix}_c2w_arkit"
    segments = sorted({frame["reset_segment"] for frame in manifest["frames"]})
    writers = {
        segment: StreamingPlyWriter(output / f"{label}_segment_{segment}.ply")
        for segment in segments
    }
    combined_writer = StreamingPlyWriter(
        output / f"{label}_combined_segments.ply"
    )
    frames_with_points = {segment: 0 for segment in segments}
    try:
        for frame in manifest["frames"]:
            points, colors = frame_world_points(
                frame,
                source_root,
                pixel_step=args.pixel_step,
                min_confidence=args.min_confidence,
                max_depth_m=args.max_depth_m,
                pose_key=pose_key_opencv,
            )
            if len(points):
                segment = frame["reset_segment"]
                writers[segment].write(points, colors)
                segment_color = (
                    np.array([40, 210, 255], dtype=np.uint8)
                    if segment == 0
                    else np.array([255, 60, 190], dtype=np.uint8)
                )
                combined_writer.write(
                    points,
                    np.broadcast_to(segment_color, colors.shape),
                )
                frames_with_points[segment] += 1
    finally:
        for writer in writers.values():
            writer.close()
        combined_writer.close()

    centers = np.asarray(
        [frame[pose_key_arkit] for frame in manifest["frames"]],
        dtype=np.float64,
    )[:, :3, 3]
    segment_colors = np.asarray(
        [
            (40, 210, 255) if frame["reset_segment"] == 0 else (255, 60, 190)
            for frame in manifest["frames"]
        ],
        dtype=np.uint8,
    )
    centers_writer = StreamingPlyWriter(output / f"{label}_camera_centers.ply")
    centers_writer.write(centers.astype(np.float32), segment_colors)
    centers_writer.close()

    overlay_indices = _overlay_indices(
        manifest["frame_count"],
        manifest["reset_after_sequence_indices"],
        args.overlay_count,
    )
    for index in overlay_indices:
        frame = manifest["frames"][index]
        write_overlay(
            frame,
            source_root,
            output / "overlays" / f"frame_{index:06d}.jpg",
            min_confidence=args.min_confidence,
            max_depth_m=args.max_depth_m,
        )

    summary = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "pixel_step": args.pixel_step,
        "min_confidence": args.min_confidence,
        "max_depth_m": args.max_depth_m,
        "pose_prefix": args.pose_prefix,
        "segments": {
            str(segment): {
                "frames_with_points": frames_with_points[segment],
                "vertices": writers[segment].count,
                "ply": str(writers[segment].path),
            }
            for segment in segments
        },
        "combined_segments_ply": str(combined_writer.path),
        "camera_centers_ply": str(centers_writer.path),
        "overlay_indices": overlay_indices,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
