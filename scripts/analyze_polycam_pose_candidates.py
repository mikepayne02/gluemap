#!/usr/bin/env python3
"""Find nonlocal pose-proximity candidates without declaring loop closures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from gluemap.datasets.polycam import load_polycam_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--pose-prefix", default="corrected")
    parser.add_argument("--max-distance-m", type=float, default=2.5)
    parser.add_argument("--max-vertical-distance-m", type=float, default=1.5)
    parser.add_argument("--min-sequence-separation", type=int, default=150)
    parser.add_argument("--bin-size", type=int, default=32)
    parser.add_argument("--max-pairs-per-frame", type=int, default=8)
    args = parser.parse_args()

    manifest = load_polycam_manifest(args.manifest)
    frames = manifest["frames"]
    pose_key = f"{args.pose_prefix}_c2w_opencv"
    poses = np.asarray([frame[pose_key] for frame in frames], dtype=np.float64)
    centers = poses[:, :3, 3]
    forwards = poses[:, :3, 2]
    tree = cKDTree(centers)

    candidates = []
    for index, center in enumerate(centers):
        neighbors = tree.query_ball_point(center, args.max_distance_m)
        local = []
        for other in neighbors:
            if other <= index:
                continue
            if other - index < args.min_sequence_separation:
                continue
            delta = centers[other] - center
            vertical_distance = abs(float(delta[1]))
            if vertical_distance > args.max_vertical_distance_m:
                continue
            distance = float(np.linalg.norm(delta))
            local.append(
                {
                    "frame_a": index,
                    "frame_b": other,
                    "segment_a": int(frames[index]["reset_segment"]),
                    "segment_b": int(frames[other]["reset_segment"]),
                    "center_distance_m": distance,
                    "vertical_distance_m": vertical_distance,
                    "forward_dot": float(
                        np.dot(forwards[index], forwards[other])
                    ),
                }
            )
        local.sort(key=lambda row: row["center_distance_m"])
        candidates.extend(local[: args.max_pairs_per_frame])

    candidates.sort(
        key=lambda row: (
            row["center_distance_m"],
            -abs(row["forward_dot"]),
            row["frame_a"],
            row["frame_b"],
        )
    )
    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "pose_proximity_candidates.csv"
    fieldnames = list(candidates[0]) if candidates else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(candidates)

    bins: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in candidates:
        key = (
            row["frame_a"] // args.bin_size,
            row["frame_b"] // args.bin_size,
        )
        bins[key].append(row)
    bands = []
    for (bin_a, bin_b), rows in bins.items():
        distances = np.asarray([row["center_distance_m"] for row in rows])
        view_dots = np.asarray([row["forward_dot"] for row in rows])
        bands.append(
            {
                "frame_range_a": [
                    bin_a * args.bin_size,
                    min(len(frames) - 1, (bin_a + 1) * args.bin_size - 1),
                ],
                "frame_range_b": [
                    bin_b * args.bin_size,
                    min(len(frames) - 1, (bin_b + 1) * args.bin_size - 1),
                ],
                "pair_count": len(rows),
                "min_center_distance_m": float(np.min(distances)),
                "median_center_distance_m": float(np.median(distances)),
                "median_forward_dot": float(np.median(view_dots)),
                "same_reset_segment": bool(
                    rows[0]["segment_a"] == rows[0]["segment_b"]
                ),
            }
        )
    bands.sort(
        key=lambda row: (
            -row["pair_count"],
            row["median_center_distance_m"],
        )
    )
    summary = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "pose_prefix": args.pose_prefix,
        "parameters": {
            "max_distance_m": args.max_distance_m,
            "max_vertical_distance_m": args.max_vertical_distance_m,
            "min_sequence_separation": args.min_sequence_separation,
            "bin_size": args.bin_size,
            "max_pairs_per_frame": args.max_pairs_per_frame,
        },
        "candidate_count": len(candidates),
        "candidate_csv": str(csv_path),
        "top_frame_bands": bands[:100],
        "warning": (
            "Pose proximity is proposal-only. Repeated stairs and rooms "
            "require appearance and depth/MapAnything verification."
        ),
    }
    (output / "pose_candidate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
