#!/usr/bin/env python3
"""Build a sparse overlapping MapAnything cover from audited frontend edges."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from gluemap.pairing.pose_depth_graph import build_graph_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("index_mapping", type=Path)
    parser.add_argument("frontend_edges", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--minimum-memberships", type=int, default=2)
    parser.add_argument(
        "--max-nontemporal-height-difference-m", type=float, default=1.25
    )
    parser.add_argument(
        "--sparse-pose-conditioning",
        action="store_true",
        help=(
            "Condition only a short temporal neighborhood around each group "
            "anchor; nonlocal revisit members remain free to correct drift."
        ),
    )
    parser.add_argument("--pose-anchor-radius", type=int, default=12)
    parser.add_argument("--max-pose-views", type=int, default=8)
    parser.add_argument(
        "--pose-exclusion",
        action="append",
        default=[],
        metavar="START:END",
        help="Inclusive manifest range whose group anchors remain pose-free.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    heights = {
        index: float(frame["corrected_c2w_opencv"][1][3])
        for index, frame in enumerate(manifest["frames"])
    }
    mapping = json.loads(args.index_mapping.read_text(encoding="utf-8"))
    manifest_indices = [int(row["manifest_index"]) for row in mapping]
    included = set(manifest_indices)
    edges = []
    rejected_cross_floor = 0
    with args.frontend_edges.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            first, second = int(row["first"]), int(row["second"])
            if first not in included or second not in included:
                continue
            reason = row["acceptance_reason"]
            if (
                reason != "temporal"
                and abs(heights[first] - heights[second])
                > args.max_nontemporal_height_difference_m
            ):
                rejected_cross_floor += 1
                continue
            edges.append(
                {
                    "first": first,
                    "second": second,
                    "kind": reason,
                    "score": float(row["score"]),
                }
            )

    groups = build_graph_groups(
        manifest_indices,
        edges,
        group_size=args.group_size,
        minimum_memberships=args.minimum_memberships,
    )
    exclusions = []
    for value in args.pose_exclusion:
        start, end = map(int, value.split(":"))
        exclusions.append((min(start, end), max(start, end)))
    if args.sparse_pose_conditioning:
        for group in groups:
            anchor = int(group["anchor_frame"])
            if any(start <= anchor <= end for start, end in exclusions):
                group["pose_conditioned_frames"] = []
                continue
            local_members = sorted(
                int(frame)
                for frame in group["frame_indices"]
                if abs(int(frame) - anchor) <= args.pose_anchor_radius
            )
            if anchor not in local_members:
                local_members.insert(0, anchor)
            if len(local_members) > args.max_pose_views:
                positions = (
                    np.linspace(0, len(local_members) - 1, args.max_pose_views)
                    .round()
                    .astype(int)
                )
                local_members = [
                    local_members[position] for position in positions
                ]
                if anchor not in local_members:
                    local_members[0] = anchor
            group["pose_conditioned_frames"] = list(
                dict.fromkeys([anchor, *local_members])
            )
    coverage = Counter()
    for group in groups:
        coverage.update(map(int, group["frame_indices"]))
    counts = np.asarray([coverage[index] for index in manifest_indices])
    result = {
        "schema_version": 1,
        "backend": "map_anything",
        "conditioning": {
            "rgb": True,
            "intrinsics": True,
            "metric_depth": True,
            "poses": "sparse_local" if args.sparse_pose_conditioning else False,
        },
        "construction": {
            "source": "audited_frontend_graph_cover",
            "group_size": args.group_size,
            "minimum_memberships": args.minimum_memberships,
            "max_nontemporal_height_difference_m": (
                args.max_nontemporal_height_difference_m
            ),
            "rejected_cross_floor_edges": rejected_cross_floor,
            "frontend_edges": str(args.frontend_edges),
            "pose_anchor_radius": args.pose_anchor_radius,
            "max_pose_views": args.max_pose_views,
            "pose_exclusions": exclusions,
        },
        "coverage": {
            "frames": len(manifest_indices),
            "groups": len(groups),
            "model_view_evaluations": int(
                sum(len(group["frame_indices"]) for group in groups)
            ),
            "minimum": int(counts.min()),
            "median": float(np.median(counts)),
            "maximum": int(counts.max()),
        },
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["coverage"], indent=2))


if __name__ == "__main__":
    main()
