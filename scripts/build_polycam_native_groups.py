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
            "poses": False,
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
