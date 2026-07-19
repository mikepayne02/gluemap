#!/usr/bin/env python3
"""Build a sparse overlapping MapAnything cover from audited frontend edges."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from gluemap.pairing.pose_depth_graph import (
    apply_group_recovery_policy,
    build_graph_groups,
    build_verified_bridge_groups,
    exclude_group_evidence_pairs,
    frontend_edge_is_group_evidence,
    select_sparse_pose_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("index_mapping", type=Path)
    parser.add_argument("frontend_edges", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--minimum-memberships", type=int, default=2)
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
    parser.add_argument(
        "--recovery-policy",
        type=Path,
        help=(
            "JSON policy for bounded larger groups around audited weak "
            "transition candidates."
        ),
    )
    args = parser.parse_args()

    mapping = json.loads(args.index_mapping.read_text(encoding="utf-8"))
    manifest_indices = [int(row["manifest_index"]) for row in mapping]
    included = set(manifest_indices)
    recovery_policy = None
    excluded_evidence_pairs: set[tuple[int, int]] = set()
    if args.recovery_policy is not None:
        recovery_policy = json.loads(
            args.recovery_policy.read_text(encoding="utf-8")
        )
        excluded_evidence_pairs = {
            tuple(sorted(map(int, pair)))
            for pair in recovery_policy.get("excluded_evidence_pairs", [])
        }
    edges = []
    verified_edges = []
    rejected_without_image_evidence = 0
    with args.frontend_edges.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            first, second = int(row["first"]), int(row["second"])
            if first not in included or second not in included:
                continue
            if not frontend_edge_is_group_evidence(row):
                rejected_without_image_evidence += 1
                continue
            reason = row["acceptance_reason"]
            edges.append(
                {
                    "first": first,
                    "second": second,
                    "kind": reason,
                    "score": float(row["score"]),
                }
            )
            if reason == "manually_verified":
                verified_edges.append(row)

    edges, rejected_by_policy = exclude_group_evidence_pairs(
        edges, excluded_evidence_pairs
    )
    verified_edges, rejected_verified_by_policy = exclude_group_evidence_pairs(
        verified_edges, excluded_evidence_pairs
    )
    if rejected_verified_by_policy > rejected_by_policy:
        raise RuntimeError("Verified-evidence exclusion accounting is invalid")

    bridge_groups = build_verified_bridge_groups(
        manifest_indices, verified_edges, group_size=args.group_size
    )

    groups = build_graph_groups(
        manifest_indices,
        edges,
        group_size=args.group_size,
        minimum_memberships=args.minimum_memberships,
        seeded_groups=bridge_groups,
    )
    recovery_groups = []
    if recovery_policy is not None:
        recovery_groups = apply_group_recovery_policy(
            groups, recovery_policy, included
        )
    recovery_names = {group["name"] for group in recovery_groups}
    exclusions = []
    for value in args.pose_exclusion:
        start, end = map(int, value.split(":"))
        exclusions.append((min(start, end), max(start, end)))
    if args.sparse_pose_conditioning:
        for group in groups:
            if group["name"] in recovery_names:
                continue
            anchor = int(group["anchor_frame"])
            if any(start <= anchor <= end for start, end in exclusions):
                group["pose_conditioned_frames"] = []
                continue
            group["pose_conditioned_frames"] = select_sparse_pose_frames(
                list(map(int, group["frame_indices"])),
                anchor,
                radius=args.pose_anchor_radius,
                maximum=args.max_pose_views,
            )
    coverage = Counter()
    for group in groups:
        coverage.update(map(int, group["frame_indices"]))
    counts = np.asarray([coverage[index] for index in manifest_indices])
    pose_conditioning_modes = sorted(
        {
            mode
            for group in groups
            if group.get("pose_conditioned_frames")
            for mode in group.get("pose_modes", ["sparse_local"])
            if mode != "none"
        }
    )
    result = {
        "schema_version": 1,
        "backend": "map_anything",
        "conditioning": {
            "rgb": True,
            "intrinsics": True,
            "metric_depth": True,
            "poses": pose_conditioning_modes or False,
        },
        "construction": {
            "source": "audited_frontend_graph_cover",
            "group_size": args.group_size,
            "minimum_memberships": args.minimum_memberships,
            "non_temporal_policy": "image_verified_or_manual",
            "rejected_without_image_evidence": (
                rejected_without_image_evidence
            ),
            "excluded_evidence_pairs": [
                list(pair) for pair in sorted(excluded_evidence_pairs)
            ],
            "rejected_by_policy": rejected_by_policy,
            "verified_bridge_groups": len(bridge_groups),
            "frontend_edges": str(args.frontend_edges),
            "pose_anchor_radius": args.pose_anchor_radius,
            "max_pose_views": args.max_pose_views,
            "pose_exclusions": exclusions,
            "recovery_policy": None
            if args.recovery_policy is None
            else str(args.recovery_policy),
            "recovery_groups": recovery_groups,
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
