#!/usr/bin/env python3
"""Freeze a conservative Polycam graph for native GLUEMAP star inference."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edge_report", type=Path)
    parser.add_argument("visual_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument(
        "--exclude-frame", type=int, action="append", default=[]
    )
    parser.add_argument("--strong-reciprocal-overlap", type=float, default=0.30)
    parser.add_argument("--minimum-visual-inliers", type=int, default=15)
    parser.add_argument(
        "--minimum-visual-inlier-ratio", type=float, default=0.30
    )
    args = parser.parse_args()

    excluded = set(args.exclude_frame)
    visual = {}
    for row in json.loads(args.visual_report.read_text(encoding="utf-8")):
        key = tuple(sorted((int(row["first"]), int(row["second"]))))
        current = visual.get(key)
        if (
            current is None
            or row["fundamental_inliers"] > current["fundamental_inliers"]
        ):
            visual[key] = row

    selected = {}
    rejected_reasons = Counter()
    with args.edge_report.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            first, second = int(row["first"]), int(row["second"])
            if first in excluded or second in excluded:
                rejected_reasons["excluded_frame"] += 1
                continue
            key = tuple(sorted((first, second)))
            kind = row["kind"]
            reciprocal = None
            visual_row = visual.get(key)
            if kind == "temporal":
                reason = "temporal"
                score = 0.90 + 0.10 * min(float(row["score"]), 1.0)
            elif kind == "verified":
                reason = "manually_verified"
                score = 1.0
            else:
                reciprocal = min(
                    float(row["forward_ratio"]), float(row["backward_ratio"])
                )
                if reciprocal >= args.strong_reciprocal_overlap:
                    reason = "strong_reciprocal_depth"
                    score = 0.75 + 0.25 * reciprocal
                elif (
                    visual_row is not None
                    and int(visual_row["fundamental_inliers"])
                    >= args.minimum_visual_inliers
                    and float(visual_row["inlier_ratio"])
                    >= args.minimum_visual_inlier_ratio
                ):
                    reason = "depth_plus_visual"
                    visual_strength = min(
                        int(visual_row["fundamental_inliers"]) / 100.0, 1.0
                    )
                    score = 0.55 + 0.25 * reciprocal + 0.20 * visual_strength
                else:
                    rejected_reasons[
                        "insufficient_reciprocal_or_visual_support"
                    ] += 1
                    continue
            candidate = {
                "first": key[0],
                "second": key[1],
                "kind": kind,
                "acceptance_reason": reason,
                "score": score,
                "reciprocal_depth_overlap": reciprocal,
                "visual_matches": None
                if visual_row is None
                else int(visual_row["matches"]),
                "visual_inliers": None
                if visual_row is None
                else int(visual_row["fundamental_inliers"]),
                "visual_inlier_ratio": None
                if visual_row is None
                else float(visual_row["inlier_ratio"]),
            }
            previous = selected.get(key)
            if previous is None or candidate["score"] > previous["score"]:
                selected[key] = candidate

    adjacency = {
        index: set()
        for index in range(args.frame_count)
        if index not in excluded
    }
    for first, second in selected:
        adjacency[first].add(second)
        adjacency[second].add(first)
    unseen = set(adjacency)
    component_sizes = []
    while unseen:
        root = unseen.pop()
        stack = [root]
        size = 1
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    size += 1
        component_sizes.append(size)
    degrees = sorted(len(neighbors) for neighbors in adjacency.values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(next(iter(selected.values())).keys())
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            sorted(
                selected.values(), key=lambda row: (row["first"], row["second"])
            )
        )

    summary = {
        "frame_count": len(adjacency),
        "excluded_frames": sorted(excluded),
        "selected_edge_count": len(selected),
        "acceptance_reasons": Counter(
            row["acceptance_reason"] for row in selected.values()
        ),
        "rejected_reasons": rejected_reasons,
        "component_count": len(component_sizes),
        "component_sizes": sorted(component_sizes, reverse=True),
        "degree": {
            "minimum": degrees[0],
            "median": degrees[len(degrees) // 2],
            "p95": degrees[int(0.95 * (len(degrees) - 1))],
            "maximum": degrees[-1],
        },
        "thresholds": {
            "strong_reciprocal_overlap": args.strong_reciprocal_overlap,
            "minimum_visual_inliers": args.minimum_visual_inliers,
            "minimum_visual_inlier_ratio": args.minimum_visual_inlier_ratio,
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
