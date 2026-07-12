#!/usr/bin/env python3
"""Build MapAnything groups from the corrected ARKit/depth frontend graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gluemap.datasets.polycam import load_polycam_manifest
from gluemap.pairing.pose_depth_graph import (
    add_verified_edges,
    build_graph_groups,
    build_pose_depth_edges,
    load_depth_frame,
    write_edge_report,
    write_group_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_config", type=Path)
    parser.add_argument("edge_report", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--verified-config", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--minimum-memberships", type=int, default=2)
    parser.add_argument("--temporal-neighbors", type=int, default=8)
    parser.add_argument("--spatial-radius-m", type=float, default=3.5)
    parser.add_argument("--depth-pixel-step", type=int, default=8)
    parser.add_argument("--min-directional-overlap", type=float, default=0.0)
    parser.add_argument("--min-mean-overlap", type=float, default=0.015)
    args = parser.parse_args()

    manifest = load_polycam_manifest(args.manifest)
    selected = manifest["frames"][args.start : args.end + 1]
    frames = [
        load_depth_frame(
            frame,
            args.source_root,
            pixel_step=args.depth_pixel_step,
            min_confidence=255,
            max_depth_m=6.0,
        )
        for frame in selected
    ]
    edges = build_pose_depth_edges(
        frames,
        temporal_neighbors=args.temporal_neighbors,
        spatial_radius_m=args.spatial_radius_m,
        min_directional_overlap=args.min_directional_overlap,
        min_mean_overlap=args.min_mean_overlap,
    )
    verified_data = json.loads(args.verified_config.read_text(encoding="utf-8"))
    verified_pairs = [
        tuple(pair)
        for group in verified_data["groups"]
        for pair in group.get("verified_correspondences", [])
    ]
    add_verified_edges(edges, verified_pairs)
    groups = build_graph_groups(
        list(range(args.start, args.end + 1)),
        edges,
        group_size=args.group_size,
        minimum_memberships=args.minimum_memberships,
        seeded_groups=verified_data["groups"],
    )
    coverage = np.zeros(args.end - args.start + 1, dtype=np.int32)
    for group in groups:
        if "frame_indices" in group:
            members = group["frame_indices"]
        else:
            members = [
                index
                for start, end in group["frame_ranges_inclusive"]
                for index in range(start, end + 1)
            ]
        for index in members:
            if args.start <= index <= args.end:
                coverage[index - args.start] += 1

    write_edge_report(edges, args.edge_report)
    output = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "pose_conditioning": False,
        "arkit_policy": {
            "group_frontend": True,
            "mapanything_conditioning": False,
            "retain_corrected_poses": True,
            "global_initialization": True,
            "final_soft_prior": True,
        },
        "segment": {"start": args.start, "end": args.end},
        "frontend": {
            "temporal_neighbors": args.temporal_neighbors,
            "spatial_radius_m": args.spatial_radius_m,
            "depth_pixel_step": args.depth_pixel_step,
            "min_directional_overlap": args.min_directional_overlap,
            "min_mean_overlap": args.min_mean_overlap,
            "edge_report": str(args.edge_report),
        },
        "coverage": {
            "minimum": int(coverage.min()),
            "median": float(np.median(coverage)),
            "maximum": int(coverage.max()),
        },
        "groups": groups,
    }
    write_group_config(output, args.output_config)
    kinds = {}
    for edge in edges:
        kinds[edge["kind"]] = kinds.get(edge["kind"], 0) + 1
    print(
        json.dumps(
            {
                "groups": len(groups),
                "edges": len(edges),
                "edge_kinds": kinds,
                "coverage": output["coverage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
