#!/usr/bin/env python3
"""Build a deterministic GlueMap pair graph from Polycam poses.

The output is a JSON file consumable by ``--pair_graph_path``. It uses local
indices after applying ``--sample-frequency``, matching GlueMap's sequential
dataset indexing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def image_list(images_path: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(images_path, "**"), recursive=True))
    return [
        p
        for p in paths
        if os.path.isfile(p)
        and p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]


def load_polycam_centers(transforms_path: Path) -> dict[str, np.ndarray]:
    with transforms_path.open() as f:
        data = json.load(f)
    centers = {}
    for frame in data.get("frames", []):
        name = Path(frame["file_path"]).name
        mat = np.asarray(frame["transform_matrix"], dtype=float)
        centers[name] = mat[:3, 3]
    return centers


def add_pair(pairs: set[tuple[int, int]], i: int, j: int) -> None:
    if i == j:
        return
    pairs.add((min(i, j), max(i, j)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-path", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-frequency", type=int, default=1)
    parser.add_argument("--sequential-window", type=int, default=24)
    parser.add_argument("--spatial-radius", type=float, default=1.25)
    parser.add_argument("--spatial-max-neighbors", type=int, default=12)
    parser.add_argument("--spatial-min-frame-gap", type=int, default=60)
    parser.add_argument(
        "--forced-pair",
        action="append",
        default=[],
        help="extra pair as local indexes or basenames: A,B",
    )
    args = parser.parse_args()

    full_paths = image_list(args.images_path)
    used_paths = full_paths[:: max(1, args.sample_frequency)]
    rel_names = [
        p.replace(args.images_path, "").strip("/") for p in used_paths
    ]
    basenames = [Path(p).name for p in used_paths]
    base_to_local = {name: i for i, name in enumerate(basenames)}

    centers_by_name = load_polycam_centers(Path(args.transforms))
    missing = [name for name in basenames if name not in centers_by_name]
    if missing:
        raise SystemExit(f"Missing Polycam centers for {missing[:10]}")

    centers = np.stack([centers_by_name[name] for name in basenames])
    n = len(centers)

    pairs: set[tuple[int, int]] = set()
    sequential_edges: set[tuple[int, int]] = set()

    seq = max(0, int(args.sequential_window))
    for i in range(n):
        for j in range(i + 1, min(n, i + seq + 1)):
            pair = (i, j)
            pairs.add(pair)
            sequential_edges.add(pair)

    if args.spatial_radius > 0 and args.spatial_max_neighbors > 0:
        dists = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=-1
        )
        min_gap = max(0, int(args.spatial_min_frame_gap))
        for i in range(n):
            candidates = [
                j
                for j in range(n)
                if j != i
                and abs(j - i) >= min_gap
                and dists[i, j] <= args.spatial_radius
            ]
            candidates.sort(key=lambda j: float(dists[i, j]))
            for j in candidates[: args.spatial_max_neighbors]:
                add_pair(pairs, i, j)

    def resolve(token: str) -> int:
        token = token.strip()
        if token.isdigit():
            return int(token)
        name = Path(token).name
        if name not in base_to_local:
            raise KeyError(f"Unknown forced-pair image: {token}")
        return base_to_local[name]

    for item in args.forced_pair:
        a, b = item.split(",", 1)
        add_pair(pairs, resolve(a), resolve(b))

    pairs_sorted = sorted(pairs)
    out = {
        "pairs": pairs_sorted,
        "sequential_edges": sorted(sequential_edges),
        "metadata": {
            "images_path": args.images_path,
            "transforms": str(args.transforms),
            "sample_frequency": args.sample_frequency,
            "num_images": n,
            "num_pairs": len(pairs_sorted),
            "sequential_window": args.sequential_window,
            "spatial_radius": args.spatial_radius,
            "spatial_max_neighbors": args.spatial_max_neighbors,
            "spatial_min_frame_gap": args.spatial_min_frame_gap,
            "images": rel_names,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(f"wrote {output}")
    print(f"images={n} pairs={len(pairs_sorted)} seq={len(sequential_edges)}")


if __name__ == "__main__":
    main()
