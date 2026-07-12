#!/usr/bin/env python3
"""Measure visual support for pose/depth-proposed Polycam overlap edges."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from lightglue import ALIKED, LightGlue
from lightglue.utils import rbd
from run_polycam_mapanything_group import _load_view

from gluemap.datasets.polycam import load_polycam_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("edge_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--anchor", type=int, action="append", default=[])
    parser.add_argument("--all-anchors", action="store_true")
    parser.add_argument("--candidates-per-anchor", type=int, default=8)
    parser.add_argument(
        "--minimum-reciprocal-overlap", type=float, default=0.10
    )
    parser.add_argument("--minimum-index-separation", type=int, default=16)
    parser.add_argument(
        "--exclude-frame", type=int, action="append", default=[]
    )
    args = parser.parse_args()

    adjacency: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    with args.edge_report.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["kind"] != "depth_overlap":
                continue
            first, second = int(row["first"]), int(row["second"])
            if first in args.exclude_frame or second in args.exclude_frame:
                continue
            reciprocal = min(
                float(row["forward_ratio"]), float(row["backward_ratio"])
            )
            mean = float(row["score"])
            adjacency[first].append((reciprocal, mean, second))
            adjacency[second].append((reciprocal, mean, first))

    anchors = (
        range(len(load_polycam_manifest(args.manifest)["frames"]))
        if args.all_anchors
        else args.anchor
    )
    if not args.all_anchors and not args.anchor:
        parser.error("provide --anchor at least once or use --all-anchors")
    pairs_by_key = {}
    for anchor in anchors:
        if anchor in args.exclude_frame:
            continue
        candidates = (
            candidate
            for candidate in adjacency[anchor]
            if abs(candidate[2] - anchor) > args.minimum_index_separation
            and candidate[0] >= args.minimum_reciprocal_overlap
        )
        for reciprocal, mean, second in sorted(candidates, reverse=True)[
            : args.candidates_per_anchor
        ]:
            key = (min(anchor, second), max(anchor, second))
            previous = pairs_by_key.get(key)
            candidate = (anchor, second, reciprocal, mean)
            if previous is None or reciprocal > previous[2]:
                pairs_by_key[key] = candidate
    pairs = list(pairs_by_key.values())

    manifest = load_polycam_manifest(args.manifest)
    extractor = (
        ALIKED(max_num_keypoints=1024, detection_threshold=0.005).eval().cuda()
    )
    matcher = LightGlue(features="aliked").eval().cuda()
    feature_cache = {}

    def features(index: int) -> dict:
        if index not in feature_cache:
            view = _load_view(
                manifest["frames"][index],
                args.source_root,
                min_confidence=1,
                include_pose=False,
                orientation="upright_cw",
            )
            image = cv2.resize(
                view["img"], (384, 512), interpolation=cv2.INTER_AREA
            )
            tensor = (
                torch.from_numpy(image).permute(2, 0, 1).float().cuda() / 255.0
            )
            feature_cache[index] = extractor.extract(tensor)
        return feature_cache[index]

    results = []
    with torch.inference_mode():
        for ordinal, (first, second, reciprocal, mean) in enumerate(pairs, 1):
            features_first, features_second = features(first), features(second)
            match_result = rbd(
                matcher({"image0": features_first, "image1": features_second})
            )
            first_unbatched, second_unbatched = (
                rbd(features_first),
                rbd(features_second),
            )
            matches = match_result["matches"].cpu().numpy()
            inliers = 0
            if len(matches) >= 8:
                points_first = (
                    first_unbatched["keypoints"][matches[:, 0]].cpu().numpy()
                )
                points_second = (
                    second_unbatched["keypoints"][matches[:, 1]].cpu().numpy()
                )
                try:
                    _, mask = cv2.findFundamentalMat(
                        points_first,
                        points_second,
                        cv2.USAC_MAGSAC,
                        1.5,
                        0.999,
                        10000,
                    )
                except cv2.error:
                    mask = None
                if mask is not None:
                    inliers = int(mask.sum())
            result = {
                "first": first,
                "second": second,
                "reciprocal_depth_overlap": reciprocal,
                "mean_depth_overlap": mean,
                "matches": int(len(matches)),
                "fundamental_inliers": inliers,
                "inlier_ratio": inliers / max(len(matches), 1),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            if ordinal % 100 == 0:
                print(
                    json.dumps({"progress": ordinal, "total": len(pairs)}),
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
