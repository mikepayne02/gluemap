"""Construct Polycam local groups from poses, frusta, and measured depth."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


@dataclass
class DepthFrame:
    index: int
    center: np.ndarray
    rotation: np.ndarray
    world_points: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    intrinsics: np.ndarray


def load_depth_frame(
    frame: dict,
    source_root: Path,
    *,
    pixel_step: int,
    min_confidence: int,
    max_depth_m: float,
) -> DepthFrame:
    paths = frame["paths"]
    with Image.open(source_root / paths["depth"]) as image:
        depth = np.asarray(image, dtype=np.float32) * 0.001
    with Image.open(source_root / paths["confidence"]) as image:
        confidence = np.asarray(image, dtype=np.uint8)
    intrinsics = np.asarray(frame["intrinsics_depth"], dtype=np.float64)
    c2w = np.asarray(frame["corrected_c2w_opencv"], dtype=np.float64)

    yy, xx = np.mgrid[0 : depth.shape[0] : pixel_step, 0 : depth.shape[1] : pixel_step]
    zz = depth[yy, xx]
    valid = (
        np.isfinite(zz)
        & (zz > 0.08)
        & (zz <= max_depth_m)
        & (confidence[yy, xx] >= min_confidence)
    )
    zz = zz[valid].astype(np.float64)
    xx = xx[valid].astype(np.float64)
    yy = yy[valid].astype(np.float64)
    camera_points = np.stack(
        [
            (xx - intrinsics[0, 2]) * zz / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) * zz / intrinsics[1, 1],
            zz,
        ],
        axis=1,
    )
    world_points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
    return DepthFrame(
        index=int(frame["sequence_index"]),
        center=c2w[:3, 3],
        rotation=c2w[:3, :3],
        world_points=world_points,
        depth=depth,
        confidence=confidence,
        intrinsics=intrinsics,
    )


def directional_depth_overlap(
    source: DepthFrame,
    target: DepthFrame,
    *,
    min_confidence: int,
    absolute_tolerance_m: float,
    relative_tolerance: float,
) -> dict[str, float | int]:
    if len(source.world_points) == 0:
        return {"visible": 0, "consistent": 0, "ratio": 0.0}
    target_points = (source.world_points - target.center) @ target.rotation
    z = target_points[:, 2]
    positive = z > 0.08
    u = target.intrinsics[0, 0] * target_points[:, 0] / np.maximum(z, 1e-8)
    u += target.intrinsics[0, 2]
    v = target.intrinsics[1, 1] * target_points[:, 1] / np.maximum(z, 1e-8)
    v += target.intrinsics[1, 2]
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    height, width = target.depth.shape
    inside = positive & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    inside_indices = np.flatnonzero(inside)
    if len(inside_indices) == 0:
        return {"visible": 0, "consistent": 0, "ratio": 0.0}
    target_depth = target.depth[y[inside], x[inside]]
    target_confidence = target.confidence[y[inside], x[inside]]
    valid_target = (
        np.isfinite(target_depth)
        & (target_depth > 0.08)
        & (target_confidence >= min_confidence)
    )
    residual = np.abs(z[inside] - target_depth)
    tolerance = absolute_tolerance_m + relative_tolerance * target_depth
    consistent = valid_target & (residual <= tolerance)
    count = int(np.count_nonzero(consistent))
    return {
        "visible": int(np.count_nonzero(valid_target)),
        "consistent": count,
        "ratio": float(count / len(source.world_points)),
    }


def build_pose_depth_edges(
    frames: list[DepthFrame],
    *,
    temporal_neighbors: int = 8,
    spatial_radius_m: float = 3.5,
    min_bidirectional_points: int = 6,
    min_directional_overlap: float = 0.0,
    min_mean_overlap: float = 0.015,
    min_confidence: int = 255,
    absolute_tolerance_m: float = 0.12,
    relative_tolerance: float = 0.04,
) -> list[dict]:
    by_index = {frame.index: frame for frame in frames}
    indices = sorted(by_index)
    edges: dict[tuple[int, int], dict] = {}
    for offset, first in enumerate(indices):
        for second in indices[offset + 1 : offset + 1 + temporal_neighbors]:
            if second - first > temporal_neighbors:
                break
            distance = float(np.linalg.norm(by_index[first].center - by_index[second].center))
            edges[(first, second)] = {
                "first": first,
                "second": second,
                "kind": "temporal",
                "center_distance_m": distance,
                "score": float(1.0 / (second - first)),
            }

    centers = np.stack([by_index[index].center for index in indices])
    for first_offset, second_offset in cKDTree(centers).query_pairs(spatial_radius_m):
        first, second = indices[first_offset], indices[second_offset]
        if (first, second) in edges:
            continue
        forward = directional_depth_overlap(
            by_index[first],
            by_index[second],
            min_confidence=min_confidence,
            absolute_tolerance_m=absolute_tolerance_m,
            relative_tolerance=relative_tolerance,
        )
        backward = directional_depth_overlap(
            by_index[second],
            by_index[first],
            min_confidence=min_confidence,
            absolute_tolerance_m=absolute_tolerance_m,
            relative_tolerance=relative_tolerance,
        )
        mean_overlap = 0.5 * (float(forward["ratio"]) + float(backward["ratio"]))
        if (
            int(forward["consistent"]) < min_bidirectional_points
            or int(backward["consistent"]) < min_bidirectional_points
            or float(forward["ratio"]) < min_directional_overlap
            or float(backward["ratio"]) < min_directional_overlap
            or mean_overlap < min_mean_overlap
        ):
            continue
        distance = float(np.linalg.norm(by_index[first].center - by_index[second].center))
        edges[(first, second)] = {
            "first": first,
            "second": second,
            "kind": "depth_overlap",
            "center_distance_m": distance,
            "forward_visible": forward["visible"],
            "forward_consistent": forward["consistent"],
            "forward_ratio": forward["ratio"],
            "backward_visible": backward["visible"],
            "backward_consistent": backward["consistent"],
            "backward_ratio": backward["ratio"],
            "score": mean_overlap,
        }
    return list(edges.values())


def add_verified_edges(edges: list[dict], verified_pairs: list[tuple[int, int]]) -> None:
    existing = {(edge["first"], edge["second"]) for edge in edges}
    for first, second in verified_pairs:
        first, second = sorted((int(first), int(second)))
        if (first, second) not in existing:
            edges.append(
                {
                    "first": first,
                    "second": second,
                    "kind": "verified",
                    "center_distance_m": None,
                    "score": 2.0,
                }
            )


def _adjacency(edges: list[dict], valid_indices: set[int]) -> dict[int, dict[int, float]]:
    adjacency = {index: {} for index in valid_indices}
    for edge in edges:
        first, second = edge["first"], edge["second"]
        if first not in valid_indices or second not in valid_indices:
            continue
        score = max(float(edge["score"]), 1e-4)
        cost = 1.0 / score
        adjacency[first][second] = min(adjacency[first].get(second, np.inf), cost)
        adjacency[second][first] = min(adjacency[second].get(first, np.inf), cost)
    return adjacency


def build_graph_groups(
    indices: list[int],
    edges: list[dict],
    *,
    group_size: int = 64,
    minimum_memberships: int = 2,
    seeded_groups: list[dict] | None = None,
) -> list[dict]:
    valid_indices = set(indices)
    adjacency = _adjacency(edges, valid_indices)
    coverage = {index: 0 for index in indices}
    groups = []
    for source in seeded_groups or []:
        group = dict(source)
        members = []
        for start, end in group["frame_ranges_inclusive"]:
            members.extend(range(start, end + 1))
        group["kind"] = "verified_bridge"
        group["pose_modes"] = ["none"]
        group["expected_view_count"] = len(members)
        groups.append(group)
        for index in members:
            if index in coverage:
                coverage[index] += 1

    candidates = {}
    for anchor in indices:
        direct_neighbors = sorted(
            adjacency[anchor],
            key=lambda index: (
                adjacency[anchor][index],
                abs(index - anchor),
                index,
            ),
        )
        members_list = [anchor, *direct_neighbors[: group_size - 1]]
        if len(members_list) < group_size:
            existing = set(members_list)
            temporal_fill = sorted(
                (index for index in indices if index not in existing),
                key=lambda index: (abs(index - anchor), index),
            )
            members_list.extend(temporal_fill[: group_size - len(members_list)])
        members = tuple(sorted(members_list))
        previous = candidates.get(members)
        if previous is None or len(adjacency[anchor]) > len(adjacency[previous]):
            candidates[members] = anchor

    remaining = dict(candidates)
    while min(coverage.values()) < minimum_memberships:
        selected = None
        selected_score = None
        for members, anchor in remaining.items():
            gain = sum(
                max(0, minimum_memberships - coverage[index])
                for index in members
            )
            undercovered = sum(
                coverage[index] < minimum_memberships for index in members
            )
            score = (gain, undercovered, len(adjacency[anchor]), -anchor)
            if selected_score is None or score > selected_score:
                selected = (anchor, members)
                selected_score = score
        if selected is None or selected_score[0] == 0:
            raise RuntimeError("Could not improve graph-group coverage")
        anchor, members = selected
        del remaining[members]
        group_number = len(groups)
        groups.append(
            {
                "name": f"graph_{group_number:03d}_anchor_{anchor:04d}",
                "kind": "pose_depth_graph",
                "anchor_frame": anchor,
                "frame_indices": list(members),
                "expected_view_count": len(members),
                "pose_modes": ["none"],
            }
        )
        for index in members:
            coverage[index] += 1
    return groups


def write_edge_report(edges: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for edge in edges for key in edge})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)


def write_group_config(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
