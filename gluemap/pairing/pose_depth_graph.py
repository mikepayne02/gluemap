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

    yy, xx = np.mgrid[
        0 : depth.shape[0] : pixel_step, 0 : depth.shape[1] : pixel_step
    ]
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
            distance = float(
                np.linalg.norm(by_index[first].center - by_index[second].center)
            )
            edges[(first, second)] = {
                "first": first,
                "second": second,
                "kind": "temporal",
                "center_distance_m": distance,
                "score": float(1.0 / (second - first)),
            }

    centers = np.stack([by_index[index].center for index in indices])
    for first_offset, second_offset in cKDTree(centers).query_pairs(
        spatial_radius_m
    ):
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
        mean_overlap = 0.5 * (
            float(forward["ratio"]) + float(backward["ratio"])
        )
        if (
            int(forward["consistent"]) < min_bidirectional_points
            or int(backward["consistent"]) < min_bidirectional_points
            or float(forward["ratio"]) < min_directional_overlap
            or float(backward["ratio"]) < min_directional_overlap
            or mean_overlap < min_mean_overlap
        ):
            continue
        distance = float(
            np.linalg.norm(by_index[first].center - by_index[second].center)
        )
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


def add_verified_edges(
    edges: list[dict], verified_pairs: list[tuple[int, int]]
) -> None:
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


def _adjacency(
    edges: list[dict], valid_indices: set[int]
) -> dict[int, dict[int, float]]:
    adjacency = {index: {} for index in valid_indices}
    for edge in edges:
        first, second = edge["first"], edge["second"]
        if first not in valid_indices or second not in valid_indices:
            continue
        score = max(float(edge["score"]), 1e-4)
        cost = 1.0 / score
        adjacency[first][second] = min(
            adjacency[first].get(second, np.inf), cost
        )
        adjacency[second][first] = min(
            adjacency[second].get(first, np.inf), cost
        )
    return adjacency


def frontend_edge_is_group_evidence(edge: dict) -> bool:
    """Return whether an audited edge may place two views in one group.

    Temporal continuity is always retained.  A non-temporal edge must have
    independent image evidence, or be one of the explicitly verified loops.
    Depth overlap computed from a drifted pose is not sufficient by itself.
    """
    if edge["acceptance_reason"] in {"temporal", "manually_verified"}:
        return True
    visual_inliers = int(edge.get("visual_inliers") or 0)
    visual_ratio = float(edge.get("visual_inlier_ratio") or 0.0)
    return visual_inliers >= 15 and visual_ratio >= 0.30


def exclude_group_evidence_pairs(
    edges: list[dict], excluded_pairs: set[tuple[int, int]]
) -> tuple[list[dict], int]:
    """Remove audited false correspondences before building any groups."""
    normalized = {tuple(sorted(map(int, pair))) for pair in excluded_pairs}
    kept = [
        edge
        for edge in edges
        if tuple(sorted((int(edge["first"]), int(edge["second"]))))
        not in normalized
    ]
    return kept, len(edges) - len(kept)


def build_verified_bridge_groups(
    indices: list[int], verified_edges: list[dict], *, group_size: int = 64
) -> list[dict]:
    """Build joint local groups around every manually verified revisit.

    Nearby verified endpoints are treated as one visit.  Each physical visit
    receives an equal temporal neighborhood, so the resulting MapAnything
    call directly observes both sides of the closure instead of hoping that a
    later feature matcher rediscovers it.
    """
    valid = set(indices)
    graph: dict[int, set[int]] = {}
    pairs = []
    for edge in verified_edges:
        first, second = sorted((int(edge["first"]), int(edge["second"])))
        if first not in valid or second not in valid:
            continue
        graph.setdefault(first, set()).add(second)
        graph.setdefault(second, set()).add(first)
        pairs.append((first, second))

    components = []
    remaining = set(graph)
    while remaining:
        root = min(remaining)
        stack = [root]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(graph[node] - component)
        remaining -= component
        components.append(sorted(component))

    groups = []
    sorted_indices = sorted(indices)
    for group_index, component in enumerate(components):
        visits: list[list[int]] = []
        for endpoint in component:
            if not visits or endpoint - visits[-1][-1] > 32:
                visits.append([endpoint])
            else:
                visits[-1].append(endpoint)
        centers = [int(round(np.mean(visit))) for visit in visits]
        quota, remainder = divmod(group_size, len(centers))
        members = set(component)
        for visit_index, center in enumerate(centers):
            count = quota + (visit_index < remainder)
            candidates = sorted(
                sorted_indices, key=lambda index: (abs(index - center), index)
            )
            members.update(candidates[:count])
        if len(members) > group_size:
            mandatory = set(component)
            optional = sorted(
                members - mandatory,
                key=lambda index: (
                    min(abs(index - center) for center in centers),
                    index,
                ),
            )
            members = mandatory | set(optional[: group_size - len(mandatory)])
        elif len(members) < group_size:
            candidates = sorted(
                (index for index in sorted_indices if index not in members),
                key=lambda index: (
                    min(abs(index - center) for center in centers),
                    index,
                ),
            )
            members.update(candidates[: group_size - len(members)])
        component_pairs = [
            list(pair)
            for pair in pairs
            if pair[0] in component and pair[1] in component
        ]
        groups.append(
            {
                "name": f"verified_bridge_{group_index:03d}",
                "kind": "verified_bridge",
                "anchor_frame": component[0],
                "frame_indices": sorted(members),
                "verified_correspondences": component_pairs,
                "expected_view_count": len(members),
                "pose_modes": ["none"],
            }
        )
    return groups


def select_centered_frames(
    indices: list[int], center: int, view_count: int
) -> list[int]:
    """Return a deterministic temporal window, shifted at dataset boundaries."""
    if view_count < 2:
        raise ValueError("Recovery groups require at least two views")
    ordered = sorted(map(int, indices))
    if center not in set(ordered):
        raise ValueError(f"Recovery center {center} is not an included frame")
    if view_count > len(ordered):
        raise ValueError(
            f"Recovery view count {view_count} exceeds {len(ordered)} frames"
        )
    center_position = ordered.index(center)
    start = center_position - view_count // 2
    start = min(max(start, 0), len(ordered) - view_count)
    return ordered[start : start + view_count]


def select_sparse_pose_frames(
    members: list[int], anchor: int, *, radius: int, maximum: int
) -> list[int]:
    """Select a deterministic local subset for optional pose conditioning."""
    if maximum < 1:
        raise ValueError("Pose conditioning requires at least one view")
    local = sorted(frame for frame in members if abs(frame - anchor) <= radius)
    if anchor not in local:
        local.append(anchor)
        local.sort()
    if len(local) > maximum:
        positions = np.linspace(0, len(local) - 1, maximum).round().astype(int)
        local = [local[position] for position in positions]
        if anchor not in local:
            local[0] = anchor
    return list(dict.fromkeys([anchor, *local]))


def apply_group_recovery_policy(
    groups: list[dict], policy: dict, valid_indices: set[int]
) -> list[dict]:
    """Add balanced bridge groups around independently verified revisits.

    A recovery candidate supplies one or more image-verified pairs joining two
    visits to the same place.  Half of the views come from temporal context
    around each side of the revisit.  This avoids both failure modes of the
    generic graph cover: a lopsided group with only a few views from one visit,
    and a large contiguous window that never jointly observes the revisit.
    """
    if int(policy.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported group recovery policy schema")
    if policy.get("layout") != "balanced_pair_context":
        raise ValueError("Recovery policy must use balanced_pair_context")
    views_per_side = int(policy.get("views_per_side", 0))
    if views_per_side < 2:
        raise ValueError("Recovery policy requires at least two views per side")

    pose_policy = policy.get("pose_conditioning", {"mode": "none"})
    pose_mode = pose_policy.get("mode", "none")
    if pose_mode not in {"none", "sparse_local"}:
        raise ValueError(f"Unsupported recovery pose mode: {pose_mode}")

    existing_names = {group["name"] for group in groups}
    ordered_indices = sorted(valid_indices)
    additions = []
    for candidate in policy.get("candidates", []):
        name = str(candidate["name"])
        if name in existing_names:
            raise ValueError(f"Duplicate recovery group name: {name}")

        evidence_pairs = [
            [int(pair[0]), int(pair[1])]
            for pair in candidate.get("evidence_pairs", [])
        ]
        if not evidence_pairs:
            raise ValueError(f"Recovery group {name} has no evidence pairs")
        if any(len(pair) != 2 for pair in evidence_pairs):
            raise ValueError(f"Recovery group {name} has a malformed pair")
        evidence_frames = {frame for pair in evidence_pairs for frame in pair}
        missing_evidence = sorted(evidence_frames - valid_indices)
        if missing_evidence:
            raise ValueError(
                f"Recovery group {name} uses excluded frames: "
                f"{missing_evidence}"
            )

        first_center = int(
            candidate.get(
                "first_center_frame",
                round(float(np.median([pair[0] for pair in evidence_pairs]))),
            )
        )
        second_center = int(
            candidate.get(
                "second_center_frame",
                round(float(np.median([pair[1] for pair in evidence_pairs]))),
            )
        )
        if first_center == second_center:
            raise ValueError(
                f"Recovery group {name} has identical visit centers"
            )
        first_members = select_centered_frames(
            ordered_indices, first_center, views_per_side
        )
        second_members = select_centered_frames(
            ordered_indices, second_center, views_per_side
        )
        members = list(dict.fromkeys([*first_members, *second_members]))
        expected_views = 2 * views_per_side
        if len(members) != expected_views:
            raise ValueError(
                f"Recovery group {name} visit contexts overlap: "
                f"{len(members)} != {expected_views}"
            )
        if not evidence_frames <= set(members):
            raise ValueError(
                f"Recovery group {name} context omits an evidence frame"
            )
        anchor = int(candidate.get("anchor_frame", evidence_pairs[0][0]))
        if anchor not in members:
            raise ValueError(
                f"Recovery anchor for {name} is outside its visit contexts"
            )
        conditioned = []
        if pose_mode == "sparse_local":
            conditioned = select_sparse_pose_frames(
                members,
                anchor,
                radius=int(pose_policy.get("anchor_radius", 12)),
                maximum=int(pose_policy.get("max_views", 8)),
            )
        groups.append(
            {
                "name": name,
                "kind": "balanced_revisit_recovery",
                "anchor_frame": anchor,
                "frame_indices": members,
                "verified_correspondences": evidence_pairs,
                "pose_conditioned_frames": conditioned,
                "expected_view_count": len(members),
                "pose_modes": [pose_mode],
            }
        )
        existing_names.add(name)
        additions.append(
            {
                "name": name,
                "anchor_frame": anchor,
                "view_count": len(members),
                "views_per_side": views_per_side,
                "visit_ranges": [
                    [first_members[0], first_members[-1]],
                    [second_members[0], second_members[-1]],
                ],
                "evidence_pairs": evidence_pairs,
                "pose_conditioned_frames": conditioned,
            }
        )

    for candidate in policy.get("local_candidates", []):
        name = str(candidate["name"])
        if name in existing_names:
            raise ValueError(f"Duplicate recovery group name: {name}")
        view_count = int(candidate.get("view_count", 64))
        center = int(candidate["center_frame"])
        members = select_centered_frames(ordered_indices, center, view_count)
        anchor = int(candidate.get("anchor_frame", center))
        if anchor not in members:
            raise ValueError(
                f"Recovery anchor for {name} is outside its temporal context"
            )
        depth_excluded = list(
            dict.fromkeys(map(int, candidate.get("depth_excluded_frames", [])))
        )
        if not set(depth_excluded) <= set(members):
            raise ValueError(
                f"Recovery group {name} excludes depth outside its members"
            )
        local_pose_policy = candidate.get(
            "pose_conditioning", {"mode": "none"}
        )
        local_pose_mode = local_pose_policy.get("mode", "none")
        if local_pose_mode not in {"none", "sparse_local", "full_local"}:
            raise ValueError(
                f"Unsupported local recovery pose mode: {local_pose_mode}"
            )
        conditioned = []
        if local_pose_mode == "sparse_local":
            conditioned = select_sparse_pose_frames(
                members,
                anchor,
                radius=int(local_pose_policy.get("anchor_radius", 12)),
                maximum=int(local_pose_policy.get("max_views", 8)),
            )
        elif local_pose_mode == "full_local":
            conditioned = list(members)
        temporal_ranges = [
            [int(bounds[0]), int(bounds[1])]
            for bounds in candidate.get("authoritative_temporal_ranges", [])
        ]
        if any(
            start >= end or not {start, end} <= set(members)
            for start, end in temporal_ranges
        ):
            raise ValueError(
                f"Recovery group {name} has an invalid temporal range"
            )
        groups.append(
            {
                "name": name,
                "kind": "local_temporal_recovery",
                "anchor_frame": anchor,
                "frame_indices": members,
                "depth_excluded_frames": depth_excluded,
                "pose_conditioned_frames": conditioned,
                "authoritative_temporal_ranges": temporal_ranges,
                "expected_view_count": len(members),
                "pose_modes": [local_pose_mode],
            }
        )
        existing_names.add(name)
        additions.append(
            {
                "name": name,
                "anchor_frame": anchor,
                "view_count": len(members),
                "temporal_range": [members[0], members[-1]],
                "depth_excluded_frames": depth_excluded,
                "authoritative_temporal_ranges": temporal_ranges,
                "pose_conditioned_frames": conditioned,
            }
        )
    return additions


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
        members = list(map(int, group.get("frame_indices", [])))
        for start, end in group.get("frame_ranges_inclusive", []):
            members.extend(range(start, end + 1))
        members = list(
            dict.fromkeys(index for index in members if index in valid_indices)
        )
        group["frame_indices"] = members
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
        if previous is None or len(adjacency[anchor]) > len(
            adjacency[previous]
        ):
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
