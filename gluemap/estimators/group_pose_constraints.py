"""Build a global camera graph from overlapping MapAnything groups."""

from __future__ import annotations

import math
from collections import defaultdict

import torch


def relative_pose(
    extrinsics: torch.Tensor, first_position: int, second_position: int
) -> torch.Tensor:
    """Return ``camera_second_from_camera_first`` for one predicted group."""
    first = extrinsics[0, first_position]
    second = extrinsics[0, second_position]
    rotation = second[:3, :3] @ first[:3, :3].T
    translation = second[:3, 3:] - rotation @ first[:3, 3:]
    return torch.cat([rotation, translation], dim=-1)


def _pose_disagreement(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[float, float]:
    rotation = first[:3, :3].double() @ second[:3, :3].double().T
    cosine = torch.clamp((torch.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_deg = math.degrees(float(torch.acos(cosine)))
    translation_m = float(
        torch.linalg.norm(first[:3, 3].double() - second[:3, 3].double())
    )
    return rotation_deg, translation_m


def _positive_score(value: torch.Tensor | float) -> float:
    """Return a finite positive edge weight without deleting support."""
    score = float(value)
    return max(score, 1e-6) if math.isfinite(score) else 1e-6


def _collect_candidates(
    predictions_dict: dict, pairs: set[tuple[int, int]]
) -> dict[tuple[int, int], list[dict]]:
    candidates: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for star_index, image_ids in enumerate(predictions_dict["indexes"]):
        positions = {
            int(image_id): position
            for position, image_id in enumerate(image_ids)
        }
        scores = predictions_dict["pose_scores"][star_index][0]
        for first, second in pairs:
            if first not in positions or second not in positions:
                continue
            first_position = positions[first]
            second_position = positions[second]
            first_score = _positive_score(scores[first_position])
            second_score = _positive_score(scores[second_position])
            score = math.sqrt(first_score * second_score)
            if not math.isfinite(score):
                score = 1e-6
            candidates[(first, second)].append(
                {
                    "star_index": star_index,
                    "first_position": first_position,
                    "second_position": second_position,
                    "pose": relative_pose(
                        predictions_dict["extrinsics"][star_index],
                        first_position,
                        second_position,
                    ),
                    "score": score,
                }
            )
    return candidates


def _group_reliability(
    temporal_candidates: dict[tuple[int, int], list[dict]],
    *,
    rotation_threshold_deg: float,
    translation_threshold_m: float,
) -> dict[int, float]:
    """Score each group by agreement on temporal pairs shared by groups."""
    agreements: dict[int, int] = defaultdict(int)
    comparisons: dict[int, int] = defaultdict(int)
    all_groups = {
        candidate["star_index"]
        for candidates in temporal_candidates.values()
        for candidate in candidates
    }
    for candidates in temporal_candidates.values():
        if len(candidates) < 2:
            continue
        for candidate in candidates:
            others = [
                other
                for other in candidates
                if other["star_index"] != candidate["star_index"]
            ]
            if not others:
                continue
            comparisons[candidate["star_index"]] += 1
            if any(
                rotation <= rotation_threshold_deg
                and translation <= translation_threshold_m
                for rotation, translation in (
                    _pose_disagreement(candidate["pose"], other["pose"])
                    for other in others
                )
            ):
                agreements[candidate["star_index"]] += 1

    # A small prior avoids declaring a sparsely overlapping group perfect or
    # unusable from one comparison. The value is only used to break local
    # candidate ties; it never creates a camera measurement.
    return {
        group: (agreements[group] + 2.0) / (comparisons[group] + 3.0)
        for group in all_groups
    }


def _select_candidate(
    candidates: list[dict],
    reliability: dict[int, float],
    *,
    rotation_threshold_deg: float,
    translation_threshold_m: float,
) -> tuple[dict, float, list[int]]:
    """Choose the actual group estimate with the strongest robust consensus."""
    weights = [
        candidate["score"] * reliability[candidate["star_index"]]
        for candidate in candidates
    ]
    costs = []
    for candidate in candidates:
        cost = 0.0
        for other, weight in zip(candidates, weights, strict=True):
            rotation, translation = _pose_disagreement(
                candidate["pose"], other["pose"]
            )
            cost += weight * (
                min(rotation / rotation_threshold_deg, 4.0)
                + min(translation / translation_threshold_m, 4.0)
            )
        costs.append(cost)
    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            costs[index],
            -weights[index],
            candidates[index]["star_index"],
        ),
    )
    selected = candidates[selected_index]
    supporting = []
    supporting_weight = 0.0
    for candidate, weight in zip(candidates, weights, strict=True):
        rotation, translation = _pose_disagreement(
            selected["pose"], candidate["pose"]
        )
        if (
            rotation <= rotation_threshold_deg
            and translation <= translation_threshold_m
        ):
            supporting.append(candidate["star_index"])
            supporting_weight += weight
    agreement_fraction = supporting_weight / max(sum(weights), 1e-12)
    return selected, agreement_fraction, supporting


def build_group_pose_constraints(
    predictions_dict: dict,
    *,
    num_images: int,
    trusted_loop_edges: set[tuple[int, int]],
    preferred_temporal_groups: dict[
        tuple[int, int], set[int]
    ] | None = None,
    rotation_threshold_deg: float = 7.5,
    translation_threshold_m: float = 0.25,
) -> list[dict]:
    """Build MapAnything-only temporal and verified-loop constraints.

    Every consecutive camera pair is measured inside each group that jointly
    predicted it. A group-level reliability score and a pairwise medoid select
    one real prediction without averaging incompatible poses. Nonlocal loops
    are admitted only when the group configuration explicitly verifies them.
    ARKit poses are not read here.
    """
    temporal_pairs = {(index, index + 1) for index in range(num_images - 1)}
    preferred_temporal_groups = {
        tuple(sorted(map(int, pair))): set(map(int, group_indices))
        for pair, group_indices in (preferred_temporal_groups or {}).items()
    }
    invalid_preferred_pairs = sorted(
        set(preferred_temporal_groups) - temporal_pairs
    )
    if invalid_preferred_pairs:
        raise ValueError(
            "Preferred group selection is only valid for consecutive "
            f"cameras: {invalid_preferred_pairs}"
        )
    loop_pairs = {
        tuple(sorted(map(int, pair)))
        for pair in trusted_loop_edges
        if abs(int(pair[0]) - int(pair[1])) > 1
    }
    requested_pairs = temporal_pairs | loop_pairs
    candidates = _collect_candidates(predictions_dict, requested_pairs)
    missing_temporal = sorted(temporal_pairs - candidates.keys())
    if missing_temporal:
        raise RuntimeError(
            "MapAnything groups do not jointly predict consecutive cameras: "
            f"{missing_temporal[:20]} ({len(missing_temporal)} total)"
        )
    missing_loops = sorted(loop_pairs - candidates.keys())
    if missing_loops:
        raise RuntimeError(
            "Verified loops are absent from their MapAnything groups: "
            f"{missing_loops}"
        )

    reliability = _group_reliability(
        {pair: candidates[pair] for pair in temporal_pairs},
        rotation_threshold_deg=rotation_threshold_deg,
        translation_threshold_m=translation_threshold_m,
    )
    constraints = []
    for kind, pairs in (
        ("mapanything_temporal", sorted(temporal_pairs)),
        ("mapanything_verified_loop", sorted(loop_pairs)),
    ):
        for first, second in pairs:
            all_candidates = candidates[(first, second)]
            preferred_groups = preferred_temporal_groups.get((first, second))
            selection_candidates = all_candidates
            if preferred_groups is not None:
                selection_candidates = [
                    candidate
                    for candidate in all_candidates
                    if candidate["star_index"] in preferred_groups
                ]
                if not selection_candidates:
                    raise RuntimeError(
                        "Preferred MapAnything group does not predict "
                        f"camera pair {(first, second)}: "
                        f"{sorted(preferred_groups)}"
                    )
            selected, agreement_fraction, supporting = _select_candidate(
                selection_candidates,
                reliability,
                rotation_threshold_deg=rotation_threshold_deg,
                translation_threshold_m=translation_threshold_m,
            )
            constraints.append(
                {
                    **selected,
                    "first": first,
                    "second": second,
                    "kind": kind,
                    "active": True,
                    "candidate_count": len(all_candidates),
                    "selection_candidate_count": len(selection_candidates),
                    "preferred_group_override": preferred_groups is not None,
                    "agreement_fraction": agreement_fraction,
                    "supporting_star_indices": supporting,
                    "selected_group_reliability": reliability[
                        selected["star_index"]
                    ],
                }
            )
    return constraints


def build_group_fragment_constraints(
    predictions_dict: dict,
    *,
    num_images: int,
    rotation_threshold_deg: float = 7.5,
    translation_threshold_m: float = 0.25,
) -> list[dict]:
    """Preserve each MapAnything group as a coherent local fragment.

    Every group contributes anchor-to-member measurements from its single
    joint prediction.  Shared cameras then align the overlapping fragments
    globally.  This keeps long-baseline geometry that would be discarded by
    reducing a group to independently selected consecutive steps.
    """
    temporal_pairs = {(index, index + 1) for index in range(num_images - 1)}
    temporal_candidates = _collect_candidates(
        predictions_dict, temporal_pairs
    )
    reliability = _group_reliability(
        temporal_candidates,
        rotation_threshold_deg=rotation_threshold_deg,
        translation_threshold_m=translation_threshold_m,
    )
    constraints = []
    for star_index, image_ids in enumerate(predictions_dict["indexes"]):
        scores = predictions_dict["pose_scores"][star_index][0]
        anchor_score = _positive_score(scores[0])
        for position in range(1, len(image_ids)):
            member_score = _positive_score(scores[position])
            score = math.sqrt(anchor_score * member_score) * reliability.get(
                star_index, 2.0 / 3.0
            )
            constraints.append(
                {
                    "star_index": star_index,
                    "first": int(image_ids[0]),
                    "second": int(image_ids[position]),
                    "first_position": 0,
                    "second_position": position,
                    "pose": relative_pose(
                        predictions_dict["extrinsics"][star_index],
                        0,
                        position,
                    ),
                    "score": max(score, 1e-6),
                    "kind": "mapanything_group_fragment",
                    "active": True,
                    "selected_group_reliability": reliability.get(
                        star_index, 2.0 / 3.0
                    ),
                }
            )
    return constraints


def iter_pose_constraints(predictions_dict: dict):
    """Yield explicit group constraints, or legacy anchor-member edges."""
    explicit = predictions_dict.get("pose_constraints")
    if explicit is not None:
        for constraint in explicit:
            if constraint["active"] and constraint["score"] > 0:
                yield constraint
        return

    for star_index, image_ids in enumerate(predictions_dict["indexes"]):
        scores = predictions_dict["pose_scores"][star_index][0]
        for position in torch.where(scores > 0)[0].tolist():
            if position == 0:
                continue
            yield {
                "star_index": star_index,
                "first": int(image_ids[0]),
                "second": int(image_ids[position]),
                "first_position": 0,
                "second_position": position,
                "pose": predictions_dict["extrinsics"][star_index][0, position],
                "score": float(scores[position]),
                "kind": "legacy_anchor_member",
                "active": True,
            }
