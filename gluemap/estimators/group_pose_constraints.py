"""Local odometry and verified-loop constraints for overlapping groups."""

from __future__ import annotations

import numpy as np
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


def build_group_pose_constraints(
    predictions_dict: dict,
    trusted_loop_edges: set[tuple[int, int]],
    trajectory_priors_c2w: list[np.ndarray],
    trajectory_break_edges: set[tuple[int, int]],
) -> list[dict]:
    """Use differential ARKit motion plus image-verified revisit links.

    ARKit positions are never absolute optimization targets.  Only adjacent
    relative motion is used, and known tracking-reset boundaries are omitted.
    Nonlocal links come from MapAnything only when imagery independently
    verified the revisit.
    """
    constraints = []
    for first in range(len(trajectory_priors_c2w) - 1):
        second = first + 1
        first_c2w = np.asarray(trajectory_priors_c2w[first], dtype=np.float64)
        second_c2w = np.asarray(trajectory_priors_c2w[second], dtype=np.float64)
        pose = np.linalg.inv(second_c2w) @ first_c2w
        is_break = (first, second) in trajectory_break_edges
        constraints.append(
            {
                "star_index": None,
                "first": first,
                "second": second,
                "first_position": None,
                "second_position": None,
                "pose": torch.from_numpy(pose[:3]),
                "score": 1.0,
                "kind": (
                    "trajectory_rotation_bridge"
                    if is_break
                    else "trajectory_odometry"
                ),
                "active": True,
            }
        )

    for star_index, image_ids in enumerate(predictions_dict["indexes"]):
        positions = {
            int(image_id): position
            for position, image_id in enumerate(image_ids)
        }
        anchor = int(image_ids[0])
        for member in map(int, image_ids[1:]):
            edge = (min(anchor, member), max(anchor, member))
            if edge not in trusted_loop_edges:
                continue
            first, second = edge
            first_pos = positions[first]
            second_pos = positions[second]
            scores = predictions_dict["pose_scores"][star_index][0]
            score = float(torch.sqrt(scores[first_pos] * scores[second_pos]))
            if not torch.isfinite(torch.tensor(score)) or score <= 0:
                continue
            extrinsics = predictions_dict["extrinsics"][star_index]
            constraints.append(
                {
                    "star_index": star_index,
                    "first": first,
                    "second": second,
                    "first_position": first_pos,
                    "second_position": second_pos,
                    "pose": relative_pose(extrinsics, first_pos, second_pos),
                    "score": score,
                    "kind": "verified_loop",
                    "active": True,
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
