import numpy as np
import torch

from gluemap.estimators.group_pose_constraints import (
    build_group_pose_constraints,
)


def _predictions():
    extrinsics = torch.eye(4).repeat(1, 4, 1, 1)
    extrinsics[0, 1, 0, 3] = -1
    extrinsics[0, 2, 0, 3] = -2
    extrinsics[0, 3, 0, 3] = -3
    return {
        "indexes": [[0, 1, 2, 3]],
        "extrinsics": [extrinsics],
        "pose_scores": [torch.ones(1, 4)],
    }


def _trajectory():
    poses = []
    for index in range(4):
        pose = np.eye(4)
        pose[0, 3] = index
        poses.append(pose)
    return poses


def test_group_constraints_keep_capture_motion_and_verified_loops_only():
    constraints = build_group_pose_constraints(
        _predictions(), {(0, 3)}, _trajectory(), {(1, 2)}
    )

    pairs = {(item["first"], item["second"]): item for item in constraints}
    assert set(pairs) == {(0, 1), (1, 2), (2, 3), (0, 3)}
    assert pairs[(0, 1)]["kind"] == "trajectory_odometry"
    assert pairs[(1, 2)]["kind"] == "trajectory_rotation_bridge"
    assert pairs[(0, 3)]["kind"] == "verified_loop"
    assert torch.allclose(
        pairs[(0, 1)]["pose"][:3, 3],
        torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64),
    )


def test_unverified_nonlocal_group_member_is_not_a_camera_constraint():
    constraints = build_group_pose_constraints(
        _predictions(), set(), _trajectory(), set()
    )

    assert (0, 3) not in {
        (item["first"], item["second"]) for item in constraints
    }
