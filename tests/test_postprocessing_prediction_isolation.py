"""Keep neural depth tracks independent of global pose pruning."""

import torch

from gluemap.controllers.gluemap_impl import (
    _apply_group_scales,
    _copy_predictions_for_global_mapping,
)


def test_pose_graph_pruning_cannot_shorten_full_neural_groups():
    predictions = {
        "indexes": [[0, 1, 2]],
        "pose_scores": [torch.ones(1, 3)],
        "extrinsics": [torch.eye(4).repeat(1, 3, 1, 1)],
        "valid_virtual": [torch.ones(1, 3, 7, dtype=torch.bool)],
        "vis": [torch.ones(1, 3, 7)],
    }

    pose_graph = _copy_predictions_for_global_mapping(predictions)
    pose_graph["indexes"][0] = pose_graph["indexes"][0][:2]
    pose_graph["valid_virtual"][0] = pose_graph["valid_virtual"][0][:, :2]
    pose_graph["pose_scores"][0][0, 2] = 0
    pose_graph["extrinsics"][0][0, 2, 0, 3] = 99

    assert predictions["indexes"][0] == [0, 1, 2]
    assert predictions["valid_virtual"][0].shape[1] == 3
    assert predictions["pose_scores"][0][0, 2] == 1
    assert predictions["extrinsics"][0][0, 2, 0, 3] == 0


def test_solved_group_scale_updates_all_full_group_geometry():
    predictions = {
        "indexes": [[0, 1, 2]],
        "points3d_virtual": [torch.full((1, 4, 3), 4.0)],
        "extrinsics": [torch.eye(4).repeat(1, 3, 1, 1)],
    }
    predictions["extrinsics"][0][:, :, :3, 3] = 6.0

    _apply_group_scales(predictions, [2.0])

    assert torch.all(predictions["points3d_virtual"][0] == 2.0)
    assert torch.all(predictions["extrinsics"][0][:, :, :3, 3] == 3.0)
