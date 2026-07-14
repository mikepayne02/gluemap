import torch

from gluemap.ff_inference.mapanything_inference import MapAnythingLocalInference


def test_mapanything_can_condition_only_selected_group_views():
    images = torch.zeros(1, 3, 3, 64, 64)
    poses = torch.eye(4).repeat(1, 3, 1, 1)
    poses[0, 2, 0, 3] = 2.0
    mask = torch.tensor([[True, False, True]])

    views = MapAnythingLocalInference._compose_input_views(
        images,
        metric_poses_c2w=poses,
        metric_pose_mask=mask,
    )

    assert "camera_poses" in views[0]
    assert "camera_poses" not in views[1]
    assert "camera_poses" in views[2]
    torch.testing.assert_close(views[2]["camera_poses"][0], poses[0, 2])
