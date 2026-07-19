"""Regression tests for incomplete global camera initialization."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import gluemap.controllers.global_merger as global_merger
from gluemap.controllers.global_merger import (
    GlobalGluer,
    _estimate_trajectory_gravity,
    _fill_missing_centers_from_priors,
    _fill_missing_centers_temporally,
    _fill_missing_rotations_temporally,
)


def test_production_graph_rejects_disconnected_confident_edges():
    extrinsics = torch.eye(4)[:3].reshape(1, 1, 3, 4).repeat(1, 2, 1, 1)
    predictions = {
        "indexes": [[0, 1]],
        "pose_scores": [torch.ones(1, 2)],
        "extrinsics": [extrinsics],
        "vis": [torch.ones(1, 2, 1)],
    }
    gluer = GlobalGluer(
        SimpleNamespace(
            valid_pose_threshold=0.05,
            is_sequential=False,
            require_complete_camera_support=True,
        )
    )
    gluer.N = 3

    with pytest.raises(RuntimeError, match="do not connect all cameras"):
        gluer._refine_graph_structure(predictions)


def test_production_graph_removes_weak_relations_after_connectivity_check(
    monkeypatch,
):
    extrinsics = torch.eye(4)[:3].reshape(1, 1, 3, 4)
    predictions = {
        "indexes": [[0, 1, 2], [1, 2]],
        "pose_scores": [
            torch.tensor([[1.0, 1.0, 0.01]]),
            torch.ones(1, 2),
        ],
        "extrinsics": [
            extrinsics.repeat(1, 3, 1, 1),
            extrinsics.repeat(1, 2, 1, 1),
        ],
        "vis": [torch.ones(1, 3, 1), torch.ones(1, 2, 1)],
    }
    gluer = GlobalGluer(
        SimpleNamespace(
            valid_pose_threshold=0.05,
            is_sequential=False,
            require_complete_camera_support=True,
        )
    )
    gluer.N = 3
    monkeypatch.setattr(gluer, "_prune_invisible_pairs", lambda *args: None)

    refined, edges = gluer._refine_graph_structure(predictions)

    assert edges == {(0, 1), (1, 2)}
    assert refined["pose_scores"][0][0, 2] == 0


def test_metric_groups_ignore_mst_scale_ratios(monkeypatch):
    predictions = {"indexes": [[0, 1], [1, 0]]}
    rotations = {0: np.eye(3), 1: np.eye(3)}
    centers = {0: np.zeros(3), 1: np.ones(3)}
    captured = {}

    monkeypatch.setattr(
        global_merger,
        "rotation_averaging_pycolmap",
        lambda *args, **kwargs: rotations,
    )
    monkeypatch.setattr(
        global_merger,
        "initialize_mst_structures",
        lambda *args, **kwargs: (centers, {0: 0.7, 1: 1.4}),
    )

    def capture_scales(*args, **kwargs):
        captured.update(kwargs["global_scales"])
        return centers

    monkeypatch.setattr(global_merger, "similarity_averaging", capture_scales)
    gluer = GlobalGluer(
        SimpleNamespace(
            valid_pose_threshold=0.05,
            is_sequential=False,
            use_ceres_rotation_averaging=False,
            fix_group_scales=True,
            require_complete_camera_support=True,
        )
    )
    gluer.N = 2
    monkeypatch.setattr(gluer, "_filter_invalid_edges", lambda *args: None)
    monkeypatch.setattr(gluer, "_prune_invisible_pairs", lambda *args: None)
    monkeypatch.setattr(gluer, "_mark_inconsistent_edges", lambda *args: None)

    gluer._global_structure_estimation(predictions)

    assert captured == {0: 1.0, 1: 1.0}


def test_rotation_filter_never_drops_mapanything_temporal_chain():
    quarter_turn = torch.from_numpy(
        Rotation.from_euler("z", 90, degrees=True).as_matrix()
    ).double()
    pose = torch.cat([quarter_turn, torch.zeros(3, 1)], dim=1)
    predictions = {
        "pose_constraints": [
            {
                "first": 0,
                "second": 1,
                "pose": pose,
                "score": 1.0,
                "kind": "mapanything_temporal",
                "active": True,
            },
            {
                "first": 0,
                "second": 1,
                "pose": pose,
                "score": 1.0,
                "kind": "mapanything_verified_loop",
                "active": True,
            },
        ]
    }
    gluer = GlobalGluer(
        SimpleNamespace(
            valid_pose_threshold=0.05,
            is_sequential=False,
            use_ceres_rotation_averaging=False,
        )
    )

    gluer._filter_invalid_edges(predictions, {0: np.eye(3), 1: np.eye(3)})

    assert predictions["pose_constraints"][0]["active"]
    assert not predictions["pose_constraints"][1]["active"]


def test_explicit_camera_graph_still_prepares_virtual_track_scores():
    predictions = {
        "indexes": [[0, 1]],
        "vis": [torch.tensor([[[0.1], [0.01]]])],
        "pose_constraints": [
            {
                "first": 0,
                "second": 1,
                "pose": torch.eye(4)[:3],
                "score": 1.0,
                "kind": "mapanything_temporal",
                "active": True,
            }
        ],
    }
    gluer = GlobalGluer(
        SimpleNamespace(valid_pose_threshold=0.05, is_sequential=False)
    )
    gluer.N = 2

    refined, edges = gluer._refine_graph_structure(predictions)

    assert edges == {(0, 1)}
    torch.testing.assert_close(
        refined["scores"][0], torch.tensor([[[0.1], [0.0]]])
    )


def test_production_explicit_graph_rejects_unsupported_camera():
    predictions = {
        "indexes": [[0, 1]],
        "vis": [torch.ones(1, 2, 1)],
        "pose_constraints": [
            {
                "first": 0,
                "second": 1,
                "pose": torch.eye(4)[:3],
                "score": 1.0,
                "kind": "mapanything_group_fragment",
                "active": True,
            }
        ],
    }
    gluer = GlobalGluer(
        SimpleNamespace(
            valid_pose_threshold=0.05,
            is_sequential=False,
            require_complete_camera_support=True,
        )
    )
    gluer.N = 3

    with pytest.raises(RuntimeError, match="do not connect all cameras"):
        gluer._refine_graph_structure(predictions)


def test_missing_centers_are_linearly_interpolated():
    centers = {
        0: np.array([0.0, 0.0, 0.0]),
        3: np.array([3.0, 6.0, 9.0]),
    }

    filled = _fill_missing_centers_temporally(centers, 4)

    np.testing.assert_allclose(filled[1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(filled[2], [2.0, 4.0, 6.0])


def test_missing_rotations_are_slerped():
    rotations = {
        0: np.eye(3),
        2: Rotation.from_euler("z", 90, degrees=True).as_matrix(),
    }

    filled = _fill_missing_rotations_temporally(rotations, 3)

    expected = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    np.testing.assert_allclose(filled[1], expected, atol=1e-7)


def test_edge_runs_copy_nearest_estimate():
    centers = {2: np.array([2.0, 4.0, 6.0])}
    rotations = {2: Rotation.from_euler("x", 15, degrees=True).as_matrix()}

    filled_centers = _fill_missing_centers_temporally(centers, 4)
    filled_rotations = _fill_missing_rotations_temporally(rotations, 4)

    for idx in (0, 1, 3):
        np.testing.assert_allclose(filled_centers[idx], centers[2])
        np.testing.assert_allclose(filled_rotations[idx], rotations[2])


def test_missing_centers_follow_curved_trajectory_prior():
    priors = []
    for center in ([0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 1, 0], [4, 0, 0]):
        pose = np.eye(4)
        pose[:3, 3] = center
        priors.append(pose)
    centers = {0: np.array([0.0, 0.0, 0.0]), 4: np.array([4.0, 0.0, 0.0])}
    rotations = {index: np.eye(3) for index in range(5)}

    filled = _fill_missing_centers_from_priors(
        centers, rotations, priors, num_images=5
    )

    np.testing.assert_allclose(filled[2], [2.0, 2.0, 0.0])
    assert filled[1][1] > 0.5
    assert filled[3][1] > 0.5


def test_gravity_is_estimated_without_changing_rotations():
    priors = [np.eye(4), np.eye(4)]
    rotations = {
        0: Rotation.from_euler("x", -5, degrees=True).as_matrix(),
        1: Rotation.from_euler("x", 5, degrees=True).as_matrix(),
    }
    originals = {key: value.copy() for key, value in rotations.items()}

    gravity_world = _estimate_trajectory_gravity(rotations, priors)

    np.testing.assert_allclose(gravity_world, [0.0, 1.0, 0.0], atol=1e-7)
    for image_id, rotation in rotations.items():
        np.testing.assert_array_equal(rotation, originals[image_id])
