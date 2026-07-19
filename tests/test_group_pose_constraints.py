import torch

from gluemap.estimators.group_pose_constraints import (
    build_group_fragment_constraints,
    build_group_pose_constraints,
)


def _group(image_ids, centers, scores=None):
    extrinsics = torch.eye(4, dtype=torch.float64).repeat(
        1, len(image_ids), 1, 1
    )
    for position, center in enumerate(centers):
        extrinsics[0, position, :3, 3] = -torch.tensor(
            center, dtype=torch.float64
        )
    return extrinsics, torch.tensor(
        [scores or [1.0] * len(image_ids)], dtype=torch.float64
    )


def _predictions(groups):
    return {
        "indexes": [group[0] for group in groups],
        "extrinsics": [group[1] for group in groups],
        "pose_scores": [group[2] for group in groups],
    }


def test_constraints_use_cross_group_consensus_for_temporal_motion():
    good_a = _group([0, 1, 2], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    good_b = _group(
        [0, 1, 2], [[0, 0, 0], [1.02, 0, 0], [2.01, 0, 0]]
    )
    bad = _group([0, 1, 2], [[0, 0, 0], [5, 0, 0], [10, 0, 0]])
    predictions = _predictions(
        [
            ([0, 1, 2], *good_a),
            ([0, 1, 2], *good_b),
            ([0, 1, 2], *bad),
        ]
    )

    constraints = build_group_pose_constraints(
        predictions, num_images=3, trusted_loop_edges=set()
    )

    assert [(item["first"], item["second"]) for item in constraints] == [
        (0, 1),
        (1, 2),
    ]
    assert all(item["kind"] == "mapanything_temporal" for item in constraints)
    assert all(item["star_index"] in {0, 1} for item in constraints)
    assert all(item["agreement_fraction"] > 0.5 for item in constraints)


def test_low_pose_score_does_not_remove_consecutive_camera_measurement():
    group = _group(
        [0, 1, 2],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        scores=[1.0, 0.0, 0.0],
    )
    predictions = _predictions([([0, 1, 2], *group)])

    constraints = build_group_pose_constraints(
        predictions, num_images=3, trusted_loop_edges=set()
    )

    assert len(constraints) == 2
    assert all(item["active"] for item in constraints)
    assert all(item["score"] > 0 for item in constraints)


def test_only_explicitly_trusted_nonlocal_pairs_become_loop_constraints():
    group = _group(
        [0, 1, 2, 3],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
    )
    predictions = _predictions([([0, 1, 2, 3], *group)])

    constraints = build_group_pose_constraints(
        predictions, num_images=4, trusted_loop_edges={(0, 3)}
    )

    loops = [
        item
        for item in constraints
        if item["kind"] == "mapanything_verified_loop"
    ]
    assert [(item["first"], item["second"]) for item in loops] == [(0, 3)]


def test_missing_temporal_pair_fails_before_global_assembly():
    group = _group([0, 2], [[0, 0, 0], [2, 0, 0]])
    predictions = _predictions([([0, 2], *group)])

    try:
        build_group_pose_constraints(
            predictions, num_images=3, trusted_loop_edges=set()
        )
    except RuntimeError as error:
        assert "consecutive cameras" in str(error)
    else:
        raise AssertionError("missing temporal support must fail")


def test_preferred_group_overrides_bad_cross_group_majority():
    bad_a = _group([0, 1], [[0, 0, 0], [5, 0, 0]])
    bad_b = _group([0, 1], [[0, 0, 0], [5.1, 0, 0]])
    recovered = _group([0, 1], [[0, 0, 0], [0.1, 0, 0]])
    predictions = _predictions(
        [([0, 1], *bad_a), ([0, 1], *bad_b), ([0, 1], *recovered)]
    )

    constraints = build_group_pose_constraints(
        predictions,
        num_images=2,
        trusted_loop_edges=set(),
        preferred_temporal_groups={(0, 1): {2}},
    )

    assert constraints[0]["star_index"] == 2
    assert constraints[0]["candidate_count"] == 3
    assert constraints[0]["selection_candidate_count"] == 1
    assert constraints[0]["preferred_group_override"]


def test_missing_preferred_group_measurement_fails():
    group = _group([0, 1], [[0, 0, 0], [1, 0, 0]])
    predictions = _predictions([([0, 1], *group)])

    try:
        build_group_pose_constraints(
            predictions,
            num_images=2,
            trusted_loop_edges=set(),
            preferred_temporal_groups={(0, 1): {4}},
        )
    except RuntimeError as error:
        assert "does not predict" in str(error)
    else:
        raise AssertionError("missing preferred measurement must fail")


def test_fragment_constraints_preserve_every_group_long_baseline():
    first = _group(
        [0, 1, 2], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    )
    second = _group(
        [1, 2, 3], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    )
    predictions = _predictions(
        [([0, 1, 2], *first), ([1, 2, 3], *second)]
    )

    constraints = build_group_fragment_constraints(
        predictions, num_images=4
    )

    assert [(item["first"], item["second"]) for item in constraints] == [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
    ]
    assert all(
        item["kind"] == "mapanything_group_fragment"
        for item in constraints
    )
    assert all(item["score"] > 0 for item in constraints)


def test_fragment_constraints_keep_support_for_nonfinite_pose_score():
    group = _group(
        [0, 1, 2],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        scores=[1.0, float("nan"), float("inf")],
    )
    constraints = build_group_fragment_constraints(
        _predictions([([0, 1, 2], *group)]), num_images=3
    )

    assert len(constraints) == 2
    assert all(
        torch.isfinite(torch.tensor(item["score"])) for item in constraints
    )
    assert all(item["score"] > 0 for item in constraints)
