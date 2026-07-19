from gluemap.pairing.pose_depth_graph import (
    apply_group_recovery_policy,
    build_graph_groups,
    build_verified_bridge_groups,
    exclude_group_evidence_pairs,
    frontend_edge_is_group_evidence,
)


def test_explicitly_rejected_evidence_never_reaches_group_builder():
    edges = [
        {"first": 1, "second": 2, "score": 1.0},
        {"first": 2, "second": 8, "score": 1.0},
    ]

    kept, rejected = exclude_group_evidence_pairs(edges, {(8, 2)})

    assert kept == [{"first": 1, "second": 2, "score": 1.0}]
    assert rejected == 1


def test_non_temporal_group_edges_require_image_evidence():
    assert frontend_edge_is_group_evidence({"acceptance_reason": "temporal"})
    assert frontend_edge_is_group_evidence(
        {"acceptance_reason": "manually_verified"}
    )
    assert frontend_edge_is_group_evidence(
        {
            "acceptance_reason": "strong_reciprocal_depth",
            "visual_inliers": "20",
            "visual_inlier_ratio": "0.5",
        }
    )
    assert not frontend_edge_is_group_evidence(
        {
            "acceptance_reason": "strong_reciprocal_depth",
            "visual_inliers": "",
            "visual_inlier_ratio": "",
        }
    )


def test_verified_revisits_are_jointly_observed_by_bridge_groups():
    indices = list(range(400))
    verified = [
        {"first": 100, "second": 300},
        {"first": 101, "second": 300},
    ]

    groups = build_verified_bridge_groups(indices, verified, group_size=64)

    assert len(groups) == 1
    members = set(groups[0]["frame_indices"])
    assert len(members) == 64
    assert {100, 101, 300} <= members
    assert groups[0]["verified_correspondences"] == [
        [100, 300],
        [101, 300],
    ]


def test_seeded_bridge_is_preserved_in_complete_cover():
    indices = list(range(96))
    edges = [
        {"first": index, "second": index + 1, "score": 1.0}
        for index in range(95)
    ]
    bridge = {
        "name": "bridge",
        "frame_indices": list(range(16)) + list(range(80, 96)),
    }

    groups = build_graph_groups(
        indices,
        edges,
        group_size=32,
        minimum_memberships=1,
        seeded_groups=[bridge],
    )

    assert groups[0]["name"] == "bridge"
    assert len(groups[0]["frame_indices"]) == 32
    covered = {frame for group in groups for frame in group["frame_indices"]}
    assert covered == set(indices)


def test_recovery_policy_adds_balanced_group_without_changing_base_cover():
    groups = [
        {"name": "keep", "frame_indices": [0, 1]},
        {"name": "also_keep", "frame_indices": [2, 3]},
    ]
    policy = {
        "schema_version": 1,
        "layout": "balanced_pair_context",
        "views_per_side": 4,
        "pose_conditioning": {"mode": "none"},
        "candidates": [
            {
                "name": "recovery",
                "evidence_pairs": [[2, 13], [3, 13]],
                "anchor_frame": 3,
            }
        ],
    }

    additions = apply_group_recovery_policy(
        groups,
        policy,
        set(range(16)),
    )

    assert groups[0] == {"name": "keep", "frame_indices": [0, 1]}
    assert groups[1] == {"name": "also_keep", "frame_indices": [2, 3]}
    assert groups[2]["kind"] == "balanced_revisit_recovery"
    assert groups[2]["anchor_frame"] == 3
    assert groups[2]["frame_indices"] == [0, 1, 2, 3, 11, 12, 13, 14]
    assert groups[2]["verified_correspondences"] == [[2, 13], [3, 13]]
    assert groups[2]["pose_conditioned_frames"] == []
    assert additions[0]["view_count"] == 8
    assert additions[0]["visit_ranges"] == [[0, 3], [11, 14]]


def test_recovery_pose_conditioning_is_policy_controlled():
    groups = []
    policy = {
        "schema_version": 1,
        "layout": "balanced_pair_context",
        "views_per_side": 4,
        "pose_conditioning": {
            "mode": "sparse_local",
            "anchor_radius": 2,
            "max_views": 3,
        },
        "candidates": [
            {
                "name": "recovery",
                "evidence_pairs": [[3, 12]],
                "anchor_frame": 3,
            }
        ],
    }

    apply_group_recovery_policy(groups, policy, set(range(16)))

    assert groups[0]["pose_conditioned_frames"] == [3, 1, 4]
    assert groups[0]["pose_modes"] == ["sparse_local"]


def test_recovery_rejects_lopsided_or_overlapping_visit_contexts():
    policy = {
        "schema_version": 1,
        "layout": "balanced_pair_context",
        "views_per_side": 4,
        "pose_conditioning": {"mode": "none"},
        "candidates": [
            {"name": "bad", "evidence_pairs": [[4, 6]]},
        ],
    }

    try:
        apply_group_recovery_policy([], policy, set(range(12)))
    except ValueError as error:
        assert "contexts overlap" in str(error)
    else:
        raise AssertionError("overlapping visit contexts must be rejected")


def test_local_recovery_records_depth_exclusions_and_preferred_range():
    groups = []
    policy = {
        "schema_version": 1,
        "layout": "balanced_pair_context",
        "views_per_side": 4,
        "pose_conditioning": {"mode": "none"},
        "candidates": [],
        "local_candidates": [
            {
                "name": "duplicate_recovery",
                "center_frame": 8,
                "view_count": 8,
                "depth_excluded_frames": [8, 9],
                "authoritative_temporal_ranges": [[7, 10]],
            }
        ],
    }

    additions = apply_group_recovery_policy(groups, policy, set(range(16)))

    assert groups[0]["kind"] == "local_temporal_recovery"
    assert groups[0]["frame_indices"] == list(range(4, 12))
    assert groups[0]["depth_excluded_frames"] == [8, 9]
    assert groups[0]["authoritative_temporal_ranges"] == [[7, 10]]
    assert groups[0]["pose_conditioned_frames"] == []
    assert additions[0]["temporal_range"] == [4, 11]


def test_local_recovery_rejects_range_outside_group():
    policy = {
        "schema_version": 1,
        "layout": "balanced_pair_context",
        "views_per_side": 4,
        "pose_conditioning": {"mode": "none"},
        "candidates": [],
        "local_candidates": [
            {
                "name": "bad",
                "center_frame": 8,
                "view_count": 4,
                "authoritative_temporal_ranges": [[7, 12]],
            }
        ],
    }

    try:
        apply_group_recovery_policy([], policy, set(range(16)))
    except ValueError as error:
        assert "invalid temporal range" in str(error)
    else:
        raise AssertionError("out-of-group temporal range must be rejected")


def test_local_recovery_can_condition_its_complete_window():
    groups = []
    policy = {
        "schema_version": 1,
        "layout": "balanced_pair_context",
        "views_per_side": 4,
        "pose_conditioning": {"mode": "none"},
        "candidates": [],
        "local_candidates": [
            {
                "name": "full_pose_recovery",
                "center_frame": 8,
                "view_count": 4,
                "pose_conditioning": {"mode": "full_local"},
            }
        ],
    }

    apply_group_recovery_policy(groups, policy, set(range(16)))

    assert groups[0]["frame_indices"] == [6, 7, 8, 9]
    assert groups[0]["pose_conditioned_frames"] == [6, 7, 8, 9]
    assert groups[0]["pose_modes"] == ["full_local"]
