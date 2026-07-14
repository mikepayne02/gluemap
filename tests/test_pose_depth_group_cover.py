from gluemap.pairing.pose_depth_graph import (
    build_graph_groups,
    build_verified_bridge_groups,
    frontend_edge_is_group_evidence,
)


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
