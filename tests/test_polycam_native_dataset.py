import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gluemap.datasets.polycam import rotate_intrinsics_cw
from gluemap.datasets.polycam_native import PolycamNativeStarDataset


def test_native_dataset_retains_upright_calibrated_intrinsics(tmp_path: Path):
    source_root = tmp_path / "source"
    dataset_directory = tmp_path / "native"
    dataset_directory.mkdir()
    (dataset_directory / "images").mkdir()

    original_intrinsics = np.array(
        [[700.0, 0.0, 510.0], [0.0, 710.0, 380.0], [0.0, 0.0, 1.0]]
    )
    frames = []
    mapping = []
    for index in range(2):
        frames.append(
            {
                "rgb_size_wh": [1024, 768],
                "intrinsics_rgb": original_intrinsics.tolist(),
                "corrected_c2w_opencv": np.eye(4).tolist(),
                "paths": {
                    "depth": f"depth/{index}.png",
                    "confidence": f"confidence/{index}.png",
                },
            }
        )
        mapping.append(
            {"manifest_index": index, "image_name": f"frame_{index:06d}.jpg"}
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame_count": len(frames),
                "depth_unit_m": 0.001,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    (dataset_directory / "index_mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )
    frontend_edges = tmp_path / "edges.csv"
    frontend_edges.write_text(
        "first,second,score,acceptance_reason\n0,1,1.0,temporal\n",
        encoding="utf-8",
    )

    dataset = PolycamNativeStarDataset(
        SimpleNamespace(num_track_per_img=16, max_neighbors=25),
        manifest_path,
        source_root,
        dataset_directory,
        frontend_edges,
    )

    expected = rotate_intrinsics_cw(original_intrinsics, (1024, 768))
    assert len(dataset.known_intrinsics) == 2
    np.testing.assert_allclose(dataset.known_intrinsics[0][0].numpy(), expected)
    np.testing.assert_allclose(dataset.known_intrinsics[1][0].numpy(), expected)


def test_native_dataset_accepts_sparse_overlapping_group_cover(tmp_path: Path):
    source_root = tmp_path / "source"
    dataset_directory = tmp_path / "native"
    dataset_directory.mkdir()
    (dataset_directory / "images").mkdir()
    frames = []
    mapping = []
    for index in range(5):
        frames.append(
            {
                "rgb_size_wh": [1024, 768],
                "intrinsics_rgb": np.eye(3).tolist(),
                "corrected_c2w_opencv": np.eye(4).tolist(),
                "paths": {
                    "depth": f"depth/{index}.png",
                    "confidence": f"confidence/{index}.png",
                },
            }
        )
        mapping.append(
            {"manifest_index": index, "image_name": f"frame_{index:06d}.jpg"}
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame_count": len(frames),
                "depth_unit_m": 0.001,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    (dataset_directory / "index_mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )
    frontend_edges = tmp_path / "edges.csv"
    frontend_edges.write_text(
        "first,second,score,acceptance_reason\n"
        "0,1,1.0,temporal\n1,2,1.0,temporal\n"
        "2,3,1.0,temporal\n3,4,1.0,temporal\n",
        encoding="utf-8",
    )
    group_config = tmp_path / "groups.json"
    group_config.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "name": "left",
                        "anchor_frame": 1,
                        "frame_indices": [0, 1, 2],
                        "pose_conditioned_frames": [1, 2],
                    },
                    {
                        "name": "right",
                        "anchor_frame": 3,
                        "frame_indices": [2, 3, 4],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = PolycamNativeStarDataset(
        SimpleNamespace(num_track_per_img=16, max_neighbors=25),
        manifest_path,
        source_root,
        dataset_directory,
        frontend_edges,
        group_config=group_config,
    )

    assert len(dataset) == 2
    assert dataset.N == 5
    assert dataset.stars[0].tolist() == [1, 0, 2]
    assert dataset.stars[1].tolist() == [3, 2, 4]
    assert dataset.group_coverage.tolist() == [1, 1, 2, 1, 1]
    assert dataset.group_pose_conditioned_members[0] == {1, 2}
    assert dataset.group_pose_conditioned_members[1] == set()
    assert dataset.refinement_edges == [(0, 1), (1, 2), (2, 3), (3, 4)]
