import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gluemap.controllers.gluemap_impl import _build_aligned_pose_priors
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


def test_pose_priors_are_aligned_to_native_similarity_gauge():
    angle = np.deg2rad(25.0)
    alignment_rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    alignment_scale = 1.7
    alignment_translation = np.array([4.0, -2.0, 3.0])
    source_centers = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.5]]
    )
    arkit_poses = []
    global_centers = {}
    for index, center in enumerate(source_centers):
        pose = np.eye(4)
        pose[:3, 3] = center
        arkit_poses.append(pose)
        global_centers[index] = (
            alignment_scale * (alignment_rotation @ center)
            + alignment_translation
        )

    names = [f"frame_{index}.jpg" for index in range(3)]
    priors = _build_aligned_pose_priors(
        arkit_poses,
        global_centers,
        names,
        position_sigma_m=0.2,
        rotation_sigma_deg=3.0,
    )

    for index, name in enumerate(names):
        np.testing.assert_allclose(
            priors[name]["center"], global_centers[index], atol=1e-12
        )
        np.testing.assert_allclose(
            priors[name]["cam_from_world_rotation"],
            alignment_rotation.T,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            priors[name]["center_sigma"], 0.2 * alignment_scale
        )
