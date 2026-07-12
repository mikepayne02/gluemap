import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gluemap.datasets.polycam import (
    PolycamDatasetError,
    arkit_c2w_to_opencv,
    backproject_z_depth,
    build_polycam_manifest,
    discover_polycam_frames,
    frame_world_points,
    project_camera_points,
    scale_intrinsics,
)


def _write_frame(
    root: Path, timestamp: int, translation: tuple[float, float, float]
):
    for directory in ("cameras", "images", "depth", "confidence"):
        (root / "keyframes" / directory).mkdir(parents=True, exist_ok=True)
    tx, ty, tz = translation
    camera = {
        "timestamp": timestamp,
        "width": 8,
        "height": 6,
        "fx": 4.0,
        "fy": 4.0,
        "cx": 3.5,
        "cy": 2.5,
        "center_depth": 1.0,
        "blur_score": 10.0,
        "t_00": 1.0,
        "t_01": 0.0,
        "t_02": 0.0,
        "t_03": tx,
        "t_10": 0.0,
        "t_11": 1.0,
        "t_12": 0.0,
        "t_13": ty,
        "t_20": 0.0,
        "t_21": 0.0,
        "t_22": 1.0,
        "t_23": tz,
    }
    (root / "keyframes" / "cameras" / f"{timestamp}.json").write_text(
        json.dumps(camera), encoding="utf-8"
    )
    Image.new("RGB", (8, 6)).save(
        root / "keyframes" / "images" / f"{timestamp}.jpg"
    )
    Image.fromarray(np.full((3, 4), 1000, dtype=np.uint16)).save(
        root / "keyframes" / "depth" / f"{timestamp}.png"
    )
    Image.fromarray(np.full((3, 4), 255, dtype=np.uint8)).save(
        root / "keyframes" / "confidence" / f"{timestamp}.png"
    )


def test_projection_round_trip_for_z_depth():
    intrinsics = np.array([[400.0, 0.0, 2.0], [0.0, 420.0, 1.0], [0, 0, 1]])
    depth = np.array([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]])
    point_map = backproject_z_depth(depth, intrinsics)
    pixels, projected_depth = project_camera_points(point_map, intrinsics)
    expected_x, expected_y = np.meshgrid(np.arange(3), np.arange(2))
    np.testing.assert_allclose(pixels[..., 0], expected_x)
    np.testing.assert_allclose(pixels[..., 1], expected_y)
    np.testing.assert_allclose(projected_depth, depth)


def test_intrinsics_scaling_and_axis_change():
    intrinsics = np.array([[800.0, 0, 400.0], [0, 600.0, 300.0], [0, 0, 1]])
    scaled = scale_intrinsics(intrinsics, (800, 600), (400, 300))
    np.testing.assert_allclose(
        scaled, np.array([[400.0, 0, 200.0], [0, 300.0, 150.0], [0, 0, 1]])
    )
    converted = arkit_c2w_to_opencv(np.eye(4))
    np.testing.assert_allclose(converted, np.diag([1.0, -1.0, -1.0, 1.0]))


def test_manifest_orders_frames_and_marks_reset(tmp_path: Path):
    _write_frame(tmp_path, 300, (5.2, 0.0, 0.0))
    _write_frame(tmp_path, 100, (0.0, 0.0, 0.0))
    _write_frame(tmp_path, 200, (0.2, 0.0, 0.0))
    manifest, report = build_polycam_manifest(
        tmp_path, reset_jump_threshold_m=2.0
    )
    assert [frame["timestamp"] for frame in manifest["frames"]] == [
        100,
        200,
        300,
    ]
    assert manifest["reset_after_sequence_indices"] == [1]
    assert [frame["reset_segment"] for frame in manifest["frames"]] == [0, 0, 1]
    assert report["associations_complete"] is True
    assert report["confidence_value_counts"] == {"255": 36}
    np.testing.assert_allclose(
        manifest["frames"][0]["intrinsics_depth"],
        np.array([[2.0, 0, 1.75], [0, 2.0, 1.25], [0, 0, 1]]),
    )


def test_discovery_rejects_missing_modality(tmp_path: Path):
    _write_frame(tmp_path, 100, (0.0, 0.0, 0.0))
    (tmp_path / "keyframes" / "confidence" / "100.png").unlink()
    with pytest.raises(PolycamDatasetError, match="association failed"):
        discover_polycam_frames(tmp_path)


def test_frame_world_points_uses_opencv_pose(tmp_path: Path):
    _write_frame(tmp_path, 100, (1.0, 2.0, 3.0))
    manifest, _ = build_polycam_manifest(tmp_path)
    points, colors = frame_world_points(
        manifest["frames"][0], tmp_path, pixel_step=1, min_confidence=255
    )
    assert points.shape == (12, 3)
    assert colors.shape == (12, 3)
    # The depth image is one metre. OpenCV +Y and +Z are flipped relative to
    # the identity ARKit camera, while the camera center remains unchanged.
    np.testing.assert_allclose(points[:, 2], 2.0)
