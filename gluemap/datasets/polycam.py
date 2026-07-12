"""Polycam RGB-D dataset discovery, manifest generation, and geometry audits."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class PolycamDatasetError(ValueError):
    """Raised when a Polycam export is incomplete or internally inconsistent."""


_MODALITIES = {
    "camera": ("cameras", ".json"),
    "image": ("images", ".jpg"),
    "depth": ("depth", ".png"),
    "confidence": ("confidence", ".png"),
}


def _files_by_timestamp(directory: Path, suffix: str) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        try:
            timestamp = int(path.stem)
        except ValueError as exc:
            raise PolycamDatasetError(
                f"Expected a numeric Polycam timestamp, got {path.name!r}"
            ) from exc
        if timestamp in files:
            raise PolycamDatasetError(
                f"Duplicate timestamp {timestamp} in {directory}"
            )
        files[timestamp] = path
    return files


def discover_polycam_frames(root: str | Path) -> list[dict[str, Path | int]]:
    """Discover complete Polycam keyframes in chronological timestamp order."""
    root = Path(root).expanduser().resolve()
    keyframes = root / "keyframes"
    if not keyframes.is_dir():
        raise PolycamDatasetError(
            f"Missing Polycam keyframes directory: {keyframes}"
        )

    indexed: dict[str, dict[int, Path]] = {}
    for name, (dirname, suffix) in _MODALITIES.items():
        directory = keyframes / dirname
        if not directory.is_dir():
            raise PolycamDatasetError(
                f"Missing Polycam {name} directory: {directory}"
            )
        indexed[name] = _files_by_timestamp(directory, suffix)

    timestamps = set(indexed["camera"])
    mismatch_messages = []
    for name, files in indexed.items():
        missing = sorted(timestamps - set(files))
        extra = sorted(set(files) - timestamps)
        if missing or extra:
            counts = f"{len(missing)} missing and {len(extra)} extra"
            mismatch_messages.append(f"{name}: {counts} timestamps")
    if mismatch_messages:
        raise PolycamDatasetError(
            "Polycam modality association failed: "
            + "; ".join(mismatch_messages)
        )
    if not timestamps:
        raise PolycamDatasetError(f"No camera records found under {keyframes}")

    return [
        {
            "timestamp": timestamp,
            **{name: files[timestamp] for name, files in indexed.items()},
        }
        for timestamp in sorted(timestamps)
    ]


def camera_to_world_arkit(camera: dict[str, Any]) -> np.ndarray:
    """Read Polycam's 3x4 ARKit camera-to-world matrix as homogeneous 4x4."""
    matrix = np.eye(4, dtype=np.float64)
    try:
        for row in range(3):
            for column in range(4):
                matrix[row, column] = float(camera[f"t_{row}{column}"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolycamDatasetError(
            "Malformed t_00..t_23 camera transform"
        ) from exc
    if not np.isfinite(matrix).all():
        raise PolycamDatasetError("Camera transform contains non-finite values")
    return matrix


def arkit_c2w_to_opencv(c2w_arkit: np.ndarray) -> np.ndarray:
    """Change camera axes from ARKit/OpenGL to OpenCV, preserving world axes."""
    axis_change = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.asarray(c2w_arkit, dtype=np.float64) @ axis_change


def scale_intrinsics(
    intrinsics: np.ndarray,
    source_size_wh: tuple[int, int],
    target_size_wh: tuple[int, int],
) -> np.ndarray:
    """Scale a 3x3 pinhole calibration between uncropped image resolutions."""
    source_width, source_height = source_size_wh
    target_width, target_height = target_size_wh
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be positive")
    result = np.asarray(intrinsics, dtype=np.float64).copy()
    result[0, :] *= target_width / source_width
    result[1, :] *= target_height / source_height
    return result


def backproject_z_depth(
    depth_z: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Back-project Z-depth to an ``(H, W, 3)`` camera-space point map."""
    depth_z = np.asarray(depth_z, dtype=np.float64)
    if depth_z.ndim != 2:
        raise ValueError("depth_z must have shape (H, W)")
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    height, width = depth_z.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    points = np.empty((height, width, 3), dtype=np.float64)
    points[..., 0] = (x - intrinsics[0, 2]) * depth_z / intrinsics[0, 0]
    points[..., 1] = (y - intrinsics[1, 2]) * depth_z / intrinsics[1, 1]
    points[..., 2] = depth_z
    return points


def project_camera_points(
    points_camera: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-space points, returning image coordinates and Z depth."""
    points_camera = np.asarray(points_camera, dtype=np.float64)
    if points_camera.shape[-1] != 3:
        raise ValueError("points_camera must end with XYZ coordinates")
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    z = points_camera[..., 2]
    if np.any(z <= 0):
        raise ValueError(
            "All projected points must have positive camera-space Z"
        )
    pixels = np.empty(points_camera.shape[:-1] + (2,), dtype=np.float64)
    pixels[..., 0] = (
        intrinsics[0, 0] * points_camera[..., 0] / z + intrinsics[0, 2]
    )
    pixels[..., 1] = (
        intrinsics[1, 1] * points_camera[..., 1] / z + intrinsics[1, 2]
    )
    return pixels, z


def _intrinsics(camera: dict[str, Any]) -> np.ndarray:
    try:
        fx = float(camera["fx"])
        fy = float(camera["fy"])
        cx = float(camera["cx"])
        cy = float(camera["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolycamDatasetError(
            "Malformed fx/fy/cx/cy camera intrinsics"
        ) from exc
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    if not np.isfinite(matrix).all() or fx <= 0 or fy <= 0:
        raise PolycamDatasetError("Camera intrinsics are invalid")
    return matrix


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def build_polycam_manifest(
    root: str | Path,
    *,
    reset_jump_threshold_m: float = 2.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a canonical manifest and calibration report from a raw export."""
    if reset_jump_threshold_m <= 0:
        raise ValueError("reset_jump_threshold_m must be positive")
    root = Path(root).expanduser().resolve()
    discovered = discover_polycam_frames(root)

    records: list[dict[str, Any]] = []
    centers = []
    rotations = []
    rgb_intrinsics = []
    center_depth_residuals = []
    confidence_counts: Counter[int] = Counter()
    rgb_sizes: Counter[tuple[int, int]] = Counter()
    depth_sizes: Counter[tuple[int, int]] = Counter()
    confidence_sizes: Counter[tuple[int, int]] = Counter()

    for index, paths in enumerate(discovered):
        camera_path = Path(paths["camera"])
        with camera_path.open("r", encoding="utf-8") as handle:
            camera = json.load(handle)
        timestamp = int(paths["timestamp"])
        if int(camera.get("timestamp", timestamp)) != timestamp:
            raise PolycamDatasetError(
                f"Camera timestamp does not match filename: {camera_path}"
            )

        c2w_arkit = camera_to_world_arkit(camera)
        rotation = c2w_arkit[:3, :3]
        centers.append(c2w_arkit[:3, 3])
        rotations.append(rotation)

        image_path = Path(paths["image"])
        depth_path = Path(paths["depth"])
        confidence_path = Path(paths["confidence"])
        with Image.open(image_path) as image:
            rgb_size = tuple(image.size)
        with Image.open(depth_path) as image:
            depth = np.asarray(image, dtype=np.float64)
            depth_size = tuple(image.size)
        with Image.open(confidence_path) as image:
            confidence = np.asarray(image, dtype=np.uint8)
            confidence_size = tuple(image.size)

        camera_size = (int(camera["width"]), int(camera["height"]))
        if rgb_size != camera_size:
            raise PolycamDatasetError(
                f"RGB size {rgb_size} disagrees with camera size {camera_size} "
                f"at timestamp {timestamp}"
            )
        if confidence_size != depth_size:
            raise PolycamDatasetError(
                f"Depth/confidence size mismatch at timestamp {timestamp}"
            )
        rgb_sizes[rgb_size] += 1
        depth_sizes[depth_size] += 1
        confidence_sizes[confidence_size] += 1
        unique_conf, counts = np.unique(confidence, return_counts=True)
        confidence_counts.update(
            {
                int(value): int(count)
                for value, count in zip(unique_conf, counts, strict=True)
            }
        )

        intrinsics_rgb = _intrinsics(camera)
        intrinsics_depth = scale_intrinsics(
            intrinsics_rgb, rgb_size, depth_size
        )
        rgb_intrinsics.append(
            [
                intrinsics_rgb[0, 0],
                intrinsics_rgb[1, 1],
                intrinsics_rgb[0, 2],
                intrinsics_rgb[1, 2],
            ]
        )

        center_depth_m = camera.get("center_depth")
        if center_depth_m is not None:
            center_y = depth.shape[0] // 2
            center_x = depth.shape[1] // 2
            measured_center_m = float(depth[center_y, center_x]) * 0.001
            if measured_center_m > 0:
                center_depth_residuals.append(
                    measured_center_m - float(center_depth_m)
                )

        records.append(
            {
                "frame_id": f"{index:06d}",
                "sequence_index": index,
                "timestamp": timestamp,
                "paths": {
                    "rgb": _relative_path(image_path, root),
                    "depth": _relative_path(depth_path, root),
                    "confidence": _relative_path(confidence_path, root),
                    "camera": _relative_path(camera_path, root),
                },
                "rgb_size_wh": list(rgb_size),
                "depth_size_wh": list(depth_size),
                "intrinsics_rgb": intrinsics_rgb.tolist(),
                "intrinsics_depth": intrinsics_depth.tolist(),
                "raw_c2w_arkit": c2w_arkit.tolist(),
                "raw_c2w_opencv": arkit_c2w_to_opencv(c2w_arkit).tolist(),
                "center_depth_m": (
                    None if center_depth_m is None else float(center_depth_m)
                ),
                "blur_score": (
                    None
                    if camera.get("blur_score") is None
                    else float(camera["blur_score"])
                ),
            }
        )

    centers_array = np.asarray(centers)
    rotations_array = np.asarray(rotations)
    step_distances = np.linalg.norm(np.diff(centers_array, axis=0), axis=1)
    reset_after = np.flatnonzero(
        step_distances > reset_jump_threshold_m
    ).tolist()
    reset_after_set = set(reset_after)
    segment = 0
    for index, record in enumerate(records):
        record["reset_segment"] = segment
        if index in reset_after_set:
            segment += 1

    rotation_orthogonality = np.linalg.norm(
        rotations_array @ np.swapaxes(rotations_array, -1, -2) - np.eye(3),
        axis=(1, 2),
    )
    rotation_determinants = np.linalg.det(rotations_array)
    intrinsics_array = np.asarray(rgb_intrinsics)

    manifest = {
        "schema_version": 1,
        "dataset_type": "polycam_arkit_rgbd",
        "source_root": str(root),
        "frame_count": len(records),
        "depth_unit_m": 0.001,
        "depth_semantics": "z_depth_assumed_pending_multiview_validation",
        "pose_conventions": {
            "raw_c2w_arkit": "ARKit/OpenGL camera-to-world",
            "raw_c2w_opencv": (
                "raw_c2w_arkit @ diag(1,-1,-1,1); OpenCV camera axes"
            ),
        },
        "reset_jump_threshold_m": reset_jump_threshold_m,
        "reset_after_sequence_indices": reset_after,
        "frames": records,
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "frame_count": len(records),
        "associations_complete": True,
        "rgb_sizes_wh": {
            f"{w}x{h}": count for (w, h), count in rgb_sizes.items()
        },
        "depth_sizes_wh": {
            f"{w}x{h}": count for (w, h), count in depth_sizes.items()
        },
        "confidence_sizes_wh": {
            f"{w}x{h}": count for (w, h), count in confidence_sizes.items()
        },
        "confidence_value_counts": {
            str(value): count
            for value, count in sorted(confidence_counts.items())
        },
        "intrinsics_rgb_fx_fy_cx_cy": {
            "median": np.median(intrinsics_array, axis=0).tolist(),
            "min": np.min(intrinsics_array, axis=0).tolist(),
            "max": np.max(intrinsics_array, axis=0).tolist(),
        },
        "camera_center_span_xyz_m": np.ptp(centers_array, axis=0).tolist(),
        "camera_step_m": _percentiles(step_distances)
        if len(step_distances)
        else None,
        "reset_jump_threshold_m": reset_jump_threshold_m,
        "reset_after_sequence_indices": reset_after,
        "reset_steps_m": [
            float(step_distances[index]) for index in reset_after
        ],
        "rotation_orthogonality_error": _percentiles(rotation_orthogonality),
        "rotation_determinant": _percentiles(rotation_determinants),
        "depth_unit_evidence": {
            "png_integer_to_m_scale": 0.001,
            "center_pixel_minus_camera_center_depth_m": (
                _percentiles(np.abs(center_depth_residuals))
                if center_depth_residuals
                else None
            ),
            "samples": len(center_depth_residuals),
        },
        "unresolved_checks": [
            "Confirm RGB/depth orientation and crop with overlay diagnostics.",
            "Confirm Z-depth versus range-along-ray using multi-view geometry.",
            (
                "Estimate and compare gravity independently in every reset "
                "segment."
            ),
        ],
    }
    return manifest, report


def write_polycam_audit(
    root: str | Path,
    output_directory: str | Path,
    *,
    reset_jump_threshold_m: float = 2.0,
) -> tuple[Path, Path]:
    """Generate ``manifest.json`` and ``calibration_report.json``."""
    manifest, report = build_polycam_manifest(
        root, reset_jump_threshold_m=reset_jump_threshold_m
    )
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    report_path = output_directory / "calibration_report.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path, report_path


def load_polycam_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a manifest produced by this module."""
    path = Path(path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise PolycamDatasetError("Unsupported Polycam manifest schema")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != manifest.get(
        "frame_count"
    ):
        raise PolycamDatasetError("Manifest frame count is inconsistent")
    return manifest


def frame_world_points(
    frame: dict[str, Any],
    source_root: str | Path,
    *,
    pixel_step: int = 1,
    min_confidence: int = 255,
    min_depth_m: float = 0.08,
    max_depth_m: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one manifest frame's measured depth and RGB into world space."""
    if pixel_step <= 0:
        raise ValueError("pixel_step must be positive")
    source_root = Path(source_root)
    paths = frame["paths"]
    with Image.open(source_root / paths["depth"]) as image:
        depth = np.asarray(image, dtype=np.float32) * 0.001
    with Image.open(source_root / paths["confidence"]) as image:
        confidence = np.asarray(image, dtype=np.uint8)
    with Image.open(source_root / paths["rgb"]) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    y, x = np.mgrid[
        0 : depth.shape[0] : pixel_step, 0 : depth.shape[1] : pixel_step
    ]
    z = depth[y, x]
    valid = (
        np.isfinite(z)
        & (z >= min_depth_m)
        & (z <= max_depth_m)
        & (confidence[y, x] >= min_confidence)
    )
    if not np.any(valid):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    intrinsics = np.asarray(frame["intrinsics_depth"], dtype=np.float64)
    sampled_depth = np.zeros_like(z, dtype=np.float64)
    sampled_depth[valid] = z[valid]
    point_map = backproject_z_depth(sampled_depth, intrinsics)
    points_camera = point_map[valid]
    c2w = np.asarray(frame["raw_c2w_opencv"], dtype=np.float64)
    points_world = (points_camera @ c2w[:3, :3].T + c2w[:3, 3]).astype(
        np.float32
    )

    rgb_height, rgb_width = rgb.shape[:2]
    depth_height, depth_width = depth.shape
    rgb_x = np.clip(
        np.round((x[valid] + 0.5) * rgb_width / depth_width - 0.5).astype(int),
        0,
        rgb_width - 1,
    )
    rgb_y = np.clip(
        np.round((y[valid] + 0.5) * rgb_height / depth_height - 0.5).astype(
            int
        ),
        0,
        rgb_height - 1,
    )
    return points_world, rgb[rgb_y, rgb_x]
