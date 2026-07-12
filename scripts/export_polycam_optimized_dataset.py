#!/usr/bin/env python3
"""Export optimized upright cameras as Nerfstudio and COLMAP datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gluemap.datasets.polycam import load_polycam_manifest


_OPENCV_FROM_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])
_RAW_CAMERA_FROM_UPRIGHT_CAMERA = np.eye(4, dtype=np.float64)
_RAW_CAMERA_FROM_UPRIGHT_CAMERA[:3, :3] = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _colmap_qvec(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("optimized_cameras", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--materialize-images", action="store_true")
    parser.add_argument("--point-cloud", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_polycam_manifest(args.manifest)
    optimized = {
        int(index): np.asarray(pose, dtype=np.float64)
        for index, pose in json.loads(
            args.optimized_cameras.read_text(encoding="utf-8")
        ).items()
    }
    output = args.output_directory.resolve()
    images_output = output / "images"
    sparse_output = output / "sparse" / "0"
    sparse_output.mkdir(parents=True, exist_ok=True)

    transforms_frames = []
    camera_lines = ["# Camera list", f"# Number of cameras: {len(optimized)}"]
    image_lines = ["# Image list", f"# Number of images: {len(optimized)}"]
    for image_id, frame_index in enumerate(sorted(optimized), start=1):
        frame = manifest["frames"][frame_index]
        upright_c2w_opencv = optimized[frame_index]
        raw_c2w_opencv = upright_c2w_opencv @ np.linalg.inv(
            _RAW_CAMERA_FROM_UPRIGHT_CAMERA
        )
        raw_c2w_opengl = raw_c2w_opencv @ _OPENCV_FROM_OPENGL
        intrinsics = np.asarray(frame["intrinsics_rgb"], dtype=np.float64)
        width, height = frame["rgb_size_wh"]
        image_name = Path(frame["paths"]["rgb"]).name
        source_image = args.source_root / frame["paths"]["rgb"]
        target_image = images_output / image_name
        if args.materialize_images:
            _link_or_copy(source_image, target_image)

        transforms_frames.append(
            {
                "file_path": f"images/{image_name}",
                "transform_matrix": raw_c2w_opengl.tolist(),
                "fl_x": float(intrinsics[0, 0]),
                "fl_y": float(intrinsics[1, 1]),
                "cx": float(intrinsics[0, 2]),
                "cy": float(intrinsics[1, 2]),
                "w": int(width),
                "h": int(height),
            }
        )
        camera_lines.append(
            f"{image_id} PINHOLE {width} {height} "
            f"{intrinsics[0, 0]} {intrinsics[1, 1]} "
            f"{intrinsics[0, 2]} {intrinsics[1, 2]}"
        )
        world_to_camera = np.linalg.inv(raw_c2w_opencv)
        qvec = _colmap_qvec(world_to_camera[:3, :3])
        tvec = world_to_camera[:3, 3]
        image_lines.append(
            f"{image_id} {' '.join(map(str, qvec))} "
            f"{' '.join(map(str, tvec))} {image_id} {image_name}"
        )
        image_lines.append("")

    transforms = {
        "camera_model": "OPENCV",
        "frames": transforms_frames,
    }
    if args.point_cloud is not None:
        point_cloud_output = output / "sparse_pc.ply"
        _link_or_copy(args.point_cloud.resolve(), point_cloud_output)
        transforms["ply_file_path"] = point_cloud_output.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "transforms.json").write_text(
        json.dumps(transforms, indent=2) + "\n", encoding="utf-8"
    )
    (sparse_output / "cameras.txt").write_text(
        "\n".join(camera_lines) + "\n", encoding="utf-8"
    )
    (sparse_output / "images.txt").write_text(
        "\n".join(image_lines) + "\n", encoding="utf-8"
    )
    (sparse_output / "points3D.txt").write_text(
        "# Empty until neural/depth points are finalized\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "frames": len(transforms_frames),
                "images_materialized": args.materialize_images,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
