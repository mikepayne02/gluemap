#!/usr/bin/env python3
"""Run conditioned MapAnything on one audited Telluride validation group."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gluemap.datasets.polycam import (
    load_polycam_manifest,
    rotate_c2w_opencv_cw,
    rotate_intrinsics_cw,
)


def _group_indices(config: dict, name: str) -> tuple[dict, list[int]]:
    matches = [group for group in config["groups"] if group["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one group named {name!r}")
    group = matches[0]
    if "frame_indices" in group:
        indices = [int(index) for index in group["frame_indices"]]
    else:
        indices = []
        for start, end in group["frame_ranges_inclusive"]:
            if end < start:
                raise ValueError(f"Invalid frame range [{start}, {end}]")
            indices.extend(range(start, end + 1))
    if len(indices) != len(set(indices)):
        raise ValueError("Group frame ranges overlap")
    expected = group.get("expected_view_count")
    if expected is not None and len(indices) != expected:
        raise ValueError(
            f"Group {name!r} has {len(indices)} views, expected {expected}"
        )
    return group, indices


def _load_view(
    frame: dict,
    source_root: Path,
    *,
    min_confidence: int,
    include_pose: bool,
    orientation: str,
) -> dict:
    paths = frame["paths"]
    with Image.open(source_root / paths["rgb"]) as image:
        rgb_image = image.convert("RGB")
    with Image.open(source_root / paths["depth"]) as image:
        depth_image = image.copy()
    with Image.open(source_root / paths["confidence"]) as image:
        confidence_image = image.copy()

    intrinsics = np.asarray(frame["intrinsics_rgb"], dtype=np.float64)
    pose = np.asarray(frame["corrected_c2w_opencv"], dtype=np.float64)
    if orientation == "upright_cw":
        source_size = rgb_image.size
        rgb_image = rgb_image.transpose(Image.Transpose.ROTATE_270)
        depth_image = depth_image.transpose(Image.Transpose.ROTATE_270)
        confidence_image = confidence_image.transpose(
            Image.Transpose.ROTATE_270
        )
        intrinsics = rotate_intrinsics_cw(intrinsics, source_size)
        pose = rotate_c2w_opencv_cw(pose)

    rgb = np.asarray(rgb_image, dtype=np.uint8)
    depth_image = depth_image.resize(
        (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
    )
    depth = np.asarray(depth_image, dtype=np.float32) * 0.001
    confidence_image = confidence_image.resize(
        (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
    )
    confidence = np.asarray(confidence_image, dtype=np.uint8)
    valid = (
        np.isfinite(depth)
        & (depth > 0.08)
        & (depth <= 6.0)
        & (confidence >= min_confidence)
    )
    measured_depth = np.where(valid, depth, 0.0).astype(np.float32)
    view = {
        "img": rgb,
        "intrinsics": intrinsics.astype(np.float32),
        "depth_z": measured_depth,
        "is_metric_scale": torch.tensor([True], dtype=torch.bool),
        "idx": [int(frame["sequence_index"])],
        "instance": [frame["frame_id"]],
    }
    if include_pose:
        view["camera_poses"] = pose.astype(np.float32)
    return view


def _cpu_prediction(prediction: dict) -> dict:
    keys = [
        "depth_z",
        "conf",
        "non_ambiguous_mask",
        "camera_poses",
        "intrinsics",
        "metric_scaling_factor",
    ]
    result = {}
    for key in keys:
        value = prediction.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if key in {"depth_z", "conf"}:
                value = value.to(torch.float16)
            elif key == "non_ambiguous_mask":
                value = value.to(torch.bool)
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("group_config", type=Path)
    parser.add_argument("group_name")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--pose-mode", choices=["corrected", "none"], required=True
    )
    parser.add_argument("--model", default="facebook/map-anything")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Override manifest source_root after copying data to a GPU host.",
    )
    parser.add_argument("--min-confidence", type=int, default=255)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--orientation",
        choices=["upright_cw", "raw"],
        default="upright_cw",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not torch.cuda.is_available():
        raise RuntimeError("This validation runner requires a CUDA GPU")
    import thirdparty.path_to_thirdparty  # noqa: F401

    mapanything_models = importlib.import_module("mapanything.models")
    mapanything_image = importlib.import_module("mapanything.utils.image")
    mapanything_class = mapanything_models.MapAnything
    preprocess_inputs = mapanything_image.preprocess_inputs

    manifest = load_polycam_manifest(args.manifest)
    config = json.loads(args.group_config.read_text(encoding="utf-8"))
    group, indices = _group_indices(config, args.group_name)
    source_root = (
        args.source_root.expanduser().resolve()
        if args.source_root is not None
        else Path(manifest["source_root"])
    )
    include_pose = args.pose_mode == "corrected"
    views = [
        _load_view(
            manifest["frames"][index],
            source_root,
            min_confidence=args.min_confidence,
            include_pose=include_pose,
            orientation=args.orientation,
        )
        for index in indices
    ]
    processed_views = preprocess_inputs(views, verbose=True)

    device = torch.device("cuda")
    model = mapanything_class.from_pretrained(args.model).to(device).eval()
    with torch.inference_mode():
        predictions = model.infer(
            processed_views,
            memory_efficient_inference=True,
            minibatch_size=args.minibatch_size,
            use_amp=True,
            amp_dtype=args.amp_dtype,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            ignore_calibration_inputs=False,
            ignore_depth_inputs=False,
            ignore_pose_inputs=not include_pose,
            ignore_depth_scale_inputs=False,
            ignore_pose_scale_inputs=not include_pose,
        )

    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "group": group,
            "frame_indices": indices,
            "pose_mode": args.pose_mode,
            "model": args.model,
            "predictions": [_cpu_prediction(pred) for pred in predictions],
        },
        output / "predictions.pt",
    )
    summary = {
        "group_name": args.group_name,
        "view_count": len(indices),
        "frame_indices": indices,
        "pose_mode": args.pose_mode,
        "model": args.model,
        "source_root": str(source_root),
        "min_confidence": args.min_confidence,
        "amp_dtype": args.amp_dtype,
        "minibatch_size": args.minibatch_size,
        "orientation": args.orientation,
        "prediction_file": str(output / "predictions.pt"),
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
