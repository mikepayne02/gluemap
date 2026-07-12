#!/usr/bin/env python3
"""Run a resumable pose-free MapAnything group batch with one model load."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from gluemap.datasets.polycam import (
    load_polycam_manifest,
    rotate_c2w_opencv_cw,
)
from run_polycam_mapanything_group import (
    _cpu_prediction,
    _group_indices,
    _load_view,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("group_config", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", default="facebook/map-anything")
    parser.add_argument("--min-confidence", type=int, default=255)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--orientation", choices=["upright_cw", "raw"], default="upright_cw")
    parser.add_argument("--group", action="append", dest="groups")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not torch.cuda.is_available():
        raise RuntimeError("This batch runner requires a CUDA GPU")
    import thirdparty.path_to_thirdparty  # noqa: F401

    mapanything_models = importlib.import_module("mapanything.models")
    preprocess_inputs = importlib.import_module(
        "mapanything.utils.image"
    ).preprocess_inputs
    manifest = load_polycam_manifest(args.manifest)
    config = json.loads(args.group_config.read_text(encoding="utf-8"))
    selected = [
        group
        for group in config["groups"]
        if args.groups is None or group["name"] in args.groups
    ]
    if args.groups is not None and len(selected) != len(set(args.groups)):
        raise ValueError("One or more requested groups are absent or duplicated")

    output_root = args.output_directory.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = mapanything_models.MapAnything.from_pretrained(args.model).to(device).eval()
    batch_start = time.perf_counter()
    summaries = []

    for ordinal, group_spec in enumerate(selected, start=1):
        name = group_spec["name"]
        group_output = output_root / name
        prediction_path = group_output / "predictions.pt"
        if prediction_path.is_file():
            print(f"[{ordinal}/{len(selected)}] skipping completed {name}", flush=True)
            summaries.append({"group_name": name, "status": "skipped"})
            continue

        group, indices = _group_indices(config, name)
        print(f"[{ordinal}/{len(selected)}] running {name} ({len(indices)} views)", flush=True)
        views = [
            _load_view(
                manifest["frames"][index],
                args.source_root,
                min_confidence=args.min_confidence,
                include_pose=False,
                orientation=args.orientation,
            )
            for index in indices
        ]
        processed_views = preprocess_inputs(views, verbose=False)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
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
                ignore_pose_inputs=True,
                ignore_depth_scale_inputs=False,
                ignore_pose_scale_inputs=True,
            )
        elapsed = time.perf_counter() - started
        group_output.mkdir(parents=True, exist_ok=True)
        input_poses = torch.from_numpy(
            np.stack(
                [manifest["frames"][index]["corrected_c2w_opencv"] for index in indices]
            ).astype(np.float32)
        )
        input_poses_upright = torch.from_numpy(
            np.stack(
                [
                    rotate_c2w_opencv_cw(
                        np.asarray(
                            manifest["frames"][index]["corrected_c2w_opencv"]
                        )
                    )
                    for index in indices
                ]
            ).astype(np.float32)
        )
        torch.save(
            {
                "group": group,
                "frame_indices": indices,
                "pose_mode": "none",
                "model": args.model,
                "input_corrected_camera_poses": input_poses,
                "input_corrected_camera_poses_upright": input_poses_upright,
                "predictions": [_cpu_prediction(prediction) for prediction in predictions],
            },
            prediction_path,
        )
        summary = {
            "group_name": name,
            "status": "completed",
            "view_count": len(indices),
            "elapsed_seconds": elapsed,
            "peak_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
            "prediction_file": str(prediction_path),
        }
        (group_output / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        del predictions, processed_views, views
        gc.collect()
        torch.cuda.empty_cache()

    batch_summary = {
        "model": args.model,
        "pose_conditioning": False,
        "gpu": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.perf_counter() - batch_start,
        "groups": summaries,
    }
    (output_root / "batch_summary.json").write_text(
        json.dumps(batch_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(batch_summary, indent=2))


if __name__ == "__main__":
    main()
