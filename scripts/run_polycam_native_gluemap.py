#!/usr/bin/env python3
"""Run conditioned local stars through GLUEMAP's native global mapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from gluemap.controllers.gluemap_impl import GluemapPipeline
from gluemap.controllers.star_inference import run_star_inference
from gluemap.datasets.polycam_native import PolycamNativeStarDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("frontend_edges", type=Path)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=["map_anything", "pi3x"], default="map_anything"
    )
    parser.add_argument("--manifest-start", type=int)
    parser.add_argument("--manifest-end", type=int)
    parser.add_argument(
        "--group-config",
        type=Path,
        help="Explicit overlapping group cover; avoids one star per frame.",
    )
    parser.add_argument("--max-neighbors", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--full-refinement", action="store_true")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Reuse star_result.pth and rebuild only the coarse global model.",
    )
    parser.add_argument(
        "--track-mode",
        choices=["SV", "V"],
        default="SV",
        help="Refinement tracks: SIFT+virtual or virtual depth tracks only.",
    )
    args_cli = parser.parse_args()

    output = args_cli.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        chosen_model=args_cli.backend,
        path_feedforward=(
            "facebook/map-anything"
            if args_cli.backend == "map_anything"
            else "/workspace/cache/pi3x/model.safetensors"
        ),
        path_tracker="",
        curr_path=str(output),
        images_path=str(args_cli.dataset_directory / "images"),
        temp_path=str(output / "tmp"),
        camera_model="PINHOLE",
        intrinsics_mode="PER_CAMERA",
        num_track_per_img=1024,
        max_num_tracks=None,
        max_neighbors=args_cli.max_neighbors,
        batch_size=1,
        num_workers=args_cli.num_workers,
        distributed=False,
        use_dummy_tracks=True,
        resume_partial=True,
        checkpoint_every=100,
        force_load=args_cli.full_refinement or args_cli.postprocess_only,
        rerun_from=None,
        coarse_only=not args_cli.full_refinement,
        output_suffix="",
        valid_pose_threshold=0.05,
        is_sequential=True,
        use_ceres_rotation_averaging=False,
        use_gt_intrinsics=False,
        gt_intrinsics_path=None,
        num_refinement_iterations=2,
        fix_intrinsics=True,
        # P means neural prior tracks. With use_dummy_tracks=True those tracks
        # are placeholders, not observations. Refine using genuine SIFT tracks
        # and MapAnything's depth-derived virtual tracks only.
        track_mode=args_cli.track_mode,
    )
    dataset = PolycamNativeStarDataset(
        args,
        args_cli.manifest,
        args_cli.source_root,
        args_cli.dataset_directory,
        args_cli.frontend_edges,
        group_config=args_cli.group_config,
        manifest_start=args_cli.manifest_start,
        manifest_end=args_cli.manifest_end,
    )
    predictions, star_timing = run_star_inference(
        args,
        dataset,
        world_size=1,
        rank=0,
        file_name="star_result.pth",
        device="cuda",
        dtype=torch.bfloat16,
    )
    dataset.query_extractors = []
    dataset._extractors_device = None
    for key, values in predictions.items():
        if isinstance(values, list):
            predictions[key] = [
                value.cpu() if isinstance(value, torch.Tensor) else value
                for value in values
            ]
    torch.cuda.empty_cache()

    dataset_pair = SimpleNamespace(
        intrinsics_mapping=dataset.intrinsics_mapping,
        known_intrinsics=dataset.known_intrinsics,
        camera_model=dataset.camera_model,
        sequential_edges=dataset.sequential_edges,
        images_list=dataset.images_list,
        images_path=dataset.images_path,
        images_shape_ori=dataset.images_shape_ori,
    )
    prediction_directory, post_timing = GluemapPipeline.run_postprocessing(
        args,
        predictions,
        dataset_pair,
        dataset,
        pairs=[tuple(map(int, pair)) for pair in dataset.pairs],
    )
    coverage = torch.as_tensor(
        getattr(dataset, "group_coverage", [1]), dtype=torch.int64
    )
    summary = {
        "backend": args_cli.backend,
        "track_mode": args_cli.track_mode,
        "pose_conditioning": False,
        "arkit_optimization_prior": False,
        "frames": dataset.N,
        "stars": len(dataset),
        "group_config": None
        if args_cli.group_config is None
        else str(args_cli.group_config),
        "coverage": {
            "minimum": int(coverage.min()),
            "median": float(coverage.median()),
            "maximum": int(coverage.max()),
        },
        "prediction_directory": prediction_directory,
        "star_timing": star_timing,
        "postprocessing_timing": post_timing,
        "native_to_manifest": dataset.native_to_manifest,
    }
    (output / "telluride_native_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key != "native_to_manifest"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
