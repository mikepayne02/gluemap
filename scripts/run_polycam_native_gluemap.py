#!/usr/bin/env python3
"""Run conditioned local stars through GLUEMAP's native global mapper."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gluemap.controllers.gluemap_impl import GluemapPipeline
from gluemap.controllers.star_inference import run_star_inference
from gluemap.datasets.polycam_native import PolycamNativeStarDataset
from gluemap.estimators.group_pose_constraints import (
    build_group_fragment_constraints,
    build_group_pose_constraints,
)


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
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--full-refinement", action="store_true")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Write or resume star_result.pth, then stop before global "
            "assembly so local groups can be validated."
        ),
    )
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
    parser.add_argument(
        "--ba-linear-solver",
        choices=["auto", "sparse_schur", "iterative_schur"],
        default="auto",
        help=(
            "Ceres linear solver for full refinement. Direct sparse Schur is "
            "more robust for dense overlapping-group camera graphs; auto "
            "preserves COLMAP's default selection."
        ),
    )
    parser.add_argument(
        "--ba-filter-iterations",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help=(
            "Bundle-adjustment/filter passes per refinement iteration. "
            "Reduce only when stricter pruning leaves a rank-deficient solve."
        ),
    )
    args_cli = parser.parse_args()
    if args_cli.inference_only and (
        args_cli.postprocess_only or args_cli.full_refinement
    ):
        parser.error(
            "--inference-only cannot be combined with postprocessing options"
        )

    output = args_cli.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args_cli.random_seed)
    np.random.seed(args_cli.random_seed)
    torch.manual_seed(args_cli.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.random_seed)
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
        # The score-weighted Ceres solver preserves reliable local group
        # rotations. PyCOLMAP's unweighted robust pass can discard every
        # incident edge for short camera runs when two overlapping groups
        # disagree, leaving otherwise valid cameras unsupported.
        use_ceres_rotation_averaging=True,
        use_gt_intrinsics=False,
        gt_intrinsics_path=None,
        num_refinement_iterations=2,
        fix_intrinsics=True,
        # Metric LiDAR conditioning gives every local prediction the same
        # physical scale.  Do not let global assembly shear groups by fitting
        # an independent arbitrary scale for each one.
        fix_group_scales=True,
        # Every production camera must be supported by MapAnything group
        # relationships. Never invent missing poses from ARKit or interpolation.
        allow_trajectory_fallback=False,
        require_complete_camera_support=True,
        # P means neural prior tracks. With use_dummy_tracks=True those tracks
        # are placeholders, not observations. Refine using genuine SIFT tracks
        # and MapAnything's depth-derived virtual tracks only.
        track_mode=args_cli.track_mode,
        ba_linear_solver=args_cli.ba_linear_solver,
        ba_filter_iterations=args_cli.ba_filter_iterations,
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
    checkpoint_path = output / "star_result.pth"
    if args_cli.postprocess_only and not checkpoint_path.is_file():
        parser.error(
            f"--postprocess-only requires an existing {checkpoint_path}"
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

    if args_cli.inference_only:
        print(
            json.dumps(
                {
                    "backend": args_cli.backend,
                    "frames": dataset.N,
                    "stars": len(dataset),
                    "completed_inference_batches": len(
                        predictions.get("indexes", [])
                    ),
                    "checkpoint": str(checkpoint_path),
                    "star_timing": star_timing,
                },
                indent=2,
            )
        )
        return

    verified_loop_edges = getattr(dataset, "group_verified_loop_edges", set())
    temporal_pose_diagnostics = build_group_pose_constraints(
        predictions,
        num_images=dataset.N,
        trusted_loop_edges=verified_loop_edges,
        preferred_temporal_groups=getattr(
            dataset, "group_preferred_temporal_groups", {}
        ),
    )
    temporal_constraints = [
        constraint
        for constraint in temporal_pose_diagnostics
        if constraint["kind"] == "mapanything_temporal"
    ]
    fragment_constraints = build_group_fragment_constraints(
        predictions, num_images=dataset.N
    )
    recovery_constraints = []
    for constraint in temporal_constraints:
        if not constraint["preferred_group_override"]:
            continue
        recovery_constraints.append(
            {**constraint, "kind": "mapanything_recovery_temporal"}
        )
    predictions["pose_constraints"] = [
        *fragment_constraints,
        *recovery_constraints,
    ]
    constraint_summary = {
        "source": "overlapping_mapanything_group_fragments",
        "arkit_used": False,
        "fragment_constraints": len(fragment_constraints),
        "temporal_diagnostic_pairs": len(temporal_constraints),
        "recovery_temporal_constraints": len(recovery_constraints),
        "verified_loop_constraints": (
            len(temporal_pose_diagnostics) - len(temporal_constraints)
        ),
        "single_group_temporal_pairs": sum(
            constraint["candidate_count"] == 1
            for constraint in temporal_constraints
        ),
        "preferred_group_temporal_pairs": sum(
            constraint["preferred_group_override"]
            for constraint in temporal_constraints
        ),
        "ambiguous_temporal_pairs": [
            [constraint["first"], constraint["second"]]
            for constraint in temporal_constraints
            if constraint["candidate_count"] > 1
            and constraint["agreement_fraction"] < 0.51
        ],
    }
    (output / "group_pose_constraint_summary.json").write_text(
        json.dumps(constraint_summary, indent=2) + "\n", encoding="utf-8"
    )

    dataset_pair = SimpleNamespace(
        intrinsics_mapping=dataset.intrinsics_mapping,
        known_intrinsics=dataset.known_intrinsics,
        pose_priors_c2w=dataset.pose_priors_c2w,
        camera_model=dataset.camera_model,
        sequential_edges=dataset.sequential_edges,
        trusted_loop_edges=verified_loop_edges,
        trajectory_break_edges=getattr(
            dataset, "trajectory_break_edges", set()
        ),
        images_list=dataset.images_list,
        images_path=dataset.images_path,
        images_shape_ori=dataset.images_shape_ori,
    )
    prediction_directory, post_timing = GluemapPipeline.run_postprocessing(
        args,
        predictions,
        dataset_pair,
        dataset,
        pairs=[
            tuple(map(int, pair))
            for pair in getattr(dataset, "refinement_edges", dataset.pairs)
        ],
    )
    coverage = torch.as_tensor(
        getattr(dataset, "group_coverage", [1]), dtype=torch.int64
    )
    conditioned_groups = sum(
        bool(members)
        for members in getattr(dataset, "group_pose_conditioned_members", [])
    )
    summary = {
        "backend": args_cli.backend,
        "track_mode": args_cli.track_mode,
        "ba_linear_solver": args_cli.ba_linear_solver,
        "ba_filter_iterations": args_cli.ba_filter_iterations,
        "pose_conditioning": conditioned_groups > 0,
        "pose_conditioned_groups": conditioned_groups,
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
