import argparse
import logging
import os
import time

import numpy as np
import torch

from gluemap.controllers.global_merger import GlobalGluer
from gluemap.controllers.global_refinement import (
    run_refinement_pipeline,
)
from gluemap.controllers.star_collection import run_star_collection
from gluemap.controllers.star_inference import run_star_inference
from gluemap.controllers.twoview_inference import run_twoview_inference
from gluemap.datasets.star import BaseStarDataset
from gluemap.datasets.twoview import BaseTwoViewDataset
from gluemap.estimators.rotation_averaging import (
    collect_relative_rotations_ministar,
)
from gluemap.estimators.track_snapping import TrackSnapping
from gluemap.estimators.virtual_tracks import VirtualTrackPreparation
from gluemap.math.scaling import (
    keep_inframes,
    rescale_intrinsics,
    rescale_tracks,
)
from gluemap.utils.colmap import (
    prepare_sift_database,
    write_to_colmap_format,
)

logger = logging.getLogger(__name__)


def _build_aligned_pose_priors(
    arkit_c2w_poses: list[np.ndarray],
    global_centers: dict[int, np.ndarray],
    image_names: list[str],
    position_sigma_m: float,
    rotation_sigma_deg: float,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Align metric ARKit poses to GLUEMAP's gauge and build soft priors."""
    source_centers = np.stack([pose[:3, 3] for pose in arkit_c2w_poses])
    target_centers = np.stack(
        [np.asarray(global_centers[index]) for index in range(len(image_names))]
    )
    source_mean = source_centers.mean(axis=0)
    target_mean = target_centers.mean(axis=0)
    source_centered = source_centers - source_mean
    target_centered = target_centers - target_mean
    left, singular_values, right_t = np.linalg.svd(
        source_centered.T @ target_centered / len(source_centers)
    )
    alignment_rotation = right_t.T @ left.T
    if np.linalg.det(alignment_rotation) < 0:
        right_t[-1] *= -1
        alignment_rotation = right_t.T @ left.T
    alignment_scale = float(
        singular_values.sum()
        / np.mean(np.sum(source_centered * source_centered, axis=1))
    )
    alignment_translation = target_mean - alignment_scale * (
        alignment_rotation @ source_mean
    )
    center_sigma = position_sigma_m * alignment_scale
    rotation_sigma = np.deg2rad(rotation_sigma_deg)
    if center_sigma <= 0 or rotation_sigma <= 0:
        raise ValueError("Pose-prior sigmas must be positive")

    priors = {}
    for image_name, arkit_pose in zip(
        image_names, arkit_c2w_poses, strict=True
    ):
        center = (
            alignment_scale
            * (alignment_rotation @ np.asarray(arkit_pose[:3, 3]))
            + alignment_translation
        )
        c2w_rotation = alignment_rotation @ np.asarray(arkit_pose[:3, :3])
        priors[image_name] = {
            "center": center,
            "cam_from_world_rotation": c2w_rotation.T,
            "center_sigma": center_sigma,
            "rotation_sigma": rotation_sigma,
        }
    logger.info(
        "Built %d ARKit pose priors after Sim3 gauge alignment "
        "(scale %.6f, center sigma %.4f native units, rotation sigma %.2f deg)",
        len(priors),
        alignment_scale,
        center_sigma,
        rotation_sigma_deg,
    )
    return priors


class GluemapPipeline:
    """
    Main GLUEMAP pipeline: twoview -> star -> global mapping -> refinement.

    Stable configuration is stored as instance attributes. Per-dataset inputs
    are passed to ``run()`` or ``run_postprocessing()``.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        world_size: int,
        rank: int,
        device: str,
        dtype: torch.dtype,
        models: dict[str, torch.nn.Module] | None = None,
    ):
        self.args = args
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.models = models

    @torch.no_grad()
    def run(
        self,
        dataset_pair: BaseTwoViewDataset,
        pairs: list[tuple[int, int]] | None = None,
    ) -> tuple[str | None, dict]:
        """
        Run the full inference pipeline.

        twoview -> star -> global mapping -> refinement.

        Only executes global mapping and refinement on rank 0.

        Returns:
            Tuple of (pred_dir, timing_dict) or (None, timing_dict)
        """
        args = self.args
        world_size = self.world_size
        rank = self.rank
        device = self.device
        models = self.models

        timing = {}
        t_pipeline_start = time.perf_counter()

        # Step 1: Two-view inference (or skip Doppelgangers and treat all
        # pairs as valid)
        if getattr(args, "skip_doppelgangers", False):
            logger.info(
                "Skipping Doppelgangers: setting all pair scores to 1.0"
            )
            global_outputs = {
                "scores": torch.ones(len(dataset_pair)),
                "pairs": dataset_pair.pairs,
            }
            twoview_timing = {"batch_times": [], "num_batches": 0, "total": 0.0}
        else:
            global_outputs, twoview_timing = run_twoview_inference(
                args,
                dataset_pair,
                world_size,
                rank,
                file_name="twoview_result.pth",
                device=device,
                preloaded_models=models,
            )
        timing["twoview_inference"] = twoview_timing

        # Step 2: Generate dataset from outputs
        t0 = time.perf_counter()
        dataset = run_star_collection(dataset_pair, global_outputs, args)
        timing["dataset_generation"] = time.perf_counter() - t0

        # Step 3: Star inference
        predictions_dict, star_timing = run_star_inference(
            args,
            dataset,
            world_size,
            rank,
            file_name="star_result.pth",
            device=device,
            preloaded_models=models,
        )
        timing["star_inference"] = star_timing

        # Release extractor models from GPU — not needed after star inference
        dataset.query_extractors = []
        dataset._extractors_device = None

        # Move predictions_dict tensors to CPU to free GPU memory before
        # postprocessing
        for key, value in predictions_dict.items():
            if isinstance(value, torch.Tensor):
                predictions_dict[key] = value.cpu()
            elif isinstance(value, list):
                predictions_dict[key] = [
                    v.cpu() if isinstance(v, torch.Tensor) else v for v in value
                ]
        torch.cuda.empty_cache()

        # Steps 4-5: Global mapping and refinement (rank 0 only)
        if rank == 0:
            pred_dir, postproc_timing = self.run_postprocessing(
                args,
                predictions_dict,
                dataset_pair,
                dataset,
                pairs=pairs,
            )
            timing["postprocessing"] = postproc_timing
            timing["total_pipeline"] = time.perf_counter() - t_pipeline_start

            # Print summary
            logger.info("[Profiling] Pipeline Summary:")
            twoview_model_load = twoview_timing.get("model_loading", 0)
            logger.info(
                f"  Two-view (load+infer): "
                f"model_load={twoview_model_load:.2f}s, "
                f"inference={twoview_timing['total']:.2f}s"
            )
            logger.info(
                f"  Dataset generation:    {timing['dataset_generation']:.2f}s"
            )
            star_model_load = star_timing.get("model_loading", 0)
            logger.info(
                f"  Star (load+infer):     "
                f"model_load={star_model_load:.2f}s, "
                f"inference={star_timing['total']:.2f}s"
            )
            logger.info(
                f"  Postprocessing:        {postproc_timing['total']:.2f}s"
            )
            logger.info(
                f"  Total pipeline:        {timing['total_pipeline']:.2f}s"
            )

            # Save per-dataset timing
            timing_path = os.path.join(args.curr_path, "pipeline_timing.pth")
            torch.save(timing, timing_path)
            logger.info(
                f"[Profiling] Per-dataset timing saved to: {timing_path}"
            )

            return pred_dir, timing

        timing["total_pipeline"] = time.perf_counter() - t_pipeline_start
        return None, timing

    @staticmethod
    def run_postprocessing(
        args: argparse.Namespace,
        predictions_dict: dict,
        dataset_pair: BaseTwoViewDataset,
        dataset: BaseStarDataset,
        pairs: list[tuple[int, int]] | None = None,
    ) -> tuple[str, dict]:
        """
        Run postprocessing: global mapping and refinement.

        This should only be called on rank 0.

        Returns:
            Tuple of (pred_dir, timing_dict)
        """
        timing = {}
        t_postproc_start = time.perf_counter()
        torch.cuda.empty_cache()

        matching_pairs = pairs if pairs is not None else dataset.pairs

        t0 = time.perf_counter()
        poses_rel, poses_rel_scores = collect_relative_rotations_ministar(
            predictions_dict
        )
        timing["collect_rotations"] = time.perf_counter() - t0

        # Step 4: Global mapping
        t0 = time.perf_counter()
        GluemapPipeline.restore_image_shape(
            predictions_dict, dataset.images_change, dataset.images_shape_ori
        )

        predictions_dict["image_index_to_star_index"] = (
            dataset.image_index_to_star_index
        )

        global_gluer = GlobalGluer(args)
        global_gluer.sequential_edges = set(
            getattr(dataset_pair, "sequential_edges", [])
        )
        (
            global_rotations,
            global_centers,
            global_intrinsics,
            valid_edges,
            predictions_dict,
        ) = global_gluer.main(
            predictions_dict,
            dataset_pair.intrinsics_mapping,
            dataset_pair.camera_model,
            dataset.N,
        )
        timing["global_mapping"] = time.perf_counter() - t0

        known_intrinsics = getattr(dataset_pair, "known_intrinsics", None)
        if known_intrinsics is not None:
            if len(known_intrinsics) != len(global_intrinsics):
                raise ValueError(
                    "Known intrinsics count does not match GLUEMAP camera "
                    f"buckets: {len(known_intrinsics)} != "
                    f"{len(global_intrinsics)}"
                )
            global_intrinsics = known_intrinsics
            logger.info(
                "Replaced model-estimated intrinsics with calibrated dataset "
                "intrinsics."
            )

        position_sigma_m = getattr(args, "pose_prior_position_sigma_m", None)
        if position_sigma_m is not None:
            dataset_pair.pose_priors = _build_aligned_pose_priors(
                dataset_pair.pose_priors_c2w,
                global_centers,
                dataset_pair.images_list,
                position_sigma_m,
                getattr(args, "pose_prior_rotation_sigma_deg", 3.0),
            )

        # Override with GT intrinsics if requested (after global mapping)
        if getattr(args, "gt_intrinsics_path", None):
            from gluemap.utils.colmap import extract_gt_intrinsics

            gt_intrinsics = extract_gt_intrinsics(
                args.gt_intrinsics_path,
                dataset_pair.images_list,
                dataset_pair.intrinsics_mapping,
            )
            for cam_id in range(len(global_intrinsics)):
                if (
                    cam_id < len(gt_intrinsics)
                    and gt_intrinsics[cam_id] is not None
                ):
                    global_intrinsics[cam_id] = gt_intrinsics[cam_id]
            logger.info(
                f"Replaced intrinsics with GT from {args.gt_intrinsics_path}"
            )

        virtual_track_preparation = VirtualTrackPreparation()
        virtual_track_preparation.main(
            predictions_dict,
            global_intrinsics,
            dataset_pair.intrinsics_mapping,
            global_rotations,
            global_centers,
        )

        # Write coarse results to COLMAP format
        t0 = time.perf_counter()
        suffix = getattr(args, "output_suffix", "")
        coarse_dir = f"coarse{suffix}"
        logger.info(
            "write_to_colmap_format: %s", args.curr_path + "/" + coarse_dir
        )
        write_to_colmap_format(
            args.curr_path + "/" + coarse_dir,
            dataset_pair.images_shape_ori,
            global_rotations,
            global_centers,
            global_intrinsics,
            dataset_pair.intrinsics_mapping,
            images_list=dataset_pair.images_list,
            camera_type=dataset_pair.camera_model,
        )
        timing["write_coarse"] = time.perf_counter() - t0

        # Early exit if coarse_only
        if getattr(args, "coarse_only", False):
            logger.info("Coarse only mode: skipping all refinement steps.")
            timing["total"] = time.perf_counter() - t_postproc_start
            return coarse_dir, timing

        track_mode = getattr(args, "track_mode", "SPV")
        t0 = time.perf_counter()
        if "S" not in track_mode:
            logger.info(f"Track mode {track_mode}: skipping SIFT database.")
        elif not (
            hasattr(args, "force_load") and args.force_load
        ) or not os.path.exists(args.curr_path + "/database_sift.db"):
            prepare_sift_database(
                args.curr_path,
                args.images_path,
                dataset_pair.images_list,
                dataset_pair.intrinsics_mapping,
                np.asarray(matching_pairs, dtype=np.int64),
                camera_model=dataset_pair.camera_model,
                skip_matching=False,
                remove_existing=True,
            )
        timing["sift_database"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if "P" not in track_mode:
            logger.info(
                f"Track mode {track_mode} has no prior tracks: "
                "skipping track snapping."
            )
        elif getattr(args, "use_dummy_tracks", False):
            # Dummy tracks repeat query-image coordinates in every view. They
            # exist only because GlobalMerger expects a visibility tensor and
            # are not valid correspondences to snap or refine against.
            logger.info("Dummy tracks enabled: skipping track snapping.")
        else:
            track_snapping = TrackSnapping(snapping_thres=1.0)
            track_snapping.main(
                args.curr_path + "/database_sift.db",
                predictions_dict,
                dataset_pair.images_shape_ori,
                dataset_pair.images_list,
            )
        timing["track_snapping"] = time.perf_counter() - t0

        # Step 5: Refinement
        t0 = time.perf_counter()
        pred_dir, refinement_timing = run_refinement_pipeline(
            args=args,
            predictions_dict=predictions_dict,
            global_rotations=global_rotations,
            global_centers=global_centers,
            global_intrinsics=global_intrinsics,
            dataset_pair=dataset_pair,
            num_images=dataset.N,
            use_triangulation_first=True,
            num_refinement_iterations=getattr(
                args, "num_refinement_iterations", 2
            ),
            track_mode=track_mode,
            pose_priors=getattr(dataset_pair, "pose_priors", None),
        )
        timing["refinement"] = time.perf_counter() - t0
        timing["refinement_detail"] = refinement_timing

        timing["total"] = time.perf_counter() - t_postproc_start

        logger.info("[Profiling] Postprocessing Summary:")
        logger.info(f"  collect_rotations: {timing['collect_rotations']:.2f}s")
        logger.info(f"  global_mapping:    {timing['global_mapping']:.2f}s")
        logger.info(f"  write_coarse:      {timing['write_coarse']:.2f}s")
        logger.info(
            f"  sift_database:     {timing.get('sift_database', 0):.2f}s"
        )
        logger.info(
            f"  track_snapping:    {timing.get('track_snapping', 0):.2f}s"
        )
        logger.info(f"  refinement:        {timing.get('refinement', 0):.2f}s")
        logger.info(f"  total:             {timing['total']:.2f}s")

        return pred_dir, timing

    @staticmethod
    def restore_image_shape(
        predictions_dict: dict,
        images_change: torch.Tensor,
        images_shape_ori: list[tuple[int, int]],
    ) -> dict:
        """Rescale tracks/intrinsics back to original image shapes in place."""
        index = range(len(predictions_dict["indexes"]))
        for idx in index:
            rescale_tracks(predictions_dict, images_change, [idx])
            keep_inframes(predictions_dict, images_shape_ori, [idx])

            if "intrinsics" in predictions_dict:
                members = predictions_dict["indexes"][idx]
                scales_curr = [
                    images_change[members[j]] for j in range(len(members))
                ]
                intrinscs_curr = [
                    predictions_dict["intrinsics"][idx][:, j]
                    for j in range(len(members))
                ]
                intrinscs_curr = torch.stack(
                    rescale_intrinsics(intrinscs_curr, scales_curr), dim=1
                )
                predictions_dict["intrinsics"][idx] = intrinscs_curr.cpu()

        return predictions_dict


# Backward-compatible module-level wrapper functions


def run_inference_pipeline(
    args: argparse.Namespace,
    dataset_pair: BaseTwoViewDataset,
    world_size: int,
    rank: int,
    device: str,
    dtype: torch.dtype,
    pairs: list[tuple[int, int]] | None = None,
    models: dict[str, torch.nn.Module] | None = None,
) -> tuple[str | None, dict]:
    """Backward-compatible wrapper for GluemapPipeline.run()."""
    pipeline = GluemapPipeline(
        args, world_size, rank, device, dtype, models=models
    )
    return pipeline.run(dataset_pair, pairs=pairs)


def run_postprocessing_pipeline(
    args: argparse.Namespace,
    predictions_dict: dict,
    dataset_pair: BaseTwoViewDataset,
    dataset: BaseStarDataset,
    pairs: list[tuple[int, int]] | None = None,
) -> tuple[str, dict]:
    """Backward-compatible wrapper for GluemapPipeline.run_postprocessing()."""
    return GluemapPipeline.run_postprocessing(
        args,
        predictions_dict,
        dataset_pair,
        dataset,
        pairs=pairs,
    )
