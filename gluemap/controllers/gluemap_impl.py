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
from gluemap.estimators.group_pose_constraints import (
    build_group_pose_constraints,
)
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


def _copy_predictions_for_global_mapping(predictions_dict: dict) -> dict:
    """Copy the small mutable subset used by global camera estimation.

    GlobalGluer rejects pose relations by pruning group members and adjusts
    relative translations during similarity averaging.  Those operations
    must not alter the full MapAnything groups later used to construct virtual
    depth tracks.
    """
    copied = {}
    for key, value in predictions_dict.items():
        if isinstance(value, list):
            copied[key] = list(value)
        else:
            copied[key] = value

    copied["indexes"] = [
        list(indexes) for indexes in predictions_dict["indexes"]
    ]
    for key in ("pose_scores", "extrinsics"):
        copied[key] = [value.clone() for value in predictions_dict[key]]
    return copied


def _set_full_group_visibility_scores(predictions_dict: dict) -> None:
    """Restore per-pixel visibility scores for unpruned MapAnything groups."""
    predictions_dict["scores"] = {
        index: torch.where(visibility > 0.05, visibility, 0.0)
        for index, visibility in enumerate(predictions_dict["vis"])
    }


def _apply_group_scales(predictions_dict: dict, scales: list[float]) -> None:
    """Apply camera-solve group scales without pruning neural observations."""
    if len(scales) != len(predictions_dict["indexes"]):
        raise ValueError("Group-scale count does not match prediction groups")
    for star_index, scale in enumerate(scales):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Invalid solved scale for group {star_index}: {scale}"
            )
        predictions_dict["points3d_virtual"][star_index] /= scale
        predictions_dict["extrinsics"][star_index][:, :, :3, 3:] /= scale


def _build_gravity_priors(
    dataset_pair,
    gravity_world: np.ndarray | None,
    angular_sigma_deg: float = 0.5,
) -> dict | None:
    """Build gravity-only BA constraints from corrected ARKit orientation."""
    priors_c2w = getattr(dataset_pair, "pose_priors_c2w", None)
    if gravity_world is None or priors_c2w is None:
        return None
    arkit_world_gravity = np.array([0.0, 1.0, 0.0])
    return {
        image_name: {
            "world_gravity": np.asarray(gravity_world, dtype=np.float64),
            "camera_gravity": np.asarray(prior, dtype=np.float64)[:3, :3].T
            @ arkit_world_gravity,
            "angular_sigma": np.deg2rad(angular_sigma_deg),
        }
        for image_name, prior in zip(
            dataset_pair.images_list, priors_c2w, strict=True
        )
    }


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

        pose_graph_predictions = _copy_predictions_for_global_mapping(
            predictions_dict
        )
        trusted_loop_edges = set(
            getattr(dataset_pair, "trusted_loop_edges", set())
        )
        if hasattr(dataset, "group_coverage"):
            pose_graph_predictions["pose_constraints"] = (
                build_group_pose_constraints(
                    pose_graph_predictions,
                    trusted_loop_edges,
                    list(dataset_pair.pose_priors_c2w),
                    set(getattr(dataset_pair, "trajectory_break_edges", set())),
                )
            )
        global_gluer = GlobalGluer(
            args,
            trajectory_priors_c2w=getattr(
                dataset_pair, "pose_priors_c2w", None
            ),
        )
        global_gluer.sequential_edges = set(
            getattr(dataset_pair, "sequential_edges", [])
        )
        (
            global_rotations,
            global_centers,
            global_intrinsics,
            valid_edges,
            pose_graph_predictions,
        ) = global_gluer.main(
            pose_graph_predictions,
            dataset_pair.intrinsics_mapping,
            dataset_pair.camera_model,
            dataset.N,
        )
        timing["global_mapping"] = time.perf_counter() - t0
        gravity_priors = _build_gravity_priors(
            dataset_pair, global_gluer.gravity_world
        )
        if "solved_group_scales" in pose_graph_predictions:
            _apply_group_scales(
                predictions_dict,
                pose_graph_predictions["solved_group_scales"],
            )
        global_gluer._mark_inconsistent_edges(
            predictions_dict, global_rotations, global_centers
        )

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

        _set_full_group_visibility_scores(predictions_dict)
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
            pose_priors=None,
            gravity_priors=gravity_priors,
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
