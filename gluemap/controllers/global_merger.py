import argparse
import logging

import networkx as nx
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from gluemap.estimators.group_pose_constraints import iter_pose_constraints
from gluemap.estimators.intrinsics_averaging import intrinsics_averaging
from gluemap.estimators.rotation_averaging import (
    rotation_averaging,
    rotation_averaging_pycolmap,
)
from gluemap.estimators.similarity_averaging import (
    similarity_averaging,
)
from gluemap.math.geometry import restore_identity
from gluemap.math.mst_initialization import initialize_mst_structures

logger = logging.getLogger(__name__)


def _contiguous_runs(values: list[int]) -> list[list[int]]:
    """Split sorted integer IDs into contiguous runs."""
    if not values:
        return []
    runs = [[values[0]]]
    for value in values[1:]:
        if value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return runs


def _fill_missing_rotations_temporally(
    rotations: dict[int, np.ndarray], num_images: int
) -> dict[int, np.ndarray]:
    """Fill unestimated rotations from adjacent trajectory frames.

    Rotation averaging can omit images whose every incident edge is rejected.
    Identity is a dangerous fallback for those images because it creates an
    apparently registered camera with an arbitrary orientation.  The image IDs
    in GLUEMAP datasets follow capture order, so short missing runs are seeded
    by SLERP between the closest estimated frames instead.
    """
    result = {idx: np.asarray(value).copy() for idx, value in rotations.items()}
    known = sorted(result)
    if not known:
        raise RuntimeError("Rotation averaging did not estimate any camera")

    missing = sorted(set(range(num_images)) - set(known))
    for run in _contiguous_runs(missing):
        first, last = run[0], run[-1]
        lower = max((idx for idx in known if idx < first), default=None)
        upper = min((idx for idx in known if idx > last), default=None)
        if lower is not None and upper is not None:
            interpolator = Slerp(
                [float(lower), float(upper)],
                Rotation.from_matrix(
                    np.stack([result[lower], result[upper]], axis=0)
                ),
            )
            interpolated = interpolator(
                np.asarray(run, dtype=float)
            ).as_matrix()
            result.update(zip(run, interpolated, strict=True))
        else:
            reference = lower if lower is not None else upper
            assert reference is not None
            for idx in run:
                result[idx] = result[reference].copy()
    return result


def _fill_missing_centers_temporally(
    centers: dict[int, np.ndarray], num_images: int
) -> dict[int, np.ndarray]:
    """Fill unestimated centers by interpolation along capture order."""
    result = {idx: np.asarray(value).copy() for idx, value in centers.items()}
    known = sorted(result)
    if not known:
        raise RuntimeError("Camera-center initialization estimated no cameras")

    missing = sorted(set(range(num_images)) - set(known))
    for run in _contiguous_runs(missing):
        first, last = run[0], run[-1]
        lower = max((idx for idx in known if idx < first), default=None)
        upper = min((idx for idx in known if idx > last), default=None)
        if lower is not None and upper is not None:
            denominator = float(upper - lower)
            for idx in run:
                alpha = (idx - lower) / denominator
                result[idx] = (1.0 - alpha) * result[lower] + alpha * result[
                    upper
                ]
        else:
            reference = lower if lower is not None else upper
            assert reference is not None
            for idx in run:
                result[idx] = result[reference].copy()
    return result


def _rotation_alignment_from_priors(
    rotations: dict[int, np.ndarray], priors_c2w: list[np.ndarray]
) -> np.ndarray:
    """Estimate the world rotation mapping ARKit into the solved frame."""
    alignments = []
    for image_id, world_to_camera in rotations.items():
        prior_c2w = np.asarray(priors_c2w[image_id], dtype=np.float64)
        solved_c2w = np.asarray(world_to_camera, dtype=np.float64).T
        alignments.append(solved_c2w @ prior_c2w[:3, :3].T)
    if not alignments:
        raise RuntimeError(
            "Cannot align trajectory priors without solved cameras"
        )
    return Rotation.from_matrix(np.stack(alignments)).mean().as_matrix()


def _fill_missing_rotations_from_priors(
    rotations: dict[int, np.ndarray],
    priors_c2w: list[np.ndarray],
    num_images: int,
) -> dict[int, np.ndarray]:
    """Fill unsupported orientations while retaining ARKit trajectory turns."""
    if len(priors_c2w) != num_images:
        raise ValueError("Trajectory-prior count does not match image count")
    result = {idx: np.asarray(value).copy() for idx, value in rotations.items()}
    known = sorted(result)
    if not known:
        raise RuntimeError("Rotation averaging did not estimate any camera")

    world_alignment = _rotation_alignment_from_priors(result, priors_c2w)
    aligned_c2w = [
        world_alignment @ np.asarray(prior)[:3, :3] for prior in priors_c2w
    ]
    endpoint_corrections = {
        idx: result[idx].T @ aligned_c2w[idx].T for idx in known
    }
    missing = sorted(set(range(num_images)) - set(known))
    for run in _contiguous_runs(missing):
        lower = max((idx for idx in known if idx < run[0]), default=None)
        upper = min((idx for idx in known if idx > run[-1]), default=None)
        if lower is not None and upper is not None:
            correction = Slerp(
                [float(lower), float(upper)],
                Rotation.from_matrix(
                    np.stack(
                        [
                            endpoint_corrections[lower],
                            endpoint_corrections[upper],
                        ]
                    )
                ),
            )(np.asarray(run, dtype=float)).as_matrix()
            for image_id, delta in zip(run, correction, strict=True):
                result[image_id] = (delta @ aligned_c2w[image_id]).T
        else:
            reference = lower if lower is not None else upper
            assert reference is not None
            delta = endpoint_corrections[reference]
            for image_id in run:
                result[image_id] = (delta @ aligned_c2w[image_id]).T
    return result


def _fill_missing_centers_from_priors(
    centers: dict[int, np.ndarray],
    rotations: dict[int, np.ndarray],
    priors_c2w: list[np.ndarray],
    num_images: int,
) -> dict[int, np.ndarray]:
    """Fill unsupported centers using the curved ARKit path, not a line."""
    if len(priors_c2w) != num_images:
        raise ValueError("Trajectory-prior count does not match image count")
    result = {idx: np.asarray(value).copy() for idx, value in centers.items()}
    known = sorted(result)
    if not known:
        raise RuntimeError("Camera-center initialization estimated no cameras")

    supported_rotations = {idx: rotations[idx] for idx in known}
    world_alignment = _rotation_alignment_from_priors(
        supported_rotations, priors_c2w
    )
    prior_centers = np.stack(
        [np.asarray(prior, dtype=np.float64)[:3, 3] for prior in priors_c2w]
    )
    aligned = (world_alignment @ prior_centers.T).T
    source = aligned[known]
    target = np.stack([result[idx] for idx in known])
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    denominator = float(np.sum(source_centered**2))
    scale = (
        float(np.sum(source_centered * target_centered)) / denominator
        if denominator > 1e-12
        else 1.0
    )
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    translation = np.median(target - scale * source, axis=0)
    aligned = scale * aligned + translation
    endpoint_offsets = {idx: result[idx] - aligned[idx] for idx in known}

    missing = sorted(set(range(num_images)) - set(known))
    for run in _contiguous_runs(missing):
        lower = max((idx for idx in known if idx < run[0]), default=None)
        upper = min((idx for idx in known if idx > run[-1]), default=None)
        if lower is not None and upper is not None:
            denominator = float(upper - lower)
            for image_id in run:
                alpha = (image_id - lower) / denominator
                offset = (1.0 - alpha) * endpoint_offsets[lower] + alpha * (
                    endpoint_offsets[upper]
                )
                result[image_id] = aligned[image_id] + offset
        else:
            reference = lower if lower is not None else upper
            assert reference is not None
            for image_id in run:
                result[image_id] = (
                    aligned[image_id] + endpoint_offsets[reference]
                )
    return result


def _estimate_trajectory_gravity(
    rotations: dict[int, np.ndarray], priors_c2w: list[np.ndarray]
) -> np.ndarray:
    """Estimate gravity in the solved world without changing camera poses."""
    arkit_world_gravity = np.array([0.0, 1.0, 0.0])
    world_estimates = []
    for image_id, world_to_camera in rotations.items():
        prior_rotation = np.asarray(priors_c2w[image_id])[:3, :3]
        camera_gravity = prior_rotation.T @ arkit_world_gravity
        world_estimates.append(np.asarray(world_to_camera).T @ camera_gravity)
    gravity_world = np.sum(world_estimates, axis=0)
    gravity_world /= np.linalg.norm(gravity_world)
    return gravity_world


class GlobalGluer:
    """Glue per-star predictions into a single global reconstruction.

    Runs graph refinement, intrinsics averaging, rotation averaging, MST-based
    similarity initialization and similarity averaging on the star-inference
    predictions to produce global rotations, camera centers and intrinsics.

    Most private helpers mutate ``predictions_dict`` in place — most notably
    ``pose_scores`` (suppress weak/inconsistent edges) and the per-star list
    entries (prune invisible neighbors).
    """

    def __init__(
        self,
        args: argparse.Namespace,
        trajectory_priors_c2w: list[np.ndarray] | None = None,
    ):
        self.max_rot_error = 5  # degrees

        self.valid_threshold_pose = (
            args.valid_pose_threshold
            if hasattr(args, "valid_pose_threshold")
            else 0.05
        )

        self.thres_consistency = np.deg2rad(10.0)  # degrees
        self.angle_threshold = 5.0  # degrees
        self.boost_sequential = bool(
            hasattr(args, "is_sequential") and args.is_sequential
        )
        self.use_ceres_rotation_averaging = getattr(
            args, "use_ceres_rotation_averaging", False
        )
        self.trajectory_priors_c2w = trajectory_priors_c2w
        self.gravity_world = None
        self.fix_group_scales = getattr(args, "fix_group_scales", False)
        self.require_complete_support = getattr(
            args, "require_complete_camera_support", False
        )

    def main(
        self,
        predictions_dict: dict,
        intrinsics_mapping: dict[int, int],
        camera_model: str,
        num_img: int,
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        list[np.ndarray | None],
        set[tuple[int, int]],
        dict,
    ]:
        """Run global gluing: graph refine + intrinsics/structure estimation.

        Args:
            predictions_dict: Star-inference outputs (mutated in place).
            intrinsics_mapping: ``{image_id: camera_type_idx}``.
            camera_model: COLMAP camera model name (e.g. ``"SIMPLE_PINHOLE"``).
            num_img: Total number of images in the dataset.

        Returns:
            ``(global_rotations, global_centers, global_intrinsics, valid_edges,
            predictions_dict)``. The returned ``predictions_dict`` is the same
            mutated object that was passed in.
        """
        self.N = num_img
        # Refine the graph structure
        predictions_dict, valid_edges = self._refine_graph_structure(
            predictions_dict
        )

        # Estimate the intrinsics
        global_intrinsics = self._estimate_intrinsics(
            predictions_dict, intrinsics_mapping, camera_model
        )

        global_rotations, global_centers = self._global_structure_estimation(
            predictions_dict,
        )

        return (
            global_rotations,
            global_centers,
            global_intrinsics,
            valid_edges,
            predictions_dict,
        )

    def _refine_graph_structure(
        self, predictions_dict: dict
    ) -> tuple[dict, set[tuple[int, int]]]:
        """Suppress weak/inconsistent edges and connect any missing components.

        Modifies ``predictions_dict`` in place (populates ``scores``, zeros
        out ``pose_scores`` for inconsistent edges, prunes invisible pairs).
        """
        if "pose_constraints" in predictions_dict:
            valid_edges = {
                (constraint["first"], constraint["second"])
                for constraint in iter_pose_constraints(predictions_dict)
            }
            return predictions_dict, valid_edges

        predictions_dict["scores"] = {}
        for idx in range(len(predictions_dict["indexes"])):
            predictions_dict["scores"][idx] = torch.where(
                predictions_dict["vis"][idx] > 0.05,
                predictions_dict["vis"][idx],
                0.0,
            )
        # Perform two way check for filtering simple outliers
        self._filter_inconsistent_edges(predictions_dict)

        # First, we want to collect the valid edges
        valid_edges = self._collect_valid_edges(predictions_dict)

        # Then, connect the missing edges
        self._connect_missing(valid_edges, predictions_dict)

        # Prune invisible pairs
        if "pose_constraints" not in predictions_dict:
            self._prune_invisible_pairs(predictions_dict)

        return predictions_dict, valid_edges

    def _filter_inconsistent_edges(self, predictions_dict: dict) -> None:
        """Zero out ``pose_scores`` for edges whose two directions disagree.

        Uses a two-way check on relative rotation and translation; both
        directions of an inconsistent edge are suppressed. Modifies
        ``predictions_dict["pose_scores"]`` in place.
        """
        indexes = range(len(predictions_dict["indexes"]))

        rel_poses = {}
        counter = 0
        for idx in indexes:
            poses = predictions_dict["extrinsics"][idx]
            N = poses.shape[1]
            idx_i = predictions_dict["indexes"][idx][0]
            for i in range(N):
                if i == 0:
                    continue
                idx_j = predictions_dict["indexes"][idx][i]
                if (idx_j, idx_i) in rel_poses:
                    pose = rel_poses[(idx_j, idx_i)][0]
                    # Compare the two relative poses
                    R12 = pose[:3, :3]
                    R21 = poses[0, i, :3, :3]

                    error_r = torch.acos(
                        torch.clamp(
                            ((R12 @ R21).trace() - 1) / 2,
                            -1.0 + 1e-6,
                            1.0 - 1e-6,
                        )
                    )

                    # R21 * t12_normed = -t21_normed
                    t12_normed = pose[:3, 3:] / torch.linalg.norm(pose[:3, 3:])
                    t21_normed = poses[0, i, :3, 3:] / torch.linalg.norm(
                        poses[0, i, :3, 3:]
                    )
                    error_t = torch.acos(
                        torch.clamp(
                            -torch.sum(t12_normed * (R12 @ t21_normed)),
                            -1.0 + 1e-6,
                            1.0 - 1e-6,
                        )
                    )

                    if (
                        error_r > self.thres_consistency
                        or error_t > 3 * self.thres_consistency
                    ):
                        # Inconsistent, suppress both directions
                        predictions_dict["pose_scores"][idx][0, i] = 0.0
                        idx_1 = rel_poses[(idx_j, idx_i)][1]
                        j_pos = rel_poses[(idx_j, idx_i)][2]

                        predictions_dict["pose_scores"][idx_1][0, j_pos] = 0.0
                        logger.debug(
                            f"Filtered inconsistent edge between {idx_i} and "
                            f"{idx_j}, rotation error: "
                            f"{np.rad2deg(error_r)} degrees, translation "
                            f"error: {np.rad2deg(error_t)} degrees"
                        )
                        counter += 1
                else:
                    rel_poses[(idx_i, idx_j)] = (poses[0, i].cpu(), idx, i)

        logger.info(f"Total number of inconsistent edges filtered: {counter}")

    # TODO: debug this part
    def _collect_valid_edges(
        self, predictions_dict: dict
    ) -> set[tuple[int, int]]:
        """Return edges whose ``pose_scores`` exceed the threshold."""
        # Here, the score already considers the n^2 visibility, so we can
        # just use the pose scores
        valid_edges = set()
        indexes = range(len(predictions_dict["indexes"]))
        for idx in indexes:
            valid_j = torch.where(
                predictions_dict["pose_scores"][idx][0]
                > self.valid_threshold_pose
            )[0]
            valid_edges.update(
                set(
                    [
                        (
                            predictions_dict["indexes"][idx][0],
                            predictions_dict["indexes"][idx][j],
                        )
                        for j in valid_j[1:]
                    ]
                )
            )

        return valid_edges

    def _connect_missing(
        self,
        valid_edges: set[tuple[int, int]],
        predictions_dict: dict,
    ) -> None:
        """Bump cross-component pose scores so the graph becomes connected.

        For every pair of images that fall in different connected components
        of ``valid_edges``, increases ``pose_scores`` by ``1e-2`` and adds the
        edge to ``valid_edges`` in place.
        """
        N = self.N

        graph = nx.Graph()
        graph.add_edges_from(valid_edges)
        components = list(nx.connected_components(graph))
        if len(components) == 1 and len(components[0]) == N:
            logger.info(
                "Edge connectivity of the graph: %d",
                nx.edge_connectivity(graph),
            )
            return

        image_to_component = {}
        for component_id, component in enumerate(components):
            for image_id in component:
                image_to_component[image_id] = component_id
        next_component = len(components)
        for image_id in range(N):
            if image_id not in image_to_component:
                image_to_component[image_id] = next_component
                next_component += 1

        for star_index, image_ids in enumerate(predictions_dict["indexes"]):
            anchor = image_ids[0]
            for position, member in enumerate(image_ids[1:], start=1):
                if image_to_component[anchor] != image_to_component[member]:
                    predictions_dict["pose_scores"][star_index][
                        0, position
                    ] += 1e-2
                    valid_edges.add((anchor, member))

    def _global_structure_estimation(
        self,
        predictions_dict: dict,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Estimate global rotations and camera centers from the refined graph.

        Runs rotation averaging (filtering invalid edges twice when using the
        Ceres backend), MST-based similarity initialization, and similarity
        averaging. Falls back to identity / zero entries for any image that
        was not estimated.

        Returns:
            ``(global_rotations, global_centers)`` covering all ``N`` images.
        """
        # Double sequential edge weights (neighboring frames with index
        # diff <= 10)
        if self.boost_sequential and "pose_constraints" not in predictions_dict:
            self._boost_sequential_edges(predictions_dict, boost_factor=2.0)

        if "pose_constraints" in predictions_dict:
            global_rotations = rotation_averaging(predictions_dict)
            self._filter_invalid_edges(predictions_dict, global_rotations)
            global_rotations = rotation_averaging(
                predictions_dict, global_rotations
            )
        elif self.use_ceres_rotation_averaging:
            # Original two-pass: RA -> filter -> RA -> filter
            global_rotations = rotation_averaging(predictions_dict)
            self._filter_invalid_edges(predictions_dict, global_rotations)
            global_rotations = rotation_averaging(
                predictions_dict, global_rotations
            )
            self._filter_invalid_edges(predictions_dict, global_rotations)
        else:
            global_rotations = rotation_averaging_pycolmap(
                predictions_dict, max_rotation_error_deg=self.max_rot_error
            )
            self._filter_invalid_edges(predictions_dict, global_rotations)

        if "pose_constraints" not in predictions_dict:
            self._prune_invisible_pairs(predictions_dict)

        missing_rotation_count = self.N - len(global_rotations)
        if missing_rotation_count:
            if self.require_complete_support:
                missing = sorted(set(range(self.N)) - set(global_rotations))
                raise RuntimeError(
                    "Rotation averaging left cameras without group support: "
                    f"{missing[:20]} ({len(missing)} total)"
                )
            if self.trajectory_priors_c2w is None:
                logger.warning(
                    "Temporally interpolating %d cameras omitted by rotation "
                    "averaging",
                    missing_rotation_count,
                )
                global_rotations = _fill_missing_rotations_temporally(
                    global_rotations, self.N
                )
            else:
                logger.info(
                    "Initializing %d unsupported camera rotations from the "
                    "curved trajectory prior",
                    missing_rotation_count,
                )
                global_rotations = _fill_missing_rotations_from_priors(
                    global_rotations, self.trajectory_priors_c2w, self.N
                )

        if self.trajectory_priors_c2w is not None:
            self.gravity_world = _estimate_trajectory_gravity(
                global_rotations, self.trajectory_priors_c2w
            )

        # Initialize the structures by maximum spanning tree
        global_centers, global_scales = initialize_mst_structures(
            predictions_dict, global_rotations
        )

        disconnected_ids = sorted(set(range(self.N)) - set(global_centers))
        if disconnected_ids:
            if self.require_complete_support:
                raise RuntimeError(
                    "Center initialization left cameras without group "
                    f"support: {disconnected_ids[:20]} "
                    f"({len(disconnected_ids)} total)"
                )
            logger.info(
                "Initializing poses for %d cameras disconnected from the "
                "center-initialization tree",
                len(disconnected_ids),
            )
            connected_rotations = {
                idx: rotation
                for idx, rotation in global_rotations.items()
                if idx not in disconnected_ids
            }
            if self.trajectory_priors_c2w is None:
                global_rotations = _fill_missing_rotations_temporally(
                    connected_rotations, self.N
                )
                global_centers = _fill_missing_centers_temporally(
                    global_centers, self.N
                )
            else:
                global_rotations = _fill_missing_rotations_from_priors(
                    connected_rotations, self.trajectory_priors_c2w, self.N
                )
                global_centers = _fill_missing_centers_from_priors(
                    global_centers,
                    global_rotations,
                    self.trajectory_priors_c2w,
                    self.N,
                )

        averaging_scales = global_scales
        if self.fix_group_scales:
            # Metric-depth-conditioned groups already share physical units.
            # MST scale ratios are only an initialization artifact and must
            # not become frozen per-group deformations.
            averaging_scales = {
                star_index: 1.0
                for star_index in range(len(predictions_dict["indexes"]))
            }
        global_centers = similarity_averaging(
            predictions_dict,
            global_rotations,
            global_centers=global_centers,
            global_scales=averaging_scales,
            max_num_iterations=200,
            fix_scales=self.fix_group_scales,
        )
        if disconnected_ids:
            # The connected cameras move during similarity averaging, while
            # disconnected cameras have no residuals.  Interpolate once more
            # from the optimized neighbors so run boundaries remain smooth.
            connected_centers = {
                idx: center
                for idx, center in global_centers.items()
                if idx not in disconnected_ids
            }
            if self.trajectory_priors_c2w is None:
                global_centers = _fill_missing_centers_temporally(
                    connected_centers, self.N
                )
            else:
                global_centers = _fill_missing_centers_from_priors(
                    connected_centers,
                    global_rotations,
                    self.trajectory_priors_c2w,
                    self.N,
                )

        # Prune the edges by the global rotations
        self._mark_inconsistent_edges(
            predictions_dict, global_rotations, global_centers
        )

        # Check whether all images are estimated
        logger.info(f"Number of images: {self.N}")
        logger.info(f"Number of global rotations: {len(global_rotations)}")
        logger.info(f"Number of global centers: {len(global_centers)}")
        if len(global_rotations) != self.N or len(global_centers) != self.N:
            raise RuntimeError(
                "Global assembly produced an incomplete camera solution: "
                f"{len(global_rotations)} rotations and "
                f"{len(global_centers)} centers for {self.N} images"
            )

        return global_rotations, global_centers

    def _estimate_intrinsics(
        self,
        predictions_dict: dict,
        intrinsics_mapping: dict[int, int],
        camera_model: str,
    ) -> list[np.ndarray | None]:
        """Run intrinsics averaging across all star predictions."""
        indexes = range(len(predictions_dict["indexes"]))
        intrinsics_all = [
            predictions_dict["intrinsics"][idx] for idx in indexes
        ]
        members = [predictions_dict["indexes"][idx] for idx in indexes]

        global_intrinsics = intrinsics_averaging(
            intrinsics_all, members, intrinsics_mapping, camera_model
        )

        return global_intrinsics

    def _prune_invisible_pairs(self, predictions_dict: dict) -> None:
        """Drop neighbors with ``pose_scores <= 0`` from each star, in place."""
        indexes = range(len(predictions_dict["indexes"]))
        for idx in indexes:
            if len(predictions_dict["indexes"][idx]) == 1:
                continue
            valid_edges_curr = (
                torch.where(predictions_dict["pose_scores"][idx][0] > 0)[0]
            ).tolist()
            if (
                len(valid_edges_curr)
                == predictions_dict["pose_scores"][idx].shape[1]
            ):
                continue
            for key in predictions_dict:
                if key == "indexes":
                    predictions_dict[key][idx] = [
                        predictions_dict[key][idx][x] for x in valid_edges_curr
                    ]
                elif (
                    key == "points3d_virtual"
                    or key == "scales"
                    or key == "star_indexes"
                    or key == "image_index_to_star_index"
                ):
                    continue
                else:
                    if predictions_dict[key][idx] is None:
                        continue
                    predictions_dict[key][idx] = predictions_dict[key][idx][
                        :, valid_edges_curr
                    ]

    def _global_relative_extrinsics(
        self,
        predictions_dict: dict,
        idx: int,
        global_rotations: dict[int, np.ndarray],
        global_centers: dict[int, np.ndarray] | None = None,
    ) -> torch.Tensor:
        """Build global extrinsics for star ``idx`` relative to its first index.

        Returns a ``(1, N, 3, 4)`` tensor.

        When ``global_centers`` is ``None``, the translation block is zero and
        only the rotation block is meaningful.
        """
        star_indexes = predictions_dict["indexes"][idx]
        rotations = torch.from_numpy(
            np.stack([global_rotations[i] for i in star_indexes])
        ).unsqueeze(0)
        if global_centers is not None:
            translations = torch.from_numpy(
                np.stack(
                    [
                        -global_rotations[i] @ global_centers[i].reshape(3, 1)
                        for i in star_indexes
                    ]
                )
            ).unsqueeze(0)
        else:
            translations = torch.zeros(
                (1, len(star_indexes), 3, 1), dtype=rotations.dtype
            )
        extrinsics = torch.cat([rotations, translations], dim=-1)
        return restore_identity(extrinsics)

    @staticmethod
    def _rotation_errors(
        rotations_global: torch.Tensor,
        rotations_local: torch.Tensor,
    ) -> torch.Tensor:
        """Per-pair geodesic rotation error in rad for ``(N, 3, 3)`` inputs."""
        diff = (
            rotations_global.double()
            @ rotations_local.transpose(-1, -2).cpu().double()
        )
        return torch.acos(
            torch.clamp(
                (torch.einsum("bii -> b", diff) - 1) / 2,
                min=-1.0 + 1e-6,
                max=1.0 - 1e-6,
            )
        )

    def _filter_invalid_edges(
        self,
        predictions_dict: dict,
        global_rotations: dict[int, np.ndarray],
    ) -> list[tuple[int, int, float, torch.Tensor]]:
        """Zero out ``pose_scores`` for edges inconsistent with global solution.

        An edge ``(i, j)`` is filtered when the relative rotation implied by
        ``global_rotations`` differs from the local rotation by more than
        ``max_rot_error`` degrees. Modifies ``predictions_dict["pose_scores"]``
        in place; returns the list of filtered ``(idx, i, score, error)``
        tuples.
        """
        if "pose_constraints" in predictions_dict:
            threshold = np.deg2rad(self.max_rot_error)
            filtered = []
            for index, constraint in enumerate(
                predictions_dict["pose_constraints"]
            ):
                if not constraint["active"]:
                    continue
                if constraint["kind"] in {
                    "trajectory_odometry",
                    "trajectory_rotation_bridge",
                }:
                    continue
                first, second = constraint["first"], constraint["second"]
                rotation_global = (
                    global_rotations[second] @ global_rotations[first].T
                )
                rotation_local = constraint["pose"][:3, :3].cpu().double()
                error = self._rotation_errors(
                    torch.from_numpy(rotation_global).unsqueeze(0),
                    rotation_local.unsqueeze(0),
                )[0]
                if error > threshold:
                    constraint["active"] = False
                    filtered.append((index, 0, constraint["score"], error))
            if filtered:
                logger.info(
                    "Filtered %d / %d explicit group constraints by rotation",
                    len(filtered),
                    len(predictions_dict["pose_constraints"]),
                )
            return filtered

        indexes = range(len(predictions_dict["indexes"]))
        thres = np.deg2rad(self.max_rot_error)
        num_filtered = 0
        num_total = 0
        filtered_index = []
        for idx in indexes:
            extrinsics_global = self._global_relative_extrinsics(
                predictions_dict, idx, global_rotations
            )
            rotations_local = predictions_dict["extrinsics"][idx][0, :, :3, :3]
            errors = self._rotation_errors(
                extrinsics_global[0, :, :3, :3], rotations_local
            )
            invalid_mask = errors > thres

            invalid_mask = invalid_mask * (
                predictions_dict["pose_scores"][idx][0].cpu() > 0
            )

            num_total += rotations_local.shape[0] - 1

            if not invalid_mask.any():
                continue

            num_filtered += invalid_mask.sum().item()

            filtered_index.extend(
                [
                    (
                        idx,
                        i,
                        predictions_dict["pose_scores"][idx][0, i].item(),
                        errors[i],
                    )
                    for i in range(len(predictions_dict["indexes"][idx]))
                    if invalid_mask[i]
                ]
            )
            predictions_dict["pose_scores"][idx][0, invalid_mask] = 0.0

        if num_filtered > 0:
            logger.info(
                f"Number of filtered edges by the rotation error / total: "
                f"{num_filtered} / {num_total}"
            )
        return filtered_index

    # Filter edges by both rotation and translation consistency
    def _mark_inconsistent_edges(
        self,
        predictions_dict: dict,
        global_rotations: dict[int, np.ndarray],
        global_centers: dict[int, np.ndarray],
    ) -> None:
        """Flag edges whose rotation or translation disagrees with global pose.

        Populates ``predictions_dict["pose_inconsistent"][idx]`` with a boolean
        mask per neighbor; the mask is then consumed downstream (the scores
        themselves are not zeroed here).
        """
        indexes = range(len(predictions_dict["indexes"]))
        thres_rot = np.deg2rad(1.0)
        thres_trans = np.deg2rad(5.0)
        num_filtered = 0
        num_total = 0
        predictions_dict["pose_inconsistent"] = {}
        for idx in indexes:
            translation_local = predictions_dict["extrinsics"][idx][
                0, :, :3, 3
            ].double()

            extrinsics_global = self._global_relative_extrinsics(
                predictions_dict, idx, global_rotations, global_centers
            )
            translation_global = extrinsics_global[0, :, :3, 3]

            # Normalize
            safe_norm_global = torch.clamp(
                torch.linalg.norm(translation_global, dim=-1, keepdim=True),
                min=1e-6,
            )
            translation_global = translation_global / safe_norm_global
            safe_norm_local = torch.clamp(
                torch.linalg.norm(translation_local, dim=-1, keepdim=True),
                min=1e-6,
            )
            translation_local = translation_local / safe_norm_local

            # Compare the directions
            errors_trans = torch.acos(
                torch.clamp(
                    torch.einsum(
                        "bi,bi->b", translation_global, translation_local
                    ),
                    min=-1.0 + 1e-6,
                    max=1.0 - 1e-6,
                )
            )

            rotations_local = predictions_dict["extrinsics"][idx][0, :, :3, :3]
            errors_rot = self._rotation_errors(
                extrinsics_global[0, :, :3, :3], rotations_local
            )
            invalid_mask = (errors_rot > thres_rot) | (
                errors_trans > thres_trans
            )
            invalid_mask = invalid_mask * (
                predictions_dict["pose_scores"][idx][0].cpu() > 0
            )
            invalid_mask[0] = False  # never filter the first one

            predictions_dict["pose_inconsistent"][idx] = invalid_mask

            num_total += translation_local.shape[0] - 1

            num_filtered += invalid_mask.sum().item()

        if num_filtered > 0:
            logger.info(
                f"Number of filtered edges by the rotation error / total: "
                f"{num_filtered} / {num_total}"
            )

    def _boost_sequential_edges(
        self,
        predictions_dict: dict,
        boost_factor: float = 2.0,
    ) -> None:
        """
        Boost weights of sequential edges (neighboring frames).

        Args:
            predictions_dict: Dictionary containing pose_scores and indexes
            boost_factor: Factor to multiply the weight by (default: 2.0)
        """
        indexes = range(len(predictions_dict["indexes"]))
        seq_edges = getattr(self, "sequential_edges", set())
        num_boosted = 0

        for idx in indexes:
            center_idx = predictions_dict["indexes"][idx][0]
            for i, neighbor_idx in enumerate(predictions_dict["indexes"][idx]):
                if i == 0:
                    continue
                edge = (
                    min(center_idx, neighbor_idx),
                    max(center_idx, neighbor_idx),
                )
                if edge in seq_edges:
                    predictions_dict["pose_scores"][idx][0, i] *= boost_factor
                    num_boosted += 1

        logger.info(
            f"Boosted {num_boosted} sequential edges by factor {boost_factor}"
        )
