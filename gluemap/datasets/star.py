import argparse
import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from lightglue import ALIKED
from PIL import Image
from tqdm import tqdm

from gluemap.datasets.base import DemoBaseDataset
from gluemap.estimators.feature_extraction import (
    get_query_points_from_extractors,
)
from gluemap.utils.load_fn import calculate_image_shapes

logger = logging.getLogger(__name__)

ARKIT_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


class BaseStarDataset(DemoBaseDataset):
    """Star-graph dataset for multi-view inference.

    Each item is a "star": one query image plus its neighbors. This base
    class only initializes settings and placeholder attributes; a
    subclass (typically built from two-view results) must populate
    ``valid_edges``, ``edge_scores``, ``sequential_edges``, ``N``,
    ``images_list``, ``images_path``, ``images_shape_ori``, and the
    optional global pose/intrinsics tables before calling
    :meth:`__post_init__`.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Set up size, query-keypoint extractor, and placeholder attrs.

        Args:
            args: Parsed CLI/config namespace. Reads optional
                ``num_track_per_img`` (default ``1024``).
        """
        self.image_size = 518
        self.patch_size = 14

        self.num_tracks = (
            args.num_track_per_img
            if hasattr(args, "num_track_per_img")
            else 1024
        )

        # Build the keypoint extractor once and reuse it across
        # __getitem__ calls. Moved to the query image's device on first use.
        # detection_threshold=0.005 matches the det_thres default used in the
        # original implementation.
        self.query_extractors = [
            ALIKED(
                max_num_keypoints=self.num_tracks, detection_threshold=0.005
            ).eval()
        ]
        self._extractors_device = None

        self.query_points: np.ndarray | None = None  # needs to be initialized
        self.valid_edges: list[tuple[int, int]] | np.ndarray | None = (
            None  # needs to be initialized
        )
        self.edge_scores: dict[tuple[int, int], float] | None = (
            None  # per-edge scores for score-based pruning
        )
        self.sequential_edges: list[tuple[int, int]] | None = (
            None  # immediate sequential neighbor edges
        )

        # Number of nodes in the star structure, needs to be initialized
        self.N: int | None = None
        self.images_list: list[str] | None = None
        self.images_path: list[str] | None = None

        self.global_rotations: list[np.ndarray] | None = None
        self.global_centers: list[np.ndarray] | None = None
        self.global_intrinsics: list[np.ndarray] | None = None
        self.intrinsics_mapping: dict[int, int] | None = None

        self.polycam_priors_enabled = False
        self.polycam_use_depth = False
        self.polycam_use_pose = False
        self.polycam_use_intrinsics = False
        self.polycam_depth_unit_scale = 0.001
        self.polycam_prior_root: Path | None = None
        self.polycam_prior_records: list[dict] | None = None

    def attach_polycam_priors(self, args: argparse.Namespace) -> None:
        """Attach reset-aware Polycam priors for MapAnything stars.

        ``transforms.json`` is treated as the only prior source. Its poses are
        Nerfstudio/ARKit camera-to-world matrices, so they are converted to
        OpenCV camera-to-world before being passed to MapAnything.
        """
        priors_path = getattr(args, "polycam_priors_path", None)
        if not priors_path:
            return

        use_depth = bool(getattr(args, "mapanything_use_polycam_depth", False))
        use_pose = bool(getattr(args, "mapanything_use_polycam_pose", False))
        use_intrinsics = bool(
            getattr(args, "mapanything_use_polycam_intrinsics", False)
        )
        if not (use_depth or use_pose or use_intrinsics):
            return

        priors_path = Path(priors_path)
        prior_root = priors_path.parent
        with priors_path.open() as f:
            transforms = json.load(f)

        frames_by_name = {
            Path(frame["file_path"]).name: frame
            for frame in transforms.get("frames", [])
        }

        records = []
        for image_name in self.images_list:
            frame = frames_by_name.get(Path(image_name).name)
            if frame is None:
                raise KeyError(
                    f"No Polycam prior for image {image_name!r} in "
                    f"{priors_path}"
                )

            K = np.array(
                [
                    [float(frame["fl_x"]), 0.0, float(frame["cx"])],
                    [0.0, float(frame["fl_y"]), float(frame["cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            c2w_arkit = np.array(frame["transform_matrix"], dtype=np.float32)
            c2w_opencv = c2w_arkit @ ARKIT_TO_OPENCV

            depth_path = (
                prior_root / frame["depth_file_path"]
                if "depth_file_path" in frame
                else None
            )
            mask_path = (
                prior_root / frame["mask_path"] if "mask_path" in frame else None
            )
            records.append(
                {
                    "intrinsics": K,
                    "camera_pose": c2w_opencv,
                    "depth_path": depth_path,
                    "mask_path": mask_path,
                }
            )

        self.polycam_priors_enabled = True
        self.polycam_use_depth = use_depth
        self.polycam_use_pose = use_pose
        self.polycam_use_intrinsics = use_intrinsics
        self.polycam_depth_unit_scale = float(
            getattr(args, "polycam_depth_unit_scale", 0.001)
        )
        self.polycam_prior_root = prior_root
        self.polycam_prior_records = records

        logger.info(
            "Attached Polycam priors from %s (depth=%s pose=%s intrinsics=%s)",
            priors_path,
            use_depth,
            use_pose,
            use_intrinsics,
        )

    def __post_init__(self) -> None:
        """Build the star structure from ``self.valid_edges``.

        Prunes per-image neighborhoods to ``MAX_NEIGHBORS`` using
        ``self.edge_scores`` (preserving sequential neighbors and
        ensuring graph connectivity), filters
        ``self.sequential_edges`` to surviving edges, builds
        ``self.stars`` (sorted by neighbor count, descending),
        ``self.image_index_to_star_index``, ``self.images_change``,
        and finally ``self.pairs`` (undirected unique edges).

        Must be called by the subclass after the placeholder attributes
        have been populated.
        """
        # Initialize any additional attributes or configurations after the
        # object is created. Collect the indexes from the pairs

        # Initialize the star structure from the valid edges
        valid_edges = np.array(self.valid_edges, dtype=np.int64)
        valid_edges = np.concatenate(
            [valid_edges, valid_edges[:, ::-1]], axis=0
        )
        valid_edges = np.unique(valid_edges, axis=0)

        # Collect the neighbors for each node
        neighbors_all = {i: set() for i in range(self.N)}
        for i in range(len(valid_edges)):
            start, end = valid_edges[i]
            neighbors_all[start].add(end)
            neighbors_all[end].add(start)

        # Prune neighbors to top-K by score, then ensure connectivity
        MAX_NEIGHBORS = getattr(self, "max_neighbors", 25)
        MAX_SEQ = min(20, MAX_NEIGHBORS - 5)
        MIN_NONSEQ = 5

        # Build sequential edge lookup set
        seq_edge_set = (
            set(self.sequential_edges) if self.sequential_edges else set()
        )
        has_sequential = len(seq_edge_set) > 0

        if self.edge_scores is not None:
            for i in range(self.N):
                if len(neighbors_all[i]) <= MAX_NEIGHBORS:
                    continue

                if has_sequential:
                    # Split neighbors into sequential and non-sequential
                    seq_nbrs = []
                    nonseq_nbrs = []
                    for j in neighbors_all[i]:
                        edge = (min(i, j), max(i, j))
                        if edge in seq_edge_set:
                            seq_nbrs.append(j)
                        else:
                            nonseq_nbrs.append(j)

                    # Sample sequential neighbors: keep first & last,
                    # uniform middle
                    seq_nbrs.sort()
                    if len(seq_nbrs) > MAX_SEQ:
                        first, last = seq_nbrs[0], seq_nbrs[-1]
                        middle = seq_nbrs[1:-1]
                        n_middle = MAX_SEQ - 2
                        indices = np.round(
                            np.linspace(0, len(middle) - 1, n_middle)
                        ).astype(int)
                        sampled_middle = [
                            middle[idx] for idx in np.unique(indices)
                        ]
                        seq_nbrs = [first] + sampled_middle + [last]

                    # Fill remaining slots with top non-sequential by score
                    remaining = max(MAX_NEIGHBORS - len(seq_nbrs), MIN_NONSEQ)
                    nonseq_scored = sorted(
                        [
                            (
                                j,
                                self.edge_scores.get(
                                    (min(i, j), max(i, j)), 0.0
                                ),
                            )
                            for j in nonseq_nbrs
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    selected_nonseq = [x[0] for x in nonseq_scored[:remaining]]
                    neighbors_all[i] = set(seq_nbrs + selected_nonseq)
                else:
                    # No sequential edges: top-K by score
                    scored = [
                        (j, self.edge_scores.get((min(i, j), max(i, j)), 0.0))
                        for j in neighbors_all[i]
                    ]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    neighbors_all[i] = set(x[0] for x in scored[:MAX_NEIGHBORS])

            # Check connectivity; add back pruned edges if disconnected
            G = nx.Graph()
            G.add_nodes_from(range(self.N))
            for i in range(self.N):
                for j in neighbors_all[i]:
                    G.add_edge(i, j)

            if not nx.is_connected(G):
                # Collect pruned edges sorted by score descending
                original_set = set()
                for e in np.array(self.valid_edges, dtype=np.int64):
                    original_set.add(
                        (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))
                    )
                current_set = set()
                for i in range(self.N):
                    for j in neighbors_all[i]:
                        current_set.add((min(i, j), max(i, j)))
                pruned_edges = original_set - current_set
                pruned_with_scores = sorted(
                    [(e, self.edge_scores.get(e, 0.0)) for e in pruned_edges],
                    key=lambda x: x[1],
                    reverse=True,
                )

                for (u, v), _score in pruned_with_scores:
                    if nx.is_connected(G):
                        break
                    # Only add the edge if it bridges two different components
                    if nx.node_connected_component(
                        G, u
                    ) != nx.node_connected_component(G, v):
                        G.add_edge(u, v)
                        neighbors_all[u].add(v)
                        neighbors_all[v].add(u)
                logger.info("Re-added edges to restore connectivity")
            else:
                logger.info(
                    "Graph is already connected after "
                    f"top-{MAX_NEIGHBORS} pruning"
                )
        else:
            # Fallback: no scores available, random selection
            for i in range(self.N):
                if len(neighbors_all[i]) > MAX_NEIGHBORS:
                    neighbors = list(neighbors_all[i])
                    index = np.random.choice(
                        len(neighbors), size=MAX_NEIGHBORS, replace=False
                    )
                    neighbors_all[i] = set(np.array(neighbors)[index])

        # Filter sequential_edges to only keep edges that survived pruning
        if self.sequential_edges is not None:
            kept_edges = set()
            for i in range(self.N):
                for j in neighbors_all[i]:
                    kept_edges.add((min(i, j), max(i, j)))
            self.sequential_edges = [
                e for e in self.sequential_edges if e in kept_edges
            ]

        # Build stars from pruned neighbors
        stars = []
        for i in tqdm(range(self.N)):
            neighbors = neighbors_all[i]
            if len(neighbors) > 0:
                neighbors = np.array(list(neighbors), dtype=np.int64)
                stars.append(np.concatenate([[i], neighbors], axis=0))

        neighbors_lens = [len(x) for x in stars]
        # sort by the length in descending order
        sorted_idx = np.argsort(neighbors_lens)[::-1]
        stars = [stars[i] for i in sorted_idx]

        # Add a mapping between star index and image index
        self.image_index_to_star_index = {
            stars[i][0]: i for i in range(len(stars))
        }
        self.stars = stars

        # calculate the image_changes
        if self.force_square:
            new_shape_hw = (518, 518)
        else:
            # find the shape
            ori_shape = self.images_shape_ori[0]
            height, width = ori_shape

            if width > height:
                new_width = 518

                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width))

            else:
                new_height = 518

                # Calculate width maintaining aspect ratio, divisible by 14
                new_width = round(width * (new_height / height))

            new_shape_hw = (new_height, new_width)

        self.images_change = calculate_image_shapes(
            self.images_shape_ori, new_shape_hw
        )

        valid_edges = valid_edges[
            valid_edges[:, 0] < valid_edges[:, 1]
        ]  # only keep valid_edges where the first index is less than the second
        valid_edges = np.unique(valid_edges, axis=0)  # remove duplicates
        self.pairs = valid_edges

    def __len__(self) -> int:
        return len(self.stars)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the ``idx``-th star and its query keypoints.

        Lazily moves the ALIKED extractor to the query image's device on
        first use, then extracts up to ``self.num_tracks`` keypoints
        from the 1024-resolution query image.

        Args:
            idx: Star index in ``[0, len(self))``.

        Returns:
            Dict with keys ``images``, ``images_change``,
            ``images_shape_ori``, ``images_1024``, ``images_change_1024``,
            ``image_paths``, ``star_indexes``, ``indexes``, and
            ``query_points``. When global poses are available,
            additionally includes ``global_rotations``,
            ``global_centers``, and ``global_intrinsics``.
        """
        indexes = self.stars[idx]
        (
            images,
            images_ori,
            images_change,
            images_1024,
            images_change_1024,
            image_paths,
        ) = self.load_images(indexes, load_1024=True)

        query_image = images_1024[:1]
        if self._extractors_device != query_image.device:
            self.query_extractors = [
                e.to(query_image.device) for e in self.query_extractors
            ]
            self._extractors_device = query_image.device
        query_points = get_query_points_from_extractors(
            query_image,
            self.query_extractors,
            max_query_num=self.num_tracks,
            strict_num=False,
        )[0]

        global_rotations = None
        global_centers = None
        global_intrinsics = None
        if self.global_rotations is not None:
            # subsample the rotations, centers, intrinsics
            global_rotations = [self.global_rotations[i] for i in indexes]
            global_centers = [self.global_centers[i] for i in indexes]
            global_intrinsics = [
                self.global_intrinsics[self.intrinsics_mapping[i]][0]
                for i in indexes
            ]

        # extract the query points for the star structure
        batch = {
            "images": images,
            "images_change": np.array(images_change),
            "images_shape_ori": np.array(
                [images_ori[i].shape[-2:] for i in range(len(images_ori))]
            ),
            "images_1024": images_1024,
            "images_change_1024": np.array(images_change_1024),
            "image_paths": image_paths,
            "star_indexes": idx,
            "indexes": indexes.astype(np.int64),
            "query_points": query_points,
        }

        if global_rotations is not None:
            batch["global_rotations"] = np.array(global_rotations)
            batch["global_centers"] = np.array(global_centers)
            batch["global_intrinsics"] = np.array(global_intrinsics)

        if self.polycam_priors_enabled:
            batch.update(
                self._load_polycam_star_priors(
                    indexes.astype(np.int64),
                    images.shape[-2:],
                    images_change,
                )
            )

        return batch

    def _load_polycam_star_priors(
        self,
        indexes: np.ndarray,
        image_hw: tuple[int, int],
        images_change: list[list[float]],
    ) -> dict[str, torch.Tensor]:
        if self.polycam_prior_records is None:
            raise RuntimeError("Polycam priors requested before attach.")

        H, W = int(image_hw[0]), int(image_hw[1])
        output: dict[str, torch.Tensor] = {}

        if self.polycam_use_intrinsics:
            intrinsics = []
            for local_i, image_idx in enumerate(indexes):
                K = self.polycam_prior_records[int(image_idx)][
                    "intrinsics"
                ].copy()
                sx, sy, x0, y0 = images_change[local_i]
                K[0, 0] *= float(sx)
                K[1, 1] *= float(sy)
                K[0, 2] = K[0, 2] * float(sx) + float(x0)
                K[1, 2] = K[1, 2] * float(sy) + float(y0)
                intrinsics.append(K)
            output["polycam_intrinsics"] = torch.from_numpy(
                np.stack(intrinsics).astype(np.float32)
            )

        if self.polycam_use_pose:
            poses = [
                self.polycam_prior_records[int(image_idx)]["camera_pose"]
                for image_idx in indexes
            ]
            output["polycam_camera_poses"] = torch.from_numpy(
                np.stack(poses).astype(np.float32)
            )

        if self.polycam_use_depth:
            depths = []
            for local_i, image_idx in enumerate(indexes):
                rec = self.polycam_prior_records[int(image_idx)]
                depth_path = rec["depth_path"]
                if depth_path is None or not depth_path.exists():
                    raise FileNotFoundError(
                        f"Missing depth prior for image index {int(image_idx)}"
                    )

                depth = np.array(Image.open(depth_path), dtype=np.float32)
                depth *= self.polycam_depth_unit_scale
                mask_path = rec["mask_path"]
                if mask_path is not None and mask_path.exists():
                    mask = np.array(Image.open(mask_path), dtype=np.uint8) > 0
                else:
                    mask = np.isfinite(depth) & (depth > 0.0)

                depth_t = torch.from_numpy(depth)[None, None]
                mask_t = torch.from_numpy(mask.astype(np.float32))[None, None]

                src_h, src_w = depth.shape
                sx, sy, x0, y0 = images_change[local_i]
                new_w = max(1, int(round(src_w * float(sx))))
                new_h = max(1, int(round(src_h * float(sy))))
                depth_rs = F.interpolate(
                    depth_t,
                    size=(new_h, new_w),
                    mode="nearest",
                )[0, 0]
                mask_rs = (
                    F.interpolate(
                        mask_t,
                        size=(new_h, new_w),
                        mode="nearest",
                    )[0, 0]
                    > 0.5
                )

                canvas = torch.zeros((H, W), dtype=torch.float32)
                x_start = int(round(float(x0)))
                y_start = int(round(float(y0)))
                x_end = min(W, x_start + new_w)
                y_end = min(H, y_start + new_h)
                if x_start < W and y_start < H and x_end > 0 and y_end > 0:
                    src_x0 = max(0, -x_start)
                    src_y0 = max(0, -y_start)
                    dst_x0 = max(0, x_start)
                    dst_y0 = max(0, y_start)
                    src_x1 = src_x0 + (x_end - dst_x0)
                    src_y1 = src_y0 + (y_end - dst_y0)
                    patch = depth_rs[src_y0:src_y1, src_x0:src_x1]
                    patch_mask = mask_rs[src_y0:src_y1, src_x0:src_x1]
                    patch = torch.where(patch_mask, patch, torch.zeros_like(patch))
                    canvas[dst_y0:y_end, dst_x0:x_end] = patch
                depths.append(canvas)

            output["polycam_depth_z"] = torch.stack(depths, dim=0)

        return output
