"""Native GLUEMAP star dataset backed by an audited Polycam RGB-D graph."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gluemap.datasets.polycam import (
    load_polycam_manifest,
    rotate_c2w_opencv_cw,
    rotate_intrinsics_cw,
    scale_intrinsics,
)
from gluemap.datasets.star import BaseStarDataset


class PolycamNativeStarDataset(BaseStarDataset):
    """One native GLUEMAP anchor star per included Polycam frame."""

    def __init__(
        self,
        args,
        manifest_path: Path,
        source_root: Path,
        dataset_directory: Path,
        frontend_edges: Path,
        *,
        manifest_start: int | None = None,
        manifest_end: int | None = None,
    ) -> None:
        super().__init__(args)
        self.manifest = load_polycam_manifest(manifest_path)
        self.source_root = source_root
        mapping = json.loads(
            (dataset_directory / "index_mapping.json").read_text(
                encoding="utf-8"
            )
        )
        if manifest_start is not None:
            mapping = [
                row
                for row in mapping
                if row["manifest_index"] >= manifest_start
            ]
        if manifest_end is not None:
            mapping = [
                row for row in mapping if row["manifest_index"] <= manifest_end
            ]
        self.native_to_manifest = [
            int(row["manifest_index"]) for row in mapping
        ]
        manifest_to_native = {
            manifest_index: native_index
            for native_index, manifest_index in enumerate(
                self.native_to_manifest
            )
        }
        original_mapping = {
            int(row["manifest_index"]): row["image_name"]
            for row in json.loads(
                (dataset_directory / "index_mapping.json").read_text(
                    encoding="utf-8"
                )
            )
        }

        valid_edges = []
        edge_scores = {}
        sequential_edges = []
        with frontend_edges.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                manifest_first, manifest_second = (
                    int(row["first"]),
                    int(row["second"]),
                )
                if (
                    manifest_first not in manifest_to_native
                    or manifest_second not in manifest_to_native
                ):
                    continue
                first = manifest_to_native[manifest_first]
                second = manifest_to_native[manifest_second]
                edge = (min(first, second), max(first, second))
                valid_edges.append(edge)
                edge_scores[edge] = max(
                    edge_scores.get(edge, 0.0), float(row["score"])
                )
                if row["acceptance_reason"] == "temporal":
                    sequential_edges.append(edge)

        self.valid_edges = np.asarray(sorted(set(valid_edges)), dtype=np.int64)
        self.edge_scores = edge_scores
        self.sequential_edges = sorted(set(sequential_edges))
        self.N = len(self.native_to_manifest)
        self.images_list = [
            original_mapping[index] for index in self.native_to_manifest
        ]
        images_root = str(dataset_directory / "images")
        self.images_path = [images_root] * self.N
        self.images_shape_ori = [(1024, 768)] * self.N
        self.force_square = False
        self.camera_model = "PINHOLE"
        self.intrinsics_mapping = {index: index for index in range(self.N)}
        self.known_intrinsics = []
        self.pose_priors_c2w = []
        for manifest_index in self.native_to_manifest:
            frame = self.manifest["frames"][manifest_index]
            source_width, source_height = frame["rgb_size_wh"]
            upright_intrinsics = rotate_intrinsics_cw(
                np.asarray(frame["intrinsics_rgb"], dtype=np.float64),
                (source_width, source_height),
            )
            self.known_intrinsics.append(
                torch.from_numpy(upright_intrinsics).float().unsqueeze(0)
            )
            self.pose_priors_c2w.append(
                rotate_c2w_opencv_cw(
                    np.asarray(frame["corrected_c2w_opencv"], dtype=np.float64)
                )
            )
        self.query_points = [None] * len(self.valid_edges)
        self.max_neighbors = getattr(args, "max_neighbors", 25)
        self.__post_init__()

    def __getitem__(self, index: int) -> dict:
        batch = super().__getitem__(index)
        target_height, target_width = batch["images"].shape[-2:]
        depths = []
        intrinsics = []
        manifest_indices = []
        for local_index, native_index in enumerate(batch["indexes"]):
            manifest_index = self.native_to_manifest[int(native_index)]
            manifest_indices.append(manifest_index)
            frame = self.manifest["frames"][manifest_index]
            with Image.open(
                self.source_root / frame["paths"]["depth"]
            ) as image:
                depth = np.asarray(image, dtype=np.float32)
            with Image.open(
                self.source_root / frame["paths"]["confidence"]
            ) as image:
                confidence = np.asarray(image, dtype=np.uint8)
            depth = np.rot90(depth, k=3).copy() * float(
                self.manifest["depth_unit_m"]
            )
            confidence = np.rot90(confidence, k=3).copy()
            depth[confidence < 1] = 0.0
            source_width, source_height = frame["rgb_size_wh"]
            scale_x, scale_y, offset_x, offset_y = batch["images_change"][
                local_index
            ]
            upright_width, upright_height = source_height, source_width
            content_width = int(round(upright_width * scale_x))
            content_height = int(round(upright_height * scale_y))
            resized_depth = F.interpolate(
                torch.from_numpy(depth)[None, None],
                size=(content_height, content_width),
                mode="nearest",
            )[0, 0]
            depth_tensor = torch.zeros(
                (target_height, target_width), dtype=resized_depth.dtype
            )
            left, top = int(round(offset_x)), int(round(offset_y))
            depth_tensor[
                top : top + content_height, left : left + content_width
            ] = resized_depth
            depths.append(depth_tensor)

            upright_intrinsics = rotate_intrinsics_cw(
                np.asarray(frame["intrinsics_rgb"], dtype=np.float64),
                (source_width, source_height),
            )
            intrinsics.append(
                torch.from_numpy(
                    scale_intrinsics(
                        upright_intrinsics,
                        (upright_width, upright_height),
                        (content_width, content_height),
                    )
                ).float()
            )
            intrinsics[-1][0, 2] += offset_x
            intrinsics[-1][1, 2] += offset_y

        batch["metric_depths"] = torch.stack(depths)
        batch["metric_intrinsics"] = torch.stack(intrinsics)
        batch["manifest_indices"] = np.asarray(manifest_indices, dtype=np.int64)
        return batch
