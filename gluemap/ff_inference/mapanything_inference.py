import torch
from mapanything.utils.geometry import closed_form_pose_inverse
from mapanything.utils.image import preprocess_inputs

from gluemap.ff_inference.local_inference import LocalInference


class MapAnythingLocalInference(LocalInference):
    """Local inference for MapAnything backbone."""

    @torch.no_grad()
    def predict(self, batch: dict) -> dict:
        """Run the MapAnything backbone on a batch of images.

        Args:
            batch: Dict with ``"images"`` of shape ``(B, N, 3, H, W)``.

        Returns:
            Dict with ``depth`` of shape ``(1, N, H, W)``, ``depth_conf`` of
            shape ``(1, N, H, W)``, ``extrinsics`` of shape ``(1, N, 4, 4)``
            and ``intrinsics`` of shape ``(1, N, 3, 3)``.
        """
        images = batch["images"].to(self.device).contiguous()
        processed_views = self._compose_input_views(
            images,
            polycam_depth_z=batch.get("polycam_depth_z"),
            polycam_camera_poses=batch.get("polycam_camera_poses"),
            polycam_intrinsics=batch.get("polycam_intrinsics"),
        )

        use_amp = torch.cuda.is_available()
        amp_dtype = (
            torch.bfloat16
            if use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        predictions = self.model.infer(
            processed_views,
            memory_efficient_inference=False,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            confidence_percentile=10,
            ignore_calibration_inputs=False,
            ignore_depth_inputs=False,
            ignore_pose_inputs=False,
            ignore_depth_scale_inputs=False,
            ignore_pose_scale_inputs=False,
        )

        return self._retrieve_result(predictions)

    @staticmethod
    def _compose_input_views(
        images: torch.Tensor,
        global_rotations: torch.Tensor | None = None,
        global_centers: torch.Tensor | None = None,
        global_intrinsics: torch.Tensor | None = None,
        polycam_depth_z: torch.Tensor | None = None,
        polycam_camera_poses: torch.Tensor | None = None,
        polycam_intrinsics: torch.Tensor | None = None,
    ) -> list[dict]:
        """Convert an image tensor into MapAnything's per-view input format.

        Each output view contains an ``"img"`` of shape ``(H, W, 3)``;
        camera poses and/or intrinsics are attached if the corresponding
        global tensors are provided. The resulting list is then run through
        ``mapanything.utils.image.preprocess_inputs``.

        Args:
            images: Image batch of shape ``(B, N, 3, H, W)`` or
                ``(N, 3, H, W)``. A leading batch dimension is added when
                missing; only the first batch entry is used.
            global_rotations: Optional rotations of shape ``(B, N, 3, 3)``
                expressed as world-from-camera matrices.
            global_centers: Optional camera centers of shape ``(B, N, 3)``.
                Required together with ``global_rotations``.
            global_intrinsics: Optional intrinsics of shape ``(B, N, 3, 3)``.
            polycam_depth_z: Optional metric depth priors of shape
                ``(B, N, H, W)`` or ``(N, H, W)`` in the same resized/padded
                image frame as ``images``.
            polycam_camera_poses: Optional OpenCV camera-to-world pose priors
                of shape ``(B, N, 4, 4)`` or ``(N, 4, 4)``.
            polycam_intrinsics: Optional intrinsics of shape ``(B, N, 3, 3)``
                or ``(N, 3, 3)`` in the same resized/padded image frame.

        Returns:
            List of view dicts pre-processed for MapAnything's ``infer`` API.
        """
        import numpy as np

        if images.ndim == 4:
            images = images.unsqueeze(0)
        if polycam_depth_z is not None and polycam_depth_z.ndim == 3:
            polycam_depth_z = polycam_depth_z.unsqueeze(0)
        if polycam_camera_poses is not None and polycam_camera_poses.ndim == 3:
            polycam_camera_poses = polycam_camera_poses.unsqueeze(0)
        if polycam_intrinsics is not None and polycam_intrinsics.ndim == 3:
            polycam_intrinsics = polycam_intrinsics.unsqueeze(0)

        input_views = []
        for i in range(images.shape[1]):
            view = {"img": images[0, i].permute(1, 2, 0)}  # (H, W, 3)

            if polycam_camera_poses is not None:
                view["camera_poses"] = polycam_camera_poses[0, i]
                view["is_metric_scale"] = True
            elif global_rotations is not None:
                camera_pose = np.eye(4, dtype=np.float32)
                camera_pose[:3, :3] = global_rotations[0, i].T
                camera_pose[:3, 3] = global_centers[0, i]
                view["camera_poses"] = camera_pose

            if polycam_intrinsics is not None:
                view["intrinsics"] = polycam_intrinsics[0, i]
            elif global_intrinsics is not None:
                view["intrinsics"] = global_intrinsics[0, i]

            if polycam_depth_z is not None:
                view["depth_z"] = polycam_depth_z[0, i]
                view["is_metric_scale"] = True

            input_views.append(view)

        input_views = preprocess_inputs(input_views)
        return input_views

    @staticmethod
    def _retrieve_result(predictions: list[dict]) -> dict[str, torch.Tensor]:
        """Stack MapAnything's per-view predictions into batched tensors.

        Per-view ``intrinsics``, ``camera_poses``, ``depth_z`` and ``conf``
        are stacked along a new view dimension and the camera-to-world poses
        are inverted via
        ``mapanything.utils.geometry.closed_form_pose_inverse`` to produce
        world-from-camera extrinsics.

        Args:
            predictions: List of per-view prediction dicts returned by
                ``model.infer``.

        Returns:
            Dict with ``depth`` of shape ``(1, N, H, W)``, ``depth_conf`` of
            shape ``(1, N, H, W)``, ``extrinsics`` of shape ``(1, N, 4, 4)``
            and ``intrinsics`` of shape ``(1, N, 3, 3)``.
        """
        extrinsics = []
        intrinsics = []
        depths = []
        confs = []

        for _view_idx, pred in enumerate(predictions):
            intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
            camera_pose_torch = pred["camera_poses"][0]  # (4, 4)
            extrinsics.append(camera_pose_torch)
            intrinsics.append(intrinsics_torch)
            depths.append(pred["depth_z"][0, :, :])  # (H, W, 1)
            confs.append(pred["conf"][0, :, :])  # (H, W)

        depth_z = torch.stack(depths, dim=0).unsqueeze(0)  # (1, N, H, W)
        conf_z = torch.stack(confs, dim=0).unsqueeze(0)  # (1, N, H, W)

        extrinsics = closed_form_pose_inverse(
            torch.stack(extrinsics, dim=0)
        ).unsqueeze(0)
        intrinsics = torch.stack(intrinsics, dim=0).unsqueeze(0)

        return {
            "depth": depth_z,
            "depth_conf": conf_z,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
        }
