import numpy as np

from gluemap.math.reprojection_error import (
    ReprojectionErrorType,
    compute_point_error,
)
from gluemap.utils.colmap import camera_from_intrinsics_matrix


def test_normalized_error_supports_two_focal_length_camera():
    intrinsics = np.array(
        [[800.0, 0.0, 384.0], [0.0, 1000.0, 512.0], [0.0, 0.0, 1.0]]
    )
    camera = camera_from_intrinsics_matrix(
        intrinsics,
        "PINHOLE",
        width=768,
        height=1024,
        camera_id=1,
    )
    world_point = np.array([0.0, 0.0, 2.0])
    observed = np.array([394.0, 512.0])

    error = compute_point_error(
        world_point,
        np.eye(3),
        np.zeros(3),
        observed,
        camera,
        ReprojectionErrorType.NORMALIZED,
    )

    np.testing.assert_allclose(error, 10.0 / 900.0)
