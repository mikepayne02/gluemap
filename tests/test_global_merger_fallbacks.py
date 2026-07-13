"""Regression tests for incomplete global camera initialization."""

import numpy as np
from scipy.spatial.transform import Rotation

from gluemap.controllers.global_merger import (
    _fill_missing_centers_temporally,
    _fill_missing_rotations_temporally,
)


def test_missing_centers_are_linearly_interpolated():
    centers = {
        0: np.array([0.0, 0.0, 0.0]),
        3: np.array([3.0, 6.0, 9.0]),
    }

    filled = _fill_missing_centers_temporally(centers, 4)

    np.testing.assert_allclose(filled[1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(filled[2], [2.0, 4.0, 6.0])


def test_missing_rotations_are_slerped():
    rotations = {
        0: np.eye(3),
        2: Rotation.from_euler("z", 90, degrees=True).as_matrix(),
    }

    filled = _fill_missing_rotations_temporally(rotations, 3)

    expected = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    np.testing.assert_allclose(filled[1], expected, atol=1e-7)


def test_edge_runs_copy_nearest_estimate():
    centers = {2: np.array([2.0, 4.0, 6.0])}
    rotations = {
        2: Rotation.from_euler("x", 15, degrees=True).as_matrix()
    }

    filled_centers = _fill_missing_centers_temporally(centers, 4)
    filled_rotations = _fill_missing_rotations_temporally(rotations, 4)

    for idx in (0, 1, 3):
        np.testing.assert_allclose(filled_centers[idx], centers[2])
        np.testing.assert_allclose(filled_rotations[idx], rotations[2])
