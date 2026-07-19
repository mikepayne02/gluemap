from types import SimpleNamespace

import numpy as np
import torch

import gluemap.estimators.rotation_averaging as rotation_module


class _FakeProblem:
    def __init__(self):
        self.parameter_blocks = set()
        self.constant_blocks = []

    def add_residual_block(self, _cost, _loss, parameters):
        self.parameter_blocks.update(map(id, parameters))

    def has_parameter_block(self, parameter):
        return id(parameter) in self.parameter_blocks

    def set_manifold(self, _parameter, _manifold):
        pass

    def set_parameter_block_constant(self, parameter):
        self.constant_blocks.append(parameter)


def test_relative_rotation_solver_fixes_global_gauge(monkeypatch):
    problem = _FakeProblem()
    monkeypatch.setattr(rotation_module.pyceres, "Problem", lambda: problem)
    monkeypatch.setattr(
        rotation_module.pyceres, "LossFunction", lambda _config: object()
    )
    monkeypatch.setattr(
        rotation_module.pyceres, "QuaternionManifold", lambda: object()
    )
    monkeypatch.setattr(
        rotation_module.pyceres,
        "SolverOptions",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        rotation_module.pyceres,
        "SolverSummary",
        lambda: SimpleNamespace(BriefReport=lambda: "ok"),
    )
    monkeypatch.setattr(rotation_module.pyceres, "solve", lambda *_args: None)
    monkeypatch.setattr(
        rotation_module.pygluemap,
        "RotationGeodesicError",
        lambda _rotation: object(),
    )

    extrinsics = torch.eye(4, dtype=torch.float64)[:3].reshape(1, 1, 3, 4)
    extrinsics = extrinsics.repeat(1, 2, 1, 1)
    predictions = {
        "indexes": [[0, 1]],
        "pose_scores": [torch.ones((1, 2), dtype=torch.float64)],
        "extrinsics": [extrinsics],
    }

    rotations = rotation_module.rotation_averaging(predictions)

    assert set(rotations) == {0, 1}
    assert len(problem.constant_blocks) == 1
    np.testing.assert_allclose(problem.constant_blocks[0], [1.0, 0.0, 0.0, 0.0])
