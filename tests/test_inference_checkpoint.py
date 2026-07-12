from types import SimpleNamespace

import torch

from gluemap.controllers.base_inference import BaseInferencePipeline


class _IndexDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {
            "star_indexes": torch.tensor(index),
            "value": torch.tensor(index),
        }


class _CheckpointPipeline(BaseInferencePipeline):
    _index_key = "star_indexes"
    _rerun_from_triggers = None
    _profiling_label = "checkpoint test"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed = []

    def _load_models(self):
        self.models = {}
        return self.models

    def _create_batch_inference(self, models):
        return None

    def _run_batch_step(self, batch_inference, batch):
        value = int(batch["value"].item())
        self.executed.append(value)
        return {"value": value}, {}

    def _pack_local_outputs(self, all_outputs, all_indices):
        return {"value": [output["value"] for output in all_outputs]}

    def _merge_gathered_outputs(self, data_list, index_mapping, dataset_size):
        raise AssertionError("Single-process test must not gather")

    def _postprocess_global_outputs(self, global_outputs, dataset):
        return global_outputs


def test_partial_checkpoint_resumes_only_missing_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    args = SimpleNamespace(
        curr_path=str(tmp_path),
        temp_path=str(tmp_path / "tmp"),
        distributed=False,
        num_workers=0,
        batch_size=1,
        force_load=False,
        rerun_from=None,
        resume_partial=True,
        checkpoint_every=2,
    )
    pipeline = _CheckpointPipeline(
        args,
        world_size=1,
        rank=0,
        file_name="star_result.pth",
        device="cpu",
        dtype=torch.float32,
    )
    partial_path = tmp_path / "star_result.pth.partial"
    pipeline._save_partial_checkpoint(
        str(partial_path),
        all_outputs=[{"value": 0}, {"value": 1}],
        all_indices=[0, 1],
        batch_times=[0.1, 0.1],
        extra_timings={},
    )

    outputs, timing = pipeline.run(_IndexDataset())

    assert pipeline.executed == [2, 3, 4]
    assert outputs["value"] == [0, 1, 2, 3, 4]
    assert outputs["star_indexes"] == [0, 1, 2, 3, 4]
    assert timing["num_batches"] == 5
    assert (tmp_path / "star_result.pth").is_file()
    assert not partial_path.exists()
