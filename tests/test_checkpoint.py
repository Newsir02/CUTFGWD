from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dataset import build_synthetic_temporal_graph
from model import PyGTGNTeacher
from train import load_teacher_from_checkpoint
from util.checkpoint import load_model_checkpoint, save_model_checkpoint


def make_bundle():
    return build_synthetic_temporal_graph(
        num_nodes=12,
        num_events=20,
        feature_dim=4,
        edge_feature_dim=2,
        num_candidates=3,
        seed=17,
    )


def make_teacher() -> PyGTGNTeacher:
    bundle = make_bundle()
    return PyGTGNTeacher(
        event_times=bundle.data.t,
        event_messages=bundle.data.msg,
        num_nodes=bundle.node_features.size(0),
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        temporal_neighbors=3,
        dropout=0.0,
    )


def make_event_batch() -> dict[str, torch.Tensor]:
    return {
        "event_index": torch.tensor([0]),
        "src": torch.tensor([0]),
        "positive_dst": torch.tensor([1]),
        "candidates": torch.tensor([[1, 2]]),
        "target": torch.tensor([0]),
        "timestamp": torch.tensor([0.0]),
        "event_time": torch.tensor([0]),
        "edge_features": torch.zeros(1, 2),
    }


def test_checkpoint_is_versioned_and_excludes_pyg_runtime_state(
    tmp_path: Path,
) -> None:
    model = make_teacher().eval()
    model.update_state(make_event_batch())
    path = tmp_path / "teacher.pt"

    save_model_checkpoint(
        path,
        model,
        model_name="PyGTGNTeacher",
        model_config=model.model_config,
        args={"stage": "teacher"},
    )
    payload = load_model_checkpoint(
        path,
        expected_model_name="PyGTGNTeacher",
    )

    assert payload["format_version"] == 1
    assert payload["model_name"] == "PyGTGNTeacher"
    assert payload["args"]["stage"] == "teacher"
    state_names = set(payload["model"])
    assert not any(name.startswith("memory.memory") for name in state_names)
    assert not any("last_update" in name for name in state_names)
    assert not any("neighbor_loader" in name for name in state_names)
    assert "event_times" not in state_names
    assert "event_messages" not in state_names


def test_checkpoint_rejects_legacy_teacher_type(tmp_path: Path) -> None:
    model = make_teacher()
    path = tmp_path / "teacher.pt"
    save_model_checkpoint(
        path,
        model,
        model_name="TGNTeacher",
        model_config=model.model_config,
        args={},
    )

    with pytest.raises(ValueError, match="PyGTGNTeacher"):
        load_model_checkpoint(path, expected_model_name="PyGTGNTeacher")


def test_checkpoint_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_model_checkpoint(
            tmp_path / "missing.pt",
            expected_model_name="PyGTGNTeacher",
        )


def test_teacher_loader_rejects_a_different_data_signature(
    tmp_path: Path,
) -> None:
    model = make_teacher()
    bundle = make_bundle()
    path = tmp_path / "teacher.pt"
    save_model_checkpoint(
        path,
        model,
        model_name="PyGTGNTeacher",
        model_config={**model.model_config, "data_signature": "dataset-a"},
        args={},
    )

    with pytest.raises(ValueError, match="data_signature"):
        load_teacher_from_checkpoint(
            path,
            bundle=bundle,
            data_signature="dataset-b",
            device=torch.device("cpu"),
        )
