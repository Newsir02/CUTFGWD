from __future__ import annotations

import pytest
import torch
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import LastNeighborLoader

from model import PyGTGNTeacher


def make_teacher() -> PyGTGNTeacher:
    return PyGTGNTeacher(
        event_times=torch.tensor([10, 20, 30, 40], dtype=torch.long),
        event_messages=torch.arange(8, dtype=torch.float).reshape(4, 2) / 10.0,
        num_nodes=5,
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        temporal_neighbors=3,
        dropout=0.0,
    )


def make_batch(
    event_indices: tuple[int, ...] = (0,),
    sources: tuple[int, ...] = (0,),
    positives: tuple[int, ...] = (1,),
    candidates: tuple[tuple[int, ...], ...] = ((1, 2),),
    targets: tuple[int, ...] = (0,),
) -> dict[str, torch.Tensor]:
    event_index = torch.tensor(event_indices, dtype=torch.long)
    event_times = torch.tensor([10, 20, 30, 40], dtype=torch.long)[event_index]
    event_messages = (
        torch.arange(8, dtype=torch.float).reshape(4, 2)[event_index] / 10.0
    )
    return {
        "event_index": event_index,
        "src": torch.tensor(sources, dtype=torch.long),
        "positive_dst": torch.tensor(positives, dtype=torch.long),
        "candidates": torch.tensor(candidates, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.long),
        "timestamp": event_index.float() / 3.0,
        "event_time": event_times,
        "edge_features": event_messages,
    }


def second_batch() -> dict[str, torch.Tensor]:
    return make_batch(
        event_indices=(1,),
        sources=(0,),
        positives=(2,),
        candidates=((2, 3),),
        targets=(0,),
    )


def test_teacher_uses_pyg_tgn_components() -> None:
    teacher = make_teacher()

    assert isinstance(teacher.memory, TGNMemory)
    assert isinstance(teacher.neighbor_loader, LastNeighborLoader)
    assert isinstance(teacher.gnn.conv, TransformerConv)


def test_teacher_can_stack_multiple_transformer_layers() -> None:
    teacher = PyGTGNTeacher(
        event_times=torch.tensor([10, 20, 30, 40], dtype=torch.long),
        event_messages=torch.arange(8, dtype=torch.float).reshape(4, 2) / 10.0,
        num_nodes=5,
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        temporal_neighbors=3,
        dropout=0.0,
        teacher_layers=2,
    ).eval()

    assert teacher.model_config["teacher_layers"] == 2
    assert len(teacher.gnn.convs) == 2
    assert all(isinstance(conv, TransformerConv) for conv in teacher.gnn.convs)
    output = teacher(make_batch(), update_memory=False)
    assert output["logits"].shape == (1, 2)


def test_teacher_scores_before_state_update() -> None:
    teacher = make_teacher().eval()
    batch = make_batch()

    output = teacher.score_candidates(batch, return_relations=True)

    assert teacher.state_is_empty
    assert not output["relation_mask"].any()
    teacher.update_state(batch)
    assert not teacher.state_is_empty
    assert int(teacher.neighbor_loader.cur_e_id) == 1
    assert torch.count_nonzero(teacher.memory.memory[0]) > 0
    assert torch.count_nonzero(teacher.memory.memory[1]) > 0


def test_forward_uses_official_batch_parallel_update_semantics() -> None:
    teacher = make_teacher().eval()
    batch = make_batch(
        event_indices=(0, 1),
        sources=(0, 0),
        positives=(1, 2),
        candidates=((1, 3), (2, 3)),
        targets=(0, 0),
    )

    output = teacher(batch, return_relations=True, update_memory=True)

    assert output["logits"].shape == (2, 2)
    assert not output["relation_mask"].any()
    assert int(teacher.neighbor_loader.cur_e_id) == 2


def test_reset_and_replay_are_deterministic() -> None:
    teacher = make_teacher().eval()
    batch = make_batch(
        event_indices=(0, 1),
        sources=(0, 0),
        positives=(1, 2),
        candidates=((1, 3), (2, 3)),
        targets=(0, 0),
    )

    first_logits = teacher(batch, update_memory=True)["logits"].detach().clone()
    first_memory = teacher.memory.memory.clone()
    first_neighbors = teacher.neighbor_loader.e_id.clone()
    teacher.reset_state()
    second_logits = teacher(batch, update_memory=True)["logits"].detach().clone()

    assert torch.allclose(first_logits, second_logits)
    assert torch.allclose(first_memory, teacher.memory.memory)
    assert torch.equal(first_neighbors, teacher.neighbor_loader.e_id)


def test_scoring_without_update_does_not_mutate_state() -> None:
    teacher = make_teacher().eval()
    before_memory = teacher.memory.memory.clone()
    before_neighbors = teacher.neighbor_loader.e_id.clone()

    output = teacher(make_batch(), update_memory=False)

    assert output["logits"].shape == (1, 2)
    assert torch.equal(before_memory, teacher.memory.memory)
    assert torch.equal(before_neighbors, teacher.neighbor_loader.e_id)
    assert teacher.state_is_empty


def test_teacher_rejects_out_of_order_batches() -> None:
    teacher = make_teacher().eval()
    teacher.update_state(make_batch())

    with pytest.raises(ValueError, match="reset_state"):
        teacher.score_candidates(make_batch(), return_relations=False)


def test_relation_output_matches_cutfgw_contract() -> None:
    teacher = make_teacher().eval()
    teacher.update_state(make_batch())

    output = teacher.score_candidates(second_batch(), return_relations=True)

    assert output["logits"].shape == (1, 2)
    assert output["relation_tokens"].shape == (1, 3, 6)
    assert output["relation_ages"].shape == (1, 3)
    assert output["relation_mask"].shape == (1, 3)
    assert output["relation_mass"].shape == (1, 3)
    assert output["relation_mask"].sum() == 1
    assert torch.isfinite(output["relation_tokens"]).all()
    assert torch.isfinite(output["relation_ages"]).all()
    assert torch.isfinite(output["relation_mass"]).all()
    assert torch.allclose(output["relation_mass"].sum(dim=1), torch.ones(1))


def test_runtime_state_and_full_event_data_are_excluded_from_state_dict() -> None:
    teacher = make_teacher()
    state_names = set(teacher.state_dict())

    assert not any(name.startswith("memory.memory") for name in state_names)
    assert not any("last_update" in name for name in state_names)
    assert not any("neighbor_loader" in name for name in state_names)
    assert "event_times" not in state_names
    assert "event_messages" not in state_names
    assert "association" not in state_names
    assert "stream_event_index" not in state_names
