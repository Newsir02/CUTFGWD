from __future__ import annotations

import torch

from model import LightSTMLPStudent
from model.layers import ResidualMLPBlock


def make_student() -> LightSTMLPStudent:
    return LightSTMLPStudent(
        node_features=torch.zeros(5, 3),
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        relation_slots=3,
        dropout=0.0,
    )


def make_batch() -> dict[str, torch.Tensor]:
    return {
        "src": torch.tensor([0]),
        "candidates": torch.tensor([[1, 2, 3]]),
        "timestamp": torch.tensor([0.2]),
        "history_nodes": torch.tensor([[1, 2]]),
        "history_times": torch.tensor([[0.0, 0.1]]),
        "history_mask": torch.tensor([[True, True]]),
    }


def test_student_is_invariant_to_graph_history_fields() -> None:
    model = make_student().eval()
    batch = make_batch()

    first = model(batch)["logits"]
    changed = {
        **batch,
        "history_nodes": torch.tensor([[4, 4]]),
        "history_times": torch.tensor([[99.0, 99.0]]),
        "history_mask": torch.tensor([[False, False]]),
    }
    second = model(changed)["logits"]

    assert torch.equal(first, second)


def test_student_uses_three_residual_temporal_blocks_by_default() -> None:
    model = make_student()

    assert len(model.temporal_blocks) == 3
    assert all(isinstance(block, ResidualMLPBlock) for block in model.temporal_blocks)


def test_student_outputs_fixed_ordered_relation_slots() -> None:
    model = make_student().eval()

    output = model(make_batch(), return_relations=True)

    assert output["logits"].shape == (1, 3)
    assert output["relation_tokens"].shape == (1, 3, 6)
    assert output["relation_ages"].shape == (1, 3)
    assert output["relation_mass_logits"].shape == (1, 3)
    assert torch.all(output["relation_ages"] >= 0.0)
    assert torch.all(output["relation_ages"] <= 1.0)
    assert torch.all(
        output["relation_ages"][:, 1:] <= output["relation_ages"][:, :-1]
    )


def test_student_requires_only_graph_free_query_fields() -> None:
    model = make_student().eval()
    graph_free_batch = {
        "src": torch.tensor([0]),
        "candidates": torch.tensor([[1, 2, 3]]),
        "timestamp": torch.tensor([0.2]),
    }

    output = model(graph_free_batch, return_relations=False)

    assert set(output) == {"logits"}
    assert torch.isfinite(output["logits"]).all()
