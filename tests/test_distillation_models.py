from __future__ import annotations

import torch

from loss import CUTFGWDDistillation
from model import LightSTMLPStudent, PyGTGNTeacher


def make_event_batch() -> dict[str, torch.Tensor]:
    return {
        "event_index": torch.tensor([0, 1, 2]),
        "src": torch.tensor([0, 0, 0]),
        "positive_dst": torch.tensor([1, 2, 3]),
        "candidates": torch.tensor(
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            dtype=torch.long,
        ),
        "target": torch.tensor([0, 1, 2]),
        "timestamp": torch.tensor([0.0, 0.5, 1.0]),
        "event_time": torch.tensor([10, 20, 30]),
        "edge_features": torch.zeros(3, 2),
    }


def test_pyg_tgn_and_lightst_backpropagate_through_cutfgw() -> None:
    node_features = torch.zeros(5, 3)
    teacher = PyGTGNTeacher(
        event_times=torch.tensor([10, 20, 30]),
        event_messages=torch.zeros(3, 2),
        num_nodes=5,
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        temporal_neighbors=3,
        dropout=0.0,
    ).eval()
    student = LightSTMLPStudent(
        node_features,
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        relation_slots=3,
        dropout=0.0,
    )
    objective = CUTFGWDDistillation(
        fgw_iterations=1,
        sinkhorn_iterations=3,
    )
    batch = make_event_batch()

    with torch.no_grad():
        teacher_output = teacher(batch, return_relations=True, update_memory=True)
    student_output = student(batch, return_relations=True)
    total, parts = objective(student_output, teacher_output, batch["target"])
    total.backward()

    assert torch.isfinite(total)
    assert student.node_embeddings.weight.grad is not None
    assert torch.isfinite(student.node_embeddings.weight.grad).all()
    assert set(parts) == {
        "total",
        "task",
        "logit",
        "rank",
        "anchor",
        "relation",
        "diversity",
    }
