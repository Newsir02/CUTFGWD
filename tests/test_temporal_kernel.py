from __future__ import annotations

import torch

from dataset import CandidateBatchLoader, build_synthetic_temporal_graph
from loss import (
    CUTFGWDDistillation,
    candidate_relation_structure_loss,
    hard_negative_margin_loss,
    pairwise_relation_kd_loss,
    support_exact_relation_content_loss,
    temporal_kernel_distillation_loss,
    temporal_relational_drift_loss,
)
from model import LightGNNStudent
from model.temporal_neighbor_store import ExactTemporalNeighborStore
from train import build_teacher


def test_exact_store_keeps_newest_events_under_hub_overflow() -> None:
    store = ExactTemporalNeighborStore(num_nodes=4, size=3)
    store.insert(
        torch.tensor([0, 0, 0, 0]),
        torch.tensor([1, 2, 3, 1]),
        event_ids=torch.tensor([5, 6, 7, 8]),
    )
    assert store.e_id[0].tolist() == [8, 7, 6]
    assert store.neighbors[0].tolist() == [1, 3, 2]
    # The reverse direction is inserted too, with the same event ids.
    assert store.e_id[1].tolist()[0] == 8


def test_exact_store_merges_old_and_new_events() -> None:
    store = ExactTemporalNeighborStore(num_nodes=3, size=2)
    store.insert(torch.tensor([0]), torch.tensor([1]), event_ids=torch.tensor([10]))
    store.insert(torch.tensor([0]), torch.tensor([2]), event_ids=torch.tensor([11]))
    assert store.e_id[0].tolist() == [11, 10]
    assert store.neighbors[0].tolist() == [2, 1]


def test_temporal_kernel_loss_renormalises_shared_support() -> None:
    student_mass = torch.tensor([[0.625, 0.375]])
    student_ids = torch.tensor([[10, 11]])
    student_mask = torch.ones(1, 2, dtype=torch.bool)
    teacher_mass = torch.tensor([[0.2, 0.5, 0.3]])
    teacher_ids = torch.tensor([[9, 10, 11]])
    teacher_mask = torch.ones(1, 3, dtype=torch.bool)

    loss = temporal_kernel_distillation_loss(
        student_mass,
        student_ids,
        student_mask,
        teacher_mass,
        teacher_ids,
        teacher_mask,
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1.0e-7)

    shifted = temporal_kernel_distillation_loss(
        torch.tensor([[1.0, 0.0]]),
        student_ids,
        student_mask,
        teacher_mass,
        teacher_ids,
        teacher_mask,
    )
    assert float(shifted) > 0.0


def test_candidate_structure_loss_is_zero_for_matching_geometry() -> None:
    teacher = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    # Different hidden dimension and isotropic scaling must not matter.
    student = 3.0 * teacher
    loss = candidate_relation_structure_loss(student, teacher)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1.0e-6)

    collapsed = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    )
    assert (
        float(
            candidate_relation_structure_loss(
                collapsed, teacher, divergence="mse"
            )
        )
        > 0.0
    )


def test_hard_negative_margin_loss() -> None:
    student = torch.tensor([[2.0, 1.0, 0.0]])
    teacher = torch.tensor([[3.0, 1.5, 0.0]])
    target = torch.tensor([0])
    matched = hard_negative_margin_loss(student, student, target)
    assert torch.isclose(matched, torch.tensor(0.0), atol=1.0e-6)
    shifted = hard_negative_margin_loss(student, teacher, target)
    assert float(shifted) > 0.0
    # Positive is the argmax for both; hard negative changes with the logits.
    student2 = torch.tensor([[0.0, 4.0, 3.0]])
    teacher2 = torch.tensor([[0.0, 5.0, 3.5]])
    assert float(hard_negative_margin_loss(student2, teacher2, target)) > 0.0


def test_support_exact_relation_content_loss() -> None:
    student = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    teacher = student.clone()
    mask = torch.ones(1, 3, dtype=torch.bool)
    matched = support_exact_relation_content_loss(student, teacher, mask)
    assert torch.isclose(matched, torch.tensor(0.0), atol=1.0e-6)
    changed = teacher.clone()
    changed[0, 2] = torch.tensor([-1.0, 0.0])
    assert float(support_exact_relation_content_loss(student, changed, mask)) > 0.0
    # Rows with fewer valid teacher slots than student slots are skipped.
    short_mask = torch.tensor([[True, True, False]])
    skipped = support_exact_relation_content_loss(student, changed, short_mask)
    assert torch.isclose(skipped, torch.tensor(0.0))


def test_pairwise_relation_kd_prefers_matching_preferences() -> None:
    teacher = torch.tensor([[2.0, 1.0, 0.0]])
    matched = pairwise_relation_kd_loss(teacher, teacher, temperature=1.0)
    reversed_preference = pairwise_relation_kd_loss(-teacher, teacher, temperature=1.0)
    assert float(matched) < float(reversed_preference)


def test_temporal_drift_loss_matches_direction() -> None:
    source_ids = torch.tensor([0, 0])
    event_index = torch.tensor([10, 20])
    cache: dict = {}
    # First observation cannot form a drift; it only fills the cache.
    first = temporal_relational_drift_loss(
        torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
        source_ids,
        event_index,
        cache,
        num_nodes=2,
        max_gap=100,
    )
    assert torch.isclose(first, torch.tensor(0.0))
    # Same drift direction => zero loss.
    aligned = temporal_relational_drift_loss(
        torch.tensor([[1.0, 4.0]]),
        torch.tensor([[0.0, 5.0]]),
        torch.tensor([0]),
        torch.tensor([30]),
        cache,
        num_nodes=2,
        max_gap=100,
    )
    assert torch.isclose(aligned, torch.tensor(0.0), atol=1.0e-6)
    # Orthogonal drift => positive loss.
    cache = {}
    temporal_relational_drift_loss(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([0]),
        torch.tensor([10]),
        cache,
        num_nodes=2,
        max_gap=100,
    )
    orthogonal = temporal_relational_drift_loss(
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([0]),
        torch.tensor([20]),
        cache,
        num_nodes=2,
        max_gap=100,
    )
    assert float(orthogonal) > 0.0


def test_kernel_distillation_backpropagates_through_attention() -> None:
    bundle = build_synthetic_temporal_graph(
        num_nodes=14,
        num_events=80,
        feature_dim=4,
        edge_feature_dim=2,
        num_candidates=3,
        history_size=4,
        seed=23,
    )
    loader = CandidateBatchLoader(bundle, "train", 8, 3, 23, 0)
    teacher = build_teacher(
        bundle,
        type(
            "Args",
            (),
            {
                "teacher_hidden": 8,
                "time_dim": 4,
                "relation_dim": 8,
                "temporal_neighbors": 4,
                "dropout": 0.0,
                "teacher_layers": 1,
                "teacher_node_features": False,
            },
        )(),
    )
    student = LightGNNStudent(
        bundle.node_features,
        hidden_dim=8,
        time_dim=4,
        relation_dim=8,
        relation_slots=3,
        dropout=0.0,
        num_blocks=2,
        neighbors=4,
        aggregation="kernel",
        kernel_hidden=8,
        event_times=bundle.data.t,
        time_span=float(bundle.data.t[-1] - bundle.data.t[0]),
    )
    objective = CUTFGWDDistillation(
        logit_weight=0.0,
        rank_weight=0.0,
        relation_weight=0.0,
        diversity_weight=0.0,
        kernel_weight=1.0,
        fgw_iterations=1,
        sinkhorn_iterations=2,
    )
    teacher.train()
    student.train()
    teacher.reset_state()
    student.reset_state()
    saw_signal = False
    for batch in loader:
        with torch.no_grad():
            teacher_output = teacher.score_candidates(batch, return_relations=True)
            teacher.update_state(batch)
        student_output = student(batch, return_relations=False)
        student.update_state(batch)
        total, parts = objective(student_output, teacher_output, batch["target"])
        if float(parts["kernel"]) > 0.0:
            total.backward()
            saw_signal = True
            break
    assert saw_signal
    assert student.attention is not None
    assert student.attention.score.weight.grad is not None
    assert torch.isfinite(student.attention.score.weight.grad).all()
