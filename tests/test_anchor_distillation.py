from __future__ import annotations

import torch

from dataset import CandidateBatchLoader, build_synthetic_temporal_graph
from loss import (
    CUTFGWDDistillation,
    anchor_relation_loss,
    masked_relation_mean,
    relation_alignment_loss,
)
from model import PyGTGNTeacher
from train import train_teacher_epoch


def make_stream():
    bundle = build_synthetic_temporal_graph(
        num_nodes=12,
        num_events=20,
        feature_dim=4,
        edge_feature_dim=2,
        num_candidates=3,
        history_size=2,
        seed=17,
    )
    loader = CandidateBatchLoader(bundle, "train", 4, 3, 17, 0)
    return bundle, loader


def make_event_batch(
    event_indices: tuple[int, ...] = (0,),
    sources: tuple[int, ...] = (0,),
    positives: tuple[int, ...] = (1,),
    candidates: tuple[tuple[int, ...], ...] = ((1, 2),),
    targets: tuple[int, ...] = (0,),
) -> dict[str, torch.Tensor]:
    event_index = torch.tensor(event_indices, dtype=torch.long)
    all_times = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    return {
        "event_index": event_index,
        "src": torch.tensor(sources, dtype=torch.long),
        "positive_dst": torch.tensor(positives, dtype=torch.long),
        "candidates": torch.tensor(candidates, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.long),
        "timestamp": event_index.float() / 3.0,
        "event_time": all_times[event_index],
        "edge_features": torch.zeros(len(event_indices), 2),
    }


def test_masked_relation_mean_ignores_padding() -> None:
    tokens = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]])
    mask = torch.tensor([[True, False]])

    pooled = masked_relation_mean(tokens, mask)

    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled, torch.tensor([[1.0, 1.0]]))


def test_anchor_relation_loss_is_zero_for_identical_representations() -> None:
    student_tokens = torch.randn(1, 2, 4)
    student_ages = torch.tensor([[0.0, 1.0]])
    teacher_tokens = student_tokens.clone()
    teacher_ages = student_ages.clone()
    teacher_mask = torch.ones(1, 2, dtype=torch.bool)

    loss = anchor_relation_loss(
        student_tokens,
        student_ages,
        teacher_tokens,
        teacher_ages,
        teacher_mask,
    )

    assert loss.item() < 1.0e-5


def test_anchor_relation_loss_backpropagates_through_age_matching() -> None:
    student_tokens = torch.randn(1, 2, 4, requires_grad=True)
    student_ages = torch.tensor([[0.0, 1.0]])
    teacher_tokens = torch.randn(1, 3, 4)
    teacher_ages = torch.tensor([[0.0, 0.5, 1.0]])
    teacher_mask = torch.tensor([[True, False, True]])

    loss = anchor_relation_loss(
        student_tokens,
        student_ages,
        teacher_tokens,
        teacher_ages,
        teacher_mask,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert student_tokens.grad is not None
    assert torch.isfinite(student_tokens.grad).all()


def test_zero_weight_objective_only_needs_student_logits() -> None:
    student_output = {"logits": torch.randn(2, 3, requires_grad=True)}
    objective = CUTFGWDDistillation(
        task_weight=1.0,
        logit_weight=0.0,
        rank_weight=0.0,
        relation_weight=0.0,
        diversity_weight=0.0,
        anchor_weight=0.0,
    )

    total, parts = objective(
        student_output,
        {},
        torch.tensor([0, 1]),
    )
    total.backward()

    assert torch.isfinite(total)
    assert student_output["logits"].grad is not None
    assert float(parts["anchor"]) == 0.0


def test_anchor_objective_backpropagates_through_cutfgw_and_anchor() -> None:
    student_tokens = torch.randn(2, 3, 4, requires_grad=True)
    student_ages = torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
    student_output = {
        "logits": torch.randn(2, 3, requires_grad=True),
        "relation_tokens": student_tokens,
        "relation_ages": student_ages,
        "relation_mass_logits": torch.randn(2, 3, requires_grad=True),
    }
    teacher_output = {
        "logits": torch.randn(2, 3),
        "relation_tokens": torch.randn(2, 4, 4),
        "relation_ages": torch.tensor([[1.0, 0.7, 0.3, 0.0]] * 2),
        "relation_mask": torch.ones(2, 4, dtype=torch.bool),
        "relation_mass": torch.rand(2, 4),
    }
    objective = CUTFGWDDistillation(
        task_weight=1.0,
        logit_weight=0.1,
        rank_weight=0.1,
        relation_weight=0.1,
        diversity_weight=0.0,
        anchor_weight=0.5,
        fgw_iterations=1,
        sinkhorn_iterations=3,
    )

    total, parts = objective(
        student_output,
        teacher_output,
        torch.tensor([0, 1]),
    )
    total.backward()

    assert torch.isfinite(total)
    assert student_tokens.grad is not None
    assert torch.isfinite(student_tokens.grad).all()
    assert float(parts["anchor"]) > 0.0


def test_relation_outputs_are_age_ordered_after_vectorization() -> None:
    teacher = PyGTGNTeacher(
        event_times=torch.tensor([10, 20, 30, 40], dtype=torch.long),
        event_messages=torch.zeros(4, 2),
        num_nodes=5,
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        temporal_neighbors=3,
        dropout=0.0,
    ).eval()

    for step, positive in enumerate((1, 2, 3)):
        teacher.update_state(
            make_event_batch(
                event_indices=(step,),
                sources=(0,),
                positives=(positive,),
                candidates=((1, 2),),
                targets=(0,),
            )
        )
    output = teacher.score_candidates(
        make_event_batch(
            event_indices=(3,),
            sources=(0,),
            positives=(1,),
            candidates=((1, 2),),
            targets=(0,),
        ),
        return_relations=True,
    )

    mask = output["relation_mask"][0]
    ages = output["relation_ages"][0][mask]
    assert int(mask.sum()) == 3
    # 槽位从旧到新：年龄应单调不增。
    assert bool((ages[:-1] >= ages[1:] - 1.0e-6).all())
    assert torch.allclose(output["relation_mass"].sum(dim=1), torch.ones(1))


def test_teacher_relation_auxiliary_loss_trains_align_head() -> None:
    teacher = PyGTGNTeacher(
        event_times=torch.tensor([10, 20, 30, 40], dtype=torch.long),
        event_messages=torch.zeros(4, 2),
        num_nodes=5,
        hidden_dim=8,
        time_dim=4,
        relation_dim=6,
        temporal_neighbors=3,
        dropout=0.0,
        relation_align=True,
    ).eval()

    teacher.update_state(make_event_batch())
    output = teacher.score_candidates(
        make_event_batch(
            event_indices=(1,),
            sources=(0,),
            positives=(2,),
            candidates=((2, 3),),
            targets=(0,),
        ),
        return_relations=True,
    )

    assert output["source_embedding"].shape == (1, 8)
    pooled = masked_relation_mean(
        output["relation_tokens"],
        output["relation_mask"],
    )
    projected = teacher.relation_align_head(pooled)
    auxiliary = relation_alignment_loss(
        projected,
        output["source_embedding"],
        output["relation_mask"].any(dim=-1),
    )
    auxiliary.backward()

    assert teacher.relation_align_head.weight.grad is not None


def test_teacher_epoch_with_relation_auxiliary_runs() -> None:
    bundle, loader = make_stream()
    teacher = PyGTGNTeacher(
        event_times=bundle.data.t,
        event_messages=bundle.data.msg,
        num_nodes=bundle.node_features.size(0),
        hidden_dim=8,
        time_dim=4,
        relation_dim=8,
        temporal_neighbors=3,
        dropout=0.0,
        relation_align=True,
    )
    optimizer = torch.optim.Adam(teacher.parameters(), lr=1.0e-3)

    loss = train_teacher_epoch(
        teacher,
        loader,
        optimizer,
        torch.device("cpu"),
        grad_clip=5.0,
        relation_aux_weight=1.0,
    )

    assert loss > 0.0
    assert not teacher.state_is_empty
