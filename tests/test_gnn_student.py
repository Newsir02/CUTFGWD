from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import CandidateBatchLoader, build_synthetic_temporal_graph
from model import LightGNNStudent
from train import main


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


def make_student(bundle) -> LightGNNStudent:
    return LightGNNStudent(
        bundle.node_features,
        hidden_dim=8,
        time_dim=4,
        relation_dim=8,
        relation_slots=3,
        dropout=0.0,
        num_blocks=2,
        neighbors=4,
    )


def test_gnn_student_forward_backward_and_state_update() -> None:
    bundle, loader = make_stream()
    student = make_student(bundle)
    optimizer = torch.optim.Adam(student.parameters(), lr=1.0e-3)
    before = student.node_embeddings.weight.detach().clone()

    student.reset_state()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        output = student(batch, return_relations=True)
        student.update_state(batch)
        loss = F.cross_entropy(output["logits"], batch["target"])
        loss.backward()
        optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(before, student.node_embeddings.weight.detach())
    assert int(student.neighbor_loader.cur_e_id) > 0


def test_gnn_student_relation_output_shapes() -> None:
    bundle, loader = make_stream()
    student = make_student(bundle).eval()
    batch = next(iter(loader))

    output = student(batch, return_relations=True)

    assert output["logits"].shape == (tuple(batch["src"].shape)[0], 3)
    assert output["relation_tokens"].shape == (tuple(batch["src"].shape)[0], 3, 8)
    assert output["relation_ages"].shape == (tuple(batch["src"].shape)[0], 3)
    assert output["relation_mass_logits"].shape == (tuple(batch["src"].shape)[0], 3)


def test_gnn_student_reset_clears_history() -> None:
    bundle, loader = make_stream()
    student = make_student(bundle)
    for batch in loader:
        student(batch, return_relations=False)
        student.update_state(batch)

    assert int(student.neighbor_loader.cur_e_id) > 0
    student.reset_state()
    assert int(student.neighbor_loader.cur_e_id) == 0


def test_gnn_student_arch_runs_end_to_end(
    tmp_path: Path,
) -> None:
    main(
        [
            "--dataset",
            "synthetic",
            "--quick",
            "--stage",
            "all",
            "--student-arch",
            "gnn",
            "--num-nodes",
            "12",
            "--num-events",
            "20",
            "--num-candidates",
            "3",
            "--feature-dim",
            "4",
            "--edge-feature-dim",
            "2",
            "--history-size",
            "2",
            "--temporal-neighbors",
            "2",
            "--student-neighbors",
            "2",
            "--batch-size",
            "4",
            "--teacher-hidden",
            "8",
            "--student-hidden",
            "8",
            "--time-dim",
            "4",
            "--relation-dim",
            "8",
            "--relation-slots",
            "3",
            "--dropout",
            "0",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert (tmp_path / "teacher.pt").is_file()
    assert (tmp_path / "student.pt").is_file()
