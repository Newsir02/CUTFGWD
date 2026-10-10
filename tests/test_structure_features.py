from __future__ import annotations

from pathlib import Path

import torch

from dataset import build_synthetic_temporal_graph, compute_structure_features
from train import build_student, main, parse_args


def test_compute_structure_features_shape_and_determinism() -> None:
    src = torch.tensor([0, 1, 0, 2, 1, 0])
    dst = torch.tensor([1, 2, 2, 3, 3, 3])
    train_mask = torch.tensor([True, True, True, True, False, False])

    features = compute_structure_features(src, dst, train_mask, num_rows=5)

    assert features.shape == (5, 5)
    assert torch.isfinite(features).all()
    # 从未在训练集出现过的节点特征应为标准化后的 0。
    assert torch.allclose(features[4], torch.zeros(5))


def test_structure_features_do_not_change_node_features() -> None:
    bundle = build_synthetic_temporal_graph(
        num_nodes=12,
        num_events=20,
        feature_dim=4,
        edge_feature_dim=2,
        seed=11,
    )

    assert bundle.structure_features is not None
    assert bundle.structure_features.shape == (bundle.node_features.size(0), 5)


def test_build_student_concatenates_structure_features() -> None:
    bundle = build_synthetic_temporal_graph(
        num_nodes=12,
        num_events=20,
        feature_dim=4,
        edge_feature_dim=2,
        num_candidates=3,
        seed=11,
    )
    args = parse_args(["--dataset", "synthetic", "--structure-features"])

    student = build_student(
        bundle.node_features,
        args,
        bundle.structure_features,
    )

    assert student.model_config["node_feature_dim"] == 4 + 5


def test_structure_features_end_to_end(
    tmp_path: Path,
) -> None:
    main(
        [
            "--dataset",
            "synthetic",
            "--quick",
            "--stage",
            "all",
            "--structure-features",
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
