from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dataset import CandidateBatchLoader, build_synthetic_temporal_graph
from model import LightSTMLPStudent, PyGTGNTeacher
from train import (
    build_student,
    build_teacher,
    compute_data_signature,
    main,
    parse_args,
    replay_teacher,
    train_supervised_student_epoch,
    train_teacher_epoch,
    validate_args,
)
from util import benchmark_latency, model_parameter_summary


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


def make_teacher(bundle) -> PyGTGNTeacher:
    return PyGTGNTeacher(
        event_times=bundle.data.t,
        event_messages=bundle.data.msg,
        num_nodes=bundle.node_features.size(0),
        hidden_dim=8,
        time_dim=4,
        relation_dim=8,
        temporal_neighbors=3,
        dropout=0.0,
    )


def test_stage_defaults_to_all() -> None:
    assert parse_args([]).stage == "all"


def test_layer_arguments_are_written_to_model_configs() -> None:
    bundle, _ = make_stream()
    args = parse_args(
        [
            "--dataset",
            "synthetic",
            "--teacher-layers",
            "2",
            "--student-layers",
            "4",
        ]
    )

    validate_args(args)
    teacher = build_teacher(bundle, args)
    student = build_student(bundle.node_features, args)

    assert args.teacher_layers == 2
    assert args.student_blocks == 4
    assert args.student_layers == 4
    assert teacher.model_config["teacher_layers"] == 2
    assert student.model_config["student_blocks"] == 4
    assert student.model_config["student_layers"] == 4


def test_teacher_parameter_summary_separates_runtime_state() -> None:
    bundle, _ = make_stream()
    teacher = make_teacher(bundle)

    summary = model_parameter_summary(teacher)

    assert summary["trainable"] == summary["all"]
    assert summary["runtime_state_elements"] > 0
    assert summary["total_with_runtime_state"] > summary["all"]


def test_data_signature_is_deterministic_and_event_sensitive() -> None:
    bundle, _ = make_stream()
    args = parse_args(["--dataset", "synthetic", "--seed", "17"])

    first = compute_data_signature(args, bundle)
    second = compute_data_signature(args, bundle)
    bundle.data.msg[0, 0] += 1.0
    changed = compute_data_signature(args, bundle)

    assert first == second
    assert first != changed


def test_distill_stage_requires_teacher_checkpoint() -> None:
    args = parse_args(["--stage", "distill", "--dataset", "synthetic"])

    with pytest.raises(ValueError, match="teacher-checkpoint"):
        validate_args(args)


def test_distill_stage_accepts_existing_teacher_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.touch()
    args = parse_args(
        [
            "--stage",
            "distill",
            "--dataset",
            "synthetic",
            "--teacher-checkpoint",
            str(checkpoint),
        ]
    )

    validate_args(args)


def test_teacher_replay_reconstructs_deterministic_state() -> None:
    bundle, loader = make_stream()
    teacher = make_teacher(bundle).eval()

    replay_teacher(teacher, [loader], torch.device("cpu"))
    first_memory = teacher.memory.memory.clone()
    first_index = int(teacher.stream_event_index)
    teacher.reset_state()
    replay_teacher(teacher, [loader], torch.device("cpu"))

    assert first_index == int(bundle.train_data.event_index[-1])
    assert torch.allclose(first_memory, teacher.memory.memory)


def test_teacher_epoch_backpropagates_with_detached_cross_batch_state() -> None:
    bundle, loader = make_stream()
    teacher = make_teacher(bundle)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=1.0e-3)

    loss = train_teacher_epoch(
        teacher,
        loader,
        optimizer,
        torch.device("cpu"),
        grad_clip=5.0,
    )

    assert loss > 0.0
    assert not teacher.state_is_empty


def test_supervised_student_epoch_does_not_need_a_teacher() -> None:
    bundle, loader = make_stream()
    student = LightSTMLPStudent(
        bundle.node_features,
        hidden_dim=8,
        time_dim=4,
        relation_dim=8,
        relation_slots=3,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=1.0e-3)
    before = student.node_embeddings.weight.detach().clone()

    loss = train_supervised_student_epoch(
        student,
        loader,
        optimizer,
        torch.device("cpu"),
        grad_clip=5.0,
    )

    assert loss > 0.0
    assert not torch.equal(before, student.node_embeddings.weight.detach())


def test_latency_benchmark_can_score_tgn_without_updating_state() -> None:
    bundle, loader = make_stream()
    teacher = make_teacher(bundle).eval()

    result = benchmark_latency(
        teacher,
        loader,
        torch.device("cpu"),
        warmup_batches=1,
        measured_batches=1,
        forward_kwargs={"update_memory": False},
    )

    assert result["query_ms"] > 0.0
    assert teacher.state_is_empty


def test_all_four_cli_stages_create_the_expected_checkpoints(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    common = [
        "--dataset",
        "synthetic",
        "--quick",
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
    ]
    separate_dir = tmp_path / "separate"
    main([*common, "--stage", "teacher", "--output-dir", str(separate_dir)])
    teacher_path = separate_dir / "teacher.pt"
    assert teacher_path.is_file()
    teacher_payload = torch.load(teacher_path, weights_only=False)
    assert teacher_payload["model_name"] == "PyGTGNTeacher"

    main(
        [
            *common,
            "--stage",
            "distill",
            "--teacher-checkpoint",
            str(teacher_path),
            "--output-dir",
            str(separate_dir),
        ]
    )
    assert (separate_dir / "student.pt").is_file()

    supervised_dir = tmp_path / "supervised"
    main([*common, "--stage", "student", "--output-dir", str(supervised_dir)])
    assert (supervised_dir / "student_supervised.pt").is_file()

    all_dir = tmp_path / "all"
    main([*common, "--stage", "all", "--output-dir", str(all_dir)])
    assert (all_dir / "teacher.pt").is_file()
    assert (all_dir / "student.pt").is_file()
    capsys.readouterr()
