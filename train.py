from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dataset import (
    CandidateBatchLoader,
    TemporalGraphBundle,
    build_synthetic_temporal_graph,
    build_tgb_temporal_graph,
)
from loss import (
    CUTFGWDDistillation,
    masked_relation_mean,
    relation_alignment_loss,
)
from model import LightGNNStudent, LightSTMLPStudent, PyGTGNTeacher
from util import (
    AverageMeter,
    RankingMetricAccumulator,
    benchmark_latency,
    evaluate_tgb,
    load_model_checkpoint,
    model_parameter_summary,
    move_to_device,
    save_model_checkpoint,
    seed_everything,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TGB temporal link distillation with a PyG TGN teacher and "
            "LightST MLP student."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("teacher", "distill", "student", "all"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dataset", type=str, default="tgbl-wiki")
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--max-real-events", type=int, default=0)
    parser.add_argument(
        "--no-tgb-download",
        action="store_false",
        dest="tgb_download",
    )
    parser.set_defaults(tgb_download=True)

    parser.add_argument("--num-nodes", type=int, default=256)
    parser.add_argument("--num-events", type=int, default=3000)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--edge-feature-dim", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--history-size", type=int, default=24)
    parser.add_argument("--temporal-neighbors", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--teacher-hidden", type=int, default=64)
    parser.add_argument("--teacher-layers", type=int, default=1)
    parser.add_argument("--student-hidden", type=int, default=32)
    parser.add_argument(
        "--student-blocks",
        "--student-layers",
        dest="student_blocks",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--student-arch",
        choices=("mlp", "gnn"),
        default="mlp",
    )
    parser.add_argument("--student-neighbors", type=int, default=8)
    parser.add_argument(
        "--structure-features",
        action="store_true",
        default=None,
        help=(
            "把离线统计的结构/时序节点特征拼接到学生输入，"
            "使图无关学生也具备结构/近期活跃度感知。"
        ),
    )
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--relation-dim", type=int, default=64)
    parser.add_argument("--relation-slots", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--teacher-lr", type=float, default=1.0e-3)
    parser.add_argument("--student-lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--logit-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--relation-weight", type=float, default=1.0)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    # 新增方法参数：默认 None，由 apply_method_configuration 按 --method 填充，
    # 从而保证 `--method baseline` 的默认行为与旧版完全一致。
    parser.add_argument("--anchor-weight", type=float, default=None)
    parser.add_argument("--teacher-relation-weight", type=float, default=None)
    parser.add_argument(
        "--method",
        choices=("baseline", "anchor_ot"),
        default="baseline",
    )
    parser.add_argument("--student-seed", type=int, default=None)
    parser.add_argument("--logit-temperature", type=float, default=2.0)
    parser.add_argument("--rank-temperature", type=float, default=1.0)
    parser.add_argument("--fused-alpha", type=float, default=0.20)
    parser.add_argument("--memory-weight", type=float, default=1.0)
    parser.add_argument("--dynamics-weight", type=float, default=0.5)
    parser.add_argument("--ot-entropy", type=float, default=0.08)
    parser.add_argument("--marginal-relaxation", type=float, default=0.8)
    parser.add_argument("--time-prior-scale", type=float, default=0.20)
    parser.add_argument("--fgw-iterations", type=int, default=4)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)

    parser.add_argument("--teacher-checkpoint", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--latency-batches", type=int, default=20)
    args = parser.parse_args(argv)
    args.student_layers = args.student_blocks
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.stage == "distill" and not args.teacher_checkpoint:
        raise ValueError("--stage distill requires --teacher-checkpoint.")
    if args.stage == "distill" and not Path(args.teacher_checkpoint).is_file():
        raise FileNotFoundError(
            f"Teacher checkpoint does not exist: {args.teacher_checkpoint}"
        )
    positive_names = {
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "feature_dim": args.feature_dim,
        "edge_feature_dim": args.edge_feature_dim,
        "num_candidates": args.num_candidates,
        "temporal_neighbors": args.temporal_neighbors,
        "batch_size": args.batch_size,
        "teacher_layers": args.teacher_layers,
        "student_blocks": args.student_blocks,
        "latency_batches": args.latency_batches,
    }
    invalid = [name for name, value in positive_names.items() if value <= 0]
    if invalid:
        raise ValueError("These arguments must be positive: " + ", ".join(invalid))
    if args.num_candidates < 2:
        raise ValueError("num-candidates must be at least 2.")
    if args.patience < 0:
        raise ValueError("patience must be non-negative.")
    if args.student_neighbors <= 0:
        raise ValueError("student-neighbors must be positive.")
    if args.teacher_hidden % 2 != 0:
        raise ValueError("teacher-hidden must be even for TransformerConv heads.")


def apply_quick_configuration(args: argparse.Namespace) -> None:
    """把实验压缩为 CPU 可运行的非官方 smoke 配置。"""

    args.num_nodes = min(args.num_nodes, 64)
    args.num_events = min(args.num_events, 120)
    args.num_candidates = min(args.num_candidates, 8)
    args.history_size = min(args.history_size, 8)
    args.temporal_neighbors = min(args.temporal_neighbors, 8)
    args.batch_size = min(args.batch_size, 24)
    args.teacher_hidden = min(args.teacher_hidden, 32)
    args.student_hidden = min(args.student_hidden, 32)
    args.relation_dim = min(args.relation_dim, 32)
    args.relation_slots = min(args.relation_slots, 4)
    args.student_neighbors = min(args.student_neighbors, 8)
    args.teacher_epochs = 1
    args.student_epochs = 1
    args.fgw_iterations = 1
    args.sinkhorn_iterations = min(args.sinkhorn_iterations, 5)
    args.latency_batches = 1
    args.student_layers = args.student_blocks
    if args.dataset != "synthetic":
        args.max_real_events = min(args.max_real_events or 120, 120)


def apply_method_configuration(args: argparse.Namespace) -> None:
    """把 --method 映射为新增权重；显式传入的参数优先。"""

    if args.method == "anchor_ot":
        if args.anchor_weight is None:
            args.anchor_weight = 0.5
        if args.teacher_relation_weight is None:
            args.teacher_relation_weight = 1.0
    else:
        if args.anchor_weight is None:
            args.anchor_weight = 0.0
        if args.teacher_relation_weight is None:
            args.teacher_relation_weight = 0.0
    if args.structure_features is None:
        args.structure_features = args.method == "anchor_ot"
    args.structure_features = bool(args.structure_features)


def resolve_student_seed(args: argparse.Namespace) -> int:
    """学生初始化/训练随机种子；未指定时回退到 --seed 以保持旧命令兼容。"""

    if args.student_seed is None:
        args.student_seed = args.seed
    return int(args.student_seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_loaders(
    args: argparse.Namespace,
) -> tuple[
    TemporalGraphBundle,
    CandidateBatchLoader,
    CandidateBatchLoader,
    CandidateBatchLoader,
]:
    if args.dataset == "synthetic":
        bundle = build_synthetic_temporal_graph(
            num_nodes=args.num_nodes,
            num_events=args.num_events,
            feature_dim=args.feature_dim,
            edge_feature_dim=args.edge_feature_dim,
            num_candidates=args.num_candidates,
            history_size=args.history_size,
            seed=args.seed,
        )
    else:
        bundle = build_tgb_temporal_graph(
            name=args.dataset,
            root=args.data_root,
            feature_dim=args.feature_dim,
            edge_feature_dim=args.edge_feature_dim,
            num_candidates=args.num_candidates,
            history_size=args.history_size,
            seed=args.seed,
            max_events=args.max_real_events,
            download=args.tgb_download,
        )
    common = {
        "bundle": bundle,
        "batch_size": args.batch_size,
        "num_candidates": args.num_candidates,
        "num_workers": args.num_workers,
    }
    train_loader = CandidateBatchLoader(
        split="train",
        seed=args.seed,
        **common,
    )
    validation_loader = CandidateBatchLoader(
        split="val",
        seed=args.seed + 1,
        **common,
    )
    test_loader = CandidateBatchLoader(
        split="test",
        seed=args.seed + 2,
        **common,
    )
    return bundle, train_loader, validation_loader, test_loader


def _update_signature_with_tensor(hasher: object, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(str(tuple(value.shape)).encode("ascii"))
    hasher.update(value.numpy().tobytes())


def compute_data_signature(
    args: argparse.Namespace,
    bundle: TemporalGraphBundle,
) -> str:
    """对模型实际使用的官方事件流、特征和切分生成稳定指纹。"""

    hasher = hashlib.sha256()
    metadata = {
        "format": "cutfgwd-pyg-temporal-v2",
        "dataset": args.dataset,
        "seed": args.seed,
        "feature_dim": args.feature_dim,
        "num_candidates": args.num_candidates,
        "max_real_events": args.max_real_events,
        "official_evaluation": bundle.official_evaluation,
    }
    hasher.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    _update_signature_with_tensor(hasher, bundle.node_features)
    for name in ("src", "dst", "t", "msg", "event_index", "timestamp"):
        hasher.update(name.encode("ascii"))
        _update_signature_with_tensor(hasher, getattr(bundle.data, name))
    for name, split_data in (
        ("train", bundle.train_data),
        ("val", bundle.val_data),
        ("test", bundle.test_data),
    ):
        hasher.update(name.encode("ascii"))
        _update_signature_with_tensor(hasher, split_data.event_index)
    return hasher.hexdigest()


def build_teacher(
    bundle: TemporalGraphBundle,
    args: argparse.Namespace,
) -> PyGTGNTeacher:
    return PyGTGNTeacher(
        event_times=bundle.data.t,
        event_messages=bundle.data.msg,
        num_nodes=int(bundle.node_features.size(0)),
        hidden_dim=args.teacher_hidden,
        time_dim=args.time_dim,
        relation_dim=args.relation_dim,
        temporal_neighbors=args.temporal_neighbors,
        dropout=args.dropout,
        teacher_layers=args.teacher_layers,
    )


def build_student(
    node_features: Tensor,
    args: argparse.Namespace,
    structure_features: Optional[Tensor] = None,
) -> nn.Module:
    student_features = node_features
    if (
        bool(getattr(args, "structure_features", False))
        and structure_features is not None
    ):
        # 拼接离线结构/时序特征；这些特征只从训练集统计，推理时无需访问图。
        student_features = torch.cat(
            [
                node_features.to(torch.float32),
                structure_features.to(torch.float32),
            ],
            dim=-1,
        )
    if args.student_arch == "gnn":
        return LightGNNStudent(
            node_features=student_features,
            hidden_dim=args.student_hidden,
            time_dim=args.time_dim,
            relation_dim=args.relation_dim,
            relation_slots=args.relation_slots,
            dropout=args.dropout,
            num_blocks=args.student_blocks,
            neighbors=args.student_neighbors,
        )
    return LightSTMLPStudent(
        node_features=student_features,
        hidden_dim=args.student_hidden,
        time_dim=args.time_dim,
        relation_dim=args.relation_dim,
        relation_slots=args.relation_slots,
        dropout=args.dropout,
        num_blocks=args.student_blocks,
    )


def load_teacher_from_checkpoint(
    checkpoint_path: str | Path,
    bundle: TemporalGraphBundle,
    data_signature: str,
    device: torch.device,
) -> PyGTGNTeacher:
    expected_config = {
        "num_nodes": int(bundle.node_features.size(0)),
        "num_events": int(bundle.data.num_events),
        "msg_dim": int(bundle.data.msg.size(1)),
        "data_signature": data_signature,
    }
    payload = load_model_checkpoint(
        checkpoint_path,
        expected_model_name="PyGTGNTeacher",
        expected_model_config=expected_config,
    )
    config = payload["model_config"]
    required = {
        "hidden_dim",
        "time_dim",
        "relation_dim",
        "temporal_neighbors",
        "dropout",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(
            "Teacher checkpoint config is missing: " + ", ".join(missing)
        )
    teacher = PyGTGNTeacher(
        event_times=bundle.data.t,
        event_messages=bundle.data.msg,
        num_nodes=int(config["num_nodes"]),
        hidden_dim=int(config["hidden_dim"]),
        time_dim=int(config["time_dim"]),
        relation_dim=int(config["relation_dim"]),
        temporal_neighbors=int(config["temporal_neighbors"]),
        dropout=float(config["dropout"]),
        teacher_layers=int(config.get("teacher_layers", 1)),
    ).to(device)
    state = payload["model"]
    try:
        teacher.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Teacher checkpoint parameters are incompatible: {exc}"
        ) from exc
    teacher.reset_state()
    return teacher


@torch.no_grad()
def evaluate_smoke(
    model: nn.Module,
    loader: CandidateBatchLoader,
    device: torch.device,
) -> Dict[str, object]:
    """在固定随机候选上运行非官方 smoke 指标。"""

    model.eval()
    loss_meter = AverageMeter()
    metrics = RankingMetricAccumulator()
    score_candidates = getattr(model, "score_candidates", None)
    update_state = getattr(model, "update_state", None)
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        if callable(score_candidates):
            output = score_candidates(batch, return_relations=False)
        else:
            output = model(batch, return_relations=False)
        loss = F.cross_entropy(output["logits"], batch["target"])
        count = int(batch["target"].numel())
        loss_meter.update(float(loss), count)
        metrics.update(output["logits"], batch["target"])
        if callable(update_state):
            update_state(batch)
    result: Dict[str, object] = metrics.compute()
    result.update(
        {
            "metric": "mrr",
            "loss": loss_meter.average,
            "official": False,
        }
    )
    return result


def evaluate_split(
    model: nn.Module,
    bundle: TemporalGraphBundle,
    loader: CandidateBatchLoader,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    if bundle.official_evaluation:
        if split not in {"val", "test"}:
            raise ValueError("Official evaluation supports val or test only.")
        return evaluate_tgb(
            model,
            bundle,
            split,
            args.batch_size,
            device,
        )
    return evaluate_smoke(model, loader, device)


@torch.no_grad()
def replay_teacher(
    teacher: PyGTGNTeacher,
    loaders: Iterable[CandidateBatchLoader],
    device: torch.device,
) -> None:
    """只回放真实正边，重建 PyG memory 与最近邻状态。"""

    teacher.eval()
    for loader in loaders:
        for raw_batch in loader:
            teacher.update_state(move_to_device(raw_batch, device))


def evaluate_teacher_from_history(
    teacher: PyGTGNTeacher,
    bundle: TemporalGraphBundle,
    history_loaders: Iterable[CandidateBatchLoader],
    target_loader: CandidateBatchLoader,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    teacher.eval()
    teacher.reset_state()
    replay_teacher(teacher, history_loaders, device)
    return evaluate_split(
        teacher,
        bundle,
        target_loader,
        split,
        args,
        device,
    )


def train_teacher_epoch(
    teacher: PyGTGNTeacher,
    loader: CandidateBatchLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    relation_aux_weight: float = 0.0,
) -> float:
    teacher.train()
    teacher.reset_state()
    # 关系对齐头始终存在；仅当辅助权重非零时才请求关系输出并计算辅助损失。
    use_relations = relation_aux_weight > 0.0
    meter = AverageMeter()
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = teacher.score_candidates(batch, return_relations=use_relations)
        loss = F.cross_entropy(output["logits"], batch["target"])
        if use_relations:
            # 自监督：池化关系表示应能重建教师打分所用的 source embedding。
            pooled = masked_relation_mean(
                output["relation_tokens"],
                output["relation_mask"],
            )
            projected = teacher.relation_align_head(pooled)
            valid_rows = output["relation_mask"].any(dim=-1)
            aux = relation_alignment_loss(
                projected,
                output["source_embedding"],
                valid_rows,
            )
            loss = loss + relation_aux_weight * aux
        # 官方 TGN 顺序：本批预测完成后才写入真实正边。
        teacher.update_state(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), grad_clip)
        optimizer.step()
        teacher.detach_state()
        meter.update(float(loss.detach()), int(batch["target"].numel()))
    return meter.average


def _copy_state_dict(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _metric_value(metrics: Dict[str, object]) -> float:
    metric_name = str(metrics["metric"])
    return float(metrics[metric_name])


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        if patience < 0:
            raise ValueError("patience must be non-negative.")
        self.patience = patience
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, metric: float, epoch: int) -> tuple[bool, bool]:
        improved = metric > self.best_metric
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return improved, should_stop


def print_early_stopping(
    name: str,
    epoch: int,
    stopper: EarlyStopping,
    metric_prefix: str,
    metric_name: str,
) -> None:
    print(
        f"Early stopping {name} at epoch {epoch:02d}: "
        f"no validation improvement for {stopper.bad_epochs} epoch(s), "
        f"best_epoch={stopper.best_epoch:02d}, "
        f"best_val_{metric_prefix}_{metric_name}={stopper.best_metric:.4f}, "
        f"patience={stopper.patience}"
    )


def pretrain_teacher(
    teacher: PyGTGNTeacher,
    bundle: TemporalGraphBundle,
    train_loader: CandidateBatchLoader,
    validation_loader: CandidateBatchLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Tensor]:
    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=args.teacher_lr,
        weight_decay=args.weight_decay,
    )
    best_metric = -1.0
    best_state: Optional[Dict[str, Tensor]] = None
    stopper = EarlyStopping(args.patience)
    for epoch in range(1, args.teacher_epochs + 1):
        train_loss = train_teacher_epoch(
            teacher,
            train_loader,
            optimizer,
            device,
            args.grad_clip,
            relation_aux_weight=float(args.teacher_relation_weight or 0.0),
        )
        validation = evaluate_split(
            teacher,
            bundle,
            validation_loader,
            "val",
            args,
            device,
        )
        metric_name = str(validation["metric"])
        prefix = "official" if validation["official"] else "smoke"
        metric_value = _metric_value(validation)
        print(
            f"Teacher {epoch:02d}/{args.teacher_epochs:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_{prefix}_{metric_name}={metric_value:.4f}"
        )
        improved, should_stop = stopper.update(metric_value, epoch)
        if improved:
            best_metric = metric_value
            best_state = _copy_state_dict(teacher)
        if should_stop:
            print_early_stopping("Teacher", epoch, stopper, prefix, metric_name)
            break
    if best_state is None:
        raise RuntimeError("Teacher training did not produce a checkpoint.")
    teacher.load_state_dict(best_state, strict=True)
    teacher.reset_state()
    return best_state


def train_student_epoch(
    student: LightSTMLPStudent,
    teacher: PyGTGNTeacher,
    objective: CUTFGWDDistillation,
    loader: CandidateBatchLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    student.train()
    teacher.eval()
    # 仅在确实需要关系张量时才请求，避免零权重分支的额外开销。
    need_teacher_relation = (
        objective.relation_weight != 0.0 or objective.anchor_weight != 0.0
    )
    need_student_relation = (
        objective.relation_weight != 0.0
        or objective.diversity_weight != 0.0
        or objective.anchor_weight != 0.0
    )
    # 有状态学生（轻量 GNN）需要按 TGN 顺序先打分后写入正边。
    update_student_state = getattr(student, "update_state", None)
    meters: Dict[str, AverageMeter] = {}
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        with torch.no_grad():
            teacher_output = teacher.score_candidates(
                batch,
                return_relations=need_teacher_relation,
            )
            teacher.update_state(batch)
        student_output = student(
            batch,
            return_relations=need_student_relation,
        )
        if callable(update_student_state):
            update_student_state(batch)
        optimizer.zero_grad(set_to_none=True)
        total, components = objective(
            student_output,
            teacher_output,
            batch["target"],
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        optimizer.step()
        count = int(batch["target"].numel())
        for name, value in components.items():
            meters.setdefault(name, AverageMeter()).update(float(value), count)
    return {name: meter.average for name, meter in meters.items()}


def distill_student(
    student: LightSTMLPStudent,
    teacher: PyGTGNTeacher,
    bundle: TemporalGraphBundle,
    train_loader: CandidateBatchLoader,
    validation_loader: CandidateBatchLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Tensor]:
    objective = CUTFGWDDistillation(
        task_weight=args.task_weight,
        logit_weight=args.logit_weight,
        rank_weight=args.rank_weight,
        relation_weight=args.relation_weight,
        diversity_weight=args.diversity_weight,
        anchor_weight=float(args.anchor_weight or 0.0),
        logit_temperature=args.logit_temperature,
        rank_temperature=args.rank_temperature,
        fused_alpha=args.fused_alpha,
        memory_weight=args.memory_weight,
        dynamics_weight=args.dynamics_weight,
        entropy=args.ot_entropy,
        marginal_relaxation=args.marginal_relaxation,
        time_prior_scale=args.time_prior_scale,
        fgw_iterations=args.fgw_iterations,
        sinkhorn_iterations=args.sinkhorn_iterations,
    ).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.student_lr,
        weight_decay=args.weight_decay,
    )
    best_metric = -1.0
    best_state: Optional[Dict[str, Tensor]] = None
    stopper = EarlyStopping(args.patience)
    reset_student_state = getattr(student, "reset_state", None)
    for epoch in range(1, args.student_epochs + 1):
        teacher.reset_state()
        if callable(reset_student_state):
            reset_student_state()
        losses = train_student_epoch(
            student,
            teacher,
            objective,
            train_loader,
            optimizer,
            device,
            args.grad_clip,
        )
        validation = evaluate_split(
            student,
            bundle,
            validation_loader,
            "val",
            args,
            device,
        )
        metric_name = str(validation["metric"])
        prefix = "official" if validation["official"] else "smoke"
        metric_value = _metric_value(validation)
        print(
            f"Student {epoch:02d}/{args.student_epochs:02d} "
            f"total={losses['total']:.4f} task={losses['task']:.4f} "
            f"logit={losses['logit']:.4f} rank={losses['rank']:.4f} "
            f"anchor={losses['anchor']:.4f} "
            f"relation={losses['relation']:.4f} "
            f"diversity={losses['diversity']:.4f} "
            f"val_{prefix}_{metric_name}={metric_value:.4f}"
        )
        improved, should_stop = stopper.update(metric_value, epoch)
        if improved:
            best_metric = metric_value
            best_state = _copy_state_dict(student)
        if should_stop:
            print_early_stopping("Student", epoch, stopper, prefix, metric_name)
            break
    if best_state is None:
        raise RuntimeError("Student distillation did not produce a checkpoint.")
    student.load_state_dict(best_state, strict=True)
    return best_state


def train_supervised_student_epoch(
    student: LightSTMLPStudent,
    loader: CandidateBatchLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    student.train()
    update_student_state = getattr(student, "update_state", None)
    meter = AverageMeter()
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = student(batch, return_relations=False)
        if callable(update_student_state):
            update_student_state(batch)
        loss = F.cross_entropy(output["logits"], batch["target"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        optimizer.step()
        meter.update(float(loss.detach()), int(batch["target"].numel()))
    return meter.average


def train_supervised_student(
    student: LightSTMLPStudent,
    bundle: TemporalGraphBundle,
    train_loader: CandidateBatchLoader,
    validation_loader: CandidateBatchLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Tensor]:
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.student_lr,
        weight_decay=args.weight_decay,
    )
    best_metric = -1.0
    best_state: Optional[Dict[str, Tensor]] = None
    stopper = EarlyStopping(args.patience)
    reset_student_state = getattr(student, "reset_state", None)
    for epoch in range(1, args.student_epochs + 1):
        if callable(reset_student_state):
            reset_student_state()
        train_loss = train_supervised_student_epoch(
            student,
            train_loader,
            optimizer,
            device,
            args.grad_clip,
        )
        validation = evaluate_split(
            student,
            bundle,
            validation_loader,
            "val",
            args,
            device,
        )
        metric_name = str(validation["metric"])
        prefix = "official" if validation["official"] else "smoke"
        metric_value = _metric_value(validation)
        print(
            f"Supervised student {epoch:02d}/{args.student_epochs:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_{prefix}_{metric_name}={metric_value:.4f}"
        )
        improved, should_stop = stopper.update(metric_value, epoch)
        if improved:
            best_metric = metric_value
            best_state = _copy_state_dict(student)
        if should_stop:
            print_early_stopping(
                "Supervised student",
                epoch,
                stopper,
                prefix,
                metric_name,
            )
            break
    if best_state is None:
        raise RuntimeError("Supervised student training did not produce a checkpoint.")
    student.load_state_dict(best_state, strict=True)
    return best_state


def freeze_teacher(teacher: PyGTGNTeacher) -> None:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


def print_metrics(name: str, metrics: Dict[str, object]) -> None:
    metric_name = str(metrics["metric"])
    prefix = "official" if metrics["official"] else "smoke"
    details = f"{prefix}_{metric_name}={_metric_value(metrics):.4f}"
    if "loss" in metrics:
        details += f", loss={float(metrics['loss']):.4f}"
    if "hits@1" in metrics:
        details += f", hits@1={float(metrics['hits@1']):.4f}"
    print(f"{name}: {details}")


def print_latency(name: str, latency: Dict[str, float]) -> None:
    print(
        f"{name} latency: mean={latency['batch_ms_mean']:.3f} ms/batch, "
        f"p95={latency['batch_ms_p95']:.3f} ms/batch, "
        f"throughput={latency['queries_per_second']:.1f} query/s"
    )


def print_model_parameters(name: str, model: nn.Module) -> None:
    summary = model_parameter_summary(model)
    print(
        f"{name} parameters: "
        f"trainable={summary['trainable']:,}, "
        f"all={summary['all']:,}, "
        f"runtime_state_elements={summary['runtime_state_elements']:,}, "
        f"total_with_runtime_state={summary['total_with_runtime_state']:,}"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.quick:
        apply_quick_configuration(args)
    apply_method_configuration(args)
    resolve_student_seed(args)
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Arguments: {json.dumps(vars(args), ensure_ascii=True, sort_keys=True)}")

    bundle, train_loader, validation_loader, test_loader = make_loaders(args)
    data_signature = compute_data_signature(args, bundle)
    print(f"Data signature: {data_signature}")
    print(
        "Evaluation protocol: "
        + ("official TGB negatives" if bundle.official_evaluation else "smoke candidates")
    )
    teacher: Optional[PyGTGNTeacher] = None
    teacher_metrics: Optional[Dict[str, object]] = None

    if args.stage in {"teacher", "all"}:
        teacher = build_teacher(bundle, args).to(device)
        print_model_parameters("Teacher", teacher)
        pretrain_teacher(
            teacher,
            bundle,
            train_loader,
            validation_loader,
            args,
            device,
        )
        teacher_path = output_dir / "teacher.pt"
        save_model_checkpoint(
            teacher_path,
            teacher,
            model_name="PyGTGNTeacher",
            model_config={**teacher.model_config, "data_signature": data_signature},
            args=vars(args),
        )
        print(f"Saved teacher checkpoint: {teacher_path}")
        teacher_metrics = evaluate_teacher_from_history(
            teacher,
            bundle,
            [train_loader, validation_loader],
            test_loader,
            "test",
            args,
            device,
        )
        print_metrics("Teacher test", teacher_metrics)

    if args.stage == "distill":
        teacher = load_teacher_from_checkpoint(
            args.teacher_checkpoint,
            bundle,
            data_signature,
            device,
        )
        print(f"Loaded teacher checkpoint: {args.teacher_checkpoint}")
        if (
            args.teacher_hidden != teacher.hidden_dim
            or args.teacher_layers != teacher.teacher_layers
        ):
            print(
                "Warning: loaded teacher config overrides CLI args: "
                f"hidden_dim={teacher.hidden_dim} (arg {args.teacher_hidden}), "
                f"teacher_layers={teacher.teacher_layers} "
                f"(arg {args.teacher_layers})."
            )
        print_model_parameters("Teacher", teacher)

    if args.stage in {"distill", "all"}:
        if teacher is None:
            raise RuntimeError("Distillation requires an initialized teacher.")
        if int(args.relation_dim) != int(teacher.relation_dim):
            raise ValueError(
                "Student relation_dim must match the teacher relation_dim: "
                f"got {args.relation_dim}, expected {teacher.relation_dim}."
            )
        # 学生独立种子：不影响已构造的数据/教师，仅控制学生初始化与 dropout。
        seed_everything(int(args.student_seed))
        freeze_teacher(teacher)
        student = build_student(bundle.node_features, args, bundle.structure_features).to(device)
        print_model_parameters("Student", student)
        distill_student(
            student,
            teacher,
            bundle,
            train_loader,
            validation_loader,
            args,
            device,
        )
        student_path = output_dir / "student.pt"
        save_model_checkpoint(
            student_path,
            student,
            model_name="LightSTMLPStudent",
            model_config={**student.model_config, "data_signature": data_signature},
            args=vars(args),
        )
        print(f"Saved student checkpoint: {student_path}")

        if teacher_metrics is None:
            teacher_metrics = evaluate_teacher_from_history(
                teacher,
                bundle,
                [train_loader, validation_loader],
                test_loader,
                "test",
                args,
                device,
            )
            print_metrics("Teacher test", teacher_metrics)
        student_metrics = evaluate_split(
            student,
            bundle,
            test_loader,
            "test",
            args,
            device,
        )
        print_metrics("Student test", student_metrics)

        teacher.eval()
        teacher.reset_state()
        replay_teacher(teacher, [train_loader, validation_loader], device)
        warmup_batches = 1 if args.quick else 5
        teacher_latency = benchmark_latency(
            teacher,
            test_loader,
            device,
            warmup_batches=warmup_batches,
            measured_batches=args.latency_batches,
            forward_kwargs={"update_memory": False},
        )
        student_latency = benchmark_latency(
            student,
            test_loader,
            device,
            warmup_batches=warmup_batches,
            measured_batches=args.latency_batches,
        )
        print_latency("Teacher", teacher_latency)
        print_latency("Student", student_latency)
        speedup = teacher_latency["query_ms"] / max(
            student_latency["query_ms"],
            1.0e-12,
        )
        print(f"Model-only query latency speedup: {speedup:.2f}x")

    if args.stage == "student":
        seed_everything(int(args.student_seed))
        student = build_student(bundle.node_features, args, bundle.structure_features).to(device)
        print_model_parameters("Student", student)
        train_supervised_student(
            student,
            bundle,
            train_loader,
            validation_loader,
            args,
            device,
        )
        student_path = output_dir / "student_supervised.pt"
        save_model_checkpoint(
            student_path,
            student,
            model_name="LightSTMLPStudent",
            model_config={**student.model_config, "data_signature": data_signature},
            args=vars(args),
        )
        print(f"Saved supervised student checkpoint: {student_path}")
        print_metrics(
            "Supervised student test",
            evaluate_split(
                student,
                bundle,
                test_loader,
                "test",
                args,
                device,
            ),
        )
        student_latency = benchmark_latency(
            student,
            test_loader,
            device,
            warmup_batches=1 if args.quick else 5,
            measured_batches=args.latency_batches,
        )
        print_latency("Student", student_latency)


if __name__ == "__main__":
    main()
