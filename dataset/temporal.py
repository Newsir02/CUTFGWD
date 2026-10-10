from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from torch import Tensor
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader


SplitName = Literal["train", "val", "test"]


@dataclass
class TemporalGraphBundle:
    """统一保存 PyG 事件流、时间切分和 TGB 官方评测句柄。"""

    dataset_name: str
    node_features: Tensor
    data: TemporalData
    train_data: TemporalData
    val_data: TemporalData
    test_data: TemporalData
    destination_nodes: Tensor
    official_evaluation: bool
    tgb_dataset: object | None = None
    structure_features: Tensor | None = None

    def split(self, name: SplitName) -> TemporalData:
        if name == "train":
            return self.train_data
        if name == "val":
            return self.val_data
        if name == "test":
            return self.test_data
        raise ValueError(f"Unknown temporal split: {name}")


def _validate_mask(name: str, mask: Tensor, event_count: int) -> Tensor:
    value = torch.as_tensor(mask, dtype=torch.bool).flatten()
    if value.shape != (event_count,):
        raise ValueError(f"{name} mask must have shape [{event_count}].")
    return value


def _quick_split_masks(event_count: int) -> tuple[Tensor, Tensor, Tensor]:
    if event_count < 3:
        raise ValueError("A temporal stream must retain at least three events.")
    train_count = max(1, int(event_count * 0.70))
    val_count = max(1, int(event_count * 0.15))
    if train_count + val_count >= event_count:
        train_count = event_count - 2
        val_count = 1
    event_ids = torch.arange(event_count)
    train_mask = event_ids < train_count
    val_mask = (event_ids >= train_count) & (event_ids < train_count + val_count)
    test_mask = event_ids >= train_count + val_count
    return train_mask, val_mask, test_mask


def _prepare_messages(data: TemporalData, event_count: int) -> Tensor:
    raw_messages = getattr(data, "msg", None)
    if raw_messages is None:
        return torch.zeros(event_count, 1)
    messages = torch.as_tensor(raw_messages, dtype=torch.float)
    if messages.dim() == 1:
        messages = messages.unsqueeze(-1)
    if messages.dim() != 2 or messages.size(0) != event_count:
        raise ValueError("Edge messages must have shape [events, features].")
    if messages.size(1) == 0:
        messages = torch.zeros(event_count, 1)
    if not bool(torch.isfinite(messages).all()):
        raise ValueError("Edge messages must contain only finite values.")
    return messages.contiguous()


def _prepare_node_features(
    node_features: object | None,
    num_nodes: int,
    feature_dim: int,
) -> Tensor:
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")
    if node_features is None:
        return torch.zeros(num_nodes, feature_dim)
    features = torch.as_tensor(node_features, dtype=torch.float)
    if features.dim() != 2 or features.size(0) < num_nodes:
        raise ValueError(
            f"Node features must contain at least {num_nodes} node rows."
        )
    if features.size(1) == 0 or not bool(torch.isfinite(features).all()):
        raise ValueError("Node features must be finite and non-empty.")
    return features.contiguous()


def compute_structure_features(
    src: Tensor,
    dst: Tensor,
    train_mask: Tensor,
    num_rows: int,
) -> Tensor:
    """从训练集事件流统计每个节点的结构/时序特征。

    这些特征只依赖训练集，推理时作为静态输入喂给图无关学生，从而在不访问图的
    前提下提供"结构感知"和"近期活跃度"信息（参考 GLNN / InfGraND / L-STEP）。
    输出已按维标准化。
    """

    train_src = src[train_mask].long()
    train_dst = dst[train_mask].long()
    out_degree = torch.zeros(num_rows, dtype=torch.float)
    in_degree = torch.zeros(num_rows, dtype=torch.float)
    out_degree.index_add_(0, train_src, torch.ones(train_src.numel()))
    in_degree.index_add_(0, train_dst, torch.ones(train_dst.numel()))
    position_count = int(train_src.numel())
    positions = torch.arange(position_count, dtype=torch.float)
    appear_node = torch.cat([train_src, train_dst])
    appear_position = torch.cat([positions, positions])
    span = max(position_count - 1, 1)
    first_seen = torch.full((num_rows,), float("inf")).scatter_reduce_(
        0,
        appear_node,
        appear_position,
        reduce="amin",
        include_self=True,
    )
    last_seen = torch.full((num_rows,), -1.0).scatter_reduce_(
        0,
        appear_node,
        appear_position,
        reduce="amax",
        include_self=True,
    )
    seen = torch.isfinite(first_seen)
    first_norm = torch.where(
        seen,
        first_seen / float(span),
        torch.zeros(()),
    )
    last_norm = torch.where(
        seen,
        last_seen.clamp_min(0.0) / float(span),
        torch.zeros(()),
    )
    activity = torch.log1p(out_degree + in_degree)
    features = torch.stack(
        [out_degree, in_degree, activity, first_norm, last_norm],
        dim=-1,
    )
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    standardized = (features - mean) / std
    # 训练集中从未出现的节点映射为中性 0，而不是 -mean/std。
    seen_any = (out_degree + in_degree) > 0.0
    standardized = torch.where(
        seen_any.unsqueeze(-1),
        standardized,
        torch.zeros(()),
    )
    return standardized.contiguous()


def build_temporal_graph_bundle(
    *,
    dataset_name: str,
    data: TemporalData,
    node_features: object | None,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    feature_dim: int,
    official_evaluation: bool,
    tgb_dataset: object | None = None,
    max_events: int = 0,
) -> TemporalGraphBundle:
    """校验官方事件流，并为教师和学生补充稳定的辅助字段。"""

    if not isinstance(data, TemporalData):
        raise TypeError("data must be a torch_geometric.data.TemporalData object.")
    data = data.clone()
    event_count = int(data.num_events)
    if event_count == 0:
        raise ValueError("The temporal event stream is empty.")

    src = torch.as_tensor(data.src, dtype=torch.long).flatten()
    dst = torch.as_tensor(data.dst, dtype=torch.long).flatten()
    event_time = torch.as_tensor(data.t, dtype=torch.long).flatten()
    if src.shape != (event_count,) or dst.shape != (event_count,):
        raise ValueError("Source and destination tensors must have shape [events].")
    if event_time.shape != (event_count,):
        raise ValueError("Event timestamps must have shape [events].")
    if bool((src < 0).any()) or bool((dst < 0).any()):
        raise ValueError("Node ids must be non-negative.")
    if event_count > 1 and bool((event_time[1:] < event_time[:-1]).any()):
        raise ValueError("TGB events must be chronological.")

    train_mask = _validate_mask("train", train_mask, event_count)
    val_mask = _validate_mask("validation", val_mask, event_count)
    test_mask = _validate_mask("test", test_mask, event_count)
    messages = _prepare_messages(data, event_count)

    if max_events > 0 and max_events < event_count:
        retained = max_events
        if retained < 3:
            raise ValueError("max_events must retain at least three events.")
        data = data[:retained]
        src = src[:retained]
        dst = dst[:retained]
        event_time = event_time[:retained]
        messages = messages[:retained]
        event_count = retained
        train_mask, val_mask, test_mask = _quick_split_masks(event_count)
        official_evaluation = False

    data.src = src
    data.dst = dst
    data.t = event_time
    data.msg = messages
    data.event_index = torch.arange(event_count, dtype=torch.long)
    time_min = event_time[0].double()
    time_span = (event_time[-1].double() - time_min).clamp_min(1.0)
    data.timestamp = ((event_time.double() - time_min) / time_span).float()

    num_nodes = int(torch.cat([src, dst]).max()) + 1
    features = _prepare_node_features(node_features, num_nodes, feature_dim)
    destination_nodes = torch.unique(dst, sorted=True)
    if destination_nodes.numel() < 2:
        raise ValueError("The destination node domain must contain at least two nodes.")
    structure_features = compute_structure_features(
        src,
        dst,
        train_mask,
        int(features.size(0)),
    )

    return TemporalGraphBundle(
        dataset_name=dataset_name,
        node_features=features,
        data=data,
        train_data=data[train_mask],
        val_data=data[val_mask],
        test_data=data[test_mask],
        destination_nodes=destination_nodes,
        official_evaluation=official_evaluation,
        tgb_dataset=tgb_dataset,
        structure_features=structure_features,
    )


def build_candidate_batch(
    batch: TemporalData,
    destination_nodes: Tensor,
    num_candidates: int,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """把连续正事件适配为固定宽度的候选排名批次。"""

    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2.")
    src = batch.src.long().cpu()
    positive_dst = batch.dst.long().cpu()
    destination_nodes = destination_nodes.long().cpu()
    candidates = torch.empty(src.numel(), num_candidates, dtype=torch.long)
    target = torch.empty(src.numel(), dtype=torch.long)

    for row, (source, positive) in enumerate(
        zip(src.tolist(), positive_dst.tolist())
    ):
        available = destination_nodes[
            (destination_nodes != source) & (destination_nodes != positive)
        ]
        if available.numel() < num_candidates - 1:
            raise ValueError("Not enough destination nodes for negative sampling.")
        order = torch.randperm(available.numel(), generator=generator)
        negatives = available[order[: num_candidates - 1]]
        positive_index = int(
            torch.randint(num_candidates, (1,), generator=generator)
        )
        candidates[row, :positive_index] = negatives[:positive_index]
        candidates[row, positive_index] = positive
        candidates[row, positive_index + 1 :] = negatives[positive_index:]
        target[row] = positive_index

    return {
        "event_index": batch.event_index.long().cpu(),
        "src": src,
        "positive_dst": positive_dst,
        "candidates": candidates,
        "target": target,
        "timestamp": batch.timestamp.float().cpu(),
        "event_time": batch.t.long().cpu(),
        "edge_features": batch.msg.float().cpu(),
    }


class CandidateBatchLoader:
    """在 PyG 连续事件加载器之上按需生成训练候选。"""

    def __init__(
        self,
        bundle: TemporalGraphBundle,
        split: SplitName,
        batch_size: int,
        num_candidates: int,
        seed: int,
        num_workers: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.bundle = bundle
        self.split_name = split
        self.num_candidates = num_candidates
        self.seed = seed
        self.dataset = bundle.split(split)
        self.temporal_loader = TemporalDataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        generator = torch.Generator().manual_seed(self.seed)
        for batch in self.temporal_loader:
            yield build_candidate_batch(
                batch,
                self.bundle.destination_nodes,
                self.num_candidates,
                generator,
            )

    def __len__(self) -> int:
        return len(self.temporal_loader)
