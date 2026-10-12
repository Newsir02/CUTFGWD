from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from dataset.temporal import TemporalGraphBundle


EvaluationSplit = Literal["val", "test"]
DEFAULT_SCORE_CHUNK = 16


def _positive_update_batch(
    batch: TemporalData,
    device: torch.device,
) -> dict[str, Tensor]:
    return {
        "event_index": batch.event_index.to(device),
        "src": batch.src.to(device),
        "positive_dst": batch.dst.to(device),
        "event_time": batch.t.to(device),
        "edge_features": batch.msg.to(device),
    }


def _score_model(model: nn.Module, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    scorer = getattr(model, "score_candidates", None)
    if callable(scorer):
        return scorer(batch, return_relations=False)
    return model(batch, return_relations=False)


def _sampled_split(
    bundle: TemporalGraphBundle,
    split: EvaluationSplit,
    max_events: int,
    stateful: bool = False,
) -> TemporalData:
    data = bundle.split(split)
    num_events = int(data.num_events)
    if max_events > 0 and num_events > max_events:
        if stateful:
            # 有状态模型（教师/有状态学生）要求事件索引连续，只能取前段。
            data = data[:max_events]
        else:
            # 无状态模型可整段时间轴均匀抽样，避免只取前段的时间偏置。
            picked = (
                torch.linspace(0, num_events - 1, steps=max_events).round().long()
            )
            mask = torch.zeros(num_events, dtype=torch.bool)
            mask[picked.unique()] = True
            data = data[mask]
    return data


def _negative_tensors(
    negative_batches: object,
    device: torch.device,
) -> list[Tensor]:
    return [
        torch.as_tensor(item, dtype=torch.long, device=device).flatten()
        for item in negative_batches
    ]


def _chunk_candidate_batch(
    positive_batch: TemporalData,
    negative_tensors: list[Tensor],
    indices: list[int],
    device: torch.device,
) -> dict[str, Tensor]:
    index = torch.as_tensor(indices, dtype=torch.long, device=device)
    width = max(int(negative_tensors[i].numel()) for i in indices)
    rows = len(indices)
    candidates = torch.zeros(rows, 1 + width, dtype=torch.long, device=device)
    candidates[:, 0] = positive_batch.dst.to(device).long()[index]
    for row, event in enumerate(indices):
        negatives = negative_tensors[event]
        candidates[row, 1 : 1 + int(negatives.numel())] = negatives
    return {
        "event_index": positive_batch.event_index.to(device).long()[index],
        "src": positive_batch.src.to(device).long()[index],
        "positive_dst": positive_batch.dst.to(device).long()[index],
        "candidates": candidates,
        "target": torch.zeros(rows, dtype=torch.long, device=device),
        "timestamp": positive_batch.timestamp.to(device).float()[index],
        "event_time": positive_batch.t.to(device).long()[index],
        "edge_features": positive_batch.msg.to(device).float()[index],
    }


@torch.no_grad()
def evaluate_tgb(
    model: nn.Module,
    bundle: TemporalGraphBundle,
    split: EvaluationSplit,
    batch_size: int,
    device: torch.device,
    max_events: int = 0,
    score_chunk: int = DEFAULT_SCORE_CHUNK,
) -> dict[str, object]:
    """使用 TGB 官方历史负样本执行 one-vs-many 动态链路评测。

    模型打分按 `score_chunk` 个事件一批向量化（默认 16），仅有评分的
    `evaluator.eval` 仍逐事件进行；同批事件共享同一份模型状态，语义与逐事件
    打分一致，但显著更快。
    """

    if split not in {"val", "test"}:
        raise ValueError("TGB official evaluation split must be 'val' or 'test'.")
    if not bundle.official_evaluation:
        raise ValueError("Official TGB evaluation is unavailable for quick data.")
    if bundle.tgb_dataset is None:
        raise ValueError("Official TGB evaluation requires the TGB dataset backend.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if score_chunk <= 0:
        raise ValueError("score_chunk must be positive.")

    try:
        from tgb.linkproppred.evaluate import Evaluator
    except ImportError as exc:
        raise ImportError("Official evaluation requires py-tgb==2.2.0.") from exc

    backend = bundle.tgb_dataset
    if split == "val":
        backend.load_val_ns()
    else:
        backend.load_test_ns()
    sampler = backend.negative_sampler
    metric = str(backend.eval_metric)
    evaluator = Evaluator(name=bundle.dataset_name)
    loader = TemporalDataLoader(
        _sampled_split(
            bundle, split, max_events, stateful=hasattr(model, "update_state")
        ),
        batch_size=batch_size,
    )
    model.eval()
    values: list[float] = []
    update_state = getattr(model, "update_state", None)

    for positive_batch in loader:
        query_kwargs = {}
        if hasattr(positive_batch, "edge_type"):
            query_kwargs["edge_type"] = positive_batch.edge_type
        negative_batches = sampler.query_batch(
            positive_batch.src,
            positive_batch.dst,
            positive_batch.t,
            split_mode=split,
            **query_kwargs,
        )
        count = int(positive_batch.num_events)
        if len(negative_batches) != count:
            raise ValueError(
                "TGB negative sampler returned a different number of event groups."
            )
        negative_tensors = _negative_tensors(negative_batches, device)
        scored: list[tuple[int, Tensor]] = []
        for start in range(0, count, score_chunk):
            indices = list(range(start, min(start + score_chunk, count)))
            candidate_batch = _chunk_candidate_batch(
                positive_batch, negative_tensors, indices, device
            )
            logits = _score_model(model, candidate_batch)["logits"]
            for row, event in enumerate(indices):
                width = int(negative_tensors[event].numel())
                scored.append((event, logits[row, : 1 + width]))
        for _, row in scored:
            result = evaluator.eval(
                {
                    "y_pred_pos": row[:1],
                    "y_pred_neg": row[1:],
                    "eval_metric": [metric],
                }
            )
            values.append(float(result[metric]))
        # 同一正事件 batch 全部打分后才让模型观察这些真实边（与逐事件版一致）。
        if callable(update_state):
            update_state(_positive_update_batch(positive_batch, device))

    if not values:
        raise ValueError(f"The TGB {split} split is empty.")
    return {
        "metric": metric,
        metric: sum(values) / len(values),
        "official": True,
        "events": len(values),
    }
