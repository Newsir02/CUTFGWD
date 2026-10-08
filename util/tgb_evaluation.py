from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from dataset.temporal import TemporalGraphBundle


EvaluationSplit = Literal["val", "test"]


def _single_event_candidates(
    batch: TemporalData,
    index: int,
    negative_destinations: object,
    device: torch.device,
) -> dict[str, Tensor]:
    negatives = torch.as_tensor(
        negative_destinations,
        dtype=torch.long,
        device=device,
    ).flatten()
    if negatives.numel() == 0:
        raise ValueError("TGB official evaluation returned no negative targets.")
    positive = batch.dst[index : index + 1].to(device)
    candidates = torch.cat([positive, negatives]).unsqueeze(0)
    return {
        "event_index": batch.event_index[index : index + 1].to(device),
        "src": batch.src[index : index + 1].to(device),
        "positive_dst": positive,
        "candidates": candidates,
        "target": torch.zeros(1, dtype=torch.long, device=device),
        "timestamp": batch.timestamp[index : index + 1].to(device),
        "event_time": batch.t[index : index + 1].to(device),
        "edge_features": batch.msg[index : index + 1].to(device),
    }


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


def _score_model(
    model: nn.Module,
    batch: dict[str, Tensor],
) -> dict[str, Tensor]:
    score_candidates = getattr(model, "score_candidates", None)
    if callable(score_candidates):
        return score_candidates(batch, return_relations=False)
    return model(batch, return_relations=False)


@torch.no_grad()
def evaluate_tgb(
    model: nn.Module,
    bundle: TemporalGraphBundle,
    split: EvaluationSplit,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    """使用 TGB 官方历史负样本执行 one-vs-many 动态链路评测。"""

    if split not in {"val", "test"}:
        raise ValueError("TGB official evaluation split must be 'val' or 'test'.")
    if not bundle.official_evaluation:
        raise ValueError("Official TGB evaluation is unavailable for quick data.")
    if bundle.tgb_dataset is None:
        raise ValueError("Official TGB evaluation requires the TGB dataset backend.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

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
    loader = TemporalDataLoader(bundle.split(split), batch_size=batch_size)
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
        if len(negative_batches) != int(positive_batch.num_events):
            raise ValueError(
                "TGB negative sampler returned a different number of event groups."
            )
        for index, negative_destinations in enumerate(negative_batches):
            candidate_batch = _single_event_candidates(
                positive_batch,
                index,
                negative_destinations,
                device,
            )
            logits = _score_model(model, candidate_batch)["logits"][0]
            result = evaluator.eval(
                {
                    "y_pred_pos": logits[:1],
                    "y_pred_neg": logits[1:],
                    "eval_metric": [metric],
                }
            )
            values.append(float(result[metric]))

        # 同一正事件 batch 全部打分后，教师才观察这些真实边。
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
