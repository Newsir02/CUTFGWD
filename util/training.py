from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn


class AverageMeter:
    def __init__(self) -> None:
        # total 累计 value * count，count 累计样本数，用于加权平均。
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        # 每个 batch 的 loss 按样本数加权，避免小 batch 影响均值。
        self.total += float(value) * count
        self.count += count

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


def move_to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    # DataLoader 返回字典 batch，这里统一搬到 CPU/GPU 目标设备。
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def count_trainable_parameters(model: nn.Module) -> int:
    # 只统计 requires_grad=True 的参数，冻结 teacher 后不会计入可训练规模。
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_runtime_state_elements(model: nn.Module) -> int:
    counter = getattr(model, "runtime_state_elements", None)
    if callable(counter):
        return int(counter())
    return 0


def model_parameter_summary(model: nn.Module) -> Dict[str, int]:
    trainable = count_trainable_parameters(model)
    all_parameters = count_all_parameters(model)
    runtime_state = count_runtime_state_elements(model)
    return {
        "trainable": trainable,
        "all": all_parameters,
        "runtime_state_elements": runtime_state,
        "total_with_runtime_state": all_parameters + runtime_state,
    }


def _synchronize(device: torch.device) -> None:
    # CUDA 调用是异步的，计时前后同步才能得到真实延迟。
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    loader: Iterable[Dict[str, Tensor]],
    device: torch.device,
    warmup_batches: int = 5,
    measured_batches: int = 20,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Measure model-only latency on full candidate batches."""

    model.eval()
    call_kwargs = dict(forward_kwargs or {})
    iterator = iter(loader)

    def next_batch() -> Dict[str, Tensor]:
        # loader 耗尽后重新开始迭代，保证 warmup/measure 批次数可控。
        nonlocal iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return move_to_device(batch, device)

    for _ in range(warmup_batches):
        # 预热阶段让 CUDA kernel 和缓存进入稳定状态，不计入最终延迟。
        model(next_batch(), return_relations=False, **call_kwargs)
    _synchronize(device)

    elapsed_batches = []
    total_queries = 0
    total_seconds = 0.0
    for _ in range(measured_batches):
        batch = next_batch()
        _synchronize(device)
        start = time.perf_counter()
        model(batch, return_relations=False, **call_kwargs)
        _synchronize(device)
        elapsed = time.perf_counter() - start
        elapsed_batches.append(elapsed)
        total_seconds += elapsed
        total_queries += int(batch["src"].size(0))

    latency = torch.tensor(elapsed_batches) * 1000.0
    # 同时返回 batch 级延迟和 query 级吞吐，便于比较 teacher/student。
    return {
        "batch_ms_mean": float(latency.mean()),
        "batch_ms_p50": float(torch.quantile(latency, 0.50)),
        "batch_ms_p95": float(torch.quantile(latency, 0.95)),
        "query_ms": 1000.0 * total_seconds / max(total_queries, 1),
        "queries_per_second": total_queries / max(total_seconds, 1.0e-12),
    }
