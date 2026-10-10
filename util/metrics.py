from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


class RankingMetricAccumulator:
    def __init__(self) -> None:
        # 累计 sum 后统一除以样本数，适合跨多个 batch 计算整体指标。
        self.reciprocal_rank_sum = 0.0
        self.hits_one_sum = 0.0
        self.hits_three_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, logits: Tensor, target: Tensor) -> None:
        # 每次 update 处理一个 batch，将 MRR 和 hits 的分子累加起来。
        order = torch.argsort(logits, dim=-1, descending=True)
        ranks = (order == target.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1] + 1
        ranks = ranks.float()
        self.reciprocal_rank_sum += float((1.0 / ranks).sum())
        self.hits_one_sum += float((ranks <= 1).float().sum())
        self.hits_three_sum += float((ranks <= 3).float().sum())
        self.count += int(target.numel())

    def compute(self) -> Dict[str, float]:
        # 防止空数据时除零；正常训练评估中 denominator 等于样本总数。
        denominator = max(self.count, 1)
        return {
            "mrr": self.reciprocal_rank_sum / denominator,
            "hits@1": self.hits_one_sum / denominator,
            "hits@3": self.hits_three_sum / denominator,
        }
