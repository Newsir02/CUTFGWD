from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import MLP
from .student import LightSTMLPStudent
from .temporal_neighbor_store import ExactTemporalNeighborStore, PyGNeighborStore


class TemporalRelationalAttention(nn.Module):
    """Single-head, age-aware attention over a node's historical interactions.

    The module is deliberately tiny (one additive-attention layer) and is
    initialised to output a uniform distribution, so enabling it never
    degrades the student at the start of training.  It is the student-side
    estimate of the teacher's *temporal relational influence* operator.
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 32) -> None:
        super().__init__()
        if hidden_dim <= 0 or attention_dim <= 0:
            raise ValueError("attention dimensions must be positive.")
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.query_projection = nn.Linear(hidden_dim, attention_dim)
        self.neighbor_projection = nn.Linear(hidden_dim, attention_dim)
        self.age_projection = nn.Linear(3, attention_dim)
        self.score = nn.Linear(attention_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    @staticmethod
    def _age_features(relative_age: Tensor, mask: Tensor) -> Tensor:
        # relative_age is already normalised to [0, 1] within each row.
        age = relative_age.clamp(0.0, 1.0)
        return torch.stack([age, age.square(), age.pow(3)], dim=-1)

    def forward(
        self,
        query: Tensor,
        neighbors: Tensor,
        relative_age: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Return normalised influence weights of shape ``[..., K]``."""

        mask = mask.bool()
        query_score = self.query_projection(query).unsqueeze(-2)
        neighbor_score = self.neighbor_projection(
            neighbors
        ) + self.age_projection(self._age_features(relative_age, mask))
        logits = self.score(torch.tanh(query_score + neighbor_score)).squeeze(-1)
        weights = F.softplus(logits) * mask.to(logits.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


class LightGNNStudent(LightSTMLPStudent):
    """一层邻居聚合的轻量时序 GNN 学生。

    邻居支持由 ``ExactTemporalNeighborStore`` 维护，保证保留的是**最近 K 条**
    交互（PyG 的 ``LastNeighborLoader`` 在单 batch 度大于 K 时会保留错误的边）。
    聚合方式可选：
      - ``mean``: 均匀均值（旧行为，便于复现）；
      - ``kernel``: 由 ``TemporalRelationalAttention`` 学习时间/内容感知的影响权重，
        该权重可被教师的时序关系注意力蒸馏监督。
    """

    def __init__(
        self,
        node_features: Tensor,
        hidden_dim: int = 48,
        time_dim: int = 16,
        relation_dim: int = 64,
        relation_slots: int = 8,
        dropout: float = 0.1,
        num_blocks: int = 3,
        neighbors: int = 8,
        aggregation: str = "mean",
        kernel_hidden: int = 32,
        event_times: Optional[Tensor] = None,
        time_span: Optional[float] = None,
        neighbor_store: str = "exact",
    ) -> None:
        super().__init__(
            node_features,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            relation_dim=relation_dim,
            relation_slots=relation_slots,
            dropout=dropout,
            num_blocks=num_blocks,
        )
        if neighbors <= 0:
            raise ValueError("Student neighbors must be positive.")
        if aggregation not in {"mean", "kernel"}:
            raise ValueError("student aggregation must be 'mean' or 'kernel'.")
        if neighbor_store not in {"exact", "pyg"}:
            raise ValueError("student neighbor_store must be 'exact' or 'pyg'.")
        self.neighbors = int(neighbors)
        self.aggregation = aggregation
        self.neighbor_store = neighbor_store
        self.model_config["student_arch"] = "gnn"
        self.model_config["student_neighbors"] = self.neighbors
        self.model_config["student_aggregation"] = self.aggregation
        self.model_config["student_kernel_hidden"] = int(kernel_hidden)
        self.model_config["student_neighbor_store"] = self.neighbor_store
        if event_times is not None:
            event_times = torch.as_tensor(event_times, dtype=torch.long).flatten()
            self.register_buffer(
                "event_times", event_times.detach().clone(), persistent=False
            )
            span = (
                float(time_span)
                if time_span is not None
                else float(
                    (event_times[-1].double() - event_times[0].double()).clamp_min(1.0)
                )
            )
            self.register_buffer(
                "time_span",
                torch.tensor(max(span, 1.0), dtype=torch.float64),
                persistent=False,
            )
        else:
            self.event_times = None
            self.time_span = None
        self.neighbor_mlp = MLP(hidden_dim, [hidden_dim], hidden_dim, dropout)
        # 初始为 0：模型起步等价于无邻居上下文的 MLP。
        self.neighbor_gate = nn.Parameter(torch.zeros(1))
        self.neighbor_loader = self._make_neighbor_store()
        self.attention: Optional[TemporalRelationalAttention] = None
        if self.aggregation == "kernel":
            if self.event_times is None:
                raise ValueError(
                    "kernel aggregation requires event_times to compute ages."
                )
            self.attention = TemporalRelationalAttention(
                hidden_dim, attention_dim=int(kernel_hidden)
            )

    def _make_neighbor_store(self):
        store_cls = (
            PyGNeighborStore if self.neighbor_store == "pyg" else ExactTemporalNeighborStore
        )
        return store_cls(
            self.num_nodes,
            size=self.neighbors,
            device=self.node_features.device,
        )

    def _apply(self, fn):
        result = super()._apply(fn)
        if hasattr(self, "neighbor_loader"):
            self.neighbor_loader = self._make_neighbor_store()
        return result

    def runtime_state_elements(self) -> int:
        total = int(self.neighbor_loader.neighbors.numel())
        total += int(self.neighbor_loader.e_id.numel())
        return total

    def _neighbor_ages(
        self,
        event_ids: Tensor,
        query_time: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if self.event_times is None:
            return torch.zeros_like(event_ids, dtype=torch.float32)
        safe_ids = event_ids.clamp_min(0)
        historical = self.event_times[safe_ids].to(torch.float64)
        span = self.time_span.clamp_min(1.0)
        age = (query_time.unsqueeze(-1).to(torch.float64) - historical) / span
        return age.clamp_min(0.0).to(torch.float32) * mask.to(torch.float32)

    @staticmethod
    def _relative_age(age: Tensor, mask: Tensor) -> Tensor:
        masked = torch.where(mask, age, torch.zeros_like(age))
        maximum = masked.amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return (masked / maximum).clamp(0.0, 1.0)

    def _aggregate_neighbors(
        self,
        query_ids: Tensor,
        query_state: Tensor,
        query_time: Tensor,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """Aggregate the newest K interactions for each query node.

        ``query_ids`` may have shape ``[B]`` (sources) or ``[B, C]``
        (candidates); the returned context has the matching shape plus the
        hidden dimension.
        """

        neighbor_ids, event_ids = self.neighbor_loader.query(query_ids)
        mask = event_ids >= 0
        safe_ids = neighbor_ids.clamp_min(0)
        neighbor_states = self._base_encode(safe_ids)
        age = self._neighbor_ages(event_ids, query_time, mask)
        relative_age = self._relative_age(age, mask)
        if self.attention is None:
            weights = mask.to(neighbor_states.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        else:
            weights = self.attention(
                query_state,
                neighbor_states,
                relative_age,
                mask,
            )
        context = (weights.unsqueeze(-1) * neighbor_states).sum(dim=-2)
        diagnostics = {
            "kernel_mass": weights,
            "kernel_event_ids": event_ids,
            "kernel_ages": age,
            "kernel_mask": mask,
        }
        return context, diagnostics

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = True,
    ) -> Dict[str, Tensor]:
        source_ids = batch["src"].long()
        candidate_ids = batch["candidates"].long()
        timestamps = batch["timestamp"]
        all_nodes = torch.cat([source_ids.reshape(-1), candidate_ids.reshape(-1)])
        if bool((all_nodes < 0).any()) or bool((all_nodes >= self.num_nodes).any()):
            raise ValueError("Student batch contains an out-of-range node id.")

        time_state = self.time_projection(self.time_encoder(timestamps))
        source_query = self._base_encode(source_ids) + time_state
        candidate_query = self._base_encode(candidate_ids) + time_state.unsqueeze(1)
        source_context, source_diagnostics = self._aggregate_neighbors(
            source_ids,
            source_query,
            batch["event_time"],
        )
        candidate_context, _ = self._aggregate_neighbors(
            candidate_ids,
            candidate_query,
            batch["event_time"].unsqueeze(-1),
        )
        source_state = (
            source_query + self.neighbor_gate * self.neighbor_mlp(source_context)
        )
        candidate_state = candidate_query + self.neighbor_gate * self.neighbor_mlp(
            candidate_context
        )
        source_state = self._apply_temporal_blocks(source_state)
        candidate_state = self._apply_temporal_blocks(candidate_state)
        outputs = self._run_heads(source_state, candidate_state, return_relations)
        if self.aggregation == "kernel":
            outputs.update(source_diagnostics)
        return outputs

    @torch.no_grad()
    def reset_state(self) -> None:
        self.neighbor_loader.reset_state()

    @torch.no_grad()
    def update_state(self, batch: Dict[str, Tensor]) -> None:
        event_ids = batch.get("event_index", None)
        self.neighbor_loader.insert(
            batch["src"].long(),
            batch["positive_dst"].long(),
            event_ids=event_ids,
        )
