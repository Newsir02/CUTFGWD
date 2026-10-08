from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch_geometric.nn.models.tgn import LastNeighborLoader

from .layers import MLP
from .student import LightSTMLPStudent


class LightGNNStudent(LightSTMLPStudent):
    """在 LightST MLP 学生之上增加一层轻量邻居聚合。

    只做一次 mean-pooling 邻居聚合（无 memory、无 attention、无多层消息传递），
    通过一个初始为 0 的可学习门控与 MLP 表示融合，因此训练初期等价于原 MLP 学生。
    关系槽分支完全复用父类，Anchor Relation / Structural OT 蒸馏仍然可用。
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
        self.neighbors = int(neighbors)
        self.model_config["student_arch"] = "gnn"
        self.model_config["student_neighbors"] = self.neighbors
        self.register_buffer(
            "association",
            torch.full((self.num_nodes,), -1, dtype=torch.long),
            persistent=False,
        )
        self.neighbor_mlp = MLP(hidden_dim, [hidden_dim], hidden_dim, dropout)
        # 初始为 0：模型起步不退化为更差的随机图模型。
        self.neighbor_gate = nn.Parameter(torch.zeros(1))
        self.neighbor_loader = LastNeighborLoader(
            self.num_nodes,
            size=self.neighbors,
            device=self.node_features.device,
        )
        self._current_context: Tensor | None = None

    def _apply(self, fn):
        result = super()._apply(fn)
        if hasattr(self, "neighbor_loader"):
            self.neighbor_loader = LastNeighborLoader(
                self.num_nodes,
                size=self.neighbors,
                device=self.node_features.device,
            )
            self.association.fill_(-1)
            self._current_context = None
        return result

    def runtime_state_elements(self) -> int:
        total = int(self.association.numel())
        loader = self.neighbor_loader
        for name in ("neighbors", "e_id", "_assoc"):
            value = getattr(loader, name, None)
            if isinstance(value, Tensor):
                total += int(value.numel())
            elif isinstance(value, dict):
                total += sum(
                    int(item.numel())
                    for item in value.values()
                    if isinstance(item, Tensor)
                )
        return total

    @staticmethod
    def _segment_mean(messages: Tensor, index: Tensor, size: int) -> Tensor:
        output = messages.new_zeros(size, messages.size(-1))
        output.index_add_(0, index, messages)
        counts = messages.new_zeros(size, 1)
        counts.index_add_(0, index, messages.new_ones(messages.size(0), 1))
        return output / counts.clamp_min(1.0)

    def _prepare_context(self, batch: Dict[str, Tensor]) -> None:
        source = batch["src"].long()
        candidates = batch["candidates"].long()
        query = torch.cat([source.reshape(-1), candidates.reshape(-1)]).unique()
        node_ids, edge_index, _ = self.neighbor_loader(query)
        base = self.node_encoder(self.node_features[node_ids]) + self.node_embeddings(
            node_ids
        )
        if edge_index.numel() > 0:
            messages = base[edge_index[0].long()]
            aggregated = self._segment_mean(
                messages,
                edge_index[1].long(),
                node_ids.numel(),
            )
        else:
            aggregated = base.new_zeros(node_ids.numel(), self.hidden_dim)
        self.association[node_ids] = torch.arange(
            node_ids.numel(),
            device=node_ids.device,
        )
        self._current_context = self.neighbor_mlp(aggregated)

    def _encode_nodes(self, node_ids: Tensor) -> Tensor:
        base = super()._encode_nodes(node_ids)
        if self._current_context is None:
            return base
        local = self.association[node_ids.long()]
        context = self._current_context[local]
        return base + self.neighbor_gate * context

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = True,
    ) -> Dict[str, Tensor]:
        self._prepare_context(batch)
        return super().forward(batch, return_relations=return_relations)

    @torch.no_grad()
    def reset_state(self) -> None:
        self.neighbor_loader.reset_state()

    @torch.no_grad()
    def update_state(self, batch: Dict[str, Tensor]) -> None:
        self.neighbor_loader.insert(
            batch["src"].long(),
            batch["positive_dst"].long(),
        )
