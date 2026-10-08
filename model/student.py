from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import MLP, ResidualMLPBlock, TimeEncoding


class LightSTMLPStudent(nn.Module):
    """受 LightST 启发的图无关时序 MLP 学生。

    学生只读取节点属性、节点 ID 和查询时间，不访问邻接关系、历史邻居或
    TGN 记忆。固定时间槽用于承接教师的变长关系蒸馏信号。
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
    ) -> None:
        super().__init__()
        if node_features.dim() != 2 or node_features.size(0) == 0:
            raise ValueError("node_features must have shape [nodes, features].")
        if hidden_dim <= 0 or relation_dim <= 0 or relation_slots <= 0:
            raise ValueError("Student dimensions and relation_slots must be positive.")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")

        self.register_buffer("node_features", node_features.detach().clone().float())
        self.num_nodes = int(node_features.size(0))
        self.hidden_dim = hidden_dim
        self.relation_slots = relation_slots
        self.model_config = {
            "hidden_dim": hidden_dim,
            "time_dim": time_dim,
            "relation_dim": relation_dim,
            "relation_slots": relation_slots,
            "dropout": dropout,
            "num_blocks": num_blocks,
            "student_blocks": num_blocks,
            "student_layers": num_blocks,
            "num_nodes": self.num_nodes,
            "node_feature_dim": int(node_features.size(1)),
        }

        feature_dim = int(node_features.size(1))
        self.node_encoder = MLP(feature_dim, [hidden_dim], hidden_dim, dropout)
        self.node_embeddings = nn.Embedding(self.num_nodes, hidden_dim)
        nn.init.normal_(self.node_embeddings.weight, std=0.02)
        self.time_encoder = TimeEncoding(time_dim)
        self.time_projection = nn.Linear(time_dim, hidden_dim)
        self.temporal_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.score_head = MLP(
            4 * hidden_dim,
            [2 * hidden_dim, hidden_dim],
            1,
            dropout,
        )

        self.slot_embeddings = nn.Parameter(
            torch.randn(relation_slots, hidden_dim) * 0.02
        )
        self.age_gaps = nn.Parameter(torch.zeros(relation_slots))
        self.slot_time_projection = nn.Linear(time_dim, hidden_dim)
        self.relation_decoder = MLP(
            hidden_dim,
            [relation_dim],
            relation_dim,
            dropout,
        )
        self.relation_mass_head = nn.Linear(relation_dim, 1)

    def _encode_nodes(self, node_ids: Tensor) -> Tensor:
        return self.node_encoder(self.node_features[node_ids]) + self.node_embeddings(
            node_ids
        )

    def _apply_temporal_blocks(self, states: Tensor) -> Tensor:
        for block in self.temporal_blocks:
            states = block(states)
        return states

    def _ordered_slot_ages(self, batch_size: int) -> Tensor:
        # 正间隔保证槽年龄从旧到新单调递减，与教师邻居队列顺序一致。
        gaps = F.softplus(self.age_gaps) + 1.0e-4
        centers = torch.cumsum(gaps, dim=0) - 0.5 * gaps
        normalized_ages = 1.0 - centers / gaps.sum()
        return normalized_ages.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = True,
    ) -> Dict[str, Tensor]:
        # 此处有意只读取三个图无关字段，历史图内容不会影响学生推理。
        source_ids = batch["src"]
        candidate_ids = batch["candidates"]
        timestamps = batch["timestamp"]
        all_nodes = torch.cat([source_ids.reshape(-1), candidate_ids.reshape(-1)])
        if bool((all_nodes < 0).any()) or bool((all_nodes >= self.num_nodes).any()):
            raise ValueError("Student batch contains an out-of-range node id.")

        time_state = self.time_projection(self.time_encoder(timestamps))
        source_state = self._encode_nodes(source_ids) + time_state
        candidate_state = self._encode_nodes(candidate_ids) + time_state.unsqueeze(1)
        source_state = self._apply_temporal_blocks(source_state)
        candidate_state = self._apply_temporal_blocks(candidate_state)

        source_expanded = source_state.unsqueeze(1).expand_as(candidate_state)
        score_inputs = torch.cat(
            [
                source_expanded,
                candidate_state,
                source_expanded * candidate_state,
                torch.abs(source_expanded - candidate_state),
            ],
            dim=-1,
        )
        outputs = {"logits": self.score_head(score_inputs).squeeze(-1)}

        if return_relations:
            slot_ages = self._ordered_slot_ages(source_state.size(0))
            slot_time = self.slot_time_projection(self.time_encoder(slot_ages))
            slot_inputs = (
                source_state.unsqueeze(1)
                + self.slot_embeddings.unsqueeze(0)
                + slot_time
            )
            relation_tokens = self.relation_decoder(slot_inputs)
            outputs.update(
                {
                    "relation_tokens": relation_tokens,
                    "relation_ages": slot_ages,
                    "relation_mass_logits": self.relation_mass_head(
                        relation_tokens
                    ).squeeze(-1),
                }
            )
        return outputs
