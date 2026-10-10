from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from .tgn import GraphAttentionEmbedding, LinkPredictor


class PyGTGNTeacher(nn.Module):
    """由 PyG 官方 TGN 组件组装的连续时间图教师。"""

    def __init__(
        self,
        event_times: Tensor,
        event_messages: Tensor,
        num_nodes: int,
        hidden_dim: int = 64,
        time_dim: int = 16,
        relation_dim: int = 64,
        temporal_neighbors: int = 24,
        dropout: float = 0.1,
        teacher_layers: int = 1,
    ) -> None:
        super().__init__()
        event_times = torch.as_tensor(event_times, dtype=torch.long).flatten()
        event_messages = torch.as_tensor(event_messages, dtype=torch.float)
        if event_times.numel() == 0:
            raise ValueError("TGN requires at least one temporal event.")
        if event_messages.dim() != 2 or event_messages.size(0) != event_times.numel():
            raise ValueError("event_messages must have shape [events, features].")
        if event_messages.size(1) == 0:
            raise ValueError("TGN event messages must have at least one feature.")
        if not bool(torch.isfinite(event_messages).all()):
            raise ValueError("TGN event messages must be finite.")
        if event_times.numel() > 1 and bool(
            (event_times[1:] < event_times[:-1]).any()
        ):
            raise ValueError("TGN event times must be chronological.")
        if num_nodes <= 0 or hidden_dim <= 0 or time_dim <= 0:
            raise ValueError("TGN node and hidden dimensions must be positive.")
        if relation_dim <= 0 or temporal_neighbors <= 0:
            raise ValueError("TGN relation dimensions must be positive.")
        if teacher_layers <= 0:
            raise ValueError("TGN teacher_layers must be positive.")
        if hidden_dim % 2 != 0:
            raise ValueError("TGN hidden_dim must be even for two attention heads.")

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim
        self.temporal_neighbors = temporal_neighbors
        self.teacher_layers = teacher_layers
        self.msg_dim = int(event_messages.size(1))
        self.model_config = {
            "num_nodes": num_nodes,
            "num_events": int(event_times.numel()),
            "msg_dim": self.msg_dim,
            "hidden_dim": hidden_dim,
            "time_dim": time_dim,
            "relation_dim": relation_dim,
            "temporal_neighbors": temporal_neighbors,
            "teacher_layers": teacher_layers,
            "dropout": dropout,
        }

        self.register_buffer(
            "event_times",
            event_times.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "event_messages",
            event_messages.detach().clone(),
            persistent=False,
        )
        # -1 表示该全局节点在当前 batch 中未被映射；避免 torch.empty 的未初始化值。
        self.register_buffer(
            "association",
            torch.full((num_nodes,), -1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "stream_event_index",
            torch.tensor(-1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "stream_timestamp",
            torch.tensor(torch.iinfo(torch.long).min, dtype=torch.long),
            persistent=False,
        )

        message_module = IdentityMessage(self.msg_dim, hidden_dim, time_dim)
        self.memory = TGNMemory(
            num_nodes,
            self.msg_dim,
            hidden_dim,
            time_dim,
            message_module=message_module,
            aggregator_module=LastAggregator(),
        )
        # 这些是数据流运行状态，检查点只保存可迁移的模型参数。
        self.memory._non_persistent_buffers_set.update(
            {"memory", "last_update", "_assoc"}
        )
        self.gnn = GraphAttentionEmbedding(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            msg_dim=self.msg_dim,
            time_encoder=self.memory.time_enc,
            dropout=dropout,
            num_layers=teacher_layers,
        )
        self.link_predictor = LinkPredictor(hidden_dim)
        relation_input_dim = hidden_dim + time_dim + self.msg_dim
        self.relation_projector = nn.Sequential(
            nn.Linear(relation_input_dim, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim),
        )
        # 关系对齐头：教师预训练时把池化关系表示映射回打分空间，
        # 从而让 relation_projector 真正获得任务相关梯度。
        self.relation_align_head = nn.Linear(relation_dim, hidden_dim)
        self.neighbor_loader = LastNeighborLoader(
            num_nodes,
            size=temporal_neighbors,
            device=self.event_times.device,
        )

    def _apply(self, fn):
        result = super()._apply(fn)
        if hasattr(self, "neighbor_loader"):
            self.neighbor_loader = LastNeighborLoader(
                self.num_nodes,
                size=self.temporal_neighbors,
                device=self.event_times.device,
            )
            self.memory.reset_state()
            self.stream_event_index.fill_(-1)
            self.stream_timestamp.fill_(torch.iinfo(torch.long).min)
        return result

    def runtime_state_elements(self) -> int:
        seen: set[int] = set()
        total = 0

        def add_tensor(value: object) -> None:
            nonlocal total
            if not isinstance(value, Tensor):
                return
            key = id(value)
            if key in seen:
                return
            seen.add(key)
            total += int(value.numel())

        add_tensor(self.memory.memory)
        add_tensor(self.memory.last_update)
        add_tensor(getattr(self.memory, "_assoc", None))
        add_tensor(self.association)
        add_tensor(self.stream_event_index)
        add_tensor(self.stream_timestamp)
        for name in ("neighbors", "e_id", "_assoc"):
            add_tensor(getattr(self.neighbor_loader, name, None))
        if hasattr(self.neighbor_loader, "cur_e_id"):
            total += 1
        return total

    def train(self, mode: bool = True):
        state_was_empty = self.state_is_empty
        result = super().train(mode)
        # PyG 在 train->eval 时会 flush 消息仓；空仓经过带偏置 GRU 后不再为零。
        if state_was_empty:
            self.memory.reset_state()
        return result

    @property
    def state_is_empty(self) -> bool:
        return bool(
            int(self.stream_event_index) < 0
            and int(self.neighbor_loader.cur_e_id) == 0
        )

    @torch.no_grad()
    def reset_state(self) -> None:
        self.memory.reset_state()
        self.neighbor_loader.reset_state()
        self.stream_event_index.fill_(-1)
        self.stream_timestamp.fill_(torch.iinfo(torch.long).min)

    def detach_state(self) -> None:
        self.memory.detach()

    def _validate_score_batch(self, batch: Dict[str, Tensor]) -> None:
        required = {
            "event_index",
            "src",
            "candidates",
            "timestamp",
            "event_time",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"TGN batch is missing fields: {', '.join(missing)}")
        event_index = batch["event_index"]
        source = batch["src"]
        candidates = batch["candidates"]
        event_time = batch["event_time"]
        batch_size = int(source.numel())
        if batch_size == 0:
            raise ValueError("TGN cannot score an empty batch.")
        if event_index.shape != (batch_size,) or event_time.shape != (batch_size,):
            raise ValueError("event_index and event_time must have shape [batch].")
        if candidates.dim() != 2 or candidates.size(0) != batch_size:
            raise ValueError("candidates must have shape [batch, candidates].")
        if batch_size > 1:
            if bool((event_index[1:] <= event_index[:-1]).any()) or bool(
                (event_time[1:] < event_time[:-1]).any()
            ):
                raise ValueError("TGN batches must be chronological.")
        node_ids = torch.cat([source.flatten(), candidates.flatten()])
        if bool((node_ids < 0).any()) or bool((node_ids >= self.num_nodes).any()):
            raise ValueError("TGN batch contains an out-of-range node id.")
        if int(self.stream_event_index) >= 0:
            if int(event_index[0]) <= int(self.stream_event_index):
                raise ValueError(
                    "Event was already observed; call reset_state() before replaying."
                )
            if int(event_time[0]) < int(self.stream_timestamp):
                raise ValueError("TGN batches must remain chronological.")

    def _relation_outputs(
        self,
        source: Tensor,
        query_time: Tensor,
        edge_index: Tensor,
        event_ids: Tensor,
        edge_attributes: Tensor,
        attention: Tensor,
        local_memory: Tensor,
    ) -> dict[str, Tensor]:
        batch_size = int(source.numel())
        slots = self.temporal_neighbors
        tokens = local_memory.new_zeros(
            batch_size,
            slots,
            self.relation_dim,
        )
        ages = local_memory.new_zeros(batch_size, slots)
        mask = torch.zeros(
            batch_size,
            slots,
            dtype=torch.bool,
            device=local_memory.device,
        )
        mass = local_memory.new_zeros(batch_size, slots)
        if event_ids.numel() == 0:
            return {
                "relation_tokens": tokens,
                "relation_ages": ages,
                "relation_mask": mask,
                "relation_mass": mass,
            }

        edge_tokens = self.relation_projector(
            torch.cat([local_memory[edge_index[0]], edge_attributes], dim=-1)
        )
        edge_mass = attention.mean(dim=-1)
        time_span = (
            self.event_times[-1].double() - self.event_times[0].double()
        ).clamp_min(1.0)

        # 向量化版本：先按事件时间稳定排序，再按目标局部节点稳定排序，
        # 得到每个节点内部从旧到新的边序列，避免逐行 Python 循环。
        destination = edge_index[1].long()
        order = torch.argsort(event_ids, stable=True)
        order = order[torch.argsort(destination[order], stable=True)]
        sorted_destination = destination[order]
        num_local = int(destination.max().item()) + 1
        counts = torch.zeros(
            num_local,
            dtype=torch.long,
            device=destination.device,
        ).scatter_add_(0, sorted_destination, torch.ones_like(sorted_destination))
        ends = torch.cumsum(counts, dim=0)
        starts = ends - counts

        local_of_row = self.association[source.long()].long()
        valid_row = local_of_row >= 0
        safe_local = local_of_row.clamp(min=0, max=num_local - 1)
        row_start = starts[safe_local]
        row_end = ends[safe_local]
        # 每个 query 行只保留窗口内最新的 temporal_neighbors 条历史边。
        available = (row_end - row_start).clamp_(max=slots)
        available = torch.where(
            valid_row,
            available,
            torch.zeros_like(available),
        )
        slot_ids = torch.arange(slots, device=local_memory.device)
        positions = (row_end - available).unsqueeze(1) + slot_ids.unsqueeze(0)
        slot_mask = slot_ids.unsqueeze(0) < available.unsqueeze(1)
        safe_positions = torch.where(
            slot_mask,
            positions,
            torch.zeros_like(positions),
        )
        edge_selection = order[safe_positions]
        tokens = edge_tokens[edge_selection] * slot_mask.unsqueeze(-1).to(
            edge_tokens.dtype
        )
        historical_time = self.event_times[event_ids[edge_selection]].double()
        selected_ages = (
            query_time.unsqueeze(1).double() - historical_time
        ).clamp_min(0.0).div(time_span)
        ages = torch.where(
            slot_mask,
            selected_ages.to(local_memory.dtype),
            torch.zeros_like(ages),
        )
        mask = slot_mask
        selected_mass = edge_mass[edge_selection].clamp_min(0.0) * slot_mask.to(
            edge_mass.dtype
        )
        mass = selected_mass / selected_mass.sum(dim=1, keepdim=True).clamp_min(
            1.0e-12
        )
        return {
            "relation_tokens": tokens,
            "relation_ages": ages,
            "relation_mask": mask,
            "relation_mass": mass,
        }

    def score_candidates(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = True,
    ) -> dict[str, Tensor]:
        self._validate_score_batch(batch)
        source = batch["src"].long()
        candidates = batch["candidates"].long()
        query_nodes = torch.cat([source, candidates.flatten()]).unique()
        node_ids, edge_index, event_ids = self.neighbor_loader(query_nodes)
        self.association[node_ids] = torch.arange(
            node_ids.numel(),
            device=node_ids.device,
        )
        local_memory, last_update = self.memory(node_ids)
        edge_time = self.event_times[event_ids]
        edge_message = self.event_messages[event_ids]
        embeddings, edge_attributes, attention = self.gnn(
            local_memory,
            last_update,
            edge_index,
            edge_time,
            edge_message,
        )
        source_embedding = embeddings[self.association[source]]
        candidate_embedding = embeddings[self.association[candidates]]
        expanded_source = source_embedding.unsqueeze(1).expand_as(
            candidate_embedding
        )
        output = {
            "logits": self.link_predictor(
                expanded_source.reshape(-1, self.hidden_dim),
                candidate_embedding.reshape(-1, self.hidden_dim),
            ).reshape_as(candidates)
        }
        if return_relations:
            output.update(
                self._relation_outputs(
                    source,
                    batch["event_time"],
                    edge_index,
                    event_ids,
                    edge_attributes,
                    attention,
                    local_memory,
                )
            )
            # 供教师关系头的自监督对齐使用，不参与 teacher 打分路径。
            output["source_embedding"] = source_embedding
        return output

    @torch.no_grad()
    def update_state(self, batch: Dict[str, Tensor]) -> None:
        required = {
            "event_index",
            "src",
            "positive_dst",
            "event_time",
            "edge_features",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"TGN update batch is missing fields: {', '.join(missing)}")
        event_index = batch["event_index"].long()
        expected = torch.arange(
            self.neighbor_loader.cur_e_id,
            self.neighbor_loader.cur_e_id + event_index.numel(),
            device=event_index.device,
        )
        if not torch.equal(event_index, expected):
            raise ValueError(
                "TGN updates must be contiguous; call reset_state() before replaying."
            )
        source = batch["src"].long()
        destination = batch["positive_dst"].long()
        event_time = batch["event_time"].long()
        edge_features = batch["edge_features"].float()
        if edge_features.shape != (source.numel(), self.msg_dim):
            raise ValueError(
                f"edge_features must have shape [batch, {self.msg_dim}]."
            )
        self.memory.update_state(
            source,
            destination,
            event_time,
            edge_features,
        )
        self.neighbor_loader.insert(source, destination)
        self.stream_event_index.copy_(event_index[-1])
        self.stream_timestamp.copy_(event_time[-1])

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = True,
        update_memory: bool = True,
    ) -> dict[str, Tensor]:
        output = self.score_candidates(batch, return_relations=return_relations)
        if update_memory:
            self.update_state(batch)
        return output
