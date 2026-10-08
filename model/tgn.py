from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import TransformerConv


class GraphAttentionEmbedding(nn.Module):
    """按照 PyG 官方 TGN 示例，用历史消息生成时间图嵌入。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        msg_dim: int,
        time_encoder: nn.Module,
        dropout: float,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError("TGN embedding dimension must be even.")
        if num_layers <= 0:
            raise ValueError("TGN embedding num_layers must be positive.")
        self.time_encoder = time_encoder
        edge_dim = msg_dim + int(time_encoder.out_channels)
        self.convs = nn.ModuleList(
            [
                TransformerConv(
                    in_channels if layer_index == 0 else out_channels,
                    out_channels // 2,
                    heads=2,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
                for layer_index in range(num_layers)
            ]
        )

    @property
    def conv(self) -> TransformerConv:
        return self.convs[0]

    def forward(
        self,
        memory: Tensor,
        last_update: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_message: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        relative_time = last_update[edge_index[0]] - edge_time
        time_features = self.time_encoder(relative_time.to(memory.dtype))
        edge_attributes = torch.cat([time_features, edge_message], dim=-1)
        embeddings = memory
        attention = edge_attributes.new_zeros(edge_index.size(1), 2)
        for conv in self.convs:
            embeddings, (_, attention) = conv(
                embeddings,
                edge_index,
                edge_attributes,
                return_attention_weights=True,
            )
        return embeddings, edge_attributes, attention


class LinkPredictor(nn.Module):
    """PyG TGN 示例使用的轻量链路预测头。"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.source_projection = nn.Linear(in_channels, in_channels)
        self.destination_projection = nn.Linear(in_channels, in_channels)
        self.output = nn.Linear(in_channels, 1)

    def forward(self, source: Tensor, destination: Tensor) -> Tensor:
        hidden = (
            self.source_projection(source)
            + self.destination_projection(destination)
        ).relu()
        return self.output(hidden).squeeze(-1)
