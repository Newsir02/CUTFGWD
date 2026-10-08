from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn


class TimeEncoding(nn.Module):
    """Fixed multi-frequency sine/cosine encoding for scalar timestamps."""

    def __init__(self, dimension: int, max_frequency: float = 64.0) -> None:
        super().__init__()
        if dimension < 2:
            raise ValueError("TimeEncoding dimension must be at least 2.")
        # 使用对数间隔频率，让编码同时覆盖慢变化和快变化的时间模式。
        half = (dimension + 1) // 2
        frequencies = torch.logspace(0.0, math.log10(max_frequency), half)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.dimension = dimension

    def forward(self, timestamps: Tensor) -> Tensor:
        # 输入可以是任意形状的时间戳，最后一维展开为 sin/cos 时间特征。
        angles = 2.0 * math.pi * timestamps.unsqueeze(-1) * self.frequencies
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return encoded[..., : self.dimension]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        # 根据输入维度、隐藏层列表和输出维度动态堆叠全连接网络。
        dimensions = [input_dim, *hidden_dims, output_dim]
        layers = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            is_last = index == len(dimensions) - 2
            if not is_last:
                # 中间层可选 LayerNorm + GELU + Dropout，输出层保持线性。
                if layer_norm:
                    layers.append(nn.LayerNorm(dimensions[index + 1]))
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class ResidualMLPBlock(nn.Module):
    """LightST 风格学生使用的预归一化残差 MLP 块。"""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.norm = nn.LayerNorm(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.network(self.norm(inputs))
