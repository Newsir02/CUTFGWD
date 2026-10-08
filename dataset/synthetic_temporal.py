from __future__ import annotations

import math

import torch
from torch_geometric.data import TemporalData

from .temporal import TemporalGraphBundle, build_temporal_graph_bundle


def build_synthetic_temporal_graph(
    num_nodes: int = 256,
    num_events: int = 3000,
    feature_dim: int = 16,
    latent_dim: int = 8,
    num_communities: int = 4,
    num_candidates: int = 16,
    history_size: int = 24,
    edge_feature_dim: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> TemporalGraphBundle:
    """生成仅供自动化 smoke 使用的连续正事件流。"""

    if num_nodes < 3:
        raise ValueError("num_nodes must be at least 3.")
    if num_events < 20:
        raise ValueError("num_events must be at least 20.")
    if edge_feature_dim <= 0:
        raise ValueError("edge_feature_dim must be positive.")
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0:
        raise ValueError("train_ratio and val_ratio must be in (0, 1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(num_nodes, latent_dim, generator=generator)
    communities = torch.randint(
        num_communities,
        (num_nodes,),
        generator=generator,
    )
    community_features = torch.nn.functional.one_hot(
        communities,
        num_classes=num_communities,
    ).float()
    base_features = torch.cat([latent, community_features], dim=-1)
    projection = torch.randn(
        base_features.size(1),
        feature_dim,
        generator=generator,
    ) / math.sqrt(float(base_features.size(1)))
    node_features = base_features @ projection
    node_features = (
        node_features - node_features.mean(0)
    ) / node_features.std(0).clamp_min(1.0e-5)

    phases = 2.0 * math.pi * torch.rand(num_nodes, generator=generator)
    all_nodes = torch.arange(num_nodes)
    sources = torch.empty(num_events, dtype=torch.long)
    destinations = torch.empty(num_events, dtype=torch.long)
    histories: list[list[int]] = [[] for _ in range(num_nodes)]
    positive_pool_size = min(num_nodes - 1, max(48, 4 * num_candidates))

    for event_index in range(num_events):
        query_time = event_index / max(1, num_events - 1)
        source = int(torch.randint(num_nodes, (1,), generator=generator))
        sources[event_index] = source
        possible = all_nodes[all_nodes != source]
        order = torch.randperm(possible.numel(), generator=generator)
        pool = possible[order[:positive_pool_size]]

        affinity = (latent[source] * latent[pool]).sum(-1) / math.sqrt(
            float(latent_dim)
        )
        same_community = (communities[pool] == communities[source]).float()
        periodic = torch.cos(
            phases[pool] - phases[source] + 2.0 * math.pi * query_time
        )
        scores = affinity + 1.20 * same_community + 0.45 * periodic
        for recency, neighbor in enumerate(
            reversed(histories[source][-history_size:])
        ):
            scores[pool == neighbor] += 2.4 * math.exp(-0.35 * recency)

        probabilities = torch.softmax(scores / 0.75, dim=0)
        positive_index = int(
            torch.multinomial(probabilities, 1, generator=generator)
        )
        positive = int(pool[positive_index])
        destinations[event_index] = positive
        histories[source].append(positive)
        histories[positive].append(source)

    event_ids = torch.arange(num_events)
    train_end = int(num_events * train_ratio)
    val_end = int(num_events * (train_ratio + val_ratio))
    data = TemporalData(
        src=sources,
        dst=destinations,
        t=event_ids.long(),
        msg=torch.zeros(num_events, edge_feature_dim),
        y=torch.ones(num_events),
    )
    return build_temporal_graph_bundle(
        dataset_name="synthetic",
        data=data,
        node_features=node_features,
        train_mask=event_ids < train_end,
        val_mask=(event_ids >= train_end) & (event_ids < val_end),
        test_mask=event_ids >= val_end,
        feature_dim=feature_dim,
        official_evaluation=False,
    )
