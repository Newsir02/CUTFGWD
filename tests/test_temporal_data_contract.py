from __future__ import annotations

import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from dataset import CandidateBatchLoader, build_synthetic_temporal_graph
from dataset.temporal import TemporalGraphBundle, build_temporal_graph_bundle


def make_tiny_bundle() -> TemporalGraphBundle:
    data = TemporalData(
        src=torch.tensor([0, 1, 0, 2]),
        dst=torch.tensor([3, 4, 4, 3]),
        t=torch.tensor([10, 20, 30, 40]),
        msg=torch.arange(8, dtype=torch.float).reshape(4, 2),
        y=torch.ones(4),
    )
    return build_temporal_graph_bundle(
        dataset_name="tiny",
        data=data,
        node_features=None,
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.tensor([False, False, True, False]),
        test_mask=torch.tensor([False, False, False, True]),
        feature_dim=3,
        official_evaluation=False,
    )


def test_candidate_loader_wraps_pyg_temporal_loader() -> None:
    bundle = make_tiny_bundle()
    loader = CandidateBatchLoader(
        bundle=bundle,
        split="train",
        batch_size=2,
        num_candidates=2,
        seed=7,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert isinstance(loader.temporal_loader, TemporalDataLoader)
    assert batch["candidates"].shape == (2, 2)
    assert torch.equal(
        batch["candidates"].gather(1, batch["target"][:, None]).squeeze(1),
        batch["positive_dst"],
    )
    assert batch["event_time"].dtype == torch.long
    assert torch.equal(batch["event_index"], torch.tensor([0, 1]))
    assert torch.all(batch["timestamp"] >= 0.0)
    assert torch.all(batch["timestamp"] <= 1.0)


def test_candidate_sampling_stays_in_destination_domain() -> None:
    bundle = make_tiny_bundle()
    loader = CandidateBatchLoader(bundle, "train", 2, 2, 11, 0)

    batch = next(iter(loader))

    assert set(batch["candidates"].flatten().tolist()) <= {3, 4}


def test_candidate_loader_is_deterministic_for_each_replay() -> None:
    bundle = make_tiny_bundle()
    loader = CandidateBatchLoader(bundle, "train", 2, 2, 19, 0)

    first = next(iter(loader))
    second = next(iter(loader))

    assert torch.equal(first["candidates"], second["candidates"])
    assert torch.equal(first["target"], second["target"])


def test_missing_edge_messages_become_one_dimensional_zeros() -> None:
    data = TemporalData(
        src=torch.tensor([0, 1, 0]),
        dst=torch.tensor([2, 3, 2]),
        t=torch.tensor([1, 2, 3]),
        y=torch.ones(3),
    )

    bundle = build_temporal_graph_bundle(
        dataset_name="without-msg",
        data=data,
        node_features=None,
        train_mask=torch.tensor([True, True, False]),
        val_mask=torch.tensor([False, False, True]),
        test_mask=torch.tensor([False, False, False]),
        feature_dim=4,
        official_evaluation=False,
    )

    assert bundle.data.msg.shape == (3, 1)
    assert torch.count_nonzero(bundle.data.msg) == 0


def test_explicit_features_preserve_nodes_missing_from_the_event_prefix() -> None:
    data = TemporalData(
        src=torch.tensor([0, 1, 0]),
        dst=torch.tensor([2, 3, 2]),
        t=torch.tensor([1, 2, 3]),
        msg=torch.zeros(3, 1),
        y=torch.ones(3),
    )
    node_features = torch.arange(10, dtype=torch.float).reshape(5, 2)

    bundle = build_temporal_graph_bundle(
        dataset_name="unobserved-node",
        data=data,
        node_features=node_features,
        train_mask=torch.tensor([True, True, False]),
        val_mask=torch.tensor([False, False, True]),
        test_mask=torch.tensor([False, False, False]),
        feature_dim=2,
        official_evaluation=False,
    )

    assert torch.equal(bundle.node_features, node_features)
    assert bundle.node_features.shape[0] == 5


def test_synthetic_stream_uses_the_temporal_data_contract() -> None:
    bundle = build_synthetic_temporal_graph(
        num_nodes=12,
        num_events=20,
        feature_dim=4,
        edge_feature_dim=2,
        seed=11,
    )

    assert isinstance(bundle.data, TemporalData)
    assert bundle.data.t.dtype == torch.long
    assert bundle.data.msg.shape == (20, 2)
    assert bundle.train_data.num_events == 14
    assert bundle.val_data.num_events == 3
    assert bundle.test_data.num_events == 3
    assert bundle.official_evaluation is False
