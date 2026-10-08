from __future__ import annotations

import torch
from torch_geometric.data import TemporalData

import dataset.tgb_temporal as tgb_temporal
from dataset.tgb_temporal import build_tgb_temporal_graph


class FakePyGLinkPropPredDataset:
    def __init__(self) -> None:
        self.train_mask = torch.tensor([True, True, True, False, False, False])
        self.val_mask = torch.tensor([False, False, False, True, True, False])
        self.test_mask = torch.tensor([False, False, False, False, False, True])
        self.node_feat = None
        self.eval_metric = "mrr"
        self.negative_sampler = object()
        self._data = TemporalData(
            src=torch.tensor([10, 10, 20, 30, 40, 50]),
            dst=torch.tensor([20, 30, 10, 40, 50, 10]),
            t=torch.tensor([100, 101, 102, 103, 104, 105]),
            msg=torch.arange(12, dtype=torch.float).reshape(6, 2),
            y=torch.ones(6),
        )

    def get_TemporalData(self) -> TemporalData:
        return self._data.clone()


def test_tgb_loader_uses_official_pyg_wrapper(monkeypatch) -> None:
    fake = FakePyGLinkPropPredDataset()
    calls: list[dict[str, object]] = []

    def create_dataset(**kwargs):
        calls.append(kwargs)
        return fake

    monkeypatch.setattr(tgb_temporal, "_create_pyg_tgb_dataset", create_dataset)

    bundle = build_tgb_temporal_graph(
        name="tgbl-wiki",
        root="datasets",
        feature_dim=4,
        download=False,
    )

    assert calls == [
        {
            "name": "tgbl-wiki",
            "root": "datasets",
            "download": False,
        }
    ]
    assert isinstance(bundle.data, TemporalData)
    assert bundle.tgb_dataset is fake
    assert bundle.official_evaluation is True
    assert bundle.data.t.dtype == torch.long
    assert torch.equal(bundle.data.src, fake._data.src)
    assert torch.equal(bundle.train_data.src, fake._data.src[fake.train_mask])
    assert torch.equal(bundle.val_data.src, fake._data.src[fake.val_mask])
    assert torch.equal(bundle.test_data.src, fake._data.src[fake.test_mask])
    assert bundle.node_features.shape == (51, 4)
    assert torch.count_nonzero(bundle.node_features) == 0


def test_tgb_quick_prefix_is_contiguous_and_not_official(monkeypatch) -> None:
    fake = FakePyGLinkPropPredDataset()
    monkeypatch.setattr(
        tgb_temporal,
        "_create_pyg_tgb_dataset",
        lambda **_: fake,
    )

    bundle = build_tgb_temporal_graph(
        name="tgbl-wiki",
        feature_dim=4,
        max_events=4,
        download=False,
    )

    assert bundle.official_evaluation is False
    assert bundle.data.num_events == 4
    assert torch.equal(bundle.data.event_index, torch.arange(4))
    assert (bundle.train_data.num_events, bundle.val_data.num_events) == (2, 1)
    assert bundle.test_data.num_events == 1
    selected = torch.cat(
        [
            bundle.train_data.event_index,
            bundle.val_data.event_index,
            bundle.test_data.event_index,
        ]
    )
    assert torch.equal(selected, torch.arange(4))


def test_tgb_loader_rejects_non_chronological_official_data(monkeypatch) -> None:
    fake = FakePyGLinkPropPredDataset()
    fake._data.t = torch.tensor([100, 102, 101, 103, 104, 105])
    monkeypatch.setattr(
        tgb_temporal,
        "_create_pyg_tgb_dataset",
        lambda **_: fake,
    )

    try:
        build_tgb_temporal_graph(download=False)
    except ValueError as exc:
        assert "chronological" in str(exc)
    else:
        raise AssertionError("Expected non-chronological TGB data to be rejected.")
