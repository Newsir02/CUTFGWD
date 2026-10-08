from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import TemporalData

from dataset.temporal import build_temporal_graph_bundle
from util.tgb_evaluation import evaluate_tgb


class FakeNegativeSampler:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query_batch(self, src, dst, t, split_mode, **kwargs):
        del src, t, kwargs
        self.queries.append(split_mode)
        return [[3] if int(positive) == 4 else [4] for positive in dst]


class FakeOfficialDataset:
    def __init__(self) -> None:
        self.eval_metric = "mrr"
        self.negative_sampler = FakeNegativeSampler()
        self.loaded_split = ""

    def load_val_ns(self) -> None:
        self.loaded_split = "val"

    def load_test_ns(self) -> None:
        self.loaded_split = "test"


class RecordingTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def score_candidates(self, batch, return_relations=False):
        del return_relations
        self.calls.append("score")
        return {"logits": -batch["candidates"].float()}

    def update_state(self, batch) -> None:
        assert batch["positive_dst"].numel() == 2
        self.calls.append("update")


def make_bundle(official: bool = True):
    backend = FakeOfficialDataset()
    data = TemporalData(
        src=torch.tensor([0, 1, 0, 1]),
        dst=torch.tensor([3, 4, 4, 3]),
        t=torch.tensor([10, 20, 30, 40]),
        msg=torch.zeros(4, 2),
        y=torch.ones(4),
    )
    bundle = build_temporal_graph_bundle(
        dataset_name="tgbl-wiki",
        data=data,
        node_features=None,
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.tensor([False, False, True, True]),
        test_mask=torch.tensor([False, False, True, True]),
        feature_dim=3,
        official_evaluation=official,
        tgb_dataset=backend,
    )
    return bundle, backend


def test_official_evaluation_queries_negatives_then_updates_teacher() -> None:
    bundle, backend = make_bundle()
    model = RecordingTeacher()

    metrics = evaluate_tgb(
        model=model,
        bundle=bundle,
        split="val",
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert backend.loaded_split == "val"
    assert backend.negative_sampler.queries == ["val"]
    assert model.calls == ["score", "score", "update"]
    assert metrics["metric"] == "mrr"
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert metrics["official"] is True


def test_official_test_split_loads_test_negatives() -> None:
    bundle, backend = make_bundle()

    evaluate_tgb(
        model=RecordingTeacher(),
        bundle=bundle,
        split="test",
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert backend.loaded_split == "test"
    assert backend.negative_sampler.queries == ["test"]


def test_quick_bundle_rejects_official_evaluation() -> None:
    bundle, _ = make_bundle(official=False)

    try:
        evaluate_tgb(
            model=RecordingTeacher(),
            bundle=bundle,
            split="test",
            batch_size=2,
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "official" in str(exc).lower()
    else:
        raise AssertionError("Expected quick data to reject official evaluation.")
