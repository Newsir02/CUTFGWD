from __future__ import annotations

import inspect
import os
from pathlib import Path

from .temporal import TemporalGraphBundle, build_temporal_graph_bundle


def _resolve_tgb_root_argument(root: str) -> str:
    """适配 py-tgb 把 root 拼接到包目录的路径语义。"""

    try:
        from tgb.utils.info import PROJ_DIR
    except ImportError as exc:
        raise ImportError(
            "Loading TGB data requires the py-tgb package."
        ) from exc

    requested_root = Path(root).expanduser()

    if not requested_root.is_absolute():
        requested_root = Path.cwd() / requested_root

    return os.path.relpath(
        requested_root.resolve(),
        Path(PROJ_DIR).resolve(),
    )


def _create_pyg_tgb_dataset(
    *,
    name: str,
    root: str,
    download: bool,
):
    try:
        from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
    except ImportError as exc:
        raise ImportError(
            "Loading real TGB datasets requires py-tgb and "
            "torch-geometric."
        ) from exc

    kwargs = {
        "name": name,
        "root": _resolve_tgb_root_argument(root),
    }

    # py-tgb 2.2.0 等新版本支持 download 参数，
    # py-tgb 2.0.0 不支持该参数。
    parameters = inspect.signature(
        PyGLinkPropPredDataset.__init__
    ).parameters

    if "download" in parameters:
        kwargs["download"] = download
    elif not download:
        raise RuntimeError(
            "The installed py-tgb version does not support "
            "download=False. Please install a newer py-tgb version "
            "or ensure the dataset already exists."
        )

    return PyGLinkPropPredDataset(**kwargs)


def build_tgb_temporal_graph(
    name: str = "tgbl-wiki",
    root: str = "datasets",
    feature_dim: int = 16,
    edge_feature_dim: int = 16,
    num_candidates: int = 16,
    history_size: int = 24,
    seed: int = 42,
    max_events: int = 0,
    download: bool = True,
) -> TemporalGraphBundle:
    """直接加载 TGB 的 PyG TemporalData，不复制或重映射事件。"""

    del edge_feature_dim, num_candidates, history_size, seed

    dataset_name = name[:-3] if name.endswith("-v2") else name

    dataset = _create_pyg_tgb_dataset(
        name=dataset_name,
        root=root,
        download=download,
    )

    return build_temporal_graph_bundle(
        dataset_name=dataset_name,
        data=dataset.get_TemporalData(),
        node_features=dataset.node_feat,
        train_mask=dataset.train_mask,
        val_mask=dataset.val_mask,
        test_mask=dataset.test_mask,
        feature_dim=feature_dim,
        official_evaluation=max_events <= 0,
        tgb_dataset=dataset,
        max_events=max_events,
    )