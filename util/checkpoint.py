from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn


CHECKPOINT_FORMAT_VERSION = 1


def save_model_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_name: str,
    model_config: Mapping[str, Any],
    args: Mapping[str, Any],
) -> None:
    """保存模型参数和可验证配置，不包含模型的非持久化运行时状态。"""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_name": model_name,
            "model": state,
            "model_config": dict(model_config),
            "args": dict(args),
        },
        checkpoint_path,
    )


def load_model_checkpoint(
    path: str | Path,
    expected_model_name: str,
    expected_model_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """读取检查点并验证版本、模型类型和指定的配置字段。"""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Model checkpoint must contain a dictionary payload.")
    required = {"format_version", "model_name", "model", "model_config", "args"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            "Model checkpoint is missing fields: " + ", ".join(missing)
        )
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format version: "
            f"{payload['format_version']}; expected {CHECKPOINT_FORMAT_VERSION}."
        )
    if payload["model_name"] != expected_model_name:
        raise ValueError(
            f"Checkpoint model is {payload['model_name']!r}; "
            f"expected {expected_model_name!r}."
        )
    if not isinstance(payload["model"], dict) or not isinstance(
        payload["model_config"], dict
    ):
        raise ValueError("Checkpoint model and model_config fields must be dictionaries.")

    if expected_model_config is not None:
        for name, expected_value in expected_model_config.items():
            actual_value = payload["model_config"].get(name)
            if actual_value != expected_value:
                raise ValueError(
                    f"Checkpoint config mismatch for {name!r}: "
                    f"got {actual_value!r}, expected {expected_value!r}."
                )
    return payload
