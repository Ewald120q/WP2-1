from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint

    raise TypeError("Checkpoint does not contain a recognizable state dictionary.")


def _strip_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def load_expert_checkpoint(
    model: nn.Module,
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> None:
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = dict(extract_state_dict(checkpoint))
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = _strip_prefix(state_dict, "module.")
    model.load_state_dict(state_dict, strict=strict)


def extract_rejector_state_dict(
    checkpoint: Any,
    *,
    joint_component: str | None = None,
) -> dict[str, torch.Tensor]:
    state_dict = dict(extract_state_dict(checkpoint))

    if joint_component:
        component_state = _strip_prefix(state_dict, f"{joint_component}.")
        if component_state:
            return component_state

    # Historical EmbeddingRejector checkpoints store a Sequential consisting
    # of SmallToEmbedding at index 0 and the actual rejector head at index 1.
    historical_prefixes = (
        "1.",
        "embedding_processing.",
        "model.1.",
        "module.1.",
    )
    for prefix in historical_prefixes:
        candidate = _strip_prefix(state_dict, prefix)
        if candidate:
            return candidate

    if state_dict and all(key.startswith("module.") for key in state_dict):
        return _strip_prefix(state_dict, "module.")

    return state_dict


def load_rejector_checkpoint(
    rejector: nn.Module,
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    joint_component: str | None = None,
) -> None:
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = extract_rejector_state_dict(
        checkpoint,
        joint_component=joint_component,
    )
    rejector.load_state_dict(state_dict, strict=strict)


def save_joint_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
    config: Mapping[str, Any],
    scheduler: Any = None,
) -> None:
    path = os.fspath(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": dict(metrics),
        "config": dict(config),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)
