from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


if __package__ in {None, ""}:
    # Permit both `python -m moe.train_joint_ensemble` and direct execution.
    _PARENT = Path(__file__).resolve().parents[1]
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))

    from moe.checkpoints import load_expert_checkpoint, load_rejector_checkpoint
    from moe.joint_ensemble import JointCascadeMoE
    from moe.loss import CascadeMoELoss
    from moe.train_helper import evaluate_hard, evaluate_soft, evaluate_topk, fit
else:
    from .checkpoints import load_expert_checkpoint, load_rejector_checkpoint
    from .joint_ensemble import JointCascadeMoE
    from .loss import CascadeMoELoss
    from .train_helper import evaluate_hard, evaluate_soft, evaluate_topk, fit

from DMTimeShardDataset import DMTimeShardDataset
from embedding_processing_models import build_embedding_processing
from training_models import models_htable
from training_utils import label_encoding


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _build_expert(
    spec: dict[str, Any],
    *,
    device: torch.device,
) -> torch.nn.Module:
    model_name = spec["model_name"]
    if model_name not in models_htable:
        raise KeyError(f"Unknown expert model: {model_name}")

    model = models_htable[model_name](
        spec.get("resolution", 256),
        mode=spec["mode"],
        dropout=spec.get("dropout", False),
        device=device,
    ).to(device)

    checkpoint = spec.get("checkpoint")
    if checkpoint:
        load_expert_checkpoint(
            model,
            checkpoint,
            map_location=device,
            strict=spec.get("strict_checkpoint", True),
        )
    return model


def _build_rejector(
    spec: dict[str, Any],
    *,
    in_channels: int,
    device: torch.device,
    joint_component: str,
) -> tuple[torch.nn.Module, str]:
    rejector, default_feature_source = build_embedding_processing(
        spec.get("model_name", "conv_mlp"),
        in_channels=in_channels,
        cnn_channels=spec.get("cnn_channels", 64),
        extra_conv=spec.get("extra_conv", False),
        pool_size=spec.get("pool_size", 7),
        hidden_dim=spec.get("hidden_dim", 64),
        dropout=spec.get("dropout", 0.0),
        pool_type=spec.get("pool_type", "max"),
    )
    rejector = rejector.to(device)

    checkpoint = spec.get("checkpoint")
    if checkpoint:
        load_rejector_checkpoint(
            rejector,
            checkpoint,
            map_location=device,
            strict=spec.get("strict_checkpoint", True),
            joint_component=joint_component,
        )

    return rejector, spec.get("feature_source", default_feature_source)


def build_joint_model(config: dict[str, Any],device: torch.device) -> JointCascadeMoE:
    model_config = config["model"]
    f_small = _build_expert(model_config["f_small"], device=device)
    f_mid = _build_expert(model_config["f_mid"], device=device)
    f_large = _build_expert(model_config["f_large"], device=device)

    r1, source_r1 = _build_rejector(
        model_config["r1"],
        in_channels=f_small.out_features,
        device=device,
        joint_component="r1",
    )
    r2, source_r2 = _build_rejector(
        model_config["r2"],
        in_channels=f_mid.out_features,
        device=device,
        joint_component="r2",
    )

    return JointCascadeMoE(
        f_small=f_small,
        f_mid=f_mid,
        f_large=f_large,
        r1=r1,
        r2=r2,
        r1_feature_source=source_r1,
        r2_feature_source=source_r2,
        temperature=model_config.get("temperature", 1.0),
        expert_fractions=model_config.get("expert_fractions"),
    ).to(device)


def _make_loader(
    dataset: DMTimeShardDataset,
    loader_config: dict[str, Any],
    *,
    shuffle: bool,
) -> DataLoader:
    num_workers = int(loader_config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(loader_config.get("batch_size", 128)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(loader_config.get("pin_memory", True)),
        "persistent_workers": (
            bool(loader_config.get("persistent_workers", True))
            and num_workers > 0
        ),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(loader_config.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def build_loaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_config = config["dataset"]
    shard_config = {
        "output_dir": dataset_config["output_dir"],
        "prefix": dataset_config["prefix"],
    }

    datasets = {}
    for split in ("train", "val", "test"):
        dataset = DMTimeShardDataset(
            shard_config,
            use_freq_time=True,
            split=split,
        )
        dataset.labels = label_encoding(dataset.labels.astype(object))
        datasets[split] = dataset

    loader_config = config.get("loader", {})
    return (
        _make_loader(datasets["train"], loader_config, shuffle=True),
        _make_loader(datasets["val"], loader_config, shuffle=False),
        _make_loader(datasets["test"], loader_config, shuffle=False),
    )


def build_optimizer( model: JointCascadeMoE, config: dict[str, Any]) -> torch.optim.Optimizer:
    training_config = config["training"]
    expert_parameters = itertools.chain(
        model.f_small.parameters(),
        model.f_mid.parameters(),
        model.f_large.parameters(),
    )
    rejector_parameters = itertools.chain(
        model.r1.parameters(),
        model.r2.parameters(),
    )
    parameter_groups = [
        {
            "params": list(expert_parameters),
            "lr": training_config["expert_learning_rate"],
            "name": "experts",
        },
        {
            "params": list(rejector_parameters),
            "lr": training_config["rejector_learning_rate"],
            "name": "rejectors",
        },
    ]

    optimizer_name = training_config.get("optimizer", "adam").lower()
    optimizer_kwargs = {
        "weight_decay": training_config.get("weight_decay", 0.0),
    }
    if optimizer_name == "adam":
        return torch.optim.Adam(parameter_groups, **optimizer_kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_loss(config: dict[str, Any], device: torch.device) -> CascadeMoELoss:
    fractions = config.get("model", {}).get("expert_fractions", {"small": 0.7, "mid": 0.21, "large": 0.09})
    target_usage = torch.tensor([fractions["small"], fractions["mid"], fractions["large"]], dtype=torch.float32, device=device)
    return CascadeMoELoss(target_usage=target_usage).to(device)


def build_scheduler(optimizer: torch.optim.Optimizer,training_config: dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_name = training_config.get("scheduler")
    gamma = training_config.get("scheduler_gamma")
    if scheduler_name is None:
        scheduler_name = "exponential" if gamma is not None else "none"

    scheduler_name = scheduler_name.lower()
    if scheduler_name in {"none", "off", "disabled"}:
        return None
    if scheduler_name == "exponential":
        if gamma is None:
            raise ValueError("scheduler_gamma must be set for the exponential scheduler.")
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if scheduler_name in {"reduce_on_plateau", "reducelronplateau"}:
        scheduler_mode = training_config.get(
            "scheduler_mode",
            training_config.get("selection_mode", "min"),
        )
        scheduler_mode = str(scheduler_mode).lower()
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=training_config.get("scheduler_factor", 0.5),
            patience=int(training_config.get("scheduler_patience", 15)),
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Run one complete joint-training experiment."""
    device = _resolve_device(config.get("device", "auto"))
    print(f"Using device: {device}")

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_joint_model(config, device)
    train_loader, val_loader, test_loader = build_loaders(config)
    loss_fn = build_loss(config, device)
    optimizer = build_optimizer(model, config)

    training_config = config["training"]
    scheduler = build_scheduler(optimizer, training_config)

    output_dir = config.get("output_dir", "./moe_runs/default")
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "run_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, indent=2)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    writer.add_text("config", json.dumps(config, indent=2), 0)

    thresholds = config.get("inference", {})
    threshold_r1 = thresholds.get("threshold_r1", 0.5)
    threshold_r2 = thresholds.get("threshold_r2", 0.5)

    try:
        history, best_path = fit(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            device,
            epochs=int(training_config["epochs"]),
            checkpoint_dir=os.path.join(output_dir, "checkpoints"),
            config=config,
            scheduler=scheduler,
            writer=writer,
            patience=training_config.get("patience"),
            gradient_clip_norm=training_config.get("gradient_clip_norm"),
            selection_metric=training_config.get("selection_metric", "total"),
            selection_mode=training_config.get("selection_mode", "min"),
            topk_noise_std=training_config.get("topk_noise_std", 0.0),
            topk_noise_start_epoch=training_config.get("topk_noise_start_epoch", 0),
            topk_noise_epochs=training_config.get("topk_noise_epochs", 0),
            expert_aux_loss_weight=training_config.get("expert_aux_loss_weight", 0.0),
            expert_aux_loss_epochs=training_config.get("expert_aux_loss_epochs", 0),
            budget_loss_weight=training_config.get("budget_loss_weight", 0.0),
            routing_loss_weight=training_config.get("routing_loss_weight", 0.0),
            only_aux_warmup=training_config.get("only_aux_warmup", False),
            aux_loss_mode=training_config.get("aux_loss_mode", "warmup_only"),
            freeze_experts_on_upper_bound=training_config.get("freeze_experts_on_upper_bound", False),
            freeze_expert_patience=int(training_config.get("freeze_expert_patience", 5)),
            freeze_expert_metric=training_config.get("freeze_expert_metric", "val_upper_bound"),
            threshold_r1=threshold_r1,
            threshold_r2=threshold_r2,
        )
    finally:
        writer.close()

    with open(
        os.path.join(output_dir, "history.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(history, handle, indent=2)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_soft = evaluate_soft(model, test_loader, loss_fn, device, description="test-soft",)
    test_topk = evaluate_topk(model, test_loader, loss_fn, device, description="test-topk",)
    test_hard = evaluate_hard(model, test_loader, device, threshold_r1=threshold_r1, threshold_r2=threshold_r2, description="test-hard",)
    print("Test soft:", test_soft)
    print("Test topk:", test_topk)
    print("Test hard:", test_hard)

    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"soft": test_soft, "topk": test_topk, "hard": test_hard}, handle, indent=2)

    return {"best_checkpoint": best_path, "test_soft": test_soft, "test_topk": test_topk, "test_hard": test_hard}


def main() -> None:
    parser = argparse.ArgumentParser(description="Jointly train the complete rejection cascade.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    args = parser.parse_args()

    config = _load_json(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
