from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoints import save_joint_checkpoint
from .joint_ensemble import JointCascadeMoE
from .loss import CascadeMoELoss


def _run_epoch(
    model: JointCascadeMoE,
    loader: DataLoader,
    loss_fn: CascadeMoELoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float | None = None,
    description: str | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_fn.train(training)

    totals: dict[str, float] = {}
    correct = 0
    sample_count = 0
    context = nullcontext() if training else torch.no_grad()

    with context:
        for batch in tqdm(loader, desc=description, leave=False):
            targets = batch["label"].to(device, non_blocking=True)
            batch_size = int(targets.shape[0])

            if training:
                optimizer.zero_grad(set_to_none=True)

            #_run_epoch wird nur verwendet, wenn evaluate_soft oder train aufgerufen wird.
            # somit okay, dass hier immer soft routing benutzt wird
            outputs = model.forward_soft_aux(batch)
            loss_values = loss_fn(outputs, targets)

            if training:
                loss_values["total"].backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                optimizer.step()

            predictions = outputs["log_probs"].argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            sample_count += batch_size

            for name, value in loss_values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size

    if sample_count == 0:
        raise ValueError("The data loader did not yield any samples.")

    metrics = {
        name: value / sample_count
        for name, value in totals.items()
    }
    metrics["accuracy"] = correct / sample_count
    metrics["samples"] = float(sample_count)
    return metrics

def train_epoch(
    model: JointCascadeMoE,
    loader: DataLoader,
    loss_fn: CascadeMoELoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip_norm: float | None = None,
    description: str | None = None,
) -> dict[str, float]:
    return _run_epoch(
        model,
        loader,
        loss_fn,
        device,
        optimizer=optimizer,
        gradient_clip_norm=gradient_clip_norm,
        description=description,
    )


def evaluate_soft(
    model: JointCascadeMoE,
    loader: DataLoader,
    loss_fn: CascadeMoELoss,
    device: torch.device,
    *,
    description: str | None = None,
) -> dict[str, float]:
    return _run_epoch(
        model,
        loader,
        loss_fn,
        device,
        optimizer=None,
        description=description,
    )


@torch.no_grad()
def evaluate_hard(
    model: JointCascadeMoE,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold_r1: float = 0.5,
    threshold_r2: float = 0.5,
    description: str | None = None,
) -> dict[str, float]:
    model.eval()
    correct = 0
    sample_count = 0
    expert_counts = torch.zeros(3, dtype=torch.long)

    for batch in tqdm(loader, desc=description, leave=False):
        targets = batch["label"].to(device, non_blocking=True)
        outputs = model.forward_hard_aux(
            batch,
            threshold_r1=threshold_r1,
            threshold_r2=threshold_r2,
        )

        predictions = outputs["log_probs"].argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        sample_count += int(targets.shape[0])
        expert_counts += torch.bincount(
            outputs["selected_expert"].detach().cpu(),
            minlength=3,
        )

    if sample_count == 0:
        raise ValueError("The data loader did not yield any samples.")

    return {
        "accuracy": correct / sample_count,
        "usage_small": float(expert_counts[0]) / sample_count,
        "usage_mid": float(expert_counts[1]) / sample_count,
        "usage_large": float(expert_counts[2]) / sample_count,
        "samples": float(sample_count),
    }


def _write_metrics(
    writer: Any,
    split: str,
    metrics: Mapping[str, float],
    epoch: int,
) -> None:
    if writer is None:
        return
    for name, value in metrics.items():
        if name != "samples":
            writer.add_scalar(f"{split}/{name}", value, epoch)


def fit(
    model: JointCascadeMoE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: CascadeMoELoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epochs: int,
    checkpoint_dir: str | os.PathLike[str],
    config: Mapping[str, Any],
    scheduler: Any = None,
    writer: Any = None,
    patience: int | None = None,
    gradient_clip_norm: float | None = None,
    selection_metric: str = "total",
    threshold_r1: float = 0.5,
    threshold_r2: float = 0.5,
) -> tuple[list[dict[str, Any]], str]:
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if patience is not None and patience <= 0:
        raise ValueError("patience must be greater than zero when provided.")

    checkpoint_dir = os.fspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "joint_cascade_moe_best.pth")

    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    stale_epochs = 0

    for epoch in range(0, epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            gradient_clip_norm=gradient_clip_norm,
            description=f"train {epoch}/{epochs-1}",
        )
        val_metrics = evaluate_soft(
            model,
            val_loader,
            loss_fn,
            device,
            description=f"val-soft {epoch}/{epochs-1}",
        )
        hard_metrics = evaluate_hard(
            model,
            val_loader,
            device,
            threshold_r1=threshold_r1,
            threshold_r2=threshold_r2,
            description=f"val-hard {epoch}/{epochs-1}",
        )

        if selection_metric not in val_metrics:
            raise KeyError(
                f"Unknown selection metric {selection_metric!r}. "
                f"Available metrics: {sorted(val_metrics)}"
            )

        _write_metrics(writer, "train", train_metrics, epoch)
        _write_metrics(writer, "val_soft", val_metrics, epoch)
        _write_metrics(writer, "val_hard", hard_metrics, epoch)

        if scheduler is not None:
            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step(val_metrics[selection_metric])
            else:
                scheduler.step()
        if writer is not None:
            for group_index, group in enumerate(optimizer.param_groups):
                writer.add_scalar(
                    f"learning_rate/group_{group_index}",
                    group["lr"],
                    epoch,
                )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val_soft": val_metrics,
            "val_hard": hard_metrics,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d}: "
            f"train soft loss={train_metrics['total']:.5f}, "
            f"train soft acc={train_metrics['accuracy']:.4f}, "
            f"val soft loss={val_metrics['total']:.5f}, "
            f"val soft acc={val_metrics['accuracy']:.4f}, "
            f"val hard acc={hard_metrics['accuracy']:.4f}, "
            "val hard usage="
            f"({hard_metrics['usage_small']:.3f}, "
            f"{hard_metrics['usage_mid']:.3f}, "
            f"{hard_metrics['usage_large']:.3f})"
        )

        current_metric = val_metrics[selection_metric]
        if current_metric < best_metric:
            best_metric = current_metric
            stale_epochs = 0
            save_joint_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    f"val_soft/{key}": value
                    for key, value in val_metrics.items()
                }
                | {
                    f"val_hard/{key}": value
                    for key, value in hard_metrics.items()
                },
                config=config,
            )
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    return history, best_path
