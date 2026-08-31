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


def _set_experts_trainable(model: JointCascadeMoE, trainable: bool) -> None:
    for expert in (model.f_small, model.f_mid, model.f_large):
        expert.requires_grad_(trainable)
        if not trainable:
            expert.eval()
    model.experts_frozen = not trainable


def _set_frozen_experts_eval(model: JointCascadeMoE) -> None:
    if getattr(model, "experts_frozen", False):
        model.f_small.eval()
        model.f_mid.eval()
        model.f_large.eval()


def _max_correct_under_budget(expert_correct: torch.Tensor, expert_counts: tuple[int, int, int]) -> int:
    # Für die budgeted upper bound reicht ein simples OR nicht mehr.
    # Wir müssen wissen, wie viele Samples man maximal richtig routen könnte, wenn die 70/21/9-Budgets fix sind.
    pattern_ids = (
        expert_correct[:, 0].to(torch.long) * 4
        + expert_correct[:, 1].to(torch.long) * 2
        + expert_correct[:, 2].to(torch.long)
    )
    # Es gibt bei drei Experten nur 8 Muster: keiner richtig, nur large richtig, mid+large richtig, usw.
    # Statt jedes Sample einzeln zu optimieren, zählen wir nur diese Muster. Das macht das Problem winzig.
    pattern_counts = torch.bincount(pattern_ids.detach().cpu(), minlength=8).tolist()
    source = 0
    pattern_offset = 1
    expert_offset = 9
    sink = 12
    capacities = [[0 for _ in range(13)] for _ in range(13)]

    # Pattern -> Experte ist nur erlaubt, wenn der Experte Samples mit diesem Muster richtig klassifiziert.
    # Experte -> Sink bekommt danach die echte Budget-Kapazität aus dem TopK-Split.
    for pattern, count in enumerate(pattern_counts):
        capacities[source][pattern_offset + pattern] = int(count)
        for expert_index in range(3):
            if pattern & (1 << (2 - expert_index)):
                capacities[pattern_offset + pattern][expert_offset + expert_index] = int(count)
    for expert_index, count in enumerate(expert_counts):
        capacities[expert_offset + expert_index][sink] = int(count)

    max_flow = 0
    while True:
        # Kleiner Edmonds-Karp auf 13 Knoten. Der gefundene Flow ist direkt:
        # "so viele Samples können unter Budget einem richtigen Experten gegeben werden".
        parent = [-1] * 13
        parent[source] = source
        queue = [source]
        for node in queue:
            for next_node, capacity in enumerate(capacities[node]):
                if capacity > 0 and parent[next_node] < 0:
                    parent[next_node] = node
                    queue.append(next_node)
                    if next_node == sink:
                        break
            if parent[sink] >= 0:
                break
        if parent[sink] < 0:
            return max_flow

        path_flow = 10**9
        node = sink
        while node != source:
            path_flow = min(path_flow, capacities[parent[node]][node])
            node = parent[node]
        node = sink
        while node != source:
            previous = parent[node]
            capacities[previous][node] -= path_flow
            capacities[node][previous] += path_flow
            node = previous
        max_flow += path_flow


def _run_epoch(
    model: JointCascadeMoE,
    loader: DataLoader,
    loss_fn: CascadeMoELoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float | None = None,
    description: str | None = None,
    route_mode: str = "topk",
    topk_noise_std: float = 0.0,
    expert_aux_loss_weight: float = 0.0,
    budget_loss_weight: float = 0.0,
    routing_loss_weight: float = 0.0,
    only_aux_warmup: bool = False,
    aux_loss_mode: str = "warmup_only",
    collect_expert_metrics: bool = False,
    collect_routed_expert_metrics: bool = False,
    collect_upper_bound_metrics: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    _set_frozen_experts_eval(model)
    loss_fn.train(training)

    totals: dict[str, float] = {}
    correct = 0
    expert_correct = torch.zeros(3, dtype=torch.long)
    expert_loss_sums = torch.zeros(3, dtype=torch.float64)
    routed_expert_correct = torch.zeros(3, dtype=torch.long)
    routed_expert_counts = torch.zeros(3, dtype=torch.long)
    upper_bound_correct = 0
    budgeted_upper_bound_correct = 0
    sample_count = 0
    context = nullcontext() if training else torch.no_grad()

    with context:
        for batch in tqdm(loader, desc=description, leave=False):
            targets = batch["label"].to(device, non_blocking=True)
            batch_size = int(targets.shape[0])

            if training:
                optimizer.zero_grad(set_to_none=True)

            if route_mode == "soft":
                outputs = model.forward_soft_aux(batch)
            elif route_mode == "topk":
                outputs = model.forward_topk_aux(batch, topk_noise_std=topk_noise_std if training else 0.0)
            else:
                raise ValueError(f"Unsupported route mode: {route_mode}")
            loss_values = loss_fn(outputs, targets, expert_aux_loss_weight=expert_aux_loss_weight if training else 0.0, budget_loss_weight=budget_loss_weight if training else 0.0, routing_loss_weight=routing_loss_weight if training else 0.0, only_aux_warmup=only_aux_warmup if training else False, aux_loss_mode=aux_loss_mode)

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
            expert_logits = outputs["expert_logits"]
            if collect_expert_metrics:
                for expert_index in range(3):
                    logits = expert_logits[:, expert_index, :]
                    expert_loss_sums[expert_index] += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").detach().cpu()
                    expert_correct[expert_index] += int((logits.argmax(dim=1) == targets).sum().item())
            if collect_routed_expert_metrics:
                selected_expert = outputs["selected_expert"].detach()
                selected_correct = predictions == targets
                for expert_index in range(3):
                    expert_mask = selected_expert == expert_index
                    routed_expert_counts[expert_index] += int(expert_mask.sum().item())
                    routed_expert_correct[expert_index] += int((selected_correct & expert_mask).sum().item())
            if collect_upper_bound_metrics:
                expert_predictions = expert_logits.argmax(dim=2)
                expert_correct_matrix = expert_predictions == targets.unsqueeze(1)
                upper_bound_correct += int(expert_correct_matrix.any(dim=1).sum().item())
                budgeted_upper_bound_correct += _max_correct_under_budget(expert_correct_matrix, model._expert_counts_for_batch(batch_size))
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
    if collect_expert_metrics:
        metrics["expert_small_loss"] = float(expert_loss_sums[0]) / sample_count
        metrics["expert_mid_loss"] = float(expert_loss_sums[1]) / sample_count
        metrics["expert_large_loss"] = float(expert_loss_sums[2]) / sample_count
        metrics["expert_small_accuracy"] = float(expert_correct[0]) / sample_count
        metrics["expert_mid_accuracy"] = float(expert_correct[1]) / sample_count
        metrics["expert_large_accuracy"] = float(expert_correct[2]) / sample_count
    if collect_routed_expert_metrics:
        metrics["expert_small_accuracy"] = float(routed_expert_correct[0]) / max(int(routed_expert_counts[0]), 1)
        metrics["expert_mid_accuracy"] = float(routed_expert_correct[1]) / max(int(routed_expert_counts[1]), 1)
        metrics["expert_large_accuracy"] = float(routed_expert_correct[2]) / max(int(routed_expert_counts[2]), 1)
        metrics["expert_small_samples"] = float(routed_expert_counts[0])
        metrics["expert_mid_samples"] = float(routed_expert_counts[1])
        metrics["expert_large_samples"] = float(routed_expert_counts[2])
    if collect_upper_bound_metrics:
        metrics["upper_bound"] = upper_bound_correct / sample_count
        metrics["budgeted_upper_bound"] = budgeted_upper_bound_correct / sample_count
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
    topk_noise_std: float = 0.0,
    expert_aux_loss_weight: float = 0.0,
    budget_loss_weight: float = 0.0,
    routing_loss_weight: float = 0.0,
    only_aux_warmup: bool = False,
    aux_loss_mode: str = "warmup_only",
) -> dict[str, float]:
    return _run_epoch(
        model,
        loader,
        loss_fn,
        device,
        optimizer=optimizer,
        gradient_clip_norm=gradient_clip_norm,
        description=description,
        route_mode="topk",
        topk_noise_std=topk_noise_std,
        expert_aux_loss_weight=expert_aux_loss_weight,
        budget_loss_weight=budget_loss_weight,
        routing_loss_weight=routing_loss_weight,
        only_aux_warmup=only_aux_warmup,
        aux_loss_mode=aux_loss_mode,
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
        route_mode="soft",
        collect_expert_metrics=True,
    )


def evaluate_topk(
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
        route_mode="topk",
        collect_routed_expert_metrics=True,
        collect_upper_bound_metrics=True,
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


def _write_metrics(writer: Any, split: str, metrics: Mapping[str, float], epoch: int) -> None:
    if writer is None:
        return
    for name, value in metrics.items():
        if name != "samples":
            writer.add_scalar(f"{split}/{name}", value, epoch)


def _split_expert_metrics(metrics: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    ordinary_metrics: dict[str, float] = {}
    expert_metrics: dict[str, float] = {}
    for name, value in metrics.items():
        if name.startswith("expert_"):
            expert_metrics[name.removeprefix("expert_")] = value
        else:
            ordinary_metrics[name] = value
    if "samples" in metrics:
        expert_metrics["samples"] = metrics["samples"]
    return ordinary_metrics, expert_metrics


def _selection_value(
    metric_name: str,
    metric_groups: Mapping[str, Mapping[str, float]],
) -> float:
    if "/" in metric_name:
        group_name, value_name = metric_name.split("/", 1)
    else:
        group_name, value_name = "val_soft", metric_name

    if group_name not in metric_groups:
        raise KeyError(
            f"Unknown selection metric group {group_name!r}. "
            f"Available groups: {sorted(metric_groups)}"
        )

    metrics = metric_groups[group_name]
    if value_name not in metrics:
        raise KeyError(
            f"Unknown selection metric {metric_name!r}. "
            f"Available metrics for {group_name!r}: {sorted(metrics)}"
        )

    return metrics[value_name]


def _is_better(current: float, best: float, mode: str) -> bool:
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError("selection_mode must be either 'min' or 'max'.")


def _annealed_topk_noise_std(base_std: float, noise_start_epoch: int, noise_epochs: int, epoch: int) -> float:
    if base_std <= 0.0 or noise_epochs <= 0 or epoch < noise_start_epoch:
        return 0.0
    noise_epoch = epoch - noise_start_epoch
    if noise_epoch >= noise_epochs:
        return 0.0
    return base_std * (1.0 - noise_epoch / noise_epochs)


def _expert_aux_loss_weight(base_weight: float, aux_epochs: int, epoch: int) -> float:
    if base_weight <= 0.0 or aux_epochs <= 0 or epoch >= aux_epochs:
        return 0.0
    return base_weight


def _upper_bound_metric_key(metric_name: str) -> str:
    metric_name = metric_name.strip()
    aliases = {
        "val_upper_bound": "upper_bound",
        "upper_bound": "upper_bound",
        "val_budgeted_upper_bound": "budgeted_upper_bound",
        "budgeted_upper_bound": "budgeted_upper_bound",
    }
    if metric_name not in aliases:
        raise ValueError("freeze_expert_metric must be 'val_upper_bound' or 'val_budgeted_upper_bound'.")
    return aliases[metric_name]


def _upper_bound_metric_value(metric_name: str, topk_metrics: Mapping[str, float]) -> float:
    return topk_metrics[_upper_bound_metric_key(metric_name)]


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
    selection_mode: str = "min",
    topk_noise_std: float = 0.0,
    topk_noise_start_epoch: int = 0,
    topk_noise_epochs: int = 0,
    expert_aux_loss_weight: float = 0.0,
    expert_aux_loss_epochs: int = 0,
    budget_loss_weight: float = 0.0,
    routing_loss_weight: float = 0.0,
    only_aux_warmup: bool = False,
    aux_loss_mode: str = "warmup_only",
    freeze_experts_on_upper_bound: bool = False,
    freeze_expert_patience: int = 5,
    freeze_expert_metric: str = "val_upper_bound",
    threshold_r1: float = 0.5,
    threshold_r2: float = 0.5,
) -> tuple[list[dict[str, Any]], str]:
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if patience is not None and patience <= 0:
        raise ValueError("patience must be greater than zero when provided.")
    if freeze_experts_on_upper_bound and freeze_expert_patience <= 0:
        raise ValueError("freeze_expert_patience must be greater than zero when enabled.")
    if freeze_experts_on_upper_bound:
        _upper_bound_metric_key(freeze_expert_metric)

    checkpoint_dir = os.fspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "joint_cascade_moe_best.pth")
    best_upper_bound_path = os.path.join(checkpoint_dir, "joint_cascade_moe_best_upper_bound.pth")

    history: list[dict[str, Any]] = []
    selection_mode = selection_mode.lower()
    if selection_mode not in {"min", "max"}:
        raise ValueError("selection_mode must be either 'min' or 'max'.")
    aux_loss_mode = aux_loss_mode.lower()
    if aux_loss_mode not in {"warmup_only", "additive"}:
        raise ValueError("aux_loss_mode must be either 'warmup_only' or 'additive'.")
    best_metric = float("inf") if selection_mode == "min" else -float("inf")
    stale_epochs = 0
    best_upper_bound_metric = -float("inf")
    upper_bound_stale_epochs = 0
    experts_frozen = bool(getattr(model, "experts_frozen", False))
    freeze_triggered_epoch: int | None = None

    for epoch in range(0, epochs):
        current_topk_noise_std = _annealed_topk_noise_std(float(topk_noise_std), int(topk_noise_start_epoch), int(topk_noise_epochs), epoch)
        current_expert_aux_loss_weight = _expert_aux_loss_weight(float(expert_aux_loss_weight), int(expert_aux_loss_epochs), epoch)
        current_only_aux_warmup = bool(only_aux_warmup) and current_expert_aux_loss_weight > 0.0
        current_budget_loss_weight = 0.0 if current_only_aux_warmup else float(budget_loss_weight)
        current_routing_loss_weight = 0.0 if current_only_aux_warmup else float(routing_loss_weight)
        current_effective_expert_aux_loss_weight = current_expert_aux_loss_weight if current_only_aux_warmup or aux_loss_mode == "additive" else 0.0
        train_metrics = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            gradient_clip_norm=gradient_clip_norm,
            description=f"train {epoch}/{epochs-1}",
            topk_noise_std=current_topk_noise_std,
            expert_aux_loss_weight=current_effective_expert_aux_loss_weight,
            budget_loss_weight=current_budget_loss_weight,
            routing_loss_weight=current_routing_loss_weight,
            only_aux_warmup=current_only_aux_warmup,
            aux_loss_mode=aux_loss_mode,
        )
        val_metrics = evaluate_soft(
            model,
            val_loader,
            loss_fn,
            device,
            description=f"val-soft {epoch}/{epochs-1}",
        )
        val_metrics, val_expert_metrics = _split_expert_metrics(val_metrics)
        topk_metrics = evaluate_topk(
            model,
            val_loader,
            loss_fn,
            device,
            description=f"val-topk {epoch}/{epochs-1}",
        )
        hard_metrics = evaluate_hard(
            model,
            val_loader,
            device,
            threshold_r1=threshold_r1,
            threshold_r2=threshold_r2,
            description=f"val-hard {epoch}/{epochs-1}",
        )

        _write_metrics(writer, "train", train_metrics, epoch)
        _write_metrics(writer, "val_soft", val_metrics, epoch)
        _write_metrics(writer, "val_expert", val_expert_metrics, epoch)
        _write_metrics(writer, "val_topk", topk_metrics, epoch)
        _write_metrics(writer, "val_hard", hard_metrics, epoch)
        if writer is not None:
            writer.add_scalar("val_upper_bound", topk_metrics["upper_bound"], epoch)
            writer.add_scalar("val_budgeted_upper_bound", topk_metrics["budgeted_upper_bound"], epoch)

        metric_groups = {
            "train": train_metrics,
            "val_soft": val_metrics,
            "val_expert": val_expert_metrics,
            "val_topk": topk_metrics,
            "val_hard": hard_metrics,
            "val": {
                "upper_bound": topk_metrics["upper_bound"],
                "budgeted_upper_bound": topk_metrics["budgeted_upper_bound"],
            },
        }
        current_metric = _selection_value(selection_metric, metric_groups)

        if scheduler is not None:
            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step(current_metric)
            else:
                scheduler.step()
        if writer is not None:
            writer.add_scalar("train/topk_noise_std", current_topk_noise_std, epoch)
            writer.add_scalar("train/expert_aux_loss_weight", current_effective_expert_aux_loss_weight, epoch)
            writer.add_scalar("train/budget_loss_weight", current_budget_loss_weight, epoch)
            writer.add_scalar("train/routing_loss_weight", current_routing_loss_weight, epoch)
            writer.add_scalar("train/only_aux_warmup", float(current_only_aux_warmup), epoch)
            writer.add_scalar("train/aux_loss_additive", float(aux_loss_mode == "additive"), epoch)
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
            "val_expert": val_expert_metrics,
            "val_topk": topk_metrics,
            "val_hard": hard_metrics,
            "val_upper_bound": topk_metrics["upper_bound"],
            "val_budgeted_upper_bound": topk_metrics["budgeted_upper_bound"],
            "experts_frozen_after_epoch": 0.0,
            "freeze_triggered_epoch": freeze_triggered_epoch,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d}: "
            f"train topk loss={train_metrics['total']:.5f}, "
            f"train topk acc={train_metrics['accuracy']:.4f}, "
            f"val soft loss={val_metrics['total']:.5f}, "
            f"val soft acc={val_metrics['accuracy']:.4f}, "
            f"val topk loss={topk_metrics['total']:.5f}, "
            f"val topk acc={topk_metrics['accuracy']:.4f}, "
            f"val upper={topk_metrics['upper_bound']:.4f}, "
            f"val budget upper={topk_metrics['budgeted_upper_bound']:.4f}, "
            f"val hard acc={hard_metrics['accuracy']:.4f}, "
            "val hard usage="
            f"({hard_metrics['usage_small']:.3f}, "
            f"{hard_metrics['usage_mid']:.3f}, "
            f"{hard_metrics['usage_large']:.3f})"
        )

        checkpoint_metrics = {
            f"val_soft/{key}": value
            for key, value in val_metrics.items()
        } | {
            f"val_expert/{key}": value
            for key, value in val_expert_metrics.items()
        } | {
            f"val_topk/{key}": value
            for key, value in topk_metrics.items()
        } | {
            "val_upper_bound": topk_metrics["upper_bound"],
            "val_budgeted_upper_bound": topk_metrics["budgeted_upper_bound"],
        } | {
            f"val_hard/{key}": value
            for key, value in hard_metrics.items()
        }

        should_stop = False
        if _is_better(current_metric, best_metric, selection_mode):
            best_metric = current_metric
            stale_epochs = 0
            save_joint_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=checkpoint_metrics,
                config=config,
            )
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= patience:
                print(f"Early stopping after epoch {epoch}.")
                should_stop = True

        if freeze_experts_on_upper_bound and not current_only_aux_warmup and not experts_frozen:
            current_upper_bound_metric = _upper_bound_metric_value(freeze_expert_metric, topk_metrics)
            if current_upper_bound_metric > best_upper_bound_metric:
                best_upper_bound_metric = current_upper_bound_metric
                upper_bound_stale_epochs = 0
                save_joint_checkpoint(
                    best_upper_bound_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=checkpoint_metrics,
                    config=config,
                )
            else:
                upper_bound_stale_epochs += 1

            if upper_bound_stale_epochs >= freeze_expert_patience:
                checkpoint = torch.load(best_upper_bound_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                _set_experts_trainable(model, False)
                optimizer.zero_grad(set_to_none=True)
                optimizer.state.clear()
                experts_frozen = True
                freeze_triggered_epoch = epoch
                print(f"Freezing experts after epoch {epoch}; restored {freeze_expert_metric} checkpoint from epoch {checkpoint['epoch']}.")

        epoch_record["experts_frozen_after_epoch"] = float(experts_frozen)
        epoch_record["freeze_triggered_epoch"] = freeze_triggered_epoch
        if writer is not None:
            writer.add_scalar("train/experts_frozen", float(experts_frozen), epoch)
            if freeze_triggered_epoch is not None:
                writer.add_scalar("train/freeze_triggered_epoch", float(freeze_triggered_epoch), epoch)
        if should_stop:
            break

    return history, best_path
