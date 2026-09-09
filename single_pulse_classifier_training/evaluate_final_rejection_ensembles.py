from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from DMTimeShardDataset import DMTimeShardDataset
from embedding_processing_models import build_embedding_processing
from moe.checkpoints import load_expert_checkpoint, load_rejector_checkpoint
from moe.train_joint_ensemble import build_joint_model
from training_models import models_htable
from training_utils import label_encoding


DATASET_PREFIX = "B0531+21_59000_48386"
REQUESTED_FRACTIONS = {"small": 0.70, "mid": 0.21, "large": 0.09}


def _path(relative: str) -> Path:
    return THIS_DIR / relative


FINAL_CHECKPOINTS = {
    "baseline": {
        "kind": "manual",
        "experts": {
            "small": {
                "label": "f_small",
                "model_name": "DM_time_binary_classificator_241002_3_GAP",
                "mode": "dmt",
                "dropout": False,
                "checkpoint": _path("final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_241002_3_GAP-014-0.764-0.740.pth"),
            },
            "mid": {
                "label": "f_mid",
                "model_name": "DM_time_binary_classificator_241002_5_GAP",
                "mode": "ft",
                "dropout": False,
                "checkpoint": _path("final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_241002_5_GAP-060-0.973-0.948.pth"),
            },
            "large": {
                "label": "f_large",
                "model_name": "DM_time_binary_classificator_resnet18",
                "mode": "dmft",
                "dropout": 0.3,
                "checkpoint": _path("final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_resnet18-003-0.993-0.993.pth"),
            },
        },
        "rejectors": {
            "r1": {
                "model_name": "conv_mlp",
                "cnn_channels": 64,
                "extra_conv": False,
                "pool_size": 7,
                "hidden_dim": 64,
                "dropout": 0.0,
                "checkpoint": _path("final_checkpoints/baseline_rejection_ensemble/prot-run_embedding_3_GAP_conv_mlp_lr1.05e-05_wd0.00e+00_drop0.0_channels64_extraFalse_pool7_hidden64_worker2_trial3-042-0.712-0.635.pth"),
            },
            "r2": {
                "model_name": "conv_mlp",
                "cnn_channels": 64,
                "extra_conv": True,
                "pool_size": 7,
                "hidden_dim": 128,
                "dropout": 0.2,
                "checkpoint": _path("final_checkpoints/baseline_rejection_ensemble/prot-run_embedding_r2_conv_mlp_lr4.51e-05_wd0.00e+00_drop0.2_channels64_extraTrue_pool7_hidden128_worker13_trial6-035-0.825-0.842.pth"),
            },
        },
    },
    "finetuned_replay": {
        "kind": "manual",
        "experts": {
            "small": {
                "label": "f_small",
                "model_name": "DM_time_binary_classificator_241002_3_GAP",
                "mode": "dmt",
                "dropout": False,
                "checkpoint": _path("final_checkpoints/finetune_checkpoints/prot-DM_time_binary_classificator_241002_3_GAP_finetune-004-0.838-0.813.pth"),
            },
            "mid": {
                "label": "f_mid",
                "model_name": "DM_time_binary_classificator_241002_5_GAP",
                "mode": "ft",
                "dropout": False,
                "checkpoint": _path("final_checkpoints/finetune_checkpoints/prot-DM_time_binary_classificator_241002_5_GAP_finetune-019-0.989-0.993.pth"),
            },
            "large": {
                "label": "f_large",
                "model_name": "DM_time_binary_classificator_resnet18",
                "mode": "dmft",
                "dropout": 0.3,
                "checkpoint": _path("final_checkpoints/finetune_checkpoints/prot-DM_time_binary_classificator_resnet18_finetune-010-0.999-0.993.pth"),
            },
        },
        "rejectors": {
            "r1": {
                "model_name": "conv_mlp",
                "cnn_channels": 64,
                "extra_conv": False,
                "pool_size": 7,
                "hidden_dim": 64,
                "dropout": 0.0,
                "checkpoint": _path("final_checkpoints/finetune_checkpoints/prot-run_embedding_r1_conv_mlp_lr1.05e-05_wd0.00e+00_drop0.0_channels64_extraFalse_pool7_hidden64_worker3_trial0-030-0.688-0.618.pth"),
            },
            "r2": {
                "model_name": "conv_mlp",
                "cnn_channels": 64,
                "extra_conv": True,
                "pool_size": 7,
                "hidden_dim": 128,
                "dropout": 0.2,
                "checkpoint": _path("final_checkpoints/finetune_checkpoints/prot-run_embedding_r2_conv_mlp_lr4.51e-05_wd0.00e+00_drop0.2_channels64_extraTrue_pool7_hidden128_worker0_trial0-017-0.800-0.834.pth"),
            },
        },
    },
    "joint_training": {
        "kind": "joint",
        "checkpoint": _path("final_checkpoints/joint_cascade_moe_best.pth"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_path("artifacts/final_rejection_ensemble_eval"))
    parser.add_argument("--dataset-dir", type=Path, default=_path("../DM_time_dataset_creator/outputs"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None, help="Smoke-test limiter; leave unset for final results.")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse per-ensemble split inference caches in output-dir/cache.")
    parser.add_argument("--torch-threads", type=int, default=None)
    return parser.parse_args()


def as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(payload), handle, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def make_loader(dataset_dir: Path, split: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = DMTimeShardDataset(
        {"output_dir": str(dataset_dir), "prefix": DATASET_PREFIX},
        use_freq_time=True,
        split=split,
    )
    dataset.labels = label_encoding(dataset.labels.astype(object))
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def build_expert(spec: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = models_htable[spec["model_name"]](
        256,
        mode=spec["mode"],
        dropout=spec.get("dropout", False),
        device=device,
    ).to(device)
    load_expert_checkpoint(model, spec["checkpoint"], map_location=device, strict=True)
    model.eval()
    return model


def build_rejector(spec: dict[str, Any], in_channels: int, device: torch.device, joint_component: str) -> torch.nn.Module:
    rejector, _feature_source = build_embedding_processing(
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
    load_rejector_checkpoint(
        rejector,
        spec["checkpoint"],
        map_location=device,
        strict=True,
        joint_component=joint_component,
    )
    rejector.eval()
    return rejector


def softmax_class1(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


def collect_manual_outputs(
    ensemble_name: str,
    spec: dict[str, Any],
    loader: DataLoader,
    split: str,
    device: torch.device,
    cache_path: Path,
    reuse_cache: bool,
    max_batches: int | None,
) -> dict[str, np.ndarray]:
    if reuse_cache and cache_path.exists():
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in cached.files}

    experts = spec["experts"]
    f_small = build_expert(experts["small"], device)
    f_mid = build_expert(experts["mid"], device)
    f_large = build_expert(experts["large"], device)
    r1 = build_rejector(spec["rejectors"]["r1"], f_small.out_features, device, "r1")
    r2 = build_rejector(spec["rejectors"]["r2"], f_mid.out_features, device, "r2")

    chunks: dict[str, list[np.ndarray]] = {
        "labels": [],
        "metadata": [],
        "small_pred": [],
        "mid_pred": [],
        "large_pred": [],
        "r1_score": [],
        "r2_score": [],
    }

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"{ensemble_name}/{split}", leave=False)):
            if max_batches is not None and batch_index >= max_batches:
                break
            labels = batch["label"].cpu().numpy().astype(np.int64)
            metadata = batch["metadata"].cpu().numpy().astype(np.float32)

            small_x = f_small._prepare_input(batch)
            small_features = f_small.classifier_features(small_x)
            small_logits = f_small.fc2(torch.flatten(f_small.gap(small_features), 1)) if hasattr(f_small, "gap") else f_small.classifier(small_x)
            r1_logits = r1(small_features)

            mid_x = f_mid._prepare_input(batch)
            mid_features = f_mid.classifier_features(mid_x)
            mid_logits = f_mid.fc2(torch.flatten(f_mid.gap(mid_features), 1)) if hasattr(f_mid, "gap") else f_mid.classifier(mid_x)
            r2_logits = r2(mid_features)

            large_logits = f_large(batch)

            chunks["labels"].append(labels)
            chunks["metadata"].append(metadata)
            chunks["small_pred"].append(small_logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
            chunks["mid_pred"].append(mid_logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
            chunks["large_pred"].append(large_logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
            chunks["r1_score"].append(softmax_class1(r1_logits).detach().cpu().numpy().astype(np.float32))
            chunks["r2_score"].append(softmax_class1(r2_logits).detach().cpu().numpy().astype(np.float32))

    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    return arrays


def collect_joint_outputs(
    ensemble_name: str,
    spec: dict[str, Any],
    loader: DataLoader,
    split: str,
    device: torch.device,
    dataset_dir: Path,
    cache_path: Path,
    reuse_cache: bool,
    max_batches: int | None,
) -> dict[str, np.ndarray]:
    if reuse_cache and cache_path.exists():
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in cached.files}

    checkpoint = torch.load(spec["checkpoint"], map_location=device)
    config = copy.deepcopy(checkpoint["config"])
    config["device"] = str(device)
    config["dataset"]["output_dir"] = str(dataset_dir)
    model = build_joint_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    chunks: dict[str, list[np.ndarray]] = {
        "labels": [],
        "metadata": [],
        "small_pred": [],
        "mid_pred": [],
        "large_pred": [],
        "r1_score": [],
        "r2_score": [],
    }

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"{ensemble_name}/{split}", leave=False)):
            if max_batches is not None and batch_index >= max_batches:
                break
            outputs = model._forward_all_aux(batch)
            expert_logits = outputs["expert_logits"]
            chunks["labels"].append(batch["label"].cpu().numpy().astype(np.int64))
            chunks["metadata"].append(batch["metadata"].cpu().numpy().astype(np.float32))
            chunks["small_pred"].append(expert_logits[:, 0, :].argmax(dim=1).detach().cpu().numpy().astype(np.int64))
            chunks["mid_pred"].append(expert_logits[:, 1, :].argmax(dim=1).detach().cpu().numpy().astype(np.int64))
            chunks["large_pred"].append(expert_logits[:, 2, :].argmax(dim=1).detach().cpu().numpy().astype(np.int64))
            chunks["r1_score"].append(outputs["rejector_probs"]["r1"].detach().cpu().numpy().astype(np.float32))
            chunks["r2_score"].append(outputs["rejector_probs"]["r2"].detach().cpu().numpy().astype(np.float32))

    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    return arrays


def desired_val_counts(n_samples: int) -> dict[str, int]:
    small = int(round(REQUESTED_FRACTIONS["small"] * n_samples))
    mid = int(round(REQUESTED_FRACTIONS["mid"] * n_samples))
    large = n_samples - small - mid
    return {"small": small, "mid": mid, "large": large}


def threshold_for_top_k(scores: np.ndarray, k: int) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if k <= 0:
        return math.inf
    if k >= scores.size:
        return -math.inf
    return float(np.partition(scores, scores.size - k)[scores.size - k])


def calibrate_thresholds(val_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    n_samples = int(val_arrays["labels"].shape[0])
    counts = desired_val_counts(n_samples)
    forwarded_count = counts["mid"] + counts["large"]
    r1_threshold = threshold_for_top_k(val_arrays["r1_score"], forwarded_count)
    r1_mask = val_arrays["r1_score"] >= r1_threshold
    r2_threshold = threshold_for_top_k(val_arrays["r2_score"][r1_mask], counts["large"])
    val_routes = route_with_thresholds(val_arrays, r1_threshold, r2_threshold)
    return {
        "r1_threshold": r1_threshold,
        "r2_threshold": r2_threshold,
        "requested_counts": counts,
        "actual_val_counts": {name: int((val_routes["selected_expert"] == idx).sum()) for idx, name in enumerate(("small", "mid", "large"))},
    }


def route_with_thresholds(arrays: dict[str, np.ndarray], r1_threshold: float, r2_threshold: float) -> dict[str, np.ndarray]:
    r1_mask = arrays["r1_score"] >= r1_threshold
    r2_mask = np.zeros_like(r1_mask, dtype=bool)
    r2_mask[r1_mask] = arrays["r2_score"][r1_mask] >= r2_threshold

    selected = np.zeros_like(arrays["labels"], dtype=np.int64)
    selected[r1_mask] = 1
    selected[r2_mask] = 2

    expert_preds = np.vstack([arrays["small_pred"], arrays["mid_pred"], arrays["large_pred"]])
    final_pred = expert_preds[selected, np.arange(selected.size)]
    return {
        "selected_expert": selected,
        "final_pred": final_pred.astype(np.int64),
        "r1_forwarded": r1_mask,
        "r2_forwarded": r2_mask,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_samples": int(y_true.size),
        "accuracy": (tp + tn) / y_true.size if y_true.size else math.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def rejector_targets(arrays: dict[str, np.ndarray], routes: dict[str, np.ndarray], rejector: str, target_rule: str) -> tuple[np.ndarray, np.ndarray, float]:
    labels = arrays["labels"]
    small_correct = arrays["small_pred"] == labels
    mid_correct = arrays["mid_pred"] == labels
    large_correct = arrays["large_pred"] == labels

    if rejector == "r1":
        mask = np.ones(labels.shape[0], dtype=bool)
        scores = arrays["r1_score"]
        threshold = routes["r1_threshold"]
        if target_rule == "current_wrong":
            target = ~small_correct
        elif target_rule == "current_wrong_next_correct":
            target = (~small_correct) & (mid_correct | large_correct)
        else:
            raise ValueError(target_rule)
    elif rejector == "r2":
        mask = routes["r1_forwarded"]
        scores = arrays["r2_score"][mask]
        threshold = routes["r2_threshold"]
        if target_rule == "current_wrong":
            target = ~mid_correct
        elif target_rule == "current_wrong_next_correct":
            target = (~mid_correct) & large_correct
        else:
            raise ValueError(target_rule)
        target = target[mask]
    else:
        raise ValueError(rejector)
    if rejector == "r1":
        target = target[mask]
        scores = scores[mask]
    return target.astype(np.int64), scores.astype(np.float64), float(threshold)


def rejector_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    positives = int((y_true == 1).sum())
    predicted_positive = scores >= threshold
    tp = int(((y_true == 1) & predicted_positive).sum())
    fp = int(((y_true == 0) & predicted_positive).sum())
    fn = int(((y_true == 1) & ~predicted_positive).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / positives if positives else math.nan
    f1 = 2.0 * precision * recall / (precision + recall) if positives and (precision + recall) else 0.0
    ap = float(average_precision_score(y_true, scores)) if positives and positives < y_true.size else math.nan
    return {
        "n_samples": int(y_true.size),
        "n_positive": positives,
        "threshold": float(threshold),
        "ap": ap,
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "f1_at_threshold": f1,
        "tp_at_threshold": tp,
        "fp_at_threshold": fp,
        "fn_at_threshold": fn,
    }


def save_pr_plot(
    plot_dir: Path,
    pr_rows: list[dict[str, Any]],
    ensemble: str,
    split: str,
    rejector: str,
    target_rule: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    if int((y_true == 1).sum()) == 0:
        return
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap = float(average_precision_score(y_true, scores)) if int((y_true == 1).sum()) < y_true.size else math.nan
    operating_pred = scores >= threshold
    op_tp = int(((y_true == 1) & operating_pred).sum())
    op_fp = int(((y_true == 0) & operating_pred).sum())
    op_pos = int((y_true == 1).sum())
    op_precision = op_tp / (op_tp + op_fp) if (op_tp + op_fp) else 0.0
    op_recall = op_tp / op_pos if op_pos else math.nan

    for idx in range(precision.shape[0]):
        pr_rows.append(
            {
                "ensemble": ensemble,
                "split": split,
                "rejector": rejector,
                "target_rule": target_rule,
                "point_index": idx,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "threshold": float(thresholds[idx]) if idx < thresholds.shape[0] else math.inf,
                "ap": ap,
                "operating_threshold": float(threshold),
            }
        )

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.step(recall, precision, where="post", label=f"AP={ap:.6f}")
    ax.scatter([op_recall], [op_precision], color="black", s=28, zorder=3, label=f"thr={threshold:.6g}")
    ax.set_title(f"{ensemble} {split} {rejector} {target_rule}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"pr_{ensemble}_{split}_{rejector}_{target_rule}.png", dpi=180)
    plt.close(fig)


def inspect_split_artifacts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_dir = _path("artifacts/splits")
    for path in sorted(split_dir.glob("*.pth")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            rows.append({"path": str(path), "load_error": str(exc)})
            continue
        if not isinstance(payload, dict):
            rows.append({"path": str(path), "payload_type": type(payload).__name__})
            continue
        for key in ("train", "val", "test"):
            if key not in payload or not isinstance(payload[key], dict):
                continue
            meta = payload[key].get("metadata", {})
            rows.append(
                {
                    "path": str(path),
                    "payload_key": key,
                    "metadata_split": meta.get("split"),
                    "n_indices": len(payload[key].get("indices", [])),
                    "dataset_output_dir": meta.get("dataset_cfg", {}).get("output_dir"),
                    "prefix": meta.get("dataset_cfg", {}).get("prefix"),
                    "key_mismatch": bool(meta.get("split") and meta.get("split") != key),
                }
            )
    return rows


def read_previous_exports(joint_checkpoint: Path) -> dict[str, Any]:
    previous: dict[str, Any] = {"validation_accuracy_exports": {}, "test_metric_exports": {}, "saved_threshold_exports": {}}
    val_files = {
        "baseline": _path("rejector_analysis/cascaded_pipeline_validation_accuracy.csv"),
        "finetuned_replay": _path("rejector_analysis_finetune/cascaded_pipeline_validation_accuracy.csv"),
    }
    for name, path in val_files.items():
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
                previous["validation_accuracy_exports"][name] = float(row["pipeline_validation_accuracy"])

    metric_files = {
        "baseline": _path("plot/snr_curves_test/metrics/baseline_rejection_ensemble_test_metrics.json"),
        "finetuned_replay": _path("plot/snr_curves_test/metrics/finetuned_rejection_ensemble_test_metrics.json"),
        "joint_training": _path("plot/snr_curves_test/metrics/joint_training_rejection_ensemble_test_metrics.json"),
        "baseline_f_small": _path("plot/snr_curves_test/metrics/f_small_test_metrics.json"),
        "baseline_f_mid": _path("plot/snr_curves_test/metrics/f_mid_test_metrics.json"),
        "baseline_f_large": _path("plot/snr_curves_test/metrics/f_large_test_metrics.json"),
    }
    for name, path in metric_files.items():
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            previous["test_metric_exports"][name] = {
                key: data[key]
                for key in ("accuracy", "precision", "recall", "f1", "count")
                if key in data
            }

    for name, path in {
        "baseline": _path("rejector_analysis/r1_r2_uncut_val_scores.npz"),
        "finetuned_replay": _path("rejector_analysis_finetune/r1_r2_uncut_val_scores_finetune.npz"),
    }.items():
        if path.exists():
            with np.load(path) as data:
                previous["saved_threshold_exports"][name] = {
                    "r1_threshold": float(data["r1_threshold"]),
                    "r2_threshold": float(data["r2_threshold"]),
                    "r1_checkpoint": str(data["r1_checkpoint"]),
                    "r2_checkpoint": str(data["r2_checkpoint"]),
                    "small_checkpoint": str(data["small_checkpoint"]),
                    "mid_checkpoint": str(data["mid_checkpoint"]),
                    "large_checkpoint": str(data["large_checkpoint"]),
                }

    if joint_checkpoint.exists():
        checkpoint = torch.load(joint_checkpoint, map_location="cpu")
        metrics = checkpoint.get("metrics", {})
        previous["validation_accuracy_exports"]["joint_training_checkpoint_val_hard"] = metrics.get("val_hard/accuracy")
        previous["validation_accuracy_exports"]["joint_training_checkpoint_val_topk"] = metrics.get("val_topk/accuracy")
    return previous


def checkpoint_audit() -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for ensemble, spec in FINAL_CHECKPOINTS.items():
        entries = []
        if spec["kind"] == "joint":
            role_paths = [("joint", spec["checkpoint"])]
        else:
            role_paths = [(role, item["checkpoint"]) for role, item in spec["experts"].items()]
            role_paths.extend((role, item["checkpoint"]) for role, item in spec["rejectors"].items())
        for role, path in role_paths:
            entries.append(
                {
                    "role": role,
                    "path": str(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "sha256": sha256_file(path) if path.exists() else None,
                }
            )
        audit[ensemble] = entries
    return audit


def evaluate_ensemble(
    ensemble: str,
    split: str,
    arrays: dict[str, np.ndarray],
    thresholds: dict[str, Any],
    plot_dir: Path,
    pr_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    routes = route_with_thresholds(arrays, thresholds["r1_threshold"], thresholds["r2_threshold"])
    routes["r1_threshold"] = thresholds["r1_threshold"]
    routes["r2_threshold"] = thresholds["r2_threshold"]
    labels = arrays["labels"]

    metric_rows: list[dict[str, Any]] = []
    for model_key, pred_key in {
        "ensemble": "final_pred",
        "f_small": "small_pred",
        "f_mid": "mid_pred",
        "f_large": "large_pred",
    }.items():
        pred = routes[pred_key] if model_key == "ensemble" else arrays[pred_key]
        row = {"ensemble": ensemble, "split": split, "model": model_key}
        row.update(classification_metrics(labels, pred))
        if model_key == "ensemble":
            for idx, usage_name in enumerate(("small", "mid", "large")):
                count = int((routes["selected_expert"] == idx).sum())
                row[f"usage_{usage_name}_count"] = count
                row[f"usage_{usage_name}_fraction"] = count / labels.size
        metric_rows.append(row)

    rejector_rows: list[dict[str, Any]] = []
    for rejector in ("r1", "r2"):
        for target_rule in ("current_wrong", "current_wrong_next_correct"):
            target, scores, threshold = rejector_targets(arrays, routes, rejector, target_rule)
            row = {
                "ensemble": ensemble,
                "split": split,
                "rejector": rejector,
                "target_rule": target_rule,
            }
            row.update(rejector_metrics(target, scores, threshold))
            rejector_rows.append(row)
            save_pr_plot(plot_dir, pr_rows, ensemble, split, rejector, target_rule, target, scores, threshold)

    summary = {
        "n_samples": int(labels.size),
        "thresholds": {
            "r1": float(thresholds["r1_threshold"]),
            "r2": float(thresholds["r2_threshold"]),
        },
        "usage_counts": {name: int((routes["selected_expert"] == idx).sum()) for idx, name in enumerate(("small", "mid", "large"))},
        "usage_fractions": {name: float((routes["selected_expert"] == idx).mean()) for idx, name in enumerate(("small", "mid", "large"))},
    }
    return metric_rows, rejector_rows, summary


def build_comparison_rows(
    metric_rows: list[dict[str, Any]],
    thresholds_rows: list[dict[str, Any]],
    previous: dict[str, Any],
) -> list[dict[str, Any]]:
    by_key = {(row["ensemble"], row["split"], row["model"]): row for row in metric_rows}
    comparison: list[dict[str, Any]] = []
    for ensemble in ("baseline", "finetuned_replay", "joint_training"):
        old_test = previous.get("test_metric_exports", {}).get(ensemble)
        if old_test:
            new = by_key.get((ensemble, "test", "ensemble"))
            if new:
                for metric in ("accuracy", "precision", "recall", "f1"):
                    comparison.append(
                        {
                            "comparison": "previous_snr_curves_test_json",
                            "ensemble": ensemble,
                            "split": "test",
                            "metric": metric,
                            "previous": old_test.get(metric),
                            "current": new.get(metric),
                            "delta_current_minus_previous": new.get(metric) - old_test.get(metric),
                        }
                    )
        old_val = previous.get("validation_accuracy_exports", {}).get(ensemble)
        if old_val is not None:
            new = by_key.get((ensemble, "val", "ensemble"))
            if new:
                comparison.append(
                    {
                        "comparison": "previous_cascaded_pipeline_validation_csv",
                        "ensemble": ensemble,
                        "split": "val",
                        "metric": "accuracy",
                        "previous": old_val,
                        "current": new.get("accuracy"),
                        "delta_current_minus_previous": new.get("accuracy") - old_val,
                    }
                )

    for export_name in ("joint_training_checkpoint_val_hard", "joint_training_checkpoint_val_topk"):
        old_val = previous.get("validation_accuracy_exports", {}).get(export_name)
        if old_val is not None:
            new = by_key.get(("joint_training", "val", "ensemble"))
            comparison.append(
                {
                    "comparison": export_name,
                    "ensemble": "joint_training",
                    "split": "val",
                    "metric": "accuracy",
                    "previous": old_val,
                    "current": new.get("accuracy") if new else None,
                    "delta_current_minus_previous": (new.get("accuracy") - old_val) if new else None,
                }
            )

    current_thr = {row["ensemble"]: row for row in thresholds_rows}
    for ensemble, saved in previous.get("saved_threshold_exports", {}).items():
        row = current_thr.get(ensemble)
        if not row:
            continue
        for rejector in ("r1", "r2"):
            metric = f"{rejector}_threshold"
            comparison.append(
                {
                    "comparison": "previous_saved_val_threshold_npz",
                    "ensemble": ensemble,
                    "split": "val",
                    "metric": metric,
                    "previous": saved.get(metric),
                    "current": row.get(metric),
                    "delta_current_minus_previous": row.get(metric) - saved.get(metric),
                }
            )
    return comparison


def write_routing_snapshot(path: Path, ensemble: str, split: str, arrays: dict[str, np.ndarray], thresholds: dict[str, Any]) -> None:
    routes = route_with_thresholds(arrays, thresholds["r1_threshold"], thresholds["r2_threshold"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ensemble", "split", "sample_index", "label", "small_pred", "mid_pred", "large_pred", "r1_score", "r2_score", "selected_expert", "final_pred"])
        for idx in range(arrays["labels"].shape[0]):
            writer.writerow(
                [
                    ensemble,
                    split,
                    idx,
                    int(arrays["labels"][idx]),
                    int(arrays["small_pred"][idx]),
                    int(arrays["mid_pred"][idx]),
                    int(arrays["large_pred"][idx]),
                    float(arrays["r1_score"][idx]),
                    float(arrays["r2_score"][idx]),
                    int(routes["selected_expert"][idx]),
                    int(routes["final_pred"][idx]),
                ]
            )


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    plot_dir = args.output_dir / "plots"

    loaders = {
        split: make_loader(args.dataset_dir, split, args.batch_size, args.num_workers)
        for split in ("val", "test")
    }

    all_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    metric_rows: list[dict[str, Any]] = []
    rejector_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for ensemble, spec in FINAL_CHECKPOINTS.items():
        print(f"\n=== {ensemble} ===")
        split_arrays: dict[str, dict[str, np.ndarray]] = {}
        for split, loader in loaders.items():
            cache_path = cache_dir / f"{ensemble}_{split}.npz"
            if spec["kind"] == "joint":
                arrays = collect_joint_outputs(ensemble, spec, loader, split, device, args.dataset_dir, cache_path, args.reuse_cache, args.max_batches)
            else:
                arrays = collect_manual_outputs(ensemble, spec, loader, split, device, cache_path, args.reuse_cache, args.max_batches)
            split_arrays[split] = arrays
        all_arrays[ensemble] = split_arrays

        thresholds = calibrate_thresholds(split_arrays["val"])
        threshold_row = {
            "ensemble": ensemble,
            "calibration_split": "val",
            "n_val_samples": int(split_arrays["val"]["labels"].shape[0]),
            "r1_threshold": thresholds["r1_threshold"],
            "r2_threshold": thresholds["r2_threshold"],
        }
        for name, count in thresholds["requested_counts"].items():
            threshold_row[f"requested_{name}_count"] = count
        for name, count in thresholds["actual_val_counts"].items():
            threshold_row[f"actual_val_{name}_count"] = count
        threshold_rows.append(threshold_row)

        summaries[ensemble] = {"thresholds": threshold_row, "splits": {}}
        for split, arrays in split_arrays.items():
            rows, r_rows, summary = evaluate_ensemble(ensemble, split, arrays, thresholds, plot_dir, pr_rows)
            metric_rows.extend(rows)
            rejector_rows.extend(r_rows)
            summaries[ensemble]["splits"][split] = summary
            write_routing_snapshot(args.output_dir / "routing_snapshots" / f"{ensemble}_{split}_routing.csv.gz", ensemble, split, arrays, thresholds)

    previous = read_previous_exports(FINAL_CHECKPOINTS["joint_training"]["checkpoint"])
    comparison_rows = build_comparison_rows(metric_rows, threshold_rows, previous)
    split_audit = inspect_split_artifacts()
    audit = {
        "device": str(device),
        "dataset": {
            "output_dir": str(args.dataset_dir),
            "prefix": DATASET_PREFIX,
            "sample_counts": {split: len(loader.dataset) for split, loader in loaders.items()},
        },
        "requested_model_fractions": REQUESTED_FRACTIONS,
        "checkpoint_audit": checkpoint_audit(),
        "split_artifact_audit": split_audit,
        "previous_exports": previous,
        "max_batches": args.max_batches,
    }

    write_csv(args.output_dir / "metrics_long.csv", metric_rows)
    write_csv(args.output_dir / "rejector_metrics_long.csv", rejector_rows)
    write_csv(args.output_dir / "thresholds.csv", threshold_rows)
    write_csv(args.output_dir / "pr_curves_long.csv", pr_rows)
    write_csv(args.output_dir / "comparison_to_previous_exports.csv", comparison_rows)
    write_csv(args.output_dir / "split_artifact_audit.csv", split_audit)
    write_json(args.output_dir / "results.json", {"metrics": metric_rows, "rejectors": rejector_rows, "thresholds": threshold_rows, "summaries": summaries, "comparison": comparison_rows})
    write_json(args.output_dir / "audit.json", audit)

    print("\nComparison table:")
    for row in comparison_rows:
        if row["metric"] == "accuracy":
            print(
                f"{row['ensemble']:18s} {row['split']:4s} {row['comparison']:42s} "
                f"prev={row['previous']:.12f} current={row['current']:.12f} "
                f"delta={row['delta_current_minus_previous']:+.12f}"
            )
    print(f"\nWrote results to: {args.output_dir}")


if __name__ == "__main__":
    main()
