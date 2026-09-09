#!/usr/bin/env python3
"""Evaluate plausible f_small checkpoints with one shared inference pipeline.

The script is intentionally evaluation-only: it loads existing checkpoints,
uses the canonical DMTimeShardDataset val/test splits, sets eval() mode, and
records paths, hashes, model configuration and metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

import training
from DMTimeShardDataset import DMTimeShardDataset
from moe.checkpoints import load_expert_checkpoint
from training_models import models_htable


ROOT = Path(__file__).resolve().parent
DATASET_CFG = {
    "output_dir": str((ROOT / "../DM_time_dataset_creator/outputs").resolve()),
    "prefix": "B0531+21_59000_48386",
}


CANDIDATES = [
    {
        "candidate": "baseline_final",
        "source": "final_checkpoints + baseline rejection scripts",
        "path": ROOT / "final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_241002_3_GAP-014-0.764-0.740.pth",
    },
    {
        "candidate": "baseline_final_original_location",
        "source": "checkpoints_new original copy referenced in benchmark notebook",
        "path": ROOT / "checkpoints_new/ch_point_DM_time_binary_classificator_241002_3_GAP_256/prot-DM_time_binary_classificator_241002_3_GAP-014-0.764-0.740.pth",
    },
    {
        "candidate": "evaluate_ipynb_commented_009_0771_0742",
        "source": "commented candidate in evaluate.ipynb",
        "path": ROOT / "checkpoints_new/ch_point_DM_time_binary_classificator_241002_3_GAP_256/prot-DM_time_binary_classificator_241002_3_GAP-009-0.771-0.742.pth",
    },
    {
        "candidate": "filename_val_near_742_epoch006",
        "source": "checkpoint filename near reported val=0.7421",
        "path": ROOT / "checkpoints_new/ch_point_DM_time_binary_classificator_241002_3_GAP_256/prot-DM_time_binary_classificator_241002_3_GAP-006-0.743-0.742.pth",
    },
    {
        "candidate": "filename_val_near_742_epoch008",
        "source": "checkpoint filename near reported val=0.7421",
        "path": ROOT / "checkpoints_new/ch_point_DM_time_binary_classificator_241002_3_GAP_256/prot-DM_time_binary_classificator_241002_3_GAP-008-0.769-0.742.pth",
    },
    {
        "candidate": "filename_test_near_7594_epoch007",
        "source": "checkpoint filename closest to reported test=0.7594 among searched names",
        "path": ROOT / "checkpoints_new/ch_point_DM_time_binary_classificator_241002_3_GAP_256/prot-DM_time_binary_classificator_241002_3_GAP-007-0.765-0.755.pth",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/fsmall_checkpoint_candidate_eval")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def make_loader(split: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = DMTimeShardDataset(DATASET_CFG, use_freq_time=True, split=split)
    dataset.labels = training.label_encoding(dataset.labels.astype(object))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = models_htable["DM_time_binary_classificator_241002_3_GAP"](
        256, mode="dmt", dropout=False, device=device
    ).to(device)
    load_expert_checkpoint(model, checkpoint_path, map_location=device)
    model.eval()
    return model


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {
        "n_samples": int(y_true.size),
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, desc: str) -> dict[str, Any]:
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    max_forward_abs_diff = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False, ncols=90):
            labels = batch["label"].to(device, non_blocking=True)
            logits_forward = model(batch)
            prepared = model._prepare_input(batch)
            logits_manual = model.classifier(prepared)
            max_forward_abs_diff = max(
                max_forward_abs_diff,
                float((logits_forward - logits_manual).abs().max().detach().cpu().item()),
            )
            preds = logits_forward.argmax(dim=1)
            y_true_parts.append(labels.detach().cpu().numpy().astype(np.int64))
            y_pred_parts.append(preds.detach().cpu().numpy().astype(np.int64))
    row = metric_row(np.concatenate(y_true_parts), np.concatenate(y_pred_parts))
    row["max_forward_vs_classifier_abs_diff"] = max_forward_abs_diff
    return row


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    loaders = {
        split: make_loader(split, args.batch_size, args.num_workers)
        for split in ("val", "test")
    }

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        path = candidate["path"]
        if not path.exists():
            audit.append({**candidate, "path": str(path), "exists": False})
            continue
        checkpoint = torch.load(path, map_location="cpu")
        file_hash = sha256_file(path)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        weight_digest = hashlib.sha256()
        for key in sorted(state_dict):
            weight_digest.update(key.encode("utf-8"))
            tensor = state_dict[key].detach().cpu().contiguous().numpy()
            weight_digest.update(tensor.tobytes())
        ckpt_info = {
            **candidate,
            "path": str(path),
            "exists": True,
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "checkpoint_accuracy_field": checkpoint.get("accuracy") if isinstance(checkpoint, dict) else None,
            "checkpoint_val_accuracy_field": checkpoint.get("val_accuracy") if isinstance(checkpoint, dict) else None,
            "sha256": file_hash,
            "state_dict_sha256": weight_digest.hexdigest(),
            "model_name": "DM_time_binary_classificator_241002_3_GAP",
            "mode": "dmt",
            "dropout": False,
            "classification_rule": "argmax(logits), positive class id=1 after label_encoding",
            "eval_mode": True,
            "dataset_output_dir": DATASET_CFG["output_dir"],
            "dataset_prefix": DATASET_CFG["prefix"],
        }
        audit.append(ckpt_info)
        model = build_model(path, device)
        for split, loader in loaders.items():
            metrics = evaluate_model(model, loader, device, f"{candidate['candidate']}/{split}")
            rows.append({**ckpt_info, "split": split, **metrics})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = output_dir / "fsmall_checkpoint_candidate_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_payload = {
        "dataset": DATASET_CFG,
        "device": str(device),
        "pipeline": {
            "model_name": "DM_time_binary_classificator_241002_3_GAP",
            "resolution": 256,
            "mode": "dmt",
            "dropout": False,
            "eval": True,
            "preprocessing": "DMTimeShardDataset(use_freq_time=True); model consumes batch['dm_time'].unsqueeze(1)",
            "classification_rule": "argmax over logits, positive class id=1 after training.label_encoding",
        },
        "audit": audit,
        "metrics": rows,
    }
    with (output_dir / "fsmall_checkpoint_candidate_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2, sort_keys=True)

    print(f"Wrote {csv_path}")
    for row in rows:
        print(
            f"{row['candidate']:<42} {row['split']:<4} "
            f"acc={row['accuracy']:.12f} f1={row['f1']:.12f} "
            f"sha={row['sha256'][:12]}"
        )


if __name__ == "__main__":
    main()
