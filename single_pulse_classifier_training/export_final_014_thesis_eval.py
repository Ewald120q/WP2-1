from __future__ import annotations

import csv
import gzip
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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from evaluate_final_rejection_ensembles import (
    FINAL_CHECKPOINTS,
    classification_metrics,
    route_with_thresholds,
    sha256_file,
)


SOURCE_DIR = THIS_DIR / "artifacts" / "final_rejection_ensemble_eval"
OUTPUT_DIR = THIS_DIR / "artifacts" / "final_014_thesis_eval"
ROUTE_NAMES = ("R1 accepted", "R2 accepted", "R2 rejected")
MODEL_FOR_ROUTE = ("f_small", "f_mid", "f_large")
PRED_KEYS = {"f_small": "small_pred", "f_mid": "mid_pred", "f_large": "large_pred"}
SNR_BIN_LABELS = ["Rest of Events"] + [f"{value}.x" for value in range(3, 9)] + ["9+"]


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


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_thresholds(path: Path) -> dict[str, dict[str, float]]:
    rows = read_csv(path)
    return {
        row["ensemble"]: {
            "r1_threshold": float(row["r1_threshold"]),
            "r2_threshold": float(row["r2_threshold"]),
        }
        for row in rows
    }


def load_arrays(ensemble: str, split: str) -> dict[str, np.ndarray]:
    cache_path = SOURCE_DIR / "cache" / f"{ensemble}_{split}.npz"
    with np.load(cache_path) as data:
        return {key: data[key] for key in data.files}


def snr_bin(value: float) -> str:
    if not np.isfinite(value):
        return "Rest of Events"
    if value >= 9.0:
        return "9+"
    rounded = int(np.rint(float(value)))
    if 3 <= rounded <= 8:
        return f"{rounded}.x"
    return "Rest of Events"


def snr_rounded_bin(value: float) -> str:
    if not np.isfinite(value):
        return "None"
    return str(int(np.rint(float(value))))


def rows_by_snr(
    ensemble: str,
    split: str,
    model: str,
    labels: np.ndarray,
    preds: np.ndarray,
    metadata: np.ndarray,
    binning: str = "thesis",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if binning == "thesis":
        labels_order = SNR_BIN_LABELS
        bins = np.array([snr_bin(value) for value in metadata[:, 0]], dtype=object)
    elif binning == "rounded":
        bins = np.array([snr_rounded_bin(value) for value in metadata[:, 0]], dtype=object)
        numeric = sorted([int(item) for item in set(bins) if item != "None"])
        labels_order = (["None"] if bool((bins == "None").any()) else []) + [str(item) for item in numeric]
    else:
        raise ValueError(binning)
    for label in labels_order:
        mask = bins == label
        if not bool(mask.any()):
            rows.append({"ensemble": ensemble, "split": split, "model": model, "binning": binning, "snr_bin": label, "n_samples": 0})
            continue
        row = {"ensemble": ensemble, "split": split, "model": model, "binning": binning, "snr_bin": label}
        row.update(classification_metrics(labels[mask], preds[mask]))
        rows.append(row)
    return rows


def routed_subset_rows(
    ensemble: str,
    split: str,
    arrays: dict[str, np.ndarray],
    routes: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = routes["selected_expert"]
    labels = arrays["labels"]
    for idx, route_name in enumerate(ROUTE_NAMES):
        model = MODEL_FOR_ROUTE[idx]
        mask = selected == idx
        pred = arrays[PRED_KEYS[model]][mask]
        row = {
            "ensemble": ensemble,
            "split": split,
            "routing_group": route_name,
            "model": model,
            "n_routed_samples": int(mask.sum()),
            "r1_threshold": routes["r1_threshold"],
            "r2_threshold": routes["r2_threshold"],
        }
        row.update(classification_metrics(labels[mask], pred))
        full = classification_metrics(labels, arrays[PRED_KEYS[model]])
        row["full_split_accuracy"] = full["accuracy"]
        row["full_split_n_samples"] = full["n_samples"]
        rows.append(row)
    return rows


def oracle_rows(ensemble: str, split: str, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    labels = arrays["labels"]
    small_ok = arrays["small_pred"] == labels
    mid_ok = arrays["mid_pred"] == labels
    large_ok = arrays["large_pred"] == labels
    variants = {
        "small_or_mid": {
            "description": "Oracle routes away from f_small exactly when f_mid is correct and f_small is wrong.",
            "correct": small_ok | mid_ok,
            "route": np.where(small_ok, 0, np.where(mid_ok, 1, 0)),
        },
        "any_three": {
            "description": "Oracle upper bound: sample is correct if at least one of f_small, f_mid, f_large is correct.",
            "correct": small_ok | mid_ok | large_ok,
            "route": np.where(small_ok, 0, np.where(mid_ok, 1, 2)),
        },
    }
    rows: list[dict[str, Any]] = []
    for variant, payload in variants.items():
        pred = labels.copy()
        pred[~payload["correct"]] = 1 - pred[~payload["correct"]]
        row = {
            "ensemble": ensemble,
            "split": split,
            "oracle_variant": variant,
            "description": payload["description"],
        }
        row.update(classification_metrics(labels, pred))
        route = payload["route"]
        for idx, name in enumerate(("small", "mid", "large")):
            count = int((route == idx).sum())
            row[f"usage_{name}_count"] = count
            row[f"usage_{name}_fraction"] = count / labels.size
        rows.append(row)
    return rows


def plot_metric_bars(path: Path, rows: list[dict[str, Any]], title: str, metric: str = "accuracy") -> None:
    labels = [f"{row['ensemble']}\n{row.get('model', row.get('oracle_variant', ''))}" for row in rows]
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.65 * len(rows)), 3.8))
    ax.bar(np.arange(len(rows)), values, color="#4c78a8")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.01, f"{value:.2%}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_snr_curves(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(SNR_BIN_LABELS))
    for key, group_rows in group_by(rows, ("ensemble", "split", "model")).items():
        ordered = {row["snr_bin"]: row for row in group_rows}
        y = [float(ordered[label]["accuracy"]) if int(ordered[label].get("n_samples", 0)) else np.nan for label in SNR_BIN_LABELS]
        ax.plot(x, y, marker="o", linewidth=1.5, label=" / ".join(key))
    ax.set_xticks(x)
    ax.set_xticklabels(SNR_BIN_LABELS, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def compare_previous_snr_exports(snr_rows: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = THIS_DIR / "plot" / "snr_curves_test" / "snr_metrics_test_summary.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        previous = json.load(handle).get("results", {})
    model_map = {
        "f_small": ("baseline", "f_small"),
        "f_mid": ("baseline", "f_mid"),
        "f_large": ("baseline", "f_large"),
        "baseline_rejection_ensemble": ("baseline", "ensemble"),
        "finetuned_rejection_ensemble": ("finetuned_replay", "ensemble"),
        "joint_training_rejection_ensemble": ("joint_training", "ensemble"),
    }
    current = {
        (row["ensemble"], row["split"], row["model"], row["snr_bin"]): row
        for row in snr_rows
        if row["split"] == "test" and row.get("binning") == "rounded"
    }
    current_metrics = {(row["ensemble"], row["split"], row["model"]): row for row in metrics}
    rows: list[dict[str, Any]] = []
    for old_name, (ensemble, model) in model_map.items():
        old_payload = previous.get(old_name)
        if not old_payload:
            continue
        old_overall = old_payload.get("overall", old_payload)
        for metric in ("accuracy", "precision", "recall", "f1"):
            row = current_metrics.get((ensemble, "test", model))
            current_overall = float(row[metric]) if row else None
            old_value = old_overall.get(metric)
            if old_value is not None and current_overall is not None:
                rows.append(
                    {
                        "comparison": "previous_snr_test_overall_json",
                        "previous_name": old_name,
                        "ensemble": ensemble,
                        "split": "test",
                        "model": model,
                        "snr_bin": "overall",
                        "metric": metric,
                        "previous": old_value,
                        "current": current_overall,
                        "delta_current_minus_previous": current_overall - old_value,
                    }
                )
        for old_bin, old_metrics in old_payload.get("per_bin", {}).items():
            label = "None" if old_bin in {"None", "null"} else str(old_bin)
            row = current.get((ensemble, "test", model, label))
            if not row:
                continue
            for metric in ("accuracy", "precision", "recall", "f1"):
                if metric in old_metrics and metric in row and row.get("n_samples", 0):
                    rows.append(
                        {
                            "comparison": "previous_snr_test_bin_json",
                            "previous_name": old_name,
                            "ensemble": ensemble,
                            "split": "test",
                            "model": model,
                            "snr_bin": label,
                            "metric": metric,
                            "previous": old_metrics[metric],
                            "current": row[metric],
                            "delta_current_minus_previous": row[metric] - old_metrics[metric],
                        }
                    )
    return rows


def group_by(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row[key]) for key in keys), []).append(row)
    return grouped


def checkpoint_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ensemble, spec in FINAL_CHECKPOINTS.items():
        if spec["kind"] == "joint":
            paths = [("joint_model", spec["checkpoint"])]
        else:
            paths = [(role, item["checkpoint"]) for role, item in spec["experts"].items()]
            paths.extend((role, item["checkpoint"]) for role, item in spec["rejectors"].items())
        for role, path in paths:
            rows.append(
                {
                    "ensemble": ensemble,
                    "role": role,
                    "path": str(path),
                    "exists": path.exists(),
                    "sha256": sha256_file(path) if path.exists() else None,
                }
            )
    rows.append(
        {
            "ensemble": "historical_single_model_chapter_candidate",
            "role": "f_small",
            "path": str(
                THIS_DIR
                / "checkpoints_new/ch_point_DM_time_binary_classificator_241002_3_GAP_256/prot-DM_time_binary_classificator_241002_3_GAP-009-0.771-0.742.pth"
            ),
            "exists": (
                THIS_DIR
                / "checkpoints_new/ch_point_DM_time_binary_classificator_241002_3_GAP_256/prot-DM_time_binary_classificator_241002_3_GAP-009-0.771-0.742.pth"
            ).exists(),
            "sha256": sha256_file(
                THIS_DIR
                / "checkpoints_new/ch_point_DM_time_binary_classificator_241002_3_GAP_256/prot-DM_time_binary_classificator_241002_3_GAP-009-0.771-0.742.pth"
            ),
            "note": "Reproduces the old thesis standalone f_small values; kept historical, not used for final ensemble evaluation.",
        }
    )
    return rows


def build_change_rows(metrics: list[dict[str, Any]], oracle: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["ensemble"], r["split"], r["model"]): r for r in metrics}
    oracle_by_key = {(r["ensemble"], r["split"], r["oracle_variant"]): r for r in oracle}
    changes = []
    old_single = {"val": 0.7421119326636905, "test": 0.7593947051186943}
    for split, old in old_single.items():
        new = by_key[("baseline", split, "f_small")]["accuracy"]
        changes.append(
            {
                "affected_statement": "Einzelmodell f_small accuracy",
                "split": split,
                "old_value": old,
                "new_value": new,
                "delta_new_minus_old": new - old,
                "old_source": "historical checkpoint 009-0.771-0.742 reproduced in fsmall_checkpoint_candidate_eval",
                "new_source": "final baseline ensemble checkpoint 014-0.764-0.740",
            }
        )
    previous_validation = {
        ("baseline", "val", "ensemble"): 0.8655947730654762,
        ("finetuned_replay", "val", "ensemble"): 0.8593343098958334,
    }
    for key, old in previous_validation.items():
        new = by_key[key]["accuracy"]
        changes.append(
            {
                "affected_statement": f"{key[0]} cascaded validation accuracy",
                "split": key[1],
                "old_value": old,
                "new_value": new,
                "delta_new_minus_old": new - old,
                "old_source": "previous cascaded_pipeline_validation_accuracy.csv",
                "new_source": "final_014_thesis_eval/full_metrics.csv",
            }
        )
    previous_oracle = {
        ("baseline", "val", "small_or_mid"): 0.9714471726190477,
        ("baseline", "val", "any_three"): 0.9993373325892857,
    }
    for key, old in previous_oracle.items():
        new = oracle_by_key[key]["accuracy"]
        changes.append(
            {
                "affected_statement": f"optimal oracle cascade accuracy {key[2]}",
                "split": key[1],
                "old_value": old,
                "new_value": new,
                "delta_new_minus_old": new - old,
                "old_source": "evaluate.ipynb Oracle output",
                "new_source": "final baseline prediction cache with checkpoint 014",
            }
        )
    return changes


def build_finetune_comparison(metrics: list[dict[str, Any]], routed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_metric = {(r["split"], r["model"], r["ensemble"]): r for r in metrics}
    for split in ("val", "test"):
        for model in ("ensemble", "f_small", "f_mid", "f_large"):
            before = by_metric[(split, model, "baseline")]
            after = by_metric[(split, model, "finetuned_replay")]
            for metric in ("accuracy", "precision", "recall", "f1"):
                rows.append(
                    {
                        "comparison": "finetuned_replay_minus_baseline",
                        "split": split,
                        "scope": "full_split",
                        "model": model,
                        "metric": metric,
                        "before_baseline": before[metric],
                        "after_finetuned_replay": after[metric],
                        "delta_after_minus_before": after[metric] - before[metric],
                    }
                )
    by_route = {(r["split"], r["routing_group"], r["ensemble"]): r for r in routed}
    for split in ("val", "test"):
        for route in ROUTE_NAMES:
            before = by_route[(split, route, "baseline")]
            after = by_route[(split, route, "finetuned_replay")]
            for metric in ("accuracy", "precision", "recall", "f1"):
                rows.append(
                    {
                        "comparison": "finetuned_replay_minus_baseline",
                        "split": split,
                        "scope": "routed_subset",
                        "routing_group": route,
                        "model": before["model"],
                        "metric": metric,
                        "before_baseline": before[metric],
                        "after_finetuned_replay": after[metric],
                        "delta_after_minus_before": after[metric] - before[metric],
                    }
                )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = OUTPUT_DIR / "plots"
    thresholds = read_thresholds(SOURCE_DIR / "thresholds.csv")
    metrics = read_csv(SOURCE_DIR / "metrics_long.csv")
    for row in metrics:
        for key, value in list(row.items()):
            if key not in {"ensemble", "split", "model"} and value != "":
                row[key] = float(value) if any(ch in value for ch in ".eE") else int(value)

    snr_rows: list[dict[str, Any]] = []
    snr_rounded_rows: list[dict[str, Any]] = []
    routed_rows: list[dict[str, Any]] = []
    oracle_all: list[dict[str, Any]] = []
    route_snapshots = OUTPUT_DIR / "routing_snapshots"
    for ensemble in ("baseline", "finetuned_replay", "joint_training"):
        for split in ("val", "test"):
            arrays = load_arrays(ensemble, split)
            routes = route_with_thresholds(
                arrays,
                thresholds[ensemble]["r1_threshold"],
                thresholds[ensemble]["r2_threshold"],
            )
            routes["r1_threshold"] = thresholds[ensemble]["r1_threshold"]
            routes["r2_threshold"] = thresholds[ensemble]["r2_threshold"]
            labels = arrays["labels"]
            selected = routes["selected_expert"]
            final_pred = routes["final_pred"]

            for model, pred in {
                "ensemble": final_pred,
                "f_small": arrays["small_pred"],
                "f_mid": arrays["mid_pred"],
                "f_large": arrays["large_pred"],
            }.items():
                snr_rows.extend(rows_by_snr(ensemble, split, model, labels, pred, arrays["metadata"]))
                snr_rounded_rows.extend(rows_by_snr(ensemble, split, model, labels, pred, arrays["metadata"], binning="rounded"))

            routed_rows.extend(routed_subset_rows(ensemble, split, arrays, routes))
            if ensemble == "baseline":
                oracle_all.extend(oracle_rows(ensemble, split, arrays))

            route_snapshots.mkdir(parents=True, exist_ok=True)
            with gzip.open(route_snapshots / f"{ensemble}_{split}_routing_minimal.csv.gz", "wt", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_index", "ensemble", "split", "snr", "label", "selected_expert", "final_pred"])
                for idx in range(labels.size):
                    writer.writerow([idx, ensemble, split, arrays["metadata"][idx, 0], int(labels[idx]), int(selected[idx]), int(final_pred[idx])])

    finetune_comparison = build_finetune_comparison(metrics, routed_rows)
    previous_snr_comparison = compare_previous_snr_exports(snr_rounded_rows, metrics)
    change_rows = build_change_rows(metrics, oracle_all)
    ckpt_rows = checkpoint_rows()

    write_csv(OUTPUT_DIR / "full_metrics.csv", metrics)
    write_csv(OUTPUT_DIR / "snr_bin_metrics.csv", snr_rows)
    write_csv(OUTPUT_DIR / "snr_rounded_metrics.csv", snr_rounded_rows)
    write_csv(OUTPUT_DIR / "routed_subset_metrics.csv", routed_rows)
    write_csv(OUTPUT_DIR / "oracle_optimal_cascade_metrics.csv", oracle_all)
    write_csv(OUTPUT_DIR / "finetune_before_after_comparison.csv", finetune_comparison)
    write_csv(OUTPUT_DIR / "previous_snr_test_export_comparison.csv", previous_snr_comparison)
    write_csv(OUTPUT_DIR / "alt_to_new_changes.csv", change_rows)
    write_csv(OUTPUT_DIR / "checkpoint_manifest.csv", ckpt_rows)

    write_json(
        OUTPUT_DIR / "summary.json",
        {
            "source_eval_dir": SOURCE_DIR,
            "thresholds": thresholds,
            "checkpoints": ckpt_rows,
            "changes": change_rows,
            "previous_snr_test_export_comparison_rows": len(previous_snr_comparison),
            "note": "Historical architecture-search results are not overwritten; this export documents the final checkpoint used in the baseline ensemble.",
            "selection_reason_for_014": "No explicit documented selection rationale was found in the searched repository artifacts.",
        },
    )

    full_plot_rows = [r for r in metrics if r["split"] == "test" and r["model"] in {"ensemble", "f_small"}]
    plot_metric_bars(plot_dir / "test_accuracy_final_models_and_ensembles.png", full_plot_rows, "Final test accuracy")
    baseline_snr = [r for r in snr_rows if r["ensemble"] == "baseline" and r["split"] == "test"]
    plot_snr_curves(plot_dir / "baseline_test_snr_bin_accuracy.png", baseline_snr, "Baseline final checkpoint 014: test SNR-bin accuracy")
    oracle_plot_rows = [r for r in oracle_all if r["split"] == "val"]
    plot_metric_bars(plot_dir / "baseline_val_oracle_bounds.png", oracle_plot_rows, "Baseline final checkpoint 014: validation oracle bounds")

    print(f"Wrote final thesis evaluation export to {OUTPUT_DIR}")
    print("Key changes:")
    for row in change_rows:
        old = float(row["old_value"])
        new = float(row["new_value"])
        print(f"- {row['affected_statement']} ({row['split']}): {old:.12f} -> {new:.12f} ({new - old:+.12f})")


if __name__ == "__main__":
    main()
