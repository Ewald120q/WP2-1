from __future__ import annotations

import csv
import gzip
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from evaluate_final_rejection_ensembles import FINAL_CHECKPOINTS, rejector_targets, route_with_thresholds, sha256_file

SOURCE_FINAL = THIS_DIR / "artifacts" / "final_rejection_ensemble_eval"
SOURCE_014 = THIS_DIR / "artifacts" / "final_014_thesis_eval"
OUT = THIS_DIR / "artifacts" / "thesis_consistency_final"
FIG = OUT / "figures"
ZIP_PATH = THIS_DIR / "artifacts" / "thesis_consistency_review.zip"

ENSEMBLES = ("baseline", "finetuned_replay", "joint_training")
SPLITS = ("val", "test")
TARGET_RULES = ("current_wrong", "current_wrong_next_correct")
REJECTORS = ("r1", "r2")
ROUTE_LABELS = ("f_small", "f_mid", "f_large")
ROUTE_NAMES = ("R1 accepted", "R2 accepted", "R2 rejected")
PRED_KEYS = {"f_small": "small_pred", "f_mid": "mid_pred", "f_large": "large_pred"}

COLORS = {
    "baseline": "#4c78a8",
    "finetuned_replay": "#f58518",
    "joint_training": "#54a24b",
    "current_wrong": "#4c78a8",
    "current_wrong_next_correct": "#b279a2",
}

PERCENT = PercentFormatter(xmax=1.0, decimals=0)
COMMA = FuncFormatter(lambda x, _pos: f"{x:.2f}".replace(".", ","))
COMMA_ONE = FuncFormatter(lambda value, pos: f"{value:.1f}".replace(".", ","))


plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 0.9,
        "lines.markersize": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(clean(payload), handle, indent=2, sort_keys=True)


def as_float_rows(rows: list[dict[str, str]], text_keys: set[str]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if key in text_keys or value == "":
                item[key] = value
                continue
            try:
                if any(ch in value for ch in ".eE"):
                    item[key] = float(value)
                else:
                    item[key] = int(value)
            except ValueError:
                item[key] = value
        converted.append(item)
    return converted


def load_arrays(ensemble: str, split: str) -> dict[str, np.ndarray]:
    with np.load(SOURCE_FINAL / "cache" / f"{ensemble}_{split}.npz") as data:
        return {key: data[key] for key in data.files}


def thresholds() -> dict[str, dict[str, float]]:
    rows = read_csv(SOURCE_FINAL / "thresholds.csv")
    return {
        row["ensemble"]: {
            "r1_threshold": float(row["r1_threshold"]),
            "r2_threshold": float(row["r2_threshold"]),
        }
        for row in rows
    }


def rejector_metric_rows(thr: dict[str, dict[str, float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for ensemble in ENSEMBLES:
        for split in SPLITS:
            arrays = load_arrays(ensemble, split)
            routes = route_with_thresholds(arrays, thr[ensemble]["r1_threshold"], thr[ensemble]["r2_threshold"])
            routes["r1_threshold"] = thr[ensemble]["r1_threshold"]
            routes["r2_threshold"] = thr[ensemble]["r2_threshold"]
            for rejector in REJECTORS:
                for target_rule in TARGET_RULES:
                    y_true, scores, threshold = rejector_targets(arrays, routes, rejector, target_rule)
                    pred = scores >= threshold
                    positives = int((y_true == 1).sum())
                    negatives = int((y_true == 0).sum())
                    tp = int(((y_true == 1) & pred).sum())
                    fp = int(((y_true == 0) & pred).sum())
                    fn = int(((y_true == 1) & ~pred).sum())
                    tn = int(((y_true == 0) & ~pred).sum())
                    precision = tp / (tp + fp) if (tp + fp) else 0.0
                    recall = tp / positives if positives else math.nan
                    f1 = 2 * precision * recall / (precision + recall) if positives and (precision + recall) else 0.0
                    ap = average_precision_score(y_true, scores) if positives and negatives else math.nan
                    roc_auc = roc_auc_score(y_true, scores) if positives and negatives else math.nan
                    rows.append(
                        {
                            "ensemble": ensemble,
                            "split": split,
                            "rejector": rejector,
                            "target_rule": target_rule,
                            "target_definition": (
                                "current classifier wrong"
                                if target_rule == "current_wrong"
                                else "current classifier wrong and at least one following classifier correct"
                            ),
                            "scope": "full split" if rejector == "r1" else "samples forwarded by r1",
                            "n_samples": int(y_true.size),
                            "n_positive": positives,
                            "n_negative": negatives,
                            "prevalence": positives / y_true.size if y_true.size else math.nan,
                            "threshold_from_val": threshold,
                            "average_precision": ap,
                            "roc_auc": roc_auc,
                            "precision_at_threshold": precision,
                            "recall_at_threshold": recall,
                            "f1_at_threshold": f1,
                            "tp_at_threshold": tp,
                            "fp_at_threshold": fp,
                            "fn_at_threshold": fn,
                            "tn_at_threshold": tn,
                        }
                    )
                    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, scores)
                    for idx in range(precision_curve.shape[0]):
                        curve_rows.append(
                            {
                                "ensemble": ensemble,
                                "split": split,
                                "rejector": rejector,
                                "target_rule": target_rule,
                                "curve": "precision_recall",
                                "point_index": idx,
                                "precision": float(precision_curve[idx]),
                                "recall": float(recall_curve[idx]),
                                "threshold": float(pr_thresholds[idx]) if idx < pr_thresholds.shape[0] else math.inf,
                            }
                        )
                    if positives and negatives:
                        fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
                        for idx in range(fpr.shape[0]):
                            curve_rows.append(
                                {
                                    "ensemble": ensemble,
                                    "split": split,
                                    "rejector": rejector,
                                    "target_rule": target_rule,
                                    "curve": "roc",
                                    "point_index": idx,
                                    "fpr": float(fpr[idx]),
                                    "tpr": float(tpr[idx]),
                                    "threshold": float(roc_thresholds[idx]),
                                }
                            )
    return rows, curve_rows


def latency_rows(thr: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ensemble in ENSEMBLES:
        for split in SPLITS:
            arrays = load_arrays(ensemble, split)
            routes = route_with_thresholds(arrays, thr[ensemble]["r1_threshold"], thr[ensemble]["r2_threshold"])
            selected = routes["selected_expert"]
            x1 = float((selected != 0).mean())
            reached_r2 = selected != 0
            x2 = float((selected == 2).sum() / reached_r2.sum()) if reached_r2.any() else 0.0
            rows.append(
                {
                    "ensemble": ensemble,
                    "split": split,
                    "formula": "0.54 + 1.24*x1 + 2.38*x1*x2",
                    "x1_r1_reject_fraction": x1,
                    "x2_r2_conditional_reject_fraction": x2,
                    "usage_small_fraction": float((selected == 0).mean()),
                    "usage_mid_fraction": float((selected == 1).mean()),
                    "usage_large_fraction": float((selected == 2).mean()),
                    "latency_ms": 0.54 + 1.24 * x1 + 2.38 * x1 * x2,
                }
            )
    return rows


def simple_classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    labels_bool = labels.astype(bool)
    preds_bool = preds.astype(bool)
    tp = int((labels_bool & preds_bool).sum())
    tn = int((~labels_bool & ~preds_bool).sum())
    fp = int((~labels_bool & preds_bool).sum())
    fn = int((labels_bool & ~preds_bool).sum())
    n = int(labels_bool.size)
    accuracy = (tp + tn) / n if n else math.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_samples": n,
        "n_positive": int(labels_bool.sum()),
        "n_negative": int((~labels_bool).sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def fixed_baseline_subset_comparison(thr: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        baseline = load_arrays("baseline", split)
        finetuned = load_arrays("finetuned_replay", split)
        routes = route_with_thresholds(baseline, thr["baseline"]["r1_threshold"], thr["baseline"]["r2_threshold"])
        selected = routes["selected_expert"]
        labels = baseline["labels"]
        for idx, route_name in enumerate(ROUTE_NAMES):
            model = ROUTE_LABELS[idx]
            pred_key = PRED_KEYS[model]
            mask = selected == idx
            before = simple_classification_metrics(labels[mask], baseline[pred_key][mask])
            after = simple_classification_metrics(labels[mask], finetuned[pred_key][mask])
            row = {
                "split": split,
                "fixed_subset_source": "baseline_route_with_baseline_val_thresholds",
                "routing_group": route_name,
                "model": model,
                "baseline_r1_threshold": thr["baseline"]["r1_threshold"],
                "baseline_r2_threshold": thr["baseline"]["r2_threshold"],
                "n_samples": before["n_samples"],
                "n_positive": before["n_positive"],
                "n_negative": before["n_negative"],
            }
            for prefix, metrics in (("baseline", before), ("finetuned_replay", after)):
                for key, value in metrics.items():
                    row[f"{prefix}_{key}"] = value
            for metric in ("accuracy", "precision", "recall", "f1"):
                row[f"delta_{metric}"] = after[metric] - before[metric]
                row[f"delta_{metric}_percentage_points"] = (after[metric] - before[metric]) * 100
            rows.append(row)
    return rows


def checkpoint_manifest() -> list[dict[str, Any]]:
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
                    "sha256": sha256_file(path) if path.exists() else "",
                    "status": "final",
                }
            )
    historical = THIS_DIR / "checkpoints_new/ch_point_DM_time_binary_classificator_241002_3_GAP_256/prot-DM_time_binary_classificator_241002_3_GAP-009-0.771-0.742.pth"
    rows.append(
        {
            "ensemble": "historical_single_model_candidate",
            "role": "f_small",
            "path": str(historical),
            "exists": historical.exists(),
            "sha256": sha256_file(historical) if historical.exists() else "",
            "status": "historical_not_final",
        }
    )
    return rows


def plot_pr_comparison(rows: list[dict[str, Any]], metrics: list[dict[str, Any]], output: Path, split: str = "test") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    metric_lookup = {(r["ensemble"], r["split"], r["rejector"], r["target_rule"]): r for r in metrics}
    for ax, rejector in zip(axes, REJECTORS):
        for ensemble in ("baseline", "joint_training"):
            points = [
                r
                for r in rows
                if r["curve"] == "precision_recall"
                and r["split"] == split
                and r["rejector"] == rejector
                and r["target_rule"] == "current_wrong"
                and r["ensemble"] == ensemble
            ]
            points.sort(key=lambda r: int(r["point_index"]))
            metric = metric_lookup[(ensemble, split, rejector, "current_wrong")]
            ax.plot(
                [p["recall"] for p in points],
                [p["precision"] for p in points],
                linewidth=1.0,
                color=COLORS[ensemble],
                label=f"{ensemble.replace('_', ' ')} AP = {metric['average_precision']:.3f}".replace(".", ","),
            )
            ax.axhline(
                metric["prevalence"],
                color=COLORS[ensemble],
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                label=f"Zufallsreferenz {ensemble.replace('_', ' ')} = {metric['prevalence']:.3f}".replace(".", ","),
            )
        ax.set_title(r"$r_1$" if rejector == "r1" else r"$r_2$", pad=3)
        ax.set_xlabel("Recall", labelpad=2)
        ax.set_ylabel("Precision", labelpad=2)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_formatter(COMMA_ONE)
        ax.yaxis.set_major_formatter(COMMA_ONE)
        ax.grid(axis="both", color="0.90", linewidth=0.7, linestyle="-")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower left", frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.91, bottom=0.20, wspace=0.22)
    fig.savefig(output)
    plt.close(fig)


def plot_pr_grid(rows: list[dict[str, Any]], metrics: list[dict[str, Any]], output: Path, split: str, target_rule: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    metric_lookup = {(r["ensemble"], r["split"], r["rejector"], r["target_rule"]): r for r in metrics}
    for ax, rejector in zip(axes, REJECTORS):
        for ensemble in ENSEMBLES:
            points = [
                r
                for r in rows
                if r["curve"] == "precision_recall"
                and r["split"] == split
                and r["rejector"] == rejector
                and r["target_rule"] == target_rule
                and r["ensemble"] == ensemble
            ]
            points.sort(key=lambda r: int(r["point_index"]))
            metric = metric_lookup[(ensemble, split, rejector, target_rule)]
            ax.plot(
                [p["recall"] for p in points],
                [p["precision"] for p in points],
                linewidth=1.0,
                color=COLORS[ensemble],
                label=f"{ensemble.replace('_', ' ')} AP = {metric['average_precision']:.3f}".replace(".", ","),
            )
            ax.axhline(
                metric["prevalence"],
                color=COLORS[ensemble],
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                label=f"Zufallsreferenz {ensemble.replace('_', ' ')} = {metric['prevalence']:.3f}".replace(".", ","),
            )
        ax.set_title(r"$r_1$" if rejector == "r1" else r"$r_2$", pad=3)
        ax.set_xlabel("Recall", labelpad=2)
        ax.set_ylabel("Precision", labelpad=2)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_formatter(COMMA_ONE)
        ax.yaxis.set_major_formatter(COMMA_ONE)
        ax.grid(axis="both", color="0.90", linewidth=0.7, linestyle="-")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower left", frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.91, bottom=0.20, wspace=0.22)
    fig.savefig(output)
    plt.close(fig)


def plot_snr(output: Path, rows: list[dict[str, Any]], split: str, model: str = "ensemble") -> None:
    wanted_bins = ["Rest of Events", "SNR < 3", "3.x", "4.x", "5.x", "6.x", "7.x", "8.x", "9+"]
    fig, ax = plt.subplots(figsize=(6.80, 2.65))
    x = np.arange(len(wanted_bins))
    for ensemble in ENSEMBLES:
        data = {
            r["snr_bin"]: r
            for r in rows
            if r["ensemble"] == ensemble and r["split"] == split and r["model"] == model and r.get("binning") == "thesis"
        }
        y = [float(data[label]["accuracy"]) if label in data and int(data[label]["n_samples"]) else np.nan for label in wanted_bins]
        ax.plot(x, y, marker="o", color=COLORS[ensemble], label=ensemble.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(wanted_bins, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"{model} {split} nach SNR-Bin")
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_routing(output: Path, full_metrics: list[dict[str, Any]], split: str) -> None:
    fig, ax = plt.subplots(figsize=(6.80, 2.55))
    x = np.arange(len(ENSEMBLES))
    bottoms = np.zeros(len(ENSEMBLES))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, label in enumerate(("small", "mid", "large")):
        vals = []
        for ensemble in ENSEMBLES:
            row = next(r for r in full_metrics if r["ensemble"] == ensemble and r["split"] == split and r["model"] == "ensemble")
            vals.append(float(row[f"usage_{label}_fraction"]))
        ax.bar(x, vals, bottom=bottoms, color=colors[idx], label=f"f_{label}")
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_", " ") for e in ENSEMBLES], rotation=20, ha="right")
    ax.set_ylabel("Modellanteil")
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_ylim(0, 1)
    ax.set_title(f"Routinganteile {split}")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_thresholds(output: Path, thr_rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(6.80, 2.55))
    x = np.arange(len(thr_rows))
    width = 0.35
    ax.bar(x - width / 2, [float(r["r1_threshold"]) for r in thr_rows], width, color="#4c78a8", label="r1")
    ax.bar(x + width / 2, [float(r["r2_threshold"]) for r in thr_rows], width, color="#f58518", label="r2")
    ax.set_xticks(x)
    ax.set_xticklabels([r["ensemble"].replace("_", " ") for r in thr_rows], rotation=20, ha="right")
    ax.set_ylabel("Threshold")
    ax.yaxis.set_major_formatter(COMMA)
    ax.set_title("Val-kalibrierte Thresholds")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_latency(output: Path, latency: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.80, 2.55))
    test_rows = [r for r in latency if r["split"] == "test"]
    x = np.arange(len(test_rows))
    ax.bar(x, [r["latency_ms"] for r in test_rows], color=[COLORS[r["ensemble"]] for r in test_rows])
    ax.set_xticks(x)
    ax.set_xticklabels([r["ensemble"].replace("_", " ") for r in test_rows], rotation=20, ha="right")
    ax.set_ylabel("Latenz [ms]")
    ax.yaxis.set_major_formatter(COMMA)
    ax.set_title("Test-Latenz aus aktuellen Routinganteilen")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_r1_reject_rate_latency(output: Path) -> None:
    reject_rate = np.linspace(0.0, 1.0, 500)
    latency_ms = 0.54 + 0.92 * reject_rate

    fig, ax = plt.subplots(figsize=(3.6, 2.3), constrained_layout=True)
    ax.plot(reject_rate, latency_ms, color="#1f78b4", linewidth=1.0)
    ax.axvline(0.30, color="0.35", linestyle="--", linewidth=1.0)

    ax.set_xlabel("Reject-Quote", labelpad=2)
    ax.set_ylabel("Latenz [ms]", labelpad=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.55, 1.50)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(COMMA_ONE)
    ax.grid(axis="both", color="0.90", linewidth=0.7, linestyle="-")

    fig.savefig(output)
    plt.close(fig)


def thesis_changes(
    full: list[dict[str, Any]],
    rejector: list[dict[str, Any]],
    latency: list[dict[str, Any]],
    fixed_subset: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_metric = {(r["ensemble"], r["split"], r["model"]): r for r in full}
    rej = {(r["ensemble"], r["split"], r["rejector"], r["target_rule"]): r for r in rejector}
    lat = {(r["ensemble"], r["split"]): r for r in latency}
    fixed = {(r["split"], r["routing_group"], r["model"]): r for r in fixed_subset}
    rows = [
        {
            "fundstelle": "Manuskript/Einzelmodellkapitel; alter Wert aus historischem Checkpoint 009",
            "alt": "f_small Val 74,21 %, Test 75,94 %",
            "neu": f"f_small Val {by_metric[('baseline','val','f_small')]['accuracy']*100:.6f} %, Test {by_metric[('baseline','test','f_small')]['accuracy']*100:.6f} %",
            "begruendung": "Finales Baseline-Ensemble verwendet f_small-Checkpoint 014, nicht 009.",
            "quelle": "full_metrics.csv, checkpoint_manifest.csv",
            "status": "ersetzen",
        },
        {
            "fundstelle": "analyse_rejectors.ipynb cell 20; rejector_analysis_new_plots.ipynb cell 21",
            "alt": "full_validation_accuracy['f_small'] = 0.7421",
            "neu": f"{by_metric[('baseline','val','f_small')]['accuracy']:.15f}",
            "begruendung": "Hart codierter Einzelmodellwert passt zum historischen Checkpoint 009, nicht zur finalen Baseline.",
            "quelle": "full_metrics.csv",
            "status": "ersetzen",
        },
        {
            "fundstelle": "analyse_rejectors_finetune.ipynb cell 20",
            "alt": "full_validation_accuracy['f_small'] = 0.7421",
            "neu": f"{by_metric[('finetuned_replay','val','f_small')]['accuracy']:.15f}",
            "begruendung": "Finetune-Routing nutzt den finalen finetuned f_small-Checkpoint; Referenzwert muss berechnet werden.",
            "quelle": "full_metrics.csv",
            "status": "ersetzen",
        },
        {
            "fundstelle": "Manuskript/Ensemble-Testvergleich",
            "alt": "Verbesserungen uneinheitlich/teilweise gerundet",
            "neu": (
                f"Baseline {by_metric[('baseline','test','ensemble')]['accuracy']*100:.6f} %, "
                f"Finetuning {by_metric[('finetuned_replay','test','ensemble')]['accuracy']*100:.6f} %, "
                f"Joint {by_metric[('joint_training','test','ensemble')]['accuracy']*100:.6f} %; "
                f"Delta Finetuning-Baseline {(by_metric[('finetuned_replay','test','ensemble')]['accuracy'] - by_metric[('baseline','test','ensemble')]['accuracy'])*100:+.6f} pp, "
                f"Delta Joint-Baseline {(by_metric[('joint_training','test','ensemble')]['accuracy'] - by_metric[('baseline','test','ensemble')]['accuracy'])*100:+.6f} pp"
            ),
            "begruendung": "Differenzen aus ungerundeten Werten; Val-kalibrierte Thresholds unverändert auf Test.",
            "quelle": "full_metrics.csv",
            "status": "ersetzen",
        },
        {
            "fundstelle": "Manuskript/PR-Aussagen und finale PR-Abbildung 9.34",
            "alt": "AP/PR-AUC uneinheitlich, teils alte Zieldefinition oder doppelte Abbildung",
            "neu": (
                f"Test AP current_wrong: Baseline r1/r2 {rej[('baseline','test','r1','current_wrong')]['average_precision']:.6f}/"
                f"{rej[('baseline','test','r2','current_wrong')]['average_precision']:.6f}; "
                f"Finetuning {rej[('finetuned_replay','test','r1','current_wrong')]['average_precision']:.6f}/"
                f"{rej[('finetuned_replay','test','r2','current_wrong')]['average_precision']:.6f}; "
                f"Joint {rej[('joint_training','test','r1','current_wrong')]['average_precision']:.6f}/"
                f"{rej[('joint_training','test','r2','current_wrong')]['average_precision']:.6f}"
            ),
            "begruendung": "Hauptdefinition ist current_wrong; current_wrong_next_correct bleibt separat als Erklaerungsziel.",
            "quelle": "rejector_metrics.csv, figures/final_pr_figure_9_34_baseline_joint_current_wrong.pdf",
            "status": "ersetzen",
        },
        {
            "fundstelle": "Manuskript/Latenzabschnitt",
            "alt": "Latenzen mit alten oder gerundeten Anteilen",
            "neu": (
                f"Formel 0.54 + 1.24*x1 + 2.38*x1*x2; Test-Latenz Baseline {lat[('baseline','test')]['latency_ms']:.6f} ms, "
                f"Finetuning {lat[('finetuned_replay','test')]['latency_ms']:.6f} ms, "
                f"Joint {lat[('joint_training','test')]['latency_ms']:.6f} ms"
            ),
            "begruendung": "x1/x2 aus aktuellen Test-Routinganteilen mit Val-Thresholds.",
            "quelle": "latency_check.csv",
            "status": "ersetzen",
        },
        {
            "fundstelle": "Manuskript/Oracle-Kaskadenanalyse; evaluate.ipynb cells 15/16/18",
            "alt": "97,14 % und 99,93 % ohne klare Trennung von echter Kaskade",
            "neu": "97,144717 % = Oracle small_or_mid; 99,933733 % = Oracle any_three; beide Val, finale Baseline-Experten.",
            "begruendung": "Keine Rejectorleistung, sondern obere Grenze aus Labels und finalen Expertenvorhersagen.",
            "quelle": "oracle_optimal_cascade_metrics.csv",
            "status": "praezisieren",
        },
        {
            "fundstelle": "Manuskript S. 69 und S. 95; Finetuning-Vergleich auf festen Baseline-Gruppen",
            "alt": "+0,12/+0,41 Prozentpunkte sowie 81,3/99,3/99,08 % aus bisher nicht identischer Teilmenge",
            "neu": (
                "Test auf festen Baseline-Routen: "
                f"R1 accepted f_small {fixed[('test','R1 accepted','f_small')]['baseline_accuracy']*100:.6f} % -> "
                f"{fixed[('test','R1 accepted','f_small')]['finetuned_replay_accuracy']*100:.6f} % "
                f"({fixed[('test','R1 accepted','f_small')]['delta_accuracy_percentage_points']:+.6f} pp); "
                f"R2 accepted f_mid {fixed[('test','R2 accepted','f_mid')]['baseline_accuracy']*100:.6f} % -> "
                f"{fixed[('test','R2 accepted','f_mid')]['finetuned_replay_accuracy']*100:.6f} % "
                f"({fixed[('test','R2 accepted','f_mid')]['delta_accuracy_percentage_points']:+.6f} pp); "
                f"R2 rejected f_large {fixed[('test','R2 rejected','f_large')]['baseline_accuracy']*100:.6f} % -> "
                f"{fixed[('test','R2 rejected','f_large')]['finetuned_replay_accuracy']*100:.6f} % "
                f"({fixed[('test','R2 rejected','f_large')]['delta_accuracy_percentage_points']:+.6f} pp)"
            ),
            "begruendung": "Beide Modellstaende wurden auf exakt denselben Sample-IDs aus dem Baseline-Routing ausgewertet; varianteneigene Routen bleiben separat.",
            "quelle": "finetune_fixed_baseline_subset_comparison.csv",
            "status": "ersetzen_pruefen",
        },
    ]
    return rows


def figure_manifest() -> list[dict[str, Any]]:
    return [
        {
            "old_figure": "Abb. 9.34 / bisherige finale PR-Abbildung",
            "new_pdf": "figures/final_pr_figure_9_34_baseline_joint_current_wrong.pdf",
            "plot_source": "generate_thesis_consistency_final.py",
            "data_source": "final_rejection_ensemble_eval/cache/*.npz via script; rejector_metrics.csv; rejector_curve_points_sampled.csv for review",
            "status": "ersetzen",
        },
        {
            "old_figure": "SNR-Testplots der finalen Ensembles",
            "new_pdf": "figures/final_snr_test_ensemble_accuracy.pdf",
            "plot_source": "generate_thesis_consistency_final.py",
            "data_source": "snr_bin_metrics.csv",
            "status": "ersetzen_falls_im_manuskript_verwendet",
        },
        {
            "old_figure": "Routinganteile finaler Ensembles",
            "new_pdf": "figures/final_routing_test_usage.pdf",
            "plot_source": "generate_thesis_consistency_final.py",
            "data_source": "full_metrics.csv",
            "status": "ersetzen_falls_im_manuskript_verwendet",
        },
        {
            "old_figure": "Threshold-/Reject-Rate-Darstellung",
            "new_pdf": "figures/final_thresholds_val_calibrated.pdf",
            "plot_source": "generate_thesis_consistency_final.py",
            "data_source": "thresholds.csv",
            "status": "ersetzen_falls_im_manuskript_verwendet",
        },
        {
            "old_figure": "Abb. 9.28",
            "new_pdf": "",
            "plot_source": "",
            "data_source": "",
            "status": "offen: Thesis-Quelle/Abbildungsdatei mit Nummerierung nicht im Repository gefunden; Wiederholung von Abb. 9.25 nicht automatisch pruefbar",
        },
    ]


def write_markdown(full: list[dict[str, Any]], rejector: list[dict[str, Any]], latency: list[dict[str, Any]], open_items: list[str]) -> None:
    baseline_test = next(r for r in full if r["ensemble"] == "baseline" and r["split"] == "test" and r["model"] == "ensemble")
    ft_test = next(r for r in full if r["ensemble"] == "finetuned_replay" and r["split"] == "test" and r["model"] == "ensemble")
    joint_test = next(r for r in full if r["ensemble"] == "joint_training" and r["split"] == "test" and r["model"] == "ensemble")
    readme = f"""# Thesis Consistency Final

Reproduktionsbefehl:

```bash
cd {THIS_DIR}
python generate_thesis_consistency_final.py
```

Grundlage: finale Inferenzartefakte aus `artifacts/final_rejection_ensemble_eval/`
und `artifacts/final_014_thesis_eval/`. Es wurden keine Trainingslaeufe gestartet
und keine Ensemble-Checkpoints geaendert.

Finale Test-Accuracy:

- Baseline: {baseline_test['accuracy'] * 100:.6f} %
- Replay-Finetuning: {ft_test['accuracy'] * 100:.6f} %
- Joint: {joint_test['accuracy'] * 100:.6f} %

Verbesserungen aus ungerundeten Werten:

- Finetuning - Baseline: {(ft_test['accuracy'] - baseline_test['accuracy']) * 100:+.6f} Prozentpunkte
- Joint - Baseline: {(joint_test['accuracy'] - baseline_test['accuracy']) * 100:+.6f} Prozentpunkte

`f_small` der finalen Baseline ist Checkpoint `014-0.764-0.740`.
Der historische Checkpoint `009-0.771-0.742` bleibt getrennt als alter
Standalone-Kandidat dokumentiert.

Offene Punkte stehen in `open_issues.md`.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    captions = """# Caption Suggestions

- Finale PR-Kurven: Variante=Baseline/Joint, Split=Test, Teilmenge=r1 gesamter Split bzw. r2 von r1 weitergeleitete Samples, Metrik=Precision-Recall, Target=current_wrong; gestrichelte Linien zeigen die jeweilige Zufallsreferenz/Praevalenz.
- Finale SNR-Metriken: Variante=Baseline/Finetuning/Joint, Split=Val oder Test, Modell/Ensemble explizit nennen, Binning=`Rest of Events` fuer nicht-finite SNR-Werte, `SNR < 3`, `3.x` bis `8.x`, `9+` nach gerundeter SNR-Zuordnung, Bezugsgröße=Samples im jeweiligen Bin.
- Routing-Teilmengen: Variante und Split nennen; `R1 accepted` wird mit `f_small`, `R2 accepted` mit `f_mid`, `R2 rejected` mit `f_large` ausgewertet; Samplezahlen und Confusion Counts angeben.
- Thresholds: Val-kalibriert auf 70/21/9 Modellanteile, ungerundet auf Test angewendet; r2-Threshold bezieht sich auf die von r1 weitergeleitete Teilmenge.
- Oracle-Werte: als obere Grenze kennzeichnen, nicht als gemessene Rejector-Kaskade; `small_or_mid` und `any_three` separat nennen.
- Latenz: Formel `0.54 + 1.24*x1 + 2.38*x1*x2`; `x1` ist r1-Weiterleitungsanteil, `x2` r2-Weiterleitungsanteil konditional auf r1.
"""
    (OUT / "caption_suggestions.md").write_text(captions, encoding="utf-8")
    (OUT / "open_issues.md").write_text("# Open Issues\n\n" + "\n".join(f"- {item}" for item in open_items) + "\n", encoding="utf-8")


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in OUT.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in {".npz", ".pth"}:
                continue
            if path.name == "rejector_curve_points.csv":
                continue
            archive.write(path, path.relative_to(OUT.parent))


def sampled_curve_rows(rows: list[dict[str, Any]], max_points_per_group: int = 500) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["ensemble"], row["split"], row["rejector"], row["target_rule"], row["curve"])
        grouped.setdefault(key, []).append(row)
    sampled: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: int(row["point_index"]))
        if len(group_rows) <= max_points_per_group:
            sampled.extend(group_rows)
            continue
        indices = np.linspace(0, len(group_rows) - 1, max_points_per_group, dtype=int)
        sampled.extend(group_rows[int(idx)] for idx in indices)
    return sampled


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), OUT / "generate_thesis_consistency_final.py")
    shutil.copy2(THIS_DIR / "export_final_014_thesis_eval.py", OUT / "export_final_014_thesis_eval.py")
    shutil.copy2(THIS_DIR / "evaluate_final_rejection_ensembles.py", OUT / "evaluate_final_rejection_ensembles.py")

    full = as_float_rows(read_csv(SOURCE_014 / "full_metrics.csv"), {"ensemble", "split", "model"})
    snr = as_float_rows(read_csv(SOURCE_014 / "snr_bin_metrics.csv"), {"ensemble", "split", "model", "binning", "snr_bin"})
    routed = as_float_rows(read_csv(SOURCE_014 / "routed_subset_metrics.csv"), {"ensemble", "split", "routing_group", "model"})
    oracle = as_float_rows(read_csv(SOURCE_014 / "oracle_optimal_cascade_metrics.csv"), {"ensemble", "split", "oracle_variant", "description"})
    threshold_rows = read_csv(SOURCE_FINAL / "thresholds.csv")
    threshold_payload = thresholds()
    rejector, curves = rejector_metric_rows(threshold_payload)
    latency = latency_rows(threshold_payload)
    fixed_subset = fixed_baseline_subset_comparison(threshold_payload)
    checkpoints = checkpoint_manifest()
    changes = thesis_changes(full, rejector, latency, fixed_subset)
    figures = figure_manifest()

    write_csv(OUT / "full_metrics.csv", full)
    write_csv(OUT / "snr_bin_metrics.csv", snr)
    shutil.copy2(SOURCE_014 / "snr_rounded_metrics.csv", OUT / "snr_rounded_metrics.csv")
    write_csv(OUT / "routed_subset_metrics.csv", routed)
    write_csv(OUT / "oracle_optimal_cascade_metrics.csv", oracle)
    write_csv(OUT / "rejector_metrics.csv", rejector)
    stale_curve_csv = OUT / "rejector_curve_points.csv"
    if stale_curve_csv.exists():
        stale_curve_csv.unlink()
    write_csv(OUT / "rejector_curve_points_sampled.csv", sampled_curve_rows(curves))
    write_csv(OUT / "thresholds.csv", threshold_rows)
    write_csv(OUT / "latency_check.csv", latency)
    shutil.copy2(SOURCE_014 / "finetune_before_after_comparison.csv", OUT / "finetune_before_after_comparison.csv")
    write_csv(OUT / "finetune_fixed_baseline_subset_comparison.csv", fixed_subset)
    shutil.copy2(SOURCE_014 / "previous_snr_test_export_comparison.csv", OUT / "previous_snr_test_export_comparison.csv")
    write_csv(OUT / "checkpoint_manifest.csv", checkpoints)
    write_csv(OUT / "thesis_changes.csv", changes)
    write_csv(OUT / "figure_manifest.csv", figures)

    plot_pr_comparison(curves, rejector, FIG / "final_pr_figure_9_34_baseline_joint_current_wrong.pdf")
    for split in SPLITS:
        for target_rule in TARGET_RULES:
            plot_pr_grid(curves, rejector, FIG / f"final_pr_{split}_{target_rule}.pdf", split, target_rule)
        plot_snr(FIG / f"final_snr_{split}_ensemble_accuracy.pdf", snr, split, "ensemble")
        plot_snr(FIG / f"final_snr_{split}_f_small_accuracy.pdf", snr, split, "f_small")
        plot_routing(FIG / f"final_routing_{split}_usage.pdf", full, split)
    plot_thresholds(FIG / "final_thresholds_val_calibrated.pdf", threshold_rows)
    plot_latency(FIG / "final_latency_test.pdf", latency)
    plot_r1_reject_rate_latency(FIG / "final_r1_reject_rate_latency.pdf")

    open_items = [
        "Keine Thesis-Quellen (.tex/.typ/.docx) mit Abbildungsnummern im Repository gefunden; Manuskriptstellen koennen nur als Aussagen/Fundstellen in Notebooks/Exports markiert werden.",
        "Kein dokumentierter Auswahlgrund fuer Checkpoint 014 gefunden; nur die finale Verdrahtung in final_checkpoints und Eval-Notebooks ist belegt.",
        "Abb. 9.28 als moegliche Wiederholung von Abb. 9.25 ist ohne Thesis-Quelle oder nummerierte Referenz-PDF nicht automatisch pruefbar.",
        "Uneinheitliches Wording 'Baseline-Rejection-Ensemble', Noise versus Rest of Events, falsche Verweise, ungeloeste Zitate, doppelter Absatz S. 100 und Prozentpunktnotation koennen mangels Thesis-Quelle nicht direkt gepatcht werden.",
        "SNR-Maximum 68/70,2 tritt in alten Notebook-Ausgaben auf; aktuelle finale Exporte verwenden gespeichertes TransientX-SNR in den Dataset-Metadaten, gerundete SNR-Bins und `Rest of Events` nur fuer nicht-finite SNR-Werte.",
        "Replay-Auswahl ist in Finetuning-Notebooks/Artefakten sichtbar, aber keine Manuskriptpassage zur inhaltlichen Auswahlbegruendung wurde gefunden.",
        "Visuelle PDF-Pruefung war im Runtime-Environment nur eingeschraenkt moeglich, weil kein PDF-Renderer (`pdftoppm`, `pdfinfo`, PyPDF/PyMuPDF) installiert ist; Dateien wurden als einseitige gueltige PDFs validiert und mit `tight_layout()` erzeugt.",
    ]
    write_markdown(full, rejector, latency, open_items)

    write_json(
        OUT / "results.json",
        {
            "full_metrics": full,
            "snr_bin_metrics_file": "snr_bin_metrics.csv",
            "routed_subset_metrics": routed,
            "oracle_optimal_cascade_metrics": oracle,
            "rejector_metrics": rejector,
            "thresholds": threshold_rows,
            "latency": latency,
            "finetune_fixed_baseline_subset_comparison": fixed_subset,
            "checkpoints": checkpoints,
            "thesis_changes": changes,
            "figure_manifest": figures,
            "open_issues": open_items,
        },
    )
    make_zip()
    print(f"Wrote {OUT}")
    print(f"Wrote {ZIP_PATH}")


if __name__ == "__main__":
    main()
