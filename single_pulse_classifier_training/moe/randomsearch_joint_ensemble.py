from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import traceback
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    from train_joint_ensemble import run_training
else:
    from .train_joint_ensemble import run_training


def set_config_value(config: dict[str, Any], path: str, value: Any) -> None:
    """Set a nested value such as ``training.expert_learning_rate``."""
    keys = path.split(".")
    current = config

    for key in keys[:-1]:
        current = current[key]

    current[keys[-1]] = value


def sample_parameter(spec: dict[str, Any], rng: random.Random) -> Any:
    distribution = spec["type"]

    if distribution == "uniform":
        return rng.uniform(spec["min"], spec["max"])

    if distribution == "log_uniform":
        return math.exp(
            rng.uniform(
                math.log(spec["min"]),
                math.log(spec["max"]),
            )
        )

    if distribution == "choice":
        return rng.choice(spec["values"])

    raise ValueError(f"Unknown random-search type: {distribution}")


def format_run_value(value: Any) -> str:
    if isinstance(value, float):
        value = f"{value:.4g}"

    return (
        str(value)
        .replace("+", "")
        .replace("-", "m")
        .replace(".", "p")
    )


def build_run_name(
    worker_id: int,
    trial: int,
    seed: int,
    sampled_parameters: dict[str, Any],
    search_space: dict[str, Any],
) -> str:
    parts = [
        "joint_moe",
        f"worker{worker_id}",
        f"trial{trial}",
        f"seed{seed}",
    ]

    for path, value in sampled_parameters.items():
        short_name = search_space[path].get(
            "name",
            path.split(".")[-1],
        )
        parts.append(f"{short_name}{format_run_value(value)}")

    return re.sub(r"[^A-Za-z0-9_.-]", "-", "_".join(parts))


def write_result(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random search for the joint MoE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as handle:
        base_config = json.load(handle)

    search_config = base_config.pop("random_search")
    search_space = search_config["parameters"]
    num_trials = args.num_trials or search_config["num_trials"]
    output_root = Path(search_config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    for trial in range(num_trials):
        seed = search_config["base_seed"] + args.worker_id * 100_000 + trial
        rng = random.Random(seed)

        trial_config = copy.deepcopy(base_config)
        sampled_parameters = {}

        for path, spec in search_space.items():
            value = sample_parameter(spec, rng)
            set_config_value(trial_config, path, value)
            sampled_parameters[path] = value

        run_name = build_run_name(
            args.worker_id,
            trial,
            seed,
            sampled_parameters,
            search_space,
        )
        topk_noise_std = trial_config.get("training", {}).get("topk_noise_std")
        if topk_noise_std is not None and "training.topk_noise_std" not in sampled_parameters:
            run_name = f"{run_name}_topknoisestd{format_run_value(topk_noise_std)}"
        topk_noise_start_epoch = trial_config.get("training", {}).get("topk_noise_start_epoch")
        if topk_noise_start_epoch is not None and "training.topk_noise_start_epoch" not in sampled_parameters:
            run_name = f"{run_name}_noisestart{format_run_value(topk_noise_start_epoch)}"
        topk_noise_epochs = trial_config.get("training", {}).get("topk_noise_epochs")
        if topk_noise_epochs is not None and "training.topk_noise_epochs" not in sampled_parameters:
            run_name = f"{run_name}_noiseep{format_run_value(topk_noise_epochs)}"
        expert_aux_loss_weight = trial_config.get("training", {}).get("expert_aux_loss_weight")
        if expert_aux_loss_weight is not None and "training.expert_aux_loss_weight" not in sampled_parameters:
            run_name = f"{run_name}_aux{format_run_value(expert_aux_loss_weight)}"
        expert_aux_loss_epochs = trial_config.get("training", {}).get("expert_aux_loss_epochs")
        if expert_aux_loss_epochs is not None and "training.expert_aux_loss_epochs" not in sampled_parameters:
            run_name = f"{run_name}_auxep{format_run_value(expert_aux_loss_epochs)}"
        aux_loss_mode = trial_config.get("training", {}).get("aux_loss_mode", "warmup_only")
        if aux_loss_mode != "warmup_only" and "training.aux_loss_mode" not in sampled_parameters:
            run_name = f"{run_name}_auxmode{aux_loss_mode}"
        budget_loss_weight = trial_config.get("training", {}).get("budget_loss_weight")
        if budget_loss_weight is not None and "training.budget_loss_weight" not in sampled_parameters:
            run_name = f"{run_name}_budget{format_run_value(budget_loss_weight)}"
        routing_loss_weight = trial_config.get("training", {}).get("routing_loss_weight")
        if routing_loss_weight is not None and "training.routing_loss_weight" not in sampled_parameters:
            run_name = f"{run_name}_routing{format_run_value(routing_loss_weight)}"
        freeze_experts_on_upper_bound = bool(trial_config.get("training", {}).get("freeze_experts_on_upper_bound", False))
        if freeze_experts_on_upper_bound and "training.freeze_experts_on_upper_bound" not in sampled_parameters:
            freeze_metric = str(trial_config.get("training", {}).get("freeze_expert_metric", "val_upper_bound")).replace("val_", "").replace("_", "")
            freeze_patience = trial_config.get("training", {}).get("freeze_expert_patience", 5)
            run_name = f"{run_name}_freeze{freeze_metric}pat{format_run_value(freeze_patience)}"
        only_aux_warmup = trial_config.get("training", {}).get("only_aux_warmup")
        if only_aux_warmup is not None and "training.only_aux_warmup" not in sampled_parameters:
            run_name = f"{run_name}_auxwarmup{int(bool(only_aux_warmup))}"
        run_dir = output_root / run_name

        trial_config["seed"] = seed
        trial_config["run_name"] = run_name
        trial_config["output_dir"] = str(run_dir)

        print(f"\nTrial {trial + 1}/{num_trials}: {run_name}", flush=True)
        print(json.dumps(sampled_parameters, indent=2), flush=True)

        result_path = run_dir / "randomsearch_result.json"
        if result_path.exists():
            with result_path.open(encoding="utf-8") as handle:
                existing_result = json.load(handle)
            if existing_result.get("status") in {"completed", "failed"}:
                print(f"Run result already exists, skipping: {result_path}", flush=True)
                continue
            print(f"Run has unfinished status {existing_result.get('status')!r}, retrying: {run_dir}", flush=True)
        elif run_dir.exists():
            print(f"Run directory exists without result, retrying: {run_dir}", flush=True)

        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "sampled_config.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(trial_config, handle, indent=2)

        result_metadata = {
            "run_name": run_name,
            "seed": seed,
            "sampled_parameters": sampled_parameters,
        }
        write_result(result_path, {"status": "running", **result_metadata})

        try:
            result = run_training(trial_config)
            result["status"] = "completed"
        except KeyboardInterrupt as error:
            result = {
                "status": "interrupted",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            print(result["traceback"], flush=True)
            result.update(result_metadata)
            write_result(result_path, result)
            raise
        except Exception as error:
            result = {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            print(result["traceback"], flush=True)

        result.update(result_metadata)
        write_result(result_path, result)


if __name__ == "__main__":
    main()
