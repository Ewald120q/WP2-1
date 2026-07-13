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
        run_dir = output_root / run_name

        trial_config["seed"] = seed
        trial_config["run_name"] = run_name
        trial_config["output_dir"] = str(run_dir)

        print(f"\nTrial {trial + 1}/{num_trials}: {run_name}")
        print(json.dumps(sampled_parameters, indent=2))

        if run_dir.exists():
            print(f"Run already exists, skipping: {run_dir}")
            continue

        run_dir.mkdir(parents=True)
        with (run_dir / "sampled_config.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(trial_config, handle, indent=2)

        try:
            result = run_training(trial_config)
            result["status"] = "completed"
        except Exception as error:
            result = {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            print(result["traceback"])

        result.update(
            {
                "run_name": run_name,
                "seed": seed,
                "sampled_parameters": sampled_parameters,
            }
        )
        with (run_dir / "randomsearch_result.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
