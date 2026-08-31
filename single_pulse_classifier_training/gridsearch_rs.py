from training import *
import copy
import random
import argparse

parser = argparse.ArgumentParser(description="Random Search Worker")
parser.add_argument("--worker_id", type=int, default=0, help="Eindeutige ID für diesen Container (z.B. 0, 1, 2)")
parser.add_argument(
    "--model",
    choices=["all", "resnet18", "resnet50"],
    default="all",
    help="Welche Architektur trainiert werden soll. Default: beide Architekturen.",
)
parser.add_argument(
    "--num_trials",
    type=int,
    default=20,
    help="Anzahl Random-Search-Runs pro Architektur.",
)
parser.add_argument("--seed", type=int, default=42, help="Basis-Seed für die Random Search.")
args = parser.parse_args()

_config = {
    "path_to_files": "/raid/outputs",
    "path_to_checkpoints": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/checkpoints_new/",
    "path_to_images": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/images_new/",
    "tensorboard_log_dir": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs_new/",
    "tensorboard": {
        "log_root": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs/tensorboard_runs_random_search_extra/",
        "experiment_name": "placeholder", 
        "run_name": "placeholder" 
    },
    "resolution": 256,
    "model_name": "DM_time_binary_classificator_resnet50",
    "files_by_resolution": {
        "256": "B0531+21_59000_48386_DM_time_dataset_realbased_training_pytorch.npy",
        "default": "B0531+21_59000_48386_DM_time_dataset_realbased_training_pytorch_{res}x{res}.npy"
    },
    "labels": "B0531+21_59000_48386_DM_time_dataset_realbased_labels.npy",
    "learning_rate": 0.0001, 
    "weight_decay": 0.0001,
    "num_epochs": 100,
    "patience": 15,
    "batch_size": 64,
    "num_workers": 4,
    "prefetch_factor": 2,
    "dataset_prefix": "B0531+21_59000_48386",
    "mode": "dmft",
    "dropout": 0.0,
    "scheduler": "reduce_on_plateau",
    "scheduler_monitor": "val_accuracy",
    "scheduler_mode": "max",
    "scheduler_factor": 0.5,
    "scheduler_patience": 10,
}

MODE = "dmft"
MODEL_BY_ALIAS = {
    "resnet18": "DM_time_binary_classificator_resnet18",
    "resnet50": "DM_time_binary_classificator_resnet50",
}


def selected_models(model_arg):
    if model_arg == "all":
        return ["resnet18", "resnet50"]
    return [model_arg]

if __name__ == "__main__":
    if args.num_trials <= 0:
        raise ValueError("--num_trials muss groesser als 0 sein.")
    
    worker_seed = args.seed + args.worker_id
    print(f"--- Starte Container mit Worker-ID {args.worker_id} und Base-Seed {worker_seed} ---")

    for model_index, model_alias in enumerate(selected_models(args.model)):
        model_name = MODEL_BY_ALIAS[model_alias]
        rng = random.Random(worker_seed + 1000 * model_index)

        for trial in range(1, args.num_trials + 1):
            lr = 10 ** rng.uniform(-5, -3)
            wd = 10 ** rng.uniform(-3, -1)
            dropout = rng.choice([0.3, 0.4, 0.5])

            config = copy.deepcopy(_config)
            config["model_name"] = model_name
            config["learning_rate"] = lr
            config["weight_decay"] = wd
            config["mode"] = MODE
            config["dropout"] = dropout

            run_name_str = f"lr{lr:.2e}_wd{wd:.2e}"

            tb_cfg = config.setdefault("tensorboard", {})
            tb_cfg["experiment_name"] = f"RandomSearch_{model_name}_dropout{dropout}-{MODE}_Container{args.worker_id}"
            tb_cfg["run_name"] = run_name_str

            print(
                f"[Container {args.worker_id} | {model_alias} | Trial {trial}/{args.num_trials}] "
                f"LR: {lr:.6f} | WD: {wd:.4f} | Drop: {dropout}"
            )

            get_model_parameters_from_config(config)
            train(config)
