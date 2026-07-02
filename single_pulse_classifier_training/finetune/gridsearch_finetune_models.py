import sys
from pathlib import Path

# needed because file is now in finetune folder
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from finetune_training import train_finetune
import copy
import random
import argparse


parser = argparse.ArgumentParser(description="Random Search Worker finetuning")
parser.add_argument("--worker_id", type=int, default=0, help="id für container")
parser.add_argument("--model", type=str, default="f_small",
                    choices=["f_small", "f_mid", "f_large"])
parser.add_argument("--num_trials", type=int, default=3)
args = parser.parse_args()


# baseline models + corresponding routes
# maybe later move this to config but for now easier here
model_configs = {
    "f_small": {
        "model_name": "DM_time_binary_classificator_241002_3_GAP",
        "dropout": False,
        "learning_rates": [
                        1e-6,
                        3e-6,
                        1e-5,
                        3e-5,
                        1e-4
                        ]
        ,
        "checkpoint": root_dir / "final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_241002_3_GAP-014-0.764-0.740.pth"
    },
    "f_mid": {
        "model_name": "DM_time_binary_classificator_241002_5_GAP",
        "dropout": False,
        "learning_rates": [
                           1e-6, 
                           3e-6, 
                           1e-5,
                           3e-5,
                           1e-4
                        ],
        "checkpoint": root_dir / "final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_241002_5_GAP-060-0.973-0.948.pth"
    },
    "f_large": {
        "model_name": "DM_time_binary_classificator_resnet18",
        "dropout": 0.3,
        "learning_rates": [
                           1.19e-7,
                           3.57e-7, 
                           1.19e-6, 
                           3.57e-6, 
                           1.19e-5
                           ],
        "checkpoint": root_dir / "final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_resnet18-003-0.993-0.993.pth"
    }
}

#small (lr: 1e-4; wd: 0; dropout: 0) -> lr:[1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
#mid (lr: 1e-4; wd: 0 dropout: 0) -> lr:[1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
#large (lr: 1.19e-05; wd: 2.47e-02; dropout: 0.3) -> lr:[1.19e-07, 3.57e-07, 1.19e-06, 3.57e-06, 1.19e-05]


_config = {
    # "path_to_files": "/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator/outputs",
    "path_to_files": "/raid/outputs",
    "path_to_finetune_datasets": str(current_dir / "finetune_datasets_NOREPLAY"),
    "path_to_checkpoints": str(root_dir / "finetune_checkpoints_classifiers_NOREPLAY"),
    "path_to_images": str(root_dir / "images_finetune_NOREPLAY"), # currently not used
    "tensorboard_log_dir": str(root_dir / "tensorboard_runs" / "tensorboard_runs_finetune_classifiers_NOREPLAY"),
    "tensorboard": {
        "log_root": str(root_dir / "tensorboard_runs" / "tensorboard_runs_finetune_classifiers_NOREPLAY"),
        "experiment_name": "placeholder",
        "run_name": "placeholder"
    },
    "resolution": 256,
    "files_by_resolution": { # legacy config stuff, dataset loader does not need it
        "256": "B0531+21_59000_48386_DM_time_dataset_realbased_training_pytorch.npy",
        "default": "B0531+21_59000_48386_DM_time_dataset_realbased_training_pytorch_{res}x{res}.npy"
    },
    "weight_decay": 0.0,
    "num_epochs": 50,
    "patience": 10,
    "batch_size": 64,
    "num_workers": 5,
    "prefetch_factor": 2,
    "dataset_prefix": "B0531+21_59000_48386",
    "dropout": 0.0
}

NUM_TRIALS = args.num_trials
MODEL = args.model


if __name__ == "__main__":
    # same random search setup as before, every container different seed
    worker_seed = 42 + args.worker_id
    random.seed(worker_seed)

    print(f"--- starte finetuning {MODEL}, worker {args.worker_id}, seed {worker_seed} ---")

    model_setup = model_configs[MODEL]

    for trial in range(0, NUM_TRIALS):

        learning_rates = model_setup["learning_rates"]
        lr = learning_rates[trial % len(learning_rates)]
        wd = _config["weight_decay"]
        dropout = model_setup["dropout"]

        config = copy.deepcopy(_config)

        # important finetune differences to old gridsearch
        config["route_name"] = MODEL
        config["model_name"] = model_setup["model_name"]
        config["pretrained_checkpoint"] = str(model_setup["checkpoint"])

        config["learning_rate"] = lr
        config["weight_decay"] = wd
        config["dropout"] = dropout

        run_name_str = f"trial{trial}_lr{lr:.2e}_wd{wd:.2e}_drop{dropout}"

        tb_cfg = config.setdefault("tensorboard", {})
        tb_cfg["experiment_name"] = f"RandomSearch_Finetune_{MODEL}_Container{args.worker_id}"
        tb_cfg["run_name"] = run_name_str

        print(f"[Container {args.worker_id} | Trial {trial}/{NUM_TRIALS}] "
              f"model {MODEL} LR: {lr:.6f} | WD: {wd:.4f} | Drop: {dropout}")

        # loads baseline checkpoint and routed datasets internally
        train_finetune(config)
