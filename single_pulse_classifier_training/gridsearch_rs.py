from training import *
import copy
import random
import argparse

parser = argparse.ArgumentParser(description="Random Search Worker")
parser.add_argument("--worker_id", type=int, default=0, help="Eindeutige ID für diesen Container (z.B. 0, 1, 2)")
args = parser.parse_args()

_config = {
    "path_to_files": "/raid/outputs",
    "path_to_checkpoints": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/checkpoints_new/",
    "path_to_images": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/images_new/",
    "tensorboard_log_dir": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs_new/",
    "tensorboard": {
        "log_root": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs_random_search_dmt_ft/",
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
    "dataset_prefix": "B0531+21_59000_48386",
    "mode": "dmft",
    "dropout": 0.0 
}

NUM_TRIALS = 3 # trys pro Container
MODEL = 'DM_time_binary_classificator_resnet18'
MODE = "ft"

if __name__ == "__main__":
    # jeder container bekommt einen eigenen Zufalls-Startpunkt
    worker_seed = 42 + args.worker_id
    random.seed(worker_seed)
    
    print(f"--- Starte Container mit Worker-ID {args.worker_id} und Base-Seed {worker_seed} ---")

    for trial in range(0, NUM_TRIALS):

        lr = 10 ** random.uniform(-5, -3)
        wd = 10 ** random.uniform(-3, -1)
        dropout = random.choice([0.3, 0.4, 0.5])

        config = copy.deepcopy(_config)
        
        config["model_name"] = MODEL
        config["learning_rate"] = lr
        config["weight_decay"] = wd
        config["mode"] = MODE
        config["dropout"] = dropout
        
        run_name_str = f"lr{lr:.2e}_wd{wd:.2e}"
        
        tb_cfg = config.setdefault("tensorboard", {})
        tb_cfg["experiment_name"] = f"RandomSearch_{MODEL}_dropout{dropout}-{MODE}_Container{args.worker_id}"
        tb_cfg["run_name"] = run_name_str

        print(f"[Container {args.worker_id} | Trial {trial}/{NUM_TRIALS}] LR: {lr:.6f} | WD: {wd:.4f} | Drop: {dropout}")

        get_model_parameters_from_config(config)
        train(config)