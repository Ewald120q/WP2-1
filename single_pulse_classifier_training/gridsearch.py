from training import *
import itertools
import copy

_config = {
    "path_to_files": "/raid/data/run4_thr2",
    "path_to_checkpoints": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/checkpoints/",
    "path_to_images": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/images/",
    "tensorboard_log_dir": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs/",
    "tensorboard": {
        "log_root": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs_grid/",
        "experiment_name": "3layer-freq_time", #!
        "run_name": "adam_lr1e-4" #!
    },
    "resolution": 256,
    "model_name": "DM_time_binary_classificator_241002_3_dropout", #!
    "files_by_resolution": {
        "256": "B0531+21_59000_48386_DM_time_dataset_realbased_training_pytorch.npy",
        "default": "B0531+21_59000_48386_DM_time_dataset_realbased_training_pytorch_{res}x{res}.npy"
    },
    "labels": "B0531+21_59000_48386_DM_time_dataset_realbased_labels.npy",
    "learning_rate": 0.0001, #!
    "weight_decay": 0.0001,
    "num_epochs": 15,
    "patience": 5,
    "batch_size": 1024,

    "dataset_prefix": "B0531+21_59000_48386",
    "use_freq_time": False # !
}

models = ["DM_time_binary_classificator_241002_3_dropout","DM_time_binary_classificator_241002_4_dropout","DM_time_binary_classificator_241002_5_dropout""DM_time_binary_classificator_241002_6_dropout"]
lrs = [1e-3, 1e-4, 1e-5]
weight_decays = [1e-4]
freq_times = [True]

if __name__ == "__main__":
    for model, lr, wd, ft in itertools.product(models, lrs, weight_decays, freq_times):
        config = copy.deepcopy(_config)
        print(f"Config: model:{model}, lr:{lr}, wd:{wd}, ft:{ft}")
        
        config["model_name"] = model
        config["learning_rate"] = lr
        config["weight_decay"] = wd
        config["use_freq_time"] = ft
        
        tb_cfg = config.setdefault("tensorboard", {})
        tb_cfg["experiment_name"] = f"{model}-{'ft' if ft else 'dmt'}"
        tb_cfg["run_name"] = f"lr{lr}_wd{wd}"

        train(config)