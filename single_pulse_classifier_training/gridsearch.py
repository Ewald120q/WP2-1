from training import *
import itertools
import copy

#"path_to_files": "/raid/outputs",
#"path_to_files": "/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator/outputs",
_config = {
    "path_to_files": "/raid/outputs",
    "path_to_checkpoints": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/checkpoints_new/",
    "path_to_images": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/images_new/",
    "tensorboard_log_dir": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs_new/",
    "tensorboard": {
        "log_root": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs_grid_08_01_01_new/",
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
    "num_epochs": 100,
    "patience": 15,
    "batch_size": 64,

    "dataset_prefix": "B0531+21_59000_48386",
    "mode": "dmt", # !
    "dropout": 0.2 #!
}

models = ['DM_time_binary_classificator_241002_3_GAP'] #['DM_time_binary_classificator_resnet50', 'DM_time_binary_classificator_resnet18'] #['DM_time_binary_classificator_241002_8_GAP','DM_time_binary_classificator_241002_7_GAP','DM_time_binary_classificator_241002_6_GAP', 'DM_time_binary_classificator_241002_5_GAP', 'DM_time_binary_classificator_241002_4_GAP', 'DM_time_binary_classificator_241002_3_GAP'] #['DM_time_binary_classificator_resnet50'] #['DM_time_binary_classificator_241002_6_GAP'] ['DM_time_binary_classificator_resnet50', 'DM_time_binary_classificator_resnet18']
lrs = [1e-4]#, 1e-4, 1e-5]
weight_decays = [0.0]
modes = ["dmt"]
dropouts = [0.0]#, 0.2, 0.5]

#run_list = [
#('DM_time_binary_classificator_241002_3_GAP', 1e-4, 1e-4, "ft", 0.0, "2"),
#('DM_time_binary_classificator_241002_3_GAP', 1e-4, 1e-4, "ft", 0.0, "3"),
#
#('DM_time_binary_classificator_241002_4_GAP', 1e-5, 1e-4, "ft", 0.0, "2"),
#('DM_time_binary_classificator_241002_4_GAP', 1e-5, 1e-4, "ft", 0.0, "3"),
#
#('DM_time_binary_classificator_241002_5_GAP', 1e-5, 1e-4, "ft", 0.0, "2"),
#('DM_time_binary_classificator_241002_5_GAP', 1e-5, 1e-4, "ft", 0.0, "3"),
#
#('DM_time_binary_classificator_241002_6_GAP', 1e-5, 1e-4, "ft", 0.0, "2"),
#('DM_time_binary_classificator_241002_6_GAP', 1e-5, 1e-4, "ft", 0.0, "3"),
#
#('DM_time_binary_classificator_241002_7_GAP', 1e-5, 1e-4, "ft", 0.0, "2"),
#('DM_time_binary_classificator_241002_7_GAP', 1e-5, 1e-4, "ft", 0.0, "3"),

#('DM_time_binary_classificator_241002_8_GAP', 1e-4, 1e-4, "ft", 0.0, "2"),
#('DM_time_binary_classificator_241002_8_GAP', 1e-4, 1e-4, "ft", 0.0, "3"),
#
#('DM_time_binary_classificator_241002_9_GAP', 1e-4, 1e-4, "ft", 0.0, "2"),
#('DM_time_binary_classificator_241002_9_GAP', 1e-4, 1e-4, "ft", 0.0, "3"),
#
#]

run_list = [
#('DM_time_binary_classificator_241002_9_GAP', 1e-4, 1e-4, "ft", 0.0, "2"),
#('DM_time_binary_classificator_241002_3_GAP', 1e-4, 1e-4, "ft", 0.0, "2"), 
#('DM_time_binary_classificator_241002_4_GAP', 1e-4, 1e-4, "ft", 0.0, "2"),
#
#('DM_time_binary_classificator_241002_9_GAP', 1e-4, 1e-4, "ft", 0.0, "3"),
#('DM_time_binary_classificator_241002_3_GAP', 1e-4, 1e-4, "ft", 0.0, "3"), 
#('DM_time_binary_classificator_241002_4_GAP', 1e-4, 1e-4, "ft", 0.0, "3"),
#
#('DM_time_binary_classificator_241002_8_GAP', 1e-4, 1e-4, "ft", 0.0, "2"),
#('DM_time_binary_classificator_241002_6_GAP', 1e-5, 1e-4, "ft", 0.0, "2"),
#('DM_time_binary_classificator_241002_5_GAP', 1e-4, 1e-4, "ft", 0.0, "2"),
#
#('DM_time_binary_classificator_241002_8_GAP', 1e-4, 1e-4, "ft", 0.0, "3"),
#('DM_time_binary_classificator_241002_6_GAP', 1e-4, 1e-4, "ft", 0.0, "3"),
#('DM_time_binary_classificator_241002_5_GAP', 1e-4, 1e-4, "ft", 0.0, "3"),

#('DM_time_binary_classificator_241002_7_GAP', 1e-4, 1e-4, "ft", 0.0, "2"),
#('DM_time_binary_classificator_241002_7_GAP', 1e-4, 1e-4, "ft", 0.0, "3"),
]


if __name__ == "__main__":
    #for lr, model, wd, mode, dropout in itertools.product(lrs, models, weight_decays, modes, dropouts):
    for (model, lr, wd, mode, dropout, run_id) in run_list:
        config = copy.deepcopy(_config)
        print(f"Config: model:{model}, lr:{lr}, wd:{wd}, mode:{mode}")
        
        config["model_name"] = model
        config["learning_rate"] = lr
        config["weight_decay"] = wd
        config["mode"] = mode
        config["dropout"] = dropout
        
        tb_cfg = config.setdefault("tensorboard", {})
        tb_cfg["experiment_name"] = f"{model}_dropout{dropout}-{mode}_100_15_runid_{run_id}"
        tb_cfg["run_name"] = f"lr{lr}_wd{wd}"

        get_model_parameters_from_config(config)
        train(config)