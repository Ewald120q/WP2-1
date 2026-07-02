#legacy code
import torch
import numpy as np

import training_models
from rejector import Rejector
import training
from ensemble import TorchRejectionEnsemble
from DMTimeShardDataset import DMTimeShardDataset

from torch.utils.data import DataLoader, TensorDataset, Subset
import math

from skrejector import SNRDT_Rejector
from rejector import EmbeddingRejector
import torch.nn as nn

#alibi rejector; wont be used at all

device = "cuda"
rejector = SNRDT_Rejector(device)

#models

small_weights = "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_241002_3_GAP-014-0.764-0.740.pth"

small_model = training_models.models_htable["DM_time_binary_classificator_241002_3_GAP"](256, mode="dmt", dropout=False, device=device).to(device)
small_model.load_state_dict(torch.load(small_weights, map_location=device)["model_state_dict"])


big_weights = "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_resnet18-003-0.993-0.993.pth"

big_model = training_models.models_htable['DM_time_binary_classificator_241002_5_GAP'](256, mode="ft", dropout=True, device=device).to(device)
big_model.load_state_dict(torch.load(big_weights, map_location=device)["model_state_dict"])


rejection_ensemble = TorchRejectionEnsemble(small_model, big_model, p=1.0, rejector=rejector, calibration=False)

#data

dataset_cfg = {
        "output_dir": "/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator/outputs",
        "prefix": "B0531+21_59000_48386",
    }

pulse_train_dataset = DMTimeShardDataset(dataset_cfg, use_freq_time=True, split="train")
pulse_train_dataset.labels = training.label_encoding(pulse_train_dataset.labels.astype(object))

pulse_val_dataset = DMTimeShardDataset(dataset_cfg, use_freq_time=True, split="val")
pulse_val_dataset.labels = training.label_encoding(pulse_val_dataset.labels.astype(object))

pulse_test_dataset = DMTimeShardDataset(dataset_cfg, use_freq_time=True, split="test")
pulse_test_dataset.labels = training.label_encoding(pulse_test_dataset.labels.astype(object))

pulse_train_loader = DataLoader(pulse_train_dataset,
                          batch_size=256,
                          shuffle=False,
                          num_workers=0)

pulse_val_loader = DataLoader(pulse_val_dataset,
                          batch_size=256,
                          shuffle=False,
                          num_workers=0)

pulse_test_loader = DataLoader(pulse_test_dataset,
                        batch_size=256,
                        shuffle=False,
                        num_workers=0)

print("Train Loader length: ", len(pulse_train_loader))
print("Val Loader length: ", len(pulse_val_loader))
print("Test Loader length: ", len(pulse_test_loader))
print("train dataset length: ", len(pulse_train_dataset))
print("val dataset length: ", len(pulse_val_dataset))
print("test dataset length: ", len(pulse_test_dataset))

routing_targets_train, routing_targets_val, pulse_train_preds_small, pulse_val_preds_small = rejection_ensemble.prepare_fit(pulse_train_loader, pulse_val_loader)

print(pulse_train_preds_small.shape)

balanced_pulse_train_loader, balanced_routing_train_targets = rejection_ensemble._splitTrainData(pulse_train_loader, routing_targets_train, pulse_train_preds_small)
balanced_pulse_val_loader, balanced_routing_val_targets = rejection_ensemble._splitTrainData(pulse_val_loader, routing_targets_val, pulse_val_preds_small)


splits_path = "./new_dm_time_splits_cascaded_r1.pth"
rejection_ensemble.save_balanced_splits(
    splits_path,
    pulse_train_loader,
    routing_targets_train,
    pulse_val_loader,
    routing_targets_val,
)
print(f"Saved unbalanced loaders and targets to {splits_path}")

balanced_splits_path = "./new_balanced_dm_time_splits_cascaded_r1.pth"
rejection_ensemble.save_balanced_splits(
    balanced_splits_path,
    balanced_pulse_train_loader,
    balanced_routing_train_targets,
    balanced_pulse_val_loader,
    balanced_routing_val_targets,
 )
print(f"Saved balanced loaders and targets to {balanced_splits_path}")