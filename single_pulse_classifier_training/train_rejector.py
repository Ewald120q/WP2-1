import torch
import numpy as np

import training_models
from rejector import Rejector
import training
from ensemble import TorchRejectionEnsemble
from DMTimeShardDataset import DMTimeShardDataset
import plotly.graph_objects as go

from torch.utils.data import DataLoader, TensorDataset, Subset
from rejection_ensemble_helper import _extract_labels, plot_snr_distributions, eval_optimal, _strip_prefix_from_state_dict, _maybe_extract_state_dict
import math

device = "cuda"

#rejector_path = "./rejector2.pth"
#rejector_path = "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/checkpoints/ch_point_DM_time_binary_classificator_241002_3_dropout_256/prot-015-0.980-0.985.pth"
rejector_path = "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/rejector_checkpoints/prot-rejector_test-011-0.544-0.510.pth"
rejector_checkpoint = torch.load(rejector_path, map_location=device)
rejector_state_dict = _strip_prefix_from_state_dict(_maybe_extract_state_dict(rejector_checkpoint))
r = training_models.models_htable["DM_time_binary_classificator_241002_3"](256, mode="dmt", dropout=False, device=device).to(device)
r.load_state_dict(rejector_state_dict)

#freeze first n-1 layers
for p in r.parameters():
    p.requires_grad = False

for p in r.fc2.parameters():
    p.requires_grad = True
    
rejector = Rejector(r, device)

small_weights = "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/checkpoints/ch_point_DM_time_binary_classificator_241002_3_dropout_256/prot-015-0.980-0.985.pth"

small_model = training_models.models_htable["DM_time_binary_classificator_241002_3"](256, mode="dmt", dropout=True, device=device).to(device)
small_model.load_state_dict(torch.load(small_weights, map_location=device)["model_state_dict"])


big_weights = "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/checkpoints/ch_point_DM_time_binary_classificator_resnet18_dropout_256/prot-014-0.972-0.977.pth"

big_model = training_models.models_htable['DM_time_binary_classificator_resnet18'](256, mode="ft", dropout=True, device=device).to(device)
big_model.load_state_dict(torch.load(big_weights, map_location=device)["model_state_dict"])


rejection_ensemble = TorchRejectionEnsemble(small_model, big_model, p=0.8, rejector=rejector, calibration=False)

dataset_cfg = {
        "output_dir": "/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator/outputs",
        "prefix": "B0531+21_59000_48386",
    }

full_train_dataset = DMTimeShardDataset(dataset_cfg, use_freq_time=True, split="train")
full_train_dataset.labels = training.label_encoding(full_train_dataset.labels.astype(object))


val_fraction = 0.1111
#val_fraction = 0.95
if not 0 < val_fraction < 1:
    raise ValueError("'val_fraction' must be between 0 and 1.")

num_train_samples = len(full_train_dataset)
if num_train_samples < 2:
    raise ValueError("Need at least 2 training samples to create a validation split.")

split_idx_val = math.floor(num_train_samples * (1 - val_fraction))
split_idx_val = min(max(split_idx_val, 1), num_train_samples - 1)

train_indices = range(0, split_idx_val)
val_indices = range(split_idx_val, num_train_samples)

pulse_train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

#train_dataset = full_train_dataset

pulse_test_dataset = DMTimeShardDataset(dataset_cfg, use_freq_time=True, split="test")
pulse_test_dataset.labels = training.label_encoding(pulse_test_dataset.labels.astype(object))

pulse_train_loader = DataLoader(pulse_train_dataset,
                          batch_size=256,
                          shuffle=False,
                          num_workers=0)
#pulse_val_loader = DataLoader(val_dataset,
#                          batch_size=256,
#                          shuffle=False,
#                          num_workers=8)
pulse_test_loader = DataLoader(pulse_test_dataset,
                        batch_size=256,
                        shuffle=False,
                        num_workers=0)

print("Train Loader length: ", len(pulse_train_loader))
#print("Val Loader length: ", len(pulse_val_loader))
print("Test Loader length: ", len(pulse_test_loader))
print("train dataset length: ", len(pulse_train_dataset))
#print("val dataset length: ", len(pulse_val_dataset))
print("test dataset length: ", len(pulse_test_dataset))

routing_targets_train, routing_targets_test, pulse_train_preds_small, pulse_test_preds_small = rejection_ensemble.prepare_fit(pulse_train_loader, pulse_test_loader)

balanced_pulse_train_loader, balanced_routing_train_targets = rejection_ensemble._splitTrainData(pulse_train_loader, routing_targets_train, pulse_train_preds_small)
balanced_pulse_test_loader, balanced_routing_test_targets = rejection_ensemble._splitTrainData(pulse_test_loader, routing_targets_test, pulse_test_preds_small)

pulse_train_dataset = pulse_train_loader.dataset
pulse_train_targets = _extract_labels(pulse_train_dataset)

pulse_idx_0 = np.where(pulse_train_targets == 0)[0]
pulse_idx_1 = np.where(pulse_train_targets == 1)[0]
print(len(pulse_idx_0)," ", len(pulse_idx_1))

balanced_routing_idx_0 = np.where(balanced_routing_train_targets == 0)[0]
balanced_routing_idx_1 = np.where(balanced_routing_train_targets == 1)[0]
print(len(balanced_routing_idx_0)," ", len(balanced_routing_idx_1))

routing_idx_0 = np.where(routing_targets_train == 0)[0]
routing_idx_1 = np.where(routing_targets_train == 1)[0]
print(len(routing_idx_0)," ", len(routing_idx_1))


rejection_ensemble.eval(train_dataloader=pulse_test_loader, test_dataloader=pulse_test_loader)