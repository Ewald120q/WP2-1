import torch
import torch.nn as nn
import numpy as np
import training_models
import training
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Rejector(nn.Module):
    
    def __init__(self, model, device):
        super(Rejector, self).__init__()
        
        self.device = device
        self.model = model
        
    #targets 0 = f_small; 1 = f_big
    def fit(self, X_train, targets_train, X_test, targets_test):
        
        train_loader = DataLoader(zip(X_train, targets_train), batch_size= 256, shuffle=False, num_workers=8)
        test_loader = DataLoader(zip(X_test, targets_test), batch_size= 256, shuffle=False, num_workers=8)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.0001)
        num_epochs = 15
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
        criterion = nn.CrossEntropyLoss()
        
        tensorboard_cfg = {
        "log_root": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs_grid_08_01_01/",
        "experiment_name": "DM_time_binary_classificator_241002_3_dropout-test",
        "run_name": "lr0.01_wd0.0001"
        }
        
        writer = SummaryWriter(log_dir=tensorboard_cfg["log_root"])
        #writer.add_text("run/config", json.dumps(config, indent=2), 0)
        #writer.add_text("run/experiment", experiment_component, 0)
        #writer.add_text("run/name", run_component, 0)
        
        history, best_test_acc, best_epoch, final_test_acc, wrong_examples, wrong_labels = training._train(
        self.model,
        train_loader,
        test_loader,
        optimizer,
        num_epochs,
        val_dataloader=None,
        criterion=criterion,
        scheduler=scheduler,
        writer=writer,
        device="cuda",
        patience=5,
        checkpoint_dir="./rejector_checkpoints",
    )