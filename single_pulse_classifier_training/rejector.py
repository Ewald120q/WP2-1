import torch
import torch.nn as nn
import numpy as np
import training_models
import training
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Rejector(nn.Module):
    """Common interface for routing rejectors.

    Sub-classes can override :meth:`prepare_inputs` when they need access to the
    raw batch (e.g. sklearn-based rejectors computing handcrafted features)
    instead of the feature tensor emitted by the small network.
    """

    def __init__(self, model, device):
        super(Rejector, self).__init__()

        self.device = device
        self.model = model

    def prepare_inputs(self, batch, features):
        """Return the object that ``predict_proba`` should consume.

        Args:
            batch: Original batch dictionary passed to the ensemble.
            features (torch.Tensor): Feature tensor computed from the small model.

        Returns:
            The value that will be forwarded to ``predict_proba``. Torch-based
            rejectors default to using the feature tensor, while sklearn based
            ones can override this method to return the raw batch instead.
        """
        return features
        
    #targets 0 = f_small; 1 = f_big
    def fit(self, pulse_X_train_dataloader, routing_targets_train, pulse_X_test_dataloader, routing_targets_test):
        
        #train_loader = DataLoader(zip(X_train, targets_train), batch_size= 256, shuffle=False, num_workers=8)
        #test_loader = DataLoader(zip(X_test, targets_test), batch_size= 256, shuffle=False, num_workers=8)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        num_epochs = 100
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
        criterion = nn.CrossEntropyLoss()
        
        tensorboard_cfg = {
        "log_root": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs_grid_08_01_01/",
        "experiment_name": "rejector-DM_time_binary_classificator_241002_3_dropout_fsasrejector",
        "run_name": "lr0.0001_wd0.0001"
        }
        
        writer = SummaryWriter(log_dir=tensorboard_cfg["log_root"])
        #writer.add_text("run/config", json.dumps(config, indent=2), 0)
        #writer.add_text("run/experiment", experiment_component, 0)
        #writer.add_text("run/name", run_component, 0)
        
        history, best_test_acc, best_epoch, final_test_acc, wrong_examples, wrong_labels = training._train(
        self.model,
        "rejector_test",
        pulse_X_train_dataloader,
        pulse_X_test_dataloader,
        optimizer,
        num_epochs,
        val_dataloader=None,
        criterion=criterion,
        scheduler=scheduler,
        writer=writer,
        device="cuda",
        patience=num_epochs,
        checkpoint_dir="./rejector_checkpoints",
        targets_train = routing_targets_train,
        targets_test = routing_targets_test
        )
        
    def predict_proba(self, x):
        return self.model.classifier(x)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        
    