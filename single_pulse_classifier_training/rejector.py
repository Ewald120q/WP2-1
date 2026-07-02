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
    def fit(self, pulse_X_train_dataloader, routing_targets_train, pulse_X_test_dataloader, routing_targets_test, lr=0.000001, weight_decay=0.0001, num_epochs=100, patience=15, writer=None, run_name="rejector_test"):
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
            min_lr=1e-7,
            verbose=True,
        )
        
        n0 = (routing_targets_train == 0).sum()
        n1 = (routing_targets_train == 1).sum()
        
        w0 = 1.0
        w1 = n0 / max(n1, 1)
        class_weights = torch.tensor([w0, w1], device=self.device, dtype=torch.float32)
        
        criterion = nn.CrossEntropyLoss(weight = class_weights)
        
        if writer is None:
            tensorboard_cfg = {
            "log_root": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs_grid_08_01_01/",
            "experiment_name": "rejector-DM_time_binary_classificator_241002_3_dropout_fsembeddingmlp",
            "run_name": run_name
            }
            
            writer = SummaryWriter(log_dir=tensorboard_cfg["log_root"])
            
        history, best_val_acc, best_epoch = training._train(
            self.model,
            run_name,
            pulse_X_train_dataloader,
            pulse_X_test_dataloader,
            optimizer,
            num_epochs,
            criterion=criterion,
            scheduler=scheduler,
            writer=writer,
            device=self.device,
            patience=patience,
            checkpoint_dir="./rejector_checkpoints",
            targets_train=routing_targets_train,
            targets_val=routing_targets_test,
        )

        return history, best_val_acc, best_epoch
        
    def predict_proba(self, x):
        return self.model.classifier(x)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        
class SmallToEmbedding(nn.Module):
    def __init__(self, f_small: nn.Module, feature_source="classifier_features"):
        super().__init__()
        self.f_small = f_small
        self.feature_source = feature_source
        self.f_small.eval()

        if feature_source not in {"classifier_features", "pooled_features"}:
            raise ValueError(
                f"Unknown feature_source '{feature_source}'. "
                "Use 'classifier_features' or 'pooled_features'."
            )

    def forward(self, batch):
        with torch.no_grad():
            x = self.f_small._prepare_input(batch)
            if self.feature_source == "pooled_features":
                return self.f_small.pooled_features(x)
            return self.f_small.classifier_features(x)

class EmbeddingRejector(Rejector):
    
    def __init__(self, f_small, embedding_processing, device, feature_source="classifier_features"):
        super(Rejector, self).__init__()
        
        self.device = device
        self.f_small = f_small
        self.f_small.eval()
        for p in self.f_small.parameters():
            p.requires_grad = False
        self.embedding_processing = embedding_processing
        self.feature_source = feature_source
        
        self.model = nn.Sequential(SmallToEmbedding(self.f_small, feature_source), self.embedding_processing)
        self.model.to(device)
        
    def predict_proba(self, x):
        return self.model(x)
    
    def prepare_inputs(self, batch, features):
        """EmbeddingRejector directly consumes the original batch."""
        return batch
