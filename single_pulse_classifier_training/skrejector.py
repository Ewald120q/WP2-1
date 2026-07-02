import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import joblib
from rejector import Rejector
import math

class SNRDT_Rejector(Rejector):
    def __init__(self, device, max_depth=1, min_samples_leaf=200, random_state=0,
                 use_abs_peak=False, snr_db=False, use_meta_snr=False):
        super().__init__(model=None, device=device)
        self.device = device

        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

        self.use_abs_peak = use_abs_peak
        self.snr_db = snr_db
        self.use_meta_snr = use_meta_snr

    def prepare_inputs(self, batch, features):
        """Override base hook to tell the ensemble we need raw batches."""
        return batch

    @staticmethod
    def approximate_SNR(x: torch.Tensor, eps: float = 1e-8, use_abs_peak: bool = False, snr_db: bool = False):
        # x: (B,1,H,W) oder (B,H,W)
        if x.dim() == 4:
            x_flat = x.squeeze(1).flatten(1)
        elif x.dim() == 3:
            x_flat = x.flatten(1)
        else:
            raise ValueError(f"Expected x with 3 or 4 dims, got shape {tuple(x.shape)}")

        med = x_flat.median(dim=1, keepdim=True).values
        mad = (x_flat - med).abs().median(dim=1).values
        sigma = 1.4826 * mad

        if use_abs_peak:
            signal = (x_flat - med).abs().max(dim=1).values
        else:
            signal = x_flat.max(dim=1).values - med.squeeze(1)

        snr = signal / (sigma + eps)
        if snr_db:
            snr = 20.0 * torch.log10(snr + eps)
        return snr  # (B,)

    def _snr_from_batch_dm_time(self, batch) -> torch.Tensor:
        x = batch["dm_time"].to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B,1,H,W)
        approx_snr = self.approximate_SNR(x, use_abs_peak=self.use_abs_peak, snr_db=self.snr_db)
        
        if self.use_meta_snr and "metadata" in batch:
            # metadata shape: (B, 5) -> Index 0 ist SNR
            meta_snr = batch["metadata"][:, 0].to(self.device)
            # Maske für NaNs, unendliche Werte oder -1 erstellen (ungültige SNR-Werte)
            fallback_mask = torch.isnan(meta_snr) | torch.isinf(meta_snr) | (meta_snr == -1)
            # Falls ungültig, nimm `approx_snr`, ansonsten `meta_snr`
            return torch.where(fallback_mask, approx_snr, meta_snr)
            
        return approx_snr

    def fit(self, pulse_X_train_dataloader, routing_targets_train,
            pulse_X_test_dataloader=None, routing_targets_test=None, **kwargs):
        X_list, y_list = [], []
        idx = 0

        if not hasattr(pulse_X_train_dataloader, "shuffle"):
            pass

        with torch.no_grad():
            for batch in pulse_X_train_dataloader:
                snr = self._snr_from_batch_dm_time(batch)  # (B,)
                bsz = snr.shape[0]

                # targets passend zur Loader-Reihenfolge slicen
                if torch.is_tensor(routing_targets_train):
                    yb = routing_targets_train[idx:idx+bsz].detach().cpu().numpy()
                else:
                    yb = np.asarray(routing_targets_train[idx:idx+bsz])

                X_list.append(snr.detach().cpu().numpy())
                y_list.append(yb)
                idx += bsz

        X = np.concatenate(X_list, axis=0).reshape(-1, 1)  # (N,1)
        y = np.concatenate(y_list, axis=0).astype(int)
        
        #class_weight = {0: 1.0, 1: 49.0}  # 98% vs. 2%
        #sample_weight = np.vectorize(class_weight.get)(y)

        self.tree.fit(X, y)

        y_pred = self.tree.predict(X)
        train_acc = self.tree.score(X, y)
        train_cm = confusion_matrix(y, y_pred)
        train_f1 = f1_score(y, y_pred, average="binary", zero_division=0)
        train_precision = precision_score(y, y_pred, average="binary", zero_division=0)
        train_recall = recall_score(y, y_pred, average="binary", zero_division=0)

        print(f"Train accuracy: {train_acc:.4f} (X: SNR; Y: routing labels)")
        print("Train confusion matrix:\n", train_cm)
        print(f"Train F1 score: {train_f1:.4f}")

        if kwargs.get("writer") is not None:
            writer = kwargs["writer"]
            writer.add_scalars('Accuracy', {'train': train_acc}, 1)
            writer.add_scalars('Precision', {'train': train_precision}, 1)
            writer.add_scalars('Recall', {'train': train_recall}, 1)
            writer.add_scalars('F1_score', {'train': train_f1}, 1)

        if pulse_X_test_dataloader is not None and routing_targets_test is not None:
            val_acc, val_f1, val_prec, val_rec = self.evaluate_rejector_with_metrics(pulse_X_test_dataloader, routing_targets_test)
            if kwargs.get("writer") is not None:
                writer.add_scalars('Accuracy', {'val': val_acc}, 1)
                writer.add_scalars('Precision', {'val': val_prec}, 1)
                writer.add_scalars('Recall', {'val': val_rec}, 1)
                writer.add_scalars('F1_score', {'val': val_f1}, 1)

        # threshhold ausgeben
        if self.tree.tree_.node_count >= 1 and self.tree.max_depth == 1:
            tau = float(self.tree.tree_.threshold[0])
            self.snr_threshold_ = tau
            
        if hasattr(self, "snr_threshold_"):
            print("threshold: ", self.snr_threshold_)

        return self
    
    def evaluate_rejector_with_metrics(self, dataloader, targets):
        X_list, y_list = [], []
        idx = 0
        with torch.no_grad():
            for batch in dataloader:
                snr = self._snr_from_batch_dm_time(batch)
                bsz = snr.shape[0]
                if torch.is_tensor(targets):
                    yb = targets[idx:idx+bsz].detach().cpu().numpy()
                else:
                    yb = np.asarray(targets[idx:idx+bsz])
                X_list.append(snr.detach().cpu().numpy())
                y_list.append(yb)
                idx += bsz
        X = np.concatenate(X_list, axis=0).reshape(-1, 1)
        y = np.concatenate(y_list, axis=0).astype(int)

        y_pred = self.tree.predict(X)
        val_acc = self.tree.score(X, y)
        val_cm = confusion_matrix(y, y_pred)
        val_prec = precision_score(y, y_pred, average="binary", zero_division=0)
        val_rec = recall_score(y, y_pred, average="binary", zero_division=0)
        val_f1 = f1_score(y, y_pred, average="binary", zero_division=0)

        print(f"Val accuracy: {val_acc:.4f} (X: SNR; Y: routing labels)")
        print("Val confusion matrix:\n", val_cm)
        print(f"Val F1 score: {val_f1:.4f}")

        return val_acc, val_f1, val_prec, val_rec
        Y = np.concatenate(y_list, axis=0).astype(int)
        
        y_pred = self.tree.predict(X)
        acc = self.tree.score(X, Y)
        f1 = f1_score(Y, y_pred, average="binary", zero_division=0)
        prec = precision_score(Y, y_pred, average="binary", zero_division=0)
        rec = recall_score(Y, y_pred, average="binary", zero_division=0)
        return acc, f1, prec, rec

    def evaluate_rejector(self, pulse_X_train_dataloader, routing_targets_train,
            pulse_X_test_dataloader=None, routing_targets_test=None):
        X_list, y_list = [], []
        idx = 0
        
        print("generate SNR labels for test data") 

        if not hasattr(pulse_X_test_dataloader, "shuffle"):
            pass

        with torch.no_grad():
            for batch in pulse_X_test_dataloader:
                snr = self._snr_from_batch_dm_time(batch)  # (B,)
                bsz = snr.shape[0]

                # Targets passend zur Loader-Reihenfolge slicen
                if torch.is_tensor(routing_targets_test):
                    yb = routing_targets_test[idx:idx+bsz].detach().cpu().numpy()
                else:
                    yb = np.asarray(routing_targets_test[idx:idx+bsz])

                X_list.append(snr.detach().cpu().numpy())
                y_list.append(yb)
                idx += bsz

        X = np.concatenate(X_list, axis=0).reshape(-1, 1)  # (N,1)
        Y = np.concatenate(y_list, axis=0).astype(int)

        print("evaluate test data")
        y_pred = self.tree.predict(X)
        test_acc = self.tree.score(X, Y)
        test_cm = confusion_matrix(Y, y_pred)
        test_f1 = f1_score(Y, y_pred, average="binary")

        print(f"SNRDTRejector Test Accuracy: {test_acc:.4f} (X: SNR; Y: routing labels)")
        print("Test confusion matrix:\n", test_cm)
        print(f"Test F1 score: {test_f1:.4f}")

        return test_acc


    def predict_proba(self, batch):
        with torch.no_grad():
            snr = self._snr_from_batch_dm_time(batch).detach().cpu().numpy().reshape(-1, 1)
        probs = self.tree.predict_proba(snr)
        return torch.from_numpy(probs).to(self.device)

    def predict(self, batch):
        with torch.no_grad():
            snr = self._snr_from_batch_dm_time(batch).detach().cpu().numpy().reshape(-1, 1)
        return self.tree.predict(snr)

    def save(self, path):
        joblib.dump(
            {
                "tree": self.tree, 
                "use_abs_peak": self.use_abs_peak, 
                "snr_db": self.snr_db,
                "use_meta_snr": self.use_meta_snr
            },
            path
        )

    def load(self, path):
        obj = joblib.load(path)
        self.tree = obj["tree"]
        self.use_abs_peak = obj.get("use_abs_peak", False)
        self.snr_db = obj.get("snr_db", False)
        self.use_meta_snr = obj.get("use_meta_snr", False)
        return self
