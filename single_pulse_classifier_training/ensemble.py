import importlib
import os
import warnings

import numpy as np
import torch
import tqdm
from scipy.special import softmax
from torch.utils.data import Subset, DataLoader

from training_utils import label_encoding


def _safe_torch_load(path, *, map_location="cpu", **kwargs):
    load_kwargs = dict(kwargs)
    if "weights_only" not in load_kwargs:
        load_kwargs["weights_only"] = False

    try:
        return torch.load(path, map_location=map_location, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        return torch.load(path, map_location=map_location, **load_kwargs)

def get_predictions(f, data_loader, return_confidences = False, return_embeddings=False, return_labels=False, pbar_desc=None):
    """
    Get predictions from a model on a given data loader.

    Args:
        f: The model to use for predictions.
        data_loader (DataLoader): The data loader containing the input data.
        return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.
        return_labels (bool, optional): Whether to return the labels. Defaults to False.
        pbar_desc (str, optional): Description for the tqdm progress bar. Defaults to None.

    Returns:
        tuple or ndarray: The predictions. If `return_embeddings` is True, returns a tuple
            containing the predictions, embeddings, and labels (if `return_labels` is True).
            If `return_embeddings` is False, returns the predictions and labels (if `return_labels` is True).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    all_emebddings = []
    labels = []
    for batch in tqdm.tqdm(data_loader, desc=pbar_desc, disable=pbar_desc is None):
        ybatch = batch["label"].to(device)
        xbatch = batch#.to(device)
        with torch.no_grad():
            if return_embeddings:
                emebddings = f.features(xbatch)
                preds = f.forward(xbatch)
                all_emebddings.append(emebddings)
            elif return_confidences:
                preds = f(xbatch)
            else:
                preds = f(xbatch)
            
            if return_labels:
                labels.extend(ybatch.detach().cpu().view(-1).tolist())

            all_preds.append(preds)
    
    all_preds = torch.vstack(all_preds).cpu().numpy()

    if return_embeddings:
        all_emebddings = torch.vstack(all_emebddings).cpu().numpy()
        if return_labels:
            return all_preds, all_emebddings, labels
        else:
            return all_preds, all_emebddings
    else:
        if return_labels:
            return all_preds, labels
        else:
            return all_preds

class TorchRejectionEnsemble():
    def __init__(self, fsmall, fbig, p, rejector, calibration = True):
        """
        Initialize a RejectionEnsemble object.

        Args:
            fsmall: The already trained small model. This object must offer classifier and features method 
            fbig: The already trained big small model. This object must offer classifier method 
            rejector: The rejector object used for rejection. This object must offer a fit and predict_proba method similar to scikit-learn models
            calibration (bool, optional): Whether to perform calibration. Defaults to True.

        """
        self.fsmall = fsmall
        self.fbig = fbig
        self.rejector = rejector
        self.p = p
        self.calibration = calibration

    def prepare_fit(self, train_loader, test_loader):
        """
        Fits the TorchRejectionEnsemble model to the given dataset.

        Args:
            dataset_loader (torch.utils.data.DataLoader): The data loader for the dataset.

        Returns:
            The result of the _fit method.

        """
        print("get fsmall train predictions")
        train_preds_small, Y_train = get_predictions(self.fsmall, train_loader, return_labels=True)
        Y_train = np.array(Y_train)

        print("get fbig train predictions")
        train_preds_big = get_predictions(self.fbig, train_loader)
        
        print("get fsmall test predictions")
        test_preds_small, Y_test = get_predictions(self.fsmall, test_loader, return_labels=True)
        Y_test = np.array(Y_test)

        print("get fbig test predictions")
        test_preds_big = get_predictions(self.fbig, test_loader)
        
        print("start fitting rejector")
        #return (train_loader, Y_train, train_preds_small, train_preds_big, test_loader, Y_test, test_preds_small, test_preds_big)
                #create pseudo labels
        argmax_train_preds_big = train_preds_big.argmax(1)
        argmax_train_preds_small = train_preds_small.argmax(1)
        routing_targets_train = []
        for i in range(train_preds_big.shape[0]):
            if argmax_train_preds_big[i] == argmax_train_preds_small[i]:
                routing_targets_train.append(0)
            else:
                if argmax_train_preds_big[i] == Y_train[i]:
                    routing_targets_train.append(1)
                else:
                    routing_targets_train.append(0)
        
        routing_targets_train = np.array(routing_targets_train)
        
        #targets arent split equally. EJECT class has a proportion of ~2% in train data, so it does not properly learn.
        #so take amount of EJECT targets, and extract same amount of non EJECT targets. Use the confidence of the model, to take the hardest ones
        
        print("TRAIN: targets before balancing. REJECT: ", np.count_nonzero(routing_targets_train == 1), "no REJECT: ", np.count_nonzero(routing_targets_train == 0))
        #X_train_dataloader, targets_train = self._splitTrainData(X_train_dataloader, targets_train, train_preds_small)   
        print("TRAIN: targets after balancing. REJECT: ",np.count_nonzero(routing_targets_train == 1),"no REJECT: ",np.count_nonzero(routing_targets_train == 0))
        
        
        #for testing rejector
        argmax_test_preds_big = test_preds_big.argmax(1)
        argmax_test_preds_small = test_preds_small.argmax(1)
        routing_targets_test = []
        for i in range(test_preds_big.shape[0]):
            if argmax_test_preds_big[i] == argmax_test_preds_small[i]:
                routing_targets_test.append(0)
            else:
                if argmax_test_preds_big[i] == Y_test[i]:
                    routing_targets_test.append(1)
                else:
                    routing_targets_test.append(0)
        
        routing_targets_test = np.array(routing_targets_test)
        
        print("TEST: targets before balancing. REJECT: ", np.count_nonzero(routing_targets_train == 1), "no REJECT: ", np.count_nonzero(routing_targets_train == 0))
        #X_test_dataloader, targets_test = self._splitTrainData(X_test_dataloader, targets_test, test_preds_small)  
        print("TEST: targets after balancing. REJECT: ",np.count_nonzero(routing_targets_train == 1),"no REJECT: ",np.count_nonzero(routing_targets_train == 0))
        
        print("created pseudo labels")
        print("WARNING: Psuedolabels are unbalanced. Please use self._splitTrainingData, to balance them")
        
        return  routing_targets_train, routing_targets_test, train_preds_small, test_preds_small

    def fit_routing(self, pulse_X_train_dataloader, routing_targets_train, pulse_X_test_dataloader, routing_targets_test):
        """
        Fits the rejection ensemble model using the given inputs. The return value can be ignored if the object is maintained after calling fit.

        Parameters:
        - X: The input data of shape (n_samples, n_features).
        - Y: The target labels of shape (n_samples,).
        - preds_small: The predictions of the small model of shape (n_samples, n_classes).
        - preds_big: The predictions of the big model of shape (n_samples, n_classes).

        Returns:
        - fsmall: The fitted small model.
        - fbig: The fitted big model.
        - rejector: The fitted rejector model, or None if no rejector is used.
        """
        
        print("start creating pseudo labels")
        if np.unique(routing_targets_test).shape[0] > 1:
            self.rejector.fit(pulse_X_train_dataloader, routing_targets_train, pulse_X_test_dataloader, routing_targets_test) #testen rejector auf test daten, um overfitting vom rejector zu vermeiden. (NUR PSEUDO LABELS!)
        else:
            self.rejector = None

        return (routing_targets_train, routing_targets_test), self.fsmall, self.fbig, self.rejector
    
    def eval_routing(self, pulse_X_train_dataloader, routing_targets_train, pulse_X_test_dataloader, routing_targets_test):
        """
        Evaluates the rejection ensemble model using the given inputs. The return value can be ignored if the object is maintained after calling fit.

        Parameters:
        - X: The input data of shape (n_samples, n_features).
        - Y: The target labels of shape (n_samples,).
        - preds_small: The predictions of the small model of shape (n_samples, n_classes).
        - preds_big: The predictions of the big model of shape (n_samples, n_classes).

        Returns:
        - fsmall: The fitted small model.
        - fbig: The fitted big model.
        - rejector: The fitted rejector model, or None if no rejector is used.
        """
        
        print("start creating pseudo labels")
        if np.unique(routing_targets_test).shape[0] > 1:
            self.rejector.eval(pulse_X_train_dataloader, routing_targets_train, pulse_X_test_dataloader, routing_targets_test) #testen rejector auf test daten, um overfitting vom rejector zu vermeiden. (NUR PSEUDO LABELS!)
        else:
            self.rejector = None

        return (routing_targets_train, routing_targets_test), self.fsmall, self.fbig, self.rejector


    def predict_proba(self, T, return_cnt=False):
        """
        Predicts the class probabilities for the input data.

        Args:
            T (torch.Tensor): The input data to be predicted.
            return_cnt (bool, optional): Whether to return the count of rejected samples. Defaults to False.

        Returns:
            torch.Tensor: The predicted class probabilities.
            int: The count of rejected samples if `return_cnt` is True, otherwise None.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #T = T.to(device)

        if self.rejector is None:
            with torch.no_grad():
                if self.p == 1:
                    preds, cnt = self.fbig(T), T.shape[0]
                else:
                    preds, cnt = self.fsmall(T), 0

                if return_cnt:
                    return preds, cnt
                else:
                    return preds
        else:
            with torch.no_grad():
                x_sfeatures = self.fsmall.features(T)
                rejector_inputs = self.rejector.prepare_inputs(T, x_sfeatures)
                r_pred = self.rejector.predict_proba(rejector_inputs)

            if isinstance(r_pred, np.ndarray):
                r_pred_tensor = torch.from_numpy(r_pred).to(device)
            elif torch.is_tensor(r_pred):
                r_pred_tensor = r_pred.to(device)
            else:
                r_pred_tensor = torch.as_tensor(r_pred, device=device)

            print(r_pred_tensor)

            if self.calibration:
                M = len(T)
                P = int(np.floor(self.p * M))
                
                # Determine indices for Ts and Tb using boolean masks
                _, Tb_sorted_indices = torch.sort(r_pred_tensor[:, 1], descending=True)
                Tb_sorted_indices = Tb_sorted_indices[:P]

                Tb_mask = torch.zeros(M, dtype=torch.bool, device=device)
                Tb_mask[Tb_sorted_indices] = True
                Ts_mask = ~Tb_mask  
            else:
                #r_pred_tensor = torch.tensor(r_pred, device=device)
                
                Ts_mask = r_pred_tensor.argmax(dim=1) == 0
                Tb_mask = r_pred_tensor.argmax(dim=1) == 1

            with torch.no_grad():
                fsmall_preds = self.fsmall.classifier(x_sfeatures[Ts_mask])
                ypred = torch.empty((T["label"].shape[0], fsmall_preds.shape[1]), dtype=fsmall_preds.dtype, device=device)
                ypred[Ts_mask] = fsmall_preds
                if not torch.all(Tb_mask == False):
                    x_bfeatures = self.fbig.features(T)
                    fbig_preds = self.fbig.classifier(x_bfeatures[Tb_mask])
                    ypred[Tb_mask] = fbig_preds
            
            if return_cnt:
                return ypred, Tb_mask.sum().item()
            else:
                return ypred
    
    def save_ensemble(self, path):
        if not path:
            raise ValueError("A valid save path must be provided.")

        path = os.fspath(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        def _export_model(model):
            if model is None:
                return None

            state_dict = model.state_dict()
            cpu_state_dict = {
                key: value.detach().cpu()
                if isinstance(value, torch.Tensor) else value
                for key, value in state_dict.items()
            }

            metadata = {
                "class_name": model.__class__.__name__,
                "module": model.__class__.__module__,
            }

            for attr in ("mode", "dropout", "resol", "input_channels", "device"):
                if hasattr(model, attr):
                    metadata[attr] = getattr(model, attr)

            return {
                "state_dict": cpu_state_dict,
                "metadata": metadata,
            }

        ensemble_payload = {
            "fsmall": _export_model(self.fsmall),
            "fbig": _export_model(self.fbig),
            "rejector": _export_model(self.rejector) if self.rejector is not None else None,
            "p": self.p,
            "calibration": self.calibration,
            "has_rejector": self.rejector is not None,
        }

        torch.save(ensemble_payload, path)

    def load_ensemble(self, path, *, map_location=None, strict=True):
        """Load ensemble components and settings from disk."""
        if not path:
            raise ValueError("A valid load path must be provided.")

        path = os.fspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No ensemble file found at '{path}'")

        map_location = map_location or torch.device("cpu")
        payload = _safe_torch_load(path, map_location=map_location)

        self.p = payload.get("p", self.p)
        self.calibration = payload.get("calibration", self.calibration)

        def _check_metadata(model, metadata, name):
            if not metadata:
                return

            expected_class = metadata.get("class_name")
            if expected_class and model.__class__.__name__ != expected_class:
                raise ValueError(
                    f"Saved `{name}` class '{expected_class}' doesn't match current '{model.__class__.__name__}'."
                )

            expected_module = metadata.get("module")
            if expected_module and model.__class__.__module__ != expected_module:
                raise ValueError(
                    f"Saved `{name}` module '{expected_module}' doesn't match current '{model.__class__.__module__}'."
                )

            for attr, expected_value in metadata.items():
                if attr in {"class_name", "module"}:
                    continue
                if hasattr(model, attr):
                    current_value = getattr(model, attr)
                    if current_value != expected_value:
                        warnings.warn(
                            f"`{name}` attribute '{attr}' differs between saved ({expected_value}) and current ({current_value}).",
                            RuntimeWarning,
                        )
                else:
                    warnings.warn(
                        f"`{name}` missing attribute '{attr}' that exists in saved metadata.",
                        RuntimeWarning,
                    )

        def _restore_component(name, current):
            component = payload.get(name)
            if component is None:
                return None
            if current is None:
                raise ValueError(
                    f"Ensemble component `{name}` is None; instantiate it before calling load_ensemble."
                )

            metadata = component.get("metadata", {})
            _check_metadata(current, metadata, name)

            state_dict = component.get("state_dict", {})
            load_info = current.load_state_dict(state_dict, strict=strict)

            if load_info and not strict:
                missing_keys = getattr(load_info, "missing_keys", None)
                unexpected_keys = getattr(load_info, "unexpected_keys", None)
                if missing_keys:
                    warnings.warn(
                        f"Missing keys when loading `{name}`: {missing_keys}",
                        RuntimeWarning,
                    )
                if unexpected_keys:
                    warnings.warn(
                        f"Unexpected keys when loading `{name}`: {unexpected_keys}",
                        RuntimeWarning,
                    )

            return current

        self.fsmall = _restore_component("fsmall", self.fsmall)
        self.fbig = _restore_component("fbig", self.fbig)

        saved_rejector = payload.get("rejector")
        has_rejector = payload.get("has_rejector", saved_rejector is not None)
        if has_rejector:
            self.rejector = _restore_component("rejector", self.rejector)
        else:
            self.rejector = None

    def eval(self, train_dataloader = None, val_dataloader = None, test_dataloader = None):
        train_acc = None
        val_acc = None
        test_acc = None
        
        train_loss = None
        val_loss = None
        test_loss = None
        
        self.rejector.eval()
        self.fsmall.eval()
        self.fbig.eval()
        
        criterion = torch.nn.CrossEntropyLoss()
        if train_dataloader is not None:
            print("evaluating model on training data")
            train_running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch in train_dataloader:
                outputs = self.predict_proba(batch)
                labels = batch["label"].to(self.rejector.device)
                
                loss = criterion(outputs.float(), torch.nn.functional.one_hot(labels, num_classes=2).float())
                train_running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            train_loss = train_running_loss / max(len(train_dataloader), 1)
            train_acc = correct_train / max(total_train, 1)
            
            print(f"train acc: {train_acc}; train loss: {train_loss}")
            
        if val_dataloader is not None:
            print("evaluating model on validation data")
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0
            
            for batch in val_dataloader:
                outputs = self.predict_proba(batch)
                labels = batch["label"].to(self.rejector.device)
                
                loss = criterion(outputs.float(), torch.nn.functional.one_hot(labels, num_classes=2).float())
                val_running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
            
            val_loss = val_running_loss / max(len(val_dataloader), 1)
            val_acc = correct_train / max(total_train, 1)
            
            print(f"val acc: {val_acc}; val loss: {val_loss}")
            
        if test_dataloader is not None:
            print("evaluating model on test data")
            test_running_loss = 0.0
            correct_test = 0
            total_test = 0
            
            for batch in test_dataloader:
                outputs = self.predict_proba(batch)
                labels = batch["label"].to(self.rejector.device)
                
                loss = criterion(outputs.float(), torch.nn.functional.one_hot(labels, num_classes=2).float())
                test_running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
            
            test_loss = test_running_loss / max(len(test_dataloader), 1)
            test_acc = correct_test / max(total_test, 1)
            
            print(f"test acc: {test_acc}; test loss: {test_loss}")
            
        return train_acc, train_loss, val_acc, val_loss, test_acc, test_loss
    

    @staticmethod
    def _resolve_base_dataset_and_indices(dataset):
        base_dataset = dataset
        indices = None

        while isinstance(base_dataset, Subset):
            current_indices = np.asarray(base_dataset.indices)
            if indices is None:
                indices = current_indices
            else:
                indices = current_indices[indices]
            base_dataset = base_dataset.dataset

        if indices is None:
            indices = np.arange(len(base_dataset))

        return base_dataset, indices.astype(np.int64)

    @staticmethod
    def _serialize_loader(loader):
        if loader is None:
            return None

        base_dataset, indices = TorchRejectionEnsemble._resolve_base_dataset_and_indices(loader.dataset)
        dataset_cfg = getattr(base_dataset, "cfg", None)
        if dataset_cfg is not None:
            dataset_cfg = dict(dataset_cfg)
        else:
            raise ValueError("Base dataset must expose a 'cfg' attribute to be serializable.")

        dataset_split = getattr(base_dataset, "split", None)
        dtype = getattr(base_dataset, "dtype", torch.float32)
        dtype_name = str(dtype).split(".")[-1]

        labels_dtype = None
        labels_numeric = True
        if hasattr(base_dataset, "labels"):
            labels_arr = np.asarray(base_dataset.labels)
            labels_dtype = str(labels_arr.dtype)
            labels_numeric = bool(np.issubdtype(labels_arr.dtype, np.number))

        metadata = {
            "dataset_class": base_dataset.__class__.__name__,
            "dataset_module": base_dataset.__class__.__module__,
            "dataset_cfg": dataset_cfg,
            "split": dataset_split,
            "use_freq_time": getattr(base_dataset, "use_freq_time", False),
            "dtype_name": dtype_name,
            "labels_dtype": labels_dtype,
            "labels_numeric": labels_numeric,
        }

        return {
            "indices": indices.tolist(),
            "batch_size": loader.batch_size,
            "num_workers": loader.num_workers,
            "pin_memory": getattr(loader, "pin_memory", False),
            "drop_last": getattr(loader, "drop_last", False),
            "shuffle": False,
            "metadata": metadata,
        }

    @staticmethod
    def _deserialize_loader(payload):
        if payload is None:
            return None

        metadata = payload.get("metadata", {})
        dataset_cfg = metadata.get("dataset_cfg")
        if dataset_cfg is None:
            raise ValueError("Serialized payload is missing dataset configuration.")

        module_name = metadata.get("dataset_module")
        class_name = metadata.get("dataset_class")
        if not module_name or not class_name:
            raise ValueError("Serialized payload is missing dataset import metadata.")

        module = importlib.import_module(module_name)
        dataset_cls = getattr(module, class_name)

        dtype_name = metadata.get("dtype_name", "float32")
        dtype = getattr(torch, dtype_name, torch.float32)

        dataset = dataset_cls(
            dataset_cfg,
            use_freq_time=metadata.get("use_freq_time", False),
            dtype=dtype,
            split=metadata.get("split", "train")
        )

        TorchRejectionEnsemble._ensure_numeric_labels(
            dataset,
            metadata=metadata,
            dataset_name=f"{class_name}({metadata.get('split', 'train')})"
        )

        subset = Subset(dataset, payload.get("indices", []))
        return DataLoader(
            subset,
            batch_size=payload.get("batch_size"),
            shuffle=payload.get("shuffle", False),
            num_workers=payload.get("num_workers", 0),
            pin_memory=payload.get("pin_memory", False),
            drop_last=payload.get("drop_last", False)
        )

    @staticmethod
    def _ensure_numeric_labels(dataset, *, metadata=None, dataset_name="dataset"):
        labels = getattr(dataset, "labels", None)
        if labels is None:
            return

        labels_arr = np.asarray(labels)
        if np.issubdtype(labels_arr.dtype, np.number):
            return

        try:
            dataset.labels = label_encoding(labels_arr.astype(object))
        except Exception as exc:
            raise ValueError(
                f"Failed to encode string labels for {dataset_name}. Original dtype={labels_arr.dtype}."
            ) from exc

    def save_balanced_splits(self, path, train_loader, train_targets, test_loader, test_targets):
        if not path:
            raise ValueError("A valid save path must be provided.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        payload = {
            "train": self._serialize_loader(train_loader),
            "test": self._serialize_loader(test_loader),
            "train_targets": np.asarray(train_targets),
            "test_targets": np.asarray(test_targets),
        }

        torch.save(payload, path)

    def load_balanced_splits(self, path):
        if not path:
            raise ValueError("A valid load path must be provided.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No balanced split file found at '{path}'.")

        payload = _safe_torch_load(path)

        train_loader = self._deserialize_loader(payload.get("train"))
        test_loader = self._deserialize_loader(payload.get("test"))
        train_targets = np.asarray(payload.get("train_targets"))
        test_targets = np.asarray(payload.get("test_targets"))

        return train_loader, train_targets, test_loader, test_targets


    def _splitTrainData(self, pulse_dataloader, routing_targets, pulse_preds):
        print("targets before balancing. REJECT: ", np.count_nonzero(routing_targets == 1), "no REJECT: ", np.count_nonzero(routing_targets == 0))
        
        pulse_dataset = pulse_dataloader.dataset
        
        routing_targets = np.asarray(routing_targets)
        pulse_preds = np.asarray(pulse_preds)

        pulse_probs = softmax(pulse_preds, axis=1)

        idx_0 = np.where(routing_targets == 0)[0]
        idx_1 = np.where(routing_targets == 1)[0]

        n0 = len(idx_0)
        n1 = len(idx_1)
        n_per_class = min(n0, n1)

        pulse_zero_probs = pulse_probs[idx_0]       # shape: [n0, num_classes]
        pulse_zero_conf =pulse_zero_probs.max(axis=1)          # max-Confidence pro Sample

        # aufsteigend sortiert, somit niedrigste conf zuerst
        pulse_zero_sorted_rel = pulse_zero_conf.argsort()
        pulse_zero_selected_rel = pulse_zero_sorted_rel[:n_per_class]

        pulse_zero_selected_idx = idx_0[pulse_zero_selected_rel]


        selected_indices = np.concatenate([pulse_zero_selected_idx, idx_1])
        np.random.shuffle(selected_indices)

        balanced_routing_targets = routing_targets[selected_indices]

        #neuen X_train_dataloader bauen
        batch_size = pulse_dataloader.batch_size
        num_workers = pulse_dataloader.num_workers
        pin_memory = getattr(pulse_dataloader, "pin_memory", False)
        drop_last = getattr(pulse_dataloader, "drop_last", False)

        balanced_pulse_dataset = Subset(pulse_dataset, selected_indices)
        balanced_pulse_loader = DataLoader(
            balanced_pulse_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        print("targets after balancing. REJECT: ",np.count_nonzero(balanced_routing_targets == 1),"no REJECT: ",np.count_nonzero(balanced_routing_targets == 0))
        
        

        return balanced_pulse_loader, balanced_routing_targets
            
            
    
    
    def forward(self, x):
        return self.predict_proba(x)
                
                
    