import numpy as np
import torch
import tqdm

def get_predictions(f, data_loader, return_embeddings=False, return_labels=False, pbar_desc=None):
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
    for xbatch, ybatch in tqdm.tqdm(data_loader, desc=pbar_desc, disable=pbar_desc is None):
        xbatch = xbatch.to(device)
        with torch.no_grad():
            if return_embeddings:
                emebddings = f.features(xbatch)
                preds = f.classifier(emebddings)
                all_emebddings.append(emebddings)
            else:
                preds = f(xbatch)
            
            if return_labels:
                labels.extend([yi for yi in ybatch])

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

    def fit(self, train_loader, test_loader):
        """
        Fits the TorchRejectionEnsemble model to the given dataset.

        Args:
            dataset_loader (torch.utils.data.DataLoader): The data loader for the dataset.

        Returns:
            The result of the _fit method.

        """
        train_preds_small, X_train = get_predictions(self.fsmall, train_loader, True)
        Y_train = [yi for _, yb in train_loader for yi in yb]

        
        train_preds_big = get_predictions(self.fbig, train_loader, False)
        
        test_preds_small, X_test = get_predictions(self.fsmall, test_loader, True)
        Y_test = [yi for _, yb in test_loader for yi in yb]

        
        test_preds_big = get_predictions(self.fbig, test_loader, False)
        

        return self._fit(X_train, Y_train, train_preds_small, train_preds_big, X_test, Y_test, test_preds_small, test_preds_big)

    def _fit(self, X_train, Y_train, train_preds_small, train_preds_big, X_test, Y_test, test_preds_small, test_preds_big):
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
        
        #create pseudo labels
        train_preds_big = train_preds_big.argmax(1)
        train_preds_small = train_preds_small.argmax(1)
        targets_train = []
        for i in range(train_preds_big.shape[0]):
            if train_preds_big[i] == train_preds_small[i]:
                targets_train.append(0)
            else:
                if train_preds_big[i] == Y_train[i]:
                    targets_train.append(1)
                else:
                    targets_train.append(0)
        
        targets_train = np.array(targets_train)
        
        #for testing rejector
        test_preds_big = test_preds_big.argmax(1)
        test_preds_small = test_preds_small.argmax(1)
        targets_test = []
        for i in range(test_preds_big.shape[0]):
            if test_preds_big[i] == test_preds_small[i]:
                targets_test.append(0)
            else:
                if test_preds_big[i] == Y_test[i]:
                    targets_test.append(1)
                else:
                    targets_test.append(0)
        
        targets_test = np.array(targets_test)
        if np.unique(targets_test).shape[0] > 1:
            self.rejector.fit(X_train, targets_train, X_test, targets_test) #testen rejector auf test daten, um overfitting vom rejector zu vermeiden. (NUR PSEUDO LABELS!)
        else:
            self.rejector = None

        return self.fsmall, self.fbig, self.rejector


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
        T = T.to(device)

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
                x_features = self.fsmall.features(T)
                r_pred = self.rejector.predict_proba(x_features.cpu().numpy())

            if self.calibration:
                M = len(T)
                P = int(np.floor(self.p * M))

                r_pred_tensor = torch.tensor(r_pred, device=device)
                
                # Determine indices for Ts and Tb using boolean masks
                _, Tb_sorted_indices = torch.sort(r_pred_tensor[:, 1], descending=True)
                Tb_sorted_indices = Tb_sorted_indices[:P]

                Tb_mask = torch.zeros(M, dtype=torch.bool, device=device)
                Tb_mask[Tb_sorted_indices] = True
                Ts_mask = ~Tb_mask  
            else:
                r_pred_tensor = torch.tensor(r_pred, device=device)
                
                Ts_mask = r_pred_tensor.argmax(dim=1) == 0
                Tb_mask = r_pred_tensor.argmax(dim=1) == 1

            with torch.no_grad():
                fsmall_preds = self.fsmall.classifier(x_features[Ts_mask])

                ypred = torch.empty((T.shape[0], fsmall_preds.shape[1]), dtype=fsmall_preds.dtype, device=device)
                ypred[Ts_mask] = fsmall_preds
                if not torch.all(Tb_mask == False):
                    fbig_preds = self.fbig(T[Tb_mask])
                    ypred[Tb_mask] = fbig_preds
            
            if return_cnt:
                return ypred, Tb_mask.sum().item()
            else:
                return ypred