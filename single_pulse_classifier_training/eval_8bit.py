import torch
import torch.nn as nn
from DMTimeShardDataset import DMTimeShardDataset
from torch.utils.data import DataLoader
from training import label_encoding
import training_models
import numpy as np
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

def evaluate_8bit():
    device = "cpu"
    
    model_name = "DM_time_binary_classificator_241002_9_GAP"
    weights_path = "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/checkpoints_new/ch_point_DM_time_binary_classificator_241002_9_GAP_256/prot-DM_time_binary_classificator_241002_9_GAP-015-0.791-0.773.pth"
    
    float_model = training_models.models_htable[model_name](256, mode="dmt", dropout=False, device=device).to(device)
    float_model.load_state_dict(torch.load(weights_path, map_location=device)["model_state_dict"])
    float_model.eval()


    float_model._prepare_input = lambda x: x

    qconfig_dict = {"": get_default_qconfig("fbgemm")}
    example_input = (torch.randn(1, 1, 256, 256),)
    
    prepared_model = prepare_fx(float_model, qconfig_dict, example_inputs=example_input)
    

    data_root = "/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator/outputs"
    dataset_cfg = {"output_dir": data_root, "prefix": "B0531+21_59000_48386"}
    
    # for calibration
    dataset_train = DMTimeShardDataset(dataset_cfg, use_freq_time=False, split="train")
    dataset_train.labels = label_encoding(dataset_train.labels.astype(object))
    loader_train = DataLoader(dataset_train, batch_size=256, shuffle=True, num_workers=4)

    # for evaluation
    dataset_val = DMTimeShardDataset(dataset_cfg, use_freq_time=False, split="val")
    dataset_val.labels = label_encoding(dataset_val.labels.astype(object))
    loader_val = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=4)

    # min/max calibration
    with torch.no_grad():
        for i, batch in enumerate(loader_train):
            if i >= 10: # 2560 samples
                break
            prepared_model(batch["dm_time"].unsqueeze(1))
            
    print("convert model to 8bit")
    quantized_model = convert_fx(prepared_model)

    tp = fp = fn = tn = 0

    # inference on 8-bit model
    print("start evaluation")
    with torch.no_grad():
        for batch in loader_val:
            x = batch["dm_time"].unsqueeze(1)
            y_true = batch["label"].numpy()

            outputs = quantized_model(x)
            y_pred_class = outputs.argmax(dim=1).numpy()

            positive_label = 1
            positive_pred = y_pred_class == positive_label
            positive_true = y_true == positive_label

            tp += np.logical_and(positive_pred, positive_true).sum()
            fp += np.logical_and(positive_pred, ~positive_true).sum()
            fn += np.logical_and(~positive_pred, positive_true).sum()
            tn += np.logical_and(~positive_pred, ~positive_true).sum()

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("-" * 50)
    print(f"PyTorch 8-Bit (INT8) Model Val Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    evaluate_8bit()
