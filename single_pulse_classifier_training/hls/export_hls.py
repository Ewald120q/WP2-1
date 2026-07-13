import torch
import hls4ml
import training_models
from DMTimeShardDataset import DMTimeShardDataset
from torch.utils.data import DataLoader
from training import label_encoding
from pathlib import Path
import numpy as np

def evaluate_hls(hls_model):
    print("start evaluation")
    data_root = "/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator/outputs"
    prefix = "B0531+21_59000_48386"
    batch_size = 256

    dataset_cfg = {
        "output_dir": data_root,
        "prefix": prefix,
    }

    dataset = DMTimeShardDataset(
        dataset_cfg,
        use_freq_time=False,
        split="val",
    )
    dataset.labels = label_encoding(dataset.labels.astype(object))
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    tp = fp = fn = tn = 0

    for batch in loader:
        x = batch["dm_time"].numpy()
        y_true = batch["label"].numpy()

        y_pred_hls = hls_model.predict(x)
        y_pred_class = np.argmax(y_pred_hls, axis=1)

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
    print(f"HLS Model Val Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("-" * 50)

def convert(evaluate=False):
    device = "cpu"

    model_name = "DM_time_binary_classificator_241002_9_GAP"
    #weights_path = "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/checkpoints_new/ch_point_DM_time_binary_classificator_241002_7_GAP_256/prot-DM_time_binary_classificator_241002_7_GAP-012-0.774-0.744.pth"
    weights_path = "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/checkpoints_new/ch_point_DM_time_binary_classificator_241002_9_GAP_256/prot-DM_time_binary_classificator_241002_9_GAP-015-0.791-0.773.pth"
    
    model = training_models.models_htable[model_name](256, mode="dmt", dropout=False, device=device).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device)["model_state_dict"])
    model.eval()

    model._prepare_input = lambda x: x
    
    del model.dropout_conv
    del model.dropout_fc
    
    model.dropout_conv = lambda x: x
    model.dropout_fc = lambda x: x

    input_shape = (1, 256, 256)


    config = hls4ml.utils.config_from_pytorch_model(
        model, 
        input_shape=input_shape,
        default_precision='ap_fixed<16,6>'
    )
    
    config['Model']['ReuseFactor'] = 256
    config['Model']['Strategy'] = 'Resource'


    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir='hls_project_9_GAP_16_bit_rf256',
        input_shape=input_shape,
        part='xc7a200tsbg484-1'
    )
    

    if evaluate:
        print("start compiling")
        hls_model.compile()    
        evaluate_hls(hls_model)
    else:
        hls_model.write()

if __name__ == "__main__":
    convert(evaluate=True)