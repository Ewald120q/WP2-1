import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from DMTimeShardDataset import *


def _slugify_component(value, default):
    """Create a filesystem-friendly component for TensorBoard paths."""
    if value is None:
        return default
    safe = ''.join(ch if ch.isalnum() or ch in ['-', '_', '.'] else '-' for ch in str(value))
    safe = safe.strip('-_.')
    return safe if safe else default

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)


# Function to dynamically select the file based on resolution
def get_filename(config, resolution):
    files_by_res = config["files_by_resolution"]
    # Check if a specific file is mapped to the resolution
    if str(resolution) in files_by_res:
        filename = files_by_res[str(resolution)]
    else:
        # Use default format with the resolution substituted in the filename
        filename = files_by_res["default"].format(res=resolution)
    return os.path.join(config["path_to_files"], filename)


# Function to save model checkpoint
def save_checkpoint(model, model_name,  optimizer, epoch, accuracy, val_accuracy, checkpoint_path):
    #os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if val_accuracy is None:
        val_accuracy = -1
    checkpoint_filename = f'prot-{model_name}-{epoch:03d}-{accuracy:.3f}-{val_accuracy:.3f}.pth'
    full_path = os.path.join(checkpoint_path, checkpoint_filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'val_accuracy': val_accuracy,
    }, full_path)
    
    return full_path

def save_metadata(metadata, labels, checkpoint_path):
    metadata_filename = f"{checkpoint_path}_metadata.npy"
    os.makedirs(os.path.dirname(metadata_filename), exist_ok=True)
    np.save(metadata_filename, metadata)
    
    if labels is not None:
        labels_filename = f"{checkpoint_path}_missedlabels.npy"
        os.makedirs(os.path.dirname(labels_filename), exist_ok=True)
        np.save(labels_filename, labels)

def save_config(config, checkpoint_path):
    config_filename = os.path.join(checkpoint_path, "config.json")
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)

# Function to encode labels into numeric format
def label_encoding(labels):
    # Check if labels are already numeric
    if labels.dtype in [np.int32, np.int64, np.int8, np.int16]:
        print("Labels are already numeric, using as-is")
        return labels
    
    # Convert string labels to numeric
    map_dict = {
        'Artefact': 0,
        'Pulse': 1
    }
    print(f"Converting string labels to numeric: {np.unique(labels)} -> {[map_dict[str(l)] for l in np.unique(labels)]}")
    return np.array([map_dict[str(i)] for i in labels])

def plot_accuracies(history, resolution, model_name, writer, config, test_acc, wrong_examples, wrong_labels, checkpoint_dir, mode, best_test_acc, best_epoch):
    # Plot training and validation loss and accuracy
    plt.clf()
    fig = plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.plot(history['test_loss'])
    plt.title(f'Model Loss: {resolution}x{resolution}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True, ls='--')
    plt.legend(['Train', 'Validation', 'Test'], loc='upper right')

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'])
    plt.plot(history['val_accuracy'])
    plt.plot(history['test_accuracy'])
    plt.title(f'Model Accuracy: {resolution}x{resolution}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True, ls='--')
    plt.legend(['Train', 'Validation', 'Test'], loc='lower right')

    # Add a common title
    plt.suptitle(f'Model {model_name} performance for {resolution}x{resolution} resolution', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    writer.add_figure('plots/loss_accuracy', fig, global_step=len(history['train_loss']))
    plt.savefig(
        os.path.join(config["path_to_images"], f'accuracy_across_epochs_for_{model_name}_{resolution}x{resolution}.jpg'),
        format='jpg',
        dpi=300
    )
    plt.close(fig)

    print(f'Test Accuracy: {test_acc}')
    save_metadata(wrong_examples, wrong_labels, checkpoint_dir)
    save_config(config, checkpoint_dir)
    
    hparam_dict = {
        'model_name': model_name,
        'resolution': resolution,
        'batch_size': config["batch_size"],
        'learning_rate': config["learning_rate"],
        'weight_decay': config["weight_decay"],
        'mode' : mode,
    }
    metric_dict = {
        'metrics/best_test_acc': best_test_acc,
        'metrics/test_acc': test_acc,
        'metrics/final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('nan'),
        'metrics/best_epoch': best_epoch,
    }
    writer.add_hparams(hparam_dict, metric_dict, run_name='hparams')

    writer.close()