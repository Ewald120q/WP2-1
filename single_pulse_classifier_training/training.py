import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from training_models import models_htable
from tqdm import tqdm
import math
from datetime import datetime

from DMTimeShardDataset import *


def _slugify_component(value, default):
    """Create a filesystem-friendly component for TensorBoard paths."""
    if value is None:
        return default
    safe = ''.join(ch if ch.isalnum() or ch in ['-', '_', '.'] else '-' for ch in str(value))
    safe = safe.strip('-_.')
    return safe if safe else default

# Function to load the configuration file
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
def save_checkpoint(model, optimizer, epoch, accuracy, val_accuracy, checkpoint_path):
    #os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint_filename = f'prot-{epoch:03d}-{accuracy:.3f}-{val_accuracy:.3f}.pth'
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

def train(config):
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.device_count()} GPU(s)")
        print(f"Current device: {torch.cuda.get_device_name()}")
    else:
        print("No GPU found, using CPU")
        
    # Extract parameters from the configuration
    resolution = config["resolution"]
    model_name = config["model_name"]
    use_freq_time = config["use_freq_time"]

    checkpoint_dir = os.path.join(config["path_to_checkpoints"], f'ch_point_{model_name}_{resolution}')

    tensorboard_cfg = config.get("tensorboard", {})
    configured_root = tensorboard_cfg.get("log_root") or config.get("tensorboard_log_dir")
    default_tensorboard_root = os.path.join(checkpoint_dir, "tensorboard")
    tensorboard_base = configured_root or default_tensorboard_root
    experiment_component = _slugify_component(
        tensorboard_cfg.get("experiment_name") or config.get("tensorboard_experiment"),
        f"{model_name}_{resolution}"
    )
    run_component = _slugify_component(
        tensorboard_cfg.get("run_name") or config.get("run_name"),
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard_log_dir = os.path.join(tensorboard_base, experiment_component, run_component)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    writer.add_text("run/config", json.dumps(config, indent=2), 0)
    writer.add_text("run/experiment", experiment_component, 0)
    writer.add_text("run/name", run_component, 0)
    
    dataset_cfg = {
    "output_dir": config["path_to_files"],
    "prefix": config["dataset_prefix"],  # e.g. basename of the filterbank
    }

    
    full_dataset = DMTimeShardDataset(dataset_cfg, use_freq_time=True)
    labels_numeric = label_encoding(full_dataset.labels.astype(object))  # only once if needed
    
    full_dataset.labels = labels_numeric

    #data is already shuffled inside shards
    indices = np.arange(len(full_dataset))
    train_idx = indices[:math.floor(len(indices)*0.7)]
    val_idx = indices[math.floor(len(indices)*0.7):]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(full_dataset, batch_size=config["batch_size"],
                            sampler=train_sampler, num_workers=config.get("num_workers", 8))
    val_loader = DataLoader(full_dataset, batch_size=config["batch_size"],
                            sampler=val_sampler, num_workers=config.get("num_workers", 8))


    print("Train Loader length: ",len(train_loader))
    print("Val Loader length: ",len(val_loader))
    print("full dataset length: ", len(full_dataset))

    # Initialize the model
    model = models_htable[model_name](resolution, use_freq_time, device).to(device)
    
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    criterion = nn.CrossEntropyLoss()

    # Training tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(config["num_epochs"]):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for _, batch in tqdm(enumerate(train_loader)):
            labels = batch["label"].to(device)
            #labels = torch.nn.functional.one_hot(labels)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs.float(), torch.nn.functional.one_hot(labels, num_classes=2).float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                #inputs = torch.unsqueeze(inputs, 1)
                labels = batch["label"].to(device)
                outputs = model(batch)
                loss = criterion(outputs.float(), torch.nn.functional.one_hot(labels, num_classes=2).float())
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_loss = val_running_loss / len(val_loader)
        val_acc = correct_val / total_val
        
        scheduler.step(val_loss)

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch + 1)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch + 1)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch + 1, train_acc, val_acc, checkpoint_dir)
            print(f'New best validation accuracy: {val_acc:.4f} - Model saved!')
            writer.add_scalar('Accuracy/best_val', best_val_acc, epoch + 1)
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config["patience"]:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    # Create history-like object for plotting
    history = {
        'loss': train_losses,
        'accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    }

    # Plot training and validation loss and accuracy
    plt.clf()
    fig = plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'Model Loss: {resolution}x{resolution}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True, ls='--')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(f'Model Accuracy: {resolution}x{resolution}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True, ls='--')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Add a common title
    plt.suptitle(f'Model {model_name} performance for {resolution}x{resolution} resolution', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    writer.add_figure('plots/loss_accuracy', fig, global_step=len(train_losses))
    plt.savefig(
        os.path.join(config["path_to_images"], f'accuracy_across_epochs_for_{model_name}_{resolution}x{resolution}.jpg'),
        format='jpg',
        dpi=300
    )
    plt.close(fig)

    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        wrong_examples = []
        wrong_labels = []
        for batch in val_loader:
            labels = batch["label"].to(device)
            metadata = batch["metadata"].to(device)
            outputs = model(batch)

            _, predicted = torch.max(outputs, 1)
            total += predicted.size(0)
            correct += (predicted == labels).sum().item()
            
            #for logging
            wrong_examples.append(metadata[predicted != labels].cpu().numpy())
            wrong_labels.append(labels[predicted != labels].cpu().numpy())
        
    test_acc = correct / total
    print(f'Test Accuracy: {test_acc}')
    writer.add_scalar('Accuracy/test', test_acc, 0)
    
    wrong_examples = np.concatenate(wrong_examples, axis=0)
    wrong_labels = np.concatenate(wrong_labels, axis=0)
    save_metadata(wrong_examples, wrong_labels, checkpoint_dir)
    save_config(config, checkpoint_dir)
    
    hparam_dict = {
        'model_name': model_name,
        'resolution': resolution,
        'batch_size': config["batch_size"],
        'learning_rate': config["learning_rate"],
        'weight_decay': config["weight_decay"],
        'use_freq_time': int(use_freq_time),
    }
    metric_dict = {
        'metrics/best_val_acc': best_val_acc,
        'metrics/test_acc': test_acc,
        'metrics/final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('nan'),
        'metrics/best_epoch': best_epoch,
    }
    writer.add_hparams(hparam_dict, metric_dict, run_name='hparams')

    writer.close()


def main():        
    # Load configuration file
    config_path = sys.argv[1]
    config = load_config(config_path)

    train(config)
    


if __name__ == "__main__":
    main()
