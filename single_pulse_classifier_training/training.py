import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from training_models import models_htable, LateFusionCombinedDMFTModel, MidFusionCombinedDMFTModel
from tqdm import tqdm
from datetime import datetime
import math
import warnings
from training_utils import _slugify_component, save_checkpoint, label_encoding,load_config, plot_accuracies

from DMTimeShardDataset import *


def _train(model, model_name, train_dataloader, test_dataloader, optimizer, num_epochs,
           *, val_dataloader=None, criterion=None, scheduler=None, writer=None,
           device=None, patience=None, checkpoint_dir=None, targets_train=None, targets_test=None):
    device = device or next(model.parameters()).device
    criterion = criterion or nn.CrossEntropyLoss()
    patience = patience if patience is not None else num_epochs

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    train_targets_tensor = None
    test_targets_tensor = None
    if targets_train is not None:
        train_targets_tensor = torch.as_tensor(targets_train, dtype=torch.long, device=device)
    if targets_test is not None:
        test_targets_tensor = torch.as_tensor(targets_test, dtype=torch.long, device=device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_offset = 0
        for batch in tqdm(train_dataloader):
            if train_targets_tensor is not None:
                batch_size = batch["label"].shape[0]
                labels = train_targets_tensor[train_offset:train_offset + batch_size]
                train_offset += batch_size
            else:
                labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs.float(), torch.nn.functional.one_hot(labels, num_classes=2).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / max(len(train_dataloader), 1)
        train_acc = correct_train / max(total_train, 1)

        model.eval()
        val_loss = None
        val_acc = None
        if val_dataloader is not None:
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                val_offset = 0
                for batch in val_dataloader:
                    if test_targets_tensor is not None:
                        batch_size = batch["label"].shape[0]
                        labels = test_targets_tensor[val_offset:val_offset + batch_size]
                        val_offset += batch_size
                    else:
                        labels = batch["label"].to(device)
                    outputs = model(batch)
                    loss = criterion(outputs.float(), torch.nn.functional.one_hot(labels, num_classes=2).float())

                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            val_loss = val_running_loss / max(len(val_dataloader), 1)
            val_acc = correct_val / max(total_val, 1)

        test_loss = None
        test_acc = None
        if test_dataloader is not None:
            test_running_loss = 0.0
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                test_offset = 0
                for batch in test_dataloader:
                    if test_targets_tensor is not None:
                        batch_size = batch["label"].shape[0]
                        labels = test_targets_tensor[test_offset:test_offset + batch_size]
                        test_offset += batch_size
                    else:
                        labels = batch["label"].to(device)
                    outputs = model(batch)
                    loss = criterion(outputs.float(), torch.nn.functional.one_hot(labels, num_classes=2).float())

                    test_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
            test_loss = test_running_loss / max(len(test_dataloader), 1)
            test_acc = correct_test / max(total_test, 1)

        #scheduler.step(test_loss)
        #scheduler.step(train_loss)

        loss_dict = {'train': train_loss}
        if val_loss is not None:
            loss_dict['val'] = val_loss
            loss_dict['test'] = test_loss
        writer.add_scalars('Loss', loss_dict, epoch + 1)
        
        acc_dict = {'train': train_acc}
        if val_acc is not None:
            acc_dict['val'] = val_acc
        if test_acc is not None:
            acc_dict['test'] = test_acc
        writer.add_scalars('Accuracy', acc_dict, epoch + 1)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        if val_losses is not None:
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}' +
              (f', Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}' if val_loss is not None else '') +
              (f', Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}' if test_loss is not None else ''))
        
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            save_checkpoint(model, model_name, optimizer, epoch + 1, train_acc, test_acc, checkpoint_dir)
            print(f'New best validation accuracy: {test_acc:.4f} - Model saved!')
            if writer is not None:
                writer.add_scalar('Accuracy/best_test', best_val_acc, epoch + 1)
        else:
            patience_counter += 1
            if patience is not None and patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    wrong_examples = np.empty((0,))
    wrong_labels = None
    final_test_acc = None
    if test_dataloader is not None:
        model.eval()
        wrong_examples_list = []
        wrong_labels_list = []
        correct = 0
        total = 0
        with torch.no_grad():
            analyze_offset = 0
            for batch in test_dataloader:
                if test_targets_tensor is not None:
                    batch_size = batch["label"].shape[0]
                    labels = test_targets_tensor[analyze_offset:analyze_offset + batch_size]
                    analyze_offset += batch_size
                else:
                    labels = batch["label"].to(device)
                metadata = batch.get("metadata")
                outputs = model(batch)
                _, predicted = torch.max(outputs, 1)
                total += predicted.size(0)
                correct += (predicted == labels).sum().item()
                if metadata is not None:
                    metadata = metadata.to(device)
                    wrong_mask = predicted != labels
                    if wrong_mask.any():
                        wrong_examples_list.append(metadata[wrong_mask].cpu().numpy())
                        wrong_labels_list.append(labels[wrong_mask].cpu().numpy())
        final_test_acc = correct / total if total > 0 else 0.0
        if writer is not None:
            writer.add_scalar('Accuracy/test', final_test_acc, 0)
        if wrong_examples_list:
            wrong_examples = np.concatenate(wrong_examples_list, axis=0)
            wrong_labels = np.concatenate(wrong_labels_list, axis=0) if wrong_labels_list else None

    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'test_loss': test_losses,
        'test_accuracy': test_accuracies,
    }

    return history, best_val_acc, best_epoch, final_test_acc, wrong_examples, wrong_labels


def _load_pretrained_model(model_key, checkpoint_path, *, resolution, mode, dropout, device):
    if model_key not in models_htable:
        raise KeyError(f"Unknown model '{model_key}' requested for late fusion.")

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint for '{model_key}' not found at '{checkpoint_path}'.")

    model = models_htable[model_key](resolution, mode, dropout, device).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        warnings.warn(f"Missing keys while loading '{model_key}': {missing}")
    if unexpected:
        warnings.warn(f"Unexpected keys while loading '{model_key}': {unexpected}")
    model.eval()
    return model


def _build_fusion_model(config, device, resolution):
    model_dmt_name = config.get("model_dmt")
    model_ft_name = config.get("model_ft")
    model_name = config["model_name"]
    if not model_dmt_name or not model_ft_name:
        raise ValueError("Late fusion requires 'model_dmt' and 'model_ft' entries in the config.")

    dmt_model = _load_pretrained_model(
        model_dmt_name,
        config.get("model_dmt_path"),
        resolution=resolution,
        mode="dmt",
        dropout=config.get("model_dmt_dropout", False),
        device=device,
    )

    ft_model = _load_pretrained_model(
        model_ft_name,
        config.get("model_ft_path"),
        resolution=resolution,
        mode="ft",
        dropout=config.get("model_ft_dropout", False),
        device=device,
    )

    k = config.get("k")

    freeze_towers = config.get("freeze_towers", True)

    if model_name == "LateFusionCombinedDMFTModel":
        fusion_model = LateFusionCombinedDMFTModel(
            device=device,
            model_dmt=dmt_model,
            model_ft=ft_model,
            k=k,
            freeze_towers=freeze_towers,
        ).to(device)
    elif model_name == "MidFusionCombinedDMFTModel":
        fusion_model = MidFusionCombinedDMFTModel(
            device=device,
            model_dmt=dmt_model,
            model_ft=ft_model,
            k=k,
            freeze_towers=freeze_towers,
        ).to(device)
    else:
        raise SyntaxError(f"{model_name} is currently not available. Use other currently supported fusion models")

    return fusion_model


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
    mode = config["mode"]
    dropout = config["dropout"]

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

    full_train_dataset = DMTimeShardDataset(dataset_cfg, use_freq_time=True, split="train")
    full_train_dataset.labels = label_encoding(full_train_dataset.labels.astype(object))

    val_fraction = config.get("val_fraction", 0.1111)
    if not 0 < val_fraction < 1:
        raise ValueError("'val_fraction' must be between 0 and 1.")

    num_train_samples = len(full_train_dataset)
    if num_train_samples < 2:
        raise ValueError("Need at least 2 training samples to create a validation split.")

    split_idx_val = math.floor(num_train_samples * (1 - val_fraction))
    split_idx_val = min(max(split_idx_val, 1), num_train_samples - 1)

    train_indices = range(0, split_idx_val)
    val_indices = range(split_idx_val, num_train_samples)

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    test_dataset = DMTimeShardDataset(dataset_cfg, use_freq_time=True, split="test")
    test_dataset.labels = label_encoding(test_dataset.labels.astype(object))

    train_loader = DataLoader(train_dataset,
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=config.get("num_workers", 8))
    val_loader = DataLoader(val_dataset,
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=config.get("num_workers", 8))
    test_loader = DataLoader(test_dataset,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            num_workers=config.get("num_workers", 8))

    print("Train Loader length: ", len(train_loader))
    print("Val Loader length: ", len(val_loader))
    print("Test Loader length: ", len(test_loader))
    print("train dataset length: ", len(train_dataset))
    print("val dataset length: ", len(val_dataset))
    print("test dataset length: ", len(test_dataset))

    combine_models = config.get("combine_model-dmt_and_model-ft", False)
    if model_name in ["LateFusionCombinedDMFTModel", "MidFusionCombinedDMFTModel"]:
        if not combine_models:
            warnings.warn("LateFusionCombinedDMFTModel selected but 'combine_model-dmt_and_model-ft' is False. Proceeding without fusion.")
        model = _build_fusion_model(config, device, resolution)
    else:
        if combine_models:
            warnings.warn("'combine_model-dmt_and_model-ft' is set but model_name is not LateFusionCombinedDMFTModel. Flag will be ignored.")
        model = models_htable[model_name](resolution, mode, dropout, device).to(device)
    
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

    history, best_val_acc, best_epoch, test_acc, wrong_examples, wrong_labels = _train(
        model,
        model_name,
        train_loader,
        test_loader,
        optimizer,
        config["num_epochs"],
        val_dataloader=val_loader,
        criterion=criterion,
        scheduler=scheduler,
        writer=writer,
        device=device,
        patience=config.get("patience"),
        checkpoint_dir=checkpoint_dir,
    )

    plot_accuracies(history, resolution, model_name, writer, config, test_acc, wrong_examples, wrong_labels, checkpoint_dir, mode, best_val_acc, best_epoch)
    

def main():        
    # Load configuration file
    config_path = sys.argv[1]
    config = load_config(config_path)

    train(config)
    


if __name__ == "__main__":
    main()
