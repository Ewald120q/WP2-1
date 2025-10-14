import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from training_models import models_htable


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
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint_filename = f'prot-{epoch:03d}-{accuracy:.3f}-{val_accuracy:.3f}.pth'
    full_path = os.path.join(checkpoint_path, checkpoint_filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'val_accuracy': val_accuracy,
    }, full_path)
    
    return full_path


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


def main():
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.device_count()} GPU(s)")
        print(f"Current device: {torch.cuda.get_device_name()}")
    else:
        print("No GPU found, using CPU")
        
    # Load configuration file
    config_path = sys.argv[1]
    config = load_config(config_path)

    # Extract parameters from the configuration
    resolution = config["resolution"]
    model_name = config["model_name"]

    # Dynamically select the data file based on resolution
    filename = get_filename(config, resolution)
    labels_file = os.path.join(config["path_to_files"], config["labels"])

    # Load the data and labels
    data = np.load(filename)
    labels = np.load(labels_file)

    # Convert data to PyTorch format (already in N, C, H, W format from dataset creator)
    data = data.astype(np.float32)
    labels = label_encoding(labels)

    # Split the dataset into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    # Initialize the model
    model = models_htable[model_name](resolution).to(device)
    
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Training tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    patience_counter = 0
    
    checkpoint_dir = os.path.join(config["path_to_checkpoints"], f'ch_point_{model_name}_{resolution}')

    # Training loop
    for epoch in range(config["num_epochs"]):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = correct_val / total_val
        
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
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch + 1, train_acc, val_acc, checkpoint_dir)
            print(f'New best validation accuracy: {val_acc:.4f} - Model saved!')
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
    plt.figure(figsize=(12, 4))

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
    plt.savefig(
        os.path.join(config["path_to_images"], f'accuracy_across_epochs_for_{model_name}_{resolution}x{resolution}.jpg'),
        format='jpg',
        dpi=300
    )

    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        test_acc = correct / total
        print(f'Test Accuracy: {test_acc}')


if __name__ == "__main__":
    main()
