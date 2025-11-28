# Model Training Pipeline

This repository contains the training pipeline for binary classification of single-pulse candidates in DM-time plane data. The pipeline is designed to preprocess data, train convolutional neural networks (CNNs), and save the results, including model checkpoints and performance plots.

## Directory Structure

- **`data/`**: Contains the training datasets and label files.
- **`checkpoints/`**: Stores the model checkpoints saved during training.
- **`images/`**: Contains plots of training and validation performance metrics.

## Requirements

The training pipeline relies on PyTorch and TensorBoard along with a few utility libraries:

- **Python** (≥3.8)
- **PyTorch** with CUDA support when available
- **TensorBoard**
- **Matplotlib**
- **NumPy**
- **scikit-learn**

Install these dependencies using `pip`:
```bash
pip install torch torchvision torchaudio matplotlib numpy scikit-learn tensorboard
```

## Configuration File

The training process is controlled using a JSON configuration file. Below is an example:

```json
{
    "path_to_files": "data/",
    "path_to_checkpoints": "checkpoints/",
    "path_to_images": "images/",
   "tensorboard_log_dir": "tensorboard_runs/",
   "run_name": "experiment_001",
   "tensorboard": {
      "log_root": "tensorboard_runs/",
      "experiment_name": "res256_dropout",
      "run_name": "adam_lr1e-4"
   },
    "resolution": 256,
    "model_name": "DM_time_binary_classificator_241002_3",
    "files_by_resolution": {
        "256": "B0531+21_59000_48386_DM_time_dataset_realbased_training.npy",
        "default": "B0531+21_59000_48386_DM_time_dataset_realbased_training_{res}x{res}.npy"
    },
    "labels": "B0531+21_59000_48386_DM_time_dataset_realbased_labels_training.npy",
    "learning_rate": 0.0001,
   "weight_decay": 0.0001,
   "batch_size": 1024,
    "num_epochs": 100,
   "patience": 5
}
```

### Key Parameters
- **`path_to_files`**: Path to the directory containing the training data and labels.
- **`path_to_checkpoints`**: Path where model checkpoints will be saved.
- **`path_to_images`**: Path to save training and validation performance plots.
- **`tensorboard_log_dir`** *(optional, legacy)*: Root folder where TensorBoard event files will be written. If omitted, it defaults to `<path_to_checkpoints>/tensorboard_runs/`.
- **`run_name`** *(optional, legacy)*: Fallback tag appended to the TensorBoard log path.
- **`tensorboard` block (optional)**:
   - **`log_root`**: Root directory that contains all experiments. Defaults to `<path_to_checkpoints>/ch_point_<model>_<resolution>/tensorboard/` to keep logs alongside checkpoints.
   - **`experiment_name`**: Folder used to group comparable runs (e.g., `res256_dropout`).
   - **`run_name`**: Per-run identifier (e.g., encode hyperparameters like `adam_lr1e-4`).
- **`resolution`**: Resolution of the DM-time data (e.g., 256x256).
- **`model_name`**: Name of the model architecture to use.
- **`files_by_resolution`**: Mapping of resolution to dataset filenames.
- **`labels`**: Filename of the label file.
- **`learning_rate`**: Learning rate for model optimization.
- **`num_epochs`**: Maximum number of training epochs.
- **`patience`**: Number of epochs without improvement to trigger early stopping.

## TensorBoard Logging

TensorBoard summaries are recorded automatically during training. Each epoch logs training and validation loss/accuracy curves along with the learning rate, and the final evaluation accuracy is added at the end of training.

1. **Configure the log directory** (optional): set `tensorboard_log_dir` and `run_name` in the config file for deterministic locations. When omitted, the trainer creates a timestamped folder under `path_to_checkpoints/tensorboard/`.
2. **Launch TensorBoard** while training (or afterwards):
```bash
tensorboard --logdir tensorboard_runs/
```
3. **Monitor metrics** in the browser to compare experiments in real time.

### Comparing multiple runs

- Keep related runs under the same `experiment_name`; each `run_name` becomes a separate curve in TensorBoard, making overlays straightforward.
- Encode important hyperparameters inside `run_name` (e.g., `adam_lr1e-4_bs1024`) to identify traces quickly.
- Hyperparameters and final metrics are logged via `add_hparams`, so the TensorBoard "HParams" dashboard can filter/sort runs without additional work.

## Workflow

1. **Load Configuration**: The pipeline loads training parameters and paths from the configuration file.

2. **Prepare Data**:
   - Dynamically selects the dataset file based on resolution.
   - Loads the DM-time data and corresponding labels.
   - Splits the data into training and validation sets (80% training, 20% validation).

3. **Train the Model**:
   - Initializes the selected model architecture.
   - Compiles the model with the Adam optimizer and sparse categorical cross-entropy loss.
   - Trains the model with early stopping and checkpoint saving.

4. **Save Results**:
   - Saves the best-performing model as a checkpoint.
   - Plots training and validation loss/accuracy over epochs and saves the plot in the `images/` directory.

## Example Usage

1. **Prepare the Environment**:
   Ensure the data, configuration file, and required dependencies are set up.

2. **Run the Training Script**:
   ```bash
   python training.py config.json
   ```

3. **Output**:
   - Checkpoints saved in `checkpoints/`.
   - Training and validation performance plots saved in `images/`.

## Outputs

- **Model Checkpoints**: Saved with filenames indicating epoch, training accuracy, and validation accuracy.
- **Performance Plots**: Visualizations of training/validation loss and accuracy across epochs.
