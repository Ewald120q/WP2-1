# TensorFlow to PyTorch Migration Guide

This document summarizes the complete migration from TensorFlow/Keras to PyTorch for the pulsar signal classification project.

## Files Modified

### 1. Model Definitions (`single_pulse_classifier_training/training_models.py`)
- **Before**: Keras Sequential models using `layers.Conv2D`, `layers.Dense`, etc.
- **After**: PyTorch `nn.Module` classes with explicit forward() methods
- **Key Changes**:
  - Replaced `from tensorflow.keras import layers, models` with `import torch.nn as nn`
  - Created separate classes for each model architecture
  - Implemented proper tensor shape calculations for flattened layers
  - Models now return raw logits (no softmax activation in forward pass)

### 2. Training Script (`single_pulse_classifier_training/training.py`)
- **Before**: TensorFlow's `model.fit()` with callbacks
- **After**: Custom PyTorch training loop with DataLoader
- **Key Changes**:
  - Replaced TensorFlow imports with PyTorch equivalents
  - Implemented manual training/validation loops
  - Added custom checkpoint saving function
  - Data format changed from channels-last (H,W,C) to channels-first (C,H,W)
  - Manual early stopping implementation
  - GPU detection using `torch.cuda.is_available()`

### 3. Inference Scripts
#### `pipeline_classifier/run_inference.py`
- **Before**: `tensorflow.keras.models.load_model()`
- **After**: Custom PyTorch model loading with state_dict
- **Key Changes**:
  - Added model architecture instantiation before loading weights
  - Replaced TensorFlow tensor operations with PyTorch equivalents
  - Added proper device handling (CPU/GPU)
  - Changed tensor format from (N,H,W,C) to (N,C,H,W)

#### `pipeline_classifier/run_pipeline_simulation.py`
- **Before**: TensorFlow model loading and inference
- **After**: PyTorch model loading and inference
- **Key Changes**:
  - Similar changes to run_inference.py
  - Added device handling for simulation mode

### 4. Configuration Files
#### `pipeline_classifier/config.json`
- **Changes**:
  - Model file extension: `.h5` → `.pth`
  - Added `model_name` and `resolution` parameters
  - Updated worker configuration from `workers_for_tensorflow` to `workers_for_pytorch`

#### `single_pulse_classifier_training/config.json`
- **No changes needed**: Configuration format remains compatible

### 5. Requirements
#### New file: `requirements_pytorch.txt`
- **Contains**: PyTorch, torchvision, and other necessary dependencies
- **Replaces**: TensorFlow dependencies

### 6. Utility Scripts
#### New file: `single_pulse_classifier_training/convert_to_pytorch.py`
- **Purpose**: Creates PyTorch model templates with random weights
- **Note**: Actual model training still required using the training script

## Key Technical Differences

### Data Format
- **TensorFlow**: Channels-last format (N, H, W, C)
- **PyTorch**: Channels-first format (N, C, H, W)

### Model Architecture
- **TensorFlow**: Sequential API with implicit input shapes
- **PyTorch**: Explicit class definition with calculated layer dimensions

### Training Process
- **TensorFlow**: High-level `model.fit()` with automatic metrics tracking
- **PyTorch**: Manual training loops with explicit forward/backward passes

### Model Saving/Loading
- **TensorFlow**: Complete model saved in `.h5` format
- **PyTorch**: State dictionary saved in `.pth` format, requires model instantiation

### GPU Handling
- **TensorFlow**: Automatic memory growth configuration
- **PyTorch**: Manual device placement with `.to(device)`

## Migration Steps for Users

1. **Install PyTorch Dependencies**:
   ```bash
   pip install -r requirements_pytorch.txt
   ```

2. **Update Model Training**:
   - Use the new `training.py` script with existing config files
   - Models will be saved in `.pth` format in the checkpoints directory

3. **Update Inference Configuration**:
   - Change model file extension in config.json from `.h5` to `.pth`
   - Add `model_name` and `resolution` parameters to config files

4. **Retrain Models**:
   - Previous TensorFlow models cannot be directly converted
   - Use `convert_to_pytorch.py` to create model templates
   - Retrain using `python training.py config.json`

5. **Test Pipeline**:
   - Run inference scripts with updated configuration
   - Verify output format matches previous results

## Performance Considerations

- **Memory Usage**: PyTorch may have different memory patterns than TensorFlow
- **Speed**: Performance should be comparable, may vary depending on hardware
- **Compatibility**: Models are now compatible with PyTorch ecosystem tools

## Backward Compatibility

- **Data Processing**: No changes to data preprocessing or candidate detection
- **Output Format**: Prediction outputs remain the same format (.npy files)
- **Configuration**: Most configuration parameters unchanged
- **Pipeline Integration**: Same command-line interfaces maintained

## Testing

After migration, verify:
1. Model training completes without errors
2. Inference produces reasonable predictions
3. Pipeline integration works with existing data processing tools
4. Performance metrics are comparable to original TensorFlow implementation