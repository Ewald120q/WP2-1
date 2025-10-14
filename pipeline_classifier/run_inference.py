import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from psrdada import Reader
from utils import load_config, normalize_image_to_255
import sys
sys.path.append('../single_pulse_classifier_training')
from training_models import models_htable

# Set up argument parser to accept configuration file path
parser = argparse.ArgumentParser(description="Inference pipeline")
parser.add_argument('-c', '--config', type=str, required=True, 
                   help="Path to configuration file")
args = parser.parse_args()

# Load configuration parameters from specified file
config = load_config(args.config)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pre-trained model
def load_pytorch_model(model_path, model_name, resolution):
    """Load PyTorch model from checkpoint"""
    model = models_htable[model_name](resolution)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# Load the model
model = load_pytorch_model(
    f'{config["path_to_models"]}{config["name_of_the_model"]}',
    config["model_name"],
    config["resolution"]
)

# Initialize DADA reader with hexadecimal key from config
reader = Reader(int(str(config["key_output"]), 16))

# Generate output filename by removing extension from filterbank name
output_filename = f"predictions_{config['name_of_the_filterbank'].split('.')[0]}.npy"

# Create memory-mapped array for efficient disk-backed storage
# This allows incremental saving without loading full array in memory
predictions_array = np.lib.format.open_memmap(
    output_filename,       # Output file path
    dtype=np.int32,          # Data type (can handle variable-length sequences)
    mode='w+',             # Read/write mode, creates new file
    shape=(config["n_spectra"],)  # Pre-allocate array size
)

# Process each spectrum in the input data
for i in trange(config["n_spectra"]):
    # Get next data page from DADA buffer
    page = reader.getNextPage()
    
    # Convert raw bytes to float32 numpy array
    data = np.frombuffer(page, dtype=np.float32)
    
    # Calculate dimensions for reshaping
    total_size = data.size
    num_dms = 256          # Fixed number of DM trials
    num_dumps = total_size // num_dms
    
    # Reshape 1D array into 2D (DM trials × time samples)
    data = data.reshape((num_dms, num_dumps))
    
    # Resize to match model input expectations if needed
    # Note: The original TensorFlow resize operation is commented out
    # as PyTorch models should be trained with the correct input size
    
    # Normalize and flip the image vertically
    normalized_image = normalize_image_to_255(data[::-1])
    
    # Convert to PyTorch tensor with proper dimensions (1, 1, H, W)
    input_tensor = torch.FloatTensor(normalized_image).unsqueeze(0).unsqueeze(0).to(device)
    
    # Run model inference
    with torch.no_grad():
        prediction = model(input_tensor)
        probabilities = F.softmax(prediction, dim=1)
        predicted_class = torch.argmax(prediction, dim=1)
    
    # Store prediction in memory-mapped array
    predictions_array[i] = int(predicted_class.cpu().item())
    
    # Mark buffer page as processed
    reader.markCleared()
    
    # Periodically flush writes to disk (every 100 spectra)
    if (i + 1) % 100 == 0:
        predictions_array.flush()

# Final flush to ensure all data is written
predictions_array.flush()

# Clean up DADA reader connection
reader.disconnect()
