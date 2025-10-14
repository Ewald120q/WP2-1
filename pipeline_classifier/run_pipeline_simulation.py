import subprocess
import argparse
import os
import tempfile
import time

from utils import run_command, load_config

parser = argparse.ArgumentParser(description='Bowtie recognition pipeline (simulation mode)')
parser.add_argument('-c', '--config', type=str, required=True, help='Config file')

args = parser.parse_args()
    
name_of_config = os.path.splitext(os.path.basename(args.config))[0]
config = load_config(args.config)

def simulate_create_buffer(buffer_size, num_blocks, key):
    """Simulate buffer creation - just print what would be done"""
    print(f"[SIMULATION] Creating buffer with key {key}, size {buffer_size}, blocks {num_blocks}")
    return None

def simulate_kill_buffer(key):
    """Simulate buffer destruction - just print what would be done"""
    print(f"[SIMULATION] Destroying buffer with key {key}")
    return None

def simulate_kill_dada_processes():
    """Simulate killing DADA processes - just print what would be done"""
    print("[SIMULATION] Killing dada_fildb and dada_dbdedispdb processes")

def preprocess_filterbank_data(filterbank_path, output_file, config):
    """
    Simulate the entire dada_fildb + dada_dbdedispdb pipeline
    by reading the filterbank file and creating simulated dedispersed data
    """
    print(f"[SIMULATION] Processing filterbank: {filterbank_path}")
    
    try:
        import numpy as np
        
        # Read filterbank file
        with open(filterbank_path, 'rb') as f:
            # Skip header (simplified - real filterbank parsing is more complex)
            header_size = 1024
            f.seek(header_size)
            
            # Read all remaining data
            raw_data = f.read()
            
        # Ensure data length is divisible by 4 (size of float32)
        bytes_per_float = 4
        usable_bytes = (len(raw_data) // bytes_per_float) * bytes_per_float
        
        if usable_bytes != len(raw_data):
            print(f"[SIMULATION] Truncating data from {len(raw_data)} to {usable_bytes} bytes to align with float32")
            raw_data = raw_data[:usable_bytes]
            
        # Convert to numpy array (make a writable copy)
        data_array = np.frombuffer(raw_data, dtype=np.float32).copy()
        
        # Simulate dedispersion processing
        # Create chunks that simulate what dada_dbdedispdb would output
        num_dms = 256
        chunk_size = config['input_buffer_size'] // 4  # 4 bytes per float32
        
        # Ensure chunk_size is compatible with num_dms
        samples_per_dm_per_chunk = chunk_size // num_dms
        if samples_per_dm_per_chunk == 0:
            # If chunk is too small, increase it
            samples_per_dm_per_chunk = 1
            chunk_size = num_dms * samples_per_dm_per_chunk
            print(f"[SIMULATION] Adjusted chunk size to {chunk_size} samples ({chunk_size * 4} bytes)")
        
        processed_chunks = []
        total_chunks = min(config['n_spectra'], len(data_array) // chunk_size)
        
        if total_chunks == 0:
            # If we don't have enough data for even one chunk, create a minimal chunk
            total_chunks = 1
            chunk_size = min(len(data_array), num_dms * 256)  # Default to 256 samples per DM
        
        print(f"[SIMULATION] Processing {total_chunks} chunks of {chunk_size} samples each...")
        
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(data_array))
            chunk = data_array[start_idx:end_idx]
            
            # Pad chunk if it's too short
            if len(chunk) < chunk_size:
                padding_needed = chunk_size - len(chunk)
                chunk = np.pad(chunk, (0, padding_needed), mode='constant', constant_values=0)
            
            # Simulate dedispersion by reshaping and adding simple delays
            samples_per_dm = len(chunk) // num_dms
            if samples_per_dm > 0:
                # Truncate to fit exact number of DM trials
                chunk_truncated = chunk[:num_dms * samples_per_dm]
                dedispersed = chunk_truncated.reshape(num_dms, samples_per_dm)
                
                # Add simulated dedispersion delays
                for dm_idx in range(num_dms):
                    shift = dm_idx // 10  # Simple delay simulation
                    if shift > 0 and shift < samples_per_dm:
                        dedispersed[dm_idx] = np.roll(dedispersed[dm_idx], shift)
                
                processed_chunks.append(dedispersed.astype(np.float32))
            else:
                # Create a minimal dedispersed chunk if samples_per_dm is 0
                minimal_chunk = np.zeros((num_dms, 1), dtype=np.float32)
                processed_chunks.append(minimal_chunk)
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"[SIMULATION] Processed {i+1}/{total_chunks} chunks")
        
        # Write processed data to output file
        with open(output_file, 'wb') as f:
            for chunk in processed_chunks:
                f.write(chunk.tobytes())
        
        print(f"[SIMULATION] Wrote {len(processed_chunks)} processed chunks to {output_file}")
        return len(processed_chunks)
        
    except ImportError:
        print("[ERROR] NumPy is required for simulation. Please install numpy in your WP2 conda environment.")
        return 0
    except Exception as e:
        print(f"[SIMULATION] Error processing filterbank: {e}")
        return 0

def create_simulation_inference_script():
    """Create a modified version of run_inference.py that works with simulated data"""
    simulation_script = '''import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from utils import load_config, normalize_image_to_255
import sys
sys.path.append('../single_pulse_classifier_training')
from training_models import models_htable

# Simulated Reader class to replace psrdada.Reader
class SimulatedDataReader:
    def __init__(self, data_file, key):
        self.data_file = data_file
        self.key = key
        self.page_count = 0
        self.file_handle = open(data_file, 'rb')
        
    def getNextPage(self):
        # Read a chunk of data (256 DMs worth)
        num_dms = 256
        chunk_size = num_dms * 256 * 4  # 256 samples per DM, 4 bytes per float32
        data = self.file_handle.read(chunk_size)
        if not data or len(data) < chunk_size:
            raise StopIteration("No more data")
        self.page_count += 1
        return data
        
    def markCleared(self):
        pass
        
    def disconnect(self):
        self.file_handle.close()
        print(f"[SIMULATION] Disconnected reader after {self.page_count} pages")

# Load PyTorch model function
def load_pytorch_model(model_path, model_name, resolution):
    """Load PyTorch model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models_htable[model_name](resolution)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

# Set up argument parser
parser = argparse.ArgumentParser(description="Inference pipeline (simulation)")
parser.add_argument('-c', '--config', type=str, required=True, help="Path to configuration file")
parser.add_argument('--data-file', type=str, required=True, help="Path to simulated data file")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

print(f"[SIMULATION] Loading model: {config['path_to_models']}{config['name_of_the_model']}")
model, device = load_pytorch_model(
    f'{config["path_to_models"]}{config["name_of_the_model"]}',
    config["model_name"],
    config["resolution"]
)

# Initialize simulated reader
reader = SimulatedDataReader(args.data_file, config["key_output"])

# Generate output filename
output_filename = f"predictions_{config['name_of_the_filterbank'].split('.')[0]}.npy"

# Create predictions array
predictions_array = np.lib.format.open_memmap(
    output_filename,
    dtype=np.int32,
    mode='w+',
    shape=(config["n_spectra"],)
)

print(f"[SIMULATION] Processing {config['n_spectra']} spectra...")

# Process each spectrum
processed_count = 0
for i in trange(config["n_spectra"]):
    try:
        # Get next data page
        page = reader.getNextPage()
        
        # Convert to numpy array
        data = np.frombuffer(page, dtype=np.float32)
        
        # Calculate dimensions
        total_size = data.size
        num_dms = 256
        num_dumps = total_size // num_dms
        
        if num_dumps > 0:
            # Reshape data (DM trials × time samples)
            data = data[:num_dms * num_dumps].reshape((num_dms, num_dumps))
            
            # Normalize and flip the image vertically
            normalized_image = normalize_image_to_255(data[::-1])
            
            # Convert to PyTorch tensor with proper dimensions (1, 1, H, W)
            input_tensor = torch.FloatTensor(normalized_image).unsqueeze(0).unsqueeze(0).to(device)
            
            # Run model inference
            with torch.no_grad():
                prediction = model(input_tensor)
                probabilities = F.softmax(prediction, dim=1)
                predicted_class = torch.argmax(prediction, dim=1)
            
            # Store prediction
            predictions_array[i] = int(predicted_class.cpu().item())
            processed_count += 1
        else:
            # If not enough data, use default prediction
            predictions_array[i] = 0
            
        # Mark as cleared
        reader.markCleared()
        
        # Flush periodically
        if (i + 1) % 100 == 0:
            predictions_array.flush()
            
    except StopIteration:
        print(f"[SIMULATION] Reached end of data at spectrum {i}")
        # Fill remaining with default predictions
        for j in range(i, config["n_spectra"]):
            predictions_array[j] = 0
        break
    except Exception as e:
        print(f"[SIMULATION] Error processing spectrum {i}: {e}")
        predictions_array[i] = 0

# Final flush
predictions_array.flush()
reader.disconnect()

print(f"[SIMULATION] Processed {processed_count} spectra successfully")
print(f"[SIMULATION] Output saved to: {output_filename}")
'''
    
    script_path = '/cephfs/users/oleksjuk/MA/WP2-1/pipeline_classifier/run_inference_simulation.py'
    with open(script_path, 'w') as f:
        f.write(simulation_script)
    
    return script_path

# Main pipeline execution
print("=== Starting Pipeline Simulation (No Singularity) ===")

# 1. Simulate creating buffers
print("Step 1: Creating buffers...")
simulate_create_buffer(config['input_buffer_size'], 16, config['key_input'])
simulate_create_buffer(config['input_buffer_size']*16, 16, config['key_output'])

# 2. Process filterbank data (simulating dada_fildb + dada_dbdedispdb)
print("Step 2: Simulating data processing...")
temp_data_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
temp_data_file.close()

filterbank_path = os.path.join(config['path_to_filterbanks'], config['name_of_the_filterbank'])
chunks_processed = preprocess_filterbank_data(filterbank_path, temp_data_file.name, config)

if chunks_processed == 0:
    print("[ERROR] Failed to process filterbank data. Exiting.")
    os.unlink(temp_data_file.name)
    exit(1)

# 3. Create and run simulation inference script
print("Step 3: Running inference...")
inference_script = create_simulation_inference_script()

# Run inference with simulation script
inference_command = f'python {inference_script} -c {args.config} --data-file {temp_data_file.name}'
print(f"[SIMULATION] Running: {inference_command}")
run_command(inference_command, wait=True)

# 4. Cleanup
print("Step 4: Cleaning up...")
simulate_kill_dada_processes()
simulate_kill_buffer(config['key_input'])
simulate_kill_buffer(config['key_output'])

# Remove temporary files
os.unlink(temp_data_file.name)
if os.path.exists(inference_script):
    os.unlink(inference_script)

print('=== Pipeline simulation completed successfully ===')