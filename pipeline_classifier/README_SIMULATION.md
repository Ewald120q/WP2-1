# Pipeline Simulation Script

## Overview

The `run_pipeline_simulation.py` script provides a simulation of the original `run_pipeline.py` that works without requiring singularity containers or special permissions. It simulates the entire pipeline including:

1. **Buffer creation/destruction** (simulated - just logs actions)
2. **dada_fildb** (reads filterbank file directly)
3. **dada_dbdedispdb** (simulates dedispersion processing)
4. **TensorFlow inference** (runs actual ML model)

## Requirements

Make sure you have the following packages installed in your WP2 conda environment:

```bash
conda activate WP2
conda install numpy tensorflow tqdm
# or using pip:
pip install numpy tensorflow tqdm
```

## Usage

Run the simulation script exactly like the original pipeline:

```bash
python run_pipeline_simulation.py -c config.json
```

## What it does differently

### Original Pipeline:
- Uses singularity containers for `dada_fildb` and `dada_dbdedispdb`
- Requires DADA buffer permissions
- Uses `psrdada` Python bindings

### Simulation Pipeline:
- Reads filterbank files directly with Python
- Simulates dedispersion processing using numpy
- Creates temporary files to simulate data flow
- Uses a custom `SimulatedDataReader` class instead of `psrdada.Reader`

## Output

The script produces the same output as the original:
- `predictions_<filterbank_name>.npy` file containing model predictions
- Console output showing progress and simulation steps

## Limitations

1. **Simplified dedispersion**: The simulation uses a basic dedispersion algorithm that may not perfectly match TransientX output
2. **No real-time processing**: Processes data sequentially rather than in parallel streams
3. **Memory usage**: May use more memory than the original DADA-based approach
4. **Performance**: Will be slower than the optimized C++ DADA tools

## Files created

The simulation script creates:
- `run_inference_simulation.py` (temporary script for inference)
- Temporary data files (automatically cleaned up)
- Same prediction output files as original

## Troubleshooting

### "Import numpy could not be resolved"
Make sure numpy is installed in your conda environment:
```bash
conda activate WP2
conda install numpy
```

### "No module named 'tensorflow'"
Install TensorFlow:
```bash
conda activate WP2
conda install tensorflow
```

### "Permission denied" errors
The simulation should not require any special permissions. If you get permission errors, check that you have write access to the current directory.

### "FileNotFoundError" for filterbank
Make sure the filterbank file specified in your config.json exists and the path is correct.

## Technical Details

The simulation works by:
1. Reading the original filterbank file specified in config
2. Simulating the data flow that would happen through DADA buffers
3. Applying simplified dedispersion (time delays based on DM values)
4. Writing processed data to a temporary file
5. Running a modified inference script that reads from the temporary file instead of DADA buffers
6. Cleaning up temporary files

This approach maintains the same interface and produces comparable results while avoiding the need for singularity containers.