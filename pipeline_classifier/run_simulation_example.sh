#!/bin/bash

# Example script to run the pipeline simulation
# This shows how to activate the correct conda environment and run the simulation

echo "=== Pipeline Simulation Example ==="
echo "Activating WP2 conda environment..."
conda activate WP2

echo "Running pipeline simulation..."
python run_pipeline_simulation.py -c config.json

echo "=== Simulation Complete ==="
echo "Check for output file: predictions_B0531+21_59000_48386.npy"