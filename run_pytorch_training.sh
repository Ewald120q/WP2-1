#!/bin/bash

# PyTorch Training Example Script
# This script demonstrates how to train the PyTorch models

echo "Setting up PyTorch training environment..."

# Change to the training directory
cd /cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training

# Check if data files exist
echo "Checking for training data..."
if [ -f "data/B0531+21_59000_48386_DM_time_dataset_realbased_training.npy" ]; then
    echo "✓ Training data found"
else
    echo "✗ Training data not found. Please ensure the following files exist in data/:"
    echo "  - B0531+21_59000_48386_DM_time_dataset_realbased_training.npy"
    echo "  - B0531+21_59000_48386_DM_time_dataset_realbased_labels_training.npy"
    echo ""
    echo "You can create training data using the DM_time_dataset_creator module."
    exit 1
fi

# Create necessary directories
echo "Creating output directories..."
mkdir -p checkpoints
mkdir -p images
mkdir -p models

echo "Starting PyTorch model training..."
echo "Using config: config.json"
echo ""

# Run training
python training.py config.json

echo ""
echo "Training completed!"
echo ""
echo "To use the trained model:"
echo "1. Copy the best checkpoint from checkpoints/ to ../pipeline_classifier/models/"
echo "2. Update ../pipeline_classifier/config.json with the model filename"
echo "3. Run inference using the pipeline scripts"

echo ""
echo "Example:"
echo "  cp checkpoints/ch_point_*/prot-*-*.pth ../pipeline_classifier/models/"
echo "  # Edit ../pipeline_classifier/config.json"
echo "  cd ../pipeline_classifier"
echo "  python run_inference.py -c config.json"