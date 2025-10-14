#!/usr/bin/env python3
"""
Script to regenerate datasets in PyTorch format using the updated DM_time_dataset_creator.
This script will create datasets with the correct (N, C, H, W) format.
"""

import os
import sys
import json
from pathlib import Path

# Add the dataset creator path
sys.path.append('/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator')

def regenerate_dataset(config_path=None):
    """
    Regenerate the dataset using the updated processor that saves in PyTorch format.
    """
    
    if config_path is None:
        config_path = '/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator/config.json'
    
    print("PyTorch Dataset Regeneration")
    print("=" * 50)
    print(f"Using config: {config_path}")
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Available configs:")
        config_dir = os.path.dirname(config_path)
        for f in os.listdir(config_dir):
            if f.endswith('.json'):
                print(f"   - {os.path.join(config_dir, f)}")
        return False
    
    # Load and display config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Input filterbank: {config.get('filterbank_path', 'N/A')}")
    print(f"Output directory: {os.getcwd()}/outputs")
    print()
    
    try:
        # Import and create processor
        from processor import DMTimeDataSetCreator
        
        print("Creating DMTimeDataSetCreator...")
        processor = DMTimeDataSetCreator(config_path)
        
        print("Starting dataset generation...")
        print("This may take a while depending on the size of your data...")
        print()
        
        # Run the processor
        processor.process()
        
        print("✅ Dataset generation completed!")
        
        # Check output files
        output_dir = os.path.join(os.getcwd(), 'outputs')
        if os.path.exists(output_dir):
            print(f"\nOutput files in {output_dir}:")
            for file in os.listdir(output_dir):
                if file.endswith('.npy'):
                    filepath = os.path.join(output_dir, file)
                    import numpy as np
                    data = np.load(filepath)
                    print(f"   📁 {file}")
                    print(f"      Shape: {data.shape}")
                    print(f"      Size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
        
        # Verify the format
        dataset_files = [f for f in os.listdir(output_dir) if f.endswith('_dataset_realbased.npy')]
        if dataset_files:
            import numpy as np
            dataset_file = os.path.join(output_dir, dataset_files[0])
            data = np.load(dataset_file)
            
            print(f"\n📊 Dataset verification:")
            print(f"   Shape: {data.shape}")
            
            if len(data.shape) == 4 and data.shape[1] == 1:
                print("   ✅ Correct PyTorch format (N, C, H, W)")
                print(f"   Samples: {data.shape[0]}")
                print(f"   Channels: {data.shape[1]}")
                print(f"   Height (DMs): {data.shape[2]}")
                print(f"   Width (Time): {data.shape[3]}")
            else:
                print("   ❌ Unexpected format - check processor updates")
        
        print(f"\n🎯 Next steps:")
        print(f"1. Copy generated files to training data directory:")
        print(f"   cp outputs/*.npy /cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/data/")
        print(f"2. Run training:")
        print(f"   cd /cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training")
        print(f"   python training.py config.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def copy_to_training_dir():
    """Copy generated dataset files to the training directory."""
    
    output_dir = '/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator/outputs'
    training_dir = '/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/data'
    
    if not os.path.exists(output_dir):
        print("❌ No outputs directory found. Generate dataset first.")
        return False
    
    # Create training data directory if it doesn't exist
    os.makedirs(training_dir, exist_ok=True)
    
    # Copy .npy files
    import shutil
    copied_files = []
    
    for file in os.listdir(output_dir):
        if file.endswith('.npy'):
            src = os.path.join(output_dir, file)
            dst = os.path.join(training_dir, file)
            shutil.copy2(src, dst)
            copied_files.append(file)
            print(f"✅ Copied {file}")
    
    if copied_files:
        print(f"\n🎯 {len(copied_files)} files copied to training directory")
        print("Ready for training!")
        return True
    else:
        print("❌ No .npy files found to copy")
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == 'copy':
            copy_to_training_dir()
            return
        else:
            config_path = sys.argv[1]
    else:
        config_path = None
    
    success = regenerate_dataset(config_path)
    
    if success:
        print("\n" + "=" * 50)
        print("Would you like to copy the files to the training directory? (y/n)")
        response = input().lower().strip()
        if response in ['y', 'yes']:
            copy_to_training_dir()


if __name__ == "__main__":
    main()