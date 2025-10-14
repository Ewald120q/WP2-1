#!/usr/bin/env python3
"""
Streamlined TensorFlow to PyTorch Dataset Converter

This script converts existing TensorFlow datasets (N, H, W) to PyTorch format (N, C, H, W)
without loading the entire dataset into memory. It processes data in chunks for memory efficiency.

Usage:
    python convert_tf_to_pytorch.py input_dataset.npy output_dataset.npy [--chunk-size 1000]
"""

import numpy as np
import os
import argparse
from tqdm import tqdm
import psutil


class StreamlinedDatasetConverter:
    """
    Converts TensorFlow format datasets to PyTorch format using memory-efficient streaming.
    """
    
    def __init__(self, input_path, output_path, chunk_size=1000):
        """
        Initialize the converter.
        
        Args:
            input_path (str): Path to input TensorFlow dataset (.npy)
            output_path (str): Path for output PyTorch dataset (.npy)
            chunk_size (int): Number of samples to process at once
        """
        self.input_path = input_path
        self.output_path = output_path
        self.chunk_size = chunk_size
        
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    def analyze_input_dataset(self):
        """
        Analyze the input dataset without loading it fully into memory.
        """
        print("📊 Analyzing input dataset...")
        
        # Use memory mapping to read shape without loading data
        data_mmap = np.load(self.input_path, mmap_mode='r')
        
        input_shape = data_mmap.shape
        input_dtype = data_mmap.dtype
        file_size_mb = os.path.getsize(self.input_path) / (1024 * 1024)
        
        print(f"   Input file: {self.input_path}")
        print(f"   Input shape: {input_shape}")
        print(f"   Input dtype: {input_dtype}")
        print(f"   File size: {file_size_mb:.1f} MB")
        
        # Validate format
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (N, H, W), got {len(input_shape)}D: {input_shape}")
        
        n_samples, height, width = input_shape
        
        # Calculate output shape (add channel dimension)
        output_shape = (n_samples, 1, height, width)  # (N, C, H, W)
        
        print(f"   Output shape: {output_shape}")
        print(f"   Conversion: (N, H, W) → (N, C, H, W)")
        
        # Estimate memory usage
        bytes_per_sample_input = height * width * np.dtype(input_dtype).itemsize
        bytes_per_sample_output = 1 * height * width * np.dtype(input_dtype).itemsize  # Same, just reshaped
        
        chunk_memory_mb = (self.chunk_size * bytes_per_sample_input) / (1024 * 1024)
        
        print(f"   Memory per chunk: {chunk_memory_mb:.1f} MB")
        print(f"   Total chunks needed: {(n_samples + self.chunk_size - 1) // self.chunk_size}")
        
        return {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'dtype': input_dtype,
            'n_samples': n_samples,
            'height': height,
            'width': width,
            'chunk_memory_mb': chunk_memory_mb
        }
    
    def check_memory_availability(self, chunk_memory_mb):
        """
        Check if system has enough memory for processing.
        """
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        required_gb = chunk_memory_mb / 1024
        
        print(f"💾 Memory check:")
        print(f"   Available: {available_gb:.1f} GB")
        print(f"   Required per chunk: {required_gb:.3f} GB")
        
        if required_gb > available_gb * 0.8:  # Use max 80% of available memory
            recommended_chunk_size = int(self.chunk_size * 0.8 * available_gb / required_gb)
            raise MemoryError(
                f"Chunk size too large! Required {required_gb:.2f} GB but only "
                f"{available_gb:.1f} GB available. Try --chunk-size {recommended_chunk_size}"
            )
        
        print(f"   ✅ Memory check passed")
    
    def convert_dataset(self):
        """
        Convert the dataset from TensorFlow to PyTorch format using streaming.
        """
        print("\n🔄 Starting streamlined conversion...")
        
        # Analyze input
        info = self.analyze_input_dataset()
        
        # Check memory
        self.check_memory_availability(info['chunk_memory_mb'])
        
        # Open input dataset with memory mapping (read-only)
        print(f"\n📖 Opening input dataset (memory-mapped)...")
        input_data = np.load(self.input_path, mmap_mode='r')
        
        # Create output dataset with memory mapping (write mode)
        print(f"📝 Creating output dataset...")
        output_data = np.lib.format.open_memmap(
            self.output_path,
            mode='w+',
            dtype=info['dtype'],
            shape=info['output_shape']
        )
        
        n_samples = info['n_samples']
        
        # Process in chunks
        print(f"\n⚡ Processing {n_samples} samples in chunks of {self.chunk_size}...")
        
        for start_idx in tqdm(range(0, n_samples, self.chunk_size), desc="Converting chunks"):
            end_idx = min(start_idx + self.chunk_size, n_samples)
            
            # Read chunk from input (N, H, W)
            chunk_input = input_data[start_idx:end_idx]
            
            # Convert to PyTorch format (N, C, H, W) by adding channel dimension
            chunk_output = np.expand_dims(chunk_input, axis=1)
            
            # Write chunk to output
            output_data[start_idx:end_idx] = chunk_output
            
            # Flush to disk periodically
            if (start_idx // self.chunk_size) % 10 == 0:
                output_data.flush()
        
        # Final flush
        output_data.flush()
        
        print(f"✅ Conversion completed!")
        return info
    
    def verify_conversion(self):
        """
        Verify the converted dataset is correct.
        """
        print(f"\n🔍 Verifying conversion...")
        
        # Load small samples for verification
        input_sample = np.load(self.input_path, mmap_mode='r')[:5]  # First 5 samples
        output_sample = np.load(self.output_path, mmap_mode='r')[:5]  # First 5 samples
        
        print(f"   Input sample shape: {input_sample.shape}")
        print(f"   Output sample shape: {output_sample.shape}")
        
        # Check if conversion is correct
        expected_output = np.expand_dims(input_sample, axis=1)
        
        if np.array_equal(output_sample, expected_output):
            print(f"   ✅ Conversion verified - data matches perfectly!")
            
            # Check file sizes
            input_size = os.path.getsize(self.input_path) / (1024 * 1024)
            output_size = os.path.getsize(self.output_path) / (1024 * 1024)
            
            print(f"   Input file size: {input_size:.1f} MB")
            print(f"   Output file size: {output_size:.1f} MB")
            print(f"   Size difference: {output_size - input_size:.1f} MB")
            
            return True
        else:
            print(f"   ❌ Verification failed - data doesn't match!")
            return False
    
    def get_conversion_summary(self):
        """
        Provide a summary of the conversion.
        """
        if not os.path.exists(self.output_path):
            print("❌ Output file not created")
            return
        
        output_data = np.load(self.output_path, mmap_mode='r')
        
        print(f"\n📋 Conversion Summary:")
        print(f"   ✅ Input format: TensorFlow (N, H, W)")
        print(f"   ✅ Output format: PyTorch (N, C, H, W)")
        print(f"   ✅ Shape: {output_data.shape}")
        print(f"   ✅ Data type: {output_data.dtype}")
        print(f"   ✅ Samples: {output_data.shape[0]:,}")
        print(f"   ✅ Channels: {output_data.shape[1]} (grayscale)")
        print(f"   ✅ Spatial size: {output_data.shape[2]}×{output_data.shape[3]}")
        print(f"   ✅ File: {self.output_path}")
        print(f"\n🎯 Ready for PyTorch training!")


def convert_labels_if_needed(input_dataset_path, labels_path=None):
    """
    Convert labels from string to numeric format if needed.
    """
    if labels_path is None:
        # Try to find labels file automatically
        base_name = input_dataset_path.replace('_dataset_realbased.npy', '')
        potential_labels = base_name + '_dataset_realbased_labels.npy'
        
        if os.path.exists(potential_labels):
            labels_path = potential_labels
        else:
            print("ℹ️  No labels file found - skipping label conversion")
            return None
    
    if not os.path.exists(labels_path):
        print(f"⚠️  Labels file not found: {labels_path}")
        return None
    
    print(f"\n🏷️  Converting labels...")
    print(f"   Input: {labels_path}")
    
    # Load labels
    labels = np.load(labels_path)
    print(f"   Original shape: {labels.shape}")
    print(f"   Original dtype: {labels.dtype}")
    print(f"   Unique values: {np.unique(labels)}")
    
    # Convert string labels to numeric if needed
    if labels.dtype.kind in ['U', 'S']:  # Unicode or byte string
        print("   Converting string labels to numeric...")
        
        label_map = {'Artefact': 0, 'Pulse': 1}
        numeric_labels = np.array([label_map.get(str(label), 0) for label in labels])
        
        # Save converted labels
        output_labels_path = labels_path.replace('.npy', '_pytorch.npy')
        np.save(output_labels_path, numeric_labels)
        
        print(f"   ✅ Converted labels saved: {output_labels_path}")
        print(f"   Conversion: {dict(zip(np.unique(labels), np.unique(numeric_labels)))}")
        
        return output_labels_path
    else:
        print("   ℹ️  Labels already numeric - no conversion needed")
        return labels_path


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(description='Convert TensorFlow datasets to PyTorch format')
    parser.add_argument('input_dataset', help='Path to input TensorFlow dataset (.npy)')
    parser.add_argument('output_dataset', help='Path for output PyTorch dataset (.npy)')
    parser.add_argument('--chunk-size', type=int, default=1000, 
                       help='Number of samples to process at once (default: 1000)')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to labels file (auto-detected if not specified)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify conversion by comparing sample data')
    
    args = parser.parse_args()
    
    print("🔄 TensorFlow to PyTorch Dataset Converter")
    print("=" * 50)
    
    try:
        # Convert dataset
        converter = StreamlinedDatasetConverter(
            args.input_dataset, 
            args.output_dataset, 
            args.chunk_size
        )
        
        info = converter.convert_dataset()
        
        # Verify if requested
        if args.verify:
            if converter.verify_conversion():
                print("✅ Verification passed!")
            else:
                print("❌ Verification failed!")
                return 1
        
        # Convert labels
        labels_output = convert_labels_if_needed(args.input_dataset, args.labels)
        
        # Summary
        converter.get_conversion_summary()
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Update training script to use: {args.output_dataset}")
        if labels_output:
            print(f"2. Update training script to use labels: {labels_output}")
        print(f"3. Remove the np.expand_dims line from training script")
        print(f"4. Run training: python training.py config.json")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())