#!/usr/bin/env python3
"""
DM-Time Dataset Visualization Tool

This script reads .npy files containing DM-time data and labels,
and provides interactive visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
import os
import argparse
from typing import Tuple, Optional

class DMTimeVisualizer:
    def __init__(self, data_path: str, labels_path: str):
        """
        Initialize the visualizer with data and label files.
        
        Args:
            data_path: Path to the .npy file containing the data
            labels_path: Path to the .npy file containing the labels
        """
        self.data_path = data_path
        self.labels_path = labels_path
        self.data = None
        self.labels = None
        self.current_index = 0
        self.fig = None
        self.ax = None
        self.im = None
        
        self.load_data()
        
    def load_data(self):
        """Load the data and labels from .npy files."""
        print(f"Loading data from {self.data_path}...")
        print(f"Loading labels from {self.labels_path}...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
            
        self.data = np.load(self.data_path)
        self.labels = np.load(self.labels_path)
        
        print(f"Data shape: {self.data.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Unique labels: {np.unique(self.labels)}")
        
        # Get label distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count} ({count/len(self.labels)*100:.1f}%)")
    
    def get_sample(self, index: int) -> Tuple[np.ndarray, str]:
        """
        Get a sample and its label at the given index.
        
        Args:
            index: Index of the sample
            
        Returns:
            Tuple of (sample_data, label)
        """
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} out of range [0, {len(self.data)-1}]")
            
        sample = self.data[index]
        label = self.labels[index]
        
        # If data has shape (1, height, width), squeeze the first dimension
        if sample.shape[0] == 1:
            sample = sample.squeeze(0)
            
        return sample, label
    
    def plot_sample(self, index: int, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot a single sample.
        
        Args:
            index: Index of the sample to plot
            ax: Optional axes to plot on
            
        Returns:
            Figure object
        """
        sample, label = self.get_sample(index)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        else:
            fig = ax.figure
            
        # Plot the DM-time data
        im = ax.imshow(sample, aspect='auto', origin='lower', cmap='viridis')
        
        # Determine color based on label
        color = 'red' if label == 'Pulse' else 'blue'
        
        ax.set_title(f"Sample {index}: {label}", fontsize=16, color=color, fontweight='bold')
        ax.set_xlabel("Time (bins)", fontsize=12)
        ax.set_ylabel("DM (bins)", fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Intensity", fontsize=12)
        
        # Add text with sample info
        info_text = f"Shape: {sample.shape}\nMin: {sample.min():.2f}\nMax: {sample.max():.2f}\nMean: {sample.mean():.2f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def interactive_plot(self):
        """Create an interactive plot with navigation buttons."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        plt.subplots_adjust(bottom=0.15)  # Make room for buttons
        
        # Initial plot
        self.update_plot()
        
        # Create navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.21, 0.02, 0.1, 0.05])
        ax_prev10 = plt.axes([0.32, 0.02, 0.1, 0.05])
        ax_next10 = plt.axes([0.43, 0.02, 0.1, 0.05])
        ax_pulse = plt.axes([0.54, 0.02, 0.15, 0.05])
        ax_artefact = plt.axes([0.7, 0.02, 0.15, 0.05])
        
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        btn_prev10 = Button(ax_prev10, 'Prev 10')
        btn_next10 = Button(ax_next10, 'Next 10')
        btn_pulse = Button(ax_pulse, 'Next Pulse')
        btn_artefact = Button(ax_artefact, 'Next Artefact')
        
        # Button callbacks
        btn_prev.on_clicked(lambda x: self.navigate(-1))
        btn_next.on_clicked(lambda x: self.navigate(1))
        btn_prev10.on_clicked(lambda x: self.navigate(-10))
        btn_next10.on_clicked(lambda x: self.navigate(10))
        btn_pulse.on_clicked(lambda x: self.find_next_label('Pulse'))
        btn_artefact.on_clicked(lambda x: self.find_next_label('Artefact'))
        
        # Add keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.show()
    
    def update_plot(self):
        """Update the current plot."""
        self.ax.clear()
        sample, label = self.get_sample(self.current_index)
        
        # Plot the DM-time data
        self.im = self.ax.imshow(sample, aspect='auto', origin='lower', cmap='viridis')
        
        # Determine color based on label
        color = 'red' if label == 'Pulse' else 'blue'
        
        self.ax.set_title(f"Sample {self.current_index}/{len(self.data)-1}: {label}", 
                         fontsize=16, color=color, fontweight='bold')
        self.ax.set_xlabel("Time (bins)", fontsize=12)
        self.ax.set_ylabel("DM (bins)", fontsize=12)
        
        # Add text with sample info
        info_text = (f"Shape: {sample.shape}\n"
                    f"Min: {sample.min():.2f}\n"
                    f"Max: {sample.max():.2f}\n"
                    f"Mean: {sample.mean():.2f}")
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add legend for label colors
        pulse_patch = mpatches.Patch(color='red', label='Pulse')
        artefact_patch = mpatches.Patch(color='blue', label='Artefact')
        self.ax.legend(handles=[pulse_patch, artefact_patch], loc='upper right')
        
        self.fig.canvas.draw()
    
    def navigate(self, step: int):
        """Navigate by the given step size."""
        new_index = self.current_index + step
        self.current_index = max(0, min(new_index, len(self.data) - 1))
        self.update_plot()
        print(f"Showing sample {self.current_index}: {self.labels[self.current_index]}")
    
    def find_next_label(self, target_label: str):
        """Find the next sample with the specified label."""
        start_index = (self.current_index + 1) % len(self.data)
        
        for i in range(len(self.data)):
            index = (start_index + i) % len(self.data)
            if self.labels[index] == target_label:
                self.current_index = index
                self.update_plot()
                print(f"Found {target_label} at index {self.current_index}")
                return
        
        print(f"No {target_label} samples found")
    
    def on_key_press(self, event):
        """Handle keyboard navigation."""
        if event.key == 'right' or event.key == 'd':
            self.navigate(1)
        elif event.key == 'left' or event.key == 'a':
            self.navigate(-1)
        elif event.key == 'up':
            self.navigate(10)
        elif event.key == 'down':
            self.navigate(-10)
        elif event.key == 'p':
            self.find_next_label('Pulse')
        elif event.key == 'r':
            self.find_next_label('Artefact')
        elif event.key == 'q':
            plt.close()
    
    def plot_random_samples(self, n_samples: int = 6):
        """Plot a grid of random samples."""
        indices = np.random.choice(len(self.data), size=n_samples, replace=False)
        
        # Determine grid layout
        cols = 3
        rows = (n_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            row = i // cols
            col = i % cols
            
            if i < n_samples:
                sample, label = self.get_sample(idx)
                
                im = axes[row, col].imshow(sample, aspect='auto', origin='lower', cmap='viridis')
                
                # Color title based on label
                color = 'red' if label == 'Pulse' else 'blue'
                axes[row, col].set_title(f"Sample {idx}: {label}", color=color, fontweight='bold')
                axes[row, col].set_xlabel("Time (bins)")
                axes[row, col].set_ylabel("DM (bins)")
                
                # Add colorbar
                plt.colorbar(im, ax=axes[row, col])
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def plot_statistics(self):
        """Plot statistics about the dataset."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Label distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        ax1.pie(counts, labels=unique_labels, autopct='%1.1f%%', colors=['blue', 'red'])
        ax1.set_title("Label Distribution")
        
        # Sample intensity histograms by label
        pulse_indices = np.where(self.labels == 'Pulse')[0]
        artefact_indices = np.where(self.labels == 'Artefact')[0]
        
        if len(pulse_indices) > 0:
            pulse_intensities = self.data[pulse_indices].flatten()
            ax2.hist(pulse_intensities, bins=50, alpha=0.7, label='Pulse', color='red', density=True)
        
        if len(artefact_indices) > 0:
            artefact_intensities = self.data[artefact_indices].flatten()
            ax2.hist(artefact_intensities, bins=50, alpha=0.7, label='Artefact', color='blue', density=True)
        
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Density")
        ax2.set_title("Intensity Distribution by Label")
        ax2.legend()
        
        # Mean intensity per sample
        mean_intensities = np.mean(self.data, axis=(1, 2, 3))
        pulse_means = mean_intensities[self.labels == 'Pulse']
        artefact_means = mean_intensities[self.labels == 'Artefact']
        
        ax3.boxplot([pulse_means, artefact_means], labels=['Pulse', 'Artefact'])
        ax3.set_ylabel("Mean Intensity")
        ax3.set_title("Mean Intensity per Sample by Label")
        
        # Sample indices
        sample_indices = np.arange(len(self.data))
        pulse_mask = self.labels == 'Pulse'
        artefact_mask = self.labels == 'Artefact'
        
        ax4.scatter(sample_indices[pulse_mask], mean_intensities[pulse_mask], 
                   c='red', alpha=0.6, label='Pulse', s=10)
        ax4.scatter(sample_indices[artefact_mask], mean_intensities[artefact_mask], 
                   c='blue', alpha=0.6, label='Artefact', s=10)
        ax4.set_xlabel("Sample Index")
        ax4.set_ylabel("Mean Intensity")
        ax4.set_title("Mean Intensity vs Sample Index")
        ax4.legend()
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Visualize DM-time dataset")
    parser.add_argument("--data", type=str, 
                       default="/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator/outputs/B0531+21_59000_48386_DM_time_dataset_realbased.npy",
                       help="Path to the data .npy file")
    parser.add_argument("--labels", type=str,
                       default="/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator/outputs/B0531+21_59000_48386_DM_time_dataset_realbased_labels.npy",
                       help="Path to the labels .npy file")
    parser.add_argument("--mode", type=str, choices=["interactive", "random", "stats", "single"],
                       default="interactive", help="Visualization mode")
    parser.add_argument("--index", type=int, default=0, help="Index for single sample mode")
    parser.add_argument("--n-samples", type=int, default=6, help="Number of samples for random mode")
    
    args = parser.parse_args()
    
    try:
        visualizer = DMTimeVisualizer(args.data, args.labels)
        
        if args.mode == "interactive":
            print("\nInteractive mode:")
            print("- Use navigation buttons or keyboard:")
            print("  - Left/Right arrows or A/D: Navigate one sample")
            print("  - Up/Down arrows: Navigate 10 samples")
            print("  - P: Find next Pulse")
            print("  - R: Find next Artefact")
            print("  - Q: Quit")
            visualizer.interactive_plot()
            
        elif args.mode == "random":
            print(f"\nShowing {args.n_samples} random samples...")
            visualizer.plot_random_samples(args.n_samples)
            
        elif args.mode == "stats":
            print("\nShowing dataset statistics...")
            visualizer.plot_statistics()
            
        elif args.mode == "single":
            print(f"\nShowing single sample at index {args.index}...")
            fig = visualizer.plot_sample(args.index)
            plt.show()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())