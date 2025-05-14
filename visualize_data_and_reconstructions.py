"""
Visualize original comodulation/coherence data and autoencoder reconstructions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from torch.utils.data import DataLoader, Subset
import argparse
from tqdm import tqdm
import warnings

# Silence warnings
warnings.filterwarnings("ignore")

# Import from project modules
from data_loader import load_mat_data
from tensor_utils import build_all_patient_tensors
from utils import get_channel_layer
from datasets import (
    ChannelLevelPhaseMagnitudeDataset,
    ChannelLevelPhaseAngleDataset,
    ChannelLevelComodDataset
)
from models import (
    PhaseMagnitudeLayerClassifier,
    PhaseAngleLayerClassifier,
    ComodLayerClassifier
)

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=1.2)

def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    import json
    with open(config_path, 'r') as f:
        return json.load(f)

def visualize_data_and_reconstruction(model_path, model_type, config, num_examples=3):
    """
    Visualize original data and its reconstruction from the autoencoder.
    
    Args:
        model_path (str): Path to the saved model
        model_type (str): Type of model ('phase_magnitude', 'phase_angle', or 'comodulation')
        config (dict): Configuration dictionary
        num_examples (int): Number of examples to visualize
    """
    # Load data
    print("Loading data...")
    parsed_data_dict = load_mat_data(config['data_path'])
    print(f"Loaded data for {len(parsed_data_dict)} patients")
    
    # Build tensors
    print("Building tensors...")
    all_tensors, paddedC = build_all_patient_tensors(parsed_data_dict, pad_value=config['pad_value'])
    
    # Create appropriate dataset based on model_type
    if model_type == 'phase_magnitude':
        dataset = ChannelLevelPhaseMagnitudeDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
        model = PhaseMagnitudeLayerClassifier(num_classes=config['num_classes'], latent_dim=64, maxC=paddedC)
    elif model_type == 'phase_angle':
        dataset = ChannelLevelPhaseAngleDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
        model = PhaseAngleLayerClassifier(num_classes=config['num_classes'], latent_dim=64, maxC=paddedC)
    elif model_type == 'comodulation':
        dataset = ChannelLevelComodDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
        model = ComodLayerClassifier(num_classes=config['num_classes'], latent_dim=64, maxC=paddedC)
    else:
        raise ValueError("model_type must be one of: 'phase_magnitude', 'phase_angle', 'comodulation'")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Function to reshape 1D data into a square matrix
    def create_matrix_from_data(data, max_dim=None):
        """Convert 1D data to a 2D square matrix for visualization"""
        if len(data.shape) > 1:  # Already 2D
            return data
            
        # Determine maximum size for the matrix
        if max_dim is None:
            # Find a square size that fits most of the data
            size = int(np.sqrt(len(data)))
        else:
            size = min(int(np.sqrt(len(data))), max_dim)
            
        # Reshape to square matrix
        matrix_data = data[:size*size].reshape(size, size)
        return matrix_data
    
    # Visualize examples
    print(f"Visualizing {num_examples} examples...")
    
    example_count = 0
    for inputs, masks, labels in loader:
        if example_count >= num_examples:
            break
            
        # Get original data and reconstructions
        with torch.no_grad():
            recon, _, _ = model(inputs, masks)
        
        # Convert to numpy arrays
        original = inputs.cpu().numpy()[0]  # shape: (channels, 1, maxC) or (channels, maxC)
        reconstruction = recon.cpu().numpy()[0]  # shape: (channels, 1, maxC) or (channels, maxC)
        mask = masks.cpu().numpy()[0]  # shape: (channels, 1, maxC) or (channels, maxC)
        label = labels.item()
        
        # Print shapes for debugging
        print(f"Original shape: {original.shape}")
        print(f"Reconstruction shape: {reconstruction.shape}")
        
        # Calculate residual (error between original and reconstruction)
        # Handle different data shapes correctly
        if len(original.shape) == 3:  # shape (channels, 1, maxC)
            residual = original - reconstruction
        else:  # shape (channels, maxC)
            residual = original - reconstruction
        
        # Create figure based on model type
        if model_type == 'comodulation':
            # Comodulation typically has many more channels (75)
            # Show a subset of channels
            num_channels_to_show = min(4, original.shape[0])
            
            # Select channels to visualize (spread them out)
            channel_indices = np.linspace(0, original.shape[0]-1, num_channels_to_show, dtype=int)
            
            fig, axs = plt.subplots(num_channels_to_show, 3, figsize=(18, 5*num_channels_to_show))
            plt.suptitle(f"Comodulation Data - Layer: {label}", fontsize=16)
            
            for i, channel_idx in enumerate(channel_indices):
                # Extract data for this channel
                if len(original.shape) == 3:
                    orig_data = original[channel_idx, 0]
                    recon_data = reconstruction[channel_idx, 0]
                    res_data = residual[channel_idx, 0]
                else:
                    orig_data = original[channel_idx]
                    recon_data = reconstruction[channel_idx]
                    res_data = residual[channel_idx]
                
                # Convert to matrices
                matrix_size = min(int(np.sqrt(len(orig_data))), 100)  # Limit to 100x100 matrix
                orig_matrix = create_matrix_from_data(orig_data, matrix_size)
                recon_matrix = create_matrix_from_data(recon_data, matrix_size)
                res_matrix = create_matrix_from_data(res_data, matrix_size)
                
                # Plot original
                im_orig = axs[i, 0].imshow(orig_matrix, cmap='viridis', aspect='auto')
                axs[i, 0].set_title(f"Original (Channel {channel_idx})")
                plt.colorbar(im_orig, ax=axs[i, 0])
                
                # Plot reconstruction
                im_recon = axs[i, 1].imshow(recon_matrix, cmap='viridis', aspect='auto')
                axs[i, 1].set_title(f"Reconstruction (Channel {channel_idx})")
                plt.colorbar(im_recon, ax=axs[i, 1])
                
                # Plot residual (error)
                im_res = axs[i, 2].imshow(res_matrix, cmap='coolwarm', aspect='auto')
                axs[i, 2].set_title(f"Residual (masked)")
                plt.colorbar(im_res, ax=axs[i, 2])
        else:
            # Phase magnitude or phase angle (typically 3 channels)
            fig, axs = plt.subplots(original.shape[0], 3, figsize=(18, 5 * original.shape[0]))
            plt.suptitle(f"{model_type.replace('_', ' ').title()} Data - Layer: {label}", fontsize=16)
            
            # Handle case with only one channel
            if original.shape[0] == 1:
                axs = np.array([axs])
            
            for i in range(original.shape[0]):
                # Extract data for this channel
                if len(original.shape) == 3:
                    orig_data = original[i, 0]
                    recon_data = reconstruction[i, 0]
                    res_data = residual[i, 0]
                else:
                    orig_data = original[i]
                    recon_data = reconstruction[i]
                    res_data = residual[i]
                
                # Convert to matrices
                matrix_size = min(int(np.sqrt(len(orig_data))), 100)  # Limit to 100x100 matrix
                orig_matrix = create_matrix_from_data(orig_data, matrix_size)
                recon_matrix = create_matrix_from_data(recon_data, matrix_size)
                res_matrix = create_matrix_from_data(res_data, matrix_size)
                
                # Plot original
                im_orig = axs[i, 0].imshow(orig_matrix, cmap='viridis', aspect='auto')
                axs[i, 0].set_title(f"Original (Fixed)")
                plt.colorbar(im_orig, ax=axs[i, 0])
                
                # Plot reconstruction
                im_recon = axs[i, 1].imshow(recon_matrix, cmap='viridis', aspect='auto')
                axs[i, 1].set_title(f"Reconstruction")
                plt.colorbar(im_recon, ax=axs[i, 1])
                
                # Plot residual (error)
                im_res = axs[i, 2].imshow(res_matrix, cmap='coolwarm', aspect='auto')
                axs[i, 2].set_title(f"Residual (masked)")
                plt.colorbar(im_res, ax=axs[i, 2])
                
                # Add patient and channel info
                plt.figtext(0.5, 0.99, f"Patient 0 | Channel {i}", ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save the figure
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(f"visualizations/{model_type}_example_{example_count}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        example_count += 1
    
    print(f"✅ Visualizations saved to 'visualizations/' directory")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize data and reconstructions')
    parser.add_argument('--model-path', type=str, 
                        default='/Users/rishabjain/Desktop/Research/LayerID/neural_data_processor/saved_models/25epochs_phase_magnitude_model.pt',
                        help='Path to the saved model')
    parser.add_argument('--model-type', type=str, default='phase_magnitude',
                        choices=['phase_magnitude', 'phase_angle', 'comodulation'],
                        help='Type of model to visualize')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--num-examples', type=int, default=3,
                        help='Number of examples to visualize')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Visualize data and reconstructions
    visualize_data_and_reconstruction(
        args.model_path, 
        args.model_type, 
        config, 
        args.num_examples
    )

if __name__ == "__main__":
    main() 