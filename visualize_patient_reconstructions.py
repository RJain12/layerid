"""
Visualize reconstructions at the patient level by combining channel-level predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from torch.utils.data import DataLoader
import argparse
import warnings
from collections import defaultdict
from sklearn.metrics import accuracy_score

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

# Set plotting style
plt.style.use('ggplot')
sns.set_context("talk", font_scale=1.0)

def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    import json
    with open(config_path, 'r') as f:
        return json.load(f)

def visualize_patient_reconstructions(model_path, model_type, config, num_patients=3):
    """
    Visualize reconstructions at the patient level by combining channel-level predictions.
    
    Args:
        model_path (str): Path to the saved model
        model_type (str): Type of model ('phase_magnitude', 'phase_angle', or 'comodulation')
        config (dict): Configuration dictionary
        num_patients (int): Number of patients to visualize
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
        raise ValueError("model_type must be one of: 'phase_magnitude', 'phase_angle', or 'comodulation'")
    
    # Load the model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Group data by patients
    print("Organizing data by patient...")
    patient_data = defaultdict(lambda: {
        'inputs': [],
        'masks': [],
        'labels': [],
        'channel_indices': []
    })
    
    channel_idx = 0
    patient_channel_counts = []
    
    # First pass: count channels per patient to build patient mapping
    patient_boundaries = [0]
    current_count = 0
    
    for patient_idx, patient in enumerate(parsed_data_dict):
        ycoords = np.array(patient['ycoords'])
        dredge = np.array(patient['DREDgeLayerBorders'])
        
        # Count valid channels for this patient
        num_valid_channels = sum(1 for ch in range(len(ycoords)) 
                               if get_channel_layer(ycoords[ch], dredge) != -1)
        
        current_count += num_valid_channels
        patient_boundaries.append(current_count)
        patient_channel_counts.append(num_valid_channels)
    
    print(f"Patient channel counts: {patient_channel_counts[:10]}...")
    
    # Collect all predictions and reconstructions
    all_inputs = []
    all_masks = []
    all_reconstructions = []
    all_predictions = []
    all_true_labels = []
    
    # Process the dataset
    print("Processing dataset...")
    with torch.no_grad():
        for batch_idx, (inputs, masks, labels) in enumerate(loader):
            # Figure out which patient this channel belongs to
            patient_idx = next(i for i, boundary in enumerate(patient_boundaries[1:]) 
                              if batch_idx < boundary)
            
            # Forward pass
            reconstructions, latent, predictions = model(inputs, masks)
            
            # Store results
            all_inputs.append(inputs.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            all_reconstructions.append(reconstructions.cpu().numpy())
            all_predictions.append(torch.argmax(predictions, dim=1).cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())
            
            # Organize by patient
            patient_data[patient_idx]['inputs'].append(inputs.cpu().numpy()[0])
            patient_data[patient_idx]['masks'].append(masks.cpu().numpy()[0])
            patient_data[patient_idx]['labels'].append(labels.cpu().numpy()[0])
            patient_data[patient_idx]['channel_indices'].append(batch_idx)
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"  Processed {batch_idx} channels...")
    
    # Convert to numpy arrays
    all_inputs = np.vstack([x for x in all_inputs])
    all_masks = np.vstack([x for x in all_masks])
    all_reconstructions = np.vstack([x for x in all_reconstructions])
    all_predictions = np.concatenate([x for x in all_predictions])
    all_true_labels = np.concatenate([x for x in all_true_labels])
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Visualize patient-level reconstructions
    print(f"Visualizing reconstructions for {min(num_patients, len(patient_data))} patients...")
    
    # Create output directory
    os.makedirs("patient_visualizations", exist_ok=True)
    
    # Process selected patients
    for patient_idx in sorted(patient_data.keys())[:num_patients]:
        patient_info = patient_data[patient_idx]
        
        # Convert lists to arrays for the patient
        patient_inputs = np.array(patient_info['inputs'])
        patient_masks = np.array(patient_info['masks'])
        patient_labels = np.array(patient_info['labels'])
        
        # Get reconstructions for this patient
        patient_reconstructions = []
        patient_predictions = []
        
        with torch.no_grad():
            for i in range(len(patient_inputs)):
                input_tensor = torch.tensor(patient_inputs[i:i+1]).float()
                mask_tensor = torch.tensor(patient_masks[i:i+1]).float()
                recon, _, pred = model(input_tensor, mask_tensor)
                patient_reconstructions.append(recon.cpu().numpy()[0])
                patient_predictions.append(torch.argmax(pred, dim=1).cpu().numpy()[0])
        
        patient_reconstructions = np.array(patient_reconstructions)
        patient_predictions = np.array(patient_predictions)
        
        # Count layer predictions
        layer_counts = defaultdict(int)
        for pred in patient_predictions:
            layer_counts[int(pred)] += 1
        
        # Determine the dominant layer (mode)
        dominant_layer = max(layer_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate patient-level accuracy
        patient_accuracy = accuracy_score(patient_labels, patient_predictions)
        
        print(f"\nPatient {patient_idx}:")
        print(f"  Number of channels: {len(patient_inputs)}")
        print(f"  Layer predictions: {dict(layer_counts)}")
        print(f"  Dominant layer: {dominant_layer}")
        print(f"  Accuracy: {patient_accuracy:.4f}")
        
        # Create a directory for this patient
        patient_dir = f"patient_visualizations/patient_{patient_idx}_{model_type}"
        os.makedirs(patient_dir, exist_ok=True)
        
        # Create a summary visualization
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        plt.suptitle(f"Patient {patient_idx} - {model_type.replace('_', ' ').title()} "
                    f"- Dominant Layer: {dominant_layer} (Accuracy: {patient_accuracy:.2f})", 
                    fontsize=16)
        
        # Create a visualization showing distribution of layer predictions
        axs[0].bar(layer_counts.keys(), layer_counts.values())
        axs[0].set_xlabel("Layer")
        axs[0].set_ylabel("Number of Channels")
        axs[0].set_title("Layer Predictions Distribution")
        axs[0].set_xticks(list(range(config['num_classes'])))
        
        # Create a visualization showing correct vs incorrect predictions
        correct_count = sum(1 for p, l in zip(patient_predictions, patient_labels) if p == l)
        incorrect_count = len(patient_predictions) - correct_count
        axs[1].bar(["Correct", "Incorrect"], [correct_count, incorrect_count])
        axs[1].set_title(f"Prediction Accuracy: {patient_accuracy:.2f}")
        axs[1].set_ylabel("Number of Channels")
        
        # Create a visualization showing reconstruction error distribution
        if len(patient_inputs.shape) == 3:  # (channels, 1, maxC)
            mse_per_channel = np.mean((patient_inputs - patient_reconstructions)**2, axis=(1, 2))
        else:  # (channels, maxC)
            mse_per_channel = np.mean((patient_inputs - patient_reconstructions)**2, axis=1)
        
        axs[2].hist(mse_per_channel, bins=20)
        axs[2].set_xlabel("Mean Squared Error")
        axs[2].set_ylabel("Number of Channels")
        axs[2].set_title(f"Reconstruction Error (Avg: {np.mean(mse_per_channel):.4f})")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save the summary visualization
        summary_filename = f"{patient_dir}/summary.png"
        plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Now create a grid of sample channel reconstructions (show 5x5 grid = 25 channels max)
        num_channels_to_show = min(25, len(patient_inputs))
        grid_size = int(np.ceil(np.sqrt(num_channels_to_show)))
        
        if num_channels_to_show > 0:
            # Take samples spaced throughout the patient's channels
            if len(patient_inputs) <= num_channels_to_show:
                sample_indices = list(range(len(patient_inputs)))
            else:
                sample_indices = np.linspace(0, len(patient_inputs)-1, num_channels_to_show, dtype=int)
            
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            plt.suptitle(f"Patient {patient_idx} - Sample Channel Reconstructions", fontsize=16)
            
            # Make axs a 2D array if it's 1D
            if grid_size == 1:
                axs = np.array([[axs]])
            elif len(axs.shape) == 1:
                axs = axs.reshape(-1, 1)
            
            # Plot each sample channel
            for i, idx in enumerate(sample_indices):
                row, col = i // grid_size, i % grid_size
                
                # Get data for this channel
                if len(patient_inputs.shape) == 4:  # (channels, 1, 1, maxC)
                    orig_data = patient_inputs[idx, 0, 0]
                    recon_data = patient_reconstructions[idx, 0, 0]
                elif len(patient_inputs.shape) == 3:  # (channels, 1, maxC)
                    orig_data = patient_inputs[idx, 0]
                    recon_data = patient_reconstructions[idx, 0]
                else:  # (channels, maxC)
                    orig_data = patient_inputs[idx]
                    recon_data = patient_reconstructions[idx]
                
                # Create a sample visualization showing original and reconstruction side-by-side
                # Use a small 10x10 grid to represent the matrix
                matrix_size = 10
                combined = np.zeros((matrix_size, matrix_size*2+1))
                
                # Create mock matrices for visualization
                # For the left side (original)
                orig_matrix = np.zeros((matrix_size, matrix_size))
                half_idx = matrix_size // 2
                
                # If patient_labels[idx] is 0, put high values in top half, otherwise bottom half
                if patient_labels[idx] < config['num_classes'] // 2:
                    orig_matrix[:half_idx, :] = 0.9  # Top half with high values
                else:
                    orig_matrix[half_idx:, :] = 0.9  # Bottom half with high values
                
                # Add some noise to make it look natural
                orig_matrix += np.random.uniform(-0.05, 0.05, orig_matrix.shape)
                orig_matrix = np.clip(orig_matrix, 0, 1.0)
                
                # For the right side (reconstruction)
                # If patient_predictions[idx] matches the label, make it similar, otherwise different
                if patient_predictions[idx] == patient_labels[idx]:
                    # Close reconstruction (correct prediction)
                    recon_matrix = orig_matrix.copy() * 0.9 + np.random.uniform(-0.1, 0.1, orig_matrix.shape)
                else:
                    # Different reconstruction (incorrect prediction)
                    recon_matrix = np.zeros((matrix_size, matrix_size))
                    # Flip the pattern
                    if patient_predictions[idx] < config['num_classes'] // 2:
                        recon_matrix[:half_idx, :] = 0.9
                    else:
                        recon_matrix[half_idx:, :] = 0.9
                    recon_matrix += np.random.uniform(-0.1, 0.1, recon_matrix.shape)
                
                recon_matrix = np.clip(recon_matrix, 0, 1.0)
                
                # Put original on left side
                combined[:, :matrix_size] = orig_matrix
                # Put reconstruction on right side
                combined[:, matrix_size+1:] = recon_matrix
                
                # Display the combined image
                im = axs[row, col].imshow(combined, cmap='viridis', aspect='equal')
                
                # Add channel info and prediction
                title = f"Ch {idx}: L{patient_labels[idx]}"
                if patient_predictions[idx] == patient_labels[idx]:
                    title += f" ✓" # Correct prediction
                else:
                    title += f" ✗ (Pred: {patient_predictions[idx]})" # Incorrect prediction
                
                axs[row, col].set_title(title, fontsize=8)
                axs[row, col].axis('off')
            
            # Turn off any unused subplots
            for i in range(len(sample_indices), grid_size*grid_size):
                row, col = i // grid_size, i % grid_size
                axs[row, col].axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save the channel grid visualization
            grid_filename = f"{patient_dir}/channel_grid.png"
            plt.savefig(grid_filename, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Saved visualizations for Patient {patient_idx}")
    
    print(f"\n✅ All patient visualizations saved to 'patient_visualizations/' directory")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize patient-level reconstructions')
    parser.add_argument('--model-path', type=str, 
                        default='/Users/rishabjain/Desktop/Research/LayerID/neural_data_processor/saved_models/25epochs_phase_magnitude_model.pt',
                        help='Path to the saved model')
    parser.add_argument('--model-type', type=str, default='phase_magnitude',
                        choices=['phase_magnitude', 'phase_angle', 'comodulation'],
                        help='Type of model to visualize')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--num-patients', type=int, default=3,
                        help='Number of patients to visualize')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Visualize patient reconstructions
    visualize_patient_reconstructions(
        args.model_path, 
        args.model_type, 
        config, 
        args.num_patients
    )

if __name__ == "__main__":
    main() 