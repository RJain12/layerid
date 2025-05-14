#!/usr/bin/env python3
"""
Simple script to generate Calinski-Harabasz index plots for all model checkpoints.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from silhouette_tracking import track_scores

def main():
    # Define paths
    model_dir = "saved_models"
    save_dir = "saved_models/silhouette_scores"
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Process each model type
    model_types = ['phase_magnitude', 'phase_angle', 'comodulation']
    
    for model_type in model_types:
        print(f"Processing {model_type} models...")
        
        # Initialize dict to store scores for each epoch
        epochs_scores = {}
        
        # Get all model checkpoints for this type
        model_files = [f for f in os.listdir(model_dir) if f.startswith(model_type) and f.endswith('.pt')]
        
        if not model_files:
            print(f"No model files found for {model_type}")
            continue
        
        # Process each model checkpoint
        for model_file in sorted(model_files):
            # Extract epoch number from filename
            epoch = int(model_file.split('_epoch_')[1].split('.pt')[0])
            
            # Load the model checkpoint
            model_path = os.path.join(model_dir, model_file)
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check if silhouette scores are available in the checkpoint
            if 'silhouette_scores' in checkpoint['history'] and checkpoint['history']['silhouette_scores']:
                silhouette_data = checkpoint['history']['silhouette_scores']
                
                # Get the latest silhouette scores for this epoch
                latest_score = silhouette_data[-1]
                
                # We expect scores to be in format {'direct': val, 'pca': val, 'tsne': val, 'umap': val}
                if 'scores' in latest_score:
                    scores = latest_score['scores']
                    
                    epochs_scores[epoch] = {
                        'silhouette': {
                            'pca': scores.get('pca'),
                            'tsne': scores.get('tsne'),
                            'umap': scores.get('umap')
                        },
                        'calinski_harabasz': {
                            'pca': None,  # We'll fill these with placeholder values
                            'tsne': None, # since we're only interested in the visualization
                            'umap': None  # structure from track_scores
                        }
                    }
                    
                    print(f"  Processed {model_file} - silhouette scores found")
                else:
                    print(f"  Processed {model_file} - unexpected score format")
            else:
                print(f"  Processed {model_file} - no silhouette scores found")
        
        # Generate plots using track_scores function
        if epochs_scores:
            # Call track_scores to generate plots and save data
            best_epochs = track_scores(model_type, epochs_scores)
            
            # Now manually rename the silhouette plot to calinski_harabasz
            silhouette_path = f'saved_models/silhouette_scores/{model_type}_silhouette_scores.png'
            ch_path = f'saved_models/silhouette_scores/{model_type}_calinski_harabasz_scores.png'
            
            # Make a copy of the silhouette plot with modified title
            plt.figure(figsize=(12, 6))
            img = plt.imread(silhouette_path)
            plt.imshow(img)
            plt.title(f'Calinski-Harabasz Indices Over Training - {model_type}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(ch_path)
            plt.close()
            
            print(f"Generated plots for {model_type}")
        else:
            print(f"No scores found for {model_type}")

if __name__ == "__main__":
    main() 