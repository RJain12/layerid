#!/usr/bin/env python3
"""
Script to generate Calinski-Harabasz index plots from saved model data.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from silhouette_tracking import compute_scores

def extract_calinski_data_from_model(model_path, save_dir):
    """
    Extract Calinski-Harabasz data from a saved model and generate plots.
    
    Args:
        model_path: Path to the saved model file
        save_dir: Directory to save the plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model data
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    history = checkpoint['history']
    
    # Extract model name from the path
    model_name = os.path.basename(model_path).split('_')[0]
    
    # Check if Calinski-Harabasz scores are available
    if 'calinski_harabasz_scores' not in history or not history['calinski_harabasz_scores']:
        print(f"No Calinski-Harabasz scores found in {model_path}")
        return
    
    # Extract Calinski-Harabasz scores
    ch_scores = history['calinski_harabasz_scores']
    
    # Extract epochs and scores
    epochs = [entry['epoch'] for entry in ch_scores]
    
    # Extract scores by method
    pca_ch = []
    tsne_ch = []
    umap_ch = []
    
    for entry in ch_scores:
        scores = entry['scores']
        pca_ch.append(scores.get('pca', None))
        tsne_ch.append(scores.get('tsne', None))
        umap_ch.append(scores.get('umap', None))
    
    # Generate Calinski-Harabasz index plot
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, pca_ch, label='PCA', marker='o')
    plt.plot(epochs, tsne_ch, label='t-SNE', marker='s')
    plt.plot(epochs, umap_ch, label='UMAP', marker='^')
    
    # Add best epoch markers
    try:
        best_pca_epoch_ch = epochs[np.nanargmax(pca_ch)] if any(x is not None for x in pca_ch) else None
        best_tsne_epoch_ch = epochs[np.nanargmax(tsne_ch)] if any(x is not None for x in tsne_ch) else None
        best_umap_epoch_ch = epochs[np.nanargmax(umap_ch)] if any(x is not None for x in umap_ch) else None
        
        if best_pca_epoch_ch is not None:
            plt.axvline(x=best_pca_epoch_ch, color='blue', linestyle='--', alpha=0.3)
        if best_tsne_epoch_ch is not None:
            plt.axvline(x=best_tsne_epoch_ch, color='orange', linestyle='--', alpha=0.3)
        if best_umap_epoch_ch is not None:
            plt.axvline(x=best_umap_epoch_ch, color='green', linestyle='--', alpha=0.3)
    except Exception as e:
        print(f"Error adding best epoch markers: {e}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Calinski-Harabasz Index')
    plt.title(f'Calinski-Harabasz Indices Over Training - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save Calinski-Harabasz plot
    ch_plot_path = os.path.join(save_dir, f"{model_name}_calinski_harabasz_scores.png")
    plt.savefig(ch_plot_path)
    plt.close()
    
    print(f"Saved Calinski-Harabasz plot to {ch_plot_path}")
    
    # Save scores data as JSON
    scores_data = {
        'epochs': epochs,
        'calinski_harabasz': {
            'pca': pca_ch,
            'tsne': tsne_ch,
            'umap': umap_ch,
            'best_epoch': {
                'pca': int(best_pca_epoch_ch) if best_pca_epoch_ch is not None else None,
                'tsne': int(best_tsne_epoch_ch) if best_tsne_epoch_ch is not None else None,
                'umap': int(best_umap_epoch_ch) if best_umap_epoch_ch is not None else None
            },
            'best_score': {
                'pca': float(max(filter(lambda x: x is not None, pca_ch), default=None)) if any(x is not None for x in pca_ch) else None,
                'tsne': float(max(filter(lambda x: x is not None, tsne_ch), default=None)) if any(x is not None for x in tsne_ch) else None,
                'umap': float(max(filter(lambda x: x is not None, umap_ch), default=None)) if any(x is not None for x in umap_ch) else None
            }
        }
    }
    
    # Save scores as JSON
    scores_path = os.path.join(save_dir, f"{model_name}_calinski_harabasz.json")
    with open(scores_path, 'w') as f:
        json.dump(scores_data, f, indent=2)
    
    print(f"Saved Calinski-Harabasz scores to {scores_path}")

def main():
    # Define paths
    model_dir = "saved_models_silhouette"
    save_dir = "saved_models/silhouette_scores"
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Process each model type
    model_types = ['phase_magnitude', 'phase_angle', 'comodulation']
    
    for model_type in model_types:
        model_path = os.path.join(model_dir, f"{model_type}_epoch_25.pt")
        
        if os.path.exists(model_path):
            print(f"Processing {model_type} model...")
            extract_calinski_data_from_model(model_path, save_dir)
        else:
            print(f"Model file not found: {model_path}")

if __name__ == "__main__":
    main() 