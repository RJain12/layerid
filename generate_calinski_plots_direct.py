#!/usr/bin/env python3
"""
Script to generate Calinski-Harabasz index plots by calculating scores directly 
from the validation set for each model.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import seaborn as sns
from tqdm import tqdm

# Import necessary modules - adjust these imports to match your codebase structure
from models import VAE, CVAE
from data_loading import get_dataloaders

def extract_latent_features(model, dataloader, device):
    """Extract latent feature representations and corresponding labels from a model"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, mask, labels in dataloader:
            data = data.to(device)
            mask = mask.to(device)
            
            # Get latent representation
            if isinstance(model, VAE) or isinstance(model, CVAE):
                mu, _ = model.encode(data, mask)
                all_features.append(mu.cpu().numpy())
            else:
                # For other model types
                features = model.get_latent(data, mask)
                all_features.append(features.cpu().numpy())
                
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    return all_features, all_labels

def compute_calinski_harabasz_scores(features, labels):
    """Compute Calinski-Harabasz scores for different dimensionality reduction methods"""
    scores = {}
    
    # Try using direct features
    try:
        scores['direct'] = float(calinski_harabasz_score(features, labels))
    except:
        scores['direct'] = None
    
    # PCA
    try:
        pca = PCA(n_components=2)
        pca_embedding = pca.fit_transform(features)
        scores['pca'] = float(calinski_harabasz_score(pca_embedding, labels))
    except:
        scores['pca'] = None
    
    # t-SNE
    try:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_embedding = tsne.fit_transform(features)
        scores['tsne'] = float(calinski_harabasz_score(tsne_embedding, labels))
    except:
        scores['tsne'] = None
    
    # UMAP
    try:
        umap = UMAP(n_components=2, random_state=42)
        umap_embedding = umap.fit_transform(features)
        scores['umap'] = float(calinski_harabasz_score(umap_embedding, labels))
    except:
        scores['umap'] = None
    
    return scores

def generate_plots(model_name, epochs_scores, save_dir):
    """Generate and save plots for Calinski-Harabasz scores"""
    # Make sure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to lists for plotting
    epochs = sorted(list(epochs_scores.keys()))
    pca_ch = [epochs_scores[epoch]['pca'] for epoch in epochs]
    tsne_ch = [epochs_scores[epoch]['tsne'] for epoch in epochs]
    umap_ch = [epochs_scores[epoch]['umap'] for epoch in epochs]
    
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
    
    return scores_data

def get_model(model_path, model_type):
    """Load a model from the specified path and set it to evaluation mode"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model based on type
    if model_type == 'phase_magnitude':
        model = CVAE(input_dim=checkpoint['model_config']['input_dim'],
                    latent_dim=checkpoint['model_config']['latent_dim'],
                    h_dims=checkpoint['model_config']['h_dims'],
                    num_classes=checkpoint['model_config']['num_classes'])
    elif model_type == 'phase_angle':
        model = CVAE(input_dim=checkpoint['model_config']['input_dim'],
                    latent_dim=checkpoint['model_config']['latent_dim'],
                    h_dims=checkpoint['model_config']['h_dims'],
                    num_classes=checkpoint['model_config']['num_classes'])
    elif model_type == 'comodulation':
        model = CVAE(input_dim=checkpoint['model_config']['input_dim'],
                    latent_dim=checkpoint['model_config']['latent_dim'],
                    h_dims=checkpoint['model_config']['h_dims'],
                    num_classes=checkpoint['model_config']['num_classes'])
    
    # Load the state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device

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
        
        # Get all model checkpoints for this type
        model_files = [f for f in os.listdir(model_dir) if f.startswith(model_type) and f.endswith('.pt')]
        
        if not model_files:
            print(f"No model files found for {model_type}")
            continue
        
        # Initialize scores dict
        ch_scores = {}
        
        # Get data loaders
        _, val_loader, _ = get_dataloaders(model_type=model_type, 
                                         batch_size=64, 
                                         train_val_ratio=0.8,
                                         shuffle=False)
        
        # Process each model checkpoint
        for model_file in tqdm(model_files):
            # Extract epoch number from filename
            epoch = int(model_file.split('_epoch_')[1].split('.pt')[0])
            
            model_path = os.path.join(model_dir, model_file)
            
            # Load the model
            model, device = get_model(model_path, model_type)
            
            # Extract features and labels
            features, labels = extract_latent_features(model, val_loader, device)
            
            # Compute Calinski-Harabasz scores
            scores = compute_calinski_harabasz_scores(features, labels)
            
            # Store scores
            ch_scores[epoch] = scores
        
        # Generate plots
        if ch_scores:
            generate_plots(model_type, ch_scores, save_dir)
        else:
            print(f"No scores calculated for {model_type}")

if __name__ == "__main__":
    main() 