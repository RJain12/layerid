import numpy as np
import torch
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns

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
            _, z, _ = model(data, mask)
            
            all_features.append(z.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    return features, labels

def compute_scores(latent_representations, labels):
    """
    Compute silhouette scores and Calinski-Harabasz indices for different embeddings.
    
    Args:
        latent_representations: The latent space representations from the model
        labels: The ground truth labels
        
    Returns:
        Dictionary containing scores for different embedding methods
    """
    # Convert to numpy if tensors
    if isinstance(latent_representations, torch.Tensor):
        latent_np = latent_representations.detach().cpu().numpy()
    else:
        latent_np = latent_representations
        
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels
    
    # Initialize scores dictionary
    scores = {
        'silhouette': {
            'pca': None,
            'tsne': None,
            'umap': None
        },
        'calinski_harabasz': {
            'pca': None,
            'tsne': None,
            'umap': None
        }
    }
    
    # Compute embeddings only if we have enough samples
    if len(latent_np) < 10:  # Minimum samples needed for scoring
        return scores
    
    # Compute PCA embedding
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(latent_np)
    
    # Compute t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embedding = tsne.fit_transform(latent_np)
    
    # Compute UMAP embedding
    reducer = UMAP(n_components=2, random_state=42)
    umap_embedding = reducer.fit_transform(latent_np)
    
    # Compute silhouette scores
    try:
        scores['silhouette']['pca'] = float(silhouette_score(pca_embedding, labels_np))
    except:
        scores['silhouette']['pca'] = None
        
    try:
        scores['silhouette']['tsne'] = float(silhouette_score(tsne_embedding, labels_np))
    except:
        scores['silhouette']['tsne'] = None
        
    try:
        scores['silhouette']['umap'] = float(silhouette_score(umap_embedding, labels_np))
    except:
        scores['silhouette']['umap'] = None
    
    # Compute Calinski-Harabasz indices
    try:
        scores['calinski_harabasz']['pca'] = float(calinski_harabasz_score(pca_embedding, labels_np))
    except:
        scores['calinski_harabasz']['pca'] = None
        
    try:
        scores['calinski_harabasz']['tsne'] = float(calinski_harabasz_score(tsne_embedding, labels_np))
    except:
        scores['calinski_harabasz']['tsne'] = None
        
    try:
        scores['calinski_harabasz']['umap'] = float(calinski_harabasz_score(umap_embedding, labels_np))
    except:
        scores['calinski_harabasz']['umap'] = None
    
    return scores

def track_scores(model_name, epochs_scores):
    """
    Track silhouette scores and Calinski-Harabasz indices over epochs and save plots/data.
    
    Args:
        model_name: Name of the model (e.g., 'phase_angle', 'phase_magnitude', 'comodulation')
        epochs_scores: Dictionary mapping epochs to score dictionaries
    """
    # Create directory for scores if it doesn't exist
    os.makedirs('saved_models/silhouette_scores', exist_ok=True)
    
    # Extract epochs and scores
    epochs = sorted(list(epochs_scores.keys()))
    
    # Extract silhouette scores
    pca_silhouette = [epochs_scores[epoch]['silhouette']['pca'] for epoch in epochs]
    tsne_silhouette = [epochs_scores[epoch]['silhouette']['tsne'] for epoch in epochs]
    umap_silhouette = [epochs_scores[epoch]['silhouette']['umap'] for epoch in epochs]
    
    # Extract Calinski-Harabasz indices
    pca_ch = [epochs_scores[epoch]['calinski_harabasz']['pca'] for epoch in epochs]
    tsne_ch = [epochs_scores[epoch]['calinski_harabasz']['tsne'] for epoch in epochs]
    umap_ch = [epochs_scores[epoch]['calinski_harabasz']['umap'] for epoch in epochs]
    
    # Generate silhouette score plot
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, pca_silhouette, label='PCA', marker='o')
    plt.plot(epochs, tsne_silhouette, label='t-SNE', marker='s')
    plt.plot(epochs, umap_silhouette, label='UMAP', marker='^')
    
    # Add best epoch markers
    best_pca_epoch = epochs[np.nanargmax(pca_silhouette)] if any(x is not None for x in pca_silhouette) else None
    best_tsne_epoch = epochs[np.nanargmax(tsne_silhouette)] if any(x is not None for x in tsne_silhouette) else None
    best_umap_epoch = epochs[np.nanargmax(umap_silhouette)] if any(x is not None for x in umap_silhouette) else None
    
    if best_pca_epoch is not None:
        plt.axvline(x=best_pca_epoch, color='blue', linestyle='--', alpha=0.3)
    if best_tsne_epoch is not None:
        plt.axvline(x=best_tsne_epoch, color='orange', linestyle='--', alpha=0.3)
    if best_umap_epoch is not None:
        plt.axvline(x=best_umap_epoch, color='green', linestyle='--', alpha=0.3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Scores Over Training - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save silhouette plot
    silhouette_plot_path = f'saved_models/silhouette_scores/{model_name}_silhouette_scores.png'
    plt.savefig(silhouette_plot_path)
    plt.close()
    
    # Generate Calinski-Harabasz index plot
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, pca_ch, label='PCA', marker='o')
    plt.plot(epochs, tsne_ch, label='t-SNE', marker='s')
    plt.plot(epochs, umap_ch, label='UMAP', marker='^')
    
    # Add best epoch markers
    best_pca_epoch_ch = epochs[np.nanargmax(pca_ch)] if any(x is not None for x in pca_ch) else None
    best_tsne_epoch_ch = epochs[np.nanargmax(tsne_ch)] if any(x is not None for x in tsne_ch) else None
    best_umap_epoch_ch = epochs[np.nanargmax(umap_ch)] if any(x is not None for x in umap_ch) else None
    
    if best_pca_epoch_ch is not None:
        plt.axvline(x=best_pca_epoch_ch, color='blue', linestyle='--', alpha=0.3)
    if best_tsne_epoch_ch is not None:
        plt.axvline(x=best_tsne_epoch_ch, color='orange', linestyle='--', alpha=0.3)
    if best_umap_epoch_ch is not None:
        plt.axvline(x=best_umap_epoch_ch, color='green', linestyle='--', alpha=0.3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Calinski-Harabasz Index')
    plt.title(f'Calinski-Harabasz Indices Over Training - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save Calinski-Harabasz plot
    ch_plot_path = f'saved_models/silhouette_scores/{model_name}_calinski_harabasz_scores.png'
    plt.savefig(ch_plot_path)
    plt.close()
    
    # Save scores data as JSON
    scores_data = {
        'epochs': epochs,
        'silhouette': {
            'pca': pca_silhouette,
            'tsne': tsne_silhouette,
            'umap': umap_silhouette,
            'best_epoch': {
                'pca': int(best_pca_epoch) if best_pca_epoch is not None else None,
                'tsne': int(best_tsne_epoch) if best_tsne_epoch is not None else None,
                'umap': int(best_umap_epoch) if best_umap_epoch is not None else None
            },
            'best_score': {
                'pca': float(max(filter(lambda x: x is not None, pca_silhouette), default=None)) if any(x is not None for x in pca_silhouette) else None,
                'tsne': float(max(filter(lambda x: x is not None, tsne_silhouette), default=None)) if any(x is not None for x in tsne_silhouette) else None,
                'umap': float(max(filter(lambda x: x is not None, umap_silhouette), default=None)) if any(x is not None for x in umap_silhouette) else None
            }
        },
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
    
    # Convert numpy values to Python native types for JSON serialization
    scores_json = json.dumps(scores_data, indent=2)
    
    # Save scores as JSON
    scores_path = f'saved_models/silhouette_scores/{model_name}_scores.json'
    with open(scores_path, 'w') as f:
        f.write(scores_json)
    
    # Print best epochs
    print(f"\nBest epochs for {model_name} based on silhouette scores:")
    print(f"  PCA: Epoch {best_pca_epoch} (score: {max(filter(lambda x: x is not None, pca_silhouette), default=None):.4f})" if best_pca_epoch is not None else "  PCA: No valid scores")
    print(f"  t-SNE: Epoch {best_tsne_epoch} (score: {max(filter(lambda x: x is not None, tsne_silhouette), default=None):.4f})" if best_tsne_epoch is not None else "  t-SNE: No valid scores")
    print(f"  UMAP: Epoch {best_umap_epoch} (score: {max(filter(lambda x: x is not None, umap_silhouette), default=None):.4f})" if best_umap_epoch is not None else "  UMAP: No valid scores")
    
    print(f"\nBest epochs for {model_name} based on Calinski-Harabasz indices:")
    print(f"  PCA: Epoch {best_pca_epoch_ch} (score: {max(filter(lambda x: x is not None, pca_ch), default=None):.1f})" if best_pca_epoch_ch is not None else "  PCA: No valid scores")
    print(f"  t-SNE: Epoch {best_tsne_epoch_ch} (score: {max(filter(lambda x: x is not None, tsne_ch), default=None):.1f})" if best_tsne_epoch_ch is not None else "  t-SNE: No valid scores")
    print(f"  UMAP: Epoch {best_umap_epoch_ch} (score: {max(filter(lambda x: x is not None, umap_ch), default=None):.1f})" if best_umap_epoch_ch is not None else "  UMAP: No valid scores")
    
    return {
        'silhouette': {
            'pca': {'epoch': best_pca_epoch, 'score': max(filter(lambda x: x is not None, pca_silhouette), default=None)},
            'tsne': {'epoch': best_tsne_epoch, 'score': max(filter(lambda x: x is not None, tsne_silhouette), default=None)},
            'umap': {'epoch': best_umap_epoch, 'score': max(filter(lambda x: x is not None, umap_silhouette), default=None)}
        },
        'calinski_harabasz': {
            'pca': {'epoch': best_pca_epoch_ch, 'score': max(filter(lambda x: x is not None, pca_ch), default=None)},
            'tsne': {'epoch': best_tsne_epoch_ch, 'score': max(filter(lambda x: x is not None, tsne_ch), default=None)},
            'umap': {'epoch': best_umap_epoch_ch, 'score': max(filter(lambda x: x is not None, umap_ch), default=None)}
        }
    }

def plot_silhouette_scores(results_path, save_path=None):
    """Plot silhouette scores over epochs"""
    # Load results
    with open(results_path, 'r') as f:
        all_scores = json.load(f)
    
    epochs = all_scores['epochs']
    
    # Extract scores for each method
    direct_scores = [score['direct'] for score in all_scores['scores']]
    pca_scores = [score['pca'] for score in all_scores['scores']]
    tsne_scores = [score['tsne'] for score in all_scores['scores']]
    umap_scores = [score['umap'] for score in all_scores['scores']]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, direct_scores, label='Direct', marker='o')
    plt.plot(epochs, pca_scores, label='PCA', marker='s')
    plt.plot(epochs, tsne_scores, label='t-SNE', marker='^')
    plt.plot(epochs, umap_scores, label='UMAP', marker='d')
    
    plt.xlabel('Epoch')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score by Dimensionality Reduction Method')
    plt.legend()
    plt.grid(True)
    
    # Find best epochs
    methods = ['Direct', 'PCA', 't-SNE', 'UMAP']
    scores_list = [direct_scores, pca_scores, tsne_scores, umap_scores]
    
    for method, scores in zip(methods, scores_list):
        best_epoch = epochs[np.argmax(scores)]
        best_score = max(scores)
        plt.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
        plt.text(best_epoch, best_score, f"Best {method}: {best_epoch}", 
                 bbox=dict(facecolor='white', alpha=0.8))
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()
    
    # Return the best epochs for each method
    best_epochs = {
        'direct': epochs[np.argmax(direct_scores)],
        'pca': epochs[np.argmax(pca_scores)],
        'tsne': epochs[np.argmax(tsne_scores)],
        'umap': epochs[np.argmax(umap_scores)]
    }
    
    return best_epochs

# Legacy function for backward compatibility with train_with_silhouette.py
def track_silhouette_scores(model, dataloader, device, epoch, results_path):
    """
    Legacy wrapper for silhouette score tracking.
    
    Args:
        model: The model for extracting latent features
        dataloader: DataLoader for validation data
        device: Device to run the model on
        epoch: Current epoch number
        results_path: Path to save results to
        
    Returns:
        Dictionary with silhouette scores for different methods
    """
    # Extract features and labels
    features, labels = extract_latent_features(model, dataloader, device)
    
    # Compute scores
    scores = compute_scores(features, labels)
    
    # Get silhouette scores
    silhouette_scores = scores['silhouette']
    
    # Create directory for the model
    model_name = results_path.split('/')[-1].replace('_silhouette.json', '')
    os.makedirs('saved_models/silhouette_scores', exist_ok=True)
    
    # Check if we already have scores from previous epochs
    epochs_scores = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
                for i, ep in enumerate(data.get('epochs', [])):
                    epochs_scores[ep] = {
                        'silhouette': {
                            'pca': data['scores'][i].get('pca'),
                            'tsne': data['scores'][i].get('tsne'),
                            'umap': data['scores'][i].get('umap')
                        },
                        'calinski_harabasz': {
                            'pca': None,
                            'tsne': None,
                            'umap': None
                        }
                    }
        except:
            pass
    
    # Add current epoch scores
    epochs_scores[epoch] = {
        'silhouette': silhouette_scores,
        'calinski_harabasz': scores['calinski_harabasz']
    }
    
    # Generate and save plots
    best_epochs = track_scores(model_name, epochs_scores)
    
    # Return the silhouette scores for backward compatibility
    return silhouette_scores 