"""
Visualization utilities for neural data processing and model training.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from matplotlib.colors import ListedColormap
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

# Imports for running as a script
from data_loader import load_mat_data
from tensor_utils import build_all_patient_tensors
from utils import get_channel_layer

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=1.2)

# Define a consistent color palette 
COLORS = sns.color_palette("Set2", 8)  # Use Set2 palette as in the notebook
LAYER_CMAP = ListedColormap(COLORS)

def plot_training_history(history, model_name, save_path=None):
    """
    Plot training and validation loss/accuracy over epochs.
    
    Args:
        history (dict): Training history dictionary
        model_name (str): Name of the model for the plot title
        save_path (str, optional): Path to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots: total loss, recon loss, cls loss, and accuracy
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot total loss
    axs[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axs[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axs[0, 0].set_title(f'{model_name} - Total Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot reconstruction loss if available
    if 'train_recon_loss' in history and 'val_recon_loss' in history:
        axs[0, 1].plot(epochs, history['train_recon_loss'], 'b-', label='Training Recon Loss')
        axs[0, 1].plot(epochs, history['val_recon_loss'], 'r-', label='Validation Recon Loss')
        axs[0, 1].set_title(f'{model_name} - Reconstruction Loss')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
    
    # Plot classification loss if available
    if 'train_cls_loss' in history and 'val_cls_loss' in history:
        axs[1, 0].plot(epochs, history['train_cls_loss'], 'b-', label='Training Class Loss')
        axs[1, 0].plot(epochs, history['val_cls_loss'], 'r-', label='Validation Class Loss')
        axs[1, 0].set_title(f'{model_name} - Classification Loss')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
    
    # Plot accuracy
    axs[1, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    axs[1, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    axs[1, 1].set_title(f'{model_name} - Accuracy')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved training history plot to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        model_name (str): Name of the model for the plot title
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix plot to {save_path}")
    else:
        plt.show()


def plot_model_comparison(results, save_path=None):
    """
    Plot bar chart comparing accuracy across different models.
    
    Args:
        results (dict): Dictionary with model names and accuracies
        save_path (str, optional): Path to save the plot
    """
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)  # Set y-axis from 0 to 1
    plt.grid(axis='y')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved model comparison plot to {save_path}")
    else:
        plt.show()


def extract_channel_features_modality(all_tensors, parsed_data_dict, modality="phase_magnitude", pad_value=99.0):
    """
    Extract channel-level row-based features from a single modality:
      - "phase_magnitude" -> shape (3, maxC, maxC)
      - "phase_angle"     -> shape (3, maxC, maxC)
      - "comod"           -> shape (75, maxC, maxC)

    Returns:
      features: list of 1D numpy arrays (one per channel)
      labels:   list of layer indices (0..7)
    """
    features = []
    labels = []

    for idx, (pmag_tensor, pang_tensor, comod_tensor) in enumerate(all_tensors):
        patient_data = parsed_data_dict[idx]
        ycoords = np.array(patient_data['ycoords'])
        borders = np.array(patient_data['DREDgeLayerBorders'])

        # pick the right tensor
        if modality == "phase_magnitude":
            cur_tensor = pmag_tensor
        elif modality == "phase_angle":
            cur_tensor = pang_tensor
        elif modality == "comod":
            cur_tensor = comod_tensor
        else:
            raise ValueError("modality must be one of: 'phase_magnitude', 'phase_angle', 'comod'")

        # ensure float
        cur_tensor = cur_tensor.astype(np.float32)

        for ch in range(len(ycoords)):
            layer = get_channel_layer(ycoords[ch], borders)
            if layer == -1:
                continue  # skip extraneous

            # row slice for this channel
            row = cur_tensor[:, ch, :]  # shape: (num_modalities, maxC)
            row[row == pad_value] = np.nan  # mask out padding

            flat = row.flatten()
            if np.isnan(flat).all():
                continue

            # local z-score
            mean = np.nanmean(flat)
            std = np.nanstd(flat) + 1e-6
            flat = (flat - mean) / std
            flat = np.nan_to_num(flat, nan=0.0)

            features.append(flat)
            labels.append(layer)

    return np.array(features), np.array(labels)


def visualize_pca(features, labels, title="PCA", save_path=None):
    """
    Run PCA on the features and visualize the results.
    
    Args:
        features (np.ndarray): Feature matrix
        labels (np.ndarray): Array of layer labels
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    X = np.array(features)
    y = np.array(labels)
    
    # Standardize features
    X = StandardScaler().fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y, palette='Set2', s=10)
    plt.title(f"{title}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Layer", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved PCA visualization to {save_path}")
    else:
        plt.show()


def visualize_umap(features, labels, title="UMAP", save_path=None):
    """
    Run UMAP on the features and visualize the results.
    
    Args:
        features (np.ndarray): Feature matrix
        labels (np.ndarray): Array of layer labels
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    X = np.array(features)
    y = np.array(labels)
    
    # Standardize features
    X = StandardScaler().fit_transform(X)
    
    # UMAP with settings from the notebook
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
    Z = reducer.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y, palette='Set2', s=10)
    plt.title(f"{title}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Layer", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved UMAP visualization to {save_path}")
    else:
        plt.show()


def visualize_tsne(features, labels, title="t-SNE", save_path=None):
    """
    Run t-SNE on the features and visualize the results.
    
    Args:
        features (np.ndarray): Feature matrix
        labels (np.ndarray): Array of layer labels
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    X = np.array(features)
    y = np.array(labels)
    
    # Standardize features
    X = StandardScaler().fit_transform(X)
    
    # t-SNE with settings from the notebook
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    Z = tsne.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y, palette='Set2', s=10)
    plt.title(f"{title}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Layer", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved t-SNE visualization to {save_path}")
    else:
        plt.show()


def visualize_latent_features(model, dataset, labels=None, title_prefix='', save_dir=None, include_silhouette=True):
    """
    Visualize latent features from a model with PCA, t-SNE, and UMAP.
    
    Args:
        model (nn.Module): Trained PyTorch model
        dataset (Dataset): Dataset with data to visualize
        labels (np.ndarray, optional): Pre-computed labels for the data
        title_prefix (str): Prefix for plot titles
        save_dir (str, optional): Directory to save visualizations
        include_silhouette (bool): Whether to include silhouette scores in plots
    """
    device = next(model.parameters()).device
    
    # Extract features and labels if not provided
    if labels is None:
        # Create a DataLoader to efficiently process the data
        dl = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Extract feature representations
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for data, mask, label in tqdm(dl, desc="Extracting features"):
                data = data.to(device)
                mask = mask.to(device)
                
                # Get latent representation
                _, z, _ = model(data, mask)
                
                features_list.append(z.cpu().numpy())
                labels_list.append(label.numpy())
        
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
    else:
        # Extract features only (labels are provided)
        dl = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Extract feature representations
        features_list = []
        
        with torch.no_grad():
            for data, mask, _ in tqdm(dl, desc="Extracting features"):
                data = data.to(device)
                mask = mask.to(device)
                
                # Get latent representation
                _, z, _ = model(data, mask)
                
                features_list.append(z.cpu().numpy())
        
        features = np.vstack(features_list)
    
    # Create figure directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Helper function for plotting and saving
    def plot_features(data, title, filename=None):
        plt.figure(figsize=(10, 8))
        
        # Get unique labels and create colormap
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))
        
        # Plot each class
        for i, label in enumerate(unique_labels):
            idx = labels == label
            plt.scatter(data[idx, 0], data[idx, 1], color=cmap(i), label=f'Layer {label}', alpha=0.7)
        
        # Calculate silhouette score if requested
        sil_score = None
        if include_silhouette:
            try:
                from sklearn.metrics import silhouette_score
                sil_score = silhouette_score(data, labels)
                title = f"{title} (Silhouette: {sil_score:.4f})"
            except:
                pass
        
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if filename and save_dir:
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()
        else:
            plt.show()
        
        return sil_score
    
    # Apply PCA
    pca_scores = {}
    try:
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(features)
        pca_var = pca.explained_variance_ratio_
        pca_title = f"{title_prefix} - PCA (Var: {pca_var[0]:.2f}, {pca_var[1]:.2f})"
        pca_scores['pca'] = plot_features(pca_data, pca_title, f"{title_prefix.lower().replace(' ', '_')}_pca.png")
    except Exception as e:
        print(f"Error plotting PCA: {e}")
    
    # Apply t-SNE
    try:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(features)
        tsne_title = f"{title_prefix} - t-SNE"
        pca_scores['tsne'] = plot_features(tsne_data, tsne_title, f"{title_prefix.lower().replace(' ', '_')}_tsne.png")
    except Exception as e:
        print(f"Error plotting t-SNE: {e}")
    
    # Apply UMAP
    try:
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        umap_data = umap_reducer.fit_transform(features)
        umap_title = f"{title_prefix} - UMAP"
        pca_scores['umap'] = plot_features(umap_data, umap_title, f"{title_prefix.lower().replace(' ', '_')}_umap.png")
    except Exception as e:
        print(f"Error plotting UMAP: {e}")
        
    return pca_scores


def visualize_training_run(prefix, config, filenames, show_best_silhouette=True):
    """
    Create visualizations from training runs.
    
    Args:
        prefix (str): Prefix for the saved files
        config (dict): Configuration dictionary
        filenames (dict): Dictionary of base filenames for each model type
        show_best_silhouette (bool): Whether to show best silhouette epochs in the plots
    """
    # Dictionary to store histories
    histories = {
        'phase_magnitude': None,
        'phase_angle': None,
        'comodulation': None
    }
    
    # Collect all histories
    for model_type, base_filename in filenames.items():
        history_path = os.path.join(config['models_dir'], f"{base_filename}_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                histories[model_type] = json.load(f)
    
    # Visualization directory
    vis_dir = os.path.join(config['models_dir'], f"{prefix}_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot losses and accuracies
    plt.figure(figsize=(15, 12))
    
    # Add column for silhouette scores if available
    include_silhouette = any(
        history and 'best_silhouette_epochs' in history and history['best_silhouette_epochs'] 
        for history in histories.values()
    )
    
    num_cols = 2
    num_rows = 2 if include_silhouette else 2
    
    # Plot training loss
    plt.subplot(num_rows, num_cols, 1)
    for model_type, history in histories.items():
        if history is None:
            continue
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], label=f"{model_type.replace('_', ' ').title()}")
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation loss
    plt.subplot(num_rows, num_cols, 2)
    for model_type, history in histories.items():
        if history is None:
            continue
        epochs = range(1, len(history['val_loss']) + 1)
        plt.plot(epochs, history['val_loss'], label=f"{model_type.replace('_', ' ').title()}")
        
        # Mark best epochs based on silhouette scores
        if show_best_silhouette and 'best_silhouette_epochs' in history and history['best_silhouette_epochs']:
            for method, data in history['best_silhouette_epochs'].items():
                if method == 'umap':  # Use UMAP for visualization as it's typically the best
                    epoch = data['epoch']
                    # Find the validation loss at this epoch
                    if epoch <= len(history['val_loss']):
                        val_loss = history['val_loss'][epoch-1]
                        plt.plot(epoch, val_loss, 'o', markersize=8,
                                label=f"{model_type.title()} Best UMAP (Ep {epoch})")
    
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training accuracy
    plt.subplot(num_rows, num_cols, 3)
    for model_type, history in histories.items():
        if history is None:
            continue
        epochs = range(1, len(history['train_acc']) + 1)
        plt.plot(epochs, history['train_acc'], label=f"{model_type.replace('_', ' ').title()}")
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    plt.subplot(num_rows, num_cols, 4)
    for model_type, history in histories.items():
        if history is None:
            continue
        epochs = range(1, len(history['val_acc']) + 1)
        plt.plot(epochs, history['val_acc'], label=f"{model_type.replace('_', ' ').title()}")
        
        # Mark best epochs based on silhouette scores
        if show_best_silhouette and 'best_silhouette_epochs' in history and history['best_silhouette_epochs']:
            for method, data in history['best_silhouette_epochs'].items():
                if method == 'umap':  # Use UMAP for visualization as it's typically the best
                    epoch = data['epoch']
                    # Find the validation accuracy at this epoch
                    if epoch <= len(history['val_acc']):
                        val_acc = history['val_acc'][epoch-1]
                        plt.plot(epoch, val_acc, 'o', markersize=8,
                                label=f"{model_type.title()} Best UMAP (Ep {epoch})")
    
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{prefix}_training_metrics.png"))
    plt.close()
    
    # Create silhouette score plots if available
    for model_type, history in histories.items():
        if history is None or 'silhouette_scores' not in history or not history['silhouette_scores']:
            continue
        
        # Extract epochs and scores
        sil_data = history['silhouette_scores']
        epochs = [entry['epoch'] for entry in sil_data]
        
        # Get scores for each method
        pca_scores = [entry['scores'].get('pca', -1) for entry in sil_data]
        tsne_scores = [entry['scores'].get('tsne', -1) for entry in sil_data]
        umap_scores = [entry['scores'].get('umap', -1) for entry in sil_data]
        direct_scores = [entry['scores'].get('direct', -1) for entry in sil_data]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot scores
        if max(pca_scores) > 0:
            plt.plot(epochs, pca_scores, 'o-', label='PCA')
        if max(tsne_scores) > 0:
            plt.plot(epochs, tsne_scores, 's-', label='t-SNE')
        if max(umap_scores) > 0:
            plt.plot(epochs, umap_scores, '^-', label='UMAP')
        if max(direct_scores) > 0:
            plt.plot(epochs, direct_scores, 'd-', label='Direct')
        
        # Mark best epochs
        if 'best_silhouette_epochs' in history:
            for method, data in history['best_silhouette_epochs'].items():
                if data['score'] > 0:
                    plt.axvline(x=data['epoch'], color='gray', linestyle='--', alpha=0.5)
                    plt.text(data['epoch'], data['score'], f"{method}: {data['epoch']}", 
                            bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title(f'Silhouette Scores - {model_type.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        silhouette_plot_path = os.path.join(vis_dir, f"{prefix}_{model_type}_silhouette_scores.png")
        plt.savefig(silhouette_plot_path)
        plt.close()
    
    # Create a summary of best epochs
    summary = {
        'models': {}
    }
    
    for model_type, history in histories.items():
        if history is None:
            continue
        
        model_summary = {
            'val_loss': min(history['val_loss']),
            'val_acc': max(history['val_acc']),
            'best_epoch_val_loss': history['val_loss'].index(min(history['val_loss'])) + 1,
            'best_epoch_val_acc': history['val_acc'].index(max(history['val_acc'])) + 1,
        }
        
        # Add silhouette information if available
        if 'best_silhouette_epochs' in history:
            model_summary['best_silhouette'] = history['best_silhouette_epochs']
        
        summary['models'][model_type] = model_summary
    
    # Save summary
    with open(os.path.join(vis_dir, f"{prefix}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary


def visualize_raw_data_from_tensors(config=None, config_path='config.json'):
    """
    Load raw data and create visualizations using dimensionality reduction techniques.
    
    This function follows the approach from layers.ipynb to extract features, applying
    local z-scoring and proper handling of padded values.
    
    Args:
        config (dict, optional): Configuration dictionary
        config_path (str, optional): Path to the configuration file
    """
    # Load configuration if not provided
    if config is None:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Load data
    print("Loading data...")
    parsed_data_dict = load_mat_data(config['data_path'])
    print(f"Loaded data for {len(parsed_data_dict)} patients")
    
    # Build tensors
    print("Building tensors...")
    all_tensors, paddedC = build_all_patient_tensors(parsed_data_dict, pad_value=config['pad_value'])
    
    # Create visualizations directory
    vis_dir = os.path.join(config['models_dir'], "raw_data_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract features for each modality
    print("Extracting features from phase magnitude data...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pmag_features, pmag_labels = extract_channel_features_modality(
            all_tensors, parsed_data_dict, modality="phase_magnitude", pad_value=config['pad_value']
        )
    
    print("Extracting features from phase angle data...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pang_features, pang_labels = extract_channel_features_modality(
            all_tensors, parsed_data_dict, modality="phase_angle", pad_value=config['pad_value']
        )
    
    print("Extracting features from comodulation data...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comod_features, comod_labels = extract_channel_features_modality(
            all_tensors, parsed_data_dict, modality="comod", pad_value=config['pad_value']
        )
    
    # Visualize each modality with different methods
    
    # Phase Magnitude
    print(f"Visualizing Phase Magnitude data (n={len(pmag_features)})...")
    modality_progress = tqdm(total=3, desc="Phase Magnitude Visualizations", leave=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        visualize_pca(
            pmag_features, pmag_labels, 
            title="Phase Magnitude (PCA)", 
            save_path=os.path.join(vis_dir, "phase_magnitude_pca.png")
        )
    modality_progress.update(1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        visualize_umap(
            pmag_features, pmag_labels, 
            title="Phase Magnitude (UMAP)", 
            save_path=os.path.join(vis_dir, "phase_magnitude_umap.png")
        )
    modality_progress.update(1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        visualize_tsne(
            pmag_features, pmag_labels, 
            title="Phase Magnitude (t-SNE)", 
            save_path=os.path.join(vis_dir, "phase_magnitude_tsne.png")
        )
    modality_progress.update(1)
    modality_progress.close()
    
    # Phase Angle
    print(f"Visualizing Phase Angle data (n={len(pang_features)})...")
    modality_progress = tqdm(total=3, desc="Phase Angle Visualizations", leave=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        visualize_pca(
            pang_features, pang_labels, 
            title="Phase Angle (PCA)", 
            save_path=os.path.join(vis_dir, "phase_angle_pca.png")
        )
    modality_progress.update(1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        visualize_umap(
            pang_features, pang_labels, 
            title="Phase Angle (UMAP)", 
            save_path=os.path.join(vis_dir, "phase_angle_umap.png")
        )
    modality_progress.update(1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        visualize_tsne(
            pang_features, pang_labels, 
            title="Phase Angle (t-SNE)", 
            save_path=os.path.join(vis_dir, "phase_angle_tsne.png")
        )
    modality_progress.update(1)
    modality_progress.close()
    
    # Comodulation
    print(f"Visualizing Comodulation data (n={len(comod_features)})...")
    modality_progress = tqdm(total=3, desc="Comodulation Visualizations", leave=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        visualize_pca(
            comod_features, comod_labels, 
            title="Comodulation (PCA)", 
            save_path=os.path.join(vis_dir, "comodulation_pca.png")
        )
    modality_progress.update(1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        visualize_umap(
            comod_features, comod_labels, 
            title="Comodulation (UMAP)", 
            save_path=os.path.join(vis_dir, "comodulation_umap.png")
        )
    modality_progress.update(1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        visualize_tsne(
            comod_features, comod_labels, 
            title="Comodulation (t-SNE)", 
            save_path=os.path.join(vis_dir, "comodulation_tsne.png")
        )
    modality_progress.update(1)
    modality_progress.close()
    
    print(f"All visualizations saved to {vis_dir}")


def main():
    """
    Main function to run the raw data visualization process.
    """
    visualize_raw_data_from_tensors()


if __name__ == "__main__":
    main() 