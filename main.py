"""
Main script for running neural data processing and model training/evaluation.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from tqdm import tqdm
import argparse
import warnings
from silhouette_tracking import compute_scores

# Silence specific deprecation and user warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*n_jobs value.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Graph is not fully connected.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*'n_iter' was renamed to 'max_iter'.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*OMP.*")

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
from visualizations import (
    visualize_training_run,
    visualize_latent_features,
    visualize_raw_data_from_tensors
)


def load_config(config_path='config.json'):
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def save_model(model, history, model_name, prefix, config):
    """
    Save a trained model and its training history.
    
    Args:
        model (nn.Module): Trained PyTorch model
        history (dict): Training history
        model_name (str): Name of the model (e.g., 'phase_magnitude')
        prefix (str): Prefix for the saved files
        config (dict): Configuration dictionary
    """
    # Create models directory if it doesn't exist
    os.makedirs(config['models_dir'], exist_ok=True)
    
    # Construct base filename
    base_filename = f"{prefix}_{model_name}"
    model_path = os.path.join(config['models_dir'], f"{base_filename}_model.pt")
    history_path = os.path.join(config['models_dir'], f"{base_filename}_history.json")
    
    # Check if files with these names already exist
    suffix = 1
    while os.path.exists(model_path) or os.path.exists(history_path):
        # Files already exist, append suffix
        base_filename = f"{prefix}_{model_name}_{suffix}"
        model_path = os.path.join(config['models_dir'], f"{base_filename}_model.pt")
        history_path = os.path.join(config['models_dir'], f"{base_filename}_history.json")
        suffix += 1
    
    # Extract best silhouette epochs if available
    best_silhouette_epochs = {}
    if 'silhouette_scores' in history and len(history['silhouette_scores']) > 0:
        # Find best epoch for each method
        for score_entry in history['silhouette_scores']:
            epoch = score_entry['epoch']
            for method, score in score_entry['scores'].items():
                if method not in best_silhouette_epochs or score > best_silhouette_epochs[method]['score']:
                    best_silhouette_epochs[method] = {'epoch': epoch, 'score': score}
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'epoch': config['num_epochs'],
        'best_silhouette_epochs': best_silhouette_epochs
    }, model_path)
    
    # Create a serializable version of history without numpy arrays
    serializable_history = {}
    for key, value in history.items():
        if key == 'silhouette_scores':
            # Make sure scores are serializable
            serializable_scores = []
            for entry in value:
                serializable_entry = {'epoch': int(entry['epoch']), 'scores': {}}
                for method, score in entry['scores'].items():
                    # Ensure all scores are native Python floats, not numpy types
                    serializable_entry['scores'][method] = float(score)
                serializable_scores.append(serializable_entry)
            serializable_history[key] = serializable_scores
        else:
            # Convert any numpy values to Python types
            if isinstance(value, list):
                # Convert all values in the list to native Python types
                serializable_history[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            else:
                serializable_history[key] = value
    
    # Add best silhouette epochs to serializable history
    serializable_history['best_silhouette_epochs'] = best_silhouette_epochs
    
    # Save history separately for easier access
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)
    
    print(f"✅ Saved {model_name} model to {model_path}")
    print(f"✅ Saved {model_name} history to {history_path}")
    
    # Return the base_filename so it can be used for saving results
    return base_filename


def save_results(results, model_name, prefix, config, base_filename=None):
    """
    Save model evaluation results.
    
    Args:
        results (dict): Evaluation results dictionary
        model_name (str): Name of the model (e.g., 'phase_magnitude')
        prefix (str): Prefix for the saved files
        config (dict): Configuration dictionary
        base_filename (str, optional): Base filename to use (to match model and history files)
    """
    # Create models directory if it doesn't exist
    os.makedirs(config['models_dir'], exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report'],
        'accuracy': float(results['accuracy'])
    }
    
    # Determine results path
    if base_filename:
        results_path = os.path.join(config['models_dir'], f"{base_filename}_results.json")
    else:
        # Construct base filename
        base_filename = f"{prefix}_{model_name}"
        results_path = os.path.join(config['models_dir'], f"{base_filename}_results.json")
        
        # Check if file with this name already exists
        suffix = 1
        while os.path.exists(results_path):
            # File already exists, append suffix
            base_filename = f"{prefix}_{model_name}_{suffix}"
            results_path = os.path.join(config['models_dir'], f"{base_filename}_results.json")
            suffix += 1
    
    # Save results
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"✅ Saved {model_name} results to {results_path}")
    
    # Return the base_filename for potential further use
    return base_filename


def create_patient_split(all_tensors, parsed_data_dict, validation_patient_idx):
    """
    Create train and validation indices based on patient split.
    
    Args:
        all_tensors (list): List of all patient tensors
        parsed_data_dict (list): List of patient data dictionaries
        validation_patient_idx (int): Index of the patient to use for validation
        
    Returns:
        tuple: (train_indices, validation_indices) for each dataset type
    """
    # Initialize indices for each dataset
    pmag_indices = []
    pang_indices = []
    comod_indices = []
    
    # Process each patient's data
    current_idx = 0
    for patient_idx, tensors in enumerate(all_tensors):
        patient = parsed_data_dict[patient_idx]
        ycoords = np.array(patient['ycoords'])
        dredge = np.array(patient['DREDgeLayerBorders'])
        
        # Count valid channels for this patient
        num_valid_channels = sum(1 for ch in range(len(ycoords)) 
                               if get_channel_layer(ycoords[ch], dredge) != -1)
        
        # Add indices for this patient's channels
        patient_indices = list(range(current_idx, current_idx + num_valid_channels))
        
        if patient_idx == validation_patient_idx:
            # Add to validation set
            pmag_indices.extend(patient_indices)
            pang_indices.extend(patient_indices)
            comod_indices.extend(patient_indices)
        else:
            # Add to train set
            pmag_indices.extend(patient_indices)
            pang_indices.extend(patient_indices)
            comod_indices.extend(patient_indices)
        
        current_idx += num_valid_channels
    
    return {
        'phase_magnitude': (pmag_indices, pmag_indices),
        'phase_angle': (pang_indices, pang_indices),
        'comodulation': (comod_indices, comod_indices)
    }


def create_dataloaders(all_tensors, parsed_data_dict, config, validation_patient_idx):
    """
    Create DataLoaders for each type of dataset using patient-based split.
    
    Args:
        all_tensors (list): List of all patient tensors
        parsed_data_dict (list): List of patient data dictionaries
        config (dict): Configuration dictionary
        validation_patient_idx (int): Index of the patient to use for validation
        
    Returns:
        dict: Dictionary of train and validation DataLoaders for each dataset type
    """
    # Create datasets
    pmag_dataset = ChannelLevelPhaseMagnitudeDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    pang_dataset = ChannelLevelPhaseAngleDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    comod_dataset = ChannelLevelComodDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    
    print(f"Created datasets: Phase Magnitude: {len(pmag_dataset)}, "
          f"Phase Angle: {len(pang_dataset)}, Comodulation: {len(comod_dataset)}")
    
    # Get train and validation indices for each dataset
    split_indices = create_patient_split(all_tensors, parsed_data_dict, validation_patient_idx)
    
    # Create dataloaders for each dataset
    dataloaders = {}
    for dataset_type, dataset in [
        ('phase_magnitude', pmag_dataset),
        ('phase_angle', pang_dataset),
        ('comodulation', comod_dataset)
    ]:
        train_indices, validation_indices = split_indices[dataset_type]
        
        # Create train and validation subsets
        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, validation_indices)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False)
        
        dataloaders[dataset_type] = {
            'train': train_loader,
            'validation': validation_loader
        }
    
    return dataloaders


def train_model(model, train_loader, val_loader, device, config, show_batch_progress=False):
    """
    Train the autoencoder model.
    
    Args:
        model (nn.Module): PyTorch model (Autoencoder with classification head)
        train_loader (DataLoader): Training set DataLoader
        val_loader (DataLoader): Validation set DataLoader
        device (torch.device): Device to train on
        config (dict): Configuration dictionary
        show_batch_progress (bool): Whether to show progress bars for individual batches
        
    Returns:
        tuple: (trained model, training history, best model state)
    """
    model_name_clean = model.model_name.replace(' ', '_').lower()
    
    # Training parameters
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    weight_decay = config.get('weight_decay', 1e-4)
    
    # Loss weights for reconstruction, KL divergence, and classification
    lambda_recon = config.get('lambda_recon', 1.0)
    lambda_kl = config.get('lambda_kl', 0.1)
    lambda_cls = config.get('lambda_cls', 1.0)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize loss functions
    recon_criterion = nn.MSELoss(reduction='mean')
    cls_criterion = nn.CrossEntropyLoss()
    
    # Initialize metrics tracking
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    
    # Initialize history dictionary for tracking metrics
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'train_cls_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'val_cls_loss': [],
        'val_acc': [],
        'silhouette_scores': [],  # Add silhouette scores tracking
        'calinski_harabasz_scores': []  # Add Calinski-Harabasz scores tracking
    }
    
    # Create directory for silhouette scores
    silhouette_dir = os.path.join('saved_models', 'silhouette_scores')
    os.makedirs(silhouette_dir, exist_ok=True)
    
    # Frequency for score calculation (default every 50 epochs)
    score_freq = config.get('silhouette_freq', 50)
    
    print(f"\n🚀 Starting training for {model.model_name} with {num_epochs} epochs")
    print(f"   Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    print(f"   Loss weights - Reconstruction: {lambda_recon}, KL: {lambda_kl}, Classification: {lambda_cls}")
    print(f"   Score calculation frequency: Every {score_freq} epochs")
    
    # Track best scores
    best_silhouette = {
        'pca': -1.0,
        'tsne': -1.0,
        'umap': -1.0,
        'direct': -1.0
    }
    
    best_calinski = {
        'pca': -1.0,
        'tsne': -1.0,
        'umap': -1.0,
        'direct': -1.0
    }
    
    best_silhouette_epoch = {
        'pca': -1,
        'tsne': -1,
        'umap': -1,
        'direct': -1
    }
    
    best_calinski_epoch = {
        'pca': -1,
        'tsne': -1,
        'umap': -1,
        'direct': -1
    }
    
    # Training loop
    epoch_bar = tqdm(range(num_epochs), desc=f"Training {model.model_name}", position=0, leave=True)
    
    for epoch in epoch_bar:
        # Training phase
        model.train()
        train_running_loss = 0.0
        train_running_recon_loss = 0.0
        train_running_kl_loss = 0.0
        train_running_cls_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Show batch progress if requested
        train_batch_iter = tqdm(train_loader, desc=f"Training", leave=False) if show_batch_progress else train_loader
        
        for inputs, masks, labels in train_batch_iter:
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            # Forward pass
            recon, z, logits = model(inputs, masks)
            
            # Reconstruction loss (masked)
            diff = recon_criterion(recon, inputs)
            masked_diff = diff * masks
            loss_recon = masked_diff.sum() / masks.sum() if masks.sum() > 0 else 0.0
            
            # KL divergence loss
            loss_kl = lambda_kl * (z**2).mean()
            
            # Classification loss
            loss_cls = cls_criterion(logits, labels)
            
            # Combined loss
            loss = loss_recon + loss_kl + lambda_cls * loss_cls
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_running_loss += loss.item()
            train_running_recon_loss += loss_recon.item()
            train_running_kl_loss += loss_kl.item()
            train_running_cls_loss += loss_cls.item()
            
            # Compute predictions
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar if showing batch progress
            if show_batch_progress:
                train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
                train_batch_iter.set_postfix_str(f"loss={loss.item():.4f}, acc={train_acc:.1f}%")
        
        # Compute epoch metrics
        epoch_train_loss = train_running_loss / len(train_loader)
        epoch_train_recon_loss = train_running_recon_loss / len(train_loader)
        epoch_train_kl_loss = train_running_kl_loss / len(train_loader)
        epoch_train_cls_loss = train_running_cls_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_recon_loss = 0.0
        val_running_kl_loss = 0.0
        val_running_cls_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Show validation progress
        val_batch_iter = tqdm(val_loader, desc=f"Validating", leave=False)
        
        with torch.no_grad():
            for inputs, masks, labels in val_batch_iter:
                inputs = inputs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                
                # Forward pass
                recon, z, logits = model(inputs, masks)
                
                # Reconstruction loss (masked)
                diff = recon_criterion(recon, inputs)
                masked_diff = diff * masks
                loss_recon = masked_diff.sum() / masks.sum() if masks.sum() > 0 else 0.0
                
                # KL divergence loss
                loss_kl = lambda_kl * (z**2).mean()
                
                # Classification loss
                loss_cls = cls_criterion(logits, labels)
                
                # Combined loss
                loss = loss_recon + loss_kl + lambda_cls * loss_cls
                
                # Update metrics
                val_running_loss += loss.item()
                val_running_recon_loss += loss_recon.item()
                val_running_kl_loss += loss_kl.item()
                val_running_cls_loss += loss_cls.item()
                
                # Compute predictions
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # Update progress bar
                val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
                val_batch_iter.set_postfix_str(f"acc={val_acc:.1f}%")
        
        # Compute validation metrics
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_recon_loss = val_running_recon_loss / len(val_loader)
        epoch_val_kl_loss = val_running_kl_loss / len(val_loader)
        epoch_val_cls_loss = val_running_cls_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_recon_loss'].append(epoch_train_recon_loss)
        history['train_kl_loss'].append(epoch_train_kl_loss)
        history['train_cls_loss'].append(epoch_train_cls_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_recon_loss'].append(epoch_val_recon_loss)
        history['val_kl_loss'].append(epoch_val_kl_loss)
        history['val_cls_loss'].append(epoch_val_cls_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Calculate scores at specific intervals or final epoch
        if (epoch + 1) % score_freq == 0 or epoch == num_epochs - 1 or epoch == 0:
            print(f"\n📊 Calculating cluster quality scores at epoch {epoch+1}...")
            
            # Extract features and labels from validation set
            try:
                # Extract features from validation dataset
                val_features, val_labels = extract_latent_features(model, val_loader.dataset, device)
                
                # Compute scores
                scores = compute_cluster_quality_scores(val_features, val_labels)
                
                # Log silhouette scores
                silhouette_scores = scores['silhouette']
                epoch_silhouette_scores = {
                    'epoch': int(epoch + 1),
                    'scores': {k: float(v) if v is not None else -1.0 for k, v in silhouette_scores.items()}
                }
                history['silhouette_scores'].append(epoch_silhouette_scores)
                
                # Log Calinski-Harabasz scores
                calinski_scores = scores['calinski_harabasz']
                epoch_calinski_scores = {
                    'epoch': int(epoch + 1),
                    'scores': {k: float(v) if v is not None else -1.0 for k, v in calinski_scores.items()}
                }
                history['calinski_harabasz_scores'].append(epoch_calinski_scores)
                
                # Update best silhouette scores
                for method, score in silhouette_scores.items():
                    if score is not None:
                        score_float = float(score)
                        if score_float > best_silhouette[method]:
                            best_silhouette[method] = score_float
                            best_silhouette_epoch[method] = int(epoch + 1)
                
                # Update best Calinski-Harabasz scores
                for method, score in calinski_scores.items():
                    if score is not None:
                        score_float = float(score)
                        if score_float > best_calinski[method]:
                            best_calinski[method] = score_float
                            best_calinski_epoch[method] = int(epoch + 1)
                
                # Print scores
                print(f"  Silhouette Scores - PCA: {silhouette_scores['pca']:.4f if silhouette_scores['pca'] is not None else 'N/A'}, "
                      f"t-SNE: {silhouette_scores['tsne']:.4f if silhouette_scores['tsne'] is not None else 'N/A'}, "
                      f"UMAP: {silhouette_scores['umap']:.4f if silhouette_scores['umap'] is not None else 'N/A'}")
                
                print(f"  Calinski-Harabasz - PCA: {calinski_scores['pca']:.1f if calinski_scores['pca'] is not None else 'N/A'}, "
                      f"t-SNE: {calinski_scores['tsne']:.1f if calinski_scores['tsne'] is not None else 'N/A'}, "
                      f"UMAP: {calinski_scores['umap']:.1f if calinski_scores['umap'] is not None else 'N/A'}")
                
                # Save scores to JSON files
                silhouette_file = os.path.join(silhouette_dir, f"{model_name_clean}_silhouette.json")
                calinski_file = os.path.join(silhouette_dir, f"{model_name_clean}_calinski_harabasz.json")
                
                # Create serializable versions of the scores
                serializable_silhouette = []
                for entry in history['silhouette_scores']:
                    serializable_entry = {
                        'epoch': int(entry['epoch']),
                        'scores': {k: float(v) for k, v in entry['scores'].items()}
                    }
                    serializable_silhouette.append(serializable_entry)
                
                serializable_calinski = []
                for entry in history['calinski_harabasz_scores']:
                    serializable_entry = {
                        'epoch': int(entry['epoch']),
                        'scores': {k: float(v) for k, v in entry['scores'].items()}
                    }
                    serializable_calinski.append(serializable_entry)
                
                with open(silhouette_file, 'w') as f:
                    json.dump(serializable_silhouette, f, indent=4)
                
                with open(calinski_file, 'w') as f:
                    json.dump(serializable_calinski, f, indent=4)
                
            except Exception as e:
                print(f"⚠️ Error calculating cluster quality scores: {e}")
        
        # Save model and visualizations every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'history': history,
                'silhouette_scores': history['silhouette_scores'],
                'calinski_harabasz_scores': history['calinski_harabasz_scores'],
                'best_silhouette': best_silhouette,
                'best_silhouette_epoch': best_silhouette_epoch,
                'best_calinski': best_calinski,
                'best_calinski_epoch': best_calinski_epoch
            }
            
            checkpoint_path = os.path.join('saved_models', f'{model_name_clean}_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"\n📦 Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
            
            # Create visualizations
            epoch_vis_dir = os.path.join(vis_dir, f'epoch_{epoch+1}')
            os.makedirs(epoch_vis_dir, exist_ok=True)
            
            try:
                # Training data visualizations
                visualize_latent_features(
                    model, train_loader.dataset, None,
                    f"{model.model_name} (Training) - Epoch {epoch+1}",
                    os.path.join(epoch_vis_dir, 'train')
                )
                
                # Validation data visualizations
                visualize_latent_features(
                    model, val_loader.dataset, None,
                    f"{model.model_name} (Validation) - Epoch {epoch+1}",
                    os.path.join(epoch_vis_dir, 'val')
                )
                
                print(f"📊 Saved visualizations for epoch {epoch+1} to {epoch_vis_dir}")
            except Exception as e:
                print(f"⚠️ Error creating visualizations for epoch {epoch+1}: {e}")
        
        # Update progress bar with all metrics
        status = {
            'tr_l': f"{epoch_train_loss:.4f}",
            'tr_a': f"{epoch_train_acc:.1f}",
            'v_l': f"{epoch_val_loss:.4f}",
            'v_a': f"{epoch_val_acc:.1f}"
        }
        
        # Format the status message
        status_str = f"tr_l={status['tr_l']}, tr_a={status['tr_a']}%, v_l={status['v_l']}, v_a={status['v_a']}%"
        epoch_bar.set_postfix_str(status_str)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'history': history,
                'silhouette_scores': history['silhouette_scores'],
                'calinski_harabasz_scores': history['calinski_harabasz_scores'],
                'best_silhouette': best_silhouette,
                'best_silhouette_epoch': best_silhouette_epoch,
                'best_calinski': best_calinski,
                'best_calinski_epoch': best_calinski_epoch
            }
            best_epoch = epoch + 1
    
    # Create plots of cluster quality scores over epochs
    if len(history['silhouette_scores']) > 0:
        try:
            # Extract epochs and silhouette scores
            epochs = [int(entry['epoch']) for entry in history['silhouette_scores']]
            pca_silhouette = [float(entry['scores']['pca']) for entry in history['silhouette_scores']]
            tsne_silhouette = [float(entry['scores']['tsne']) for entry in history['silhouette_scores']]
            umap_silhouette = [float(entry['scores']['umap']) for entry in history['silhouette_scores']]
            
            # Create the silhouette scores plot
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, pca_silhouette, 'o-', label='PCA')
            plt.plot(epochs, tsne_silhouette, 's-', label='t-SNE')
            plt.plot(epochs, umap_silhouette, '^-', label='UMAP')
            
            plt.xlabel('Epoch')
            plt.ylabel('Silhouette Score')
            plt.title(f'Silhouette Scores for {model.model_name}')
            plt.legend()
            plt.grid(True)
            
            # Mark best epochs for each method
            for method, best_epoch in best_silhouette_epoch.items():
                if best_epoch > 0 and method != 'direct':
                    plt.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
            
            # Save plot
            silhouette_plot_path = os.path.join(silhouette_dir, f"{model_name_clean}_silhouette_scores.png")
            plt.savefig(silhouette_plot_path)
            plt.close()
            print(f"📈 Saved silhouette score plot to {silhouette_plot_path}")
            
            # Extract Calinski-Harabasz scores
            if len(history['calinski_harabasz_scores']) > 0:
                pca_calinski = [float(entry['scores']['pca']) for entry in history['calinski_harabasz_scores']]
                tsne_calinski = [float(entry['scores']['tsne']) for entry in history['calinski_harabasz_scores']]
                umap_calinski = [float(entry['scores']['umap']) for entry in history['calinski_harabasz_scores']]
                
                # Create the Calinski-Harabasz plot
                plt.figure(figsize=(12, 8))
                plt.plot(epochs, pca_calinski, 'o-', label='PCA')
                plt.plot(epochs, tsne_calinski, 's-', label='t-SNE')
                plt.plot(epochs, umap_calinski, '^-', label='UMAP')
                
                plt.xlabel('Epoch')
                plt.ylabel('Calinski-Harabasz Index')
                plt.title(f'Calinski-Harabasz Indices for {model.model_name}')
                plt.legend()
                plt.grid(True)
                
                # Mark best epochs for each method
                for method, best_epoch in best_calinski_epoch.items():
                    if best_epoch > 0 and method != 'direct':
                        plt.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
                
                # Save plot
                calinski_plot_path = os.path.join(silhouette_dir, f"{model_name_clean}_calinski_harabasz_scores.png")
                plt.savefig(calinski_plot_path)
                plt.close()
                print(f"📈 Saved Calinski-Harabasz index plot to {calinski_plot_path}")
        except Exception as e:
            print(f"⚠️ Error creating cluster quality score plots: {e}")
    
    print(f"\n✨ Best validation loss achieved at epoch {best_epoch}")
    print(f"🔍 Best silhouette scores: PCA: {best_silhouette['pca']:.4f} (epoch {best_silhouette_epoch['pca']}), "
          f"t-SNE: {best_silhouette['tsne']:.4f} (epoch {best_silhouette_epoch['tsne']}), "
          f"UMAP: {best_silhouette['umap']:.4f} (epoch {best_silhouette_epoch['umap']})")
    
    print(f"🔍 Best Calinski-Harabasz indices: PCA: {best_calinski['pca']:.1f} (epoch {best_calinski_epoch['pca']}), "
          f"t-SNE: {best_calinski['tsne']:.1f} (epoch {best_calinski_epoch['tsne']}), "
          f"UMAP: {best_calinski['umap']:.1f} (epoch {best_calinski_epoch['umap']})")
    
    return model, history, best_model_state


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the validation set.
    
    Args:
        model (nn.Module): Trained PyTorch model (Autoencoder with classification head)
        dataloader (DataLoader): Validation set DataLoader
        device (torch.device): Device to evaluate on
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, masks, labels in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            # Model returns (recon, z, logits)
            _, _, logits = model(inputs, masks)
            preds = torch.argmax(logits, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Calculate confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': report['accuracy']
    }


def extract_latent_features(model, dataset, device):
    """
    Extract latent feature representations and corresponding labels from a model
    
    Args:
        model (nn.Module): Trained PyTorch model
        dataset (torch.utils.data.Dataset): Dataset to extract features from
        device (torch.device): Device to use
        
    Returns:
        tuple: (features, labels)
    """
    model.eval()
    all_features = []
    all_labels = []
    
    # Create DataLoader to batch the computations
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for data, mask, labels in tqdm(dataloader, desc="Extracting features", leave=False):
            data = data.to(device)
            mask = mask.to(device)
            
            # Get latent representation (recon, z, logits)
            _, z, _ = model(data, mask)
            
            all_features.append(z.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    return features, labels


def compute_cluster_quality_scores(features, labels):
    """
    Compute silhouette scores and Calinski-Harabasz indices using different dimensionality reduction techniques
    
    Args:
        features (numpy.ndarray): Feature matrix
        labels (numpy.ndarray): Class labels
        
    Returns:
        dict: Dictionary with silhouette scores and Calinski-Harabasz indices
    """
    # Initialize scores dictionary
    scores = {
        'silhouette': {},
        'calinski_harabasz': {}
    }
    
    # Compute direct silhouette score if dimensionality is reasonable
    if features.shape[1] <= 50:  # Use direct features if dimension is not too high
        try:
            scores['silhouette']['direct'] = float(silhouette_score(features, labels))
        except ValueError:
            scores['silhouette']['direct'] = -1.0  # Invalid score
        
        # Compute direct Calinski-Harabasz index
        try:
            scores['calinski_harabasz']['direct'] = float(calinski_harabasz_score(features, labels))
        except ValueError:
            scores['calinski_harabasz']['direct'] = -1.0  # Invalid score
    else:
        scores['silhouette']['direct'] = -1.0  # Skip for high dimensions
        scores['calinski_harabasz']['direct'] = -1.0  # Skip for high dimensions
    
    # Apply PCA
    try:
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)
        scores['silhouette']['pca'] = float(silhouette_score(reduced_features, labels))
        scores['calinski_harabasz']['pca'] = float(calinski_harabasz_score(reduced_features, labels))
    except Exception as e:
        print(f"PCA quality score calculation error: {e}")
        scores['silhouette']['pca'] = -1.0
        scores['calinski_harabasz']['pca'] = -1.0
    
    # Apply t-SNE
    try:
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(features)
        scores['silhouette']['tsne'] = float(silhouette_score(reduced_features, labels))
        scores['calinski_harabasz']['tsne'] = float(calinski_harabasz_score(reduced_features, labels))
    except Exception as e:
        print(f"t-SNE quality score calculation error: {e}")
        scores['silhouette']['tsne'] = -1.0
        scores['calinski_harabasz']['tsne'] = -1.0
    
    # Apply UMAP
    try:
        umap_reducer = UMAP(n_components=2, random_state=42)
        reduced_features = umap_reducer.fit_transform(features)
        scores['silhouette']['umap'] = float(silhouette_score(reduced_features, labels))
        scores['calinski_harabasz']['umap'] = float(calinski_harabasz_score(reduced_features, labels))
    except Exception as e:
        print(f"UMAP quality score calculation error: {e}")
        scores['silhouette']['umap'] = -1.0
        scores['calinski_harabasz']['umap'] = -1.0
    
    return scores


def main():
    """
    Main function to run the neural data processor.
    
    This function loads data, creates datasets and dataloaders, trains models,
    evaluates them, and saves results.
    """
    # Set console colors for better visibility
    class Colors:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Data Processor')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--use-optimized', action='store_true',
                        help='Use optimized configs for each model type if available')
    parser.add_argument('--visualize-raw', action='store_true',
                        help='Generate visualizations of raw data (overrides config setting)')
    parser.add_argument('--no-visualize-raw', action='store_true',
                        help='Skip visualizations of raw data (overrides config setting)')
    parser.add_argument('--show-batch-progress', action='store_true',
                        help='Show progress bars for individual batches during all epochs (not just the first)')
    parser.add_argument('--skip-latent-vis', action='store_true',
                        help='Skip latent feature visualizations (useful if visualization is causing errors)')
    parser.add_argument('--silhouette-freq', type=int, default=50,
                        help='Frequency (in epochs) for calculating silhouette scores (default: 50)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Add silhouette frequency to config
    config['silhouette_freq'] = args.silhouette_freq
    
    # Load optimized configs if specified
    model_configs = {
        'phase_magnitude': config.copy(),
        'phase_angle': config.copy(),
        'comodulation': config.copy()
    }
    
    if args.use_optimized:
        print(f"{Colors.HEADER}🔍 Checking for optimized configurations...{Colors.ENDC}")
        for model_type in model_configs.keys():
            optimized_config_path = f'config_{model_type}_optimized.json'
            if os.path.exists(optimized_config_path):
                model_configs[model_type] = load_config(optimized_config_path)
                # Ensure silhouette_freq is in the optimized config
                model_configs[model_type]['silhouette_freq'] = args.silhouette_freq
                print(f"{Colors.GREEN}  ✓ Using optimized config for {model_type}: {optimized_config_path}{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}  ⚠ No optimized config found for {model_type}, using default{Colors.ENDC}")
    
    # Ask for a prefix for saved models (default: timestamp)
    prefix = input(f"{Colors.CYAN}Enter a prefix for saved files (e.g., 10epochs_test1): {Colors.ENDC}")
    if not prefix:
        from datetime import datetime
        prefix = datetime.now().strftime("%m%d_%H%M")
    
    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    print(f"{Colors.BOLD}💻 Using {device} device{Colors.ENDC}")
    
    # Create directories for saving results
    os.makedirs(config['models_dir'], exist_ok=True)
    
    # Create visualization directory
    vis_dir = os.path.join(config['models_dir'], f"{prefix}_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(os.path.join(vis_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(vis_dir, "validation"), exist_ok=True)
    
    # Create directory for silhouette scores
    silhouette_dir = os.path.join(config['models_dir'], f"{prefix}_silhouette")
    os.makedirs(silhouette_dir, exist_ok=True)
    
    # Load data
    print(f"{Colors.HEADER}📊 Loading data...{Colors.ENDC}")
    parsed_data_dict = load_mat_data(config['data_path'])
    print(f"{Colors.GREEN}  ✓ Loaded data for {len(parsed_data_dict)} patients{Colors.ENDC}")
    
    # Build tensors
    print(f"{Colors.HEADER}🔧 Building tensors...{Colors.ENDC}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_tensors, paddedC = build_all_patient_tensors(parsed_data_dict, pad_value=config['pad_value'])
    print(f"{Colors.GREEN}  ✓ Built {len(all_tensors)} patient tensors with maxC={paddedC}{Colors.ENDC}")
    
    # Determine whether to visualize raw data, with command-line overriding config
    visualize_raw = config.get('visualize_raw_data', False)
    if args.visualize_raw:
        visualize_raw = True
    if args.no_visualize_raw:
        visualize_raw = False
    
    # Visualize raw data if configured to do so
    if visualize_raw:
        print(f"{Colors.HEADER}📈 Generating raw data visualizations...{Colors.ENDC}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            visualize_raw_data_from_tensors(config)
        print(f"{Colors.GREEN}  ✓ Raw data visualizations complete{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}ℹ️ Skipping raw data visualizations (set visualize_raw_data=true in config or use --visualize-raw flag to enable){Colors.ENDC}")
    
    # Use the last patient as validation set
    validation_patient_idx = len(parsed_data_dict) - 1
    print(f"{Colors.CYAN}ℹ️ Using patient {validation_patient_idx + 1} as validation set{Colors.ENDC}")
    
    # Create dataloaders
    print(f"{Colors.HEADER}🔄 Creating dataloaders...{Colors.ENDC}")
    dataloaders = create_dataloaders(all_tensors, parsed_data_dict, config, validation_patient_idx)
    
    # Get latent dimension from config
    pmag_latent_dim = model_configs['phase_magnitude'].get('latent_dim', 64)
    pang_latent_dim = model_configs['phase_angle'].get('latent_dim', 64)
    comod_latent_dim = model_configs['comodulation'].get('latent_dim', 64)
    
    # Create raw datasets for the visualizations
    pmag_dataset = ChannelLevelPhaseMagnitudeDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    pang_dataset = ChannelLevelPhaseAngleDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    comod_dataset = ChannelLevelComodDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    
    print(f"{Colors.GREEN}  ✓ Created datasets: Phase Magnitude: {len(pmag_dataset)}, Phase Angle: {len(pang_dataset)}, Comodulation: {len(comod_dataset)}{Colors.ENDC}")
    
    # Get indices for train/validation split from our dataloaders
    pmag_val_indices = dataloaders['phase_magnitude']['validation'].dataset.indices
    pmag_train_indices = dataloaders['phase_magnitude']['train'].dataset.indices
    
    pang_val_indices = dataloaders['phase_angle']['validation'].dataset.indices
    pang_train_indices = dataloaders['phase_angle']['train'].dataset.indices
    
    comod_val_indices = dataloaders['comodulation']['validation'].dataset.indices
    comod_train_indices = dataloaders['comodulation']['train'].dataset.indices
    
    # Create training and validation subsets for visualization
    pmag_val_subset = Subset(pmag_dataset, pmag_val_indices)
    pmag_train_subset = Subset(pmag_dataset, pmag_train_indices)
    
    pang_val_subset = Subset(pang_dataset, pang_val_indices)
    pang_train_subset = Subset(pang_dataset, pang_train_indices)
    
    comod_val_subset = Subset(comod_dataset, comod_val_indices)
    comod_train_subset = Subset(comod_dataset, comod_train_indices)
    
    # We need to extract the raw labels for each dataset
    # Extract labels in a more efficient way with progress bars
    print(f"{Colors.HEADER}📋 Extracting labels for visualization...{Colors.ENDC}")
    
    def extract_labels(dataset, desc):
        labels = []
        with torch.no_grad():
            for _, _, label in tqdm(DataLoader(dataset, batch_size=32, shuffle=False), 
                                    desc=desc, leave=False):
                labels.extend(label.numpy())
        return np.array(labels)
    
    # For validation data
    pmag_val_labels = extract_labels(pmag_val_subset, "Phase Magnitude (Val)")
    pang_val_labels = extract_labels(pang_val_subset, "Phase Angle (Val)")
    comod_val_labels = extract_labels(comod_val_subset, "Comodulation (Val)")
    
    # For training data
    pmag_train_labels = extract_labels(pmag_train_subset, "Phase Magnitude (Train)")
    pang_train_labels = extract_labels(pang_train_subset, "Phase Angle (Train)")
    comod_train_labels = extract_labels(comod_train_subset, "Comodulation (Train)")
    
    # Create autoencoder models with proper maxC dimension
    print(f"{Colors.HEADER}🧠 Creating models...{Colors.ENDC}")
    pmag_model = PhaseMagnitudeLayerClassifier(
        num_classes=config['num_classes'],
        latent_dim=pmag_latent_dim,
        maxC=paddedC
    )
    pang_model = PhaseAngleLayerClassifier(
        num_classes=config['num_classes'],
        latent_dim=pang_latent_dim,
        maxC=paddedC
    )
    comod_model = ComodLayerClassifier(
        num_classes=config['num_classes'],
        latent_dim=comod_latent_dim,
        maxC=paddedC
    )
    
    # Print model configurations when using optimized configs
    if args.use_optimized:
        print(f"{Colors.HEADER}📝 Model configurations:{Colors.ENDC}")
        for model_type, model_config in model_configs.items():
            print(f"  {model_type.capitalize()}:")
            print(f"    - Learning rate: {model_config.get('learning_rate', config['learning_rate'])}")
            print(f"    - Batch size: {model_config.get('batch_size', config['batch_size'])}")
            print(f"    - Latent dim: {model_config.get('latent_dim', 64)}")
            print(f"    - Lambda cls: {model_config.get('lambda_cls', 1.0)}")
            if 'model_params' in model_config:
                if 'dropout_rate' in model_config['model_params']:
                    print(f"    - Dropout rate: {model_config['model_params']['dropout_rate']}")
            if 'optimizer_params' in model_config:
                if 'weight_decay' in model_config['optimizer_params']:
                    print(f"    - Weight decay: {model_config['optimizer_params']['weight_decay']}")
    
    # Train and evaluate models
    print(f"{Colors.HEADER}🔥 Starting model training and evaluation...{Colors.ENDC}")
    
    print(f"{Colors.BOLD}▶️ Training Phase Magnitude model...{Colors.ENDC}")
    pmag_model, pmag_history, pmag_best_model_state = train_model(pmag_model, dataloaders['phase_magnitude']['train'], 
                              dataloaders['phase_magnitude']['validation'], device, 
                              model_configs['phase_magnitude'], args.show_batch_progress)
    
    print(f"{Colors.HEADER}📊 Evaluating Phase Magnitude model...{Colors.ENDC}")
    pmag_results = evaluate_model(pmag_model, dataloaders['phase_magnitude']['validation'], device)
    pmag_base_filename = save_model(pmag_model, pmag_history, 'phase_magnitude', prefix, config)
    save_results(pmag_results, 'phase_magnitude', prefix, config, pmag_base_filename)
    
    # Visualize latent features for Phase Magnitude model
    if not args.skip_latent_vis:
        print(f"{Colors.HEADER}📊 Visualizing Phase Magnitude latent features...{Colors.ENDC}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Validation data
                visualize_latent_features(pmag_model, pmag_val_subset, pmag_val_labels, 
                                        "Phase Magnitude (Validation)", os.path.join(vis_dir, "validation"))
                # Training data
                visualize_latent_features(pmag_model, pmag_train_subset, pmag_train_labels, 
                                        "Phase Magnitude (Training)", os.path.join(vis_dir, "training"))
        except Exception as e:
            print(f"{Colors.WARNING}⚠️ Error during Phase Magnitude visualization: {e}{Colors.ENDC}")
            print(f"{Colors.WARNING}⚠️ Continuing with next model...{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}ℹ️ Skipping Phase Magnitude latent features visualization (--skip-latent-vis flag set){Colors.ENDC}")
    
    print(f"{Colors.BOLD}▶️ Training Phase Angle model...{Colors.ENDC}")
    pang_model, pang_history, pang_best_model_state = train_model(pang_model, dataloaders['phase_angle']['train'], 
                              dataloaders['phase_angle']['validation'], device, 
                              model_configs['phase_angle'], args.show_batch_progress)
    
    print(f"{Colors.HEADER}📊 Evaluating Phase Angle model...{Colors.ENDC}")
    pang_results = evaluate_model(pang_model, dataloaders['phase_angle']['validation'], device)
    pang_base_filename = save_model(pang_model, pang_history, 'phase_angle', prefix, config)
    save_results(pang_results, 'phase_angle', prefix, config, pang_base_filename)
    
    # Visualize latent features for Phase Angle model
    if not args.skip_latent_vis:
        print(f"{Colors.HEADER}📊 Visualizing Phase Angle latent features...{Colors.ENDC}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Validation data
                visualize_latent_features(pang_model, pang_val_subset, pang_val_labels, 
                                        "Phase Angle (Validation)", os.path.join(vis_dir, "validation"))
                # Training data
                visualize_latent_features(pang_model, pang_train_subset, pang_train_labels, 
                                        "Phase Angle (Training)", os.path.join(vis_dir, "training"))
        except Exception as e:
            print(f"{Colors.WARNING}⚠️ Error during Phase Angle visualization: {e}{Colors.ENDC}")
            print(f"{Colors.WARNING}⚠️ Continuing with next model...{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}ℹ️ Skipping Phase Angle latent features visualization (--skip-latent-vis flag set){Colors.ENDC}")
    
    print(f"{Colors.BOLD}▶️ Training Comodulation model...{Colors.ENDC}")
    comod_model, comod_history, comod_best_model_state = train_model(comod_model, dataloaders['comodulation']['train'], 
                               dataloaders['comodulation']['validation'], device, 
                               model_configs['comodulation'], args.show_batch_progress)
    
    print(f"{Colors.HEADER}📊 Evaluating Comodulation model...{Colors.ENDC}")
    comod_results = evaluate_model(comod_model, dataloaders['comodulation']['validation'], device)
    comod_base_filename = save_model(comod_model, comod_history, 'comodulation', prefix, config)
    save_results(comod_results, 'comodulation', prefix, config, comod_base_filename)
    
    # Visualize latent features for Comodulation model
    if not args.skip_latent_vis:
        print(f"{Colors.HEADER}📊 Visualizing Comodulation latent features...{Colors.ENDC}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Validation data
                visualize_latent_features(comod_model, comod_val_subset, comod_val_labels, 
                                        "Comodulation (Validation)", os.path.join(vis_dir, "validation"))
                # Training data
                visualize_latent_features(comod_model, comod_train_subset, comod_train_labels, 
                                        "Comodulation (Training)", os.path.join(vis_dir, "training"))
        except Exception as e:
            print(f"{Colors.WARNING}⚠️ Error during Comodulation visualization: {e}{Colors.ENDC}")
            print(f"{Colors.WARNING}⚠️ Continuing with final visualizations...{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}ℹ️ Skipping Comodulation latent features visualization (--skip-latent-vis flag set){Colors.ENDC}")
    
    # Generate all visualizations
    print(f"{Colors.HEADER}📈 Generating final training visualizations...{Colors.ENDC}")
    
    # Collect all the base filenames
    filenames = {
        'phase_magnitude': pmag_base_filename,
        'phase_angle': pang_base_filename,
        'comodulation': comod_base_filename
    }
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            visualize_training_run(prefix, config, filenames)
    except Exception as e:
        print(f"{Colors.WARNING}⚠️ Error generating final visualizations: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}⚠️ Some visualizations may be missing.{Colors.ENDC}")
    
    # Print best epochs based on silhouette scores
    print(f"{Colors.HEADER}🔍 Best epochs based on silhouette scores:{Colors.ENDC}")
    
    for model_name, history in [
        ('Phase Magnitude', pmag_history), 
        ('Phase Angle', pang_history), 
        ('Comodulation', comod_history)
    ]:
        if 'silhouette_scores' in history and len(history['silhouette_scores']) > 0:
            # Find best epoch for each method
            best_epochs = {}
            for score_entry in history['silhouette_scores']:
                epoch = score_entry['epoch']
                for method, score in score_entry['scores'].items():
                    if method not in best_epochs or score > best_epochs[method]['score']:
                        best_epochs[method] = {'epoch': epoch, 'score': score}
            
            print(f"  {model_name}:")
            for method, data in best_epochs.items():
                if data['score'] > 0:  # Only show valid scores
                    print(f"    - {method.capitalize()}: Epoch {data['epoch']} (score: {data['score']:.4f})")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Training complete! All models and visualizations saved.{Colors.ENDC}")
    print(f"{Colors.CYAN}💾 Results are in: {config['models_dir']}/{prefix}_*{Colors.ENDC}")
    print(f"{Colors.CYAN}🖼️ Visualizations are in: {vis_dir}{Colors.ENDC}")
    print(f"{Colors.CYAN}📊 Silhouette scores are in: {silhouette_dir}{Colors.ENDC}")


if __name__ == "__main__":
    main()
