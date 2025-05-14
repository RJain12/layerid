"""
Hyperparameter optimization for the neural data processor models.

This script uses Optuna to perform efficient hyperparameter search for the models.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import Trial
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Import from our modules
from data_loader import load_mat_data
from tensor_utils import build_all_patient_tensors
from utils import get_channel_layer
from datasets.phase_magnitude import ChannelLevelPhaseMagnitudeDataset
from datasets.phase_angle import ChannelLevelPhaseAngleDataset
from datasets.comodulation import ChannelLevelComodDataset
from models import (
    PhaseMagnitudeLayerClassifier,
    PhaseAngleLayerClassifier,
    ComodLayerClassifier
)
from visualizations import (
    visualize_training_run,
    plot_model_comparison
)


def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_dataloaders(config, model_type, train_ratio=0.8, batch_size=None):
    """
    Create train and validation dataloaders for a specific model type.
    
    Args:
        config (dict): Configuration dictionary
        model_type (str): One of 'phase_magnitude', 'phase_angle', 'comodulation'
        train_ratio (float): Ratio of data to use for training
        batch_size (int, optional): Batch size to override config
        
    Returns:
        tuple: (train_loader, val_loader, train_subset, val_subset)
    """
    # Load data
    parsed_data_dict = load_mat_data(config['data_path'])
    
    # Build tensors (same as in main.py)
    all_tensors, paddedC = build_all_patient_tensors(parsed_data_dict, pad_value=config['pad_value'])
    
    # Create appropriate dataset
    if model_type == 'phase_magnitude':
        dataset = ChannelLevelPhaseMagnitudeDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    elif model_type == 'phase_angle':
        dataset = ChannelLevelPhaseAngleDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    elif model_type == 'comodulation':
        dataset = ChannelLevelComodDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Split into train and validation
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    batch_size = batch_size or config['batch_size']
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_subset, val_subset, paddedC


def objective(trial: Trial, model_type, config, device, num_epochs=None):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial (Trial): Optuna trial object
        model_type (str): Type of model to optimize ('phase_magnitude', 'phase_angle', 'comodulation')
        config (dict): Base configuration
        device (torch.device): Device to train on
        num_epochs (int, optional): Number of epochs to override config
        
    Returns:
        float: Validation accuracy (to be maximized)
    """
    # Define hyperparameters to optimize with expanded ranges
    lr = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128, 256, 512])
    lambda_cls = trial.suggest_float('lambda_cls', 0.01, 100.0, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.7)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
    
    # Get dataloaders with suggested batch size
    train_loader, val_loader, _, _, paddedC = get_dataloaders(
        config, model_type, train_ratio=0.8, batch_size=batch_size
    )
    
    # Create appropriate model
    if model_type == 'phase_magnitude':
        model = PhaseMagnitudeLayerClassifier(
            num_classes=config['num_classes'],
            latent_dim=latent_dim,
            maxC=paddedC
        )
    elif model_type == 'phase_angle':
        model = PhaseAngleLayerClassifier(
            num_classes=config['num_classes'],
            latent_dim=latent_dim,
            maxC=paddedC
        )
    elif model_type == 'comodulation':
        model = ComodLayerClassifier(
            num_classes=config['num_classes'],
            latent_dim=latent_dim,
            maxC=paddedC
        )
    
    # Custom Dropout - add dropout to the model
    # We'll modify the forward method to apply dropout to the latent space
    original_forward = model.forward
    
    def forward_with_dropout(x, mask=None):
        recon, z, logits = original_forward(x, mask)
        if model.training:
            z = torch.nn.functional.dropout(z, p=dropout_rate, training=True)
            # Recompute logits with dropout-applied features
            logits = model.classifier(z)
        return recon, z, logits
    
    model.forward = forward_with_dropout
    
    # Move model to device
    model.to(device)
    
    # Set up training
    cls_criterion = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Training loop with more epochs
    num_epochs = 100  # Fixed number of epochs for thorough training
    best_val_acc = 0.0
    
    # Print trial hyperparameters
    print(f"\n\nTrial {trial.number+1}:")
    print(f"  Learning rate: {lr:.6f}")
    print(f"  Batch size: {batch_size}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Lambda cls: {lambda_cls:.4f}")
    print(f"  Dropout rate: {dropout_rate:.4f}")
    print(f"  Weight decay: {weight_decay:.8f}")
    print("-" * 50)
    
    # Use tqdm for progress tracking
    for epoch in tqdm(range(num_epochs), desc=f"Training {model_type}", unit="epoch"):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for batch progress in first epoch
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                         leave=False, unit="batch") if epoch == 0 else train_loader
        
        for inputs, masks, labels in batch_iter:
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon, z, logits = model(inputs, masks)
            
            # Reconstruction loss (masked)
            diff = recon_criterion(recon, inputs)
            masked_diff = diff * masks
            loss_recon = masked_diff.sum() / masks.sum() if masks.sum() > 0 else 0.0
            
            # Classification loss
            loss_cls = cls_criterion(logits, labels)
            
            # Combined loss
            loss = loss_recon + lambda_cls * loss_cls
            
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total if total > 0 else 0.0
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, masks, labels in val_loader:
                inputs = inputs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                
                # Forward pass
                _, _, logits = model(inputs, masks)
                
                # Compute predictions
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Print epoch results every 10 epochs or at the end
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1}/{num_epochs} - Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
        
        # Report intermediate values to Optuna
        trial.report(val_acc, epoch)
        
        # Early stopping if validation accuracy hasn't improved in 20 epochs
        if trial.should_prune():
            print(f"  Trial pruned at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()
        
        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc


def optimize_hyperparameters(model_type, n_trials=50, study_name=None, config_path='config.json'):
    """
    Run hyperparameter optimization for a specific model type.
    
    Args:
        model_type (str): Type of model to optimize ('phase_magnitude', 'phase_angle', 'comodulation')
        n_trials (int): Number of optimization trials to run
        study_name (str, optional): Name for the Optuna study
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Best hyperparameters
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else
                         "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create study name if not provided
    if study_name is None:
        study_name = f"{model_type}_optimization"
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # We want to maximize validation accuracy
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Start optimization
    print(f"\n{'='*50}")
    print(f"Starting optimization for {model_type} model with {n_trials} trials")
    print(f"{'='*50}")
    
    study.optimize(
        lambda trial: objective(trial, model_type, config, device),
        n_trials=n_trials
    )
    
    # Print optimization summary
    print("\nOptimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Visualize optimization results
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(f"param_importances_{model_type}.png")
        
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(f"optimization_history_{model_type}.png")
        
        for param in study.best_params:
            fig = optuna.visualization.plot_slice(study, params=[param])
            fig.write_image(f"slice_{param}_{model_type}.png")
    except:
        print("Warning: Could not generate visualization plots (requires plotly)")
    
    # Save best hyperparameters to file
    os.makedirs('hyperparameter_results', exist_ok=True)
    best_params = study.best_params
    best_params['val_accuracy'] = study.best_value
    
    with open(f'hyperparameter_results/{model_type}_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    return best_params


def apply_best_hyperparameters(model_type, best_params, output_config_path=None, config_path='config.json'):
    """
    Apply best hyperparameters to a new configuration file.
    
    Args:
        model_type (str): Type of model to optimize ('phase_magnitude', 'phase_angle', 'comodulation')
        best_params (dict): Best hyperparameters from optimization
        output_config_path (str, optional): Path to save the new configuration
        config_path (str): Path to the base configuration file
        
    Returns:
        dict: Updated configuration
    """
    # Load base configuration
    config = load_config(config_path)
    
    # Apply hyperparameters
    config['learning_rate'] = best_params['learning_rate']
    config['batch_size'] = best_params['batch_size']
    config['latent_dim'] = best_params['latent_dim']
    config['lambda_cls'] = best_params['lambda_cls']
    
    # Make sure visualize_raw_data setting is preserved or defaulted to false
    if 'visualize_raw_data' not in config:
        config['visualize_raw_data'] = False
    
    # Additional parameters that may need special handling
    config['optimizer_params'] = {
        'weight_decay': best_params['weight_decay']
    }
    config['model_params'] = {
        'dropout_rate': best_params['dropout_rate']
    }
    
    # Save updated configuration
    if output_config_path is None:
        output_config_path = f'config_{model_type}_optimized.json'
    
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Updated configuration saved to {output_config_path}")
    
    return config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for neural data processor')
    parser.add_argument('--model', type=str, required=True, choices=['phase_magnitude', 'phase_angle', 'comodulation', 'all'],
                        help='Model type to optimize')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--apply', action='store_true', help='Apply best hyperparameters to config')
    args = parser.parse_args()
    
    # Optimize hyperparameters
    if args.model == 'all':
        # Optimize all model types
        model_types = ['phase_magnitude', 'phase_angle', 'comodulation']
        best_params = {}
        
        for i, model_type in enumerate(model_types):
            print(f"\n\n{'='*50}")
            print(f"Optimizing {model_type} model ({i+1}/{len(model_types)})")
            print(f"{'='*50}")
            
            best_params[model_type] = optimize_hyperparameters(
                model_type=model_type,
                n_trials=args.trials,
                config_path=args.config
            )
            
            if args.apply:
                apply_best_hyperparameters(
                    model_type=model_type,
                    best_params=best_params[model_type],
                    output_config_path=f'config_{model_type}_optimized.json',
                    config_path=args.config
                )
    else:
        # Optimize specific model type
        best_params = optimize_hyperparameters(
            model_type=args.model,
            n_trials=args.trials,
            config_path=args.config
        )
        
        if args.apply:
            apply_best_hyperparameters(
                model_type=args.model,
                best_params=best_params,
                output_config_path=f'config_{args.model}_optimized.json',
                config_path=args.config
            )


if __name__ == "__main__":
    main() 