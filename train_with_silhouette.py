"""
Training script with silhouette score tracking for neural data processor models.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

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
    visualize_training_run
)
from silhouette_tracking import track_silhouette_scores, plot_silhouette_scores


def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_checkpoint(state, filename):
    """Save a checkpoint of the model."""
    torch.save(state, filename)


def create_model_and_dataset(model_type, config, parsed_data_dict, all_tensors, paddedC):
    """Create a model and dataset based on the model type."""
    if model_type == 'phase_magnitude':
        dataset = ChannelLevelPhaseMagnitudeDataset(all_tensors, parsed_data_dict)
        model = PhaseMagnitudeLayerClassifier(
            num_classes=config['num_classes'], 
            latent_dim=config['latent_dim'],
            maxC=paddedC
        )
    elif model_type == 'phase_angle':
        dataset = ChannelLevelPhaseAngleDataset(all_tensors, parsed_data_dict)
        model = PhaseAngleLayerClassifier(
            num_classes=config['num_classes'], 
            latent_dim=config['latent_dim'],
            maxC=paddedC
        )
    elif model_type == 'comodulation':
        dataset = ChannelLevelComodDataset(all_tensors, parsed_data_dict)
        model = ComodLayerClassifier(
            num_classes=config['num_classes'], 
            latent_dim=config['latent_dim'],
            maxC=paddedC
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, dataset


def train_model(model, train_loader, val_loader, config, optimizer, device, save_dir, model_type):
    """Train the model with silhouette score tracking."""
    # Set up loss functions and training parameters
    cls_criterion = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss(reduction='none')
    lambda_cls = config['lambda_cls']
    num_epochs = config['num_epochs']
    
    # Setup for tracking metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Path for silhouette scores
    silhouette_path = os.path.join(save_dir, f"{model_type}_silhouette_scores.json")
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, masks, labels in tqdm(train_loader, desc="Training"):
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
            
            running_loss += loss.item()
            
            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, masks, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                
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
                
                val_loss += loss.item()
                
                # Compute predictions
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # Store for confusion matrix
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Calculate silhouette scores
        if (epoch+1) % config.get('silhouette_freq', 50) == 0 or epoch == 0 or epoch == num_epochs-1:
            print(f"Calculating silhouette scores for epoch {epoch+1}...")
            silhouette_scores = track_silhouette_scores(
                model=model,
                dataloader=val_loader,
                device=device,
                epoch=epoch+1,
                results_path=silhouette_path
            )
            print(f"Silhouette scores: {silhouette_scores}")
        
        # Save checkpoint
        if (epoch+1) % config.get('save_freq', 50) == 0 or epoch == num_epochs-1:
            checkpoint_path = os.path.join(save_dir, f"{model_type}_epoch_{epoch+1}.pt")
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': {
                    'train_loss': train_losses,
                    'val_loss': val_losses,
                    'train_acc': train_accs,
                    'val_acc': val_accs,
                }
            }, checkpoint_path)
            
            # Save confusion matrix at this checkpoint
            cm = confusion_matrix(all_labels, all_preds)
            visualize_dir = os.path.join(save_dir, "visualizations", f"epoch_{epoch+1}")
            os.makedirs(visualize_dir, exist_ok=True)
            
            # Generate visualizations for this checkpoint
            visualize_training_run(
                model=model,
                val_loader=val_loader,
                device=device,
                save_path=visualize_dir,
                epoch=epoch+1,
                model_type=model_type
            )
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
    
    # Plot silhouette scores at the end
    silhouette_plot_path = os.path.join(save_dir, f"{model_type}_silhouette_scores.png")
    best_epochs = plot_silhouette_scores(silhouette_path, silhouette_plot_path)
    
    # Save final history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'best_epoch': best_epoch,
        'best_silhouette_epochs': best_epochs
    }
    
    history_path = os.path.join(save_dir, f"{model_type}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Also plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.axvline(x=best_epoch-1, color='gray', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.axvline(x=best_epoch-1, color='gray', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_type}_training_curves.png"))
    plt.close()
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train a neural data processor model with silhouette score tracking")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['phase_magnitude', 'phase_angle', 'comodulation', 'all'],
                        help='Model type to train')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='saved_models_silhouette',
                        help='Directory to save models and results')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else
                         "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    parsed_data_dict = load_mat_data(config['data_path'])
    all_tensors, paddedC = build_all_patient_tensors(parsed_data_dict, pad_value=config['pad_value'])
    
    # Determine which models to train
    models_to_train = ['phase_magnitude', 'phase_angle', 'comodulation'] if args.model == 'all' else [args.model]
    
    # Train each model
    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_type} model")
        print(f"{'='*50}")
        
        # Create model and dataset
        model, dataset = create_model_and_dataset(model_type, config, parsed_data_dict, all_tensors, paddedC)
        model.to(device)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False
        )
        
        # Set up optimizer
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )
        
        # Train the model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader, 
            config=config,
            optimizer=optimizer,
            device=device,
            save_dir=args.save_dir,
            model_type=model_type
        )
        
        print(f"\nTraining completed for {model_type} model")
        print(f"Best validation accuracy: {max(history['val_acc']):.4f} at epoch {history['best_epoch']}")
        print(f"Best silhouette score epochs: {history['best_silhouette_epochs']}")


if __name__ == "__main__":
    main() 