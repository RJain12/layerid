import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from data_loader import load_mat_data
from tensor_utils import build_all_patient_tensors
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
from utils import get_channel_layer
from visualizations import visualize_training_run

def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_device(config):
    """Set up device (CPU/GPU) based on configuration."""
    if config['gpu_settings']['use_gpu'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['gpu_settings']['gpu_id']}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def create_dataloader(dataset, config):
    """Create DataLoader with GPU-optimized settings if available."""
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['gpu_settings']['num_workers'] if config['gpu_settings']['use_gpu'] else 0,
        pin_memory=config['gpu_settings']['pin_memory'] if config['gpu_settings']['use_gpu'] else False
    )

def train_model(model, train_loader, val_loader, config, device, model_type):
    """Train the model with GPU support."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        for inputs, masks, labels in train_loader_tqdm:
            # Move data to device
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            reconstructions, latent, predictions = model(inputs, masks)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_loader_tqdm.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(train_correct/train_total):.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]')
            for inputs, masks, labels in val_loader_tqdm:
                # Move data to device
                inputs = inputs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                
                reconstructions, latent, predictions = model(inputs, masks)
                loss = criterion(predictions, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_loader_tqdm.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(val_correct/val_total):.4f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': val_accuracy
            }
        
        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    return model, history, best_model_state

def main():
    parser = argparse.ArgumentParser(description='Train neural layer classifier')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--model-type', type=str, choices=['phase_magnitude', 'phase_angle', 'comodulation'],
                      default='phase_magnitude', help='Type of model to train')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up device
    device = setup_device(config)
    
    # Load and prepare data
    print("Loading data...")
    parsed_data_dict = load_mat_data(config['data_path'])
    print(f"Loaded data for {len(parsed_data_dict)} patients")
    
    # Build tensors
    print("Building tensors...")
    all_tensors, paddedC = build_all_patient_tensors(parsed_data_dict, pad_value=config['pad_value'])
    
    # Create appropriate dataset based on model_type
    if args.model_type == 'phase_magnitude':
        dataset = ChannelLevelPhaseMagnitudeDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
        model = PhaseMagnitudeLayerClassifier(num_classes=config['num_classes'], latent_dim=config['latent_dim'], maxC=paddedC)
    elif args.model_type == 'phase_angle':
        dataset = ChannelLevelPhaseAngleDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
        model = PhaseAngleLayerClassifier(num_classes=config['num_classes'], latent_dim=config['latent_dim'], maxC=paddedC)
    elif args.model_type == 'comodulation':
        dataset = ChannelLevelComodDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
        model = ComodLayerClassifier(num_classes=config['num_classes'], latent_dim=config['latent_dim'], maxC=paddedC)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = create_dataloader(train_dataset, config)
    val_loader = create_dataloader(val_dataset, config)
    
    # Train model
    print(f"\nTraining {args.model_type} model...")
    trained_model, history, best_model_state = train_model(model, train_loader, val_loader, config, device, args.model_type)
    
    # Save model and history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{config['saved_models_path']}{timestamp}_{args.model_type}_model.pt"
    history_filename = f"{config['saved_models_path']}{timestamp}_{args.model_type}_history.npz"
    
    # Save best model
    torch.save(best_model_state, model_filename)
    print(f"\nSaved best model to {model_filename}")
    
    # Save training history
    np.savez(history_filename, **history)
    print(f"Saved training history to {history_filename}")
    
    # Visualize training results
    print("\nGenerating training visualizations...")
    visualize_training_run(timestamp, config, {
        args.model_type: f"{timestamp}_{args.model_type}"
    })

if __name__ == "__main__":
    main() 