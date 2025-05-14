import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader import load_mat_data
from tensor_utils import build_all_patient_tensors
from datasets.phase_magnitude import ChannelLevelPhaseMagnitudeDataset
from datasets.phase_angle import ChannelLevelPhaseAngleDataset
from datasets.comodulation import ChannelLevelComodDataset
from models import (
    PhaseMagnitudeLayerClassifier,
    PhaseAngleLayerClassifier,
    ComodLayerClassifier
)

def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def generate_confusion_matrix(model_path, model, dataset, device):
    # Load the model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Get all predictions and true labels
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for data, mask, labels in dataloader:
            data = data.to(device)
            mask = mask.to(device)
            
            _, _, logits = model(data, mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return cm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    parsed_data_dict = load_mat_data('parsedData.mat')
    all_tensors, paddedC = build_all_patient_tensors(parsed_data_dict)
    
    # Model configurations
    configs = [
        {
            'name': 'phase_magnitude',
            'epoch': 400,
            'dataset_class': ChannelLevelPhaseMagnitudeDataset,
            'model_class': PhaseMagnitudeLayerClassifier,
            'model_params': {'num_classes': 8, 'latent_dim': 128}
        },
        {
            'name': 'phase_angle',
            'epoch': 400,
            'dataset_class': ChannelLevelPhaseAngleDataset,
            'model_class': PhaseAngleLayerClassifier,
            'model_params': {'num_classes': 8, 'latent_dim': 128}
        },
        {
            'name': 'comodulation',
            'epoch': 200,
            'dataset_class': ChannelLevelComodDataset,
            'model_class': ComodLayerClassifier,
            'model_params': {'num_classes': 8, 'latent_dim': 512}
        }
    ]
    
    for config in configs:
        print(f"\nProcessing {config['name']} model...")
        
        # Create dataset
        dataset = config['dataset_class'](all_tensors, parsed_data_dict)
        
        # Create and load model
        model = config['model_class'](
            maxC=paddedC,
            **config['model_params']
        ).to(device)
        
        model_path = f"saved_models/{config['name']}_epoch_{config['epoch']}.pt"
        print(f"Loading model from {model_path}")
        
        # Generate confusion matrix
        cm = generate_confusion_matrix(model_path, model, dataset, device)
        
        # Plot and save
        save_path = f"saved_models/visualizations/epoch_{config['epoch']}/train/{config['name']}_confusion_matrix.png"
        plot_confusion_matrix(cm, f"{config['name']} (Epoch {config['epoch']})", save_path)
        
        print(f"Generated confusion matrix for {config['name']} at epoch {config['epoch']}")
        print(f"Saved to {save_path}")

if __name__ == '__main__':
    main() 