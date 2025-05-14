import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

from models import (
    PhaseMagnitudeLayerClassifier,
    PhaseAngleLayerClassifier,
    ComodLayerClassifier
)
from data_loader import load_mat_data
from tensor_utils import build_all_patient_tensors
from datasets import (
    ChannelLevelPhaseMagnitudeDataset,
    ChannelLevelPhaseAngleDataset,
    ChannelLevelComodDataset
)

def extract_latent_features(model, dataloader, device):
    """Extract latent features from a model for all samples in a dataloader."""
    model.eval()
    features = []
    labels = []
    patient_indices = []
    
    with torch.no_grad():
        for inputs, masks, layer_label in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            # Get latent features
            _, z, _ = model(inputs, masks)
            features.append(z.cpu().numpy())
            labels.extend(layer_label.numpy())
            
            # Get patient indices from the masks
            # The mask will be all zeros for padding in other patients
            patient_idx = torch.where(masks.sum(dim=(1,2,3)) > 0)[0][0].item()
            patient_indices.extend([patient_idx] * inputs.shape[0])
    
    return np.vstack(features), np.array(labels), np.array(patient_indices)

def main():
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("Loading data...")
    parsed_data_dict = load_mat_data(config['data_path'])
    all_tensors, paddedC = build_all_patient_tensors(parsed_data_dict, pad_value=config['pad_value'])
    
    # Create datasets
    pmag_dataset = ChannelLevelPhaseMagnitudeDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    pang_dataset = ChannelLevelPhaseAngleDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    comod_dataset = ChannelLevelComodDataset(all_tensors, parsed_data_dict, pad_value=config['pad_value'])
    
    # Create dataloaders
    batch_size = 32
    pmag_loader = DataLoader(pmag_dataset, batch_size=batch_size, shuffle=False)
    pang_loader = DataLoader(pang_dataset, batch_size=batch_size, shuffle=False)
    comod_loader = DataLoader(comod_dataset, batch_size=batch_size, shuffle=False)
    
    # Load trained models
    print("Loading trained models...")
    
    # Phase Magnitude model
    pmag_model = PhaseMagnitudeLayerClassifier(
        num_classes=config['num_classes'],
        latent_dim=128,
        maxC=paddedC
    ).to(device)
    pmag_checkpoint = torch.load('saved_models/1000r_phase_magnitude_model.pt', map_location=device)
    pmag_model.load_state_dict(pmag_checkpoint['model_state_dict'])
    
    # Phase Angle model
    pang_model = PhaseAngleLayerClassifier(
        num_classes=config['num_classes'],
        latent_dim=128,
        maxC=paddedC
    ).to(device)
    pang_checkpoint = torch.load('saved_models/1000r_phase_angle_model.pt', map_location=device)
    pang_model.load_state_dict(pang_checkpoint['model_state_dict'])
    
    # Comodulation model
    comod_model = ComodLayerClassifier(
        num_classes=config['num_classes'],
        latent_dim=512,
        maxC=paddedC
    ).to(device)
    comod_checkpoint = torch.load('saved_models/1000r_comodulation_model.pt', map_location=device)
    comod_model.load_state_dict(comod_checkpoint['model_state_dict'])
    
    # Extract latent features
    print("Extracting latent features...")
    pmag_features, pmag_labels, pmag_patients = extract_latent_features(pmag_model, pmag_loader, device)
    pang_features, pang_labels, pang_patients = extract_latent_features(pang_model, pang_loader, device)
    comod_features, comod_labels, comod_patients = extract_latent_features(comod_model, comod_loader, device)
    
    # Combine features
    combined_features = np.hstack([pmag_features, pang_features, comod_features])
    labels = pmag_labels  # All labels should be the same
    patient_indices = pmag_patients  # All patient indices should be the same
    
    # Split into train and validation sets
    # Use the last patient as validation, just like in the original training
    val_patient_idx = len(parsed_data_dict) - 1
    train_mask = patient_indices != val_patient_idx
    val_mask = patient_indices == val_patient_idx
    
    X_train = combined_features[train_mask]
    y_train = labels[train_mask]
    X_val = combined_features[val_mask]
    y_val = labels[val_mask]
    
    # Train GLM
    print("Training GLM...")
    glm = LogisticRegression(multi_class='multinomial', max_iter=1000)
    glm.fit(X_train, y_train)
    
    # Make predictions on validation set
    val_predictions = glm.predict(X_val)
    
    # Print validation results
    print("\nValidation Results:")
    print("Accuracy:", accuracy_score(y_val, val_predictions))
    print("\nClassification Report:")
    print(classification_report(y_val, val_predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, val_predictions))
    
    # Save the GLM model
    import joblib
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(glm, 'saved_models/combined_glm.joblib')
    print("\nSaved GLM model to saved_models/combined_glm.joblib")
    
    # Save validation results
    results = {
        'accuracy': float(accuracy_score(y_val, val_predictions)),
        'classification_report': classification_report(y_val, val_predictions, output_dict=True),
        'confusion_matrix': confusion_matrix(y_val, val_predictions).tolist()
    }
    
    with open('saved_models/glm_validation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved validation results to saved_models/glm_validation_results.json")

if __name__ == '__main__':
    main() 