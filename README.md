# Neural Data Processing for Cortical Layer Identification

This project is a modular Python implementation of neural data processing for identifying cortical layers from neural recordings. It analyzes neural data stored in MATLAB format, extracts features, trains autoencoder models, and visualizes the results.

## Overview

The LayerID project provides a framework for processing neural data and identifying cortical layers using deep learning methods. The system takes neural recordings as input and processes them through three different feature types:

1. **Phase Magnitude** - Magnitude of phase synchrony between channels
2. **Phase Angle** - Angular differences in phase synchrony
3. **Comodulation** - Cross-frequency coupling between different frequency bands

For each of these feature types, the framework trains a specialized autoencoder model with a classification head that can identify the cortical layer of each recording channel.

## Codebase Structure

The codebase is organized into several modules, each with a specific purpose:

### Core Modules

- `main.py`: Entry point and orchestration of the entire pipeline
- `data_loader.py`: Functions for loading and parsing MATLAB data files
- `tensor_utils.py`: Utilities for creating and manipulating neural data tensors
- `utils.py`: Common utility functions used across modules

### Model Framework

- `models/`: Neural network model definitions
  - `layer_classifier.py`: Defines the autoencoder models with classification heads
  
### Dataset Handlers

- `datasets/`: Dataset classes for different neural feature types
  - `base.py`: Base dataset class with common functionality
  - `phase_magnitude.py`: Dataset for phase magnitude features
  - `phase_angle.py`: Dataset for phase angle features
  - `comodulation.py`: Dataset for comodulation features

### Visualization Tools

- `visualizations.py`: Functions for generating training and evaluation visualizations
- `visualize_coherence_matrices.py`: Tools for visualizing coherence matrices
- `visualize_data_and_reconstructions.py`: Tools for visualizing raw data and model reconstructions
- `visualize_patient_reconstructions.py`: Tools for visualizing patient-specific reconstructions

### Model Evaluation

- `silhouette_tracking.py`: Functions for computing silhouette scores and tracking clustering quality
- `generate_confusion_matrices.py`: Generate confusion matrices for model evaluation

### Hyperparameter Optimization

- `hyperparameter_optimization.py`: Optimize model hyperparameters using Optuna
- `optimize_all.py`: Script to optimize all model types
- `train.py`: Simplified training script for optimization

### Distributed Computing

- `run_on_runpod.sh`: Script for running the code on RunPod compute platform
- `runpod_hyperopt.py`: RunPod-specific hyperparameter optimization
- `runpod_train_silhouette.py`: RunPod-specific silhouette training

## Data Flow

The data flow through the system follows these steps:

1. **Data Loading**: MATLAB data is loaded using `data_loader.py`
2. **Tensor Construction**: Raw data is organized into tensors using `tensor_utils.py`
3. **Dataset Creation**: Tensors are processed into channel-level datasets
4. **Model Training**: Autoencoder models are trained on these datasets
5. **Evaluation**: Models are evaluated using classification metrics and clustering quality
6. **Visualization**: Results are visualized for interpretation

## Model Architecture

The core architecture uses supervised autoencoders with a classification head:

```
Input Data -> Convolutional Encoder -> Latent Space -> Convolutional Decoder -> Reconstructed Data
                                     |
                                     v
                                Classification Head
                                     v
                                Layer Prediction
```

- **Encoder**: Convolutional neural network that compresses input data into a latent representation
- **Decoder**: Deconvolutional network that reconstructs the input from the latent representation
- **Classification Head**: Fully connected layer that predicts layer identity from latent features

The model is trained with a combined loss function:

```
Total Loss = Reconstruction Loss + KL Regularization + Classification Loss
```

## Configuration System

The project uses a flexible configuration system with JSON files:

- `config.json`: Default configuration for all models
- `config_{feature_type}_optimized.json`: Optimized configurations for each feature type

Key configuration parameters include:
- Learning rate
- Batch size
- Latent dimension
- Number of epochs
- Classification loss weight
- Dropout rate
- Weight decay

## Usage Instructions

### Basic Usage

1. Place your `parsedData.mat` file in the root directory
2. (Optional) Modify `config.json` to adjust model parameters
3. Run the main script:

```bash
python main.py
```

4. When prompted, enter a prefix for the saved model files (e.g., '10epochs_test1')

### Command-Line Arguments

The main script supports several command-line arguments:

```bash
# Use a different configuration file
python main.py --config custom_config.json

# Use optimized configs for each model type (if available)
python main.py --use-optimized

# Force raw data visualization
python main.py --visualize-raw

# Skip raw data visualization
python main.py --no-visualize-raw

# Show batch progress for all epochs
python main.py --show-batch-progress

# Skip latent feature visualizations
python main.py --skip-latent-vis

# Set frequency for calculating silhouette scores
python main.py --silhouette-freq 50
```

### Hyperparameter Optimization

To optimize model hyperparameters:

```bash
# Optimize for a specific model type
python hyperparameter_optimization.py --model phase_magnitude --trials 50

# Optimize for all model types
python hyperparameter_optimization.py --model all --trials 30

# Apply best parameters to config files
python hyperparameter_optimization.py --model phase_magnitude --trials 50 --apply
```

To optimize all models and generate optimized config files:

```bash
python optimize_all.py --trials 30
```

Then train using the optimized configs:

```bash
python main.py --use-optimized
```

## Neural Data Format

The code expects a MATLAB file named `parsedData.mat` that contains:

- **Phase Magnitude Data**:
  - `LFPphasemagnitude`: Phase magnitude for Local Field Potentials 
  - `LFGphasemagnitude`: Phase magnitude for Local Field Gradients
  - `CSDphasemagnitude`: Phase magnitude for Current Source Density

- **Phase Angle Data**:
  - `LFPphaseangle`: Phase angles for Local Field Potentials 
  - `LFGphaseangle`: Phase angles for Local Field Gradients
  - `CSDphaseangle`: Phase angles for Current Source Density

- **Comodulation Data**:
  - `lfpLFPcoModCorrs`: Comodulation correlations between LFP frequencies
  - `lfpLFGcoModCorrs`: Comodulation correlations between LFP and LFG
  - `lfpCSDcoModCorrs`: Comodulation correlations between LFP and CSD

- **Recording Metadata**:
  - `ycoords`: Y-coordinates for each recording channel
  - `DREDgeLayerBorders`: Borders between cortical layers

## Output and Results

After training, the system produces:

### Model Files
- `{prefix}_{model_type}_model.pt`: Trained PyTorch model
- `{prefix}_{model_type}_history.json`: Training metrics history
- `{prefix}_{model_type}_results.json`: Evaluation results

### Visualizations
- Training and validation loss/accuracy curves
- Confusion matrices for layer classification
- Model comparison charts
- Latent space visualizations using PCA, t-SNE, and UMAP
- Silhouette score and Calinski-Harabasz index plots for clustering quality

### Evaluation Metrics
- Classification accuracy
- Confusion matrix
- Silhouette scores (for clustering quality)
- Calinski-Harabasz index (for clustering quality)

## Cluster Quality Metrics

The system tracks two important metrics to evaluate how well the model separates different cortical layers:

1. **Silhouette Score**: Measures how similar objects are to their own cluster compared to other clusters
2. **Calinski-Harabasz Index**: Ratio of between-cluster dispersion to within-cluster dispersion

These metrics are calculated for different dimensionality reduction techniques:
- Direct on latent features
- After PCA reduction
- After t-SNE reduction
- After UMAP reduction

## Training Optimizations

The training process includes several optimizations:

- Batch-level progress tracking
- Silhouette score calculation at configurable intervals
- Model checkpointing every 50 epochs
- Early stopping with patience
- Learning rate scheduling
- Balancing of reconstruction, KL divergence, and classification losses

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.