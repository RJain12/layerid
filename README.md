# Neural Data Processing

This project is a modular Python implementation of neural data processing code, designed for analyzing neural data stored in MATLAB format.

## Project Structure

- `main.py`: Main script for running neural data processing and model training/evaluation
- `data_loader.py`: Module for loading and converting MATLAB data structures
- `tensor_utils.py`: Utility functions for creating and manipulating neural data tensors
- `utils.py`: Common utility functions used across modules
- `config.json`: Configuration file for model parameters and data paths
- `visualizations.py`: Module for generating training and evaluation visualizations
- `hyperparameter_optimization.py`: Script for optimizing model hyperparameters
- `saved_models/`: Directory containing trained models and training histories
- `datasets/`: Directory containing dataset classes for different types of neural data
  - `base.py`: Base dataset class with common utilities
  - `phase_magnitude.py`: Dataset for phase magnitude features
  - `phase_angle.py`: Dataset for phase angle features
  - `comodulation.py`: Dataset for comodulation features
- `models/`: Directory containing neural network models for layer classification
  - `layer_classifier.py`: Neural network models for layer classification

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- matplotlib
- seaborn
- umap-learn
- optuna (for hyperparameter optimization)
- plotly (optional, for optimization visualizations)

## Usage

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

# Force raw data visualization (regardless of config setting)
python main.py --visualize-raw

# Skip raw data visualization (regardless of config setting)
python main.py --no-visualize-raw

# Show batch progress bars for all epochs, not just the first
python main.py --show-batch-progress

# Skip latent feature visualizations (useful if visualization is causing errors)
python main.py --skip-latent-vis
```

## Hyperparameter Optimization

To optimize hyperparameters for better model performance, use the `hyperparameter_optimization.py` script:

```bash
# Optimize hyperparameters for a specific model type
python hyperparameter_optimization.py --model phase_magnitude --trials 50

# Optimize hyperparameters for all model types
python hyperparameter_optimization.py --model all --trials 30

# Apply the best hyperparameters to a new config file
python hyperparameter_optimization.py --model phase_magnitude --trials 50 --apply
```

This will create optimized configuration files (`config_{model_type}_optimized.json`) that you can use for training:

```bash
python main.py --config config_phase_magnitude_optimized.json
```

### Using Optimized Configs for All Models

To optimize hyperparameters for all model types at once and generate optimized config files, you can use the provided `optimize_all.py` script:

```bash
# Run optimization for all model types (30 trials each)
python optimize_all.py --trials 30

# Run with custom number of trials
python optimize_all.py --trials 50
```

This script will generate optimized config files for each model type (`config_phase_magnitude_optimized.json`, `config_phase_angle_optimized.json`, and `config_comodulation_optimized.json`).

To train using these optimized configs:

```bash
python main.py --use-optimized
```

This will automatically use the optimized config files for each model type if available, falling back to the default config for any model types that don't have optimized configs.

The hyperparameter optimization includes:
- Learning rate
- Batch size
- Latent space dimension
- Classification loss weight (lambda_cls)
- Dropout rate
- Weight decay

Optimization results and visualizations are saved in the `hyperparameter_results/` directory.

## Data Format

The code expects a MATLAB file named `parsedData.mat` that contains neural data in a specific format. The data should include:

- Phase magnitude data (LFPphasemagnitude, LFGphasemagnitude, CSDphasemagnitude)
- Phase angle data (LFPphaseangle, LFGphaseangle, CSDphaseangle)
- Comodulation data (lfpLFPcoModCorrs, lfpLFGcoModCorrs, lfpCSDcoModCorrs)
- Channel y-coordinates (ycoords)
- Layer borders (dredge)

## Configuration

The project uses a `config.json` file for configuration. The following parameters can be adjusted:

- `data_path`: Path to the MATLAB data file (default: 'parsedData.mat')
- `batch_size`: Batch size for training (default: 32)
- `num_epochs`: Number of training epochs (default: 50)
- `learning_rate`: Learning rate for model training (default: 0.001)
- `num_classes`: Number of output classes (default: 8)
- `pad_value`: Value used for padding tensors (default: 99.0)
- `models_dir`: Directory for saving trained models (default: 'saved_models')
- `visualize_raw_data`: Whether to generate visualizations of raw data (default: false)

## Saved Models

After training, the following files are saved in the `saved_models` directory:

- `{prefix}_{model_type}_model.pt`: PyTorch model state dictionary
- `{prefix}_{model_type}_history.json`: Training history (loss and accuracy)
- `{prefix}_{model_type}_results.json`: Evaluation results (confusion matrix, classification report)

Where `{prefix}` is the user-provided prefix and `{model_type}` is one of:
- `phase_magnitude`
- `phase_angle`
- `comodulation`

## Visualizations

The project automatically generates visualizations for each training run in a subdirectory named `{prefix}_visualizations` within the `saved_models` directory. The following visualizations are created:

### Training Visualizations
- `{prefix}_{model_type}_history.png`: Training and validation loss/accuracy over epochs
- `{prefix}_{model_type}_confusion.png`: Confusion matrix heatmap
- `{prefix}_model_comparison.png`: Bar chart comparing accuracy across different models

### Latent Feature Visualizations
For each model, the latent features are extracted and visualized using dimensionality reduction techniques. These are saved in separate directories for training and validation data to facilitate comparison:

#### Training Data (7 patients)
Located in `{prefix}_visualizations/training/`:
- `{model_type}_latent_pca.png`: PCA visualization of the model's latent features for training data
- `{model_type}_latent_umap.png`: UMAP visualization of the model's latent features for training data
- `{model_type}_latent_tsne.png`: t-SNE visualization of the model's latent features for training data

#### Validation Data (1 patient)
Located in `{prefix}_visualizations/validation/`:
- `{model_type}_latent_pca.png`: PCA visualization of the model's latent features for validation data
- `{model_type}_latent_umap.png`: UMAP visualization of the model's latent features for validation data
- `{model_type}_latent_tsne.png`: t-SNE visualization of the model's latent features for validation data

This separation allows for comparison between how the model represents data from the training set versus the validation set, which can help identify issues like overfitting.

### Raw Data Visualizations
The raw data is also visualized using the same dimensionality reduction techniques:
- `phase_magnitude_pca.png`, `phase_magnitude_umap.png`, `phase_magnitude_tsne.png`: Visualizations of phase magnitude data
- `phase_angle_pca.png`, `phase_angle_umap.png`, `phase_angle_tsne.png`: Visualizations of phase angle data
- `comodulation_pca.png`, `comodulation_umap.png`, `comodulation_tsne.png`: Visualizations of comodulation data

These visualizations help in understanding the data distribution, model performance, and comparing different training runs. They also allow for the analysis of how well the model separates different cortical layers in both the raw data and latent feature spaces.

### Training Optimization

The training process includes the following optimizations:

- Validation is performed only every 10 epochs and on the final epoch to speed up training
- Progress bars show key metrics (training loss, training accuracy, validation loss, validation accuracy)
- Batch-level progress bars show training progress (can be enabled for all epochs with `--show-batch-progress`)
- A checkmark (✓) indicator shows when validation was actually performed