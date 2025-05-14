#!/usr/bin/env python3
"""
Training script for RunPod with silhouette score tracking.
"""

import os
import sys
import torch
import argparse
import time
import json
from train_with_silhouette import main as train_main

def setup_config_for_silhouette(config_path, output_path):
    """
    Modifies the config file to add silhouette tracking parameters
    """
    # Load existing config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add/update silhouette parameters
    config['silhouette_freq'] = 50  # Calculate silhouette score every 50 epochs
    config['save_freq'] = 50        # Save checkpoints every 50 epochs
    
    # Save updated config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return output_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run training with silhouette score tracking on RunPod')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['phase_magnitude', 'phase_angle', 'comodulation', 'all'],
                      help='Type of model to train')
    parser.add_argument('--config_path', type=str, default='config.json',
                      help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='/workspace/saved_models_silhouette',
                      help='Directory to save models and results')
    args = parser.parse_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Ensure we're in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create necessary directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('/workspace/visualizations', exist_ok=True)

    # Setup config for silhouette tracking
    silhouette_config_path = os.path.join(
        os.path.dirname(args.config_path),
        'config_silhouette.json'
    )
    config_path = setup_config_for_silhouette(args.config_path, silhouette_config_path)

    # Print start time
    start_time = time.time()
    print(f"\nStarting training with silhouette tracking at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model type: {args.model_type}")
    print(f"Config path: {config_path}")
    print(f"Save directory: {args.save_dir}")
    print("-" * 50)

    try:
        # Run training with modified sys.argv
        sys.argv = [
            'train_with_silhouette.py',
            f'--model={args.model_type}',
            f'--config={config_path}',
            f'--save_dir={args.save_dir}'
        ]
        train_main()
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Print completion time and duration
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration/3600:.2f} hours")

if __name__ == '__main__':
    main() 