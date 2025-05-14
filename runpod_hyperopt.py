#!/usr/bin/env python3
"""
Hyperparameter optimization script for RunPod.
This script handles the RunPod-specific setup and runs the hyperparameter optimization.
"""

import os
import sys
import torch
import argparse
import time
from hyperparameter_optimization import optimize_hyperparameters

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization on RunPod')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['phase_magnitude', 'phase_angle', 'comodulation'],
                      help='Type of model to optimize')
    parser.add_argument('--n_trials', type=int, default=50,
                      help='Number of trials to run')
    parser.add_argument('--study_name', type=str, default=None,
                      help='Name for the Optuna study')
    parser.add_argument('--config_path', type=str, default='config.json',
                      help='Path to configuration file')
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
    os.makedirs('/workspace/saved_models', exist_ok=True)
    os.makedirs('/workspace/visualizations', exist_ok=True)

    # Print start time
    start_time = time.time()
    print(f"\nStarting hyperparameter optimization at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model type: {args.model_type}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Study name: {args.study_name or 'default'}")
    print("-" * 50)

    try:
        # Run hyperparameter optimization
        optimize_hyperparameters(
            model_type=args.model_type,
            n_trials=args.n_trials,
            study_name=args.study_name,
            config_path=args.config_path
        )
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        raise
    finally:
        # Print completion time and duration
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nOptimization completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration/3600:.2f} hours")
        print(f"Average time per trial: {duration/args.n_trials/60:.2f} minutes")

if __name__ == '__main__':
    main() 