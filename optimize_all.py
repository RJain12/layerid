#!/usr/bin/env python

"""
Script to optimize hyperparameters for all model types and generate optimized config files.
This is a convenient wrapper around hyperparameter_optimization.py
"""

import subprocess
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimize all model types')
    parser.add_argument('--trials', type=int, default=30, 
                        help='Number of optimization trials for each model (default: 30)')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Base configuration file to use (default: config.json)')
    args = parser.parse_args()
    
    # Model types to optimize
    model_types = ['phase_magnitude', 'phase_angle', 'comodulation']
    
    # Run optimization for each model type
    for model_type in model_types:
        print(f"\n{'-' * 80}")
        print(f"Optimizing {model_type} model...")
        print(f"{'-' * 80}\n")
        
        # Build command
        cmd = [
            'python', 'hyperparameter_optimization.py',
            '--model', model_type,
            '--trials', str(args.trials),
            '--config', args.config,
            '--apply'
        ]
        
        # Run the command
        subprocess.run(cmd)
    
    print(f"\n{'-' * 80}")
    print("All optimizations complete!")
    print("You can now run training with all optimized configs using:")
    print("python main.py --use-optimized")
    print(f"{'-' * 80}\n")

if __name__ == "__main__":
    main() 