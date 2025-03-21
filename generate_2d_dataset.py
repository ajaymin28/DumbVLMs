"""
@vi-sri
This script generates 2D shape benchmark datasets based on a specified configuration.
It supports different scenarios such as shape matching, rotation reasoning, and odd one out.
"""

import argparse
import os
import yaml 
from tools.data_generation.Two_D import ShapeBenchmarkGenerator
from tools.data_generation.Two_D import (
    ShapeMatchingScenario,
    RotationReasoningScenario,
    OddOneOutScenario
)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate shape benchmark datasets")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="benchmark_output",
        help="Directory to save generated datasets"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get scenario type
    scenario_type = config.get("scenario_type", "shape_matching")
    
    # Initialize the appropriate scenario generator
    if scenario_type == "shape_matching":
        scenario = ShapeMatchingScenario(config, args.output_dir, args.seed)
    elif scenario_type == "rotation_reasoning":
        scenario = RotationReasoningScenario(config, args.output_dir, args.seed)
    elif scenario_type == "odd_one_out":
        scenario = OddOneOutScenario(config, args.output_dir, args.seed)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    # Generate the dataset
    scenario.generate()
    
    print(f"Dataset generation completed. Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()