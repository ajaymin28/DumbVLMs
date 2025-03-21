# DumbVLMs

A benchmark dataset for evaluating spatial reasoning, association, and matching capabilities in Vision Language Models (VLMs).

## Overview

DumbVLM-Bench provides a standardized way to assess how well Vision Language Models understand and reason about spatial relationships in images. The benchmark focuses on three key capabilities:

1. **Spatial Reasoning**: Understanding relative positions and orientations of objects
2. **Spatial Association**: Connecting objects based on spatial proximity or relationships
3. **Spatial Matching**: Identifying corresponding objects or patterns across multiple images
4. **Cross Modal Reasoning**: 

## Repository Structure

```
DumbVLMs/
├── data/
│   ├── raw/                    # Original image datasets
│   ├── processed/              # Preprocessed images
│   ├── annotations/            # Human-annotated spatial relationships
│   └── challenge_sets/         # Specialized test suites
├── benchmark/
│   ├── tasks/                  # Task definitions
│   │   ├── relative_position/  # "If Z were to be placed to the right of Y and between X and Z, Is X to the left of Y?" tasks
│   │   ├── distance_reasoning/ # "If you were to place object B in the middle of A and C, Will it be closer to E or D ?" tasks  
│   │   ├── spatial_grouping/   # "Which objects belong together ?" tasks
│   │   └── cross_modal_mapping/ # "If A is mapped to T and B is mapped to E, what can C be mapped to ?" tasks
│   ├── metrics/                # Evaluation metrics
│   └── submissions/            # Example submission formatting
├── baselines/
│   ├── models/                 # Implementation of baseline models
│   ├── configs/                # Configuration files
│   └── results/                # Baseline performance results
├── tools/
│   ├── visualization/          # Tools for visualizing spatial relationships
│   ├── data_generation/        # Scripts for generating synthetic data
│   └── evaluation/             # Evaluation scripts
├── docs/
│   ├── tasks/                  # Detailed task descriptions
│   ├── metrics/                # Metric documentation
│   ├── leaderboard/            # Leaderboard setup
│   └── getting_started.md      # Quick start guide
├── tests/                      # Unit and integration tests
├── experiments/                  # Example usage and analyses
├── setup.py                    # Package installation
├── requirements.txt            # Dependencies
├── LICENSE                     # License information
└── README.md                   # Main documentation
```

## Key Features

- **Diverse Challenge Sets**: Covers various aspects of spatial understanding
- **Controlled Difficulty Levels**: Gradual progression from simple to complex spatial reasoning
- **Diagnostic Categories**: Identifies specific strengths and weaknesses in VLM capabilities
- **Standardized Evaluation Protocol**: Ensures fair comparison across different models
- **Human Performance Baseline**: Includes human performance metrics for comparison

## Getting Started

### Installation

```bash
git clone https://github.com/ajaymin28/DumbVLMs
cd DumbVLMs
pip install -e .
```

### Basic Usage

```python
from spatialvlm_bench import BenchmarkSuite, TaskLoader, Evaluator

# Load a specific task suite
task_suite = TaskLoader.load("relative_position")

# Run a model on the benchmark
results = BenchmarkSuite.evaluate(model, task_suite)

# Generate evaluation report
Evaluator.generate_report(results, output_dir="./results")
```

## Data Generation - 2D

Generate datasets using configuration files:

```bash
python generate_2d_dataset.py --config configs/scenario1_shape_matching.yaml --output_dir output --seed 42
```

### Command Line Arguments

- `--config`: Path to YAML configuration file (required)
- `--output_dir`: Directory to save generated datasets (default: "benchmark_output")
- `--seed`: Random seed for reproducibility (optional)

## Scenarios

### Scenario 1: Shape Matching

Generates pairs of shapes (numbered in red, lettered in green) that need to be matched based on their geometric properties despite different positions and rotations.

#### Parameters:

- `num_shapes`: Number of shapes to match (2-5)
- `spatial_complexity`: Complexity of spatial arrangement
  - `easy`: Simple column-based arrangement
  - `medium`: Random placement clustered in the center
  - `hard`: Random placement spread across the image
- `rotation_degree`: Rotation settings
  - `default`: No rotation
  - `random`: Random rotation for each shape
  - Custom angle (e.g., "90"): Fixed rotation angle applied to gallery shapes

#### Output:

- Images with numbered red shapes and lettered green shapes
- Ground truth matching files (no ground truth displayed in images)

### Scenario 2: Rotation Reasoning

Generates images with a single query shape and multiple gallery shapes, with prompts asking which gallery shape the query would match after rotation.

#### Parameters:

- `gallery_size`: Number of gallery shapes to include
- `rotation_angles`: List of rotation angles to test (e.g., [90, 180, 270])

#### Output:

- Images with a query shape (red with "?") and gallery shapes (green with letters)
- Prompt files with the question "If the shape in Red is rotated by X degrees, what shape would it match to?"
- Ground truth files with the correct answer letter

### Scenario 3: Odd One Out

Generates sequences of shapes where one shape differs from the others based on a specific criterion.

#### Parameters:

- `sequence_length`: Number of shapes in the sequence
- `oddity_criteria`: List of criteria for identifying the odd shape
  - `shape`: Different number of sides
  - `color`: Different color
  - `rotation_angle`: Different rotation
  - `size`: Different size

#### Output:

- Images with sequences of lettered shapes where one differs
- Prompt files with the question "Pick the odd one out from the image. Give only the alphabet of the diagram as answer."
- Ground truth files with the correct answer letter

## Configuration Files

Sample configuration files are provided in the `configs/` directory:

- `scenario1_shape_matching.yaml`: Basic shape matching (easy difficulty)
- `scenario1_shape_matching_medium.yaml`: Medium difficulty shape matching
- `scenario1_shape_matching_hard.yaml`: Hard difficulty shape matching
- `scenario2_rotation_reasoning.yaml`: Rotation reasoning task
- `scenario3_odd_one_out.yaml`: Odd one out task

## Baselines

The repository includes baseline implementations and results for:

- CLIP
- GPT-4V
- Claude 3 Opus
- Gemini Pro Vision
- LLaVA
- CogVLM