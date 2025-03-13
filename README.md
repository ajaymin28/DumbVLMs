# DumbVLMs

A benchmark dataset for evaluating spatial reasoning, association, and matching capabilities in Vision Language Models (VLMs).

## Overview

SpatialVLM-Bench provides a standardized way to assess how well Vision Language Models understand and reason about spatial relationships in images. The benchmark focuses on three key capabilities:

1. **Spatial Reasoning**: Understanding relative positions and orientations of objects
2. **Spatial Association**: Connecting objects based on spatial proximity or relationships
3. **Spatial Matching**: Identifying corresponding objects or patterns across multiple images

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
│   │   ├── relative_position/  # "Is X to the left of Y?" tasks
│   │   ├── distance_reasoning/ # "Which object is closer to X?" tasks  
│   │   ├── spatial_grouping/   # "Which objects belong together?" tasks
│   │   └── cross_modal_mapping/ # "Find the corresponding object" tasks
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

## Task Categories

### 1. Relative Position Understanding

Tests a model's ability to understand spatial relationships like "above," "below," "left of," "right of," "inside," "outside," etc.

### 2. Distance Estimation

Evaluates how well models can judge relative distances between objects in a scene.

### 3. Spatial Grouping

Assesses models on grouping objects based on spatial proximity or arrangement patterns.

### 4. Cross-Image Mapping

Tests the ability to find corresponding objects or regions across multiple images based on spatial configuration.

## Evaluation Metrics

- **Accuracy**: Percentage of correct spatial relationship judgments
- **Consistency**: Agreement of model predictions across variations of the same scene
- **Robustness**: Performance under visual perturbations (lighting, viewpoint, occlusion)
- **Fine-grained Analysis**: Breakdown of performance by relationship type and scene complexity

## Baselines

The repository includes baseline implementations and results for:

- CLIP
- GPT-4V
- Claude 3 Opus
- Gemini Pro Vision
- LLaVA
- CogVLM

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use SpatialVLM-Bench in your research, please cite:

```
@misc{spatialvlmbench2025,
  author = {Your Name},
  title = {SpatialVLM-Bench: A Benchmark for Spatial Reasoning in Vision Language Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  url = {https://github.com/yourusername/SpatialVLM-Bench}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.