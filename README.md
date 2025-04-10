# HDR Image Grouping Pipeline

A robust solution for grouping HDR images from real estate photography by camera angle, regardless of exposure levels.

## Overview

This pipeline addresses the challenges of grouping HDR exposure brackets from unstructured, real-world photography, where:
- File names cannot be relied upon
- Time-based grouping has many edge cases
- Bracketing may be inconsistent or incomplete
- Accidental shots and duplicates are common

## Features

- **Exposure-agnostic scene matching**: Groups images by visual content, not timestamps or filenames
- **Robust to exposure variations**: Can match extremely dark and bright images of the same scene
- **Handles messy real-world data**: Detects duplicates, accidental shots, and incomplete brackets
- **Modular design**: Each component can be replaced or improved independently
- **Comprehensive validation**: Multiple quality metrics ensure reliable grouping
- **Detailed output**: Organized files and metadata for downstream processing

## Pipeline Steps

1. **Ingest & Preprocess**: Normalizes inputs without assumptions about naming or time structure
2. **Robust Scene Embedding**: Creates exposure-agnostic image fingerprints using deep learning
3. **Scene-Level Clustering**: Groups images based on visual content, ignoring exposure differences
4. **Exposure Sequence Identification**: Sorts and validates exposure brackets within each scene
5. **Final Cleanup & Validation**: Ensures high-confidence groupings and flags problematic cases
6. **Output & Logging**: Delivers structured results for HDR merging or review

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hdr-grouping.git
cd hdr-grouping

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python model_pipeline.py /path/to/images --output-dir ./output --log-level INFO
```

### As a Module

```python
from model_pipeline import HDRGroupingPipeline

# Initialize pipeline
pipeline = HDRGroupingPipeline(output_dir="./output")

# Process a directory of images
results = pipeline.process_directory("/path/to/images")

# Access results
print(f"Found {results['valid_clusters']} valid HDR clusters")
print(f"Output saved to {results['output_dir']}")
```

## Configuration

The pipeline can be configured through a JSON file:

```bash
python model_pipeline.py /path/to/images --config config.json
```

Example configuration:

```json
{
  "ingest": {
    "target_size": [512, 512],
    "max_workers": 4
  },
  "embedding": {
    "model_name": "facebook/dinov2-small",
    "use_gpu": true,
    "dimensionality_reduction": "pca",
    "target_dims": 128
  },
  "clustering": {
    "min_cluster_size": 2,
    "cluster_selection_epsilon": 0.1
  }
}
```

## Requirements

- Python 3.12+
- PyTorch
- OpenCV
- scikit-learn
- HDBSCAN
- UMAP
- And others listed in requirements.txt

## Output Structure

```
output/
├── cluster_0/
│   ├── 00_EV-3.0_IMG001.jpg
│   ├── 01_EV-1.5_IMG002.jpg
│   ├── 02_EV0.0_IMG003.jpg
│   ├── 03_EV+1.5_IMG004.jpg
│   └── metadata.json
├── cluster_1/
│   └── ...
├── unassigned/
│   └── ...
├── thumbnails/
│   └── ...
├── metadata.json
├── report.json
└── summary.txt
```
