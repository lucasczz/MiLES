# MiLES: Multi-Level Spatial Embedding Sharing for Trajectory User Linking

This repository contains the implementation of the paper:

> **MiLES: Multi-Level Spatial Embedding Sharing for Enhanced Online Trajectory-User Linking**

## Overview

MiLES (Multi-Level Spatial Embedding Sharing) is a novel approach for Trajectory User Linking (TUL) that significantly improves the performance over existing embedding techniques in an online learning setting by leveraging a hierarchical spatial embedding technique.

## Installation

This project uses Python and can be installed via pip:

```bash
# Clone or download and unpack the repository

# Install the package
pip install -e .
```

## MiLES implementation

The MiLES embedding module is implemented in `src.embedding.MilesEmbedding`. The cell indices `[cell0, cell1, ...]` with which the higher level embeddings are selected are computed during data loading using `src.data.discretization.discretize_coordinates` for ease of implementation.

## Datasets

Follow these steps to download and prepare the datasets used in our experiments:

### Foursquare Dataset

```bash
# Download the dataset
gdown 1VE67jfSN-GV6fkCfuyBEQJzk_qy-SfOk

# Create the raw data directory
mkdir -p data/raw

# Extract the dataset
unzip -j dataset_tsmc2014.zip -d data/raw
```

### GeoLife Dataset

```bash
# Download the dataset
curl "https://download.microsoft.com/download/f/4/8/f4894aa5-fdbc-481e-9285-d5f8c4c4f039/Geolife%20Trajectories%201.3.zip" --output GeoLife.zip

# Extract the dataset
unzip GeoLife.zip -d "data/raw/Geolife Trajectories 1.3"
```

### Preprocessing

After downloading and extracting both datasets, run the preprocessing script:

```bash
# Preprocess the datasets
python src/data/prep_datasets.py
```

This will prepare the datasets in the format required by our models.

## Usage

### Device Configuration

All experiment scripts automatically detect and use available GPUs. You can customize the device configuration using command-line arguments:

```bash
# Use specific GPUs (by index)
python tools/test_TUL.py --gpus 0 1 2

# Adjust workers per device (default: 2)
python tools/test_TUL.py --workers-per-device 4
```

**Device Configuration Options:**
- `--gpus`: Specify GPU indices to use (e.g., `--gpus 0 1 2`). If not specified, all available GPUs are used automatically.
- `--workers-per-device`: Number of workers per GPU/device (default: 2).

**Examples:**
```bash
# Auto-detect all GPUs with default settings
python tools/test_TUL.py

# Use only GPUs 0 and 1 with 3 workers each
python tools/test_TUL.py --gpus 0 1 --workers-per-device 3
```

### Running Evaluations

The following scripts can be used to reproduce our experiments:

```bash
# Run all TUL models with MiLES + other methods
python tools/test_TUL.py

# Run ablation studies
python tools/ablate_all.py

# Evaluate different grid resolutions
python tools/test_grid_resolution.py

# Evaluate weighting of individual embedding levels
python tools/test_level_weights.py

# Test runtimes
python tools/test_walltime.py

# Run all destination prediction models with MiLES + other methods
python tools/test_DP.py
```

All evaluation scripts support the same device configuration options described above.

### Generating Tables and Figures

We provide Jupyter notebooks to generate the tables and figures from our paper:

- `notebooks/tables.ipynb` - Generate main result and wall time tables, plus the appendix tables (variance, Wilcoxon significance, parameter breakdown, level-learning-rate, and dimension-budget tables)
- `notebooks/ablation_all.ipynb` - Generate ablation study figures
- `notebooks/grid_resolution.ipynb` - Generate grid resolution analysis
- `notebooks/level_weights.ipynb` - Generate embedding level weights analysis

## Project Structure

```
MiLES/
├── src/               # Implementations of TUL models and MiLES
├── tools/             # Evaluation scripts
├── notebooks/         # Jupyter notebooks for figure generation
├── data/              # Data directory (see Data section)
├── pub/               # TeX files and figures of the MiLES publication
├── pyproject.toml     # Project configuration
├── README.md          # This file
└── LICENSE            # License information
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, please open an issue.
