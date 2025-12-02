# Surface Classification Enhancement Using Estimated Power Consumption Data

**Authors:** Łukasz Kędzierski, Nina Janus, Bartłomiej Cybulski, Paweł Smyczyński and Grzegorz Granosik
**Status:** Published
**Paper Link:** <https://www.sciencedirect.com/science/article/pii/S095219762503091X>

## Overview

This repository contains the experimental code for research on improving autonomous mobile robot control with robust surface classification methods. We investigate the application of convolutional neural networks for recognition of terrain types using IMU and estimated power consumption data and provide comprehensive comparisons with XGBoost.

Our work demonstrates that IMU data is sufficient for satisfactory terrain classification and information about power expenditure can further enhance it. All experiments were performed using [a custom dataset](https://data.mendeley.com/datasets/j73s4z6mnv/1) available for reproducibility and future research.

## Key Features

- PyTorch implementation of CNN architecture for surface type classification
- Comprehensive comparison with [XGBoost](https://xgboost.readthedocs.io/en/stable/)
- Custom dataset with public availability
- Reproducible experiments with fixed random seeds
- GPU-optimized training with CPU fallback support

## Installation

### Dependencies

This project uses Python and requires the following core dependencies:
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [SciPy](https://scipy.org/)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)
- [PyTorch](https://pytorch.org/)

All dependencies are managed through `pyproject.toml`. 

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lukasz-kedzierski/surface-classification
cd surface-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in editable mode:
```bash
pip install -e .
```

This installs all dependencies and makes the `src` modules importable throughout the project.

4. Install PyTorch:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

### Hardware Requirements

- **Recommended:** NVIDIA GPU with CUDA support

## Dataset

The custom dataset used in this research is publicly available for download in raw form:

**Download Link:** <https://data.mendeley.com/datasets/j73s4z6mnv/1>  
**Size:** 5.65 GB  
**Format:** ROS bag files

After downloading, place the dataset in the root directory ([recommended structure](#project-structure)). The preprocessing scripts expect a path relative to root.

## Usage

### Quick Start

1. **Download the dataset:**

2. **Prepare the dataset:**
```bash
python src/data_processing/preprocessing.py
```

3. **Run baseline experiments:**
```bash
python src/experiments/launchers/launcher_xgb.py --config-file xgb_tuning_generalized.yaml --script-name surface_classification_xgb_tuning.py
```

4. **Train CNN models:**
```bash
python src/experiments/launchers/launcher_cnn.py --config-file cnn_tuning_generalized.yaml --script-name surface_classification_cnn_tuning.py
```

5. **Generate results and comparisons:**
```bash
python src/result_processing/tuning_results.py
python src/result_processing/statistical_analysis.py
```

### Reproducibility

All experiments use fixed random seeds for reproducibility. Results should be identical across runs on similar hardware configurations.

### Key Scripts

- `src/data_processing/preprocessing.py` - Data preprocessing pipeline
- `src/experiments/cv/surface_classification_cnn_cv.py` - CNN model CV training
- `src/experiments/tuning/` - CNN and XGB CV tuning scripts with best model save
- `src/experiments/launchers/` - Launcher scripts for paper experiments
- `src/experiments/training/surface_classification_cnn_training.py` - CNN model training script for deployment
- `src/result_processing/` - Dataset analysis; Comprehensive model comparison and results generation

### Expected Runtime

Execution time was measured using the following hardware:
- Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz
- 128 GB 2400 MT/s
- 12 GB NVIDIA GeForce RTX 3060

| Script | GPU Time | Notes |
|--------|----------|-------|
| Data preprocessing | ~65 min | ~45 min for extracting, ~20 min for processing |
| XGB threshold analysis | ~10 min | 40 splits |
| CNN CV experiments | ~3.0 h | 9 parallel processes (one per experiment), 10 splits |
| CNN tuning experiments | ~4.0-4.5 h | 9 parallel processes (one per experiment), 40 splits |
| XGB tuning experiments | ~1.0-3.0 h | 9 parallel processes (one per experiment), 40 splits |

## Project Structure

```
├── src/                      # Main source code
│   ├── data_processing/       # Data processing modules
│   ├── models/                # Model implementations
│   ├── experiments/           # Training and evaluation scripts
│   ├── result_processing/     # Result processing modules
│   └── utils/                 # Helper functions
├── configs/                  # Experiment configuration files
├── data/                     # Dataset storage
│   ├── train_set/             # Training dataset
│   │   ├── raw/                # Original dataset
│   │   └── processed/          # Preprocessed data
│   └── odom_erros.ods         # Odometry error values
├── results/                  # Experimental outputs
│   ├── figures/               # Generated plots and figures
│   ├── models/                # Saved model checkpoints
│   └── logs/                  # Training logs
```

## Citing

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{KEDZIERSKI2026113060,
title = {Surface classification enhancement using estimated power consumption data},
journal = {Engineering Applications of Artificial Intelligence},
volume = {163},
pages = {113060},
year = {2026},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.113060},
author = {Łukasz Kędzierski and Nina Janus and Bartłomiej Cybulski and Paweł Smyczyński and Grzegorz Granosik},
}
```

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Contact

For questions about the code or dataset, please contact:
- Łukasz Kędzierski: <249911@edu.p.lodz.pl>
- Nina Janus: <249910@edu.p.lodz.pl>

---

<!-- **Note:** This repository is actively maintained during the review process. Updates and improvements are ongoing. -->
