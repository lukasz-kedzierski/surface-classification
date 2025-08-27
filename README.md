# Surface Classification Enhancement Using Estimated Power Consumption Data

**Authors:** Łukasz Kędzierski, Nina Janus, Bartłomiej Cybulski, Paweł Smyczyński and Grzegorz Granosik  
**Status:** Under Review (Major Revision)  
**Paper Link:** [Link when available]

## Overview

This repository contains the experimental code for research on improving autonomous mobile robot control with robust surface classification methods. We investigate the application of convolutional neural networks for recognition of terrain types using IMU and estimated power consumption data and provide comprehensive comparisons with XGBoost.

Our work demonstrates that IMU data is sufficient for satisfactory terrain classification and information about power expenditure can further enhance it. All experiments were performed using [a custom dataset](https://tulodz-my.sharepoint.com/:f:/g/personal/202715_edu_p_lodz_pl/Em1T2WBmJT1NvsVm7FuZdJYB-d-HaB4iCnT79G592e8QtQ) available for reproducibility and future research.

## Key Features

- Implementation of CNN architecture for surface type classification
- Comprehensive comparison with [XGBoost](https://xgboost.readthedocs.io/en/stable/)
- Custom dataset with public availability
- Reproducible experiments with fixed random seeds
- GPU-optimized training with CPU fallback support

## Installation

### Dependencies

This project uses Python and requires the following core dependencies:
- pandas
- numpy
- scipy
- scikit-learn
- pytorch

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

### Hardware Requirements

- **Recommended:** NVIDIA GPU with CUDA support
- **Minimum:** CPU (training will be significantly slower)
- **Memory:** [To be determined based on experiments]

## Dataset

The custom dataset used in this research is publicly available for download:

**Download Link:** <https://tulodz-my.sharepoint.com/:f:/g/personal/202715_edu_p_lodz_pl/Em1T2WBmJT1NvsVm7FuZdJYB-d-HaB4iCnT79G592e8QtQ>  
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
python src/experiments/run_baselines.py
```

4. **Train CNN models:**
```bash
python src/experiments/train_cnn.py
```

5. **Generate results and comparisons:**
```bash
python src/experiments/evaluate_models.py
```

### Reproducibility

All experiments use fixed random seeds for reproducibility. Results should be identical across runs on similar hardware configurations.

### Key Scripts

- `src/data/preprocessing.py` - Data preprocessing pipeline
- `src/experiments/run_baselines.py` - Traditional ML baseline experiments
- `src/experiments/train_cnn.py` - CNN model training and evaluation
- `src/experiments/evaluate_models.py` - Comprehensive model comparison and results generation
- `scripts/` - Utility scripts for setup and batch processing

## Project Structure

```
├── src/                    # Main source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model implementations
│   ├── experiments/       # Training and evaluation scripts
│   └── utils/             # Helper functions
├── data/                  # Dataset storage
│   ├── raw/              # Original dataset
│   └── processed/        # Preprocessed data
├── results/               # Experimental outputs
│   ├── figures/          # Generated plots and figures
│   ├── models/           # Saved model checkpoints
│   └── logs/             # Training logs
├── notebooks/             # Jupyter notebooks for analysis
├── scripts/               # Utility scripts
└── tests/                 # Unit tests
```

## Citing

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{[citation-key],
  title={[Paper Title]},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={[Year]},
  note={Under Review}
}
```

## License

[License information]

## Contact

For questions about the code or dataset, please contact:
- Łukasz Kędzierski: <249911@edu.p.lodz.pl>
- Nina Janus: <249910@edu.p.lodz.pl>

---

**Note:** This repository is actively maintained during the review process. Updates and improvements are ongoing.