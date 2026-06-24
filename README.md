# Crack Detection and Segmentation Suite

This repository provides a comprehensive framework for crack detection and segmentation in infrastructure images. It integrates state-of-the-art Deep Learning models, Probabilistic Circuits (PCNet), and Evolutionary Algorithms to provide robust and interpretable solutions for structural health monitoring.

## 🚀 Key Features

- **Multi-Model Support**:
  - **Deep Learning**: U-Net, U-Net++, CrackFormer, DeepCrack, ConvAutoEncoder.
  - **Probabilistic Circuits (PCNet)**: Standard PCNet, Attentional PCNet, and Multi-Head Attentional PCNet for tractable and robust segmentation.
  - **GAN-based Models**: Integration with UP-CrackNet for high-fidelity segmentation.
- **Evolutionary Architecture Search**: Uses **Grammatical Evolution (GE)** to automatically discover and optimize Probabilistic Circuit structures.
- **Flexible Training Modes**: Supports Supervised, Unsupervised, and Evolutionary training pipelines.
- **Advanced Loss Functions**: Includes DiceLoss, FocalLoss, soft-clDice (centerline Dice), and specialized Unsupervised PCNet losses.
- **Comprehensive Evaluation**: Metrics include Dice Coefficient, IoU, Centerline IoU (clIoU), Accuracy, Precision, Recall, and F1-Score.
- **Configuration-Driven**: All experiments are defined via JSON configuration files for easy reproducibility and hyperparameter tuning.

## 📁 Project Structure

```text
crack-detection/
├── data/                  # Dataset directory (DeepCrack, CrackSeg9k, etc.)
├── models/                # Saved model weights (.pth, .pkl)
├── logs/                  # Experiment logs and visualizations
└── src/
    ├── config/            # JSON configuration files for experiments
    ├── probabilistic_circuits/ # PCNet and Grammatical Evolution logic
    ├── upcracknet/        # UP-CrackNet implementation and losses
    ├── experiment.py      # Core experiment orchestration and launchers
    ├── networks.py        # standard DL network architectures
    ├── main.py            # Entry point for running experiments
    ├── crackdata_processing.py # Data loading and preprocessing
    └── plots.py           # Visualization and plotting utilities
```

## 🛠️ Getting Started

### Prerequisites

- Python 3.10+
- PyTorch
- Additional dependencies: `numpy`, `matplotlib`, `opencv-python`, `scikit-learn`, `tqdm`, `json5`.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NielsenErik/Crack_Segementation.git
   cd crack-detection
   ```
2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## 🏃 Running Experiments

Experiments are launched using `src/main.py` with a path to a configuration file.

### Basic Training

To run a standard U-Net experiment:
```bash
python src/main.py src/config/unet.json
```

### PCNet with Grammatical Evolution

To evolve a PCNet structure:
```bash
python src/main.py src/config/pcnet_ge.json
```

### UP-CrackNet (GAN)

To run the UP-CrackNet pipeline:
```bash
python src/main.py src/config/upcracknet.json
```

## ⚙️ Configuration

Configuration files allow you to customize:
- `algorithm`: The model/approach to use (e.g., `unet`, `ge_pcnet`, `upcracknet`).
- `data_path`: Path to the training images.
- `epochs`, `batch_size`, `learning_rate`.
- `resize_size`: Image resolution for processing.
- `ge_parameters`: Population size, genome length, mutation rates for evolution.

## 📊 Evaluation & Visualization

Results are saved in the `logs/` directory, including:
- **Metrics**: CSV/JSON files containing detailed per-epoch and final test metrics.
- **Visualizations**: Side-by-side comparisons of Input, Ground Truth, and Predicted Masks.
- **Models**: Saved weights for the best-performing models.

---
