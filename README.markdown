# RAIE Transformer: Recency-Augmented Transformer for Cricket Ball Trajectory Prediction

![RAIE Transformer Overview](images/raie_transformer_overview.png)

This repository contains the implementation of the **RAIE Transformer**, a novel architecture designed for automated sports ball trajectory prediction in cricket, as described in the paper *"RAIE Transformer: A Recency-Augmented Transformer Architecture for Automated Sports Ball Trajectory Prediction in Cricket"* (IEEE, 2025). The model leverages **Recency-Augmented Input Embeddings (RAIE)** and a lightweight transformer architecture to achieve accurate trajectory predictions using short input sequences and moderate frame rates, making it suitable for real-time sports analytics and live broadcasting.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

The RAIE Transformer addresses the challenges of predicting cricket ball trajectories by incorporating temporal recency features into the input embeddings, prioritizing recent frames while preserving long-term context. It integrates a **YOLO-based detection pipeline** for real-time ball tracking and an end-to-end prediction framework, achieving superior performance compared to baseline models like Base Transformer, LSTM, and RNN.

### Key Metrics
- **Test Rollout MSE**: 3.9515
- **Average Displacement Error (ADE)**: 2.3508
- **Final Displacement Error (FDE)**: 2.5595

The model was trained and evaluated on a curated dataset of **4,400 manually annotated frames** from **212 cricket deliveries**, sourced from public YouTube videos.

## Key Features
- **Recency-Augmented Input Embeddings (RAIE)**: Enhances prediction accuracy by emphasizing recent motion patterns.
- **Automated Pipeline**: Uses YOLOv11n for real-time ball detection, eliminating the need for manual annotations.
- **Lightweight Architecture**: Optimized for moderate-FPS videos, reducing computational requirements.
- **Open Dataset**: Provides CSV annotations for 4,400 frames, compliant with copyright and ethical standards.
- **Reproducible Results**: Includes scripts for preprocessing, training, evaluation, and visualization.

## Repository Structure
```
RAIE-Transformer-Cricket/
├── src/
│   ├── model/
│   │   ├── raie_transformer.py    # RAIE Transformer implementation
│   │   ├── yolo_detection.py      # YOLO-based ball detection
│   ├── preprocessing/
│   │   ├── preprocess.py         # Data preprocessing pipeline
│   │   ├── annotation_tool.py    # Custom annotation tool
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script (MSE, ADE, FDE)
│   ├── visualize.py              # Visualization script for trajectories
├── data/
│   ├── annotations.csv           # Full dataset annotations
│   ├── sample_data.csv           # Sample dataset for testing
│   ├── dataset.md                # Dataset description
├── models/
│   ├── pretrained_model.pth      # Pre-trained model weights
│   ├── config.yaml               # Model and training configurations
├── images/
│   ├── raie_transformer_overview.png  # Model architecture diagram
│   ├── trajectory_plot.png       # Sample trajectory visualization
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # License file
```

## Installation

### Prerequisites
- Python 3.8+
- Git
- Optional: CUDA-enabled GPU for faster training/inference

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/RAIE-Transformer-Cricket.git](https://github.com/HiteshDereddy/RAIE-Transformer.git
   cd RAIE-Transformer
   ```

2. **Install Dependencies**:
   Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   Key dependencies include:
   - `torch`
   - `opencv-python`
   - `ultralytics` (for YOLOv11n)
   - `numpy`, `pandas`, `matplotlib`


3. **Optional: Conda Environment**:
   Create a Conda environment for reproducibility:
   ```bash
   conda env create -f environment.yml
   conda activate raie-transformer
   ```

## Dataset

The dataset comprises **4,400 manually annotated frames** from **212 cricket deliveries**, sourced from public YouTube videos:
- **Matches**:
  - "Lancashire vs Somerset DAY ONE LV= Insurance County Championship"
  - "Vertu Motors Live: Yorkshire v Leicestershire – Vitality T20 Blast"
- **Source**: Official Yorkshire Cricket YouTube channel
- **Annotations**: CSV files with columns: `Frame ID`, `Delivery ID`, `Ball centroid coordinates (x, y)`, `Timestamp`
- **Diversity**: Includes fast, spin, and swing deliveries for varied motion patterns

### Files
- Coming Soon

**Note**: Only derived annotations (CSV files) are shared to comply with copyright and ethical standards. Raw video files are not included.

## Usage

### 1. Preprocessing
Prepare the dataset for training:
```bash
python src/preprocessing/preprocess.py --input data/annotations.csv --output data/processed_data.csv
```

### 2. Training
Train the RAIE Transformer model:
```bash
python src/train.py --data data/processed_data.csv --model_config models/config.yaml --output_dir models/
```

### 3. Inference
Perform trajectory prediction:
```bash
python src/evaluate.py --model models/pretrained_model.pth --data data/processed_data.csv --output results/trajectories
```

### 4. Visualization
Generate trajectory plots (similar to Figure 2 in the paper):
```bash
python src/visualize.py --input results/trajectories --output results/trajectory_plots
```

### 5. Full Pipeline
Run the end-to-end pipeline (detection + prediction):
```bash
bash scripts/run_pipeline.sh
```

## Results

The RAIE Transformer outperforms baseline models across key metrics:
- **Test Rollout MSE**: 3.9515
- **ADE**: 2.3508
- **FDE**: 2.5595

### Comparison with Baselines
| Model            | MSE     | ADE     | FDE     |
|------------------|---------|---------|---------|
| RAIE Transformer | 3.9515  | 2.3508  | 2.5595  |
| Base Transformer | 12.3241 | 4.0023  | 7.3452  |
| LSTM             | 91.0440 | 10.0999 | 13.2344 |
| RNN              | 85.1120 | 10.0912 | 11.1628 |

See `results/evaluation_results.md` for detailed results and ablation studies.

## Visualizations

Below are sample visualizations of predicted vs. true trajectories (as in Figure 2 of the paper):

![Trajectory Plot](images/trajectory_plot.png)
*Caption*: Predicted (red) vs. true (blue) trajectories for a cricket delivery, showing close alignment with slight divergence post-bounce.

![Frame-by-Frame Visualization](images/frame_by_frame.png)
*Caption*: Frame-by-frame comparison of predicted (green) and true (red) trajectories, capturing bounce dynamics.

To generate similar visualizations, run:
```bash
python src/visualize.py --input results/trajectories --output results/trajectory_plots
```
