# GDS-MVC

Guided Disentangled Semantic Multi-View Clustering.

This repository contains the official implementation of the GDS-MVC model for multi-view clustering with learnable disentanglement and contrastive learning.

## Overview

GDS-MVC is a deep multi-view clustering framework that:
- **Disentangles** view-specific features into shared semantic representations and private complementary representations via a gating mechanism with multi-head attention
- **Fuses** multi-view shared features using per-view channel-wise SE enhancement with learnable view-level weights
- **Aligns** cluster assignments across views via distribution alignment loss and structure-guided contrastive learning

## Supported Datasets

| Dataset | Views | Samples | Classes |
|---------|-------|---------|---------|
| BDGP | 2 | 2,500 | 5 |
| CCV | 3 | 6,773 | 20 |
| Fashion | 3 | 10,000 | 10 |
| Caltech_5V | 5 | 1,400 | 7 |
| Hdigit | 2 | 10,000 | 10 |
| Cifar100 | 3 | 50,000 | 100 |
| cifar10 | 3 | 50,000 | 10 |
| Youtubeface | 5 | 101,499 | 31 |
| Prokaryotic | 3 | 551 | 4 |
| Synthetic3d | 3 | 600 | 3 |
| WebKB | 2 | 1,051 | 2 |
| Cora | 4 | 2,708 | 7 |
| NGs | 3 | 500 | 5 |
| MNIST-USPS | 2 | 5,000 | 10 |
| Noisy-MNIST | 2 | 30,000 | 10 |
| NUSWIDE | 5 | 5,000 | 5 |
| SUNRGBD | 2 | 10,335 | 45 |
| YouTubeVideo | 3 | 101,499 | 31 |
| ALOI | 4 | 10,800 | 100 |
| bbcsport | 2 | 544 | 5 |
| DHA | 2 | 483 | 23 |
| handwritten | 6 | 2,000 | 10 |
| caltech101_7 | 6 | 1,474 | 7 |

## Installation

```bash
git clone <this-repo-url>
cd GDS-MVC
pip install -r requirements.txt
```

## Data Preparation

Place `.mat` and `.npy` data files under a `data/` directory. Each dataset file should contain:
- `X1`, `X2`, ... `Xk` — feature matrices for each view
- `Y` — ground-truth labels
- For CCV: `STIP.npy`, `SIFT.npy`, `MFCC.npy`, `label.npy`

Download links for the datasets will be provided upon publication.

## Usage

### Training

```bash
# Train on WebKB (default)
python train.py

# Train on a specific dataset
python train.py --dataset BDGP

# Custom hyperparameters
python train.py --dataset Fashion --mse_epochs 200 --con_epochs 50 --lambda_1 0.01 --lambda_2 0.001 --lambda_3 0.001
```

### Hyperparameter Grid Search

Modify the loop values in `train.py` to sweep over multiple parameter combinations:
```python
for args.mse_epochs in [50, 100, 200]:
    for args.lambda_1 in [0.001, 0.01, 0.1]:
        ...
```

Results are saved as `.mat` files in `result_8_8/`.

## File Structure

```
GDSMVC/
├── train.py          # Main training script with grid search
├── network.py        # Model architecture (encoder, decoder, gating, fusion)
├── loss.py           # Loss functions (contrastive, alignment, uncertainty)
├── dataloader.py     # Dataset loaders for all supported datasets
├── metric.py         # Evaluation metrics (ACC, NMI, ARI, PUR)
├── requirements.txt  # Python dependencies
├── LICENSE           # MIT License
└── README.md         # This file
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name | WebKB |
| `--batch_size` | Batch size | 256 |
| `--mse_epochs` | Pre-training epochs | 100 |
| `--con_epochs` | Contrastive training epochs | 100 |
| `--low_feature_dim` | Encoder output dimension | 512 |
| `--high_feature_dim` | Projection head dimension | 128 |
| `--temperature_f` | Temperature for feature contrast | 0.8 |
| `--temperature_l` | Temperature for label contrast | 0.5 |
| `--lambda_1` | Weight for orthogonal/alignment/uncertainty loss | 0.01 |
| `--lambda_2` | Weight for label alignment loss | 0.001 |
| `--lambda_3` | Weight for structure-guided contrastive loss | 0.001 |
| `--learning_rate` | Learning rate | 0.0003 |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
