# MG-STGNN: Mechanism-Guided Spatiotemporal Graph Neural Network for CBAM Policy Evaluation

## Overview
This repository contains the code and data processing pipeline for the paper:

**"Evaluating Carbon Border Adjustment Effects in Interconnected European Electricity Markets via Mechanism-Guided Graph Learning"**

*Submitted to Applied Energy*

MG-STGNN is a spatiotemporal graph neural network that separates three market mechanism channels — merit-order dispatch, cross-border trade arbitrage, and carbon-cost pass-through — to analyze the EU Carbon Border Adjustment Mechanism's projected effects on interconnected European electricity markets.

## Key Features
- Channel-separated message passing with three mechanism-typed edge types
- Dynamic adjacency construction from hourly cross-border flow data
- Dual-target prediction (electricity price + carbon intensity)
- CBAM counterfactual scenario rollout with mechanism decomposition
- Uncertainty quantification via MC Dropout + conformal prediction

## Requirements
- Python 3.11+
- PyTorch 2.1+
- CUDA 12.1+ (recommended)
- See `requirements.txt` for full dependency list

## Installation
```bash
git clone https://github.com/[username]/MG-STGNN-CBAM.git
cd MG-STGNN-CBAM
conda create -n mg-stgnn python=3.11
conda activate mg-stgnn
pip install -r requirements.txt
```

## Data
The electricity market data is sourced from the ENTSO-E Transparency Platform via the Fraunhofer ISE Energy-Charts API. See `data/README.md` for download instructions and preprocessing steps.

**Countries covered:** AT, BE, CH, CZ, DE, DK, ES, FR, HU, IT, NL, NO, PL, SE, SK
**Period:** 2019–2024 (hourly resolution, ~52,600 observations per country)

## Usage

### Training
```bash
python src/train.py --config configs/default.yaml
```

### CBAM Scenario Rollout
```bash
python src/cbam_rollout.py --gamma 1.0 --checkpoint checkpoints/best_model.pt
```

### Mechanism Attribution
```bash
python src/mechanism_attribution.py --checkpoint checkpoints/best_model.pt
```

## Model Architecture

The model processes 15 country nodes through:
1. **Merit-Order (MO)** self-loop edges: domestic dispatch reshuffling
2. **Trade-Arbitrage (TA)** inter-node edges: cross-border trade adjustment
3. **Carbon-Cost (CC)** inter-node edges: direct CBAM cost transmission

Followed by temporal convolutional encoding (TCN) and dual-target prediction heads.

## Results Summary

| Metric | MG-STGNN | Best Baseline |
|---|---|---|
| Price RMSE (EUR/MWh) | **24.47** | 25.96 (XGBoost) |
| CI RMSE (kg CO₂/MWh) | **21.63** | 21.88 (XGBoost) |
| Peak-RMSE | **49.68** | 57.90 (Spatial-lag) |

## Citation
```bibtex
@article{shen2025mgstgnn,
  title={Evaluating Carbon Border Adjustment Effects in Interconnected European Electricity Markets via Mechanism-Guided Graph Learning},
  author={Shen, Jiachen and Shi, Jian and Wang, Dan and Han, Zhu},
  journal={Applied Energy},
  year={2025},
  note={Under review}
}
```

## License
This project is licensed under the MIT License — see `LICENSE` for details.
