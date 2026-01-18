# RM-Optimizer

A comprehensive framework for analyzing reward models through loss landscape geometry, RL coupling dynamics, and data efficiency metrics.

## Features

- **Loss Landscape Analysis**: Hessian eigenspectrum computation for understanding reward model geometry
- **RL Coupling**: Policy simulation and RL-readiness scoring
- **Data Efficiency**: Active learning for preference data collection
- **Multi-Optimizer Comparison**: Compare Adam, AdamW, SGD, Muon, Lion

## Installation

```bash
# Clone and install
git clone <repository>
cd rm-optimizer
pip install -e ".[dev]"
```

## Quick Start

```bash
# Train a reward model
rm-opt train --model deberta --optimizer muon --epochs 5

# Analyze Hessian eigenspectrum
rm-opt analyze --checkpoint model.ckpt --type hessian

# Compare optimizers
rm-opt compare --optimizers adam,muon,lion --seeds 3
```

## Hardware Requirements

- **Recommended**: NVIDIA H100 80GB
- **Minimum**: NVIDIA A100 40GB or RTX 4090 24GB

## Project Structure

```
rm_optimizer/
├── core/           # Base classes and data loading
├── training/       # PyTorch Lightning modules
├── landscape/      # Hessian and loss landscape analysis
├── rl_coupling/    # RL readiness and policy simulation
├── data_efficiency/# Active learning
└── interfaces/     # CLI and dashboard
```

## License

MIT
