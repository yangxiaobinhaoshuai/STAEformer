# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STAEformer (Spatio-Temporal Adaptive Embedding Transformer) is a PyTorch implementation for traffic forecasting, published at CIKM 2023. It achieves SOTA on standard traffic benchmarks by using learnable adaptive spatio-temporal embeddings combined with temporal and spatial self-attention.

## Commands

This project uses **uv** for dependency management (Python 3.12).

```bash
# Install dependencies
uv sync

# Train a model
cd model/
python train.py -d METRLA -g 0   # -d: dataset name, -g: GPU ID (use -1 for CPU)

# Available datasets: METRLA, PEMSBAY, PEMS03, PEMS04, PEMS07, PEMS08
```

There are no test scripts or linting configurations defined in this project.

## Architecture

### Data Flow

```
data/<DATASET>/{data.npz, index.npz}
    → lib/data_prepare.py: normalize + split into train/val/test DataLoaders
    → model/train.py: training loop with early stopping
    → model/STAEformer.py: forward pass
    → ../logs/ (training logs) and ../saved_models/ (checkpoints)
```

**Input shape**: `(batch, in_steps, num_nodes, 3)` — channels are traffic value, time-of-day, day-of-week
**Output shape**: `(batch, out_steps, num_nodes, 1)`

### Model (`model/STAEformer.py`)

The model stacks multiple `SelfAttentionLayer` blocks (each containing `AttentionLayer` + feed-forward). The key innovation is **adaptive embedding**: a learnable `(num_nodes, in_steps, adaptive_embedding_dim)` tensor that encodes spatio-temporal patterns without explicit graph structure.

Embedding inputs are summed before the transformer stack:
- Input projection (linear)
- Time-of-day embedding (lookup, 288 steps/day)
- Day-of-week embedding (lookup, 7 days)
- Spatial embedding (optional, per-node learnable)
- Adaptive embedding (always on, spatio-temporal learnable)

Attention runs all temporal layers first, then all spatial layers (not interleaved). `num_layers` controls the depth of each group separately, so the total transformer depth is `2 * num_layers`.

**Output projection**: when `use_mixed_proj=True` (default), flattens `(in_steps, model_dim)` and projects directly to `(out_steps, output_dim)` — this is the key difference from a standard encoder-decoder.

### Configuration (`model/STAEformer.yaml`)

All hyperparameters are dataset-specific YAML entries. Key fields:
- `model_args`: embedding dims, num_heads, num_layers, dropout
- Training: lr, batch_size, max_epochs, early_stop, milestones
- Data: time_of_day, day_of_week flags

Loss function selection is hardcoded in `train.py`: MaskedMAELoss for METRLA/PEMSBAY, HuberLoss for PEMS03-08.

### Key Files

| File | Purpose |
|------|---------|
| `model/STAEformer.py` | Model definition (`AttentionLayer`, `SelfAttentionLayer`, `STAEformer`) |
| `model/train.py` | Training pipeline: `train()`, `eval_model()`, `test_model()` |
| `model/STAEformer.yaml` | Per-dataset hyperparameters |
| `lib/data_prepare.py` | `get_dataloaders_from_index_data()` — data loading and normalization |
| `lib/utils.py` | `StandardScaler`, `MaskedMAELoss`, `seed_everything()` |
| `lib/metrics.py` | `RMSE_MAE_MAPE()` and individual metric functions |
