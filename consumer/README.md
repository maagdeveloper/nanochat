# nanochat on Consumer GPUs 🐣

A lightweight, reproducible end-to-end workflow
built on Andrej Karpathy's nanochat experiment_refactor.

## Goals

- Train small language models on consumer GPUs.
- Support ClimbMix and custom datasets.
- Configure tokenizer, pretraining, and SFT.
- Reuse nanochat's original training engine.
- Export complete models and tokenizers to Hugging Face.
- Validate native vs Hugging Face inference parity.

## Development

The original nanochat implementation remains the source of truth.

Training and data processing run as Python scripts.
Notebooks are reserved for exploration and visualization.

Initial target: NVIDIA RTX 3060 12GB.