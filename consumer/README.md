# nanochatn't 🐣

*Because apparently, nano wasn't small enough.*

A lightweight, reproducible workflow inside [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat), designed to make language model training and Hugging Face export accessible on consumer GPUs.

Built on the `experiment_refactor` branch. The original nanochat architecture and training engine remain the source of truth.

## 🎯 Goals

- Train small language models on consumer GPUs.
- Support ClimbMix and custom datasets.
- Configure tokenization, packing, pretraining, and eventually SFT.
- Reuse nanochat's original training and evaluation scripts.
- Export faithful models and fast tokenizers to Hugging Face Transformers.
- Verify native/HF weights, logits, KV caching, and generation.
- Keep the workflow simple, reproducible, and easy to inspect.

## 🖥️ Hardware

Initial target: **NVIDIA RTX 3060 (12 GB VRAM)**, using a single GPU.

## ⚙️ Development Philosophy

- Reuse the original nanochat implementation; don't rewrite its training engine.
- Run training and data processing through Python scripts.
- Reserve notebooks for exploration, visualization, and analysis.
- Add only what solves a problem we've actually encountered.

## 🤗 Hugging Face Compatibility

Transformers already supports NanoChat. However, the official converter we tested **runs successfully but does not faithfully preserve** checkpoints from `experiment_refactor` at commit `07620a2`.

On our 36.7M-parameter checkpoint, the tested converter skipped seven tensors: two value embeddings, two VE gates, `resid_lambdas`, `x0_lambdas`, and `backout_lambda`. Its output was loadable but had different logits. Our exporter preserved all **33/33 native tensors**, with exact native/HF logits and cached-logits parity in the tested cases.

The exported model uses custom code (`trust_remote_code=True`), a Hugging Face Fast Tokenizer, left-padding support, and native stopping-token IDs. These findings apply to the tested revisions, not necessarily future Transformers releases. Chat/SFT export and legacy NanoChat checkpoint compatibility are still planned.

See [HF_COMPATIBILITY.md](HF_COMPATIBILITY.md) for the pinned converter revision, skipped tensors, and numerical evidence.

## 💬 A Personal Note

Honestly, Karpathy, every time I think I need to implement something, I discover you've already built it into nanochat.

Almost nothing has needed to be rewritten from scratch. It's a testament to how thoughtfully the project was designed.

This journey is less about reinventing the wheel and more about understanding, adapting, and making nanochat work on my hardware.

## 🚀 Getting Started

From the **nanochat repository root**, activate a Python environment with nanochat's dependencies installed, then review [consumer/config.toml](config.toml). The commands below use its settings by default.

### 1. Prepare the dataset

```bash
python -m consumer.scripts.prepare_data
```

### 2. Train or reuse the tokenizer

```bash
python -m consumer.scripts.train_tokenizer
```

### 3. Pack the dataset

```bash
python -m consumer.scripts.pack_dataset
```

### 4. Pretrain

Check the effective command and data compatibility before starting:

```bash
python -m consumer.scripts.train --dry-run
python -m consumer.scripts.train
```

Use a **new `training.model_tag`** for a new run. Existing checkpoints and logs are not silently overwritten.

### 5. Export the base model to Hugging Face

```bash
python -m consumer.scripts.export_hf --dry-run
python -m consumer.scripts.export_hf
```

The exporter also accepts `--checkpoint`, `--tokenizer-dir`, and `--output` to convert compatible `experiment_refactor` checkpoints independently of the training configuration. Existing export directories are not overwritten.

**Status:** Base pretraining and Hugging Face export are verified. The SFT workflow and chat template are not yet included in these instructions.

## 📌 Roadmap

- [x] Dataset preparation and configurable tokenization.
- [x] Versioned tokenizer and packing artifacts.
- [x] Pretraining, evaluation, checkpointing, and resume.
- [x] Native inference and KV-cache validation.
- [x] Hugging Face CausalLM and Fast Tokenizer export.
- [x] Native/HF weight, logits, cache, and generation parity tests.
- [x] Standalone Hugging Face model loading and batched left-padding tests.
- [x] Document the tested official converter compatibility gap.
- [ ] SFT training, chat template, and chat model export.
- [ ] Artifact metadata and reproducibility improvements.
- [ ] Interactive training and evaluation visualizations.
- [ ] NeoCorpus integration.
- [ ] Safe cache cleanup.
- [ ] Compatibility testing with older NanoChat architectures.
- [ ] Model cards, documentation, and Hugging Face publishing.

## 🙏 Credits

This is an adaptation of [karpathy/nanochat](https://github.com/karpathy/nanochat), not an independent reimplementation. Credit for the original architecture, training pipeline, and foundational work belongs to Andrej Karpathy and the original contributors.

Original copyright notices and applicable MIT license requirements are preserved.
