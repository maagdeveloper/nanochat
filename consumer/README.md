# nanochatn't 🐣

Basically: "NanoChat? Nah, not happening!" 😄

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

## 🔐 Reproducibility and Artifact Safety

Consumer assigns versioned tokenizer/packing views and verifies SHA-256 manifests for the selected raw Parquet files, native tokenizer artifacts, packed shards, and relevant native preparation source files. This is an integrity/reuse safeguard, not proof that an unverified legacy cache was created by the recorded implementation.

The training wrapper also supports explicit checkpoint resume with guards against missing artifacts, incompatible selected metadata, and rewinding past newer checkpoints. **Continuous-vs-resume training equivalence remains to be tested**; the latest reported frozen-gradient optimizer replay (33/33 compared updated tensors, max difference 0.0) is a narrower check and its test code was not included in this review.

See [underreview.md](underreview.md) for exact implementation status, reported test scope, and open checks.

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

From the **nanochat repository root**, activate a Python environment with nanochat's dependencies installed, then review [consumer/config.toml](config.toml). The commands below use its settings by default. Data preparation, packing, and training accept `--config path/to/custom.toml`; automatic Hugging Face export currently uses the default config (or explicit checkpoint/tokenizer paths).

### 1. Prepare the dataset

```bash
python -m consumer.scripts.prepare_data
```

### 2. Prepare the Tokenizer and Packed Dataset

Train the tokenizer (if missing), then tokenize and pack the
dataset using nanochat's original `scripts.base_prepare` pipeline.

```bash
python -m consumer.scripts.pack_dataset
```

### 3. Pretrain

Check the effective command and data compatibility before starting:

```bash
python -m consumer.scripts.train --dry-run
python -m consumer.scripts.train
```

Use a **new `training.model_tag`** for a new run. Existing checkpoints and logs are not silently overwritten. To resume a compatible run at an existing checkpoint step:

```bash
python -m consumer.scripts.train --resume-from-step 200
```

Replace `200` with a checkpoint step that actually exists for that run. Resume guards are implemented; full uninterrupted-vs-resume training equivalence is still an open test. `--dry-run` performs read-only artifact validation, including hashing, so it may take time on large datasets.

### 4. Export the base model to Hugging Face

```bash
python -m consumer.scripts.export_hf --dry-run
python -m consumer.scripts.export_hf
```

The exporter also accepts `--checkpoint`, `--tokenizer-dir`, and `--output` to convert compatible `experiment_refactor` checkpoints independently of the training configuration. Existing export directories are not overwritten.

**Status:** Base pretraining and Hugging Face export have previously reported passing tests for the pinned `experiment_refactor` checkpoint. The current HF port is inference-only and supports full causal attention (`window_pattern="L"`) only; sliding-window `S` and SFT/chat export are not yet implemented.

## 📌 Roadmap

- [x] Dataset preparation and configurable tokenization.
- [x] Versioned tokenizer and packing artifacts.
- [x] Native pretraining/evaluation and checkpointing; consumer resume CLI and safety guards implemented.
- [ ] Controlled continuous-vs-resume training equivalence test (model + optimizer + step/schedule + data/RNG state).
- [x] Native inference and KV-cache validation.
- [x] Hugging Face CausalLM and Fast Tokenizer export.
- [x] Native/HF weight, logits, cache, and generation parity tests.
- [x] Standalone Hugging Face model loading and batched left-padding tests.
- [x] Document the tested official converter compatibility gap.
- [ ] Native SFT smoke test, exact chat template, and HF chat model export.
- [x] Raw/tokenizer/packed/native-source SHA-256 manifest checks implemented.
- [ ] Artifact recovery/integration negative tests and complete config validation.
- [ ] Interactive training and evaluation visualizations.
- [ ] NeoCorpus integration.
- [ ] Safe cache cleanup.
- [ ] Compatibility testing with older NanoChat architectures.
- [ ] Model cards, documentation, and Hugging Face publishing.

## 🙏 Credits

This is an adaptation of [karpathy/nanochat](https://github.com/karpathy/nanochat), not an independent reimplementation. Credit for the original architecture, training pipeline, and foundational work belongs to Andrej Karpathy and the original contributors.

Original copyright notices and applicable MIT license requirements are preserved.
