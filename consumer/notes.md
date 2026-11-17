# nanochatn't 🐣 — Engineering Notes

> Architecture decisions and durable principles. For status, open tests, and historical evidence see [underreview.md](underreview.md).
> Updated: 2026-11-17, based on the supplied consumer snapshot. No new native GPU tests were run in this review.

## Architecture rule — Consumer is a thin wrapper

**Consumer owns configuration, artifact identity, and orchestration. Nanochat owns the native implementation.**

The goal is not a second nanochat; it is a reproducible, approachable workflow for running native nanochat experiments on consumer hardware.

**Keep:**
- `config.toml` as the primary workflow configuration entry point, while making explicit config overrides consistent where supported.
- Versioned tokenizer and packed dataset views, config fingerprints, and verified cache reuse.
- Independently runnable preparation, training, and export stages.
- Native implementation behavior and narrowly scoped parity tests.

**Prefer:**
- Invoking upstream functions and scripts rather than copying algorithms.
- Minimal adapters for configuration, paths, validation, checkpoint safety, and logs.
- Changes upstream can flow through without a second implementation to maintain.

**Avoid:**
- Changing native nanochat only to support consumer-specific behavior.
- Reimplementing tokenization, packing, training, or inference algorithms in consumer.
- Abstractions that are not needed for a tested use case.
- Reusing caches whose actual bytes or native implementation provenance cannot be verified.

**Maintenance rule:** When an upstream internal detail changes with a stable interface, consumer should usually benefit on new/verified artifacts. A changed interface may require a small compatibility adapter.

## Decisions in the current supplied implementation

1. **Native data preparation remains native.** `prepare_data.py` calls `harness.dataset.materialize()`. `pack_dataset.py` invokes `scripts.base_prepare` for tokenizer training and packed shards; preparation is intentionally a combined stage.
2. **Native pretraining remains native.** `train.py` delegates to `scripts.base_train`, adding consumer CLI/config selection, artifact checks, run logging, overwrite protection, and an explicit resume option.
3. **Artifact names alone are not identity proofs.** Config-derived names are kept, but SHA-256 locks bind selected Parquet inputs, tokenizer output files, packed payloads/headers, and relevant native preparation source bytes.
4. **Recovery markers are not provenance proofs.** `.consumer_prepare_pending.json` identifies an interrupted preparation attempt; verified output locks are still required before reuse.
5. **Hugging Face is an explicit compatibility port.** `NanoChatRefactorForCausalLM` must mirror native computations and be guarded by weight/logit/cache/generation parity checks. It does not automatically inherit upstream architecture changes.
6. **The current HF port is inference-only and full-attention-only.** `labels` is unsupported and `window_pattern="S"` is not implemented. Do not advertise generic sliding-window compatibility.
7. **Checks are evidence-scoped.** A test of one checkpoint, depth, attention pattern, hardware/software stack, or decoding mode is not proof for all variants.

## Hugging Face findings (recorded earlier)

- The pinned official converter **ran successfully** but skipped seven `experiment_refactor` tensors, and produced different logits from the native model in the tested comparison. This is an observed compatibility gap, **not** an official-converter crash and not a claim about all future versions.
- The consumer exporter was reported to preserve all 33 native tensors and achieve native/HF parity in the specified base-model tests. See [HF_COMPATIBILITY.md](HF_COMPATIBILITY.md) for exact revisions and numerical results.
- Fast tokenizer, stopping tokens, generation, and selected left-padding/cached cases were reported tested; more cache/mask/architecture coverage is still needed.

## Optimizer replay result — reported 2026-11-17

The latest supplied test output states:

```text
Exact updated tensors: 33/33
Maximum absolute difference: 0.000000000000e+00
FROZEN-GRADIENT OPTIMIZER REPLAY: COMPLETED
```

**Interpretation:** two updates using frozen gradients produced equal compared model tensors in that test. This does **not** by itself establish optimizer-buffer equality, forward/backward equivalence, RNG/data-order restoration, or uninterrupted-versus-resumed training equivalence. The test script and full setup were not supplied, so record this as a **reported result**, not a newly rerun independent verification.

## Next work, one focused test at a time

- [ ] Finish `config.py` validation for `[model]` and `[training]` without rejecting native-supported sentinel values.
- [ ] Add focused CPU/no-GPU CLI tests for config selection and safe resume guards.
- [ ] Run controlled interrupted-versus-continuous training equivalence on a disposable experiment (including optimizer state and data/RNG position where possible).
- [ ] Exercise artifact-lock mutation and interrupted-recovery cases in temporary caches.
- [ ] Read native SFT preparation/chat-rendering sources; test assistant-only loss masks before downloads or a GPU smoke run.
- [ ] Implement HF SFT/chat-template export only after native SFT semantics are verified.

**Definition of done:** minimal duplicated code, explicit scope and provenance, safe reuse, and parity claims backed by repeatable evidence.

*Less duplicated logic. More native nanochat.*
