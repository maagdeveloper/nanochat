# nanochatn't 🐣 — Under Review

> Internal engineering ledger: observed evidence, current code, limitations, and next checks—not a release checklist.
> Updated: 2026-11-17. Upstream reference recorded for the tested baseline: `experiment_refactor` at `07620a2`.
> This update reviews the **supplied consumer source snapshot** and prior recorded test results. It does not claim a fresh GPU run or inspection of the latest fork HEAD.

## North star

**Consumer owns configuration, artifact identity, and orchestration. Nanochat owns native implementation.**

- **Delegate, don't reproduce.** Reuse native preparation and training scripts; do not fork their algorithms for convenience.
- Keep versioned dataset views and fail-closed reuse checks.
- Do not modify upstream nanochat only to accommodate `consumer/`.
- If upstream changes an internal algorithm without changing its interface, consumer should normally inherit the improvement when artifacts are appropriately regenerated.
- **Explicit exception:** standalone Hugging Face modeling is a compatibility port, not a thin wrapper. Guard it with version-scoped parity tests.
- Make one controlled change and run a focused test. Preserve reference caches and checkpoints until migration is independently checked.

### Evidence/status key

- **HISTORIC VERIFIED** — a passed test documented before this update; not rerun here.
- **REPORTED RESULT** — output supplied in conversation; test code, setup, and full state comparison not available in this snapshot.
- **IMPLEMENTED / UNVERIFIED** — the supplied code has a path/guard, but this review did not execute its integration test.
- **CONFIRMED GAP** — missing behavior or a demonstrated limitation of supplied code.
- **TO TEST / TO REVIEW** — follow-up evidence or design decision needed.
- **INTENTIONAL** — accepted, documented scope restriction.

## 1. Evidence ledger

| Area | Status | Evidence and scope |
| --- | --- | --- |
| Native and consumer cold/warm preparation | HISTORIC VERIFIED | Earlier ledger reports warm reuse, cold bootstrap, and real-data fresh preparation. |
| Packed/tokenizer parity after refactor | HISTORIC VERIFIED | Four `.bin` shards, `tokenizer.pkl`, and `token_bytes.pt` were reported byte-for-byte equal; tokenizer ID **693393400**. Train rows **67,361**, validation rows **22,236**, train token positions **137,955,328**. |
| Native base training | HISTORIC VERIFIED | Depth 4, 2048 context, checkpoint step 528; prior training/evaluation and inference exercised. |
| Native/HF base export | HISTORIC VERIFIED (tested scope) | Reported 33/33 tensors preserved; full/cached logits parity, greedy generation, standalone loading, and selected left-padding tests. Not a general architecture proof. |
| Official HF converter comparison | HISTORIC VERIFIED (pinned revisions) | Official converter completed but skipped seven tensors, and its exported logits differed. See `HF_COMPATIBILITY.md`. |
| Frozen-gradient optimizer replay | REPORTED RESULT | Identical update A/B: **33/33 updated tensors equal**, max absolute difference **0.0**. The test code, exact setup, and optimizer-state comparison were not supplied. |
| Raw-input/config identity mismatch | CONFIRMED GAP (naming) | `plan_artifacts()` derives names from settings/recipes, not source bytes; mutation alone does not change the name. |
| Raw/tokenizer/packed/native locks | IMPLEMENTED / UNVERIFIED (current snapshot) | `artifacts.py` contains SHA-256 manifests and `require_native_provenance()` transitively checks packed, tokenizer, and raw inputs. Full integration was not rerun here. |
| Consumer resume launcher | IMPLEMENTED / UNVERIFIED (current snapshot) | `train.py` supports `--resume-from-step` with checkpoint, log, artifact identity, and metadata guards. No end-to-end resume-equivalence result was supplied. |

**Do not conflate these checks.** Byte-for-byte packing parity, native/HF forward parity, a one-step frozen-gradient optimizer replay, and interrupted-vs-uninterrupted **training** equivalence prove different properties. The final one remains open.

## 2. Current implementation and remaining work

### A. Artifact identity, integrity, and provenance — implemented guards; integration and recovery open

**Present in the supplied code:**

- `plan_artifacts()` defines config-derived tokenizer and packed view names.
- `.consumer_raw_inputs.json` hashes the exact selected Parquet bytes.
- `.consumer_tokenizer.json` binds `tokenizer.pkl`, `token_bytes.pt`, native tokenizer ID, and raw lock.
- `.consumer_packed.json` hashes expected `.bin` files, checks headers/rows, links, and tokenizer lock.
- `.consumer_native.json` binds packed-lock bytes and hashes of relevant native preparation sources; Git HEAD is recorded as context, **not** the equality criterion.
- `consumer.scripts.train` calls `require_native_provenance(cfg)` before proceeding; that check transitively validates the other locks. `--dry-run` is non-mutating, but full-file hashing may be expensive.
- `consumer.scripts.pack_dataset` invokes native `scripts.base_prepare` and uses `.consumer_prepare_pending.json` to recognize an interrupted consumer-initiated preparation. The pending marker is an **intent/recovery marker**, not proof of provenance.

**Still to test/review:**

- [ ] End-to-end negative tests: change raw bytes, tokenizer bytes, packed payload, and a native preparation source; each consuming path must fail closed.
- [ ] Warm reuse, clean cold bootstrap, interrupted preparation recovery, mismatched pending marker, partial files, changed symlinks, and invalid/absent locks in isolated temporary caches.
- [ ] Confirm an existing artifact cannot be silently retroactively blessed. `adopt_verified_raw_inputs(cfg)` creates a lock for a current snapshot; it does **not** independently establish that legacy outputs were generated from those bytes.
- [ ] Measure full SHA-256 verification overhead on larger datasets; decide whether read-only dry-run needs an explicit lightweight inspection mode, **without** bypassing integrity in a real run.
- [ ] Decide whether stable config-derived names plus strict manifests remain the preferred policy versus content-addressed names. Do not rename validated reference caches casually.
- [ ] Verify each entry point, including default and explicit-path export, has an appropriate artifact identity/provenance policy. The explicit-path export currently checks checkpoint metadata/tokenizer presence but does not call `require_native_provenance()`.

### B. Native preparation ownership — architecture implemented

- **IMPLEMENTED:** `prepare_data.py` delegates downloading to `harness.dataset.materialize()`; `pack_dataset.py` delegates tokenizer training and packing to `scripts.base_prepare` via subprocess.
- **INTENTIONAL:** Tokenizer training and packing are one native preparation stage. No duplicated tokenizer/packer implementation should be reintroduced in consumer.
- **HISTORIC VERIFIED:** Earlier cold/warm tests and four-shard byte comparisons support output parity for the tested preparation revision.
- **TO REVIEW:** Keep `prepare_data.py` limited to config, early checks, and download worker selection. Document the native convention that, for custom datasets, the last lexically sorted Parquet is validation.
- **TO REVIEW:** Native `base_prepare.py` is treated as a top-level executable, not an import-safe API. Maintain subprocess delegation unless upstream exposes a supported interface.

### C. Config, training, and resume — launcher added, reproducibility not yet established

- **IMPLEMENTED:** `train.py` accepts `--config` and `--resume-from-step N` (positive integer). It checks native artifact locks and packed headers/tokenizer ID, guards missing resume files, refuses rewind when newer checkpoints exist, and compares selected checkpoint metadata and the run's first log line.
- **IMPLEMENTED:** A new run refuses to overwrite existing logs/model checkpoints; resume appends to the existing log.
- **CONFIRMED GAP:** `config.py` validates required dataset/tokenizer/packing fields but does not comprehensively validate `[model]` and `[training]` or their value types before they are indexed.
- **CONFIRMED RESTRICTION:** `train.py` enforces `model.max_seq_len == packing.row_len`. Review whether native allows a shorter context than the packed row; do not remove the restriction without confirming loader and masking semantics.
- **TO REVIEW:** Training, preparation, and export should document config selection consistently. `export_hf.py` currently reads only the default config in its automatic mode; explicit `--checkpoint`, `--tokenizer-dir`, and `--output` work around that limitation.
- **TO REVIEW:** Decide whether resume validation should also compare `eval_every`, `eval_tokens`, and `save_every`, and inspect what native checkpoint restore actually preserves: model, optimizer state, scheduler/step, RNG, and data cursor/order.
- **TO TEST:** Run a controlled continuous-vs-resume equivalence experiment. Avoid claiming bitwise equality across multiple steps based on the frozen-gradient result.
- **TO REVIEW:** Respect native CLI sentinel values (e.g., a disabled eval interval), instead of blanket positive-integer checks.

### D. Hugging Face compatibility — validated base path, explicit scope limits

- **HISTORIC VERIFIED (pinned):** The official converter at the recorded revision **completed** and produced a loadable model, but skipped seven `experiment_refactor` tensors. The consumer port was reported to preserve 33/33 with tested native/HF forward and cached parity. See `HF_COMPATIBILITY.md`.
- **OPEN FEATURE GAP — HF Sliding Window Support**
  - Current HF port supports `window_pattern="L"` only.
  - Support native `S`, `L`, and mixed window patterns.
  - Match nanochat's `compute_window_sizes()` behavior exactly.
  - Respect `short_seq_len` and per-layer attention windows.
  - Preserve GQA, RoPE, KV Cache, and left-padding behavior.
  - Verify native/HF logits, cached decoding, and generation parity.
  - Test `L`, `S`, and mixed patterns such as `SSSL`.
  - Do not modify upstream nanochat.
  - Complete before declaring generic HF Base Export support.
  - Defer implementation until the HF Export milestone.
- **INTENTIONAL:** Hugging Face `forward(labels=...)` raises `NotImplementedError`: training stays native for now.
- **TO REVIEW:** Fail early for unsupported window patterns or invalid shape/rotary assumptions at configuration/initialization where appropriate.
- **TO TEST:** Additional supported shapes, prefill versus incremental cached decode, multi-step batched left-padding, positional/mask handling, intermediate activations and per-layer KV contents.
- **TO REVIEW:** Checkpoint/export metadata and explicit-path tokenizer matching need a clear verified contract before publication.
- Do **not** generalize the converter finding to other Transformers versions or legacy NanoChat architectures.

### E. SFT / chat workflow — next functional milestone

No SFT wrapper, chat template, or SFT export implementation was provided in this snapshot.

1. [ ] Read native `scripts/chat_prepare.py`, `scripts/chat_train.py`, and conversation-rendering/loss-mask contracts **at the exact fork revision**.
2. [ ] Test token IDs and assistant-only loss mask on a tiny in-memory conversation, with no large dataset download.
3. [ ] Run an isolated native SFT smoke test, preserving base checkpoints.
4. [ ] Validate native chat inference/evaluation.
5. [ ] Implement an exact HF chat template; export SFT checkpoint and recheck native/HF parity.

## 3. Agreed next sequence

1. **Document current state** (this update): keep old test evidence but distinguish it from code inspection and new reported replay output.
2. **Close low-risk config gaps:** validate `[model]` and `[training]`, inspect supported native sentinel semantics, and add no-GPU unit/CLI tests for valid/invalid configs.
3. **Prove resume equivalence:** frozen input/seed/schedule, uninterrupted vs checkpoint+resume; compare model tensors, optimizer buffers/state, step, LR schedule, data position, and RNG as available. Start with the smallest isolated test; never overwrite step-528 reference artifacts.
4. **Audit artifact recovery/provenance** with temporary caches and negative mutation tests before relying on reuse at scale.
5. **Begin native SFT** with data/mask contract tests and a controlled smoke run.
6. **Extend HF parity coverage** (cache, padding, intermediate states, supported variants) before publishing chat weights.
7. **Consider cache-cleanup tooling** only with dry-run, explicit targets, and rollback; preserve reference backup until separately verified.

## 4. Safety and repository bookkeeping

- Historic reference backup: `~/.cache/nanochat-backup-20261115-054456/` (reported previously; **not inspected here**). Same-disk moves do not provide off-disk disaster recovery. Do not delete it.
- Historic shared raw Parquet symlinks were `shard_00000`, `shard_00001`, `shard_00002`, and `shard_06542`; source moves/deletion can break those links.
- Tested settings on the recorded reference run: 3 training shards, vocabulary 32,768, row length 2,048, 2 packing workers, depth 4.
- Keep `consumer/notes.md` for enduring architecture decisions; use **this file** for risks, status changes, and test evidence.
- No automatic downloads, cache deletion, retraining, or modifications to upstream native implementation during documentation/test reviews.

---

**Acceptance question:** *If Karpathy improves a native internal algorithm without changing its interface, will consumer inherit it on a correctly regenerated, verified artifact set?* If not, document the exception.
