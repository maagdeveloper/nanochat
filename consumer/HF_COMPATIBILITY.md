# Hugging Face Compatibility — experiment_refactor

## Tested versions

- **nanochat:** `experiment_refactor`, commit `07620a2`
- **Transformers:** `5.14.1`
- **Official converter revision:** `3a134bc37f7c22c801b618e289b0255425f22647`
- **Converter SHA-256:** `58a6ca93c5b2344fac613a7dbb75195f102250a7ef6e907e5465eb7df3e51858`
- **Checkpoint:** `nanochatnt_d4_ctx2048_v1`, step 528

## Findings

The official NanoChat converter completes successfully (exit code `0`) and produces a loadable Hugging Face model.

However, it reports skipping seven checkpoint tensors unsupported by its target implementation:

- `value_embeds.1.weight`
- `value_embeds.3.weight`
- `transformer.h.1.attn.ve_gate.weight`
- `transformer.h.3.attn.ve_gate.weight`
- `resid_lambdas`
- `x0_lambdas`
- `backout_lambda`

### Numerical comparison

| Metric | Official converter | nanochatn't |
|---|---:|---:|
| Parameters | 19,922,944 | 36,700,217 |
| Native weight preservation | Incomplete | 33/33 |
| Native logits parity | Failed | Exact in tested cases |
| Native cached logits parity | Not verified | Exact in tested cases |
| Native greedy generation parity | Not verified | Passed |

Across three shared prompts, maximum absolute logits differences between the two exported models ranged from **10.6241 to 12.0502**. Next-token predictions differed on two prompts.

The official conversion also casts model weights to BF16, so this comparison does not isolate the effects of individual architectural differences.

## Conclusion

The tested official converter produces a loadable model, but does not faithfully preserve the `experiment_refactor` architecture.

nanochatn't provides an alternative conversion path for this specific architecture, preserving all 33 native tensors and achieving verified native/HF parity.

These findings apply to the tested revisions and should not be generalized to future Transformers releases or older NanoChat architectures.