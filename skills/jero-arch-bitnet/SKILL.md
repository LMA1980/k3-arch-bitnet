---
name: jero-arch-bitnet
description: "Implement and validate BitNet (bitnet, bitnet-b1.58, and bitnet-25) ArchRecipe execution, SubLN int8 dynamic quantization, and on-the-fly TQ1_0/TQ2_0 BitLinear forward passes in jero-arch-bitnet."
---

# `jero-arch-bitnet` Architecture Recipe Invariants

- Implement `jero_core::ArchRecipe` for BitNet architectures (`bitnet`, `bitnet-b1.58`, and `bitnet-25`), hydrating hyperparameters via `BitNetRecipe::from_gguf(&GgufFile<'_>)` and validating weights via `validate_model`.
- Enforce canonical BitNet norm and `SubLN` placement cleanly without fallback loops: `blk.{l}.attn_norm.weight` (`[hidden_dim]`) before Q/K/V, `blk.{l}.attn_sub_norm.weight` (`[total_q]`) before `attn_output`, `blk.{l}.ffn_norm.weight` (`[hidden_dim]`) before gate/up, and `blk.{l}.ffn_sub_norm.weight` (`[ffn_dim]`) before `ffn_down`.
- Execute full `0..=position` causal GQA via `ExecutionContext::kv_view_mut` (`KvCacheView`) when `KvGeometry` is attached, respecting optional `sliding_window`.
- Decode `token_embd.weight` across `F32`/`F16`/`BF16`/`Q8_0`/`TQ1_0`/`TQ2_0`/`I2_S`, apply `output_norm.weight` once (no duplicate `SubLN`), and fall back to tied `token_embd.weight` when `output.weight` is absent.
- Support `QuantType::I2_S`, `TQ1_0`, `TQ2_0`, `Q8_0`, `F32`, `F16`, and `BF16` weights while preserving zero heap allocations inside the layer loop.
- Use `usize` for tensor slice indexing and memory offsets, and `u32` for architectural hyperparameters.
