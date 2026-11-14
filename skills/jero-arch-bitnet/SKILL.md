---
name: jero-arch-bitnet
description: "Implement and validate BitNet (bitnet, bitnet-b1.58, and bitnet-25) ArchRecipe execution, SubLN int8 dynamic quantization, load-time TQ1_0/TQ2_0/I2_S 128-byte 8x16 i8 tile repacking, compact tied token_embd/LM-head storage, and Llama-3 128k stop contracts in jero-arch-bitnet."
---

# `jero-arch-bitnet` Architecture Recipe Invariants

- Implement `jero_core::ArchRecipe` for BitNet architectures (`bitnet`, `bitnet-b1.58`, and `bitnet-25`), hydrating hyperparameters via `BitNetRecipe::from_gguf(&GgufFile<'_>)` and validating weights via `validate_model`.
- Enforce canonical BitNet norm and `SubLN` placement cleanly without fallback loops: `blk.{l}.attn_norm.weight` (`[hidden_dim]`) before Q/K/V, `blk.{l}.attn_sub_norm.weight` (`[total_q]`) before `attn_output`, `blk.{l}.ffn_norm.weight` (`[hidden_dim]`) before gate/up, and `blk.{l}.ffn_sub_norm.weight` (`[ffn_dim]`) before `ffn_down`.
- Repack `TQ1_0`, `TQ2_0`, `I2_S`, and `Q8_0` projection weights once at load time inside `BitNetModelCache` into `128-byte`-aligned `8x16` `i8` tiles alongside cached `f32` `SubLN`/output norms, preserving bit-exact `i32` ternary dot accumulation parity without per-step block re-unpacking.
- Keep tied/untied `128,256 x 2560` `F16`/`BF16`/`Q8_0` `token_embd.weight` / `output.weight` tables in compact `128-byte`-aligned storage (`dense_f32_max_elems = 0`, zero full-table `f32` expansion), and validate `BitNetVocabContract` bounds including Llama-3 `<|eot_id|>` (`128009`) and `<|end_of_text|>` stops.
- Execute full `0..=position` causal GQA via `ExecutionContext::kv_view_mut` (`KvCacheView`) when `KvGeometry` is attached, respecting optional `sliding_window`.
- Apply `output_norm.weight` once (no duplicate `SubLN`), fall back to tied `token_embd.weight` when `output.weight` is absent, and keep zero heap allocations inside the transformer layer loop.
- Use `usize` for tensor slice indexing and memory offsets, and `u32` for architectural hyperparameters and token identifiers.
