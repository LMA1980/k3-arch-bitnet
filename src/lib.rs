//! `jero-arch-bitnet`: `BitNet` architecture recipe for the `SpacemiT` K3 platform.
//!
//! Implements [`jero_core::ArchRecipe`] for Microsoft `BitNet` architectures (`bitnet`, `bitnet-b1.58`,
//! and `bitnet-25`), supporting `SubLN` (`RMSNorm` followed by dynamic per-token int8 `absmax`
//! quantization), `ReLU^2` / `SwiGLU` gated FFNs, and on-the-fly ternary unpacking
//! (`TQ1_0`, `TQ2_0`) for zero-expansion `BitLinear` execution on `A100` (`smt.vmadot`)
//! and `X100` (RVV) compute clusters per ADR-023 and ADR-024.

pub mod recipe;

pub use recipe::BitNetRecipe;

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::{ArchRecipe, ExecutionContext, QuantType};

    #[derive(Debug)]
    struct EmptyWeights;

    impl jero_core::ModelWeights for EmptyWeights {
        fn tensor(&self, name: &str) -> Result<jero_core::TensorView<'_>, jero_core::K3Error> {
            Err(jero_core::K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "not found in dummy weights".to_owned(),
            })
        }
        fn tensor_count(&self) -> usize {
            0
        }
        fn total_weight_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_bitnet_recipe_architecture_ids() {
        let bitnet = BitNetRecipe::new();
        assert_eq!(bitnet.architecture_id(), "bitnet");
        assert_eq!(bitnet.recipe_version(), "1.0.0");
        assert_eq!(bitnet.target_layouts(), &["native", "ime_tile_8x16"]);
        assert!(bitnet.supports_quant(QuantType::Q8_0));
        assert!(bitnet.supports_quant(QuantType::F32));
        assert!(bitnet.supports_quant(QuantType::F16));
        assert!(bitnet.supports_quant(QuantType::BF16));
        assert!(bitnet.use_relu2);

        let b158 = BitNetRecipe::new_b158();
        assert_eq!(b158.architecture_id(), "bitnet-b1.58");
        assert_eq!(b158.recipe_version(), "1.0.0");
        assert!(b158.use_relu2);

        let b25 = BitNetRecipe::new_25();
        assert_eq!(b25.architecture_id(), "bitnet-25");
        assert_eq!(b25.recipe_version(), "1.0.0");
        assert!(!b25.use_relu2);
    }

    #[test]
    fn test_bitnet_recipe_execution_flow() {
        let mut recipe = BitNetRecipe::new();
        recipe.num_layers = 1;
        if cfg!(miri) {
            recipe.hidden_dim = 64;
            recipe.ffn_dim = 128;
            recipe.n_heads_q = 4;
            recipe.n_heads_kv = 2;
            recipe.head_dim = 16;
        }
        let prompt_tokens = [1u32, 2, 3];
        let mut logits = [0.0f32; 16];
        let mut kv_cache = [0.0f32; 64];
        let mut ctx = ExecutionContext::new(&prompt_tokens, 32, &mut logits, &mut kv_cache)
            .expect("valid context");

        let weights = EmptyWeights;
        recipe
            .prefill_step(&mut ctx, &weights)
            .expect("prefill succeeds");
        assert_eq!(ctx.position(), 3);

        ctx.prepare_decode(42).expect("prepare decode succeeds");
        recipe
            .decode_step(&mut ctx, &weights)
            .expect("decode succeeds");
        assert_eq!(ctx.position(), 4);
    }

    #[test]
    fn test_bitnet_recipe_descriptors() {
        let b = BitNetRecipe::new();
        assert_eq!(b.architecture_id(), "bitnet");
        assert_eq!(b.recipe_version(), "1.0.0");
        assert_eq!(b.target_layouts(), &["native", "ime_tile_8x16"]);
        assert!(b.supports_quant(QuantType::TQ1_0));
        assert!(b.supports_quant(QuantType::TQ2_0));
        assert!(b.supports_quant(QuantType::Q8_0));
        assert!(b.supports_quant(QuantType::F32));
        assert!(b.supports_quant(QuantType::F16));
        assert!(b.supports_quant(QuantType::BF16));
        assert!(!b.supports_quant(QuantType::Q4_0));
        assert!(!b.supports_quant(QuantType::Q4_K_M));
        assert!(!b.supports_quant(QuantType::Q6_K));
        assert!(b.use_relu2);

        let b158 = BitNetRecipe::new_b158();
        assert_eq!(b158.architecture_id(), "bitnet-b1.58");
        assert_eq!(b158.recipe_version(), "1.0.0");
        assert!(b158.supports_quant(QuantType::TQ1_0));
        assert!(b158.supports_quant(QuantType::TQ2_0));
        assert!(b158.use_relu2);

        let b25 = BitNetRecipe::new_25();
        assert_eq!(b25.architecture_id(), "bitnet-25");
        assert_eq!(b25.recipe_version(), "1.0.0");
        assert!(b25.supports_quant(QuantType::TQ1_0));
        assert!(b25.supports_quant(QuantType::TQ2_0));
        assert!(!b25.use_relu2);
    }

    #[derive(Debug, Default)]
    struct MapWeights {
        tensors: std::collections::BTreeMap<String, (QuantType, Vec<usize>, Vec<u8>)>,
    }

    impl MapWeights {
        fn insert(&mut self, name: &str, quant: QuantType, shape: Vec<usize>, data: Vec<u8>) {
            self.tensors.insert(name.to_owned(), (quant, shape, data));
        }
    }

    impl jero_core::ModelWeights for MapWeights {
        fn tensor(&self, name: &str) -> Result<jero_core::TensorView<'_>, jero_core::K3Error> {
            let (stored_name, (quant, shape, data)) =
                self.tensors.get_key_value(name).ok_or_else(|| {
                    jero_core::K3Error::TensorBoundsError {
                        tensor_name: name.to_owned(),
                        offset: 0,
                        byte_len: 0,
                        buffer_len: 0,
                        reason: "tensor not found".to_owned(),
                    }
                })?;
            jero_core::TensorView::from_bytes(stored_name.as_str(), *quant, shape, 0, data)
        }

        fn tensor_count(&self) -> usize {
            self.tensors.len()
        }

        fn total_weight_bytes(&self) -> usize {
            self.tensors.values().map(|(_, _, b)| b.len()).sum()
        }
    }

    fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn push_kv_str(buf: &mut Vec<u8>, key: &str, val: &str) {
        push_gguf_str(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes());
        push_gguf_str(buf, val);
    }

    fn push_kv_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
        push_gguf_str(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }

    fn push_kv_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
        push_gguf_str(buf, key);
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }

    fn build_gguf_bytes(
        arch: &str,
        kvs: &[(&str, u32)],
        f32_kvs: &[(&str, f32)],
        tensors: &[(&str, QuantType, &[u64], &[u8])],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&((1 + kvs.len() + f32_kvs.len()) as u64).to_le_bytes());
        push_kv_str(&mut buf, "general.architecture", arch);
        for &(k, v) in kvs {
            push_kv_u32(&mut buf, k, v);
        }
        for &(k, v) in f32_kvs {
            push_kv_f32(&mut buf, k, v);
        }
        let mut rel_offset = 0u64;
        for &(name, quant, shape, data) in tensors {
            push_gguf_str(&mut buf, name);
            buf.extend_from_slice(
                &u32::try_from(shape.len())
                    .expect("tensor rank fits u32")
                    .to_le_bytes(),
            );
            for &dim in shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&quant.ggml_type_id().to_le_bytes());
            buf.extend_from_slice(&rel_offset.to_le_bytes());
            let aligned = data.len().div_ceil(32) * 32;
            rel_offset += aligned as u64;
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for &(_, _, _, data) in tensors {
            buf.extend_from_slice(data);
            while buf.len() % 32 != 0 {
                buf.push(0);
            }
        }
        buf
    }

    fn make_i2s_block_matrix(rows: usize, cols: usize, seed: u8, scale: f32) -> Vec<u8> {
        assert_eq!(cols % 128, 0);
        let blocks_per_row = cols / 128;
        let payload = rows * blocks_per_row * 32;
        let mut out = vec![0u8; payload + 32];
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let off = (r * blocks_per_row + b) * 32;
                for gp in 0..32 {
                    let v = seed
                        .wrapping_add(u8::try_from(r % 251).unwrap_or(0))
                        .wrapping_add(u8::try_from(gp).unwrap_or(0));
                    let c0 = v % 3;
                    let c1 = v.wrapping_add(1) % 3;
                    let c2 = v.wrapping_add(2) % 3;
                    let c3 = v.wrapping_mul(2) % 3;
                    out[off + gp] = (c0 << 6) | (c1 << 4) | (c2 << 2) | c3;
                }
            }
        }
        out[payload..payload + 4].copy_from_slice(&scale.to_le_bytes());
        out
    }

    fn make_f32_norm(dim: usize, val: f32) -> Vec<u8> {
        let mut out = Vec::with_capacity(dim * 4);
        for _ in 0..dim {
            out.extend_from_slice(&val.to_le_bytes());
        }
        out
    }

    #[test]
    fn test_bitnet_recipe_from_gguf_and_validate_model() {
        let zeros_f32 = vec![0u8; 128 * 4 * 4];
        let gguf_bytes = build_gguf_bytes(
            "bitnet-b1.58",
            &[
                ("bitnet-b1.58.block_count", 2),
                ("bitnet-b1.58.embedding_length", 128),
                ("bitnet-b1.58.feed_forward_length", 256),
                ("bitnet-b1.58.attention.head_count", 4),
                ("bitnet-b1.58.attention.head_count_kv", 2),
                ("bitnet-b1.58.vocab_size", 4),
            ],
            &[
                ("bitnet-b1.58.attention.layer_norm_rms_epsilon", 1e-5),
                ("bitnet-b1.58.rope.freq_base", 500_000.0),
            ],
            &[(
                "token_embd.weight",
                QuantType::F32,
                &[128, 4],
                zeros_f32.as_slice(),
            )],
        );
        let gguf = jero_core::GgufFile::parse(&gguf_bytes).expect("parse bitnet gguf");
        let recipe = BitNetRecipe::from_gguf(&gguf).expect("hydrate recipe");
        assert_eq!(recipe.architecture_id(), "bitnet-b1.58");
        assert_eq!(recipe.num_layers, 2);
        assert_eq!(recipe.hidden_dim, 128);
        assert_eq!(recipe.ffn_dim, 256);
        assert_eq!(recipe.n_heads_q, 4);
        assert_eq!(recipe.n_heads_kv, 2);
        assert_eq!(recipe.head_dim, 32);
        assert_eq!(recipe.vocab_size, 4);
        assert!(recipe.use_relu2);
        assert!(recipe.supports_quant(QuantType::I2_S));
        recipe
            .validate_model(gguf.metadata(), &EmptyWeights)
            .expect("validate_model accepts valid metadata");

        let b25_bytes = build_gguf_bytes(
            "bitnet-25",
            &[
                ("bitnet-25.block_count", 1),
                ("bitnet-25.embedding_length", 128),
                ("bitnet-25.feed_forward_length", 128),
                ("bitnet-25.attention.head_count", 4),
                ("bitnet-25.attention.head_count_kv", 2),
                ("bitnet-25.vocab_size", 4),
            ],
            &[],
            &[],
        );
        let gguf25 = jero_core::GgufFile::parse(&b25_bytes).expect("parse bitnet-25 gguf");
        let recipe25 = BitNetRecipe::from_gguf(&gguf25).expect("hydrate bitnet-25");
        assert!(!recipe25.use_relu2);

        let llama_bytes = build_gguf_bytes(
            "llama",
            &[
                ("llama.block_count", 1),
                ("llama.embedding_length", 128),
                ("llama.attention.head_count", 4),
                ("llama.attention.head_count_kv", 2),
            ],
            &[],
            &[],
        );
        let gguf_llama = jero_core::GgufFile::parse(&llama_bytes).expect("parse llama gguf");
        assert!(matches!(
            BitNetRecipe::from_gguf(&gguf_llama),
            Err(jero_core::K3Error::CapabilityMismatch { .. })
        ));
        assert!(matches!(
            recipe.validate_model(gguf_llama.metadata(), &EmptyWeights),
            Err(jero_core::K3Error::CapabilityMismatch { .. })
        ));

        let q4_bytes = build_gguf_bytes(
            "bitnet-b1.58",
            &[
                ("bitnet-b1.58.block_count", 2),
                ("bitnet-b1.58.embedding_length", 128),
                ("bitnet-b1.58.feed_forward_length", 256),
                ("bitnet-b1.58.attention.head_count", 4),
                ("bitnet-b1.58.attention.head_count_kv", 2),
            ],
            &[],
            &[(
                "blk.0.attn_q.weight",
                QuantType::Q4_0,
                &[128, 128],
                &vec![0u8; 128 * 128 / 32 * 18],
            )],
        );
        let gguf_q4 = jero_core::GgufFile::parse(&q4_bytes).expect("parse q4 gguf");
        assert!(matches!(
            recipe.validate_model(gguf_q4.metadata(), &EmptyWeights),
            Err(jero_core::K3Error::CapabilityMismatch { .. })
        ));
    }

    fn build_i2s_unit_weights() -> MapWeights {
        let mut embd_f16 = Vec::with_capacity(4 * 128 * 2);
        for tok in 0u16..4 {
            for idx in 0u16..128 {
                let sign_bit = if ((tok + idx) % 2) == 0 {
                    0u16
                } else {
                    0x8000u16
                };
                let bits = sign_bit | (0x3800u16 + ((tok * 17 + idx) % 5) * 0x0100u16);
                embd_f16.extend_from_slice(&bits.to_le_bytes());
            }
        }

        let mut weights = MapWeights::default();
        weights.insert("token_embd.weight", QuantType::F16, vec![128, 4], embd_f16);
        for (name, dim) in [
            ("output_norm.weight", 128),
            ("blk.0.attn_norm.weight", 128),
            ("blk.0.attn_sub_norm.weight", 128),
            ("blk.0.ffn_norm.weight", 128),
            ("blk.0.ffn_sub_norm.weight", 128),
        ] {
            weights.insert(name, QuantType::F32, vec![dim], make_f32_norm(dim, 1.0));
        }
        for (seed, name, rows, cols) in [
            (1u8, "blk.0.attn_q.weight", 128, 128),
            (2, "blk.0.attn_k.weight", 64, 128),
            (3, "blk.0.attn_v.weight", 64, 128),
            (4, "blk.0.attn_output.weight", 128, 128),
            (5, "blk.0.ffn_gate.weight", 128, 128),
            (6, "blk.0.ffn_up.weight", 128, 128),
            (7, "blk.0.ffn_down.weight", 128, 128),
        ] {
            weights.insert(
                name,
                QuantType::I2_S,
                vec![cols, rows],
                make_i2s_block_matrix(rows, cols, seed, 0.25),
            );
        }
        weights
    }

    #[test]
    fn test_bitnet_recipe_i2s_kv_view_and_tied_output_linear_scale() {
        let mut recipe = BitNetRecipe::new_b158();
        recipe.num_layers = 1;
        recipe.hidden_dim = 128;
        recipe.ffn_dim = 128;
        recipe.n_heads_q = 4;
        recipe.n_heads_kv = 2;
        recipe.head_dim = 32;
        recipe.vocab_size = 4;

        let mut weights = build_i2s_unit_weights();

        let geom = jero_core::KvGeometry::new(1, 2, 32, 4, None).expect("valid kv geometry");
        let run_seq = |tokens: &[u32], w: &MapWeights| -> Vec<f32> {
            let mut logits = [0.0f32; 4];
            let mut kv_buf = vec![0.0f32; geom.required_f32_len().expect("kv len")];
            let mut ctx = ExecutionContext::new(tokens, 4, &mut logits, &mut kv_buf)
                .expect("ctx")
                .with_kv_geometry(geom)
                .expect("attach kv geometry");
            recipe.prefill_step(&mut ctx, w).expect("prefill succeeds");
            logits.to_vec()
        };

        let logits_a = if cfg!(miri) {
            run_seq(&[0], &weights)
        } else {
            let logits_a = run_seq(&[0, 2], &weights);
            let logits_b = run_seq(&[1, 2], &weights);
            assert!(
                logits_a
                    .iter()
                    .zip(&logits_b)
                    .any(|(a, b)| (a - b).abs() > 1e-4)
            );
            logits_a
        };
        assert!(logits_a.iter().all(|v| v.is_finite()));

        if !cfg!(miri) {
            // Scaling output_norm.weight by 2.0 must scale F16 tied LM-head logits by 2.0 (no double RMSNorm).
            weights.insert(
                "output_norm.weight",
                QuantType::F32,
                vec![128],
                make_f32_norm(128, 2.0),
            );
            let logits_scaled = run_seq(&[0, 2], &weights);
            for (base, scaled) in logits_a.iter().zip(&logits_scaled) {
                assert!((scaled - base * 2.0).abs() <= 1e-4);
            }
        }
    }
}
