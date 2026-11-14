//! M4 Wave 2 integration verification (`#19`–`#23`):
//! - Load-time `TQ1_0`, `TQ2_0`, `I2_S`, and `Q8_0` repacking into `128-byte` `8x16` `i8` tiles
//! - Zero block re-unpacking during repeated prefill/decode execution
//! - Bit-exact `i32` ternary dot accumulation and `f32` logit parity across formats & tile shapes
//! - Compact tied/untied `F16`/`BF16`/`Q8_0` `token_embd` / LM-head storage and memory bounds
//! - ABA-safe model anchor invalidation, live norm mutation, and Llama-3 `128,256` `<|eot_id|>` contracts

use jero_arch_bitnet::{
    BitNetModelCache, BitNetRecipe, BitNetVocabContract, CompactLmHeadMatrix,
    LLAMA3_BITNET_VOCAB_SIZE, LLAMA3_END_OF_TEXT_TOKEN_ID, LLAMA3_EOT_TOKEN_ID,
    RepackedTernaryMatrix,
};
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, K3Error, KvGeometry, ModelWeights, QuantType,
    TensorView,
};
use std::collections::BTreeMap;

#[derive(Default, Clone)]
struct MapWeights(BTreeMap<String, (QuantType, Vec<usize>, Vec<u8>)>);

impl MapWeights {
    fn insert(&mut self, name: &str, quant: QuantType, shape: Vec<usize>, data: Vec<u8>) {
        self.0.insert(name.to_owned(), (quant, shape, data));
    }
}

impl ModelWeights for MapWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let (key, (quant, shape, bytes)) =
            self.0
                .get_key_value(name)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: "tensor not found in MapWeights".to_owned(),
                })?;
        TensorView::from_bytes(key, *quant, shape, 0, bytes)
    }

    fn tensor_count(&self) -> usize {
        self.0.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.0.values().map(|(_, _, b)| b.len()).sum()
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
    let exp = i32::try_from((bits >> 23) & 0xff).unwrap_or(0) - 127 + 15;
    let mant = u16::try_from((bits >> 13) & 0x3ff).unwrap_or(0);
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        sign | (u16::try_from(exp).unwrap_or(0) << 10) | mant
    }
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    u16::try_from(v.to_bits() >> 16).unwrap_or(0)
}

fn make_f32_vec(dim: usize, val: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(dim * 4);
    for _ in 0..dim {
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

fn make_half_table(rows: usize, cols: usize, bf16: bool, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        for c in 0..cols {
            let phase = u16::try_from((r * 7 + c * 3 + seed) % 23).unwrap_or(0);
            let val = (f32::from(phase) - 11.0) * 0.0625;
            let bits = if bf16 {
                f32_to_bf16_bits(val)
            } else {
                f32_to_f16_bits(val)
            };
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    out
}

fn make_q8_0_matrix(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0);
    let bpr = cols / 32;
    let mut out = Vec::with_capacity(rows * bpr * 34);
    for r in 0..rows {
        for b in 0..bpr {
            let s_idx = u16::try_from((r + b + seed) % 5).unwrap_or(0);
            let scale = f32_to_f16_bits(0.0625 + f32::from(s_idx) * 0.03125);
            out.extend_from_slice(&scale.to_le_bytes());
            for k in 0..32 {
                let q = i16::try_from((r * 11 + b * 7 + k * 5 + seed) % 61).unwrap_or(0) - 30;
                out.push(i8::try_from(q).unwrap_or(0).to_le_bytes()[0]);
            }
        }
    }
    out
}

fn make_tq1_0_matrix(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 256, 0);
    let bpr = cols / 256;
    let mut out = Vec::with_capacity(rows * bpr * 54);
    for r in 0..rows {
        for b in 0..bpr {
            let s_idx = u16::try_from((r + b + seed) % 4).unwrap_or(0);
            let scale = f32_to_f16_bits(0.125 + f32::from(s_idx) * 0.0625);
            out.extend_from_slice(&scale.to_le_bytes());
            for p in 0..51 {
                let mut val = 0_u8;
                let mut pow = 1_u8;
                for j in 0..5 {
                    let code = u8::try_from((r * 7 + b * 3 + p + j + seed) % 3).unwrap_or(0);
                    val += code * pow;
                    pow = pow.saturating_mul(3);
                }
                out.push(val);
            }
            out.push(u8::try_from((r + b + seed) % 3).unwrap_or(0));
        }
    }
    out
}

fn make_tq2_0_matrix(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 256, 0);
    let bpr = cols / 256;
    let mut out = Vec::with_capacity(rows * bpr * 66);
    for r in 0..rows {
        for b in 0..bpr {
            let s_idx = u16::try_from((r + b + seed) % 3).unwrap_or(0);
            let scale = f32_to_f16_bits(0.25 + f32::from(s_idx) * 0.125);
            out.extend_from_slice(&scale.to_le_bytes());
            for p in 0..64 {
                let mut byte = 0_u8;
                for k in 0..4 {
                    let code = u8::try_from((r * 5 + b + p + k + seed) % 3).unwrap_or(0);
                    byte |= (code & 0x03) << (k * 2);
                }
                out.push(byte);
            }
        }
    }
    out
}

fn make_i2s_matrix(rows: usize, cols: usize, seed: usize, scale: f32) -> Vec<u8> {
    assert_eq!(cols % 128, 0);
    let bpr = cols / 128;
    let mut out = Vec::with_capacity(rows * bpr * 32 + 32);
    for r in 0..rows {
        for b in 0..bpr {
            for g in 0..32 {
                let mut packed = 0_u8;
                for slot in 0..4 {
                    let code = u8::try_from((r * 3 + b * 5 + g + slot + seed) % 3).unwrap_or(0);
                    packed |= (code & 0x03) << (6 - slot * 2);
                }
                out.push(packed);
            }
        }
    }
    let sbytes = scale.to_le_bytes();
    for _ in 0..8 {
        out.extend_from_slice(&sbytes);
    }
    out
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x3ff);
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mag = f32::from(u16::try_from(mant).unwrap_or(0)) * (1.0_f32 / 16_777_216.0_f32);
            if sign == 1 { -mag } else { mag }
        }
    } else if exp == 31 {
        0.0
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

fn independent_oracle_block_dots(
    quant: QuantType,
    raw: &[u8],
    rows_in_tensor: usize,
    row: usize,
    cols: usize,
    act_i8: &[i8],
) -> Vec<(i32, f32)> {
    match quant {
        QuantType::TQ1_0 => {
            let bpr = cols / 256;
            let mut out = Vec::with_capacity(bpr);
            for b in 0..bpr {
                let off = (row * bpr + b) * 54;
                let scale = f16_bits_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
                let mut w = [0_i8; 256];
                for p in 0..51 {
                    let mut byte = raw[off + 2 + p];
                    for j in 0..5 {
                        w[p * 5 + j] = i8::try_from(byte % 3).unwrap_or(0) - 1;
                        byte /= 3;
                    }
                }
                w[255] = i8::try_from(raw[off + 53] % 3).unwrap_or(0) - 1;
                let dot: i32 = w
                    .iter()
                    .zip(&act_i8[b * 256..(b + 1) * 256])
                    .map(|(&wi, &ai)| i32::from(wi) * i32::from(ai))
                    .sum();
                out.push((dot, scale));
            }
            out
        }
        QuantType::TQ2_0 => {
            let bpr = cols / 256;
            let mut out = Vec::with_capacity(bpr);
            for b in 0..bpr {
                let off = (row * bpr + b) * 66;
                let scale = f16_bits_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
                let mut w = [0_i8; 256];
                for p in 0..64 {
                    let byte = raw[off + 2 + p];
                    for k in 0..4 {
                        w[p * 4 + k] = i8::try_from((byte >> (k * 2)) & 0x03).unwrap_or(0) - 1;
                    }
                }
                let dot: i32 = w
                    .iter()
                    .zip(&act_i8[b * 256..(b + 1) * 256])
                    .map(|(&wi, &ai)| i32::from(wi) * i32::from(ai))
                    .sum();
                out.push((dot, scale));
            }
            out
        }
        QuantType::I2_S => {
            let bpr = cols / 128;
            let p = rows_in_tensor * bpr * 32;
            let scale = f32::from_le_bytes([raw[p], raw[p + 1], raw[p + 2], raw[p + 3]]);
            let mut out = Vec::with_capacity(bpr);
            for b in 0..bpr {
                let off = (row * bpr + b) * 32;
                let mut w = [0_i8; 128];
                for g in 0..32 {
                    let packed = raw[off + g];
                    w[g] = i8::try_from((packed >> 6) & 0x03).unwrap_or(0) - 1;
                    w[32 + g] = i8::try_from((packed >> 4) & 0x03).unwrap_or(0) - 1;
                    w[64 + g] = i8::try_from((packed >> 2) & 0x03).unwrap_or(0) - 1;
                    w[96 + g] = i8::try_from(packed & 0x03).unwrap_or(0) - 1;
                }
                let dot: i32 = w
                    .iter()
                    .zip(&act_i8[b * 128..(b + 1) * 128])
                    .map(|(&wi, &ai)| i32::from(wi) * i32::from(ai))
                    .sum();
                out.push((dot, scale));
            }
            out
        }
        QuantType::Q8_0 => {
            let bpr = cols / 32;
            let mut out = Vec::with_capacity(bpr);
            for b in 0..bpr {
                let off = (row * bpr + b) * 34;
                let scale = f16_bits_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
                let mut dot = 0_i32;
                for k in 0..32 {
                    let wi = i8::from_le_bytes([raw[off + 2 + k]]);
                    dot += i32::from(wi) * i32::from(act_i8[b * 32 + k]);
                }
                out.push((dot, scale));
            }
            out
        }
        other => panic!("unsupported oracle quant {other:?}"),
    }
}

fn build_wave2_two_layer_model(seed: usize) -> (BitNetRecipe, MapWeights) {
    let mut recipe = BitNetRecipe::new_b158();
    recipe.num_layers = 2;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 256;
    recipe.n_heads_q = 4;
    recipe.n_heads_kv = 2;
    recipe.head_dim = 64;
    recipe.vocab_size = 16;

    let mut weights = MapWeights::default();
    weights.insert(
        "token_embd.weight",
        QuantType::F16,
        vec![256, 16],
        make_half_table(16, 256, false, seed),
    );
    weights.insert(
        "output_norm.weight",
        QuantType::F32,
        vec![256],
        make_f32_vec(256, 1.0),
    );

    for l in 0..2 {
        weights.insert(
            &format!("blk.{l}.attn_norm.weight"),
            QuantType::F32,
            vec![256],
            make_f32_vec(256, 1.0),
        );
        weights.insert(
            &format!("blk.{l}.attn_sub_norm.weight"),
            QuantType::F32,
            vec![256],
            make_f32_vec(256, 1.0),
        );
        weights.insert(
            &format!("blk.{l}.ffn_norm.weight"),
            QuantType::F32,
            vec![256],
            make_f32_vec(256, 1.0),
        );
        weights.insert(
            &format!("blk.{l}.ffn_sub_norm.weight"),
            QuantType::F32,
            vec![256],
            make_f32_vec(256, 1.0),
        );

        let ls = seed + l * 31;
        weights.insert(
            &format!("blk.{l}.attn_q.weight"),
            QuantType::TQ1_0,
            vec![256, 256],
            make_tq1_0_matrix(256, 256, ls + 1),
        );
        weights.insert(
            &format!("blk.{l}.attn_k.weight"),
            QuantType::I2_S,
            vec![256, 128],
            make_i2s_matrix(128, 256, ls + 2, 0.25),
        );
        weights.insert(
            &format!("blk.{l}.attn_v.weight"),
            QuantType::I2_S,
            vec![256, 128],
            make_i2s_matrix(128, 256, ls + 3, 0.25),
        );
        weights.insert(
            &format!("blk.{l}.attn_output.weight"),
            QuantType::TQ2_0,
            vec![256, 256],
            make_tq2_0_matrix(256, 256, ls + 4),
        );
        weights.insert(
            &format!("blk.{l}.ffn_gate.weight"),
            QuantType::TQ1_0,
            vec![256, 256],
            make_tq1_0_matrix(256, 256, ls + 5),
        );
        weights.insert(
            &format!("blk.{l}.ffn_up.weight"),
            QuantType::TQ2_0,
            vec![256, 256],
            make_tq2_0_matrix(256, 256, ls + 6),
        );
        weights.insert(
            &format!("blk.{l}.ffn_down.weight"),
            QuantType::I2_S,
            vec![256, 256],
            make_i2s_matrix(256, 256, ls + 7, 0.3125),
        );
    }

    (recipe, weights)
}

#[test]
fn test_wave2_bit_exact_i32_and_f32_ternary_dot_parity_across_tile_boundaries() {
    for &cols in &[256_usize, 512] {
        let act_i8: Vec<i8> = (0..cols)
            .map(|i| {
                let centered = i16::try_from((i * 29 + 11) % 210).unwrap_or(0) - 105;
                i8::try_from(centered).unwrap_or(0)
            })
            .collect();
        let act_scale = 0.015_625_f32;

        for &rows in &[1_usize, 7, 8, 9, 17, 65] {
            for (quant, data) in [
                (QuantType::TQ1_0, make_tq1_0_matrix(rows, cols, 3)),
                (QuantType::TQ2_0, make_tq2_0_matrix(rows, cols, 9)),
                (QuantType::I2_S, make_i2s_matrix(rows, cols, 15, 0.375)),
                (QuantType::Q8_0, make_q8_0_matrix(rows, cols, 21)),
            ] {
                let shape = [cols, rows];
                let view =
                    TensorView::from_bytes("w", quant, &shape, 0, &data).expect("tensor view");
                let repacked =
                    RepackedTernaryMatrix::from_tensor(&view, rows, cols).expect("repack ternary");
                assert!(repacked.is_128_byte_aligned());

                for r in 0..rows {
                    let got = repacked.dot_blocks_i32(r, &act_i8).expect("dot_blocks_i32");
                    let oracle =
                        independent_oracle_block_dots(quant, &data, rows, r, cols, &act_i8);
                    assert_eq!(got.len(), oracle.len());
                    for (blk, ((gd, gs), (od, os))) in got.iter().zip(&oracle).enumerate() {
                        assert_eq!(gd, od, "{quant:?} rows={rows} cols={cols} r={r} blk={blk}");
                        assert_eq!(
                            gs.to_bits(),
                            os.to_bits(),
                            "{quant:?} scale rows={rows} cols={cols} r={r} blk={blk}"
                        );
                    }
                }

                let mut out_repacked = vec![0.0_f32; rows];
                repacked
                    .bitlinear_gemv(&mut out_repacked, &act_i8, act_scale)
                    .expect("bitlinear_gemv");

                for (r, &actual_f32) in out_repacked.iter().enumerate() {
                    let oracle =
                        independent_oracle_block_dots(quant, &data, rows, r, cols, &act_i8);
                    let mut ref_sum = 0.0_f32;
                    for (dot, w_scale) in oracle {
                        #[expect(clippy::cast_precision_loss, reason = "oracle test comparison")]
                        {
                            ref_sum += (dot as f32) * (w_scale * act_scale);
                        }
                    }
                    assert_eq!(
                        actual_f32.to_bits(),
                        ref_sum.to_bits(),
                        "bit-exact f32 mismatch {quant:?} rows={rows} cols={cols} r={r}"
                    );
                }

                #[cfg(feature = "with-kernels")]
                {
                    let mut rvv_out = vec![0.0_f32; rows];
                    let mut ime_out = vec![0.0_f32; rows];
                    jero_kernels_rvv::bitlinear_i8_rvv(&mut rvv_out, &view, &act_i8, act_scale)
                        .expect("bitlinear_i8_rvv");
                    jero_kernels_rvv::bitlinear_i8_ime(&mut ime_out, &view, &act_i8, act_scale)
                        .expect("bitlinear_i8_ime");
                    for r in 0..rows {
                        assert_eq!(out_repacked[r].to_bits(), rvv_out[r].to_bits());
                        assert_eq!(out_repacked[r].to_bits(), ime_out[r].to_bits());
                    }
                }
            }
        }
    }
}

#[test]
fn test_wave2_preload_multi_step_zero_reunpack_and_exact_logit_parity() {
    let (recipe, weights) = build_wave2_two_layer_model(11);
    let mut cache = BitNetModelCache::new();
    recipe.preload_cache(&weights, &mut cache).expect("preload");
    let load_stats = cache.stats();
    assert_eq!(load_stats.norm_decodes, 9);
    assert_eq!(load_stats.weight_repacks, 14);
    assert_eq!(load_stats.lm_head_repacks, 1);
    let initial_unpacks = load_stats.unpack_block_calls;
    assert!(initial_unpacks > 0);

    let geom = KvGeometry::new(2, 2, 64, 16, None).expect("geom");
    let mut kv_cached = vec![0.0_f32; geom.required_f32_len().expect("kv")];
    let mut logits_cached = [0.0_f32; 16];
    let mut ctx_cached = ExecutionContext::new(&[0, 3, 5], 16, &mut logits_cached, &mut kv_cached)
        .expect("ctx")
        .with_kv_geometry(geom)
        .expect("kv view");
    recipe
        .prefill_step_with_cache(&mut ctx_cached, &weights, &mut cache)
        .expect("prefill");

    let mut decode_history = Vec::new();
    for tok in [1_u32, 2, 4, 6] {
        ctx_cached.prepare_decode(tok).expect("stage decode");
        recipe
            .decode_step_with_cache(&mut ctx_cached, &weights, &mut cache)
            .expect("decode");
        decode_history.push(ctx_cached.logits().to_vec());
    }

    let post_stats = cache.stats();
    assert_eq!(post_stats.norm_decodes, load_stats.norm_decodes);
    assert_eq!(post_stats.weight_repacks, load_stats.weight_repacks);
    assert_eq!(post_stats.lm_head_repacks, load_stats.lm_head_repacks);
    assert_eq!(
        post_stats.unpack_block_calls, initial_unpacks,
        "repeated prefill + 4x decode must perform zero incremental block unpacks"
    );
    assert!(post_stats.weight_hits >= 14 * 7);
    assert!(post_stats.norm_hits >= 9 * 7);
    assert!(post_stats.lm_head_hits >= 12);

    // Verify zero-budget streaming oracle produces identical step-by-step logits.
    let mut uncached = BitNetModelCache::with_max_bytes(0);
    let mut kv_ref = vec![0.0_f32; geom.required_f32_len().expect("kv")];
    let mut logits_ref = [0.0_f32; 16];
    let mut ctx_ref = ExecutionContext::new(&[0, 3, 5], 16, &mut logits_ref, &mut kv_ref)
        .expect("ctx ref")
        .with_kv_geometry(geom)
        .expect("kv ref");
    recipe
        .prefill_step_with_cache(&mut ctx_ref, &weights, &mut uncached)
        .expect("prefill ref");
    for (step, &tok) in [1_u32, 2, 4, 6].iter().enumerate() {
        ctx_ref.prepare_decode(tok).expect("stage ref");
        recipe
            .decode_step_with_cache(&mut ctx_ref, &weights, &mut uncached)
            .expect("decode ref");
        for (c, (&a, &b)) in decode_history[step]
            .iter()
            .zip(ctx_ref.logits())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "cached vs uncached logit mismatch step {step} vocab {c}"
            );
        }
    }
    assert!(uncached.stats().budget_skips > 0);
}

#[test]
fn test_wave2_compact_token_embd_and_lm_head_never_expands_f16_bf16_q8() {
    let rows = 32;
    let cols = 128;
    let hidden: Vec<f32> = (0..cols)
        .map(|i| {
            let v = i16::try_from((i * 13) % 31).unwrap_or(0) - 15;
            f32::from(v) * 0.0625
        })
        .collect();

    for (quant, data, is_half) in [
        (QuantType::F16, make_half_table(rows, cols, false, 1), true),
        (QuantType::BF16, make_half_table(rows, cols, true, 2), true),
        (QuantType::Q8_0, make_q8_0_matrix(rows, cols, 3), false),
        (QuantType::I2_S, make_i2s_matrix(rows, cols, 4, 0.25), false),
    ] {
        let shape = [cols, rows];
        let view = TensorView::from_bytes("token_embd.weight", quant, &shape, 0, &data)
            .expect("embd view");
        let (compact, _) =
            CompactLmHeadMatrix::from_tensor_counted(&view, rows, cols).expect("compact");
        assert!(compact.is_128_byte_aligned());
        assert_eq!(compact.is_compact_half(), is_half);
        assert!(compact.is_compact_storage(), "{quant:?} must stay compact");
        assert!(
            compact.aligned_bytes() <= rows * cols.div_ceil(64) * 128 + 128,
            "{quant:?} exceeded compact alignment envelope: {} bytes",
            compact.aligned_bytes()
        );

        let mut emb_row = vec![0.0_f32; cols];
        compact.embed_row(5, &mut emb_row).expect("embed row 5");
        assert!(emb_row.iter().all(|v| v.is_finite()));

        let mut logits = vec![0.0_f32; rows];
        compact
            .project_logits(&mut logits, &hidden)
            .expect("project");
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    // Confirm canonical 128,256 x 2560 F16 compact sizing is ~626 MiB (not 1.31 GiB f32).
    let compact_128k_f16_bytes = LLAMA3_BITNET_VOCAB_SIZE * (2560_usize.div_ceil(64) * 128);
    let expanded_128k_f32_bytes = LLAMA3_BITNET_VOCAB_SIZE * 2560 * 4;
    assert_eq!(compact_128k_f16_bytes, 656_670_720);
    assert!(compact_128k_f16_bytes < expanded_128k_f32_bytes / 2 + 1_000_000);
}

#[test]
fn test_wave2_aba_model_swap_and_live_norm_mutation_invalidation() {
    let (recipe_a, mut weights_a) = build_wave2_two_layer_model(19);
    let (recipe_b, weights_b) = build_wave2_two_layer_model(77);

    let mut cache = BitNetModelCache::new();
    let geom = KvGeometry::new(2, 2, 64, 8, None).expect("geom");
    let run =
        |recipe: &BitNetRecipe, weights: &MapWeights, cache: &mut BitNetModelCache| -> Vec<f32> {
            let mut kv = vec![0.0_f32; geom.required_f32_len().expect("kv")];
            let mut logits = [0.0_f32; 16];
            let mut ctx = ExecutionContext::new(&[1, 2], 8, &mut logits, &mut kv)
                .expect("ctx")
                .with_kv_geometry(geom)
                .expect("kv view");
            recipe
                .prefill_step_with_cache(&mut ctx, weights, cache)
                .expect("prefill");
            logits.to_vec()
        };

    let base_logits = run(&recipe_a, &weights_a, &mut cache);
    let after_a = cache.stats();

    // Mutate output_norm.weight by 2.0x in-place: FNV-1a norm identity detects change while keeping
    // the 14 layer projection matrices cached.
    weights_a.insert(
        "output_norm.weight",
        QuantType::F32,
        vec![256],
        make_f32_vec(256, 2.0),
    );
    let scaled_logits = run(&recipe_a, &weights_a, &mut cache);
    let after_norm_mut = cache.stats();
    assert_eq!(after_norm_mut.norm_decodes, after_a.norm_decodes + 1);
    assert_eq!(after_norm_mut.weight_repacks, after_a.weight_repacks);
    for (b, s) in base_logits.iter().zip(&scaled_logits) {
        assert!((s - b * 2.0).abs() <= 1e-4, "{s} != {b} * 2.0");
    }

    // Switching to model B triggers anchor invalidation and produces distinct B logits.
    let logits_b = run(&recipe_b, &weights_b, &mut cache);
    let after_b = cache.stats();
    assert_eq!(after_b.model_invalidations, 1);
    assert!(
        base_logits
            .iter()
            .zip(&logits_b)
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "model swap must not reuse stale weights"
    );
}

fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

enum GgufTestKv<'a> {
    Str(&'a str),
    U32(u32),
    StrArray(&'a [&'a str]),
}

fn build_synthetic_gguf_kv(arch: &str, kvs: &[(&str, GgufTestKv<'_>)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3_u32.to_le_bytes());
    buf.extend_from_slice(&0_u64.to_le_bytes());
    buf.extend_from_slice(&((kvs.len() + 1) as u64).to_le_bytes());

    push_gguf_str(&mut buf, "general.architecture");
    buf.extend_from_slice(&8_u32.to_le_bytes());
    push_gguf_str(&mut buf, arch);

    for (k, v) in kvs {
        push_gguf_str(&mut buf, k);
        match v {
            GgufTestKv::Str(s) => {
                buf.extend_from_slice(&8_u32.to_le_bytes());
                push_gguf_str(&mut buf, s);
            }
            GgufTestKv::U32(n) => {
                buf.extend_from_slice(&4_u32.to_le_bytes());
                buf.extend_from_slice(&n.to_le_bytes());
            }
            GgufTestKv::StrArray(arr) => {
                buf.extend_from_slice(&9_u32.to_le_bytes());
                buf.extend_from_slice(&8_u32.to_le_bytes());
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                for item in *arr {
                    push_gguf_str(&mut buf, item);
                }
            }
        }
    }
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf
}

#[test]
fn test_wave2_llama3_128k_vocab_and_eot_id_stop_contract() {
    let mut tokens = vec!["t"; LLAMA3_BITNET_VOCAB_SIZE];
    tokens[128_000] = "<|begin_of_text|>";
    tokens[usize::try_from(LLAMA3_END_OF_TEXT_TOKEN_ID).unwrap()] = "<|end_of_text|>";
    tokens[usize::try_from(LLAMA3_EOT_TOKEN_ID).unwrap()] = "<|eot_id|>";

    let raw = build_synthetic_gguf_kv(
        "bitnet-b1.58",
        &[
            (
                "bitnet-b1.58.vocab_size",
                GgufTestKv::U32(u32::try_from(LLAMA3_BITNET_VOCAB_SIZE).unwrap()),
            ),
            ("bitnet-b1.58.block_count", GgufTestKv::U32(30)),
            ("bitnet-b1.58.embedding_length", GgufTestKv::U32(2560)),
            ("bitnet-b1.58.feed_forward_length", GgufTestKv::U32(6912)),
            ("bitnet-b1.58.attention.head_count", GgufTestKv::U32(20)),
            ("bitnet-b1.58.attention.head_count_kv", GgufTestKv::U32(5)),
            ("tokenizer.ggml.model", GgufTestKv::Str("gpt2")),
            ("tokenizer.ggml.tokens", GgufTestKv::StrArray(&tokens)),
            (
                "tokenizer.ggml.eos_token_id",
                GgufTestKv::U32(LLAMA3_END_OF_TEXT_TOKEN_ID),
            ),
            (
                "tokenizer.ggml.eot_token_id",
                GgufTestKv::U32(LLAMA3_EOT_TOKEN_ID),
            ),
        ],
    );
    let gguf = GgufFile::parse(&raw).expect("parse 128k BitNet GGUF");
    let recipe = BitNetRecipe::from_gguf(&gguf).expect("hydrate 128k recipe");
    let contract = recipe.vocab_contract(&gguf).expect("vocab contract");

    assert!(contract.is_llama3_128k);
    assert_eq!(contract.vocab_size, LLAMA3_BITNET_VOCAB_SIZE);
    assert_eq!(contract.eot_piece_token_id, Some(LLAMA3_EOT_TOKEN_ID));
    assert_eq!(
        contract.end_of_text_token_id,
        Some(LLAMA3_END_OF_TEXT_TOKEN_ID)
    );
    contract
        .expect_llama3_128k_stops()
        .expect("Llama-3 128k stops must validate");
    assert!(
        contract
            .validate_stop_token(LLAMA3_EOT_TOKEN_ID)
            .expect("valid eot")
    );
    assert!(
        contract
            .validate_stop_token(LLAMA3_END_OF_TEXT_TOKEN_ID)
            .expect("valid eos")
    );
    assert!(!contract.validate_stop_token(42).expect("valid token"));
    assert!(
        contract.validate_stop_token(128_256).is_err(),
        "OOB token must fail closed"
    );

    // Disagreeing eot_token_id vs <|eot_id|> token string index must fail closed.
    let bad_raw = build_synthetic_gguf_kv(
        "bitnet-b1.58",
        &[
            (
                "bitnet-b1.58.vocab_size",
                GgufTestKv::U32(u32::try_from(LLAMA3_BITNET_VOCAB_SIZE).unwrap()),
            ),
            ("tokenizer.ggml.tokens", GgufTestKv::StrArray(&tokens)),
            ("tokenizer.ggml.eot_token_id", GgufTestKv::U32(128_008)),
        ],
    );
    let bad_gguf = GgufFile::parse(&bad_raw).expect("parse bad gguf");
    assert!(
        BitNetVocabContract::from_gguf(&bad_gguf, LLAMA3_BITNET_VOCAB_SIZE).is_err(),
        "mismatched <|eot_id|> index must fail closed"
    );

    // Truncated token_embd.weight row count must fail closed on weight validation.
    let mut bad_weights = MapWeights::default();
    bad_weights.insert(
        "token_embd.weight",
        QuantType::F16,
        vec![256, 4],
        make_half_table(4, 256, false, 1),
    );
    assert!(contract.validate_weights(&bad_weights, 256).is_err());
}

#[test]
fn test_wave2_preload_global_arch_recipe_zero_reunpack() {
    BitNetModelCache::clear_global();
    let (recipe, weights) = build_wave2_two_layer_model(29);
    BitNetModelCache::preload_global(&recipe, &weights).expect("preload global");
    let load_stats = BitNetModelCache::global_stats();
    assert_eq!(load_stats.norm_decodes, 9);
    assert_eq!(load_stats.weight_repacks, 14);
    assert_eq!(load_stats.lm_head_repacks, 1);
    let initial_unpacks = load_stats.unpack_block_calls;
    assert!(initial_unpacks > 0);

    let geom = KvGeometry::new(2, 2, 64, 16, None).expect("geom");
    let mut kv = vec![0.0_f32; geom.required_f32_len().expect("kv")];
    let mut logits = [0.0_f32; 16];
    let mut ctx = ExecutionContext::new(&[1, 4], 16, &mut logits, &mut kv)
        .expect("ctx")
        .with_kv_geometry(geom)
        .expect("kv view");

    recipe
        .prefill_step(&mut ctx, &weights)
        .expect("global prefill");
    ctx.prepare_decode(7).expect("stage 7");
    recipe
        .decode_step(&mut ctx, &weights)
        .expect("global decode 7");
    ctx.prepare_decode(9).expect("stage 9");
    recipe
        .decode_step(&mut ctx, &weights)
        .expect("global decode 9");

    let post_stats = BitNetModelCache::global_stats();
    assert_eq!(post_stats.unpack_block_calls, initial_unpacks);
    assert_eq!(post_stats.norm_decodes, load_stats.norm_decodes);
    assert_eq!(post_stats.weight_repacks, load_stats.weight_repacks);
    assert!(post_stats.weight_hits >= 14 * 4);

    // Compare against uncached oracle.
    let mut oracle_cache = BitNetModelCache::with_max_bytes(0);
    let mut kv_ref = vec![0.0_f32; geom.required_f32_len().expect("kv")];
    let mut logits_ref = [0.0_f32; 16];
    let mut ctx_ref = ExecutionContext::new(&[1, 4], 16, &mut logits_ref, &mut kv_ref)
        .expect("ctx ref")
        .with_kv_geometry(geom)
        .expect("kv ref");
    recipe
        .prefill_step_with_cache(&mut ctx_ref, &weights, &mut oracle_cache)
        .expect("oracle prefill");
    ctx_ref.prepare_decode(7).expect("stage 7 ref");
    recipe
        .decode_step_with_cache(&mut ctx_ref, &weights, &mut oracle_cache)
        .expect("oracle decode 7");
    ctx_ref.prepare_decode(9).expect("stage 9 ref");
    recipe
        .decode_step_with_cache(&mut ctx_ref, &weights, &mut oracle_cache)
        .expect("oracle decode 9");

    for (i, (&a, &b)) in ctx.logits().iter().zip(ctx_ref.logits()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "global cache logit mismatch {i}");
    }
    BitNetModelCache::clear_global();
    assert_eq!(BitNetModelCache::global_stats().resident_bytes, 0);
}
