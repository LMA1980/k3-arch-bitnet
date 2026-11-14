//! M4 Wave 3 integration verification (`#24`–`#28`):
//! - Batched `[M, hidden_dim]` prefill GEMM across cached/streaming `TQ1_0`, `TQ2_0`, `I2_S`, `Q8_0`
//! - Shared `BatchedSubLnActivations` (`1x` Pre-QKV across Q/K/V, `1x` Pre-FFN across gate/up)
//! - Proportional `XSMTVDotII` `8x16` tile invocation scaling (`ime_vmadot_count()`)
//! - Batched `NeoX` `RoPE` (`rope_theta`), `20:5` GQA causal prefill, sliding window, & `KvCacheView`
//! - Numerical equivalence (`max_abs_diff < 1e-4`) across batched, chunked, & sequential decode

use jero_arch_bitnet::{
    BatchedSubLnActivations, BitNetModelCache, BitNetRecipe, RepackedTernaryMatrix,
};
use jero_core::{
    ArchRecipe, ExecutionContext, K3Error, KvCacheView, KvGeometry, ModelWeights, QuantType,
    TensorView,
};
use std::collections::BTreeMap;
use std::sync::RwLock;

static IME_LOCK: RwLock<()> = RwLock::new(());

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

fn make_f32_norm_vec(dim: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(dim * 4);
    for c in 0..dim {
        let phase = u16::try_from((c * 13 + seed) % 17).unwrap_or(0);
        let val = 0.75 + f32::from(phase) * 0.03125;
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

fn make_matrix_for_quant(quant: QuantType, rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    match quant {
        QuantType::TQ1_0 => make_tq1_0_matrix(rows, cols, seed),
        QuantType::TQ2_0 => make_tq2_0_matrix(rows, cols, seed),
        QuantType::I2_S => make_i2s_matrix(rows, cols, seed, 0.25),
        QuantType::Q8_0 => make_q8_0_matrix(rows, cols, seed),
        other => panic!("unexpected test quant {other:?}"),
    }
}

fn make_subln_inputs(m: usize, cols: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
    let mut input = Vec::with_capacity(m * cols);
    for r in 0..m {
        for c in 0..cols {
            let phase = i16::try_from((r * 19 + c * 11 + seed) % 41).unwrap_or(0) - 20;
            input.push(f32::from(phase) * 0.0625 + 0.015_625);
        }
    }
    let mut norm_weight = Vec::with_capacity(cols);
    for c in 0..cols {
        let phase = u16::try_from((c * 7 + seed) % 11).unwrap_or(0);
        norm_weight.push(0.8 + f32::from(phase) * 0.04);
    }
    (input, norm_weight)
}

fn insert_gqa_layer_weights(
    weights: &mut MapWeights,
    layer: usize,
    recipe: &BitNetRecipe,
    uniform_sibling_quants: bool,
    seed: usize,
) {
    let total_q = recipe.n_heads_q * recipe.head_dim;
    let total_kv = recipe.n_heads_kv * recipe.head_dim;
    let hidden = recipe.hidden_dim;
    let ffn = recipe.ffn_dim;
    let ls = seed + layer * 47;

    weights.insert(
        &format!("blk.{layer}.attn_norm.weight"),
        QuantType::F32,
        vec![hidden],
        make_f32_norm_vec(hidden, ls + 1),
    );
    weights.insert(
        &format!("blk.{layer}.attn_sub_norm.weight"),
        QuantType::F32,
        vec![total_q],
        make_f32_norm_vec(total_q, ls + 2),
    );
    weights.insert(
        &format!("blk.{layer}.ffn_norm.weight"),
        QuantType::F32,
        vec![hidden],
        make_f32_norm_vec(hidden, ls + 3),
    );
    weights.insert(
        &format!("blk.{layer}.ffn_sub_norm.weight"),
        QuantType::F32,
        vec![ffn],
        make_f32_norm_vec(ffn, ls + 4),
    );

    let (k_quant, v_quant, gate_quant, up_quant) = if uniform_sibling_quants {
        (
            QuantType::TQ1_0,
            QuantType::TQ1_0,
            QuantType::TQ2_0,
            QuantType::TQ2_0,
        )
    } else {
        (
            QuantType::I2_S,
            QuantType::TQ2_0,
            QuantType::TQ1_0,
            QuantType::I2_S,
        )
    };

    weights.insert(
        &format!("blk.{layer}.attn_q.weight"),
        QuantType::TQ1_0,
        vec![hidden, total_q],
        make_tq1_0_matrix(total_q, hidden, ls + 10),
    );
    weights.insert(
        &format!("blk.{layer}.attn_k.weight"),
        k_quant,
        vec![hidden, total_kv],
        make_matrix_for_quant(k_quant, total_kv, hidden, ls + 11),
    );
    weights.insert(
        &format!("blk.{layer}.attn_v.weight"),
        v_quant,
        vec![hidden, total_kv],
        make_matrix_for_quant(v_quant, total_kv, hidden, ls + 12),
    );
    // `total_q = 320` is 32-aligned, exercising `Q8_0` batched GEMM alongside ternary layers.
    weights.insert(
        &format!("blk.{layer}.attn_output.weight"),
        QuantType::Q8_0,
        vec![total_q, hidden],
        make_q8_0_matrix(hidden, total_q, ls + 13),
    );
    weights.insert(
        &format!("blk.{layer}.ffn_gate.weight"),
        gate_quant,
        vec![hidden, ffn],
        make_matrix_for_quant(gate_quant, ffn, hidden, ls + 14),
    );
    weights.insert(
        &format!("blk.{layer}.ffn_up.weight"),
        up_quant,
        vec![hidden, ffn],
        make_matrix_for_quant(up_quant, ffn, hidden, ls + 15),
    );
    weights.insert(
        &format!("blk.{layer}.ffn_down.weight"),
        QuantType::I2_S,
        vec![ffn, hidden],
        make_i2s_matrix(hidden, ffn, ls + 16, 0.3125),
    );
}

fn build_20_5_gqa_model(
    swiglu: bool,
    untied_output: bool,
    uniform_sibling_quants: bool,
    seed: usize,
) -> (BitNetRecipe, MapWeights) {
    let mut recipe = if swiglu {
        BitNetRecipe::new_25()
    } else {
        BitNetRecipe::new_b158()
    };
    recipe.num_layers = 2;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 256;
    recipe.n_heads_q = 20;
    recipe.n_heads_kv = 5;
    recipe.head_dim = 16;
    recipe.vocab_size = 32;
    recipe.rope_theta = 500_000.0;

    let mut weights = MapWeights::default();
    weights.insert(
        "token_embd.weight",
        QuantType::F16,
        vec![recipe.hidden_dim, recipe.vocab_size],
        make_half_table(recipe.vocab_size, recipe.hidden_dim, false, seed),
    );
    if untied_output {
        weights.insert(
            "output.weight",
            QuantType::BF16,
            vec![recipe.hidden_dim, recipe.vocab_size],
            make_half_table(recipe.vocab_size, recipe.hidden_dim, true, seed + 500),
        );
    }
    weights.insert(
        "output_norm.weight",
        QuantType::F32,
        vec![recipe.hidden_dim],
        make_f32_norm_vec(recipe.hidden_dim, seed + 1),
    );

    for layer in 0..recipe.num_layers {
        insert_gqa_layer_weights(&mut weights, layer, &recipe, uniform_sibling_quants, seed);
    }

    (recipe, weights)
}

struct ModelHarness<'a> {
    recipe: &'a BitNetRecipe,
    weights: &'a MapWeights,
    cache: Option<BitNetModelCache>,
    n_ctx: usize,
    geom: Option<KvGeometry>,
}

impl<'a> ModelHarness<'a> {
    fn new(
        recipe: &'a BitNetRecipe,
        weights: &'a MapWeights,
        use_cache: bool,
        n_ctx: usize,
        geom: Option<KvGeometry>,
    ) -> Self {
        let cache = if use_cache {
            let mut c = BitNetModelCache::new();
            recipe.preload_cache(weights, &mut c).expect("preload");
            Some(c)
        } else {
            None
        };
        Self {
            recipe,
            weights,
            cache,
            n_ctx,
            geom,
        }
    }

    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        want_logits: bool,
        kv_buf: &mut [f32],
    ) -> Vec<f32> {
        let mut logits = if want_logits {
            vec![0.0_f32; self.recipe.vocab_size]
        } else {
            Vec::new()
        };
        let base = ExecutionContext::new_prefill_chunk(
            tokens,
            self.n_ctx,
            start_position,
            want_logits,
            &mut logits,
            kv_buf,
        )
        .expect("prefill chunk ctx");
        let mut ctx = match self.geom {
            Some(g) => base.with_kv_geometry(g).expect("attach kv geometry"),
            None => base,
        };
        if let Some(ref mut cached) = self.cache {
            self.recipe
                .prefill_step_with_cache(&mut ctx, self.weights, cached)
                .expect("cached prefill");
        } else {
            self.recipe
                .prefill_step(&mut ctx, self.weights)
                .expect("prefill");
        }
        logits
    }

    fn decode_sequence(
        &mut self,
        tokens: &[u32],
        start_position: usize,
        kv_buf: &mut [f32],
    ) -> Vec<Vec<f32>> {
        let mut per_step_logits = Vec::with_capacity(tokens.len());
        for (step_idx, &tok) in tokens.iter().enumerate() {
            let seed_arr = [tok];
            let mut step_logits = vec![0.0_f32; self.recipe.vocab_size];
            let base = ExecutionContext::new(&seed_arr, self.n_ctx, &mut step_logits, kv_buf)
                .expect("decode ctx")
                .with_position(start_position + step_idx)
                .expect("decode position");
            let mut ctx = match self.geom {
                Some(g) => base.with_kv_geometry(g).expect("decode kv geometry"),
                None => base,
            };
            ctx.prepare_decode(tok).expect("prepare decode");
            if let Some(ref mut cached) = self.cache {
                self.recipe
                    .decode_step_with_cache(&mut ctx, self.weights, cached)
                    .expect("cached decode");
            } else {
                self.recipe
                    .decode_step(&mut ctx, self.weights)
                    .expect("decode");
            }
            per_step_logits.push(step_logits);
        }
        per_step_logits
    }

    fn sequential_prefill_then_decode(&mut self, prompt: &[u32], kv_buf: &mut [f32]) -> Vec<f32> {
        let mut logits = self.prefill_chunk(&prompt[..1], 0, true, kv_buf);
        let remaining = self.decode_sequence(&prompt[1..], 1, kv_buf);
        if let Some(last) = remaining.into_iter().last() {
            logits = last;
        }
        logits
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn assert_all_finite_nontrivial(vals: &[f32], label: &str) {
    assert!(vals.iter().all(|v| v.is_finite()), "{label} had non-finite");
    let peak = vals.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(peak > 1e-4, "{label} collapsed near zero (peak={peak})");
}

fn assert_kv_buffers_equivalent(
    geom: Option<KvGeometry>,
    num_layers: usize,
    total_positions: usize,
    buf_first: &mut [f32],
    buf_second: &mut [f32],
    buf_oracle: &mut [f32],
) {
    if let Some(g) = geom {
        let mut view_first = KvCacheView::new(buf_first, g).expect("view first");
        let mut view_second = KvCacheView::new(buf_second, g).expect("view second");
        let mut view_oracle = KvCacheView::new(buf_oracle, g).expect("view oracle");
        let kv_dim = g.kv_dim().expect("kv_dim");
        let active_elems = total_positions * kv_dim;
        for layer in 0..num_layers {
            let (first_k, first_v) = view_first.layer_kv_mut(layer).expect("kv first");
            let (second_k, second_v) = view_second.layer_kv_mut(layer).expect("kv second");
            let (oracle_k, oracle_v) = view_oracle.layer_kv_mut(layer).expect("kv oracle");
            let batched_key_diff =
                max_abs_diff(&first_k[..active_elems], &oracle_k[..active_elems]);
            let batched_val_diff =
                max_abs_diff(&first_v[..active_elems], &oracle_v[..active_elems]);
            let chunked_key_diff =
                max_abs_diff(&second_k[..active_elems], &oracle_k[..active_elems]);
            let chunked_val_diff =
                max_abs_diff(&second_v[..active_elems], &oracle_v[..active_elems]);
            assert!(
                batched_key_diff < 1e-4,
                "layer {layer} K batched-seq diff={batched_key_diff}"
            );
            assert!(
                batched_val_diff < 1e-4,
                "layer {layer} V batched-seq diff={batched_val_diff}"
            );
            assert!(
                chunked_key_diff < 1e-4,
                "layer {layer} K chunked-seq diff={chunked_key_diff}"
            );
            assert!(
                chunked_val_diff < 1e-4,
                "layer {layer} V chunked-seq diff={chunked_val_diff}"
            );
        }
    } else {
        let raw_chunk_diff = max_abs_diff(buf_first, buf_second);
        let raw_oracle_diff = max_abs_diff(buf_first, buf_oracle);
        assert!(
            raw_chunk_diff < 1e-4,
            "raw KV chunked diff={raw_chunk_diff}"
        );
        assert!(
            raw_oracle_diff < 1e-4,
            "raw KV oracle diff={raw_oracle_diff}"
        );
    }
}

#[test]
fn test_wave3_bitlinear_gemm_non_aligned_tiles_and_formats_parity() {
    let _shared = IME_LOCK
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for &cols in &[256_usize, 512] {
        for &m in &[1_usize, 7, 8, 9, 17] {
            let (input, norm_weight) = make_subln_inputs(m, cols, m * 17 + cols);
            let mut act = BatchedSubLnActivations::from_subln(&input, &norm_weight, m, cols, 1e-5)
                .expect("subln batch");
            assert_eq!(act.m(), m);
            assert_eq!(act.cols(), cols);
            assert_eq!(act.m_tiles(), m.div_ceil(8));

            for &rows in &[1_usize, 7, 8, 9, 17, 65] {
                for quant in [
                    QuantType::TQ1_0,
                    QuantType::TQ2_0,
                    QuantType::I2_S,
                    QuantType::Q8_0,
                ] {
                    let data = make_matrix_for_quant(quant, rows, cols, rows * 13 + cols);
                    let shape = [cols, rows];
                    let view =
                        TensorView::from_bytes("w", quant, &shape, 0, &data).expect("tensor view");
                    let repacked =
                        RepackedTernaryMatrix::from_tensor(&view, rows, cols).expect("repack");

                    let mut gemm_out = vec![0.0_f32; m * rows];
                    repacked
                        .bitlinear_gemm(&mut gemm_out, &mut act)
                        .expect("bitlinear_gemm");

                    let mut gemv_row = vec![0.0_f32; rows];
                    for tok in 0..m {
                        repacked
                            .bitlinear_gemv(&mut gemv_row, act.row_i8(tok), act.row_scale(tok))
                            .expect("bitlinear_gemv");
                        let row_slice = &gemm_out[tok * rows..(tok + 1) * rows];
                        let diff = max_abs_diff(row_slice, &gemv_row);
                        assert!(
                            diff < 1e-5,
                            "{quant:?} m={m} rows={rows} cols={cols} tok={tok} diff={diff}"
                        );
                    }
                }
            }
        }
    }

    // Sibling matrices sharing identical block geometry pack activations exactly once.
    let (input, norm_weight) = make_subln_inputs(9, 256, 73);
    let mut shared_act =
        BatchedSubLnActivations::from_subln(&input, &norm_weight, 9, 256, 1e-5).expect("act");
    for proj in 0..3 {
        let data = make_tq1_0_matrix(64, 256, proj * 19 + 5);
        let view =
            TensorView::from_bytes("sib", QuantType::TQ1_0, &[256, 64], 0, &data).expect("view");
        let repacked = RepackedTernaryMatrix::from_tensor(&view, 64, 256).expect("repack");
        let mut out = vec![0.0_f32; 9 * 64];
        repacked
            .bitlinear_gemm(&mut out, &mut shared_act)
            .expect("shared gemm");
    }
    assert_eq!(shared_act.pack_calls(), 1);
}

#[cfg(feature = "with-kernels")]
#[test]
fn test_wave3_gemm_increments_ime_vmadot_count_proportionally() {
    let _guard = IME_LOCK
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let cols = 256_usize;
    let expected_k_chunks = cols / 16;

    for quant in [
        QuantType::TQ1_0,
        QuantType::TQ2_0,
        QuantType::I2_S,
        QuantType::Q8_0,
    ] {
        for &m in &[1_usize, 8, 9, 17] {
            let (input, norm_weight) = make_subln_inputs(m, cols, m * 37);
            let mut act = BatchedSubLnActivations::from_subln(&input, &norm_weight, m, cols, 1e-5)
                .expect("subln batch");

            for &rows in &[8_usize, 16, 17] {
                let data = make_matrix_for_quant(quant, rows, cols, rows * 11 + m);
                let shape = [cols, rows];
                let view = TensorView::from_bytes("ime_w", quant, &shape, 0, &data).expect("view");
                let repacked =
                    RepackedTernaryMatrix::from_tensor(&view, rows, cols).expect("repack");
                let mut out = vec![0.0_f32; m * rows];

                let before = jero_kernels_rvv::ime::ime_vmadot_count();
                repacked
                    .bitlinear_gemm(&mut out, &mut act)
                    .expect("bitlinear_gemm");
                let after = jero_kernels_rvv::ime::ime_vmadot_count();

                let expected_tiles =
                    u64::try_from(m.div_ceil(8) * rows.div_ceil(8) * expected_k_chunks)
                        .expect("tile count fits u64");
                assert_eq!(
                    after - before,
                    expected_tiles,
                    "{quant:?} m={m} rows={rows} cols={cols}: ime_vmadot_count delta mismatch"
                );
            }
        }
    }
}

#[test]
fn test_wave3_prefill_step_with_stats_subln_dedup_and_final_token_lm_head() {
    let _shared = IME_LOCK
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tokens: Vec<u32> = (0..17)
        .map(|i| u32::try_from((i * 5 + 3) % 32).unwrap_or(0))
        .collect();

    for (untied, uniform_siblings) in [(true, true), (false, false)] {
        let (recipe, weights) = build_20_5_gqa_model(false, untied, uniform_siblings, 42);
        let geom = KvGeometry::new(
            recipe.num_layers,
            recipe.n_heads_kv,
            recipe.head_dim,
            32,
            None,
        )
        .expect("geom");

        let mut cached = BitNetModelCache::new();
        recipe
            .preload_cache(&weights, &mut cached)
            .expect("preload");
        let mut streaming = BitNetModelCache::with_max_bytes(0);

        for &m in &[1_usize, 7, 8, 9, 17] {
            let prompt = &tokens[..m];
            let mut kv_cached = vec![0.0_f32; geom.required_f32_len().expect("kv")];
            let mut logits_cached = vec![0.0_f32; recipe.vocab_size];
            let mut ctx_cached =
                ExecutionContext::new(prompt, 32, &mut logits_cached, &mut kv_cached)
                    .expect("ctx")
                    .with_kv_geometry(geom)
                    .expect("geom");

            let stats_cached = recipe
                .prefill_step_with_stats(&mut ctx_cached, &weights, Some(&mut cached))
                .expect("prefill cached");

            assert_eq!(stats_cached.prompt_tokens, m);
            assert_eq!(stats_cached.pre_qkv_subln_calls, recipe.num_layers);
            assert_eq!(stats_cached.pre_o_subln_calls, recipe.num_layers);
            assert_eq!(stats_cached.pre_ffn_subln_calls, recipe.num_layers);
            assert_eq!(stats_cached.pre_down_subln_calls, recipe.num_layers);
            assert_eq!(stats_cached.total_subln_calls(), recipe.num_layers * 4);
            assert_eq!(stats_cached.projection_calls, recipe.num_layers * 7);
            assert_eq!(stats_cached.lm_head_tokens_projected, 1);
            if uniform_siblings {
                assert_eq!(
                    stats_cached.tile_pack_calls,
                    stats_cached.total_subln_calls()
                );
            } else {
                assert!(stats_cached.tile_pack_calls >= stats_cached.total_subln_calls());
            }

            let mut kv_stream = vec![0.0_f32; geom.required_f32_len().expect("kv")];
            let mut logits_stream = vec![0.0_f32; recipe.vocab_size];
            let mut ctx_stream =
                ExecutionContext::new(prompt, 32, &mut logits_stream, &mut kv_stream)
                    .expect("ctx")
                    .with_kv_geometry(geom)
                    .expect("geom");
            let stats_stream = recipe
                .prefill_step_with_stats(&mut ctx_stream, &weights, Some(&mut streaming))
                .expect("prefill stream");
            assert_eq!(stats_stream.pre_qkv_subln_calls, recipe.num_layers);
            assert_eq!(stats_stream.pre_ffn_subln_calls, recipe.num_layers);
            assert_eq!(stats_stream.lm_head_tokens_projected, 1);

            let diff = max_abs_diff(&logits_cached, &logits_stream);
            assert!(
                diff < 1e-5,
                "cached vs streaming prefill mismatch (untied={untied}, m={m}): diff={diff}"
            );
            assert_all_finite_nontrivial(&logits_cached, "prefill logits");
        }

        // `compute_logits = false` skips `output_norm` + `project_lm_head` cleanly.
        let mut kv_skip = vec![0.0_f32; geom.required_f32_len().expect("kv")];
        let mut ctx_skip =
            ExecutionContext::new_prefill_chunk(&tokens[..8], 32, 0, false, &mut [], &mut kv_skip)
                .expect("chunk ctx")
                .with_kv_geometry(geom)
                .expect("geom");
        let skip_stats = recipe
            .prefill_step_with_stats(&mut ctx_skip, &weights, Some(&mut cached))
            .expect("skip logits prefill");
        assert_eq!(skip_stats.lm_head_tokens_projected, 0);
        assert_eq!(skip_stats.projection_calls, recipe.num_layers * 7);
    }
}

#[test]
fn test_wave3_compute_logits_false_preserves_reused_buffer_and_overflow_preserves_kv() {
    let _shared = IME_LOCK
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (recipe, weights) = build_20_5_gqa_model(true, true, false, 19);
    let mut cached = BitNetModelCache::new();
    recipe
        .preload_cache(&weights, &mut cached)
        .expect("preload cache");
    let tokens: Vec<u32> = (0..8)
        .map(|i| u32::try_from((i * 5 + 1) % recipe.vocab_size).unwrap_or(0))
        .collect();
    let n_ctx = 32_usize;
    let geom = KvGeometry::new(
        recipe.num_layers,
        recipe.n_heads_kv,
        recipe.head_dim,
        n_ctx,
        None,
    )
    .expect("geom");

    let sentinel_val = -123.5_f32;
    for use_cached in [true, false] {
        let mut sentinel_logits = vec![sentinel_val; recipe.vocab_size];
        let mut kv_sentinel = vec![0.0_f32; geom.required_f32_len().expect("kv")];
        let mut ctx_sentinel = ExecutionContext::new_prefill_chunk(
            &tokens,
            n_ctx,
            0,
            false,
            &mut sentinel_logits,
            &mut kv_sentinel,
        )
        .expect("chunk sentinel ctx")
        .with_kv_geometry(geom)
        .expect("geom");
        let stats = recipe
            .prefill_step_with_stats(
                &mut ctx_sentinel,
                &weights,
                use_cached.then_some(&mut cached),
            )
            .expect("skip logits with reused buffer");
        assert_eq!(stats.lm_head_tokens_projected, 0);
        assert_eq!(stats.projection_calls, recipe.num_layers * 7);
        assert_eq!(ctx_sentinel.position(), tokens.len());
        assert!(
            sentinel_logits
                .iter()
                .all(|&v| v.to_bits() == sentinel_val.to_bits()),
            "compute_logits=false overwrote reused logits buffer (cached={use_cached})"
        );
    }

    // `start_pos + prompt_len > n_ctx` fails before mutating KV storage.
    let small_n_ctx = 8_usize;
    let small_geom = KvGeometry::new(
        recipe.num_layers,
        recipe.n_heads_kv,
        recipe.head_dim,
        small_n_ctx,
        None,
    )
    .expect("small geom");
    let mut kv_guard = vec![0.0_f32; small_geom.required_f32_len().expect("kv")];
    let mut guard_logits = vec![sentinel_val; recipe.vocab_size];
    let mut ctx_overflow =
        ExecutionContext::new(&tokens[..5], small_n_ctx, &mut guard_logits, &mut kv_guard)
            .expect("first chunk ctx")
            .with_kv_geometry(small_geom)
            .expect("geom");
    let _ = recipe
        .prefill_step_with_stats(&mut ctx_overflow, &weights, Some(&mut cached))
        .expect("first 5 tokens fit within n_ctx=8");
    assert_eq!(ctx_overflow.position(), 5);
    let kv_after_first = ctx_overflow.kv_cache_mut().to_vec();

    let overflow_err = recipe
        .prefill_step_with_stats(&mut ctx_overflow, &weights, Some(&mut cached))
        .expect_err("second 5-token prefill from pos=5 overflows n_ctx=8");
    assert!(matches!(
        overflow_err,
        K3Error::ContextLengthExceeded {
            requested_tokens: 10,
            max_context_tokens: 8,
        }
    ));
    assert_eq!(ctx_overflow.position(), 5);
    assert!(
        kv_guard
            .iter()
            .zip(&kv_after_first)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "KV buffer mutated despite n_ctx overflow"
    );
}

fn verify_equivalences_for_model(swiglu: bool, with_geometry: bool, use_cache: bool, seed: usize) {
    let (recipe, weights) = build_20_5_gqa_model(swiglu, true, false, seed);
    let tokens: Vec<u32> = (0..17)
        .map(|i| u32::try_from((i * 7 + seed) % recipe.vocab_size).unwrap_or(0))
        .collect();
    let follow_tokens = [4_u32, 19_u32];
    let n_ctx = 32_usize;

    let geom = KvGeometry::new(
        recipe.num_layers,
        recipe.n_heads_kv,
        recipe.head_dim,
        n_ctx,
        None,
    )
    .expect("geom");
    let kv_len = geom.required_f32_len().expect("kv len");
    let attached_geom = with_geometry.then_some(geom);
    let mut harness = ModelHarness::new(&recipe, &weights, use_cache, n_ctx, attached_geom);

    // Path A: Full batched prefill (M = 17, start_pos = 0) + 2 follow-up decode steps.
    let mut kv_batched = vec![0.0_f32; kv_len];
    let logits_batched = harness.prefill_chunk(&tokens, 0, true, &mut kv_batched);
    let decodes_batched = harness.decode_sequence(&follow_tokens, tokens.len(), &mut kv_batched);

    // Path B: Chunked batched prefill ([0..7], [7..15], [15..17]) + 2 follow-up decode steps.
    let mut kv_chunked = vec![0.0_f32; kv_len];
    let mut logits_chunked = Vec::new();
    for (start, end, want_logits) in [(0_usize, 7_usize, false), (7, 15, true), (15, 17, true)] {
        let out = harness.prefill_chunk(&tokens[start..end], start, want_logits, &mut kv_chunked);
        if want_logits {
            logits_chunked = out;
        }
    }
    let decodes_chunked = harness.decode_sequence(&follow_tokens, tokens.len(), &mut kv_chunked);

    // Path C: Sequential single-token reference (prefill [0..1] + decode 1..17 + 2 decode steps).
    let mut kv_seq = vec![0.0_f32; kv_len];
    let logits_seq = harness.sequential_prefill_then_decode(&tokens, &mut kv_seq);
    let decodes_seq = harness.decode_sequence(&follow_tokens, tokens.len(), &mut kv_seq);

    assert_all_finite_nontrivial(&logits_batched, "batched prefill logits");
    let batched_prefill_diff = max_abs_diff(&logits_batched, &logits_seq);
    let chunked_prefill_diff = max_abs_diff(&logits_chunked, &logits_seq);
    assert!(
        batched_prefill_diff < 1e-4,
        "batched vs seq prefill logits diverged (swiglu={swiglu}, geom={with_geometry}, cache={use_cache}): {batched_prefill_diff}"
    );
    assert!(
        chunked_prefill_diff < 1e-4,
        "chunked vs seq prefill logits diverged (swiglu={swiglu}, geom={with_geometry}, cache={use_cache}): {chunked_prefill_diff}"
    );

    for step in 0..follow_tokens.len() {
        let step_batched_diff = max_abs_diff(&decodes_batched[step], &decodes_seq[step]);
        let step_chunked_diff = max_abs_diff(&decodes_chunked[step], &decodes_seq[step]);
        assert!(
            step_batched_diff < 1e-4,
            "follow decode {step} batched vs seq diff={step_batched_diff}"
        );
        assert!(
            step_chunked_diff < 1e-4,
            "follow decode {step} chunked vs seq diff={step_chunked_diff}"
        );
    }

    assert_kv_buffers_equivalent(
        attached_geom,
        recipe.num_layers,
        tokens.len() + follow_tokens.len(),
        &mut kv_batched,
        &mut kv_chunked,
        &mut kv_seq,
    );
}

#[test]
fn test_wave3_multi_layer_gqa_rope_kv_chunk_and_decode_equivalence() {
    let _shared = IME_LOCK
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // ReLU^2 (`bitnet-b1.58`) with KvGeometry + cached tiles
    verify_equivalences_for_model(false, true, true, 17);
    // SwiGLU (`bitnet-25`) with KvGeometry + uncached streaming
    verify_equivalences_for_model(true, true, false, 29);
    // Raw KV fallback without attached KvGeometry
    verify_equivalences_for_model(false, false, true, 41);
}

#[test]
fn test_wave3_batched_prefill_sliding_window_causal_parity() {
    let _shared = IME_LOCK
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (recipe, weights) = build_20_5_gqa_model(true, true, true, 63);
    let n_ctx = 16_usize;
    let geom = KvGeometry::new(
        recipe.num_layers,
        recipe.n_heads_kv,
        recipe.head_dim,
        n_ctx,
        Some(4),
    )
    .expect("geom");
    let kv_len = geom.required_f32_len().expect("kv_len");
    let mut harness = ModelHarness::new(&recipe, &weights, true, n_ctx, Some(geom));

    let prompt_a = [1_u32, 3, 5, 7, 9, 11, 13, 15, 17];
    let mut prompt_prefix_mutated = prompt_a;
    // Mutate token 0 (outside window 4 relative to token 8: active window covers positions 5..=8).
    prompt_prefix_mutated[0] = 28;

    let mut kv_a = vec![0.0_f32; kv_len];
    let logits_a = harness.prefill_chunk(&prompt_a, 0, true, &mut kv_a);
    let mut kv_mut = vec![0.0_f32; kv_len];
    let logits_mut = harness.prefill_chunk(&prompt_prefix_mutated, 0, true, &mut kv_mut);
    let window_leak = max_abs_diff(&logits_a, &logits_mut);
    assert!(
        window_leak < 1e-5,
        "sliding_window=4 leaked out-of-window token 0 into token 8 logits: diff={window_leak}"
    );

    let mut kv_chunked = vec![0.0_f32; kv_len];
    let _ = harness.prefill_chunk(&prompt_a[..5], 0, false, &mut kv_chunked);
    let logits_chunked = harness.prefill_chunk(&prompt_a[5..], 5, true, &mut kv_chunked);

    let mut kv_seq = vec![0.0_f32; kv_len];
    let logits_seq = harness.sequential_prefill_then_decode(&prompt_a, &mut kv_seq);

    let batched_diff = max_abs_diff(&logits_a, &logits_seq);
    let chunked_diff = max_abs_diff(&logits_chunked, &logits_seq);
    assert!(
        batched_diff < 1e-4,
        "SWA batched vs seq diff={batched_diff}"
    );
    assert!(
        chunked_diff < 1e-4,
        "SWA chunked vs seq diff={chunked_diff}"
    );
}
