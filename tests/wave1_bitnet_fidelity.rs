//! Integration fidelity suite for `BitNet` recipe hydration, `SubLN` placement,
//! full causal GQA over `KvCacheView`, multi-quant embeddings, and tied LM heads.

use std::collections::BTreeMap;

use jero_arch_bitnet::BitNetRecipe;
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, K3Error, KvGeometry, ModelWeights, QuantType,
    TensorView,
};

#[derive(Debug, Default)]
struct TestWeights {
    tensors: BTreeMap<String, (QuantType, Vec<usize>, Vec<u8>)>,
}

impl TestWeights {
    fn insert(
        &mut self,
        name: impl Into<String>,
        quant: QuantType,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) {
        self.tensors.insert(name.into(), (quant, shape, data));
    }
}

impl ModelWeights for TestWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let (key, (quant, shape, data)) =
            self.tensors
                .get_key_value(name)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("tensor '{name}' not found"),
                })?;
        TensorView::from_bytes(key.as_str(), *quant, shape.as_slice(), 0, data.as_slice())
    }

    fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.tensors.values().map(|(_, _, b)| b.len()).sum()
    }
}

#[derive(Debug, Default)]
struct EmptyWeights;

impl ModelWeights for EmptyWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "empty weights".to_owned(),
        })
    }

    fn tensor_count(&self) -> usize {
        0
    }

    fn total_weight_bytes(&self) -> usize {
        0
    }
}

fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

enum KvVal<'a> {
    U32(u32),
    F32(f32),
    Str(&'a str),
    StrArray(&'a [&'a str]),
}

fn build_gguf(kvs: &[(&str, KvVal<'_>)], tensors: &[(&str, QuantType, &[u64], usize)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());

    for (key, val) in kvs {
        push_gguf_str(&mut buf, key);
        match val {
            KvVal::U32(v) => {
                buf.extend_from_slice(&4u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            KvVal::F32(v) => {
                buf.extend_from_slice(&6u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            KvVal::Str(s) => {
                buf.extend_from_slice(&8u32.to_le_bytes());
                push_gguf_str(&mut buf, s);
            }
            KvVal::StrArray(items) => {
                buf.extend_from_slice(&9u32.to_le_bytes());
                buf.extend_from_slice(&8u32.to_le_bytes());
                buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for s in *items {
                    push_gguf_str(&mut buf, s);
                }
            }
        }
    }

    let mut rel_offset = 0u64;
    let mut total_payload = 0usize;
    for &(name, quant, shape, byte_len) in tensors {
        push_gguf_str(&mut buf, name);
        let rank = u32::try_from(shape.len()).expect("rank fits u32");
        buf.extend_from_slice(&rank.to_le_bytes());
        for &dim in shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&quant.ggml_type_id().to_le_bytes());
        buf.extend_from_slice(&rel_offset.to_le_bytes());
        let aligned = byte_len.div_ceil(32) * 32;
        rel_offset += aligned as u64;
        total_payload += aligned;
    }

    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.resize(buf.len() + total_payload, 0);
    buf
}

fn encode_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).expect("sign fits u16");
    let exp_raw = i32::try_from((bits >> 23) & 0xFF).expect("exp fits i32");
    let mant = bits & 0x007F_FFFF;
    if val == 0.0 {
        return sign;
    }
    let exp_f16 = exp_raw - 127 + 15;
    if exp_f16 >= 31 {
        return sign | 0x7C00;
    }
    if exp_f16 <= 0 {
        return sign;
    }
    let e = u16::try_from(exp_f16).expect("exp fits u16");
    let m = u16::try_from(mant >> 13).expect("mant fits u16");
    sign | (e << 10) | m
}

fn decode_f16(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x03FF);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let val = f32::from(u16::try_from(mant).expect("10-bit mantissa fits u16"))
            * f32::from_bits(0x3380_0000);
        return if sign != 0 { -val } else { val };
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

fn encode_bf16(val: f32) -> u16 {
    u16::try_from((val.to_bits() >> 16) & 0xFFFF).expect("bf16 upper bits fit u16")
}

fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn encode_f32_vec(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_f16_vec(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&encode_f16(v).to_le_bytes());
    }
    out
}

fn encode_bf16_vec(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&encode_bf16(v).to_le_bytes());
    }
    out
}

fn ternary_pattern(seed: usize, r: usize, c: usize) -> i8 {
    let mut x = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (seed as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    match x % 3 {
        0 => -1,
        1 => 0,
        _ => 1,
    }
}

fn encode_i2s_matrix(rows: usize, cols: usize, seed: usize, scale: f32) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(cols % 128, 0);
    let blocks_per_row = cols / 128;
    let payload = rows * blocks_per_row * 32;
    let mut raw = vec![0u8; payload + 32];
    let mut ref_w = vec![0.0f32; rows * cols];

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let boff = (r * blocks_per_row + b) * 32;
            let col_base = b * 128;
            for gp in 0..32 {
                let w0 = ternary_pattern(seed, r, col_base + gp);
                let w1 = ternary_pattern(seed, r, col_base + 32 + gp);
                let w2 = ternary_pattern(seed, r, col_base + 64 + gp);
                let w3 = ternary_pattern(seed, r, col_base + 96 + gp);
                let code = |w: i8| u8::try_from(w + 1).expect("ternary code 0..=2");
                raw[boff + gp] = (code(w0) << 6) | (code(w1) << 4) | (code(w2) << 2) | code(w3);
                ref_w[r * cols + col_base + gp] = f32::from(w0) * scale;
                ref_w[r * cols + col_base + 32 + gp] = f32::from(w1) * scale;
                ref_w[r * cols + col_base + 64 + gp] = f32::from(w2) * scale;
                ref_w[r * cols + col_base + 96 + gp] = f32::from(w3) * scale;
            }
        }
    }
    raw[payload..payload + 4].copy_from_slice(&scale.to_le_bytes());
    (raw, ref_w)
}

fn encode_tq1_0_matrix(
    rows: usize,
    cols: usize,
    seed: usize,
    scale_f16: f32,
) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(cols % 256, 0);
    let blocks_per_row = cols / 256;
    let d_bits = encode_f16(scale_f16);
    let d = decode_f16(d_bits);
    let mut raw = vec![0u8; rows * blocks_per_row * 54];
    let mut ref_w = vec![0.0f32; rows * cols];

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let boff = (r * blocks_per_row + b) * 54;
            raw[boff..boff + 2].copy_from_slice(&d_bits.to_le_bytes());
            let col_base = b * 256;
            for byte_idx in 0..51 {
                let mut packed = 0u8;
                let mut mul = 1u8;
                for j in 0..5 {
                    let c = col_base + byte_idx * 5 + j;
                    let w = ternary_pattern(seed, r, c);
                    ref_w[r * cols + c] = f32::from(w) * d;
                    let t = u8::try_from(w + 1).expect("trit 0..=2");
                    packed += t * mul;
                    mul = mul.saturating_mul(3);
                }
                raw[boff + 2 + byte_idx] = packed;
            }
            let last_w = ternary_pattern(seed, r, col_base + 255);
            ref_w[r * cols + col_base + 255] = f32::from(last_w) * d;
            raw[boff + 53] = u8::try_from(last_w + 1).expect("trit 0..=2");
        }
    }
    (raw, ref_w)
}

fn encode_tq2_0_matrix(
    rows: usize,
    cols: usize,
    seed: usize,
    scale_f16: f32,
) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(cols % 256, 0);
    let blocks_per_row = cols / 256;
    let d_bits = encode_f16(scale_f16);
    let d = decode_f16(d_bits);
    let mut raw = vec![0u8; rows * blocks_per_row * 66];
    let mut ref_w = vec![0.0f32; rows * cols];

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let boff = (r * blocks_per_row + b) * 66;
            raw[boff..boff + 2].copy_from_slice(&d_bits.to_le_bytes());
            let col_base = b * 256;
            for byte_idx in 0..64 {
                let base_c = col_base + byte_idx * 4;
                let w0 = ternary_pattern(seed, r, base_c);
                let w1 = ternary_pattern(seed, r, base_c + 1);
                let w2 = ternary_pattern(seed, r, base_c + 2);
                let w3 = ternary_pattern(seed, r, base_c + 3);
                for (k, &w) in [w0, w1, w2, w3].iter().enumerate() {
                    ref_w[r * cols + base_c + k] = f32::from(w) * d;
                }
                let code = |w: i8| u8::try_from(w + 1).expect("code 0..=2");
                raw[boff + 2 + byte_idx] =
                    code(w0) | (code(w1) << 2) | (code(w2) << 4) | (code(w3) << 6);
            }
        }
    }
    (raw, ref_w)
}

fn encode_q8_0_matrix(
    rows: usize,
    cols: usize,
    seed: usize,
    scale_f16: f32,
) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(cols % 32, 0);
    let blocks_per_row = cols / 32;
    let d_bits = encode_f16(scale_f16);
    let d = decode_f16(d_bits);
    let mut raw = vec![0u8; rows * blocks_per_row * 34];
    let mut ref_w = vec![0.0f32; rows * cols];

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let boff = (r * blocks_per_row + b) * 34;
            raw[boff..boff + 2].copy_from_slice(&d_bits.to_le_bytes());
            for j in 0..32 {
                let c = b * 32 + j;
                let idx_val = i32::try_from((r * 13 + c * 7 + seed * 5) % 41).expect("mod 41");
                let q = i8::try_from(idx_val - 20).expect("q8 value fits i8");
                #[expect(clippy::cast_sign_loss, reason = "two's complement i8 storage")]
                {
                    raw[boff + 2 + j] = q as u8;
                }
                ref_w[r * cols + c] = f32::from(q) * d;
            }
        }
    }
    (raw, ref_w)
}

fn encode_matrix_by_quant(
    quant: QuantType,
    rows: usize,
    cols: usize,
    seed: usize,
) -> (Vec<u8>, Vec<f32>) {
    match quant {
        QuantType::I2_S => encode_i2s_matrix(rows, cols, seed, 0.25),
        QuantType::TQ1_0 => encode_tq1_0_matrix(rows, cols, seed, 0.25),
        QuantType::TQ2_0 => encode_tq2_0_matrix(rows, cols, seed, 0.25),
        QuantType::Q8_0 => encode_q8_0_matrix(rows, cols, seed, 0.05),
        other => panic!("unexpected projection quant in helper: {other:?}"),
    }
}

fn ref_subln_bitlinear(
    input: &[f32],
    norm_w: &[f32],
    eps: f32,
    weight_rows: &[f32],
    out_dim: usize,
) -> Vec<f32> {
    let dim = input.len();
    #[expect(clippy::cast_precision_loss, reason = "test dims fit f32")]
    let rms = ((input.iter().map(|x| x * x).sum::<f32>()) / dim as f32 + eps).sqrt();
    let mut normed = vec![0.0f32; dim];
    let mut max_abs = 0.0f32;
    for i in 0..dim {
        normed[i] = (input[i] / rms) * norm_w[i];
        max_abs = max_abs.max(normed[i].abs());
    }
    let scale = max_abs / 127.0;
    let act_q: Vec<f32> = if scale > 1e-12 {
        normed
            .iter()
            .map(|&v| ((v / scale).round().clamp(-128.0, 127.0)) * scale)
            .collect()
    } else {
        vec![0.0f32; dim]
    };

    let mut out = vec![0.0f32; out_dim];
    for r in 0..out_dim {
        let mut dot = 0.0f32;
        for c in 0..dim {
            dot += weight_rows[r * dim + c] * act_q[c];
        }
        out[r] = dot;
    }
    out
}

fn ref_rope(vec: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for p in 0..half {
            #[expect(clippy::cast_precision_loss, reason = "test indices fit f32")]
            let freq = theta.powf(-(2.0 * p as f32) / head_dim as f32);
            #[expect(clippy::cast_precision_loss, reason = "pos fits f32")]
            let angle = (pos as f32) * freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = vec[base + 2 * p];
            let x1 = vec[base + 2 * p + 1];
            vec[base + 2 * p] = x0 * cos - x1 * sin;
            vec[base + 2 * p + 1] = x0 * sin + x1 * cos;
        }
    }
}

fn verify_bitnet_25_and_validation_rejections(recipe: &BitNetRecipe) {
    let tokens = ["<unk>", "hello", "world", "bitnet", "k3", "rvv"];
    let gate_bytes = 512 / 128 * 32 * 1024 + 32;
    let raw_swiglu = build_gguf(
        &[
            ("general.architecture", KvVal::Str("bitnet-25")),
            ("bitnet-25.block_count", KvVal::U32(2)),
            ("bitnet-25.embedding_length", KvVal::U32(512)),
            ("bitnet-25.attention.head_count", KvVal::U32(8)),
            ("bitnet-25.attention.head_count_kv", KvVal::U32(2)),
            ("tokenizer.ggml.tokens", KvVal::StrArray(&tokens)),
        ],
        &[(
            "blk.0.ffn_gate.weight",
            QuantType::I2_S,
            &[512, 1024],
            gate_bytes,
        )],
    );
    let parsed_swiglu = GgufFile::parse(&raw_swiglu).expect("parse bitnet-25 gguf");
    let recipe_25 = BitNetRecipe::from_gguf(&parsed_swiglu).expect("hydrate bitnet-25");
    assert_eq!(recipe_25.architecture_id(), "bitnet-25");
    assert!(!recipe_25.use_relu2);
    assert_eq!(recipe_25.ffn_dim, 1024);
    assert_eq!(recipe_25.vocab_size, tokens.len());

    let bad_arch = build_gguf(
        &[
            ("general.architecture", KvVal::Str("qwen2")),
            ("qwen2.block_count", KvVal::U32(2)),
            ("qwen2.embedding_length", KvVal::U32(512)),
            ("qwen2.attention.head_count", KvVal::U32(8)),
            ("qwen2.attention.head_count_kv", KvVal::U32(2)),
            ("qwen2.vocab_size", KvVal::U32(32)),
        ],
        &[],
    );
    let gguf_bad_arch = GgufFile::parse(&bad_arch).expect("parse qwen2 gguf");
    assert!(matches!(
        BitNetRecipe::from_gguf(&gguf_bad_arch),
        Err(K3Error::CapabilityMismatch { .. })
    ));

    let mismatched_heads = build_gguf(
        &[
            ("general.architecture", KvVal::Str("bitnet-b1.58")),
            ("bitnet-b1.58.block_count", KvVal::U32(30)),
            ("bitnet-b1.58.embedding_length", KvVal::U32(2560)),
            ("bitnet-b1.58.feed_forward_length", KvVal::U32(6912)),
            ("bitnet-b1.58.attention.head_count", KvVal::U32(16)),
            ("bitnet-b1.58.attention.head_count_kv", KvVal::U32(4)),
            ("bitnet-b1.58.vocab_size", KvVal::U32(128_256)),
        ],
        &[],
    );
    let gguf_mismatch = GgufFile::parse(&mismatched_heads).expect("parse mismatched");
    assert!(matches!(
        recipe.validate_model(gguf_mismatch.metadata(), &EmptyWeights),
        Err(K3Error::CapabilityMismatch { .. })
    ));
}

#[test]
fn test_from_gguf_bitnet_2b4t_and_validate_model_contracts() {
    let raw_2b4t = build_gguf(
        &[
            ("general.architecture", KvVal::Str("bitnet-b1.58")),
            ("bitnet-b1.58.block_count", KvVal::U32(30)),
            ("bitnet-b1.58.embedding_length", KvVal::U32(2560)),
            ("bitnet-b1.58.feed_forward_length", KvVal::U32(6912)),
            ("bitnet-b1.58.attention.head_count", KvVal::U32(20)),
            ("bitnet-b1.58.attention.head_count_kv", KvVal::U32(5)),
            ("bitnet-b1.58.vocab_size", KvVal::U32(128_256)),
            ("bitnet-b1.58.rope.freq_base", KvVal::F32(500_000.0)),
            (
                "bitnet-b1.58.attention.layer_norm_rms_epsilon",
                KvVal::F32(1e-5),
            ),
        ],
        &[],
    );
    let parsed_2b4t = GgufFile::parse(&raw_2b4t).expect("parse 2B-4T metadata");
    let recipe = BitNetRecipe::from_gguf(&parsed_2b4t).expect("hydrate BitNet-b1.58-2B-4T");
    assert_eq!(recipe.architecture_id(), "bitnet-b1.58");
    assert_eq!(recipe.num_layers, 30);
    assert_eq!(recipe.hidden_dim, 2560);
    assert_eq!(recipe.ffn_dim, 6912);
    assert_eq!(recipe.n_heads_q, 20);
    assert_eq!(recipe.n_heads_kv, 5);
    assert_eq!(recipe.head_dim, 128);
    assert_eq!(recipe.vocab_size, 128_256);
    assert!((recipe.rope_theta - 500_000.0).abs() <= f32::EPSILON);
    assert!(recipe.use_relu2);
    recipe
        .validate_model(parsed_2b4t.metadata(), &EmptyWeights)
        .expect("validate_model accepts 2B-4T metadata");

    verify_bitnet_25_and_validation_rejections(&recipe);

    for bad_quant in [QuantType::Q4_0, QuantType::Q4_K_M, QuantType::Q6_K] {
        assert!(!recipe.supports_quant(bad_quant));
    }
    for ok_quant in [
        QuantType::I2_S,
        QuantType::TQ1_0,
        QuantType::TQ2_0,
        QuantType::Q8_0,
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
    ] {
        assert!(recipe.supports_quant(ok_quant));
    }
}

struct SublnFixture {
    weights: TestWeights,
    embd_vals: Vec<f32>,
    attn_norm: Vec<f32>,
    attn_sub_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_sub_norm: Vec<f32>,
    out_norm: Vec<f32>,
    q_ref: Vec<f32>,
    k_ref: Vec<f32>,
    v_ref: Vec<f32>,
    o_ref: Vec<f32>,
    g_ref: Vec<f32>,
    u_ref: Vec<f32>,
    d_ref: Vec<f32>,
}

fn build_subln_fixture(proj_quant: QuantType) -> SublnFixture {
    let (hidden_dim, ffn_dim, total_q, total_kv, vocab_size) = (256, 512, 256, 64, 8);
    #[expect(clippy::cast_precision_loss, reason = "test indices fit f32")]
    let make_nonuniform_norm = |len: usize, seed: usize| -> Vec<f32> {
        (0..len)
            .map(|i| 0.75 + (((i * 19 + seed * 7) % 13) as f32) * 0.05)
            .collect()
    };

    let mut weights = TestWeights::default();
    let mut embd_vals = vec![0.0f32; vocab_size * hidden_dim];
    for (i, slot) in embd_vals.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "test index fits f32")]
        {
            *slot = (((i * 29 + 7) % 53) as f32) * 0.04 - 1.0;
        }
    }
    weights.insert(
        "token_embd.weight",
        QuantType::F32,
        vec![hidden_dim, vocab_size],
        encode_f32_vec(&embd_vals),
    );

    let attn_norm: Vec<f32> = make_nonuniform_norm(hidden_dim, 1)
        .into_iter()
        .map(|val| decode_f16(encode_f16(val)))
        .collect();
    let attn_sub_norm: Vec<f32> = make_nonuniform_norm(total_q, 2)
        .into_iter()
        .map(|val| decode_bf16(encode_bf16(val)))
        .collect();
    let ffn_norm = make_nonuniform_norm(hidden_dim, 3);
    let ffn_sub_norm: Vec<f32> = make_nonuniform_norm(ffn_dim, 4)
        .into_iter()
        .map(|val| decode_f16(encode_f16(val)))
        .collect();
    let out_norm = make_nonuniform_norm(hidden_dim, 5);

    weights.insert(
        "blk.0.attn_norm.weight",
        QuantType::F16,
        vec![hidden_dim],
        encode_f16_vec(&attn_norm),
    );
    weights.insert(
        "blk.0.attn_sub_norm.weight",
        QuantType::BF16,
        vec![total_q],
        encode_bf16_vec(&attn_sub_norm),
    );
    weights.insert(
        "blk.0.ffn_norm.weight",
        QuantType::F32,
        vec![hidden_dim],
        encode_f32_vec(&ffn_norm),
    );
    weights.insert(
        "blk.0.ffn_sub_norm.weight",
        QuantType::F16,
        vec![ffn_dim],
        encode_f16_vec(&ffn_sub_norm),
    );
    weights.insert(
        "output_norm.weight",
        QuantType::F32,
        vec![hidden_dim],
        encode_f32_vec(&out_norm),
    );

    let (q_raw, q_ref) = encode_matrix_by_quant(proj_quant, total_q, hidden_dim, 11);
    let (k_raw, k_ref) = encode_matrix_by_quant(proj_quant, total_kv, hidden_dim, 12);
    let (v_raw, v_ref) = encode_matrix_by_quant(proj_quant, total_kv, hidden_dim, 13);
    let (o_raw, o_ref) = encode_matrix_by_quant(proj_quant, hidden_dim, total_q, 14);
    let (g_raw, g_ref) = encode_matrix_by_quant(proj_quant, ffn_dim, hidden_dim, 15);
    let (u_raw, u_ref) = encode_matrix_by_quant(proj_quant, ffn_dim, hidden_dim, 16);
    let (d_raw, d_ref) = encode_matrix_by_quant(proj_quant, hidden_dim, ffn_dim, 17);
    for (name, shape, raw) in [
        ("blk.0.attn_q.weight", vec![hidden_dim, total_q], q_raw),
        ("blk.0.attn_k.weight", vec![hidden_dim, total_kv], k_raw),
        ("blk.0.attn_v.weight", vec![hidden_dim, total_kv], v_raw),
        ("blk.0.attn_output.weight", vec![total_q, hidden_dim], o_raw),
        ("blk.0.ffn_gate.weight", vec![hidden_dim, ffn_dim], g_raw),
        ("blk.0.ffn_up.weight", vec![hidden_dim, ffn_dim], u_raw),
        ("blk.0.ffn_down.weight", vec![ffn_dim, hidden_dim], d_raw),
    ] {
        weights.insert(name, proj_quant, shape, raw);
    }

    SublnFixture {
        weights,
        embd_vals,
        attn_norm,
        attn_sub_norm,
        ffn_norm,
        ffn_sub_norm,
        out_norm,
        q_ref,
        k_ref,
        v_ref,
        o_ref,
        g_ref,
        u_ref,
        d_ref,
    }
}

fn compute_reference_hidden(fixture: &SublnFixture, use_relu2: bool) -> Vec<f32> {
    let (hidden_dim, ffn_dim, n_q, n_kv, head_dim) = (256, 512, 8, 2, 32);
    let (total_q, total_kv) = (256, 64);

    let mut hidden_ref = fixture.embd_vals[3 * hidden_dim..4 * hidden_dim].to_vec();
    let mut q_proj = ref_subln_bitlinear(
        &hidden_ref,
        &fixture.attn_norm,
        1e-5,
        &fixture.q_ref,
        total_q,
    );
    let mut k_proj = ref_subln_bitlinear(
        &hidden_ref,
        &fixture.attn_norm,
        1e-5,
        &fixture.k_ref,
        total_kv,
    );
    let v_proj = ref_subln_bitlinear(
        &hidden_ref,
        &fixture.attn_norm,
        1e-5,
        &fixture.v_ref,
        total_kv,
    );
    ref_rope(&mut q_proj, n_q, head_dim, 0, 500_000.0);
    ref_rope(&mut k_proj, n_kv, head_dim, 0, 500_000.0);
    let mut attn_out = vec![0.0f32; total_q];
    let group = n_q / n_kv;
    for head_idx in 0..n_q {
        let kv_idx = head_idx / group;
        attn_out[head_idx * head_dim..(head_idx + 1) * head_dim]
            .copy_from_slice(&v_proj[kv_idx * head_dim..(kv_idx + 1) * head_dim]);
    }
    let o_proj = ref_subln_bitlinear(
        &attn_out,
        &fixture.attn_sub_norm,
        1e-5,
        &fixture.o_ref,
        hidden_dim,
    );
    for (dst, &delta) in hidden_ref.iter_mut().zip(&o_proj) {
        *dst += delta;
    }
    let gate_vec = ref_subln_bitlinear(
        &hidden_ref,
        &fixture.ffn_norm,
        1e-5,
        &fixture.g_ref,
        ffn_dim,
    );
    let up_vec = ref_subln_bitlinear(
        &hidden_ref,
        &fixture.ffn_norm,
        1e-5,
        &fixture.u_ref,
        ffn_dim,
    );
    let ffn_inter: Vec<f32> = gate_vec
        .iter()
        .zip(&up_vec)
        .map(|(&gate_val, &up_val)| {
            if use_relu2 {
                gate_val.max(0.0).powi(2) * up_val
            } else {
                let sig = 1.0 / (1.0 + (-gate_val).exp());
                (gate_val * sig) * up_val
            }
        })
        .collect();
    let d_proj = ref_subln_bitlinear(
        &ffn_inter,
        &fixture.ffn_sub_norm,
        1e-5,
        &fixture.d_ref,
        hidden_dim,
    );
    for (dst, &delta) in hidden_ref.iter_mut().zip(&d_proj) {
        *dst += delta;
    }
    #[expect(clippy::cast_precision_loss, reason = "hidden_dim fits f32")]
    let out_rms =
        ((hidden_ref.iter().map(|val| val * val).sum::<f32>()) / hidden_dim as f32 + 1e-5).sqrt();
    for (h_slot, &norm_val) in hidden_ref.iter_mut().zip(&fixture.out_norm) {
        *h_slot = (*h_slot / out_rms) * norm_val;
    }
    hidden_ref
}

fn verify_canonical_subln_for_quant(proj_quant: QuantType, use_relu2: bool) {
    let (hidden_dim, ffn_dim, n_q, n_kv, head_dim) = (256, 512, 8, 2, 32);
    let vocab_size = 8;
    let fixture = build_subln_fixture(proj_quant);

    let recipe = BitNetRecipe {
        architecture: if use_relu2 {
            "bitnet-b1.58"
        } else {
            "bitnet-25"
        },
        num_layers: 1,
        hidden_dim,
        ffn_dim,
        n_heads_q: n_q,
        n_heads_kv: n_kv,
        head_dim,
        vocab_size,
        rms_eps: 1e-5,
        rope_theta: 500_000.0,
        use_relu2,
    };

    let geom = KvGeometry::new(1, n_kv, head_dim, 4, None).expect("geom");
    let mut kv_buf = vec![0.0f32; geom.required_f32_len().expect("kv len")];
    let mut logits = vec![0.0f32; vocab_size];
    let prompt = [3u32];
    let mut ctx = ExecutionContext::new(&prompt, 4, &mut logits, &mut kv_buf)
        .expect("ctx")
        .with_kv_geometry(geom)
        .expect("attach geom");
    recipe
        .prefill_step(&mut ctx, &fixture.weights)
        .expect("prefill succeeds");

    let hidden_ref = compute_reference_hidden(&fixture, use_relu2);
    for (tok, (&got, row)) in logits
        .iter()
        .zip(fixture.embd_vals.chunks_exact(hidden_dim))
        .enumerate()
    {
        let expected: f32 = hidden_ref
            .iter()
            .zip(row)
            .map(|(&lhs, &rhs)| lhs * rhs)
            .sum();
        let err = (got - expected).abs();
        assert!(
            err <= 5e-2,
            "quant={proj_quant:?} use_relu2={use_relu2} tok={tok} got={got} expected={expected} err={err}"
        );
    }
}

fn populate_layer_projections(
    weights: &mut TestWeights,
    proj_quant: QuantType,
    recipe: &BitNetRecipe,
    seed_base: usize,
) {
    let total_q = recipe.n_heads_q * recipe.head_dim;
    let total_kv = recipe.n_heads_kv * recipe.head_dim;
    for (offset, name, rows, cols) in [
        (1, "blk.0.attn_q.weight", total_q, recipe.hidden_dim),
        (2, "blk.0.attn_k.weight", total_kv, recipe.hidden_dim),
        (3, "blk.0.attn_v.weight", total_kv, recipe.hidden_dim),
        (4, "blk.0.attn_output.weight", recipe.hidden_dim, total_q),
        (
            5,
            "blk.0.ffn_gate.weight",
            recipe.ffn_dim,
            recipe.hidden_dim,
        ),
        (6, "blk.0.ffn_up.weight", recipe.ffn_dim, recipe.hidden_dim),
        (
            7,
            "blk.0.ffn_down.weight",
            recipe.hidden_dim,
            recipe.ffn_dim,
        ),
    ] {
        let (raw, _) = encode_matrix_by_quant(proj_quant, rows, cols, seed_base + offset);
        weights.insert(name, proj_quant, vec![cols, rows], raw);
    }
}

#[test]
fn test_canonical_subln_placement_matches_exact_reference_across_quants() {
    for (proj_quant, use_relu2) in [
        (QuantType::I2_S, true),
        (QuantType::TQ1_0, true),
        (QuantType::TQ2_0, false),
        (QuantType::Q8_0, false),
    ] {
        verify_canonical_subln_for_quant(proj_quant, use_relu2);
    }

    let recipe = BitNetRecipe {
        hidden_dim: 256,
        ffn_dim: 512,
        n_heads_q: 8,
        n_heads_kv: 2,
        head_dim: 32,
        vocab_size: 8,
        num_layers: 1,
        ..BitNetRecipe::new_b158()
    };
    let mut legacy_weights = TestWeights::default();
    legacy_weights.insert(
        "token_embd.weight",
        QuantType::F32,
        vec![256, 8],
        vec![0u8; 8 * 256 * 4],
    );
    for (name, dim) in [
        ("output_norm.weight", 256),
        ("blk.0.attn_norm.weight", 256),
        ("blk.0.attn_sub_norm.weight", 256),
        ("blk.0.ffn_norm.weight", 256),
        ("blk.0.ffn_down_sub_norm.weight", 512),
    ] {
        legacy_weights.insert(
            name,
            QuantType::F32,
            vec![dim],
            encode_f32_vec(&vec![1.0; dim]),
        );
    }
    populate_layer_projections(&mut legacy_weights, QuantType::I2_S, &recipe, 90);
    let mut logits = vec![0.0f32; 8];
    let mut kv = vec![0.0f32; 2 * 4 * 64];
    let prompt = [1u32];
    let mut ctx = ExecutionContext::new(&prompt, 4, &mut logits, &mut kv).expect("ctx");
    assert!(matches!(
        recipe.prefill_step(&mut ctx, &legacy_weights),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn test_causal_gqa_kv_view_prefill_decode_equivalence_and_sliding_window() {
    let recipe = BitNetRecipe {
        architecture: "bitnet-b1.58",
        num_layers: 1,
        hidden_dim: 256,
        ffn_dim: 256,
        n_heads_q: 8,
        n_heads_kv: 2,
        head_dim: 32,
        vocab_size: 8,
        rms_eps: 1e-5,
        rope_theta: 500_000.0,
        use_relu2: true,
    };

    let mut weights = TestWeights::default();
    let mut embd_vals = vec![0.0f32; 8 * 256];
    for (i, slot) in embd_vals.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "test index fits f32")]
        {
            *slot = (((i * 37 + 11) % 61) as f32) * 0.05 - 1.5;
        }
    }
    weights.insert(
        "token_embd.weight",
        QuantType::F32,
        vec![256, 8],
        encode_f32_vec(&embd_vals),
    );
    for (name, dim) in [
        ("output_norm.weight", 256),
        ("blk.0.attn_norm.weight", 256),
        ("blk.0.attn_sub_norm.weight", 256),
        ("blk.0.ffn_norm.weight", 256),
        ("blk.0.ffn_sub_norm.weight", 256),
    ] {
        weights.insert(
            name,
            QuantType::F32,
            vec![dim],
            encode_f32_vec(&vec![1.0; dim]),
        );
    }
    populate_layer_projections(&mut weights, QuantType::TQ1_0, &recipe, 20);

    let run_full = |tokens: &[u32], window: Option<usize>| -> Vec<f32> {
        let geom = KvGeometry::new(1, 2, 32, 8, window).expect("geom");
        let mut logits = vec![0.0f32; 8];
        let mut kv = vec![0.0f32; geom.required_f32_len().expect("len")];
        let mut ctx = ExecutionContext::new(tokens, 8, &mut logits, &mut kv)
            .expect("ctx")
            .with_kv_geometry(geom)
            .expect("geom");
        recipe.prefill_step(&mut ctx, &weights).expect("prefill");
        logits
    };

    let logits_prefill = run_full(&[1, 2, 3], None);
    let geom = KvGeometry::new(1, 2, 32, 8, None).expect("geom");
    let mut logits_step = vec![0.0f32; 8];
    let mut kv_step = vec![0.0f32; geom.required_f32_len().expect("len")];
    let seed_tok = [1u32];
    let mut ctx_step = ExecutionContext::new(&seed_tok, 8, &mut logits_step, &mut kv_step)
        .expect("ctx")
        .with_kv_geometry(geom)
        .expect("geom");
    recipe.prefill_step(&mut ctx_step, &weights).expect("p0");
    ctx_step.prepare_decode(2).expect("d1 prep");
    recipe.decode_step(&mut ctx_step, &weights).expect("d1");
    ctx_step.prepare_decode(3).expect("d2 prep");
    recipe.decode_step(&mut ctx_step, &weights).expect("d2");
    for (p, s) in logits_prefill.iter().zip(&logits_step) {
        assert!((p - s).abs() <= 1e-4, "prefill={p} decode={s}");
    }

    let logits_mutated = run_full(&[6, 2, 3], None);
    assert!(
        logits_prefill
            .iter()
            .zip(&logits_mutated)
            .any(|(a, b)| (a - b).abs() > 1e-2)
    );

    let logits_win_a = run_full(&[1, 2, 3], Some(2));
    let logits_win_b = run_full(&[6, 2, 3], Some(2));
    for (a, b) in logits_win_a.iter().zip(&logits_win_b) {
        assert!((a - b).abs() <= 1e-5, "windowed causal leak: a={a} b={b}");
    }
}

#[test]
fn test_multiquant_embeddings_tied_head_and_linear_output_norm() {
    let recipe = BitNetRecipe {
        architecture: "bitnet-b1.58",
        num_layers: 1,
        hidden_dim: 256,
        ffn_dim: 256,
        n_heads_q: 8,
        n_heads_kv: 2,
        head_dim: 32,
        vocab_size: 6,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        use_relu2: true,
    };

    for emb_quant in [
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q8_0,
        QuantType::I2_S,
        QuantType::TQ1_0,
        QuantType::TQ2_0,
    ] {
        let mut weights = TestWeights::default();
        let make_vals = |mul: usize| -> Vec<f32> {
            (0..6 * 256)
                .map(|i| {
                    #[expect(clippy::cast_precision_loss, reason = "index fits f32")]
                    {
                        ((i * mul % 31) as f32) * 0.05 - 0.75
                    }
                })
                .collect()
        };
        let emb_bytes = match emb_quant {
            QuantType::F16 => encode_f16_vec(&make_vals(17)),
            QuantType::BF16 => encode_bf16_vec(&make_vals(19)),
            QuantType::Q8_0 => encode_q8_0_matrix(6, 256, 31, 0.05).0,
            QuantType::I2_S => encode_i2s_matrix(6, 256, 32, 0.25).0,
            QuantType::TQ1_0 => encode_tq1_0_matrix(6, 256, 33, 0.25).0,
            QuantType::TQ2_0 => encode_tq2_0_matrix(6, 256, 34, 0.25).0,
            _ => unreachable!(),
        };
        weights.insert("token_embd.weight", emb_quant, vec![256, 6], emb_bytes);
        for name in [
            "blk.0.attn_norm.weight",
            "blk.0.attn_sub_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_sub_norm.weight",
        ] {
            weights.insert(
                name,
                QuantType::F32,
                vec![256],
                encode_f32_vec(&vec![1.0; 256]),
            );
        }
        populate_layer_projections(&mut weights, QuantType::I2_S, &recipe, 40);

        let run_with_norm_scale = |scale: f32, w: &mut TestWeights| -> Vec<f32> {
            w.insert(
                "output_norm.weight",
                QuantType::F32,
                vec![256],
                encode_f32_vec(&vec![scale; 256]),
            );
            let geom = KvGeometry::new(1, 2, 32, 4, None).expect("geom");
            let mut logits = vec![0.0f32; 6];
            let mut kv = vec![0.0f32; geom.required_f32_len().expect("len")];
            let tok = [2u32];
            let mut ctx = ExecutionContext::new(&tok, 4, &mut logits, &mut kv)
                .expect("ctx")
                .with_kv_geometry(geom)
                .expect("geom");
            recipe.prefill_step(&mut ctx, w).expect("prefill");
            logits
        };

        let logits_1x = run_with_norm_scale(1.0, &mut weights);
        assert!(logits_1x.iter().all(|v| v.is_finite()));
        assert!(logits_1x.iter().any(|v| v.abs() > 1e-3));

        if matches!(
            emb_quant,
            QuantType::F16 | QuantType::BF16 | QuantType::Q8_0
        ) {
            let logits_2x = run_with_norm_scale(2.0, &mut weights);
            for (l1, l2) in logits_1x.iter().zip(&logits_2x) {
                assert!(
                    (l2 - 2.0 * l1).abs() <= 1e-4,
                    "emb_quant={emb_quant:?} non-linear output_norm scaling: 1x={l1} 2x={l2}"
                );
            }
        }
    }
}
