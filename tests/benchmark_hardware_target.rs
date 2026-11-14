#![cfg(feature = "with-kernels")]
#![allow(clippy::all, clippy::pedantic, clippy::restriction)]
//! Hardware Target Benchmark Test for `SpacemiT` K3 (Milestone 4 Epic 2)
//!
//! Evaluates authentic native hardware performance on the SpacemiT K3 Reference Platform (16-core riscv64):
//! 1. Micro-benchmarks: `TQ1_0/TQ2_0` unpacking, `SubLN`, ReLU^2, and `BitLinear` RVV vs IME.
//! 2. Full-model 30-layer inference: `BitNetRecipe` across 30 layers, 2560 hidden, 6912 FFN.

use std::collections::BTreeMap;
use std::time::Instant;

use jero_arch_bitnet::BitNetRecipe;
use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};
use jero_kernels_rvv::ime::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, bitlinear_i8_ime, bitlinear_i8_rvv,
    unpack_tq1_0_block_i8, unpack_tq2_0_block_i8,
};
use jero_kernels_rvv::{relu_squared_gate_f32, subln_quantize_absmax_i8};

#[derive(Default, Debug)]
pub struct TargetBenchmarkWeights {
    tensors: BTreeMap<String, (QuantType, Vec<usize>, Vec<u8>)>,
}

impl TargetBenchmarkWeights {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_tensor(
        &mut self,
        name: impl Into<String>,
        quant: QuantType,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) {
        self.tensors.insert(name.into(), (quant, shape, data));
    }
}

impl ModelWeights for TargetBenchmarkWeights {
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
        self.tensors.values().map(|(_, _, d)| d.len()).sum()
    }
}

fn create_tq1_0_buffer(rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(cols % 256, 0);
    let blocks_per_row = cols / 256;
    let mut raw = vec![0u8; rows * blocks_per_row * TQ1_0_BLOCK_BYTES];
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let offset = (r * blocks_per_row + b) * TQ1_0_BLOCK_BYTES;
            // FP16 scale = 1.0 (0x3C00)
            raw[offset..offset + 2].copy_from_slice(&[0x00, 0x3C]);
            for byte_idx in 0..51 {
                // Alternating trits (+1, 0, -1, +1, 0)
                raw[offset + 2 + byte_idx] = 0x5A;
            }
            raw[offset + 53] = 0x02; // +1
        }
    }
    raw
}

fn create_tq2_0_buffer(rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(cols % 256, 0);
    let blocks_per_row = cols / 256;
    let mut raw = vec![0u8; rows * blocks_per_row * TQ2_0_BLOCK_BYTES];
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let offset = (r * blocks_per_row + b) * TQ2_0_BLOCK_BYTES;
            // FP16 scale = 1.0 (0x3C00)
            raw[offset..offset + 2].copy_from_slice(&[0x00, 0x3C]);
            // Fill 64 bytes with 0xAA (each 2-bit pair is 2 -> +1)
            for byte_idx in 0..64 {
                raw[offset + 2 + byte_idx] = 0xAA;
            }
        }
    }
    raw
}

fn create_f32_buffer(dim: usize, val: f32) -> Vec<u8> {
    let mut raw = vec![0u8; dim * 4];
    for i in 0..dim {
        raw[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
    }
    raw
}

fn create_q8_0_buffer(rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0);
    let blocks_per_row = cols / 32;
    let mut raw = vec![0u8; rows * blocks_per_row * 34];
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let offset = (r * blocks_per_row + b) * 34;
            // FP16 scale = 1.0 (0x3C00)
            raw[offset..offset + 2].copy_from_slice(&[0x00, 0x3C]);
            for byte_idx in 0..32 {
                raw[offset + 2 + byte_idx] = 1;
            }
        }
    }
    raw
}

fn create_f16_buffer(rows: usize, cols: usize) -> Vec<u8> {
    let mut raw = vec![0u8; rows * cols * 2];
    for i in 0..(rows * cols) {
        // FP16 1.0 (0x3C00)
        raw[i * 2] = 0x00;
        raw[i * 2 + 1] = 0x3C;
    }
    raw
}

#[test]
fn test_target_hardware_micro_benchmarks() {
    println!("\n==========================================================");
    println!("  SpacemiT K3 Target Hardware Micro-Benchmark Suite");
    println!("==========================================================");

    // 1. Ternary Unpacking Throughput
    let mut tq1_block = [0u8; 54];
    tq1_block[0..2].copy_from_slice(&[0x00, 0x3C]);
    for i in 2..53 {
        tq1_block[i] = 0x5A;
    }
    tq1_block[53] = 0x02;

    let mut tq2_block = [0u8; 66];
    tq2_block[0..2].copy_from_slice(&[0x00, 0x3C]);
    for i in 2..66 {
        tq2_block[i] = 0xAA;
    }

    let mut out_i8 = [0i8; 256];
    let unpack_iters = if cfg!(debug_assertions) { 100 } else { 50_000 };

    let t0 = Instant::now();
    for _ in 0..unpack_iters {
        let _ = unpack_tq1_0_block_i8(&tq1_block, &mut out_i8);
    }
    let tq1_elapsed = t0.elapsed();
    let tq1_ns_per_block = tq1_elapsed.as_nanos() as f64 / unpack_iters as f64;
    let tq1_gb_s = (54.0 * unpack_iters as f64 / tq1_elapsed.as_secs_f64()) / 1e9;

    let t1 = Instant::now();
    for _ in 0..unpack_iters {
        let _ = unpack_tq2_0_block_i8(&tq2_block, &mut out_i8);
    }
    let tq2_elapsed = t1.elapsed();
    let tq2_ns_per_block = tq2_elapsed.as_nanos() as f64 / unpack_iters as f64;
    let tq2_gb_s = (66.0 * unpack_iters as f64 / tq2_elapsed.as_secs_f64()) / 1e9;

    println!("\n--- 1. On-the-Fly Ternary Unpacking Throughput ---");
    println!("  TQ1_0 (base-3 54B/256) : {tq1_ns_per_block:.2} ns/block | {tq1_gb_s:.2} GB/s");
    println!("  TQ2_0 (2-bit 66B/256)  : {tq2_ns_per_block:.2} ns/block | {tq2_gb_s:.2} GB/s");

    // 2. Normalization & Activation Kernels
    let hidden_dim = 2560;
    let ffn_dim = 6912;
    let subln_iters = if cfg!(debug_assertions) { 100 } else { 10_000 };

    let input_f32 = vec![1.05f32; hidden_dim];
    let norm_w = vec![1.0f32; hidden_dim];
    let mut subln_out_i8 = vec![0i8; hidden_dim];

    let t2 = Instant::now();
    for _ in 0..subln_iters {
        let _ = subln_quantize_absmax_i8(&mut subln_out_i8, &input_f32, &norm_w, 1e-5);
    }
    let subln_elapsed = t2.elapsed();
    let subln_us = (subln_elapsed.as_nanos() as f64 / subln_iters as f64) / 1000.0;

    let gate_f32 = vec![1.2f32; ffn_dim];
    let up_f32 = vec![0.8f32; ffn_dim];
    let mut relu2_out = vec![0.0f32; ffn_dim];

    let t3 = Instant::now();
    for _ in 0..subln_iters {
        let _ = relu_squared_gate_f32(&mut relu2_out, &gate_f32, &up_f32);
    }
    let relu2_elapsed = t3.elapsed();
    let relu2_us = (relu2_elapsed.as_nanos() as f64 / subln_iters as f64) / 1000.0;

    println!("\n--- 2. Normalization & Activation Kernels ---");
    println!("  SubLN Absmax (2560D)   : {subln_us:.3} µs/call");
    println!("  ReLU^2 Gating (6912D)  : {relu2_us:.3} µs/call");

    // 3. BitLinear Matrix-Vector GEMV (RVV vs IME & Bit-Identity)
    let shapes = [
        ("Single Row (1 x 256)", 1, 256, QuantType::TQ1_0, 10_000),
        ("Tile 8x256 (8 x 256)", 8, 256, QuantType::TQ1_0, 5_000),
        ("Attn Proj (2560 x 2560)", 2560, 2560, QuantType::TQ1_0, 100),
        ("FFN Proj (2560 x 6912)", 6912, 2560, QuantType::TQ2_0, 50),
    ];

    println!("\n--- 3. BitLinear GEMV (X100 RVV vs A100 IME) ---");
    for (label, rows, cols, quant, iters) in shapes {
        let iters = if cfg!(debug_assertions) { 2 } else { iters };
        let raw_w = if quant == QuantType::TQ1_0 {
            create_tq1_0_buffer(rows, cols)
        } else {
            create_tq2_0_buffer(rows, cols)
        };
        let shape = [cols, rows];
        let weights = TensorView::from_bytes("gemv", quant, &shape, 0, &raw_w).unwrap();
        let act = vec![1i8; cols];
        let mut out_rvv = vec![0.0f32; rows];
        let mut out_ime = vec![0.0f32; rows];

        let t_rvv = Instant::now();
        for _ in 0..iters {
            bitlinear_i8_rvv(&mut out_rvv, &weights, &act, 0.05).unwrap();
        }
        let rvv_dur = t_rvv.elapsed();
        let rvv_ns = rvv_dur.as_nanos() as f64 / iters as f64;

        let t_ime = Instant::now();
        for _ in 0..iters {
            bitlinear_i8_ime(&mut out_ime, &weights, &act, 0.05).unwrap();
        }
        let ime_dur = t_ime.elapsed();
        let ime_ns = ime_dur.as_nanos() as f64 / iters as f64;

        let speedup = (rvv_ns - ime_ns) / rvv_ns * 100.0;
        let mut bit_identical = true;
        for i in 0..rows {
            if out_rvv[i].to_bits() != out_ime[i].to_bits() {
                bit_identical = false;
                break;
            }
        }

        println!(
            "  {label:<24} | RVV: {rvv_ns:>10.1} ns | IME: {ime_ns:>10.1} ns | Speedup: {speedup:>5.1}% | Bit-Identical: {bit_identical}"
        );
        assert!(bit_identical, "RVV and IME must be 100% bit-identical!");
    }
}

#[test]
fn test_target_hardware_full_model_inference() {
    println!("\n==========================================================");
    println!("  SpacemiT K3 Target Hardware Full-Model 30-Layer Benchmark");
    println!("==========================================================");

    let (
        num_layers,
        hidden_dim,
        ffn_dim,
        n_heads_q,
        n_heads_kv,
        head_dim,
        total_q,
        total_kv,
        vocab_size,
        n_runs,
        n_gen_tokens,
    ) = if cfg!(debug_assertions) {
        (2, 256, 512, 4, 1, 64, 256, 64, 512, 1, 1)
    } else {
        (30, 2560, 6912, 20, 5, 128, 2560, 640, 32000, 3, 2)
    };

    let prompt_tokens = (0..4)
        .map(|x| (x * 7 + 3) % vocab_size as u32)
        .collect::<Vec<_>>();

    let model_variants = [
        ("bitnet-b1.58 (30L)", QuantType::TQ1_0, true),
        ("bitnet-b1.58 (30L)", QuantType::TQ2_0, true),
        ("bitnet-25 (SwiGLU, 30L)", QuantType::TQ1_0, false),
        ("bitnet-b1.58 (30L)", QuantType::Q8_0, true),
        ("Reference F16 (30L)", QuantType::F16, true),
    ];

    for (name, quant, use_relu2) in model_variants {
        println!("\n----------------------------------------------------------");
        println!("  Evaluating: {name} [{quant:?}]");
        println!("----------------------------------------------------------");

        // Build 30-layer model weights
        let t_build = Instant::now();
        let mut weights = TargetBenchmarkWeights::new();

        weights.insert_tensor(
            "token_embd.weight",
            QuantType::F32,
            vec![hidden_dim, vocab_size],
            create_f32_buffer(vocab_size * hidden_dim, 0.25),
        );
        weights.insert_tensor(
            "output_norm.weight",
            QuantType::F32,
            vec![hidden_dim],
            create_f32_buffer(hidden_dim, 1.0),
        );

        let out_bytes = match quant {
            QuantType::TQ1_0 => create_tq1_0_buffer(vocab_size, hidden_dim),
            QuantType::TQ2_0 => create_tq2_0_buffer(vocab_size, hidden_dim),
            QuantType::Q8_0 => create_q8_0_buffer(vocab_size, hidden_dim),
            _ => create_f16_buffer(vocab_size, hidden_dim),
        };
        weights.insert_tensor(
            "output.weight",
            quant,
            vec![hidden_dim, vocab_size],
            out_bytes,
        );

        for l in 0..num_layers {
            weights.insert_tensor(
                format!("blk.{l}.attn_norm.weight"),
                QuantType::F32,
                vec![hidden_dim],
                create_f32_buffer(hidden_dim, 1.0),
            );
            weights.insert_tensor(
                format!("blk.{l}.attn_sub_norm.weight"),
                QuantType::F32,
                vec![total_q],
                create_f32_buffer(total_q, 1.0),
            );
            weights.insert_tensor(
                format!("blk.{l}.ffn_norm.weight"),
                QuantType::F32,
                vec![hidden_dim],
                create_f32_buffer(hidden_dim, 1.0),
            );
            weights.insert_tensor(
                format!("blk.{l}.ffn_sub_norm.weight"),
                QuantType::F32,
                vec![ffn_dim],
                create_f32_buffer(ffn_dim, 1.0),
            );

            let q_raw = match quant {
                QuantType::TQ1_0 => create_tq1_0_buffer(total_q, hidden_dim),
                QuantType::TQ2_0 => create_tq2_0_buffer(total_q, hidden_dim),
                QuantType::Q8_0 => create_q8_0_buffer(total_q, hidden_dim),
                _ => create_f16_buffer(total_q, hidden_dim),
            };
            weights.insert_tensor(
                format!("blk.{l}.attn_q.weight"),
                quant,
                vec![hidden_dim, total_q],
                q_raw,
            );

            let k_raw = match quant {
                QuantType::TQ1_0 => create_tq1_0_buffer(total_kv, hidden_dim),
                QuantType::TQ2_0 => create_tq2_0_buffer(total_kv, hidden_dim),
                QuantType::Q8_0 => create_q8_0_buffer(total_kv, hidden_dim),
                _ => create_f16_buffer(total_kv, hidden_dim),
            };
            weights.insert_tensor(
                format!("blk.{l}.attn_k.weight"),
                quant,
                vec![hidden_dim, total_kv],
                k_raw,
            );

            let v_raw = match quant {
                QuantType::TQ1_0 => create_tq1_0_buffer(total_kv, hidden_dim),
                QuantType::TQ2_0 => create_tq2_0_buffer(total_kv, hidden_dim),
                QuantType::Q8_0 => create_q8_0_buffer(total_kv, hidden_dim),
                _ => create_f16_buffer(total_kv, hidden_dim),
            };
            weights.insert_tensor(
                format!("blk.{l}.attn_v.weight"),
                quant,
                vec![hidden_dim, total_kv],
                v_raw,
            );

            let o_raw = match quant {
                QuantType::TQ1_0 => create_tq1_0_buffer(hidden_dim, total_q),
                QuantType::TQ2_0 => create_tq2_0_buffer(hidden_dim, total_q),
                QuantType::Q8_0 => create_q8_0_buffer(hidden_dim, total_q),
                _ => create_f16_buffer(hidden_dim, total_q),
            };
            weights.insert_tensor(
                format!("blk.{l}.attn_output.weight"),
                quant,
                vec![total_q, hidden_dim],
                o_raw,
            );

            let gate_raw = match quant {
                QuantType::TQ1_0 => create_tq1_0_buffer(ffn_dim, hidden_dim),
                QuantType::TQ2_0 => create_tq2_0_buffer(ffn_dim, hidden_dim),
                QuantType::Q8_0 => create_q8_0_buffer(ffn_dim, hidden_dim),
                _ => create_f16_buffer(ffn_dim, hidden_dim),
            };
            weights.insert_tensor(
                format!("blk.{l}.ffn_gate.weight"),
                quant,
                vec![hidden_dim, ffn_dim],
                gate_raw,
            );

            let up_raw = match quant {
                QuantType::TQ1_0 => create_tq1_0_buffer(ffn_dim, hidden_dim),
                QuantType::TQ2_0 => create_tq2_0_buffer(ffn_dim, hidden_dim),
                QuantType::Q8_0 => create_q8_0_buffer(ffn_dim, hidden_dim),
                _ => create_f16_buffer(ffn_dim, hidden_dim),
            };
            weights.insert_tensor(
                format!("blk.{l}.ffn_up.weight"),
                quant,
                vec![hidden_dim, ffn_dim],
                up_raw,
            );

            let down_raw = match quant {
                QuantType::TQ1_0 => create_tq1_0_buffer(hidden_dim, ffn_dim),
                QuantType::TQ2_0 => create_tq2_0_buffer(hidden_dim, ffn_dim),
                QuantType::Q8_0 => create_q8_0_buffer(hidden_dim, ffn_dim),
                _ => create_f16_buffer(hidden_dim, ffn_dim),
            };
            weights.insert_tensor(
                format!("blk.{l}.ffn_down.weight"),
                quant,
                vec![ffn_dim, hidden_dim],
                down_raw,
            );
        }

        let total_bytes = weights.total_weight_bytes();
        let total_mb = total_bytes as f64 / 1_000_000.0;
        println!(
            "  Weight Memory (mlock)  : {total_mb:.1} MB (built in {:.2?})",
            t_build.elapsed()
        );

        let recipe = BitNetRecipe {
            architecture: if use_relu2 {
                "bitnet-b1.58"
            } else {
                "bitnet-25"
            },
            num_layers,
            hidden_dim,
            ffn_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            vocab_size,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            use_relu2,
        };

        // Multi-run evaluation across N runs at greedy T=0
        let mut prefill_tok_s = Vec::new();
        let mut decode_tok_s = Vec::new();
        let mut ttft_ms = Vec::new();
        let mut total_wall_ms = Vec::new();
        let mut run_digests = Vec::new();

        for r in 0..n_runs {
            let mut logits = vec![0.0f32; vocab_size];
            let kv_len = 2 * total_kv * 128 * num_layers;
            let mut kv_cache = vec![0.0f32; kv_len];
            let mut ctx =
                ExecutionContext::new(&prompt_tokens, 128, &mut logits, &mut kv_cache).unwrap();

            let t_start = Instant::now();
            recipe.prefill_step(&mut ctx, &weights).expect("prefill");
            let t_prefill = t_start.elapsed();
            let ttft = t_prefill.as_secs_f64() * 1000.0;
            let prefill_throughput = prompt_tokens.len() as f64 / t_prefill.as_secs_f64();

            let mut generated = Vec::new();
            let t_decode_start = Instant::now();
            for _step in 0..n_gen_tokens {
                let next_token = (ctx.position() as u32 * 13 + 5) % vocab_size as u32;
                ctx.prepare_decode(next_token).unwrap();
                recipe.decode_step(&mut ctx, &weights).expect("decode");
                generated.push(ctx.logits()[0].to_bits());
            }
            let t_decode = t_decode_start.elapsed();
            let decode_throughput = n_gen_tokens as f64 / t_decode.as_secs_f64();
            let total_dur = t_start.elapsed();

            prefill_tok_s.push(prefill_throughput);
            decode_tok_s.push(decode_throughput);
            ttft_ms.push(ttft);
            total_wall_ms.push(total_dur.as_secs_f64() * 1000.0);
            run_digests.push(generated);

            println!(
                "  Run {}: TTFT: {:>7.1} ms | Prefill: {:>5.1} tok/s | Decode: {:>5.1} tok/s | Wall: {:>7.1} ms",
                r + 1,
                ttft,
                prefill_throughput,
                decode_throughput,
                total_dur.as_secs_f64() * 1000.0
            );
        }

        // Verify 100% bit-identical determinism across runs
        let determinism_verified = if n_runs >= 2 {
            run_digests.windows(2).all(|w| w[0] == w[1])
        } else {
            true
        };
        println!("  Determinism (N={n_runs})      : Verified = {determinism_verified}");
        assert!(
            determinism_verified,
            "Greedy execution must be bit-identical across runs!"
        );

        let mean_prefill = prefill_tok_s.iter().sum::<f64>() / n_runs as f64;
        let mean_decode = decode_tok_s.iter().sum::<f64>() / n_runs as f64;
        let mean_ttft = ttft_ms.iter().sum::<f64>() / n_runs as f64;
        let stddev_ttft = (ttft_ms
            .iter()
            .map(|&x| (x - mean_ttft).powi(2))
            .sum::<f64>()
            / n_runs as f64)
            .sqrt();

        println!(
            "  Summary Mean           : Prefill: {mean_prefill:.1} tok/s | Decode: {mean_decode:.1} tok/s | TTFT: {mean_ttft:.1} ± {stddev_ttft:.2} ms"
        );
    }
}
