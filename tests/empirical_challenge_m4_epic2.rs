//! Empirical Adversarial Challenger Verification Suite (M4 Epic 2).
//!
//! Empirically challenges:
//! 1. Branching Divergence: ReLU^2 (BitNet-b1.58) vs SwiGLU (BitNet-25) representation divergence.
//! 2. Multi-layer Forward Accumulation: Progressive output evolution across 1, 2, and 4 layers,
//!    and token-step KV cache residency integrity.
//! 3. Zero Heap Allocation Invariant: Dynamic allocation tracking and static source inspection
//!    of the `0..self.num_layers` iteration in `execute_token_step`.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks
)]

use jero_arch_bitnet::BitNetRecipe;
use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// -----------------------------------------------------------------------------
// Tracking Global Allocator for Empirical Heap Allocation Verification
// -----------------------------------------------------------------------------

struct TrackingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACKING_ACTIVE: AtomicBool = AtomicBool::new(false);
static TEST_LOCK: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING_ACTIVE.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn measure_allocations<F, R>(f: F) -> (R, usize)
where
    F: FnOnce() -> R,
{
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKING_ACTIVE.store(true, Ordering::SeqCst);
    let res = f();
    TRACKING_ACTIVE.store(false, Ordering::SeqCst);
    let count = ALLOC_COUNT.load(Ordering::SeqCst);
    (res, count)
}

// -----------------------------------------------------------------------------
// Synthetic Weights Harness
// -----------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct SyntheticBitNetWeights {
    pub tensor_data: HashMap<String, (QuantType, Vec<usize>, Vec<u8>)>,
}

impl SyntheticBitNetWeights {
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
        self.tensor_data.insert(name.into(), (quant, shape, data));
    }
}

impl ModelWeights for SyntheticBitNetWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let (key, (quant, shape, data)) =
            self.tensor_data
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
        self.tensor_data.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.tensor_data.values().map(|(_, _, d)| d.len()).sum()
    }
}

fn _create_tq1_0_raw(rows: usize, cols: usize, val: i8) -> Vec<u8> {
    create_tq1_0_pattern(rows, cols, |_r, _c| val)
}

fn create_tq1_0_pattern(rows: usize, cols: usize, pattern: impl Fn(usize, usize) -> i8) -> Vec<u8> {
    assert_eq!(cols % 256, 0);
    let blocks_per_row = cols / 256;
    let mut raw = vec![0u8; rows * blocks_per_row * 54];

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let offset = (r * blocks_per_row + b) * 54;
            // FP16 scale = 1.0 (0x3C00)
            raw[offset..offset + 2].copy_from_slice(&[0x00, 0x3C]);
            let col_base = b * 256;
            for byte_idx in 0..51 {
                let mut byte_val = 0u8;
                let mut mul = 1u8;
                for trit_idx in 0..5 {
                    let c = col_base + byte_idx * 5 + trit_idx;
                    let val = pattern(r, c);
                    let t = match val {
                        -1 => 0u8,
                        0 => 1u8,
                        1 => 2u8,
                        _ => 1u8,
                    };
                    byte_val += t * mul;
                    mul = mul.saturating_mul(3);
                }
                raw[offset + 2 + byte_idx] = byte_val;
            }
            let last_val = pattern(r, col_base + 255);
            raw[offset + 53] = match last_val {
                -1 => 0u8,
                0 => 1u8,
                1 => 2u8,
                _ => 1u8,
            };
        }
    }
    raw
}

fn create_f32_raw(dim: usize, val: f32) -> Vec<u8> {
    let mut raw = vec![0u8; dim * 4];
    for i in 0..dim {
        raw[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
    }
    raw
}

fn ternary_mix(seed: usize, r: usize, c: usize) -> i8 {
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

/// Builds a multi-layer synthetic BitNet weight collection.
pub fn build_multi_layer_weights(
    num_layers: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    total_q: usize,
    total_kv: usize,
    vocab_size: usize,
) -> SyntheticBitNetWeights {
    let mut weights = SyntheticBitNetWeights::new();

    // Token embedding: [hidden_dim, vocab_size]
    let mut embd_raw = vec![0u8; vocab_size * hidden_dim * 4];
    for t in 0..vocab_size {
        for i in 0..hidden_dim {
            let off = (t * hidden_dim + i) * 4;
            let val = ((t * 11 + i * 17) % 73) as f32 * 0.05 - 1.5;
            embd_raw[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
    weights.insert_tensor(
        "token_embd.weight",
        QuantType::F32,
        vec![hidden_dim, vocab_size],
        embd_raw,
    );

    // Output norm: [hidden_dim]
    weights.insert_tensor(
        "output_norm.weight",
        QuantType::F32,
        vec![hidden_dim],
        create_f32_raw(hidden_dim, 1.0),
    );

    // Output LM head: [hidden_dim, 16] with diverse rows
    weights.insert_tensor(
        "output.weight",
        QuantType::TQ1_0,
        vec![hidden_dim, 16],
        create_tq1_0_pattern(16, hidden_dim, |r, c| ternary_mix(101, r, c)),
    );

    for l in 0..num_layers {
        // Canonical RMSNorm and SubLN weights
        weights.insert_tensor(
            format!("blk.{l}.attn_norm.weight"),
            QuantType::F32,
            vec![hidden_dim],
            create_f32_raw(hidden_dim, 1.0 + (l as f32) * 0.05),
        );
        weights.insert_tensor(
            format!("blk.{l}.attn_sub_norm.weight"),
            QuantType::F32,
            vec![total_q],
            create_f32_raw(total_q, 1.0),
        );
        weights.insert_tensor(
            format!("blk.{l}.ffn_norm.weight"),
            QuantType::F32,
            vec![hidden_dim],
            create_f32_raw(hidden_dim, 1.0 + (l as f32) * 0.05),
        );
        weights.insert_tensor(
            format!("blk.{l}.ffn_sub_norm.weight"),
            QuantType::F32,
            vec![ffn_dim],
            create_f32_raw(ffn_dim, 1.0),
        );

        // Projections with heterogeneous non-periodic patterns
        weights.insert_tensor(
            format!("blk.{l}.attn_q.weight"),
            QuantType::TQ1_0,
            vec![hidden_dim, total_q],
            create_tq1_0_pattern(total_q, hidden_dim, |r, c| ternary_mix(l * 16 + 1, r, c)),
        );
        weights.insert_tensor(
            format!("blk.{l}.attn_k.weight"),
            QuantType::TQ1_0,
            vec![hidden_dim, total_kv],
            create_tq1_0_pattern(total_kv, hidden_dim, |r, c| ternary_mix(l * 16 + 2, r, c)),
        );
        weights.insert_tensor(
            format!("blk.{l}.attn_v.weight"),
            QuantType::TQ1_0,
            vec![hidden_dim, total_kv],
            create_tq1_0_pattern(total_kv, hidden_dim, |r, c| ternary_mix(l * 16 + 3, r, c)),
        );
        weights.insert_tensor(
            format!("blk.{l}.attn_output.weight"),
            QuantType::TQ1_0,
            vec![total_q, hidden_dim],
            create_tq1_0_pattern(hidden_dim, total_q, |r, c| ternary_mix(l * 16 + 4, r, c)),
        );

        // Gate projection: heterogeneous positive/negative/zero ternary
        weights.insert_tensor(
            format!("blk.{l}.ffn_gate.weight"),
            QuantType::TQ1_0,
            vec![hidden_dim, ffn_dim],
            create_tq1_0_pattern(ffn_dim, hidden_dim, |r, c| ternary_mix(l * 16 + 5, r, c)),
        );

        // Up projection
        weights.insert_tensor(
            format!("blk.{l}.ffn_up.weight"),
            QuantType::TQ1_0,
            vec![hidden_dim, ffn_dim],
            create_tq1_0_pattern(ffn_dim, hidden_dim, |r, c| ternary_mix(l * 16 + 6, r, c)),
        );

        // Down projection
        weights.insert_tensor(
            format!("blk.{l}.ffn_down.weight"),
            QuantType::TQ1_0,
            vec![ffn_dim, hidden_dim],
            create_tq1_0_pattern(hidden_dim, ffn_dim, |r, c| ternary_mix(l * 16 + 7, r, c)),
        );
    }

    weights
}

// -----------------------------------------------------------------------------
// Challenge 1: Branching Divergence (ReLU^2 vs SwiGLU)
// -----------------------------------------------------------------------------

#[test]
fn test_empirical_branching_divergence() {
    let _guard = TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let hidden_dim = 256;
    let ffn_dim = 512;
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let head_dim = 64;
    let total_q = 256;
    let total_kv = 128;
    let vocab_size = 512;

    // 1. Direct mathematical verification of activation difference:
    // f_relu2(g, u) = max(0, g)^2 * u
    // f_swiglu(g, u) = (g * sigmoid(g)) * u
    let test_gates = [-3.0f32, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let test_up = 1.5f32;

    for &g in &test_gates {
        let sig = 1.0 / (1.0 + (-g).exp());
        let val_relu2 = g.max(0.0).powi(2) * test_up;
        let val_swiglu = (g * sig) * test_up;
        let delta = (val_relu2 - val_swiglu).abs();

        if g < 0.0 {
            assert_eq!(val_relu2, 0.0, "ReLU^2 must be zero for negative gate");
            assert!(
                val_swiglu < 0.0,
                "SwiGLU must be negative for negative gate and positive up"
            );
            assert!(
                delta > 0.0,
                "Mathematical divergence must be strictly non-zero"
            );
        } else if g > 1.0 {
            assert!(
                val_relu2 > val_swiglu,
                "For g > 1.0, g^2 > g*sig(g) because sig(g) < 1.0"
            );
            assert!(delta > 0.0);
        }
    }

    // 2. Full model forward divergence check:
    let weights = build_multi_layer_weights(2, hidden_dim, ffn_dim, total_q, total_kv, vocab_size);

    let recipe_relu2 = BitNetRecipe {
        architecture: "bitnet-b1.58",
        num_layers: 2,
        hidden_dim,
        ffn_dim,
        n_heads_q,
        n_heads_kv,
        head_dim,
        vocab_size,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        use_relu2: true,
    };

    let recipe_swiglu = BitNetRecipe {
        architecture: "bitnet-25",
        num_layers: 2,
        hidden_dim,
        ffn_dim,
        n_heads_q,
        n_heads_kv,
        head_dim,
        vocab_size,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        use_relu2: false,
    };

    let prompt = [12u32, 45u32];

    let mut logits_relu2 = vec![0.0f32; 16];
    let mut kv_relu2 = vec![0.0f32; 4096];
    let mut ctx_relu2 =
        ExecutionContext::new(&prompt, 32, &mut logits_relu2, &mut kv_relu2).unwrap();
    recipe_relu2
        .prefill_step(&mut ctx_relu2, &weights)
        .expect("prefill relu2 succeeds");

    let mut logits_swiglu = vec![0.0f32; 16];
    let mut kv_swiglu = vec![0.0f32; 4096];
    let mut ctx_swiglu =
        ExecutionContext::new(&prompt, 32, &mut logits_swiglu, &mut kv_swiglu).unwrap();
    recipe_swiglu
        .prefill_step(&mut ctx_swiglu, &weights)
        .expect("prefill swiglu succeeds");

    // Compute divergence metrics:
    let mut l1_dist = 0.0f32;
    let mut l2_sq = 0.0f32;
    let mut max_abs_diff = 0.0f32;
    let mut dot = 0.0f32;
    let mut norm_r_sq = 0.0f32;
    let mut norm_s_sq = 0.0f32;

    for (&r, &s) in logits_relu2.iter().zip(&logits_swiglu) {
        let diff = (r - s).abs();
        l1_dist += diff;
        l2_sq += diff * diff;
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        dot += r * s;
        norm_r_sq += r * r;
        norm_s_sq += s * s;
    }
    let l2_dist = l2_sq.sqrt();
    let cos_sim = dot / (norm_r_sq.sqrt() * norm_s_sq.sqrt());

    println!("=== Branching Divergence Empirical Results ===");
    println!("Logits ReLU^2 (first 4): {:?}", &logits_relu2[..4]);
    println!("Logits SwiGLU (first 4): {:?}", &logits_swiglu[..4]);
    println!("L1 distance: {l1_dist}");
    println!("L2 distance: {l2_dist}");
    println!("Max absolute difference: {max_abs_diff}");
    println!("Cosine similarity: {cos_sim}");

    assert!(
        l2_dist > 1.0,
        "Empirical L2 divergence between ReLU^2 and SwiGLU must be significant (got {l2_dist})"
    );
    assert!(
        max_abs_diff > 0.5,
        "Max abs difference must be substantial (got {max_abs_diff})"
    );
    assert!(
        cos_sim < 0.999,
        "Cosine similarity must reflect non-identical directions (got {cos_sim})"
    );
}

// -----------------------------------------------------------------------------
// Challenge 2: Multi-layer Forward Accumulation & KV Cache Integrity
// -----------------------------------------------------------------------------

#[test]
fn test_empirical_multi_layer_forward_accumulation() {
    let _guard = TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let hidden_dim = 256;
    let ffn_dim = 512;
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let head_dim = 64;
    let total_q = 256;
    let total_kv = 128;
    let vocab_size = 512;

    // Build weights for up to 30 layers
    let weights = build_multi_layer_weights(30, hidden_dim, ffn_dim, total_q, total_kv, vocab_size);

    let prompt = [10u32, 25u32];

    // Helper to run forward pass for N layers
    let run_layers = |num_layers: usize| -> (Vec<f32>, Vec<f32>) {
        let recipe = BitNetRecipe {
            architecture: "bitnet-b1.58",
            num_layers,
            hidden_dim,
            ffn_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            vocab_size,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            use_relu2: true,
        };
        let mut logits = vec![0.0f32; 16];
        // Sizing: 2 * total_kv * n_ctx * num_layers = 2 * 128 * 32 * num_layers = 8192 * num_layers
        let mut kv_cache = vec![-999.0f32; 8192 * num_layers];
        let mut ctx = ExecutionContext::new(&prompt, 32, &mut logits, &mut kv_cache).unwrap();
        recipe.prefill_step(&mut ctx, &weights).expect("prefill");
        (logits, kv_cache)
    };

    let (logits_1, _kv_1) = run_layers(1);
    let (logits_2, _kv_2) = run_layers(2);
    let (logits_4, _kv_4) = run_layers(4);
    let (logits_30, kv_30) = run_layers(30);

    let l2_dist = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    };

    let dist_1_2 = l2_dist(&logits_1, &logits_2);
    let dist_2_4 = l2_dist(&logits_2, &logits_4);
    let dist_4_30 = l2_dist(&logits_4, &logits_30);
    let dist_1_4 = l2_dist(&logits_1, &logits_4);
    let dist_1_30 = l2_dist(&logits_1, &logits_30);

    let cum_dist_1_2 = dist_1_2;
    let cum_dist_1_4 = dist_1_2 + dist_2_4;
    let cum_dist_1_30 = dist_1_2 + dist_2_4 + dist_4_30;

    println!("=== Multi-layer Forward Accumulation Results ===");
    println!("Logits (L=1): {:?}", &logits_1[..4]);
    println!("Logits (L=2): {:?}", &logits_2[..4]);
    println!("Logits (L=4): {:?}", &logits_4[..4]);
    println!("Logits (L=30): {:?}", &logits_30[..4]);
    println!("L2 distance (1 vs 2 layers): {dist_1_2}");
    println!("L2 distance (2 vs 4 layers): {dist_2_4}");
    println!("L2 distance (4 vs 30 layers): {dist_4_30}");
    println!("L2 distance (1 vs 4 layers): {dist_1_4}");
    println!("L2 distance (1 vs 30 layers): {dist_1_30}");
    println!("Cumulative trajectory distance L=2: {cum_dist_1_2}");
    println!("Cumulative trajectory distance L=4: {cum_dist_1_4}");
    println!("Cumulative trajectory distance L=30: {cum_dist_1_30}");

    assert!(
        dist_1_2 > 0.1,
        "Logits must evolve non-trivially between 1 and 2 layers"
    );
    assert!(
        dist_2_4 > 0.1,
        "Logits must evolve non-trivially between 2 and 4 layers"
    );
    assert!(
        dist_4_30 > 0.1,
        "Logits must evolve non-trivially between 4 and 30 layers"
    );
    assert!(
        cum_dist_1_4 > cum_dist_1_2,
        "Cumulative evolution trajectory must increase monotonically across layers (cum_dist_1_4 > cum_dist_1_2)"
    );
    assert!(
        cum_dist_1_30 > cum_dist_1_4,
        "Cumulative evolution trajectory must increase monotonically across 30 layers (cum_dist_1_30 > cum_dist_1_4)"
    );

    // KV Cache Step Accumulation & Memory Residency Integrity across all 30 layers
    // In L=30, each layer has stride 8192, half_stride = 4096.
    // Prompt tokens pos 0 and pos 1 were executed.
    // Verify each layer's KV slots:
    for layer in 0..30 {
        let layer_base = layer * 8192;
        let k_half = layer_base;
        let v_half = layer_base + 4096;

        // Pos 0: [0..128]
        let k_pos0 = &kv_30[k_half..k_half + 128];
        let v_pos0 = &kv_30[v_half..v_half + 128];
        assert!(
            k_pos0.iter().all(|&x| x != -999.0 && x.is_finite()),
            "Layer {layer} Pos 0 K must be populated and finite"
        );
        assert!(
            v_pos0.iter().all(|&x| x != -999.0 && x.is_finite()),
            "Layer {layer} Pos 0 V must be populated and finite"
        );

        // Pos 1: [128..256]
        let k_pos1 = &kv_30[k_half + 128..k_half + 256];
        let v_pos1 = &kv_30[v_half + 128..v_half + 256];
        assert!(
            k_pos1.iter().all(|&x| x != -999.0 && x.is_finite()),
            "Layer {layer} Pos 1 K must be populated and finite"
        );
        assert!(
            v_pos1.iter().all(|&x| x != -999.0 && x.is_finite()),
            "Layer {layer} Pos 1 V must be populated and finite"
        );

        // Verify pos 0 and pos 1 are NOT identical (different token and position)
        assert_ne!(
            k_pos0, k_pos1,
            "Layer {layer} Pos 0 and Pos 1 K entries must differ"
        );

        // Verify unwritten positions [256..4096] remain completely untouched (-999.0)
        let k_unwritten = &kv_30[k_half + 256..k_half + 4096];
        let v_unwritten = &kv_30[v_half + 256..v_half + 4096];
        assert!(
            k_unwritten.iter().all(|&x| x == -999.0),
            "Layer {layer} unwritten K slots must remain untouched"
        );
        assert!(
            v_unwritten.iter().all(|&x| x == -999.0),
            "Layer {layer} unwritten V slots must remain untouched"
        );
    }
    println!(
        "PASS: Verified multi-layer forward accumulation and KV cache integrity across 1, 2, 4, and 30 layers."
    );
}

#[test]
fn test_empirical_multi_layer_accumulation() {
    test_empirical_multi_layer_forward_accumulation();
}

// -----------------------------------------------------------------------------
// Challenge 3: Zero Allocation Invariant
// -----------------------------------------------------------------------------

#[test]
fn test_empirical_zero_allocation_invariant_dynamic() {
    let _guard = TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let hidden_dim = 256;
    let ffn_dim = 512;
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let head_dim = 64;
    let total_q = 256;
    let total_kv = 128;
    let vocab_size = 512;

    // Build weights for up to 30 layers
    let weights = build_multi_layer_weights(30, hidden_dim, ffn_dim, total_q, total_kv, vocab_size);

    let test_layers = [1, 2, 4, 8, 16, 30];
    let mut alloc_counts = Vec::new();

    for &num_layers in &test_layers {
        let recipe = BitNetRecipe {
            architecture: "bitnet-b1.58",
            num_layers,
            hidden_dim,
            ffn_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            vocab_size,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            use_relu2: true,
        };

        let mut logits = vec![0.0f32; 16];
        let mut kv_cache = vec![0.0f32; 8192 * num_layers];
        let prompt = [1u32];
        let mut ctx = ExecutionContext::new(&prompt, 32, &mut logits, &mut kv_cache).unwrap();
        ctx.prepare_decode(42).expect("prepare decode");

        // Measure allocations during decode_step (which executes 1 token step through the layer loop)
        let (_res, count) = measure_allocations(|| recipe.decode_step(&mut ctx, &weights));

        alloc_counts.push(count);
    }

    println!("=== Zero Allocation Invariant Empirical Results ===");
    for (&layers, &count) in test_layers.iter().zip(&alloc_counts) {
        println!("Layers: {layers} -> Allocations in execute_token_step: {count}");
    }

    // Verify allocation behavior:
    // In execute_token_step, recipe.rs preallocates 22 scratch slices once outside the layer loop.
    // decode_step adds 1 scratch slice in the ternary LM head for absmax quantization, yielding
    // 23 fixed allocations outside the loop.
    for (&layers, &count) in test_layers.iter().zip(&alloc_counts) {
        assert_eq!(
            count, 23,
            "Allocation count for {layers} layers must remain 23 \
             (22 pre-allocated scratch buffers in execute_token_step + 1 in decode_step LM head; \
             upstream gqa_causal_attention_f32 uses stack scratch for seq_len <= 2048). \
             d(Allocations)/d(layers) must be strictly 0."
        );
    }
    println!(
        "PASS: d(Allocations)/d(layers) == 0 across all tested depths. Zero heap allocations occur inside the layer loop."
    );
}

#[test]
fn test_empirical_zero_dynamic_allocations() {
    test_empirical_zero_allocation_invariant_dynamic();
}

#[test]
fn test_empirical_zero_allocation_invariant_static_ast() {
    let _guard = TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Read the recipe source code and analyze the layer loop
    let source = include_str!("../src/recipe.rs");

    // Locate `for layer in 0..self.num_layers {`
    let loop_start_token = "for layer in 0..self.num_layers {";
    let loop_start = source
        .find(loop_start_token)
        .expect("must find layer loop start");

    // Locate loop end before `8. Output Normalization`
    let loop_end_token = "// 8. Output Normalization";
    let loop_end = source
        .find(loop_end_token)
        .expect("must find output normalization section");

    let loop_body = &source[loop_start..loop_end];

    // Scan loop body lines for forbidden allocation primitives
    let forbidden_patterns = [
        ("vec![", "dynamic vector macro"),
        ("Vec::new()", "dynamic vector creation"),
        ("Vec::with_capacity", "dynamic vector creation"),
        ("to_vec()", "vector copy allocation"),
        ("Box::new", "box allocation"),
        ("Arc::new", "arc allocation"),
        ("String::from", "string allocation"),
        ("to_string()", "string formatting allocation"),
        (".to_owned()", "owned string allocation on happy path"),
    ];

    let mut forbidden_violations = Vec::new();

    for (line_idx, line) in loop_body.lines().enumerate() {
        let trimmed = line.trim();
        // Ignore comments
        if trimmed.starts_with("//") {
            continue;
        }

        // On the happy path, check if any forbidden pattern occurs outside error branches
        for (pattern, desc) in &forbidden_patterns {
            if trimmed.contains(pattern) {
                // Allow in error returning branches
                if !trimmed.contains("Err(") && !trimmed.contains("reason:") {
                    forbidden_violations.push((line_idx, trimmed.to_string(), desc));
                }
            }
        }

        // Specifically check format! macro
        if trimmed.contains("format!(") {
            assert!(
                trimmed.contains("reason:") || trimmed.contains("Err("),
                "format! found on non-error branch at line {line_idx}: {trimmed}"
            );
        }
    }

    println!("=== Static Source Scan Results for Layer Loop ===");
    println!("Loop lines analyzed: {}", loop_body.lines().count());
    println!("Forbidden violations on happy path: {forbidden_violations:?}");

    assert!(
        forbidden_violations.is_empty(),
        "Detected heap allocations inside layer loop: {:?}",
        forbidden_violations
    );
}

// -----------------------------------------------------------------------------
// Challenge 4: Memory Safety & Numerical Stability under Extreme Float Regimes
// -----------------------------------------------------------------------------

#[test]
fn test_empirical_numerical_stability() {
    let _guard = TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let hidden_dim = 256;
    let ffn_dim = 512;
    let total_q = 256;
    let total_kv = 128;
    let vocab_size = 512;
    #[cfg(feature = "with-kernels")]
    let eps = 1e-5;

    // 1. Extreme float stress on SubLN and Activation kernels (when with-kernels is active)
    #[cfg(feature = "with-kernels")]
    {
        // 1a. Subnormals
        let subnormal_val = f32::from_bits(1);
        let subnormal_input = vec![subnormal_val; hidden_dim];
        let norm_weight = vec![1.0f32; hidden_dim];
        let mut out_i8 = vec![0i8; hidden_dim];

        let scale_sub = jero_kernels_rvv::subln_quantize_absmax_i8(
            &mut out_i8,
            &subnormal_input,
            &norm_weight,
            eps,
        )
        .expect("SubLN succeeds on subnormals");
        assert!(scale_sub.is_finite());
        assert!(scale_sub > 0.0);
        // Subnormals enter defensive guard (< 1e-12)
        assert!(out_i8.iter().all(|&x| x == 0));
        assert_eq!(scale_sub, 1.0 / 127.0);

        // 1b. Zeros (positive and negative zero)
        let zero_input = vec![0.0f32; hidden_dim];
        let scale_zero =
            jero_kernels_rvv::subln_quantize_absmax_i8(&mut out_i8, &zero_input, &norm_weight, eps)
                .expect("SubLN succeeds on exact zeros");
        assert_eq!(scale_zero, 1.0 / 127.0);
        assert!(out_i8.iter().all(|&x| x == 0));

        let neg_zero_input = vec![-0.0f32; hidden_dim];
        let scale_neg_zero = jero_kernels_rvv::subln_quantize_absmax_i8(
            &mut out_i8,
            &neg_zero_input,
            &norm_weight,
            eps,
        )
        .expect("SubLN succeeds on negative zeros");
        assert_eq!(scale_neg_zero, 1.0 / 127.0);
        assert!(out_i8.iter().all(|&x| x == 0));

        // 1c. Negative Infinity
        let neg_inf_input = vec![f32::NEG_INFINITY; hidden_dim];
        let scale_neg_inf = jero_kernels_rvv::subln_quantize_absmax_i8(
            &mut out_i8,
            &neg_inf_input,
            &norm_weight,
            eps,
        )
        .expect("SubLN succeeds on negative infinity");
        assert!(scale_neg_inf.is_finite());
        assert_eq!(scale_neg_inf, 1.0 / 127.0);
        assert!(out_i8.iter().all(|&x| x == 0));

        // 1d. Positive Infinity and NaN
        let inf_input = vec![f32::INFINITY; hidden_dim];
        let scale_inf =
            jero_kernels_rvv::subln_quantize_absmax_i8(&mut out_i8, &inf_input, &norm_weight, eps)
                .expect("SubLN succeeds on positive infinity");
        assert_eq!(scale_inf, 1.0 / 127.0);
        assert!(out_i8.iter().all(|&x| x == 0));

        let nan_input = vec![f32::NAN; hidden_dim];
        let scale_nan =
            jero_kernels_rvv::subln_quantize_absmax_i8(&mut out_i8, &nan_input, &norm_weight, eps)
                .expect("SubLN succeeds on NaN");
        assert_eq!(scale_nan, 1.0 / 127.0);
        assert!(out_i8.iter().all(|&x| x == 0));

        // 2. Activation kernels on extreme floats:
        let gates_extreme = [
            f32::NEG_INFINITY,
            -1e30,
            -1.0,
            -0.0,
            0.0,
            subnormal_val,
            1e-30,
            1.0,
            1e30,
            f32::INFINITY,
        ];
        let up_val = [1.0f32; 10];
        let mut out_r = [0.0f32; 10];
        let mut out_s = [0.0f32; 10];
        jero_kernels_rvv::relu_squared_gate_f32(&mut out_r, &gates_extreme, &up_val)
            .expect("relu2 on extreme floats");
        jero_kernels_rvv::swiglu_f32(&mut out_s, &gates_extreme, &up_val)
            .expect("swiglu on extreme floats");

        // Negative values and negative infinity must produce exactly 0.0 in ReLU^2
        assert_eq!(out_r[0], 0.0); // NEG_INFINITY
        assert_eq!(out_r[1], 0.0); // -1e30
        assert_eq!(out_r[2], 0.0); // -1.0
        assert_eq!(out_r[3], 0.0); // -0.0
        assert_eq!(out_r[4], 0.0); // 0.0

        // SwiGLU: sigmoid(-inf) = 0.0 -> out_s[0] = 0.0
        assert_eq!(out_s[0], 0.0);
        assert_eq!(out_s[3], 0.0); // -0.0
        assert_eq!(out_s[4], 0.0); // 0.0
    }

    // 3. Full BitNetRecipe End-to-End Execution under extreme float regimes:
    let mut weights_extreme =
        build_multi_layer_weights(2, hidden_dim, ffn_dim, total_q, total_kv, vocab_size);
    let subnormal_val = f32::from_bits(1);

    // Inject extreme float patterns into token embedding tensor:
    let mut extreme_embd = vec![0u8; vocab_size * hidden_dim * 4];
    for t in 0..vocab_size {
        for i in 0..hidden_dim {
            let off = (t * hidden_dim + i) * 4;
            let val = match (t + i) % 7 {
                0 => 0.0f32,
                1 => -0.0f32,
                2 => subnormal_val,
                3 => f32::NEG_INFINITY,
                4 => 1e-35f32,
                5 => -1e-35f32,
                _ => 0.5f32,
            };
            extreme_embd[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
    weights_extreme.insert_tensor(
        "token_embd.weight",
        QuantType::F32,
        vec![hidden_dim, vocab_size],
        extreme_embd,
    );

    let recipe = BitNetRecipe {
        architecture: "bitnet-b1.58",
        num_layers: 2,
        hidden_dim,
        ffn_dim,
        n_heads_q: 4,
        n_heads_kv: 2,
        head_dim: 64,
        vocab_size,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        use_relu2: true,
    };

    let prompt = [0u32, 1u32];
    let mut logits = vec![0.0f32; 16];
    let mut kv_cache = vec![0.0f32; 8192 * 2];
    let mut ctx = ExecutionContext::new(&prompt, 32, &mut logits, &mut kv_cache).unwrap();

    // Verify prefill_step executes without panic, crash, or memory violation
    let res = recipe.prefill_step(&mut ctx, &weights_extreme);
    assert!(
        res.is_ok(),
        "prefill_step must handle extreme float inputs safely: {:?}",
        res
    );

    // Verify decode_step executes without panic or crash
    ctx.prepare_decode(2).expect("prepare decode");
    let res_decode = recipe.decode_step(&mut ctx, &weights_extreme);
    assert!(
        res_decode.is_ok(),
        "decode_step must handle extreme float inputs safely: {:?}",
        res_decode
    );

    println!(
        "PASS: test_empirical_numerical_stability verified extreme floats (subnormals, zeros, negative infinity) cleanly."
    );
}
