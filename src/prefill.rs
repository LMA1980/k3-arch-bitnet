//! Batched multi-token prefill execution, deduplicated `SubLN` activation packing,
//! and `8x16` tile GEMM dispatch for [`BitNetRecipe`].
//!
//! Replaces token-by-token prefill iteration with `[M, K]` matrix projections across cached
//! `128-byte`-aligned ternary/`Q8_0` tiles (or zero-expansion streaming tiles when uncached):
//! - **Shared Pre-QKV (`3x -> 1x`) & Pre-FFN (`2x -> 1x`) `SubLN`**: normalizes, absmax-quantizes,
//!   and packs `[blk][mt][ch][8][16]` activation tiles once per shared input buffer.
//! - **Batched `NeoX` `RoPE` & Causal `GQA` Prefill**: rotates positions `start_pos..start_pos + M`,
//!   appends `KV` chunks cleanly into `KvCacheView` or raw slices, and evaluates causal grouped-query
//!   attention via `gqa_causal_attention_prefill_f32` (or portable scalar parity fallback).
//! - **Single-Token LM Head Projection**: applies final `output_norm` and `project_lm_head` only to
//!   the last prompt token (`M - 1`).

use crate::recipe::{BitNetRecipe, KvStorage};
use crate::repack_cache::{
    Aligned128, BitNetModelCache, RepackedTernaryMatrix, TILE_BYTES, TILE_C, TILE_K, TILE_M,
    TILE_N, extract_i2s_trailer, is_block_repack_quant, unpack_source_block,
};
use jero_core::{ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};
use std::fmt::Write as _;
use std::sync::Arc;

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    AttentionMaskConfig, gqa_causal_attention_prefill_f32, ime::gemm_i8_ime, relu_squared_gate_f32,
    swiglu_f32,
};

const MAX_BLOCK_K_CHUNKS: usize = 16;

/// Observable counters emitted by batched multi-token prefill execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefillSubLnStats {
    /// Batched `SubLN` normalization/quantization passes before `Q/K/V` (`1` per layer vs `3 * M`).
    pub pre_qkv_subln_calls: usize,
    /// Batched `SubLN` normalization/quantization passes before `attn_output` (`1` per layer).
    pub pre_o_subln_calls: usize,
    /// Batched `SubLN` normalization/quantization passes before `ffn_gate/ffn_up` (`1` per layer).
    pub pre_ffn_subln_calls: usize,
    /// Batched `SubLN` normalization/quantization passes before `ffn_down` (`1` per layer).
    pub pre_down_subln_calls: usize,
    /// Actual `8x16` `i8` activation tile packing passes executed (`<= 1` per shared activation group).
    pub tile_pack_calls: usize,
    /// Total prompt token rows processed in the prefill batch (`M`).
    pub prompt_tokens: usize,
    /// Total `BitLinear` linear projections dispatched (`7` per transformer layer).
    pub projection_calls: usize,
    /// Prompt tokens projected through `output_norm` and the LM head (`1` for token `M - 1`).
    pub lm_head_tokens_projected: usize,
}

impl PrefillSubLnStats {
    /// Returns the total count of batched `SubLN` quantization passes across all stages.
    #[must_use]
    pub const fn total_subln_calls(&self) -> usize {
        self.pre_qkv_subln_calls
            + self.pre_o_subln_calls
            + self.pre_ffn_subln_calls
            + self.pre_down_subln_calls
    }
}

/// Reusable batched `[M, K]` `SubLN` `i8` activations paired with per-token `absmax / 127`
/// scales and lazily-packed `128-byte`-aligned `8x16` activation tiles.
///
/// Shared across sibling projections that read identical normalized inputs (`attn_q`/`attn_k`/`attn_v`
/// and `ffn_gate`/`ffn_up`), eliminating redundant `RMSNorm`, `absmax` quantization, and `8x16`
/// tile packing passes.
#[derive(Debug, Clone, Default)]
pub struct BatchedSubLnActivations {
    m: usize,
    cols: usize,
    m_tiles: usize,
    act_i8: Vec<i8>,
    act_scales: Vec<f32>,
    packed_tiles: Vec<Aligned128>,
    packed_block_k_chunks: Option<usize>,
    pack_calls: usize,
    tile_acc_scratch: Vec<f32>,
}

impl BatchedSubLnActivations {
    /// Creates an empty reusable activation batch buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            m: 0,
            cols: 0,
            m_tiles: 0,
            act_i8: Vec::new(),
            act_scales: Vec::new(),
            packed_tiles: Vec::new(),
            packed_block_k_chunks: None,
            pack_calls: 0,
            tile_acc_scratch: Vec::new(),
        }
    }

    /// Pre-allocates activation, packed tile, and accumulator buffers outside the layer loop.
    #[must_use]
    pub fn with_capacity(max_m: usize, max_cols: usize) -> Self {
        let m_tiles = max_m.div_ceil(TILE_M);
        let k_chunks = max_cols.div_ceil(TILE_K);
        Self {
            m: 0,
            cols: 0,
            m_tiles: 0,
            act_i8: vec![0_i8; max_m * max_cols],
            act_scales: vec![0.0_f32; max_m],
            packed_tiles: vec![Aligned128::zeroed(); m_tiles * k_chunks],
            packed_block_k_chunks: None,
            pack_calls: 0,
            tile_acc_scratch: vec![0.0_f32; m_tiles * TILE_C],
        }
    }

    /// Runs `SubLN` (`RMSNorm` + dynamic per-token `absmax / 127` int8 quantization) over `[m, cols]`
    /// and returns a newly allocated [`BatchedSubLnActivations`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] when dimensions are zero or slices mismatch.
    pub fn from_subln(
        input: &[f32],
        norm_weight: &[f32],
        m: usize,
        cols: usize,
        eps: f32,
    ) -> Result<Self, K3Error> {
        let mut batch = Self::with_capacity(m, cols);
        batch.quantize_subln_into(input, norm_weight, m, cols, eps)?;
        Ok(batch)
    }

    /// Normalizes and quantizes `[m, cols]` in-place, invalidating cached tile packing for reuse
    /// across subsequent sibling `BitLinear` projections.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `m == 0`, `cols == 0`, or slice lengths mismatch.
    pub fn quantize_subln_into(
        &mut self,
        input: &[f32],
        norm_weight: &[f32],
        m: usize,
        cols: usize,
        eps: f32,
    ) -> Result<(), K3Error> {
        let expected = m
            .checked_mul(cols)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "batched_subln".to_owned(),
                offset: 0,
                byte_len: input.len(),
                buffer_len: usize::MAX,
                reason: format!("m ({m}) * cols ({cols}) overflow"),
            })?;
        if m == 0 || cols == 0 || input.len() != expected || norm_weight.len() != cols {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "batched_subln".to_owned(),
                offset: 0,
                byte_len: input.len(),
                buffer_len: expected,
                reason: format!(
                    "batched SubLN dimension mismatch: m={m}, cols={cols}, input={}, norm_weight={}",
                    input.len(),
                    norm_weight.len()
                ),
            });
        }

        if self.act_i8.len() < expected {
            self.act_i8.resize(expected, 0);
        }
        if self.act_scales.len() < m {
            self.act_scales.resize(m, 0.0);
        }
        let m_tiles = m.div_ceil(TILE_M);
        let acc_needed = m_tiles * TILE_C;
        if self.tile_acc_scratch.len() < acc_needed {
            self.tile_acc_scratch.resize(acc_needed, 0.0);
        }

        for row in 0..m {
            let row_off = row * cols;
            self.act_scales[row] = BitNetRecipe::subln(
                &mut self.act_i8[row_off..row_off + cols],
                &input[row_off..row_off + cols],
                norm_weight,
                eps,
            )?;
        }

        self.m = m;
        self.cols = cols;
        self.m_tiles = m_tiles;
        self.packed_block_k_chunks = None;
        Ok(())
    }

    /// Number of token rows (`M`) in the active batch.
    #[must_use]
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Feature width (`K`) of each token row.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Number of `TILE_M = 8` activation tiles covering `m`.
    #[must_use]
    pub const fn m_tiles(&self) -> usize {
        self.m_tiles
    }

    /// Per-token dynamic `absmax / 127` quantization scales (`[M]`).
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.act_scales[..self.m]
    }

    /// Quantized `i8` matrix row for token `row` (`0 <= row < M`).
    #[must_use]
    pub fn row_i8(&self, row: usize) -> &[i8] {
        let start = row * self.cols;
        &self.act_i8[start..start + self.cols]
    }

    /// Quantization scale for token `row` (`0 <= row < M`).
    #[must_use]
    pub fn row_scale(&self, row: usize) -> f32 {
        self.act_scales[row]
    }

    /// Total `8x16` tile packing operations performed on this batch instance.
    #[must_use]
    pub const fn pack_calls(&self) -> usize {
        self.pack_calls
    }

    /// Packs `[m, cols]` quantized `i8` rows into `128-byte`-aligned `[blk][mt][ch][8][16]` tiles
    /// once per unique `block_k_chunks` layout, returning `1` when packed or `0` on reuse.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] when `cols` is not divisible by `blocks_per_row * block_k_chunks * TILE_K`.
    pub fn ensure_packed_for(
        &mut self,
        blocks_per_row: usize,
        block_k_chunks: usize,
    ) -> Result<usize, K3Error> {
        let expected_cols = blocks_per_row
            .checked_mul(block_k_chunks)
            .and_then(|v| v.checked_mul(TILE_K))
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "batched_act_tiles".to_owned(),
                offset: 0,
                byte_len: self.cols,
                buffer_len: 0,
                reason: "block tile geometry overflow".to_owned(),
            })?;
        if self.m == 0 || expected_cols != self.cols {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "batched_act_tiles".to_owned(),
                offset: 0,
                byte_len: self.cols,
                buffer_len: expected_cols,
                reason: format!(
                    "activation width {} != expected block geometry {expected_cols} (m={})",
                    self.cols, self.m
                ),
            });
        }
        if self.packed_block_k_chunks == Some(block_k_chunks) {
            return Ok(0);
        }

        let total_tiles = blocks_per_row * self.m_tiles * block_k_chunks;
        if self.packed_tiles.len() < total_tiles {
            self.packed_tiles.resize(total_tiles, Aligned128::zeroed());
        }
        let dst_i8 = Aligned128::slice_as_i8_mut(&mut self.packed_tiles[..total_tiles]);
        dst_i8.fill(0);

        let block_elems = block_k_chunks * TILE_K;
        for row in 0..self.m {
            let mt = row / TILE_M;
            let r_in_tile = row % TILE_M;
            let src_row_base = row * self.cols;
            for blk in 0..blocks_per_row {
                let blk_src_base = src_row_base + blk * block_elems;
                let blk_dst_base = blk * self.m_tiles * block_k_chunks;
                for ch in 0..block_k_chunks {
                    let tile_idx = blk_dst_base + mt * block_k_chunks + ch;
                    let dst_off = tile_idx * TILE_BYTES + r_in_tile * TILE_K;
                    let src_off = blk_src_base + ch * TILE_K;
                    dst_i8[dst_off..dst_off + TILE_K]
                        .copy_from_slice(&self.act_i8[src_off..src_off + TILE_K]);
                }
            }
        }

        self.packed_block_k_chunks = Some(block_k_chunks);
        self.pack_calls += 1;
        Ok(1)
    }
}

#[derive(Clone, Copy)]
struct TileStepShape {
    m: usize,
    m_tiles: usize,
    block_k_chunks: usize,
    tile_rows: usize,
}

#[inline]
fn accumulate_dot_tile(
    acc_mt: &mut [f32],
    dot_c: &[i32; TILE_C],
    act_scales_mt: &[f32],
    w_scales_nt: &[f32],
    tile_rows: usize,
) {
    for (i, &act_scale) in act_scales_mt.iter().enumerate() {
        let row_off = i * TILE_N;
        for j in 0..tile_rows {
            let effective_scale = w_scales_nt[j] * act_scale;
            #[expect(
                clippy::cast_precision_loss,
                reason = "block int32 accumulation fits safely in f32 mantissa"
            )]
            {
                acc_mt[row_off + j] += (dot_c[row_off + j] as f32) * effective_scale;
            }
        }
    }
}

#[cfg(not(feature = "with-kernels"))]
#[inline]
fn compute_tile_batch_dots(
    a_tile_blk: &[i8],
    b_blk: &[i8],
    k_chunks: usize,
    active_m: usize,
    dot_c: &mut [i32; TILE_C],
) {
    dot_c.fill(0);
    for ch in 0..k_chunks {
        let a_base = ch * TILE_BYTES;
        let b_base = ch * TILE_BYTES;
        for i in 0..active_m {
            let a_row = &a_tile_blk[a_base + i * TILE_K..a_base + (i + 1) * TILE_K];
            for j in 0..TILE_N {
                let b_row = &b_blk[b_base + j * TILE_K..b_base + (j + 1) * TILE_K];
                let mut acc = 0_i32;
                for k in 0..TILE_K {
                    acc += i32::from(a_row[k]) * i32::from(b_row[k]);
                }
                dot_c[i * TILE_N + j] += acc;
            }
        }
    }
}

#[inline]
fn execute_block_tile_step(
    acc: &mut [f32],
    a_blk: &[i8],
    b_blk: &[i8],
    shape: TileStepShape,
    act_scales: &[f32],
    w_scales: &[f32],
) {
    #[cfg(feature = "with-kernels")]
    {
        gemm_i8_ime(
            a_blk,
            shape.m_tiles,
            b_blk,
            0..1,
            shape.block_k_chunks,
            |mt, _, dot_c| {
                let m_base = mt * TILE_M;
                let active_m = (shape.m - m_base).min(TILE_M);
                accumulate_dot_tile(
                    &mut acc[mt * TILE_C..(mt + 1) * TILE_C],
                    dot_c,
                    &act_scales[m_base..m_base + active_m],
                    w_scales,
                    shape.tile_rows,
                );
            },
        );
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let blk_stride = shape.block_k_chunks * TILE_BYTES;
        let mut dot_c = [0_i32; TILE_C];
        for mt in 0..shape.m_tiles {
            let m_base = mt * TILE_M;
            let active_m = (shape.m - m_base).min(TILE_M);
            compute_tile_batch_dots(
                &a_blk[mt * blk_stride..(mt + 1) * blk_stride],
                b_blk,
                shape.block_k_chunks,
                active_m,
                &mut dot_c,
            );
            accumulate_dot_tile(
                &mut acc[mt * TILE_C..(mt + 1) * TILE_C],
                &dot_c,
                &act_scales[m_base..m_base + active_m],
                w_scales,
                shape.tile_rows,
            );
        }
    }
}

#[inline]
fn scatter_tile_outputs(
    out: &mut [f32],
    acc: &[f32],
    m: usize,
    m_tiles: usize,
    out_rows: usize,
    r_base: usize,
    tile_rows: usize,
) {
    for mt in 0..m_tiles {
        let m_base = mt * TILE_M;
        let active_m = (m - m_base).min(TILE_M);
        let acc_mt = &acc[mt * TILE_C..(mt + 1) * TILE_C];
        for i in 0..active_m {
            let dst_off = (m_base + i) * out_rows + r_base;
            out[dst_off..dst_off + tile_rows]
                .copy_from_slice(&acc_mt[i * TILE_N..i * TILE_N + tile_rows]);
        }
    }
}

fn validate_gemm_shape(
    out_len: usize,
    weights_rows: usize,
    weights_cols: usize,
    act: &BatchedSubLnActivations,
    tensor_name: &str,
) -> Result<usize, K3Error> {
    let m = act.m();
    if m == 0 || act.cols() != weights_cols || out_len % m != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: out_len * size_of::<f32>(),
            buffer_len: weights_rows * m * size_of::<f32>(),
            reason: format!(
                "GEMM shape mismatch: m={m}, act_cols={}, weights=[{weights_rows}, {weights_cols}], out_len={out_len}",
                act.cols()
            ),
        });
    }
    let out_rows = out_len / m;
    if out_rows > weights_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: out_len * size_of::<f32>(),
            buffer_len: weights_rows * m * size_of::<f32>(),
            reason: format!("GEMM output rows ({out_rows}) exceed weights rows ({weights_rows})"),
        });
    }
    Ok(out_rows)
}

/// Executes batched `[M, K] x [out_rows, K]^T -> [M, out_rows]` over a pre-repacked
/// [`RepackedTernaryMatrix`] using `gemm_i8_ime` (`with-kernels`) or portable `8x16` tiles,
/// returning the number of newly triggered activation packing passes (`0` or `1`).
pub(crate) fn bitlinear_gemm_repacked(
    out: &mut [f32],
    weights: &RepackedTernaryMatrix,
    act: &mut BatchedSubLnActivations,
) -> Result<usize, K3Error> {
    let out_rows = validate_gemm_shape(
        out.len(),
        weights.rows(),
        weights.cols(),
        act,
        "RepackedTernaryMatrix::bitlinear_gemm",
    )?;
    if out_rows == 0 {
        return Ok(0);
    }

    let blocks_per_row = weights.blocks_per_row();
    let block_k_chunks = weights.block_k_chunks();
    let newly_packed = act.ensure_packed_for(blocks_per_row, block_k_chunks)?;

    let m = act.m();
    let m_tiles = act.m_tiles();
    let active_n_tiles = out_rows.div_ceil(TILE_N);
    let acc_len = m_tiles * TILE_C;

    // Split disjoint field borrows so `acc` and immutable packed slices coexist without aliasing.
    let BatchedSubLnActivations {
        packed_tiles,
        act_scales,
        tile_acc_scratch,
        ..
    } = act;
    let total_tiles = blocks_per_row * m_tiles * block_k_chunks;
    let all_a_i8 = Aligned128::slice_as_i8(&packed_tiles[..total_tiles]);
    let blk_a_span = m_tiles * block_k_chunks * TILE_BYTES;
    let act_scales_active = &act_scales[..m];
    let acc = &mut tile_acc_scratch[..acc_len];

    for nt in 0..active_n_tiles {
        let r_base = nt * TILE_N;
        let tile_rows = (out_rows - r_base).min(TILE_N);
        acc.fill(0.0);
        let shape = TileStepShape {
            m,
            m_tiles,
            block_k_chunks,
            tile_rows,
        };

        for blk in 0..blocks_per_row {
            let a_blk = &all_a_i8[blk * blk_a_span..(blk + 1) * blk_a_span];
            let b_blk = weights.tile_block_i8(nt, blk);
            let w_scales = weights.tile_block_scales(nt, blk);
            execute_block_tile_step(acc, a_blk, b_blk, shape, act_scales_active, w_scales);
        }

        scatter_tile_outputs(out, acc, m, m_tiles, out_rows, r_base, tile_rows);
    }

    Ok(newly_packed)
}

struct StreamingBlockLayout {
    quant: QuantType,
    block_elems: usize,
    block_bytes: usize,
    blocks_per_row: usize,
    block_k_chunks: usize,
    i2_s_scale: f32,
}

fn validate_streaming_block_layout(
    view: &TensorView<'_>,
    out_rows: usize,
    cols: usize,
) -> Result<StreamingBlockLayout, K3Error> {
    let quant = view.quant();
    let (block_elems, block_bytes, trailer_bytes) = BitNetRecipe::quant_block_layout(quant)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: view.name().to_owned(),
            offset: view.offset(),
            byte_len: view.byte_len(),
            buffer_len: view.as_bytes().len(),
            reason: format!("unsupported block quant in streaming prefill GEMM: {quant:?}"),
        })?;
    if out_rows == 0 || cols == 0 || cols % block_elems != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: view.name().to_owned(),
            offset: view.offset(),
            byte_len: view.byte_len(),
            buffer_len: view.as_bytes().len(),
            reason: format!(
                "streaming prefill GEMM shape [{out_rows}, {cols}] incompatible with {quant:?} block {block_elems}"
            ),
        });
    }
    let total_rows = BitNetRecipe::matrix_row_col_dims(view, cols, out_rows)?;
    let blocks_per_row = cols / block_elems;
    let block_k_chunks = block_elems / TILE_K;
    debug_assert!(block_k_chunks <= MAX_BLOCK_K_CHUNKS);

    let payload_bytes = total_rows * blocks_per_row * block_bytes;
    let required_bytes = payload_bytes + trailer_bytes;
    let raw = view.as_bytes();
    if raw.len() < required_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: view.name().to_owned(),
            offset: view.offset(),
            byte_len: view.byte_len(),
            buffer_len: raw.len(),
            reason: format!(
                "weight buffer length {} is smaller than required {required_bytes} bytes",
                raw.len()
            ),
        });
    }
    let i2_s_scale = if quant == QuantType::I2_S {
        extract_i2s_trailer(raw, payload_bytes, view)?
    } else {
        0.0_f32
    };
    Ok(StreamingBlockLayout {
        quant,
        block_elems,
        block_bytes,
        blocks_per_row,
        block_k_chunks,
        i2_s_scale,
    })
}

fn unpack_streaming_tile_col(
    raw: &[u8],
    layout: &StreamingBlockLayout,
    r_base: usize,
    blk: usize,
    tile_rows: usize,
    b_i8: &mut [i8],
    w_buf: &mut [i8; 256],
) -> Result<[f32; TILE_N], K3Error> {
    b_i8.fill(0);
    let mut w_scales = [0.0_f32; TILE_N];
    for (j, w_scale_slot) in w_scales.iter_mut().enumerate().take(tile_rows) {
        let row = r_base + j;
        let boff = (row * layout.blocks_per_row + blk) * layout.block_bytes;
        *w_scale_slot = unpack_source_block(
            layout.quant,
            &raw[boff..boff + layout.block_bytes],
            layout.i2_s_scale,
            &mut w_buf[..layout.block_elems],
        )?;
        for ch in 0..layout.block_k_chunks {
            let dst_off = ch * TILE_BYTES + j * TILE_K;
            let src_off = ch * TILE_K;
            b_i8[dst_off..dst_off + TILE_K].copy_from_slice(&w_buf[src_off..src_off + TILE_K]);
        }
    }
    Ok(w_scales)
}

/// Streams a quantized block matrix (`TQ1_0`, `TQ2_0`, `I2_S`, `Q8_0`) in `8`-row x `1`-block
/// stack-resident tiles across all `M` tokens without expanding the full weight matrix in memory.
fn streaming_block_repack_gemm(
    out: &mut [f32],
    view: &TensorView<'_>,
    out_rows: usize,
    act: &mut BatchedSubLnActivations,
) -> Result<usize, K3Error> {
    let layout = validate_streaming_block_layout(view, out_rows, act.cols())?;
    if act.m() == 1 {
        // Avoid allocating and discarding temporary 8x16 B tiles for a single uncached token.
        BitNetRecipe::dispatch_bitlinear(
            &mut out[..out_rows],
            view,
            act.row_i8(0),
            act.row_scale(0),
        )?;
        return Ok(0);
    }
    let newly_packed = act.ensure_packed_for(layout.blocks_per_row, layout.block_k_chunks)?;
    let raw = view.as_bytes();
    let m = act.m();
    let m_tiles = act.m_tiles();
    let active_n_tiles = out_rows.div_ceil(TILE_N);
    let acc_len = m_tiles * TILE_C;

    let BatchedSubLnActivations {
        packed_tiles,
        act_scales,
        tile_acc_scratch,
        ..
    } = act;
    let total_tiles = layout.blocks_per_row * m_tiles * layout.block_k_chunks;
    let all_a_i8 = Aligned128::slice_as_i8(&packed_tiles[..total_tiles]);
    let blk_a_span = m_tiles * layout.block_k_chunks * TILE_BYTES;
    let act_scales_active = &act_scales[..m];
    let acc = &mut tile_acc_scratch[..acc_len];

    let mut b_tile_buf = [Aligned128::zeroed(); MAX_BLOCK_K_CHUNKS];
    let mut w_buf = [0_i8; 256];

    for nt in 0..active_n_tiles {
        let r_base = nt * TILE_N;
        let tile_rows = (out_rows - r_base).min(TILE_N);
        acc.fill(0.0);
        let shape = TileStepShape {
            m,
            m_tiles,
            block_k_chunks: layout.block_k_chunks,
            tile_rows,
        };

        for blk in 0..layout.blocks_per_row {
            let b_i8 = Aligned128::slice_as_i8_mut(&mut b_tile_buf[..layout.block_k_chunks]);
            let w_scales =
                unpack_streaming_tile_col(raw, &layout, r_base, blk, tile_rows, b_i8, &mut w_buf)?;
            let a_blk = &all_a_i8[blk * blk_a_span..(blk + 1) * blk_a_span];
            execute_block_tile_step(acc, a_blk, b_i8, shape, act_scales_active, &w_scales);
        }

        scatter_tile_outputs(out, acc, m, m_tiles, out_rows, r_base, tile_rows);
    }

    Ok(newly_packed)
}

#[derive(Clone, Copy)]
enum DummyFallbackKind {
    UniformQkvGate,
    AttnOutputResidual,
    FfnUp,
    FfnDownResidual,
}

enum ProjBatchTarget<'a> {
    Overwrite {
        out: &'a mut [f32],
        dummy: DummyFallbackKind,
    },
    ResidualAdd {
        hidden: &'a mut [f32],
        scratch: &'a mut [f32],
        dummy: DummyFallbackKind,
    },
}

fn dispatch_proj_batch(
    cache: Option<&mut BitNetModelCache>,
    weights: &dyn ModelWeights,
    name: &str,
    act: &mut BatchedSubLnActivations,
    stats: &mut PrefillSubLnStats,
    target: ProjBatchTarget<'_>,
) -> Result<(), K3Error> {
    let m = act.m();
    let cols = act.cols();
    let (out_buf, add_residual_to, dummy_kind) = match target {
        ProjBatchTarget::Overwrite { out, dummy } => (out, None, dummy),
        ProjBatchTarget::ResidualAdd {
            hidden,
            scratch,
            dummy,
        } => {
            if weights.contains_tensor(name) {
                (scratch, Some(hidden), dummy)
            } else {
                (hidden, None, dummy)
            }
        }
    };
    let out_rows = out_buf.len() / m;

    if weights.contains_tensor(name) {
        stats.projection_calls += 1;
        let view = weights.tensor(name)?;
        let cached: Option<Arc<RepackedTernaryMatrix>> = match cache {
            Some(c) => c.cached_projection_batched(&view, name, out_rows, cols)?,
            None => None,
        };
        if let Some(matrix) = cached {
            let newly_packed = bitlinear_gemm_repacked(out_buf, &matrix, act)?;
            stats.tile_pack_calls += newly_packed;
        } else if is_block_repack_quant(view.quant()) {
            let newly_packed = streaming_block_repack_gemm(out_buf, &view, out_rows, act)?;
            stats.tile_pack_calls += newly_packed;
        } else {
            for row_m in 0..m {
                let dst = &mut out_buf[row_m * out_rows..(row_m + 1) * out_rows];
                BitNetRecipe::dispatch_bitlinear(
                    dst,
                    &view,
                    act.row_i8(row_m),
                    act.row_scale(row_m),
                )?;
            }
        }
        if let Some(hidden) = add_residual_to {
            for (h, &p) in hidden.iter_mut().zip(out_buf.iter()) {
                *h += p;
            }
        }
        return Ok(());
    }

    if weights.tensor_count() == 0 {
        for row_m in 0..m {
            let act_row = act.row_i8(row_m);
            let act_scale = act.row_scale(row_m);
            let dst = &mut out_buf[row_m * out_rows..(row_m + 1) * out_rows];
            for (i, slot) in dst.iter_mut().enumerate() {
                let scaled = f32::from(act_row[i % cols]) * act_scale;
                match dummy_kind {
                    DummyFallbackKind::UniformQkvGate => *slot = scaled * 0.1 + 0.5,
                    DummyFallbackKind::AttnOutputResidual => *slot += scaled * 0.05,
                    DummyFallbackKind::FfnUp => *slot = scaled * 0.05 + 0.25,
                    DummyFallbackKind::FfnDownResidual => *slot += scaled * 0.01,
                }
            }
        }
        return Ok(());
    }

    Err(K3Error::TensorBoundsError {
        tensor_name: name.to_owned(),
        offset: 0,
        byte_len: 0,
        buffer_len: 0,
        reason: format!("tensor '{name}' not found in weights"),
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "GQA causal prefill takes geometry, KV slices, position range, and scratch"
)]
fn dispatch_gqa_prefill_attention(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    start_pos: usize,
    scale: f32,
    sliding_window: Option<usize>,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        let config =
            AttentionMaskConfig::causal(start_pos, scale).with_sliding_window(sliding_window);
        gqa_causal_attention_prefill_f32(
            out,
            q,
            k_cache,
            v_cache,
            m,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq_len,
            config,
            scores_scratch,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        dispatch_gqa_prefill_scalar(
            out,
            q,
            k_cache,
            v_cache,
            m,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq_len,
            start_pos,
            scale,
            sliding_window,
            scores_scratch,
        )
    }
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "portable scalar causal GQA prefill fallback mirrors with-kernels semantics"
)]
fn dispatch_gqa_prefill_scalar(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    start_pos: usize,
    scale: f32,
    sliding_window: Option<usize>,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    if n_heads_q == 0 || n_heads_kv == 0 || head_dim == 0 || m == 0 || total_seq_len == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: "GQA prefill dimensions must be non-zero".to_owned(),
        });
    }
    if n_heads_q % n_heads_kv != 0 || !scale.is_finite() || sliding_window == Some(0) {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: "invalid GQA head grouping, scale, or sliding window".to_owned(),
        });
    }
    let query_end = start_pos.saturating_add(m);
    if query_end > total_seq_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_start_pos".to_owned(),
            offset: start_pos,
            byte_len: m,
            buffer_len: total_seq_len,
            reason: format!("start_pos + m ({query_end}) exceeds total_seq_len ({total_seq_len})"),
        });
    }
    let kv_width = n_heads_kv * head_dim;
    let q_width = n_heads_q * head_dim;
    let req_kv = total_seq_len * kv_width;
    if q.len() != m * q_width
        || out.len() != m * q_width
        || k_cache.len() < req_kv
        || v_cache.len() < req_kv
        || scores_scratch.len() < query_end
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_prefill".to_owned(),
            offset: 0,
            byte_len: req_kv * size_of::<f32>(),
            buffer_len: k_cache.len().min(v_cache.len()) * size_of::<f32>(),
            reason: "GQA prefill buffer length mismatch".to_owned(),
        });
    }

    let group_size = n_heads_q / n_heads_kv;
    for kv_h in 0..n_heads_kv {
        let kv_head_offset = kv_h * head_dim;
        for m_idx in 0..m {
            let query_pos = start_pos + m_idx;
            let valid_end = (query_pos + 1).min(total_seq_len);
            let valid_start =
                sliding_window.map_or(0, |w| query_pos.saturating_add(1).saturating_sub(w));
            let active_len = valid_end.saturating_sub(valid_start);
            let score_buf = &mut scores_scratch[..active_len];

            for g in 0..group_size {
                let q_h = kv_h * group_size + g;
                let q_slice = &q[(m_idx * n_heads_q + q_h) * head_dim..][..head_dim];

                let mut max_score = f32::NEG_INFINITY;
                for (slot, s) in score_buf.iter_mut().zip(valid_start..valid_end) {
                    let kv_off = s * kv_width + kv_head_offset;
                    let k_slice = &k_cache[kv_off..kv_off + head_dim];
                    let score: f32 = q_slice
                        .iter()
                        .zip(k_slice)
                        .map(|(&qv, &kv)| qv * kv)
                        .sum::<f32>()
                        * scale;
                    *slot = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                if active_len > 0 {
                    let mut sum = 0.0_f32;
                    for slot in score_buf.iter_mut() {
                        let w = if max_score.is_finite() {
                            (*slot - max_score).exp()
                        } else {
                            1.0_f32
                        };
                        *slot = w;
                        sum += w;
                    }
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "active sequence window fits cleanly in f32"
                    )]
                    let inv_sum = if sum.is_finite() && sum > 0.0 {
                        1.0_f32 / sum
                    } else {
                        1.0_f32 / (active_len as f32)
                    };
                    for slot in score_buf.iter_mut() {
                        *slot *= inv_sum;
                    }
                }

                let out_slice = &mut out[(m_idx * n_heads_q + q_h) * head_dim..][..head_dim];
                out_slice.fill(0.0);
                for (s, &weight) in (valid_start..valid_end).zip(score_buf.iter()) {
                    if weight == 0.0 {
                        continue;
                    }
                    let kv_off = s * kv_width + kv_head_offset;
                    let v_slice = &v_cache[kv_off..kv_off + head_dim];
                    for (dst, &val) in out_slice.iter_mut().zip(v_slice) {
                        *dst += weight * val;
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct LayerKvGeom {
    layer: usize,
    num_layers: usize,
    start_pos: usize,
    m: usize,
    n_q: usize,
    n_kv: usize,
    head_d_even: usize,
    attn_scale: f32,
    weights_empty: bool,
}

struct LayerAttnBuffers<'a> {
    k_batch: &'a [f32],
    v_batch: &'a [f32],
    q_batch: &'a [f32],
    attn_out: &'a mut [f32],
    attn_scores: &'a mut [f32],
}

fn update_kv_view_and_attention(
    geom: LayerKvGeom,
    bufs: &mut LayerAttnBuffers<'_>,
    kv_view: &mut jero_core::KvCacheView<'_>,
    name_buf: &mut String,
) -> Result<(), K3Error> {
    let total_kv = geom.n_kv * geom.head_d_even;
    let total_seq_len = geom.start_pos + geom.m;
    if kv_view.kv_dim() != total_kv {
        name_buf.clear();
        let _ = write!(name_buf, "kv_cache.layer.{}", geom.layer);
        return Err(K3Error::TensorBoundsError {
            tensor_name: name_buf.clone(),
            offset: 0,
            byte_len: total_kv * size_of::<f32>(),
            buffer_len: kv_view.kv_dim() * size_of::<f32>(),
            reason: format!(
                "KvGeometry kv_dim {} does not match total_kv {total_kv}",
                kv_view.kv_dim()
            ),
        });
    }
    let _ = kv_view.valid_token_range(total_seq_len)?;
    for row_m in 0..geom.m {
        let pos = geom.start_pos + row_m;
        let (k_slot, v_slot) = kv_view.write_pos_range(geom.layer, pos)?;
        let src_off = row_m * total_kv;
        k_slot.copy_from_slice(&bufs.k_batch[src_off..src_off + total_kv]);
        v_slot.copy_from_slice(&bufs.v_batch[src_off..src_off + total_kv]);
    }
    let sliding_window = kv_view.geometry().sliding_window;
    let (k_plane, v_plane) = kv_view.layer_kv(geom.layer)?;
    let req_elems = total_seq_len * total_kv;
    dispatch_gqa_prefill_attention(
        bufs.attn_out,
        bufs.q_batch,
        &k_plane[..req_elems],
        &v_plane[..req_elems],
        geom.m,
        geom.n_q,
        geom.n_kv,
        geom.head_d_even,
        total_seq_len,
        geom.start_pos,
        geom.attn_scale,
        sliding_window,
        bufs.attn_scores,
    )
}

fn update_kv_raw_and_attention(
    geom: LayerKvGeom,
    bufs: &mut LayerAttnBuffers<'_>,
    kv_cache: &mut [f32],
    name_buf: &mut String,
) -> Result<(), K3Error> {
    let total_kv = geom.n_kv * geom.head_d_even;
    let total_q = geom.n_q * geom.head_d_even;
    let total_seq_len = geom.start_pos + geom.m;
    let layer_stride = kv_cache.len().checked_div(geom.num_layers).unwrap_or(0);
    let half_stride = layer_stride / 2;
    let req_hist_elems = total_seq_len.saturating_mul(total_kv);

    if half_stride >= req_hist_elems && req_hist_elems > 0 {
        let layer_start = geom.layer * layer_stride;
        let layer_buf = &mut kv_cache[layer_start..layer_start + layer_stride];
        let write_off = geom.start_pos * total_kv;
        let batch_span = geom.m * total_kv;
        layer_buf[write_off..write_off + batch_span].copy_from_slice(&bufs.k_batch[..batch_span]);
        layer_buf[half_stride + write_off..half_stride + write_off + batch_span]
            .copy_from_slice(&bufs.v_batch[..batch_span]);
        let k_hist = &layer_buf[..req_hist_elems];
        let v_hist = &layer_buf[half_stride..half_stride + req_hist_elems];
        return dispatch_gqa_prefill_attention(
            bufs.attn_out,
            bufs.q_batch,
            k_hist,
            v_hist,
            geom.m,
            geom.n_q,
            geom.n_kv,
            geom.head_d_even,
            total_seq_len,
            geom.start_pos,
            geom.attn_scale,
            None,
            bufs.attn_scores,
        );
    }

    if geom.weights_empty {
        for row_m in 0..geom.m {
            let pos = geom.start_pos + row_m;
            let k_row = &bufs.k_batch[row_m * total_kv..(row_m + 1) * total_kv];
            let v_row = &bufs.v_batch[row_m * total_kv..(row_m + 1) * total_kv];
            if half_stride > 0 {
                let layer_start = geom.layer * layer_stride;
                let layer_end = (layer_start + layer_stride).min(kv_cache.len());
                let layer_buf = &mut kv_cache[layer_start..layer_end];
                let k_copy = total_kv.min(half_stride);
                let pos_off = (pos * k_copy) % half_stride;
                let k_end = (pos_off + k_copy).min(half_stride);
                let actual_copy = k_end - pos_off;
                layer_buf[pos_off..pos_off + actual_copy].copy_from_slice(&k_row[..actual_copy]);
                layer_buf[half_stride + pos_off..half_stride + pos_off + actual_copy]
                    .copy_from_slice(&v_row[..actual_copy]);
            }
            dispatch_gqa_prefill_attention(
                &mut bufs.attn_out[row_m * total_q..(row_m + 1) * total_q],
                &bufs.q_batch[row_m * total_q..(row_m + 1) * total_q],
                k_row,
                v_row,
                1,
                geom.n_q,
                geom.n_kv,
                geom.head_d_even,
                1,
                0,
                geom.attn_scale,
                None,
                &mut bufs.attn_scores[..1],
            )?;
        }
        return Ok(());
    }

    name_buf.clear();
    let _ = write!(name_buf, "kv_cache");
    Err(K3Error::TensorBoundsError {
        tensor_name: name_buf.clone(),
        offset: 0,
        byte_len: req_hist_elems * 2 * geom.num_layers * size_of::<f32>(),
        buffer_len: size_of_val(kv_cache),
        reason: format!(
            "kv_cache half-layer stride {half_stride} is smaller than required {req_hist_elems} elements for end position {total_seq_len}"
        ),
    })
}

fn update_kv_and_causal_attention(
    geom: LayerKvGeom,
    bufs: &mut LayerAttnBuffers<'_>,
    kv_storage: &mut KvStorage<'_>,
    name_buf: &mut String,
) -> Result<(), K3Error> {
    bufs.attn_out.fill(0.0);
    match kv_storage {
        KvStorage::View(kv_view) => update_kv_view_and_attention(geom, bufs, kv_view, name_buf),
        KvStorage::Raw(kv_cache) => update_kv_raw_and_attention(geom, bufs, kv_cache, name_buf),
    }
}

#[cfg_attr(
    not(feature = "with-kernels"),
    expect(
        clippy::unnecessary_wraps,
        reason = "with-kernels activation helpers return Result"
    )
)]
fn apply_ffn_activation_batch(
    use_relu2: bool,
    ffn_inter: &mut [f32],
    gate: &[f32],
    up: &[f32],
) -> Result<(), K3Error> {
    if use_relu2 {
        #[cfg(feature = "with-kernels")]
        {
            relu_squared_gate_f32(ffn_inter, gate, up)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            for ((dst, &g), &u) in ffn_inter.iter_mut().zip(gate).zip(up) {
                let gp = g.max(0.0);
                *dst = gp * gp * u;
            }
            Ok(())
        }
    } else {
        #[cfg(feature = "with-kernels")]
        {
            swiglu_f32(ffn_inter, gate, up)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            for ((dst, &g), &u) in ffn_inter.iter_mut().zip(gate).zip(up) {
                let sig = if g <= -20.0 {
                    0.0
                } else if g >= 20.0 {
                    1.0
                } else {
                    1.0 / (1.0 + (-g).exp())
                };
                *dst = (g * sig) * u;
            }
            Ok(())
        }
    }
}

impl BitNetRecipe {
    /// Executes [`jero_core::ArchRecipe::prefill_step`] and returns batched `SubLN` and tile-packing telemetry.
    ///
    /// When `cache` is `None` and the process-global cache is active, the global cache is used.
    ///
    /// # Errors
    ///
    /// Propagates [`K3Error`] on empty prompt, KV geometry mismatch, or invalid tensors.
    pub fn prefill_step_with_stats(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        cache: Option<&mut BitNetModelCache>,
    ) -> Result<PrefillSubLnStats, K3Error> {
        match cache {
            Some(c) => self.prefill_step_batched_stats(ctx, weights, Some(c)),
            None if BitNetModelCache::is_global_active() => {
                BitNetModelCache::with_global(|global| {
                    self.prefill_step_batched_stats(ctx, weights, Some(global))
                })
            }
            None => self.prefill_step_batched_stats(ctx, weights, None),
        }
    }

    pub(crate) fn prefill_step_batched_stats(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        mut cache: Option<&mut BitNetModelCache>,
    ) -> Result<PrefillSubLnStats, K3Error> {
        let prompt_tokens = ctx.prompt_tokens().to_vec();
        if prompt_tokens.is_empty() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: 0,
                max_context_tokens: ctx.n_ctx(),
            });
        }
        if let Some(c) = cache.as_deref_mut() {
            c.ensure_model_anchor(weights)?;
        }

        let prompt_len = prompt_tokens.len();
        let start_pos = ctx.position();
        let (last_hidden, mut stats) = {
            let mut kv_storage = match ctx.kv_view_mut() {
                Some(view) => KvStorage::View(view),
                None => KvStorage::Raw(ctx.kv_cache_mut()),
            };
            self.run_batched_prefill_with_cache(
                &prompt_tokens,
                start_pos,
                &mut kv_storage,
                weights,
                cache.as_deref_mut(),
            )?
        };

        ctx.advance_position(prompt_len)?;
        match cache {
            Some(c) => c.project_lm_head(weights, self, ctx.logits_mut(), &last_hidden, None)?,
            None => self.project_lm_head(ctx.logits_mut(), &last_hidden, None, weights)?,
        }
        stats.lm_head_tokens_projected = usize::from(!ctx.logits().is_empty());
        Ok(stats)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "batched multi-token transformer loop pre-allocates scratch once before layers"
    )]
    pub(crate) fn run_batched_prefill_with_cache(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kv_storage: &mut KvStorage<'_>,
        weights: &dyn ModelWeights,
        mut cache: Option<&mut BitNetModelCache>,
    ) -> Result<(Vec<f32>, PrefillSubLnStats), K3Error> {
        let m = tokens.len();
        let hidden_dim = self.hidden_dim.max(16);
        let ffn_dim = self.ffn_dim.max(16);
        let head_d = self.head_dim.max(2);
        let head_d_even = if head_d % 2 == 0 { head_d } else { head_d - 1 };
        let n_q = self.n_heads_q.max(1);
        let n_kv = self.n_heads_kv.max(1);
        let total_q = n_q * head_d_even;
        let total_kv = n_kv * head_d_even;
        let total_seq_len = start_pos + m;
        let eps = self.rms_eps.max(1e-6);
        #[expect(clippy::cast_precision_loss, reason = "head dimension fits in f32")]
        let attn_scale = 1.0_f32 / (head_d_even as f32).sqrt();
        let weights_empty = weights.tensor_count() == 0;

        let mut hidden = vec![0.0_f32; m * hidden_dim];
        for (row_m, &token_id) in tokens.iter().enumerate() {
            Self::embed_token_opt(
                cache.as_deref_mut(),
                weights,
                token_id,
                &mut hidden[row_m * hidden_dim..(row_m + 1) * hidden_dim],
            )?;
        }

        // Pre-allocate all layer scratch buffers once outside `0..self.num_layers`.
        let mut norm_w_hidden = vec![1.0_f32; hidden_dim];
        let mut norm_w_total_q = vec![1.0_f32; total_q];
        let mut norm_w_ffn = vec![1.0_f32; ffn_dim];

        let mut act_hidden = BatchedSubLnActivations::with_capacity(m, hidden_dim);
        let mut act_total_q = BatchedSubLnActivations::with_capacity(m, total_q);
        let mut act_ffn = BatchedSubLnActivations::with_capacity(m, ffn_dim);

        let mut q_batch = vec![0.0_f32; m * total_q];
        let mut k_batch = vec![0.0_f32; m * total_kv];
        let mut v_batch = vec![0.0_f32; m * total_kv];

        let mut attn_out = vec![0.0_f32; m * total_q];
        let mut attn_scores = vec![0.0_f32; total_seq_len.max(1)];
        let mut o_proj = vec![0.0_f32; m * hidden_dim];

        let mut gate = vec![0.0_f32; m * ffn_dim];
        let mut up = vec![0.0_f32; m * ffn_dim];
        let mut ffn_inter = vec![0.0_f32; m * ffn_dim];
        let mut d_proj = vec![0.0_f32; m * hidden_dim];

        let mut name_buf = String::with_capacity(64);
        let mut stats = PrefillSubLnStats {
            prompt_tokens: m,
            ..PrefillSubLnStats::default()
        };

        for layer in 0..self.num_layers {
            // 1. Pre-Attention Norm & Deduplicated SubLN (3x -> 1x) + Batched Q, K, V GEMM
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_norm.weight");
            Self::load_norm_opt(cache.as_deref_mut(), weights, &name_buf, &mut norm_w_hidden)?;

            act_hidden.quantize_subln_into(&hidden, &norm_w_hidden, m, hidden_dim, eps)?;
            stats.pre_qkv_subln_calls += 1;

            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_q.weight");
            dispatch_proj_batch(
                cache.as_deref_mut(),
                weights,
                &name_buf,
                &mut act_hidden,
                &mut stats,
                ProjBatchTarget::Overwrite {
                    out: &mut q_batch,
                    dummy: DummyFallbackKind::UniformQkvGate,
                },
            )?;

            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_k.weight");
            dispatch_proj_batch(
                cache.as_deref_mut(),
                weights,
                &name_buf,
                &mut act_hidden,
                &mut stats,
                ProjBatchTarget::Overwrite {
                    out: &mut k_batch,
                    dummy: DummyFallbackKind::UniformQkvGate,
                },
            )?;

            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_v.weight");
            dispatch_proj_batch(
                cache.as_deref_mut(),
                weights,
                &name_buf,
                &mut act_hidden,
                &mut stats,
                ProjBatchTarget::Overwrite {
                    out: &mut v_batch,
                    dummy: DummyFallbackKind::UniformQkvGate,
                },
            )?;

            // 2. Batched NeoX RoPE over [start_pos..start_pos + m]
            for row_m in 0..m {
                let pos = start_pos + row_m;
                self.apply_rope_heads(
                    &mut q_batch[row_m * total_q..(row_m + 1) * total_q],
                    head_d_even,
                    pos,
                    total_seq_len,
                )?;
                self.apply_rope_heads(
                    &mut k_batch[row_m * total_kv..(row_m + 1) * total_kv],
                    head_d_even,
                    pos,
                    total_seq_len,
                )?;
            }

            // 3. Chunked KV Cache Write & Causal Multi-Token GQA Prefill
            let kv_geom = LayerKvGeom {
                layer,
                num_layers: self.num_layers,
                start_pos,
                m,
                n_q,
                n_kv,
                head_d_even,
                attn_scale,
                weights_empty,
            };
            let mut kv_bufs = LayerAttnBuffers {
                k_batch: &k_batch,
                v_batch: &v_batch,
                q_batch: &q_batch,
                attn_out: &mut attn_out,
                attn_scores: &mut attn_scores,
            };
            update_kv_and_causal_attention(kv_geom, &mut kv_bufs, kv_storage, &mut name_buf)?;

            // 4. SubLN before Output Projection + Residual
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_sub_norm.weight");
            Self::load_norm_opt(
                cache.as_deref_mut(),
                weights,
                &name_buf,
                &mut norm_w_total_q,
            )?;
            act_total_q.quantize_subln_into(&attn_out, &norm_w_total_q, m, total_q, eps)?;
            stats.pre_o_subln_calls += 1;

            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_output.weight");
            dispatch_proj_batch(
                cache.as_deref_mut(),
                weights,
                &name_buf,
                &mut act_total_q,
                &mut stats,
                ProjBatchTarget::ResidualAdd {
                    hidden: &mut hidden,
                    scratch: &mut o_proj,
                    dummy: DummyFallbackKind::AttnOutputResidual,
                },
            )?;

            // 5. Pre-FFN Norm & Deduplicated SubLN (2x -> 1x) + Batched Gate & Up GEMM
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.ffn_norm.weight");
            Self::load_norm_opt(cache.as_deref_mut(), weights, &name_buf, &mut norm_w_hidden)?;

            act_hidden.quantize_subln_into(&hidden, &norm_w_hidden, m, hidden_dim, eps)?;
            stats.pre_ffn_subln_calls += 1;

            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.ffn_gate.weight");
            dispatch_proj_batch(
                cache.as_deref_mut(),
                weights,
                &name_buf,
                &mut act_hidden,
                &mut stats,
                ProjBatchTarget::Overwrite {
                    out: &mut gate,
                    dummy: DummyFallbackKind::UniformQkvGate,
                },
            )?;

            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.ffn_up.weight");
            dispatch_proj_batch(
                cache.as_deref_mut(),
                weights,
                &name_buf,
                &mut act_hidden,
                &mut stats,
                ProjBatchTarget::Overwrite {
                    out: &mut up,
                    dummy: DummyFallbackKind::FfnUp,
                },
            )?;

            // 6. Batched FFN Gated Activation (ReLU^2 vs SwiGLU)
            apply_ffn_activation_batch(self.use_relu2, &mut ffn_inter, &gate, &up)?;

            // 7. SubLN before Down Projection + Residual
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.ffn_sub_norm.weight");
            Self::load_norm_opt(cache.as_deref_mut(), weights, &name_buf, &mut norm_w_ffn)?;
            act_ffn.quantize_subln_into(&ffn_inter, &norm_w_ffn, m, ffn_dim, eps)?;
            stats.pre_down_subln_calls += 1;

            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.ffn_down.weight");
            dispatch_proj_batch(
                cache.as_deref_mut(),
                weights,
                &name_buf,
                &mut act_ffn,
                &mut stats,
                ProjBatchTarget::ResidualAdd {
                    hidden: &mut hidden,
                    scratch: &mut d_proj,
                    dummy: DummyFallbackKind::FfnDownResidual,
                },
            )?;
        }

        // 8. Output Normalization ONLY for the final token row `M - 1`.
        let last_off = (m - 1) * hidden_dim;
        let mut last_hidden = hidden[last_off..last_off + hidden_dim].to_vec();
        if weights.contains_tensor("output_norm.weight") {
            Self::load_norm_opt(cache, weights, "output_norm.weight", &mut norm_w_hidden)?;
            let sum_sq: f32 = last_hidden.iter().map(|&x| x * x).sum();
            if sum_sq.is_finite() {
                #[expect(clippy::cast_precision_loss, reason = "dimension fits in f32")]
                let inv_rms = 1.0_f32 / (sum_sq / (hidden_dim as f32) + eps).sqrt();
                if inv_rms.is_finite() {
                    for (dst, &nw) in last_hidden.iter_mut().zip(&norm_w_hidden) {
                        *dst *= inv_rms * nw;
                    }
                } else {
                    last_hidden.fill(0.0);
                }
            } else {
                last_hidden.fill(0.0);
            }
        } else if weights.tensor_count() > 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "output_norm.weight".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "tensor 'output_norm.weight' not found in weights".to_owned(),
            });
        }

        Ok((last_hidden, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repack_cache::RepackedTernaryMatrix;
    use jero_core::{ExecutionContext, KvGeometry, TensorView};

    fn encode_q8_0_row(values: &[i8], scale: f32) -> Vec<u8> {
        assert_eq!(values.len() % 32, 0);
        let f16_bits = half_from_f32(scale);
        let mut raw = Vec::with_capacity((values.len() / 32) * 34);
        for chunk in values.chunks_exact(32) {
            raw.extend_from_slice(&f16_bits.to_le_bytes());
            for &v in chunk {
                #[expect(clippy::cast_sign_loss, reason = "Q8_0 encodes signed i8 bytes")]
                raw.push(v as u8);
            }
        }
        raw
    }

    fn half_from_f32(v: f32) -> u16 {
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

    #[test]
    fn test_prefill_gemm_non_aligned_m_tiles_matches_gemv_exact() {
        let rows = if cfg!(miri) { 9_usize } else { 12_usize }; // covers 1 full N-tile + trailing partial N-tile
        let cols = if cfg!(miri) { 32_usize } else { 64_usize }; // 1 or 2 Q8_0 blocks per row
        let mut raw = Vec::new();
        for r in 0..rows {
            let vals: Vec<i8> = (0..cols)
                .map(|c| {
                    let rem = i8::try_from((r * 7 + c * 5) % 15).unwrap_or(0);
                    let v = rem - 7;
                    if v == 0 { 1 } else { v }
                })
                .collect();
            raw.extend_from_slice(&encode_q8_0_row(&vals, 0.25));
        }
        let shape = [cols, rows];
        let view = TensorView::from_bytes("w.q8", QuantType::Q8_0, &shape, 0, &raw)
            .expect("valid q8 tensor");
        let repacked = RepackedTernaryMatrix::from_tensor(&view, rows, cols).expect("repack");

        let m = 9_usize; // spans M-tile 0 (rows 0..8) and M-tile 1 (row 8)
        let norm_w = vec![1.0_f32; cols];
        let input: Vec<f32> = (0..m * cols)
            .map(|idx| (f32::from(u8::try_from(idx % 23).unwrap_or(0)) - 11.0) * 0.125)
            .collect();

        let mut act =
            BatchedSubLnActivations::from_subln(&input, &norm_w, m, cols, 1e-5).expect("subln");
        let mut out_batch = vec![0.0_f32; m * rows];
        repacked
            .bitlinear_gemm(&mut out_batch, &mut act)
            .expect("batched gemm");
        assert_eq!(act.pack_calls(), 1);

        // Reuse same packed batch for a second call; pack_calls must remain 1.
        let mut out_batch_2 = vec![0.0_f32; m * rows];
        repacked
            .bitlinear_gemm(&mut out_batch_2, &mut act)
            .expect("reused batch gemm");
        assert_eq!(act.pack_calls(), 1);

        let mut ref_row = vec![0.0_f32; rows];
        for row_m in 0..m {
            repacked
                .bitlinear_gemv(&mut ref_row, act.row_i8(row_m), act.row_scale(row_m))
                .expect("gemv ref");
            for col in 0..rows {
                assert_eq!(
                    out_batch[row_m * rows + col].to_bits(),
                    ref_row[col].to_bits(),
                    "bit mismatch at m={row_m}, row={col}"
                );
                assert_eq!(
                    out_batch_2[row_m * rows + col].to_bits(),
                    ref_row[col].to_bits()
                );
            }
        }
    }

    #[cfg(feature = "with-kernels")]
    #[test]
    fn test_prefill_gemm_increments_ime_vmadot_count_proportionally() {
        let rows = 16_usize;
        let cols = 64_usize;
        let mut raw = Vec::new();
        for _ in 0..rows {
            raw.extend_from_slice(&encode_q8_0_row(&[1_i8; 64], 0.5));
        }
        let shape = [cols, rows];
        let view = TensorView::from_bytes("w.ime", QuantType::Q8_0, &shape, 0, &raw)
            .expect("valid q8 tensor");
        let repacked = RepackedTernaryMatrix::from_tensor(&view, rows, cols).expect("repack");

        let m = 9_usize; // ceil(9/8) = 2 m_tiles, ceil(16/8) = 2 n_tiles, 64/16 = 4 k_chunks -> 16 calls
        let norm_w = vec![1.0_f32; cols];
        let input = vec![0.5_f32; m * cols];
        let mut act =
            BatchedSubLnActivations::from_subln(&input, &norm_w, m, cols, 1e-5).expect("subln");
        let mut out = vec![0.0_f32; m * rows];

        let before = jero_kernels_rvv::ime::ime_vmadot_count();
        repacked
            .bitlinear_gemm(&mut out, &mut act)
            .expect("ime batch");
        let delta = jero_kernels_rvv::ime::ime_vmadot_count() - before;
        let expected = (m.div_ceil(TILE_M) * rows.div_ceil(TILE_N) * (cols / TILE_K)) as u64;
        // Global IME counter may also advance from concurrent `--lib` tests; exact equality is
        // verified serially under a mutex in integration tests.
        assert!(
            delta >= expected,
            "ime_vmadot_count delta {delta} must include all {expected} tiles"
        );
    }

    #[test]
    fn test_prefill_step_with_stats_deduplicates_qkv_and_ffn() {
        struct EmptyWeights;
        impl ModelWeights for EmptyWeights {
            fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
                Err(K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: "empty".to_owned(),
                })
            }
            fn tensor_count(&self) -> usize {
                0
            }
            fn total_weight_bytes(&self) -> usize {
                0
            }
        }

        let mut recipe = BitNetRecipe::new();
        recipe.num_layers = 2;
        recipe.hidden_dim = 32;
        recipe.ffn_dim = 64;
        recipe.n_heads_q = 4;
        recipe.n_heads_kv = 2;
        recipe.head_dim = 8;
        recipe.vocab_size = 16;

        let geom = KvGeometry::new(2, 2, 8, 8, None).expect("geom");
        let mut kv = vec![0.0_f32; geom.required_f32_len().expect("kv")];
        let mut logits = [0.0_f32; 16];
        let mut ctx = ExecutionContext::new(&[1, 2, 3], 8, &mut logits, &mut kv)
            .expect("ctx")
            .with_kv_geometry(geom)
            .expect("kv view");

        let stats = recipe
            .prefill_step_with_stats(&mut ctx, &EmptyWeights, None)
            .expect("batched empty weights prefill");
        assert_eq!(stats.prompt_tokens, 3);
        assert_eq!(stats.pre_qkv_subln_calls, 2);
        assert_eq!(stats.pre_ffn_subln_calls, 2);
        assert_eq!(stats.total_subln_calls(), 8);
        assert_eq!(stats.lm_head_tokens_projected, 1);
        assert_eq!(ctx.position(), 3);
    }
}
