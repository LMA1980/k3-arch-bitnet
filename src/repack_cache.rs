//! Load-time `128-byte` `8x16` `i8` tile repacking, cached `SubLN` norm vectors,
//! compact tied `token_embd` / LM-head storage, and Llama-3 `128k` stop-token contract checks
//! for [`BitNetRecipe`].

use crate::recipe::BitNetRecipe;
use jero_core::{
    ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error, ModelWeights, QuantType, TensorView,
};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

#[cfg(feature = "with-kernels")]
pub use jero_kernels_rvv::Aligned128;

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    QuantizedActQ8, RepackedWeight,
    ime::{TILE_C, TILE_K, TILE_N, gemm_i8_ime, unpack_tq1_0_block_i8, unpack_tq2_0_block_i8},
    quant::{read_i2_s_trailer_scale, unpack_i2_s_block_i8},
};

#[cfg(not(feature = "with-kernels"))]
/// Logical N rows per `8x16` weight tile.
pub const TILE_N: usize = 8;
#[cfg(not(feature = "with-kernels"))]
/// Logical K depth per `8x16` tile chunk.
pub const TILE_K: usize = 16;
#[cfg(not(feature = "with-kernels"))]
/// Accumulator tile elements (`8 x 8 = 64` `i32` slots).
pub const TILE_C: usize = 8 * TILE_N;

/// Byte alignment enforced on all repacked weight and compact embedding payloads.
pub const WEIGHT_TILE_ALIGNMENT_BYTES: usize = 128;

/// Byte size of one `8 x 16` `i8` weight tile.
pub const TILE_BYTES: usize = 128;

/// Default element limit above which `F16`/`BF16` tables stay compact `16-bit` (`0` = always compact).
pub const DEFAULT_DENSE_F32_MAX_ELEMENTS: usize = 0;

/// Default upper bound on cached repacked weights (`24 GiB`, inside a `28 GiB` memlock budget).
pub const DEFAULT_MAX_REPACK_CACHE_BYTES: usize = 24 * 1024 * 1024 * 1024;

/// Canonical Llama-3 / `BitNet-b1.58-2B-4T` vocabulary size (`128,256`).
pub const LLAMA3_BITNET_VOCAB_SIZE: usize = 128_256;

/// Canonical Llama-3 `<|eot_id|>` token identifier (`128009`).
pub const LLAMA3_EOT_TOKEN_ID: u32 = 128_009;

/// Canonical Llama-3 `<|end_of_text|>` token identifier (`128001`).
pub const LLAMA3_END_OF_TEXT_TOKEN_ID: u32 = 128_001;

#[cfg(not(feature = "with-kernels"))]
/// `128-byte`-aligned tile / row block when `with-kernels` is disabled.
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aligned128(pub [u8; 128]);

#[cfg(not(feature = "with-kernels"))]
impl Aligned128 {
    /// Zero-initialized `128-byte` aligned block.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self([0_u8; 128])
    }

    /// Reinterprets `Aligned128` blocks as signed `i8` tile bytes.
    #[must_use]
    pub fn slice_as_i8(buf: &[Self]) -> &[i8] {
        // SAFETY: `i8` has alignment 1, size 1, and every bit pattern is valid.
        unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<i8>(), buf.len() * 128) }
    }

    /// Mutable signed `i8` view over `Aligned128` blocks.
    #[must_use]
    pub fn slice_as_i8_mut(buf: &mut [Self]) -> &mut [i8] {
        // SAFETY: `i8` has alignment 1, size 1, and every bit pattern is valid.
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<i8>(), buf.len() * 128) }
    }

    /// Interprets `Aligned128` blocks as native-endian `u16` half elements.
    #[must_use]
    pub fn slice_as_u16(buf: &[Self]) -> &[u16] {
        // SAFETY: `Aligned128` is 128-byte aligned (>= align_of::<u16>()) and size 128 = 64 * 2.
        unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<u16>(), buf.len() * 64) }
    }

    /// Mutable native-endian `u16` view over `Aligned128` blocks.
    #[must_use]
    pub fn slice_as_u16_mut(buf: &mut [Self]) -> &mut [u16] {
        // SAFETY: `Aligned128` is 128-byte aligned (>= align_of::<u16>()) and size 128 = 64 * 2.
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u16>(), buf.len() * 64) }
    }

    /// Interprets `Aligned128` blocks as native-endian `f32` elements.
    #[must_use]
    pub fn slice_as_f32(buf: &[Self]) -> &[f32] {
        // SAFETY: `Aligned128` is 128-byte aligned (>= align_of::<f32>()) and size 128 = 32 * 4.
        unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<f32>(), buf.len() * 32) }
    }

    /// Mutable native-endian `f32` view over `Aligned128` blocks.
    #[must_use]
    pub fn slice_as_f32_mut(buf: &mut [Self]) -> &mut [f32] {
        // SAFETY: `Aligned128` is 128-byte aligned (>= align_of::<f32>()) and size 128 = 32 * 4.
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<f32>(), buf.len() * 32) }
    }
}

const fn assert_aligned128_layout() {
    assert!(size_of::<Aligned128>() == WEIGHT_TILE_ALIGNMENT_BYTES);
    assert!(align_of::<Aligned128>() == WEIGHT_TILE_ALIGNMENT_BYTES);
}
const _: () = assert_aligned128_layout();

/// Global singleton cache backing default [`jero_core::ArchRecipe`] calls.
static GLOBAL_BITNET_CACHE: Mutex<Option<BitNetModelCache>> = Mutex::new(None);
/// Fast-path lock-free flag avoiding global mutex/HashMap initialization until preloaded.
static GLOBAL_CACHE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// ABA-safe identity descriptor for a backing [`TensorView`] buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorIdentity {
    /// Backing slice base address (`*const u8::addr()`).
    pub data_ptr: usize,
    /// Backing slice length in bytes.
    pub byte_len: usize,
    /// Tensor quantization format.
    pub quant: QuantType,
    /// Number of logical dimensions (`shape.len()`).
    pub ndims: usize,
    /// Up to 4 leading dimension sizes.
    pub shape: [usize; 4],
    /// Fast `FNV-1a` content fingerprint guarding against in-place mutation or heap reuse.
    pub fingerprint64: u64,
}

impl TensorIdentity {
    /// Builds an ABA-safe identity from a borrowed [`TensorView`].
    #[must_use]
    pub fn from_view(view: &TensorView<'_>) -> Self {
        let bytes = view.as_bytes();
        let src_shape = view.shape();
        let mut shape = [0_usize; 4];
        for (dst, &s) in shape.iter_mut().zip(src_shape) {
            *dst = s;
        }
        Self {
            data_ptr: bytes.as_ptr().addr(),
            byte_len: bytes.len(),
            quant: view.quant(),
            ndims: src_shape.len(),
            shape,
            fingerprint64: fnv1a_tensor_fingerprint(bytes, view.quant(), src_shape),
        }
    }
}

#[inline]
fn fnv1a_mix_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn fnv1a_tensor_fingerprint(bytes: &[u8], quant: QuantType, shape: &[usize]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    hash = fnv1a_mix_bytes(hash, &(quant as u32).to_le_bytes());
    hash = fnv1a_mix_bytes(hash, &bytes.len().to_le_bytes());
    for &dim in shape {
        hash = fnv1a_mix_bytes(hash, &dim.to_le_bytes());
    }

    // Hash full content for small buffers (all 1D SubLN norms and compact unit tensors),
    // and sample head, tail, and evenly spaced interior windows for large weight matrices.
    if bytes.len() <= 4096 {
        return fnv1a_mix_bytes(hash, bytes);
    }

    let head = bytes.len().min(64);
    hash = fnv1a_mix_bytes(hash, &bytes[..head]);
    if bytes.len() > 64 {
        let tail_start = bytes.len() - 64;
        hash = fnv1a_mix_bytes(hash, &bytes[tail_start..]);
    }
    let span = bytes.len() - 16;
    let step = (span / 16).max(1);
    for idx in 0..16_usize {
        let off = idx.saturating_mul(step).min(span);
        hash = fnv1a_mix_bytes(hash, &bytes[off..off + 16]);
    }
    hash
}

/// Telemetry counters recorded by [`BitNetModelCache`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BitNetCacheStats {
    /// Number of times `SubLN` / norm `f32` vectors were decoded from raw GGUF storage.
    pub norm_decodes: u64,
    /// Number of cache hits for decoded `SubLN` / norm `f32` vectors.
    pub norm_hits: u64,
    /// Number of times `BitLinear` projection matrices were repacked into `8x16` `i8` tiles.
    pub weight_repacks: u64,
    /// Number of cache hits reusing pre-repacked `BitLinear` `8x16` `i8` tiles.
    pub weight_hits: u64,
    /// Number of times a compact `token_embd` / LM-head matrix was built or cached.
    pub lm_head_repacks: u64,
    /// Number of cache hits for compact `token_embd` / LM-head storage.
    pub lm_head_hits: u64,
    /// Total quantized weight blocks unpacked during repacking (zero incremental calls during infer).
    pub unpack_block_calls: u64,
    /// Number of full cache invalidations triggered by model anchor changes.
    pub model_invalidations: u64,
    /// Number of matrices kept on streaming fallback because caching exceeded `max_cache_bytes`.
    pub budget_skips: u64,
    /// Total resident cache bytes tracked inside this [`BitNetModelCache`].
    pub resident_bytes: usize,
}

/// Tokenizer vocabulary and Llama-3 `<|eot_id|>` / `<|end_of_text|>` boundary contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitNetVocabContract {
    /// Expected model vocabulary size (`vocab_size > 0`).
    pub vocab_size: usize,
    /// `tokenizer.ggml.eos_token_id`, if present.
    pub eos_token_id: Option<u32>,
    /// `tokenizer.ggml.eot_token_id`, if present.
    pub eot_token_id: Option<u32>,
    /// Index of `"<|end_of_text|>"` (or `"<|endoftext|>"`) in `tokenizer.ggml.tokens`, if present.
    pub end_of_text_token_id: Option<u32>,
    /// Index of `"<|eot_id|>"` in `tokenizer.ggml.tokens`, if present.
    pub eot_piece_token_id: Option<u32>,
    /// Number of entries in `tokenizer.ggml.tokens`, if present.
    pub token_list_len: Option<usize>,
    /// `true` when `vocab_size == 128_256` (`LLAMA3_BITNET_VOCAB_SIZE`).
    pub is_llama3_128k: bool,
}

impl BitNetVocabContract {
    /// Parses and validates vocabulary bounds and stop tokens from GGUF metadata.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `expected_vocab_size == 0`, any stop token ID
    /// falls outside `0..expected_vocab_size`, `tokenizer.ggml.tokens` length mismatches
    /// `expected_vocab_size`, or `<|eot_id|>` disagrees with `tokenizer.ggml.eot_token_id`.
    pub fn from_gguf(gguf: &GgufFile<'_>, expected_vocab_size: usize) -> Result<Self, K3Error> {
        let arch = gguf.architecture();
        if expected_vocab_size == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: "expected vocab_size must be positive".to_owned(),
            });
        }

        let parsed_eos = gguf
            .metadata_value("tokenizer.ggml.eos_token_id")
            .and_then(GgufValue::as_u32);
        let parsed_eot = gguf
            .metadata_value("tokenizer.ggml.eot_token_id")
            .and_then(GgufValue::as_u32);

        validate_token_in_vocab(
            "tokenizer.ggml.eos_token_id",
            parsed_eos,
            expected_vocab_size,
            arch,
        )?;
        validate_token_in_vocab(
            "tokenizer.ggml.eot_token_id",
            parsed_eot,
            expected_vocab_size,
            arch,
        )?;

        let mut token_list_len = None;
        let mut eot_piece_token_id = None;
        let mut end_of_text_token_id = None;

        if let Some(GgufValue::Array(GgufArray::String(tokens))) =
            gguf.metadata_value("tokenizer.ggml.tokens")
        {
            token_list_len = Some(tokens.len());
            if tokens.len() != expected_vocab_size {
                return Err(K3Error::CapabilityMismatch {
                    architecture: arch.to_owned(),
                    reason: format!(
                        "tokenizer.ggml.tokens length {} does not match expected vocab_size {expected_vocab_size}",
                        tokens.len()
                    ),
                });
            }

            // In Llama-3 128k GGUFs, special control tokens live at 128_000..128_256; scan that tail
            // first before scanning earlier slots.
            let tail_start = if tokens.len() >= 128_000 { 128_000 } else { 0 };
            for (rel_idx, &tok) in tokens[tail_start..].iter().enumerate() {
                let idx = u32::try_from(tail_start + rel_idx).ok();
                if tok == "<|eot_id|>" && eot_piece_token_id.is_none() {
                    eot_piece_token_id = idx;
                } else if (tok == "<|end_of_text|>" || tok == "<|endoftext|>")
                    && end_of_text_token_id.is_none()
                {
                    end_of_text_token_id = idx;
                }
            }
            if tail_start > 0 && (eot_piece_token_id.is_none() || end_of_text_token_id.is_none()) {
                for (abs_idx, &tok) in tokens[..tail_start].iter().enumerate() {
                    let idx = u32::try_from(abs_idx).ok();
                    if tok == "<|eot_id|>" && eot_piece_token_id.is_none() {
                        eot_piece_token_id = idx;
                    } else if (tok == "<|end_of_text|>" || tok == "<|endoftext|>")
                        && end_of_text_token_id.is_none()
                    {
                        end_of_text_token_id = idx;
                    }
                    if eot_piece_token_id.is_some() && end_of_text_token_id.is_some() {
                        break;
                    }
                }
            }
        }

        if let (Some(explicit_eot), Some(piece_eot)) = (parsed_eot, eot_piece_token_id)
            && explicit_eot != piece_eot
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!(
                    "tokenizer.ggml.eot_token_id ({explicit_eot}) disagrees with '<|eot_id|>' index ({piece_eot})"
                ),
            });
        }

        Ok(Self {
            vocab_size: expected_vocab_size,
            eos_token_id: parsed_eos,
            eot_token_id: parsed_eot,
            end_of_text_token_id,
            eot_piece_token_id,
            token_list_len,
            is_llama3_128k: expected_vocab_size == LLAMA3_BITNET_VOCAB_SIZE,
        })
    }

    /// Asserts that Llama-3 `128,256`-vocab `<|eot_id|>` (`128009`) and `<|end_of_text|>` stops
    /// match canonical IDs inside `0..128_256`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `vocab_size != 128_256`, no canonical `<|eot_id|>`
    /// stop matches `128009`, or any declared stop ID lies outside `0..vocab_size`.
    pub fn expect_llama3_128k_stops(&self) -> Result<(), K3Error> {
        if self.vocab_size != LLAMA3_BITNET_VOCAB_SIZE {
            return Err(K3Error::CapabilityMismatch {
                architecture: "bitnet".to_owned(),
                reason: format!(
                    "expected Llama-3 vocab_size {LLAMA3_BITNET_VOCAB_SIZE}, got {}",
                    self.vocab_size
                ),
            });
        }
        let resolved_eot = self.eot_piece_token_id.or(self.eot_token_id);
        if resolved_eot != Some(LLAMA3_EOT_TOKEN_ID) {
            return Err(K3Error::CapabilityMismatch {
                architecture: "bitnet".to_owned(),
                reason: format!(
                    "expected Llama-3 '<|eot_id|>' stop at token {LLAMA3_EOT_TOKEN_ID}, found {resolved_eot:?}"
                ),
            });
        }
        if let Some(eos) = self.eos_token_id
            && usize::try_from(eos).is_ok_and(|id| id >= self.vocab_size)
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "bitnet".to_owned(),
                reason: format!(
                    "eos_token_id {eos} out of bounds for vocab {}",
                    self.vocab_size
                ),
            });
        }
        Ok(())
    }

    /// Validates whether `token_id` is in `0..self.vocab_size` and returns `true` when it is a
    /// configured stop token (`eos_token_id`, `eot_token_id`, `<|eot_id|>`, or `<|end_of_text|>`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `token_id >= self.vocab_size`.
    pub fn validate_stop_token(&self, token_id: u32) -> Result<bool, K3Error> {
        let idx = usize::try_from(token_id).map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "vocab_contract".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: self.vocab_size,
            reason: format!("token_id {token_id} exceeds usize"),
        })?;
        if idx >= self.vocab_size {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "vocab_contract".to_owned(),
                offset: idx,
                byte_len: 0,
                buffer_len: self.vocab_size,
                reason: format!(
                    "token_id {token_id} is out of bounds for vocab_size {}",
                    self.vocab_size
                ),
            });
        }
        Ok(self.eos_token_id == Some(token_id)
            || self.eot_token_id == Some(token_id)
            || self.end_of_text_token_id == Some(token_id)
            || self.eot_piece_token_id == Some(token_id))
    }

    /// Verifies that `token_embd.weight` (and optional `output.weight`) row counts match `self.vocab_size`
    /// and column width matches `hidden_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if tensor shapes or sizes are inconsistent.
    pub fn validate_weights(
        &self,
        weights: &dyn ModelWeights,
        hidden_dim: usize,
    ) -> Result<(), K3Error> {
        if weights.tensor_count() == 0 {
            return Ok(());
        }
        for (name, required) in [("token_embd.weight", true), ("output.weight", false)] {
            if !weights.contains_tensor(name) {
                if required {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: name.to_owned(),
                        offset: 0,
                        byte_len: 0,
                        buffer_len: 0,
                        reason: format!(
                            "required tensor '{name}' missing during vocab contract check"
                        ),
                    });
                }
                continue;
            }
            let view = weights.tensor(name)?;
            let total_rows = BitNetRecipe::matrix_row_col_dims(&view, hidden_dim, self.vocab_size)?;
            if total_rows != self.vocab_size {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: view.offset(),
                    byte_len: view.byte_len(),
                    buffer_len: view.as_bytes().len(),
                    reason: format!(
                        "tensor '{name}' row count {total_rows} does not match vocab_size {}",
                        self.vocab_size
                    ),
                });
            }
        }
        Ok(())
    }
}

fn validate_token_in_vocab(
    key: &str,
    token_id: Option<u32>,
    vocab_size: usize,
    arch: &str,
) -> Result<(), K3Error> {
    if let Some(id) = token_id {
        let valid = usize::try_from(id).is_ok_and(|idx| idx < vocab_size);
        if !valid {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!("{key} ({id}) exceeds expected vocab_size ({vocab_size})"),
            });
        }
    }
    Ok(())
}

/// Pre-repacked `128-byte`-aligned `8x16` `i8` ternary / block-quantized projection matrix.
///
/// Unpacks `TQ1_0`, `TQ2_0`, `I2_S`, and `Q8_0` weight blocks once at load time into `[n_tile][blk][k_chunk][8][16]`
/// `i8` tiles alongside per-block `f32` scales so inference preserves bit-exact `i32` dot accumulation parity
/// without re-unpacking blocks per token.
#[derive(Debug, Clone)]
pub struct RepackedTernaryMatrix {
    quant: QuantType,
    rows: usize,
    cols: usize,
    n_tiles: usize,
    blocks_per_row: usize,
    block_elems: usize,
    block_k_chunks: usize,
    tiles: Vec<Aligned128>,
    block_scales: Vec<f32>,
}

/// Reusable `8x16` `i8` activation tile scratch buffer allocated outside the transformer layer loop.
#[derive(Debug, Clone, Default)]
pub struct BitLinearActScratch {
    a_tiles: Vec<Aligned128>,
}

impl BitLinearActScratch {
    /// Creates an empty activation tile scratch buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            a_tiles: Vec::new(),
        }
    }

    /// Pre-allocates tile capacity for up to `max_cols` quantized `i8` activations.
    #[must_use]
    pub fn with_max_cols(max_cols: usize) -> Self {
        let k_chunks = max_cols.div_ceil(TILE_K);
        Self {
            a_tiles: vec![Aligned128::zeroed(); k_chunks],
        }
    }

    /// Grows internal scratch capacity if needed to cover at least `max_cols` columns.
    pub fn ensure_max_cols(&mut self, max_cols: usize) {
        let k_chunks = max_cols.div_ceil(TILE_K);
        if self.a_tiles.len() < k_chunks {
            self.a_tiles.resize(k_chunks, Aligned128::zeroed());
        }
    }

    fn prepare_block_tiles(&mut self, act_i8: &[i8], blocks_per_row: usize, block_k_chunks: usize) {
        let total_chunks = blocks_per_row * block_k_chunks;
        if self.a_tiles.len() < total_chunks {
            self.a_tiles.resize(total_chunks, Aligned128::zeroed());
        }
        let dst = Aligned128::slice_as_i8_mut(&mut self.a_tiles[..total_chunks]);
        for ch in 0..total_chunks {
            let chunk = &act_i8[ch * TILE_K..(ch + 1) * TILE_K];
            let ch_base = ch * TILE_BYTES;
            dst[ch_base..ch_base + TILE_K].copy_from_slice(chunk);
            dst[ch_base + TILE_K..ch_base + TILE_BYTES].fill(0);
        }
    }
}

impl RepackedTernaryMatrix {
    /// Repacks `view` into `128-byte`-aligned `8x16` `i8` tiles and per-block `f32` scales.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `view.quant()` is unsupported, dimensions mismatch,
    /// or backing bytes are truncated.
    pub fn from_tensor(view: &TensorView<'_>, rows: usize, cols: usize) -> Result<Self, K3Error> {
        let (mat, _) = Self::from_tensor_counted(view, rows, cols)?;
        Ok(mat)
    }

    /// Repacks `view` and returns the matrix together with the exact count of unpacked source blocks.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `view.quant()` is unsupported, dimensions mismatch,
    /// or backing bytes are truncated.
    pub fn from_tensor_counted(
        view: &TensorView<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<(Self, u64), K3Error> {
        let quant = view.quant();
        let (block_elems, block_bytes, trailer_bytes) = BitNetRecipe::quant_block_layout(quant)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: view.as_bytes().len(),
                reason: format!("unsupported quant for RepackedTernaryMatrix: {quant:?}"),
            })?;
        if rows == 0 || cols == 0 || cols % block_elems != 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: view.as_bytes().len(),
                reason: format!(
                    "RepackedTernaryMatrix dimensions [{rows}, {cols}] invalid for {quant:?} block size {block_elems}"
                ),
            });
        }
        let total_rows = BitNetRecipe::matrix_row_col_dims(view, cols, rows)?;
        let blocks_per_row = cols / block_elems;
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

        let n_tiles = rows.div_ceil(TILE_N);
        let block_k_chunks = block_elems / TILE_K;
        let mut tiles = vec![Aligned128::zeroed(); n_tiles * blocks_per_row * block_k_chunks];
        let mut block_scales = vec![0.0_f32; n_tiles * blocks_per_row * TILE_N];
        let tile_i8 = Aligned128::slice_as_i8_mut(&mut tiles);
        let mut w_buf = [0_i8; 256];
        let mut unpack_calls = 0_u64;

        for nt in 0..n_tiles {
            let r_base = nt * TILE_N;
            let tile_rows = (rows - r_base).min(TILE_N);
            for blk in 0..blocks_per_row {
                let scale_base = (nt * blocks_per_row + blk) * TILE_N;
                let tile_blk_base = (nt * blocks_per_row + blk) * block_k_chunks;
                for j in 0..tile_rows {
                    let row = r_base + j;
                    let boff = (row * blocks_per_row + blk) * block_bytes;
                    let scale = unpack_source_block(
                        quant,
                        &raw[boff..boff + block_bytes],
                        i2_s_scale,
                        &mut w_buf[..block_elems],
                    )?;
                    unpack_calls += 1;
                    block_scales[scale_base + j] = scale;
                    for ch in 0..block_k_chunks {
                        let dst_off = (tile_blk_base + ch) * TILE_BYTES + j * TILE_K;
                        let src_off = ch * TILE_K;
                        tile_i8[dst_off..dst_off + TILE_K]
                            .copy_from_slice(&w_buf[src_off..src_off + TILE_K]);
                    }
                }
            }
        }

        Ok((
            Self {
                quant,
                rows,
                cols,
                n_tiles,
                blocks_per_row,
                block_elems,
                block_k_chunks,
                tiles,
                block_scales,
            },
            unpack_calls,
        ))
    }

    /// Quantization type of the source weight tensor.
    #[must_use]
    pub const fn quant(&self) -> QuantType {
        self.quant
    }

    /// Number of `TILE_N = 8` output-row tiles.
    #[must_use]
    pub const fn n_tiles(&self) -> usize {
        self.n_tiles
    }

    /// Output row count (`N`).
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Input column width (`K`).
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Source quantized blocks along each row (`cols / block_elems`).
    #[must_use]
    pub const fn blocks_per_row(&self) -> usize {
        self.blocks_per_row
    }

    /// Source elements per scale block (`256` for `TQ1_0`/`TQ2_0`, `128` for `I2_S`, `32` for `Q8_0`).
    #[must_use]
    pub const fn block_elems(&self) -> usize {
        self.block_elems
    }

    /// `16-element` `IME`/`RVV` chunks per scale block.
    #[must_use]
    pub const fn block_k_chunks(&self) -> usize {
        self.block_k_chunks
    }

    /// Total byte size of `128-byte`-aligned `8x16` `i8` tiles.
    #[must_use]
    pub fn aligned_bytes(&self) -> usize {
        self.tiles.len() * TILE_BYTES
    }

    /// Total bytes resident in cache (`aligned_bytes() + block_scales` bytes).
    #[must_use]
    pub fn total_resident_bytes(&self) -> usize {
        self.aligned_bytes() + self.block_scales.len() * size_of::<f32>()
    }

    /// Base pointer to the `128-byte`-aligned tile buffer.
    #[must_use]
    pub fn payload_ptr(&self) -> *const u8 {
        self.tiles.as_ptr().cast::<u8>()
    }

    /// Returns `true` when the tile buffer address is aligned to `128` bytes.
    #[must_use]
    pub fn is_128_byte_aligned(&self) -> bool {
        self.payload_ptr().addr() % WEIGHT_TILE_ALIGNMENT_BYTES == 0
    }

    /// Raw slice of `128-byte`-aligned tiles (`[n_tile][blk][k_chunk][8][16]`).
    #[must_use]
    pub fn tiles(&self) -> &[Aligned128] {
        &self.tiles
    }

    /// Per-block `f32` scales indexed as `[(nt * blocks_per_row + blk) * 8 + row_in_tile]`.
    #[must_use]
    pub fn block_scales(&self) -> &[f32] {
        &self.block_scales
    }

    /// Reconstructs logical weight row `row` into `out` (`out.len() == self.cols()`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `row >= self.rows()` or `out.len() != self.cols()`.
    pub fn dequantize_row(&self, row: usize, out: &mut [f32]) -> Result<(), K3Error> {
        if row >= self.rows || out.len() != self.cols {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "RepackedTernaryMatrix::dequantize_row".to_owned(),
                offset: row,
                byte_len: size_of_val(out),
                buffer_len: self.cols * size_of::<f32>(),
                reason: format!(
                    "invalid row ({row}/{}) or output width ({}/{})",
                    self.rows,
                    out.len(),
                    self.cols
                ),
            });
        }
        let nt = row / TILE_N;
        let j = row % TILE_N;
        let tile_i8 = Aligned128::slice_as_i8(&self.tiles);
        for blk in 0..self.blocks_per_row {
            let scale = self.block_scales[(nt * self.blocks_per_row + blk) * TILE_N + j];
            let tile_blk_base = (nt * self.blocks_per_row + blk) * self.block_k_chunks;
            for ch in 0..self.block_k_chunks {
                let src_off = (tile_blk_base + ch) * TILE_BYTES + j * TILE_K;
                let dst_off = blk * self.block_elems + ch * TILE_K;
                for k in 0..TILE_K {
                    out[dst_off + k] = f32::from(tile_i8[src_off + k]) * scale;
                }
            }
        }
        Ok(())
    }

    /// Returns exact `(dot_i32, block_scale)` pairs for every block in `row` using repacked `8x16`
    /// tiles without calling block unpackers.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `row >= self.rows()` or `act_i8.len() != self.cols()`.
    pub fn dot_blocks_i32(&self, row: usize, act_i8: &[i8]) -> Result<Vec<(i32, f32)>, K3Error> {
        if row >= self.rows || act_i8.len() != self.cols {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "RepackedTernaryMatrix::dot_blocks_i32".to_owned(),
                offset: row,
                byte_len: act_i8.len(),
                buffer_len: self.cols,
                reason: format!(
                    "invalid row ({row}/{}) or activation length ({}/{})",
                    self.rows,
                    act_i8.len(),
                    self.cols
                ),
            });
        }
        let mut scratch = BitLinearActScratch::with_max_cols(self.cols);
        scratch.prepare_block_tiles(act_i8, self.blocks_per_row, self.block_k_chunks);

        let nt = row / TILE_N;
        let j = row % TILE_N;
        let a_i8 = Aligned128::slice_as_i8(&scratch.a_tiles);
        let b_i8 = Aligned128::slice_as_i8(&self.tiles);
        let blk_stride = self.block_k_chunks * TILE_BYTES;
        let mut dots = Vec::with_capacity(self.blocks_per_row);

        for blk in 0..self.blocks_per_row {
            let a_blk = &a_i8[blk * blk_stride..(blk + 1) * blk_stride];
            let b_idx = (nt * self.blocks_per_row + blk) * blk_stride;
            let b_blk = &b_i8[b_idx..b_idx + blk_stride];
            let dot_c = compute_tile_row0_dots(a_blk, b_blk, self.block_k_chunks);
            let scale = self.block_scales[(nt * self.blocks_per_row + blk) * TILE_N + j];
            dots.push((dot_c[j], scale));
        }
        Ok(dots)
    }

    /// Executes `BitLinear` matrix-vector product `out = W_repacked * (act_i8 * act_scale)` from
    /// pre-repacked `128-byte` `8x16` `i8` tiles.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `out.len() > self.rows()` or `act_i8.len() != self.cols()`.
    pub fn bitlinear_gemv(
        &self,
        out: &mut [f32],
        act_i8: &[i8],
        act_scale: f32,
    ) -> Result<(), K3Error> {
        let mut scratch = BitLinearActScratch::with_max_cols(self.cols);
        self.bitlinear_gemv_with_scratch(out, act_i8, act_scale, &mut scratch)
    }

    /// Executes `BitLinear` GEMV using a caller-owned reusable [`BitLinearActScratch`] buffer.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `out.len() > self.rows()` or `act_i8.len() != self.cols()`.
    pub fn bitlinear_gemv_with_scratch(
        &self,
        out: &mut [f32],
        act_i8: &[i8],
        act_scale: f32,
        scratch: &mut BitLinearActScratch,
    ) -> Result<(), K3Error> {
        if out.len() > self.rows || act_i8.len() != self.cols {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "RepackedTernaryMatrix::bitlinear_gemv".to_owned(),
                offset: 0,
                byte_len: out.len(),
                buffer_len: self.rows,
                reason: format!(
                    "GEMV shape mismatch: out {} <= {}, act {} == {}",
                    out.len(),
                    self.rows,
                    act_i8.len(),
                    self.cols
                ),
            });
        }
        if out.is_empty() {
            return Ok(());
        }

        scratch.prepare_block_tiles(act_i8, self.blocks_per_row, self.block_k_chunks);
        let a_i8 = Aligned128::slice_as_i8(&scratch.a_tiles);
        let b_i8 = Aligned128::slice_as_i8(&self.tiles);
        let blk_stride = self.block_k_chunks * TILE_BYTES;
        let active_n_tiles = out.len().div_ceil(TILE_N);

        for nt in 0..active_n_tiles {
            let r_base = nt * TILE_N;
            let tile_rows = (out.len() - r_base).min(TILE_N);
            let mut row_sums = [0.0_f32; TILE_N];

            for blk in 0..self.blocks_per_row {
                let a_blk = &a_i8[blk * blk_stride..(blk + 1) * blk_stride];
                let b_idx = (nt * self.blocks_per_row + blk) * blk_stride;
                let b_blk = &b_i8[b_idx..b_idx + blk_stride];
                let dot_c = compute_tile_row0_dots(a_blk, b_blk, self.block_k_chunks);
                let scale_base = (nt * self.blocks_per_row + blk) * TILE_N;

                for j in 0..tile_rows {
                    let effective_scale = self.block_scales[scale_base + j] * act_scale;
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "block int32 accumulation fits safely in f32 mantissa"
                    )]
                    {
                        row_sums[j] += (dot_c[j] as f32) * effective_scale;
                    }
                }
            }
            out[r_base..r_base + tile_rows].copy_from_slice(&row_sums[..tile_rows]);
        }

        Ok(())
    }

    /// Executes per-32-element `Q8_0 x Q8_0` matrix-vector multiplication for non-kernel `Q8_0` LM heads.
    #[cfg(not(feature = "with-kernels"))]
    fn gemv_q8_per32_blocks(&self, out: &mut [f32], hidden: &[f32]) -> Result<(), K3Error> {
        if out.len() > self.rows || hidden.len() != self.cols || self.quant != QuantType::Q8_0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "RepackedTernaryMatrix::gemv_q8_per32_blocks".to_owned(),
                offset: 0,
                byte_len: out.len(),
                buffer_len: self.rows,
                reason: "Q8_0 LM head projection dimension mismatch".to_owned(),
            });
        }
        let mut x_q = vec![0_i8; self.cols];
        let mut x_scales = vec![0.0_f32; self.blocks_per_row];
        for blk in 0..self.blocks_per_row {
            let sub = &hidden[blk * 32..(blk + 1) * 32];
            x_scales[blk] = quantize_q8_block32(sub, &mut x_q[blk * 32..(blk + 1) * 32]);
        }
        let mut scratch = BitLinearActScratch::with_max_cols(self.cols);
        scratch.prepare_block_tiles(&x_q, self.blocks_per_row, self.block_k_chunks);

        let a_i8 = Aligned128::slice_as_i8(&scratch.a_tiles);
        let b_i8 = Aligned128::slice_as_i8(&self.tiles);
        let blk_stride = self.block_k_chunks * TILE_BYTES;
        let active_n_tiles = out.len().div_ceil(TILE_N);

        for nt in 0..active_n_tiles {
            let r_base = nt * TILE_N;
            let tile_rows = (out.len() - r_base).min(TILE_N);
            let mut row_sums = [0.0_f32; TILE_N];
            for blk in 0..self.blocks_per_row {
                let a_blk = &a_i8[blk * blk_stride..(blk + 1) * blk_stride];
                let b_idx = (nt * self.blocks_per_row + blk) * blk_stride;
                let b_blk = &b_i8[b_idx..b_idx + blk_stride];
                let dot_c = compute_tile_row0_dots(a_blk, b_blk, self.block_k_chunks);
                let scale_base = (nt * self.blocks_per_row + blk) * TILE_N;
                for j in 0..tile_rows {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "32-element int32 block sum fits in f32 mantissa"
                    )]
                    {
                        row_sums[j] +=
                            (dot_c[j] as f32) * self.block_scales[scale_base + j] * x_scales[blk];
                    }
                }
            }
            out[r_base..r_base + tile_rows].copy_from_slice(&row_sums[..tile_rows]);
        }
        Ok(())
    }
}

#[cfg(not(feature = "with-kernels"))]
fn quantize_q8_block32(src: &[f32], dst: &mut [i8]) -> f32 {
    let mut amax = 0.0_f32;
    for &v in src {
        if v.is_finite() {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
    }
    if amax <= 0.0 || !amax.is_finite() {
        dst.fill(0);
        return 0.0;
    }
    let scale = amax / 127.0;
    let inv = 1.0 / scale;
    for (out, &v) in dst.iter_mut().zip(src) {
        let q = if v.is_finite() {
            (v * inv).round().clamp(-127.0, 127.0)
        } else {
            0.0
        };
        #[expect(clippy::cast_possible_truncation, reason = "clamped to [-127, 127]")]
        {
            *out = q as i8;
        }
    }
    scale
}

#[inline]
fn compute_tile_row0_dots(a_blk: &[i8], b_blk: &[i8], k_chunks: usize) -> [i32; TILE_N] {
    #[cfg(feature = "with-kernels")]
    {
        let mut dot_c = [0_i32; TILE_C];
        gemm_i8_ime(a_blk, 1, b_blk, 0..1, k_chunks, |_, _, tile| {
            dot_c = *tile;
        });
        let mut out = [0_i32; TILE_N];
        out.copy_from_slice(&dot_c[..TILE_N]);
        out
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let mut out = [0_i32; TILE_N];
        for ch in 0..k_chunks {
            let a_row = &a_blk[ch * TILE_BYTES..ch * TILE_BYTES + TILE_K];
            let b_base = ch * TILE_BYTES;
            for j in 0..TILE_N {
                let b_row = &b_blk[b_base + j * TILE_K..b_base + (j + 1) * TILE_K];
                let mut acc = 0_i32;
                for k in 0..TILE_K {
                    acc += i32::from(a_row[k]) * i32::from(b_row[k]);
                }
                out[j] += acc;
            }
        }
        out
    }
}

fn extract_i2s_trailer(
    raw: &[u8],
    payload_bytes: usize,
    view: &TensorView<'_>,
) -> Result<f32, K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        if raw.len() < payload_bytes + 32 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: raw.len(),
                reason: "I2_S buffer missing 32-byte trailer scale".to_owned(),
            });
        }
        Ok(read_i2_s_trailer_scale(&raw[payload_bytes..]))
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        BitNetRecipe::extract_i2s_scale(raw, payload_bytes, view)
    }
}

fn unpack_source_block(
    quant: QuantType,
    block_raw: &[u8],
    i2_s_scale: f32,
    w_out: &mut [i8],
) -> Result<f32, K3Error> {
    match quant {
        QuantType::TQ1_0 => {
            let blk: &[u8; 54] = block_raw
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: "TQ1_0".to_owned(),
                    offset: 0,
                    byte_len: block_raw.len(),
                    buffer_len: 54,
                    reason: "invalid TQ1_0 block byte length".to_owned(),
                })?;
            let dst: &mut [i8; 256] = w_out.try_into().expect("256-element TQ1_0 slice");
            #[cfg(feature = "with-kernels")]
            {
                Ok(unpack_tq1_0_block_i8(blk, dst))
            }
            #[cfg(not(feature = "with-kernels"))]
            {
                Ok(BitNetRecipe::unpack_tq1_0_block_i8(blk, dst))
            }
        }
        QuantType::TQ2_0 => {
            let blk: &[u8; 66] = block_raw
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: "TQ2_0".to_owned(),
                    offset: 0,
                    byte_len: block_raw.len(),
                    buffer_len: 66,
                    reason: "invalid TQ2_0 block byte length".to_owned(),
                })?;
            let dst: &mut [i8; 256] = w_out.try_into().expect("256-element TQ2_0 slice");
            #[cfg(feature = "with-kernels")]
            {
                Ok(unpack_tq2_0_block_i8(blk, dst))
            }
            #[cfg(not(feature = "with-kernels"))]
            {
                Ok(BitNetRecipe::unpack_tq2_0_block_i8(blk, dst))
            }
        }
        QuantType::I2_S => {
            let blk: &[u8; 32] = block_raw
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: "I2_S".to_owned(),
                    offset: 0,
                    byte_len: block_raw.len(),
                    buffer_len: 32,
                    reason: "invalid I2_S block byte length".to_owned(),
                })?;
            let dst: &mut [i8; 128] = w_out.try_into().expect("128-element I2_S slice");
            #[cfg(feature = "with-kernels")]
            unpack_i2_s_block_i8(blk, dst);
            #[cfg(not(feature = "with-kernels"))]
            BitNetRecipe::unpack_i2s_block_i8(blk, dst);
            Ok(i2_s_scale)
        }
        QuantType::Q8_0 => {
            let scale = BitNetRecipe::convert_f16(u16::from_le_bytes([block_raw[0], block_raw[1]]));
            #[expect(
                clippy::cast_possible_wrap,
                reason = "Q8_0 payload stores signed i8 values"
            )]
            for (dst, &q) in w_out.iter_mut().zip(&block_raw[2..34]) {
                *dst = q as i8;
            }
            Ok(scale)
        }
        _ => Err(K3Error::TensorBoundsError {
            tensor_name: "unpack_source_block".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: format!("unsupported block quant {quant:?}"),
        }),
    }
}

/// Compact `128-byte`-aligned embedding / LM-head storage avoiding `f32` table expansion.
#[derive(Debug, Clone)]
pub enum CompactLmHeadMatrix {
    /// Kernel-backed compact [`RepackedWeight`] (`DenseHalf` for `F16`/`BF16`, `BlockScaledI8` for `Q8_0`).
    #[cfg(feature = "with-kernels")]
    Repacked(RepackedWeight),
    /// Portable `128-byte`-aligned compact `F16`/`BF16` rows when `with-kernels` is disabled.
    #[cfg(not(feature = "with-kernels"))]
    DenseHalf {
        /// Whether stored `u16` values are `BF16` (`true`) or `IEEE 754 F16` (`false`).
        is_bf16: bool,
        /// Source quantization (`F16` or `BF16`).
        quant: QuantType,
        /// Vocabulary row count.
        rows: usize,
        /// Hidden width per row.
        cols: usize,
        /// `Aligned128` blocks per row (`cols.div_ceil(64)`).
        blocks_per_row: usize,
        /// `128-byte`-aligned `u16` rows.
        data: Vec<Aligned128>,
    },
    /// Portable `128-byte`-aligned `Q8_0` `8x16` `i8` tiles when `with-kernels` is disabled.
    #[cfg(not(feature = "with-kernels"))]
    BlockQ8(RepackedTernaryMatrix),
    /// Portable `128-byte`-aligned `F32` rows when `with-kernels` is disabled.
    #[cfg(not(feature = "with-kernels"))]
    DenseF32 {
        /// Vocabulary row count.
        rows: usize,
        /// Hidden width per row.
        cols: usize,
        /// `Aligned128` blocks per row (`cols.div_ceil(32)`).
        blocks_per_row: usize,
        /// `128-byte`-aligned `f32` rows.
        data: Vec<Aligned128>,
    },
    /// Ternary `TQ1_0`/`TQ2_0`/`I2_S` `8x16` `i8` tiles for ternary LM heads.
    Ternary(RepackedTernaryMatrix),
}

impl CompactLmHeadMatrix {
    /// Constructs compact `128-byte`-aligned storage for `token_embd.weight` or `output.weight`,
    /// keeping `F16`/`BF16` in compact `16-bit` mode (`dense_f32_max_elems = 0`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if the tensor quant or buffer bounds are invalid.
    pub fn from_tensor_counted(
        view: &TensorView<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<(Self, u64), K3Error> {
        let quant = view.quant();
        match quant {
            QuantType::TQ1_0 | QuantType::TQ2_0 | QuantType::I2_S => {
                let (mat, calls) = RepackedTernaryMatrix::from_tensor_counted(view, rows, cols)?;
                Ok((Self::Ternary(mat), calls))
            }
            QuantType::F16 | QuantType::BF16 | QuantType::Q8_0 | QuantType::F32 => {
                #[cfg(feature = "with-kernels")]
                {
                    let repacked = RepackedWeight::from_tensor_with_half_limit(
                        view,
                        rows,
                        cols,
                        DEFAULT_DENSE_F32_MAX_ELEMENTS,
                    )?;
                    let unpack_calls = if quant == QuantType::Q8_0 {
                        u64::try_from(rows * (cols / 32)).unwrap_or(0)
                    } else {
                        0
                    };
                    Ok((Self::Repacked(repacked), unpack_calls))
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    Self::build_portable(view, rows, cols)
                }
            }
            _ => Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: view.as_bytes().len(),
                reason: format!("unsupported quant for CompactLmHeadMatrix: {quant:?}"),
            }),
        }
    }

    #[cfg(not(feature = "with-kernels"))]
    fn build_portable(
        view: &TensorView<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<(Self, u64), K3Error> {
        let quant = view.quant();
        if quant == QuantType::Q8_0 {
            let (mat, calls) = RepackedTernaryMatrix::from_tensor_counted(view, rows, cols)?;
            return Ok((Self::BlockQ8(mat), calls));
        }
        let total_rows = BitNetRecipe::matrix_row_col_dims(view, cols, rows)?;
        let raw = view.as_bytes();
        let elem_bytes = if quant == QuantType::F32 { 4 } else { 2 };
        let required = total_rows * cols * elem_bytes;
        if raw.len() < required {
            return Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: raw.len(),
                reason: format!(
                    "compact table buffer {} is smaller than required {required} bytes",
                    raw.len()
                ),
            });
        }
        if quant == QuantType::F32 {
            let blocks_per_row = cols.div_ceil(32);
            let mut data = vec![Aligned128::zeroed(); rows * blocks_per_row];
            for r in 0..rows {
                let row_f32 = Aligned128::slice_as_f32_mut(
                    &mut data[r * blocks_per_row..(r + 1) * blocks_per_row],
                );
                let src_base = r * cols * 4;
                for (c, dst) in row_f32.iter_mut().enumerate().take(cols) {
                    let off = src_base + c * 4;
                    let v =
                        f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                    *dst = if v.is_finite() { v } else { 0.0 };
                }
            }
            return Ok((
                Self::DenseF32 {
                    rows,
                    cols,
                    blocks_per_row,
                    data,
                },
                0,
            ));
        }

        let blocks_per_row = cols.div_ceil(64);
        let mut data = vec![Aligned128::zeroed(); rows * blocks_per_row];
        for r in 0..rows {
            let row_u16 = Aligned128::slice_as_u16_mut(
                &mut data[r * blocks_per_row..(r + 1) * blocks_per_row],
            );
            let src_base = r * cols * 2;
            for (c, dst) in row_u16.iter_mut().enumerate().take(cols) {
                let off = src_base + c * 2;
                *dst = u16::from_le_bytes([raw[off], raw[off + 1]]);
            }
        }
        Ok((
            Self::DenseHalf {
                is_bf16: quant == QuantType::BF16,
                quant,
                rows,
                cols,
                blocks_per_row,
                data,
            },
            0,
        ))
    }

    /// Source quantization format.
    #[must_use]
    pub const fn quant(&self) -> QuantType {
        match self {
            #[cfg(feature = "with-kernels")]
            Self::Repacked(w) => w.quant(),
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseHalf { quant, .. } => *quant,
            #[cfg(not(feature = "with-kernels"))]
            Self::BlockQ8(_) => QuantType::Q8_0,
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseF32 { .. } => QuantType::F32,
            Self::Ternary(t) => t.quant(),
        }
    }

    /// Number of stored vocabulary rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        match self {
            #[cfg(feature = "with-kernels")]
            Self::Repacked(w) => w.rows(),
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseHalf { rows, .. } | Self::DenseF32 { rows, .. } => *rows,
            #[cfg(not(feature = "with-kernels"))]
            Self::BlockQ8(m) => m.rows(),
            Self::Ternary(m) => m.rows(),
        }
    }

    /// Column width (`hidden_dim`).
    #[must_use]
    pub const fn cols(&self) -> usize {
        match self {
            #[cfg(feature = "with-kernels")]
            Self::Repacked(w) => w.cols(),
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseHalf { cols, .. } | Self::DenseF32 { cols, .. } => *cols,
            #[cfg(not(feature = "with-kernels"))]
            Self::BlockQ8(m) => m.cols(),
            Self::Ternary(m) => m.cols(),
        }
    }

    /// Bytes occupied by the `128-byte`-aligned weight payload.
    #[must_use]
    pub fn aligned_bytes(&self) -> usize {
        match self {
            #[cfg(feature = "with-kernels")]
            Self::Repacked(w) => w.aligned_bytes(),
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseHalf { data, .. } | Self::DenseF32 { data, .. } => data.len() * TILE_BYTES,
            #[cfg(not(feature = "with-kernels"))]
            Self::BlockQ8(m) => m.aligned_bytes(),
            Self::Ternary(m) => m.aligned_bytes(),
        }
    }

    /// Base pointer to the `128-byte`-aligned weight payload.
    #[must_use]
    pub fn payload_ptr(&self) -> *const u8 {
        match self {
            #[cfg(feature = "with-kernels")]
            Self::Repacked(w) => w.payload_ptr(),
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseHalf { data, .. } | Self::DenseF32 { data, .. } => {
                data.as_ptr().cast::<u8>()
            }
            #[cfg(not(feature = "with-kernels"))]
            Self::BlockQ8(m) => m.payload_ptr(),
            Self::Ternary(m) => m.payload_ptr(),
        }
    }

    /// Returns `true` when the underlying payload is `128-byte`-aligned.
    #[must_use]
    pub fn is_128_byte_aligned(&self) -> bool {
        self.payload_ptr().addr() % WEIGHT_TILE_ALIGNMENT_BYTES == 0
    }

    /// Returns `true` when `F16`/`BF16` weights are kept in compact `u16` storage rather than expanded `f32`.
    #[must_use]
    pub const fn is_compact_half(&self) -> bool {
        match self {
            #[cfg(feature = "with-kernels")]
            Self::Repacked(w) => w.is_compact_half(),
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseHalf { .. } => true,
            #[cfg(not(feature = "with-kernels"))]
            Self::BlockQ8(_) | Self::DenseF32 { .. } => false,
            Self::Ternary(_) => false,
        }
    }

    /// Returns `true` when weights remain compact (`DenseHalf`, `Q8_0`, or ternary tiles) rather than `f32`.
    #[must_use]
    pub const fn is_compact_storage(&self) -> bool {
        match self {
            #[cfg(feature = "with-kernels")]
            Self::Repacked(w) => !matches!(w.quant(), QuantType::F32),
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseF32 { .. } => false,
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseHalf { .. } | Self::BlockQ8(_) => true,
            Self::Ternary(_) => true,
        }
    }

    /// Reconstructs one embedding vector (`row` in `0..self.rows()`) into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `row >= self.rows()` or `out.len() != self.cols()`.
    pub fn embed_row(&self, row: usize, out: &mut [f32]) -> Result<(), K3Error> {
        match self {
            #[cfg(feature = "with-kernels")]
            Self::Repacked(w) => w.dequantize_row(row, out),
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseHalf {
                is_bf16,
                rows,
                cols,
                blocks_per_row,
                data,
                ..
            } => {
                if row >= *rows || out.len() != *cols {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: "CompactLmHeadMatrix::embed_row".to_owned(),
                        offset: row,
                        byte_len: out.len(),
                        buffer_len: *cols,
                        reason: "embedding row or width out of bounds".to_owned(),
                    });
                }
                let row_u16 = Aligned128::slice_as_u16(
                    &data[row * blocks_per_row..(row + 1) * blocks_per_row],
                );
                for (dst, &bits) in out.iter_mut().zip(&row_u16[..*cols]) {
                    *dst = if *is_bf16 {
                        BitNetRecipe::convert_bf16(bits)
                    } else {
                        BitNetRecipe::convert_f16(bits)
                    };
                }
                Ok(())
            }
            #[cfg(not(feature = "with-kernels"))]
            Self::BlockQ8(m) => m.dequantize_row(row, out),
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseF32 {
                rows,
                cols,
                blocks_per_row,
                data,
            } => {
                if row >= *rows || out.len() != *cols {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: "CompactLmHeadMatrix::embed_row".to_owned(),
                        offset: row,
                        byte_len: out.len(),
                        buffer_len: *cols,
                        reason: "embedding row or width out of bounds".to_owned(),
                    });
                }
                let row_f32 = Aligned128::slice_as_f32(
                    &data[row * blocks_per_row..(row + 1) * blocks_per_row],
                );
                out.copy_from_slice(&row_f32[..*cols]);
                Ok(())
            }
            Self::Ternary(m) => m.dequantize_row(row, out),
        }
    }

    /// Projects `hidden` directly from compact storage into `logits` (`logits.len() <= self.rows()`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if dimensions mismatch.
    pub fn project_logits(&self, logits: &mut [f32], hidden: &[f32]) -> Result<(), K3Error> {
        if logits.is_empty() {
            return Ok(());
        }
        if logits.len() > self.rows() || hidden.len() != self.cols() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "CompactLmHeadMatrix::project_logits".to_owned(),
                offset: 0,
                byte_len: logits.len(),
                buffer_len: self.rows(),
                reason: format!(
                    "LM head projection mismatch: logits {} <= {}, hidden {} == {}",
                    logits.len(),
                    self.rows(),
                    hidden.len(),
                    self.cols()
                ),
            });
        }
        match self {
            #[cfg(feature = "with-kernels")]
            Self::Repacked(w) => {
                let act = QuantizedActQ8::quantize_row(hidden)?;
                w.gemv_rows(logits, &act, 0..logits.len())
            }
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseHalf {
                is_bf16,
                cols,
                blocks_per_row,
                data,
                ..
            } => {
                for (r, slot) in logits.iter_mut().enumerate() {
                    let row_u16 = Aligned128::slice_as_u16(
                        &data[r * blocks_per_row..(r + 1) * blocks_per_row],
                    );
                    let mut sum = 0.0_f32;
                    for (&bits, &x) in row_u16[..*cols].iter().zip(hidden) {
                        let w = if *is_bf16 {
                            BitNetRecipe::convert_bf16(bits)
                        } else {
                            BitNetRecipe::convert_f16(bits)
                        };
                        sum += w * x;
                    }
                    *slot = sum;
                }
                Ok(())
            }
            #[cfg(not(feature = "with-kernels"))]
            Self::BlockQ8(m) => m.gemv_q8_per32_blocks(logits, hidden),
            #[cfg(not(feature = "with-kernels"))]
            Self::DenseF32 {
                cols,
                blocks_per_row,
                data,
                ..
            } => {
                for (r, slot) in logits.iter_mut().enumerate() {
                    let row_f32 = Aligned128::slice_as_f32(
                        &data[r * blocks_per_row..(r + 1) * blocks_per_row],
                    );
                    *slot = row_f32[..*cols]
                        .iter()
                        .zip(hidden)
                        .map(|(&w, &x)| w * x)
                        .sum();
                }
                Ok(())
            }
            Self::Ternary(m) => {
                let mut act_i8 = vec![0_i8; hidden.len()];
                let scale = BitNetRecipe::quantize_absmax_i8(&mut act_i8, hidden)?;
                m.bitlinear_gemv(logits, &act_i8, scale)
            }
        }
    }
}

struct CachedNorm {
    identity: TensorIdentity,
    values: Arc<[f32]>,
    byte_len: usize,
}

struct CachedProjection {
    identity: TensorIdentity,
    matrix: Arc<RepackedTernaryMatrix>,
    byte_len: usize,
}

struct CachedLmHead {
    key: String,
    identity: TensorIdentity,
    matrix: Arc<CompactLmHeadMatrix>,
    byte_len: usize,
}

/// Model-scoped cache holding pre-repacked `128-byte` `8x16` `i8` projection tiles,
/// pre-decoded `SubLN` norm vectors, and compact tied/untied `token_embd` / LM-head storage.
pub struct BitNetModelCache {
    anchor: Option<TensorIdentity>,
    norms: HashMap<String, CachedNorm>,
    projections: HashMap<String, CachedProjection>,
    token_embd: Option<CachedLmHead>,
    lm_head: Option<CachedLmHead>,
    act_scratch: BitLinearActScratch,
    populate_on_miss: bool,
    max_cache_bytes: usize,
    stats: BitNetCacheStats,
}

impl Default for BitNetModelCache {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for BitNetModelCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BitNetModelCache")
            .field("anchor", &self.anchor)
            .field("cached_norms", &self.norms.len())
            .field("cached_projections", &self.projections.len())
            .field("has_token_embd", &self.token_embd.is_some())
            .field("has_lm_head", &self.lm_head.is_some())
            .field("act_scratch", &self.act_scratch)
            .field("populate_on_miss", &self.populate_on_miss)
            .field("max_cache_bytes", &self.max_cache_bytes)
            .field("stats", &self.stats)
            .finish()
    }
}

impl BitNetModelCache {
    /// Creates a new cache with [`DEFAULT_MAX_REPACK_CACHE_BYTES`] (`24 GiB`).
    #[must_use]
    pub fn new() -> Self {
        Self::with_max_bytes(DEFAULT_MAX_REPACK_CACHE_BYTES)
    }

    /// Creates a cache bounded by `max_cache_bytes`.
    #[must_use]
    pub fn with_max_bytes(max_cache_bytes: usize) -> Self {
        Self {
            anchor: None,
            norms: HashMap::new(),
            projections: HashMap::new(),
            token_embd: None,
            lm_head: None,
            act_scratch: BitLinearActScratch::new(),
            populate_on_miss: true,
            max_cache_bytes,
            stats: BitNetCacheStats::default(),
        }
    }

    /// Global singleton default: read-only lookup on un-preloaded models so cold `decode_step`
    /// keeps zero dynamic cache-insertion allocations until [`Self::preload_global`] runs.
    #[must_use]
    fn new_deferred() -> Self {
        Self {
            populate_on_miss: false,
            ..Self::new()
        }
    }

    /// Runs `f` with exclusive access to the process-global [`BitNetModelCache`].
    pub fn with_global<R>(f: impl FnOnce(&mut Self) -> R) -> R {
        let mut guard = GLOBAL_BITNET_CACHE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let cache = guard.get_or_insert_with(Self::new_deferred);
        f(cache)
    }

    /// Returns `true` when the process-global cache has been populated by preload or validation.
    #[must_use]
    pub fn is_global_active() -> bool {
        GLOBAL_CACHE_ACTIVE.load(Ordering::Acquire)
    }

    /// Preloads `recipe` + `weights` into the process-global [`BitNetModelCache`] at load time.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if required tensors are malformed or invalid.
    pub fn preload_global(
        recipe: &BitNetRecipe,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        Self::with_global(|cache| cache.preload_model(recipe, weights))?;
        if weights.tensor_count() > 0 {
            GLOBAL_CACHE_ACTIVE.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Returns a snapshot of the global cache telemetry counters.
    #[must_use]
    pub fn global_stats() -> BitNetCacheStats {
        Self::with_global(|cache| cache.stats())
    }

    /// Drops all entries and resets telemetry in the global cache.
    pub fn clear_global() {
        GLOBAL_CACHE_ACTIVE.store(false, Ordering::Release);
        Self::with_global(Self::clear);
    }

    /// Current cache telemetry counters.
    #[must_use]
    pub const fn stats(&self) -> BitNetCacheStats {
        self.stats
    }

    /// Configured memory ceiling in bytes.
    #[must_use]
    pub const fn max_cache_bytes(&self) -> usize {
        self.max_cache_bytes
    }

    /// Current cached resident bytes.
    #[must_use]
    pub const fn resident_bytes(&self) -> usize {
        self.stats.resident_bytes
    }

    /// Clears cached norms, projections, and embeddings without resetting cumulative stats.
    pub fn invalidate_entries(&mut self) {
        self.anchor = None;
        self.norms.clear();
        self.projections.clear();
        self.token_embd = None;
        self.lm_head = None;
        self.stats.resident_bytes = 0;
    }

    /// Clears all entries and resets cumulative counters.
    pub fn clear(&mut self) {
        self.invalidate_entries();
        self.stats = BitNetCacheStats::default();
    }

    /// Verifies the model identity anchor (`token_embd.weight` -> `blk.0.attn_q.weight` ->
    /// `output_norm.weight`) and invalidates stale entries if a different model is bound.
    ///
    /// # Errors
    ///
    /// Propagates weight lookup errors other than absent anchor candidates.
    pub fn ensure_model_anchor(&mut self, weights: &dyn ModelWeights) -> Result<(), K3Error> {
        if weights.tensor_count() == 0 {
            if self.anchor.is_some() {
                self.invalidate_entries();
                self.stats.model_invalidations += 1;
            }
            return Ok(());
        }
        let next_anchor = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "output_norm.weight",
        ]
        .into_iter()
        .find(|name| weights.contains_tensor(name))
        .map(|name| weights.tensor(name).map(|v| TensorIdentity::from_view(&v)))
        .transpose()?;

        if let Some(next) = next_anchor {
            if self.anchor.is_some_and(|prev| prev != next) {
                self.invalidate_entries();
                self.stats.model_invalidations += 1;
            }
            self.anchor = Some(next);
        }
        Ok(())
    }

    /// Pre-decodes all `SubLN`/output norms, repacks all ternary/Q8 projection matrices, and loads
    /// compact `token_embd` (and untied `output.weight`, if present) up front.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if required tensors are malformed or missing.
    pub fn preload_model(
        &mut self,
        recipe: &BitNetRecipe,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        if weights.tensor_count() == 0 {
            return Ok(());
        }
        self.ensure_model_anchor(weights)?;

        let hidden_dim = recipe.hidden_dim.max(16);
        let ffn_dim = recipe.ffn_dim.max(16);
        let head_d_even = {
            let hd = recipe.head_dim.max(2);
            if hd % 2 == 0 { hd } else { hd - 1 }
        };
        let total_q = recipe.n_heads_q.max(1) * head_d_even;
        let total_kv = recipe.n_heads_kv.max(1) * head_d_even;
        self.act_scratch
            .ensure_max_cols(hidden_dim.max(total_q).max(ffn_dim));

        if weights.contains_tensor("token_embd.weight") {
            let embd = weights.tensor("token_embd.weight")?;
            let total_rows = BitNetRecipe::matrix_row_col_dims(&embd, hidden_dim, 1)?;
            let _ =
                self.get_or_load_table("token_embd.weight", &embd, total_rows, hidden_dim, true)?;
        }
        if weights.contains_tensor("output.weight") {
            let out_w = weights.tensor("output.weight")?;
            let total_rows = BitNetRecipe::matrix_row_col_dims(&out_w, hidden_dim, 1)?;
            let _ =
                self.get_or_load_table("output.weight", &out_w, total_rows, hidden_dim, false)?;
        }

        if weights.contains_tensor("output_norm.weight") {
            let _ =
                self.get_or_decode_norm_into_cache(weights, "output_norm.weight", hidden_dim)?;
        }

        let mut name_buf = String::with_capacity(64);
        for layer in 0..recipe.num_layers {
            for (suffix, dim) in [
                ("attn_norm.weight", hidden_dim),
                ("attn_sub_norm.weight", total_q),
                ("ffn_norm.weight", hidden_dim),
                ("ffn_sub_norm.weight", ffn_dim),
            ] {
                name_buf.clear();
                let _ = write!(&mut name_buf, "blk.{layer}.{suffix}");
                if weights.contains_tensor(&name_buf) {
                    let _ = self.get_or_decode_norm_into_cache(weights, &name_buf, dim)?;
                }
            }

            for (suffix, rows, cols) in [
                ("attn_q.weight", total_q, hidden_dim),
                ("attn_k.weight", total_kv, hidden_dim),
                ("attn_v.weight", total_kv, hidden_dim),
                ("attn_output.weight", hidden_dim, total_q),
                ("ffn_gate.weight", ffn_dim, hidden_dim),
                ("ffn_up.weight", ffn_dim, hidden_dim),
                ("ffn_down.weight", hidden_dim, ffn_dim),
            ] {
                name_buf.clear();
                let _ = write!(&mut name_buf, "blk.{layer}.{suffix}");
                if weights.contains_tensor(&name_buf) {
                    let view = weights.tensor(&name_buf)?;
                    if is_block_repack_quant(view.quant()) {
                        let _ = self.get_or_repack_projection(&name_buf, &view, rows, cols)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Loads `SubLN`/`RMSNorm` weight `name` into `out` via the decoded `f32` cache.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if the norm tensor is missing or invalid.
    pub fn load_norm_slice(
        &mut self,
        weights: &dyn ModelWeights,
        name: &str,
        out: &mut [f32],
    ) -> Result<(), K3Error> {
        if weights.tensor_count() == 0 {
            out.fill(1.0);
            return Ok(());
        }
        if !self.populate_on_miss {
            if let Some(entry) = self.norms.get(name)
                && entry.values.len() == out.len()
            {
                let tensor = weights.tensor(name)?;
                let identity = TensorIdentity::from_view(&tensor);
                if entry.identity == identity {
                    self.stats.norm_hits += 1;
                    out.copy_from_slice(&entry.values);
                    return Ok(());
                }
            }
            return BitNetRecipe::load_norm_weight_slice(weights, name, None, out);
        }
        let cached = self.get_or_decode_norm_into_cache(weights, name, out.len())?;
        out.copy_from_slice(&cached);
        Ok(())
    }

    fn get_or_decode_norm_into_cache(
        &mut self,
        weights: &dyn ModelWeights,
        name: &str,
        dim: usize,
    ) -> Result<Arc<[f32]>, K3Error> {
        let tensor = weights.tensor(name)?;
        let identity = TensorIdentity::from_view(&tensor);
        if let Some(entry) = self.norms.get(name)
            && entry.identity == identity
            && entry.values.len() == dim
        {
            self.stats.norm_hits += 1;
            return Ok(Arc::clone(&entry.values));
        }

        let mut decoded = vec![1.0_f32; dim];
        BitNetRecipe::load_norm_weight_slice(weights, name, None, &mut decoded)?;
        self.stats.norm_decodes += 1;

        let byte_len = dim * size_of::<f32>();
        let values: Arc<[f32]> = Arc::from(decoded);
        if let Some(old) = self.norms.insert(
            name.to_owned(),
            CachedNorm {
                identity,
                values: Arc::clone(&values),
                byte_len,
            },
        ) {
            self.stats.resident_bytes = self.stats.resident_bytes.saturating_sub(old.byte_len);
        }
        self.stats.resident_bytes = self.stats.resident_bytes.saturating_add(byte_len);
        Ok(values)
    }

    fn get_or_repack_projection(
        &mut self,
        name: &str,
        view: &TensorView<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<Option<Arc<RepackedTernaryMatrix>>, K3Error> {
        let identity = TensorIdentity::from_view(view);
        if let Some(entry) = self.projections.get(name)
            && entry.identity == identity
            && entry.matrix.rows() == rows
            && entry.matrix.cols() == cols
        {
            self.stats.weight_hits += 1;
            return Ok(Some(Arc::clone(&entry.matrix)));
        }

        let estimated = estimate_repacked_projection_bytes(view.quant(), rows, cols);
        let existing_bytes = self.projections.get(name).map_or(0, |e| e.byte_len);
        let projected_total = self
            .stats
            .resident_bytes
            .saturating_sub(existing_bytes)
            .saturating_add(estimated);
        if projected_total > self.max_cache_bytes {
            self.stats.budget_skips += 1;
            return Ok(None);
        }

        let (mat, unpack_calls) = RepackedTernaryMatrix::from_tensor_counted(view, rows, cols)?;
        let byte_len = mat.total_resident_bytes();
        let matrix = Arc::new(mat);
        if let Some(old) = self.projections.insert(
            name.to_owned(),
            CachedProjection {
                identity,
                matrix: Arc::clone(&matrix),
                byte_len,
            },
        ) {
            self.stats.resident_bytes = self.stats.resident_bytes.saturating_sub(old.byte_len);
        }
        self.stats.resident_bytes = self.stats.resident_bytes.saturating_add(byte_len);
        self.stats.weight_repacks += 1;
        self.stats.unpack_block_calls += unpack_calls;
        Ok(Some(matrix))
    }

    /// Executes a cached `BitLinear` projection when ternary/Q8 weights fit within cache budget,
    /// falling back cleanly on dense weights or budget skip.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if weights are malformed or dimensions mismatch.
    pub fn dispatch_projection(
        &mut self,
        weights: &dyn ModelWeights,
        name: &str,
        out: &mut [f32],
        act_i8: &[i8],
        act_scale: f32,
    ) -> Result<(), K3Error> {
        let view = weights.tensor(name)?;
        if is_block_repack_quant(view.quant()) {
            if !self.populate_on_miss {
                let identity = TensorIdentity::from_view(&view);
                let Self {
                    projections,
                    act_scratch,
                    stats,
                    ..
                } = self;
                if let Some(entry) = projections.get(name)
                    && entry.identity == identity
                    && entry.matrix.rows() == out.len()
                    && entry.matrix.cols() == act_i8.len()
                {
                    stats.weight_hits += 1;
                    return entry.matrix.bitlinear_gemv_with_scratch(
                        out,
                        act_i8,
                        act_scale,
                        act_scratch,
                    );
                }
                return BitNetRecipe::dispatch_bitlinear(out, &view, act_i8, act_scale);
            }
            if let Some(cached) =
                self.get_or_repack_projection(name, &view, out.len(), act_i8.len())?
            {
                return cached.bitlinear_gemv_with_scratch(
                    out,
                    act_i8,
                    act_scale,
                    &mut self.act_scratch,
                );
            }
        }
        BitNetRecipe::dispatch_bitlinear(out, &view, act_i8, act_scale)
    }

    fn find_cached_table(
        &self,
        key: &str,
        identity: TensorIdentity,
        rows: usize,
        cols: usize,
        is_token_embd: bool,
    ) -> Option<&Arc<CompactLmHeadMatrix>> {
        if is_token_embd {
            if let Some(entry) = &self.token_embd
                && entry.identity == identity
                && entry.matrix.rows() == rows
                && entry.matrix.cols() == cols
            {
                return Some(&entry.matrix);
            }
            return None;
        }
        if let Some(entry) = &self.lm_head
            && entry.key == key
            && entry.identity == identity
            && entry.matrix.rows() == rows
            && entry.matrix.cols() == cols
        {
            return Some(&entry.matrix);
        }
        if let Some(embd) = &self.token_embd
            && embd.identity == identity
            && embd.matrix.rows() == rows
            && embd.matrix.cols() == cols
        {
            return Some(&embd.matrix);
        }
        None
    }

    /// Returns a compact cached handle for `token_embd.weight` or `output.weight` when within budget,
    /// sharing the same `Arc<CompactLmHeadMatrix>` whenever tied embeddings match identity.
    fn get_or_load_table(
        &mut self,
        key: &str,
        view: &TensorView<'_>,
        rows: usize,
        cols: usize,
        is_token_embd: bool,
    ) -> Result<Option<Arc<CompactLmHeadMatrix>>, K3Error> {
        let identity = TensorIdentity::from_view(view);
        if let Some(matrix) = self.find_cached_table(key, identity, rows, cols, is_token_embd) {
            let cached = Arc::clone(matrix);
            self.stats.lm_head_hits += 1;
            return Ok(Some(cached));
        }

        let estimated = estimate_compact_table_bytes(view.quant(), rows, cols);
        let prev_slot_bytes = if is_token_embd {
            self.token_embd.as_ref().map_or(0, |e| e.byte_len)
        } else {
            self.lm_head.as_ref().map_or(0, |e| e.byte_len)
        };
        let projected_total = self
            .stats
            .resident_bytes
            .saturating_sub(prev_slot_bytes)
            .saturating_add(estimated);
        if projected_total > self.max_cache_bytes {
            self.stats.budget_skips += 1;
            return Ok(None);
        }

        let (mat, unpack_calls) = CompactLmHeadMatrix::from_tensor_counted(view, rows, cols)?;
        let byte_len = mat.aligned_bytes();
        let matrix = Arc::new(mat);
        let entry = CachedLmHead {
            key: key.to_owned(),
            identity,
            matrix: Arc::clone(&matrix),
            byte_len,
        };
        if is_token_embd {
            if let Some(old) = self.token_embd.replace(entry) {
                self.stats.resident_bytes = self.stats.resident_bytes.saturating_sub(old.byte_len);
            }
        } else if let Some(old) = self.lm_head.replace(entry) {
            self.stats.resident_bytes = self.stats.resident_bytes.saturating_sub(old.byte_len);
        }
        self.stats.resident_bytes = self.stats.resident_bytes.saturating_add(byte_len);
        self.stats.lm_head_repacks += 1;
        self.stats.unpack_block_calls += unpack_calls;
        Ok(Some(matrix))
    }

    /// Reads one token embedding row into `hidden` using compact cached storage when available.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `token_id` is out of bounds or `token_embd.weight`
    /// is invalid/missing.
    pub fn embed_token(
        &mut self,
        weights: &dyn ModelWeights,
        token_id: u32,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        if weights.tensor_count() == 0 {
            return BitNetRecipe::embed_token_streaming(weights, token_id, hidden);
        }
        let embd = weights.tensor("token_embd.weight")?;
        let row = usize::try_from(token_id).map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "token_embd.weight".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: embd.as_bytes().len(),
            reason: format!("token_id {token_id} exceeds usize"),
        })?;
        let total_rows =
            BitNetRecipe::matrix_row_col_dims(&embd, hidden.len(), row.saturating_add(1))?;
        if !self.populate_on_miss {
            let identity = TensorIdentity::from_view(&embd);
            if let Some(compact) = self.find_cached_table(
                "token_embd.weight",
                identity,
                total_rows,
                hidden.len(),
                true,
            ) {
                let cached = Arc::clone(compact);
                self.stats.lm_head_hits += 1;
                return cached.embed_row(row, hidden);
            }
            return BitNetRecipe::decode_row_f32(&embd, row, hidden);
        }
        if let Some(compact) =
            self.get_or_load_table("token_embd.weight", &embd, total_rows, hidden.len(), true)?
        {
            return compact.embed_row(row, hidden);
        }
        BitNetRecipe::decode_row_f32(&embd, row, hidden)
    }

    /// Projects logits from `hidden` via compact cached LM-head / tied `token_embd.weight` storage,
    /// falling back to streaming when the cache budget cannot hold the table.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on invalid shapes or missing LM-head weights.
    pub fn project_lm_head(
        &mut self,
        weights: &dyn ModelWeights,
        recipe: &BitNetRecipe,
        logits: &mut [f32],
        hidden: &[f32],
        token_seed: Option<u32>,
    ) -> Result<(), K3Error> {
        if logits.is_empty() {
            return Ok(());
        }
        if weights.tensor_count() == 0 {
            return recipe.project_lm_head(logits, hidden, token_seed, weights);
        }

        let (key, head_weight) = if weights.contains_tensor("output.weight") {
            ("output.weight", weights.tensor("output.weight")?)
        } else if weights.contains_tensor("token_embd.weight") {
            ("token_embd.weight", weights.tensor("token_embd.weight")?)
        } else {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "output.weight".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "neither 'output.weight' nor tied 'token_embd.weight' found in weights"
                    .to_owned(),
            });
        };

        if !self.populate_on_miss {
            let total_rows =
                BitNetRecipe::matrix_row_col_dims(&head_weight, hidden.len(), logits.len())?;
            let identity = TensorIdentity::from_view(&head_weight);
            if let Some(compact) = self.find_cached_table(
                key,
                identity,
                total_rows,
                hidden.len(),
                key == "token_embd.weight",
            ) {
                let cached = Arc::clone(compact);
                self.stats.lm_head_hits += 1;
                return cached.project_logits(logits, hidden);
            }
            return recipe.project_lm_head(logits, hidden, token_seed, weights);
        }

        if let Some(compact) =
            self.get_or_load_compact_for_head(key, &head_weight, logits.len(), hidden.len())?
        {
            return compact.project_logits(logits, hidden);
        }
        recipe.project_lm_head(logits, hidden, token_seed, weights)
    }

    fn get_or_load_compact_for_head(
        &mut self,
        key: &str,
        view: &TensorView<'_>,
        min_rows: usize,
        hidden_dim: usize,
    ) -> Result<Option<Arc<CompactLmHeadMatrix>>, K3Error> {
        if !BitNetRecipe::supports_quant_for_cache(view.quant()) {
            return Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: view.as_bytes().len(),
                reason: format!(
                    "unsupported quant type for BitLinear LM head: {:?}",
                    view.quant()
                ),
            });
        }
        let total_rows = BitNetRecipe::matrix_row_col_dims(view, hidden_dim, min_rows)?;
        let is_token_embd = key == "token_embd.weight";
        self.get_or_load_table(key, view, total_rows, hidden_dim, is_token_embd)
    }
}

/// Extension methods on [`BitNetRecipe`] wiring [`BitNetModelCache`] and [`BitNetVocabContract`].
impl BitNetRecipe {
    pub(crate) fn embed_token_streaming(
        weights: &dyn ModelWeights,
        token_id: u32,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        if weights.tensor_count() == 0 {
            #[expect(clippy::cast_precision_loss, reason = "token modulo fits in f32")]
            let base_val = ((token_id % 17) as f32) * 0.1 + 0.5;
            hidden.fill(base_val);
            return Ok(());
        }
        let embd = weights.tensor("token_embd.weight")?;
        let row = usize::try_from(token_id).map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "token_embd.weight".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: embd.as_bytes().len(),
            reason: format!("token_id {token_id} exceeds usize"),
        })?;
        let _total_rows = Self::matrix_row_col_dims(&embd, hidden.len(), row.saturating_add(1))?;
        Self::decode_row_f32(&embd, row, hidden)
    }

    pub(crate) fn embed_token_opt(
        cache: Option<&mut BitNetModelCache>,
        weights: &dyn ModelWeights,
        token_id: u32,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        match cache {
            Some(c) => c.embed_token(weights, token_id, hidden),
            None => Self::embed_token_streaming(weights, token_id, hidden),
        }
    }

    pub(crate) fn load_norm_opt(
        cache: Option<&mut BitNetModelCache>,
        weights: &dyn ModelWeights,
        name: &str,
        out: &mut [f32],
    ) -> Result<(), K3Error> {
        match cache {
            Some(c) => c.load_norm_slice(weights, name, out),
            None => Self::load_norm_weight_slice(weights, name, None, out),
        }
    }

    pub(crate) fn dispatch_proj_opt(
        cache: Option<&mut BitNetModelCache>,
        weights: &dyn ModelWeights,
        name: &str,
        out: &mut [f32],
        act_i8: &[i8],
        act_scale: f32,
    ) -> Result<(), K3Error> {
        if let Some(c) = cache {
            c.dispatch_projection(weights, name, out, act_i8, act_scale)
        } else {
            let view = weights.tensor(name)?;
            Self::dispatch_bitlinear(out, &view, act_i8, act_scale)
        }
    }

    pub(crate) const fn supports_quant_for_cache(quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q8_0
                | QuantType::I2_S
                | QuantType::TQ1_0
                | QuantType::TQ2_0
        )
    }

    /// Parses and validates the [`BitNetVocabContract`] against `gguf`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `self.vocab_size` or stop tokens fail validation.
    pub fn vocab_contract(&self, gguf: &GgufFile<'_>) -> Result<BitNetVocabContract, K3Error> {
        BitNetVocabContract::from_gguf(gguf, self.vocab_size)
    }

    /// Pre-decodes norms and repacks all ternary/Q8 projection matrices into `cache` at load time.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if required tensors are missing or invalid.
    pub fn preload_cache(
        &self,
        weights: &dyn ModelWeights,
        cache: &mut BitNetModelCache,
    ) -> Result<(), K3Error> {
        cache.preload_model(self, weights)
    }

    /// Executes `prefill_step` using the caller-provided [`BitNetModelCache`].
    ///
    /// # Errors
    ///
    /// Propagates [`K3Error`] on empty prompt, KV geometry mismatch, or invalid tensors.
    pub fn prefill_step_with_cache(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        cache: &mut BitNetModelCache,
    ) -> Result<(), K3Error> {
        self.prefill_step_with_opt_cache(ctx, weights, Some(cache))
    }

    pub(crate) fn prefill_step_with_opt_cache(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        mut cache: Option<&mut BitNetModelCache>,
    ) -> Result<(), K3Error> {
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
        let seq_len = start_pos + prompt_len;

        let last_hidden = self.run_tokens_with_cache(
            &prompt_tokens,
            start_pos,
            seq_len,
            ctx,
            weights,
            cache.as_deref_mut(),
        )?;
        ctx.advance_position(prompt_len)?;
        match cache {
            Some(c) => c.project_lm_head(weights, self, ctx.logits_mut(), &last_hidden, None),
            None => self.project_lm_head(ctx.logits_mut(), &last_hidden, None, weights),
        }
    }

    /// Executes `decode_step` using the caller-provided [`BitNetModelCache`].
    ///
    /// # Errors
    ///
    /// Propagates [`K3Error`] on missing staged decode token, KV mismatch, or invalid tensors.
    pub fn decode_step_with_cache(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        cache: &mut BitNetModelCache,
    ) -> Result<(), K3Error> {
        self.decode_step_with_opt_cache(ctx, weights, Some(cache))
    }

    pub(crate) fn decode_step_with_opt_cache(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        mut cache: Option<&mut BitNetModelCache>,
    ) -> Result<(), K3Error> {
        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;
        if let Some(c) = cache.as_deref_mut() {
            c.ensure_model_anchor(weights)?;
        }
        let pos = ctx.position();
        let seq_len = pos + 1;
        let hidden =
            self.run_tokens_with_cache(&[token], pos, seq_len, ctx, weights, cache.as_deref_mut())?;
        ctx.advance_position(1)?;
        match cache {
            Some(c) => c.project_lm_head(weights, self, ctx.logits_mut(), &hidden, Some(token)),
            None => self.project_lm_head(ctx.logits_mut(), &hidden, Some(token), weights),
        }
    }
}

const fn is_block_repack_quant(quant: QuantType) -> bool {
    matches!(
        quant,
        QuantType::TQ1_0 | QuantType::TQ2_0 | QuantType::I2_S | QuantType::Q8_0
    )
}

fn estimate_repacked_projection_bytes(quant: QuantType, rows: usize, cols: usize) -> usize {
    let block_elems = BitNetRecipe::quant_block_layout(quant).map_or(256, |(e, _, _)| e);
    let n_tiles = rows.div_ceil(TILE_N);
    let blocks_per_row = cols.div_ceil(block_elems);
    let block_k_chunks = block_elems.div_ceil(TILE_K);
    let tile_bytes = n_tiles * blocks_per_row * block_k_chunks * TILE_BYTES;
    let scale_bytes = n_tiles * blocks_per_row * TILE_N * size_of::<f32>();
    tile_bytes + scale_bytes
}

fn estimate_compact_table_bytes(quant: QuantType, rows: usize, cols: usize) -> usize {
    match quant {
        QuantType::F16 | QuantType::BF16 => rows * cols.div_ceil(64) * TILE_BYTES,
        QuantType::F32 => rows * cols.div_ceil(32) * TILE_BYTES,
        QuantType::Q8_0 | QuantType::TQ1_0 | QuantType::TQ2_0 | QuantType::I2_S => {
            estimate_repacked_projection_bytes(quant, rows, cols)
        }
        _ => rows * cols * 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::{GgufFile, QuantType};
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct TestWeights(BTreeMap<String, (QuantType, Vec<usize>, Vec<u8>)>);

    impl TestWeights {
        fn insert(&mut self, name: &str, quant: QuantType, shape: Vec<usize>, data: Vec<u8>) {
            self.0.insert(name.to_owned(), (quant, shape, data));
        }
    }

    impl ModelWeights for TestWeights {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            let (key, (q, s, d)) =
                self.0
                    .get_key_value(name)
                    .ok_or_else(|| K3Error::TensorBoundsError {
                        tensor_name: name.to_owned(),
                        offset: 0,
                        byte_len: 0,
                        buffer_len: 0,
                        reason: "missing test tensor".to_owned(),
                    })?;
            TensorView::from_bytes(key, *q, s, 0, d)
        }

        fn tensor_count(&self) -> usize {
            self.0.len()
        }

        fn total_weight_bytes(&self) -> usize {
            self.0.values().map(|(_, _, b)| b.len()).sum()
        }
    }

    fn push_gguf_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    enum KvItem<'a> {
        Str(&'a str),
        U32(u32),
        StrArray(&'a [&'a str]),
    }

    fn build_vocab_gguf(arch: &str, kvs: &[(&str, KvItem<'_>)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3_u32.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&((kvs.len() + 1) as u64).to_le_bytes());

        push_gguf_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8_u32.to_le_bytes()); // string
        push_gguf_str(&mut buf, arch);

        for (key, val) in kvs {
            push_gguf_str(&mut buf, key);
            match val {
                KvItem::Str(s) => {
                    buf.extend_from_slice(&8_u32.to_le_bytes());
                    push_gguf_str(&mut buf, s);
                }
                KvItem::U32(v) => {
                    buf.extend_from_slice(&4_u32.to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                KvItem::StrArray(items) => {
                    buf.extend_from_slice(&9_u32.to_le_bytes()); // array
                    buf.extend_from_slice(&8_u32.to_le_bytes()); // string element type
                    buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for s in *items {
                        push_gguf_str(&mut buf, s);
                    }
                }
            }
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf
    }

    fn f32_to_f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
        let raw_exp = i32::try_from((bits >> 23) & 0xff).unwrap_or(0);
        let exp = raw_exp - 127 + 15;
        let mant = u16::try_from((bits >> 13) & 0x3ff).unwrap_or(0);
        if exp <= 0 {
            sign
        } else if exp >= 31 {
            sign | 0x7c00
        } else {
            let exp_u16 = u16::try_from(exp).unwrap_or(0);
            sign | (exp_u16 << 10) | mant
        }
    }

    fn make_tq1_0_matrix(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
        let bpr = cols / 256;
        let mut out = Vec::with_capacity(rows * bpr * 54);
        for r in 0..rows {
            for b in 0..bpr {
                let rem = u8::try_from((r + b) % 4).unwrap_or(0);
                let scale = f32_to_f16_bits(0.125 + f32::from(rem) * 0.0625);
                out.extend_from_slice(&scale.to_le_bytes());
                for p in 0..51 {
                    let mut val = 0_u8;
                    let mut pow = 1_u8;
                    for j in 0..5 {
                        let code = u8::try_from((r * 7 + b * 3 + p + j + usize::from(seed)) % 3)
                            .unwrap_or(0);
                        val += code * pow;
                        pow = pow.saturating_mul(3);
                    }
                    out.push(val);
                }
                out.push(u8::try_from((r + b + usize::from(seed)) % 3).unwrap_or(0));
            }
        }
        out
    }

    fn make_tq2_0_matrix(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
        let bpr = cols / 256;
        let mut out = Vec::with_capacity(rows * bpr * 66);
        for r in 0..rows {
            for b in 0..bpr {
                let rem = u8::try_from((r + b) % 3).unwrap_or(0);
                let scale = f32_to_f16_bits(0.25 + f32::from(rem) * 0.125);
                out.extend_from_slice(&scale.to_le_bytes());
                for p in 0..64 {
                    let mut byte = 0_u8;
                    for k in 0..4 {
                        let code =
                            u8::try_from((r * 5 + b + p + k + usize::from(seed)) % 3).unwrap_or(0);
                        byte |= (code & 0x03) << (k * 2);
                    }
                    out.push(byte);
                }
            }
        }
        out
    }

    fn make_i2s_matrix(rows: usize, cols: usize, seed: u8, scale: f32) -> Vec<u8> {
        let bpr = cols / 128;
        let mut out = Vec::with_capacity(rows * bpr * 32 + 32);
        for r in 0..rows {
            for b in 0..bpr {
                for g in 0..32 {
                    let mut packed = 0_u8;
                    for slot in 0..4 {
                        let code = u8::try_from((r * 3 + b * 5 + g + slot + usize::from(seed)) % 3)
                            .unwrap_or(0);
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

    fn make_q8_0_matrix(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
        let bpr = cols / 32;
        let mut out = Vec::with_capacity(rows * bpr * 34);
        for r in 0..rows {
            for b in 0..bpr {
                let scale = f32_to_f16_bits(
                    0.0625 + f32::from(u8::try_from((r + b) % 4).unwrap_or(0)) * 0.03125,
                );
                out.extend_from_slice(&scale.to_le_bytes());
                for j in 0..32 {
                    let v = i16::try_from((r * 7 + b * 3 + j + usize::from(seed)) % 21)
                        .unwrap_or(0)
                        - 10;
                    out.extend_from_slice(&[i8::try_from(v).unwrap_or(0).to_le_bytes()[0]]);
                }
            }
        }
        out
    }

    fn oracle_block_dot_i32(
        view: &TensorView<'_>,
        row: usize,
        cols: usize,
        act_i8: &[i8],
    ) -> Vec<(i32, f32)> {
        let raw = view.as_bytes();
        let (belem, bbytes, _) = BitNetRecipe::quant_block_layout(view.quant()).expect("layout");
        let bpr = cols / belem;
        let rows_in_tensor = view.shape()[1];
        let i2s_scale = if view.quant() == QuantType::I2_S {
            let p = rows_in_tensor * bpr * bbytes;
            f32::from_le_bytes([raw[p], raw[p + 1], raw[p + 2], raw[p + 3]])
        } else {
            0.0
        };

        let mut res = Vec::with_capacity(bpr);
        for b in 0..bpr {
            let off = (row * bpr + b) * bbytes;
            let blk = &raw[off..off + bbytes];
            let mut w = vec![0_i8; belem];
            let scale = unpack_source_block(view.quant(), blk, i2s_scale, &mut w).expect("unpack");
            let act_sub = &act_i8[b * belem..(b + 1) * belem];
            let dot: i32 = w
                .iter()
                .zip(act_sub)
                .map(|(&wi, &ai)| i32::from(wi) * i32::from(ai))
                .sum();
            res.push((dot, scale));
        }
        res
    }

    #[test]
    fn test_repacked_ternary_bit_exact_i32_and_f32_parity_across_formats() {
        let cols = 256;
        let rows = 9; // exercises partial trailing tile (8 + 1)
        let act_i8: Vec<i8> = (0..cols)
            .map(|i| {
                let v = i16::try_from((i * 19 + 7) % 180).unwrap_or(0) - 90;
                i8::try_from(v).unwrap_or(0)
            })
            .collect();
        let act_scale = 0.03125_f32;

        for (quant, data) in [
            (QuantType::TQ1_0, make_tq1_0_matrix(rows, cols, 3)),
            (QuantType::TQ2_0, make_tq2_0_matrix(rows, cols, 5)),
            (QuantType::I2_S, make_i2s_matrix(rows, cols, 7, 0.375)),
        ] {
            let shape = [cols, rows];
            let view =
                TensorView::from_bytes("proj.weight", quant, &shape, 0, &data).expect("view");
            let (mat, unpack_calls) = RepackedTernaryMatrix::from_tensor_counted(&view, rows, cols)
                .expect("repack succeeds");
            assert_eq!(
                unpack_calls,
                u64::try_from(rows * (cols / mat.block_elems())).unwrap()
            );
            assert!(
                mat.is_128_byte_aligned(),
                "128B alignment required for {quant:?}"
            );

            for r in [0, 1, rows - 1] {
                let actual_dots = mat.dot_blocks_i32(r, &act_i8).expect("dot_blocks_i32");
                let oracle_dots = oracle_block_dot_i32(&view, r, cols, &act_i8);
                assert_eq!(actual_dots.len(), oracle_dots.len());
                for ((ad, as_), (od, os_)) in actual_dots.iter().zip(&oracle_dots) {
                    assert_eq!(ad, od, "i32 dot mismatch for {quant:?} row {r}");
                    assert_eq!(
                        as_.to_bits(),
                        os_.to_bits(),
                        "scale mismatch for {quant:?} row {r}"
                    );
                }
            }

            let mut cached_out = vec![0.0_f32; rows];
            mat.bitlinear_gemv(&mut cached_out, &act_i8, act_scale)
                .expect("gemv");

            let mut baseline_out = vec![0.0_f32; rows];
            BitNetRecipe::dispatch_bitlinear(&mut baseline_out, &view, &act_i8, act_scale)
                .expect("baseline dispatch");

            for r in 0..rows {
                assert_eq!(
                    cached_out[r].to_bits(),
                    baseline_out[r].to_bits(),
                    "bit-exact f32 output mismatch for {quant:?} row {r}"
                );
            }
        }
    }

    #[test]
    fn test_compact_lm_head_stays_compact_for_f16_and_q8() {
        let rows = 16;
        let cols = 64;
        let mut f16_bytes = Vec::with_capacity(rows * cols * 2);
        for i in 0..rows * cols {
            let rem = f32::from(u8::try_from(i % 13).unwrap_or(0));
            f16_bytes.extend_from_slice(&f32_to_f16_bits((rem - 6.0) * 0.25).to_le_bytes());
        }
        let shape = [cols, rows];
        let view =
            TensorView::from_bytes("token_embd.weight", QuantType::F16, &shape, 0, &f16_bytes)
                .expect("view");
        let (compact, calls) =
            CompactLmHeadMatrix::from_tensor_counted(&view, rows, cols).expect("compact F16");
        assert_eq!(calls, 0);
        assert!(compact.is_128_byte_aligned());
        assert!(compact.is_compact_half());
        assert!(compact.is_compact_storage());
        assert!(
            compact.aligned_bytes() < rows * cols * 4,
            "compact F16 table must stay strictly smaller than f32 expansion"
        );

        let hidden: Vec<f32> = (0..cols)
            .map(|i| f32::from(u16::try_from(i).unwrap_or(0)) * 0.05 - 0.5)
            .collect();
        let mut logits = vec![0.0_f32; rows];
        compact
            .project_logits(&mut logits, &hidden)
            .expect("project logits");

        let mut row_tmp = vec![0.0_f32; cols];
        for (r, &actual) in logits.iter().enumerate() {
            BitNetRecipe::decode_row_f32(&view, r, &mut row_tmp).expect("decode row");
            let expected: f32 = row_tmp.iter().zip(&hidden).map(|(w, h)| w * h).sum();
            assert!(
                (actual - expected).abs() < 1e-4,
                "row {r}: {actual} vs {expected}"
            );
        }
    }

    fn build_one_layer_cache_fixture() -> (BitNetRecipe, TestWeights) {
        let mut recipe = BitNetRecipe::new_b158();
        recipe.num_layers = 1;
        recipe.hidden_dim = 32;
        recipe.ffn_dim = 32;
        recipe.n_heads_q = 2;
        recipe.n_heads_kv = 1;
        recipe.head_dim = 16;
        recipe.vocab_size = 4;

        let mut weights = TestWeights::default();
        let mut embd = Vec::with_capacity(4 * 32 * 2);
        for i in 0..4 * 32 {
            let rem = f32::from(u8::try_from(i % 11).unwrap_or(0));
            embd.extend_from_slice(&f32_to_f16_bits(rem * 0.05 - 0.2).to_le_bytes());
        }
        weights.insert("token_embd.weight", QuantType::F16, vec![32, 4], embd);

        let f32_norm: Vec<u8> = std::iter::repeat_n(1.0_f32.to_le_bytes(), 32)
            .flatten()
            .collect();
        for name in [
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_sub_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_sub_norm.weight",
        ] {
            weights.insert(name, QuantType::F32, vec![32], f32_norm.clone());
        }
        let specs = [
            (
                "blk.0.attn_q.weight",
                vec![32, 32],
                make_q8_0_matrix(32, 32, 1),
            ),
            (
                "blk.0.attn_k.weight",
                vec![32, 16],
                make_q8_0_matrix(16, 32, 2),
            ),
            (
                "blk.0.attn_v.weight",
                vec![32, 16],
                make_q8_0_matrix(16, 32, 3),
            ),
            (
                "blk.0.attn_output.weight",
                vec![32, 32],
                make_q8_0_matrix(32, 32, 4),
            ),
            (
                "blk.0.ffn_gate.weight",
                vec![32, 32],
                make_q8_0_matrix(32, 32, 5),
            ),
            (
                "blk.0.ffn_up.weight",
                vec![32, 32],
                make_q8_0_matrix(32, 32, 6),
            ),
            (
                "blk.0.ffn_down.weight",
                vec![32, 32],
                make_q8_0_matrix(32, 32, 7),
            ),
        ];
        for (name, shape, data) in specs {
            weights.insert(name, QuantType::Q8_0, shape, data);
        }
        (recipe, weights)
    }

    #[test]
    fn test_bitnet_model_cache_hits_and_zero_reunpack_on_repeat_inference() {
        let (recipe, weights) = build_one_layer_cache_fixture();
        let mut cache = BitNetModelCache::new();
        recipe
            .preload_cache(&weights, &mut cache)
            .expect("preload succeeds");
        let after_load = cache.stats();
        assert_eq!(after_load.norm_decodes, 5);
        assert_eq!(after_load.weight_repacks, 7);
        assert_eq!(after_load.lm_head_repacks, 1);
        assert!(after_load.unpack_block_calls > 0);

        let geom = jero_core::KvGeometry::new(1, 1, 16, 4, None).expect("geom");
        let mut logits = [0.0_f32; 4];
        let mut kv_buf = vec![0.0_f32; geom.required_f32_len().expect("kv len")];
        let mut ctx = ExecutionContext::new(&[0], 4, &mut logits, &mut kv_buf)
            .expect("ctx")
            .with_kv_geometry(geom)
            .expect("kv view");
        recipe
            .prefill_step_with_cache(&mut ctx, &weights, &mut cache)
            .expect("prefill succeeds");

        ctx.prepare_decode(1).expect("prepare decode");
        recipe
            .decode_step_with_cache(&mut ctx, &weights, &mut cache)
            .expect("decode succeeds");

        let after_infer = cache.stats();
        assert_eq!(after_infer.norm_decodes, after_load.norm_decodes);
        assert_eq!(after_infer.weight_repacks, after_load.weight_repacks);
        assert_eq!(after_infer.lm_head_repacks, after_load.lm_head_repacks);
        assert_eq!(
            after_infer.unpack_block_calls, after_load.unpack_block_calls,
            "inference must not re-unpack weight blocks"
        );
        assert!(after_infer.norm_hits >= 10);
        assert!(after_infer.weight_hits >= 14);
        assert!(after_infer.lm_head_hits >= 4);

        // Compact 2-row ternary dispatch caching across TQ1_0, TQ2_0, and I2_S.
        let mut ternary_weights = TestWeights::default();
        ternary_weights.insert(
            "tq1",
            QuantType::TQ1_0,
            vec![256, 2],
            make_tq1_0_matrix(2, 256, 1),
        );
        ternary_weights.insert(
            "tq2",
            QuantType::TQ2_0,
            vec![256, 2],
            make_tq2_0_matrix(2, 256, 2),
        );
        ternary_weights.insert(
            "i2s",
            QuantType::I2_S,
            vec![128, 2],
            make_i2s_matrix(2, 128, 3, 0.25),
        );
        let mut tcache = BitNetModelCache::new();
        let act_256 = [3_i8; 256];
        for (name, cols) in [("tq1", 256), ("tq2", 256), ("i2s", 128)] {
            let mut out = [0.0_f32; 2];
            tcache
                .dispatch_projection(&ternary_weights, name, &mut out, &act_256[..cols], 0.125)
                .expect("initial ternary repack dispatch");
        }
        let mid = tcache.stats();
        assert_eq!(mid.weight_repacks, 3);
        for (name, cols) in [("tq1", 256), ("tq2", 256), ("i2s", 128)] {
            let mut out = [0.0_f32; 2];
            tcache
                .dispatch_projection(&ternary_weights, name, &mut out, &act_256[..cols], 0.125)
                .expect("cached ternary dispatch");
        }
        let fin = tcache.stats();
        assert_eq!(fin.weight_repacks, 3);
        assert_eq!(fin.weight_hits, 3);
        assert_eq!(fin.unpack_block_calls, mid.unpack_block_calls);
    }

    #[test]
    fn test_bitnet_vocab_contract_stop_bounds_and_mismatch_fail_closed() {
        let mut tokens = vec!["<pad>"; 16];
        tokens[1] = "<|begin_of_text|>";
        tokens[2] = "<|end_of_text|>";
        tokens[9] = "<|eot_id|>";
        let raw = build_vocab_gguf(
            "bitnet-b1.58",
            &[
                ("bitnet-b1.58.vocab_size", KvItem::U32(16)),
                ("tokenizer.ggml.model", KvItem::Str("gpt2")),
                ("tokenizer.ggml.tokens", KvItem::StrArray(&tokens)),
                ("tokenizer.ggml.eos_token_id", KvItem::U32(2)),
                ("tokenizer.ggml.eot_token_id", KvItem::U32(9)),
            ],
        );
        let gguf = GgufFile::parse(&raw).expect("parse gguf");
        let contract = BitNetVocabContract::from_gguf(&gguf, 16).expect("valid contract");
        assert_eq!(contract.eot_piece_token_id, Some(9));
        assert_eq!(contract.end_of_text_token_id, Some(2));
        assert!(contract.validate_stop_token(9).expect("valid stop"));
        assert!(!contract.validate_stop_token(5).expect("valid normal token"));
        assert!(
            contract.validate_stop_token(16).is_err(),
            "OOB token must fail closed"
        );
        assert!(
            contract.expect_llama3_128k_stops().is_err(),
            "16 != 128_256"
        );

        let bad_raw = build_vocab_gguf(
            "bitnet-b1.58",
            &[
                ("bitnet-b1.58.vocab_size", KvItem::U32(16)),
                ("tokenizer.ggml.eos_token_id", KvItem::U32(99)),
            ],
        );
        let bad = GgufFile::parse(&bad_raw).expect("parse");
        assert!(
            BitNetVocabContract::from_gguf(&bad, 16).is_err(),
            "EOS OOB must fail closed"
        );
    }
}
