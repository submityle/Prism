//! ASTC (Adaptive Scalable Texture Compression) decode.
//!
//! ASTC packs a configurable footprint (4x4 up to 12x12) into a fixed 128-bit
//! block, so unlike the BC family it needs a block-mode parse before the colour
//! data can be read. This module is built up in GPU-validatable milestones:
//!
//! 1. **Void-extent LDR** -- constant-colour blocks (no Integer Sequence
//!    Encoding). Implemented here and parity-checked on hardware.
//! 2. BISE (trit/quint Integer Sequence Encoding) -- *follow-up*.
//! 3. Single-partition LDR 4x4 (block-mode + weights + endpoints) -- *follow-up*.
//! 4. Multi-partition / dual-plane, then HDR, then larger footprints.
//!
//! Decode is pure integer/`f32` arithmetic (no AI/ML path); every landed stage
//! is proven bit-for-bit against the platform ASTC hardware decoder via the
//! `prism_render_material_gpu` block oracle.
//!
//! # References
//! * Khronos Data Format Specification 1.3, "ASTC Compressed Texture Formats".
//! * Vulkan `VK_FORMAT_ASTC_4x4_UNORM_BLOCK`.

mod bise;
mod block_mode;
mod block_reader;
mod cem;
mod color_unquant;
mod encode;
mod endpoints;
mod hdr_endpoints;
mod infill;
mod multi_partition;
mod multi_partition_hdr;
mod partition;
mod quant_mode;
mod single_partition;
mod trit_quint;
mod void_extent;
mod weight_unquant;
mod weights;

pub use encode::encode_astc_single_partition_4x4_ldr;
pub use void_extent::{decode_astc_void_extent_hdr, decode_astc_void_extent_ldr};
pub use weights::{decode_astc_4x4_weights, decode_astc_4x4_weights_ise};

/// Errors returned by the ASTC decoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstcError {
    /// A reserved or malformed encoding that no profile defines.
    Reserved,
    /// An HDR (FP16) block was passed to an LDR-only decode path.
    UnsupportedHdr,
    /// A non-void-extent block was passed to the void-extent-only 4x4 decoder;
    /// removed once the general single-partition path lands.
    UnsupportedBlockMode,
    /// Reserved for a BISE integer sequence whose quantisation range is not
    /// yet decodable. Trit and quint ranges now decode (see `trit_quint`), so
    /// this is currently unused on the decode path and kept for forward
    /// compatibility with future reserved encodings.
    UnsupportedIse,
}

/// Decode a single 4x4 ASTC **LDR** block to sixteen `RGBA8` texels.
///
/// Dispatches void-extent (constant-colour) blocks and single-partition LDR
/// weighted blocks: all ten LDR Colour Endpoint Modes, any colour
/// quantisation, any weight grid (resampled to the 4x4 footprint by the
/// Khronos bilinear infill), and both single- and dual-plane weights. Blocks
/// outside that subset (multi-partition and the six HDR CEMs) return an
/// [`AstcError`] rather than approximate pixels, until their own milestones
/// land.
///
/// # Errors
/// Propagates [`AstcError`] from the selected decode path, or
/// [`AstcError::UnsupportedBlockMode`] for block types not yet implemented.
pub fn decode_astc_4x4_ldr(block: &[u8; 16]) -> Result<[[u8; 4]; 16], AstcError> {
    if void_extent::is_void_extent(block) {
        decode_astc_void_extent_ldr(block)
    } else {
        // Partition count is the 2-bit field at block bits [11, 13); 0 => a
        // single partition, 1..=3 => 2..=4 partitions. Route each to its
        // dedicated decoder.
        let partition_count = ((u32::from(block[1]) >> 3) & 0x3) + 1;
        if partition_count == 1 {
            single_partition::decode_single_partition_4x4_ldr(block)
        } else {
            multi_partition::decode_multi_partition_4x4_ldr(block)
        }
    }
}

/// Decode a single 4x4 ASTC **HDR** block to sixteen `RGBA` texels in `f32`.
///
/// Dispatches HDR void-extent (constant-colour FP16) blocks and single-
/// partition HDR weighted blocks: the six HDR Colour Endpoint Modes (2, 3, 7,
/// 11, 14, 15), any colour quantisation, any weight grid resampled to the 4x4
/// footprint by the Khronos bilinear infill, and both single- and dual-plane
/// weights. RGB lanes are logarithmic (LNS); alpha is logarithmic except for
/// CEM 14 (linear LDR alpha).
///
/// Multi-partition HDR blocks (2/3/4 partitions) decode under the HDR profile
/// with any mix of partition Colour Endpoint Modes -- all-HDR, all-LDR, and
/// mixed LDR/HDR -- the LDR partitions being widened into the 16-bit linear
/// HDR domain (`x257`) exactly as the reference HDR-profile decode does.
///
/// # Errors
/// Propagates [`AstcError`] from the selected decode path, or
/// [`AstcError::UnsupportedBlockMode`] for block types not yet implemented.
pub fn decode_astc_4x4_hdr(block: &[u8; 16]) -> Result<[[f32; 4]; 16], AstcError> {
    if void_extent::is_void_extent(block) {
        decode_astc_void_extent_hdr(block)
    } else {
        // Partition count is the 2-bit field at block bits [11, 13); 0 => a
        // single partition, 1..=3 => 2..=4 partitions (any LDR/HDR mix).
        let partition_count = ((u32::from(block[1]) >> 3) & 0x3) + 1;
        if partition_count == 1 {
            hdr_endpoints::decode_single_partition_4x4_hdr(block)
        } else {
            multi_partition_hdr::decode_multi_partition_4x4_hdr(block)
        }
    }
}

/// Decode a single ASTC **LDR** block for an arbitrary 2D footprint `bx` x `by`
/// (4..=12 per axis) to `bx * by` `RGBA8` texels.
///
/// Returns a fixed 144-entry array (the largest 12x12 footprint) together with
/// the number of valid texels (`bx * by`); callers read `out[..count]` in
/// row-major order (`texel = y * bx + x`). This avoids a heap allocation on the
/// decode hot path.
///
/// Dispatches void-extent (constant-colour, replicated to the footprint),
/// single-partition and multi-partition (2/3/4) LDR weighted blocks across all
/// footprints: all ten LDR CEMs, any colour quantisation, any legal weight
/// grid, single- or dual-plane. Unsupported block types return an
/// [`AstcError`] rather than approximate pixels.
///
/// # Errors
/// Returns [`AstcError::Reserved`] for an out-of-range footprint, or propagates
/// [`AstcError`] from the selected decode path.
pub fn decode_astc_ldr(
    block: &[u8; 16],
    bx: u32,
    by: u32,
) -> Result<([[u8; 4]; 144], usize), AstcError> {
    let texels = (bx as usize) * (by as usize);
    if texels == 0 || texels > 144 {
        return Err(AstcError::Reserved);
    }
    let mut out = [[0u8; 4]; 144];
    if void_extent::is_void_extent(block) {
        // Void-extent blocks are a single constant colour; replicate it across
        // the whole footprint.
        let c = decode_astc_void_extent_ldr(block)?[0];
        for slot in out[..texels].iter_mut() {
            *slot = c;
        }
    } else {
        let partition_count = ((u32::from(block[1]) >> 3) & 0x3) + 1;
        if partition_count == 1 {
            single_partition::decode_single_partition_ldr(block, bx, by, &mut out[..texels])?;
        } else {
            multi_partition::decode_multi_partition_ldr(block, bx, by, &mut out[..texels])?;
        }
    }
    Ok((out, texels))
}

/// Decode a single ASTC **HDR** block for an arbitrary 2D footprint `bx` x `by`
/// (4..=12 per axis) to `bx * by` `RGBA` `f32` texels.
///
/// Returns a fixed 144-entry array (the largest 12x12 footprint) together with
/// the number of valid texels (`bx * by`); callers read `out[..count]` in
/// row-major order (`texel = y * bx + x`). This avoids a heap allocation on the
/// decode hot path.
///
/// Dispatches HDR void-extent (constant-colour FP16, replicated to the
/// footprint), single-partition and multi-partition (2/3/4) HDR weighted
/// blocks across all footprints: the six HDR CEMs (with any LDR/HDR partition
/// mix), any colour quantisation, any legal weight grid, single- or
/// dual-plane. Unsupported block types return an [`AstcError`] rather than
/// approximate pixels.
///
/// # Errors
/// Returns [`AstcError::Reserved`] for an out-of-range footprint, or propagates
/// [`AstcError`] from the selected decode path.
pub fn decode_astc_hdr(
    block: &[u8; 16],
    bx: u32,
    by: u32,
) -> Result<([[f32; 4]; 144], usize), AstcError> {
    let texels = (bx as usize) * (by as usize);
    if texels == 0 || texels > 144 {
        return Err(AstcError::Reserved);
    }
    let mut out = [[0.0f32; 4]; 144];
    if void_extent::is_void_extent(block) {
        // Void-extent blocks are a single constant colour; replicate it across
        // the whole footprint.
        let c = decode_astc_void_extent_hdr(block)?[0];
        for slot in out[..texels].iter_mut() {
            *slot = c;
        }
    } else {
        let partition_count = ((u32::from(block[1]) >> 3) & 0x3) + 1;
        if partition_count == 1 {
            hdr_endpoints::decode_single_partition_hdr(block, bx, by, &mut out[..texels])?;
        } else {
            multi_partition_hdr::decode_multi_partition_hdr(block, bx, by, &mut out[..texels])?;
        }
    }
    Ok((out, texels))
}
