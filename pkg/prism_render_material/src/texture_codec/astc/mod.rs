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
mod endpoints;
mod quant_mode;
mod single_partition;
mod trit_quint;
mod void_extent;
mod weight_unquant;
mod weights;

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
/// Dispatches void-extent (constant-colour) blocks and single-partition,
/// single-plane 4x4-grid CEM 8 (LDR direct RGB, QUANT_256) weighted blocks.
/// Blocks outside that subset (multi-partition, dual-plane, non-4x4 grids,
/// other CEMs or non-identity colour quantisation) return an [`AstcError`]
/// rather than approximate pixels, until their own milestones land.
///
/// # Errors
/// Propagates [`AstcError`] from the selected decode path, or
/// [`AstcError::UnsupportedBlockMode`] for block types not yet implemented.
pub fn decode_astc_4x4_ldr(block: &[u8; 16]) -> Result<[[u8; 4]; 16], AstcError> {
    if void_extent::is_void_extent(block) {
        decode_astc_void_extent_ldr(block)
    } else {
        single_partition::decode_single_partition_4x4_ldr(block)
    }
}
