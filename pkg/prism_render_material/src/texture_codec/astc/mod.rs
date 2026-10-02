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
mod block_reader;
mod void_extent;

pub use void_extent::{decode_astc_void_extent_hdr, decode_astc_void_extent_ldr};

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
    /// A BISE integer sequence used a trit or quint quantisation range, which
    /// is parsed but not yet GPU-validated and so not yet emitted.
    UnsupportedIse,
}

/// Decode a single 4x4 ASTC **LDR** block to sixteen `RGBA8` texels.
///
/// Currently dispatches void-extent (constant-colour) blocks; general
/// single-partition weighted blocks return [`AstcError::UnsupportedBlockMode`]
/// until that milestone lands.
///
/// # Errors
/// Propagates [`AstcError`] from the selected decode path, or
/// [`AstcError::UnsupportedBlockMode`] for block types not yet implemented.
pub fn decode_astc_4x4_ldr(block: &[u8; 16]) -> Result<[[u8; 4]; 16], AstcError> {
    if void_extent::is_void_extent(block) {
        decode_astc_void_extent_ldr(block)
    } else {
        Err(AstcError::UnsupportedBlockMode)
    }
}
