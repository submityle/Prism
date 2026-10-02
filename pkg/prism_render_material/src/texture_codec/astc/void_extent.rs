//! ASTC void-extent (constant-colour) block decode.
//!
//! A void-extent block encodes a single colour that fills the whole footprint,
//! plus an optional rectangle over which that colour is guaranteed constant
//! (a sampler optimisation hint, irrelevant to a single-block decode). It is
//! the only ASTC block type that carries no Integer-Sequence-Encoded weights,
//! which makes it the natural first milestone: pure field extraction, no BISE.
//!
//! Layout (Khronos Data Format Specification 1.3, section "Void-extent
//! Blocks"), all fields LSB-first in the 128-bit little-endian block:
//! * bits `[0..9)`  -- void-extent signature, must be `0b1_1111_1100`.
//! * bit  `9`       -- dynamic-range flag: `0` = LDR (UNORM16), `1` = HDR.
//! * bits `[10..12)`-- reserved, must be `0b11`.
//! * bits `[12..64)`-- four 13-bit void-extent coordinates (hint only).
//! * bits `[64..80)`  -- red,   16-bit UNORM (LDR) / FP16 (HDR).
//! * bits `[80..96)`  -- green.
//! * bits `[96..112)` -- blue.
//! * bits `[112..128)`-- alpha.

use super::block_reader::read_bits;
use super::AstcError;

/// The 9-bit signature that marks a void-extent block.
pub(super) const VOID_EXTENT_SIGNATURE: u32 = 0b1_1111_1100;

/// Returns `true` if `block`'s low 9 bits are the void-extent signature.
#[must_use]
pub(super) fn is_void_extent(block: &[u8; 16]) -> bool {
    read_bits(block, 0, 9) == VOID_EXTENT_SIGNATURE
}

/// Convert a 16-bit UNORM channel to 8-bit using the same rounding the GPU
/// oracle applies: `round(v / 65535 * 255)`.
#[inline]
fn unorm16_to_unorm8(v: u32) -> u8 {
    let f = (v as f32) / 65535.0 * 255.0 + 0.5;
    // `f` is always in [0, 255.5]; truncation of a non-negative value rounds.
    f as u8
}

/// Decode a void-extent LDR block to sixteen identical `RGBA8` texels.
///
/// # Errors
/// * [`AstcError::Reserved`] if the signature bits are not the void-extent
///   pattern.
/// * [`AstcError::UnsupportedHdr`] if the dynamic-range flag selects HDR
///   (FP16 void extents are a later milestone).
pub fn decode_astc_void_extent_ldr(block: &[u8; 16]) -> Result<[[u8; 4]; 16], AstcError> {
    if !is_void_extent(block) {
        return Err(AstcError::Reserved);
    }
    if read_bits(block, 9, 1) == 1 {
        return Err(AstcError::UnsupportedHdr);
    }
    let r = unorm16_to_unorm8(read_bits(block, 64, 16));
    let g = unorm16_to_unorm8(read_bits(block, 80, 16));
    let b = unorm16_to_unorm8(read_bits(block, 96, 16));
    let a = unorm16_to_unorm8(read_bits(block, 112, 16));
    Ok([[r, g, b, a]; 16])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture_codec::astc::AstcError;

    /// Build a minimal LDR void-extent block with the given 16-bit channels and
    /// a "no coordinates" extent (all-ones per the spec's degenerate hint).
    fn ldr_block(r: u16, g: u16, b: u16, a: u16) -> [u8; 16] {
        let mut blk = [0u8; 16];
        // bits [0..9) signature, bit 9 = 0 (LDR), bits [10..12) = 0b11 reserved,
        // bits [12..64) = all ones (degenerate extent coordinates).
        // Compose the low 64 bits then store little-endian.
        let mut lo: u64 = 0;
        lo |= VOID_EXTENT_SIGNATURE as u64; // [0..9)
                                            // bit 9 stays 0 (LDR)
        lo |= 0b11u64 << 10; // reserved [10..12)
        lo |= ((1u64 << 52) - 1) << 12; // [12..64) all ones
        blk[0..8].copy_from_slice(&lo.to_le_bytes());
        blk[8..10].copy_from_slice(&r.to_le_bytes());
        blk[10..12].copy_from_slice(&g.to_le_bytes());
        blk[12..14].copy_from_slice(&b.to_le_bytes());
        blk[14..16].copy_from_slice(&a.to_le_bytes());
        blk
    }

    #[test]
    fn decodes_constant_colour_to_all_texels() {
        let blk = ldr_block(0xFFFF, 0x8000, 0x0000, 0xFFFF);
        let out = decode_astc_void_extent_ldr(&blk).unwrap();
        for texel in out {
            assert_eq!(texel, [255, 128, 0, 255]);
        }
    }

    #[test]
    fn endpoints_round_like_the_gpu() {
        // 0x0101 / 65535 * 255 = 1.0039..  -> rounds to 1.
        let blk = ldr_block(0x0101, 0x00FF, 0x0100, 0x0000);
        let out = decode_astc_void_extent_ldr(&blk).unwrap();
        assert_eq!(out[0], [1, 1, 1, 0]);
    }

    #[test]
    fn hdr_flag_is_rejected() {
        let mut blk = ldr_block(0, 0, 0, 0);
        blk[1] |= 1 << (9 - 8); // set bit 9 (byte 1, bit 1)
        assert_eq!(
            decode_astc_void_extent_ldr(&blk),
            Err(AstcError::UnsupportedHdr)
        );
    }

    #[test]
    fn non_void_extent_signature_is_rejected() {
        let blk = [0u8; 16];
        assert_eq!(decode_astc_void_extent_ldr(&blk), Err(AstcError::Reserved));
    }
}
