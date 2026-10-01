//! Public block decoders for the BC1/BC3/BC4/BC5 texture-compression formats.
//!
//! These cover the AAA texture staples: BC1 (opaque / 1-bit-alpha albedo), BC3
//! (albedo + smooth alpha), BC4 (single-channel masks: roughness, AO, height),
//! and BC5 (two-channel tangent-space normals). Each decoder expands one 4x4
//! block into sixteen `RGBA8` texels so a [`TexelSource`](crate::TexelSource)
//! over streaming virtual-texture pages can feed the manual sampler. BC6H/BC7
//! (multi-mode HDR / high-quality) are intentionally out of scope here.
//!
//! Built entirely on [`super::color_block`] and [`super::alpha_block`]; pure
//! integer decode with no AI/ML path, GPU-twin reproducible within the formats'
//! documented +/-1 LSB interpolation tolerance.
//!
//! # Conventions
//! * Output is row-major `RGBA8`, texel `t = y*4 + x`, `t in [0, 16)`.
//! * Channel mapping: BC4 -> value in `R`, `G=B=0`, `A=255`; BC5 -> channel 0
//!   in `R`, channel 1 in `G`, `B=0`, `A=255` (callers reconstruct normal `Z`).
//! * Block sizes: BC1/BC4 are 8 bytes; BC3/BC5 are 16 bytes.
//!
//! # References
//! * Khronos Data Format Spec 1.3 (S3TC/RGTC); Vulkan `VK_FORMAT_BC*`.

use super::alpha_block::decode_channel_block;
use super::color_block::decode_color_block;

/// Decode one 8-byte BC1 block to 16 `RGBA8` texels (1-bit punch-through alpha).
#[must_use]
pub fn decode_bc1(block: &[u8; 8]) -> [[u8; 4]; 16] {
    decode_color_block(block, true)
}

/// Decode one 16-byte BC3 block: BC1-style colour (always opaque) plus a
/// BC4-style 8-bit alpha channel.
#[must_use]
pub fn decode_bc3(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut alpha_bytes = [0u8; 8];
    alpha_bytes.copy_from_slice(&block[0..8]);
    let mut color_bytes = [0u8; 8];
    color_bytes.copy_from_slice(&block[8..16]);

    let alpha = decode_channel_block(&alpha_bytes);
    // BC3 colour never uses the punch-through transparent mode.
    let color = decode_color_block(&color_bytes, false);

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        *texel = [color[t][0], color[t][1], color[t][2], alpha[t]];
    }
    out
}

/// Decode one 8-byte BC4 block to 16 `RGBA8` texels with the value in `R`.
#[must_use]
pub fn decode_bc4(block: &[u8; 8]) -> [[u8; 4]; 16] {
    let r = decode_channel_block(block);
    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        *texel = [r[t], 0, 0, 255];
    }
    out
}

/// Decode one 16-byte BC5 block to 16 `RGBA8` texels with channel 0 in `R` and
/// channel 1 in `G` (tangent-space normal XY).
#[must_use]
pub fn decode_bc5(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut ch0 = [0u8; 8];
    ch0.copy_from_slice(&block[0..8]);
    let mut ch1 = [0u8; 8];
    ch1.copy_from_slice(&block[8..16]);

    let r = decode_channel_block(&ch0);
    let g = decode_channel_block(&ch1);
    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        *texel = [r[t], g[t], 0, 255];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc1_opaque_block_is_fully_opaque() {
        // c0 > c1 (white > black) -> 4-colour opaque; every texel alpha 255.
        let block = [0xFF, 0xFF, 0x00, 0x00, 0, 0, 0, 0];
        let out = decode_bc1(&block);
        assert!(out.iter().all(|t| t[3] == 255));
        assert_eq!(out[0], [255, 255, 255, 255]);
    }

    #[test]
    fn bc1_punchthrough_block_has_transparent_texels() {
        // c0 <= c1 and index 3 -> transparent black.
        let block = [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let out = decode_bc1(&block);
        assert!(out.iter().any(|t| t[3] == 0));
    }

    #[test]
    fn bc3_alpha_is_independent_of_color() {
        // Alpha block: r0=0,r1=255, all indices 1 -> alpha 255 everywhere.
        // Color block: white endpoints, indices 0.
        let mut block = [0u8; 16];
        block[0] = 0;
        block[1] = 255;
        // every 3-bit index = 1 across 16 texels (48-bit word of repeated 001).
        let mut bits: u64 = 0;
        for t in 0..16 {
            bits |= 1u64 << (3 * t);
        }
        let bb = bits.to_le_bytes();
        block[2..8].copy_from_slice(&bb[0..6]);
        // color: white/white, indices 0
        block[8] = 0xFF;
        block[9] = 0xFF;
        block[10] = 0xFF;
        block[11] = 0xFF;
        let out = decode_bc3(&block);
        assert!(out.iter().all(|t| t[3] == 255), "alpha not uniform: {out:?}");
    }

    #[test]
    fn bc4_puts_value_in_red_only() {
        let block = [128, 128, 0, 0, 0, 0, 0, 0]; // r0==r1 -> flat 128
        let out = decode_bc4(&block);
        assert!(out.iter().all(|t| t == &[128, 0, 0, 255]));
    }

    #[test]
    fn bc5_fills_red_and_green() {
        let mut block = [0u8; 16];
        block[0] = 200; // ch0 r0
        block[1] = 200; // ch0 r1 (flat 200)
        block[8] = 50; // ch1 r0
        block[9] = 50; // ch1 r1 (flat 50)
        let out = decode_bc5(&block);
        assert!(out.iter().all(|t| t == &[200, 50, 0, 255]));
    }
}
