//! BC2/DXT3 block encoder: compress a 4x4 `RGBA8` tile to a 16-byte block.
//!
//! BC2 is the *sharp-alpha* sibling of BC3: coverage is stored as sixteen
//! explicit 4-bit alpha nibbles (no interpolation), so hard mask edges do not
//! ring the way BC3's interpolated alpha can. It remains a popular bake target
//! for cutout masks and decals with crisp coverage boundaries. The block is:
//!
//! * bytes `0..8`  -- sixteen little-endian 4-bit alpha nibbles (texel `t` at
//!   bits `[4t, 4t+3]`), the inverse of [`decode_bc2`](crate::decode_bc2)'s
//!   `a4 * 17` nibble replication;
//! * bytes `8..16` -- a BC1/DXT1 colour block over the tile *RGB*, always the
//!   opaque four-colour mode (coverage lives in the alpha half).
//!
//! Each texel alpha is quantised to its nearest nibble, `round(a * 15 / 255)`,
//! which the decoder expands back by replication; alpha values that are already
//! multiples of 17 (including the hard `0`/`255` terminals) round-trip exactly.
//! The colour half reuses [`encode_bc1`](crate::encode_bc1) with alpha forced
//! opaque, exactly as the BC3 encoder does. Pure integer arithmetic -- no
//! AI/ML -- so the result is deterministic and a GPU twin reproduces it
//! bit-for-bit.
//!
//! # References
//! * Khronos Data Format Spec 1.3, BC2 (RGB + explicit 4-bit alpha) layout.
//! * `DirectXTex` / `NVTT` / `squish` DXT3 encoders.

use super::encode_bc1;

/// Quantise an 8-bit alpha to its nearest 4-bit nibble (`round(a * 15 / 255)`).
///
/// The decoder reconstructs `nibble * 17`, so this is the error-minimising
/// inverse and is exact for every multiple of 17.
fn quantize_nibble(alpha: u8) -> u8 {
    ((u32::from(alpha) * 15 + 127) / 255) as u8
}

/// Encode one 4x4 `RGBA8` tile as a 16-byte BC2/DXT3 block.
///
/// Alpha is stored as explicit 4-bit nibbles (sharp, no interpolation) and the
/// colour channels as an opaque BC1 block; the two are concatenated in the
/// on-disk BC2 order (alpha first, colour second).
#[must_use]
pub fn encode_bc2(tile: &[[u8; 4]; 16]) -> [u8; 16] {
    let mut alpha_word: u64 = 0;
    for (t, texel) in tile.iter().enumerate() {
        let nibble = u64::from(quantize_nibble(texel[3]) & 0xF);
        alpha_word |= nibble << (4 * t);
    }

    // Force opaque so `encode_bc1` picks the four-colour mode: BC2 carries
    // coverage in the explicit-alpha half, never in the colour block.
    let mut opaque = *tile;
    for texel in &mut opaque {
        texel[3] = 255;
    }
    let color_block = encode_bc1(&opaque);

    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&alpha_word.to_le_bytes());
    out[8..16].copy_from_slice(&color_block);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_bc1, decode_bc2};

    fn ramp_tile() -> [[u8; 4]; 16] {
        let mut tile = [[0u8; 4]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            let v = (t * 17) as u8;
            *texel = [v, 255 - v, v / 2, (t * 17) as u8];
        }
        tile
    }

    #[test]
    fn multiples_of_17_alpha_round_trip_exactly() {
        // Alpha nibbles reconstruct via *17, so inputs on that lattice are exact.
        let tile = ramp_tile();
        let out = decode_bc2(&encode_bc2(&tile));
        for (t, texel) in out.iter().enumerate() {
            assert_eq!(texel[3], (t * 17) as u8, "alpha texel {t}");
        }
    }

    #[test]
    fn hard_alpha_edges_are_preserved() {
        // A checkerboard of 0/255 alpha must stay perfectly binary (sharp mask).
        let mut tile = [[10u8, 20, 30, 0]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            texel[3] = if t % 2 == 0 { 255 } else { 0 };
        }
        let out = decode_bc2(&encode_bc2(&tile));
        for (t, texel) in out.iter().enumerate() {
            let expected = if t % 2 == 0 { 255 } else { 0 };
            assert_eq!(texel[3], expected, "alpha texel {t}");
        }
    }

    #[test]
    fn alpha_quantization_is_nearest() {
        // Arbitrary alpha reconstructs within half a nibble step (<= 9).
        let mut tile = [[0u8, 0, 0, 0]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            texel[3] = (t * 16 + 3) as u8;
        }
        let out = decode_bc2(&encode_bc2(&tile));
        for (t, texel) in out.iter().enumerate() {
            let want = i32::from((t * 16 + 3) as u8);
            assert!((i32::from(texel[3]) - want).abs() <= 9, "alpha texel {t}");
        }
    }

    #[test]
    fn colour_half_matches_opaque_bc1() {
        let tile = ramp_tile();
        let block = encode_bc2(&tile);
        let mut opaque = tile;
        for texel in &mut opaque {
            texel[3] = 255;
        }
        let bc1 = decode_bc1(&encode_bc1(&opaque));
        let bc2 = decode_bc2(&block);
        for (t, texel) in bc2.iter().enumerate() {
            assert_eq!(
                [texel[0], texel[1], texel[2]],
                [bc1[t][0], bc1[t][1], bc1[t][2]],
                "rgb texel {t}"
            );
        }
    }

    #[test]
    fn colour_block_is_four_colour_mode() {
        // BC2 colour never uses punch-through even when the tile is transparent.
        let mut tile = ramp_tile();
        for texel in &mut tile {
            texel[3] = 0;
        }
        let block = encode_bc2(&tile);
        let c0 = u16::from_le_bytes([block[8], block[9]]);
        let c1 = u16::from_le_bytes([block[10], block[11]]);
        assert!(c0 >= c1, "colour must be four-colour mode: {c0} vs {c1}");
    }

    #[test]
    fn encoding_is_deterministic() {
        let tile = ramp_tile();
        assert_eq!(encode_bc2(&tile), encode_bc2(&tile));
    }
}
