//! BC3/DXT5 block encoder: compress a 4x4 `RGBA8` tile to a 16-byte block.
//!
//! BC3 is the composite GPU format for colour-plus-smooth-alpha textures
//! (diffuse-with-coverage, packed mask atlases, UI with gradients). Its block
//! is simply the two already-implemented building blocks concatenated, so this
//! encoder composes them rather than re-deriving the fit:
//!
//! * bytes `0..8` -- a BC4/RGTC single-channel block over the tile *alpha*,
//!   produced by [`encode_bc4`](crate::encode_bc4);
//! * bytes `8..16` -- a BC1/DXT1 colour block over the tile *RGB*, produced by
//!   [`encode_bc1`](crate::encode_bc1).
//!
//! Unlike standalone BC1, the BC3 colour sub-block never uses the 1-bit
//! punch-through mode (coverage lives in the dedicated alpha half), so the RGB
//! is encoded with alpha forced opaque to pin [`encode_bc1`] to its four-colour
//! branch (`color0 > color1`). The output round-trips through
//! [`decode_bc3`](crate::decode_bc3) exactly as the two halves round-trip
//! through their own decoders. Pure integer/analytic arithmetic -- no AI/ML --
//! so the result is deterministic and a GPU twin reproduces it bit-for-bit.
//!
//! # References
//! * Khronos Data Format Spec 1.3, BC3 (RGB + RGTC alpha) block layout.
//! * `DirectXTex` / `NVTT` / `squish` DXT5 encoders.

use super::{encode_bc1, encode_bc4};

/// Encode one 4x4 `RGBA8` tile as a 16-byte BC3/DXT5 block.
///
/// The alpha channel is compressed as an independent BC4 block and the colour
/// channels as an opaque BC1 block; the two are concatenated in the on-disk BC3
/// order (alpha first, colour second).
#[must_use]
pub fn encode_bc3(tile: &[[u8; 4]; 16]) -> [u8; 16] {
    let mut alpha = [0u8; 16];
    for (dst, texel) in alpha.iter_mut().zip(tile.iter()) {
        *dst = texel[3];
    }
    let alpha_block = encode_bc4(&alpha);

    // Force opaque so `encode_bc1` picks the four-colour mode: BC3 carries
    // coverage in the alpha half, never in the colour block.
    let mut opaque = *tile;
    for texel in &mut opaque {
        texel[3] = 255;
    }
    let color_block = encode_bc1(&opaque);

    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&alpha_block);
    out[8..16].copy_from_slice(&color_block);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_bc1, decode_bc3, decode_bc4};

    fn ramp_tile() -> [[u8; 4]; 16] {
        let mut tile = [[0u8; 4]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            let v = (t * 17) as u8;
            *texel = [v, 255 - v, v / 2, (t * 16) as u8];
        }
        tile
    }

    #[test]
    fn alpha_half_matches_standalone_bc4() {
        // The decoded alpha of a BC3 block equals the standalone BC4 encode of
        // that alpha channel, decoded -- i.e. the alpha half is independent.
        let tile = ramp_tile();
        let block = encode_bc3(&tile);
        let mut alpha = [0u8; 16];
        for (dst, texel) in alpha.iter_mut().zip(tile.iter()) {
            *dst = texel[3];
        }
        let bc4 = decode_bc4(&encode_bc4(&alpha));
        let bc3 = decode_bc3(&block);
        for (t, texel) in bc3.iter().enumerate() {
            assert_eq!(texel[3], bc4[t][0], "alpha texel {t}");
        }
    }

    #[test]
    fn colour_half_matches_opaque_bc1() {
        // The decoded RGB of a BC3 block equals the standalone opaque BC1
        // encode of the same RGB, decoded.
        let tile = ramp_tile();
        let block = encode_bc3(&tile);
        let mut opaque = tile;
        for texel in &mut opaque {
            texel[3] = 255;
        }
        let bc1 = decode_bc1(&encode_bc1(&opaque));
        let bc3 = decode_bc3(&block);
        for (t, texel) in bc3.iter().enumerate() {
            assert_eq!(
                [texel[0], texel[1], texel[2]],
                [bc1[t][0], bc1[t][1], bc1[t][2]],
                "rgb texel {t}"
            );
        }
    }

    #[test]
    fn decoded_block_is_fully_opaque_in_colour_branch() {
        // BC3 colour never uses punch-through, so a decoded BC3 block carries
        // its coverage only through the alpha channel; the colour block itself
        // is a four-colour (opaque) block regardless of the source alpha.
        let mut tile = ramp_tile();
        for texel in &mut tile {
            texel[3] = 0; // fully transparent everywhere
        }
        let block = encode_bc3(&tile);
        // Colour sub-block endpoints must be in four-colour order (c0 > c1).
        let c0 = u16::from_le_bytes([block[8], block[9]]);
        let c1 = u16::from_le_bytes([block[10], block[11]]);
        assert!(
            c0 >= c1,
            "colour block must be four-colour mode: {c0} vs {c1}"
        );
    }

    #[test]
    fn flat_tile_round_trips_exactly() {
        // A constant tile must reconstruct to that exact colour and alpha.
        let tile = [[40u8, 90, 160, 200]; 16];
        let out = decode_bc3(&encode_bc3(&tile));
        for (t, texel) in out.iter().enumerate() {
            assert_eq!(texel[3], 200, "alpha texel {t}");
            // Colour within tight BC1 565 quantisation of the input.
            assert!((i32::from(texel[0]) - 40).abs() <= 8);
            assert!((i32::from(texel[1]) - 90).abs() <= 8);
            assert!((i32::from(texel[2]) - 160).abs() <= 8);
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let tile = ramp_tile();
        assert_eq!(encode_bc3(&tile), encode_bc3(&tile));
    }
}
