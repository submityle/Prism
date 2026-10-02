//! BC5/RGTC2 block encoder: compress the two leading channels of a 4x4
//! `RGBA8` tile to a 16-byte block.
//!
//! BC5 is the AAA tangent-space normal-map format (and the general two-channel
//! format for packed scalar pairs such as metallic+roughness). Its block is two
//! independent BC4/RGTC single-channel blocks concatenated, so this encoder
//! composes [`encode_bc4`](crate::encode_bc4) twice rather than re-deriving the
//! fit:
//!
//! * bytes `0..8`  -- channel 0 (`R`, e.g. normal `X`);
//! * bytes `8..16` -- channel 1 (`G`, e.g. normal `Y`).
//!
//! The output round-trips through [`decode_bc5`](crate::decode_bc5) exactly as
//! each half round-trips through [`decode_bc4`](crate::decode_bc4). The `Z`
//! component of a normal is reconstructed at sample time from `X`/`Y`
//! (see the `normal_map` module), so it is intentionally not stored. Pure
//! integer arithmetic -- no AI/ML -- so the result is deterministic and a GPU
//! twin reproduces it bit-for-bit.
//!
//! # References
//! * Khronos Data Format Spec 1.3, RGTC2/BC5 block layout.
//! * `DirectXTex` / `NVTT` BC5 and 3Dc normal-map encoders.

use super::encode_bc4;

/// Encode channels `R`/`G` of one 4x4 `RGBA8` tile as a 16-byte BC5 block.
///
/// Channel 0 (`R`) occupies bytes `0..8` and channel 1 (`G`) bytes `8..16`,
/// each a standalone BC4/RGTC block. Channels `B`/`A` are ignored.
#[must_use]
pub fn encode_bc5(tile: &[[u8; 4]; 16]) -> [u8; 16] {
    let mut ch0 = [0u8; 16];
    let mut ch1 = [0u8; 16];
    for (t, texel) in tile.iter().enumerate() {
        ch0[t] = texel[0];
        ch1[t] = texel[1];
    }
    let block0 = encode_bc4(&ch0);
    let block1 = encode_bc4(&ch1);

    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&block0);
    out[8..16].copy_from_slice(&block1);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_bc4, decode_bc5};

    fn xy_tile() -> [[u8; 4]; 16] {
        let mut tile = [[0u8; 4]; 16];
        for (t, texel) in tile.iter_mut().enumerate() {
            // Independent ramps in X and Y so a channel swap would be caught.
            *texel = [(t * 16) as u8, (255 - t * 16) as u8, 0, 255];
        }
        tile
    }

    #[test]
    fn each_half_matches_standalone_bc4() {
        let tile = xy_tile();
        let block = encode_bc5(&tile);
        let mut ch0 = [0u8; 16];
        let mut ch1 = [0u8; 16];
        for (t, texel) in tile.iter().enumerate() {
            ch0[t] = texel[0];
            ch1[t] = texel[1];
        }
        let d0 = decode_bc4(&encode_bc4(&ch0));
        let d1 = decode_bc4(&encode_bc4(&ch1));
        let d5 = decode_bc5(&block);
        for (t, texel) in d5.iter().enumerate() {
            assert_eq!(texel[0], d0[t][0], "R texel {t}");
            assert_eq!(texel[1], d1[t][0], "G texel {t}");
        }
    }

    #[test]
    fn channels_are_not_swapped() {
        // Constant distinct channels must land in their own slots.
        let tile = [[200u8, 40, 0, 255]; 16];
        let out = decode_bc5(&encode_bc5(&tile));
        for (t, texel) in out.iter().enumerate() {
            assert!((i32::from(texel[0]) - 200).abs() <= 1, "R texel {t}");
            assert!((i32::from(texel[1]) - 40).abs() <= 1, "G texel {t}");
            // Decoder pins B=0, A=255 for BC5.
            assert_eq!(texel[2], 0);
            assert_eq!(texel[3], 255);
        }
    }

    #[test]
    fn flat_tile_round_trips_exactly() {
        let tile = [[123u8, 210, 0, 255]; 16];
        let out = decode_bc5(&encode_bc5(&tile));
        for (t, texel) in out.iter().enumerate() {
            assert_eq!(texel[0], 123, "R texel {t}");
            assert_eq!(texel[1], 210, "G texel {t}");
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let tile = xy_tile();
        assert_eq!(encode_bc5(&tile), encode_bc5(&tile));
    }
}
