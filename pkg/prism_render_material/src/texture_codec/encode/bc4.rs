//! BC4/RGTC single-channel block encoder: compress a 4x4 `R8` tile to an
//! 8-byte block.
//!
//! This is the inverse of
//! [`decode_bc4`](crate::decode_bc4) (and the shared single-channel block
//! behind the alpha half of BC3 and both channels of BC5). AAA pipelines bake
//! roughness, metallic, occlusion, height and other scalar maps to BC4 (4 bpp,
//! single channel) offline, so a correct encoder is the second member of the
//! texture *compression* path after [`encode_bc1`](crate::encode_bc1).
//!
//! The encoder evaluates both block modes and keeps the lower-error one:
//!
//! * **Eight-value mode** (`r0 > r1`): endpoints are the tile `max`/`min`;
//!   six interpolants span the interior in sevenths. Best for smooth scalar
//!   data.
//! * **Six-value mode** (`r0 <= r1`): endpoints are `min`/`max`, four
//!   interpolants span the interior in fifths, and two palette slots are the
//!   hard `0`/`255` terminals. Best when the block genuinely contains those
//!   extremes (e.g. a binary mask), since it spends fewer codes on them.
//!
//! Endpoints are the exact per-tile `min`/`max`, so the reconstructed palette
//! is a superset of the true extrema; each texel then takes its nearest
//! palette entry. The output round-trips through
//! [`decode_bc4`](crate::decode_bc4) exactly as encoded. Pure integer
//! arithmetic -- no AI/ML -- so the result is deterministic and a GPU twin
//! encoder reproduces it bit-for-bit.
//!
//! # References
//! * Khronos Data Format Spec 1.3, RGTC/BC4 block layout.
//! * Vulkan `VK_FORMAT_BC4_*` / D3D `DXGI_FORMAT_BC4_*` definitions.

use super::super::alpha_block::channel_palette;

/// Pack two endpoints and sixteen 3-bit indices into an 8-byte block.
fn pack_block(r0: u8, r1: u8, idx: &[u8; 16]) -> [u8; 8] {
    let mut bits: u64 = 0;
    for (t, &i) in idx.iter().enumerate() {
        bits |= u64::from(i & 0x7) << (3 * t);
    }
    let b = bits.to_le_bytes();
    [r0, r1, b[0], b[1], b[2], b[3], b[4], b[5]]
}

/// Assign each texel its nearest palette entry; return `(indices, sq_error)`.
fn assign(tile: &[u8; 16], palette: &[u8; 8]) -> ([u8; 16], u64) {
    let mut idx = [0u8; 16];
    let mut err = 0u64;
    for (t, &v) in tile.iter().enumerate() {
        let mut best = 0usize;
        let mut best_d = u32::MAX;
        for (i, &pv) in palette.iter().enumerate() {
            let d = i32::from(v) - i32::from(pv);
            let d = (d * d) as u32;
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        idx[t] = best as u8;
        err += u64::from(best_d);
    }
    (idx, err)
}

/// Encode one 4x4 single-channel tile (row-major, texel `t = y*4 + x`) to an
/// 8-byte BC4/RGTC block.
///
/// Both block modes are tried and the lower-squared-error block is kept; the
/// output round-trips through [`decode_bc4`](crate::decode_bc4).
#[must_use]
pub fn encode_bc4(tile: &[u8; 16]) -> [u8; 8] {
    let mut lo = 255u8;
    let mut hi = 0u8;
    for &v in tile {
        lo = lo.min(v);
        hi = hi.max(v);
    }

    // Six-value mode (r0 <= r1): spends two codes on the hard 0/255 terminals.
    let pal6 = channel_palette(lo, hi);
    let (idx6, err6) = assign(tile, &pal6);
    let mut best = (pack_block(lo, hi, &idx6), err6);

    // Eight-value mode (r0 > r1): all eight codes interpolate; only valid when
    // the block is not constant.
    if hi > lo {
        let pal8 = channel_palette(hi, lo);
        let (idx8, err8) = assign(tile, &pal8);
        if err8 < best.1 {
            best = (pack_block(hi, lo, &idx8), err8);
        }
    }

    best.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_bc4;

    /// Round-trip a single-channel tile and project the decoded R channel.
    fn rt(tile: &[u8; 16]) -> [u8; 16] {
        let decoded = decode_bc4(&encode_bc4(tile));
        core::array::from_fn(|t| decoded[t][0])
    }

    fn ssd(a: &[u8; 16], b: &[u8; 16]) -> u64 {
        let mut e = 0u64;
        for t in 0..16 {
            let d = i32::from(a[t]) - i32::from(b[t]);
            e += (d * d) as u64;
        }
        e
    }

    #[test]
    fn constant_block_round_trips_exactly() {
        let tile = [137u8; 16];
        assert_eq!(rt(&tile), tile, "constant block must be lossless");
    }

    #[test]
    fn endpoints_are_preserved() {
        // A block containing both extremes must decode something that reaches
        // near both ends.
        let mut tile = [40u8; 16];
        tile[0] = 250;
        tile[1] = 5;
        let decoded = rt(&tile);
        let dmax = decoded.iter().copied().max().unwrap();
        let dmin = decoded.iter().copied().min().unwrap();
        assert!(dmax >= 240, "max endpoint lost: {dmax}");
        assert!(dmin <= 10, "min endpoint lost: {dmin}");
    }

    #[test]
    fn smooth_ramp_error_is_bounded() {
        // A full-range ramp spans eight-value mode's palette; with only eight
        // levels across 0..255 each texel must land within one half palette
        // step (255/7 ~= 36 -> ~18) of the original.
        let tile: [u8; 16] = core::array::from_fn(|t| (t * 17) as u8); // 0..255
        let decoded = rt(&tile);
        let worst = (0..16)
            .map(|t| (i32::from(tile[t]) - i32::from(decoded[t])).unsigned_abs())
            .max()
            .unwrap();
        assert!(
            worst <= 19,
            "ramp worst texel error {worst}, ssd {}",
            ssd(&tile, &decoded)
        );
    }

    #[test]
    fn binary_mask_is_lossless() {
        // 0/255 mask: six-value mode has both terminals exactly, so it is
        // lossless.
        let mut tile = [0u8; 16];
        for t in (0..16).step_by(2) {
            tile[t] = 255;
        }
        assert_eq!(rt(&tile), tile, "binary mask must be lossless");
    }

    #[test]
    fn encoding_is_deterministic() {
        let tile: [u8; 16] = core::array::from_fn(|t| ((t * 29 + 7) & 0xff) as u8);
        assert_eq!(encode_bc4(&tile), encode_bc4(&tile));
    }

    #[test]
    fn all_zero_and_all_full_are_lossless() {
        for v in [0u8, 255u8] {
            let tile = [v; 16];
            assert_eq!(rt(&tile), tile, "v={v}");
        }
    }

    #[test]
    fn decoded_values_stay_within_block_extrema() {
        // No decoded texel should overshoot the block's own min/max range
        // beyond interpolation rounding (palette is built from those endpoints).
        let tile: [u8; 16] = core::array::from_fn(|t| (30 + t * 11) as u8); // 30..195
        let decoded = rt(&tile);
        for &v in &decoded {
            assert!(v <= 200, "overshoot high: {v}");
            assert!(v >= 28, "overshoot low: {v}");
        }
    }

    #[test]
    fn two_level_block_is_lossless() {
        // Only two distinct interior values -> both are palette entries.
        let mut tile = [60u8; 16];
        for texel in tile.iter_mut().skip(8) {
            *texel = 190;
        }
        assert_eq!(rt(&tile), tile, "two-level block must be lossless");
    }
}
