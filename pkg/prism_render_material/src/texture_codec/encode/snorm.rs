//! BC4/BC5 `SNORM` (signed RGTC/RGTC2) single- and two-channel block encoders:
//! compress a 4x4 signed `i8` tile to an 8- or 16-byte block.
//!
//! These are the inverses of
//! [`decode_bc4_signed`](crate::decode_bc4_signed) /
//! [`decode_bc5_signed`](crate::decode_bc5_signed) and the signed sibling of
//! [`encode_bc4`](crate::encode_bc4) / [`encode_bc5`](crate::encode_bc5). AAA
//! pipelines bake object-space normals, signed displacement, and motion-vector
//! channels to `BC4_SNORM`/`BC5_SNORM` (4 bpp per channel) offline, so a
//! correct signed encoder completes the RGTC family on the compression path.
//!
//! The signed block byte layout and 3-bit index word are identical to the
//! unsigned block; only the endpoint *interpretation* differs (two's-complement
//! `i8`, with the reserved `-128` pattern meaning `-127`). The encoder mirrors
//! the unsigned [`encode_bc4`](crate::encode_bc4):
//!
//! * **Eight-value mode** (`r0 > r1`): endpoints are the tile `max`/`min`;
//!   six interpolants span the interior in sevenths.
//! * **Six-value mode** (`r0 <= r1`): endpoints are `min`/`max`, four
//!   interpolants span the interior in fifths, and two palette slots are the
//!   hard `-127`/`127` terminals.
//!
//! Endpoints are the exact per-tile signed `min`/`max` (clamped to the
//! symmetric `-127..=127` range so they are representable), so the palette is a
//! superset of the true extrema; each texel takes its nearest palette entry.
//! The output round-trips through
//! [`decode_bc4_signed`](crate::decode_bc4_signed) exactly as encoded. Pure
//! integer arithmetic -- no AI/ML -- so the result is deterministic and a GPU
//! twin encoder reproduces it bit-for-bit.
//!
//! # References
//! * Khronos Data Format Spec 1.3, RGTC signed block layout.
//! * Vulkan `VK_FORMAT_BC4_SNORM_BLOCK` / `VK_FORMAT_BC5_SNORM_BLOCK`;
//!   D3D `DXGI_FORMAT_BC4_SNORM` / `DXGI_FORMAT_BC5_SNORM`.

use super::super::snorm_block::signed_channel_palette;

/// Clamp a raw `i8` input to the symmetric representable range `-127..=127`
/// (the reserved `-128` pattern is not a valid signed-normalized value).
#[inline]
fn clamp_sym(v: i8) -> i8 {
    v.max(-127)
}

/// Pack two signed endpoints and sixteen 3-bit indices into an 8-byte block.
fn pack_block(r0: i8, r1: i8, idx: &[u8; 16]) -> [u8; 8] {
    let mut bits: u64 = 0;
    for (t, &i) in idx.iter().enumerate() {
        bits |= u64::from(i & 0x7) << (3 * t);
    }
    let b = bits.to_le_bytes();
    let e0 = r0.to_le_bytes()[0];
    let e1 = r1.to_le_bytes()[0];
    [e0, e1, b[0], b[1], b[2], b[3], b[4], b[5]]
}

/// Assign each texel its nearest signed palette entry; return
/// `(indices, sq_error)`.
fn assign(tile: &[i8; 16], palette: &[i8; 8]) -> ([u8; 16], u64) {
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

/// Encode one 4x4 signed single-channel tile (row-major, texel `t = y*4 + x`,
/// each value clamped to `-127..=127`) to an 8-byte `BC4_SNORM` block.
///
/// Both block modes are tried and the lower-squared-error block is kept; the
/// output round-trips through [`decode_bc4_signed`](crate::decode_bc4_signed).
#[must_use]
pub fn encode_bc4_signed(tile: &[i8; 16]) -> [u8; 8] {
    let mut lo = 127i8;
    let mut hi = -127i8;
    for &raw in tile {
        let v = clamp_sym(raw);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let clamped: [i8; 16] = core::array::from_fn(|t| clamp_sym(tile[t]));

    // Six-value mode (r0 <= r1): spends two codes on the hard -127/127 terminals.
    let pal6 = signed_channel_palette(lo, hi);
    let (idx6, err6) = assign(&clamped, &pal6);
    let mut best = (pack_block(lo, hi, &idx6), err6);

    // Eight-value mode (r0 > r1): all eight codes interpolate; only valid when
    // the block is not constant.
    if hi > lo {
        let pal8 = signed_channel_palette(hi, lo);
        let (idx8, err8) = assign(&clamped, &pal8);
        if err8 < best.1 {
            best = (pack_block(hi, lo, &idx8), err8);
        }
    }

    best.0
}

/// Encode one 4x4 signed two-channel tile (`[R, G]` per texel) to a 16-byte
/// `BC5_SNORM` block: the first 8 bytes are the `R` channel block, the last 8
/// the `G` channel block.
///
/// Each channel is an independent [`encode_bc4_signed`]; the output round-trips
/// through [`decode_bc5_signed`](crate::decode_bc5_signed).
#[must_use]
pub fn encode_bc5_signed(tile: &[[i8; 2]; 16]) -> [u8; 16] {
    let r: [i8; 16] = core::array::from_fn(|t| tile[t][0]);
    let g: [i8; 16] = core::array::from_fn(|t| tile[t][1]);
    let rb = encode_bc4_signed(&r);
    let gb = encode_bc4_signed(&g);
    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&rb);
    out[8..16].copy_from_slice(&gb);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_bc4_signed, decode_bc5_signed};

    fn rt(tile: &[i8; 16]) -> [i8; 16] {
        decode_bc4_signed(&encode_bc4_signed(tile))
    }

    fn ssd(a: &[i8; 16], b: &[i8; 16]) -> u64 {
        let mut e = 0u64;
        for t in 0..16 {
            let d = i32::from(a[t]) - i32::from(b[t]);
            e += (d * d) as u64;
        }
        e
    }

    #[test]
    fn constant_block_round_trips_exactly() {
        let tile = [-37i8; 16];
        assert_eq!(rt(&tile), tile, "constant signed block must be lossless");
    }

    #[test]
    fn reserved_minus_128_clamps_to_minus_127() {
        // -128 is not representable; it must clamp to -127 and round-trip there.
        let tile = [i8::MIN; 16];
        assert!(rt(&tile).iter().all(|&v| v == -127));
    }

    #[test]
    fn endpoints_reach_both_extremes() {
        let mut tile = [0i8; 16];
        tile[0] = 120;
        tile[1] = -120;
        let decoded = rt(&tile);
        let dmax = decoded.iter().copied().max().unwrap();
        let dmin = decoded.iter().copied().min().unwrap();
        assert!(dmax >= 110, "max endpoint lost: {dmax}");
        assert!(dmin <= -110, "min endpoint lost: {dmin}");
    }

    #[test]
    fn full_range_ramp_error_is_bounded() {
        // A signed ramp across the full range; with eight levels each texel
        // lands within one half palette step.
        let tile: [i8; 16] = core::array::from_fn(|t| {
            let v = -112 + (t as i32) * 224 / 15;
            v.clamp(-127, 127) as i8
        });
        let decoded = rt(&tile);
        let e = ssd(&tile, &decoded);
        assert!(e <= 16 * 20 * 20, "ramp error too high: {e}");
    }

    #[test]
    fn bc5_signed_encodes_both_channels_independently() {
        // Flat R = 70, flat G = -95 -> each channel is a constant block.
        let tile: [[i8; 2]; 16] = core::array::from_fn(|_| [70, -95]);
        let decoded = decode_bc5_signed(&encode_bc5_signed(&tile));
        assert!(decoded.iter().all(|t| t == &[70, -95]));
    }

    #[test]
    fn encode_is_deterministic() {
        let tile: [i8; 16] = core::array::from_fn(|t| ((t as i32) * 11 - 80) as i8);
        assert_eq!(encode_bc4_signed(&tile), encode_bc4_signed(&tile));
    }
}
