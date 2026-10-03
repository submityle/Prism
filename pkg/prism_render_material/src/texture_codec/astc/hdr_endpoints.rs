//! ASTC HDR colour-endpoint decode (the six HDR Colour Endpoint Modes).
//!
//! ASTC's HDR CEMs encode endpoints as packed, modal bit-fields that expand to
//! 12-bit magnitudes, which are then left-shifted into a 16-bit internal
//! representation (`value << 4`). Unlike the LDR path those 16-bit values are
//! **not** UNORM: the RGB lanes (and the HDR alpha lane) are logarithmic (LNS)
//! and are converted to FP16 by [`lns_to_sf16`] after weight interpolation.
//!
//! | CEM | Format                         | integers | this module |
//! |-----|--------------------------------|----------|-------------|
//! | 2   | HDR luminance, large range     | 2        | yes         |
//! | 3   | HDR luminance, small range     | 2        | yes         |
//! | 7   | HDR RGB base + scale           | 4        | yes         |
//! | 11  | HDR RGB direct                 | 6        | yes         |
//! | 14  | HDR RGB + LDR alpha            | 8        | yes         |
//! | 15  | HDR RGB + HDR alpha            | 8        | yes         |
//!
//! Every routine is transcribed from the ARM `astcenc` reference decoder
//! (`astcenc_color_unquantize.cpp`, Apache-2.0): `hdr_rgbo_unpack`,
//! `hdr_rgb_unpack`, `hdr_luminance_{small,large}_range_unpack`,
//! `hdr_alpha_unpack`, `hdr_rgb_ldr_alpha_unpack`, `hdr_rgb_hdr_alpha_unpack`
//! and the FP16 conversions `lns_to_sf16` / `unorm16_to_sf16`
//! (`astcenc_vecmathlib.h`). The reference emits a default HDR alpha of
//! `0x7800` (= FP16 1.0 after LNS) for the RGB-only HDR modes, matching a
//! hardware HDR-profile decode.
//!
//! Pure integer arithmetic -- no AI/ML path.

use super::block_mode::decode_block_mode_2d;
use super::cem::cem_is_ldr;
use super::endpoints::decode_cem_color_vals;
use super::infill::{infill_dual_plane_4x4, infill_weights_4x4};
use super::AstcError;
use crate::texture_codec::half_bits_to_f32;

/// A pair of HDR endpoint colours in the 16-bit internal representation, plus a
/// per-channel LNS mask. A channel whose mask bit is set is logarithmic and is
/// converted with [`lns_to_sf16`]; a clear bit is a linear UNORM16 lane
/// converted with [`unorm16_to_sf16`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HdrEndpoints {
    /// Endpoint 0, 16-bit internal values (R, G, B, A).
    pub e0: [i32; 4],
    /// Endpoint 1, 16-bit internal values (R, G, B, A).
    pub e1: [i32; 4],
    /// Per-channel LNS (HDR) mask: `true` = logarithmic lane.
    pub lns: [bool; 4],
}

/// Reference `safe_signed_lsh`: left-shift as unsigned to avoid signed-overflow
/// UB, then reinterpret as signed.
#[inline]
fn safe_signed_lsh(val: i32, shift: u32) -> i32 {
    ((val as u32).wrapping_shl(shift)) as i32
}

/// Convert a post-interpolation UNORM16 value `[0, 65535]` to an FP16 bit
/// pattern in `[0, 1]` (reference `unorm16_to_sf16`).
#[must_use]
pub(super) fn unorm16_to_sf16(p: i32) -> u16 {
    let p = (p & 0xFFFF) as u32;
    if p == 0xFFFF {
        return 0x3C00; // exactly 1.0
    }
    if p < 4 {
        // Tiny values map linearly into the FP16 subnormal range (p << 8).
        return (p << 8) as u16;
    }
    // Count leading zeros over the 16-bit value, matching `clz(p) - 16`.
    let lz = (p as u16).leading_zeros() as i32;
    let mut v = p.wrapping_mul(1u32 << ((lz + 1) as u32));
    v &= 0xFFFF;
    v >>= 6;
    v |= ((14 - lz) as u32) << 10;
    v as u16
}

/// Convert a post-interpolation 16-bit LNS value to an FP16 bit pattern
/// (reference `lns_to_sf16`), clamped to `0x7BFF` (largest finite half).
#[must_use]
pub(super) fn lns_to_sf16(p: i32) -> u16 {
    let p = (p & 0xFFFF) as i32;
    let mc = p & 0x7FF;
    let ec = p >> 11;

    let mt = if mc < 512 {
        mc * 3
    } else if mc < 1536 {
        mc * 4 - 512
    } else {
        mc * 5 - 2048
    };

    let res = (ec << 10) | (mt >> 3);
    res.min(0x7BFF) as u16
}

/// CEM 2 -- HDR luminance, large range (reference
/// `hdr_luminance_large_range_unpack`).
#[inline]
fn hdr_luminance_large_range(v: [i32; 2]) -> ([i32; 4], [i32; 4]) {
    let (v0, v1) = (v[0], v[1]);
    let (y0, y1) = if v1 >= v0 {
        (v0 << 4, v1 << 4)
    } else {
        ((v1 << 4) + 8, (v0 << 4) - 8)
    };
    (
        [y0 << 4, y0 << 4, y0 << 4, 0x7800],
        [y1 << 4, y1 << 4, y1 << 4, 0x7800],
    )
}

/// CEM 3 -- HDR luminance, small range (reference
/// `hdr_luminance_small_range_unpack`).
#[inline]
fn hdr_luminance_small_range(v: [i32; 2]) -> ([i32; 4], [i32; 4]) {
    let (v0, v1) = (v[0], v[1]);
    let (y0, mut y1) = if v0 & 0x80 != 0 {
        (((v1 & 0xE0) << 4) | ((v0 & 0x7F) << 2), (v1 & 0x1F) << 2)
    } else {
        (((v1 & 0xF0) << 4) | ((v0 & 0x7F) << 1), (v1 & 0x0F) << 1)
    };
    y1 += y0;
    if y1 > 0xFFF {
        y1 = 0xFFF;
    }
    (
        [y0 << 4, y0 << 4, y0 << 4, 0x7800],
        [y1 << 4, y1 << 4, y1 << 4, 0x7800],
    )
}

/// CEM 7 -- HDR RGB base + scale (reference `hdr_rgbo_unpack`).
#[inline]
fn hdr_rgbo(input: [i32; 4]) -> ([i32; 4], [i32; 4]) {
    let (v0, v1, v2, v3) = (input[0], input[1], input[2], input[3]);

    let modeval = ((v0 & 0xC0) >> 6) | (((v1 & 0x80) >> 7) << 2) | (((v2 & 0x80) >> 7) << 3);

    let (majcomp, mode) = if (modeval & 0xC) != 0xC {
        (modeval >> 2, modeval & 3)
    } else if modeval != 0xF {
        (modeval & 3, 4)
    } else {
        (0, 5)
    };

    let mut red = v0 & 0x3F;
    let mut green = v1 & 0x1F;
    let mut blue = v2 & 0x1F;
    let mut scale = v3 & 0x1F;

    let bit0 = (v1 >> 6) & 1;
    let bit1 = (v1 >> 5) & 1;
    let bit2 = (v2 >> 6) & 1;
    let bit3 = (v2 >> 5) & 1;
    let bit4 = (v3 >> 7) & 1;
    let bit5 = (v3 >> 6) & 1;
    let bit6 = (v3 >> 5) & 1;

    let ohcomp = 1 << mode;

    if ohcomp & 0x30 != 0 {
        green |= bit0 << 6;
    }
    if ohcomp & 0x3A != 0 {
        green |= bit1 << 5;
    }
    if ohcomp & 0x30 != 0 {
        blue |= bit2 << 6;
    }
    if ohcomp & 0x3A != 0 {
        blue |= bit3 << 5;
    }
    if ohcomp & 0x3D != 0 {
        scale |= bit6 << 5;
    }
    if ohcomp & 0x2D != 0 {
        scale |= bit5 << 6;
    }
    if ohcomp & 0x04 != 0 {
        scale |= bit4 << 7;
    }
    if ohcomp & 0x3B != 0 {
        red |= bit4 << 6;
    }
    if ohcomp & 0x04 != 0 {
        red |= bit3 << 6;
    }
    if ohcomp & 0x10 != 0 {
        red |= bit5 << 7;
    }
    if ohcomp & 0x0F != 0 {
        red |= bit2 << 7;
    }
    if ohcomp & 0x05 != 0 {
        red |= bit1 << 8;
    }
    if ohcomp & 0x0A != 0 {
        red |= bit0 << 8;
    }
    if ohcomp & 0x05 != 0 {
        red |= bit0 << 9;
    }
    if ohcomp & 0x02 != 0 {
        red |= bit6 << 9;
    }
    if ohcomp & 0x01 != 0 {
        red |= bit3 << 10;
    }
    if ohcomp & 0x02 != 0 {
        red |= bit5 << 10;
    }

    const SHAMTS: [u32; 6] = [1, 1, 2, 3, 4, 5];
    let shamt = SHAMTS[mode as usize];
    red <<= shamt;
    green <<= shamt;
    blue <<= shamt;
    scale <<= shamt;

    if mode != 5 {
        green = red - green;
        blue = red - blue;
    }

    match majcomp {
        1 => core::mem::swap(&mut red, &mut green),
        2 => core::mem::swap(&mut red, &mut blue),
        _ => {}
    }

    let mut red0 = red - scale;
    let mut green0 = green - scale;
    let mut blue0 = blue - scale;

    red = red.max(0);
    green = green.max(0);
    blue = blue.max(0);
    red0 = red0.max(0);
    green0 = green0.max(0);
    blue0 = blue0.max(0);

    (
        [red0 << 4, green0 << 4, blue0 << 4, 0x7800],
        [red << 4, green << 4, blue << 4, 0x7800],
    )
}

/// CEM 11 -- HDR RGB direct (reference `hdr_rgb_unpack`).
#[inline]
fn hdr_rgb(input: [i32; 6]) -> ([i32; 4], [i32; 4]) {
    let (v0, v1, v2, v3, v4, v5) = (input[0], input[1], input[2], input[3], input[4], input[5]);

    let modeval = ((v1 & 0x80) >> 7) | (((v2 & 0x80) >> 7) << 1) | (((v3 & 0x80) >> 7) << 2);
    let majcomp = ((v4 & 0x80) >> 7) | (((v5 & 0x80) >> 7) << 1);

    if majcomp == 3 {
        return (
            [v0 << 8, v2 << 8, (v4 & 0x7F) << 9, 0x7800],
            [v1 << 8, v3 << 8, (v5 & 0x7F) << 9, 0x7800],
        );
    }

    let mut a = v0 | ((v1 & 0x40) << 2);
    let mut b0 = v2 & 0x3F;
    let mut b1 = v3 & 0x3F;
    let mut c = v1 & 0x3F;
    let mut d0 = v4 & 0x7F;
    let mut d1 = v5 & 0x7F;

    const DBITS_TAB: [u32; 8] = [7, 6, 7, 6, 5, 6, 5, 6];
    let dbits = DBITS_TAB[modeval as usize];

    let bit0 = (v2 >> 6) & 1;
    let bit1 = (v3 >> 6) & 1;
    let bit2 = (v4 >> 6) & 1;
    let bit3 = (v5 >> 6) & 1;
    let bit4 = (v4 >> 5) & 1;
    let bit5 = (v5 >> 5) & 1;

    let ohmod = 1 << modeval;
    if ohmod & 0xA4 != 0 {
        a |= bit0 << 9;
    }
    if ohmod & 0x8 != 0 {
        a |= bit2 << 9;
    }
    if ohmod & 0x50 != 0 {
        a |= bit4 << 9;
    }
    if ohmod & 0x50 != 0 {
        a |= bit5 << 10;
    }
    if ohmod & 0xA0 != 0 {
        a |= bit1 << 10;
    }
    if ohmod & 0xC0 != 0 {
        a |= bit2 << 11;
    }
    if ohmod & 0x4 != 0 {
        c |= bit1 << 6;
    }
    if ohmod & 0xE8 != 0 {
        c |= bit3 << 6;
    }
    if ohmod & 0x20 != 0 {
        c |= bit2 << 7;
    }
    if ohmod & 0x5B != 0 {
        b0 |= bit0 << 6;
        b1 |= bit1 << 6;
    }
    if ohmod & 0x12 != 0 {
        b0 |= bit2 << 7;
        b1 |= bit3 << 7;
    }
    if ohmod & 0xAF != 0 {
        d0 |= bit4 << 5;
        d1 |= bit5 << 5;
    }
    if ohmod & 0x5 != 0 {
        d0 |= bit2 << 6;
        d1 |= bit3 << 6;
    }

    // Sign-extend d0/d1 from `dbits` bits.
    let sx_shamt = 32 - dbits;
    d0 = safe_signed_lsh(d0, sx_shamt) >> sx_shamt;
    d1 = safe_signed_lsh(d1, sx_shamt) >> sx_shamt;

    // Expand all values to 12 bits.
    let val_shamt = ((modeval >> 1) ^ 3) as u32;
    a = safe_signed_lsh(a, val_shamt);
    b0 = safe_signed_lsh(b0, val_shamt);
    b1 = safe_signed_lsh(b1, val_shamt);
    c = safe_signed_lsh(c, val_shamt);
    d0 = safe_signed_lsh(d0, val_shamt);
    d1 = safe_signed_lsh(d1, val_shamt);

    let mut red1 = a;
    let mut green1 = a - b0;
    let mut blue1 = a - b1;
    let mut red0 = a - c;
    let mut green0 = a - b0 - c - d0;
    let mut blue0 = a - b1 - c - d1;

    red0 = red0.clamp(0, 4095);
    green0 = green0.clamp(0, 4095);
    blue0 = blue0.clamp(0, 4095);
    red1 = red1.clamp(0, 4095);
    green1 = green1.clamp(0, 4095);
    blue1 = blue1.clamp(0, 4095);

    match majcomp {
        1 => {
            core::mem::swap(&mut red0, &mut green0);
            core::mem::swap(&mut red1, &mut green1);
        }
        2 => {
            core::mem::swap(&mut red0, &mut blue0);
            core::mem::swap(&mut red1, &mut blue1);
        }
        _ => {}
    }

    (
        [red0 << 4, green0 << 4, blue0 << 4, 0x7800],
        [red1 << 4, green1 << 4, blue1 << 4, 0x7800],
    )
}

/// Reference `hdr_alpha_unpack`: unpack the two HDR alpha endpoints (used by
/// CEM 15). Returns the two 16-bit internal alpha values.
#[inline]
fn hdr_alpha(input: [i32; 2]) -> (i32, i32) {
    let mut v6 = input[0];
    let mut v7 = input[1];

    let modeval = ((v6 >> 7) & 1) | ((v7 >> 6) & 2);
    v6 &= 0x7F;
    v7 &= 0x7F;

    let (out0, out1) = if modeval == 3 {
        (v6 << 5, v7 << 5)
    } else {
        // Transfer 1-3 high bits of v7 to make an 8-10 bit base.
        v6 |= (v7 << (modeval + 1)) & 0x780;
        // Extract remaining 4-6 bits and unbias to a signed delta.
        v7 &= 0x3F >> modeval;
        v7 ^= 32 >> modeval;
        v7 -= 32 >> modeval;
        // Scale base to 12 bits and delta to 6-10 bits.
        v6 <<= 4 - modeval;
        v7 = safe_signed_lsh(v7, (4 - modeval) as u32);
        v7 = (v6 + v7).clamp(0, 0xFFF);
        (v6, v7)
    };

    (out0 << 4, out1 << 4)
}

/// Dispatch an HDR CEM to its endpoint unpack, returning the two 16-bit
/// internal endpoints and the per-channel LNS mask. `vals` holds the
/// unquantized colour integers (`0..=255`), exactly as the LDR path produces
/// them.
///
/// The RGB-only HDR modes (2/3/7/11) default alpha to the HDR-profile constant
/// `0x7800` (FP16 1.0). CEM 14 carries a linear LDR alpha (`value * 257`); CEM
/// 15 carries a logarithmic HDR alpha.
#[must_use]
pub(super) fn unpack_hdr_endpoints(cem: u32, vals: &[u8]) -> HdrEndpoints {
    let v = |i: usize| i32::from(vals[i]);
    match cem {
        2 => {
            let (e0, e1) = hdr_luminance_large_range([v(0), v(1)]);
            HdrEndpoints {
                e0,
                e1,
                lns: [true, true, true, true],
            }
        }
        3 => {
            let (e0, e1) = hdr_luminance_small_range([v(0), v(1)]);
            HdrEndpoints {
                e0,
                e1,
                lns: [true, true, true, true],
            }
        }
        7 => {
            let (e0, e1) = hdr_rgbo([v(0), v(1), v(2), v(3)]);
            HdrEndpoints {
                e0,
                e1,
                lns: [true, true, true, true],
            }
        }
        11 => {
            let (e0, e1) = hdr_rgb([v(0), v(1), v(2), v(3), v(4), v(5)]);
            HdrEndpoints {
                e0,
                e1,
                lns: [true, true, true, true],
            }
        }
        14 => {
            // HDR RGB + LDR alpha: RGB from CEM 11; alpha is a linear UNORM
            // byte expanded by replication (`v * 257`) and decoded as UNORM16.
            let (mut e0, mut e1) = hdr_rgb([v(0), v(1), v(2), v(3), v(4), v(5)]);
            e0[3] = v(6) * 257;
            e1[3] = v(7) * 257;
            HdrEndpoints {
                e0,
                e1,
                lns: [true, true, true, false],
            }
        }
        15 => {
            // HDR RGB + HDR alpha: RGB from CEM 11; alpha via `hdr_alpha`.
            let (mut e0, mut e1) = hdr_rgb([v(0), v(1), v(2), v(3), v(4), v(5)]);
            let (a0, a1) = hdr_alpha([v(6), v(7)]);
            e0[3] = a0;
            e1[3] = a1;
            HdrEndpoints {
                e0,
                e1,
                lns: [true, true, true, true],
            }
        }
        _ => unreachable!("non-HDR CEM {cem} reached unpack_hdr_endpoints"),
    }
}

/// Interpolate one HDR colour lane between its two 16-bit internal endpoints
/// using an ASTC weight `w` in `0..=64`, then convert to `f32`.
///
/// Mirrors the reference `lerp_color_int` + `decode_texel` for a hardware HDR
/// decode (the `u8_mask` bit-replication branch is inactive here because this
/// path targets the FP16/FP32 output, not `decode_unorm8`): interpolate in the
/// 16-bit integer domain `(c0*(64-w) + c1*w + 32) >> 6`, then map a
/// logarithmic (`lns`) lane through [`lns_to_sf16`] and a linear lane through
/// [`unorm16_to_sf16`] before expanding the FP16 bit pattern to `f32`.
#[inline]
pub(super) fn lerp_hdr_lane(e0: i32, e1: i32, w: u32, lns: bool) -> f32 {
    let w = w as i32;
    let color = (e0 * (64 - w) + e1 * w + 32) >> 6;
    let half = if lns {
        lns_to_sf16(color)
    } else {
        unorm16_to_sf16(color)
    };
    half_bits_to_f32(half)
}

/// Decode a single-partition 4x4 ASTC **HDR** `block` to sixteen `RGBA` texels
/// in `f32`, row-major (`texel = y * 4 + x`).
///
/// Handles the six HDR Colour Endpoint Modes (2/3/7/11/14/15), any colour
/// quantisation, any weight grid resampled to the 4x4 footprint by the Khronos
/// bilinear infill, and both single- and dual-plane weights. The RGB lanes are
/// always logarithmic (LNS); CEM 14 carries a linear LDR alpha and every other
/// mode a logarithmic (or default `0x7800`) alpha.
///
/// # Errors
/// Returns [`AstcError::UnsupportedBlockMode`] for an LDR CEM, a grid that
/// exceeds the 4x4 footprint, or a multi-partition block (its own milestone),
/// and propagates [`AstcError`] from the endpoint/weight decode. No unsupported
/// block is decoded to approximate pixels.
pub(super) fn decode_single_partition_4x4_hdr(
    block: &[u8; 16],
) -> Result<[[f32; 4]; 16], AstcError> {
    let mode = (u16::from(block[1]) << 8 | u16::from(block[0])) & 0x07FF;
    let bm = decode_block_mode_2d(mode).ok_or(AstcError::UnsupportedBlockMode)?;

    // The weight grid must fit inside the 4x4 texel footprint; an oversized
    // grid is an illegal block mode that conformant hardware rejects.
    if bm.weights_x > 4 || bm.weights_y > 4 {
        return Err(AstcError::UnsupportedBlockMode);
    }

    // Partition count is the 2-bit field at block bits [11, 13); 0 => single.
    let partition_count = ((u32::from(block[1]) >> 3) & 0x3) + 1;
    if partition_count != 1 {
        return Err(AstcError::UnsupportedBlockMode);
    }

    // CEM is the 4-bit field at block bits [13, 17). Only the six HDR CEMs are
    // decoded here; an LDR CEM belongs to the LDR path.
    let cem = ((u32::from(block[1]) >> 5) & 0x7) | ((u32::from(block[2]) & 1) << 3);
    if cem_is_ldr(cem) {
        return Err(AstcError::UnsupportedBlockMode);
    }

    let (vals, integer_count) = decode_cem_color_vals(block, bm.weight_bits, cem, bm.dual_plane)?;
    let ep = unpack_hdr_endpoints(cem, &vals[..integer_count as usize]);

    let mut out = [[0.0f32; 4]; 16];
    if bm.dual_plane {
        let below_weights_pos = 128 - bm.weight_bits;
        let ccs = super::block_reader::read_bits(block, below_weights_pos - 2, 2);
        let (plane0, plane1) =
            infill_dual_plane_4x4(block, bm.weights_x, bm.weights_y, bm.weight_levels)?;
        for texel in 0..16usize {
            for c in 0..4usize {
                let w = u32::from(if c as u32 == ccs {
                    plane1[texel]
                } else {
                    plane0[texel]
                });
                out[texel][c] = lerp_hdr_lane(ep.e0[c], ep.e1[c], w, ep.lns[c]);
            }
        }
    } else {
        let weights = infill_weights_4x4(block, bm.weights_x, bm.weights_y, bm.weight_levels)?;
        for (texel, &wq) in weights.iter().enumerate() {
            let w = u32::from(wq);
            for c in 0..4usize {
                out[texel][c] = lerp_hdr_lane(ep.e0[c], ep.e1[c], w, ep.lns[c]);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture_codec::half_bits_to_f32;

    #[test]
    fn unorm16_endpoints_map_like_the_reference() {
        assert_eq!(unorm16_to_sf16(0), 0x0000);
        assert_eq!(unorm16_to_sf16(0xFFFF), 0x3C00); // 1.0
                                                     // 0x8000 (0.5) -> FP16 0x3800.
        assert_eq!(half_bits_to_f32(unorm16_to_sf16(0x8000)), 0.5);
    }

    #[test]
    fn lns_zero_and_one() {
        // LNS 0 -> 0.0; LNS 0x7800 -> FP16 1.0 (the RGB default-alpha constant).
        assert_eq!(lns_to_sf16(0), 0x0000);
        assert_eq!(half_bits_to_f32(lns_to_sf16(0x7800)), 1.0);
    }

    #[test]
    fn hdr_luminance_large_range_is_monotonic() {
        // v1 >= v0: y0 = v0<<4, y1 = v1<<4; grey broadcast, alpha = 0x7800.
        // Reference `hdr_luminance_large_range_unpack`: y = v<<4, then the
        // output lane is y<<4 (12-bit magnitude expanded into the 16-bit
        // internal representation). v0=10 -> 10<<8 = 2560; v1=200 -> 51200.
        let ep = unpack_hdr_endpoints(2, &[10, 200]);
        assert_eq!(ep.e0, [2560, 2560, 2560, 0x7800]);
        assert_eq!(ep.e1, [51200, 51200, 51200, 0x7800]);
        assert_eq!(ep.lns, [true, true, true, true]);
    }

    #[test]
    fn hdr_rgb_direct_major3_passthrough() {
        // Force majcomp == 3 (v4 and v5 high bits set) -> direct pass-through.
        let ep = unpack_hdr_endpoints(11, &[0x10, 0x20, 0x18, 0x28, 0x80, 0x80]);
        // red0 = v0<<8, green0 = v2<<8, blue0 = (v4&0x7f)<<9 = 0.
        assert_eq!(ep.e0, [0x10 << 8, 0x18 << 8, 0, 0x7800]);
        assert_eq!(ep.e1, [0x20 << 8, 0x28 << 8, 0, 0x7800]);
    }

    #[test]
    fn hdr_rgb_ldr_alpha_scales_alpha_linearly() {
        let ep = unpack_hdr_endpoints(14, &[0x10, 0x20, 0x18, 0x28, 0x80, 0x80, 100, 200]);
        assert_eq!(ep.e0[3], 100 * 257);
        assert_eq!(ep.e1[3], 200 * 257);
        assert_eq!(ep.lns, [true, true, true, false]);
    }
}
