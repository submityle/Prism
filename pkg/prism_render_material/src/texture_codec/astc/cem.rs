//! ASTC Colour Endpoint Mode (CEM) endpoint assembly.
//!
//! Once the colour integers have been read from the block and unquantized to
//! 8-bit components, the Colour Endpoint Mode decides how they are grouped into
//! the two endpoint colours. ASTC defines sixteen CEMs; this module implements
//! the ten LDR modes:
//!
//! | CEM | Encoding                                   | integers |
//! |-----|--------------------------------------------|----------|
//! | 0   | Luminance direct                           | 2        |
//! | 1   | Luminance base+offset                      | 2        |
//! | 4   | Luminance + alpha direct                   | 4        |
//! | 5   | Luminance + alpha base+offset              | 4        |
//! | 6   | RGB base + scale                           | 4        |
//! | 8   | RGB direct                                 | 6        |
//! | 9   | RGB base + offset (delta)                  | 6        |
//! | 10  | RGB base + scale, two alpha                | 6        |
//! | 12  | RGBA direct                                | 8        |
//! | 13  | RGBA base + offset (delta)                 | 8        |
//!
//! The six HDR modes (2, 3, 7, 11, 14, 15) are a later milestone and are
//! rejected upstream before reaching this module.
//!
//! Every unpack routine is transcribed from the ARM `astcenc` reference decoder
//! (`astcenc_color_unquantize.cpp`, Apache-2.0). The reference operates on
//! already-unquantized integers in `0..=255` for direct lanes; delta/offset
//! lanes carry signed intermediates, so the maths here is done in `i32` and
//! clamped back to `0..=255` exactly where the reference clamps.
//!
//! Pure integer arithmetic -- no AI/ML path.

use super::endpoints::Endpoints;

/// An endpoint colour during unpack: signed so delta/offset intermediates and
/// blue-uncontraction sums do not wrap. Clamped to `0..=255` on output.
type V4 = [i32; 4];

/// Number of colour integers a CEM needs: `(endpoint_class + 1) * 2`, where the
/// endpoint class is `cem >> 2`.
#[inline]
pub(super) fn cem_integer_count(cem: u32) -> u32 {
    ((cem >> 2) + 1) * 2
}

/// Whether `cem` is one of the ten LDR modes this module decodes. The HDR modes
/// (2, 3, 7, 11, 14, 15) return `false`.
#[inline]
pub(super) fn cem_is_ldr(cem: u32) -> bool {
    matches!(cem, 0 | 1 | 4 | 5 | 6 | 8 | 9 | 10 | 12 | 13)
}

/// Blue-uncontraction (reference `uncontract_color`): recover full-range R and
/// G from a blue-contracted endpoint; B and A pass through unchanged.
#[inline]
fn uncontract(c: V4) -> V4 {
    [(c[0] + c[2]) >> 1, (c[1] + c[2]) >> 1, c[2], c[3]]
}

/// Signed sum of the R, G and B lanes (reference `hadd_rgb_s`).
#[inline]
fn hadd_rgb(c: V4) -> i32 {
    c[0] + c[1] + c[2]
}

/// Clamp an [`V4`] to an RGBA8 colour.
#[inline]
fn to_rgba8(c: V4) -> [u8; 4] {
    [
        c[0].clamp(0, 255) as u8,
        c[1].clamp(0, 255) as u8,
        c[2].clamp(0, 255) as u8,
        c[3].clamp(0, 255) as u8,
    ]
}

/// Reference `bit_transfer_signed(input0, input1)`: move the high bit of
/// `input0` into `input1`, then reduce `input0` to a sign-extended 6-bit
/// offset. Faithful to the per-lane bit maths; callers mirror the reference's
/// argument order.
#[inline]
fn bit_transfer_signed(input0: &mut V4, input1: &mut V4) {
    for lane in 0..4 {
        input1[lane] = (input1[lane] >> 1) | (input0[lane] & 0x80);
        input0[lane] = (input0[lane] >> 1) & 0x3F;
        if input0[lane] & 0x20 != 0 {
            input0[lane] -= 0x40;
        }
    }
}

/// Reference `rgba_unpack`: direct RGBA with blue-uncontraction + endpoint swap
/// when endpoint 0 is the brighter of the two.
#[inline]
fn rgba_unpack(mut input0: V4, mut input1: V4) -> (V4, V4) {
    if hadd_rgb(input0) > hadd_rgb(input1) {
        input0 = uncontract(input0);
        input1 = uncontract(input1);
        core::mem::swap(&mut input0, &mut input1);
    }
    (input0, input1)
}

/// Reference `rgba_delta_unpack`: base + signed offset, with blue-uncontraction
/// + swap keyed on the sign of the offset's RGB sum.
#[inline]
fn rgba_delta_unpack(mut input0: V4, mut input1: V4) -> (V4, V4) {
    // Reference call order is `bit_transfer_signed(input1, input0)`.
    bit_transfer_signed(&mut input1, &mut input0);

    let rgb_sum = hadd_rgb(input1);
    for lane in 0..4 {
        input1[lane] += input0[lane];
    }
    if rgb_sum < 0 {
        input0 = uncontract(input0);
        input1 = uncontract(input1);
        core::mem::swap(&mut input0, &mut input1);
    }
    (input0, input1)
}

/// Reference `rgb_scale_unpack`: `output1` is the base colour, `output0` the
/// base scaled by `scale / 256`.
#[inline]
fn rgb_scale_unpack(base: V4, scale: i32) -> (V4, V4) {
    let output1 = [base[0], base[1], base[2], 255];
    let output0 = [
        (base[0] * scale) >> 8,
        (base[1] * scale) >> 8,
        (base[2] * scale) >> 8,
        255,
    ];
    (output0, output1)
}

/// Decode `cem`'s two endpoint colours from the unquantized colour integers
/// `vals`.
///
/// `vals.len()` must equal [`cem_integer_count`] for `cem`, and `cem` must be
/// an LDR mode ([`cem_is_ldr`]); both are guaranteed by the caller.
pub(super) fn unpack_endpoints(cem: u32, vals: &[u8]) -> Endpoints {
    let v: [i32; 8] = {
        let mut a = [0i32; 8];
        for (slot, val) in a.iter_mut().zip(vals.iter()) {
            *slot = i32::from(*val);
        }
        a
    };

    let (e0, e1): (V4, V4) = match cem {
        // Luminance direct.
        0 => ([v[0], v[0], v[0], 255], [v[1], v[1], v[1], 255]),

        // Luminance base + offset.
        1 => {
            let l0 = (v[0] >> 2) | (v[1] & 0xC0);
            let l1 = (l0 + (v[1] & 0x3F)).min(255);
            ([l0, l0, l0, 255], [l1, l1, l1, 255])
        }

        // Luminance + alpha direct.
        4 => ([v[0], v[0], v[0], v[2]], [v[1], v[1], v[1], v[3]]),

        // Luminance + alpha base + offset.
        5 => {
            let mut lum0 = v[0];
            let mut lum1 = v[1];
            let mut alpha0 = v[2];
            let mut alpha1 = v[3];

            lum0 |= (lum1 & 0x80) << 1;
            alpha0 |= (alpha1 & 0x80) << 1;
            lum1 &= 0x7F;
            alpha1 &= 0x7F;
            if lum1 & 0x40 != 0 {
                lum1 -= 0x80;
            }
            if alpha1 & 0x40 != 0 {
                alpha1 -= 0x80;
            }
            lum0 >>= 1;
            lum1 >>= 1;
            alpha0 >>= 1;
            alpha1 >>= 1;
            lum1 += lum0;
            alpha1 += alpha0;
            lum1 = lum1.clamp(0, 255);
            alpha1 = alpha1.clamp(0, 255);

            ([lum0, lum0, lum0, alpha0], [lum1, lum1, lum1, alpha1])
        }

        // RGB base + scale: integers are (R, G, B, scale).
        6 => rgb_scale_unpack([v[0], v[1], v[2], 0], v[3]),

        // RGB direct: shares `rgba_unpack`, then forces alpha opaque.
        8 => {
            let (mut e0, mut e1) = rgba_unpack([v[0], v[2], v[4], 0], [v[1], v[3], v[5], 0]);
            e0[3] = 255;
            e1[3] = 255;
            (e0, e1)
        }

        // RGB base + offset.
        9 => rgba_unpack_rgb_delta(v),

        // RGB base + scale, two alpha: integers are (R, G, B, scale, A0, A1).
        10 => {
            let base = [v[0], v[1], v[2], v[4]];
            let scale = v[3];
            let alpha1 = v[5];
            let output1 = [v[0], v[1], v[2], alpha1];
            let output0 = [
                (base[0] * scale) >> 8,
                (base[1] * scale) >> 8,
                (base[2] * scale) >> 8,
                base[3],
            ];
            (output0, output1)
        }

        // RGBA direct.
        12 => rgba_unpack([v[0], v[2], v[4], v[6]], [v[1], v[3], v[5], v[7]]),

        // RGBA base + offset.
        13 => rgba_delta_unpack([v[0], v[2], v[4], v[6]], [v[1], v[3], v[5], v[7]]),

        // HDR modes are rejected upstream; `cem_is_ldr` guards this match.
        _ => unreachable!("non-LDR CEM {cem} reached unpack_endpoints"),
    };

    Endpoints {
        e0: to_rgba8(e0),
        e1: to_rgba8(e1),
    }
}

/// RGB base + offset (CEM 9): shares `rgba_delta_unpack` then forces alpha opaque.
#[inline]
fn rgba_unpack_rgb_delta(v: [i32; 8]) -> (V4, V4) {
    let (mut e0, mut e1) = rgba_delta_unpack([v[0], v[2], v[4], 0], [v[1], v[3], v[5], 0]);
    e0[3] = 255;
    e1[3] = 255;
    (e0, e1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_counts_follow_endpoint_class() {
        assert_eq!(cem_integer_count(0), 2);
        assert_eq!(cem_integer_count(1), 2);
        assert_eq!(cem_integer_count(4), 4);
        assert_eq!(cem_integer_count(6), 4);
        assert_eq!(cem_integer_count(8), 6);
        assert_eq!(cem_integer_count(10), 6);
        assert_eq!(cem_integer_count(12), 8);
        assert_eq!(cem_integer_count(13), 8);
    }

    #[test]
    fn ldr_mode_classification() {
        for cem in [0u32, 1, 4, 5, 6, 8, 9, 10, 12, 13] {
            assert!(cem_is_ldr(cem), "CEM {cem} should be LDR");
        }
        for cem in [2u32, 3, 7, 11, 14, 15] {
            assert!(!cem_is_ldr(cem), "CEM {cem} is HDR");
        }
    }

    #[test]
    fn luminance_direct_broadcasts_to_grey() {
        let ep = unpack_endpoints(0, &[40, 200]);
        assert_eq!(ep.e0, [40, 40, 40, 255]);
        assert_eq!(ep.e1, [200, 200, 200, 255]);
    }

    #[test]
    fn luminance_alpha_direct_keeps_alpha() {
        let ep = unpack_endpoints(4, &[10, 220, 30, 240]);
        assert_eq!(ep.e0, [10, 10, 10, 30]);
        assert_eq!(ep.e1, [220, 220, 220, 240]);
    }

    #[test]
    fn rgb_direct_matches_the_cem8_path() {
        // hadd(e0) <= hadd(e1): pass-through, alpha forced opaque.
        let ep = unpack_endpoints(8, &[10, 200, 20, 210, 30, 220]);
        assert_eq!(ep.e0, [10, 20, 30, 255]);
        assert_eq!(ep.e1, [200, 210, 220, 255]);
    }

    #[test]
    fn rgba_direct_carries_alpha() {
        let ep = unpack_endpoints(12, &[10, 200, 20, 210, 30, 220, 40, 230]);
        assert_eq!(ep.e0, [10, 20, 30, 40]);
        assert_eq!(ep.e1, [200, 210, 220, 230]);
    }

    #[test]
    fn rgb_scale_darkens_endpoint0() {
        // scale 128/256 => output0 is roughly half the base; output1 is base.
        let ep = unpack_endpoints(6, &[100, 150, 200, 128]);
        assert_eq!(ep.e1, [100, 150, 200, 255]);
        // output0 = (base * 128) >> 8 == base / 2.
        assert_eq!(ep.e0, [50, 75, 100, 255]);
    }

    #[test]
    fn zero_offset_delta_leaves_base_unchanged() {
        // RGB delta: offset lanes zero (even values) => endpoints identical to
        // the base after bit transfer. v0..v5 = (base,0) interleaved.
        // base = 40 -> stored as 40 (even) so low bit (transferred) is 0.
        let ep = unpack_endpoints(9, &[80, 0, 100, 0, 120, 0]);
        // base lane = (80>>1)|(0&0x80) = 40; offset = (0>>1)&0x3f = 0.
        assert_eq!(ep.e0, [40, 50, 60, 255]);
        assert_eq!(ep.e1, [40, 50, 60, 255]);
    }
}
