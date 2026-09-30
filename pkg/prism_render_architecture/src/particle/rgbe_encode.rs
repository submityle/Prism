//! Shared-exponent `HDR` packing: Radiance `RGBE` (`[u8; 4]`) and `RGB9E5`
//! (`u32`), the two color formats that trade a per-pixel exponent for a large
//! dynamic range at a fraction of a full `f32x3`'s footprint (design §27).
//!
//! Emissive particles, bloom sources, and light-shaft cookies routinely carry
//! radiance well above `1.0`, so a `unorm8` color channel clips them and a full
//! `f32x3` wastes bandwidth. Both codecs here store one exponent shared across
//! the three color channels plus a small per-channel mantissa:
//!
//! * **Radiance `RGBE`** (the classic `.hdr`/`.pic` on-disk encoding): four
//!   bytes, an 8-bit mantissa per channel and a single 8-bit exponent biased by
//!   128. Round-trips to roughly 1% relative error on the brightest channel.
//! * **`RGB9E5`** (the Khronos `GL_EXT_texture_shared_exponent` format, laid
//!   out as `E5B9G9R9`): a `u32` with a 9-bit mantissa per channel and a 5-bit
//!   shared exponent biased by 15. More precise than `RGBE` and directly
//!   sampleable by the `GPU`.
//!
//! This module draws a hard line against [`super::compression`]: that module
//! owns the independent-channel codecs (`fp16`, octahedral, `unorm`/`snorm`) and
//! has *no* shared-exponent format. Nothing here reuses its types or functions;
//! shared-exponent packing lives only in this file.
//!
//! Every routine is closed-form and uses at most `f32::floor`/`f32::sqrt` plus
//! integer bit assembly. The exponent is separated straight out of the
//! `IEEE754` layout via `f32::to_bits`, and each `2^n` is rebuilt by writing the
//! biased exponent field back with `f32::from_bits`; no `frexp`, `ldexp`,
//! `exp2`, or any other transcendental is called, so this `CPU` reference stays
//! bit-reproducible against a future `GPU` kernel. Negative and `NaN` inputs are
//! guarded to `0.0` so a malformed color can never emit `NaN` or an
//! out-of-range integer.

use alloc::vec::Vec;

/// Radiance `RGBE` biases its stored 8-bit exponent by this amount.
pub const RGBE_EXP_BIAS: i32 = 128;

/// Radiance `RGBE` keeps an 8-bit mantissa per channel.
pub const RGBE_MANTISSA_BITS: i32 = 8;

/// Colors whose largest channel falls below this are encoded as pure black,
/// matching the classic Radiance reference implementation.
pub const RGBE_MIN: f32 = 1e-32;

/// `RGB9E5` keeps a 9-bit mantissa per channel (`N` in the Khronos spec).
pub const RGB9E5_MANTISSA_BITS: i32 = 9;

/// `RGB9E5` biases its 5-bit shared exponent by this amount (`B` / `EPBIAS`).
pub const RGB9E5_EXP_BIAS: i32 = 15;

/// The largest shared exponent an `RGB9E5` word can hold (`2^5 - 1`).
pub const RGB9E5_MAX_EXP: i32 = 31;

/// The largest finite value `RGB9E5` can represent, `(2^N - 1)/2^N * 2^(EMAX-B)`
/// which works out to exactly `65408.0` (`MAXVAL` in the Khronos spec).
pub const RGB9E5_MAX_VALUE: f32 = 65_408.0;

/// Rebuilds `2^n` as an exact `f32` purely by integer bit assembly.
///
/// For `n` in the normal-`f32` exponent range the biased exponent field is
/// written directly; subnormal powers (`n` in `-149..=-127`) set the single
/// mantissa bit that names them. Anything above the normal range saturates to
/// infinity and anything below underflows to `0.0`. This is the `exp2`/`ldexp`
/// replacement the module's no-transcendental rule requires.
#[must_use]
fn pow2_i32(n: i32) -> f32 {
    if (-126..=127).contains(&n) {
        let biased = (n + 127) as u32;
        f32::from_bits(biased << 23)
    } else if n > 127 {
        f32::INFINITY
    } else if (-149..=-127).contains(&n) {
        let shift = (n + 149) as u32;
        f32::from_bits(1_u32 << shift)
    } else {
        0.0
    }
}

/// Extracts the unbiased base-2 exponent of a positive `f32`, i.e.
/// `floor(log2(x))`, straight from its `IEEE754` exponent field.
///
/// For a normal `x = 1.m * 2^e` this is exactly `e`. Zeros and subnormals lack a
/// meaningful exponent field, so they report a value far below any bias used
/// here, which lets the callers clamp them to the black bucket without ever
/// evaluating a real logarithm.
#[must_use]
fn floor_log2(x: f32) -> i32 {
    let biased = (x.to_bits() >> 23) & 0x0000_00ff;
    if biased == 0 {
        // Zero or subnormal: smaller than every exponent the callers clamp to.
        -128
    } else {
        biased as i32 - 127
    }
}

/// Returns the largest of the three color channels.
///
/// This is the value that pins the shared exponent for both codecs. `NaN`
/// channels are ignored in favor of a real number where possible, following
/// `f32::max`.
#[must_use]
pub fn max_channel(rgb: [f32; 3]) -> f32 {
    rgb[0].max(rgb[1]).max(rgb[2])
}

/// Replaces negative or `NaN` inputs with `0.0`, leaving valid magnitudes alone.
#[must_use]
fn sanitize(x: f32) -> f32 {
    if x.is_nan() || x <= 0.0 {
        0.0
    } else {
        x
    }
}

/// Floors a non-negative product into a color byte, clamping to `0..=255`.
#[must_use]
fn channel_byte(v: f32) -> u8 {
    let clamped = v.clamp(0.0, 255.0);
    clamped as u8
}

/// Encodes a linear `HDR` color into Radiance `RGBE` (`[R, G, B, E]`).
///
/// The brightest channel sets a shared power-of-two exponent; each channel then
/// stores an 8-bit mantissa relative to it. Colors below [`RGBE_MIN`] (and any
/// negative or `NaN` input) collapse to `[0, 0, 0, 0]`, the canonical black.
#[must_use]
pub fn rgbe_encode(rgb: [f32; 3]) -> [u8; 4] {
    let r = sanitize(rgb[0]);
    let g = sanitize(rgb[1]);
    let b = sanitize(rgb[2]);
    let m = max_channel([r, g, b]);
    if m < RGBE_MIN {
        return [0, 0, 0, 0];
    }
    // frexp: m = mantissa * 2^exp with mantissa in [0.5, 1); for a normal m the
    // exponent is (biased field) - 126. m >= RGBE_MIN is always normal.
    let frexp_exp = floor_log2(m) + 1;
    // Clamp so the stored byte (frexp_exp + bias) stays inside 0..=255; extreme
    // radiance saturates rather than wrapping the exponent as the naive C does.
    let clamped_exp = frexp_exp.clamp(-RGBE_EXP_BIAS, 255 - RGBE_EXP_BIAS);
    // scale = 2^(mantissa_bits - exp): maps the brightest channel into [128,256).
    let scale = pow2_i32(RGBE_MANTISSA_BITS - clamped_exp);
    [
        channel_byte(r * scale),
        channel_byte(g * scale),
        channel_byte(b * scale),
        (clamped_exp + RGBE_EXP_BIAS) as u8,
    ]
}

/// Decodes a Radiance `RGBE` quad back into a linear `HDR` color.
///
/// A zero exponent byte is the black sentinel and decodes to `[0, 0, 0]`.
#[must_use]
pub fn rgbe_decode(rgbe: [u8; 4]) -> [f32; 3] {
    let exp = rgbe[3];
    if exp == 0 {
        return [0.0, 0.0, 0.0];
    }
    // Undo the mantissa scale: value = byte * 2^(exp - bias - mantissa_bits).
    let scale = pow2_i32(i32::from(exp) - RGBE_EXP_BIAS - RGBE_MANTISSA_BITS);
    [
        f32::from(rgbe[0]) * scale,
        f32::from(rgbe[1]) * scale,
        f32::from(rgbe[2]) * scale,
    ]
}

/// Clamps one channel into the `RGB9E5` representable range, mapping negatives
/// and `NaN` to `0.0` and everything above [`RGB9E5_MAX_VALUE`] down to it.
#[must_use]
fn clamp_rgb9e5(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, RGB9E5_MAX_VALUE)
    }
}

/// Rounds `value / denom` to the nearest 9-bit mantissa via `floor(x + 0.5)`.
#[must_use]
fn quantize_mantissa(value: f32, denom: f32) -> u32 {
    let m = (value / denom + 0.5).floor();
    m as u32
}

/// Encodes a linear `HDR` color into a Khronos `RGB9E5` (`E5B9G9R9`) `u32`.
///
/// Follows the `GL_EXT_texture_shared_exponent` reference: clamp to
/// [`RGB9E5_MAX_VALUE`], derive the 5-bit shared exponent from the brightest
/// channel, then round each channel to a 9-bit mantissa. The top 5 bits hold the
/// exponent, then blue, green, and red mantissas in descending bit order.
#[must_use]
pub fn rgb9e5_encode(rgb: [f32; 3]) -> u32 {
    let rc = clamp_rgb9e5(rgb[0]);
    let gc = clamp_rgb9e5(rgb[1]);
    let bc = clamp_rgb9e5(rgb[2]);
    let maxc = max_channel([rc, gc, bc]);

    // exp_shared = max(-B-1, floor(log2(maxc))) + 1 + B, clamped into [0, EMAX].
    let raw = floor_log2(maxc).max(-RGB9E5_EXP_BIAS - 1) + 1 + RGB9E5_EXP_BIAS;
    let mut exp_shared = raw;
    let mut denom = pow2_i32(exp_shared - RGB9E5_EXP_BIAS - RGB9E5_MANTISSA_BITS);

    // Rounding the brightest channel can carry into the next octave.
    let maxm = (maxc / denom + 0.5).floor();
    if maxm >= 512.0 {
        exp_shared += 1;
        denom *= 2.0;
    }
    exp_shared = exp_shared.clamp(0, RGB9E5_MAX_EXP);

    let rm = quantize_mantissa(rc, denom);
    let gm = quantize_mantissa(gc, denom);
    let bm = quantize_mantissa(bc, denom);
    let exp_bits = (exp_shared as u32) & 0x0000_001f;

    (exp_bits << 27) | (bm << 18) | (gm << 9) | rm
}

/// Decodes a Khronos `RGB9E5` (`E5B9G9R9`) `u32` back into a linear `HDR` color.
#[must_use]
pub fn rgb9e5_decode(value: u32) -> [f32; 3] {
    let exp_shared = (value >> 27) & 0x0000_001f;
    let rm = value & 0x0000_01ff;
    let gm = (value >> 9) & 0x0000_01ff;
    let bm = (value >> 18) & 0x0000_01ff;
    let scale = pow2_i32(exp_shared as i32 - RGB9E5_EXP_BIAS - RGB9E5_MANTISSA_BITS);
    [rm as f32 * scale, gm as f32 * scale, bm as f32 * scale]
}

/// Encodes a slice of `HDR` colors into Radiance `RGBE` quads.
#[must_use]
pub fn rgbe_encode_batch(colors: &[[f32; 3]]) -> Vec<[u8; 4]> {
    colors.iter().copied().map(rgbe_encode).collect()
}

/// Encodes a slice of `HDR` colors into Khronos `RGB9E5` words.
#[must_use]
pub fn rgb9e5_encode_batch(colors: &[[f32; 3]]) -> Vec<u32> {
    colors.iter().copied().map(rgb9e5_encode).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only float comparison tolerance (production code never compares
    /// `f32` for equality).
    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn rel_err(actual: f32, expected: f32) -> f32 {
        if expected.abs() <= CMP_EPS {
            actual.abs()
        } else {
            ((actual - expected) / expected).abs()
        }
    }

    #[test]
    fn rgbe_zero_maps_to_zero() {
        assert_eq!(rgbe_encode([0.0, 0.0, 0.0]), [0, 0, 0, 0]);
        assert_eq!(rgbe_decode([0, 0, 0, 0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn rgbe_below_min_maps_to_zero() {
        assert_eq!(rgbe_encode([1e-34, 1e-35, 0.0]), [0, 0, 0, 0]);
    }

    #[test]
    fn rgbe_negative_and_nan_guarded_to_zero() {
        let encoded = rgbe_encode([-4.0, f32::NAN, -1.0]);
        assert_eq!(encoded, [0, 0, 0, 0]);
        // A single valid channel still encodes; the bad ones read back as 0.
        let mixed = rgbe_decode(rgbe_encode([-1.0, 2.0, f32::NAN]));
        assert!(approx(mixed[0], 0.0, 1e-2));
        assert!(approx(mixed[1], 2.0, 2e-2));
        assert!(approx(mixed[2], 0.0, 1e-2));
    }

    #[test]
    fn rgbe_known_white() {
        // [1,1,1] -> mantissa 128 each, exponent byte 129.
        assert_eq!(rgbe_encode([1.0, 1.0, 1.0]), [128, 128, 128, 129]);
    }

    #[test]
    fn rgbe_known_half_quarter_eighth() {
        // Exact powers of two share exponent byte 128 and round-trip perfectly.
        let encoded = rgbe_encode([0.5, 0.25, 0.125]);
        assert_eq!(encoded, [128, 64, 32, 128]);
        let back = rgbe_decode(encoded);
        assert!(approx(back[0], 0.5, CMP_EPS));
        assert!(approx(back[1], 0.25, CMP_EPS));
        assert!(approx(back[2], 0.125, CMP_EPS));
    }

    #[test]
    fn rgbe_roundtrip_relative_error() {
        let samples = [
            [0.9, 0.8, 0.7],
            [12.0, 10.0, 8.0],
            [3.5, 3.4, 3.6],
            [100.0, 90.0, 110.0],
            [0.03, 0.028, 0.031],
        ];
        for c in samples {
            let back = rgbe_decode(rgbe_encode(c));
            for (k, (&got, &want)) in back.iter().zip(c.iter()).enumerate() {
                assert!(
                    rel_err(got, want) < 1.0e-2,
                    "channel {k} of {c:?} -> {back:?}"
                );
            }
        }
    }

    #[test]
    fn rgbe_shared_exponent_small_channel_quantizes_to_zero() {
        // A tiny channel next to a bright one loses to the shared exponent.
        let encoded = rgbe_encode([8.0, 0.01, 0.0]);
        assert_eq!(encoded[1], 0);
        assert_eq!(encoded[2], 0);
        assert!(encoded[0] > 0);
    }

    #[test]
    fn rgbe_shared_exponent_is_single_byte() {
        // Three channels of the same magnitude report one exponent for all.
        let encoded = rgbe_encode([6.0, 6.0, 6.0]);
        assert_eq!(encoded[0], encoded[1]);
        assert_eq!(encoded[1], encoded[2]);
    }

    #[test]
    fn rgbe_monotonic_in_brightness() {
        let dim = rgbe_decode(rgbe_encode([2.0, 2.0, 2.0]))[0];
        let mid = rgbe_decode(rgbe_encode([4.0, 4.0, 4.0]))[0];
        let bright = rgbe_decode(rgbe_encode([8.0, 8.0, 8.0]))[0];
        assert!(dim < mid);
        assert!(mid < bright);
    }

    #[test]
    fn rgbe_batch_matches_scalar() {
        let colors = [[1.0, 2.0, 3.0], [0.5, 0.25, 0.125], [0.0, 0.0, 0.0]];
        let batch = rgbe_encode_batch(&colors);
        assert_eq!(batch.len(), colors.len());
        for (packed, c) in batch.iter().zip(colors.iter()) {
            assert_eq!(*packed, rgbe_encode(*c));
        }
    }

    #[test]
    fn rgbe_empty_batch_is_empty() {
        assert!(rgbe_encode_batch(&[]).is_empty());
    }

    #[test]
    fn rgb9e5_zero_maps_to_zero() {
        assert_eq!(rgb9e5_encode([0.0, 0.0, 0.0]), 0);
        assert_eq!(rgb9e5_decode(0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn rgb9e5_negative_and_nan_guarded() {
        assert_eq!(rgb9e5_encode([-1.0, -2.0, -3.0]), 0);
        let back = rgb9e5_decode(rgb9e5_encode([-1.0, 4.0, f32::NAN]));
        assert!(approx(back[0], 0.0, 1e-2));
        assert!(approx(back[1], 4.0, 4e-2));
        assert!(approx(back[2], 0.0, 1e-2));
    }

    #[test]
    fn rgb9e5_known_white() {
        // [1,1,1]: exponent 16, mantissa 256 per channel.
        let encoded = rgb9e5_encode([1.0, 1.0, 1.0]);
        assert_eq!((encoded >> 27) & 0x0000_001f, 16);
        assert_eq!(encoded & 0x0000_01ff, 256);
        let back = rgb9e5_decode(encoded);
        for &channel in &back {
            assert!(approx(channel, 1.0, CMP_EPS));
        }
    }

    #[test]
    fn rgb9e5_known_half_quarter_eighth() {
        let encoded = rgb9e5_encode([0.5, 0.25, 0.125]);
        let back = rgb9e5_decode(encoded);
        assert!(approx(back[0], 0.5, CMP_EPS));
        assert!(approx(back[1], 0.25, CMP_EPS));
        assert!(approx(back[2], 0.125, CMP_EPS));
    }

    #[test]
    fn rgb9e5_roundtrip_precise() {
        let samples = [
            [0.9, 0.8, 0.7],
            [12.0, 10.0, 8.0],
            [3.5, 3.4, 3.6],
            [100.0, 90.0, 110.0],
            [0.03, 0.028, 0.031],
            [1000.0, 500.0, 250.0],
        ];
        for c in samples {
            let back = rgb9e5_decode(rgb9e5_encode(c));
            for (k, (&got, &want)) in back.iter().zip(c.iter()).enumerate() {
                assert!(
                    rel_err(got, want) < 3.0e-3,
                    "channel {k} of {c:?} -> {back:?}"
                );
            }
        }
    }

    #[test]
    fn rgb9e5_clamps_to_max_value() {
        // Everything above MAXVAL saturates to the same encoded word.
        let a = rgb9e5_encode([RGB9E5_MAX_VALUE, RGB9E5_MAX_VALUE, RGB9E5_MAX_VALUE]);
        let b = rgb9e5_encode([1.0e30, 1.0e30, 1.0e30]);
        assert_eq!(a, b);
        // Exponent must not overflow past the 5-bit field.
        assert!((a >> 27) & 0x0000_001f <= RGB9E5_MAX_EXP as u32);
        let back = rgb9e5_decode(a);
        assert!(approx(back[0], RGB9E5_MAX_VALUE, 200.0));
    }

    #[test]
    fn rgb9e5_max_value_constant() {
        // (2^9 - 1) / 2^9 * 2^(31 - 15) = 511 * 128 = 65408.
        assert!(approx(RGB9E5_MAX_VALUE, 65_408.0, CMP_EPS));
    }

    #[test]
    fn rgb9e5_bit_layout_channels_are_positioned() {
        // Only red is bright: red mantissa nonzero, green/blue zero.
        let encoded = rgb9e5_encode([500.0, 0.0, 0.0]);
        assert!(encoded & 0x0000_01ff > 0);
        assert_eq!((encoded >> 9) & 0x0000_01ff, 0);
        assert_eq!((encoded >> 18) & 0x0000_01ff, 0);
        // Only blue bright: blue mantissa nonzero, red/green zero.
        let blue = rgb9e5_encode([0.0, 0.0, 500.0]);
        assert_eq!(blue & 0x0000_01ff, 0);
        assert_eq!((blue >> 9) & 0x0000_01ff, 0);
        assert!((blue >> 18) & 0x0000_01ff > 0);
    }

    #[test]
    fn rgb9e5_exponent_in_high_bits() {
        // A brighter color must not shrink the shared exponent field.
        let dim = (rgb9e5_encode([1.0, 1.0, 1.0]) >> 27) & 0x0000_001f;
        let bright = (rgb9e5_encode([1000.0, 1000.0, 1000.0]) >> 27) & 0x0000_001f;
        assert!(bright > dim);
    }

    #[test]
    fn rgb9e5_shared_exponent_small_channel_quantizes_to_zero() {
        let encoded = rgb9e5_encode([2000.0, 0.01, 0.0]);
        assert_eq!((encoded >> 9) & 0x0000_01ff, 0);
        assert_eq!((encoded >> 18) & 0x0000_01ff, 0);
        assert!(encoded & 0x0000_01ff > 0);
    }

    #[test]
    fn rgb9e5_monotonic_in_brightness() {
        let dim = rgb9e5_decode(rgb9e5_encode([2.0, 2.0, 2.0]))[0];
        let mid = rgb9e5_decode(rgb9e5_encode([20.0, 20.0, 20.0]))[0];
        let bright = rgb9e5_decode(rgb9e5_encode([200.0, 200.0, 200.0]))[0];
        assert!(dim < mid);
        assert!(mid < bright);
    }

    #[test]
    fn rgb9e5_batch_matches_scalar() {
        let colors = [[1.0, 2.0, 3.0], [0.5, 0.25, 0.125], [0.0, 0.0, 0.0]];
        let batch = rgb9e5_encode_batch(&colors);
        assert_eq!(batch.len(), colors.len());
        for (packed, c) in batch.iter().zip(colors.iter()) {
            assert_eq!(*packed, rgb9e5_encode(*c));
        }
    }

    #[test]
    fn rgb9e5_empty_batch_is_empty() {
        assert!(rgb9e5_encode_batch(&[]).is_empty());
    }

    #[test]
    fn max_channel_returns_largest() {
        assert!(approx(max_channel([1.0, 5.0, 3.0]), 5.0, CMP_EPS));
        assert!(approx(max_channel([-1.0, -5.0, -3.0]), -1.0, CMP_EPS));
        assert!(approx(max_channel([2.0, 2.0, 2.0]), 2.0, CMP_EPS));
    }

    #[test]
    fn pow2_reconstructs_exact_powers() {
        assert!(approx(pow2_i32(0), 1.0, CMP_EPS));
        assert!(approx(pow2_i32(1), 2.0, CMP_EPS));
        assert!(approx(pow2_i32(-1), 0.5, CMP_EPS));
        assert!(approx(pow2_i32(8), 256.0, CMP_EPS));
        assert!(approx(pow2_i32(10), 1024.0, CMP_EPS));
    }

    #[test]
    fn floor_log2_matches_exponent() {
        assert_eq!(floor_log2(1.0), 0);
        assert_eq!(floor_log2(2.0), 1);
        assert_eq!(floor_log2(3.9), 1);
        assert_eq!(floor_log2(0.5), -1);
        assert_eq!(floor_log2(0.0), -128);
    }

    #[test]
    fn rgbe_and_rgb9e5_do_not_share_encoding() {
        // The two codecs disagree bit-for-bit; they are independent formats.
        let color = [3.0, 2.0, 1.0];
        let radiance = rgbe_encode(color);
        let khronos = rgb9e5_encode(color);
        // Reinterpret the RGBE bytes as a u32 purely to assert inequality.
        let radiance_word = u32::from_le_bytes(radiance);
        assert_ne!(radiance_word, khronos);
    }
}
