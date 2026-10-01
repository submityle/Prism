//! Normalized fixed-point quantization: the device-free contract that converts
//! author-facing `f32` values into the normalized integer vertex/texture
//! formats a `GPU` consumes (`UNORM`/`SNORM`), and back, with the exact
//! round-to-nearest and endpoint rules of the `D3D`, `Vulkan`, and `OpenGL`
//! specifications (design §27 attribute compression / vertex-attribute
//! packing). Every function is a pure, total `CPU` reference that a future
//! shader kernel can be validated against bit-for-bit.
//!
//! # Scope and demarcation
//!
//! This module is the *normalized-integer format* member of the packing
//! family and is deliberately distinct from its two neighbors:
//!
//! * [`super::bit_pack_u32`] is a *generic bit-field* packer: it relocates the
//!   raw bits of already-integer values into a dense `u32` stream with no
//!   numeric transform. It neither clamps a real range nor rounds. This module
//!   instead maps a *continuous* `f32` range (`[0, 1]` for `UNORM`, `[-1, 1]`
//!   for `SNORM`) onto a bounded integer lattice, which is a lossy numeric
//!   quantization, not a bit relocation.
//! * [`super::fixed_point_q16`] is a *signed `Q16.16`* number type whose scale
//!   (`2^16`) and range (`[-32768, 32768)`) target deterministic simulation
//!   arithmetic. This module's scale is instead `2^N - 1` (`UNORM`) or
//!   `2^(N-1) - 1` (`SNORM`), chosen so the real endpoints `0`, `1`, and `-1`
//!   land *exactly* on integer endpoints — the property texture/vertex formats
//!   require, and the property `Q16.16` does not provide.
//!
//! # `UNORM` (unsigned normalized)
//!
//! A real value in `[0, 1]` maps onto the integer range `[0, 2^N - 1]`. Packing
//! clamps to `[0, 1]`, scales by the maximum code `2^N - 1`, and rounds to the
//! nearest integer via `floor(x * MAX + 0.5)` (round-half-up; only
//! [`f32::floor`] is used). Unpacking divides by the maximum code, so `0`
//! unpacks to `0.0` and the maximum code unpacks to exactly `1.0`.
//!
//! # `SNORM` (signed normalized)
//!
//! Following the `Vulkan`/`D3D` convention, a real value in `[-1, 1]` maps onto
//! the *symmetric* integer range `[-(2^(N-1) - 1), 2^(N-1) - 1]` — for 8 bits
//! that is `[-127, 127]`, not `[-128, 127]`. The extra two's-complement code
//! (`-128` for `i8`, `-32768` for `i16`) is redundant: on unpack it maps to the
//! same `-1.0` as `-127`/`-32767` because the result is clamped with
//! `max(v / MAX, -1.0)`. Packing clamps to `[-1, 1]`, then rounds *half away
//! from zero* so the quantizer is symmetric about zero: positive magnitudes use
//! `floor(y + 0.5)` and negative magnitudes negate the rounded absolute value
//! (`-floor(-y + 0.5)`), still using only [`f32::floor`]. The result is clamped
//! to `[-MAX, MAX]`, which is what keeps `-1.0` at `-127`/`-32767` rather than
//! the reserved `-128`/`-32768`.
//!
//! # Vector helpers (`RGBA`)
//!
//! [`pack_unorm8x4`] packs four `UNORM8` channels into one `u32` in
//! least-significant-byte-first (`LSB`-first) order: channel 0 (red) occupies
//! bits `0..=7`, channel 1 (green) bits `8..=15`, channel 2 (blue) bits
//! `16..=23`, and channel 3 (alpha) the most-significant byte (`MSB`), bits
//! `24..=31`. This matches the little-endian memory layout of a
//! `R8G8B8A8_UNORM` texel. [`pack_unorm16x2`] likewise places channel 0 in the
//! low 16 bits and channel 1 in the high 16 bits.
//!
//! # Numeric discipline
//!
//! The only `f32` operations used are [`f32::floor`], [`f32::clamp`],
//! [`f32::min`], [`f32::max`], and [`f32::abs`]; no transcendental or rounding
//! intrinsic (`round`, `ceil`, `powf`, …) appears. Float equality (`==`/`!=`)
//! is never used — round-trip identities are asserted on integers and
//! reconstruction accuracy with an epsilon bound.

/// Maximum integer code of a `UNORM8` channel, as `f32` (`2^8 - 1`).
pub const UNORM8_MAX: f32 = 255.0;

/// Maximum integer code of a `UNORM16` channel, as `f32` (`2^16 - 1`).
pub const UNORM16_MAX: f32 = 65535.0;

/// Maximum (and negated minimum) integer code of a `SNORM8` channel, as `f32`
/// (`2^7 - 1`). The symmetric range is `[-127, 127]`.
pub const SNORM8_MAX: f32 = 127.0;

/// Maximum (and negated minimum) integer code of a `SNORM16` channel, as `f32`
/// (`2^15 - 1`). The symmetric range is `[-32767, 32767]`.
pub const SNORM16_MAX: f32 = 32767.0;

/// Quantizes a clamped `UNORM` value to the nearest integer code as `f32`.
///
/// Clamps `x` into `[0, 1]`, scales by `max`, and applies round-half-up with
/// `floor(x * max + 0.5)`. The caller casts the result into the target integer
/// width.
fn quantize_unorm(x: f32, max: f32) -> f32 {
    (x.clamp(0.0, 1.0) * max + 0.5).floor()
}

/// Rounds `y` to the nearest integer with ties broken *away from zero*, using
/// only [`f32::floor`].
///
/// Positive inputs use `floor(y + 0.5)`; negative inputs round the magnitude
/// and negate (`-floor(-y + 0.5)`), which keeps the quantizer symmetric about
/// zero as the `SNORM` specification requires.
fn round_half_away_from_zero(y: f32) -> f32 {
    if y >= 0.0 {
        (y + 0.5).floor()
    } else {
        -((-y) + 0.5).floor()
    }
}

/// Quantizes a clamped `SNORM` value to the nearest integer code as `f32`.
///
/// Clamps `x` into `[-1, 1]`, scales by `max`, rounds half away from zero, and
/// clamps the code into `[-max, max]` so the reserved most-negative code is
/// never produced.
fn quantize_snorm(x: f32, max: f32) -> f32 {
    let scaled = x.clamp(-1.0, 1.0) * max;
    round_half_away_from_zero(scaled).clamp(-max, max)
}

/// Packs a `UNORM8` value: `[0, 1]` real maps to `[0, 255]` integer.
#[must_use]
pub fn pack_unorm8(x: f32) -> u8 {
    quantize_unorm(x, UNORM8_MAX) as u8
}

/// Packs a `UNORM16` value: `[0, 1]` real maps to `[0, 65535]` integer.
#[must_use]
pub fn pack_unorm16(x: f32) -> u16 {
    quantize_unorm(x, UNORM16_MAX) as u16
}

/// Unpacks a `UNORM8` code into `[0, 1]`.
#[must_use]
pub fn unpack_unorm8(v: u8) -> f32 {
    f32::from(v) / UNORM8_MAX
}

/// Unpacks a `UNORM16` code into `[0, 1]`.
#[must_use]
pub fn unpack_unorm16(v: u16) -> f32 {
    f32::from(v) / UNORM16_MAX
}

/// Packs a `SNORM8` value: `[-1, 1]` real maps to `[-127, 127]` integer.
#[must_use]
pub fn pack_snorm8(x: f32) -> i8 {
    quantize_snorm(x, SNORM8_MAX) as i8
}

/// Packs a `SNORM16` value: `[-1, 1]` real maps to `[-32767, 32767]` integer.
#[must_use]
pub fn pack_snorm16(x: f32) -> i16 {
    quantize_snorm(x, SNORM16_MAX) as i16
}

/// Unpacks a `SNORM8` code into `[-1, 1]`.
///
/// The reserved code `-128` maps to `-1.0`, matching `-127`.
#[must_use]
pub fn unpack_snorm8(v: i8) -> f32 {
    (f32::from(v) / SNORM8_MAX).max(-1.0)
}

/// Unpacks a `SNORM16` code into `[-1, 1]`.
///
/// The reserved code `-32768` maps to `-1.0`, matching `-32767`.
#[must_use]
pub fn unpack_snorm16(v: i16) -> f32 {
    (f32::from(v) / SNORM16_MAX).max(-1.0)
}

/// Packs four `UNORM8` channels into one `u32`, `LSB`-first (red in bits
/// `0..=7`, alpha in the `MSB`). See the [module documentation](self).
#[must_use]
pub fn pack_unorm8x4(v: [f32; 4]) -> u32 {
    let r = u32::from(pack_unorm8(v[0]));
    let g = u32::from(pack_unorm8(v[1]));
    let b = u32::from(pack_unorm8(v[2]));
    let a = u32::from(pack_unorm8(v[3]));
    r | (g << 8) | (b << 16) | (a << 24)
}

/// Unpacks a `LSB`-first `RGBA8` `u32` into four `UNORM` channels in `[0, 1]`.
#[must_use]
pub fn unpack_unorm8x4(p: u32) -> [f32; 4] {
    [
        unpack_unorm8((p & 0xFF) as u8),
        unpack_unorm8(((p >> 8) & 0xFF) as u8),
        unpack_unorm8(((p >> 16) & 0xFF) as u8),
        unpack_unorm8(((p >> 24) & 0xFF) as u8),
    ]
}

/// Packs two `UNORM16` channels into one `u32`: channel 0 in the low 16 bits,
/// channel 1 in the high 16 bits.
#[must_use]
pub fn pack_unorm16x2(v: [f32; 2]) -> u32 {
    let x = u32::from(pack_unorm16(v[0]));
    let y = u32::from(pack_unorm16(v[1]));
    x | (y << 16)
}

/// Unpacks a packed `UNORM16x2` `u32` into two channels in `[0, 1]`.
#[must_use]
pub fn unpack_unorm16x2(p: u32) -> [f32; 2] {
    [
        unpack_unorm16((p & 0xFFFF) as u16),
        unpack_unorm16(((p >> 16) & 0xFFFF) as u16),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shared epsilon for reconstruction-accuracy assertions (float equality
    /// `==` is forbidden by the module contract).
    const EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn unorm8_lower_endpoint_is_zero() {
        assert_eq!(pack_unorm8(0.0), 0);
    }

    #[test]
    fn unorm8_upper_endpoint_is_max() {
        assert_eq!(pack_unorm8(1.0), 255);
    }

    #[test]
    fn unorm16_lower_endpoint_is_zero() {
        assert_eq!(pack_unorm16(0.0), 0);
    }

    #[test]
    fn unorm16_upper_endpoint_is_max() {
        assert_eq!(pack_unorm16(1.0), 65535);
    }

    #[test]
    fn unorm8_half_rounds_to_128() {
        // floor(0.5 * 255 + 0.5) = floor(128.0) = 128.
        assert_eq!(pack_unorm8(0.5), 128);
    }

    #[test]
    fn unorm8_clamps_negative_to_zero() {
        assert_eq!(pack_unorm8(-0.3), 0);
    }

    #[test]
    fn unorm8_clamps_above_one_to_max() {
        assert_eq!(pack_unorm8(2.0), 255);
    }

    #[test]
    fn unorm16_clamps_out_of_range() {
        assert_eq!(pack_unorm16(-5.0), 0);
        assert_eq!(pack_unorm16(3.0), 65535);
    }

    #[test]
    fn snorm8_positive_endpoint_is_127() {
        assert_eq!(pack_snorm8(1.0), 127);
    }

    #[test]
    fn snorm8_negative_endpoint_is_minus_127() {
        assert_eq!(pack_snorm8(-1.0), -127);
    }

    #[test]
    fn snorm8_zero_is_zero() {
        assert_eq!(pack_snorm8(0.0), 0);
    }

    #[test]
    fn snorm8_clamps_below_minus_one() {
        assert_eq!(pack_snorm8(-2.0), -127);
    }

    #[test]
    fn snorm8_clamps_above_one() {
        assert_eq!(pack_snorm8(2.0), 127);
    }

    #[test]
    fn snorm16_positive_endpoint_is_max() {
        assert_eq!(pack_snorm16(1.0), 32767);
    }

    #[test]
    fn snorm16_negative_endpoint_is_min_symmetric() {
        assert_eq!(pack_snorm16(-1.0), -32767);
    }

    #[test]
    fn snorm8_rounding_is_symmetric_about_zero() {
        // Equal magnitudes must map to negated codes (round half away from
        // zero), unlike a naive `floor(x * MAX - 0.5)` which would be biased.
        let x = 50.4 / 127.0;
        assert_eq!(pack_snorm8(x), 50);
        assert_eq!(pack_snorm8(-x), -50);
    }

    #[test]
    fn unorm8_round_trips_every_code() {
        let mut v: u16 = 0;
        while v <= 255 {
            let code = v as u8;
            assert_eq!(pack_unorm8(unpack_unorm8(code)), code);
            v += 1;
        }
    }

    #[test]
    fn unorm16_round_trips_sampled_codes() {
        for code in [0u16, 1, 128, 32768, 40000, 65534, 65535] {
            assert_eq!(pack_unorm16(unpack_unorm16(code)), code);
        }
    }

    #[test]
    fn snorm8_round_trips_every_valid_code() {
        let mut c: i16 = -127;
        while c <= 127 {
            let code = c as i8;
            assert_eq!(pack_snorm8(unpack_snorm8(code)), code);
            c += 1;
        }
    }

    #[test]
    fn snorm16_round_trips_sampled_codes() {
        for code in [-32767i16, -20000, -1, 0, 1, 20000, 32767] {
            assert_eq!(pack_snorm16(unpack_snorm16(code)), code);
        }
    }

    #[test]
    fn unpack_unorm8_matches_reference_ratio() {
        assert!(approx(unpack_unorm8(128), 128.0 / 255.0, EPS));
    }

    #[test]
    fn unpack_unorm16_matches_reference_ratio() {
        assert!(approx(unpack_unorm16(40000), 40000.0 / 65535.0, EPS));
    }

    #[test]
    fn unpack_snorm8_reserved_code_maps_to_minus_one() {
        assert!(approx(unpack_snorm8(-128), -1.0, EPS));
        assert!(approx(unpack_snorm8(-127), -1.0, EPS));
    }

    #[test]
    fn unpack_snorm16_reserved_code_maps_to_minus_one() {
        assert!(approx(unpack_snorm16(-32768), -1.0, EPS));
    }

    #[test]
    fn unorm8x4_round_trips_each_channel() {
        let codes = [12u8, 200, 0, 255];
        let packed = pack_unorm8x4([
            unpack_unorm8(codes[0]),
            unpack_unorm8(codes[1]),
            unpack_unorm8(codes[2]),
            unpack_unorm8(codes[3]),
        ]);
        let out = unpack_unorm8x4(packed);
        for (channel, &code) in out.iter().zip(codes.iter()) {
            assert_eq!(pack_unorm8(*channel), code);
        }
    }

    #[test]
    fn unorm8x4_byte_order_is_lsb_first() {
        // Red = 0x11 in the LSB, alpha = 0x44 in the MSB.
        let packed = pack_unorm8x4([
            unpack_unorm8(0x11),
            unpack_unorm8(0x22),
            unpack_unorm8(0x33),
            unpack_unorm8(0x44),
        ]);
        assert_eq!(packed, 0x4433_2211);
    }

    #[test]
    fn unorm16x2_round_trips_and_orders_channels() {
        let packed = pack_unorm16x2([unpack_unorm16(0x1234), unpack_unorm16(0xABCD)]);
        assert_eq!(packed & 0xFFFF, 0x1234);
        assert_eq!(packed >> 16, 0xABCD);
        let out = unpack_unorm16x2(packed);
        assert_eq!(pack_unorm16(out[0]), 0x1234);
        assert_eq!(pack_unorm16(out[1]), 0xABCD);
    }

    #[test]
    fn unorm8_quantization_error_within_bound() {
        // Max round-to-nearest error of a UNORM8 is 1 / (2 * MAX).
        let bound = 1.0 / (2.0 * UNORM8_MAX) + EPS;
        let mut state: u32 = 0x1234_5678;
        let mut count: u32 = 0;
        while count < 10_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = ((state >> 8) as f32) / ((1u32 << 24) as f32);
            let r = unpack_unorm8(pack_unorm8(x));
            assert!((r - x).abs() <= bound);
            count += 1;
        }
    }

    #[test]
    fn snorm8_quantization_error_within_bound() {
        // Max round-to-nearest error of a SNORM8 is 1 / (2 * MAX).
        let bound = 1.0 / (2.0 * SNORM8_MAX) + EPS;
        let mut state: u32 = 0x9E37_79B9;
        let mut count: u32 = 0;
        while count < 10_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = ((state >> 8) as f32) / ((1u32 << 24) as f32);
            let x = u * 2.0 - 1.0;
            let r = unpack_snorm8(pack_snorm8(x));
            assert!((r - x).abs() <= bound);
            count += 1;
        }
    }

    #[test]
    fn unpack_outputs_stay_in_unit_ranges() {
        assert!((0.0..=1.0).contains(&unpack_unorm8(200)));
        assert!((-1.0..=1.0).contains(&unpack_snorm8(-100)));
        assert!((0.0..=1.0).contains(&unpack_unorm16(50000)));
        assert!((-1.0..=1.0).contains(&unpack_snorm16(-30000)));
    }
}
