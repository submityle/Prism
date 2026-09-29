//! `HDR` to `LDR` `tonemap` operators for the particle post/compositing stack
//! (design §16, §21).
//!
//! A `tonemap` compresses an open-ended high-dynamic-range (`HDR`) radiance
//! signal into the closed `[0, 1]` low-dynamic-range (`LDR`) range a display
//! consumes. Production post stacks (Unreal's `ACES` filmic curve, `Frostbite`'s
//! exposure + filmic pipeline, Unity `HDRP`'s `Reinhard`/`ACES` selector) all
//! evaluate the same three quantities per pixel: an *exposure* scale, a *tone
//! curve*, and a *display-`gamma` encode*. This module owns the
//! `CPU`-verifiable maths of that contract and packs the parameters into the
//! `std430` block a `GPU` post pass binds.
//!
//! # Determinism
//!
//! Every curve here is a rational polynomial (`Reinhard`, `ACES`) or a
//! square-root `gamma` approximation, so evaluation touches only `+ - * /`,
//! [`f32::sqrt`], and integer shifts. No transcendental function
//! (`sin`/`cos`/`exp`/`ln`/`powf`) and no `f32::round`/`f32::ceil` is ever
//! called, matching the determinism contract of the sibling
//! [`super::vignette_mask`] module so a future `GPU` evaluation reproduces the
//! `CPU` result bit for bit.
//!
//! # Pipeline
//!
//! [`TonemapParams::map`] runs the canonical order: multiply by exposure, apply
//! the selected [`TonemapOperator`] per channel (clamped into `[0, 1]`), then
//! encode to the display `gamma` via [`linear_to_srgb_approx`]. The output is
//! always a valid `LDR` `RGB` triple in `[0, 1]`.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Denominators with magnitude below this are treated as zero so evaluation
/// falls back to a defined result instead of dividing by (near) zero or
/// propagating `NaN`.
const MIN_DENOM: f32 = 1e-6;

/// A `white_point` is clamped up to at least this before squaring, so a
/// degenerate zero white cannot blow the extended-`Reinhard` denominator up.
const MIN_WHITE: f32 = 1e-4;

/// The largest exposure stop magnitude honored by [`exposure_from_stops`]; a
/// `2^31` factor already exceeds any physical exposure and keeps the shift in
/// range.
const MAX_STOPS: i32 = 31;

/// A numerically guarded division: returns `0.0` when the denominator is within
/// [`MIN_DENOM`] of zero, otherwise the true quotient. This keeps every curve
/// finite for pathological inputs without a transcendental fallback.
#[must_use]
fn safe_div(numerator: f32, denominator: f32) -> f32 {
    if denominator.abs() < MIN_DENOM {
        0.0
    } else {
        numerator / denominator
    }
}

/// Scales a linear `RGB` triple by a scalar `exposure` multiplier, the first
/// step of the `tonemap` pipeline. Exposure is a plain per-channel product, so
/// it is exactly linear: doubling `exposure` doubles every channel.
#[must_use]
pub fn apply_exposure(rgb: [f32; 3], exposure: f32) -> [f32; 3] {
    [rgb[0] * exposure, rgb[1] * exposure, rgb[2] * exposure]
}

/// The multiplicative exposure factor for an integer number of photographic
/// *stops* (`2^stops`), computed with an integer bit shift rather than the
/// forbidden [`f32::powf`]. Positive stops brighten (`+1` doubles), negative
/// stops darken (`-1` halves), and `0` is the identity `1.0`. The magnitude is
/// clamped to [`MAX_STOPS`] so the shift stays in range.
#[must_use]
pub fn exposure_from_stops(stops: i32) -> f32 {
    let magnitude = stops.saturating_abs().min(MAX_STOPS);
    let shift = u32::try_from(magnitude).unwrap_or(0);
    let factor_bits = 1u32 << shift;
    #[expect(
        clippy::cast_precision_loss,
        reason = "a power-of-two u32 is represented exactly by f32"
    )]
    let factor = factor_bits as f32;
    if stops >= 0 {
        factor
    } else {
        safe_div(1.0, factor)
    }
}

/// The classic `Reinhard` tone curve `x / (1 + x)`.
///
/// Maps `[0, inf)` monotonically into `[0, 1)`: `reinhard(0) = 0` and the value
/// approaches (but never reaches) `1.0` as `x` grows without bound.
#[must_use]
pub fn reinhard(x: f32) -> f32 {
    safe_div(x, 1.0 + x)
}

/// The extended `Reinhard` tone curve `x * (1 + x / white^2) / (1 + x)`.
///
/// Unlike the plain [`reinhard`], it lets the artist pick the luminance
/// `white` that should map to `1.0`: any input equal to `white` yields exactly
/// `1.0`, values below it stay below `1.0`, and values above it exceed `1.0`
/// (the caller clamps for display). `white` is floored at [`MIN_WHITE`] so the
/// squared denominator never collapses.
#[must_use]
pub fn reinhard_extended(x: f32, white: f32) -> f32 {
    let w = white.max(MIN_WHITE);
    let white_sq = w * w;
    let numerator = x * (1.0 + safe_div(x, white_sq));
    safe_div(numerator, 1.0 + x)
}

/// The Narkowicz `ACES` filmic approximation, a rational polynomial fit to the
/// `ACES` reference tone curve:
/// `clamp((x * (2.51*x + 0.03)) / (x * (2.43*x + 0.59) + 0.14), 0, 1)`.
///
/// Monotonic on `[0, inf)`, maps black to black, and saturates smoothly toward
/// `1.0`, giving the filmic shoulder/toe that plain `Reinhard` lacks.
#[must_use]
pub fn aces_film(x: f32) -> f32 {
    let numerator = x * (2.51 * x + 0.03);
    let denominator = x * (2.43 * x + 0.59) + 0.14;
    safe_div(numerator, denominator).clamp(0.0, 1.0)
}

/// Encodes a linear channel into the display `gamma` domain with a square-root
/// approximation of the `sRGB` transfer (`gamma` ~= 2.0), avoiding the
/// forbidden [`f32::powf`]. The input is clamped into `[0, 1]` first, so the
/// result is always a valid `LDR` value: `linear_to_srgb_approx(0) = 0` and
/// `linear_to_srgb_approx(1) = 1`.
#[must_use]
pub fn linear_to_srgb_approx(linear: f32) -> f32 {
    linear.clamp(0.0, 1.0).sqrt()
}

/// The exact inverse of [`linear_to_srgb_approx`]: squares an encoded channel
/// back into linear space (`gamma` ~= 2.0 decode). Clamps into `[0, 1]` first,
/// so `srgb_to_linear(linear_to_srgb_approx(x)) == x` for `x` in `[0, 1]`.
#[must_use]
pub fn srgb_to_linear(encoded: f32) -> f32 {
    let e = encoded.clamp(0.0, 1.0);
    e * e
}

/// Which tone curve [`TonemapParams::map`] applies per channel (design §21).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TonemapOperator {
    /// Plain [`reinhard`]: cheap, no artist white control.
    Reinhard,
    /// [`reinhard_extended`] with an artist-chosen white luminance.
    ReinhardExtended,
    /// The Narkowicz [`aces_film`] filmic approximation.
    Aces,
}

impl TonemapOperator {
    /// A stable numeric code for hashing, `std430` packing, and round-tripping.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            TonemapOperator::Reinhard => 0,
            TonemapOperator::ReinhardExtended => 1,
            TonemapOperator::Aces => 2,
        }
    }

    /// The inverse of [`TonemapOperator::code`]; `None` for an unknown code.
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        let operator = match code {
            0 => TonemapOperator::Reinhard,
            1 => TonemapOperator::ReinhardExtended,
            2 => TonemapOperator::Aces,
            _ => return None,
        };
        Some(operator)
    }

    /// Applies this operator to a single (already exposed) linear channel,
    /// threading the `white` point through only when the operator uses it.
    #[must_use]
    pub fn apply(self, x: f32, white: f32) -> f32 {
        match self {
            TonemapOperator::Reinhard => reinhard(x),
            TonemapOperator::ReinhardExtended => reinhard_extended(x, white),
            TonemapOperator::Aces => aces_film(x),
        }
    }
}

/// The number of scalar fields packed into the `std430` block: `exposure`,
/// `white_point`, and the operator code.
const TONEMAP_FIELD_COUNT: usize = 3;

/// The parameters a `GPU` `tonemap` post pass evaluates against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TonemapParams {
    /// Linear exposure multiplier applied before the tone curve.
    pub exposure: f32,
    /// White-luminance control for [`TonemapOperator::ReinhardExtended`];
    /// ignored by the other operators.
    pub white_point: f32,
    /// Which tone curve to apply per channel.
    pub operator: TonemapOperator,
}

impl TonemapParams {
    /// Byte size of the `std430` packing: three scalars padded up to a single
    /// `vec4` slot so the block honors the 16-byte `std430` base alignment the
    /// `GPU` kernel expects.
    pub const STD430_SIZE: usize = TONEMAP_FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

    /// Builds a parameter block from its fields.
    #[must_use]
    pub const fn new(exposure: f32, white_point: f32, operator: TonemapOperator) -> Self {
        Self {
            exposure,
            white_point,
            operator,
        }
    }

    /// Runs the full `tonemap` pipeline on a linear `HDR` `RGB` triple: multiply
    /// by [`TonemapParams::exposure`], apply the [`TonemapParams::operator`] per
    /// channel (clamped into `[0, 1]`), then encode to display `gamma` via
    /// [`linear_to_srgb_approx`]. The result is always a valid `LDR` triple in
    /// `[0, 1]`.
    #[must_use]
    pub fn map(&self, rgb: [f32; 3]) -> [f32; 3] {
        let exposed = apply_exposure(rgb, self.exposure);
        let mut out = [0.0f32; 3];
        for (slot, &channel) in out.iter_mut().zip(exposed.iter()) {
            let toned = self
                .operator
                .apply(channel, self.white_point)
                .clamp(0.0, 1.0);
            *slot = linear_to_srgb_approx(toned);
        }
        out
    }

    /// Packs the parameters into their `std430` uniform-block bytes.
    ///
    /// Laid out little-endian as `exposure` (`f32`), `white_point` (`f32`), the
    /// operator code (`u32`), and one `u32` of zero padding so the block spans a
    /// single `vec4` slot ([`TonemapParams::STD430_SIZE`] bytes).
    #[must_use]
    pub fn to_std430(&self) -> [u8; Self::STD430_SIZE] {
        let mut bytes = [0u8; Self::STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.exposure.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.white_point.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.operator.code().to_le_bytes());
        bytes
    }

    /// Packs a slice of parameter blocks into one contiguous `std430` byte
    /// buffer (element stride [`TonemapParams::STD430_SIZE`]), the layout a
    /// `GPU` storage array of `tonemap` params binds.
    #[must_use]
    pub fn pack_slice(params: &[Self]) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(Self::STD430_SIZE * params.len());
        for p in params {
            buffer.extend_from_slice(&p.to_std430());
        }
        buffer
    }

    /// Total `std430` byte size of a storage buffer holding `count` packed
    /// [`TonemapParams`] blocks, clamped up to a single element per the shared
    /// [`storage_bytes`] rule.
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(Self::STD430_SIZE, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the `f32` comparisons used by the tests; direct
    /// `==` on floating point is intentionally avoided.
    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    /// A small helper to read a little-endian `f32` back out of packed bytes.
    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[offset..offset + 4]);
        f32::from_le_bytes(buf)
    }

    /// A small helper to read a little-endian `u32` back out of packed bytes.
    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[offset..offset + 4]);
        u32::from_le_bytes(buf)
    }

    #[test]
    fn reinhard_is_monotonic_and_saturates() {
        assert!(approx(reinhard(0.0), 0.0));
        // Strictly increasing across a wide input range.
        let mut previous = -1.0;
        for i in 0..=50 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small loop index maps exactly to f32"
            )]
            let x = i as f32 * 0.5;
            let y = reinhard(x);
            assert!(y > previous, "reinhard must increase: {y} <= {previous}");
            assert!(y < 1.0, "reinhard stays below one: {y}");
            previous = y;
        }
        // Large input approaches one.
        assert!(reinhard(1.0e6) > 0.999);
    }

    #[test]
    fn reinhard_extended_maps_white_to_one() {
        let white = 4.0;
        assert!(approx(reinhard_extended(white, white), 1.0));
        assert!(approx(reinhard_extended(0.0, white), 0.0));
        // Below white stays below one; above white exceeds one.
        assert!(reinhard_extended(2.0, white) < 1.0);
        assert!(reinhard_extended(8.0, white) > 1.0);
        // Monotonic across the range.
        let mut previous = -1.0;
        for i in 0..=50 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small loop index maps exactly to f32"
            )]
            let x = i as f32 * 0.25;
            let y = reinhard_extended(x, white);
            assert!(y > previous, "extended must increase: {y} <= {previous}");
            previous = y;
        }
    }

    #[test]
    fn aces_is_monotonic_bounded_black_to_black() {
        assert!(approx(aces_film(0.0), 0.0));
        let mut previous = -1.0;
        for i in 0..=100 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small loop index maps exactly to f32"
            )]
            let x = i as f32 * 0.2;
            let y = aces_film(x);
            assert!((0.0..=1.0).contains(&y), "aces stays in unit range: {y}");
            assert!(
                y >= previous - CMP_EPS,
                "aces must be non-decreasing: {y} < {previous}"
            );
            previous = y;
        }
        // Bright input saturates toward one.
        assert!(aces_film(1.0e3) > 0.99);
    }

    #[test]
    fn exposure_scales_linearly() {
        let rgb = [0.1, 0.4, 0.8];
        let scaled = apply_exposure(rgb, 2.0);
        assert!(approx(scaled[0], 0.2));
        assert!(approx(scaled[1], 0.8));
        assert!(approx(scaled[2], 1.6));
        // Doubling exposure doubles the result exactly.
        let single = apply_exposure(rgb, 1.5);
        let doubled = apply_exposure(rgb, 3.0);
        for (s, d) in single.iter().zip(doubled.iter()) {
            assert!(approx(s * 2.0, *d));
        }
    }

    #[test]
    fn exposure_from_stops_uses_powers_of_two() {
        assert!(approx(exposure_from_stops(0), 1.0));
        assert!(approx(exposure_from_stops(1), 2.0));
        assert!(approx(exposure_from_stops(3), 8.0));
        assert!(approx(exposure_from_stops(-1), 0.5));
        assert!(approx(exposure_from_stops(-2), 0.25));
        // The extreme negative stop does not panic on negation.
        assert!(exposure_from_stops(i32::MIN) > 0.0);
    }

    #[test]
    fn srgb_round_trips_within_epsilon() {
        assert!(approx(linear_to_srgb_approx(0.0), 0.0));
        assert!(approx(linear_to_srgb_approx(1.0), 1.0));
        assert!(approx(srgb_to_linear(0.0), 0.0));
        for i in 0..=20 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small loop index maps exactly to f32"
            )]
            let x = i as f32 / 20.0;
            let round = srgb_to_linear(linear_to_srgb_approx(x));
            assert!(approx(round, x), "round trip failed: {round} != {x}");
        }
    }

    #[test]
    fn srgb_encode_is_monotonic() {
        let mut previous = -1.0;
        for i in 0..=20 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small loop index maps exactly to f32"
            )]
            let x = i as f32 / 20.0;
            let y = linear_to_srgb_approx(x);
            assert!(y > previous, "encode must increase: {y} <= {previous}");
            previous = y;
        }
    }

    #[test]
    fn map_output_is_in_unit_range_for_every_operator() {
        for operator in [
            TonemapOperator::Reinhard,
            TonemapOperator::ReinhardExtended,
            TonemapOperator::Aces,
        ] {
            let params = TonemapParams::new(1.5, 4.0, operator);
            for scale in [0.0, 0.5, 2.0, 50.0, 1.0e4] {
                let out = params.map([scale, scale * 0.5, scale * 2.0]);
                for c in out {
                    assert!(
                        (0.0..=1.0).contains(&c),
                        "map output out of range for {operator:?}: {c}"
                    );
                }
            }
        }
    }

    #[test]
    fn map_is_monotonic_in_input_brightness() {
        let params = TonemapParams::new(1.0, 4.0, TonemapOperator::Aces);
        let mut previous = -1.0;
        for i in 0..=100 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small loop index maps exactly to f32"
            )]
            let luma = i as f32 * 0.1;
            let out = params.map([luma, luma, luma]);
            // Grey input keeps channels equal.
            assert!(approx(out[0], out[1]) && approx(out[1], out[2]));
            assert!(
                out[0] >= previous - CMP_EPS,
                "map must be non-decreasing: {} < {previous}",
                out[0]
            );
            previous = out[0];
        }
    }

    #[test]
    fn map_is_deterministic() {
        let params = TonemapParams::new(1.25, 3.0, TonemapOperator::ReinhardExtended);
        let rgb = [0.3, 1.7, 5.0];
        let first = params.map(rgb);
        let second = params.map(rgb);
        assert_eq!(first, second);
    }

    #[test]
    fn operator_code_round_trips() {
        for operator in [
            TonemapOperator::Reinhard,
            TonemapOperator::ReinhardExtended,
            TonemapOperator::Aces,
        ] {
            assert_eq!(TonemapOperator::from_code(operator.code()), Some(operator));
        }
        assert_eq!(TonemapOperator::from_code(99), None);
    }

    #[test]
    fn std430_layout_matches_contract() {
        let params = TonemapParams::new(1.5, 4.0, TonemapOperator::Aces);
        let bytes = params.to_std430();
        assert_eq!(bytes.len(), TonemapParams::STD430_SIZE);
        assert_eq!(TonemapParams::STD430_SIZE, VEC4_STRIDE);
        assert_eq!(TonemapParams::STD430_SIZE % VEC4_STRIDE, 0);
        // Fields decode back to their source values.
        assert!(approx(read_f32(&bytes, 0), params.exposure));
        assert!(approx(read_f32(&bytes, 4), params.white_point));
        assert_eq!(read_u32(&bytes, 8), params.operator.code());
        // Trailing padding word is zero.
        assert_eq!(read_u32(&bytes, 12), 0);
    }

    #[test]
    fn std430_storage_and_slice_packing() {
        assert_eq!(
            TonemapParams::gpu_storage_bytes(1),
            TonemapParams::STD430_SIZE
        );
        // An empty pool still reserves one element.
        assert_eq!(
            TonemapParams::gpu_storage_bytes(0),
            TonemapParams::STD430_SIZE
        );
        let params = [
            TonemapParams::new(1.0, 2.0, TonemapOperator::Reinhard),
            TonemapParams::new(2.0, 4.0, TonemapOperator::Aces),
        ];
        let packed = TonemapParams::pack_slice(&params);
        assert_eq!(packed.len(), TonemapParams::STD430_SIZE * params.len());
        assert_eq!(read_u32(&packed, 8), TonemapOperator::Reinhard.code());
        assert_eq!(
            read_u32(&packed, TonemapParams::STD430_SIZE + 8),
            TonemapOperator::Aces.code()
        );
    }
}
