//! ASC CDL and lift / gamma / gain colour-grading golden references.
//!
//! This module implements the two primary per-channel grading transfers every
//! colourist stack exposes, as backend-neutral CPU golden references:
//!
//! * **ASC CDL** (American Society of Cinematographers Color Decision List) — the
//!   standardised Slope / Offset / Power (SOP) transfer `out = (in * slope +
//!   offset) ^ power`. It is the interchange format that travels through the
//!   film pipeline, so matching it exactly keeps the engine's grade identical to
//!   the one authored in a dailies or DI suite.
//! * **Lift / Gamma / Gain** — the three colour wheels of a `DaVinci`-style
//!   corrector. *Lift* raises the shadow floor while pinning white, *Gain*
//!   scales the highlight ceiling while pinning black, and *Gamma* is the
//!   midtone power. The neutral `lift = 0`, `gamma = 1`, `gain = 1` is the
//!   identity.
//!
//! Both transfers are per-channel: the same scalar maths is applied to R, G and
//! B independently, so the [`bevy_math::Vec3`] variants are thin wrappers over
//! the scalar [`asc_cdl_channel`] / [`lift_gamma_gain_channel`] kernels.
//!
//! # Conventions
//! * Pure deterministic `f32` maths — no RNG, IO, GPU, `unsafe` or allocation.
//! * The only transcendental is the power curve, taken via
//!   [`bevy_math::ops::powf`]; negative bases are clamped to `0` first because
//!   `powf` of a negative base with a fractional exponent is `NaN`.
//! * Every exponent (`power`, `1 / gamma`) is forced non-negative and finite,
//!   and `gamma` is floored away from zero, so no input can produce `NaN` or an
//!   infinity.
//! * Mirrored arm-for-arm by the GPU twin; all maths stays in `f32`.

use bevy_math::Vec3;
use bevy_math::ops;

/// Smallest magnitude a `gamma` divisor is allowed to take before the midtone
/// power `1 / gamma` is formed, so a zero or near-zero gamma cannot explode the
/// exponent to infinity.
const MIN_GAMMA: f32 = 1.0e-4;

/// Largest exponent fed to [`ops::powf`]. Clamping both the CDL `power` and the
/// `1 / gamma` midtone exponent keeps the curve finite for any user input.
const MAX_EXPONENT: f32 = 1.0e4;

/// Sanitise an exponent into the finite, non-negative range `powf` can evaluate
/// without overflowing to infinity or returning `NaN`.
///
/// Non-finite inputs collapse to the identity exponent `1`.
#[must_use]
fn sanitize_exponent(exponent: f32) -> f32 {
    if exponent.is_finite() {
        exponent.clamp(0.0, MAX_EXPONENT)
    } else {
        1.0
    }
}

/// Raise a non-negative base to an exponent with full defensive clamping.
///
/// The base is clamped to `>= 0` so a negative value can never reach `powf`
/// (which would yield `NaN` for a fractional exponent), and the result is
/// clamped back to a finite non-negative number.
#[must_use]
fn safe_powf(base: f32, exponent: f32) -> f32 {
    let b = if base.is_finite() { base.max(0.0) } else { 0.0 };
    let e = sanitize_exponent(exponent);
    let r = ops::powf(b, e);
    if r.is_finite() { r.max(0.0) } else { 0.0 }
}

// --- ASC CDL --------------------------------------------------------------

/// Single-channel ASC CDL transfer `out = (in * slope + offset) ^ power`.
///
/// This is the SOP (Slope / Offset / Power) kernel standardised by the ASC. The
/// intermediate `in * slope + offset` is clamped to `>= 0` before the power so
/// the curve is defined for every input, matching the reference ASC behaviour
/// of clipping sub-black values at the power stage.
///
/// The identity parameters `slope = 1`, `offset = 0`, `power = 1` return the
/// input unchanged. With `slope > 0` and `power > 0` the transfer is monotonic
/// non-decreasing in `color`.
#[must_use]
pub fn asc_cdl_channel(color: f32, slope: f32, offset: f32, power: f32) -> f32 {
    let s = if slope.is_finite() { slope } else { 1.0 };
    let o = if offset.is_finite() { offset } else { 0.0 };
    let c = if color.is_finite() { color } else { 0.0 };
    let graded = c * s + o;
    safe_powf(graded, power)
}

/// Per-channel ASC CDL transfer applied to an RGB triple.
///
/// Each channel is graded independently by [`asc_cdl_channel`] with its own
/// slope, offset and power. The identity `slope = (1,1,1)`, `offset = (0,0,0)`,
/// `power = (1,1,1)` returns `color` unchanged.
#[must_use]
pub fn asc_cdl(color: Vec3, slope: Vec3, offset: Vec3, power: Vec3) -> Vec3 {
    Vec3::new(
        asc_cdl_channel(color.x, slope.x, offset.x, power.x),
        asc_cdl_channel(color.y, slope.y, offset.y, power.y),
        asc_cdl_channel(color.z, slope.z, offset.z, power.z),
    )
}

/// Per-channel ASC CDL transfer driven by scalar SOP parameters shared across
/// all three channels (the common "achromatic" grade).
///
/// Equivalent to [`asc_cdl`] with each parameter broadcast to all channels.
#[must_use]
pub fn asc_cdl_uniform(color: Vec3, slope: f32, offset: f32, power: f32) -> Vec3 {
    asc_cdl(color, Vec3::splat(slope), Vec3::splat(offset), Vec3::splat(power))
}

/// ASC CDL saturation stage applied *after* the SOP transfer.
///
/// The ASC CDL spec adds a single scalar saturation that mixes each channel
/// toward the Rec. 709 luma: `out = luma + sat * (in - luma)`. `sat = 1` is the
/// identity, `sat = 0` is fully desaturated to luma. Negative saturation is
/// clamped to `0` and the result is clamped to `>= 0`.
#[must_use]
pub fn asc_cdl_saturation(color: Vec3, saturation: f32) -> Vec3 {
    let sat = if saturation.is_finite() { saturation.max(0.0) } else { 1.0 };
    let luma = rec709_luma(color);
    let out = Vec3::splat(luma) + (color - Vec3::splat(luma)) * sat;
    Vec3::new(out.x.max(0.0), out.y.max(0.0), out.z.max(0.0))
}

/// Full ASC CDL node: SOP transfer followed by the scalar saturation stage, in
/// the canonical ASC order.
#[must_use]
pub fn asc_cdl_full(
    color: Vec3,
    slope: Vec3,
    offset: Vec3,
    power: Vec3,
    saturation: f32,
) -> Vec3 {
    asc_cdl_saturation(asc_cdl(color, slope, offset, power), saturation)
}

/// Rec. 709 luma weights (linear `sRGB` primaries), used by the ASC CDL
/// saturation stage.
pub const REC709_LUMA_WEIGHTS: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// Rec. 709 relative luma of a linear RGB sample.
#[must_use]
pub fn rec709_luma(color: Vec3) -> f32 {
    color.dot(REC709_LUMA_WEIGHTS)
}

// --- Lift / Gamma / Gain --------------------------------------------------

/// Single-channel lift / gamma / gain transfer.
///
/// The three colour wheels are composed in the canonical order:
///
/// 1. **Lift** raises the shadows while pinning white: `l = in + lift * (1 -
///    in)`, so `in = 0` maps to `lift` and `in = 1` stays at `1`.
/// 2. **Gain** scales the highlights while pinning black: `g = l * gain`.
/// 3. **Gamma** applies the midtone power: `out = g ^ (1 / gamma)`.
///
/// The neutral `lift = 0`, `gamma = 1`, `gain = 1` is the exact identity. The
/// base of the power is clamped to `>= 0` and `gamma` is floored at
/// [`MIN_GAMMA`] so the transfer is always finite.
#[must_use]
pub fn lift_gamma_gain_channel(color: f32, lift: f32, gamma: f32, gain: f32) -> f32 {
    let c = if color.is_finite() { color } else { 0.0 };
    let lift = if lift.is_finite() { lift } else { 0.0 };
    let gain = if gain.is_finite() { gain } else { 1.0 };
    let gamma = if gamma.is_finite() { gamma.max(MIN_GAMMA) } else { 1.0 };

    let lifted = c + lift * (1.0 - c);
    let gained = lifted * gain;
    safe_powf(gained, 1.0 / gamma)
}

/// Per-channel lift / gamma / gain transfer applied to an RGB triple.
///
/// Each channel is graded independently by [`lift_gamma_gain_channel`]. The
/// neutral `lift = (0,0,0)`, `gamma = (1,1,1)`, `gain = (1,1,1)` returns `color`
/// unchanged.
#[must_use]
pub fn lift_gamma_gain(color: Vec3, lift: Vec3, gamma: Vec3, gain: Vec3) -> Vec3 {
    Vec3::new(
        lift_gamma_gain_channel(color.x, lift.x, gamma.x, gain.x),
        lift_gamma_gain_channel(color.y, lift.y, gamma.y, gain.y),
        lift_gamma_gain_channel(color.z, lift.z, gamma.z, gain.z),
    )
}

/// Lift / gamma / gain driven by scalar parameters shared across all channels.
#[must_use]
pub fn lift_gamma_gain_uniform(color: Vec3, lift: f32, gamma: f32, gain: f32) -> Vec3 {
    lift_gamma_gain(
        color,
        Vec3::splat(lift),
        Vec3::splat(gamma),
        Vec3::splat(gain),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    fn approx3(a: Vec3, b: Vec3) {
        approx(a.x, b.x);
        approx(a.y, b.y);
        approx(a.z, b.z);
    }

    #[test]
    fn asc_cdl_identity_params_are_identity_map() {
        let one = Vec3::ONE;
        let zero = Vec3::ZERO;
        for &c in &[0.0_f32, 0.18, 0.5, 0.9, 1.0, 4.0] {
            let color = Vec3::splat(c);
            approx3(asc_cdl(color, one, zero, one), color);
        }
    }

    #[test]
    fn asc_cdl_channel_identity() {
        for &c in &[0.0_f32, 0.25, 0.5, 1.0, 3.0] {
            approx(asc_cdl_channel(c, 1.0, 0.0, 1.0), c);
        }
    }

    #[test]
    fn asc_cdl_matches_closed_form() {
        // (0.5 * 1.5 + 0.1) ^ 2.0 = 0.85^2 = 0.7225.
        approx(asc_cdl_channel(0.5, 1.5, 0.1, 2.0), 0.7225);
    }

    #[test]
    fn asc_cdl_is_monotonic_for_positive_slope_power() {
        let slope = Vec3::splat(1.3);
        let offset = Vec3::splat(-0.05);
        let power = Vec3::splat(1.8);
        let mut prev = -1.0_f32;
        let mut x = 0.0_f32;
        while x <= 1.0 {
            let v = asc_cdl(Vec3::splat(x), slope, offset, power).x;
            assert!(v >= prev - EPS, "not monotonic at {x}: {v} < {prev}");
            prev = v;
            x += 0.05;
        }
    }

    #[test]
    fn asc_cdl_clamps_negative_base() {
        // in * slope + offset = -0.5 -> clamped to 0 before the power.
        approx(asc_cdl_channel(0.0, 1.0, -0.5, 2.0), 0.0);
        let out = asc_cdl(Vec3::ZERO, Vec3::ONE, Vec3::splat(-1.0), Vec3::splat(0.5));
        approx3(out, Vec3::ZERO);
    }

    #[test]
    fn asc_cdl_never_nan() {
        // Negative base with fractional power would be NaN without clamping.
        let out = asc_cdl(
            Vec3::splat(-1.0),
            Vec3::splat(2.0),
            Vec3::splat(-3.0),
            Vec3::splat(0.5),
        );
        assert!(out.x.is_finite() && out.y.is_finite() && out.z.is_finite());
    }

    #[test]
    fn asc_cdl_saturation_neutral_is_identity() {
        let color = Vec3::new(0.2, 0.6, 0.9);
        approx3(asc_cdl_saturation(color, 1.0), color);
    }

    #[test]
    fn asc_cdl_saturation_zero_collapses_to_luma() {
        let color = Vec3::new(0.2, 0.6, 0.9);
        let luma = rec709_luma(color);
        approx3(asc_cdl_saturation(color, 0.0), Vec3::splat(luma));
    }

    #[test]
    fn asc_cdl_uniform_matches_broadcast() {
        let color = Vec3::new(0.1, 0.4, 0.8);
        let a = asc_cdl_uniform(color, 1.2, 0.03, 1.4);
        let b = asc_cdl(
            color,
            Vec3::splat(1.2),
            Vec3::splat(0.03),
            Vec3::splat(1.4),
        );
        approx3(a, b);
    }

    #[test]
    fn asc_cdl_full_identity() {
        let color = Vec3::new(0.3, 0.5, 0.7);
        approx3(
            asc_cdl_full(color, Vec3::ONE, Vec3::ZERO, Vec3::ONE, 1.0),
            color,
        );
    }

    #[test]
    fn lgg_identity_params_are_identity_map() {
        for &c in &[0.0_f32, 0.18, 0.5, 0.9, 1.0] {
            let color = Vec3::splat(c);
            approx3(
                lift_gamma_gain(color, Vec3::ZERO, Vec3::ONE, Vec3::ONE),
                color,
            );
        }
    }

    #[test]
    fn lgg_channel_identity() {
        for &c in &[0.0_f32, 0.25, 0.5, 1.0] {
            approx(lift_gamma_gain_channel(c, 0.0, 1.0, 1.0), c);
        }
    }

    #[test]
    fn lgg_lift_pins_white_and_raises_black() {
        // Pure lift: black -> lift, white -> 1.
        approx(lift_gamma_gain_channel(0.0, 0.2, 1.0, 1.0), 0.2);
        approx(lift_gamma_gain_channel(1.0, 0.2, 1.0, 1.0), 1.0);
    }

    #[test]
    fn lgg_gain_pins_black_and_scales_white() {
        // Pure gain: black -> 0, white -> gain.
        approx(lift_gamma_gain_channel(0.0, 0.0, 1.0, 1.5), 0.0);
        approx(lift_gamma_gain_channel(1.0, 0.0, 1.0, 1.5), 1.5);
    }

    #[test]
    fn lgg_gamma_matches_power() {
        // Pure gamma on 0.5 with gamma = 2 -> 0.5 ^ 0.5 = sqrt(0.5).
        approx(lift_gamma_gain_channel(0.5, 0.0, 2.0, 1.0), 0.5_f32.sqrt());
    }

    #[test]
    fn lgg_is_monotonic() {
        let mut prev = -1.0_f32;
        let mut x = 0.0_f32;
        while x <= 1.0 {
            let v = lift_gamma_gain_channel(x, 0.1, 1.5, 1.2);
            assert!(v >= prev - EPS, "not monotonic at {x}");
            prev = v;
            x += 0.05;
        }
    }

    #[test]
    fn lgg_handles_zero_gamma_without_nan() {
        let v = lift_gamma_gain_channel(0.5, 0.0, 0.0, 1.0);
        assert!(v.is_finite());
    }

    #[test]
    fn lgg_uniform_matches_broadcast() {
        let color = Vec3::new(0.2, 0.5, 0.8);
        let a = lift_gamma_gain_uniform(color, 0.05, 1.3, 1.1);
        let b = lift_gamma_gain(
            color,
            Vec3::splat(0.05),
            Vec3::splat(1.3),
            Vec3::splat(1.1),
        );
        approx3(a, b);
    }
}
