//! Temporal auto-exposure / eye adaptation — the `CPU`-verifiable reference for
//! how the particle `HDR` shading front end converges its exposure over time
//! (design §16-§21).
//!
//! Production renderers do not snap exposure to the value implied by the
//! current frame's brightness; that would make the image flicker as bright and
//! dark objects sweep through view. Instead they *ease* a running "current
//! exposure" toward the frame's *target* exposure over several frames, exactly
//! as a human eye takes time to adjust when walking from sunlight into a cave.
//! Unreal's eye-adaptation pass and `Frostbite`'s auto-exposure both do this
//! with two different rates — a faster *brightening* (adapting to a brighter
//! scene) and a slower *darkening* — and clamp the result between an artist
//! `min`/`max` `EV` with an `exposure_compensation` bias on top. This module
//! owns the `CPU`-verifiable maths of that convergence contract and packs its
//! parameters into the `std430` block a future `GPU` eye-adaptation kernel
//! binds.
//!
//! # Frame-rate-independent temporal convergence
//!
//! The physically motivated law is exponential relaxation,
//! `current += (target - current) * (1 - exp(-speed * dt))`, whose blend factor
//! is independent of how the frame time `dt` is chopped up: two half-steps
//! compose into (approximately) one full step. The determinism-locked contract
//! layer forbids `exp`/`ln`/`powf`, so this file approximates that blend factor
//! with the *rational* `rate_factor(dt, speed) = 1 - 1 / (1 + speed * dt)`
//! (equivalently `speed*dt / (1 + speed*dt)`). It shares the qualitative shape
//! of `1 - exp(-speed*dt)`: it is `0` at `dt = 0` (the exposure does not move on
//! a zero-length frame), it rises monotonically toward `1` as `dt` grows (a long
//! frame lands essentially on the target), and it stays strictly inside `[0, 1]`
//! so the update is always a convex blend of `current` and `target`. A convex
//! blend can never overshoot, so convergence is monotone and the steady state
//! (`current == target`) is a genuine fixed point that never oscillates.
//!
//! # Deliberately *not* rebuilding the histogram
//!
//! This module **consumes** an already-computed average `luminance` (the mean
//! that [`super::luminance_hist`] reads out of its `log2`-`EV` histogram). It
//! does **not** build, bin, or trim a histogram, and it never recomputes the
//! average — that stage is owned end to end by [`super::luminance_hist`], and
//! duplicating it here would risk the two paths disagreeing bit for bit. Nor
//! does this file apply a tone curve (that is [`super::tonemap`]) or a bloom
//! bright-pass (that is [`super::bloom_threshold`]). Its whole job is the three
//! links in between: average `luminance` -> *target* `EV`, *target* `EV` plus a
//! previous `EV` and `dt` -> the temporally converged *current* `EV`, and an
//! `EV` -> the linear exposure scale a shader multiplies into radiance.
//!
//! # Determinism
//!
//! The `log2` needed to place a `luminance` ratio on the `EV` axis and the
//! `exp2` needed to turn an `EV` back into a linear scale are reconstructed from
//! the `IEEE`-754 field layout of `f32`: the biased exponent supplies the
//! integer part exactly and a quadratic closes the fractional octave. Every
//! other operation is add / subtract / multiply / guarded scalar divide plus
//! [`f32::floor`]. No transcendental function, no lookup table, and no
//! input-dependent iteration count, so a future `GPU` kernel reproduces the
//! `CPU` result.

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};
use alloc::vec::Vec;

/// Number of scalar `f32` fields packed into the [`ExposureAdaptConfig`]
/// `std430` block: `min_ev`, `max_ev`, `up_speed`, `down_speed`, and
/// `exposure_compensation`.
const CONFIG_FIELD_COUNT: usize = 5;

/// Denominators (and guarded divisors such as an average `luminance` or a
/// middle-grey `key`) with magnitude below this are treated as (near) zero so
/// no routine divides by zero or emits `NaN`. Relational comparisons against
/// this constant replace bare `==`/`!=` on floating-point values in the
/// non-test code.
const MIN_DENOM: f32 = 1e-6;

/// Quadratic correction coefficient for the `log2` mantissa term:
/// `log2(1 + f) ≈ f + LOG2_MANTISSA_K * f * (1 - f)` for `f` in `[0, 1)`.
///
/// The value matches the octave midpoint so the residual error stays well under
/// `0.01 EV`; powers of two (`f == 0`) are reproduced exactly.
const LOG2_MANTISSA_K: f32 = 0.346_573_6;

/// Linear coefficient of the quadratic `exp2` fractional fit
/// `2^f ≈ 1 + EXP2_FRAC_C1 * f + EXP2_FRAC_C2 * f * f` for `f` in `[0, 1)`.
///
/// The two coefficients sum to `1` so the fit is exact at the octave endpoints
/// (`f = 0 -> 1`, `f = 1 -> 2`) and matches the octave midpoint between them.
const EXP2_FRAC_C1: f32 = 0.656_854_2;

/// Quadratic coefficient of the `exp2` fractional fit (see [`EXP2_FRAC_C1`]).
const EXP2_FRAC_C2: f32 = 0.343_145_8;

/// `2^23`, the width of the `f32` mantissa field, as a float divisor.
const MANTISSA_SCALE: f32 = 8_388_608.0;

/// Sentinel `EV` returned by [`approx_log2`] for non-positive / subnormal
/// inputs, far below any physical `HDR` exposure so such degenerate ratios pin
/// the target to the darkest end after clamping.
const NEG_EV_FLOOR: f32 = -1000.0;

/// Clamps `value` into `[lo, hi]` without `f32::clamp`'s `min > max` panic and
/// without the `manual_range_contains` shape: `lo` wins on the low side, `hi`
/// on the high side, and a degenerate `lo > hi` collapses to `hi`.
#[must_use]
fn clamp(value: f32, lo: f32, hi: f32) -> f32 {
    value.max(lo).min(hi)
}

/// Integer-exponent `log2` approximation built from the `f32` bit pattern.
///
/// For a positive normal `x = (1 + f) * 2^e` the biased exponent field gives
/// `e` exactly and the mantissa fraction `f in [0, 1)` is closed with the
/// quadratic `log2(1 + f) ≈ f + LOG2_MANTISSA_K * f * (1 - f)`. Non-positive or
/// subnormal inputs return [`NEG_EV_FLOOR`] rather than `-inf`/`NaN`. No
/// `ln`/`log2` floating-point function is called.
#[must_use]
fn approx_log2(x: f32) -> f32 {
    if x <= 0.0 {
        return NEG_EV_FLOOR;
    }
    let bits = x.to_bits();
    let exp_field = i32::try_from((bits >> 23) & 0xff).unwrap_or(0);
    if exp_field == 0 {
        // Subnormal: below the smallest normal exposure ratio we ever place.
        return NEG_EV_FLOOR;
    }
    let mantissa = bits & 0x007f_ffff;
    #[expect(
        clippy::cast_precision_loss,
        reason = "mantissa < 2^23 is represented exactly in f32"
    )]
    let frac = mantissa as f32 / MANTISSA_SCALE;
    let log_mant = frac + LOG2_MANTISSA_K * frac * (1.0 - frac);
    #[expect(
        clippy::cast_precision_loss,
        reason = "an unbiased f32 exponent is in [-126, 127], exact in f32"
    )]
    let exponent = (exp_field - 127) as f32;
    exponent + log_mant
}

/// Integer-exponent `exp2` (`2^x`) approximation, the inverse of
/// [`approx_log2`], exposed as the public `EV` -> linear exposure scale.
///
/// The integer part `floor(x)` is written straight into the `f32` exponent
/// field (an integer power of two, hence exact), and the fractional octave is
/// closed with the quadratic `2^f ≈ 1 + EXP2_FRAC_C1 * f + EXP2_FRAC_C2 * f*f`.
/// Inputs beyond the representable exponent range saturate to `0.0` /
/// [`f32::MAX`] instead of producing a subnormal or infinity. No `exp`/`powf`
/// function is called: the integer power of two is materialized by a bit shift
/// into the exponent field.
#[must_use]
pub fn ev_to_exposure_scale(ev: f32) -> f32 {
    let floor = ev.floor();
    let frac = ev - floor;
    let mantissa = 1.0 + EXP2_FRAC_C1 * frac + EXP2_FRAC_C2 * frac * frac;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "floor(ev) is an integral f32; out-of-range magnitudes are handled below"
    )]
    let exponent = floor as i32;
    if exponent > 127 {
        return f32::MAX;
    }
    if exponent < -126 {
        return 0.0;
    }
    #[expect(
        clippy::cast_sign_loss,
        reason = "exponent + 127 lies in [1, 254], always non-negative"
    )]
    let field = (exponent + 127) as u32;
    let scale = f32::from_bits(field << 23);
    mantissa * scale
}

/// Frame-rate-independent temporal blend factor in `[0, 1]`.
///
/// Returns `rate = 1 - 1 / (1 + speed * dt)`, the rational stand-in for the
/// exponential relaxation weight `1 - exp(-speed * dt)`. Both `dt` and `speed`
/// are floored at `0` (time never runs backward and a negative rate is
/// meaningless), so `speed * dt >= 0`, the denominator is `>= 1`, and the
/// result lands in `[0, 1)`. A final [`clamp`] guards against rounding nudging
/// the value a hair outside the interval.
///
/// * `dt = 0` -> `0.0`: a zero-length frame does not move the exposure.
/// * `dt -> inf` (or `speed -> inf`) -> `-> 1.0`: a long frame lands on target.
#[must_use]
pub fn rate_factor(dt: f32, speed: f32) -> f32 {
    let dt = dt.max(0.0);
    let speed = speed.max(0.0);
    let product = speed * dt;
    let raw = 1.0 - 1.0 / (1.0 + product);
    clamp(raw, 0.0, 1.0)
}

/// Configuration for temporal auto-exposure / eye adaptation (design §16).
///
/// The running exposure is expressed in `EV` (`log2` stops) and confined to
/// `[min_ev, max_ev]`. Adaptation uses two rates: [`up_speed`](Self::up_speed)
/// when the exposure is climbing toward a brighter target and
/// [`down_speed`](Self::down_speed) when it is falling toward a darker one,
/// mirroring the asymmetric light/dark adaptation of the eye.
/// [`exposure_compensation`](Self::exposure_compensation) is an artist `EV`
/// bias added to the metered target before clamping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureAdaptConfig {
    /// Darkest allowed exposure, in `EV` (`log2` stops).
    pub min_ev: f32,
    /// Brightest allowed exposure, in `EV` (`log2` stops).
    pub max_ev: f32,
    /// Convergence speed while *brightening* (target above current). Larger is
    /// faster; `0` freezes the exposure. Floored at `0` in [`rate_factor`].
    pub up_speed: f32,
    /// Convergence speed while *darkening* (target below current). Larger is
    /// faster; `0` freezes the exposure. Floored at `0` in [`rate_factor`].
    pub down_speed: f32,
    /// Artist `EV` bias added to the metered target exposure before clamping.
    pub exposure_compensation: f32,
}

impl Default for ExposureAdaptConfig {
    /// A neutral photographic default: a wide `[-8, +8] EV` window, a faster
    /// brightening rate than darkening rate (the eye opens up more slowly than
    /// it stops down, so darkening reads as the slower motion here), and no
    /// exposure compensation.
    fn default() -> Self {
        Self {
            min_ev: -8.0,
            max_ev: 8.0,
            up_speed: 3.0,
            down_speed: 1.0,
            exposure_compensation: 0.0,
        }
    }
}

impl ExposureAdaptConfig {
    /// Byte size of the `std430` packing: five scalars rounded up to whole
    /// `vec4` slots so the block honors the 16-byte `std430` base alignment.
    /// Five scalars occupy two `vec4` slots (32 bytes) with a three-scalar
    /// padding tail.
    pub const STD430_SIZE: usize = CONFIG_FIELD_COUNT.div_ceil(4) * (U32_STRIDE * 4);

    /// Builds a configuration from its fields.
    #[must_use]
    pub const fn new(
        min_ev: f32,
        max_ev: f32,
        up_speed: f32,
        down_speed: f32,
        exposure_compensation: f32,
    ) -> Self {
        Self {
            min_ev,
            max_ev,
            up_speed,
            down_speed,
            exposure_compensation,
        }
    }

    /// Returns the ordered exposure window `(lo, hi)` with `lo <= hi`, so a
    /// mis-ordered `min_ev > max_ev` never makes a downstream [`clamp`] behave
    /// oddly.
    #[must_use]
    fn ev_window(&self) -> (f32, f32) {
        let lo = self.min_ev.min(self.max_ev);
        let hi = self.min_ev.max(self.max_ev);
        (lo, hi)
    }

    /// Clamps an `EV` into this configuration's `[min_ev, max_ev]` window.
    #[must_use]
    pub fn clamp_ev(&self, ev: f32) -> f32 {
        let (lo, hi) = self.ev_window();
        clamp(ev, lo, hi)
    }

    /// The metered *target* exposure, in `EV`, for a frame whose average
    /// `luminance` is `avg_luminance` against a middle-grey `key`.
    ///
    /// The exposure scale that maps the average onto the key is `key /
    /// avg_luminance`; its `EV` is `log2(key / avg_luminance)`. This module does
    /// **not** compute `avg_luminance` — that mean is read out of
    /// [`super::luminance_hist`]'s histogram and handed in. A brighter frame
    /// (larger `avg_luminance`) yields a smaller ratio and therefore a lower
    /// (more negative) target `EV`, i.e. the exposure stops down. The artist
    /// [`exposure_compensation`](Self::exposure_compensation) is added and the
    /// result is clamped into `[min_ev, max_ev]`. Both `avg_luminance` and `key`
    /// are floored at [`MIN_DENOM`] so the ratio is finite and positive.
    #[must_use]
    pub fn target_exposure(&self, avg_luminance: f32, key: f32) -> f32 {
        let avg = avg_luminance.max(MIN_DENOM);
        let key = key.max(MIN_DENOM);
        let metered = approx_log2(key / avg) + self.exposure_compensation;
        self.clamp_ev(metered)
    }

    /// Advances the running exposure one temporal step.
    ///
    /// Returns `current_ev + (target_ev - current_ev) * rate`, where `rate` is
    /// the frame-rate-independent [`rate_factor`] built from `dt` and the
    /// direction-appropriate speed: [`up_speed`](Self::up_speed) when the
    /// exposure climbs toward a brighter target (`target_ev > current_ev`) and
    /// [`down_speed`](Self::down_speed) when it falls toward a darker one. The
    /// blend factor lies in `[0, 1]`, so the step is a convex combination that
    /// moves toward the target without overshooting; the returned `EV` is
    /// clamped into `[min_ev, max_ev]`. With `dt = 0` the exposure does not
    /// move, and once `current_ev == target_ev` the step is a fixed point.
    #[must_use]
    pub fn adapt_step(&self, current_ev: f32, target_ev: f32, dt: f32) -> f32 {
        let speed = if target_ev > current_ev {
            self.up_speed
        } else {
            self.down_speed
        };
        let rate = rate_factor(dt, speed);
        let next = current_ev + (target_ev - current_ev) * rate;
        self.clamp_ev(next)
    }

    /// Convenience: meter the target for `avg_luminance`/`key` and immediately
    /// take one [`adapt_step`](Self::adapt_step) from `current_ev`, returning
    /// the new running exposure `EV`. Equivalent to
    /// `adapt_step(current_ev, target_exposure(avg_luminance, key), dt)`.
    #[must_use]
    pub fn adapt_toward_luminance(
        &self,
        current_ev: f32,
        avg_luminance: f32,
        key: f32,
        dt: f32,
    ) -> f32 {
        let target = self.target_exposure(avg_luminance, key);
        self.adapt_step(current_ev, target, dt)
    }

    /// Packs the configuration into its `std430` uniform-block bytes.
    ///
    /// Laid out little-endian as `min_ev`, `max_ev`, `up_speed`, `down_speed`,
    /// `exposure_compensation` (five `f32`), followed by three `f32` of zero
    /// padding so the block spans two `vec4` slots
    /// ([`ExposureAdaptConfig::STD430_SIZE`] bytes).
    #[must_use]
    pub fn to_std430(&self) -> [u8; Self::STD430_SIZE] {
        let mut bytes = [0u8; Self::STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.min_ev.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.max_ev.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.up_speed.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.down_speed.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.exposure_compensation.to_le_bytes());
        bytes
    }

    /// Packs a slice of configuration blocks into one contiguous `std430` byte
    /// buffer (element stride [`ExposureAdaptConfig::STD430_SIZE`]), the layout
    /// a `GPU` storage array of eye-adaptation configs binds.
    #[must_use]
    pub fn pack_slice(configs: &[Self]) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(Self::STD430_SIZE * configs.len());
        for c in configs {
            buffer.extend_from_slice(&c.to_std430());
        }
        buffer
    }

    /// Total `std430` byte size of a storage buffer holding `count` packed
    /// [`ExposureAdaptConfig`] blocks, clamped up to a single element per the
    /// shared [`storage_bytes`] rule.
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(Self::STD430_SIZE, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparing floating-point results; replaces the
    /// forbidden bare `==`/`!=` on `f32` in the assertions below.
    const CMP_EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn cfg() -> ExposureAdaptConfig {
        ExposureAdaptConfig::new(-8.0, 8.0, 3.0, 1.0, 0.0)
    }

    #[test]
    fn rate_factor_zero_dt_is_zero() {
        assert!(approx_eq(rate_factor(0.0, 5.0), 0.0, CMP_EPS));
    }

    #[test]
    fn rate_factor_large_dt_approaches_one() {
        let r = rate_factor(1.0e6, 4.0);
        assert!(r < 1.0);
        assert!(r > 1.0 - 1.0e-5);
    }

    #[test]
    fn rate_factor_is_monotone_in_dt() {
        let mut prev = rate_factor(0.0, 2.0);
        for step in 1..64 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "small loop index is exact in f32"
            )]
            let dt = step as f32 * 0.05;
            let r = rate_factor(dt, 2.0);
            assert!(r > prev);
            assert!(r >= 0.0);
            assert!(r <= 1.0);
            prev = r;
        }
    }

    #[test]
    fn rate_factor_clamps_negative_inputs_to_zero() {
        assert!(approx_eq(rate_factor(-1.0, 5.0), 0.0, CMP_EPS));
        assert!(approx_eq(rate_factor(1.0, -5.0), 0.0, CMP_EPS));
    }

    #[test]
    fn adapt_step_moves_toward_target() {
        let c = cfg();
        let current = 0.0;
        let target = 4.0;
        let next = c.adapt_step(current, target, 0.1);
        // Strictly between the start and the target: moved, but no overshoot.
        assert!(next > current);
        assert!(next < target);
    }

    #[test]
    fn adapt_step_converges_monotonically() {
        let c = cfg();
        let target: f32 = 5.0;
        let mut current: f32 = -3.0;
        let mut prev_gap = (target - current).abs();
        for _ in 0..200 {
            current = c.adapt_step(current, target, 0.05);
            let gap = (target - current).abs();
            // Gap never grows and never crosses the target: a monotone,
            // overshoot-free approach that flattens out at the float limit.
            assert!(gap <= prev_gap);
            assert!(current <= target);
            prev_gap = gap;
        }
        assert!(approx_eq(current, target, 1.0e-3));
    }

    #[test]
    fn adapt_step_up_and_down_use_different_speeds() {
        // up_speed (3.0) is faster than down_speed (1.0): a brightening step of
        // a given gap covers more ground than a darkening step of the same gap.
        let c = cfg();
        let dt = 0.1;
        let up_move = c.adapt_step(0.0, 2.0, dt) - 0.0;
        let down_move = 0.0 - c.adapt_step(0.0, -2.0, dt);
        assert!(up_move > down_move);
    }

    #[test]
    fn adapt_step_zero_dt_does_not_move() {
        let c = cfg();
        let current = 1.25;
        let next = c.adapt_step(current, 6.0, 0.0);
        assert!(approx_eq(next, current, CMP_EPS));
    }

    #[test]
    fn adapt_step_large_dt_lands_near_target() {
        let c = cfg();
        let target = 3.5;
        let next = c.adapt_step(-2.0, target, 1.0e6);
        assert!(approx_eq(next, target, 1.0e-3));
    }

    #[test]
    fn adapt_step_steady_state_is_a_fixed_point() {
        let c = cfg();
        let mut current = 2.0;
        for _ in 0..50 {
            let next = c.adapt_step(current, 2.0, 0.1);
            assert!(approx_eq(next, current, CMP_EPS));
            current = next;
        }
    }

    #[test]
    fn target_exposure_clamps_to_min_and_max() {
        let c = cfg();
        // A pitch-dark frame wants a huge positive EV, clamped to max_ev.
        let bright_target = c.target_exposure(1.0e-4, 0.18);
        assert!(approx_eq(bright_target, c.max_ev, CMP_EPS));
        // A blown-out frame wants a hugely negative EV, clamped to min_ev.
        let dark_target = c.target_exposure(1.0e4, 0.18);
        assert!(approx_eq(dark_target, c.min_ev, CMP_EPS));
    }

    #[test]
    fn target_exposure_is_monotone_in_luminance() {
        // Wider window so neither sample saturates the clamp.
        let c = ExposureAdaptConfig::new(-30.0, 30.0, 3.0, 1.0, 0.0);
        let dim = c.target_exposure(0.05, 0.18);
        let bright = c.target_exposure(0.8, 0.18);
        // Brighter average -> lower (more negative) target exposure.
        assert!(bright < dim);
    }

    #[test]
    fn target_exposure_key_match_is_zero_ev() {
        let c = ExposureAdaptConfig::new(-30.0, 30.0, 3.0, 1.0, 0.0);
        // average == key -> ratio 1 -> log2(1) == 0 EV.
        let ev = c.target_exposure(0.18, 0.18);
        assert!(approx_eq(ev, 0.0, 1.0e-3));
    }

    #[test]
    fn exposure_compensation_shifts_target() {
        let base = ExposureAdaptConfig::new(-30.0, 30.0, 3.0, 1.0, 0.0);
        let plus = ExposureAdaptConfig::new(-30.0, 30.0, 3.0, 1.0, 2.0);
        let ev0 = base.target_exposure(0.18, 0.18);
        let ev1 = plus.target_exposure(0.18, 0.18);
        assert!(approx_eq(ev1 - ev0, 2.0, 1.0e-3));
    }

    #[test]
    fn ev_to_exposure_scale_exact_powers_of_two() {
        assert!(approx_eq(ev_to_exposure_scale(0.0), 1.0, CMP_EPS));
        assert!(approx_eq(ev_to_exposure_scale(1.0), 2.0, CMP_EPS));
        assert!(approx_eq(ev_to_exposure_scale(2.0), 4.0, CMP_EPS));
        assert!(approx_eq(ev_to_exposure_scale(-1.0), 0.5, CMP_EPS));
        assert!(approx_eq(ev_to_exposure_scale(-2.0), 0.25, CMP_EPS));
    }

    #[test]
    fn ev_to_exposure_scale_saturates_out_of_range() {
        assert!(approx_eq(ev_to_exposure_scale(200.0), f32::MAX, CMP_EPS));
        assert!(approx_eq(ev_to_exposure_scale(-200.0), 0.0, CMP_EPS));
    }

    #[test]
    fn log2_exp2_round_trip_within_tolerance() {
        // approx_log2 then ev_to_exposure_scale should recover the input to a
        // small relative tolerance across a few octaves.
        let samples = [0.05f32, 0.18, 0.5, 1.0, 2.0, 7.5, 30.0];
        for &x in &samples {
            let ev = approx_log2(x);
            let back = ev_to_exposure_scale(ev);
            let rel = (back - x).abs() / x;
            assert!(rel < 1.5e-2);
        }
    }

    #[test]
    fn approx_log2_is_exact_on_powers_of_two() {
        assert!(approx_eq(approx_log2(1.0), 0.0, CMP_EPS));
        assert!(approx_eq(approx_log2(2.0), 1.0, CMP_EPS));
        assert!(approx_eq(approx_log2(4.0), 2.0, CMP_EPS));
        assert!(approx_eq(approx_log2(0.5), -1.0, CMP_EPS));
        // Non-positive inputs pin to the darkest sentinel.
        assert!(approx_eq(approx_log2(0.0), NEG_EV_FLOOR, CMP_EPS));
        assert!(approx_eq(approx_log2(-3.0), NEG_EV_FLOOR, CMP_EPS));
    }

    #[test]
    fn std430_layout_size_and_bytes() {
        assert_eq!(ExposureAdaptConfig::STD430_SIZE, 32);
        let c = ExposureAdaptConfig::new(-6.0, 5.0, 2.5, 0.75, 1.5);
        let bytes = c.to_std430();
        assert_eq!(bytes.len(), ExposureAdaptConfig::STD430_SIZE);
        assert_eq!(&bytes[0..4], &(-6.0f32).to_le_bytes());
        assert_eq!(&bytes[4..8], &5.0f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &2.5f32.to_le_bytes());
        assert_eq!(&bytes[12..16], &0.75f32.to_le_bytes());
        assert_eq!(&bytes[16..20], &1.5f32.to_le_bytes());
        // The three-scalar tail is zero padding.
        assert_eq!(&bytes[20..32], &[0u8; 12]);
    }

    #[test]
    fn pack_slice_and_storage_bytes_agree() {
        let c = ExposureAdaptConfig::default();
        let packed = ExposureAdaptConfig::pack_slice(&[c, c, c]);
        assert_eq!(packed.len(), ExposureAdaptConfig::STD430_SIZE * 3);
        assert_eq!(
            ExposureAdaptConfig::gpu_storage_bytes(3),
            ExposureAdaptConfig::STD430_SIZE * 3
        );
        // Empty pool still reserves one element per the shared storage rule.
        assert_eq!(
            ExposureAdaptConfig::gpu_storage_bytes(0),
            ExposureAdaptConfig::STD430_SIZE
        );
    }

    #[test]
    fn adapt_toward_luminance_matches_two_step_composition() {
        let c = cfg();
        let target = c.target_exposure(0.02, 0.18);
        let combined = c.adapt_toward_luminance(0.0, 0.02, 0.18, 0.1);
        let manual = c.adapt_step(0.0, target, 0.1);
        assert!(approx_eq(combined, manual, CMP_EPS));
    }

    #[test]
    fn cmp_eps_flags_near_and_far_values() {
        assert!(approx_eq(1.0, 1.0 + CMP_EPS * 0.5, CMP_EPS));
        assert!(!approx_eq(1.0, 1.0 + CMP_EPS * 10.0, CMP_EPS));
    }
}
