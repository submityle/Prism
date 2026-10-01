//! Grain blending curves and ISO→strength mapping — CPU golden reference.
//!
//! Having a zero-mean grain field (from [`super::grain`]) is only half the
//! story: it must be *composited* into the image in a way that (a) never pushes
//! a pixel outside the displayable `[0, 1]` range and (b) respects the signal,
//! so grain modulates the picture rather than simply adding a flat haze. This
//! module provides two such composites plus the photographic ISO→strength law:
//!
//! * [`overlay`] — the classic "overlay" blend, a soft-light-like S-curve that
//!   is contrast-preserving: a neutral grain of `0.5` is an exact identity and
//!   the `0` / `1` endpoints are fixed, so results stay in `[0, 1]`.
//! * [`apply_grain`] — overlay-composites a signed grain into a colour, lerped
//!   by `strength`; the deviation from the base grows monotonically with
//!   `strength`.
//! * [`gain_blend`] — a signal-dependent *multiplicative* composite where the
//!   grain amplitude scales with the signal (brighter areas take proportionally
//!   more grain), another physically-motivated option.
//! * [`iso_to_strength`] — a logarithmic ISO→strength curve: strength rises
//!   monotonically from `0` at base ISO toward `1` at the top of the range.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, IO, GPU, global state, or `unsafe`.
//! * `base` colours / signals are sanitized to `[0, 1]`; outputs stay in
//!   `[0, 1]`.
//! * `grain` is a signed perturbation in `[-1, 1]`; `grain = 0` is a strict
//!   identity for every composite.
//! * `strength` is clamped to `[0, 1]`; `strength = 0` is a strict identity.
//! * The only transcendental ([`iso_to_strength`]) goes through
//!   [`bevy_math::ops`].

use bevy_math::{ops, Vec3};

/// Lowest ISO of the strength ramp (strength `0`).
pub const ISO_MIN: f32 = 100.0;

/// Highest ISO of the strength ramp (strength `1`).
pub const ISO_MAX: f32 = 12_800.0;

/// Replaces a non-finite scalar with `fallback`.
#[inline]
#[must_use]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Sanitizes a scalar to `[0, 1]`, mapping non-finite input to `0`.
#[inline]
#[must_use]
fn unit(x: f32) -> f32 {
    finite_or(x, 0.0).clamp(0.0, 1.0)
}

/// Converts a signed grain in `[-1, 1]` to an overlay operand in `[0, 1]`,
/// where `0.5` is the neutral (identity) value.
#[inline]
#[must_use]
fn grain_to_operand(grain: f32) -> f32 {
    0.5 + 0.5 * finite_or(grain, 0.0).clamp(-1.0, 1.0)
}

/// Photoshop-style **overlay** blend of a base and a blend value, both in
/// `[0, 1]`, returning a value in `[0, 1]`.
///
/// `overlay(b, g) = 2·b·g` for `b < 0.5`, else `1 - 2·(1-b)·(1-g)`. This is a
/// contrast-preserving S-curve: `g = 0.5` is an exact identity, and the `b = 0`
/// / `b = 1` endpoints are fixed, which is precisely what keeps grained results
/// inside the displayable range.
#[must_use]
pub fn overlay(base: f32, blend: f32) -> f32 {
    let b = unit(base);
    let g = unit(blend);
    let out = if b < 0.5 {
        2.0 * b * g
    } else {
        1.0 - 2.0 * (1.0 - b) * (1.0 - g)
    };
    out.clamp(0.0, 1.0)
}

/// Overlay-composites a signed scalar `grain` into a single `base` channel,
/// linearly blended by `strength`.
///
/// Returns `lerp(base, overlay(base, operand), strength)` where `operand`
/// recentres the signed grain so `grain = 0` is neutral. Because the overlaid
/// value does not depend on `strength`, the deviation from `base` is
/// `strength · |overlay - base|`, i.e. monotonically non-decreasing in
/// `strength`. The result stays in `[0, 1]`.
#[must_use]
pub fn overlay_grain(base: f32, grain: f32, strength: f32) -> f32 {
    let b = unit(base);
    let s = unit(strength);
    let overlaid = overlay(b, grain_to_operand(grain));
    (b + s * (overlaid - b)).clamp(0.0, 1.0)
}

/// Overlay-composites a single scalar `grain` into an RGB `color`.
///
/// The same monochrome grain modulates all three channels (film grain is a
/// luminance phenomenon, not a per-channel one), via [`overlay_grain`]. The
/// result is clamped to `[0, 1]` per channel. `strength = 0` and `grain = 0`
/// are both strict identities.
#[must_use]
pub fn apply_grain(color: Vec3, grain: f32, strength: f32) -> Vec3 {
    Vec3::new(
        overlay_grain(color.x, grain, strength),
        overlay_grain(color.y, grain, strength),
        overlay_grain(color.z, grain, strength),
    )
}

/// Signal-dependent **multiplicative** grain composite for one channel.
///
/// The grain amplitude scales with the signal: `base + strength · grain · base`,
/// so a black pixel is untouched and brighter pixels take proportionally more
/// grain (the behaviour of multiplicative sensor/grain gain). Clamped to
/// `[0, 1]`; the deviation grows monotonically with `strength`.
#[must_use]
pub fn gain_blend(base: f32, grain: f32, strength: f32) -> f32 {
    let b = unit(base);
    let g = finite_or(grain, 0.0).clamp(-1.0, 1.0);
    let s = unit(strength);
    (b + s * g * b).clamp(0.0, 1.0)
}

/// Signal-dependent multiplicative grain composite for an RGB colour.
///
/// Applies [`gain_blend`] per channel with a shared monochrome grain.
#[must_use]
pub fn apply_gain_blend(color: Vec3, grain: f32, strength: f32) -> Vec3 {
    Vec3::new(
        gain_blend(color.x, grain, strength),
        gain_blend(color.y, grain, strength),
        gain_blend(color.z, grain, strength),
    )
}

/// Maps an ISO value to a grain strength in `[0, 1]` on a logarithmic ramp.
///
/// `strength = clamp(ln(iso / ISO_MIN) / ln(ISO_MAX / ISO_MIN), 0, 1)`. ISO is
/// a logarithmic (stops-based) quantity, so a log ramp gives perceptually even
/// steps: strength is `0` at or below [`ISO_MIN`], rises monotonically, and
/// saturates to `1` at or above [`ISO_MAX`]. Non-finite ISO yields `0`.
#[must_use]
pub fn iso_to_strength(iso: f32) -> f32 {
    let i = finite_or(iso, ISO_MIN).max(ISO_MIN);
    let num = ops::ln(i / ISO_MIN);
    let den = ops::ln(ISO_MAX / ISO_MIN);
    (num / den).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Overlay is an identity at the neutral blend `0.5`, fixes the `0` / `1`
    /// base endpoints, and always returns a value in `[0, 1]`.
    #[test]
    fn overlay_neutral_endpoints_and_range() {
        for i in 0..=20 {
            let b = i as f32 / 20.0;
            assert!((overlay(b, 0.5) - b).abs() < 1.0e-6, "neutral failed at b={b}");
        }
        for i in 0..=20 {
            let g = i as f32 / 20.0;
            assert_eq!(overlay(0.0, g), 0.0, "black endpoint moved, g={g}");
            assert_eq!(overlay(1.0, g), 1.0, "white endpoint moved, g={g}");
        }
        for bi in 0..=10 {
            for gi in 0..=10 {
                let o = overlay(bi as f32 / 10.0, gi as f32 / 10.0);
                assert!((0.0..=1.0).contains(&o), "o={o}");
            }
        }
    }

    /// `grain = 0` and `strength = 0` are both strict identities for the
    /// overlay composite.
    #[test]
    fn apply_grain_identities() {
        let color = Vec3::new(0.2, 0.5, 0.8);
        let id_grain = apply_grain(color, 0.0, 1.0);
        assert!((id_grain - color).abs().max_element() < 1.0e-6, "grain=0 not identity: {id_grain:?}");
        let id_strength = apply_grain(color, 0.7, 0.0);
        assert!((id_strength - color).abs().max_element() < 1.0e-6, "strength=0 not identity: {id_strength:?}");
    }

    /// Grained output never leaves `[0, 1]`, even for out-of-range inputs and
    /// extreme grain / strength.
    #[test]
    fn apply_grain_stays_in_unit_range() {
        let inputs = [
            Vec3::new(0.0, 0.5, 1.0),
            Vec3::new(-1.0, 2.0, f32::NAN),
            Vec3::splat(f32::INFINITY),
        ];
        for &c in &inputs {
            for &grain in &[-1.0_f32, -0.3, 0.0, 0.6, 1.0] {
                for &s in &[0.0_f32, 0.5, 1.0] {
                    let out = apply_grain(c, grain, s);
                    assert!(
                        (0.0..=1.0).contains(&out.x)
                            && (0.0..=1.0).contains(&out.y)
                            && (0.0..=1.0).contains(&out.z),
                        "out={out:?} c={c:?} grain={grain} s={s}"
                    );
                }
            }
        }
    }

    /// The deviation from the base grows monotonically with `strength` for a
    /// fixed non-neutral grain.
    #[test]
    fn apply_grain_is_monotonic_in_strength() {
        let base = 0.4_f32;
        let grain = 0.8_f32;
        let mut prev = 0.0_f32;
        let mut prev_s = -1.0_f32;
        for i in 0..=10 {
            let s = i as f32 / 10.0;
            let dev = (overlay_grain(base, grain, s) - base).abs();
            assert!(dev + 1.0e-6 >= prev, "non-monotone: s={s} dev={dev} prev={prev}");
            assert!(s > prev_s);
            prev = dev;
            prev_s = s;
        }
    }

    /// The gain blend leaves black untouched, scales with the signal, stays in
    /// range, and grows monotonically with strength.
    #[test]
    fn gain_blend_behaviour() {
        assert_eq!(gain_blend(0.0, 1.0, 1.0), 0.0, "black must stay black");
        // Brighter signal takes a larger absolute grain for the same grain/strength.
        let dim = (gain_blend(0.2, 1.0, 0.5) - 0.2).abs();
        let bright = (gain_blend(0.8, 1.0, 0.5) - 0.8).abs();
        assert!(bright > dim, "signal dependence failed: dim={dim} bright={bright}");
        // Range + strength monotonicity.
        let mut prev = 0.0_f32;
        for i in 0..=10 {
            let s = i as f32 / 10.0;
            let v = gain_blend(0.5, -0.6, s);
            assert!((0.0..=1.0).contains(&v), "v={v}");
            let dev = (v - 0.5).abs();
            assert!(dev + 1.0e-6 >= prev, "non-monotone dev={dev} prev={prev}");
            prev = dev;
        }
    }

    /// `apply_gain_blend` stays in `[0, 1]` for arbitrary inputs.
    #[test]
    fn apply_gain_blend_in_range() {
        let out = apply_gain_blend(Vec3::new(-2.0, 0.5, 3.0), 1.0, 1.0);
        assert!(
            (0.0..=1.0).contains(&out.x)
                && (0.0..=1.0).contains(&out.y)
                && (0.0..=1.0).contains(&out.z),
            "out={out:?}"
        );
    }

    /// ISO→strength is `0` at/below base, `1` at/above max, monotonically
    /// increasing in between, and always within `[0, 1]`.
    #[test]
    fn iso_to_strength_is_monotonic_ramp() {
        assert_eq!(iso_to_strength(ISO_MIN), 0.0);
        assert_eq!(iso_to_strength(50.0), 0.0, "sub-min ISO clamps to zero");
        assert!((iso_to_strength(ISO_MAX) - 1.0).abs() < 1.0e-6);
        assert_eq!(iso_to_strength(25_600.0), 1.0, "above-max ISO clamps to one");
        assert_eq!(iso_to_strength(f32::NAN), 0.0);

        let mut prev = -1.0_f32;
        let mut iso = ISO_MIN;
        while iso <= ISO_MAX {
            let s = iso_to_strength(iso);
            assert!((0.0..=1.0).contains(&s), "s={s} iso={iso}");
            assert!(s + 1.0e-7 >= prev, "non-monotone s={s} prev={prev} iso={iso}");
            prev = s;
            iso *= 1.4142;
        }
    }
}
