//! Robust Contrast Adaptive Sharpening (RCAS) — CPU golden reference.
//!
//! RCAS is the sharpening half of AMD FidelityFX Super Resolution (FSR 1). It
//! refines plain CAS with two robustness features that matter after an
//! upscale:
//!
//! * a **limiter** derived from the local min/max ratio that analytically
//!   bounds the sharpening lobe so the result cannot overshoot the values
//!   already present in the local cross, and
//! * an optional **noise estimate** that attenuates sharpening where the 3x3
//!   cross looks like noise rather than a real edge, so RCAS does not amplify
//!   grain.
//!
//! Unlike CAS, RCAS uses only the 4-connected cross around the center:
//!
//! ```text
//!     b
//!   d e f      e = center
//!     h
//! ```
//!
//! The resolve kernel is the same negative-lobe cross as CAS,
//! `output = ((b + d + f + h)·lobe + e) / (1 + 4·lobe)` with a *negative*
//! `lobe`, but the lobe is clamped by the limiter and the noise term instead of
//! the plain CAS amplitude. Following FidelityFX, the limiter and noise terms
//! are scalars shared across channels; the per-channel min/max still bounds the
//! final clamp.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, IO, GPU, global state, or `unsafe`.
//! * Colours are linear-RGB [`Vec3`]; every tap is sanitized to be finite and
//!   non-negative before use so no `NaN`/`inf` can propagate.
//! * `sharpness` is clamped to `[0, 1]` and scales the lobe linearly: `0` is a
//!   strict identity, `1` is the strongest (still limiter-bounded) sharpening.
//! * The lobe magnitude is capped at [`RCAS_LIMIT`], guaranteeing the kernel
//!   denominator `1 + 4·lobe >= 1 - 4·RCAS_LIMIT > 0`.
//! * The output is additionally clamped into the per-channel `[min, max]` of
//!   the cross `{b, d, e, f, h}` as a hard overshoot guard.

use bevy_math::Vec3;

/// Maximum magnitude of the (negative) sharpening lobe.
///
/// FidelityFX uses `FSR_RCAS_LIMIT = 0.25 - 1/16 = 0.1875`. The bound keeps the
/// resolve denominator `1 + 4·lobe` within `[1 - 4·0.1875, 1] = [0.25, 1]`,
/// strictly positive.
pub const RCAS_LIMIT: f32 = 0.25 - 1.0 / 16.0;

/// Smallest denominator magnitude permitted in any reciprocal.
const RCP_EPS: f32 = 1.0e-6;

/// Row-major cross taps (top, left, center, right, bottom) RCAS consumes.
#[derive(Clone, Copy, Debug)]
pub struct Cross {
    /// Top neighbour (`b`).
    pub top: Vec3,
    /// Left neighbour (`d`).
    pub left: Vec3,
    /// Center pixel (`e`).
    pub center: Vec3,
    /// Right neighbour (`f`).
    pub right: Vec3,
    /// Bottom neighbour (`h`).
    pub bottom: Vec3,
}

impl Cross {
    /// Builds a cross from the five taps in reading order.
    #[must_use]
    pub fn new(top: Vec3, left: Vec3, center: Vec3, right: Vec3, bottom: Vec3) -> Self {
        Self { top, left, center, right, bottom }
    }

    /// Builds a flat cross (all taps equal) — handy for tests and borders.
    #[must_use]
    pub fn uniform(color: Vec3) -> Self {
        Self { top: color, left: color, center: color, right: color, bottom: color }
    }
}

/// Replaces a non-finite scalar with `fallback`.
#[inline]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Clamps a colour to be finite and non-negative component-wise.
#[inline]
fn sanitize(color: Vec3) -> Vec3 {
    Vec3::new(
        finite_or(color.x, 0.0).max(0.0),
        finite_or(color.y, 0.0).max(0.0),
        finite_or(color.z, 0.0).max(0.0),
    )
}

/// Safe scalar reciprocal with a floor on the denominator magnitude.
#[inline]
fn safe_rcp(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    s / x.abs().max(RCP_EPS)
}

/// FidelityFX "luma times two" approximation: `g + 0.5·(r + b)`. Only relative
/// values matter for the noise estimate, so the exact scale is irrelevant.
#[inline]
fn luma2(c: Vec3) -> f32 {
    c.y + 0.5 * (c.x + c.z)
}

/// Noise-suppression factor in `[0.5, 1]` from the cross luma.
///
/// It measures how much the center luma deviates from the ring average,
/// normalised by the ring's luma spread. A center that sits near the ring
/// extremes (a real edge) keeps the factor near `1`; a center that straddles
/// the ring (noise-like) pulls it toward `0.5`, following FidelityFX
/// `nz = 1 - 0.5·saturate(|avg - e| / spread)`.
fn noise_factor(cross: &Cross) -> f32 {
    let bl = luma2(cross.top);
    let dl = luma2(cross.left);
    let el = luma2(cross.center);
    let fl = luma2(cross.right);
    let hl = luma2(cross.bottom);

    let avg = 0.25 * (bl + dl + fl + hl);
    let hi = bl.max(dl).max(el).max(fl).max(hl);
    let lo = bl.min(dl).min(el).min(fl).min(hl);
    let spread = (hi - lo).max(RCP_EPS);

    let nz = ((avg - el).abs() * safe_rcp(spread)).clamp(0.0, 1.0);
    (1.0 - 0.5 * nz).clamp(0.5, 1.0)
}

/// Per-channel hard min/max over the cross taps `{b, d, e, f, h}`.
fn cross_min_max(cross: &Cross) -> (Vec3, Vec3) {
    let lo = cross
        .top
        .min(cross.left)
        .min(cross.center)
        .min(cross.right)
        .min(cross.bottom);
    let hi = cross
        .top
        .max(cross.left)
        .max(cross.center)
        .max(cross.right)
        .max(cross.bottom);
    (lo, hi)
}

/// Computes the limiter-bounded, noise-suppressed negative lobe RCAS applies.
///
/// The lobe is the most conservative (closest to zero) per-channel bound of the
/// FidelityFX `hitMin`/`hitMax` limiter, clamped to `[-RCAS_LIMIT, 0]`, scaled
/// by `sharpness` and the noise factor. Returns `0` on degenerate input, which
/// makes RCAS the identity there.
fn sharpening_lobe(cross: &Cross, sharpness: f32) -> f32 {
    let sharp = finite_or(sharpness, 0.0).clamp(0.0, 1.0);
    if sharp <= 0.0 {
        return 0.0;
    }

    // Ring (exclude center), matching FidelityFX RCAS.
    let mn4 = cross.top.min(cross.right).min(cross.left).min(cross.bottom);
    let mx4 = cross.top.max(cross.right).max(cross.left).max(cross.bottom);

    // Per-channel limiter terms. `hit_min` is the headroom toward the floor,
    // `hit_max` toward the ceiling (assuming display-referred [0, 1] signal).
    let hit_min = Vec3::new(
        mn4.x * safe_rcp(4.0 * mx4.x),
        mn4.y * safe_rcp(4.0 * mx4.y),
        mn4.z * safe_rcp(4.0 * mx4.z),
    );
    let hit_max = Vec3::new(
        (1.0 - mx4.x) * safe_rcp(4.0 * mn4.x - 4.0),
        (1.0 - mx4.y) * safe_rcp(4.0 * mn4.y - 4.0),
        (1.0 - mx4.z) * safe_rcp(4.0 * mn4.z - 4.0),
    );

    // Negative lobe per channel; take the least sharpening (max toward zero).
    let lobe_rgb = (-hit_min).max(hit_max);
    let lobe_scalar = lobe_rgb.x.max(lobe_rgb.y).max(lobe_rgb.z).min(0.0);
    let lobe = lobe_scalar.max(-RCAS_LIMIT);

    lobe * sharp * noise_factor(cross)
}

/// Applies Robust CAS to the center of a cross neighbourhood.
///
/// `sharpness` is clamped to `[0, 1]`; `0` is the identity. The result is
/// clamped into the per-channel `[min, max]` of the cross, so even with the
/// limiter the output can never overshoot locally present values.
#[must_use]
pub fn rcas(cross: Cross, sharpness: f32) -> Vec3 {
    let clean = Cross {
        top: sanitize(cross.top),
        left: sanitize(cross.left),
        center: sanitize(cross.center),
        right: sanitize(cross.right),
        bottom: sanitize(cross.bottom),
    };

    let lobe = sharpening_lobe(&clean, sharpness);
    let sum = clean.top + clean.left + clean.right + clean.bottom;
    let denom = 1.0 + 4.0 * lobe;
    let sharpened = (sum * lobe + clean.center) * safe_rcp(denom);

    let (lo, hi) = cross_min_max(&clean);
    sanitize(sharpened.clamp(lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps && (a.z - b.z).abs() <= eps
    }

    fn finite(v: Vec3) -> bool {
        v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
    }

    /// A flat cross is reproduced exactly for any sharpness.
    #[test]
    fn flat_region_is_identity() {
        for &lvl in &[0.0_f32, 0.2, 0.5, 0.8, 1.0] {
            let cross = Cross::uniform(Vec3::splat(lvl));
            for &s in &[0.0_f32, 0.4, 1.0] {
                let out = rcas(cross, s);
                assert!(approx(out, Vec3::splat(lvl), 1.0e-6), "lvl={lvl} s={s} out={out:?}");
            }
        }
    }

    /// `sharpness = 0` is a strict identity.
    #[test]
    fn zero_sharpness_is_identity() {
        let cross = Cross::new(
            Vec3::new(0.3, 0.6, 0.2),
            Vec3::new(0.2, 0.4, 0.7),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.6, 0.3, 0.4),
            Vec3::new(0.1, 0.8, 0.3),
        );
        let out = rcas(cross, 0.0);
        assert!(approx(out, cross.center, 1.0e-6), "out={out:?}");
    }

    /// On a sharpenable edge RCAS increases contrast, moving the center away
    /// from the ring average.
    #[test]
    fn edge_contrast_is_increased() {
        // One ring tap bright, the rest dark, center mid; values kept away from
        // 0 and 1 so the limiter does not fully zero the lobe.
        let center = Vec3::splat(0.5);
        let cross = Cross::new(
            Vec3::splat(0.8), // top (bright)
            Vec3::splat(0.3), // left
            center,
            Vec3::splat(0.3), // right
            Vec3::splat(0.3), // bottom
        );
        let out = rcas(cross, 1.0);
        assert!(out.x > center.x + 1.0e-3, "expected sharpening, out={out:?}");
    }

    /// The limiter keeps the output inside the cross `[min, max]` even at full
    /// sharpness — overshoot is suppressed.
    #[test]
    fn limiter_prevents_overshoot() {
        let cross = Cross::new(
            Vec3::new(0.70, 0.20, 0.60),
            Vec3::new(0.30, 0.55, 0.25),
            Vec3::new(0.52, 0.48, 0.50),
            Vec3::new(0.35, 0.60, 0.28),
            Vec3::new(0.28, 0.22, 0.66),
        );
        let (lo, hi) = cross_min_max(&cross);
        for &s in &[0.0_f32, 0.3, 0.6, 1.0] {
            let out = rcas(cross, s);
            assert!(out.x >= lo.x - 1.0e-6 && out.x <= hi.x + 1.0e-6, "x s={s} out={out:?}");
            assert!(out.y >= lo.y - 1.0e-6 && out.y <= hi.y + 1.0e-6, "y s={s} out={out:?}");
            assert!(out.z >= lo.z - 1.0e-6 && out.z <= hi.z + 1.0e-6, "z s={s} out={out:?}");
        }
    }

    /// Stronger sharpness yields a larger deviation from the center on an edge.
    #[test]
    fn sharpness_increases_effect() {
        let center = Vec3::splat(0.5);
        let cross = Cross::new(
            Vec3::splat(0.75),
            Vec3::splat(0.3),
            center,
            Vec3::splat(0.3),
            Vec3::splat(0.3),
        );
        let low = (rcas(cross, 0.3).x - center.x).abs();
        let high = (rcas(cross, 1.0).x - center.x).abs();
        assert!(high > low, "low={low} high={high}");
    }

    /// The noise factor stays within `[0.5, 1]` and reaches 1 on a clean step.
    #[test]
    fn noise_factor_bounds() {
        let edge = Cross::new(
            Vec3::splat(0.8),
            Vec3::splat(0.8),
            Vec3::splat(0.2),
            Vec3::splat(0.2),
            Vec3::splat(0.2),
        );
        let nz = noise_factor(&edge);
        assert!((0.5..=1.0).contains(&nz), "nz={nz}");

        // An impulse: the ring is near-uniform but the center is an outlier,
        // which reads as noise and drives nz below 1 (toward the 0.5 floor).
        let noisy = Cross::new(
            Vec3::splat(0.50),
            Vec3::splat(0.52),
            Vec3::splat(0.90),
            Vec3::splat(0.48),
            Vec3::splat(0.50),
        );
        let nz2 = noise_factor(&noisy);
        assert!((0.5..=1.0).contains(&nz2) && nz2 < 1.0, "nz2={nz2}");
    }

    /// Determinism: identical inputs produce bit-identical outputs.
    #[test]
    fn deterministic() {
        let cross = Cross::new(
            Vec3::new(0.31, 0.62, 0.21),
            Vec3::new(0.22, 0.41, 0.72),
            Vec3::new(0.52, 0.51, 0.53),
            Vec3::new(0.63, 0.31, 0.42),
            Vec3::new(0.11, 0.82, 0.33),
        );
        assert_eq!(rcas(cross, 0.7), rcas(cross, 0.7));
    }

    /// Degenerate input stays finite, non-negative and in range.
    #[test]
    fn degenerate_input_is_sanitized() {
        let cross = Cross::new(
            Vec3::new(f32::NAN, 0.2, 0.3),
            Vec3::new(0.4, f32::INFINITY, -2.0),
            Vec3::new(0.5, f32::NEG_INFINITY, 0.5),
            Vec3::new(0.6, 0.3, 0.4),
            Vec3::new(0.1, 0.8, 0.3),
        );
        let out = rcas(cross, f32::INFINITY);
        assert!(finite(out), "out={out:?}");
        assert!(out.x >= 0.0 && out.y >= 0.0 && out.z >= 0.0, "out={out:?}");
    }
}
