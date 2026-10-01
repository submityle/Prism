//! Contrast Adaptive Sharpening (CAS) — CPU golden reference.
//!
//! CAS is AMD FidelityFX's single-pass, overshoot-free sharpening filter. It
//! raises local contrast where an image is *soft* and backs off where the
//! image is already crisp or near the signal limits, so it sharpens detail
//! without the ringing halos a fixed unsharp-mask produces. This module is the
//! backend-neutral numerical reference the WESL/GPU twin must reproduce under
//! real-device parity.
//!
//! The filter inspects the 3x3 neighbourhood of one pixel:
//!
//! ```text
//!   a b c
//!   d e f      e = center
//!   g h i
//! ```
//!
//! and performs three steps per colour channel, mirroring FidelityFX:
//!
//! 1. **Soft min/max.** A "soft" minimum/maximum is formed by summing the
//!    minimum over the cross `{b, d, e, f, h}` with the minimum over the full
//!    ring (and likewise for the maximum). This doubles the magnitude, so the
//!    values live in `[0, 2]` for display-referred `[0, 1]` input and behave
//!    like a gently low-passed extremum that is less twitchy than a hard
//!    min/max.
//! 2. **Adaptive amplitude.** `amp = sqrt(saturate(min(mn, 2 - mx) / mx))`
//!    measures how much headroom the signal has toward both the floor (`0`) and
//!    the ceiling (`1`). Flat or clipped regions yield a small effective weight;
//!    mid-range detail yields the strongest sharpening.
//! 3. **Cross-shaped sharpen.** A negative-lobe cross kernel
//!    `(e - w·(b + d + f + h)) / (1 - 4w)` increases contrast between the
//!    center and its four edge neighbours. The output is finally clamped into
//!    the real per-channel `[min, max]` of the 3x3 ring, which is the CAS
//!    guarantee of *no overshoot* beyond values already present locally.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, IO, GPU, global state, or `unsafe`.
//! * Colours are linear-RGB [`Vec3`]; every tap is sanitized to be finite and
//!   non-negative before use so no `NaN`/`inf` can propagate.
//! * `sqrt` uses the inherent method; no other transcendental is needed here.
//! * `sharpness` is clamped to `[0, 1]`. At `sharpness = 0` the weight is
//!   exactly zero and the filter is the identity (output equals the center).
//! * The sharpen weight is `w = CAS_WEIGHT_MAX · sharpness · amp` per channel,
//!   bounded so the kernel denominator `1 - 4w` stays strictly positive.

use bevy_math::Vec3;

/// Row-major index of the center tap within a [`Neighborhood3x3`] array.
pub const CENTER: usize = 4;

/// Peak sharpen-weight magnitude at `sharpness = 1`.
///
/// FidelityFX shapes its negative lobe as `-1 / lerp(8, 5, sharpness)`, whose
/// strongest magnitude is `1/5 = 0.2`. We expose the same ceiling but scale it
/// linearly by `sharpness` so that `sharpness = 0` is a true identity, which is
/// convenient for a golden reference and A/B testing. The bound guarantees
/// `1 - 4w >= 1 - 4·0.2 = 0.2 > 0`, so the kernel never divides by zero.
pub const CAS_WEIGHT_MAX: f32 = 0.2;

/// Smallest denominator permitted when forming `1 / max` for the amplitude.
const RCP_EPS: f32 = 1.0e-6;

/// A 3x3 colour neighbourhood in row-major order (`[a, b, c, d, e, f, g, h, i]`
/// with the center at index [`CENTER`]).
pub type Neighborhood3x3 = [Vec3; 9];

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

/// Component-wise reciprocal with a floor on the denominator magnitude, so a
/// zero or tiny channel can never produce `inf`.
#[inline]
fn safe_recip(v: Vec3) -> Vec3 {
    Vec3::new(
        1.0 / v.x.max(RCP_EPS),
        1.0 / v.y.max(RCP_EPS),
        1.0 / v.z.max(RCP_EPS),
    )
}

/// Component-wise square root of a non-negative colour.
#[inline]
fn sqrt3(v: Vec3) -> Vec3 {
    Vec3::new(v.x.max(0.0).sqrt(), v.y.max(0.0).sqrt(), v.z.max(0.0).sqrt())
}

/// Sanitizes every tap of a neighbourhood up front.
#[inline]
fn sanitized(n: &Neighborhood3x3) -> Neighborhood3x3 {
    let mut out = [Vec3::ZERO; 9];
    for (o, &c) in out.iter_mut().zip(n.iter()) {
        *o = sanitize(c);
    }
    out
}

/// Soft (doubled) per-channel min/max of the 3x3 neighbourhood as used by CAS:
/// the extremum over the cross `{b, d, e, f, h}` added to the extremum over the
/// full ring. Returns `(soft_min, soft_max)`.
fn soft_min_max(n: &Neighborhood3x3) -> (Vec3, Vec3) {
    let (a, b, c) = (n[0], n[1], n[2]);
    let (d, e, f) = (n[3], n[4], n[5]);
    let (g, h, i) = (n[6], n[7], n[8]);

    let cross_min = b.min(d).min(e).min(f).min(h);
    let full_min = cross_min.min(a).min(c).min(g).min(i);
    let soft_min = cross_min + full_min;

    let cross_max = b.max(d).max(e).max(f).max(h);
    let full_max = cross_max.max(a).max(c).max(g).max(i);
    let soft_max = cross_max + full_max;

    (soft_min, soft_max)
}

/// Hard per-channel min/max over all nine taps. Used only to clamp the result
/// into the locally present range (the CAS no-overshoot guarantee).
fn hard_min_max(n: &Neighborhood3x3) -> (Vec3, Vec3) {
    let mut lo = n[0];
    let mut hi = n[0];
    for &c in n.iter().skip(1) {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    (lo, hi)
}

/// Per-channel adaptive sharpening amplitude in `[0, 1]`:
/// `amp = sqrt(saturate(min(mn, 2 - mx) / mx))`.
///
/// `mn`/`mx` are the *soft* (doubled) extrema from [`soft_min_max`], so the
/// `2 - mx` term measures headroom to the display ceiling. Clipped or
/// out-of-`[0, 1]` input drives the amplitude toward zero, which keeps the
/// filter from amplifying already-saturated regions.
#[must_use]
pub fn adaptive_amplitude(neighborhood: &Neighborhood3x3) -> Vec3 {
    let clean = sanitized(neighborhood);
    let (mn, mx) = soft_min_max(&clean);
    let headroom = (Vec3::splat(2.0) - mx).max(Vec3::ZERO);
    let numerator = mn.min(headroom);
    let ratio = (numerator * safe_recip(mx)).clamp(Vec3::ZERO, Vec3::ONE);
    sqrt3(ratio)
}

/// Applies Contrast Adaptive Sharpening to the center of a 3x3 neighbourhood.
///
/// `sharpness` is clamped to `[0, 1]`; `0` is the identity. The result is
/// clamped into the per-channel `[min, max]` of the neighbourhood so it can
/// never overshoot values already present locally.
#[must_use]
pub fn cas(neighborhood: Neighborhood3x3, sharpness: f32) -> Vec3 {
    let clean = sanitized(&neighborhood);
    let sharp = finite_or(sharpness, 0.0).clamp(0.0, 1.0);

    let amp = adaptive_amplitude(&clean);
    // Per-channel positive weight; bounded by CAS_WEIGHT_MAX so 1 - 4w > 0.
    let w = amp * (CAS_WEIGHT_MAX * sharp);

    let center = clean[CENTER];
    let cross_sum = clean[1] + clean[3] + clean[5] + clean[7];

    let numerator = center - w * cross_sum;
    let denom = Vec3::ONE - 4.0 * w;
    let sharpened = numerator * safe_recip(denom);

    let (lo, hi) = hard_min_max(&clean);
    sanitize(sharpened.clamp(lo, hi))
}

/// Convenience constructor for a [`Neighborhood3x3`] from a single flat colour
/// (every tap equal). Handy for tests and for callers padding image borders.
#[must_use]
pub fn uniform_neighborhood(color: Vec3) -> Neighborhood3x3 {
    [color; 9]
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

    /// A flat region must be reproduced exactly (no ringing, no DC shift).
    #[test]
    fn flat_region_is_preserved() {
        for &lvl in &[0.0_f32, 0.1, 0.5, 0.9, 1.0] {
            let n = uniform_neighborhood(Vec3::splat(lvl));
            for &s in &[0.0_f32, 0.3, 0.7, 1.0] {
                let out = cas(n, s);
                assert!(approx(out, Vec3::splat(lvl), 1.0e-6), "lvl={lvl} s={s} out={out:?}");
            }
        }
    }

    /// `sharpness = 0` is a strict identity (center passed through unchanged).
    #[test]
    fn zero_sharpness_is_identity() {
        let n = [
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.4, 0.1, 0.9),
            Vec3::new(0.2, 0.5, 0.1),
            Vec3::new(0.7, 0.3, 0.2),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.3, 0.8, 0.4),
            Vec3::new(0.9, 0.1, 0.6),
            Vec3::new(0.2, 0.4, 0.7),
            Vec3::new(0.6, 0.3, 0.1),
        ];
        let out = cas(n, 0.0);
        assert!(approx(out, n[CENTER], 1.0e-6), "out={out:?}");
    }

    /// On an edge (center brighter than its cross neighbours, with headroom in
    /// the corners) CAS must push the center *up*, increasing local contrast.
    #[test]
    fn edge_contrast_is_increased() {
        let center = Vec3::splat(0.5);
        let n = [
            Vec3::splat(1.0), // a corner provides max headroom
            Vec3::splat(0.2), // b cross
            Vec3::splat(1.0), // c corner
            Vec3::splat(0.2), // d cross
            center,           // e
            Vec3::splat(0.2), // f cross
            Vec3::splat(1.0), // g corner
            Vec3::splat(0.2), // h cross
            Vec3::splat(1.0), // i corner
        ];
        let out = cas(n, 1.0);
        assert!(out.x > center.x + 1.0e-3, "expected sharpening, out={out:?}");
    }

    /// The output can never leave the per-channel neighbourhood `[min, max]`
    /// (the CAS overshoot-free guarantee), for any sharpness.
    #[test]
    fn output_stays_within_neighborhood_range() {
        let n = [
            Vec3::new(0.05, 0.90, 0.20),
            Vec3::new(0.60, 0.10, 0.70),
            Vec3::new(0.30, 0.40, 0.95),
            Vec3::new(0.80, 0.20, 0.05),
            Vec3::new(0.45, 0.55, 0.50),
            Vec3::new(0.10, 0.85, 0.35),
            Vec3::new(0.95, 0.05, 0.60),
            Vec3::new(0.25, 0.65, 0.75),
            Vec3::new(0.70, 0.30, 0.15),
        ];
        let (lo, hi) = hard_min_max(&n);
        for &s in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let out = cas(n, s);
            assert!(out.x >= lo.x - 1.0e-6 && out.x <= hi.x + 1.0e-6, "x s={s} out={out:?}");
            assert!(out.y >= lo.y - 1.0e-6 && out.y <= hi.y + 1.0e-6, "y s={s} out={out:?}");
            assert!(out.z >= lo.z - 1.0e-6 && out.z <= hi.z + 1.0e-6, "z s={s} out={out:?}");
        }
    }

    /// Determinism: identical inputs yield bit-identical outputs.
    #[test]
    fn deterministic() {
        let n = [
            Vec3::new(0.11, 0.22, 0.33),
            Vec3::new(0.44, 0.15, 0.91),
            Vec3::new(0.21, 0.53, 0.12),
            Vec3::new(0.73, 0.31, 0.24),
            Vec3::new(0.52, 0.57, 0.58),
            Vec3::new(0.31, 0.82, 0.41),
            Vec3::new(0.93, 0.12, 0.61),
            Vec3::new(0.23, 0.41, 0.72),
            Vec3::new(0.61, 0.33, 0.14),
        ];
        assert_eq!(cas(n, 0.6), cas(n, 0.6));
    }

    /// Degenerate input (NaN/inf taps, out-of-range sharpness) stays finite and
    /// in range rather than propagating garbage.
    #[test]
    fn degenerate_input_is_sanitized() {
        let n = [
            Vec3::new(f32::NAN, 0.2, 0.3),
            Vec3::new(0.4, f32::INFINITY, -1.0),
            Vec3::new(0.2, 0.5, 0.1),
            Vec3::new(0.7, 0.3, 0.2),
            Vec3::new(0.5, f32::NEG_INFINITY, 0.5),
            Vec3::new(0.3, 0.8, 0.4),
            Vec3::new(0.9, 0.1, 0.6),
            Vec3::new(0.2, 0.4, 0.7),
            Vec3::new(0.6, 0.3, 0.1),
        ];
        let out = cas(n, f32::NAN);
        assert!(finite(out), "out={out:?}");
        assert!(out.x >= 0.0 && out.y >= 0.0 && out.z >= 0.0, "out={out:?}");
    }

    /// Amplitude is well-defined and bounded in `[0, 1]`, including on black.
    #[test]
    fn amplitude_is_bounded() {
        let black = uniform_neighborhood(Vec3::ZERO);
        let amp0 = adaptive_amplitude(&black);
        assert!(approx(amp0, Vec3::ZERO, 1.0e-6), "amp0={amp0:?}");

        let n = [
            Vec3::splat(0.2),
            Vec3::splat(0.3),
            Vec3::splat(0.2),
            Vec3::splat(0.3),
            Vec3::splat(0.5),
            Vec3::splat(0.3),
            Vec3::splat(0.2),
            Vec3::splat(0.3),
            Vec3::splat(0.2),
        ];
        let amp = adaptive_amplitude(&n);
        assert!((0.0..=1.0).contains(&amp.x), "amp={amp:?}");
    }

    /// Stronger sharpness moves the center further from its original value on
    /// a sharpenable edge (monotone response).
    #[test]
    fn sharpness_increases_effect() {
        let center = Vec3::splat(0.5);
        let n = [
            Vec3::splat(0.9),
            Vec3::splat(0.25),
            Vec3::splat(0.9),
            Vec3::splat(0.25),
            center,
            Vec3::splat(0.25),
            Vec3::splat(0.9),
            Vec3::splat(0.25),
            Vec3::splat(0.9),
        ];
        let low = (cas(n, 0.25).x - center.x).abs();
        let high = (cas(n, 1.0).x - center.x).abs();
        assert!(high > low, "low={low} high={high}");
    }
}
