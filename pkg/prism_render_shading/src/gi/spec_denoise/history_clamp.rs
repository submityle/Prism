//! Specular history clamping and accumulation control — CPU golden.
//!
//! Temporal accumulation of a specular signal must be far more cautious than
//! for diffuse: a glossy highlight is view-dependent and can change completely
//! in a single frame as the camera, a normal, or the roughness shifts.  Letting
//! stale history leak through produces ghosting trails behind reflections.
//! Production specular denoisers (NVIDIA NRD ReBLUR / ReLAX specular) guard the
//! accumulator with four mechanisms, all reproduced here as deterministic
//! reference pure functions:
//!
//! 1. **Colour-space clipping** — reject history that leaves the current
//!    neighbourhood's colour AABB.  [`clip_history_aabb`] uses the Karis/DICE
//!    line-clip toward the neighbourhood mean (sharper than a per-channel
//!    clamp), and [`variance_aabb`] builds the box from mean ± γ·σ moments.
//! 2. **Roughness-aware history length** — [`roughness_history_length`] grants
//!    rough surfaces long accumulation (their lobe is stable) while mirrors get
//!    a short window so they stay responsive.
//! 3. **Lobe-shift rejection** — [`lobe_rejection`] compares the inter-frame
//!    normal change against the GGX lobe half-angle ([`lobe_half_angle`]); a
//!    normal shift larger than the lobe means the history is looking at a
//!    different reflection and is discarded.
//! 4. **Fast-history clamp** — [`fast_history_factor`] detects divergence
//!    between a short "fast" and long "slow" luminance history and boosts the
//!    new sample's weight, suppressing lag on genuine lighting changes.
//!
//! [`specular_accumulation`] fuses these into a single blend weight + clamped
//! colour for the caller's exponential moving average.
//!
//! # Conventions
//! * Linear RGB `f32` radiance throughout, matching the WESL/GPU twin.
//! * `roughness ∈ [0, 1]` perceptual; GGX `alpha = roughness²` via
//!   [`crate::gi::spec_gi::ggx_lobe::roughness_to_alpha`], which this module
//!   calls so the lobe width stays consistent with the BRDF reference.
//! * Transcendentals go through [`bevy_math::ops`]; `sqrt` is inherent `f32`.
//! * Every helper is pure (no RNG / IO / GPU / globals / `unsafe`), defends
//!   against degeneracy (empty neighbourhoods, zero normals, non-finite input),
//!   and never returns `NaN`; weights are clamped to `[0, 1]`.

use bevy_math::{ops, Vec3};

use crate::gi::spec_gi::ggx_lobe::roughness_to_alpha;

/// Tunables for the specular history guard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryClampParams {
    /// Half-width of the colour AABB in standard deviations (ReLAX γ).  Smaller
    /// → tighter clamp, less ghosting but more variance.
    pub clamp_sigma: f32,
    /// Minimum accumulated frames (mirror end of the schedule).
    pub min_frames: u32,
    /// Maximum accumulated frames (rough end of the schedule).
    pub max_frames: u32,
    /// Multiplier on the lobe half-angle beyond which history is fully
    /// rejected.  `1` rejects exactly at the lobe edge; larger is more lenient.
    pub lobe_tolerance: f32,
    /// Sensitivity of the fast/slow divergence detector; larger reacts to
    /// smaller luminance gaps.
    pub fast_sensitivity: f32,
}

impl Default for HistoryClampParams {
    fn default() -> Self {
        Self {
            clamp_sigma: 2.0,
            min_frames: 2,
            max_frames: 32,
            lobe_tolerance: 2.0,
            fast_sensitivity: 4.0,
        }
    }
}

/// Build a colour-space AABB `(min, max) = mean ± σ·sigma` from neighbourhood
/// moments (clamped non-negative so a bright floor never inverts the box).
#[must_use]
pub fn variance_aabb(mean: Vec3, std: Vec3, sigma: f32) -> (Vec3, Vec3) {
    let m = sanitize_rgb(mean);
    let s = sanitize_rgb(std) * sigma.max(0.0);
    let lo = (m - s).max(Vec3::ZERO);
    let hi = m + s;
    (lo, hi)
}

/// Clip `history` into the colour AABB `[aabb_min, aabb_max]` by moving it along
/// the line toward the box centre (Karis/DICE "clip-to-AABB").
///
/// Line clipping keeps the clamped colour on the segment between the history
/// and the neighbourhood centre, which preserves hue far better than an
/// independent per-channel clamp.  A history already inside the box is
/// returned unchanged; a degenerate (zero-extent) box returns the centre.
#[must_use]
pub fn clip_history_aabb(history: Vec3, aabb_min: Vec3, aabb_max: Vec3) -> Vec3 {
    let h = sanitize_rgb(history);
    let lo = sanitize_rgb(aabb_min);
    let hi = sanitize_rgb(aabb_max);
    let center = (lo + hi) * 0.5;
    let extent = (hi - lo) * 0.5;
    let offset = h - center;

    // Largest axis-wise ratio |offset| / extent; if <= 1 the point is inside.
    let mut max_ratio = 0.0_f32;
    for axis in 0..3 {
        let e = extent[axis];
        let o = offset[axis].abs();
        if e <= 1.0e-12 {
            // Degenerate axis: any deviation is "infinitely" outside.
            if o > 1.0e-12 {
                max_ratio = f32::INFINITY;
            }
        } else {
            max_ratio = max_ratio.max(o / e);
        }
    }

    if !max_ratio.is_finite() {
        return center;
    }
    if max_ratio <= 1.0 {
        return h;
    }
    center + offset / max_ratio
}

/// Accumulated-frame budget for a given roughness, in `[min_frames, max_frames]`.
///
/// Monotonically increasing in roughness: a mirror gets `min_frames` (fast,
/// responsive history) while a rough surface earns `max_frames` (long, stable
/// accumulation).  The schedule is linear in roughness and rounds to the
/// nearest frame.
#[must_use]
pub fn roughness_history_length(roughness: f32, params: &HistoryClampParams) -> u32 {
    let r = roughness.clamp(0.0, 1.0);
    let lo = params.min_frames.max(1);
    let hi = params.max_frames.max(lo);
    let span = (hi - lo) as f32;
    let frames = lo as f32 + span * r;
    // Round-half-up; result is already within [lo, hi] by construction.
    let rounded = (frames + 0.5) as u32;
    rounded.clamp(lo, hi)
}

/// GGX specular lobe half-angle (radians) for a given roughness.
///
/// Derived from the GGX width `alpha = roughness²`: `atan(alpha)` is the angle
/// at which the microfacet distribution has fallen off substantially.  Returns
/// a tiny but non-zero angle for a mirror (so comparisons never divide by zero)
/// and widens monotonically with roughness.
#[must_use]
pub fn lobe_half_angle(roughness: f32) -> f32 {
    let alpha = roughness_to_alpha(roughness);
    ops::atan(alpha).max(1.0e-4)
}

/// History-keep weight in `[0, 1]` from the inter-frame normal shift.
///
/// Returns `1` when the current and historical normals coincide and falls to
/// `0` once the angle between them exceeds `lobe_tolerance · lobe_half_angle`.
/// A normal shift wider than the lobe means the pixel now reflects a different
/// part of the scene, so its history must be dropped.  The falloff is a smooth
/// `cos`-shaped ramp over the tolerance window.
#[must_use]
pub fn lobe_rejection(
    normal_curr: Vec3,
    normal_prev: Vec3,
    roughness: f32,
    params: &HistoryClampParams,
) -> f32 {
    let a = safe_normalize(normal_curr);
    let b = safe_normalize(normal_prev);
    let cos = a.dot(b).clamp(-1.0, 1.0);
    let angle = ops::acos(cos);
    let window = lobe_half_angle(roughness) * params.lobe_tolerance.max(1.0e-4);
    if angle <= 0.0 {
        return 1.0;
    }
    if angle >= window {
        return 0.0;
    }
    // Raised-cosine ramp: 1 at angle 0, 0 at angle = window, smooth in between.
    let t = (angle / window).clamp(0.0, 1.0);
    (0.5 * (1.0 + ops::cos(core::f32::consts::PI * t))).clamp(0.0, 1.0)
}

/// Divergence weight in `[0, 1]` between a short "fast" and long "slow"
/// luminance history.
///
/// Returns `1` when the two agree (converged, trust the slow history) and
/// decays toward `0` as they diverge (a genuine lighting change the slow
/// history has not caught up to), signalling the caller to lean on the new
/// sample.  Normalised by the signal magnitude so it is exposure-robust.
#[must_use]
pub fn fast_history_factor(fast_luma: f32, slow_luma: f32, params: &HistoryClampParams) -> f32 {
    let f = sanitize_scalar(fast_luma).max(0.0);
    let s = sanitize_scalar(slow_luma).max(0.0);
    let diff = (f - s).abs();
    let denom = (f + s) * 0.5 + 1.0e-4;
    let rel = diff / denom;
    stable_exp(-params.fast_sensitivity.max(0.0) * rel)
}

/// Blend weight for the *new* sample (ReLAX α), in `[0, 1]`.
///
/// `age` is the pre-update accumulated frame count; the base weight is
/// `1 / min(age + 1, frame_budget)` so the running average seeds instantly and
/// then floors at `1 / frame_budget`.  `frame_budget` comes from
/// [`roughness_history_length`].  The weight is pushed toward `1` (discard
/// history) by low `confidence`, lobe rejection, and fast/slow divergence.
#[must_use]
pub fn specular_blend_weight(
    age: u32,
    frame_budget: u32,
    confidence: f32,
    lobe_keep: f32,
    fast_keep: f32,
) -> f32 {
    let budget = frame_budget.max(1);
    let effective = (age + 1).min(budget);
    let base = 1.0 / effective as f32;
    // History trust is the product of the three keep terms, each in [0, 1].
    let keep = sanitize_scalar(confidence).clamp(0.0, 1.0)
        * sanitize_scalar(lobe_keep).clamp(0.0, 1.0)
        * sanitize_scalar(fast_keep).clamp(0.0, 1.0);
    // alpha = 1 - keep·(1 - base): full keep -> base EMA; zero keep -> alpha 1.
    let alpha = 1.0 - keep * (1.0 - base);
    alpha.clamp(0.0, 1.0)
}

/// Result of guarding one specular history pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClampResult {
    /// History colour after colour-space clipping.
    pub clamped_history: Vec3,
    /// New-sample blend weight (α) for the EMA `lerp(history, sample, α)`.
    pub blend_weight: f32,
    /// Accumulated-frame budget used (for the caller to carry forward).
    pub frame_budget: u32,
}

/// Full specular history guard for one pixel.
///
/// Clips the history into the neighbourhood colour AABB, derives the roughness
/// frame budget, evaluates lobe-shift and fast/slow divergence, and folds the
/// reprojection `confidence` into a final blend weight.  All outputs are finite.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn specular_accumulation(
    history: Vec3,
    neighbourhood_mean: Vec3,
    neighbourhood_std: Vec3,
    age: u32,
    roughness: f32,
    confidence: f32,
    normal_curr: Vec3,
    normal_prev: Vec3,
    fast_luma: f32,
    slow_luma: f32,
    params: &HistoryClampParams,
) -> ClampResult {
    let (lo, hi) = variance_aabb(neighbourhood_mean, neighbourhood_std, params.clamp_sigma);
    let clamped_history = clip_history_aabb(history, lo, hi);
    let frame_budget = roughness_history_length(roughness, params);
    let lobe_keep = lobe_rejection(normal_curr, normal_prev, roughness, params);
    let fast_keep = fast_history_factor(fast_luma, slow_luma, params);
    let blend_weight = specular_blend_weight(age, frame_budget, confidence, lobe_keep, fast_keep);
    ClampResult {
        clamped_history: sanitize_rgb(clamped_history),
        blend_weight,
        frame_budget,
    }
}

// ---------------------------------------------------------------------------
// Defensive numeric helpers (private).
// ---------------------------------------------------------------------------

/// Numerically safe `exp` on `(-inf, 0]`; never returns `NaN`/`+inf`.
#[must_use]
fn stable_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    ops::exp(x.clamp(-80.0, 0.0))
}

/// Normalise `v`, falling back to `+Z` for zero-length / non-finite input.
#[must_use]
fn safe_normalize(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > 1.0e-24 {
        v / len_sq.sqrt()
    } else {
        Vec3::Z
    }
}

/// Replace any non-finite component of an RGB triple with `0`, clamped `≥ 0`.
#[must_use]
fn sanitize_rgb(c: Vec3) -> Vec3 {
    Vec3::new(finite_or_zero(c.x), finite_or_zero(c.y), finite_or_zero(c.z)).max(Vec3::ZERO)
}

/// Replace a non-finite scalar with `0`.
#[must_use]
fn sanitize_scalar(x: f32) -> f32 {
    finite_or_zero(x)
}

#[must_use]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    fn params() -> HistoryClampParams {
        HistoryClampParams::default()
    }

    #[test]
    fn variance_aabb_brackets_mean() {
        let (lo, hi) = variance_aabb(Vec3::splat(1.0), Vec3::splat(0.25), 2.0);
        assert!((lo - Vec3::splat(0.5)).length() < EPS);
        assert!((hi - Vec3::splat(1.5)).length() < EPS);
    }

    #[test]
    fn clip_identity_inside_box() {
        let lo = Vec3::splat(0.0);
        let hi = Vec3::splat(1.0);
        let inside = Vec3::new(0.3, 0.7, 0.5);
        let out = clip_history_aabb(inside, lo, hi);
        assert!((out - inside).length() < EPS);
    }

    #[test]
    fn clip_pulls_outside_point_onto_box() {
        let lo = Vec3::splat(0.0);
        let hi = Vec3::splat(1.0);
        let outside = Vec3::new(2.0, 0.5, 0.5);
        let out = clip_history_aabb(outside, lo, hi);
        // Result must lie within the (slightly padded) box and be closer to it.
        assert!(out.x <= 1.0 + 1e-4 && out.x >= -1e-4);
        assert!((out - outside).length() > 0.0);
        // On the line toward centre, so y,z move proportionally toward 0.5.
        assert!(out.x <= outside.x);
    }

    #[test]
    fn clip_degenerate_box_returns_centre() {
        let p = Vec3::splat(0.5);
        let out = clip_history_aabb(Vec3::splat(5.0), p, p);
        assert!((out - p).length() < EPS);
    }

    #[test]
    fn history_length_monotone_in_roughness() {
        let p = params();
        let mirror = roughness_history_length(0.0, &p);
        let mid = roughness_history_length(0.5, &p);
        let rough = roughness_history_length(1.0, &p);
        assert_eq!(mirror, p.min_frames);
        assert_eq!(rough, p.max_frames);
        assert!(mirror <= mid && mid <= rough);
    }

    #[test]
    fn lobe_half_angle_monotone_and_mirror_is_tiny() {
        let mirror = lobe_half_angle(0.0);
        let mid = lobe_half_angle(0.5);
        let rough = lobe_half_angle(1.0);
        assert!(mirror > 0.0);
        assert!(mirror < mid && mid < rough);
        // roughness=1 -> alpha=1 -> atan(1) = pi/4.
        assert!((rough - core::f32::consts::FRAC_PI_4).abs() < 1e-3);
    }

    #[test]
    fn lobe_rejection_keeps_aligned_and_drops_wide_shift() {
        let p = params();
        let keep = lobe_rejection(Vec3::Z, Vec3::Z, 0.1, &p);
        assert!((keep - 1.0).abs() < EPS);
        // A 90° normal shift is far outside any lobe -> full rejection.
        let drop = lobe_rejection(Vec3::Z, Vec3::X, 0.1, &p);
        assert!(drop.abs() < EPS);
    }

    #[test]
    fn lobe_rejection_is_monotone_in_angle() {
        let p = params();
        let n0 = Vec3::Z;
        let small = Vec3::new(0.05, 0.0, 1.0);
        let large = Vec3::new(0.2, 0.0, 1.0);
        let w_small = lobe_rejection(n0, small, 0.6, &p);
        let w_large = lobe_rejection(n0, large, 0.6, &p);
        assert!(w_small >= w_large);
        assert!((0.0..=1.0).contains(&w_small));
        assert!((0.0..=1.0).contains(&w_large));
    }

    #[test]
    fn fast_history_detects_divergence() {
        let p = params();
        let agree = fast_history_factor(1.0, 1.0, &p);
        let diverge = fast_history_factor(5.0, 1.0, &p);
        assert!((agree - 1.0).abs() < EPS);
        assert!(diverge < agree);
        assert!((0.0..=1.0).contains(&diverge));
    }

    #[test]
    fn blend_weight_full_trust_is_running_average() {
        // confidence=keep=1 -> alpha = 1/min(age+1, budget).
        let a = specular_blend_weight(0, 32, 1.0, 1.0, 1.0);
        let b = specular_blend_weight(1, 32, 1.0, 1.0, 1.0);
        let c = specular_blend_weight(5, 32, 1.0, 1.0, 1.0);
        assert!((a - 1.0).abs() < EPS);
        assert!((b - 0.5).abs() < EPS);
        assert!((c - 1.0 / 6.0).abs() < EPS);
    }

    #[test]
    fn blend_weight_zero_trust_discards_history() {
        let alpha = specular_blend_weight(100, 32, 0.0, 1.0, 1.0);
        assert!((alpha - 1.0).abs() < EPS);
        let alpha2 = specular_blend_weight(100, 32, 1.0, 0.0, 1.0);
        assert!((alpha2 - 1.0).abs() < EPS);
    }

    #[test]
    fn blend_weight_monotone_in_confidence() {
        let hi = specular_blend_weight(10, 32, 1.0, 1.0, 1.0);
        let lo = specular_blend_weight(10, 32, 0.5, 1.0, 1.0);
        // Lower confidence -> larger alpha (more new sample).
        assert!(lo >= hi);
    }

    #[test]
    fn specular_accumulation_is_finite_on_degenerate_inputs() {
        let p = params();
        let r = specular_accumulation(
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::INFINITY),
            Vec3::splat(-1.0),
            7,
            2.0,
            f32::NAN,
            Vec3::ZERO,
            Vec3::ZERO,
            f32::INFINITY,
            -3.0,
            &p,
        );
        assert!(r.clamped_history.is_finite());
        assert!(r.blend_weight.is_finite());
        assert!((0.0..=1.0).contains(&r.blend_weight));
        assert!(r.frame_budget >= p.min_frames && r.frame_budget <= p.max_frames);
    }

    #[test]
    fn determinism() {
        let p = params();
        let a = specular_accumulation(
            Vec3::splat(2.0),
            Vec3::splat(1.0),
            Vec3::splat(0.3),
            4,
            0.4,
            0.8,
            Vec3::Z,
            Vec3::new(0.1, 0.0, 1.0),
            1.2,
            1.0,
            &p,
        );
        let b = specular_accumulation(
            Vec3::splat(2.0),
            Vec3::splat(1.0),
            Vec3::splat(0.3),
            4,
            0.4,
            0.8,
            Vec3::Z,
            Vec3::new(0.1, 0.0, 1.0),
            1.2,
            1.0,
            &p,
        );
        assert_eq!(a, b);
    }
}
