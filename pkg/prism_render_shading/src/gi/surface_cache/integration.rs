//! Confidence-weighted temporal integration, disocclusion reset, and spatial
//! filtering for the surfel radiance cache.
//!
//! Surfel radiance is accumulated across frames with a confidence-weighted
//! exponential moving average (the surface-cache analogue of the ReBLUR
//! temporal accumulation in [`crate::gi::denoise::reblur`]): the new-sample
//! blend weight `alpha = 1 / min(sample_count + 1, max_samples)` starts at `1`
//! (snap to the first sample), falls as confidence grows, and floors at
//! `1 / max_samples` so the cache keeps adapting to changing lighting.  When a
//! surfel is *disoccluded* — its anchor jumps off the previous disc or its
//! normal flips — the stale history is discarded and the entry reseeds from the
//! fresh sample.  A final bilateral spatial pass pulls energy between
//! geometrically compatible neighbours to hide the per-surfel Monte-Carlo
//! noise, weighting each neighbour by [`Surfel::geometric_weight`].
//!
//! # Conventions
//! * Radiance is linear RGB `Vec3`; it is sanitised to be finite and
//!   non-negative on every store.  `sample_count` is a frame-count confidence
//!   capped at `max_samples`.
//! * Disocclusion compares the previous and current surfel geometry: the anchor
//!   displacement measured against the surfel radius must stay within
//!   `position_tolerance`, and the normals must agree within `normal_tolerance`
//!   (a cosine floor).  Either failure forces a reset.
//! * The spatial filter is energy-preserving for a locally constant signal:
//!   when every neighbour carries the same radiance the output equals it
//!   exactly, and a fully incompatible neighbourhood (all weights zero) falls
//!   back to the centre radiance rather than emitting `NaN`.
//! * Transcendentals go through [`bevy_math::ops`]; every function is a
//!   deterministic pure function (no RNG / IO / GPU / unsafe).

use bevy_math::Vec3;

use super::surfel::{CoverageParams, Surfel};

/// Persistent per-surfel radiance accumulator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfelCacheEntry {
    /// Accumulated linear-RGB irradiance (finite, non-negative).
    pub radiance: Vec3,
    /// Confidence as an accumulated frame count, capped at `max_samples`.
    pub sample_count: u32,
}

impl SurfelCacheEntry {
    /// Seed a fresh entry from a single sample (confidence `1`).
    #[must_use]
    pub fn from_sample(radiance: Vec3) -> Self {
        Self {
            radiance: sanitize_rgb(radiance),
            sample_count: 1,
        }
    }
}

/// Parameters governing temporal accumulation and disocclusion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalParams {
    /// Maximum confidence; caps the EMA weight at `1 / max_samples` so the
    /// cache never freezes.
    pub max_samples: u32,
    /// Anchor-displacement tolerance as a fraction of the surfel radius; a jump
    /// beyond `position_tolerance * radius` triggers a disocclusion reset.
    pub position_tolerance: f32,
    /// Minimum `dot(prev_normal, curr_normal)` to treat the surfel as the same
    /// surface; below this the entry resets.
    pub normal_tolerance: f32,
}

impl Default for TemporalParams {
    fn default() -> Self {
        Self {
            max_samples: 32,
            position_tolerance: 0.5,
            normal_tolerance: 0.5,
        }
    }
}

/// New-sample blend weight for a given confidence.
///
/// `alpha = 1 / min(sample_count + 1, max_samples)`: `1` on the first sample,
/// decreasing as confidence accrues, floored at `1 / max_samples`.
#[must_use]
pub fn confidence_alpha(sample_count: u32, max_samples: u32) -> f32 {
    let cap = max_samples.max(1);
    let n = (sample_count + 1).min(cap);
    1.0 / n as f32
}

/// Whether the current surfel geometry has drifted far enough from the previous
/// surfel to invalidate its history (a disocclusion).
///
/// Returns `true` when the anchor displacement exceeds
/// `position_tolerance * radius` or the normals disagree beyond
/// `normal_tolerance`.  Degenerate inputs are treated conservatively as a
/// disocclusion so stale radiance is never trusted.
#[must_use]
pub fn is_disoccluded(prev: &Surfel, curr: &Surfel, params: &TemporalParams) -> bool {
    let offset = (curr.position - prev.position).length();
    if !offset.is_finite() {
        return true;
    }
    let radius = prev.radius.max(curr.radius);
    let max_offset = radius * params.position_tolerance.max(0.0);
    if offset > max_offset {
        return true;
    }
    let cos = prev.normal.dot(curr.normal);
    if !cos.is_finite() || cos < params.normal_tolerance {
        return true;
    }
    false
}

/// Advance a surfel's temporal accumulation by one frame.
///
/// `prev` is the previous `(entry, surfel)` state or `None` on first sight.  On
/// disocclusion (or `None`) the sample reseeds a fresh entry; otherwise the
/// previous radiance is blended toward `new_radiance` via
/// [`confidence_alpha`] and the confidence advances toward `max_samples`.
#[must_use]
pub fn integrate_radiance(
    prev: Option<(SurfelCacheEntry, Surfel)>,
    curr: &Surfel,
    new_radiance: Vec3,
    params: &TemporalParams,
) -> SurfelCacheEntry {
    let sample = sanitize_rgb(new_radiance);
    let Some((entry, prev_surfel)) = prev else {
        return SurfelCacheEntry::from_sample(sample);
    };
    if is_disoccluded(&prev_surfel, curr, params) {
        return SurfelCacheEntry::from_sample(sample);
    }
    let alpha = confidence_alpha(entry.sample_count, params.max_samples);
    let radiance = sanitize_rgb(entry.radiance).lerp(sample, alpha);
    let sample_count = (entry.sample_count + 1).min(params.max_samples.max(1));
    SurfelCacheEntry {
        radiance: sanitize_rgb(radiance),
        sample_count,
    }
}

/// Bilateral spatial filter over neighbouring surfels.
///
/// Returns a geometry-weighted average of the centre radiance and its
/// `neighbours` (`(surfel, radiance)` pairs), where each neighbour weight is
/// [`Surfel::geometric_weight`] from the centre.  The centre always contributes
/// unit weight so the filter degrades to identity when every neighbour is
/// incompatible; a locally constant signal is reproduced exactly.
#[must_use]
pub fn spatial_filter(
    center: &Surfel,
    center_radiance: Vec3,
    neighbours: &[(Surfel, Vec3)],
    params: &CoverageParams,
) -> Vec3 {
    let center_rgb = sanitize_rgb(center_radiance);
    let mut sum = center_rgb;
    let mut weight = 1.0_f32;
    for (surfel, radiance) in neighbours {
        let w = center.geometric_weight(surfel, params);
        if w <= 0.0 {
            continue;
        }
        sum += sanitize_rgb(*radiance) * w;
        weight += w;
    }
    if weight <= 1.0e-12 {
        center_rgb
    } else {
        sanitize_rgb(sum / weight)
    }
}

/// Replace any non-finite component of an RGB triple with `0` and clamp to be
/// non-negative.
#[must_use]
fn sanitize_rgb(c: Vec3) -> Vec3 {
    Vec3::new(
        finite_or_zero(c.x),
        finite_or_zero(c.y),
        finite_or_zero(c.z),
    )
    .max(Vec3::ZERO)
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

    fn surfel() -> Surfel {
        Surfel::new(Vec3::ZERO, Vec3::Z, 1.0)
    }

    #[test]
    fn alpha_decreases_and_floors() {
        let a0 = confidence_alpha(0, 32); // 1/min(1,32) = 1
        let a1 = confidence_alpha(1, 32); // 1/2
        let a5 = confidence_alpha(5, 32); // 1/6
        let floor = confidence_alpha(1000, 32); // 1/32
        assert!((a0 - 1.0).abs() < 1e-6);
        assert!((a1 - 0.5).abs() < 1e-6);
        assert!((a5 - 1.0 / 6.0).abs() < 1e-6);
        assert!((floor - 1.0 / 32.0).abs() < 1e-6);
        assert!(a0 > a1 && a1 > a5 && a5 > floor);
    }

    #[test]
    fn first_sight_seeds_entry() {
        let e = integrate_radiance(None, &surfel(), Vec3::splat(2.0), &TemporalParams::default());
        assert_eq!(e.sample_count, 1);
        assert!((e.radiance - Vec3::splat(2.0)).length() < 1e-6);
    }

    #[test]
    fn ema_converges_to_constant_signal() {
        let params = TemporalParams::default();
        let s = surfel();
        let target = Vec3::new(0.5, 1.0, 2.0);
        // Seed from a wrong (zero) history; the EMA must drag it onto target.
        let mut entry = SurfelCacheEntry::from_sample(Vec3::ZERO);
        for _ in 0..256 {
            entry = integrate_radiance(Some((entry, s)), &s, target, &params);
        }
        assert!((entry.radiance - target).length() < 1e-2, "{:?}", entry.radiance);
        assert_eq!(entry.sample_count, params.max_samples);
    }

    #[test]
    fn disocclusion_resets_on_position_jump() {
        let params = TemporalParams::default();
        let prev = surfel();
        // Current surfel jumped two radii away: history must be discarded.
        let curr = Surfel::new(Vec3::new(2.0, 0.0, 0.0), Vec3::Z, 1.0);
        assert!(is_disoccluded(&prev, &curr, &params));
        let entry = SurfelCacheEntry {
            radiance: Vec3::splat(5.0),
            sample_count: 30,
        };
        let out = integrate_radiance(Some((entry, prev)), &curr, Vec3::splat(1.0), &params);
        assert_eq!(out.sample_count, 1);
        assert!((out.radiance - Vec3::splat(1.0)).length() < 1e-6);
    }

    #[test]
    fn disocclusion_resets_on_normal_flip() {
        let params = TemporalParams::default();
        let prev = surfel();
        let curr = Surfel::new(Vec3::ZERO, Vec3::NEG_Z, 1.0);
        assert!(is_disoccluded(&prev, &curr, &params));
    }

    #[test]
    fn no_disocclusion_for_small_motion() {
        let params = TemporalParams::default();
        let prev = surfel();
        let curr = Surfel::new(Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.05, 0.0, 1.0), 1.0);
        assert!(!is_disoccluded(&prev, &curr, &params));
        let entry = SurfelCacheEntry {
            radiance: Vec3::splat(5.0),
            sample_count: 10,
        };
        let out = integrate_radiance(Some((entry, prev)), &curr, Vec3::splat(1.0), &params);
        assert_eq!(out.sample_count, 11);
        // Blended, not reset: stays between history and sample.
        assert!(out.radiance.x < 5.0 && out.radiance.x > 1.0);
    }

    #[test]
    fn spatial_filter_preserves_constant_signal() {
        let center = surfel();
        let params = CoverageParams::default();
        let c = Vec3::new(0.3, 0.6, 0.9);
        let neighbours = [
            (Surfel::new(Vec3::new(0.1, 0.0, 0.0), Vec3::Z, 1.0), c),
            (Surfel::new(Vec3::new(-0.1, 0.1, 0.0), Vec3::Z, 1.0), c),
        ];
        let out = spatial_filter(&center, c, &neighbours, &params);
        assert!((out - c).length() < 1e-6, "{out:?}");
    }

    #[test]
    fn spatial_filter_identity_without_neighbours() {
        let center = surfel();
        let c = Vec3::new(0.2, 0.4, 0.8);
        let out = spatial_filter(&center, c, &[], &CoverageParams::default());
        assert_eq!(out, c);
    }

    #[test]
    fn spatial_filter_ignores_incompatible_neighbours() {
        let center = surfel();
        let c = Vec3::splat(1.0);
        // Far away + opposed normal => zero weight, output unchanged.
        let far = (Surfel::new(Vec3::new(10.0, 0.0, 0.0), Vec3::NEG_Z, 1.0), Vec3::splat(50.0));
        let out = spatial_filter(&center, c, &[far], &CoverageParams::default());
        assert!((out - c).length() < 1e-6);
    }

    #[test]
    fn spatial_filter_pulls_toward_neighbour() {
        let center = surfel();
        let params = CoverageParams::default();
        let neighbours = [(Surfel::new(Vec3::new(0.05, 0.0, 0.0), Vec3::Z, 1.0), Vec3::splat(2.0))];
        let out = spatial_filter(&center, Vec3::ZERO, &neighbours, &params);
        // Weighted mean of 0 and 2 with positive neighbour weight lies in (0, 2).
        assert!(out.x > 0.0 && out.x < 2.0, "{out:?}");
    }

    #[test]
    fn integration_sanitizes_nan_inputs() {
        let params = TemporalParams::default();
        let out = integrate_radiance(None, &surfel(), Vec3::splat(f32::NAN), &params);
        assert!(out.radiance.is_finite());
        assert_eq!(out.radiance, Vec3::ZERO);
    }

    #[test]
    fn spatial_filter_sanitizes_nan() {
        let center = surfel();
        let n = [(Surfel::new(Vec3::new(0.05, 0.0, 0.0), Vec3::Z, 1.0), Vec3::splat(f32::NAN))];
        let out = spatial_filter(&center, Vec3::splat(1.0), &n, &CoverageParams::default());
        assert!(out.is_finite());
    }

    #[test]
    fn determinism() {
        let params = TemporalParams::default();
        let s = surfel();
        let entry = SurfelCacheEntry::from_sample(Vec3::splat(0.5));
        let a = integrate_radiance(Some((entry, s)), &s, Vec3::splat(1.0), &params);
        let b = integrate_radiance(Some((entry, s)), &s, Vec3::splat(1.0), &params);
        assert_eq!(a, b);
    }
}
