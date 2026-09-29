//! Breaking-wave detection and spray emission planning.
//!
//! A wave breaks (whitecaps, plunging crest) when its surface steepens past
//! what gravity can hold: the local displacement steepness climbs, the
//! choppy-displacement Jacobian collapses toward a fold, and the crest
//! curvature spikes. This module turns those three per-sample metrics into a
//! [`BreakingClass`] and a normalized breaking intensity, then plans two
//! downstream effects: a foam source strength written into the dynamic foam
//! field (see `foam`), and an arc-spray particle burst handed to the `Ember`
//! particle engine for crest spume and wind-blown mist.
//!
//! The classifier is a pure, deterministic set of threshold comparisons; no
//! float equality, and the only arithmetic is add/mul/div plus clamping. Spray
//! counts scale linearly with intensity so the plan is exactly reproducible
//! frame to frame.

use super::{Vec3, EPS};

/// Per-sample surface metrics feeding the breaking classifier.
///
/// `steepness` is the displacement-gradient magnitude (dimensionless slope),
/// `jacobian` is the horizontal choppy-displacement Jacobian (`< 0` means the
/// surface has folded over itself), and `curvature` is the crest's local mean
/// curvature magnitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakingSample {
    /// Displacement-gradient steepness (slope magnitude).
    pub steepness: f32,
    /// Choppy-displacement Jacobian (a fold when at or below zero).
    pub jacobian: f32,
    /// Local crest curvature magnitude.
    pub curvature: f32,
}

/// Thresholds above/below which each metric contributes to breaking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakingCriteria {
    /// Steepness at which cresting begins; contribution saturates at twice it.
    pub steepness_threshold: f32,
    /// Jacobian at or below which a fold is counted; a value of `0` is a hard
    /// fold, small positive values catch imminent folds.
    pub jacobian_fold_threshold: f32,
    /// Curvature at which the crest is sharp enough to spume.
    pub curvature_threshold: f32,
    /// Intensity at or above which a sample is classified as fully breaking.
    pub breaking_intensity: f32,
}

/// Qualitative breaking state of a surface sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreakingClass {
    /// Below all thresholds: a smooth, unbroken surface.
    Calm,
    /// Steep but not yet folded: a sharpening crest (choppy, pre-whitecap).
    Cresting,
    /// Folded / high-intensity: an actively breaking, foam-seeding crest.
    Breaking,
}

/// Saturating ramp: `0` at or below `edge`, `1` at or above `2 * edge`.
fn ramp_above(value: f32, edge: f32) -> f32 {
    let scale = if edge > EPS { edge } else { EPS };
    ((value - edge) / scale).clamp(0.0, 1.0)
}

/// Saturating ramp for a metric that grows as it drops below `edge`.
///
/// `0` at or above `edge`, rising to `1` as the value falls to zero, and
/// clamped at `1` for negative values (a completed fold).
fn ramp_below(value: f32, edge: f32) -> f32 {
    let scale = if edge > EPS { edge } else { EPS };
    ((edge - value) / scale).clamp(0.0, 1.0)
}

/// Normalized breaking intensity in `0..=1` from the three metrics.
///
/// The average of the steepness, fold, and curvature ramps. It rises
/// monotonically as steepness or curvature increase and as the Jacobian falls
/// toward a fold, so a sharper, more compressed crest always scores higher.
#[must_use]
pub fn breaking_intensity(sample: BreakingSample, criteria: BreakingCriteria) -> f32 {
    let steep = ramp_above(sample.steepness, criteria.steepness_threshold);
    let fold = ramp_below(sample.jacobian, criteria.jacobian_fold_threshold);
    let curve = ramp_above(sample.curvature, criteria.curvature_threshold);
    ((steep + fold + curve) / 3.0).clamp(0.0, 1.0)
}

/// Classifies a surface sample into a [`BreakingClass`].
///
/// A folded Jacobian or an intensity at/above the breaking threshold is
/// [`BreakingClass::Breaking`]; otherwise any steepness past its threshold is
/// [`BreakingClass::Cresting`]; everything else is [`BreakingClass::Calm`].
#[must_use]
pub fn classify_breaking(sample: BreakingSample, criteria: BreakingCriteria) -> BreakingClass {
    let intensity = breaking_intensity(sample, criteria);
    let folded = sample.jacobian <= criteria.jacobian_fold_threshold;
    if folded || intensity >= criteria.breaking_intensity {
        BreakingClass::Breaking
    } else if sample.steepness > criteria.steepness_threshold {
        BreakingClass::Cresting
    } else {
        BreakingClass::Calm
    }
}

/// Foam-source strength (per second) a breaking sample writes into the foam
/// field. Zero unless the sample is [`BreakingClass::Breaking`]; otherwise it
/// scales with breaking intensity and the caller's `max_rate`.
#[must_use]
pub fn foam_source_strength(
    sample: BreakingSample,
    criteria: BreakingCriteria,
    max_rate: f32,
) -> f32 {
    match classify_breaking(sample, criteria) {
        BreakingClass::Breaking => breaking_intensity(sample, criteria) * max_rate.max(0.0),
        BreakingClass::Calm | BreakingClass::Cresting => 0.0,
    }
}

/// A planned crest-spray particle burst for the `Ember` particle engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SprayEmission {
    /// Number of spray particles to spawn this frame.
    pub count: u32,
    /// Initial velocity: tangential launch along the crest plus an upward jet.
    pub velocity: Vec3,
}

impl SprayEmission {
    /// The empty burst (no spray).
    pub const NONE: Self = Self {
        count: 0,
        velocity: Vec3::ZERO,
    };
}

/// Plans the crest-spray burst for a breaking sample.
///
/// Only [`BreakingClass::Breaking`] samples spew spray. The particle count is
/// `round(intensity * max_count)`, and the launch velocity blends the surface
/// tangent (crest flow direction) with the surface normal (the upward jet),
/// each weighted by `jet_speed`. A calm or merely cresting sample returns
/// [`SprayEmission::NONE`].
#[must_use]
pub fn plan_spray(
    sample: BreakingSample,
    criteria: BreakingCriteria,
    tangent: Vec3,
    normal: Vec3,
    jet_speed: f32,
    max_count: u32,
) -> SprayEmission {
    if classify_breaking(sample, criteria) != BreakingClass::Breaking {
        return SprayEmission::NONE;
    }
    let intensity = breaking_intensity(sample, criteria);
    // Deterministic round-to-nearest without float equality on the fraction.
    let count = (intensity * (max_count as f32) + 0.5) as u32;
    let dir = tangent
        .scale(0.5)
        .add(normal.scale(0.5))
        .normalize_or_zero();
    SprayEmission {
        count,
        velocity: dir.scale(jet_speed * intensity),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CRITERIA: BreakingCriteria = BreakingCriteria {
        steepness_threshold: 1.0,
        jacobian_fold_threshold: 0.2,
        curvature_threshold: 2.0,
        breaking_intensity: 0.5,
    };

    fn sample(steepness: f32, jacobian: f32, curvature: f32) -> BreakingSample {
        BreakingSample {
            steepness,
            jacobian,
            curvature,
        }
    }

    #[test]
    fn calm_surface_is_classified_calm_with_zero_intensity() {
        let s = sample(0.2, 1.0, 0.5);
        assert_eq!(classify_breaking(s, CRITERIA), BreakingClass::Calm);
        assert_eq!(breaking_intensity(s, CRITERIA), 0.0);
        assert_eq!(foam_source_strength(s, CRITERIA, 5.0), 0.0);
    }

    #[test]
    fn steep_but_unfolded_crest_is_cresting() {
        // Steepness just past threshold, Jacobian still healthy, low curvature.
        let s = sample(1.2, 1.0, 0.5);
        assert_eq!(classify_breaking(s, CRITERIA), BreakingClass::Cresting);
    }

    #[test]
    fn folded_jacobian_forces_breaking() {
        let s = sample(0.1, 0.0, 0.1);
        assert_eq!(classify_breaking(s, CRITERIA), BreakingClass::Breaking);
    }

    #[test]
    fn intensity_is_monotonic_in_each_metric() {
        let base = sample(1.0, 0.2, 2.0);
        let steeper = sample(1.6, 0.2, 2.0);
        let folded = sample(1.0, -0.1, 2.0);
        let sharper = sample(1.0, 0.2, 3.0);
        let b = breaking_intensity(base, CRITERIA);
        assert!(breaking_intensity(steeper, CRITERIA) >= b);
        assert!(breaking_intensity(folded, CRITERIA) >= b);
        assert!(breaking_intensity(sharper, CRITERIA) >= b);
    }

    #[test]
    fn intensity_stays_in_unit_range() {
        for &st in &[0.0, 0.5, 1.0, 3.0, 10.0] {
            for &jac in &[-1.0, 0.0, 0.2, 1.0] {
                for &cur in &[0.0, 1.0, 2.0, 8.0] {
                    let i = breaking_intensity(sample(st, jac, cur), CRITERIA);
                    assert!((0.0..=1.0).contains(&i), "intensity out of range: {i}");
                }
            }
        }
    }

    #[test]
    fn foam_source_is_nonnegative_and_scales_with_intensity() {
        let weak = sample(1.0, 0.1, 2.0);
        let strong = sample(3.0, -0.5, 8.0);
        let fw = foam_source_strength(weak, CRITERIA, 10.0);
        let fs = foam_source_strength(strong, CRITERIA, 10.0);
        assert!(fw >= 0.0 && fs >= 0.0);
        assert!(fs >= fw);
    }

    #[test]
    fn spray_only_fires_when_breaking_and_count_scales() {
        let tangent = Vec3::new(1.0, 0.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        // Calm: no spray.
        let calm = plan_spray(sample(0.1, 1.0, 0.1), CRITERIA, tangent, normal, 4.0, 100);
        assert_eq!(calm, SprayEmission::NONE);
        // Full break: maximal count and a real launch velocity.
        let full = plan_spray(sample(4.0, -1.0, 10.0), CRITERIA, tangent, normal, 4.0, 100);
        assert_eq!(full.count, 100);
        assert!(full.velocity.length() > 0.0);
        // Partial break: fewer particles than the full burst.
        let partial = plan_spray(sample(0.1, 0.0, 0.1), CRITERIA, tangent, normal, 4.0, 100);
        assert!(partial.count <= full.count);
    }

    #[test]
    fn spray_direction_is_normalized_scaled_by_intensity() {
        let tangent = Vec3::new(1.0, 0.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let s = sample(4.0, -1.0, 10.0);
        let burst = plan_spray(s, CRITERIA, tangent, normal, 2.0, 10);
        // Intensity is 1 here, jet_speed 2 => speed 2 along the 45-degree jet.
        let speed = burst.velocity.length();
        assert!((speed - 2.0).abs() < 1e-3, "unexpected jet speed: {speed}");
    }
}
