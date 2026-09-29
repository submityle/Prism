//! Cloud-shadow casting and `god ray` (crepuscular / volumetric light-beam)
//! injection planning (design section 12).
//!
//! Clouds both receive and cast shadows. This module owns the *scheduling and
//! weighting* of two effects and deliberately does **not** re-implement the
//! shared virtual shadow map: it computes the light-space `transmittance` a
//! cloud column casts (for terrain / lower-layer shadowing) and the per-sample
//! weights of the screen-space `god ray` march that injects volumetric light
//! beams around the sun. All quantities are unit fractions or bounded sums, so
//! the effect can be dialed by exposure without going black or blowing out.
//!
//! Cloud shadow `transmittance` is the Beer-Lambert survival term
//! `exp(-optical_depth)`; the `god ray` weights form a geometric series in the
//! sample index (a decaying march away from the light), whose partial sums are
//! bounded by `weight / (1 - decay)`. Everything is pure and deterministic and
//! reaches no transcendental intrinsic beyond the shared
//! [`super::math::exp_approx`], so it matches the `GPU` `WESL` kernel bit for
//! bit on the reference path.

use super::math::{exp_approx, pow_approx, saturate};

/// Cloud-shadow cascade / march configuration.
///
/// The cascades split the shadow range the way the shared virtual shadow map
/// does; this struct only records the counts and scales the cloud pass needs,
/// never the shadow-map storage itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudShadowConfig {
    /// Number of shadow cascades the cloud pass writes into.
    pub cascade_count: u32,
    /// Farthest distance (world units) cloud shadows are cast.
    pub max_shadow_distance: f32,
    /// Samples marched along the light ray when accumulating optical depth.
    pub step_count: u32,
    /// Multiplier applied to sampled density before accumulation.
    pub density_scale: f32,
}

impl Default for CloudShadowConfig {
    fn default() -> Self {
        Self {
            cascade_count: 4,
            max_shadow_distance: 20_000.0,
            step_count: 16,
            density_scale: 1.0,
        }
    }
}

/// Beer-Lambert cloud-shadow `transmittance` for an accumulated optical depth.
///
/// Negative optical depths are clamped to zero (a physical column never adds
/// light), so the result is monotone non-increasing in optical depth and always
/// in `[0, 1]`.
#[must_use]
pub fn shadow_transmittance(optical_depth: f32) -> f32 {
    let od = if optical_depth > 0.0 {
        optical_depth
    } else {
        0.0
    };
    saturate(exp_approx(-od))
}

/// Accumulates optical depth along the light ray and returns the surviving
/// `transmittance`.
///
/// Each of `density_samples` contributes `max(density, 0) * step` to the
/// optical depth; an empty sample slice yields zero optical depth and thus a
/// `transmittance` of `1`. Denser or more numerous samples only ever lower the
/// result, so it is monotone non-increasing in the accumulated density, and it
/// is always in `[0, 1]`.
#[must_use]
pub fn accumulate_shadow(density_samples: &[f32], step: f32) -> f32 {
    let step = if step > 0.0 { step } else { 0.0 };
    let mut optical_depth = 0.0;
    for &d in density_samples {
        let density = if d > 0.0 { d } else { 0.0 };
        optical_depth += density * step;
    }
    shadow_transmittance(optical_depth)
}

/// `god ray` (crepuscular light-beam) march configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GodRayConfig {
    /// Number of radial samples marched from the fragment toward the light.
    pub sample_count: u32,
    /// Per-sample geometric decay in `[0, 1]`; smaller means faster falloff.
    pub decay: f32,
    /// Overall beam weight (the first sample's contribution) in `[0, 1]`.
    pub weight: f32,
    /// Exposure multiplier applied to the composited beam.
    pub exposure: f32,
}

impl Default for GodRayConfig {
    fn default() -> Self {
        Self {
            sample_count: 64,
            decay: 0.96,
            weight: 0.6,
            exposure: 1.0,
        }
    }
}

/// Weight of the `god ray` sample at `sample_index`.
///
/// The weight is `weight * decay^index`, a geometric decay away from the light.
/// `decay` and `weight` are saturated into `[0, 1]`, so the result is always in
/// `[0, 1]`, is monotone non-increasing in `sample_index`, and the partial sums
/// over any number of samples are bounded by `weight / (1 - decay)`.
#[must_use]
pub fn god_ray_weight(sample_index: u32, cfg: GodRayConfig) -> f32 {
    let decay = saturate(cfg.decay);
    let weight = saturate(cfg.weight);
    saturate(weight * pow_approx(decay, sample_index as f32))
}

/// Screen-space mask gating `god ray` injection at a fragment.
///
/// Volumetric beams are visible where light reaches the fragment (high
/// `shadow_transmittance`) *and* there is medium to scatter off (nonzero
/// density). The mask is the product of the two saturated terms, so it is
/// always in `[0, 1]` and vanishes in either full shadow or clear air.
#[must_use]
pub fn scattering_mask(shadow_transmittance: f32, density: f32) -> f32 {
    saturate(shadow_transmittance) * saturate(density)
}

#[cfg(test)]
mod tests {
    use super::super::math::EPS;
    use super::*;

    #[test]
    fn shadow_transmittance_is_unit_ranged_and_monotone() {
        let mut prev = shadow_transmittance(0.0);
        assert_eq!(prev, 1.0);
        let mut od = 0.0;
        while od <= 20.0 {
            let t = shadow_transmittance(od);
            assert!((0.0..=1.0).contains(&t));
            assert!(t <= prev + EPS, "not monotone at od={od}: {t} > {prev}");
            prev = t;
            od += 0.25;
        }
        // Negative optical depth clamps to full transmittance.
        assert_eq!(shadow_transmittance(-5.0), 1.0);
    }

    #[test]
    fn accumulate_empty_samples_is_full_transmittance() {
        assert_eq!(accumulate_shadow(&[], 4.0), 1.0);
        // Zero step also leaves transmittance untouched.
        assert_eq!(accumulate_shadow(&[1.0, 1.0, 1.0], 0.0), 1.0);
    }

    #[test]
    fn accumulate_is_monotone_non_increasing_in_density() {
        let step = 2.0;
        let light = accumulate_shadow(&[0.1, 0.1, 0.1], step);
        let heavy = accumulate_shadow(&[0.9, 0.9, 0.9], step);
        assert!(heavy <= light);
        assert!((0.0..=1.0).contains(&light));
        assert!((0.0..=1.0).contains(&heavy));
        // Negative samples are treated as empty (no negative optical depth).
        assert_eq!(accumulate_shadow(&[-1.0, -2.0], step), 1.0);
        // Adding a sample never raises transmittance.
        let a = accumulate_shadow(&[0.3, 0.3], step);
        let b = accumulate_shadow(&[0.3, 0.3, 0.3], step);
        assert!(b <= a + EPS);
    }

    #[test]
    fn god_ray_weight_is_unit_ranged_and_decreasing() {
        let cfg = GodRayConfig {
            sample_count: 32,
            decay: 0.8,
            weight: 1.0,
            exposure: 1.0,
        };
        let mut prev = god_ray_weight(0, cfg);
        assert!((0.0..=1.0).contains(&prev));
        for i in 1..cfg.sample_count {
            let w = god_ray_weight(i, cfg);
            assert!((0.0..=1.0).contains(&w));
            assert!(w <= prev + EPS, "weight rose at i={i}: {w} > {prev}");
            prev = w;
        }
    }

    #[test]
    fn god_ray_weight_partial_sum_is_bounded() {
        let cfg = GodRayConfig {
            sample_count: 128,
            decay: 0.75,
            weight: 1.0,
            exposure: 1.0,
        };
        let mut sum = 0.0;
        for i in 0..cfg.sample_count {
            sum += god_ray_weight(i, cfg);
        }
        // Geometric bound weight / (1 - decay) with a small approximation slack.
        let bound = cfg.weight / (1.0 - cfg.decay);
        assert!(sum <= bound + 1.0e-2, "sum {sum} exceeds bound {bound}");
    }

    #[test]
    fn scattering_mask_is_unit_ranged() {
        assert_eq!(scattering_mask(1.0, 1.0), 1.0);
        assert_eq!(scattering_mask(0.0, 1.0), 0.0);
        assert_eq!(scattering_mask(1.0, 0.0), 0.0);
        // Out-of-range inputs saturate rather than panic.
        let m = scattering_mask(5.0, -3.0);
        assert!((0.0..=1.0).contains(&m));
        let m2 = scattering_mask(0.5, 0.5);
        assert_eq!(m2, 0.25);
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(shadow_transmittance(3.3), shadow_transmittance(3.3));
        assert_eq!(
            accumulate_shadow(&[0.2, 0.4, 0.6], 1.5),
            accumulate_shadow(&[0.2, 0.4, 0.6], 1.5)
        );
        let cfg = GodRayConfig::default();
        assert_eq!(god_ray_weight(7, cfg), god_ray_weight(7, cfg));
        assert_eq!(
            god_ray_weight(7, cfg).to_bits(),
            god_ray_weight(7, cfg).to_bits()
        );
    }

    #[test]
    fn out_of_range_inputs_do_not_panic() {
        // Huge index flushes the weight toward zero without panic.
        let cfg = GodRayConfig::default();
        let w = god_ray_weight(u32::MAX, cfg);
        assert!((0.0..=1.0).contains(&w));
        // Degenerate decay / weight stay in range.
        let bad = GodRayConfig {
            sample_count: 4,
            decay: 5.0,
            weight: -2.0,
            exposure: 1.0,
        };
        for i in 0..bad.sample_count {
            assert!((0.0..=1.0).contains(&god_ray_weight(i, bad)));
        }
        // Huge optical depth flushes transmittance to zero.
        assert_eq!(shadow_transmittance(1.0e9), 0.0);
    }
}
