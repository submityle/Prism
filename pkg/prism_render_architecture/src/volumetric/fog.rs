#![forbid(unsafe_code)]
//! Unified volumetric fog and condensation trails (design section 9f).
//!
//! The volumetric engine reuses the same ray-march / `froxel` framework to
//! cover near-ground participating media that is continuous with the sky
//! clouds: height fog, valley / shore fog, and the line-shaped `contrail`
//! sources that aircraft leave at altitude. This module owns the deterministic,
//! allocation-free authoring curves for that medium; the `GPU` `WESL` kernels
//! mirror the same math with native intrinsics.
//!
//! Every routine is a pure function with clamped inputs, so it never panics and
//! never escapes its documented range. Height fog density decays monotonically
//! with altitude (an exponential barometric-style falloff), `transmittance`
//! stays in `0..=1` and is exactly `1` at zero distance, the `contrail`
//! diffusion kernel is analytically normalized so its cross-section integrates
//! to one, and the `froxel` injection weight stays in `0..=1` so the shared
//! `froxel` volume is only *added to*, never overwritten. The only permitted
//! float intrinsic is `f32::sqrt` (used for the Gaussian normalization
//! constant); every transcendental routes through the hand-rolled
//! [`super::math`].

use super::math::{exp_approx, saturate, EPS, TWO_PI};

/// Base cross-section half-width of a freshly formed `contrail`, in world
/// units, before any diffusion widening from [`contrail_spread`].
const CONTRAIL_BASE_SIGMA: f32 = 1.0;

/// Per-second widening rate of a `contrail` cross-section under diffusion and
/// wind shear, used by [`contrail_spread`].
const CONTRAIL_DIFFUSION_RATE: f32 = 0.5;

/// Parameters of an exponential height-fog layer (design section 9f, 地表雾).
///
/// The layer is densest at sea level and thins with altitude; above
/// `max_height` it is culled to exactly zero so the fog volume has a hard,
/// artist-controlled ceiling.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HeightFogParams {
    /// Extinction coefficient at sea level (altitude `0`); floored at `0`.
    pub density_at_sea_level: f32,
    /// Exponential falloff rate per world unit of altitude; floored at `0`.
    pub falloff: f32,
    /// Altitude ceiling above which the fog density is culled to zero.
    pub max_height: f32,
}

/// Height-fog extinction density at a given `altitude`.
///
/// The density decays monotonically (non-increasing) with altitude via
/// `density_at_sea_level * exp(-falloff * altitude)`, and is culled to exactly
/// `0` at or above `max_height`. Negative altitudes are treated as sea level so
/// the value never exceeds `density_at_sea_level`. This is an extinction
/// coefficient, not a probability, so it is only floored at `0` (not clamped to
/// `1`); [`fog_transmittance`] converts it into a bounded `transmittance`.
#[must_use]
pub fn height_fog_density(altitude: f32, params: HeightFogParams) -> f32 {
    let base = params.density_at_sea_level.max(0.0);
    let falloff = params.falloff.max(0.0);
    let ceiling = params.max_height.max(0.0);
    let altitude = altitude.max(0.0);
    if altitude >= ceiling {
        return 0.0;
    }
    base * exp_approx(-falloff * altitude)
}

/// `Beer-Lambert` `transmittance` through a uniform fog slab.
///
/// Returns `exp(-density * distance)` saturated into `0..=1`. Negative inputs
/// are floored at `0`, so a zero distance (or zero density) yields exactly `1`
/// and larger optical depths decay monotonically toward `0`.
#[must_use]
pub fn fog_transmittance(distance: f32, density: f32) -> f32 {
    let optical_depth = density.max(0.0) * distance.max(0.0);
    saturate(exp_approx(-optical_depth))
}

/// A line-shaped `contrail` condensation source that widens and fades with age
/// (design section 9f, `contrail` 凝结尾迹).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Contrail {
    /// Age in seconds since the trail formed; drives diffusion widening.
    pub age: f32,
    /// Authored base cross-section width in world units.
    pub width: f32,
    /// Diffusion coefficient scaling how fast the trail spreads with age.
    pub diffusion: f32,
}

/// Effective Gaussian standard deviation of a `contrail` cross-section.
///
/// Combines the authored half-width with an age-scaled diffusion term and is
/// floored at [`EPS`] so the kernel normalization never divides by (near) zero.
#[must_use]
fn effective_sigma(contrail: Contrail) -> f32 {
    let half_width = 0.5 * contrail.width.max(0.0);
    let spread = contrail.diffusion.max(0.0) * contrail.age.max(0.0);
    (half_width + spread).max(EPS)
}

/// Normalized diffusion kernel value at a cross-section `offset`.
///
/// A unit-area Gaussian: `1 / (sigma * sqrt(2*pi)) * exp(-0.5 * (offset/sigma)^2)`,
/// where `sigma` is the [`effective_sigma`]. Because the analytic normalization
/// constant is applied, integrating this kernel across the full cross-section
/// yields `1`, so injecting a `contrail` conserves its total mass regardless of
/// how much it has diffused. The value is always non-negative.
#[must_use]
pub fn contrail_kernel(offset: f32, contrail: Contrail) -> f32 {
    let sigma = effective_sigma(contrail);
    let norm = 1.0 / (sigma * TWO_PI.sqrt());
    let z = offset / sigma;
    norm * exp_approx(-0.5 * z * z)
}

/// Cross-section spread (half-width proxy) of a `contrail` at a given `age`.
///
/// Increases monotonically with age from [`CONTRAIL_BASE_SIGMA`] at a rate of
/// [`CONTRAIL_DIFFUSION_RATE`] per second, modelling diffusion and wind shear.
/// Negative ages are floored at zero.
#[must_use]
pub fn contrail_spread(age: f32) -> f32 {
    CONTRAIL_BASE_SIGMA + CONTRAIL_DIFFUSION_RATE * age.max(0.0)
}

/// Injection weight for a `froxel` depth slice into the shared `froxel` volume.
///
/// Near slices (small `depth_slice`) receive full weight and far slices taper
/// to zero, so the fog only *adds* energy to the shared volume without
/// overwriting deeper contributions. Returns `0` when `slice_count` is zero and
/// clamps `depth_slice` to the slice count, so the result is always in `0..=1`
/// and never panics.
#[must_use]
pub fn froxel_injection_weight(depth_slice: u32, slice_count: u32) -> f32 {
    if slice_count == 0 {
        return 0.0;
    }
    let slice = depth_slice.min(slice_count);
    saturate(1.0 - slice as f32 / slice_count as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn height_fog_density_decays_monotonically_and_is_non_negative() {
        let params = HeightFogParams {
            density_at_sea_level: 0.8,
            falloff: 0.02,
            max_height: 1000.0,
        };
        let mut prev = height_fog_density(0.0, params);
        let mut alt = 0.0;
        while alt <= 900.0 {
            let d = height_fog_density(alt, params);
            assert!(d >= 0.0, "fog density went negative: {d}");
            assert!(
                d <= prev + EPS,
                "fog density increased with altitude at {alt}"
            );
            prev = d;
            alt += 25.0;
        }
        // Ceiling culls the fog to exactly zero.
        assert!(height_fog_density(1000.0, params).abs() < EPS);
        assert!(height_fog_density(5000.0, params).abs() < EPS);
        // Negative altitude is treated as sea level.
        assert!((height_fog_density(-50.0, params) - 0.8).abs() < EPS);
    }

    #[test]
    fn fog_transmittance_stays_in_unit_range_and_is_one_at_zero_distance() {
        assert!(
            (fog_transmittance(0.0, 1.0) - 1.0).abs() < EPS,
            "T(0) must be 1"
        );
        assert!(
            (fog_transmittance(10.0, 0.0) - 1.0).abs() < EPS,
            "zero density is 1"
        );
        let mut prev = 1.0;
        let mut dist = 0.0;
        while dist <= 20.0 {
            let t = fog_transmittance(dist, 0.3);
            assert!((0.0..=1.0).contains(&t), "transmittance out of range: {t}");
            assert!(
                t <= prev + EPS,
                "transmittance increased with distance at {dist}"
            );
            prev = t;
            dist += 0.5;
        }
        // Negative inputs are guarded.
        assert!((fog_transmittance(-5.0, 0.5) - 1.0).abs() < EPS);
        assert!((fog_transmittance(5.0, -0.5) - 1.0).abs() < EPS);
    }

    #[test]
    fn contrail_kernel_integrates_to_one() {
        // A moderately diffused trail; integrate the normalized cross-section.
        let contrail = Contrail {
            age: 4.0,
            width: 2.0,
            diffusion: 0.3,
        };
        let dx = 0.005;
        let mut integral = 0.0;
        let mut offset = -40.0;
        while offset <= 40.0 {
            integral += contrail_kernel(offset, contrail) * dx;
            // The kernel is always non-negative.
            assert!(
                contrail_kernel(offset, contrail) >= 0.0,
                "kernel went negative"
            );
            offset += dx;
        }
        assert!(
            (integral - 1.0).abs() < 0.02,
            "kernel integral not unit: {integral}"
        );
    }

    #[test]
    fn contrail_kernel_degenerate_inputs_do_not_panic() {
        let degenerate = Contrail {
            age: 0.0,
            width: 0.0,
            diffusion: 0.0,
        };
        let k = contrail_kernel(0.0, degenerate);
        assert!(
            k.is_finite() && k >= 0.0,
            "degenerate kernel not finite: {k}"
        );
    }

    #[test]
    fn contrail_spread_increases_monotonically_with_age() {
        let mut prev = contrail_spread(0.0);
        let mut age = 0.0;
        while age <= 60.0 {
            let s = contrail_spread(age);
            assert!(s + EPS >= prev, "spread not monotonic at age {age}");
            prev = s;
            age += 1.0;
        }
        // Negative age floors at the base sigma.
        assert!((contrail_spread(-5.0) - CONTRAIL_BASE_SIGMA).abs() < EPS);
    }

    #[test]
    fn froxel_injection_weight_stays_in_unit_range() {
        let slices = 16;
        let mut prev = froxel_injection_weight(0, slices);
        let mut i = 0;
        while i <= slices + 4 {
            let w = froxel_injection_weight(i, slices);
            assert!((0.0..=1.0).contains(&w), "froxel weight out of range: {w}");
            assert!(w <= prev + EPS, "froxel weight not non-increasing at {i}");
            prev = w;
            i += 1;
        }
        // Zero slice count is guarded.
        assert!(froxel_injection_weight(3, 0).abs() < EPS);
    }

    #[test]
    fn fog_routines_are_deterministic() {
        let params = HeightFogParams {
            density_at_sea_level: 0.6,
            falloff: 0.01,
            max_height: 800.0,
        };
        assert_eq!(
            height_fog_density(123.0, params),
            height_fog_density(123.0, params)
        );
        assert_eq!(fog_transmittance(7.0, 0.4), fog_transmittance(7.0, 0.4));
        let contrail = Contrail {
            age: 2.0,
            width: 1.5,
            diffusion: 0.2,
        };
        assert_eq!(
            contrail_kernel(0.7, contrail),
            contrail_kernel(0.7, contrail)
        );
        assert_eq!(
            froxel_injection_weight(5, 16),
            froxel_injection_weight(5, 16)
        );
    }
}
