//! `aerial perspective` atmosphere hookup for the volumetric cloud subsystem
//! (design section 8, testability section 16).
//!
//! The atmosphere itself is a *shared base service* (the material "atmosphere =
//! lighting data service" tier): it owns the pre-integrated `transmittance` /
//! multi-scatter / `aerial perspective` `LUT`s. The volumetric cloud engine
//! only **consumes** those `LUT`s to blend distant clouds with the scene; it
//! never reimplements or writes the atmosphere `LUT`. This module therefore
//! holds only the read-only *hookup* math:
//!
//! - [`aerial_perspective_weight`] — the distance-driven blend weight in
//!   `0..=1`, monotone non-decreasing with view distance, shaped by a
//!   normalised exponential so it is exactly `0` at the camera and `1` at the
//!   far plane.
//! - [`blend_with_atmosphere`] — an energy-conserving, read-only composite of a
//!   cloud colour with sampled atmosphere in-scatter; every output channel is a
//!   convex combination of its inputs, so the blend never over-exposes.
//! - [`AtmosphereCoupling`] — a small config (in-scatter scale + `transmittance`
//!   floor) whose [`AtmosphereCoupling::apply`] wraps the blend, still sampling
//!   the shared atmosphere only.
//! - [`AerialPerspectiveParams`] — a per-sample distance/altitude record with a
//!   combined [`AerialPerspectiveParams::weight`] that also attenuates the blend
//!   with altitude (thinner air aloft).
//!
//! All functions are pure, deterministic, and never panic (inputs are clamped).
//! The determinism policy allows only [`f32::sqrt`] among the float intrinsics,
//! so the exponential fade routes through [`super::math::exp_approx`]. The `GPU`
//! `WESL` resolve uses the native froxel `LUT`; this `CPU` reference exists so
//! the hookup is verifiable in the sandbox.

#![forbid(unsafe_code)]

use super::math::{clamp, exp_approx, saturate};
use super::{Vec3, EPS};

/// Extinction strength of the normalised exponential `aerial perspective`
/// fade: larger values push the fade toward the near plane. Chosen so the
/// unnormalised opacity `1 - e^{-k}` at the far plane is already near one.
pub const AERIAL_EXTINCTION: f32 = 4.0;

/// Atmospheric scale height in metres, the `e`-folding altitude over which the
/// `aerial perspective` contribution thins as a cloud sample rises.
pub const SCALE_HEIGHT_M: f32 = 8000.0;

/// The distance-driven `aerial perspective` blend weight in `0..=1`.
///
/// `distance` and `max_distance` are in metres. The weight is a *normalised*
/// exponential opacity `(1 - e^{-k t}) / (1 - e^{-k})` with `t = distance /
/// max_distance` clamped to `0..=1` and `k` = [`AERIAL_EXTINCTION`]. This is
/// monotone non-decreasing in `distance` (the exponential is monotone), exactly
/// `0` at the camera (`distance = 0`) and exactly `1` at the far plane
/// (`distance = max_distance`), mirroring the froxel fade without reading or
/// rewriting the shared atmosphere `LUT`. A non-positive `max_distance`
/// degenerates safely: any positive distance maps to `1`, otherwise `0`.
#[must_use]
pub fn aerial_perspective_weight(distance: f32, max_distance: f32) -> f32 {
    if max_distance <= EPS {
        return if distance > 0.0 { 1.0 } else { 0.0 };
    }
    let t = saturate(distance / max_distance);
    let numer = 1.0 - exp_approx(-AERIAL_EXTINCTION * t);
    let denom = 1.0 - exp_approx(-AERIAL_EXTINCTION);
    saturate(numer / denom)
}

/// Read-only, energy-conserving blend of a cloud colour with sampled
/// atmosphere in-scatter (the `aerial perspective` composite).
///
/// `cloud_color` is the cloud's out-scattered radiance, `cloud_transmittance`
/// (`0..=1`) is how much of the distant sky shows *through* the cloud,
/// `inscatter` is the airlight sampled from the shared atmosphere service, and
/// `weight` (`0..=1`, typically from [`aerial_perspective_weight`]) is the
/// distance fade. The result is computed as two nested convex combinations:
///
/// 1. composite the cloud over the airlight it transmits
///    (`lerp(cloud_color, inscatter, transmittance)`), then
/// 2. fade that composite toward the airlight with view distance
///    (`lerp(.., inscatter, weight)`).
///
/// Because both steps are convex combinations, every output channel stays
/// within the interval spanned by `cloud_color` and `inscatter`, so the blend
/// is energy-conserving and can never over-expose beyond its brightest source.
/// `weight` and `cloud_transmittance` are saturated, so out-of-range inputs are
/// clamped rather than panicking.
#[must_use]
pub fn blend_with_atmosphere(
    cloud_color: Vec3,
    cloud_transmittance: f32,
    inscatter: Vec3,
    weight: f32,
) -> Vec3 {
    let bg = saturate(cloud_transmittance);
    let w = saturate(weight);
    let composited = cloud_color.lerp(inscatter, bg);
    composited.lerp(inscatter, w)
}

/// Read-only coupling config for the `aerial perspective` hookup: it scales the
/// sampled in-scatter and floors the cloud `transmittance` before blending, and
/// only ever *samples* the shared atmosphere / froxel `LUT` (it never rewrites
/// the atmosphere itself).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmosphereCoupling {
    /// Multiplier applied to sampled in-scatter before blending; saturated to
    /// `0..=1` so a config can only *attenuate* airlight, never amplify it
    /// past the sampled radiance (keeps [`AtmosphereCoupling::apply`] bounded).
    pub inscatter_scale: f32,
    /// Minimum cloud `transmittance` used by the blend, in `0..=1`; a floor
    /// above zero keeps a sliver of sky showing through even the densest cloud.
    pub transmittance_floor: f32,
}

impl AtmosphereCoupling {
    /// The identity coupling: full in-scatter, no `transmittance` floor.
    pub const IDENTITY: Self = Self {
        inscatter_scale: 1.0,
        transmittance_floor: 0.0,
    };

    /// Builds a coupling, saturating both parameters into `0..=1` so the
    /// resulting blend is always bounded regardless of caller input.
    #[must_use]
    pub fn new(inscatter_scale: f32, transmittance_floor: f32) -> Self {
        Self {
            inscatter_scale: saturate(inscatter_scale),
            transmittance_floor: saturate(transmittance_floor),
        }
    }

    /// Applies the coupling: floors the cloud `transmittance`, scales the
    /// sampled `inscatter`, then defers to [`blend_with_atmosphere`].
    ///
    /// Both parameters are re-saturated here (the public fields may have been
    /// set directly), so the scaled in-scatter never exceeds the sampled
    /// radiance and the composite stays energy-conserving and never panics.
    #[must_use]
    pub fn apply(
        &self,
        cloud_color: Vec3,
        cloud_transmittance: f32,
        inscatter: Vec3,
        weight: f32,
    ) -> Vec3 {
        let floor = saturate(self.transmittance_floor);
        let transmittance = clamp(cloud_transmittance, floor, 1.0);
        let scaled = inscatter.scale(saturate(self.inscatter_scale));
        blend_with_atmosphere(cloud_color, transmittance, scaled, weight)
    }
}

impl Default for AtmosphereCoupling {
    /// The [`AtmosphereCoupling::IDENTITY`] coupling.
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Per-sample `aerial perspective` inputs: how far the cloud sample is and how
/// high it sits, both in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AerialPerspectiveParams {
    /// View distance from the camera to the cloud sample, in metres.
    pub distance: f32,
    /// Altitude of the cloud sample above the ground, in metres.
    pub altitude: f32,
}

impl AerialPerspectiveParams {
    /// Builds the parameters from a `distance` and `altitude` (metres).
    #[must_use]
    pub fn new(distance: f32, altitude: f32) -> Self {
        Self { distance, altitude }
    }

    /// Altitude attenuation factor in `(0, 1]`: `e^{-altitude / H}` with `H` =
    /// [`SCALE_HEIGHT_M`]. It decreases monotonically with altitude (thinner
    /// air aloft scatters less), is exactly `1` at the ground, and never
    /// reaches zero. Negative altitudes are clamped to the ground value.
    #[must_use]
    pub fn altitude_attenuation(&self) -> f32 {
        exp_approx(-self.altitude.max(0.0) / SCALE_HEIGHT_M)
    }

    /// Combined `aerial perspective` weight in `0..=1`: the distance weight
    /// from [`aerial_perspective_weight`] scaled by [`altitude_attenuation`].
    /// For a fixed altitude it is monotone non-decreasing in `distance`.
    ///
    /// [`altitude_attenuation`]: AerialPerspectiveParams::altitude_attenuation
    #[must_use]
    pub fn weight(&self, max_distance: f32) -> f32 {
        saturate(
            aerial_perspective_weight(self.distance, max_distance) * self.altitude_attenuation(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for equality of computed reals in these tests.
    const TOL: f32 = 1e-4;

    /// A far plane distance shared by the distance-fade tests.
    const MAX_D: f32 = 1000.0;

    #[test]
    fn weight_stays_in_range_and_clamps_out_of_range() {
        // Includes negative and beyond-far-plane distances: all clamp, none panic.
        let samples = [-500.0, -1.0, 0.0, 1.0, 250.0, 500.0, 999.0, 1000.0, 5000.0];
        let mut i = 0;
        while i < samples.len() {
            let w = aerial_perspective_weight(samples[i], MAX_D);
            assert!((0.0..=1.0).contains(&w), "weight {w} out of range");
            i += 1;
        }
    }

    #[test]
    fn weight_is_monotone_non_decreasing_in_distance() {
        let mut prev = aerial_perspective_weight(0.0, MAX_D);
        let mut i = 1;
        while i <= 100 {
            let d = (i as f32) / 100.0 * MAX_D;
            let cur = aerial_perspective_weight(d, MAX_D);
            assert!(cur >= prev - TOL, "weight must be non-decreasing at d={d}");
            prev = cur;
            i += 1;
        }
    }

    #[test]
    fn weight_hits_its_endpoints() {
        assert!(
            aerial_perspective_weight(0.0, MAX_D) < TOL,
            "near plane ~ 0"
        );
        assert!(
            (aerial_perspective_weight(MAX_D, MAX_D) - 1.0).abs() < 1e-2,
            "far plane ~ 1"
        );
    }

    #[test]
    fn weight_degenerate_max_distance_is_safe() {
        assert_eq!(aerial_perspective_weight(0.0, 0.0), 0.0);
        assert_eq!(aerial_perspective_weight(5.0, 0.0), 1.0);
        assert_eq!(aerial_perspective_weight(-5.0, -3.0), 0.0);
    }

    #[test]
    fn blend_endpoints_and_stays_bounded() {
        let cloud = Vec3::new(0.8, 0.6, 0.2);
        let air = Vec3::new(0.2, 0.4, 0.9);
        // weight = 1 fades fully to the airlight.
        let far = blend_with_atmosphere(cloud, 0.5, air, 1.0);
        assert!((far.x - air.x).abs() < TOL);
        assert!((far.y - air.y).abs() < TOL);
        assert!((far.z - air.z).abs() < TOL);
        // Every channel of the composite stays within [0, 1] for in-range inputs.
        let mut wi = 0;
        while wi <= 10 {
            let w = (wi as f32) / 10.0;
            let mut ti = 0;
            while ti <= 10 {
                let t = (ti as f32) / 10.0;
                let out = blend_with_atmosphere(cloud, t, air, w);
                assert!((0.0..=1.0).contains(&out.x), "over-exposed x {}", out.x);
                assert!((0.0..=1.0).contains(&out.y), "over-exposed y {}", out.y);
                assert!((0.0..=1.0).contains(&out.z), "over-exposed z {}", out.z);
                ti += 1;
            }
            wi += 1;
        }
    }

    #[test]
    fn blend_is_deterministic() {
        let cloud = Vec3::new(0.7, 0.3, 0.5);
        let air = Vec3::new(0.1, 0.2, 0.8);
        let a = blend_with_atmosphere(cloud, 0.4, air, 0.6);
        let b = blend_with_atmosphere(cloud, 0.4, air, 0.6);
        assert_eq!(a, b);
    }

    #[test]
    fn coupling_applies_floor_and_scale_and_stays_bounded() {
        let cloud = Vec3::new(0.9, 0.7, 0.4);
        let air = Vec3::new(0.2, 0.3, 0.6);
        let coupling = AtmosphereCoupling::new(0.5, 0.25);
        // A cloud transmittance below the floor is lifted to the floor.
        let expected = blend_with_atmosphere(cloud, 0.25, air.scale(0.5), 0.7);
        assert_eq!(coupling.apply(cloud, 0.0, air, 0.7), expected);
        // Out-of-range scale/floor are saturated; output stays in [0, 1].
        let clamped = AtmosphereCoupling::new(5.0, 2.0);
        let out = clamped.apply(cloud, 3.0, air, -1.0);
        assert!((0.0..=1.0).contains(&out.x));
        assert!((0.0..=1.0).contains(&out.y));
        assert!((0.0..=1.0).contains(&out.z));
        assert_eq!(AtmosphereCoupling::default(), AtmosphereCoupling::IDENTITY);
    }

    #[test]
    fn params_weight_is_bounded_and_monotone_in_distance() {
        // Altitude attenuation is in (0, 1] and decreases with altitude.
        let ground = AerialPerspectiveParams::new(500.0, 0.0);
        let aloft = AerialPerspectiveParams::new(500.0, 12000.0);
        assert!((ground.altitude_attenuation() - 1.0).abs() < TOL);
        let ga = ground.altitude_attenuation();
        let aa = aloft.altitude_attenuation();
        assert!(aa > 0.0 && aa < ga, "attenuation must thin with altitude");
        // Combined weight monotone in distance for a fixed altitude.
        let mut prev = AerialPerspectiveParams::new(0.0, 2000.0).weight(MAX_D);
        let mut i = 1;
        while i <= 50 {
            let d = (i as f32) / 50.0 * MAX_D;
            let cur = AerialPerspectiveParams::new(d, 2000.0).weight(MAX_D);
            assert!((0.0..=1.0).contains(&cur));
            assert!(cur >= prev - TOL, "combined weight must be non-decreasing");
            prev = cur;
            i += 1;
        }
    }
}
