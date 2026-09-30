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

use alloc::vec;

use super::math::{clamp, exp_approx, saturate};
use super::spectral::{spectral_to_rgb, sunset_reddening, SpectralBands};
use super::{Vec3, EPS};

/// Extinction strength of the normalised exponential `aerial perspective`
/// fade: larger values push the fade toward the near plane. Chosen so the
/// unnormalised opacity `1 - e^{-k}` at the far plane is already near one.
pub const AERIAL_EXTINCTION: f32 = 4.0;

/// Atmospheric scale height in metres, the `e`-folding altitude over which the
/// `aerial perspective` contribution thins as a cloud sample rises.
pub const SCALE_HEIGHT_M: f32 = 8000.0;

/// Solar altitude (radians) used by [`AtmosphereCoupling::IDENTITY`] and the
/// two-argument [`AtmosphereCoupling::new`]: the sun sits at the zenith so
/// [`sunset_inscatter_tint`] returns the neutral `(1, 1, 1)` tint and the
/// coupling reduces exactly to the untinted blend. Any altitude at or above the
/// spectral reddening cut-off would do; the zenith is the unambiguous default.
pub const NO_TWILIGHT_ALTITUDE: f32 = core::f32::consts::FRAC_PI_2;

/// Short-wavelength (blue bucket) weight of the fixed warm twilight spectrum
/// consumed by [`sunset_inscatter_tint`]. It is the smallest of the three so
/// blue is the first channel scattered out of the airlight as the sun sets
/// (the `Rayleigh` reddening signature).
const TWILIGHT_BLUE_WEIGHT: f32 = 0.2;

/// Mid-wavelength (green bucket) weight of the warm twilight spectrum.
const TWILIGHT_GREEN_WEIGHT: f32 = 0.5;

/// Long-wavelength (red bucket) weight of the warm twilight spectrum; the
/// largest of the three, so the surviving airlight is red-dominated at the
/// horizon.
const TWILIGHT_RED_WEIGHT: f32 = 1.0;

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

/// The spectral sunset / twilight tint for the atmosphere in-scatter (design
/// section 8b), as a per-channel multiplier in `0..=1`.
///
/// `sun_altitude` is the sun's angular altitude in radians (see
/// [`sunset_reddening`]): zero at the horizon, positive above it. The reddening
/// amount it returns drives a linear interpolation from the neutral tint
/// `(1, 1, 1)` (sun high, no twilight) toward a warm twilight tint derived
/// entirely from the read-only spectral helpers: a fixed short-to-long
/// twilight [`SpectralBands`] spectrum is collapsed to linear `RGB` via
/// [`spectral_to_rgb`], then rescaled so its brightest channel is exactly one.
///
/// Because every channel of the warm tint is therefore in `0..=1`, and the
/// neutral tint is one, the interpolated tint is in `0..=1` on every channel:
/// multiplying the sampled airlight by it can only *attenuate* (never amplify)
/// a channel, so the downstream [`blend_with_atmosphere`] composite stays
/// energy-conserving. As the sun drops toward the horizon the blue and green
/// channels are attenuated faster than red, warming the airlight. This is the
/// only place the atmosphere hookup consumes the spectral module; it is pure,
/// deterministic, and never `panic`s.
#[must_use]
pub fn sunset_inscatter_tint(sun_altitude: f32) -> Vec3 {
    let reddening = sunset_reddening(sun_altitude);
    let twilight = SpectralBands::new(vec![
        TWILIGHT_BLUE_WEIGHT,
        TWILIGHT_GREEN_WEIGHT,
        TWILIGHT_RED_WEIGHT,
    ]);
    let warm = spectral_to_rgb(&twilight);
    let peak = warm.x.max(warm.y).max(warm.z).max(EPS);
    let warm_tint = warm.scale(1.0 / peak);
    Vec3::splat(1.0).lerp(warm_tint, reddening)
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
    /// Solar altitude (radians) driving the spectral sunset tint applied to the
    /// sampled in-scatter (see [`sunset_inscatter_tint`]). At or above
    /// [`NO_TWILIGHT_ALTITUDE`] the tint is neutral and the coupling reduces to
    /// the untinted blend; lower altitudes warm the airlight toward the horizon.
    pub sun_altitude: f32,
}

impl AtmosphereCoupling {
    /// The identity coupling: full in-scatter, no `transmittance` floor.
    pub const IDENTITY: Self = Self {
        inscatter_scale: 1.0,
        transmittance_floor: 0.0,
        sun_altitude: NO_TWILIGHT_ALTITUDE,
    };

    /// Builds a coupling, saturating both parameters into `0..=1` so the
    /// resulting blend is always bounded regardless of caller input.
    #[must_use]
    pub fn new(inscatter_scale: f32, transmittance_floor: f32) -> Self {
        Self {
            inscatter_scale: saturate(inscatter_scale),
            transmittance_floor: saturate(transmittance_floor),
            sun_altitude: NO_TWILIGHT_ALTITUDE,
        }
    }

    /// Returns a copy of this coupling with its `sun_altitude` (radians) set,
    /// enabling the spectral sunset tint in [`AtmosphereCoupling::apply`]. The
    /// altitude is stored verbatim; [`sunset_inscatter_tint`] clamps it, so any
    /// value is safe.
    #[must_use]
    pub fn with_sun_altitude(mut self, sun_altitude: f32) -> Self {
        self.sun_altitude = sun_altitude;
        self
    }

    /// Applies the coupling: floors the cloud `transmittance`, scales the
    /// sampled `inscatter`, applies the spectral sunset tint for the configured
    /// `sun_altitude`, then defers to [`blend_with_atmosphere`].
    ///
    /// Both scalar parameters are re-saturated here (the public fields may have
    /// been set directly), and the sunset tint is a per-channel multiplier in
    /// `0..=1` (see [`sunset_inscatter_tint`]), so the tinted, scaled in-scatter
    /// never exceeds the sampled radiance and the composite stays
    /// energy-conserving and never panics. At [`NO_TWILIGHT_ALTITUDE`] the tint
    /// is neutral and this reduces exactly to the untinted scale-and-blend.
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
        let tint = sunset_inscatter_tint(self.sun_altitude);
        let scaled = inscatter.scale(saturate(self.inscatter_scale)).mul(tint);
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

    #[test]
    fn sunset_tint_is_neutral_when_the_sun_is_high() {
        // At the default no-twilight altitude the tint collapses to (1, 1, 1),
        // so the coupling reduces exactly to the untinted blend.
        let t = sunset_inscatter_tint(NO_TWILIGHT_ALTITUDE);
        assert!((t.x - 1.0).abs() < TOL, "red neutral, got {}", t.x);
        assert!((t.y - 1.0).abs() < TOL, "green neutral, got {}", t.y);
        assert!((t.z - 1.0).abs() < TOL, "blue neutral, got {}", t.z);
        // Well above the reddening cut-off is equally neutral.
        let high = sunset_inscatter_tint(1.0);
        assert!((high.x - 1.0).abs() < TOL);
        assert!((high.y - 1.0).abs() < TOL);
        assert!((high.z - 1.0).abs() < TOL);
    }

    #[test]
    fn sunset_tint_is_warm_and_bounded_at_the_horizon() {
        // At the horizon the tint is fully warm: red is the (unit) peak, and
        // green/blue are attenuated with blue attenuated at least as much as
        // green. Every channel stays in [0, 1].
        let h = sunset_inscatter_tint(0.0);
        for c in [h.x, h.y, h.z] {
            assert!((0.0..=1.0).contains(&c), "tint channel {c} out of range");
        }
        assert!(
            (h.x - 1.0).abs() < TOL,
            "red must be the unit peak, got {}",
            h.x
        );
        assert!(h.y <= h.x + TOL, "green must not exceed red");
        assert!(
            h.z <= h.y + TOL,
            "blue must be attenuated at least as much as green"
        );
        assert!(
            h.y < h.x - TOL,
            "green must be visibly attenuated at the horizon"
        );
        assert!(
            h.z < h.y - TOL,
            "blue must be visibly attenuated at the horizon"
        );
    }

    #[test]
    fn sunset_tint_blue_recovers_monotonically_as_the_sun_rises() {
        // Blue is the most scattered channel at sunset; as the sun climbs it
        // must recover monotonically toward one, and red stays pinned at the
        // unit peak throughout.
        let mut prev = sunset_inscatter_tint(-0.3);
        let mut i = 1;
        while i <= 100 {
            let altitude = -0.3 + (i as f32) / 100.0 * (NO_TWILIGHT_ALTITUDE + 0.3);
            let cur = sunset_inscatter_tint(altitude);
            for c in [cur.x, cur.y, cur.z] {
                assert!((0.0..=1.0).contains(&c), "tint channel {c} out of range");
            }
            assert!((cur.x - 1.0).abs() < TOL, "red must stay at the unit peak");
            assert!(
                cur.z >= prev.z - TOL,
                "blue must not darken as the sun rises"
            );
            assert!(
                cur.y >= prev.y - TOL,
                "green must not darken as the sun rises"
            );
            prev = cur;
            i += 1;
        }
    }

    #[test]
    fn coupling_sunset_tint_attenuates_airlight_but_stays_bounded() {
        // With a fully opaque cloud the composite is just the (scaled, tinted)
        // airlight, so we can read the tint straight off the output.
        let air = Vec3::splat(0.8);
        let neutral = AtmosphereCoupling::new(1.0, 0.0);
        let sunset = AtmosphereCoupling::new(1.0, 0.0).with_sun_altitude(0.0);
        let neutral_out = neutral.apply(Vec3::ZERO, 1.0, air, 0.0);
        let sunset_out = sunset.apply(Vec3::ZERO, 1.0, air, 0.0);
        // The neutral coupling passes the airlight through untouched.
        assert!((neutral_out.x - 0.8).abs() < TOL);
        assert!((neutral_out.y - 0.8).abs() < TOL);
        assert!((neutral_out.z - 0.8).abs() < TOL);
        // The sunset coupling keeps red, and attenuates green then blue harder.
        assert!(
            (sunset_out.x - 0.8).abs() < TOL,
            "red preserved, got {}",
            sunset_out.x
        );
        assert!(sunset_out.y < neutral_out.y, "green must be attenuated");
        assert!(sunset_out.z < sunset_out.y, "blue must be attenuated most");
        for c in [sunset_out.x, sunset_out.y, sunset_out.z] {
            assert!((0.0..=1.0).contains(&c), "sunset output {c} over-exposed");
        }
    }

    #[test]
    fn with_sun_altitude_only_touches_the_tint_field() {
        let base = AtmosphereCoupling::new(0.5, 0.25);
        let tinted = base.with_sun_altitude(0.0);
        assert_eq!(tinted.inscatter_scale, base.inscatter_scale);
        assert_eq!(tinted.transmittance_floor, base.transmittance_floor);
        assert_eq!(tinted.sun_altitude, 0.0);
    }
}
