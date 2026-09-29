//! Underwater volume: extinction, colour shift, phase, god rays, visibility.
//!
//! Below the surface the water is a participating medium: light is absorbed and
//! scattered along every path, so distant geometry fades, warm colours vanish
//! first, and shafts of light (god rays) glow where the surface focuses the sun
//! into the haze. This module provides the pure, deterministic parameter math
//! feeding the shared froxel volume — extinction, per-channel colour shift, the
//! `Henyey-Greenstein` phase function, a bounded multiple-scattering boost, god
//! ray in-scatter, and a visibility test.
//!
//! Extinction follows `Beer-Lambert`, evaluated with the crate's monotone
//! [`exp_approx`](super::exp_approx) so transmittance stays in `0..=1` and only
//! `sqrt` is ever used. There are no `f32` equality tests and no AI/ML.

use super::{exp_approx, EPS, PI};

/// Per-channel extinction (absorption plus out-scatter) coefficients.
///
/// Real water attenuates red far faster than blue, so `r > g > b` is the usual
/// ordering and is what drives the characteristic blue-green depth shift.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbExtinction {
    /// Extinction coefficient for the red channel.
    pub r: f32,
    /// Extinction coefficient for the green channel.
    pub g: f32,
    /// Extinction coefficient for the blue channel.
    pub b: f32,
}

/// A linear RGB colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbColor {
    /// Red component.
    pub r: f32,
    /// Green component.
    pub g: f32,
    /// Blue component.
    pub b: f32,
}

/// `Beer-Lambert` transmittance over a path of `distance` at `extinction`.
///
/// Returns `exp(-extinction * distance)` via the monotone
/// [`exp_approx`](super::exp_approx), a value in `0..=1` that falls
/// monotonically as either the path length or the extinction grows. Negative
/// inputs are floored to zero, so transmittance never exceeds one.
#[must_use]
pub fn beer_lambert_transmittance(extinction: f32, distance: f32) -> f32 {
    let e = extinction.max(0.0);
    let d = distance.max(0.0);
    exp_approx(-(e * d)).clamp(0.0, 1.0)
}

/// Applies the depth-dependent colour shift to a surface colour.
///
/// Each channel is multiplied by its own `Beer-Lambert` transmittance over
/// `depth`. Because red extinction is largest, the red channel collapses first,
/// then green, leaving the deep blue-green cast that reads as underwater. Every
/// output channel is non-negative and no brighter than its input.
#[must_use]
pub fn depth_color_shift(color: RgbColor, extinction: RgbExtinction, depth: f32) -> RgbColor {
    RgbColor {
        r: color.r.max(0.0) * beer_lambert_transmittance(extinction.r, depth),
        g: color.g.max(0.0) * beer_lambert_transmittance(extinction.g, depth),
        b: color.b.max(0.0) * beer_lambert_transmittance(extinction.b, depth),
    }
}

/// The `Henyey-Greenstein` scattering phase function.
///
/// Returns `(1 - g^2) / (4*PI * (1 + g^2 - 2*g*cos_theta)^(3/2))`, the standard
/// single-lobe phase used for turbid water. The asymmetry `g` in `(-1, 1)` is
/// clamped just inside the open interval to keep the lobe finite; `g > 0` is
/// forward scattering (peaks toward `cos_theta = 1`), `g < 0` backward, and
/// `g = 0` isotropic. The value is always non-negative. The `3/2` power is
/// evaluated as `d * sqrt(d)` so only `sqrt` is used.
#[must_use]
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let gg = g.clamp(-0.999, 0.999);
    let c = cos_theta.clamp(-1.0, 1.0);
    let denom_base = (1.0 + gg * gg - 2.0 * gg * c).max(EPS);
    let denom = 4.0 * PI * denom_base * denom_base.sqrt();
    (1.0 - gg * gg) / denom
}

/// A bounded multiple-scattering amplification of single-scatter radiance.
///
/// Sums the geometric series of successive scattering orders,
/// `single / (1 - albedo)`, the closed form of `single * (1 + a + a^2 + ...)`.
/// The single-scattering `albedo` is clamped below one so the turbid-water glow
/// stays finite; clearer water (low albedo) barely lifts the radiance while
/// milky water (high albedo) boosts it strongly. The result is non-negative.
#[must_use]
pub fn multiple_scatter_boost(single: f32, albedo: f32) -> f32 {
    let a = albedo.clamp(0.0, 0.999);
    single.max(0.0) / (1.0 - a)
}

/// God-ray in-scattered radiance accumulated along a light shaft.
///
/// Models the light that scatters toward the eye along a shaft of length
/// `path_length` as `surface_light * scatter_albedo * (1 - transmittance)`,
/// where the transmittance is the `Beer-Lambert` term over the shaft. The
/// factor `1 - transmittance` is the fraction of the beam that has scattered by
/// the shaft's end, so the glow rises monotonically with path length and
/// saturates. The result is non-negative.
#[must_use]
pub fn godray_inscatter(
    surface_light: f32,
    scatter_albedo: f32,
    extinction: f32,
    path_length: f32,
) -> f32 {
    let light = surface_light.max(0.0);
    let albedo = scatter_albedo.clamp(0.0, 1.0);
    let transmittance = beer_lambert_transmittance(extinction, path_length);
    light * albedo * (1.0 - transmittance)
}

/// Whether geometry at `distance` stays above the visibility `threshold`.
///
/// Compares the `Beer-Lambert` transmittance over `distance` against a
/// transmittance `threshold` in `0..=1`; returns `true` while the target is
/// still discernible through the haze. This avoids taking a logarithm (the
/// crate forbids `ln`) yet still expresses a turbidity-driven visibility range.
#[must_use]
pub fn is_visible(extinction: f32, distance: f32, threshold: f32) -> bool {
    let t = threshold.clamp(0.0, 1.0);
    beer_lambert_transmittance(extinction, distance) >= t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transmittance_decays_and_stays_in_unit_range() {
        assert!((beer_lambert_transmittance(0.0, 100.0) - 1.0).abs() < EPS);
        let near = beer_lambert_transmittance(0.5, 1.0);
        let far = beer_lambert_transmittance(0.5, 10.0);
        assert!(far < near, "more path attenuates more");
        assert!(far >= 0.0 && near <= 1.0);
    }

    #[test]
    fn red_fades_before_blue_with_depth() {
        let color = RgbColor {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        };
        let ext = RgbExtinction {
            r: 0.6,
            g: 0.2,
            b: 0.05,
        };
        let shifted = depth_color_shift(color, ext, 5.0);
        assert!(shifted.r < shifted.g, "red collapses before green");
        assert!(shifted.g < shifted.b, "green collapses before blue");
        assert!(shifted.r >= 0.0 && shifted.b <= 1.0);
    }

    #[test]
    fn phase_is_non_negative_and_forward_peaked() {
        let forward = henyey_greenstein(1.0, 0.6);
        let backward = henyey_greenstein(-1.0, 0.6);
        assert!(forward > backward, "positive g peaks forward");
        assert!(backward >= 0.0);
        // Isotropic g = 0 is the uniform 1 / (4*PI).
        let iso = henyey_greenstein(0.3, 0.0);
        assert!((iso - 1.0 / (4.0 * PI)).abs() < 1e-4);
    }

    #[test]
    fn multiple_scatter_boost_grows_with_albedo_and_is_bounded() {
        let clear = multiple_scatter_boost(1.0, 0.1);
        let milky = multiple_scatter_boost(1.0, 0.9);
        assert!(milky > clear);
        assert!(clear >= 1.0, "boost never below single scatter");
        // Albedo at one clamps instead of dividing by zero.
        assert!(multiple_scatter_boost(1.0, 1.0).is_finite());
    }

    #[test]
    fn godray_rises_with_path_and_saturates() {
        let short = godray_inscatter(1.0, 0.5, 0.3, 1.0);
        let long = godray_inscatter(1.0, 0.5, 0.3, 20.0);
        assert!(long > short, "longer shaft scatters more toward the eye");
        assert!(short >= 0.0);
        // Bounded by surface_light * albedo.
        assert!(long <= 1.0 * 0.5 + EPS);
    }

    #[test]
    fn visibility_shrinks_with_turbidity() {
        // Clear water: a nearby target is visible.
        assert!(is_visible(0.05, 5.0, 0.5));
        // Turbid water: the same target drops below threshold.
        assert!(!is_visible(1.0, 5.0, 0.5));
    }
}
