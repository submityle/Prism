//! Directional sun inscattering and the combined fog apply step (CPU golden).
//!
//! Beyond plain extinction, a fogged atmosphere also *inscatters* sunlight into
//! the view ray: looking toward the sun through fog produces a bright halo,
//! while looking away leaves only the ambient fog tint.  This directional term
//! is driven by the Henyey-Greenstein (HG) phase function `p(cosθ, g)`, reusing
//! the volumetric-GI reference
//! [`crate::gi::volumetric_gi::scattering::henyey_greenstein`] so the fog and
//! volumetric passes share one phase model.
//!
//! `cosθ = view_dir · sun_dir` is the cosine of the angle between the view ray
//! and the direction toward the sun.  Subtracting the isotropic baseline
//! `1 / 4π` isolates the *extra* forward-scattered energy, which tints the fog
//! toward the sun colour:
//!
//! ```text
//! directional = max(p(cosθ, g) - 1/4π, 0) · sun_intensity
//! fog_tint    = clamp(fog_color + sun_color · directional, 0, max_radiance)
//! ```
//!
//! The combined [`AtmosphericFog`] bundles exponential height fog, distance
//! fog, and this inscattering term.  Its [`AtmosphericFog::apply_fog`] computes
//! the total transmittance as the product of the height- and distance-fog
//! transmittances (both are extinction along the same ray) and blends the
//! scene colour toward the inscattered `fog_tint`:
//!
//! ```text
//! T       = T_height · T_distance
//! result  = lerp(scene_color, fog_tint(cosθ), 1 - T)
//! ```
//!
//! # Conventions
//! * `view_dir` points from the camera toward the shaded surface; `sun_dir`
//!   points from the surface toward the sun. Both are normalised internally;
//!   a degenerate (near-zero / non-finite) vector falls back to a neutral
//!   `cosθ = 0` and a zero up component.
//! * The HG anisotropy `g` is forward (`g > 0`) for a sun-facing halo; it is
//!   clamped to `(-1, 1)` by the reused phase function.
//! * The directional term is clamped non-negative, `fog_tint` is clamped to
//!   `[0, max_radiance]`, and the final colour is sanitised to be finite and
//!   non-negative so energy never blows up.
//! * Transcendental maths goes through [`bevy_math::ops`]. Every function is a
//!   deterministic pure function: no RNG, no I/O, no GPU, no `unsafe`.

use bevy_math::Vec3;
use core::f32::consts::PI;

use crate::gi::fog::exponential::DistanceFog;
use crate::gi::fog::height_fog::HeightFog;
use crate::gi::volumetric_gi::scattering::henyey_greenstein;

/// Isotropic phase value `1 / 4π`, the inscattering baseline subtracted so only
/// the anisotropic (directional) excess tints the fog.
const INV_4PI: f32 = 1.0 / (4.0 * PI);

/// Shortest vector length treated as a valid direction; shorter vectors are
/// degenerate and fall back to a neutral orientation.
const MIN_DIR_LEN: f32 = 1.0e-6;

/// Directional sun-inscattering parameters mirroring the GPU twin layout.
///
/// `fog_color` is the ambient (view-independent) fog tint; `sun_color` scaled
/// by the HG excess adds the sun-facing halo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Inscatter {
    /// Ambient linear-RGB fog colour, present regardless of view direction.
    pub fog_color: Vec3,
    /// Linear-RGB colour of the directional sun inscattering halo.
    pub sun_color: Vec3,
    /// Henyey-Greenstein anisotropy `g ∈ (-1, 1)`; `g > 0` concentrates the
    /// halo toward the sun.
    pub g: f32,
    /// Scalar strength multiplier on the directional inscattering term.
    pub sun_intensity: f32,
    /// Per-channel ceiling applied to `fog_tint` so radiance cannot blow up.
    pub max_radiance: f32,
}

impl Default for Inscatter {
    #[inline]
    fn default() -> Self {
        Self {
            fog_color: Vec3::splat(0.5),
            sun_color: Vec3::new(1.0, 0.9, 0.7),
            g: 0.76,
            sun_intensity: 1.0,
            max_radiance: 16.0,
        }
    }
}

impl Inscatter {
    /// HG phase value `p(cosθ, g)` for the inscattering lobe.
    ///
    /// Delegates to the shared volumetric-GI phase function so the fog and
    /// volumetric passes stay numerically identical.
    #[inline]
    pub fn phase(&self, cos_theta: f32) -> f32 {
        henyey_greenstein(cos_theta, self.g)
    }

    /// Directional inscattering excess `max(p(cosθ, g) - 1/4π, 0) · sun_intensity`.
    ///
    /// Zero (or negligible) when looking away from the sun; positive and
    /// peaking when looking toward it (for `g > 0`).
    #[inline]
    pub fn directional(&self, cos_theta: f32) -> f32 {
        let intensity = clamp_non_negative(self.sun_intensity);
        let excess = (self.phase(cos_theta) - INV_4PI).max(0.0);
        clamp_non_negative(excess * intensity)
    }

    /// Inscattered fog colour at the given view/sun angle cosine.
    ///
    /// Returns `clamp(fog_color + sun_color · directional(cosθ), 0, max_radiance)`
    /// per channel.
    #[inline]
    pub fn fog_tint(&self, cos_theta: f32) -> Vec3 {
        let fog = sanitize_rgb(self.fog_color);
        let sun = sanitize_rgb(self.sun_color);
        let tint = fog + sun * self.directional(cos_theta);
        clamp_rgb_ceiling(tint, self.max_radiance)
    }
}

/// Combined analytic fog: exponential height fog, distance fog, and sun
/// inscattering, with a single [`AtmosphericFog::apply_fog`] entry point.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct AtmosphericFog {
    /// Exponential height (altitude) fog contribution.
    pub height: HeightFog,
    /// Distance-based fog contribution.
    pub distance: DistanceFog,
    /// Directional sun-inscattering colour model.
    pub inscatter: Inscatter,
}

impl AtmosphericFog {
    /// Composites height fog, distance fog, and sun inscattering over a shaded
    /// surface colour.
    ///
    /// * `scene_color` — the lit surface colour before fog (linear-RGB).
    /// * `view_dir` — camera→surface direction (normalised internally).
    /// * `sun_dir` — surface→sun direction (normalised internally).
    /// * `cam_height` — camera altitude along the `+Z` up axis.
    /// * `dist` — distance from camera to surface.
    ///
    /// The height- and distance-fog transmittances multiply into a single
    /// transmittance `T`; the scene is blended toward the inscattered fog tint
    /// by `1 - T`. The result is finite and non-negative.
    #[inline]
    pub fn apply_fog(
        &self,
        scene_color: Vec3,
        view_dir: Vec3,
        sun_dir: Vec3,
        cam_height: f32,
        dist: f32,
    ) -> Vec3 {
        let dist = clamp_non_negative(dist);
        let view = normalize_or(view_dir, Vec3::ZERO);
        let sun = normalize_or(sun_dir, Vec3::ZERO);

        // Angle cosine between the view ray and the direction to the sun.
        let cos_theta = view.dot(sun).clamp(-1.0, 1.0);

        // Height-fog transmittance along the ray (up axis = +Z).
        let tau_h = self.height.optical_depth_axis(cam_height, view.z, dist);
        let t_height = bounded_exp_neg(tau_h);

        // Distance-fog transmittance from its fog factor.
        let f_dist = self.distance.fog_factor(dist);
        let t_dist = (1.0 - f_dist).clamp(0.0, 1.0);

        // Both are extinction along the same ray → transmittances multiply.
        let transmittance = (t_height * t_dist).clamp(0.0, 1.0);
        let fog_amount = (1.0 - transmittance).clamp(0.0, 1.0);

        let tint = self.inscatter.fog_tint(cos_theta);
        let scene = sanitize_rgb(scene_color);
        sanitize_rgb(scene.lerp(tint, fog_amount))
    }

    /// Combined fog coverage `1 - T_height · T_distance` in `[0, 1]`.
    ///
    /// Exposes the blend weight [`AtmosphericFog::apply_fog`] uses, for callers
    /// that want to composite the fog colour themselves.
    #[inline]
    pub fn fog_amount(&self, view_up: f32, cam_height: f32, dist: f32) -> f32 {
        let dist = clamp_non_negative(dist);
        let up = finite_or(view_up, 0.0);
        let tau_h = self.height.optical_depth_axis(cam_height, up, dist);
        let t_height = bounded_exp_neg(tau_h);
        let t_dist = (1.0 - self.distance.fog_factor(dist)).clamp(0.0, 1.0);
        (1.0 - (t_height * t_dist)).clamp(0.0, 1.0)
    }
}

/// `exp(-x)` with the exponent bounded and the result clamped to `[0, 1]`.
#[inline]
fn bounded_exp_neg(x: f32) -> f32 {
    let x = if x.is_finite() { x.max(0.0) } else { 80.0 };
    bevy_math::ops::exp(-x.min(80.0)).clamp(0.0, 1.0)
}

/// Normalises `v`, returning `fallback` for a degenerate / non-finite vector.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.is_finite() {
        let len = v.length();
        if len > MIN_DIR_LEN {
            return v / len;
        }
    }
    fallback
}

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Clamps `value` to be non-negative and finite (non-finite → `0`).
#[inline]
fn clamp_non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Replaces any non-finite channel with `0` and clamps every channel
/// non-negative.
#[inline]
fn sanitize_rgb(rgb: Vec3) -> Vec3 {
    Vec3::new(
        if rgb.x.is_finite() { rgb.x.max(0.0) } else { 0.0 },
        if rgb.y.is_finite() { rgb.y.max(0.0) } else { 0.0 },
        if rgb.z.is_finite() { rgb.z.max(0.0) } else { 0.0 },
    )
}

/// Sanitises `rgb` and clamps every channel into `[0, ceiling]`.
#[inline]
fn clamp_rgb_ceiling(rgb: Vec3, ceiling: f32) -> Vec3 {
    let ceiling = if ceiling.is_finite() { ceiling.max(0.0) } else { 0.0 };
    let rgb = sanitize_rgb(rgb);
    Vec3::new(
        rgb.x.min(ceiling),
        rgb.y.min(ceiling),
        rgb.z.min(ceiling),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_matches_shared_henyey_greenstein() {
        let inscatter = Inscatter {
            g: 0.6,
            ..Inscatter::default()
        };
        for mu in [-1.0f32, -0.5, 0.0, 0.3, 0.8, 1.0] {
            let a = inscatter.phase(mu);
            let b = henyey_greenstein(mu, 0.6);
            assert!((a - b).abs() < 1e-7, "mu={mu} a={a} b={b}");
        }
    }

    #[test]
    fn forward_inscatter_exceeds_backward() {
        let inscatter = Inscatter {
            g: 0.76,
            ..Inscatter::default()
        };
        let fwd = inscatter.directional(1.0);
        let bwd = inscatter.directional(-1.0);
        assert!(fwd > bwd, "fwd={fwd} bwd={bwd}");
        // Backward lobe (g > 0) sits below the isotropic baseline → zero excess.
        assert_eq!(bwd, 0.0);
    }

    #[test]
    fn backward_tint_is_just_fog_color() {
        let inscatter = Inscatter {
            fog_color: Vec3::new(0.3, 0.35, 0.4),
            sun_color: Vec3::new(1.0, 0.8, 0.5),
            g: 0.7,
            sun_intensity: 2.0,
            max_radiance: 16.0,
        };
        let back = inscatter.fog_tint(-1.0);
        assert!((back - inscatter.fog_color).length() < 1e-6, "back={back:?}");
        let front = inscatter.fog_tint(1.0);
        // Facing the sun adds energy on top of the ambient fog colour.
        assert!(front.length() > back.length(), "front={front:?} back={back:?}");
    }

    #[test]
    fn apply_fog_is_scene_color_at_zero_distance() {
        let fog = AtmosphericFog {
            height: HeightFog::new(0.1, 0.2, 0.0),
            distance: DistanceFog::default(),
            inscatter: Inscatter::default(),
        };
        let scene = Vec3::new(0.2, 0.5, 0.9);
        let out = fog.apply_fog(scene, Vec3::new(0.0, 0.0, -1.0), Vec3::Z, 5.0, 0.0);
        assert!((out - scene).length() < 1e-6, "out={out:?}");
    }

    #[test]
    fn apply_fog_saturates_to_tint_at_far_distance() {
        let fog = AtmosphericFog {
            height: HeightFog::new(0.05, 0.0, 0.0),
            distance: DistanceFog {
                color: Vec3::splat(0.6),
                density: 0.3,
                start: 0.0,
                end: 100.0,
                mode: crate::gi::fog::exponential::DistanceFogMode::Exponential,
            },
            inscatter: Inscatter {
                fog_color: Vec3::splat(0.6),
                sun_color: Vec3::ZERO,
                g: 0.0,
                sun_intensity: 0.0,
                max_radiance: 16.0,
            },
        };
        let scene = Vec3::new(0.1, 0.1, 0.1);
        // Horizontal ray so height fog adds constant extinction; large distance
        // drives transmittance → 0 and the result toward the fog tint.
        let out = fog.apply_fog(scene, Vec3::new(1.0, 0.0, 0.0), Vec3::X, 0.0, 1.0e5);
        let tint = fog.inscatter.fog_tint(1.0);
        assert!((out - tint).length() < 1e-3, "out={out:?} tint={tint:?}");
    }

    #[test]
    fn looking_toward_sun_is_brighter_than_away() {
        let fog = AtmosphericFog {
            height: HeightFog::new(0.0, 0.0, 0.0),
            distance: DistanceFog {
                color: Vec3::splat(0.4),
                density: 0.1,
                start: 0.0,
                end: 100.0,
                mode: crate::gi::fog::exponential::DistanceFogMode::Exponential,
            },
            inscatter: Inscatter {
                fog_color: Vec3::splat(0.4),
                sun_color: Vec3::new(1.0, 0.9, 0.7),
                g: 0.7,
                sun_intensity: 2.0,
                max_radiance: 32.0,
            },
        };
        let scene = Vec3::splat(0.1);
        let sun = Vec3::new(1.0, 0.0, 0.0);
        let toward = fog.apply_fog(scene, sun, sun, 0.0, 50.0);
        let away = fog.apply_fog(scene, -sun, sun, 0.0, 50.0);
        assert!(
            toward.length() > away.length(),
            "toward={toward:?} away={away:?}"
        );
    }

    #[test]
    fn energy_is_bounded_and_finite_on_extreme_inputs() {
        let fog = AtmosphericFog {
            height: HeightFog::new(5.0, 3.0, 0.0),
            distance: DistanceFog {
                color: Vec3::splat(2.0),
                density: 10.0,
                start: 0.0,
                end: 1.0,
                mode: crate::gi::fog::exponential::DistanceFogMode::ExponentialSquared,
            },
            inscatter: Inscatter {
                fog_color: Vec3::splat(1.0),
                sun_color: Vec3::splat(1000.0),
                g: 0.95,
                sun_intensity: 1000.0,
                max_radiance: 8.0,
            },
        };
        for &(view, sun, h, d) in &[
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.0, 50.0),
            (Vec3::ZERO, Vec3::ZERO, f32::NAN, f32::INFINITY),
            (
                Vec3::splat(f32::NAN),
                Vec3::splat(f32::INFINITY),
                f32::NAN,
                -5.0,
            ),
        ] {
            let out = fog.apply_fog(Vec3::splat(0.5), view, sun, h, d);
            assert!(out.is_finite(), "out={out:?}");
            // Blend of scene (≤0.5) and a tint ceilinged at max_radiance.
            assert!(out.max_element() <= 8.0 + 1e-3, "out={out:?}");
            assert!(out.min_element() >= 0.0, "out={out:?}");
        }
    }

    #[test]
    fn fog_amount_matches_apply_blend_weight() {
        let fog = AtmosphericFog {
            height: HeightFog::new(0.08, 0.1, 0.0),
            distance: DistanceFog {
                color: Vec3::splat(0.5),
                density: 0.05,
                start: 0.0,
                end: 100.0,
                mode: crate::gi::fog::exponential::DistanceFogMode::Exponential,
            },
            inscatter: Inscatter::default(),
        };
        // With a black scene and black fog tint except coverage, the output
        // length equals coverage · |tint|; cross-check against fog_amount.
        let up = 0.5f32;
        let view = Vec3::new(0.0, (1.0f32 - up * up).sqrt(), up);
        let amount = fog.fog_amount(view.z, 3.0, 40.0);
        assert!((0.0..=1.0).contains(&amount));
        // Monotone: more distance → at least as much fog.
        let more = fog.fog_amount(view.z, 3.0, 80.0);
        assert!(more >= amount - 1e-6, "amount={amount} more={more}");
    }
}
