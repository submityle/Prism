//! Particle lighting integration: wrap-diffuse, Beer-Lambert self-shadowing,
//! and premultiplied-alpha compositing — CPU golden reference.
//!
//! Once a billboard fragment has a reconstructed normal (see
//! [`crate::gi::particle_shade::billboard`]) it can be lit like a tiny piece of
//! surface.  Particle media are thin and translucent, so the integration here
//! differs from opaque surface shading in three ways:
//!
//! 1. **Wrap / half-Lambert diffuse** — hard Lambert `max(N·L, 0)` makes the
//!    unlit hemisphere of a soft puff look like a flat cutout.  Wrapping the
//!    cosine (`(N·L * (1 - w) + w)` normalised, with half-Lambert the `w = 0.5`
//!    special case) lets light bleed around the silhouette, the standard look
//!    for smoke and dust.
//! 2. **Beer-Lambert self-shadowing** — a particle's own optical thickness
//!    attenuates light travelling through it; the transmittance
//!    `T = exp(-density * thickness)` darkens denser or thicker puffs.  This
//!    reuses [`crate::gi::volumetric_gi::scattering::beer_lambert`] so the
//!    particle and volumetric passes share one transmittance definition.
//! 3. **Premultiplied-alpha compositing** — translucent particles blend with
//!    `src = premultiplied color`, `dst *= (1 - alpha)`; emissive adds *after*
//!    premultiplication because it is independent of coverage opacity.
//!
//! * [`lambert`] / [`half_lambert`] / [`wrap_diffuse`] — the diffuse response
//!   terms from a clamped or wrapped `N·L`.
//! * [`self_shadow_transmittance`] — Beer-Lambert through-particle attenuation.
//! * [`diffuse_radiance`] — wrapped diffuse × light colour × self-shadow.
//! * [`premultiply`] / [`composite_over`] — premultiplied-alpha helpers.
//! * [`integrate`] — albedo diffuse + emissive combined into a premultiplied
//!   RGB and the compositing weight.
//!
//! # Conventions
//! * `n_dot_l` is `clamp(N·L, -1, 1)` of unit vectors; the diffuse terms map it
//!   to `[0, 1]`.  `wrap` is in `[0, 1]`: `0` is hard Lambert, `1` fully wrapped.
//! * Colours are linear-RGB [`Vec3`]; `alpha` / `density` / `thickness` are
//!   clamped non-negative, `alpha` to `[0, 1]`.
//! * "Premultiplied" means the RGB already carries the coverage factor
//!   (`rgb * alpha`); emissive is added on top un-multiplied-by-coverage.
//! * Every function is a deterministic pure function (no RNG / I/O / GPU /
//!   `unsafe`).  Transcendental maths flows through
//!   [`crate::gi::volumetric_gi::scattering::beer_lambert`]; every result is
//!   finite and never `NaN`.
//!
//! # References
//! * V. Miller / Valve, "Half Lambert" diffuse wrap (Half-Life 2 shading).
//! * NVIDIA GPU Gems 3, Ch. 23 "High-Speed, Off-Screen Particles"
//!   (particle lighting + soft shadows).
//! * T. Porter, T. Duff, "Compositing Digital Images", SIGGRAPH 1984
//!   (premultiplied-alpha `over`).

use crate::gi::volumetric_gi::scattering::beer_lambert;
use bevy_math::Vec3;

/// Clamp a scalar to `[0, 1]`, mapping non-finite input to `0`.
#[inline]
fn saturate(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Clamp a scalar to non-negative, mapping non-finite input to `0`.
#[inline]
fn non_negative(x: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

/// Clamp every channel of a colour to a finite non-negative value.
#[inline]
fn sanitize_rgb(c: Vec3) -> Vec3 {
    Vec3::new(non_negative(c.x), non_negative(c.y), non_negative(c.z))
}

/// A directional light reaching the particle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleLight {
    /// Unit direction **toward** the light (the `L` of `N·L`).
    pub direction: Vec3,
    /// Linear-RGB radiance / colour of the light (non-negative).
    pub color: Vec3,
}

impl ParticleLight {
    /// Construct a light, normalising the direction (zero → `+Y`) and clamping
    /// the colour to non-negative.
    #[inline]
    pub fn new(direction: Vec3, color: Vec3) -> Self {
        let dir = if direction.length_squared().is_finite()
            && direction.length_squared() > 1.0e-12
        {
            direction.normalize()
        } else {
            Vec3::Y
        };
        Self {
            direction: dir,
            color: sanitize_rgb(color),
        }
    }
}

/// Hard Lambertian cosine term `max(N·L, 0)`.
///
/// `n_dot_l` is clamped to `[-1, 1]`; the result is in `[0, 1]`.
#[inline]
pub fn lambert(n_dot_l: f32) -> f32 {
    let c = if n_dot_l.is_finite() {
        n_dot_l.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    c.max(0.0)
}

/// Half-Lambert diffuse `(N·L * 0.5 + 0.5)^2`.
///
/// The classic Valve wrap: remaps `N·L` from `[-1, 1]` to `[0, 1]` and squares
/// it to keep contrast, so even back-facing fragments receive a soft fill.
/// Always in `[0, 1]`.
#[inline]
pub fn half_lambert(n_dot_l: f32) -> f32 {
    let c = if n_dot_l.is_finite() {
        n_dot_l.clamp(-1.0, 1.0)
    } else {
        -1.0
    };
    let h = c * 0.5 + 0.5;
    saturate(h * h)
}

/// Generalised wrap diffuse with wrap factor `wrap in [0, 1]`.
///
/// Computes `max(N·L + wrap, 0) / (1 + wrap)`, the energy-normalised wrap-around
/// term: `wrap = 0` is hard [`lambert`], larger `wrap` pulls the terminator
/// around the silhouette.  Always in `[0, 1]`.
#[inline]
pub fn wrap_diffuse(n_dot_l: f32, wrap: f32) -> f32 {
    let c = if n_dot_l.is_finite() {
        n_dot_l.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let w = saturate(wrap);
    let num = (c + w).max(0.0);
    saturate(num / (1.0 + w))
}

/// Beer-Lambert self-shadow transmittance through the particle.
///
/// Returns `T = exp(-density * thickness) in [0, 1]` via the shared
/// [`beer_lambert`] reference: `1` for a thin / sparse puff, approaching `0`
/// as optical thickness grows.  Both inputs are clamped non-negative.
#[inline]
pub fn self_shadow_transmittance(density: f32, thickness: f32) -> f32 {
    beer_lambert(non_negative(density), non_negative(thickness)).clamp(0.0, 1.0)
}

/// Wrapped diffuse radiance reaching the particle surface.
///
/// `wrap_diffuse(N·L, wrap)` scaled by the light colour and the self-shadow
/// transmittance for `density` / `thickness`.  Returns a finite non-negative
/// linear-RGB radiance.
#[inline]
pub fn diffuse_radiance(
    normal: Vec3,
    light: ParticleLight,
    wrap: f32,
    density: f32,
    thickness: f32,
) -> Vec3 {
    let n = if normal.length_squared() > 1.0e-12 {
        normal.normalize()
    } else {
        return Vec3::ZERO;
    };
    let n_dot_l = n.dot(light.direction);
    let diffuse = wrap_diffuse(n_dot_l, wrap);
    let shadow = self_shadow_transmittance(density, thickness);
    sanitize_rgb(light.color * (diffuse * shadow))
}

/// Premultiply a colour by coverage `alpha` (`rgb * alpha`).
///
/// `alpha` is saturated to `[0, 1]`; the colour is clamped non-negative.
#[inline]
pub fn premultiply(color: Vec3, alpha: f32) -> Vec3 {
    sanitize_rgb(color) * saturate(alpha)
}

/// Premultiplied-alpha `over` composite: `src + dst * (1 - src_alpha)`.
///
/// `src` is assumed already premultiplied; `dst` is the background colour.
/// `src_alpha` is saturated.  Returns a finite non-negative colour.
#[inline]
pub fn composite_over(src: Vec3, src_alpha: f32, dst: Vec3) -> Vec3 {
    let a = saturate(src_alpha);
    sanitize_rgb(src) + sanitize_rgb(dst) * (1.0 - a)
}

/// Integrate albedo diffuse + emissive into premultiplied RGB and a weight.
///
/// `albedo` is the particle's base colour; it is lit by `radiance`
/// (e.g. [`diffuse_radiance`]) and the product is premultiplied by `alpha`.
/// `emissive` is added after premultiplication (self-emission is independent of
/// coverage opacity but is still scaled by `alpha` so a fully transparent
/// fragment contributes nothing).  Returns `(premultiplied_rgb, alpha)` ready
/// for [`composite_over`].
#[inline]
pub fn integrate(
    albedo: Vec3,
    radiance: Vec3,
    emissive: Vec3,
    alpha: f32,
) -> (Vec3, f32) {
    let a = saturate(alpha);
    let lit = sanitize_rgb(albedo) * sanitize_rgb(radiance);
    let rgb = (lit + sanitize_rgb(emissive)) * a;
    (sanitize_rgb(rgb), a)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-5;

    #[test]
    fn lambert_clamps_backface_to_zero() {
        assert!((lambert(1.0) - 1.0).abs() < TOL);
        assert!((lambert(0.5) - 0.5).abs() < TOL);
        assert!(lambert(-0.3).abs() < TOL);
        assert!(lambert(f32::NAN).is_finite());
    }

    #[test]
    fn half_lambert_lifts_backface() {
        // Back-facing (N·L = -1) -> 0, grazing (0) -> 0.25, facing (1) -> 1.
        assert!(half_lambert(-1.0).abs() < TOL);
        assert!((half_lambert(0.0) - 0.25).abs() < TOL);
        assert!((half_lambert(1.0) - 1.0).abs() < TOL);
        // Always lifts the unlit side above hard Lambert.
        assert!(half_lambert(-0.5) >= lambert(-0.5));
    }

    #[test]
    fn wrap_diffuse_reduces_to_lambert_at_zero_wrap() {
        for &c in &[-1.0, -0.5, 0.0, 0.3, 1.0] {
            assert!((wrap_diffuse(c, 0.0) - lambert(c)).abs() < TOL, "c={c}");
        }
    }

    #[test]
    fn wrap_diffuse_softens_terminator() {
        // With wrap, grazing/back angles gain some light vs hard Lambert.
        assert!(wrap_diffuse(0.0, 0.5) > lambert(0.0));
        assert!(wrap_diffuse(-0.2, 0.5) >= lambert(-0.2));
        // Full facing stays normalised to 1.
        assert!((wrap_diffuse(1.0, 0.5) - 1.0).abs() < TOL);
        // Stays in range.
        for &w in &[0.0, 0.25, 0.5, 1.0] {
            for &c in &[-1.0, -0.4, 0.0, 0.7, 1.0] {
                let v = wrap_diffuse(c, w);
                assert!((0.0..=1.0).contains(&v), "oob w={w} c={c} v={v}");
            }
        }
    }

    #[test]
    fn self_shadow_matches_beer_lambert() {
        // Zero optical thickness fully transmits.
        assert!((self_shadow_transmittance(0.0, 10.0) - 1.0).abs() < TOL);
        assert!((self_shadow_transmittance(5.0, 0.0) - 1.0).abs() < TOL);
        // Shares the scattering reference exactly.
        let t = self_shadow_transmittance(1.5, 2.0);
        assert!((t - beer_lambert(1.5, 2.0)).abs() < TOL);
        // Denser puff transmits less.
        assert!(self_shadow_transmittance(2.0, 2.0) < self_shadow_transmittance(0.5, 2.0));
    }

    #[test]
    fn diffuse_radiance_scales_with_light_and_shadow() {
        let n = Vec3::Z;
        let light = ParticleLight::new(Vec3::Z, Vec3::splat(2.0));
        let full = diffuse_radiance(n, light, 0.0, 0.0, 0.0);
        // N·L = 1, no self-shadow -> radiance == light colour.
        assert!((full - Vec3::splat(2.0)).length() < TOL);
        // Add optical thickness -> dimmer.
        let dim = diffuse_radiance(n, light, 0.0, 2.0, 2.0);
        assert!(dim.x < full.x && dim.x > 0.0);
    }

    #[test]
    fn diffuse_radiance_backface_wraps() {
        let n = Vec3::Z;
        // Slightly back-of-grazing so wrap can lift it but hard Lambert cannot.
        let light = ParticleLight::new(Vec3::new(0.5, 0.0, -0.866), Vec3::ONE);
        // Hard Lambert: back-facing is dark.
        assert!(diffuse_radiance(n, light, 0.0, 0.0, 0.0).length() < TOL);
        // Wrapped: receives some light.
        assert!(diffuse_radiance(n, light, 1.0, 0.0, 0.0).length() > 0.0);
    }

    #[test]
    fn light_constructor_normalizes_and_sanitizes() {
        let l = ParticleLight::new(Vec3::new(0.0, 3.0, 0.0), Vec3::new(-1.0, 2.0, f32::NAN));
        assert!((l.direction - Vec3::Y).length() < TOL);
        assert!(l.color.x >= 0.0 && l.color.y >= 0.0 && l.color.z >= 0.0);
        // Degenerate direction falls back to +Y.
        let z = ParticleLight::new(Vec3::ZERO, Vec3::ONE);
        assert!((z.direction - Vec3::Y).length() < TOL);
    }

    #[test]
    fn premultiply_scales_color_by_alpha() {
        let c = premultiply(Vec3::splat(1.0), 0.5);
        assert!((c - Vec3::splat(0.5)).length() < TOL);
        assert!(premultiply(Vec3::ONE, 2.0).x <= 1.0 + TOL);
    }

    #[test]
    fn composite_over_blends_premultiplied() {
        // Opaque src hides dst.
        let c = composite_over(Vec3::new(0.4, 0.0, 0.0), 1.0, Vec3::splat(0.9));
        assert!((c - Vec3::new(0.4, 0.0, 0.0)).length() < TOL);
        // Transparent src keeps dst.
        let d = composite_over(Vec3::ZERO, 0.0, Vec3::splat(0.7));
        assert!((d - Vec3::splat(0.7)).length() < TOL);
        // Half coverage mixes.
        let src = premultiply(Vec3::ONE, 0.5);
        let e = composite_over(src, 0.5, Vec3::ZERO);
        assert!((e - Vec3::splat(0.5)).length() < TOL);
    }

    #[test]
    fn integrate_premultiplies_and_adds_emissive() {
        let (rgb, a) = integrate(Vec3::ONE, Vec3::splat(0.5), Vec3::new(0.2, 0.0, 0.0), 0.5);
        // lit = 0.5; +emissive(0.2,0,0) -> (0.7,0.5,0.5); *alpha 0.5.
        assert!((a - 0.5).abs() < TOL);
        assert!((rgb - Vec3::new(0.35, 0.25, 0.25)).length() < TOL);
    }

    #[test]
    fn integrate_zero_alpha_is_transparent() {
        let (rgb, a) = integrate(Vec3::ONE, Vec3::ONE, Vec3::ONE, 0.0);
        assert!(rgb.length() < TOL);
        assert!(a.abs() < TOL);
    }

    #[test]
    fn no_nan_on_pathological_inputs() {
        let bad = Vec3::new(f32::NAN, f32::INFINITY, -f32::INFINITY);
        let light = ParticleLight::new(bad, bad);
        assert!(diffuse_radiance(bad, light, f32::NAN, f32::NAN, f32::NAN).is_finite());
        assert!(premultiply(bad, f32::NAN).is_finite());
        assert!(composite_over(bad, f32::NAN, bad).is_finite());
        let (rgb, a) = integrate(bad, bad, bad, f32::NAN);
        assert!(rgb.is_finite() && a.is_finite());
        assert!(self_shadow_transmittance(f32::NAN, f32::NAN).is_finite());
        assert!(half_lambert(f32::NAN).is_finite());
        assert!(wrap_diffuse(f32::NAN, f32::NAN).is_finite());
    }
}
