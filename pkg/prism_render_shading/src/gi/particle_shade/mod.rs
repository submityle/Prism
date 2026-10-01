//! Soft-particle shading CPU golden reference.
//!
//! Backend-neutral, GPU-free numerical reference for shading billboarded
//! particles (smoke, fire, dust, sparks, magic).  A particle is a flat
//! camera-facing quad, so robust shading needs three ingredients that opaque
//! surface shading does not: it must hide the hard seam where the quad clips
//! scene geometry, synthesise a plausible normal from the quad's UVs, and blend
//! translucently with correct premultiplied alpha.  Those three concerns map to
//! the submodules:
//!
//! * [`soft_depth`] — soft-particle depth fade: linearise the hardware depth
//!   buffer, fade opacity as the fragment approaches the scene surface behind
//!   it, and dissolve particles that drift too close to the camera.
//! * [`billboard`] — spherical / cylindrical billboard normal reconstruction
//!   with a circular coverage mask, the camera-facing orthonormal basis, and
//!   the billboard→world normal rotation.
//! * [`lighting`] — wrap / half-Lambert diffuse, Beer-Lambert self-shadowing
//!   (reusing [`crate::gi::volumetric_gi::scattering`]), and
//!   premultiplied-alpha compositing / emissive integration.
//!
//! The high-level [`shade_particle`] stitches these together: reconstruct a
//! world-space normal from the quad UV and view direction, light it, fold in
//! the depth and camera fades plus the sprite coverage into a single alpha, and
//! return premultiplied colour + alpha ready for an `over` composite.
//!
//! # Conventions
//! * `uv` is the quad coordinate in `[0, 1]^2` with the disc centre at
//!   `(0.5, 0.5)`; `view_dir` points **from the particle toward the camera**.
//! * `scene_depth` / `particle_depth` are **linear eye distances** (positive);
//!   raw depth-buffer samples should be linearised with
//!   [`soft_depth::linearize_depth01`] first.
//! * Colours are linear-RGB [`Vec3`]; the returned colour is premultiplied by
//!   the final alpha and both are finite and non-negative.
//! * Every item is a deterministic pure function (no RNG / I/O / GPU / globals
//!   / `unsafe`); defensive clamping guarantees no `NaN` or infinity.
//!
//! # References
//! * T. Lorach, "Soft Particles", NVIDIA, 2007.
//! * T. Umenhoffer et al., "Spherical Billboards", 2006.
//! * NVIDIA GPU Gems 3, Ch. 23, "High-Speed, Off-Screen Particles".

pub mod billboard;
pub mod lighting;
pub mod soft_depth;

pub use billboard::{
    billboard_basis_from_view, billboard_normal_world, camera_facing_basis, cylindrical_normal,
    spherical_normal, spherical_normal_soft, view_normal_to_world, view_normal_to_world_mat,
    BillboardNormal,
};
pub use lighting::{
    composite_over, diffuse_radiance, half_lambert, integrate, lambert, premultiply,
    self_shadow_transmittance, wrap_diffuse, ParticleLight,
};
pub use soft_depth::{
    camera_fade, depth_opacity, linearize_depth01, linearize_depth_ndc, soft_fade,
    soft_fade_linear,
};

use bevy_math::{Vec2, Vec3};

/// Parameters controlling [`shade_particle`], with sensible AAA defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleShadeParams {
    /// Base (albedo) colour of the particle, linear-RGB.
    pub albedo: Vec3,
    /// Self-emission added after lighting, linear-RGB.
    pub emissive: Vec3,
    /// Base coverage opacity (e.g. the sprite texture alpha) in `[0, 1]`.
    pub base_alpha: f32,
    /// Diffuse wrap factor in `[0, 1]`: `0` hard Lambert, `1` fully wrapped.
    pub wrap: f32,
    /// Optical density used for Beer-Lambert self-shadowing (non-negative).
    pub density: f32,
    /// Optical thickness along the light for self-shadowing (non-negative).
    pub thickness: f32,
    /// Soft-particle intersection fade distance in view units (non-negative).
    pub fade_distance: f32,
    /// Camera-fade fully-opaque radius (eye distance), the farther radius.
    pub camera_fade_start: f32,
    /// Camera-fade fully-transparent radius (eye distance), the nearer radius.
    pub camera_fade_end: f32,
}

impl Default for ParticleShadeParams {
    #[inline]
    fn default() -> Self {
        Self {
            albedo: Vec3::ONE,
            emissive: Vec3::ZERO,
            base_alpha: 1.0,
            wrap: 0.5,
            density: 1.0,
            thickness: 1.0,
            fade_distance: 1.0,
            camera_fade_start: 0.5,
            camera_fade_end: 0.1,
        }
    }
}

/// Result of [`shade_particle`]: premultiplied colour and final coverage alpha.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleShadeResult {
    /// Premultiplied linear-RGB colour (already scaled by `alpha`).
    pub color: Vec3,
    /// Final coverage opacity in `[0, 1]` for an `over` composite.
    pub alpha: f32,
}

/// Build a right-handed camera-facing basis `(right, up, forward)` from a view
/// direction alone.
///
/// `forward` is the normalised `view_dir` (particle → camera).  `right` / `up`
/// are chosen via a world-up reference (`+Y`, or `+X` when `forward` is nearly
/// vertical) and orthonormalised through [`camera_facing_basis`].  A degenerate
/// `view_dir` falls back to the world axes.
#[inline]
fn basis_from_view_dir(view_dir: Vec3) -> (Vec3, Vec3, Vec3) {
    let forward = if view_dir.length_squared().is_finite()
        && view_dir.length_squared() > 1.0e-12
    {
        view_dir.normalize()
    } else {
        return (Vec3::X, Vec3::Y, Vec3::Z);
    };
    // Pick a world-up reference not parallel to `forward`.
    let up_ref = if forward.y.abs() < 0.999 { Vec3::Y } else { Vec3::X };
    let right = up_ref.cross(forward);
    // `camera_facing_basis` re-derives `forward = right × up` and
    // orthonormalises, matching the `+Z toward camera` convention.
    camera_facing_basis(right, forward.cross(right))
}

/// High-level soft-particle shade: normal reconstruction + lighting + fades.
///
/// Reconstructs a world-space spherical-billboard normal from `uv` and the
/// camera-facing basis derived from `view_dir`, lights it with `light` using
/// wrapped diffuse and Beer-Lambert self-shadowing, then multiplies the sprite
/// disc coverage, the soft-particle intersection fade (from `particle_depth`
/// vs `scene_depth`) and the near-camera fade into a single alpha.  Returns the
/// premultiplied colour and that alpha.  Fragments outside the sprite disc, or
/// fully faded, come back with `alpha = 0` and black colour.
#[inline]
pub fn shade_particle(
    uv: Vec2,
    view_dir: Vec3,
    light: ParticleLight,
    scene_depth: f32,
    particle_depth: f32,
    params: &ParticleShadeParams,
) -> ParticleShadeResult {
    // 1. Reconstruct the billboard normal + circular coverage.
    let local = spherical_normal(uv);
    let (right, up, forward) = basis_from_view_dir(view_dir);
    let world_normal = view_normal_to_world(local.normal, right, up, forward);

    // 2. Light the reconstructed normal (wrapped diffuse × self-shadow).
    let radiance = diffuse_radiance(
        world_normal,
        light,
        params.wrap,
        params.density,
        params.thickness,
    );

    // 3. Combine coverage sources into one alpha.
    let fade = depth_opacity(
        particle_depth,
        scene_depth,
        params.fade_distance,
        params.camera_fade_start,
        params.camera_fade_end,
    );
    let base = if params.base_alpha.is_finite() {
        params.base_alpha.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let alpha = (local.alpha * fade * base).clamp(0.0, 1.0);

    // 4. Integrate albedo diffuse + emissive, premultiplied by the alpha.
    let (color, alpha) = integrate(params.albedo, radiance, params.emissive, alpha);
    ParticleShadeResult { color, alpha }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-5;

    #[test]
    fn defaults_are_sane() {
        let p = ParticleShadeParams::default();
        assert!((p.albedo - Vec3::ONE).length() < TOL);
        assert_eq!(p.emissive, Vec3::ZERO);
        assert!((0.0..=1.0).contains(&p.wrap));
        assert!(p.camera_fade_start >= p.camera_fade_end);
    }

    #[test]
    fn basis_from_view_dir_is_orthonormal() {
        for dir in [
            Vec3::Z,
            Vec3::new(0.3, 0.2, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ] {
            let (r, u, f) = basis_from_view_dir(dir);
            for v in [r, u, f] {
                assert!((v.length() - 1.0).abs() < TOL, "non-unit for {dir:?}");
            }
            assert!(r.dot(u).abs() < 1.0e-4);
            assert!(r.dot(f).abs() < 1.0e-4);
            assert!(u.dot(f).abs() < 1.0e-4);
        }
    }

    #[test]
    fn facing_light_gives_lit_opaque_fragment() {
        // Centre fragment, camera looking down +Z, light from the camera.
        let params = ParticleShadeParams::default();
        let light = ParticleLight::new(Vec3::Z, Vec3::splat(2.0));
        let r = shade_particle(
            Vec2::new(0.5, 0.5),
            Vec3::Z,
            light,
            100.0, // scene far behind
            10.0,  // particle well in front of camera
            &params,
        );
        assert!(r.alpha > 0.0);
        assert!(r.color.length() > 0.0);
        assert!(r.color.is_finite());
    }

    #[test]
    fn outside_disc_is_discarded() {
        let params = ParticleShadeParams::default();
        let light = ParticleLight::new(Vec3::Z, Vec3::ONE);
        // Corner (0,0) is outside the inscribed disc.
        let r = shade_particle(Vec2::ZERO, Vec3::Z, light, 100.0, 10.0, &params);
        assert!(r.alpha.abs() < TOL);
        assert!(r.color.length() < TOL);
    }

    #[test]
    fn intersection_fades_alpha_to_zero() {
        let params = ParticleShadeParams::default();
        let light = ParticleLight::new(Vec3::Z, Vec3::ONE);
        // Particle sitting exactly on the scene surface -> soft fade 0.
        let r = shade_particle(Vec2::new(0.5, 0.5), Vec3::Z, light, 10.0, 10.0, &params);
        assert!(r.alpha.abs() < TOL);
    }

    #[test]
    fn near_camera_fade_dissolves() {
        let params = ParticleShadeParams::default();
        let light = ParticleLight::new(Vec3::Z, Vec3::ONE);
        // Depth below camera_fade_end -> transparent even if clear of scene.
        let r = shade_particle(
            Vec2::new(0.5, 0.5),
            Vec3::Z,
            light,
            100.0,
            0.05,
            &params,
        );
        assert!(r.alpha.abs() < TOL);
    }

    #[test]
    fn emissive_shows_even_in_shadow() {
        let mut params = ParticleShadeParams::default();
        params.emissive = Vec3::new(1.0, 0.2, 0.0);
        // Light pointing away so diffuse is minimal; emissive still contributes.
        let light = ParticleLight::new(Vec3::new(0.0, 0.0, -1.0), Vec3::ZERO);
        let r = shade_particle(Vec2::new(0.5, 0.5), Vec3::Z, light, 100.0, 10.0, &params);
        assert!(r.color.x > 0.0);
    }

    #[test]
    fn result_is_premultiplied() {
        let mut params = ParticleShadeParams::default();
        params.base_alpha = 0.5;
        let light = ParticleLight::new(Vec3::Z, Vec3::splat(2.0));
        let r = shade_particle(Vec2::new(0.5, 0.5), Vec3::Z, light, 100.0, 10.0, &params);
        // Premultiplied colour never exceeds alpha * lit; sanity: color <= something.
        assert!(r.alpha > 0.0 && r.alpha <= 0.5 + TOL);
        assert!(r.color.is_finite());
    }

    #[test]
    fn no_nan_on_pathological_inputs() {
        let bad = Vec3::new(f32::NAN, f32::INFINITY, -f32::INFINITY);
        let params = ParticleShadeParams {
            albedo: bad,
            emissive: bad,
            base_alpha: f32::NAN,
            wrap: f32::NAN,
            density: f32::NAN,
            thickness: f32::NAN,
            fade_distance: f32::NAN,
            camera_fade_start: f32::NAN,
            camera_fade_end: f32::NAN,
        };
        let light = ParticleLight::new(bad, bad);
        let r = shade_particle(
            Vec2::new(f32::NAN, f32::INFINITY),
            bad,
            light,
            f32::NAN,
            f32::NAN,
            &params,
        );
        assert!(r.color.is_finite());
        assert!(r.alpha.is_finite());
    }
}
