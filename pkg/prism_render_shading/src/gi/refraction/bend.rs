//! Snell view-ray refraction and thickness-scaled screen-space UV bending.
//!
//! This module is the CPU golden reference for the *geometric* half of
//! screen-space refraction: it bends the camera's view ray through a dielectric
//! interface and turns the refracted direction into a displacement of the
//! background buffer's sample coordinate (UV).  A transmissive surface (glass,
//! a gem, a water volume) does not emit its own colour; it re-samples whatever
//! was already rendered *behind* it, shifted sideways by the refraction, so the
//! background appears to warp.  The companion modules handle the two remaining
//! halves: [`crate::gi::refraction::absorption`] attenuates the transmitted
//! radiance along the path, and [`crate::gi::refraction::rough`] blurs and
//! disperses the sample for frosted / chromatic glass.
//!
//! The job here is deliberately narrow and disjoint from the other refraction
//! references in the engine:
//!
//! * It is *not* the physical cornea Snell solve of [`crate::gi::eye`]; that
//!   module traces a curved biological interface.
//! * It is *not* the wet-surface reflection/refraction *blend* of
//!   [`crate::gi::material`]; that module mixes a reflected and a refracted
//!   lobe for a thin water film.
//!
//! Here we assume a flat-slab approximation of a thick transmissive body of a
//! given `thickness`: the refracted ray is projected onto the screen plane and
//! the background UV is pushed in that direction by an artist-and-physics
//! driven strength that grows with both the slab `thickness` and the magnitude
//! of the index-of-refraction contrast across the interface.
//!
//! # Conventions
//! * Direction vectors use the common real-time convention: `view` is the
//!   incident view ray that points *from the eye into the surface*, and
//!   `normal` points *out of* the surface toward the eye (the incident medium).
//!   Refracted directions are returned as unit vectors in the same view space,
//!   whose `+z` axis points toward the camera and whose `xy` plane is parallel
//!   to the screen.
//! * `eta` is the relative index `n_incident / n_transmitted`.  Total internal
//!   reflection (TIR), only possible when `n_incident > n_transmitted`, is
//!   reported as `None` from [`refract`].
//! * UV coordinates live in `[0, 1]^2`; every UV produced here is clamped into
//!   that range so a caller can sample the background buffer unconditionally.
//! * `no_std`: math via `bevy_math`; square roots via the `f32::sqrt` method.
//!   This module needs no transcendental functions and allocates nothing.
//! * All inputs are defensively sanitised — indices of refraction are clamped
//!   to the physical dielectric range `[1, inf)`, thickness and projection
//!   scale to `>= 0`, degenerate directions fall back to canonical axes — so
//!   every result is finite: no `NaN`, no infinity, no division by zero.
//! * Every function is a deterministic, allocation-free pure function: no RNG,
//!   no I/O, no GPU, no global state.

use bevy_math::{Vec2, Vec3};

/// Index of refraction of air / vacuum, the usual incident medium.
pub const AIR_IOR: f32 = 1.0;

/// Smallest index of refraction accepted; physical dielectrics have `n >= 1`.
const MIN_IOR: f32 = 1.0;

/// Floor applied to the view-space depth of the refracted ray when deriving a
/// path length, so a grazing (`z -> 0`) ray cannot produce an unbounded shift.
const MIN_AXIAL: f32 = 1.0e-4;

/// Clamps an index of refraction to the physical dielectric range `[1, inf)`.
///
/// Non-finite inputs collapse to [`MIN_IOR`].
#[inline]
fn clamp_ior(n: f32) -> f32 {
    if n.is_finite() {
        n.max(MIN_IOR)
    } else {
        MIN_IOR
    }
}

/// Sanitises a magnitude that must be non-negative (thickness, scale factors).
///
/// Non-finite inputs collapse to `0`.
#[inline]
fn sanitize_nonneg(x: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

/// Clamps a single UV scalar to `[0, 1]`; non-finite inputs collapse to `0`.
#[inline]
fn sanitize_unit(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Normalises `v`, returning `fallback` for a degenerate (near-zero) input.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        fallback
    }
}

/// Clamps a UV coordinate into the sampleable range `[0, 1]^2`.
///
/// Both components are sanitised independently; non-finite components become
/// `0`.  Callers can therefore feed the result straight into a background
/// buffer fetch without a bounds check.
#[inline]
pub fn clamp_uv(uv: Vec2) -> Vec2 {
    Vec2::new(sanitize_unit(uv.x), sanitize_unit(uv.y))
}

/// Snell refraction of a view ray through a dielectric interface.
///
/// `view` is the incident ray pointing *into* the surface, `normal` points
/// *out* toward the incident medium, and `eta = n_incident / n_transmitted` is
/// the relative index.  Returns the unit refracted direction, or `None` on
/// total internal reflection — when the radicand
/// `k = 1 - eta^2 * (1 - cos_i^2)` is negative, which can only happen going
/// from a denser into a rarer medium (`eta > 1`).
///
/// Uses the standard closed form `t = eta*I + (eta*cos_i - sqrt(k))*N` with
/// `cos_i = -(N·I)`.  Inputs are normalised defensively: a degenerate `view`
/// falls back to `-Z` (straight into the screen) and a degenerate `normal` to
/// `+Z` (facing the camera), and `eta` is clamped to `>= 0`.
#[inline]
pub fn refract(view: Vec3, normal: Vec3, eta: f32) -> Option<Vec3> {
    let i = normalize_or(view, Vec3::NEG_Z);
    let n = normalize_or(normal, Vec3::Z);
    let eta = if eta.is_finite() { eta.max(0.0) } else { 1.0 };

    let cos_i = -n.dot(i);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        None
    } else {
        let t = eta * i + (eta * cos_i - k.sqrt()) * n;
        Some(normalize_or(t, i))
    }
}

/// Screen-plane (tangential) direction of a refracted ray.
///
/// Returns the unit-length `xy` component of `refracted`, i.e. the direction in
/// which the background UV should be pushed.  A ray travelling straight toward
/// the camera (`xy ~= 0`, no lateral bending) yields [`Vec2::ZERO`], meaning no
/// displacement — the correct degenerate behaviour.
#[inline]
pub fn refracted_tangent(refracted: Vec3) -> Vec2 {
    let xy = Vec2::new(refracted.x, refracted.y);
    let len_sq = xy.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        xy * len_sq.sqrt().recip()
    } else {
        Vec2::ZERO
    }
}

/// Heuristic magnitude of the background-UV displacement.
///
/// Models the flat-slab intuition that a thicker body, and a sharper
/// index-of-refraction contrast across its surface, both warp the background
/// more.  Returns `thickness * |n_out - n_in|`, with the indices clamped to the
/// physical dielectric range and `thickness` to `>= 0`.  When the two media
/// match (`n_in == n_out`) the strength is `0`: there is no interface to
/// refract through, so the background is sampled straight.
#[inline]
pub fn displacement_strength(thickness: f32, ior_in: f32, ior_out: f32) -> f32 {
    let t = sanitize_nonneg(thickness);
    let delta = (clamp_ior(ior_out) - clamp_ior(ior_in)).abs();
    t * delta
}

/// Approximate optical path length through a slab of a given `thickness`.
///
/// The refracted ray crosses a slab whose faces are separated by `thickness`
/// along the view `z` axis; the travelled distance is `thickness / cos(theta)`
/// where `theta` is the refraction angle, i.e. `thickness / |refracted.z|`.
/// The axial term is floored by [`MIN_AXIAL`] so a grazing ray yields a large
/// but finite length rather than infinity.  This is the distance the
/// absorption reference integrates Beer-Lambert attenuation over.
#[inline]
pub fn slab_path_length(refracted: Vec3, thickness: f32) -> f32 {
    let t = sanitize_nonneg(thickness);
    let axial = refracted.z.abs().max(MIN_AXIAL);
    t / axial
}

/// A background-buffer bend: the shifted sample UV plus refraction metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackgroundBend {
    /// Clamped `[0, 1]^2` UV at which to sample the background buffer.
    pub uv: Vec2,
    /// Unit refracted direction, or `None` on total internal reflection.
    pub refracted: Option<Vec3>,
    /// `true` when the interface totally internally reflected the view ray; in
    /// that case `uv` falls back to `base_uv` (there is no transmitted sample).
    pub total_internal_reflection: bool,
}

/// Bends the background sample UV for a transmissive surface fragment.
///
/// Given the fragment's own screen UV (`base_uv`), the incident `view` ray and
/// surface `normal`, the indices on either side of the interface, the body
/// `thickness`, and a `projection_scale` that converts the view-space
/// tangential shift into UV units (bundling focal length and aspect), this
/// returns where to sample the already-rendered background.
///
/// The displacement direction is the screen-plane projection of the refracted
/// ray ([`refracted_tangent`]); its magnitude is [`displacement_strength`]
/// scaled by `projection_scale`.  On total internal reflection there is no
/// transmitted ray, so the UV falls back to `base_uv` and the flag is set.  The
/// returned UV is always clamped into `[0, 1]^2`.
#[inline]
pub fn bend_background_uv(
    base_uv: Vec2,
    view: Vec3,
    normal: Vec3,
    ior_in: f32,
    ior_out: f32,
    thickness: f32,
    projection_scale: f32,
) -> BackgroundBend {
    let eta = clamp_ior(ior_in) / clamp_ior(ior_out);
    match refract(view, normal, eta) {
        None => BackgroundBend {
            uv: clamp_uv(base_uv),
            refracted: None,
            total_internal_reflection: true,
        },
        Some(refracted) => {
            let dir = refracted_tangent(refracted);
            let strength = displacement_strength(thickness, ior_in, ior_out)
                * sanitize_nonneg(projection_scale);
            let uv = clamp_uv(base_uv + dir * strength);
            BackgroundBend {
                uv,
                refracted: Some(refracted),
                total_internal_reflection: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-4;

    fn assert_finite_vec3(v: Vec3) {
        assert!(v.x.is_finite() && v.y.is_finite() && v.z.is_finite(), "{v:?}");
    }

    fn assert_finite_vec2(v: Vec2) {
        assert!(v.x.is_finite() && v.y.is_finite(), "{v:?}");
    }

    #[test]
    fn refracted_direction_is_unit_length() {
        // Oblique ray into a denser medium (air -> glass, eta < 1).
        let view = Vec3::new(0.3, -0.2, -1.0).normalize();
        let normal = Vec3::Z;
        let r = refract(view, normal, AIR_IOR / 1.5).expect("no TIR into denser medium");
        assert_finite_vec3(r);
        assert!((r.length() - 1.0).abs() < EPS, "len = {}", r.length());
    }

    #[test]
    fn entering_denser_medium_never_tirs() {
        // Going rare -> dense (eta < 1) can never total-internal-reflect, even
        // at near-grazing incidence.
        let view = Vec3::new(0.999, 0.0, -0.045).normalize();
        let normal = Vec3::Z;
        assert!(refract(view, normal, AIR_IOR / 1.8).is_some());
    }

    #[test]
    fn dense_to_rare_tirs_past_critical_angle() {
        // Glass -> air, eta = 1.5. Critical angle ~41.8 deg; a 60-deg ray TIRs.
        let eta = 1.5f32;
        let theta = 60.0f32.to_radians();
        let view = Vec3::new(theta.sin(), 0.0, -theta.cos());
        let normal = Vec3::Z;
        assert_eq!(refract(view, normal, eta), None);
    }

    #[test]
    fn normal_incidence_passes_straight_through() {
        let r = refract(Vec3::NEG_Z, Vec3::Z, AIR_IOR / 1.5).unwrap();
        // Straight-on refraction does not bend: direction stays along -Z.
        assert!((r - Vec3::NEG_Z).length() < EPS, "{r:?}");
        // ...and therefore produces no tangential displacement.
        assert!(refracted_tangent(r).length() < EPS);
    }

    #[test]
    fn degenerate_inputs_stay_finite() {
        let r = refract(Vec3::ZERO, Vec3::ZERO, f32::NAN).unwrap();
        assert_finite_vec3(r);
        assert!((r.length() - 1.0).abs() < EPS);
    }

    #[test]
    fn tir_falls_back_to_base_uv() {
        let base = Vec2::new(0.5, 0.5);
        let theta = 70.0f32.to_radians();
        let view = Vec3::new(theta.sin(), 0.0, -theta.cos());
        let bend = bend_background_uv(base, view, Vec3::Z, 1.5, AIR_IOR, 2.0, 0.5);
        assert!(bend.total_internal_reflection);
        assert_eq!(bend.refracted, None);
        assert_eq!(bend.uv, base);
    }

    #[test]
    fn uv_is_clamped_into_unit_square() {
        // A huge strength would push the UV far outside [0, 1]; it must clamp.
        let base = Vec2::new(0.5, 0.5);
        let view = Vec3::new(0.6, 0.0, -0.8).normalize();
        let bend = bend_background_uv(base, view, Vec3::Z, AIR_IOR, 1.5, 1.0e3, 1.0e3);
        assert_finite_vec2(bend.uv);
        assert!(bend.uv.x >= 0.0 && bend.uv.x <= 1.0);
        assert!(bend.uv.y >= 0.0 && bend.uv.y <= 1.0);
    }

    #[test]
    fn displacement_grows_with_thickness() {
        let thin = displacement_strength(0.5, AIR_IOR, 1.5);
        let thick = displacement_strength(2.0, AIR_IOR, 1.5);
        assert!(thick > thin);
        assert!(thin >= 0.0);
    }

    #[test]
    fn displacement_grows_with_ior_contrast() {
        let soft = displacement_strength(1.0, AIR_IOR, 1.2);
        let hard = displacement_strength(1.0, AIR_IOR, 1.9);
        assert!(hard > soft);
    }

    #[test]
    fn matched_media_do_not_displace() {
        assert_eq!(displacement_strength(5.0, 1.5, 1.5), 0.0);
        let base = Vec2::new(0.4, 0.6);
        let view = Vec3::new(0.3, 0.1, -1.0).normalize();
        let bend = bend_background_uv(base, view, Vec3::Z, 1.5, 1.5, 5.0, 1.0);
        assert_eq!(bend.uv, base);
    }

    #[test]
    fn zero_thickness_leaves_uv_unchanged() {
        let base = Vec2::new(0.25, 0.75);
        let view = Vec3::new(0.4, -0.2, -1.0).normalize();
        let bend = bend_background_uv(base, view, Vec3::Z, AIR_IOR, 1.5, 0.0, 2.0);
        assert_eq!(bend.uv, base);
        assert!(!bend.total_internal_reflection);
    }

    #[test]
    fn tangent_points_along_screen_plane_bend() {
        // A ray bent toward +x should push the UV toward +x.
        let base = Vec2::new(0.5, 0.5);
        let view = Vec3::new(0.5, 0.0, -0.866).normalize();
        let bend = bend_background_uv(base, view, Vec3::Z, AIR_IOR, 1.5, 0.5, 1.0);
        assert!(bend.uv.x > base.x);
        assert!((bend.uv.y - base.y).abs() < EPS);
    }

    #[test]
    fn slab_path_length_is_finite_at_grazing() {
        // A nearly tangential refracted ray (z -> 0) must not blow up.
        let grazing = Vec3::new(1.0, 0.0, 0.0);
        let len = slab_path_length(grazing, 1.0);
        assert!(len.is_finite());
        assert!(len > 0.0);
    }

    #[test]
    fn slab_path_length_grows_with_thickness() {
        let r = Vec3::new(0.3, 0.0, -0.954).normalize();
        assert!(slab_path_length(r, 2.0) > slab_path_length(r, 1.0));
    }
}
