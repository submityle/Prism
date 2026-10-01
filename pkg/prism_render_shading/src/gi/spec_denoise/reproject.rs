//! Specular history reprojection along the *virtual* reflection point — CPU golden.
//!
//! A mirror highlight does not live on the surface: it is the image of a point
//! sitting `hit_distance` *behind* the reflector along the reflected view ray
//! (its "virtual" world position).  Reprojecting a specular history buffer from
//! the surface position — as a diffuse denoiser does — smears glossy
//! reflections into the classic "reflection lag" trail as the camera moves.
//! Production specular denoisers (NVIDIA NRD ReBLUR / ReLAX specular) instead
//! reproject along this virtual point, apply a *parallax* correction derived
//! from the camera's motion relative to that virtual image, and modulate the
//! reprojection trust by surface roughness.
//!
//! This module is the backend-neutral, deterministic reference for that stage:
//!
//! 1. [`virtual_reflection_point`] builds the virtual world position, blending
//!    between pure surface tracking (rough → the lobe is wide, the highlight
//!    tracks the surface) and the full virtual image (smooth → mirror parallax).
//! 2. [`view_parallax`] measures how far the camera swept around that point
//!    between frames (the tangent of the subtended angle).
//! 3. [`reprojection_confidence`] turns roughness + parallax into a `[0, 1]`
//!    trust weight: smooth surfaces lose confidence rapidly under parallax
//!    (their virtual image races across the screen), rough ones barely care.
//! 4. [`blend_reprojected_position`] mixes the surface-motion reprojection with
//!    the virtual-motion reprojection by the same roughness schedule.
//!
//! # Conventions
//! * World space is right-handed `f32` to match the WESL/GPU twin bit-for-bit.
//!   `view_dir` always points *from the surface toward the camera* and is
//!   normalised defensively on entry.
//! * `roughness ∈ [0, 1]` is perceptual (Disney); callers may map it to GGX
//!   `alpha` via [`crate::gi::spec_gi::ggx_lobe::roughness_to_alpha`].
//! * Transcendentals go through [`bevy_math::ops`] (never `f32::exp`), per the
//!   crate-wide no-`std` numerical contract; `sqrt` uses inherent `f32::sqrt`.
//! * Every helper is a deterministic pure function (no RNG / IO / GPU / globals
//!   / `unsafe`).  Degenerate input (zero-length vectors, non-finite depths,
//!   negative distances) falls back to a safe identity rather than emitting
//!   `NaN`; every result is finite.

use bevy_math::{ops, Vec3};

/// Tunables for the specular reprojection trust model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReprojectParams {
    /// How sharply reprojection confidence collapses under parallax for a
    /// *mirror* (`roughness = 0`).  Larger → smooth reflections reject history
    /// more eagerly when the camera moves.
    pub parallax_sensitivity: f32,
    /// Exponent of the `(1 - roughness)` virtual-history schedule.  Larger →
    /// only very smooth surfaces follow the virtual image; everything else
    /// tracks the surface.
    pub virtual_exponent: f32,
    /// Floor on confidence so a converged rough surface never fully discards
    /// its (cheap-to-trust) history.
    pub min_confidence: f32,
}

impl Default for ReprojectParams {
    fn default() -> Self {
        Self {
            parallax_sensitivity: 8.0,
            virtual_exponent: 2.0,
            min_confidence: 0.0,
        }
    }
}

/// Reflect incident direction `i` about unit normal `n`: `i - 2 (i·n) n`.
///
/// `i` is the direction of travel of the incident ray; for a view ray arriving
/// at the surface pass `-view_dir`.
#[must_use]
pub fn reflect(i: Vec3, n: Vec3) -> Vec3 {
    let nn = safe_normalize(n);
    i - nn * (2.0 * i.dot(nn))
}

/// Fraction of the full virtual (hit-distance) parallax a reflection exhibits
/// at a given roughness, in `[0, 1]`.
///
/// Returns `1` for a perfect mirror (`roughness = 0`, the highlight is the
/// virtual image and moves with full parallax) and falls monotonically toward
/// `0` as roughness grows (the lobe widens until the highlight effectively
/// sticks to the surface).  The schedule is `(1 - roughness)^virtual_exponent`.
#[must_use]
pub fn virtual_history_amount(roughness: f32, params: &ReprojectParams) -> f32 {
    let r = roughness.clamp(0.0, 1.0);
    let e = params.virtual_exponent.max(0.0);
    ops::powf((1.0 - r).max(0.0), e).clamp(0.0, 1.0)
}

/// NRD/Frostbite specular *dominant* factor in `[0, 1]`: how far the dominant
/// specular direction leans from the normal toward the mirror reflection.
///
/// At `roughness = 0` the dominant direction is the mirror reflection (`≈ 1`);
/// as roughness grows it retreats toward the normal.  Grazing angles
/// (`n_dot_v → 0`) push it back toward the reflection.  Used to orient the
/// virtual point and to weight parallax.
#[must_use]
pub fn specular_dominant_factor(n_dot_v: f32, roughness: f32) -> f32 {
    let ndv = n_dot_v.clamp(0.0, 1.0);
    let r = roughness.clamp(0.0, 1.0);
    // Lazarov/NRD closed form; the `ln` argument stays in [0.408, 39.41] for
    // r ∈ [0, 1] so it is always strictly positive.
    let a = 0.298_475 * ops::ln(39.411_5 - 39.002_9 * r);
    let f = ops::powf((1.0 - ndv).max(0.0), 10.864_9) * (1.0 - a) + a;
    f.clamp(0.0, 1.0)
}

/// World-space virtual reflection point used to reproject specular history.
///
/// The mirror image of the reflected scene sits `hit_distance` along the
/// reflected view ray.  For glossy surfaces the effective parallax shrinks, so
/// the virtual distance is scaled by [`virtual_history_amount`]: at
/// `roughness = 0` this returns `surface_pos + R · hit_distance` (full virtual
/// image); at `roughness = 1` it collapses onto `surface_pos` (pure surface
/// tracking).  `view_dir` points from the surface toward the camera.
#[must_use]
pub fn virtual_reflection_point(
    surface_pos: Vec3,
    view_dir: Vec3,
    normal: Vec3,
    hit_distance: f32,
    roughness: f32,
    params: &ReprojectParams,
) -> Vec3 {
    let v = safe_normalize(view_dir);
    let n = safe_normalize(normal);
    let r = reflect(-v, n);
    let dist = sanitize_scalar(hit_distance).max(0.0);
    let amount = virtual_history_amount(roughness, params);
    sanitize_vec(surface_pos) + r * (dist * amount)
}

/// Tangent of the angle subtended at `point` by the camera's inter-frame
/// motion — the ReBLUR specular parallax measure (always `≥ 0`).
///
/// `0` when the camera did not move (or moved straight along the view ray);
/// grows as the camera orbits the point.  Returns `0` for degenerate geometry
/// (camera coincident with the point) so downstream weights stay finite.
#[must_use]
pub fn view_parallax(prev_cam_pos: Vec3, curr_cam_pos: Vec3, point: Vec3) -> f32 {
    let p = sanitize_vec(point);
    let a = sanitize_vec(curr_cam_pos) - p;
    let b = sanitize_vec(prev_cam_pos) - p;
    let la = a.length();
    let lb = b.length();
    if la <= 1.0e-12 || lb <= 1.0e-12 {
        return 0.0;
    }
    let cos = (a.dot(b) / (la * lb)).clamp(-1.0, 1.0);
    let sin = (1.0 - cos * cos).max(0.0).sqrt();
    // tan(theta) = sin/cos; clamp cos away from zero and saturate at a large
    // but finite value so a 180° sweep does not explode to +inf.
    let tan = sin / cos.abs().max(1.0e-4);
    if tan.is_finite() {
        tan.min(1.0e4)
    } else {
        1.0e4
    }
}

/// Reprojection confidence in `[min_confidence, 1]` for the virtual history.
///
/// Combines roughness and [`view_parallax`]: a mirror (`roughness = 0`) decays
/// as `exp(-parallax · parallax_sensitivity)`, while rough surfaces — whose
/// virtual image barely moves — keep almost full trust.  The effective
/// sensitivity is scaled by [`virtual_history_amount`] so the roughness
/// schedule is shared with the position blend.
#[must_use]
pub fn reprojection_confidence(roughness: f32, parallax: f32, params: &ReprojectParams) -> f32 {
    let px = sanitize_scalar(parallax).max(0.0);
    let amount = virtual_history_amount(roughness, params);
    let sensitivity = params.parallax_sensitivity.max(0.0) * amount;
    let conf = stable_exp(-sensitivity * px);
    let floor = params.min_confidence.clamp(0.0, 1.0);
    (conf.max(floor)).clamp(0.0, 1.0)
}

/// Blend the surface-motion reprojection with the virtual-motion reprojection.
///
/// `surface_reproj` is where the *surface* point reprojects to under the
/// previous view (good for rough, lobe-dominated reflections); `virtual_reproj`
/// is where the *virtual image* reprojects (good for smooth mirrors).  The mix
/// is [`virtual_history_amount`], so smooth surfaces follow the virtual image
/// and rough ones follow the surface.  Returns a finite world position.
#[must_use]
pub fn blend_reprojected_position(
    surface_reproj: Vec3,
    virtual_reproj: Vec3,
    roughness: f32,
    params: &ReprojectParams,
) -> Vec3 {
    let amount = virtual_history_amount(roughness, params);
    let s = sanitize_vec(surface_reproj);
    let v = sanitize_vec(virtual_reproj);
    sanitize_vec(s.lerp(v, amount))
}

/// Everything the reprojection stage produces for one pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reprojection {
    /// World position to sample the specular history buffer at.
    pub sample_position: Vec3,
    /// The virtual reflection point used to measure parallax.
    pub virtual_point: Vec3,
    /// Tangent-of-angle parallax at the virtual point.
    pub parallax: f32,
    /// Trust weight for the reprojected history in `[0, 1]`.
    pub confidence: f32,
}

/// Full specular reprojection for one pixel.
///
/// Builds the virtual reflection point, measures camera parallax against it,
/// derives the confidence, and blends the surface/virtual reprojected sample
/// position.  `surface_reproj` is the (externally computed) reprojected surface
/// position; the virtual reprojected position is approximated by the virtual
/// point itself (its world location is view-independent).  All outputs are
/// finite.
#[must_use]
pub fn reproject_specular(
    surface_pos: Vec3,
    surface_reproj: Vec3,
    view_dir: Vec3,
    normal: Vec3,
    hit_distance: f32,
    roughness: f32,
    prev_cam_pos: Vec3,
    curr_cam_pos: Vec3,
    params: &ReprojectParams,
) -> Reprojection {
    let virtual_point =
        virtual_reflection_point(surface_pos, view_dir, normal, hit_distance, roughness, params);
    let parallax = view_parallax(prev_cam_pos, curr_cam_pos, virtual_point);
    let confidence = reprojection_confidence(roughness, parallax, params);
    let sample_position =
        blend_reprojected_position(surface_reproj, virtual_point, roughness, params);
    Reprojection {
        sample_position,
        virtual_point,
        parallax,
        confidence,
    }
}

// ---------------------------------------------------------------------------
// Defensive numeric helpers (private).
// ---------------------------------------------------------------------------

/// Numerically safe `exp` on `(-inf, 0]`; never returns `NaN`/`+inf`.
#[must_use]
fn stable_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    ops::exp(x.clamp(-80.0, 0.0))
}

/// Normalise `v`, falling back to `+Z` for zero-length / non-finite input.
#[must_use]
fn safe_normalize(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > 1.0e-24 {
        v / len_sq.sqrt()
    } else {
        Vec3::Z
    }
}

/// Replace any non-finite component of a position with `0`.
#[must_use]
fn sanitize_vec(v: Vec3) -> Vec3 {
    Vec3::new(finite_or_zero(v.x), finite_or_zero(v.y), finite_or_zero(v.z))
}

/// Replace a non-finite scalar with `0`.
#[must_use]
fn sanitize_scalar(x: f32) -> f32 {
    finite_or_zero(x)
}

#[must_use]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    fn params() -> ReprojectParams {
        ReprojectParams::default()
    }

    #[test]
    fn reflect_mirrors_about_normal() {
        // Straight-down incident ray reflects straight up about an up normal.
        let r = reflect(Vec3::NEG_Z, Vec3::Z);
        assert!((r - Vec3::Z).length() < EPS);
    }

    #[test]
    fn mirror_virtual_point_marches_full_hit_distance() {
        // Camera above a floor looking down; view points up, normal up.
        let surface = Vec3::ZERO;
        let view = Vec3::Z;
        let n = Vec3::Z;
        let vp = virtual_reflection_point(surface, view, n, 3.0, 0.0, &params());
        // Mirror reflection of a straight-down view about up is straight up.
        assert!((vp - Vec3::new(0.0, 0.0, 3.0)).length() < EPS);
    }

    #[test]
    fn rough_virtual_point_collapses_to_surface() {
        let surface = Vec3::new(1.0, 2.0, 3.0);
        let vp = virtual_reflection_point(surface, Vec3::Z, Vec3::Z, 5.0, 1.0, &params());
        assert!((vp - surface).length() < EPS);
    }

    #[test]
    fn virtual_history_amount_is_monotone_decreasing() {
        let p = params();
        let a0 = virtual_history_amount(0.0, &p);
        let a3 = virtual_history_amount(0.3, &p);
        let a7 = virtual_history_amount(0.7, &p);
        let a1 = virtual_history_amount(1.0, &p);
        assert!((a0 - 1.0).abs() < EPS);
        assert!(a1.abs() < EPS);
        assert!(a0 > a3 && a3 > a7 && a7 > a1);
    }

    #[test]
    fn dominant_factor_in_unit_range_and_rough_leans_to_normal() {
        // Smoother surface => dominant direction closer to mirror reflection
        // (larger factor) at a fixed, non-grazing view.
        let smooth = specular_dominant_factor(0.9, 0.02);
        let rough = specular_dominant_factor(0.9, 0.9);
        assert!((0.0..=1.0).contains(&smooth));
        assert!((0.0..=1.0).contains(&rough));
        assert!(smooth > rough);
    }

    #[test]
    fn parallax_zero_when_camera_static() {
        let cam = Vec3::new(0.0, 0.0, 5.0);
        let px = view_parallax(cam, cam, Vec3::ZERO);
        assert!(px.abs() < EPS);
    }

    #[test]
    fn parallax_grows_with_camera_sweep() {
        let point = Vec3::ZERO;
        let prev = Vec3::new(0.0, 0.0, 5.0);
        let small = view_parallax(prev, Vec3::new(0.5, 0.0, 5.0), point);
        let large = view_parallax(prev, Vec3::new(3.0, 0.0, 5.0), point);
        assert!(large > small);
        assert!(small > 0.0);
    }

    #[test]
    fn confidence_decreases_with_parallax_for_mirror() {
        let p = params();
        let c0 = reprojection_confidence(0.0, 0.0, &p);
        let c1 = reprojection_confidence(0.0, 0.1, &p);
        let c2 = reprojection_confidence(0.0, 0.5, &p);
        assert!((c0 - 1.0).abs() < EPS);
        assert!(c0 > c1 && c1 > c2);
        assert!((0.0..=1.0).contains(&c2));
    }

    #[test]
    fn confidence_higher_for_rough_at_equal_parallax() {
        let p = params();
        let mirror = reprojection_confidence(0.0, 0.5, &p);
        let rough = reprojection_confidence(0.9, 0.5, &p);
        assert!(rough > mirror);
    }

    #[test]
    fn blend_follows_roughness_schedule() {
        let p = params();
        let surf = Vec3::ZERO;
        let virt = Vec3::new(10.0, 0.0, 0.0);
        // Mirror -> virtual, rough -> surface.
        let mirror = blend_reprojected_position(surf, virt, 0.0, &p);
        let rough = blend_reprojected_position(surf, virt, 1.0, &p);
        assert!((mirror - virt).length() < EPS);
        assert!((rough - surf).length() < EPS);
    }

    #[test]
    fn reproject_specular_is_finite_on_degenerate_inputs() {
        let p = params();
        let r = reproject_specular(
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::INFINITY),
            Vec3::ZERO,
            Vec3::ZERO,
            -5.0,
            2.0,
            Vec3::splat(f32::NAN),
            Vec3::ZERO,
            &p,
        );
        assert!(r.sample_position.is_finite());
        assert!(r.virtual_point.is_finite());
        assert!(r.parallax.is_finite());
        assert!(r.confidence.is_finite());
        assert!((0.0..=1.0).contains(&r.confidence));
    }

    #[test]
    fn determinism() {
        let p = params();
        let a = reproject_specular(
            Vec3::ZERO,
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
            2.0,
            0.3,
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.3, 0.0, 5.0),
            &p,
        );
        let b = reproject_specular(
            Vec3::ZERO,
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
            2.0,
            0.3,
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.3, 0.0, 5.0),
            &p,
        );
        assert_eq!(a, b);
    }
}
