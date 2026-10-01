//! Contact shadows — screen-space & SDF ray-marched occlusion (CPU golden).
//!
//! Shadow maps and ray-traced area shadows capture coarse occlusion well but
//! lose the thin, high-frequency *contact* darkening where objects touch — the
//! grounding cue that stops geometry from looking like it floats.  Real-time
//! engines recover it with a short ray march toward the light, testing a cheap
//! proxy of the scene for an occluder a few metres away.  This module is the
//! backend-neutral reference for the two standard proxies:
//!
//! * [`screen_space_contact`] marches the ray in screen space against a linear
//!   depth buffer sampled through a caller-supplied callback.  A hit is a texel
//!   whose stored depth lies between the ray's current depth plus a self-shadow
//!   bias and that depth plus a finite `thickness` (so the ray cannot be
//!   occluded by a surface it has already passed behind).  The result fades near
//!   the screen edge and near the march's end for temporal stability.
//! * [`sdf_soft_contact`] marches the ray against an analytic signed-distance
//!   occluder and accumulates Quilez's penumbra estimate
//!   `vis = min(vis, hardness * d / t)`, giving a cheap soft contact shadow with
//!   distance-correct hardening.
//!
//! Both return **occlusion** in `[0, 1]` (`0` lit, `1` fully shadowed), so they
//! compose multiplicatively with the main shadow term as `1 - occlusion`.
//!
//! # Conventions
//! * Depths are *linear* view-space distances that increase away from the eye,
//!   in the same units the GPU depth-linearisation twin produces.
//! * Screen coordinates are UV in `[0, 1]^2`; `(0,0)` is one corner, `(1,1)` the
//!   opposite, matching the sampler the GPU uses.
//! * The self-shadow bias and thickness are expressed in the same linear-depth
//!   units; [`clamp_self_shadow_bias`] keeps the bias strictly below the
//!   thickness so the acceptance window never collapses or inverts.
//! * Every function is a deterministic pure function — the only external input
//!   is the user's sampling callback — and defends against zero steps,
//!   non-finite depths and degenerate windows, never returning `NaN`.

use bevy_math::{ops, Vec2};

/// Clamps a self-shadow depth bias into a safe sub-thickness range.
///
/// The acceptance window for a contact hit is `(depth + bias, depth + thickness)`.
/// For that window to be non-empty the bias must stay below the thickness; this
/// forces `bias` into `[0, 0.9 * thickness]` (and treats a non-finite or
/// non-positive thickness as a tiny positive floor).  Returning the clamped bias
/// keeps callers from silently producing an empty or inverted window.
#[inline]
pub fn clamp_self_shadow_bias(bias: f32, thickness: f32) -> f32 {
    let thickness = if thickness.is_finite() {
        thickness.max(1.0e-6)
    } else {
        1.0e-6
    };
    let bias = if bias.is_finite() { bias.max(0.0) } else { 0.0 };
    bias.min(0.9 * thickness)
}

/// Smooth edge-fade so a march running off-screen does not pop.
///
/// Returns `1.0` well inside the frame and ramps to `0.0` within `margin` of any
/// UV border using a smoothstep.  A non-positive or non-finite margin disables
/// the fade (returns `1.0`).
#[inline]
pub fn screen_edge_fade(uv: Vec2, margin: f32) -> f32 {
    if !(margin > 0.0) || !margin.is_finite() {
        return 1.0;
    }
    let edge = uv.x.min(uv.y).min(1.0 - uv.x).min(1.0 - uv.y);
    smoothstep01(edge / margin)
}

/// Classic Hermite `smoothstep` of a value already normalised to `[0, 1]`.
#[inline]
fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Immutable parameters shared by the contact-shadow marchers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactShadowParams {
    /// Number of ray-march steps, clamped to `>= 1`.
    pub steps: u32,
    /// Self-shadow bias in linear-depth units (kept below `thickness`).
    pub bias: f32,
    /// Occluder thickness in linear-depth units; surfaces deeper than
    /// `depth + thickness` are treated as *behind* the ray and ignored.
    pub thickness: f32,
    /// Smoothstep edge-fade margin in UV units (`0` disables the fade).
    pub edge_fade: f32,
}

impl ContactShadowParams {
    /// Returns a copy with every field clamped into a valid range.
    #[inline]
    pub fn sanitized(self) -> Self {
        Self {
            steps: self.steps.max(1),
            bias: clamp_self_shadow_bias(self.bias, self.thickness),
            thickness: if self.thickness.is_finite() {
                self.thickness.max(1.0e-6)
            } else {
                1.0e-6
            },
            edge_fade: if self.edge_fade.is_finite() {
                self.edge_fade.max(0.0)
            } else {
                0.0
            },
        }
    }
}

impl Default for ContactShadowParams {
    #[inline]
    fn default() -> Self {
        Self {
            steps: 16,
            bias: 0.02,
            thickness: 0.5,
            edge_fade: 0.1,
        }
    }
}

/// Marches a screen-space contact shadow against a linear depth buffer.
///
/// The ray starts at `start_uv`/`start_depth` (the shading point) and advances
/// to `end_uv`/`end_depth` (a point a short world-space distance toward the
/// light, pre-projected by the caller).  At each of `params.steps` interior
/// samples the ray's interpolated UV and depth are compared against the scene
/// depth returned by `scene_depth`:
///
/// * `diff = ray_depth - scene_depth` is how far the ray is *behind* the stored
///   surface.
/// * A hit requires `bias < diff < thickness`: the surface must be in front of
///   the ray by more than the bias (rejecting self-shadowing) yet thin enough
///   that the ray has not already emerged on the far side.
///
/// On the first hit the function returns `fade`, the product of the screen-edge
/// fade at the hit UV and a smooth ramp down over the final quarter of the march
/// (so distant, grazing hits fade out).  With no hit it returns `0.0` (lit).
/// Samples whose UV leaves `[0, 1]^2` or whose depth is non-finite are skipped.
pub fn screen_space_contact<F>(
    params: ContactShadowParams,
    start_uv: Vec2,
    start_depth: f32,
    end_uv: Vec2,
    end_depth: f32,
    scene_depth: F,
) -> f32
where
    F: Fn(Vec2) -> f32,
{
    let params = params.sanitized();
    if !start_depth.is_finite() || !end_depth.is_finite() {
        return 0.0;
    }
    if !start_uv.is_finite() || !end_uv.is_finite() {
        return 0.0;
    }

    let steps = params.steps;
    let inv_steps = 1.0 / steps as f32;
    for i in 1..=steps {
        let t = i as f32 * inv_steps;
        let uv = start_uv.lerp(end_uv, t);
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
            // Left the frame; a screen-space march cannot test outside it.
            break;
        }
        let ray_depth = start_depth + (end_depth - start_depth) * t;
        let scene = scene_depth(uv);
        if !scene.is_finite() {
            continue;
        }
        let diff = ray_depth - scene;
        if diff > params.bias && diff < params.thickness {
            // Fade the hit near the screen edge and over the march tail.
            let edge = screen_edge_fade(uv, params.edge_fade);
            // Ramp down across the last quarter of the ray so far hits soften.
            let tail = 1.0 - smoothstep01((t - 0.75) / 0.25);
            return (edge * tail).clamp(0.0, 1.0);
        }
    }
    0.0
}

/// Marches a soft contact shadow against an analytic signed-distance occluder.
///
/// `sdf` returns the signed distance from a point to the nearest occluder (`<= 0`
/// inside).  Starting a small `start_offset` along `dir` from `origin` (to avoid
/// self-intersection), the ray sphere-traces up to `max_distance`, accumulating
/// Quilez's penumbra bound `vis = min(vis, hardness * d / t)` where `d` is the
/// clearance at distance `t`.  A clearance `<= 0` is a solid hit and returns full
/// occlusion `1.0`.  Otherwise the returned occlusion is `1 - vis`, a soft edge
/// whose width grows with distance and shrinks with `hardness`.
///
/// `dir` is normalised internally; a degenerate (zero/non-finite) direction or a
/// non-positive `max_distance` returns `0.0` (lit).  `hardness` is clamped to a
/// positive floor so a zero never erases the penumbra.
pub fn sdf_soft_contact<F>(
    origin: Vec2,
    dir: Vec2,
    max_distance: f32,
    start_offset: f32,
    hardness: f32,
    steps: u32,
    sdf: F,
) -> f32
where
    F: Fn(Vec2) -> f32,
{
    let len = dir.length();
    if !(len > 0.0) || !len.is_finite() || !origin.is_finite() {
        return 0.0;
    }
    if !(max_distance > 0.0) || !max_distance.is_finite() {
        return 0.0;
    }
    let dir = dir / len;
    let hardness = if hardness.is_finite() {
        hardness.max(1.0e-3)
    } else {
        1.0e-3
    };
    let start_offset = if start_offset.is_finite() {
        start_offset.max(0.0)
    } else {
        0.0
    };
    let steps = steps.max(1);

    let mut vis = 1.0_f32;
    let mut t = start_offset.min(max_distance);
    for _ in 0..steps {
        if t >= max_distance {
            break;
        }
        let p = origin + dir * t;
        let d = sdf(p);
        if !d.is_finite() {
            break;
        }
        if d <= 0.0 {
            // Solid hit: fully occluded.
            return 1.0;
        }
        vis = vis.min(hardness * d / t.max(1.0e-6));
        // Advance by the clearance (sphere tracing), with a small floor so the
        // march always terminates in `steps` iterations.
        t += d.max(max_distance * 1.0e-3);
    }
    (1.0 - vis.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// Analytic occlusion of a ray by a single sphere (test/parity helper).
///
/// Returns `1.0` if the ray `origin + t*dir`, `t ∈ (start_offset, max_distance)`,
/// enters the sphere of the given `center`/`radius`, else `0.0`.  Used to
/// validate [`sdf_soft_contact`] against a closed-form occluder.  Degenerate
/// inputs return `0.0`.
pub fn analytic_sphere_occlusion(
    origin: Vec2,
    dir: Vec2,
    max_distance: f32,
    start_offset: f32,
    center: Vec2,
    radius: f32,
) -> f32 {
    let len = dir.length();
    if !(len > 0.0) || !len.is_finite() || !(max_distance > 0.0) || !(radius > 0.0) {
        return 0.0;
    }
    let dir = dir / len;
    let oc = origin - center;
    // Solve |oc + t*dir|^2 = radius^2  =>  t^2 + 2 b t + c = 0 (dir unit).
    let b = oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return 0.0;
    }
    let sqrt_d = disc.sqrt();
    let t0 = -b - sqrt_d;
    let t1 = -b + sqrt_d;
    let lo = start_offset.max(0.0);
    // A hit lies in (lo, max_distance) for either root, or the ray starts inside.
    let hit = (t0 > lo && t0 < max_distance)
        || (t1 > lo && t1 < max_distance)
        || (t0 <= lo && t1 >= lo);
    if hit {
        1.0
    } else {
        0.0
    }
}

/// Converts a world-space march length and light direction into the screen-space
/// end-point for [`screen_space_contact`].
///
/// Given the start UV/depth and a per-unit-length screen delta `uv_per_unit` and
/// depth delta `depth_per_unit` (both supplied by the caller's projection), this
/// returns `(end_uv, end_depth)` after marching `length` world units.  It exists
/// so the projection math has one documented, tested home.  A non-finite or
/// non-positive length returns the start unchanged.
#[inline]
pub fn project_march_end(
    start_uv: Vec2,
    start_depth: f32,
    uv_per_unit: Vec2,
    depth_per_unit: f32,
    length: f32,
) -> (Vec2, f32) {
    if !(length > 0.0) || !length.is_finite() {
        return (start_uv, start_depth);
    }
    let end_uv = start_uv + uv_per_unit * length;
    let scale = if depth_per_unit.is_finite() {
        depth_per_unit
    } else {
        0.0
    };
    (end_uv, start_depth + scale * length)
}

/// Converts a smooth hardness exponent into an equivalent Quilez `hardness`
/// factor via `exp2`, so callers can specify softness on a perceptual log scale.
///
/// `softness` in `[0, 1]` maps to a hardness in `[max, min]`: `0` is sharp
/// (large factor), `1` is soft (small factor).  The mapping is monotone and
/// defended against non-finite input.
#[inline]
pub fn softness_to_hardness(softness: f32, min_hardness: f32, max_hardness: f32) -> f32 {
    let s = if softness.is_finite() {
        softness.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let lo = min_hardness.max(1.0e-3);
    let hi = max_hardness.max(lo);
    // Interpolate in log2 space for perceptual uniformity.
    let log_lo = ops::log2(lo);
    let log_hi = ops::log2(hi);
    ops::exp2(log_hi + (log_lo - log_hi) * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_clamped_below_thickness() {
        assert!((clamp_self_shadow_bias(10.0, 1.0) - 0.9).abs() < 1e-6);
        assert_eq!(clamp_self_shadow_bias(-1.0, 1.0), 0.0);
        assert!(clamp_self_shadow_bias(f32::NAN, 1.0) == 0.0);
        // Non-finite thickness floors to a positive window.
        assert!(clamp_self_shadow_bias(1.0, f32::NAN) >= 0.0);
    }

    #[test]
    fn edge_fade_is_one_in_center_zero_at_border() {
        assert!((screen_edge_fade(Vec2::splat(0.5), 0.1) - 1.0).abs() < 1e-6);
        assert!(screen_edge_fade(Vec2::new(0.0, 0.5), 0.1) < 1e-6);
        // Disabled fade returns 1.
        assert_eq!(screen_edge_fade(Vec2::new(0.0, 0.0), 0.0), 1.0);
    }

    #[test]
    fn screen_space_detects_occluder_in_window() {
        // Flat occluder sits 0.1 units in front of the ray's depth across the
        // march; within (bias=0.02, thickness=0.5) so it should shadow.
        let params = ContactShadowParams {
            steps: 8,
            bias: 0.02,
            thickness: 0.5,
            edge_fade: 0.0,
        };
        let occ = screen_space_contact(
            params,
            Vec2::new(0.5, 0.5),
            1.0,
            Vec2::new(0.6, 0.5),
            1.0,
            |_uv| 0.9, // scene surface 0.1 in front of ray depth 1.0
        );
        assert!(occ > 0.5, "expected shadow, got {occ}");
    }

    #[test]
    fn screen_space_ignores_surface_behind_thickness() {
        let params = ContactShadowParams {
            steps: 8,
            bias: 0.02,
            thickness: 0.2,
            edge_fade: 0.0,
        };
        // Surface is 0.5 in front -> beyond thickness 0.2 -> no shadow.
        let occ = screen_space_contact(
            params,
            Vec2::new(0.5, 0.5),
            1.0,
            Vec2::new(0.6, 0.5),
            1.0,
            |_uv| 0.5,
        );
        assert_eq!(occ, 0.0);
    }

    #[test]
    fn screen_space_ignores_self_within_bias() {
        let params = ContactShadowParams {
            steps: 8,
            bias: 0.1,
            thickness: 0.5,
            edge_fade: 0.0,
        };
        // Surface only 0.05 in front -> within bias -> self, no shadow.
        let occ = screen_space_contact(
            params,
            Vec2::new(0.5, 0.5),
            1.0,
            Vec2::new(0.6, 0.5),
            1.0,
            |_uv| 0.95,
        );
        assert_eq!(occ, 0.0);
    }

    #[test]
    fn screen_space_non_finite_inputs_are_lit() {
        let p = ContactShadowParams::default();
        assert_eq!(
            screen_space_contact(p, Vec2::splat(0.5), f32::NAN, Vec2::splat(0.6), 1.0, |_| 0.9),
            0.0
        );
    }

    #[test]
    fn sdf_hits_solid_occluder() {
        // A disc of radius 0.3 centred ahead of the ray fully occludes.
        let occ = sdf_soft_contact(
            Vec2::ZERO,
            Vec2::new(1.0, 0.0),
            5.0,
            0.01,
            8.0,
            64,
            |p| (p - Vec2::new(2.0, 0.0)).length() - 0.3,
        );
        assert!((occ - 1.0).abs() < 1e-6, "expected full occlusion, got {occ}");
    }

    #[test]
    fn sdf_misses_distant_occluder() {
        // Disc far off the ray's path -> essentially lit.
        let occ = sdf_soft_contact(
            Vec2::ZERO,
            Vec2::new(1.0, 0.0),
            5.0,
            0.01,
            8.0,
            64,
            |p| (p - Vec2::new(2.0, 10.0)).length() - 0.3,
        );
        assert!(occ < 0.05, "expected lit, got {occ}");
    }

    #[test]
    fn sdf_grazing_occluder_is_soft() {
        // Disc that just grazes the ray path yields a partial (soft) occlusion.
        let near = sdf_soft_contact(
            Vec2::ZERO,
            Vec2::new(1.0, 0.0),
            5.0,
            0.01,
            8.0,
            128,
            |p| (p - Vec2::new(2.0, 0.35)).length() - 0.3,
        );
        assert!(near > 0.0 && near < 1.0, "expected soft edge, got {near}");
    }

    #[test]
    fn sdf_degenerate_direction_is_lit() {
        let occ = sdf_soft_contact(Vec2::ZERO, Vec2::ZERO, 5.0, 0.0, 8.0, 16, |_| 1.0);
        assert_eq!(occ, 0.0);
    }

    #[test]
    fn analytic_sphere_matches_sdf_hit_decision() {
        let origin = Vec2::ZERO;
        let dir = Vec2::new(1.0, 0.0);
        let center = Vec2::new(2.0, 0.0);
        let radius = 0.3;
        let analytic = analytic_sphere_occlusion(origin, dir, 5.0, 0.01, center, radius);
        let sdf = sdf_soft_contact(origin, dir, 5.0, 0.01, 8.0, 64, |p| {
            (p - center).length() - radius
        });
        assert_eq!(analytic, 1.0);
        assert!((sdf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn project_march_end_scales_linearly() {
        let (uv, d) = project_march_end(
            Vec2::new(0.5, 0.5),
            1.0,
            Vec2::new(0.1, 0.0),
            0.2,
            2.0,
        );
        assert!((uv.x - 0.7).abs() < 1e-6);
        assert!((d - 1.4).abs() < 1e-6);
        // Degenerate length returns the start.
        let (uv0, d0) =
            project_march_end(Vec2::new(0.5, 0.5), 1.0, Vec2::new(0.1, 0.0), 0.2, -1.0);
        assert_eq!((uv0, d0), (Vec2::new(0.5, 0.5), 1.0));
    }

    #[test]
    fn softness_mapping_is_monotone() {
        let sharp = softness_to_hardness(0.0, 1.0, 64.0);
        let soft = softness_to_hardness(1.0, 1.0, 64.0);
        let mid = softness_to_hardness(0.5, 1.0, 64.0);
        assert!((sharp - 64.0).abs() < 1e-3);
        assert!((soft - 1.0).abs() < 1e-3);
        assert!(mid < sharp && mid > soft);
    }

    #[test]
    fn params_sanitized_clamps_all_fields() {
        let p = ContactShadowParams {
            steps: 0,
            bias: 100.0,
            thickness: -1.0,
            edge_fade: f32::NAN,
        }
        .sanitized();
        assert_eq!(p.steps, 1);
        assert!(p.thickness > 0.0);
        assert!(p.bias <= 0.9 * p.thickness);
        assert_eq!(p.edge_fade, 0.0);
    }
}
