//! Reflection-probe / local cubemap fallback — CPU golden.
//!
//! Screen-space reflection can only reflect what is on screen; it misses
//! off-screen geometry and rays that leave the frustum.  Engines fall back to a
//! prefiltered **reflection probe** (a local cubemap captured from a fixed
//! point) for those rays.  A probe captured at one point but sampled at another
//! shows obvious parallax error, so the reflected ray is **parallax-corrected**:
//! it is intersected against a simple proxy volume (box or sphere) approximating
//! the room, and the direction is re-pointed from the probe centre to that
//! intersection (Lagarde & Zanuttini, *Local Image-Based Lighting with
//! Parallax-Corrected Cubemaps*).  The final reflection is a confidence- and
//! roughness-weighted blend of the traced (SSR) colour and the probe colour.
//!
//! This module is the backend-neutral reference for that fallback:
//!
//! * [`parallax_correct_box`] / [`parallax_correct_sphere`] re-point a reflected
//!   ray at the surface of an AABB or sphere proxy.
//! * [`blend_weights`] / [`blend_reflection`] combine traced and probe colours
//!   under a confidence/roughness-driven, normalised weight.
//! * [`roughness_to_mip`] selects the prefiltered mip for a roughness, and
//!   [`probe_sample_oct`] returns the octahedral lookup UV (reusing the sibling
//!   [`crate::gi::world_space::octahedral`] mapping) plus that mip.
//!
//! # Conventions
//! * `no_std`: math via `bevy_math`; transcendentals via [`bevy_math::ops`].
//! * Positions, directions and proxy bounds are world-space [`Vec3`]s in a
//!   right-handed frame.  Corrected directions are returned as unit vectors.
//! * Blend weights always satisfy `w_traced + w_probe == 1` and lie in `[0, 1]`.
//! * `roughness` is a perceptual roughness in `[0, 1]`; the selected mip grows
//!   monotonically with it over `[0, mip_count - 1]`.
//! * Every function is deterministic and defensive: degenerate (zero) inputs
//!   fall back to a finite result and never produce `NaN`.

use bevy_math::{Vec2, Vec3};

use crate::gi::world_space::octahedral::dir_to_oct;

/// Smallest component magnitude treated as a real ray-direction slope when
/// intersecting a proxy; smaller is treated as parallel (no crossing on that
/// axis).
const MIN_SLOPE: f32 = 1.0e-6;

/// Result of a parallax correction: the re-pointed direction and the proxy
/// intersection it was built from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParallaxHit {
    /// Unit direction from the probe centre to the proxy intersection.
    pub direction: Vec3,
    /// World-space proxy intersection point.
    pub hit_point: Vec3,
    /// Parametric distance along the reflected ray to [`ParallaxHit::hit_point`].
    pub distance: f32,
}

/// Normalises `v`, falling back to `fallback` for a degenerate (zero) input.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > f32::MIN_POSITIVE && len_sq.is_finite() {
        v * len_sq.sqrt().recip()
    } else {
        fallback
    }
}

/// Parallax-corrects a reflected ray against an axis-aligned box proxy.
///
/// Intersects the ray `position + t · reflection` (`t ≥ 0`) with the AABB
/// `[box_min, box_max]`, then returns the direction from `probe_center` to that
/// exit point.  This is the standard parallax-corrected local-cubemap fix: the
/// returned direction is what the probe cubemap should be sampled with so the
/// reflection lines up with the room geometry.
///
/// A zero (or non-finite) `reflection`, or a degenerate box, falls back to the
/// normalised input direction (or `+Y`) so the result is always a finite unit
/// vector.
pub fn parallax_correct_box(
    position: Vec3,
    reflection: Vec3,
    box_min: Vec3,
    box_max: Vec3,
    probe_center: Vec3,
) -> ParallaxHit {
    let dir = normalize_or(reflection, Vec3::Y);
    let lo = box_min.min(box_max);
    let hi = box_min.max(box_max);

    // Per-axis parameters to the two slabs; a near-parallel axis contributes no
    // finite exit distance (its `t` is pushed to +∞).
    let t_exit = slab_exit(position.x, dir.x, lo.x, hi.x)
        .min(slab_exit(position.y, dir.y, lo.y, hi.y))
        .min(slab_exit(position.z, dir.z, lo.z, hi.z));

    if !t_exit.is_finite() || t_exit <= 0.0 {
        return ParallaxHit {
            direction: dir,
            hit_point: position,
            distance: 0.0,
        };
    }

    let hit_point = position + dir * t_exit;
    let direction = normalize_or(hit_point - probe_center, dir);
    ParallaxHit {
        direction,
        hit_point,
        distance: t_exit,
    }
}

/// Farthest forward exit parameter of a 1-D slab `[lo, hi]` for a ray
/// `o + t·d`, or `+∞` when the ray is parallel to the slab.
#[inline]
fn slab_exit(o: f32, d: f32, lo: f32, hi: f32) -> f32 {
    if d.abs() <= MIN_SLOPE {
        // Parallel: no exit crossing on this axis.
        return f32::INFINITY;
    }
    let inv = 1.0 / d;
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    // The exit of this slab is the larger of the two plane intersections.
    t0.max(t1)
}

/// Parallax-corrects a reflected ray against a sphere proxy.
///
/// Intersects the ray `position + t · reflection` (`t ≥ 0`) with the sphere of
/// radius `radius` centred at `sphere_center`, taking the forward exit point,
/// then returns the direction from `probe_center` to it.  Degenerate inputs
/// (zero direction, non-positive radius, ray origin that yields no forward
/// intersection) fall back to the normalised direction.
pub fn parallax_correct_sphere(
    position: Vec3,
    reflection: Vec3,
    sphere_center: Vec3,
    radius: f32,
    probe_center: Vec3,
) -> ParallaxHit {
    let dir = normalize_or(reflection, Vec3::Y);
    let fallback = ParallaxHit {
        direction: dir,
        hit_point: position,
        distance: 0.0,
    };
    if !(radius > 0.0) {
        return fallback;
    }

    // Solve |o + t d - c|^2 = r^2 for the forward exit root (dir is unit, so
    // the quadratic's leading coefficient is 1).
    let m = position - sphere_center;
    let b = m.dot(dir);
    let c = m.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return fallback;
    }
    let sqrt_disc = disc.sqrt();
    // Exit root is the larger of (-b ± sqrt_disc).
    let t_exit = -b + sqrt_disc;
    if !t_exit.is_finite() || t_exit <= 0.0 {
        return fallback;
    }

    let hit_point = position + dir * t_exit;
    let direction = normalize_or(hit_point - probe_center, dir);
    ParallaxHit {
        direction,
        hit_point,
        distance: t_exit,
    }
}

/// Normalised blend weights between the traced (SSR) and probe reflections.
///
/// The traced result is trusted when it is confident and the surface is sharp,
/// so `w_traced = clamp(confidence) · (1 - clamp(roughness))` and
/// `w_probe = 1 - w_traced`.  This is monotonically non-decreasing in
/// confidence and non-increasing in roughness, and the pair always sums to `1`.
///
/// Returns `(w_traced, w_probe)`, both in `[0, 1]`.
#[inline]
pub fn blend_weights(confidence: f32, roughness: f32) -> (f32, f32) {
    let c = confidence.clamp(0.0, 1.0);
    let r = roughness.clamp(0.0, 1.0);
    let w_traced = (c * (1.0 - r)).clamp(0.0, 1.0);
    (w_traced, 1.0 - w_traced)
}

/// Blends a traced and a probe RGB radiance under [`blend_weights`].
///
/// Returns `w_traced · traced + w_probe · probe`, a finite non-negative-safe
/// linear combination (inputs are used as-is; weights are the normalised pair).
#[inline]
pub fn blend_reflection(traced: Vec3, probe: Vec3, confidence: f32, roughness: f32) -> Vec3 {
    let (wt, wp) = blend_weights(confidence, roughness);
    traced * wt + probe * wp
}

/// Maps a roughness to a prefiltered probe mip level.
///
/// Linearly maps `roughness ∈ [0, 1]` onto `[0, mip_count - 1]`, so a mirror
/// (`roughness = 0`) samples the sharp level 0 and the roughest surface samples
/// the coarsest level.  The result is clamped to the valid mip range and is
/// monotonically non-decreasing in roughness.  A `mip_count` of `0` is treated
/// as `1` (only level 0 exists).
#[inline]
pub fn roughness_to_mip(roughness: f32, mip_count: u32) -> f32 {
    let max_mip = (mip_count.max(1) - 1) as f32;
    (roughness.clamp(0.0, 1.0) * max_mip).clamp(0.0, max_mip)
}

/// Octahedral lookup UV and mip for sampling a reflection probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeLookup {
    /// Octahedral UV in `[0, 1]^2` for the (corrected) direction.
    pub uv: Vec2,
    /// Prefiltered mip level selected from the roughness.
    pub mip: f32,
}

/// Computes the octahedral lookup for a probe sample.
///
/// Encodes `direction` with the shared [`dir_to_oct`] mapping and selects the
/// prefiltered mip via [`roughness_to_mip`].  A degenerate (zero) direction is
/// mapped to the `+Z` pole (the octahedral map's safe centre) so the lookup is
/// always finite.
#[inline]
pub fn probe_sample_oct(direction: Vec3, roughness: f32, mip_count: u32) -> ProbeLookup {
    let dir = normalize_or(direction, Vec3::Z);
    ProbeLookup {
        uv: dir_to_oct(dir),
        mip: roughness_to_mip(roughness, mip_count),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_intersection_hits_known_point() {
        // Unit box centred at the origin; probe at the centre.
        let lo = Vec3::splat(-1.0);
        let hi = Vec3::splat(1.0);
        let center = Vec3::ZERO;

        // Ray from the origin along +X exits at (1, 0, 0).
        let hit = parallax_correct_box(Vec3::ZERO, Vec3::X, lo, hi, center);
        assert!((hit.hit_point - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
        assert!((hit.direction - Vec3::X).length() < 1e-5);
        assert!((hit.distance - 1.0).abs() < 1e-5);

        // Offset origin along the ray still exits at the +X wall.
        let hit2 = parallax_correct_box(Vec3::new(0.5, 0.0, 0.0), Vec3::X, lo, hi, center);
        assert!((hit2.hit_point - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);

        // Diagonal exits at the corner edge (1, 1, 0).
        let diag = Vec3::new(1.0, 1.0, 0.0).normalize();
        let hit3 = parallax_correct_box(Vec3::ZERO, diag, lo, hi, center);
        assert!((hit3.hit_point - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn box_correction_repoints_from_offset_probe() {
        let lo = Vec3::splat(-2.0);
        let hi = Vec3::splat(2.0);
        // Probe is offset from the shading point, so the corrected direction
        // differs from the raw reflection direction.
        let probe = Vec3::new(-1.0, 0.0, 0.0);
        let hit = parallax_correct_box(Vec3::new(1.0, 0.0, 0.0), Vec3::X, lo, hi, probe);
        assert!((hit.hit_point - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-5);
        // From probe (-1,0,0) to (2,0,0) is still +X here.
        assert!((hit.direction - Vec3::X).length() < 1e-5);
        assert!((hit.direction.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sphere_intersection_hits_known_point() {
        let center = Vec3::ZERO;
        // Unit sphere, ray from origin along +X exits at (1, 0, 0).
        let hit = parallax_correct_sphere(Vec3::ZERO, Vec3::X, center, 1.0, center);
        assert!((hit.hit_point - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
        assert!((hit.direction - Vec3::X).length() < 1e-5);
        assert!((hit.distance - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sphere_miss_falls_back() {
        // Ray starts outside and points away: no forward intersection.
        let hit = parallax_correct_sphere(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::X,
            Vec3::ZERO,
            1.0,
            Vec3::ZERO,
        );
        // Falls back to the normalised direction, finite and unit.
        assert!((hit.direction - Vec3::X).length() < 1e-5);
        assert!(hit.direction.is_finite());
    }

    #[test]
    fn blend_weights_normalise_and_are_monotone() {
        // Sum is always 1.
        for &c in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            for &r in &[0.0f32, 0.3, 0.6, 1.0] {
                let (wt, wp) = blend_weights(c, r);
                assert!((wt + wp - 1.0).abs() < 1e-6, "sum {} + {}", wt, wp);
                assert!((0.0..=1.0).contains(&wt));
                assert!((0.0..=1.0).contains(&wp));
            }
        }
        // Full confidence + mirror → fully traced.
        assert_eq!(blend_weights(1.0, 0.0), (1.0, 0.0));
        // Zero confidence → fully probe.
        assert_eq!(blend_weights(0.0, 0.5), (0.0, 1.0));

        // Monotone: traced weight rises with confidence, falls with roughness.
        let a = blend_weights(0.3, 0.4).0;
        let b = blend_weights(0.7, 0.4).0;
        assert!(b >= a);
        let c0 = blend_weights(0.8, 0.2).0;
        let c1 = blend_weights(0.8, 0.9).0;
        assert!(c1 <= c0);
    }

    #[test]
    fn blend_reflection_interpolates_endpoints() {
        let traced = Vec3::new(1.0, 0.0, 0.0);
        let probe = Vec3::new(0.0, 0.0, 1.0);
        // Fully traced.
        assert!((blend_reflection(traced, probe, 1.0, 0.0) - traced).length() < 1e-6);
        // Fully probe.
        assert!((blend_reflection(traced, probe, 0.0, 0.0) - probe).length() < 1e-6);
        // Midpoint weight (confidence 0.5, mirror) → 0.5 each.
        let mid = blend_reflection(traced, probe, 0.5, 0.0);
        assert!((mid - Vec3::new(0.5, 0.0, 0.5)).length() < 1e-6);
    }

    #[test]
    fn roughness_to_mip_is_monotone_and_bounded() {
        let mips = 6u32;
        let mut prev = -1.0f32;
        for i in 0..=10u32 {
            let r = i as f32 / 10.0;
            let m = roughness_to_mip(r, mips);
            assert!(m >= 0.0 && m <= (mips - 1) as f32, "mip {} out of range", m);
            assert!(m >= prev - 1e-6, "mip not monotone: {} then {}", prev, m);
            prev = m;
        }
        assert!((roughness_to_mip(0.0, mips) - 0.0).abs() < 1e-6);
        assert!((roughness_to_mip(1.0, mips) - 5.0).abs() < 1e-6);
        // Degenerate mip_count is treated as a single level.
        assert!((roughness_to_mip(0.8, 0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn probe_sample_oct_matches_octahedral_and_mip() {
        let dir = Vec3::new(0.2, 0.5, -0.3).normalize();
        let look = probe_sample_oct(dir, 0.5, 5);
        assert_eq!(look.uv, dir_to_oct(dir));
        assert!((look.mip - roughness_to_mip(0.5, 5)).abs() < 1e-6);
        // Zero direction is safe (maps to the +Z pole, centre of the square).
        let look0 = probe_sample_oct(Vec3::ZERO, 0.2, 5);
        assert!(look0.uv.is_finite());
        assert!((look0.uv - dir_to_oct(Vec3::Z)).length() < 1e-6);
    }

    #[test]
    fn zero_direction_box_is_clamped() {
        let hit = parallax_correct_box(
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            Vec3::ZERO,
        );
        assert!(hit.direction.is_finite());
        assert!((hit.direction.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn determinism() {
        let a = parallax_correct_box(
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.4, -0.2, 0.9),
            Vec3::splat(-3.0),
            Vec3::splat(3.0),
            Vec3::new(0.0, 0.5, 0.0),
        );
        let b = parallax_correct_box(
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.4, -0.2, 0.9),
            Vec3::splat(-3.0),
            Vec3::splat(3.0),
            Vec3::new(0.0, 0.5, 0.0),
        );
        assert_eq!(a, b);
    }
}
