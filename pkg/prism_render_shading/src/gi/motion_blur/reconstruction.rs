//! McGuire reconstruction-filter motion blur — CPU golden reference.
//!
//! This is the backend-neutral reference for the gather stage of McGuire et al.
//! (2012), *"A Reconstruction Filter for Plausible Motion Blur"*.  For a center
//! pixel it walks a small set of taps placed by
//! [`crate::gi::motion_blur::sampling`] along the **dominant tile velocity**
//! (the NeighborMax result produced upstream by [`crate::gi::motion`]), and
//! accumulates a weighted average of their colors using the cone/cylinder/soft
//! depth primitives in [`crate::gi::motion_blur::weights`].
//!
//! The dominant velocity is supplied by the caller as a parameter — this module
//! deliberately does **not** compute tiles itself; it *consumes* the concept.
//!
//! ## Weighting (per tap `Y`, center `X`)
//! Let `vx = |X.velocity|`, `vy = |Y.velocity|`, and `dist` the pixel distance
//! from `X` to `Y`.  With `f = soft(X.depth, Y.depth)` (`X` in front) and
//! `b = soft(Y.depth, X.depth)` (`Y` in front), the tap weight is
//!
//! ```text
//! w =  f * cone(dist, vy)              // Y is the fast foreground smearing onto X
//!    + b * cone(dist, vx)              // X is the fast foreground smearing onto Y
//!    + cylinder(dist, vy, vx) * 2.0    // both uniformly blurred along the line
//! ```
//!
//! A center tap is always included with a small positive weight so the output is
//! defined even when every gather weight is zero, and so a sharp pixel keeps a
//! presence through its own blur.  The final color is the weight-normalised sum.
//!
//! # Conventions
//! * Velocities are in **pixels**; depths use *smaller = nearer* (pass a
//!   negated / reversed depth if your buffer is reverse-Z).
//! * If the dominant velocity is below the half-pixel threshold the pixel is not
//!   moving, so the center color is returned unchanged.
//! * Deterministic pure function: no RNG/IO/GPU/unsafe.  The caller supplies a
//!   gather closure `sample_at(offset_px) -> PixelSample`; this module performs
//!   the single sampling allocation and nothing else.
//! * Every returned channel is finite; non-finite taps are sanitized so no
//!   `NaN`/`inf` can escape.

use bevy_math::{Vec2, Vec3};

use super::sampling::along_velocity_offsets_thresholded;
use super::weights::{cone, cylinder, soft_depth_compare, velocity_magnitude};

/// Default number of taps gathered along the velocity line.
pub const DEFAULT_SAMPLE_COUNT: u32 = 15;

/// Default soft-z transition width (in depth units) for foreground/background
/// classification.
pub const DEFAULT_SOFT_Z_EXTENT: f32 = 1.0;

/// Default half-pixel "is it moving?" threshold (pixels).
pub const DEFAULT_HALF_VELOCITY_THRESHOLD_PX: f32 = 0.5;

/// Minimum velocity magnitude (pixels) used as a cone radius floor so a nearly
/// static tap still contributes a sane, bounded weight to its own location.
const MIN_CONE_RADIUS_PX: f32 = 0.5;

/// A single per-pixel sample fed to the reconstruction filter.
///
/// Mirrors the GPU twin's gather payload: linear `color`, screen-space
/// `velocity` (pixels), and scene `depth` (smaller = nearer).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelSample {
    /// Linear radiance / color at the pixel.
    pub color: Vec3,
    /// Screen-space velocity in pixels.
    pub velocity: Vec2,
    /// Scene depth; smaller is nearer.
    pub depth: f32,
}

impl PixelSample {
    /// Builds a sample, sanitizing every field so no `NaN`/`inf` enters the
    /// filter (non-finite color/velocity components become `0.0`; non-finite
    /// depth becomes `0.0`).
    #[inline]
    pub fn new(color: Vec3, velocity: Vec2, depth: f32) -> Self {
        Self {
            color: sanitize_vec3(color),
            velocity: sanitize_vec2(velocity),
            depth: if depth.is_finite() { depth } else { 0.0 },
        }
    }
}

/// Tunable parameters for the reconstruction filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconstructionParams {
    /// Number of taps along the velocity line (floored to `1`).
    pub sample_count: u32,
    /// Soft-z transition width for depth classification (clamped `> 0`).
    pub soft_z_extent: f32,
    /// Velocity magnitude (pixels) below which the pixel is "not moving".
    pub half_velocity_threshold_px: f32,
    /// Deterministic per-pixel jitter in `[0, 1)` (see
    /// [`crate::gi::motion_blur::sampling::interleaved_gradient_jitter`]).
    pub jitter: f32,
}

impl Default for ReconstructionParams {
    #[inline]
    fn default() -> Self {
        Self {
            sample_count: DEFAULT_SAMPLE_COUNT,
            soft_z_extent: DEFAULT_SOFT_Z_EXTENT,
            half_velocity_threshold_px: DEFAULT_HALF_VELOCITY_THRESHOLD_PX,
            jitter: 0.0,
        }
    }
}

/// Replaces non-finite components of a [`Vec2`] with `0.0`.
#[inline]
fn sanitize_vec2(v: Vec2) -> Vec2 {
    Vec2::new(
        if v.x.is_finite() { v.x } else { 0.0 },
        if v.y.is_finite() { v.y } else { 0.0 },
    )
}

/// Replaces non-finite components of a [`Vec3`] with `0.0`.
#[inline]
fn sanitize_vec3(v: Vec3) -> Vec3 {
    Vec3::new(
        if v.x.is_finite() { v.x } else { 0.0 },
        if v.y.is_finite() { v.y } else { 0.0 },
        if v.z.is_finite() { v.z } else { 0.0 },
    )
}

/// Computes the center tap weight.
///
/// McGuire gives the center a modest, velocity-aware weight so a fast pixel does
/// not drown out its own color while a slow pixel keeps near-unit presence.  We
/// use `1 / max(|vx|, MIN_CONE_RADIUS_PX)`, which is always finite and `> 0`.
#[inline]
fn center_weight(center_velocity: Vec2) -> f32 {
    let v = velocity_magnitude(center_velocity).max(MIN_CONE_RADIUS_PX);
    1.0 / v
}

/// Full reconstruction returning both the resolved color and the total weight.
///
/// Exposed for tests and callers that want to inspect confidence; most callers
/// use [`reconstruct`].  `sample_at(offset_px)` must return the [`PixelSample`]
/// at `center + offset_px` (in pixels).
pub fn reconstruct_weighted<F>(
    center: PixelSample,
    dominant_velocity: Vec2,
    params: ReconstructionParams,
    sample_at: F,
) -> (Vec3, f32)
where
    F: Fn(Vec2) -> PixelSample,
{
    let center = PixelSample::new(center.color, center.velocity, center.depth);
    let dominant_velocity = sanitize_vec2(dominant_velocity);
    let extent = params.soft_z_extent;
    let threshold = if params.half_velocity_threshold_px.is_finite() {
        params.half_velocity_threshold_px.max(0.0)
    } else {
        DEFAULT_HALF_VELOCITY_THRESHOLD_PX
    };

    // Not moving: the dominant velocity is below the half-pixel threshold.
    if velocity_magnitude(dominant_velocity) < threshold {
        let w = center_weight(center.velocity);
        return (center.color, w);
    }

    // Start from the center tap so the accumulator is never empty.
    let w_center = center_weight(center.velocity);
    let mut color_sum = center.color * w_center;
    let mut weight_sum = w_center;

    let offsets = along_velocity_offsets_thresholded(
        dominant_velocity,
        params.sample_count,
        params.jitter,
        threshold,
    );

    let vx = velocity_magnitude(center.velocity).max(MIN_CONE_RADIUS_PX);

    for offset in offsets {
        // The center tap is already accounted for; skip the (near) zero offset.
        if offset.length() <= 1.0e-6 {
            continue;
        }
        let tap = sample_at(offset);
        let tap = PixelSample::new(tap.color, tap.velocity, tap.depth);

        let dist = offset.length();
        let vy = velocity_magnitude(tap.velocity).max(MIN_CONE_RADIUS_PX);

        // Soft depth classification (smaller depth = nearer / in front).
        let f = soft_depth_compare(center.depth, tap.depth, extent);
        let b = soft_depth_compare(tap.depth, center.depth, extent);

        let w = f * cone(dist, vy)
            + b * cone(dist, vx)
            + cylinder(dist, vy, vx) * 2.0;
        let w = if w.is_finite() { w.max(0.0) } else { 0.0 };

        color_sum += tap.color * w;
        weight_sum += w;
    }

    let weight_sum = if weight_sum.is_finite() && weight_sum > 0.0 {
        weight_sum
    } else {
        // Should not happen (center weight is positive) but stay defensive.
        return (center.color, w_center.max(MIN_CONE_RADIUS_PX));
    };

    let resolved = sanitize_vec3(color_sum / weight_sum);
    (resolved, weight_sum)
}

/// Reconstructs the motion-blurred color for one pixel.
///
/// `center` is the pixel being resolved; `dominant_velocity` is the tile
/// NeighborMax velocity passing through it (pixels); `sample_at(offset_px)`
/// gathers the neighbour at `center + offset_px`.
///
/// Returns the center color unchanged when the dominant velocity is below the
/// half-pixel threshold; otherwise returns the weight-normalised McGuire gather.
/// The result is always finite.
#[inline]
pub fn reconstruct<F>(
    center: PixelSample,
    dominant_velocity: Vec2,
    params: ReconstructionParams,
    sample_at: F,
) -> Vec3
where
    F: Fn(Vec2) -> PixelSample,
{
    reconstruct_weighted(center, dominant_velocity, params, sample_at).0
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-4;

    fn approx_eq(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a - b).length() < eps
    }

    #[test]
    fn static_scene_returns_center_color() {
        let center = PixelSample::new(Vec3::new(0.2, 0.4, 0.6), Vec2::ZERO, 1.0);
        // Dominant velocity below threshold -> no blur.
        let out = reconstruct(center, Vec2::new(0.1, 0.0), ReconstructionParams::default(), |_| {
            PixelSample::new(Vec3::new(9.0, 9.0, 9.0), Vec2::ZERO, 1.0)
        });
        assert!(approx_eq(out, center.color, EPS), "got {out:?}");
    }

    #[test]
    fn constant_field_is_smooth_and_preserves_color() {
        // A uniform moving field: every tap shares the same color/velocity/depth,
        // so the weighted average must reproduce that color exactly.
        let color = Vec3::new(0.7, 0.3, 0.1);
        let velocity = Vec2::new(12.0, 0.0);
        let center = PixelSample::new(color, velocity, 2.0);
        let out = reconstruct(center, velocity, ReconstructionParams::default(), |_| {
            PixelSample::new(color, velocity, 2.0)
        });
        assert!(approx_eq(out, color, EPS), "got {out:?}");
    }

    #[test]
    fn total_weight_is_positive() {
        let center = PixelSample::new(Vec3::new(1.0, 1.0, 1.0), Vec2::new(10.0, 2.0), 1.0);
        let (_, w) = reconstruct_weighted(
            center,
            Vec2::new(10.0, 2.0),
            ReconstructionParams::default(),
            |_| PixelSample::new(Vec3::ZERO, Vec2::new(10.0, 2.0), 1.0),
        );
        assert!(w > 0.0, "weight sum should be positive, got {w}");
    }

    #[test]
    fn deterministic_across_calls() {
        let center = PixelSample::new(Vec3::new(0.5, 0.25, 0.75), Vec2::new(8.0, 6.0), 1.0);
        let params = ReconstructionParams { jitter: 0.42, ..Default::default() };
        let field = |o: Vec2| {
            // A deterministic synthetic field varying with position.
            let shade = 0.5 + 0.1 * o.x - 0.05 * o.y;
            PixelSample::new(Vec3::splat(shade.clamp(0.0, 1.0)), Vec2::new(8.0, 6.0), 1.0)
        };
        let a = reconstruct(center, Vec2::new(8.0, 6.0), params, field);
        let b = reconstruct(center, Vec2::new(8.0, 6.0), params, field);
        assert_eq!(a, b);
    }

    #[test]
    fn fast_foreground_blurs_over_static_background() {
        // Background (center) is static and far; a fast red foreground passes in
        // front of it.  The resolved color should pull toward the foreground.
        let bg = PixelSample::new(Vec3::new(0.0, 0.0, 1.0), Vec2::ZERO, 10.0);
        let fg_color = Vec3::new(1.0, 0.0, 0.0);
        let out = reconstruct(bg, Vec2::new(20.0, 0.0), ReconstructionParams::default(), |_| {
            // Foreground: nearer (smaller depth) and fast.
            PixelSample::new(fg_color, Vec2::new(20.0, 0.0), 1.0)
        });
        // Red channel must have increased relative to the pure background.
        assert!(out.x > 0.1, "foreground did not bleed in: {out:?}");
        assert!(out.is_finite());
    }

    #[test]
    fn non_finite_taps_are_sanitized() {
        let center = PixelSample::new(Vec3::new(0.5, 0.5, 0.5), Vec2::new(10.0, 0.0), 1.0);
        let out = reconstruct(center, Vec2::new(10.0, 0.0), ReconstructionParams::default(), |_| {
            PixelSample {
                color: Vec3::new(f32::NAN, f32::INFINITY, 0.5),
                velocity: Vec2::new(f32::NAN, 10.0),
                depth: f32::INFINITY,
            }
        });
        assert!(out.x.is_finite() && out.y.is_finite() && out.z.is_finite());
    }

    #[test]
    fn sample_count_floored_and_bounded() {
        let center = PixelSample::new(Vec3::ONE, Vec2::new(5.0, 0.0), 1.0);
        let params = ReconstructionParams { sample_count: 0, ..Default::default() };
        let out = reconstruct(center, Vec2::new(5.0, 0.0), params, |_| {
            PixelSample::new(Vec3::ONE, Vec2::new(5.0, 0.0), 1.0)
        });
        assert!(approx_eq(out, Vec3::ONE, EPS));
    }
}
