//! Relief mapping — linear search plus binary-search refinement (CPU golden).
//!
//! Relief mapping sharpens the parallax-occlusion idea.  Where plain [POM]
//! marches a fixed stack of layers and linearly interpolates the crossing,
//! relief mapping first does a coarse *linear search* to bracket the layer
//! interval the view ray crosses the surface in, then runs a handful of
//! *binary-search* refinements inside that interval to pin the intersection
//! down to near machine precision.  This is the Policarpo–Oliveira
//! *Real-Time Relief Mapping* (2005) construction, and the reference the
//! WESL/GPU twin's relief pass must reproduce.
//!
//! The height-field and tangent-space conventions match [`crate::gi::parallax`]
//! and the [POM] module: `height_at(uv) -> f32` returns the surface height in
//! `[0, 1]` (`1` on the polygon plane, `0` at the deepest valley), the march
//! works in depth `= 1 - height` increasing downward from the plane, the
//! tangent-space view direction `view_ts` points surface-to-eye with `+z` along
//! the normal, and the ray samples `base_uv - t * p` at depth `t` for the
//! parallax vector `p = (view_ts.xy / view_ts.z) * height_scale`.
//!
//! [POM]: crate::gi::parallax::pom
//! [`crate::gi::parallax`]: crate::gi::parallax
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; `sqrt` uses the inherent
//!   method.  No `f32::exp()`-style free functions.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Height is `[0, 1]` with `1` at the polygon plane; depth is `1 - height`.
//! * The linear search brackets the crossing; the binary search bisects that
//!   bracket, each step keeping the "above surface" and "below surface" ends.
//! * Defensive clamping everywhere: non-finite inputs, a degenerate or
//!   back-facing view direction, and empty depth ranges all fall back to "no
//!   displacement" so no `NaN`/`inf` ever escapes.

use bevy_math::{Vec2, Vec3};

/// Smallest `view_ts.z` treated as a usable, front-facing view direction.
const MIN_VIEW_Z: f32 = 1.0e-4;

/// Lower bound on the coarse linear-search layer count.
const MIN_LINEAR_STEPS: u32 = 1;

/// Upper bound on the coarse linear-search layer count.
const MAX_LINEAR_STEPS: u32 = 1024;

/// Upper bound on the binary-search refinement count (ample for `f32`).
const MAX_BINARY_STEPS: u32 = 32;

/// Tuning for a [`relief_map`] march.
///
/// Field layout mirrors the packed `u32`/`f32` constants the GPU twin consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReliefConfig {
    /// Coarse linear-search layers used to bracket the crossing interval.
    pub linear_steps: u32,
    /// Binary-search refinements run inside the bracketed interval.
    pub binary_steps: u32,
    /// Scales height-field units into tangent-plane texture units.
    pub height_scale: f32,
}

impl Default for ReliefConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl ReliefConfig {
    /// A sensible default: 16 linear steps, 8 binary refinements, unit height.
    pub const DEFAULT: Self = Self {
        linear_steps: 16,
        binary_steps: 8,
        height_scale: 1.0,
    };

    /// Builds a configuration, clamping every field to a finite, sane range.
    pub fn new(linear_steps: u32, binary_steps: u32, height_scale: f32) -> Self {
        let linear_steps = linear_steps.clamp(MIN_LINEAR_STEPS, MAX_LINEAR_STEPS);
        let binary_steps = binary_steps.min(MAX_BINARY_STEPS);
        let height_scale = if height_scale.is_finite() {
            height_scale.max(0.0)
        } else {
            0.0
        };
        Self {
            linear_steps,
            binary_steps,
            height_scale,
        }
    }
}

/// Outcome of a [`relief_map`] march.
///
/// All fields are finite and in range.  When the ray never crosses the surface
/// (`hit == false`) the coordinate is left at `base_uv` and `depth` is `0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReliefSample {
    /// Refined, parallax-displaced texture coordinate to shade with.
    pub uv: Vec2,
    /// Depth of the crossing in `[0, 1]` (`0` at the plane, `1` deepest).
    pub depth: f32,
    /// Whether the ray actually crossed the surface within the height field.
    pub hit: bool,
}

impl ReliefSample {
    fn miss(base_uv: Vec2) -> Self {
        Self {
            uv: base_uv,
            depth: 0.0,
            hit: false,
        }
    }
}

/// Marches the view ray through the height field with a linear search followed
/// by binary-search refinement, returning the refined parallax hit.
///
/// `base_uv` is the texture coordinate at the polygon plane, `view_ts` the
/// tangent-space surface-to-eye direction (`+z` along the normal, need not be
/// normalised), and `height_at` the height-field closure (values outside
/// `[0, 1]` are clamped).
pub fn relief_map<F>(base_uv: Vec2, view_ts: Vec3, cfg: ReliefConfig, height_at: F) -> ReliefSample
where
    F: Fn(Vec2) -> f32,
{
    let cfg = ReliefConfig::new(cfg.linear_steps, cfg.binary_steps, cfg.height_scale);
    let base_uv = sanitize_vec2(base_uv);

    let view = sanitize_vec3(view_ts);
    let len = view.length();
    if len <= MIN_VIEW_Z {
        return ReliefSample::miss(base_uv);
    }
    let view = view / len;
    if view.z <= MIN_VIEW_Z {
        return ReliefSample::miss(base_uv);
    }

    let parallax = Vec2::new(view.x, view.y) / view.z * cfg.height_scale;
    let sample_depth = |depth: f32| -> f32 { depth_at(&height_at, base_uv - parallax * depth) };

    let n = cfg.linear_steps;
    let layer_depth = 1.0 / n as f32;

    // --- Coarse linear search: find the first layer where the ray reaches or
    // passes under the surface. ---
    let mut cur_depth = 0.0f32;
    let mut cur_surface = sample_depth(cur_depth);
    let mut prev_depth = 0.0f32;

    // The plane itself may already be at/under the surface (height == 1).
    if cur_depth >= cur_surface {
        return ReliefSample {
            uv: base_uv,
            depth: cur_surface.clamp(0.0, 1.0),
            hit: true,
        };
    }

    let mut crossed = false;
    for _ in 0..n {
        prev_depth = cur_depth;
        cur_depth += layer_depth;
        cur_surface = sample_depth(cur_depth);
        if cur_depth >= cur_surface {
            crossed = true;
            break;
        }
    }

    if !crossed {
        // Bounded fields always cross by `depth == 1`; reaching here means the
        // closure stayed strictly above the ray (degenerate), so retreat.
        return ReliefSample::miss(base_uv);
    }

    // --- Binary-search refinement inside the bracket [prev_depth, cur_depth].
    // `lo` is kept above the surface, `hi` at/under it. ---
    let mut lo = prev_depth;
    let mut hi = cur_depth;
    for _ in 0..cfg.binary_steps {
        let mid = 0.5 * (lo + hi);
        let surf = sample_depth(mid);
        if mid < surf {
            // Still above the surface: the crossing is deeper.
            lo = mid;
        } else {
            // At or under the surface: the crossing is shallower.
            hi = mid;
        }
    }

    let depth = (0.5 * (lo + hi)).clamp(0.0, 1.0);
    let uv = sanitize_vec2(base_uv - parallax * depth);
    ReliefSample {
        uv,
        depth,
        hit: true,
    }
}

/// Surface depth (`1 - height`) at `uv`, with the height clamped to `[0, 1]`
/// and non-finite samples treated as the plane (`depth = 0`).
fn depth_at<F>(height_at: &F, uv: Vec2) -> f32
where
    F: Fn(Vec2) -> f32,
{
    let h = height_at(uv);
    let h = if h.is_finite() { h.clamp(0.0, 1.0) } else { 1.0 };
    (1.0 - h).clamp(0.0, 1.0)
}

/// Replaces any non-finite component of a `Vec2` with `0`.
fn sanitize_vec2(v: Vec2) -> Vec2 {
    Vec2::new(finite_or_zero(v.x), finite_or_zero(v.y))
}

/// Replaces any non-finite component of a `Vec3` with `0`.
fn sanitize_vec3(v: Vec3) -> Vec3 {
    Vec3::new(finite_or_zero(v.x), finite_or_zero(v.y), finite_or_zero(v.z))
}

/// Returns `x` when finite, otherwise `0`.
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
    use bevy_math::ops;

    /// A ramp whose height falls off along `+x`: `height = clamp(1 - slope*x)`.
    fn ramp(slope: f32) -> impl Fn(Vec2) -> f32 {
        move |uv: Vec2| (1.0 - slope * uv.x).clamp(0.0, 1.0)
    }

    fn flat(h: f32) -> impl Fn(Vec2) -> f32 {
        move |_uv: Vec2| h
    }

    fn view_from_angle(deg: f32) -> Vec3 {
        let r = deg * core::f32::consts::PI / 180.0;
        Vec3::new(ops::sin(r), 0.0, ops::cos(r)).normalize()
    }

    /// Closed-form crossing depth for `ramp(slope)` with parallax `px` along x
    /// starting at `x0`, valid while the intersection stays in the ramp's
    /// unclamped region.  Solve `t = slope*(x0 - t*px)`.
    fn analytic_depth(slope: f32, x0: f32, px: f32) -> f32 {
        slope * x0 / (1.0 + slope * px)
    }

    #[test]
    fn flat_top_surface_hits_at_plane() {
        let s = relief_map(Vec2::new(0.3, 0.4), view_from_angle(40.0), ReliefConfig::DEFAULT, flat(1.0));
        assert!(s.hit);
        assert!(s.depth.abs() < 1e-6, "depth {}", s.depth);
        assert!((s.uv - Vec2::new(0.3, 0.4)).length() < 1e-6);
    }

    #[test]
    fn matches_analytic_ramp_intersection() {
        let deg = 45.0f32;
        let slope = 0.6f32;
        let x0 = 0.5f32;
        let view = view_from_angle(deg);
        let px = (view.x / view.z) * ReliefConfig::DEFAULT.height_scale;
        let expected = analytic_depth(slope, x0, px);

        let s = relief_map(Vec2::new(x0, 0.5), view, ReliefConfig::new(16, 16, 1.0), ramp(slope));
        assert!(s.hit);
        assert!((s.depth - expected).abs() < 1e-3, "got {} expected {}", s.depth, expected);
    }

    #[test]
    fn error_shrinks_with_more_binary_steps() {
        let deg = 50.0f32;
        let slope = 0.7f32;
        let x0 = 0.5f32;
        let view = view_from_angle(deg);
        let px = (view.x / view.z) * 1.0;
        let expected = analytic_depth(slope, x0, px);

        let coarse = relief_map(Vec2::new(x0, 0.5), view, ReliefConfig::new(8, 1, 1.0), ramp(slope));
        let fine = relief_map(Vec2::new(x0, 0.5), view, ReliefConfig::new(8, 20, 1.0), ramp(slope));
        let e_coarse = (coarse.depth - expected).abs();
        let e_fine = (fine.depth - expected).abs();
        assert!(e_fine <= e_coarse, "fine {} should beat coarse {}", e_fine, e_coarse);
        assert!(e_fine < 2e-4, "fine error too large: {}", e_fine);
    }

    #[test]
    fn displacement_grows_monotonically_with_view_tilt() {
        // For this ramp the crossing *depth* falls as the ray leans into
        // shallower relief, but the horizontal texture displacement `t * px`
        // rises monotonically with tilt, which is the robust parallax invariant.
        let base = Vec2::new(0.5, 0.5);
        let shift = |deg: f32| {
            let s = relief_map(base, view_from_angle(deg), ReliefConfig::DEFAULT, ramp(0.5));
            (base.x - s.uv.x).max(0.0)
        };
        let a = shift(10.0);
        let b = shift(35.0);
        let c = shift(60.0);
        assert!(a < b && b < c, "displacements not monotone: {} {} {}", a, b, c);
    }

    #[test]
    fn displaces_toward_minus_view_x() {
        let base = Vec2::new(0.5, 0.5);
        let s = relief_map(base, view_from_angle(55.0), ReliefConfig::DEFAULT, ramp(0.7));
        assert!(s.hit);
        assert!(s.uv.x < base.x, "expected shift toward -x: {}", s.uv.x);
        assert!((s.uv.y - base.y).abs() < 1e-6);
    }

    #[test]
    fn degenerate_view_retreats() {
        let base = Vec2::new(0.2, 0.8);
        let a = relief_map(base, Vec3::ZERO, ReliefConfig::DEFAULT, ramp(0.6));
        let b = relief_map(base, Vec3::new(1.0, 0.0, 0.0), ReliefConfig::DEFAULT, ramp(0.6));
        assert!(!a.hit && a.uv == base);
        assert!(!b.hit && b.uv == base);
    }

    #[test]
    fn results_are_finite_and_bounded() {
        for deg in [5.0f32, 25.0, 45.0, 65.0, 85.0] {
            let s = relief_map(Vec2::new(0.5, 0.5), view_from_angle(deg), ReliefConfig::DEFAULT, ramp(0.8));
            assert!(s.uv.is_finite());
            assert!(s.depth.is_finite() && (0.0..=1.0).contains(&s.depth));
        }
    }

    #[test]
    fn non_finite_inputs_never_produce_nan() {
        let s = relief_map(
            Vec2::new(f32::NAN, 0.5),
            Vec3::new(0.0, f32::INFINITY, 1.0),
            ReliefConfig::new(0, 999, f32::NAN),
            ramp(0.5),
        );
        assert!(s.uv.is_finite() && s.depth.is_finite());
    }

    #[test]
    fn results_are_deterministic() {
        let v = view_from_angle(37.0);
        let a = relief_map(Vec2::new(0.45, 0.55), v, ReliefConfig::DEFAULT, ramp(0.65));
        let b = relief_map(Vec2::new(0.45, 0.55), v, ReliefConfig::DEFAULT, ramp(0.65));
        assert_eq!(a, b);
    }
}
