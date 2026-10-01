//! Parallax occlusion mapping (POM) — CPU golden reference.
//!
//! A single textured polygon is flat, yet its height map describes a whole
//! field of bumps, grooves, and ridges standing above and below the polygon
//! plane.  Parallax occlusion mapping fakes that relief without extra geometry
//! by walking the view ray *through* the height field and shifting the sampled
//! texture coordinate to wherever the ray first dips below the surface.  This
//! module is the backend-neutral reference for that walk, following the layered
//! ray march of Tatarchuk's *Dynamic Parallax Occlusion Mapping* (ATI, 2006)
//! and the steep-parallax lineage of Kaneko et al.
//!
//! The height field is handed in as a closure `height_at(uv) -> f32`.  By this
//! module's convention the value is the surface *height above the valley floor*
//! in `[0, 1]`: `1` is the ridge sitting on the polygon plane and `0` is the
//! deepest valley.  Internally the march works in *depth* measured downward
//! from the polygon plane, `depth = 1 - height`, so the plane is `depth = 0`
//! and the deepest valley is `depth = 1`.
//!
//! The tangent-space view direction `view_ts` points from the shaded surface
//! toward the eye, with `+z` along the geometric normal.  The total texture
//! shift accumulated between the plane and full depth is the parallax vector
//! `p = (view_ts.xy / view_ts.z) * height_scale`; at a march depth `t` the ray
//! samples `base_uv - t * p`.  The march steps down one layer at a time until
//! the ray depth meets or passes the surface depth, then does one linear
//! interpolation between the straddling layers for the crossing coordinate.
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; `sqrt` uses the inherent
//!   method.  No `f32::exp()`-style free functions.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Height is `[0, 1]` with `1` at the polygon plane; depth is `1 - height`.
//! * The layer count adapts to view obliquity: grazing rays (small `view_ts.z`)
//!   use up to `max_layers`, head-on rays as few as `min_layers`.
//! * Defensive clamping everywhere: non-finite inputs, a degenerate or
//!   back-facing view direction, and empty depth ranges all fall back to "no
//!   displacement" so no `NaN`/`inf` ever escapes.

use bevy_math::{ops, Vec2, Vec3};

/// Smallest `view_ts.z` treated as a usable, front-facing view direction.
///
/// Below this the ray is effectively parallel to the polygon plane, the
/// parallax vector `view_ts.xy / view_ts.z` would blow up, and the march is
/// meaningless, so the reference returns the undisplaced coordinate instead.
const MIN_VIEW_Z: f32 = 1.0e-4;

/// Smallest tolerated layer count, regardless of configuration.
const MIN_LAYER_FLOOR: f32 = 1.0;

/// Largest layer count the reference will ever march, bounding work and keeping
/// the step size representable.
const MAX_LAYER_CEIL: f32 = 1024.0;

/// Tuning for a [`parallax_occlusion`] march.
///
/// The field layout mirrors the packed `f32`/`u32` constants the GPU twin
/// consumes.  Construct via [`PomConfig::new`] (which sanitises the inputs) or
/// take [`PomConfig::DEFAULT`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PomConfig {
    /// Layer count used for head-on views (ray along the normal).
    pub min_layers: f32,
    /// Layer count used for the most grazing views.
    pub max_layers: f32,
    /// Scales height-field units into tangent-plane texture units; the larger
    /// it is, the deeper the apparent relief and the longer the parallax shift.
    pub height_scale: f32,
}

impl Default for PomConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl PomConfig {
    /// A sensible default: 8 head-on layers, 32 grazing layers, unit height.
    pub const DEFAULT: Self = Self {
        min_layers: 8.0,
        max_layers: 32.0,
        height_scale: 1.0,
    };

    /// Builds a configuration, clamping every field to a finite, sane range.
    ///
    /// Non-finite or non-positive layer counts fall back to the default pair;
    /// `max_layers` is lifted to at least `min_layers`.  A non-finite
    /// `height_scale` becomes `0` (a flat surface, i.e. no parallax).
    pub fn new(min_layers: f32, max_layers: f32, height_scale: f32) -> Self {
        let min_layers = sanitize_layers(min_layers, Self::DEFAULT.min_layers);
        let max_layers = sanitize_layers(max_layers, Self::DEFAULT.max_layers).max(min_layers);
        let height_scale = if height_scale.is_finite() {
            height_scale.max(0.0)
        } else {
            0.0
        };
        Self {
            min_layers,
            max_layers,
            height_scale,
        }
    }
}

/// Outcome of a [`parallax_occlusion`] march.
///
/// All fields are finite and in range.  When the ray never dips below the
/// surface (`hit == false`) the coordinate is left at `base_uv` and `depth` is
/// `0`, i.e. the shader should treat the texel as lying on the polygon plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PomSample {
    /// Parallax-displaced texture coordinate to shade with.
    pub uv: Vec2,
    /// Depth of the crossing in `[0, 1]` (`0` at the plane, `1` deepest).
    pub depth: f32,
    /// Whether the ray actually crossed the surface within the height field.
    pub hit: bool,
    /// Number of layers the march used (after view-dependent adaptation).
    pub layers: u32,
}

impl PomSample {
    /// The "no displacement" result: sit on the plane at `base_uv`.
    fn miss(base_uv: Vec2, layers: u32) -> Self {
        Self {
            uv: base_uv,
            depth: 0.0,
            hit: false,
            layers,
        }
    }
}

/// Walks the view ray through the height field and returns the parallax hit.
///
/// `base_uv` is the texture coordinate at the polygon plane.  `view_ts` is the
/// tangent-space direction from the surface toward the eye (`+z` along the
/// normal); it need not be normalised.  `height_at` returns the surface height
/// in `[0, 1]` for any coordinate (values outside the range are clamped).
///
/// The march uses a view-adaptive layer count, compares the ray depth against
/// the sampled surface depth layer by layer, and linearly interpolates between
/// the last two layers once the ray crosses under the surface.
pub fn parallax_occlusion<F>(base_uv: Vec2, view_ts: Vec3, cfg: PomConfig, height_at: F) -> PomSample
where
    F: Fn(Vec2) -> f32,
{
    let cfg = PomConfig::new(cfg.min_layers, cfg.max_layers, cfg.height_scale);
    let base_uv = sanitize_vec2(base_uv);

    // Reject degenerate / back-facing view directions up front.
    let view = sanitize_vec3(view_ts);
    let len = view.length();
    if len <= MIN_VIEW_Z {
        return PomSample::miss(base_uv, cfg.min_layers as u32);
    }
    let view = view / len;
    if view.z <= MIN_VIEW_Z {
        return PomSample::miss(base_uv, cfg.min_layers as u32);
    }

    // View-adaptive layer count: `view.z` is the cosine with the normal, so a
    // head-on ray (cos -> 1) uses `min_layers`, a grazing ray `max_layers`.
    let cos_view = view.z.clamp(0.0, 1.0);
    let layer_f = cfg.max_layers + (cfg.min_layers - cfg.max_layers) * cos_view;
    let layer_count = ops::floor(layer_f + 0.5).clamp(MIN_LAYER_FLOOR, MAX_LAYER_CEIL);
    let n = layer_count as u32;
    let layer_depth = 1.0 / layer_count;

    // Total texture shift from plane to full depth, and the per-layer step.
    let parallax = Vec2::new(view.x, view.y) / view.z * cfg.height_scale;
    let delta_uv = parallax * layer_depth;

    let sample_depth = |uv: Vec2| -> f32 { depth_at(&height_at, uv) };

    // Layered ray march from the plane downward.
    let mut cur_layer_depth = 0.0f32;
    let mut cur_uv = base_uv;
    let mut cur_surface = sample_depth(cur_uv);

    // Previous-layer state, needed for the crossing interpolation.
    let mut prev_layer_depth = 0.0f32;
    let mut prev_uv = base_uv;
    let mut prev_surface = cur_surface;

    let mut crossed = cur_layer_depth >= cur_surface;
    let mut steps = 0u32;
    while !crossed && steps < n {
        prev_layer_depth = cur_layer_depth;
        prev_uv = cur_uv;
        prev_surface = cur_surface;

        cur_layer_depth += layer_depth;
        cur_uv -= delta_uv;
        cur_surface = sample_depth(cur_uv);

        crossed = cur_layer_depth >= cur_surface;
        steps += 1;
    }

    if !crossed {
        return PomSample::miss(base_uv, n);
    }

    // Signed gaps between the ray depth and the surface at the two layers.
    // `after >= 0` (ray at/under the surface now) and `before <= 0` (ray above
    // the surface previously); their difference is strictly positive.
    let after = cur_layer_depth - cur_surface;
    let before = prev_layer_depth - prev_surface;
    let denom = after - before;
    let weight = if denom.abs() > f32::MIN_POSITIVE {
        (after / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Interpolate coordinate and depth across the straddled layers.
    let uv = prev_uv * weight + cur_uv * (1.0 - weight);
    let depth = (prev_layer_depth * weight + cur_layer_depth * (1.0 - weight)).clamp(0.0, 1.0);

    PomSample {
        uv: sanitize_vec2(uv),
        depth,
        hit: true,
        layers: n,
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

/// Clamps a configured layer count to a finite, positive value.
fn sanitize_layers(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value >= MIN_LAYER_FLOOR {
        value.min(MAX_LAYER_CEIL)
    } else {
        fallback
    }
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

    /// A flat surface at height `h` everywhere.
    fn flat(h: f32) -> impl Fn(Vec2) -> f32 {
        move |_uv: Vec2| h
    }

    /// A ramp whose height falls off along `+x`: `height = clamp(1 - slope*x)`.
    /// Deeper (smaller height) as `x` grows, so a ray leaning toward `+x`
    /// penetrates and shifts the coordinate.
    fn ramp(slope: f32) -> impl Fn(Vec2) -> f32 {
        move |uv: Vec2| (1.0 - slope * uv.x).clamp(0.0, 1.0)
    }

    fn view_from_angle(deg: f32) -> Vec3 {
        let r = deg * core::f32::consts::PI / 180.0;
        // Tilt toward +x; +z stays along the normal.
        Vec3::new(ops::sin(r), 0.0, ops::cos(r)).normalize()
    }

    #[test]
    fn flat_top_surface_has_no_displacement() {
        // Height 1 everywhere == the whole plane sits at depth 0.
        let s = parallax_occlusion(Vec2::new(0.3, 0.7), view_from_angle(45.0), PomConfig::DEFAULT, flat(1.0));
        assert!((s.uv - Vec2::new(0.3, 0.7)).length() < 1e-6, "uv {:?}", s.uv);
        assert!(s.depth.abs() < 1e-6, "depth {}", s.depth);
        assert!(s.hit, "a plane at depth 0 is an immediate hit");
    }

    #[test]
    fn constant_depth_surface_is_finite_and_bounded() {
        // A flat but *sunken* surface (height 0.5) still resolves cleanly.
        let s = parallax_occlusion(Vec2::ZERO, view_from_angle(30.0), PomConfig::DEFAULT, flat(0.5));
        assert!(s.uv.is_finite());
        assert!(s.depth.is_finite() && (0.0..=1.0).contains(&s.depth));
    }

    #[test]
    fn ramp_displaces_along_minus_view_x() {
        // View leans toward +x, so the parallax vector is +x and the sampled
        // coordinate shifts toward -x.
        let s = parallax_occlusion(Vec2::new(0.5, 0.5), view_from_angle(50.0), PomConfig::DEFAULT, ramp(0.8));
        assert!(s.hit);
        assert!(s.uv.x < 0.5, "expected shift toward -x, got {}", s.uv.x);
        assert!((s.uv.y - 0.5).abs() < 1e-6, "y should be untouched: {}", s.uv.y);
    }

    #[test]
    fn displacement_grows_monotonically_with_view_tilt() {
        let base = Vec2::new(0.5, 0.5);
        let shift = |deg: f32| {
            let s = parallax_occlusion(base, view_from_angle(deg), PomConfig::DEFAULT, ramp(0.6));
            (base.x - s.uv.x).max(0.0)
        };
        let a = shift(15.0);
        let b = shift(35.0);
        let c = shift(60.0);
        assert!(a < b, "15deg {} should shift less than 35deg {}", a, b);
        assert!(b < c, "35deg {} should shift less than 60deg {}", b, c);
    }

    #[test]
    fn near_vertical_view_barely_displaces() {
        let base = Vec2::new(0.5, 0.5);
        let s = parallax_occlusion(base, view_from_angle(0.5), PomConfig::DEFAULT, ramp(0.6));
        assert!((base.x - s.uv.x).abs() < 5e-3, "near-vertical shift too large: {}", s.uv.x);
    }

    #[test]
    fn degenerate_view_returns_plane() {
        let base = Vec2::new(0.2, 0.9);
        // Zero vector and a grazing (z ~ 0) vector both bail out.
        let a = parallax_occlusion(base, Vec3::ZERO, PomConfig::DEFAULT, ramp(0.6));
        let b = parallax_occlusion(base, Vec3::new(1.0, 0.0, 0.0), PomConfig::DEFAULT, ramp(0.6));
        assert_eq!(a.uv, base);
        assert!(!a.hit);
        assert_eq!(b.uv, base);
        assert!(!b.hit);
    }

    #[test]
    fn non_finite_inputs_never_produce_nan() {
        let base = Vec2::new(f32::NAN, 0.5);
        let view = Vec3::new(f32::INFINITY, 0.0, 1.0);
        let s = parallax_occlusion(base, view, PomConfig::new(f32::NAN, -3.0, f32::NAN), flat(0.5));
        assert!(s.uv.is_finite());
        assert!(s.depth.is_finite());
    }

    #[test]
    fn adaptive_layer_count_tracks_view_angle() {
        let head_on = parallax_occlusion(Vec2::ZERO, view_from_angle(1.0), PomConfig::DEFAULT, ramp(0.5));
        let grazing = parallax_occlusion(Vec2::ZERO, view_from_angle(80.0), PomConfig::DEFAULT, ramp(0.5));
        assert!(head_on.layers <= grazing.layers, "head-on {} grazing {}", head_on.layers, grazing.layers);
        assert!(head_on.layers >= 8 && grazing.layers <= 32);
    }

    #[test]
    fn results_are_deterministic() {
        let v = view_from_angle(42.0);
        let a = parallax_occlusion(Vec2::new(0.4, 0.6), v, PomConfig::DEFAULT, ramp(0.7));
        let b = parallax_occlusion(Vec2::new(0.4, 0.6), v, PomConfig::DEFAULT, ramp(0.7));
        assert_eq!(a, b);
    }

    #[test]
    fn config_new_lifts_max_above_min_and_floors_scale() {
        let c = PomConfig::new(20.0, 4.0, -1.0);
        assert!(c.max_layers >= c.min_layers);
        assert_eq!(c.height_scale, 0.0);
    }
}
