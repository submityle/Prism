//! Screen-space reflections (`SSR`): the `CPU`-verifiable golden reference for
//! the `view`/screen-space ray-march that mirrors nearby geometry into glossy
//! particle surfaces (design §16-§21).
//!
//! A wet decal, a puddle sprite, or a polished metal flake should pick up the
//! scene around it, but a full reflection probe is far too expensive per
//! particle. Production engines instead reuse what is already on screen: from
//! the shaded fragment they reflect the view ray about the surface normal, walk
//! that reflected ray forward through `view` space, project each step to screen
//! `UV`, read the scene depth buffer there, and accept the first step whose
//! `view`-space depth crosses *behind* the stored surface within a thickness
//! window. The recovered `UV` then indexes the already-shaded colour buffer, and
//! a confidence term fades the sample where the reflection is unreliable. Every
//! step here lives in `view`-space depth (larger means farther from the camera)
//! and pure vector algebra, so a future `GPU` kernel can reproduce this
//! reference bit for bit.
//!
//! The pieces are: (1) [`reflect`] mirrors an incident direction about a normal
//! with the textbook `reflect(v, n) = v - 2 * dot(v, n) * n` identity — no
//! transcendental call, just a dot product and a scale; (2) [`Projection`] maps
//! a `view`-space point to screen `UV` through a pinhole (`NDC` then the
//! half-scale-bias to `UV`); (3) [`trace_reflection`] performs the fixed-step
//! `DDA`/linear ray-march, brackets the first front-to-back depth crossing
//! against the `(0 .. thickness)` acceptance window, and runs a bisection
//! *secant refine* to land precisely on the surface; (4) [`screen_edge_fade`],
//! [`grazing_fade`], and [`distance_fade`] combine into the reflection
//! confidence (screen-border `smoothstep`, a fade for rays that point back
//! toward the viewer and therefore sample off-screen content, and a march-budget
//! falloff); and (5) [`SsrParams`] / [`SsrHit`] gather the tunables and the
//! packed hit result in `std430`-friendly records whose `vec4` alignment follows
//! the shared strides from [`super::gpu_layout`].
//!
//! Deliberately out of scope and distinct from the siblings: screen-space
//! *contact shadows* march toward the light for a binary occlusion factor
//! ([`super::contact_shadow`]); signed-distance-field shadows sphere-trace an
//! `SDF` ([`super::distance_field_shadow`]); hierarchical-`Z` occlusion
//! *culling* lives in [`super::occlusion`]. This file marches the reflected ray
//! to recover a reflection `UV` plus confidence and nothing else — no motion
//! vectors, no temporal accumulation — and imports no sibling particle module
//! beyond [`super::gpu_layout`]. Surface normals and scene depth arrive as
//! inputs, so it never depends on `normal_reconstruct`.
//!
//! Only `f32::sqrt` / `f32::floor`, `f32::clamp`, integer arithmetic, and
//! integer `div_ceil` are used — no transcendental functions — so results stay
//! deterministic and platform independent.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Soft-edge / denominator guard below which a `smoothstep` interval collapses
/// to a hard step, so no divide-by-zero can produce a `NaN`.
const MIN_EDGE: f32 = 1e-6;

/// Minimum vector length treated as non-degenerate when normalizing, so a
/// zero-length direction can never divide by zero.
const MIN_NORM: f32 = 1e-6;

/// Minimum positive `view`-space depth in front of the pinhole; points at or
/// behind it cannot be projected to a finite `UV`.
const NEAR_MIN: f32 = 1e-4;

/// Minimum march step length treated as making progress, guarding the
/// `max_distance / max_steps` division against a degenerate schedule.
const MIN_STEP: f32 = 1e-6;

/// Byte stride of one [`SsrParams`] record in a `std430` storage buffer.
///
/// The six scalars pack into two `vec4` slots: `vec4(max_steps, max_distance,
/// thickness, refine_steps)` followed by `vec4(edge_fade, grazing_bias, pad,
/// pad)`.
pub const SSR_PARAMS_STRIDE: usize = 2 * VEC4_STRIDE;

/// Byte stride of one packed [`SsrHit`] record in a `std430` storage buffer.
///
/// The result packs into two `vec4` slots: `vec4(uv.x, uv.y, confidence,
/// hit_distance)` followed by `vec4(hit_flag, pad, pad, pad)`.
pub const SSR_HIT_STRIDE: usize = 2 * VEC4_STRIDE;

/// Clamps a scalar into the `0..=1` range without branching on equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Hermite `smoothstep` from `edge0` to `edge1` evaluated at `x`.
///
/// Returns `0.0` at or below `edge0`, `1.0` at or above `edge1`, and the
/// `t * t * (3 - 2 * t)` interpolation in between. A degenerate (near-equal)
/// interval collapses to a hard step at `edge1` rather than dividing by zero.
#[must_use]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span < MIN_EDGE {
        return if x < edge1 { 0.0 } else { 1.0 };
    }
    let t = clamp01((x - edge0) / span);
    t * t * (3.0 - 2.0 * t)
}

/// A two-component screen-space coordinate (typically a `UV` in `0..=1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    /// Horizontal component.
    pub x: f32,
    /// Vertical component.
    pub y: f32,
}

impl Vec2 {
    /// The origin `(0, 0)`.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Creates a coordinate from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// A three-component `view`-space vector (position or direction).
///
/// The camera sits at the origin looking down `+z`, so a larger `z` is farther
/// from the camera — the same depth convention the scene depth buffer uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// `X` component.
    pub x: f32,
    /// `Y` component.
    pub y: f32,
    /// `Z` component (`view`-space depth; larger is farther).
    pub z: f32,
}

impl Vec3 {
    /// Creates a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum (named `plus` to avoid the [`core::ops::Add`] trait
    /// convention that `clippy::should_implement_trait` would flag).
    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Component-wise difference (named `minus` to avoid the
    /// [`core::ops::Sub`] trait convention).
    #[must_use]
    pub fn minus(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Scales every component by `s`.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot product with `other`.
    #[must_use]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Euclidean length (the only `sqrt` on this type).
    #[must_use]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns the unit-length direction, or the vector unchanged when it is
    /// shorter than [`MIN_NORM`] (so a degenerate direction can never divide by
    /// zero or produce a `NaN`).
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < MIN_NORM {
            self
        } else {
            self.scale(1.0 / len)
        }
    }
}

/// Mirrors an incident direction about a surface normal.
///
/// Implements the textbook identity `reflect(v, n) = v - 2 * dot(v, n) * n`
/// using only a dot product and a scale — no transcendental call. When `n` is
/// unit length this preserves the length of `v`, flips the sign of the
/// normal-aligned component (so the angle of incidence equals the angle of
/// reflection), and leaves the tangential component untouched.
#[must_use]
pub fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    let d = incident.dot(normal);
    incident.minus(normal.scale(2.0 * d))
}

/// A pinhole projection from `view` space to screen `UV`.
///
/// The focal lengths are the `NDC`-per-`view`-unit scales along each axis (the
/// cotangent-of-half-`FoV` terms of a perspective matrix), supplied as inputs so
/// this module needs no trigonometry. A `view`-space point `p` projects to
/// `ndc = focal * p.xy / p.z`, then to `uv = 0.5 + 0.5 * ndc`, mapping the
/// screen centre to `(0.5, 0.5)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    /// Horizontal `NDC`-per-`view`-unit scale.
    pub focal_x: f32,
    /// Vertical `NDC`-per-`view`-unit scale.
    pub focal_y: f32,
}

impl Projection {
    /// Creates a projection from its focal scales.
    #[must_use]
    pub const fn new(focal_x: f32, focal_y: f32) -> Self {
        Self { focal_x, focal_y }
    }

    /// Projects a `view`-space point to screen `UV`, or [`None`] when the point
    /// is at or behind the pinhole (`z <= NEAR_MIN`) and therefore has no finite
    /// screen position.
    #[must_use]
    pub fn project(self, p: Vec3) -> Option<Vec2> {
        if p.z < NEAR_MIN {
            return None;
        }
        let ndc_x = self.focal_x * p.x / p.z;
        let ndc_y = self.focal_y * p.y / p.z;
        Some(Vec2::new(0.5 + 0.5 * ndc_x, 0.5 + 0.5 * ndc_y))
    }
}

/// Screen-border confidence fade for a reflection `UV`.
///
/// A recovered `UV` near the screen edge indexes colour that is about to leave
/// the frame, so its reflection is unreliable. Each axis is faded with a
/// `smoothstep` ramp of width `edge` at both borders, and the two axes multiply.
/// At an exact border the fade is `0.0`; well inside the frame it is `1.0`. A
/// non-positive `edge` disables the ramp and simply returns `1.0` inside the
/// `0..=1` box and `0.0` outside it.
#[must_use]
pub fn screen_edge_fade(uv: Vec2, edge: f32) -> f32 {
    if edge < MIN_EDGE {
        let inside = (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y);
        return if inside { 1.0 } else { 0.0 };
    }
    let fx = smoothstep(0.0, edge, uv.x) * smoothstep(0.0, edge, 1.0 - uv.x);
    let fy = smoothstep(0.0, edge, uv.y) * smoothstep(0.0, edge, 1.0 - uv.y);
    clamp01(fx * fy)
}

/// Confidence fade for reflected rays that point back toward the viewer.
///
/// A reflection whose direction heads back toward the camera samples content
/// that lives *behind* the viewer and is therefore absent from the screen, so
/// `SSR` cannot resolve it. The alignment `facing = dot(reflect_dir, view_dir)`
/// is positive when the reflected ray continues into the scene (reliable) and
/// negative when it turns back toward the camera (unreliable). A `smoothstep`
/// across `(-bias .. bias)` fades from `0.0` for strongly backward rays to
/// `1.0` for forward rays. Both directions are expected to be unit length.
#[must_use]
pub fn grazing_fade(reflect_dir: Vec3, view_dir: Vec3, bias: f32) -> f32 {
    let facing = reflect_dir.dot(view_dir);
    smoothstep(-bias, bias, facing)
}

/// March-budget confidence fade based on how far along the ray a hit landed.
///
/// A hit near the shaded fragment is trustworthy; one found only after nearly
/// exhausting the march has accumulated projection and depth error, so its
/// confidence should taper toward the end of the search radius. The fade is
/// `1 - smoothstep(0, 1, hit_distance / max_distance)`, so it is `1.0` at the
/// origin and decreases monotonically to `0.0` at `max_distance`. A non-positive
/// `max_distance` yields `0.0`.
#[must_use]
pub fn distance_fade(hit_distance: f32, max_distance: f32) -> f32 {
    if max_distance < MIN_STEP {
        return 0.0;
    }
    let frac = hit_distance / max_distance;
    1.0 - smoothstep(0.0, 1.0, frac)
}

/// Builds the per-step `view`-space march distances along the reflected ray.
///
/// Entry `k` is `(k + 1) * max_distance / max_steps`, the distance at which the
/// `k`-th `DDA`/linear step reads the depth buffer. The schedule mirrors the
/// fixed stepping inside [`trace_reflection`] so callers can pre-plan or debug
/// the sample positions. A zero step count yields an empty schedule.
#[must_use]
pub fn march_schedule(params: &SsrParams) -> Vec<f32> {
    if params.max_steps == 0 {
        return Vec::new();
    }
    let step_len = params.max_distance / params.max_steps as f32;
    let mut out = Vec::with_capacity(params.max_steps as usize);
    let mut t = 0.0_f32;
    for _ in 0..params.max_steps {
        t += step_len;
        out.push(t);
    }
    out
}

/// Number of `GPU` workgroups needed to cover `pixel_count` invocations at
/// `group_size` threads each, via integer `div_ceil` (never zero-sized).
#[must_use]
pub fn dispatch_groups(pixel_count: u32, group_size: u32) -> u32 {
    pixel_count.div_ceil(group_size.max(1))
}

/// Tunables for the screen-space reflection ray-march (design §16-§21).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrParams {
    /// Number of fixed `DDA`/linear steps taken along the reflected ray.
    pub max_steps: u32,
    /// `view`-space distance the march covers from the shaded fragment along
    /// the reflected ray; also the denominator of the march-budget fade.
    pub max_distance: f32,
    /// Width of the acceptance window: a front-to-back depth crossing counts as
    /// a hit only when the depth gap at the behind sample is below this,
    /// otherwise the ray is assumed to have tunneled behind a thin silhouette.
    pub thickness: f32,
    /// Number of bisection *secant refine* iterations run after a crossing is
    /// bracketed, tightening the hit toward the exact surface.
    pub refine_steps: u32,
    /// Screen-border `smoothstep` width used by [`screen_edge_fade`].
    pub edge_fade: f32,
    /// `smoothstep` half-width used by [`grazing_fade`] for backward-pointing
    /// reflected rays.
    pub grazing_bias: f32,
}

impl SsrParams {
    /// Creates reflection parameters from all fields.
    #[must_use]
    pub const fn new(
        max_steps: u32,
        max_distance: f32,
        thickness: f32,
        refine_steps: u32,
        edge_fade: f32,
        grazing_bias: f32,
    ) -> Self {
        Self {
            max_steps,
            max_distance,
            thickness,
            refine_steps,
            edge_fade,
            grazing_bias,
        }
    }

    /// Traces the reflected ray for these parameters; see [`trace_reflection`]
    /// for the full contract.
    #[must_use]
    pub fn trace<F>(
        &self,
        projection: Projection,
        origin_view: Vec3,
        normal_view: Vec3,
        sample_depth: F,
    ) -> SsrHit
    where
        F: Fn(Vec2) -> Option<f32>,
    {
        trace_reflection(self, projection, origin_view, normal_view, sample_depth)
    }

    /// Packs the parameters into their `std430` `vec4`-aligned word layout.
    ///
    /// Layout: `[max_steps, max_distance, thickness, refine_steps, edge_fade,
    /// grazing_bias, pad, pad]` as raw `u32` words (the four `f32` fields via
    /// `f32::to_bits`) — two `vec4` slots, matching [`SSR_PARAMS_STRIDE`]. The
    /// trailing words are padding.
    #[must_use]
    pub fn to_std430(&self) -> [u32; 8] {
        [
            self.max_steps,
            self.max_distance.to_bits(),
            self.thickness.to_bits(),
            self.refine_steps,
            self.edge_fade.to_bits(),
            self.grazing_bias.to_bits(),
            0,
            0,
        ]
    }
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`SsrParams`] records (clamp-to-one-element rule from [`storage_bytes`]).
#[must_use]
pub fn ssr_params_buffer_bytes(count: usize) -> usize {
    storage_bytes(SSR_PARAMS_STRIDE, count)
}

/// The recovered reflection sample: where the reflected ray hit, how far along
/// it landed, and how much to trust the result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrHit {
    /// Screen `UV` of the reflection sample (meaningful only when `hit`).
    pub uv: Vec2,
    /// Confidence in `0..=1` folding the edge, grazing, and distance fades;
    /// `0.0` on a miss.
    pub confidence: f32,
    /// `view`-space distance along the reflected ray at which the hit landed;
    /// `0.0` on a miss.
    pub hit_distance: f32,
    /// Whether a valid front-to-back crossing was found within the thickness
    /// window before the march was exhausted.
    pub hit: bool,
}

impl SsrHit {
    /// The empty result returned when no valid reflection is found.
    #[must_use]
    pub fn miss() -> Self {
        Self {
            uv: Vec2::ZERO,
            confidence: 0.0,
            hit_distance: 0.0,
            hit: false,
        }
    }

    /// Packs the hit into its `std430` `vec4`-aligned word layout.
    ///
    /// Layout: `[uv.x, uv.y, confidence, hit_distance, hit_flag, pad, pad,
    /// pad]` as raw `u32` words (the four `f32` fields via `f32::to_bits`, the
    /// boolean as `0`/`1`) — two `vec4` slots, matching [`SSR_HIT_STRIDE`].
    #[must_use]
    pub fn to_std430(&self) -> [u32; 8] {
        [
            self.uv.x.to_bits(),
            self.uv.y.to_bits(),
            self.confidence.to_bits(),
            self.hit_distance.to_bits(),
            u32::from(self.hit),
            0,
            0,
            0,
        ]
    }
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`SsrHit`] records (clamp-to-one-element rule from [`storage_bytes`]).
#[must_use]
pub fn ssr_hit_buffer_bytes(count: usize) -> usize {
    storage_bytes(SSR_HIT_STRIDE, count)
}

/// Traces a reflected ray through `view`/screen space and returns the recovered
/// reflection sample.
///
/// The incident view direction is `view_dir = normalize(origin_view)` (from the
/// pinhole at the origin toward the shaded fragment). It is mirrored about the
/// unit `normal_view` via [`reflect`] to give the march direction. The ray is
/// then stepped in `max_steps` equal `view`-space increments of `max_distance /
/// max_steps`. At each step the sample position is projected with `projection`
/// and its `UV` fed to `sample_depth`, which returns the scene `view`-space
/// depth there (or [`None`] when the `UV` is off-screen or has no stored
/// surface). The signed depth gap `diff = ray_z - scene_z` is negative while the
/// ray is in front of the stored surface and positive once it passes behind.
///
/// A hit is the first step where `diff` crosses from `<= 0` (in front) to `> 0`
/// (behind) **and** the behind-sample gap is below `thickness`; a larger gap
/// means the ray skipped past a thin silhouette into distant background and is
/// rejected as penetration. Once bracketed, `refine_steps` bisection *secant*
/// iterations tighten the crossing between the two straddling steps for a
/// precise reflection `UV`. The confidence multiplies [`screen_edge_fade`],
/// [`grazing_fade`], and [`distance_fade`].
///
/// Returns [`SsrHit::miss`] when there is no valid crossing — including a
/// zero-step or degenerate schedule, a ray that runs off-screen, or a march that
/// is exhausted without ever passing behind a nearby surface.
#[must_use]
pub fn trace_reflection<F>(
    params: &SsrParams,
    projection: Projection,
    origin_view: Vec3,
    normal_view: Vec3,
    sample_depth: F,
) -> SsrHit
where
    F: Fn(Vec2) -> Option<f32>,
{
    if params.max_steps == 0 {
        return SsrHit::miss();
    }
    let step_len = params.max_distance / params.max_steps as f32;
    if step_len < MIN_STEP {
        return SsrHit::miss();
    }

    let view_dir = origin_view.normalize();
    let reflect_dir = reflect(view_dir, normal_view.normalize()).normalize();

    // Signed depth gap and screen UV at march distance `t`, or None when the
    // sample projects off-screen or lands on empty depth.
    let eval = |t: f32| -> Option<(Vec2, f32)> {
        let pos = origin_view.plus(reflect_dir.scale(t));
        let uv = projection.project(pos)?;
        let scene_z = sample_depth(uv)?;
        Some((uv, pos.z - scene_z))
    };

    let mut prev = eval(0.0);
    let mut prev_t = 0.0_f32;
    let mut t = 0.0_f32;
    for _ in 0..params.max_steps {
        t += step_len;
        let cur = eval(t);
        match (prev, cur) {
            (Some((_, prev_diff)), Some((_, cur_diff))) => {
                let crossed = prev_diff <= 0.0 && cur_diff > 0.0;
                if crossed && cur_diff < params.thickness {
                    let (hit_uv, hit_t) = refine_crossing(prev_t, t, params.refine_steps, &eval);
                    return finalize_hit(hit_uv, hit_t, reflect_dir, view_dir, params);
                }
            }
            _ => {
                // The ray left the screen or the depth field: stop marching.
                break;
            }
        }
        prev = cur;
        prev_t = t;
    }
    SsrHit::miss()
}

/// Bisection *secant refine* of a bracketed front-to-back crossing.
///
/// `lo_t` is the last in-front distance and `hi_t` the first behind distance.
/// Each iteration evaluates the midpoint and keeps the half that still straddles
/// the crossing, so `hi_t` converges onto the surface from behind. The returned
/// `UV` is the projection at the refined distance. Midpoints that fall off-screen
/// end the refinement early and keep the current behind bound.
fn refine_crossing<F>(mut lo_t: f32, mut hi_t: f32, iterations: u32, eval: &F) -> (Vec2, f32)
where
    F: Fn(f32) -> Option<(Vec2, f32)>,
{
    let mut hit = eval(hi_t);
    for _ in 0..iterations {
        let mid_t = (lo_t + hi_t) * 0.5;
        match eval(mid_t) {
            Some((uv, diff)) => {
                if diff > 0.0 {
                    hi_t = mid_t;
                    hit = Some((uv, diff));
                } else {
                    lo_t = mid_t;
                }
            }
            None => break,
        }
    }
    let uv = hit.map_or(Vec2::ZERO, |(uv, _)| uv);
    (uv, hi_t)
}

/// Folds the fade terms into an [`SsrHit`] for a bracketed, refined crossing.
#[must_use]
fn finalize_hit(
    uv: Vec2,
    hit_distance: f32,
    reflect_dir: Vec3,
    view_dir: Vec3,
    params: &SsrParams,
) -> SsrHit {
    let edge = screen_edge_fade(uv, params.edge_fade);
    let graze = grazing_fade(reflect_dir, view_dir, params.grazing_bias);
    let dist = distance_fade(hit_distance, params.max_distance);
    SsrHit {
        uv,
        confidence: clamp01(edge * graze * dist),
        hit_distance,
        hit: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the `f32` assertions in this module's tests.
    const CMP_EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// A flat wall at constant `view`-space depth `w`, visible only where the
    /// `UV` stays inside the `0..=1` screen box.
    fn flat_wall(w: f32) -> impl Fn(Vec2) -> Option<f32> {
        move |uv: Vec2| {
            if (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y) {
                Some(w)
            } else {
                None
            }
        }
    }

    /// Generous default parameters: many steps, a wide search, a small thickness
    /// window, and moderate refinement.
    fn base_params() -> SsrParams {
        SsrParams::new(64, 8.0, 0.5, 24, 0.1, 0.25)
    }

    // ----- reflect vector identities -------------------------------------

    #[test]
    fn reflect_flips_normal_component() {
        // Incident along +z, normal along -z: the ray bounces straight back.
        let v = Vec3::new(0.0, 0.0, 1.0);
        let n = Vec3::new(0.0, 0.0, -1.0);
        let r = reflect(v, n);
        assert!(approx_eq(r.x, 0.0));
        assert!(approx_eq(r.y, 0.0));
        assert!(approx_eq(r.z, -1.0));
    }

    #[test]
    fn reflect_incidence_equals_reflection() {
        // The normal-aligned component flips sign: dot(r, n) == -dot(v, n).
        let v = Vec3::new(0.3, -0.7, 0.9);
        let n = Vec3::new(0.2, 0.5, -0.4).normalize();
        let r = reflect(v, n);
        assert!(approx(r.dot(n), -v.dot(n), 1e-5));
    }

    #[test]
    fn reflect_preserves_tangential_component() {
        // Removing the normal-aligned part leaves the same tangential vector.
        let v = Vec3::new(1.0, 2.0, -0.5);
        let n = Vec3::new(-0.3, 0.8, 0.1).normalize();
        let r = reflect(v, n);
        let tangent_v = v.minus(n.scale(v.dot(n)));
        let tangent_r = r.minus(n.scale(r.dot(n)));
        assert!(approx(tangent_v.x, tangent_r.x, 1e-5));
        assert!(approx(tangent_v.y, tangent_r.y, 1e-5));
        assert!(approx(tangent_v.z, tangent_r.z, 1e-5));
    }

    #[test]
    fn reflect_preserves_length_for_unit_normal() {
        let v = Vec3::new(0.6, -1.3, 2.1);
        let n = Vec3::new(0.5, 0.5, 0.8).normalize();
        let r = reflect(v, n);
        assert!(approx(r.length(), v.length(), 1e-4));
    }

    #[test]
    fn reflect_twice_is_identity_for_unit_normal() {
        let v = Vec3::new(0.9, 0.2, -1.1);
        let n = Vec3::new(0.1, -0.4, 0.9).normalize();
        let r2 = reflect(reflect(v, n), n);
        assert!(approx(r2.x, v.x, 1e-5));
        assert!(approx(r2.y, v.y, 1e-5));
        assert!(approx(r2.z, v.z, 1e-5));
    }

    // ----- vector helpers -------------------------------------------------

    #[test]
    fn vector_dot_and_length_are_consistent() {
        let a = Vec3::new(3.0, 0.0, 4.0);
        assert!(approx_eq(a.dot(a), 25.0));
        assert!(approx_eq(a.length(), 5.0));
    }

    #[test]
    fn normalize_yields_unit_length_and_guards_zero() {
        let a = Vec3::new(0.0, 0.0, 2.0).normalize();
        assert!(approx_eq(a.length(), 1.0));
        // A near-zero vector is returned unchanged rather than dividing by zero.
        let z = Vec3::new(0.0, 0.0, 0.0).normalize();
        assert!(approx_eq(z.length(), 0.0));
    }

    // ----- projection -----------------------------------------------------

    #[test]
    fn projection_maps_axis_point_to_screen_centre() {
        let proj = Projection::new(1.0, 1.0);
        let uv = proj.project(Vec3::new(0.0, 0.0, 5.0)).expect("in front");
        assert!(approx_eq(uv.x, 0.5));
        assert!(approx_eq(uv.y, 0.5));
    }

    #[test]
    fn projection_rejects_points_at_or_behind_the_pinhole() {
        let proj = Projection::new(1.0, 1.0);
        assert!(proj.project(Vec3::new(0.2, 0.1, -1.0)).is_none());
        assert!(proj.project(Vec3::new(0.2, 0.1, 0.0)).is_none());
    }

    // ----- ray-march hit --------------------------------------------------

    #[test]
    fn planar_hit_recovers_the_projected_crossing_uv() {
        let params = base_params();
        let proj = Projection::new(1.0, 1.0);
        let origin = Vec3::new(0.0, 0.0, 2.0);
        // A steep normal (mostly lateral, slightly toward the camera) bends the
        // forward view ray into a forward + lateral reflected ray.
        let normal = Vec3::new(0.6, 0.0, -0.4).normalize();
        let wall_z = 3.0;
        let hit = params.trace(proj, origin, normal, flat_wall(wall_z));
        assert!(hit.hit);

        // Analytic crossing: the flat wall sits at wall_z, so the ray hits when
        // its own z reaches wall_z. Reconstruct the expected UV independently.
        let view_dir = origin.normalize();
        let reflect_dir = reflect(view_dir, normal).normalize();
        let t_star = (wall_z - origin.z) / reflect_dir.z;
        let expected = proj
            .project(origin.plus(reflect_dir.scale(t_star)))
            .expect("crossing is on-screen");
        assert!(approx(hit.uv.x, expected.x, 1e-3));
        assert!(approx(hit.uv.y, expected.y, 1e-3));
        assert!((0.0..=1.0).contains(&hit.confidence));
    }

    #[test]
    fn thickness_window_rejects_penetration() {
        let proj = Projection::new(1.0, 1.0);
        let origin = Vec3::new(0.0, 0.0, 2.0);
        let normal = Vec3::new(0.6, 0.0, -0.4).normalize();
        // A wall nearer than the origin: the ray starts behind it and marches
        // farther away, so it never crosses front-to-back -> penetration, miss.
        let near_wall = flat_wall(1.0);
        let hit = base_params().trace(proj, origin, normal, near_wall);
        assert!(!hit.hit);
        assert!(approx_eq(hit.confidence, 0.0));
    }

    #[test]
    fn coarse_overshoot_beyond_thickness_is_rejected() {
        // Few steps + a tiny thickness make the first behind-sample overshoot
        // the surface by more than the window allows -> rejected as tunneling.
        let coarse = SsrParams::new(3, 8.0, 0.01, 0, 0.1, 0.25);
        let proj = Projection::new(1.0, 1.0);
        let origin = Vec3::new(0.0, 0.0, 2.0);
        let normal = Vec3::new(0.6, 0.0, -0.4).normalize();
        let hit = coarse.trace(proj, origin, normal, flat_wall(3.0));
        assert!(!hit.hit);
        // The same geometry with a fine march and wide window does hit.
        let fine = SsrParams::new(256, 8.0, 0.5, 16, 0.1, 0.25);
        assert!(fine.trace(proj, origin, normal, flat_wall(3.0)).hit);
    }

    #[test]
    fn secant_refine_converges_closer_to_the_surface() {
        let proj = Projection::new(1.0, 1.0);
        let origin = Vec3::new(0.0, 0.0, 2.0);
        let normal = Vec3::new(0.6, 0.0, -0.4).normalize();
        let wall_z = 3.0;
        let view_dir = origin.normalize();
        let reflect_dir = reflect(view_dir, normal).normalize();
        let t_star = (wall_z - origin.z) / reflect_dir.z;

        let coarse = SsrParams::new(16, 8.0, 2.0, 0, 0.1, 0.25);
        let refined = SsrParams::new(16, 8.0, 2.0, 30, 0.1, 0.25);
        let hc = coarse.trace(proj, origin, normal, flat_wall(wall_z));
        let hr = refined.trace(proj, origin, normal, flat_wall(wall_z));
        assert!(hc.hit && hr.hit);
        let err_coarse = (hc.hit_distance - t_star).abs();
        let err_refined = (hr.hit_distance - t_star).abs();
        assert!(err_refined < err_coarse);
        assert!(err_refined < 1e-3);
    }

    #[test]
    fn exhausted_march_without_crossing_reports_miss() {
        // The wall lies well beyond the search radius, so the ray never reaches
        // it before the step budget runs out.
        let params = SsrParams::new(8, 1.0, 0.5, 8, 0.1, 0.25);
        let proj = Projection::new(1.0, 1.0);
        let origin = Vec3::new(0.0, 0.0, 2.0);
        let normal = Vec3::new(0.6, 0.0, -0.4).normalize();
        let hit = params.trace(proj, origin, normal, flat_wall(50.0));
        assert!(!hit.hit);
        assert!(approx_eq(hit.confidence, 0.0));
    }

    #[test]
    fn zero_steps_reports_miss() {
        let params = SsrParams::new(0, 8.0, 0.5, 8, 0.1, 0.25);
        let proj = Projection::new(1.0, 1.0);
        let origin = Vec3::new(0.0, 0.0, 2.0);
        let normal = Vec3::new(0.6, 0.0, -0.4).normalize();
        assert!(!params.trace(proj, origin, normal, flat_wall(3.0)).hit);
    }

    // ----- fades ----------------------------------------------------------

    #[test]
    fn screen_edge_fade_vanishes_at_the_border() {
        let edge = 0.1;
        assert!(approx_eq(screen_edge_fade(Vec2::new(0.0, 0.5), edge), 0.0));
        assert!(approx_eq(screen_edge_fade(Vec2::new(1.0, 0.5), edge), 0.0));
        assert!(approx_eq(screen_edge_fade(Vec2::new(0.5, 0.0), edge), 0.0));
    }

    #[test]
    fn screen_edge_fade_is_full_in_the_interior() {
        assert!(approx_eq(screen_edge_fade(Vec2::new(0.5, 0.5), 0.1), 1.0));
        // A disabled ramp is a hard inside/outside test.
        assert!(approx_eq(screen_edge_fade(Vec2::new(0.5, 0.5), 0.0), 1.0));
        assert!(approx_eq(screen_edge_fade(Vec2::new(1.5, 0.5), 0.0), 0.0));
    }

    #[test]
    fn grazing_fade_drops_for_backward_rays() {
        let view_dir = Vec3::new(0.0, 0.0, 1.0);
        // A reflected ray heading back toward the camera samples off-screen.
        let backward = Vec3::new(0.0, 0.0, -1.0);
        assert!(approx_eq(grazing_fade(backward, view_dir, 0.25), 0.0));
    }

    #[test]
    fn grazing_fade_is_high_for_forward_rays() {
        let view_dir = Vec3::new(0.0, 0.0, 1.0);
        let forward = Vec3::new(0.2, 0.0, 0.98).normalize();
        assert!(grazing_fade(forward, view_dir, 0.25) > 0.9);
    }

    #[test]
    fn distance_fade_prefers_near_hits() {
        assert!(distance_fade(0.2, 1.0) > distance_fade(0.8, 1.0));
        assert!(approx_eq(distance_fade(0.0, 1.0), 1.0));
        assert!(approx_eq(distance_fade(1.0, 1.0), 0.0));
        // A degenerate search radius yields no confidence.
        assert!(approx_eq(distance_fade(0.5, 0.0), 0.0));
    }

    #[test]
    fn smoothstep_endpoints_and_degenerate_span() {
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.0), 0.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 1.0), 1.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.5), 0.5));
        assert!(approx_eq(smoothstep(2.0, 2.0, 1.9), 0.0));
        assert!(approx_eq(smoothstep(2.0, 2.0, 2.0), 1.0));
    }

    #[test]
    fn confidence_stays_in_the_unit_interval() {
        let params = base_params();
        let proj = Projection::new(1.0, 1.0);
        let normal = Vec3::new(0.6, 0.0, -0.4).normalize();
        let mut z = 2.5;
        while z <= 5.0 {
            let hit = params.trace(proj, Vec3::new(0.0, 0.0, 2.0), normal, flat_wall(z));
            assert!((0.0..=1.0).contains(&hit.confidence));
            z += 0.25;
        }
    }

    // ----- std430 packing -------------------------------------------------

    #[test]
    fn ssr_params_std430_layout_and_bytes() {
        assert_eq!(SSR_PARAMS_STRIDE, 32);
        let params = SsrParams::new(32, 4.5, 0.25, 12, 0.15, 0.5);
        let packed = params.to_std430();
        assert_eq!(packed[0], 32);
        assert_eq!(packed[1], 4.5f32.to_bits());
        assert_eq!(packed[2], 0.25f32.to_bits());
        assert_eq!(packed[3], 12);
        assert_eq!(packed[4], 0.15f32.to_bits());
        assert_eq!(packed[5], 0.5f32.to_bits());
        assert_eq!(packed[6], 0);
        assert_eq!(packed[7], 0);

        assert_eq!(ssr_params_buffer_bytes(3), 96);
        assert_eq!(ssr_params_buffer_bytes(0), SSR_PARAMS_STRIDE);
    }

    #[test]
    fn ssr_hit_std430_layout_and_bytes() {
        assert_eq!(SSR_HIT_STRIDE, 32);
        let hit = SsrHit {
            uv: Vec2::new(0.75, 0.25),
            confidence: 0.5,
            hit_distance: 2.5,
            hit: true,
        };
        let packed = hit.to_std430();
        assert_eq!(packed[0], 0.75f32.to_bits());
        assert_eq!(packed[1], 0.25f32.to_bits());
        assert_eq!(packed[2], 0.5f32.to_bits());
        assert_eq!(packed[3], 2.5f32.to_bits());
        assert_eq!(packed[4], 1);
        assert_eq!(packed[5], 0);
        assert_eq!(packed[6], 0);
        assert_eq!(packed[7], 0);

        // A miss packs a zero flag.
        assert_eq!(SsrHit::miss().to_std430()[4], 0);
        assert_eq!(ssr_hit_buffer_bytes(4), 128);
        assert_eq!(ssr_hit_buffer_bytes(0), SSR_HIT_STRIDE);
    }

    #[test]
    fn march_schedule_is_uniform_and_bounded() {
        let params = SsrParams::new(4, 8.0, 0.5, 8, 0.1, 0.25);
        let schedule = march_schedule(&params);
        assert_eq!(schedule.len(), 4);
        for pair in schedule.windows(2) {
            assert!(approx_eq(pair[1] - pair[0], 2.0));
        }
        assert!(approx_eq(schedule[3], 8.0));
        assert!(march_schedule(&SsrParams::new(0, 8.0, 0.5, 8, 0.1, 0.25)).is_empty());
    }

    #[test]
    fn dispatch_groups_round_up() {
        assert_eq!(dispatch_groups(0, 64), 0);
        assert_eq!(dispatch_groups(1, 64), 1);
        assert_eq!(dispatch_groups(64, 64), 1);
        assert_eq!(dispatch_groups(65, 64), 2);
        assert_eq!(dispatch_groups(10, 0), 10);
    }
}
