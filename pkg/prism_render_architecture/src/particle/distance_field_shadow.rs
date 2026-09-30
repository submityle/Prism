//! World-space signed-distance-field (`SDF`) soft shadows: the
//! `CPU`-verifiable reference for the sphere-tracing cone soft shadow that
//! grounds particles against baked scene distance fields, aligned with the
//! `UE` / `Frostbite` distance-field soft-shadow model (design §16-§21).
//!
//! A distant emitter cannot afford a per-light shadow-map render for every
//! particle, so production engines bake the static scene into a signed-distance
//! field and *sphere trace* the shadow ray through it. Starting a short
//! distance off the receiver, the march repeatedly samples the nearest surface
//! distance `d` at the current point and advances by exactly `d` (the largest
//! step guaranteed not to tunnel through geometry). The penumbra falls out for
//! free: the ratio `k * d / t` of the closest approach `d` to the travelled
//! distance `t` is an angular cone half-width, and the running minimum of that
//! ratio over the whole march is the soft-shadow visibility. A ray that grazes
//! an occluder returns a soft grey; a ray that hits one returns black; a ray
//! that stays clear returns white. This is the classic Inigo-Quilez cone
//! formulation that `UE`'s distance-field shadows and `Frostbite`'s global
//! distance field both build on, reproduced here without any of their code.
//!
//! The pieces are: (1) [`SdfGrid`], a baked scalar distance field with
//! trilinear reconstruction, so the march has a concrete, testable distance
//! oracle (any `impl Fn(point) -> f32` also works, e.g. an analytic primitive);
//! (2) [`sphere_trace_shadow`], the cone soft-shadow march itself, plus a
//! dithered wrapper [`sphere_trace_shadow_dithered`] whose sub-step offset comes
//! from a self-contained integer hash; (3) [`ShadowCascades`], the
//! receiver-distance cascade (`LOD`) selector that returns a cascade index and a
//! normalized cross-fade weight so two cascades blend seamlessly at their
//! boundary, together with [`mip_lod_for_distance`] which picks a coarser `SDF`
//! mip for far receivers via an exponent-bits `log2`; and (4)
//! [`DistanceFieldShadowParams`] / [`CascadeSelection`], the `std430`-friendly
//! packings whose `vec4` / `vec2` alignment follows the shared strides from
//! [`super::gpu_layout`].
//!
//! Deliberately out of scope, and owned by siblings this file never touches:
//! screen-space contact shadows (the short-range `SSCS` march in
//! [`super::contact_shadow`]), hierarchical-Z occlusion *culling* (the
//! `HzbPyramid` in [`super::occlusion`]), and soft-particle depth fade
//! ([`super::soft_particle`]). This module does world-space `SDF` sphere-tracing
//! soft shadows and cascade selection and nothing else, and imports no sibling
//! particle module beyond [`super::gpu_layout`].
//!
//! Only `f32::sqrt` / `f32::floor`, integer arithmetic (including `div_ceil`),
//! and integer hashing are used — no transcendental functions (`sin` / `cos` /
//! `exp` / `ln` / `powf`), and `log2` is read straight from the `f32` exponent
//! bits — so a future `GPU` kernel that samples the same field in the same order
//! reproduces this reference bit for bit.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC2_STRIDE, VEC4_STRIDE};

/// Absolute tolerance for the `f32` comparison guards in this module.
///
/// The crate bans `==` / `!=` on floating point, so degenerate-extent
/// detection, `smoothstep` interval collapse, and near-zero direction / step
/// guards all compare a magnitude against this epsilon instead.
const CMP_EPS: f32 = 1e-6;

/// `2^32` as an `f32`: the normalizing span turning a `u32` hash word into a
/// unit-interval fraction.
const U32_SPAN: f32 = 4_294_967_296.0;

/// Byte stride of one [`DistanceFieldShadowParams`] record in a `std430`
/// storage buffer.
///
/// The five scalars pack into two `vec4` slots: `vec4(max_steps, softness_k,
/// min_t, max_t)` followed by `vec4(surface_eps, pad, pad, pad)`.
pub const DISTANCE_FIELD_SHADOW_PARAMS_STRIDE: usize = 2 * VEC4_STRIDE;

/// Byte stride of one packed [`CascadeSelection`] record (`vec2(index, blend)`)
/// in a `std430` storage buffer.
pub const CASCADE_SELECTION_STRIDE: usize = VEC2_STRIDE;

/// A 3D vector as a bare `[f32; 3]`; the module hand-rolls the little math it
/// needs rather than depending on any shared vector type.
pub type Vec3 = [f32; 3];

/// Clamps a scalar into the `0..=1` range without branching on equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Component-wise sum of two vectors.
#[must_use]
fn add3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales a vector by a scalar.
#[must_use]
fn scale3(a: Vec3, s: f32) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product of two vectors.
#[must_use]
fn dot3(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean length of a vector (the only `sqrt` in the module).
#[must_use]
fn length3(a: Vec3) -> f32 {
    dot3(a, a).sqrt()
}

/// Returns the unit-length direction, or the zero vector when the input is
/// shorter than [`CMP_EPS`] (so a degenerate direction can never divide by zero
/// or produce a `NaN`).
#[must_use]
fn normalize_or_zero(a: Vec3) -> Vec3 {
    let len = length3(a);
    if len < CMP_EPS {
        [0.0, 0.0, 0.0]
    } else {
        scale3(a, 1.0 / len)
    }
}

/// Linear interpolation `a + (b - a) * t`.
#[must_use]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Hermite `smoothstep` from `edge0` to `edge1` evaluated at `x`.
///
/// Returns `0.0` at or below `edge0`, `1.0` at or above `edge1`, and the
/// `t * t * (3 - 2 * t)` interpolation in between. A degenerate (near-equal)
/// interval collapses to a hard step at `edge1` rather than dividing by zero.
#[must_use]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span < CMP_EPS {
        return if x < edge1 { 0.0 } else { 1.0 };
    }
    let t = clamp01((x - edge0) / span);
    t * t * (3.0 - 2.0 * t)
}

/// Integer avalanche hash mixing a `u32` seed into a well-distributed `u32`.
///
/// A self-contained mixer (no dependency on any noise or determinism module):
/// xor-shifts and odd-constant multiplies give a deterministic,
/// platform-independent word.
#[must_use]
fn hash_u32(seed: u32) -> u32 {
    let mut x = seed;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Normalizes a `u32` hash word into a `0..1` unit fraction.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "a hash word maps to the nearest representable unit fraction; exactness is not required for dither"
)]
fn to_unit(word: u32) -> f32 {
    (word as f32) / U32_SPAN
}

/// Deterministic sub-step jitter offset in `[0, 1)` for a per-ray `seed`.
///
/// The march always starts at the same `min_t`, so a whole tile of rays would
/// otherwise sample identical positions and leave stair-stepped penumbra edges.
/// Offsetting each ray's start by a hashed fraction of one step decorrelates
/// neighbours while staying bit-reproducible for a given `seed`.
#[must_use]
pub fn jitter_offset(seed: u32) -> f32 {
    to_unit(hash_u32(seed))
}

/// Reads `floor(log2(x))` straight from the `f32` exponent bits.
///
/// For a normalized positive `x` the biased exponent minus `127` is exactly the
/// floor of the base-two logarithm (`x` in `[1, 2)` yields `0`, `[2, 4)` yields
/// `1`, and so on). Inputs at or below zero, or below one, yield `0`; this
/// avoids the banned `ln` / `log2` transcendental calls entirely.
#[must_use]
fn log2_floor_from_bits(x: f32) -> u32 {
    if x < 1.0 {
        return 0;
    }
    let biased = (x.to_bits() >> 23) & 0xff;
    let exponent = i32::try_from(biased).unwrap_or(127) - 127;
    u32::try_from(exponent).unwrap_or(0)
}

/// Picks the `SDF` mip `LOD` for a receiver at `receiver_dist`, given the
/// distance `base_dist` covered by mip `0`.
///
/// Far receivers can be shadowed from a coarser (lower-resolution) `SDF` mip
/// without visible error, so the level grows as `floor(log2(receiver_dist /
/// base_dist))` — one extra mip per doubling of distance. A non-positive
/// `base_dist` or a receiver inside the mip-`0` range yields level `0`.
#[must_use]
pub fn mip_lod_for_distance(receiver_dist: f32, base_dist: f32) -> u32 {
    if base_dist < CMP_EPS {
        return 0;
    }
    let ratio = receiver_dist / base_dist;
    if ratio <= 1.0 {
        return 0;
    }
    log2_floor_from_bits(ratio)
}

/// Number of `GPU` workgroups needed to cover `ray_count` invocations at
/// `group_size` threads each, via integer `div_ceil` (never zero-sized).
#[must_use]
pub fn dispatch_groups(ray_count: u32, group_size: u32) -> u32 {
    ray_count.div_ceil(group_size.max(1))
}

/// Casts a small grid dimension to `f32` for coordinate math.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "SDF grid dimensions are far below 2^24, so the conversion is exact"
)]
fn dim_to_f32(d: usize) -> f32 {
    d as f32
}

/// Casts a pre-clamped, floored, non-negative coordinate to a grid index.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "input is clamped to [0, dim - 1] and floored, so it is a valid non-negative index"
)]
fn floor_to_index(x: f32) -> usize {
    x as usize
}

/// A baked scalar signed-distance field sampled with trilinear reconstruction.
///
/// The grid is stored row-major (`X`-fastest) over an axis-aligned box: texel
/// `(i, j, k)` lives at `data[i + j * nx + k * nx * ny]` and its distance is
/// negative inside geometry, positive outside. World points map into grid
/// coordinates through the `[min, max]` box (the `UE` / `Niagara` volume-bounds
/// convention) and clamp to the border, so a sample outside the box reads the
/// nearest face rather than reading out of bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct SdfGrid {
    dims: [usize; 3],
    min: Vec3,
    max: Vec3,
    data: Vec<f32>,
}

impl SdfGrid {
    /// Builds a grid from its dimensions, world-space bounds, and row-major
    /// distance samples.
    ///
    /// Returns `None` when any dimension is zero or when `data.len()` does not
    /// equal `nx * ny * nz` (checked with saturating-free integer arithmetic),
    /// so a malformed field can never be sampled.
    #[must_use]
    pub fn new(dims: [usize; 3], min: Vec3, max: Vec3, data: Vec<f32>) -> Option<Self> {
        if dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
            return None;
        }
        let expected = dims[0].checked_mul(dims[1])?.checked_mul(dims[2])?;
        if data.len() != expected {
            return None;
        }
        Some(Self {
            dims,
            min,
            max,
            data,
        })
    }

    /// The grid dimensions `[nx, ny, nz]`.
    #[must_use]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Reads the raw distance at integer texel `(x, y, z)`; callers guarantee
    /// the indices are in range.
    #[must_use]
    fn at(&self, x: usize, y: usize, z: usize) -> f32 {
        let idx = x + y * self.dims[0] + z * self.dims[0] * self.dims[1];
        self.data[idx]
    }

    /// Trilinearly samples the signed distance at world point `p`.
    ///
    /// The point is mapped into grid space, clamped to `[0, dim - 1]` on each
    /// axis (border clamp), split into an integer base cell and a fractional
    /// weight, and reconstructed with the standard eight-corner trilinear blend.
    /// A degenerate axis whose `[min, max]` extent is below [`CMP_EPS`] collapses
    /// to grid coordinate zero instead of dividing by zero.
    #[must_use]
    pub fn sample(&self, p: Vec3) -> f32 {
        let mut base = [0usize; 3];
        let mut frac = [0.0f32; 3];
        for axis in 0..3 {
            let last = self.dims[axis] - 1;
            let last_f = dim_to_f32(last);
            let extent = self.max[axis] - self.min[axis];
            let coord = if extent < CMP_EPS {
                0.0
            } else {
                ((p[axis] - self.min[axis]) / extent) * last_f
            };
            let clamped = coord.clamp(0.0, last_f);
            let floored = clamped.floor();
            let idx = floor_to_index(floored).min(last);
            base[axis] = idx;
            frac[axis] = clamped - floored;
        }
        let x0 = base[0];
        let y0 = base[1];
        let z0 = base[2];
        let x1 = (x0 + 1).min(self.dims[0] - 1);
        let y1 = (y0 + 1).min(self.dims[1] - 1);
        let z1 = (z0 + 1).min(self.dims[2] - 1);

        let c00 = lerp(self.at(x0, y0, z0), self.at(x1, y0, z0), frac[0]);
        let c10 = lerp(self.at(x0, y1, z0), self.at(x1, y1, z0), frac[0]);
        let c01 = lerp(self.at(x0, y0, z1), self.at(x1, y0, z1), frac[0]);
        let c11 = lerp(self.at(x0, y1, z1), self.at(x1, y1, z1), frac[0]);

        let c0 = lerp(c00, c10, frac[1]);
        let c1 = lerp(c01, c11, frac[1]);
        lerp(c0, c1, frac[2])
    }
}

/// Tunables for the `SDF` sphere-tracing cone soft shadow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceFieldShadowParams {
    /// Maximum number of sphere-trace iterations before the march gives up.
    pub max_steps: u32,
    /// Cone-softness factor `k`: larger values narrow the penumbra toward a hard
    /// shadow, smaller values widen it.
    pub softness_k: f32,
    /// Distance along the ray at which the march starts, lifting it off the
    /// receiver to avoid self-shadow acne.
    pub min_t: f32,
    /// Distance along the ray at which the march stops (the light's reach).
    pub max_t: f32,
    /// Surface threshold: a sampled distance below this counts as a hit (fully
    /// shadowed) and also floors each step so the march always makes progress.
    pub surface_eps: f32,
}

impl DistanceFieldShadowParams {
    /// Creates sphere-trace parameters from all fields.
    #[must_use]
    pub const fn new(
        max_steps: u32,
        softness_k: f32,
        min_t: f32,
        max_t: f32,
        surface_eps: f32,
    ) -> Self {
        Self {
            max_steps,
            softness_k,
            min_t,
            max_t,
            surface_eps,
        }
    }

    /// Packs the parameters into their `std430` `vec4`-aligned word layout.
    ///
    /// Layout: `[max_steps, softness_k, min_t, max_t, surface_eps, pad, pad,
    /// pad]` as raw `u32` words (the four `f32` fields via `f32::to_bits`) — two
    /// `vec4` slots, matching [`DISTANCE_FIELD_SHADOW_PARAMS_STRIDE`]. The
    /// trailing words are padding.
    #[must_use]
    pub fn to_std430(&self) -> [u32; 8] {
        [
            self.max_steps,
            self.softness_k.to_bits(),
            self.min_t.to_bits(),
            self.max_t.to_bits(),
            self.surface_eps.to_bits(),
            0,
            0,
            0,
        ]
    }
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`DistanceFieldShadowParams`] records, via the shared clamp-to-one-element
/// rule from [`storage_bytes`].
#[must_use]
pub fn distance_field_shadow_params_buffer_bytes(count: usize) -> usize {
    storage_bytes(DISTANCE_FIELD_SHADOW_PARAMS_STRIDE, count)
}

/// Total byte size of a `std430` storage buffer holding `count` raw cascade
/// split distances (one `f32` word each), via [`storage_bytes`].
#[must_use]
pub fn cascade_split_buffer_bytes(count: usize) -> usize {
    storage_bytes(U32_STRIDE, count)
}

/// Marches a shadow ray through an `SDF` and returns the cone soft-shadow
/// visibility in `0..=1`.
///
/// `sample` is the distance oracle (an [`SdfGrid`] sampler or any analytic
/// `impl Fn(point) -> f32`). Starting at `t = min_t` from `origin` along the
/// normalized `dir`, each step reads the nearest surface distance `d` at the
/// current point. A `d` below `surface_eps` means the ray reached geometry, so
/// the receiver is fully shadowed and the function returns `0.0`. Otherwise the
/// step folds the cone ratio `softness_k * d / t` into a running minimum — the
/// closest angular approach to any occluder — and advances by `d` (floored at
/// `surface_eps` so the march always progresses). The march ends at `max_t` or
/// after `max_steps` iterations, returning the accumulated minimum clamped to
/// `0..=1`.
///
/// The result follows the convention **`1.0` = fully lit / unoccluded** and
/// **`0.0` = fully shadowed**, with soft grey values in the penumbra. A
/// degenerate direction shorter than [`CMP_EPS`], a `max_steps` of zero, or a
/// `min_t` already past `max_t` all leave the receiver fully lit (`1.0`).
#[must_use]
pub fn sphere_trace_shadow(
    sample: impl Fn(Vec3) -> f32,
    origin: Vec3,
    dir: Vec3,
    params: &DistanceFieldShadowParams,
) -> f32 {
    let unit = normalize_or_zero(dir);
    if length3(unit) < CMP_EPS {
        return 1.0;
    }
    let step_floor = params.surface_eps.max(CMP_EPS);
    let mut visibility = 1.0f32;
    let mut t = params.min_t.max(step_floor);
    let steps = usize::try_from(params.max_steps).unwrap_or(usize::MAX);
    for _ in 0..steps {
        if t >= params.max_t {
            break;
        }
        let point = add3(origin, scale3(unit, t));
        let dist = sample(point);
        if dist < params.surface_eps {
            return 0.0;
        }
        let cone = params.softness_k * dist / t;
        visibility = visibility.min(cone);
        t += dist.max(step_floor);
    }
    clamp01(visibility)
}

/// Dithered [`sphere_trace_shadow`]: offsets the ray's start distance by a
/// hashed sub-step fraction (see [`jitter_offset`]) so a tile of rays does not
/// share the exact same sampling positions and band the penumbra.
///
/// The offset is a fraction of `min_t`, so it never pushes the start past
/// `max_t`; every other guarantee of [`sphere_trace_shadow`] is preserved.
#[must_use]
pub fn sphere_trace_shadow_dithered(
    sample: impl Fn(Vec3) -> f32,
    origin: Vec3,
    dir: Vec3,
    params: &DistanceFieldShadowParams,
    seed: u32,
) -> f32 {
    let mut jittered = *params;
    jittered.min_t = params.min_t + jitter_offset(seed) * params.min_t;
    sphere_trace_shadow(sample, origin, dir, &jittered)
}

/// The cascade a receiver landed in, plus a normalized cross-fade to the next
/// cascade.
///
/// `index` is the selected cascade and `next_index` the one it fades toward
/// (equal to `index` for the last cascade). `blend` in `0..=1` is the weight of
/// `next_index`; the primary weight is `1 - blend`, so the two always sum to
/// exactly one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CascadeSelection {
    /// Index of the selected cascade.
    pub index: usize,
    /// Index of the cascade being cross-faded toward.
    pub next_index: usize,
    /// Cross-fade weight of `next_index` in `0..=1`.
    pub blend: f32,
}

impl CascadeSelection {
    /// Weight of the primary cascade (`1 - blend`).
    #[must_use]
    pub fn primary_weight(&self) -> f32 {
        1.0 - self.blend
    }

    /// Weight of the next cascade (`blend`).
    #[must_use]
    pub fn next_weight(&self) -> f32 {
        self.blend
    }

    /// Packs the selection into a `std430` `vec2` word pair `[index, blend]`,
    /// the index as a `u32` and the blend via `f32::to_bits`, matching
    /// [`CASCADE_SELECTION_STRIDE`].
    #[must_use]
    pub fn to_std430(&self) -> [u32; 2] {
        let index = u32::try_from(self.index).unwrap_or(u32::MAX);
        [index, self.blend.to_bits()]
    }
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`CascadeSelection`] records, via [`storage_bytes`].
#[must_use]
pub fn cascade_selection_buffer_bytes(count: usize) -> usize {
    storage_bytes(CASCADE_SELECTION_STRIDE, count)
}

/// A cascaded shadow ladder selected by receiver distance (the distance-field
/// analogue of cascaded shadow maps).
///
/// Each cascade covers out to an ascending *far distance*; a receiver picks the
/// first cascade whose far distance still contains it. Near each cascade's far
/// edge a blend band cross-fades into the next cascade so the transition is not
/// a visible seam. The bands are expressed as a fraction of each cascade's own
/// span, so a wide far cascade blends over a proportionally wider region.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowCascades {
    far_distances: Vec<f32>,
    blend_fraction: f32,
}

impl ShadowCascades {
    /// Builds a cascade ladder from ascending `far_distances` and a
    /// `blend_fraction` (the share of each cascade's span used to cross-fade at
    /// its far edge, clamped to `0..=1`).
    #[must_use]
    pub fn new(far_distances: Vec<f32>, blend_fraction: f32) -> Self {
        Self {
            far_distances,
            blend_fraction: clamp01(blend_fraction),
        }
    }

    /// Number of cascades in the ladder.
    #[must_use]
    pub fn count(&self) -> usize {
        self.far_distances.len()
    }

    /// Selects the cascade for a receiver at `receiver_dist` and the normalized
    /// cross-fade toward the next cascade.
    ///
    /// The chosen `index` is the first cascade whose far distance is at least
    /// `receiver_dist`, or the last cascade when the receiver is beyond them
    /// all. Within the last `blend_fraction` of the chosen cascade's span the
    /// `blend` weight `smoothstep`s from `0` up to `1` at the far edge; the last
    /// cascade and an empty ladder never blend. The returned weights always sum
    /// to one (see [`CascadeSelection`]).
    #[must_use]
    pub fn select(&self, receiver_dist: f32) -> CascadeSelection {
        let count = self.far_distances.len();
        if count == 0 {
            return CascadeSelection {
                index: 0,
                next_index: 0,
                blend: 0.0,
            };
        }
        let mut index = count - 1;
        for (i, &far) in self.far_distances.iter().enumerate() {
            if receiver_dist <= far {
                index = i;
                break;
            }
        }
        if index + 1 >= count {
            return CascadeSelection {
                index,
                next_index: index,
                blend: 0.0,
            };
        }
        let near_edge = if index == 0 {
            0.0
        } else {
            self.far_distances[index - 1]
        };
        let far_edge = self.far_distances[index];
        let span = far_edge - near_edge;
        let band = span * self.blend_fraction;
        let blend = if band < CMP_EPS {
            0.0
        } else {
            smoothstep(far_edge - band, far_edge, receiver_dist)
        };
        CascadeSelection {
            index,
            next_index: index + 1,
            blend,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Absolute tolerance for the `f32` assertions in this module's tests.
    const TEST_EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPS
    }

    /// Analytic sphere `SDF`: signed distance from `p` to a sphere at `center`.
    fn sphere_sdf(p: Vec3, center: Vec3, radius: f32) -> f32 {
        let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - radius
    }

    fn base_params() -> DistanceFieldShadowParams {
        DistanceFieldShadowParams::new(64, 8.0, 0.05, 20.0, 1e-3)
    }

    #[test]
    fn ray_into_occluder_is_fully_shadowed() {
        // A unit sphere sits two units up the ray; marching straight at it hits.
        let params = base_params();
        let sample = |p: Vec3| sphere_sdf(p, [0.0, 0.0, 2.0], 0.5);
        let v = sphere_trace_shadow(sample, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &params);
        assert!(approx_eq(v, 0.0));
    }

    #[test]
    fn ray_clear_of_geometry_is_fully_lit() {
        // The sphere is far off to the side of a ray heading up +Z: no occlusion.
        let params = base_params();
        let sample = |p: Vec3| sphere_sdf(p, [50.0, 50.0, 2.0], 0.5);
        let v = sphere_trace_shadow(sample, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &params);
        assert!(approx_eq(v, 1.0));
    }

    #[test]
    fn grazing_ray_yields_soft_penumbra() {
        // A ray that passes just beside the sphere returns a partial grey.
        let params = base_params();
        let sample = |p: Vec3| sphere_sdf(p, [0.7, 0.0, 4.0], 0.5);
        let v = sphere_trace_shadow(sample, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &params);
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn penumbra_is_monotonic_in_softness() {
        // Larger softness_k narrows the penumbra: visibility must not decrease.
        let sample = |p: Vec3| sphere_sdf(p, [0.7, 0.0, 4.0], 0.5);
        let soft = DistanceFieldShadowParams::new(64, 2.0, 0.05, 20.0, 1e-3);
        let sharp = DistanceFieldShadowParams::new(64, 16.0, 0.05, 20.0, 1e-3);
        let vs = sphere_trace_shadow(sample, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &soft);
        let vh = sphere_trace_shadow(sample, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &sharp);
        assert!(vh >= vs - TEST_EPS);
    }

    #[test]
    fn closer_grazing_darkens_more() {
        // A ray skimming nearer the surface has a smaller closest ratio, so it
        // is at least as dark as one passing farther out.
        let params = base_params();
        let near = |p: Vec3| sphere_sdf(p, [0.6, 0.0, 4.0], 0.5);
        let far = |p: Vec3| sphere_sdf(p, [0.9, 0.0, 4.0], 0.5);
        let vn = sphere_trace_shadow(near, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &params);
        let vf = sphere_trace_shadow(far, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &params);
        assert!(vn <= vf + TEST_EPS);
    }

    #[test]
    fn zero_direction_stays_fully_lit() {
        let params = base_params();
        let sample = |p: Vec3| sphere_sdf(p, [0.0, 0.0, 2.0], 0.5);
        let v = sphere_trace_shadow(sample, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], &params);
        assert!(approx_eq(v, 1.0));
    }

    #[test]
    fn zero_steps_stays_fully_lit() {
        let params = DistanceFieldShadowParams::new(0, 8.0, 0.05, 20.0, 1e-3);
        let sample = |p: Vec3| sphere_sdf(p, [0.0, 0.0, 2.0], 0.5);
        let v = sphere_trace_shadow(sample, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &params);
        assert!(approx_eq(v, 1.0));
    }

    #[test]
    fn min_t_past_max_t_stays_fully_lit() {
        let params = DistanceFieldShadowParams::new(64, 8.0, 30.0, 20.0, 1e-3);
        let sample = |p: Vec3| sphere_sdf(p, [0.0, 0.0, 2.0], 0.5);
        let v = sphere_trace_shadow(sample, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &params);
        assert!(approx_eq(v, 1.0));
    }

    #[test]
    fn dithered_march_stays_in_range_and_hits() {
        let params = base_params();
        let sample = |p: Vec3| sphere_sdf(p, [0.0, 0.0, 2.0], 0.5);
        for seed in 0..8u32 {
            let v = sphere_trace_shadow_dithered(
                sample,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                &params,
                seed,
            );
            assert!((0.0..=1.0).contains(&v));
        }
        // A direct hit remains fully shadowed regardless of the small start jitter.
        let v =
            sphere_trace_shadow_dithered(sample, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], &params, 12345);
        assert!(approx_eq(v, 0.0));
    }

    #[test]
    fn jitter_offset_is_unit_ranged() {
        for seed in 0..64u32 {
            let j = jitter_offset(seed);
            assert!((0.0..1.0).contains(&j));
        }
    }

    #[test]
    fn cascade_selection_picks_by_distance() {
        let cascades = ShadowCascades::new(vec![10.0, 40.0, 120.0], 0.0);
        assert_eq!(cascades.count(), 3);
        assert_eq!(cascades.select(5.0).index, 0);
        assert_eq!(cascades.select(25.0).index, 1);
        assert_eq!(cascades.select(100.0).index, 2);
        // Beyond the last far distance clamps to the last cascade.
        assert_eq!(cascades.select(500.0).index, 2);
    }

    #[test]
    fn cascade_blend_weights_sum_to_one() {
        let cascades = ShadowCascades::new(vec![10.0, 40.0, 120.0], 0.25);
        for &d in &[3.0f32, 9.5, 25.0, 38.0, 100.0, 300.0] {
            let sel = cascades.select(d);
            assert!(approx_eq(sel.primary_weight() + sel.next_weight(), 1.0));
            assert!((0.0..=1.0).contains(&sel.blend));
        }
    }

    #[test]
    fn cascade_blends_toward_next_at_far_edge() {
        let cascades = ShadowCascades::new(vec![10.0, 40.0], 0.5);
        // Deep inside cascade 0: no blend.
        let inner = cascades.select(2.0);
        assert_eq!(inner.index, 0);
        assert!(approx_eq(inner.blend, 0.0));
        // At the far edge of cascade 0: fully faded into cascade 1.
        let edge = cascades.select(10.0);
        assert_eq!(edge.index, 0);
        assert_eq!(edge.next_index, 1);
        assert!(approx_eq(edge.blend, 1.0));
    }

    #[test]
    fn last_cascade_and_empty_ladder_never_blend() {
        let cascades = ShadowCascades::new(vec![10.0, 40.0], 0.5);
        let last = cascades.select(60.0);
        assert_eq!(last.index, 1);
        assert_eq!(last.next_index, 1);
        assert!(approx_eq(last.blend, 0.0));

        let empty = ShadowCascades::new(Vec::new(), 0.5);
        assert_eq!(empty.count(), 0);
        let sel = empty.select(5.0);
        assert_eq!(sel.index, 0);
        assert!(approx_eq(sel.primary_weight() + sel.next_weight(), 1.0));
    }

    #[test]
    fn params_std430_layout_is_exact() {
        let params = DistanceFieldShadowParams::new(64, 8.0, 0.05, 20.0, 1e-3);
        let words = params.to_std430();
        assert_eq!(words[0], 64);
        assert_eq!(words[1], 8.0f32.to_bits());
        assert_eq!(words[2], 0.05f32.to_bits());
        assert_eq!(words[3], 20.0f32.to_bits());
        assert_eq!(words[4], 1e-3f32.to_bits());
        assert_eq!(words[5], 0);
        assert_eq!(words[6], 0);
        assert_eq!(words[7], 0);
        assert_eq!(DISTANCE_FIELD_SHADOW_PARAMS_STRIDE, 32);
    }

    #[test]
    fn params_buffer_bytes_follow_stride() {
        assert_eq!(distance_field_shadow_params_buffer_bytes(0), 32);
        assert_eq!(distance_field_shadow_params_buffer_bytes(4), 128);
        assert_eq!(cascade_split_buffer_bytes(0), U32_STRIDE);
        assert_eq!(cascade_split_buffer_bytes(8), 32);
    }

    #[test]
    fn cascade_selection_std430_layout() {
        let sel = CascadeSelection {
            index: 2,
            next_index: 3,
            blend: 0.25,
        };
        let words = sel.to_std430();
        assert_eq!(words[0], 2);
        assert_eq!(words[1], 0.25f32.to_bits());
        assert_eq!(CASCADE_SELECTION_STRIDE, 8);
        assert_eq!(cascade_selection_buffer_bytes(0), 8);
        assert_eq!(cascade_selection_buffer_bytes(3), 24);
    }

    #[test]
    fn mip_lod_grows_by_one_per_doubling() {
        assert_eq!(mip_lod_for_distance(5.0, 10.0), 0);
        assert_eq!(mip_lod_for_distance(10.0, 10.0), 0);
        assert_eq!(mip_lod_for_distance(15.0, 10.0), 0);
        assert_eq!(mip_lod_for_distance(20.0, 10.0), 1);
        assert_eq!(mip_lod_for_distance(41.0, 10.0), 2);
        assert_eq!(mip_lod_for_distance(90.0, 10.0), 3);
        // A degenerate base yields level zero rather than dividing by zero.
        assert_eq!(mip_lod_for_distance(100.0, 0.0), 0);
    }

    #[test]
    fn dispatch_groups_round_up() {
        assert_eq!(dispatch_groups(0, 64), 0);
        assert_eq!(dispatch_groups(1, 64), 1);
        assert_eq!(dispatch_groups(64, 64), 1);
        assert_eq!(dispatch_groups(65, 64), 2);
        // A zero group size is clamped to one instead of dividing by zero.
        assert_eq!(dispatch_groups(10, 0), 10);
    }

    #[test]
    fn grid_rejects_malformed_input() {
        assert!(SdfGrid::new([0, 2, 2], [0.0; 3], [1.0; 3], vec![0.0; 0]).is_none());
        assert!(SdfGrid::new([2, 2, 2], [0.0; 3], [1.0; 3], vec![0.0; 7]).is_none());
        assert!(SdfGrid::new([2, 2, 2], [0.0; 3], [1.0; 3], vec![0.0; 8]).is_some());
    }

    #[test]
    fn grid_trilinear_hits_nodes_and_midpoints() {
        // 2x2x2 grid over the unit cube; corner distances chosen so the field is
        // a linear ramp in X (0 at x=0, 1 at x=1).
        let data = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let grid = SdfGrid::new([2, 2, 2], [0.0; 3], [1.0; 3], data).expect("valid grid");
        assert_eq!(grid.dims(), [2, 2, 2]);
        // Exact node reads.
        assert!(approx_eq(grid.sample([0.0, 0.0, 0.0]), 0.0));
        assert!(approx_eq(grid.sample([1.0, 0.0, 0.0]), 1.0));
        // Midpoint of the ramp.
        assert!(approx_eq(grid.sample([0.5, 0.5, 0.5]), 0.5));
        // Outside the box clamps to the border rather than reading out of range.
        assert!(approx_eq(grid.sample([-5.0, 0.0, 0.0]), 0.0));
        assert!(approx_eq(grid.sample([5.0, 0.0, 0.0]), 1.0));
    }

    #[test]
    fn grid_degenerate_axis_collapses_without_nan() {
        // A zero-extent X axis must not divide by zero; the sample stays finite.
        let data = vec![0.25; 8];
        let grid = SdfGrid::new([2, 2, 2], [0.0; 3], [0.0, 1.0, 1.0], data).expect("valid grid");
        let v = grid.sample([0.3, 0.5, 0.5]);
        assert!(approx_eq(v, 0.25));
    }

    #[test]
    fn grid_drives_sphere_trace() {
        // Bake a sphere SDF into a grid and confirm the march through it hits.
        let dims = [17usize, 17, 17];
        let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
        let mut data = vec![0.0f32; nx * ny * nz];
        let center = [2.0f32, 2.0, 2.0];
        let radius = 0.8f32;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let p = [
                        (i as f32) / (nx as f32 - 1.0) * 4.0,
                        (j as f32) / (ny as f32 - 1.0) * 4.0,
                        (k as f32) / (nz as f32 - 1.0) * 4.0,
                    ];
                    data[i + j * nx + k * nx * ny] = sphere_sdf(p, center, radius);
                }
            }
        }
        let grid = SdfGrid::new(dims, [0.0; 3], [4.0; 3], data).expect("valid grid");
        let params = DistanceFieldShadowParams::new(128, 8.0, 0.05, 6.0, 2e-2);
        // Ray from the origin straight at the baked sphere: shadowed.
        let hit = sphere_trace_shadow(
            |p| grid.sample(p),
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            &params,
        );
        assert!(hit < 0.5);
        // Ray parallel but far to the side: lit.
        let miss = sphere_trace_shadow(
            |p| grid.sample(p),
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            &params,
        );
        assert!(miss > 0.5);
    }

    #[test]
    fn cmp_eps_tolerance_guards_smoothstep_collapse() {
        // A blend fraction so small the band is below CMP_EPS collapses to a hard
        // step instead of dividing by a near-zero span.
        let cascades = ShadowCascades::new(vec![10.0, 40.0], 1e-9);
        let sel = cascades.select(9.9999);
        assert!(approx_eq(sel.blend, 0.0));
        assert!(approx_eq(sel.primary_weight() + sel.next_weight(), 1.0));
    }
}
