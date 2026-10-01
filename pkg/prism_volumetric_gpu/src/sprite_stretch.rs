//! `wgpu` compute twin of the motion-stretch billboard *sizing* golden
//! ([`sprite_stretch`](prism_render_architecture::particle::sprite_stretch),
//! design §15).
//!
//! The particle subsystem answers two narrow, purely geometric questions per
//! sprite: how large is a velocity-stretched quad (its anisotropic half-extents
//! and the four world-space corners oriented along motion), and how does a
//! previous-to-current *trail* quad lay out. The canonical real-time-rendering
//! form derives the stretched half-length as
//! `(base_half + speed * stretch_scale)` clamped to `[min_length, max_length]`,
//! the half-width as `base_half * width_scale`, and lays the corners out at
//! `center ± forward * half_length ± binormal * half_width`, with
//! degenerate-safe fallbacks (`+Y` forward, a deterministic perpendicular
//! binormal) so no result is ever `NaN`.
//!
//! The `CPU` golden
//! [`sprite_stretch`](prism_render_architecture::particle::sprite_stretch) owns
//! that math; [`GpuSpriteStretch`] is the on-device twin that runs one thread
//! per query and returns the same values per sprite, in input order. A single
//! query drives every portable pure function at once: the per-query
//! [`SpriteStretchResult`] carries the
//! [`StretchParams::stretched_size`](prism_render_architecture::particle::sprite_stretch::StretchParams::stretched_size)
//! half-extents, the
//! [`StretchedQuad::aspect`](prism_render_architecture::particle::sprite_stretch::StretchedQuad::aspect)
//! of those extents, the
//! [`velocity_stretched_corners`](prism_render_architecture::particle::sprite_stretch::velocity_stretched_corners)
//! quad, the
//! [`stretched_from_prev`](prism_render_architecture::particle::sprite_stretch::stretched_from_prev)
//! trail and a direct
//! [`stretched_corners`](prism_render_architecture::particle::sprite_stretch::stretched_corners)
//! call. A passing real-device parity test is therefore direct evidence the
//! ported kernel clamps the same window, scales the same width, normalizes the
//! same axes, picks the same degenerate perpendicular and lays the same corners
//! the reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! Every portable pure function is reproduced:
//!
//! * [`StretchParams::stretched_size`](prism_render_architecture::particle::sprite_stretch::StretchParams::stretched_size)
//!   — the zero-velocity isotropic fallback (`length_squared < EPS`), the
//!   `base_half + speed * stretch_scale` half-length clamped to the window and
//!   the `base_half * width_scale` half-width.
//! * [`StretchedQuad::aspect`](prism_render_architecture::particle::sprite_stretch::StretchedQuad::aspect)
//!   — `half_length / half_width`, with the isotropic `1.0` fallback when the
//!   half-width is below `EPS`.
//! * [`stretched_corners`](prism_render_architecture::particle::sprite_stretch::stretched_corners)
//!   — the normalized forward/binormal axes with their `+Y` and
//!   deterministic-perpendicular fallbacks, laid out so the centroid is exactly
//!   `center`.
//! * [`velocity_stretched_corners`](prism_render_architecture::particle::sprite_stretch::velocity_stretched_corners)
//!   — the convenience composition of `stretched_size` and `stretched_corners`
//!   with `velocity` as the forward axis.
//! * [`stretched_from_prev`](prism_render_architecture::particle::sprite_stretch::stretched_from_prev)
//!   — the trail quad whose length is the travel distance and whose near/far
//!   edges sit at `center`/`prev_position`, spread by `± half_width`.
//!
//! The private `perpendicular_to` helper is reproduced verbatim and exercised
//! through the degenerate-binormal fallbacks of the two corner builders.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `sqrt`, `+ - * /` and unsigned integer index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, `acos`, built-in
//! `smoothstep` or optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The only floating-point primitive beyond ordinary
//! arithmetic is `sqrt` (for speeds, trail lengths and axis normalization),
//! matching the reference, which also uses only `f32::sqrt`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of dots, one or more
//! `sqrt`s, clamps, scales and adds, so `CPU` and `GPU` evaluate the same
//! closed form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance on the continuous fields (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) and keeps every fixture well clear of the decision
//! boundaries (the `EPS` zero-velocity / zero-axis branch splits, the
//! clamp-window edges and the `perpendicular_to` axis-selection ties) so a
//! legal fused multiply-add cannot flip a branch.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sprite_stretch`;
//! canonical velocity-stretch billboard sizing plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::sprite_stretch::{
    StretchParams, StretchedQuad, TrailQuad,
};
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` sprite-stretch kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden
/// [`sprite_stretch`](prism_render_architecture::particle::sprite_stretch)
/// function by function; see the module documentation for the algorithm.
const SPRITE_STRETCH_WGSL: &str = r#"
// Sprite motion-stretch twin: one thread per sprite reproduces the whole
// portable sprite_stretch surface -- stretched_size, StretchedQuad::aspect,
// stretched_corners, velocity_stretched_corners and stretched_from_prev -- and
// writes every field of the combined SpriteStretchResult. It mirrors the CPU
// golden `particle::sprite_stretch` function for function, uses only the
// portable core-WGSL subset (min/max/clamp/abs/sqrt and + - * / plus unsigned
// compares, no transcendental and no built-in smoothstep), and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::sprite_stretch;
// no third-party engine source or derived code.

struct Params {
    // Number of valid sprite queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 112-byte std430 stride matching the host `GpuQuery`, grouped into
// seven 16-byte lanes.
struct Query {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    base_half: f32,
    velocity_x: f32,
    velocity_y: f32,
    velocity_z: f32,
    pad_v: f32,
    binormal_x: f32,
    binormal_y: f32,
    binormal_z: f32,
    pad_b: f32,
    prev_x: f32,
    prev_y: f32,
    prev_z: f32,
    trail_half_width: f32,
    forward_x: f32,
    forward_y: f32,
    forward_z: f32,
    pad_f: f32,
    half_length_in: f32,
    half_width_in: f32,
    stretch_scale: f32,
    min_length: f32,
    max_length: f32,
    width_scale: f32,
    qpad0: f32,
    qpad1: f32,
}

// One result. 208-byte std430 stride matching the host `GpuResult`: a scalar
// lane (two half-extents, the aspect, the trail length) followed by three
// four-corner blocks (velocity-stretched, trail, direct), each corner an
// (x, y, z, pad) 16-byte lane.
struct SpriteStretch {
    size_half_length: f32,
    size_half_width: f32,
    aspect: f32,
    trail_length: f32,
    vc0_x: f32, vc0_y: f32, vc0_z: f32, vc0_p: f32,
    vc1_x: f32, vc1_y: f32, vc1_z: f32, vc1_p: f32,
    vc2_x: f32, vc2_y: f32, vc2_z: f32, vc2_p: f32,
    vc3_x: f32, vc3_y: f32, vc3_z: f32, vc3_p: f32,
    tc0_x: f32, tc0_y: f32, tc0_z: f32, tc0_p: f32,
    tc1_x: f32, tc1_y: f32, tc1_z: f32, tc1_p: f32,
    tc2_x: f32, tc2_y: f32, tc2_z: f32, tc2_p: f32,
    tc3_x: f32, tc3_y: f32, tc3_z: f32, tc3_p: f32,
    dc0_x: f32, dc0_y: f32, dc0_z: f32, dc0_p: f32,
    dc1_x: f32, dc1_y: f32, dc1_z: f32, dc1_p: f32,
    dc2_x: f32, dc2_y: f32, dc2_z: f32, dc2_p: f32,
    dc3_x: f32, dc3_y: f32, dc3_z: f32, dc3_p: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<SpriteStretch>;

// Module tolerance for near-zero speed / axis comparisons, matching the
// reference `sprite_stretch::EPS`. Compared against a squared magnitude so a
// forbidden f32 `==`/`!=` is never used.
const EPS: f32 = 1.0e-6;

// Vec3 length threshold for normalization, matching the reference
// `particle::EPS_LEN_SQ`. Deliberately tighter than `EPS`: it guards the
// reciprocal `sqrt` in `normalize_or_zero`.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Returns the unit vector along `v`, or the zero vector when `v` is
// (numerically) zero, mirroring `Vec3::normalize_or_zero` (threshold
// `EPS_LEN_SQ`), so normalization never yields NaN.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// A deterministic unit vector perpendicular to `v`, or +X when `v` is
// (numerically) zero. Crosses `v` with whichever world axis is least aligned
// with it (smallest squared component, ties resolved to the first in x, y, z
// order by `<=`), exactly as the reference `perpendicular_to`.
fn perpendicular_to(v: vec3<f32>) -> vec3<f32> {
    let ax = v.x * v.x;
    let ay = v.y * v.y;
    let az = v.z * v.z;
    var reference = vec3<f32>(0.0, 0.0, 1.0);
    if (ax <= ay && ax <= az) {
        reference = vec3<f32>(1.0, 0.0, 0.0);
    } else if (ay <= az) {
        reference = vec3<f32>(0.0, 1.0, 0.0);
    }
    let perp = normalize_or_zero(cross(v, reference));
    if (dot(perp, perp) > EPS) {
        return perp;
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    let center = vec3<f32>(q.center_x, q.center_y, q.center_z);
    let velocity = vec3<f32>(q.velocity_x, q.velocity_y, q.velocity_z);
    let binormal_in = vec3<f32>(q.binormal_x, q.binormal_y, q.binormal_z);
    let prev = vec3<f32>(q.prev_x, q.prev_y, q.prev_z);
    let forward_in = vec3<f32>(q.forward_x, q.forward_y, q.forward_z);

    // --- StretchParams::stretched_size(velocity, base_half) ---
    var size_hl = q.base_half;
    var size_hw = q.base_half;
    if (dot(velocity, velocity) >= EPS) {
        let speed = sqrt(dot(velocity, velocity));
        let raw_length = q.base_half + speed * q.stretch_scale;
        size_hl = clamp(raw_length, q.min_length, q.max_length);
        size_hw = q.base_half * q.width_scale;
    }

    // --- StretchedQuad::aspect() on the stretched size ---
    var aspect = 1.0;
    if (abs(size_hw) >= EPS) {
        aspect = size_hl / size_hw;
    }

    // --- velocity_stretched_corners = stretched_corners(center, velocity, ...) ---
    let vc = stretched_corners(center, velocity, binormal_in, size_hl, size_hw);

    // --- stretched_from_prev(center, prev, binormal, trail_half_width) ---
    let travel = center - prev;
    let trail_length = sqrt(dot(travel, travel));
    var trail_binormal = normalize_or_zero(binormal_in);
    if (dot(trail_binormal, trail_binormal) <= EPS) {
        trail_binormal = perpendicular_to(travel);
    }
    let bw = trail_binormal * q.trail_half_width;
    let tc0 = center + bw;
    let tc1 = center - bw;
    let tc2 = prev + bw;
    let tc3 = prev - bw;

    // --- direct stretched_corners(center, forward_in, binormal, hl_in, hw_in) ---
    let dc = stretched_corners(center, forward_in, binormal_in, q.half_length_in, q.half_width_in);

    results[idx].size_half_length = size_hl;
    results[idx].size_half_width = size_hw;
    results[idx].aspect = aspect;
    results[idx].trail_length = trail_length;

    results[idx].vc0_x = vc.c0.x; results[idx].vc0_y = vc.c0.y; results[idx].vc0_z = vc.c0.z; results[idx].vc0_p = 0.0;
    results[idx].vc1_x = vc.c1.x; results[idx].vc1_y = vc.c1.y; results[idx].vc1_z = vc.c1.z; results[idx].vc1_p = 0.0;
    results[idx].vc2_x = vc.c2.x; results[idx].vc2_y = vc.c2.y; results[idx].vc2_z = vc.c2.z; results[idx].vc2_p = 0.0;
    results[idx].vc3_x = vc.c3.x; results[idx].vc3_y = vc.c3.y; results[idx].vc3_z = vc.c3.z; results[idx].vc3_p = 0.0;

    results[idx].tc0_x = tc0.x; results[idx].tc0_y = tc0.y; results[idx].tc0_z = tc0.z; results[idx].tc0_p = 0.0;
    results[idx].tc1_x = tc1.x; results[idx].tc1_y = tc1.y; results[idx].tc1_z = tc1.z; results[idx].tc1_p = 0.0;
    results[idx].tc2_x = tc2.x; results[idx].tc2_y = tc2.y; results[idx].tc2_z = tc2.z; results[idx].tc2_p = 0.0;
    results[idx].tc3_x = tc3.x; results[idx].tc3_y = tc3.y; results[idx].tc3_z = tc3.z; results[idx].tc3_p = 0.0;

    results[idx].dc0_x = dc.c0.x; results[idx].dc0_y = dc.c0.y; results[idx].dc0_z = dc.c0.z; results[idx].dc0_p = 0.0;
    results[idx].dc1_x = dc.c1.x; results[idx].dc1_y = dc.c1.y; results[idx].dc1_z = dc.c1.z; results[idx].dc1_p = 0.0;
    results[idx].dc2_x = dc.c2.x; results[idx].dc2_y = dc.c2.y; results[idx].dc2_z = dc.c2.z; results[idx].dc2_p = 0.0;
    results[idx].dc3_x = dc.c3.x; results[idx].dc3_y = dc.c3.y; results[idx].dc3_z = dc.c3.z; results[idx].dc3_p = 0.0;
}

// Four world-space corners, returned as a plain struct so the kernel can share
// the builder between the velocity and direct paths.
struct Corners {
    c0: vec3<f32>,
    c1: vec3<f32>,
    c2: vec3<f32>,
    c3: vec3<f32>,
}

// Builds the four world-space corners of a stretched sprite quad, mirroring the
// reference `stretched_corners`: the forward axis is normalized (a zero axis
// falls back to +Y) and the binormal is normalized (a collapsed binormal falls
// back to a deterministic perpendicular of the forward axis), then the corners
// are laid out at center +/- forward * half_length +/- binormal * half_width.
fn stretched_corners(
    center: vec3<f32>,
    forward_axis: vec3<f32>,
    binormal_in: vec3<f32>,
    half_length: f32,
    half_width: f32,
) -> Corners {
    var forward = normalize_or_zero(forward_axis);
    if (dot(forward, forward) <= EPS) {
        forward = vec3<f32>(0.0, 1.0, 0.0);
    }
    var binormal = normalize_or_zero(binormal_in);
    if (dot(binormal, binormal) <= EPS) {
        binormal = perpendicular_to(forward);
    }
    let fl = forward * half_length;
    let fw = binormal * half_width;
    var out: Corners;
    out.c0 = center + fl + fw;
    out.c1 = center + fl - fw;
    out.c2 = center - fl + fw;
    out.c3 = center - fl - fw;
    return out;
}
"#;

/// Inputs for one sprite motion-stretch query (design §15). A single query
/// drives every portable pure function so the result can be compared against
/// the whole golden surface at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpriteStretchQuery {
    /// The sprite's world-space centre.
    pub center: Vec3,
    /// The particle velocity; its length drives the stretch and it is the
    /// forward axis of the velocity-stretched quad.
    pub velocity: Vec3,
    /// The caller's in-plane width direction (typically `velocity × view`),
    /// shared by the velocity, trail and direct corner builders.
    pub binormal: Vec3,
    /// The particle's previous-frame position for the trail quad.
    pub prev_position: Vec3,
    /// The explicit forward axis for the direct `stretched_corners` call.
    pub forward_axis: Vec3,
    /// Isotropic base half-extent.
    pub base_half: f32,
    /// Explicit half-length for the direct `stretched_corners` call.
    pub half_length: f32,
    /// Explicit half-width for the direct `stretched_corners` call.
    pub half_width: f32,
    /// Half-width spread for the trail quad.
    pub trail_half_width: f32,
    /// The velocity-stretch parameters.
    pub params: StretchParams,
}

/// The device-computed result for one [`SpriteStretchQuery`], decoded into the
/// reference golden types for a direct parity comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpriteStretchResult {
    /// `params.stretched_size(velocity, base_half)`.
    pub size: StretchedQuad,
    /// `size.aspect()`.
    pub aspect: f32,
    /// `velocity_stretched_corners(center, velocity, binormal, base_half, params)`.
    pub velocity_corners: [Vec3; 4],
    /// `stretched_from_prev(center, prev_position, binormal, trail_half_width)`.
    pub trail: TrailQuad,
    /// `stretched_corners(center, forward_axis, binormal, half_length, half_width)`.
    pub corners: [Vec3; 4],
}

/// `std430` upload layout for one [`SpriteStretchQuery`], `112` bytes grouped
/// into seven `16`-byte lanes matching `Query` in [`SPRITE_STRETCH_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    base_half: f32,
    velocity_x: f32,
    velocity_y: f32,
    velocity_z: f32,
    pad_v: f32,
    binormal_x: f32,
    binormal_y: f32,
    binormal_z: f32,
    pad_b: f32,
    prev_x: f32,
    prev_y: f32,
    prev_z: f32,
    trail_half_width: f32,
    forward_x: f32,
    forward_y: f32,
    forward_z: f32,
    pad_f: f32,
    half_length_in: f32,
    half_width_in: f32,
    stretch_scale: f32,
    min_length: f32,
    max_length: f32,
    width_scale: f32,
    qpad0: f32,
    qpad1: f32,
}

impl GpuQuery {
    /// Packs a [`SpriteStretchQuery`] into the `std430` upload layout.
    fn from_query(query: &SpriteStretchQuery) -> GpuQuery {
        let c = query.center;
        let v = query.velocity;
        let b = query.binormal;
        let p = query.prev_position;
        let f = query.forward_axis;
        GpuQuery {
            center_x: c.x,
            center_y: c.y,
            center_z: c.z,
            base_half: query.base_half,
            velocity_x: v.x,
            velocity_y: v.y,
            velocity_z: v.z,
            pad_v: 0.0,
            binormal_x: b.x,
            binormal_y: b.y,
            binormal_z: b.z,
            pad_b: 0.0,
            prev_x: p.x,
            prev_y: p.y,
            prev_z: p.z,
            trail_half_width: query.trail_half_width,
            forward_x: f.x,
            forward_y: f.y,
            forward_z: f.z,
            pad_f: 0.0,
            half_length_in: query.half_length,
            half_width_in: query.half_width,
            stretch_scale: query.params.stretch_scale,
            min_length: query.params.min_length,
            max_length: query.params.max_length,
            width_scale: query.params.width_scale,
            qpad0: 0.0,
            qpad1: 0.0,
        }
    }
}

/// `std430` readback layout for one [`SpriteStretchResult`], `208` bytes: a
/// scalar lane followed by three four-corner blocks, matching `SpriteStretch`
/// in [`SPRITE_STRETCH_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    size_half_length: f32,
    size_half_width: f32,
    aspect: f32,
    trail_length: f32,
    velocity_corners: [[f32; 4]; 4],
    trail_corners: [[f32; 4]; 4],
    direct_corners: [[f32; 4]; 4],
}

impl GpuResult {
    /// Decodes the raw device record into the reference golden types.
    fn into_result(self) -> SpriteStretchResult {
        SpriteStretchResult {
            size: StretchedQuad {
                half_length: self.size_half_length,
                half_width: self.size_half_width,
            },
            aspect: self.aspect,
            velocity_corners: corners_from_lanes(self.velocity_corners),
            trail: TrailQuad {
                corners: corners_from_lanes(self.trail_corners),
                length: self.trail_length,
            },
            corners: corners_from_lanes(self.direct_corners),
        }
    }
}

/// Converts four `(x, y, z, pad)` device lanes into four [`Vec3`] corners.
fn corners_from_lanes(lanes: [[f32; 4]; 4]) -> [Vec3; 4] {
    [
        Vec3::new(lanes[0][0], lanes[0][1], lanes[0][2]),
        Vec3::new(lanes[1][0], lanes[1][1], lanes[1][2]),
        Vec3::new(lanes[2][0], lanes[2][1], lanes[2][2]),
        Vec3::new(lanes[3][0], lanes[3][1], lanes[3][2]),
    ]
}

/// `std430` uniform layout for the dispatch parameters, `16` bytes matching
/// `Params` in [`SPRITE_STRETCH_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable sprite motion-stretch pipeline.
pub struct GpuSpriteStretch {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSpriteStretch {
    /// Compiles the sprite motion-stretch kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSpriteStretch {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sprite_stretch"),
            source: ShaderSource::Wgsl(SPRITE_STRETCH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sprite_stretch_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sprite_stretch_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sprite_stretch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSpriteStretch {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the sprite motion-stretch evaluation for every query, returning one
    /// [`SpriteStretchResult`] per query in input order.
    ///
    /// The returned result for query `q` reproduces, field for field, the
    /// golden
    /// [`q.params.stretched_size(q.velocity, q.base_half)`](prism_render_architecture::particle::sprite_stretch::StretchParams::stretched_size),
    /// its [`aspect`](prism_render_architecture::particle::sprite_stretch::StretchedQuad::aspect),
    /// [`velocity_stretched_corners`](prism_render_architecture::particle::sprite_stretch::velocity_stretched_corners),
    /// [`stretched_from_prev`](prism_render_architecture::particle::sprite_stretch::stretched_from_prev)
    /// and a direct
    /// [`stretched_corners`](prism_render_architecture::particle::sprite_stretch::stretched_corners).
    /// An empty `queries` slice yields an empty result — storage buffers cannot
    /// be zero-sized, so it is handled by an early return and no dispatch.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[SpriteStretchQuery],
    ) -> Vec<SpriteStretchResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sprite_stretch_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sprite_stretch_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sprite_stretch_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sprite_stretch_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sprite_stretch_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sprite_stretch_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sprite_stretch_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch of `64`-wide
            // workgroups.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_results = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
            .into_iter()
            .map(GpuResult::into_result)
            .collect()
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
