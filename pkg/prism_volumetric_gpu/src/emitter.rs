//! `wgpu` compute twin of the particle-emission numeric contract
//! ([`emitter`](prism_render_architecture::particle::emitter), particle design
//! §8.2 *Emission*).
//!
//! The `CPU` golden
//! [`emitter`](prism_render_architecture::particle::emitter) turns an authored
//! spawn rate and a set of bursts into a concrete count of new particles, then
//! samples a distribution shape for each new particle's local spawn position and
//! emission direction and composes its initial velocity. Every piece of that
//! contract is a pure, deterministic, transcendental-free function: the shapes
//! are sampled by *rejection* inside a cube or square (using only comparisons
//! and a final robust normalize) rather than the textbook `sin`/`cos`/`cbrt`
//! formulations, so the same stream of unit samples always produces the same
//! spawn and the `CPU` reference and this `GPU` twin agree. [`GpuEmitter`] is the
//! on-device twin: one thread resolves one query, so a passing real-device
//! parity test is direct evidence the ported kernel evaluates the same closed
//! form the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-element numeric primitive is reproduced through a tagged
//! [`EmitterQuery`]: one variant per reference routine. The randomness a shape
//! sampler consumes is budgeted into a fixed-length sample array (at most
//! [`MAX_EMITTER_SAMPLES`] values) the host fills before dispatch, and the burst
//! set is budgeted into a fixed-length array (at most [`MAX_EMITTER_BURSTS`]
//! entries), mirroring how the reference reads its sample and burst slices. The
//! twinned routines are the unit-disk and unit-ball rejection samplers
//! ([`sample_shape`](prism_render_architecture::particle::emitter::sample_shape)
//! internals), the shape sampler itself, the fractional-carry rate accumulator
//! ([`SpawnAccumulator::accumulate`](prism_render_architecture::particle::emitter::SpawnAccumulator::accumulate)),
//! the burst-window reduction
//! ([`bursts_in_window`](prism_render_architecture::particle::emitter::bursts_in_window)),
//! the inherited-velocity scale
//! ([`inherited_velocity`](prism_render_architecture::particle::emitter::inherited_velocity)),
//! the initial-state composition
//! ([`build_spawn`](prism_render_architecture::particle::emitter::build_spawn)),
//! and the combined per-frame count
//! ([`Emitter::spawn_count`](prism_render_architecture::particle::emitter::Emitter::spawn_count)).
//!
//! # Correctness model
//!
//! Each sampler reads unit samples in the same order the reference does (a
//! wrapping cursor over the budgeted array, `0.5` for an empty slice), applies
//! the same bounded rejection loop, and finishes with the same robust normalize
//! (one `sqrt`), so `CPU` and `GPU` walk the identical branch and evaluate the
//! identical closed form. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every
//! continuous lane; the integer spawn counts are carried as exact-integer `f32`
//! payloads and compared exactly after rounding.
//!
//! # Degenerate inputs
//!
//! Every guard matches the reference: a non-positive or non-finite rate or step
//! spawns nothing and leaves the carry untouched, a backward or empty burst
//! window fires nothing, a non-positive radius or height collapses to a point,
//! and a zero sampled vector falls back to the `+Z` emission axis rather than
//! dividing. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `min`, `max`, `floor`, `dot`, `sqrt` and `+ - * /` — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no inverse trigonometry, no builtin `smoothstep`,
//! no `round`, and no `cbrt`, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Every loop is bounded (`MAX_REJECTION_TRIES`, [`MAX_EMITTER_BURSTS`]),
//! so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! # Honest host boundary
//!
//! The variable-length container logic is deliberately *not* twinned, because it
//! is host-side allocation / indexing rather than per-element kernel math:
//! [`allocate_spawns`](prism_render_architecture::particle::emitter::allocate_spawns)
//! and
//! [`Emitter::allocate`](prism_render_architecture::particle::emitter::Emitter::allocate)
//! draw slots from the owning pool's free list into a `Vec`, and the stateful
//! [`UnitCursor`](prism_render_architecture::particle::emitter::UnitCursor)
//! borrows a sample slice. The host feeds the twinned samplers a fixed-length
//! sample budget instead, and the twinned accumulator takes the carry explicitly
//! as a carry-in and returns the carry-out, so the stateful rate accumulation is
//! reproduced as a pure function.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::emitter::{
    BurstSpawn, EmitterShape, SpawnParams, SpawnSample,
};
use prism_render_architecture::particle::{SimSpace, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter`；无第三方引擎源码或衍生代码。
const WORKGROUP_SIZE: u32 = 64;

/// Upper bound on the unit samples one shape-sampling query carries. The
/// reference consumes at most [`MAX_EMITTER_SAMPLES`] values per spawn (eight
/// rejection tries of three components for the unit-ball sampler), and the
/// wrapping cursor reads modulo the live length, so this budget reproduces the
/// reference sampler without a variable-length device array.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter`；无第三方引擎源码或衍生代码。
pub const MAX_EMITTER_SAMPLES: usize = 32;

/// Upper bound on the bursts one window query carries. The host budgets the
/// emitter's burst list into at most this many entries before dispatch,
/// mirroring the reference scan over its burst slice.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter`；无第三方引擎源码或衍生代码。
pub const MAX_EMITTER_BURSTS: usize = 16;

/// The portable core-`WGSL` emission kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` branches on a
/// per-query op code into the `CPU` golden
/// [`emitter`](prism_render_architecture::particle::emitter) routines; see the
/// module documentation for the algorithm.
const EMITTER_WGSL: &str = r#"
// Particle-emission twin: one thread per query reproduces one reference numeric
// routine selected by `op`. It mirrors the CPU golden particle::emitter term
// for term, uses only the portable core-WGSL subset (abs/clamp/min/max/floor/
// dot/sqrt and + - * /), needs no transcendental call (the shapes are sampled by
// rejection, exactly as the reference avoids sin/cos/cbrt), and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12. Every loop
// is bounded (MAX_REJECTION_TRIES, MAX_BURSTS), so the kernel provably
// terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::emitter；
// 无第三方引擎源码或衍生代码。

// Squared-length floor for a degenerate direction, matching EPS_LEN_SQ.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Maximum rejection attempts before a shape sampler falls back to its axis.
const MAX_REJECTION_TRIES: u32 = 8u;
// Largest burst count the budgeted burst arrays can describe.
const MAX_BURSTS: u32 = 16u;
// Saturating-count sentinel, matching u32::MAX (f32 and u32 forms).
const U32_MAX_F: f32 = 4294967295.0;
const U32_MAX_U: u32 = 4294967295u;
// Local +Z, the canonical emission axis and degenerate-shape fallback.
const EMIT_AXIS: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
// Shape classification codes shared with the host encoder.
const SHAPE_POINT: u32 = 0u;
const SHAPE_SPHERE: u32 = 1u;
const SHAPE_BOX: u32 = 2u;
// Simulation-space codes shared with the host encoder (Local == 0).
const SIM_LOCAL: u32 = 0u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Geometry vectors; each vec3 carries a trailing pad lane to stay 16-byte
    // aligned on device.
    half_extents: vec3<f32>,
    pad_he: f32,
    origin: vec3<f32>,
    pad_or: f32,
    emitter_velocity: vec3<f32>,
    pad_ev: f32,
    sample_position: vec3<f32>,
    pad_sp: f32,
    sample_direction: vec3<f32>,
    pad_sd: f32,
    // Budgeted unit samples and burst arrays for the sampler / window ops.
    samples: array<f32, 32>,
    burst_times: array<f32, 16>,
    burst_counts: array<u32, 16>,
    // Scalar inputs; a field a given op does not name is ignored.
    radius: f32,
    base_radius: f32,
    height: f32,
    rate: f32,
    dt: f32,
    time: f32,
    carry_in: f32,
    speed: f32,
    inherit_velocity: f32,
    start: f32,
    end: f32,
    // Integer inputs / controls.
    shape_kind: u32,
    surface_only: u32,
    sample_len: u32,
    burst_count: u32,
    sim_space: u32,
    op: u32,
    pad_q0: f32,
    pad_q1: f32,
    pad_q2: f32,
}

struct Result {
    // Up to eight scalar outputs; the interpretation depends on the query op.
    // Vectors fill consecutive lanes (position in v0..v2, a second vector in
    // v3..v5); counts are carried in v0 as exact-integer f32 with the carry-out
    // in v1.
    v0: f32,
    v1: f32,
    v2: f32,
    v3: f32,
    v4: f32,
    v5: f32,
    v6: f32,
    v7: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

struct SpawnSample {
    position: vec3<f32>,
    direction: vec3<f32>,
}

// Next unit sample in [0, 1], wrapping; 0.5 when the slice is empty. Mirrors
// `UnitCursor::next_unit`.
fn next_unit(q: Query, idx: ptr<function, u32>) -> f32 {
    if (q.sample_len == 0u) {
        return 0.5;
    }
    let v = q.samples[(*idx) % q.sample_len];
    *idx = (*idx) + 1u;
    return v;
}

// Next sample mapped to [-1, 1]; mirrors `UnitCursor::next_signed`.
fn next_signed(q: Query, idx: ptr<function, u32>) -> f32 {
    return next_unit(q, idx) * 2.0 - 1.0;
}

// Robust normalize; mirrors `Vec3::normalize_or_zero` (EPS_LEN_SQ threshold).
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Point in the unit disk by rejection; mirrors `sample_unit_disk`.
fn sample_unit_disk(q: Query, idx: ptr<function, u32>) -> vec2<f32> {
    var i = 0u;
    loop {
        if (i >= MAX_REJECTION_TRIES) {
            break;
        }
        let x = next_signed(q, idx);
        let y = next_signed(q, idx);
        if (x * x + y * y <= 1.0) {
            return vec2<f32>(x, y);
        }
        i = i + 1u;
    }
    return vec2<f32>(0.0, 0.0);
}

// Point in the unit ball by rejection; mirrors `sample_unit_ball`.
fn sample_unit_ball(q: Query, idx: ptr<function, u32>) -> vec3<f32> {
    var i = 0u;
    loop {
        if (i >= MAX_REJECTION_TRIES) {
            break;
        }
        let x = next_signed(q, idx);
        let y = next_signed(q, idx);
        let z = next_signed(q, idx);
        let p = vec3<f32>(x, y, z);
        if (dot(p, p) <= 1.0) {
            return p;
        }
        i = i + 1u;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Samples a spawn placement for the query's shape; mirrors `sample_shape`. A
// zero sampled vector (dot(p, p) not strictly positive) falls back to the +Z
// emission axis, exactly as the reference compares against the zero vector.
fn sample_shape(q: Query, idx: ptr<function, u32>) -> SpawnSample {
    var s: SpawnSample;
    s.position = vec3<f32>(0.0, 0.0, 0.0);
    s.direction = EMIT_AXIS;
    if (q.shape_kind == SHAPE_POINT) {
        return s;
    } else if (q.shape_kind == SHAPE_SPHERE) {
        let p = sample_unit_ball(q, idx);
        var dir: vec3<f32>;
        if (dot(p, p) > 0.0) {
            dir = normalize_or_zero(p);
        } else {
            dir = EMIT_AXIS;
        }
        let r = max(q.radius, 0.0);
        if (q.surface_only != 0u) {
            s.position = dir * r;
        } else {
            s.position = p * r;
        }
        s.direction = dir;
        return s;
    } else if (q.shape_kind == SHAPE_BOX) {
        let x = next_signed(q, idx);
        let y = next_signed(q, idx);
        let z = next_signed(q, idx);
        s.position = vec3<f32>(x * q.half_extents.x, y * q.half_extents.y, z * q.half_extents.z);
        s.direction = EMIT_AXIS;
        return s;
    } else {
        let d = sample_unit_disk(q, idx);
        let base = vec3<f32>(d.x * q.base_radius, d.y * q.base_radius, max(q.height, 0.0));
        var dir: vec3<f32>;
        if (dot(base, base) > 0.0) {
            dir = normalize_or_zero(base);
        } else {
            dir = EMIT_AXIS;
        }
        s.position = vec3<f32>(0.0, 0.0, 0.0);
        s.direction = dir;
        return s;
    }
}

// Fractional-carry rate accumulator; mirrors `SpawnAccumulator::accumulate`.
// Returns the whole count in `.x` and the carried remainder in `.y`.
fn accumulate(carry_in: f32, rate: f32, dt: f32) -> vec2<f32> {
    if (rate > 0.0 && dt > 0.0) {
        var carry = carry_in + rate * dt;
        var whole: u32;
        if (carry >= U32_MAX_F) {
            whole = U32_MAX_U;
        } else {
            whole = u32(carry);
        }
        carry = carry - f32(whole);
        return vec2<f32>(f32(whole), carry);
    }
    return vec2<f32>(0.0, carry_in);
}

// Sums the bursts firing in the half-open window (start, end]; mirrors
// `bursts_in_window`. A backward or empty window fires nothing.
fn bursts_in_window(q: Query, start: f32, end: f32) -> u32 {
    if (end > start) {
        var total = 0u;
        var i = 0u;
        loop {
            if (i >= q.burst_count || i >= MAX_BURSTS) {
                break;
            }
            let t = q.burst_times[i];
            if (t > start && t <= end) {
                let c = q.burst_counts[i];
                if (total > U32_MAX_U - c) {
                    total = U32_MAX_U;
                } else {
                    total = total + c;
                }
            }
            i = i + 1u;
        }
        return total;
    }
    return 0u;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.v0 = 0.0;
    out.v1 = 0.0;
    out.v2 = 0.0;
    out.v3 = 0.0;
    out.v4 = 0.0;
    out.v5 = 0.0;
    out.v6 = 0.0;
    out.v7 = 0.0;

    if (q.op == 0u) {
        var cursor = 0u;
        let d = sample_unit_disk(q, &cursor);
        out.v0 = d.x;
        out.v1 = d.y;
    } else if (q.op == 1u) {
        var cursor = 0u;
        let p = sample_unit_ball(q, &cursor);
        out.v0 = p.x;
        out.v1 = p.y;
        out.v2 = p.z;
    } else if (q.op == 2u) {
        var cursor = 0u;
        let s = sample_shape(q, &cursor);
        out.v0 = s.position.x;
        out.v1 = s.position.y;
        out.v2 = s.position.z;
        out.v3 = s.direction.x;
        out.v4 = s.direction.y;
        out.v5 = s.direction.z;
    } else if (q.op == 3u) {
        let a = accumulate(q.carry_in, q.rate, q.dt);
        out.v0 = a.x;
        out.v1 = a.y;
    } else if (q.op == 4u) {
        let n = bursts_in_window(q, q.start, q.end);
        out.v0 = f32(n);
    } else if (q.op == 5u) {
        let v = q.emitter_velocity * q.inherit_velocity;
        out.v0 = v.x;
        out.v1 = v.y;
        out.v2 = v.z;
    } else if (q.op == 6u) {
        var pos: vec3<f32>;
        if (q.sim_space == SIM_LOCAL) {
            pos = q.sample_position;
        } else {
            pos = q.origin + q.sample_position;
        }
        let vel = q.sample_direction * q.speed + q.emitter_velocity * q.inherit_velocity;
        out.v0 = pos.x;
        out.v1 = pos.y;
        out.v2 = pos.z;
        out.v3 = vel.x;
        out.v4 = vel.y;
        out.v5 = vel.z;
    } else {
        let a = accumulate(q.carry_in, q.rate, q.dt);
        let from_rate = u32(a.x);
        let start = q.time - q.dt;
        let from_bursts = bursts_in_window(q, start, q.time);
        var total: u32;
        if (from_rate > U32_MAX_U - from_bursts) {
            total = U32_MAX_U;
        } else {
            total = from_rate + from_bursts;
        }
        out.v0 = f32(total);
        out.v1 = a.y;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`EMITTER_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// Every `vec3` lane carries a trailing pad word so each stays `16`-byte aligned
/// on device; the trailing pad words fill the final `16`-byte slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Box half-extents.
    half_extents: [f32; 3],
    /// Pad lane after `half_extents`.
    pad_he: f32,
    /// Emitter origin for the world/hybrid spawn offset.
    origin: [f32; 3],
    /// Pad lane after `origin`.
    pad_or: f32,
    /// Emitter velocity for inheritance.
    emitter_velocity: [f32; 3],
    /// Pad lane after `emitter_velocity`.
    pad_ev: f32,
    /// Sampled local spawn position for the build-spawn op.
    sample_position: [f32; 3],
    /// Pad lane after `sample_position`.
    pad_sp: f32,
    /// Sampled emission direction for the build-spawn op.
    sample_direction: [f32; 3],
    /// Pad lane after `sample_direction`.
    pad_sd: f32,
    /// Budgeted unit samples consumed by the shape samplers.
    samples: [f32; MAX_EMITTER_SAMPLES],
    /// Budgeted burst timestamps.
    burst_times: [f32; MAX_EMITTER_BURSTS],
    /// Budgeted burst counts.
    burst_counts: [u32; MAX_EMITTER_BURSTS],
    /// Sphere radius.
    radius: f32,
    /// Cone base radius.
    base_radius: f32,
    /// Cone height.
    height: f32,
    /// Spawn rate (particles per second) for the accumulator.
    rate: f32,
    /// Frame step in seconds.
    dt: f32,
    /// Emitter-local time for the spawn-count window.
    time: f32,
    /// Carried fractional remainder fed into the accumulator.
    carry_in: f32,
    /// Initial speed along the emission direction.
    speed: f32,
    /// Inherited-velocity fraction (also the standalone inherit factor).
    inherit_velocity: f32,
    /// Burst-window lower bound (exclusive) for the standalone window op.
    start: f32,
    /// Burst-window upper bound (inclusive) for the standalone window op.
    end: f32,
    /// Shape classification code (`0` point, `1` sphere, `2` box, `3` cone).
    shape_kind: u32,
    /// Whether a sphere samples its shell (`1`) or fills its volume (`0`).
    surface_only: u32,
    /// Number of live unit samples (`0` yields the constant-`0.5` cursor).
    sample_len: u32,
    /// Number of live bursts in the budgeted arrays.
    burst_count: u32,
    /// Simulation-space code (`0` local, `1` world, `2` hybrid).
    sim_space: u32,
    /// Op classification code (`0..=7`).
    op: u32,
    /// Padding lane.
    pad_q0: f32,
    /// Padding lane.
    pad_q1: f32,
    /// Padding lane.
    pad_q2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: up to eight scalar outputs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Output lane `0`.
    v0: f32,
    /// Output lane `1`.
    v1: f32,
    /// Output lane `2`.
    v2: f32,
    /// Output lane `3`.
    v3: f32,
    /// Output lane `4`.
    v4: f32,
    /// Output lane `5`.
    v5: f32,
    /// Output lane `6`.
    v6: f32,
    /// Output lane `7`.
    v7: f32,
}

/// One tagged query selecting which reference routine the kernel evaluates.
///
/// There is one variant per twinned `CPU` golden routine; a field a variant does
/// not name is ignored. The discriminant order matches the `u32` op codes the
/// kernel branches on.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmitterQuery {
    /// A rejection-sampled point in the unit disk, matching the `sample_unit_disk`
    /// internal of
    /// [`sample_shape`](prism_render_architecture::particle::emitter::sample_shape).
    SampleUnitDisk {
        /// Budgeted unit samples the wrapping cursor reads.
        samples: [f32; MAX_EMITTER_SAMPLES],
        /// Number of live samples.
        sample_len: u32,
    },
    /// A rejection-sampled point in the unit ball, matching the `sample_unit_ball`
    /// internal of
    /// [`sample_shape`](prism_render_architecture::particle::emitter::sample_shape).
    SampleUnitBall {
        /// Budgeted unit samples the wrapping cursor reads.
        samples: [f32; MAX_EMITTER_SAMPLES],
        /// Number of live samples.
        sample_len: u32,
    },
    /// A full shape placement, matching
    /// [`sample_shape`](prism_render_architecture::particle::emitter::sample_shape).
    SampleShape {
        /// Distribution shape to sample.
        shape: EmitterShape,
        /// Budgeted unit samples the wrapping cursor reads.
        samples: [f32; MAX_EMITTER_SAMPLES],
        /// Number of live samples.
        sample_len: u32,
    },
    /// The fractional-carry rate accumulation, matching
    /// [`SpawnAccumulator::accumulate`](prism_render_architecture::particle::emitter::SpawnAccumulator::accumulate).
    Accumulate {
        /// Carried remainder from the previous frame.
        carry_in: f32,
        /// Spawn rate in particles per second.
        rate_per_second: f32,
        /// Frame step in seconds.
        dt: f32,
    },
    /// The burst-window reduction, matching
    /// [`bursts_in_window`](prism_render_architecture::particle::emitter::bursts_in_window).
    BurstsInWindow {
        /// Budgeted bursts to scan.
        bursts: [BurstSpawn; MAX_EMITTER_BURSTS],
        /// Number of live bursts.
        burst_count: u32,
        /// Window lower bound (exclusive).
        start: f32,
        /// Window upper bound (inclusive).
        end: f32,
    },
    /// The inherited-velocity scale, matching
    /// [`inherited_velocity`](prism_render_architecture::particle::emitter::inherited_velocity).
    InheritedVelocity {
        /// Emitter velocity to scale.
        emitter_velocity: Vec3,
        /// Inheritance fraction.
        factor: f32,
    },
    /// The initial-state composition, matching
    /// [`build_spawn`](prism_render_architecture::particle::emitter::build_spawn).
    BuildSpawn {
        /// Emitter origin.
        origin: Vec3,
        /// Emitter velocity for inheritance.
        emitter_velocity: Vec3,
        /// Sampled local placement.
        sample: SpawnSample,
        /// Authored per-spawn parameters.
        params: SpawnParams,
    },
    /// The combined per-frame spawn count, matching
    /// [`Emitter::spawn_count`](prism_render_architecture::particle::emitter::Emitter::spawn_count).
    SpawnCount {
        /// Carried remainder from the previous frame.
        carry_in: f32,
        /// Spawn rate in particles per second.
        rate_per_second: f32,
        /// Frame step in seconds.
        dt: f32,
        /// Budgeted bursts to scan over `(time - dt, time]`.
        bursts: [BurstSpawn; MAX_EMITTER_BURSTS],
        /// Number of live bursts.
        burst_count: u32,
        /// Emitter-local time at the end of the frame.
        time: f32,
    },
}

impl EmitterQuery {
    /// The `u32` op code the kernel branches on, matching the variant order.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter`；无第三方引擎源码或衍生代码。
    const fn code(&self) -> u32 {
        match self {
            EmitterQuery::SampleUnitDisk { .. } => 0,
            EmitterQuery::SampleUnitBall { .. } => 1,
            EmitterQuery::SampleShape { .. } => 2,
            EmitterQuery::Accumulate { .. } => 3,
            EmitterQuery::BurstsInWindow { .. } => 4,
            EmitterQuery::InheritedVelocity { .. } => 5,
            EmitterQuery::BuildSpawn { .. } => 6,
            EmitterQuery::SpawnCount { .. } => 7,
        }
    }
}

/// One decoded result, one variant per [`EmitterQuery`] variant.
///
/// Continuous lanes are returned as plain component arrays so the parity test
/// can compare them within tolerance; the integer spawn counts are returned as
/// exact `u32` values.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmitterResult {
    /// A point in the unit disk.
    SampleUnitDisk {
        /// The `(x, y)` disk coordinate.
        point: [f32; 2],
    },
    /// A point in the unit ball.
    SampleUnitBall {
        /// The `(x, y, z)` ball coordinate.
        point: [f32; 3],
    },
    /// A sampled shape placement.
    SampleShape {
        /// Local spawn position offset.
        position: [f32; 3],
        /// Unit emission direction (or `+Z` for a degenerate shape).
        direction: [f32; 3],
    },
    /// A fractional-carry accumulation.
    Accumulate {
        /// Whole particles to spawn this frame.
        count: u32,
        /// Remainder carried to the next frame.
        carry: f32,
    },
    /// A burst-window sum.
    BurstsInWindow {
        /// Particles released by the bursts in the window.
        count: u32,
    },
    /// An inherited velocity.
    InheritedVelocity {
        /// The scaled emitter velocity.
        velocity: [f32; 3],
    },
    /// A composed initial state.
    BuildSpawn {
        /// Initial position.
        position: [f32; 3],
        /// Initial velocity: emission plus inherited motion.
        velocity: [f32; 3],
    },
    /// A combined per-frame spawn count.
    SpawnCount {
        /// Total particles to spawn this frame.
        count: u32,
        /// Remainder carried to the next frame by the rate accumulator.
        carry: f32,
    },
}

/// Maps an [`EmitterShape`] to the `u32` code the kernel and encoder share.
#[must_use]
const fn shape_code(shape: EmitterShape) -> u32 {
    match shape {
        EmitterShape::Point => 0,
        EmitterShape::Sphere { .. } => 1,
        EmitterShape::Box { .. } => 2,
        EmitterShape::Cone { .. } => 3,
    }
}

/// Maps a [`SimSpace`] to the `u32` code the kernel and encoder share.
#[must_use]
const fn sim_space_code(space: SimSpace) -> u32 {
    match space {
        SimSpace::Local => 0,
        SimSpace::World => 1,
        SimSpace::Hybrid => 2,
    }
}

/// Encodes one [`EmitterQuery`] into its `std430` [`GpuQuery`] slot, zeroing the
/// lanes the chosen op does not read.
fn encode_query(query: &EmitterQuery) -> GpuQuery {
    let mut gpu = GpuQuery {
        half_extents: [0.0; 3],
        pad_he: 0.0,
        origin: [0.0; 3],
        pad_or: 0.0,
        emitter_velocity: [0.0; 3],
        pad_ev: 0.0,
        sample_position: [0.0; 3],
        pad_sp: 0.0,
        sample_direction: [0.0; 3],
        pad_sd: 0.0,
        samples: [0.0; MAX_EMITTER_SAMPLES],
        burst_times: [0.0; MAX_EMITTER_BURSTS],
        burst_counts: [0; MAX_EMITTER_BURSTS],
        radius: 0.0,
        base_radius: 0.0,
        height: 0.0,
        rate: 0.0,
        dt: 0.0,
        time: 0.0,
        carry_in: 0.0,
        speed: 0.0,
        inherit_velocity: 0.0,
        start: 0.0,
        end: 0.0,
        shape_kind: 0,
        surface_only: 0,
        sample_len: 0,
        burst_count: 0,
        sim_space: 0,
        op: query.code(),
        pad_q0: 0.0,
        pad_q1: 0.0,
        pad_q2: 0.0,
    };
    match *query {
        EmitterQuery::SampleUnitDisk {
            samples,
            sample_len,
        }
        | EmitterQuery::SampleUnitBall {
            samples,
            sample_len,
        } => {
            gpu.samples = samples;
            gpu.sample_len = sample_len;
        }
        EmitterQuery::SampleShape {
            shape,
            samples,
            sample_len,
        } => {
            gpu.samples = samples;
            gpu.sample_len = sample_len;
            gpu.shape_kind = shape_code(shape);
            encode_shape(&mut gpu, shape);
        }
        EmitterQuery::Accumulate {
            carry_in,
            rate_per_second,
            dt,
        } => {
            gpu.carry_in = carry_in;
            gpu.rate = rate_per_second;
            gpu.dt = dt;
        }
        EmitterQuery::BurstsInWindow {
            bursts,
            burst_count,
            start,
            end,
        } => {
            encode_bursts(&mut gpu, &bursts, burst_count);
            gpu.start = start;
            gpu.end = end;
        }
        EmitterQuery::InheritedVelocity {
            emitter_velocity,
            factor,
        } => {
            gpu.emitter_velocity = [emitter_velocity.x, emitter_velocity.y, emitter_velocity.z];
            gpu.inherit_velocity = factor;
        }
        EmitterQuery::BuildSpawn {
            origin,
            emitter_velocity,
            sample,
            params,
        } => {
            gpu.origin = [origin.x, origin.y, origin.z];
            gpu.emitter_velocity = [emitter_velocity.x, emitter_velocity.y, emitter_velocity.z];
            gpu.sample_position = [sample.position.x, sample.position.y, sample.position.z];
            gpu.sample_direction = [sample.direction.x, sample.direction.y, sample.direction.z];
            gpu.speed = params.speed;
            gpu.inherit_velocity = params.inherit_velocity;
            gpu.sim_space = sim_space_code(params.sim_space);
        }
        EmitterQuery::SpawnCount {
            carry_in,
            rate_per_second,
            dt,
            bursts,
            burst_count,
            time,
        } => {
            gpu.carry_in = carry_in;
            gpu.rate = rate_per_second;
            gpu.dt = dt;
            gpu.time = time;
            encode_bursts(&mut gpu, &bursts, burst_count);
        }
    }
    gpu
}

/// Writes the shape-specific scalar lanes for a [`SampleShape`](EmitterQuery::SampleShape)
/// query into `gpu`.
fn encode_shape(gpu: &mut GpuQuery, shape: EmitterShape) {
    match shape {
        EmitterShape::Point => {}
        EmitterShape::Sphere {
            radius,
            surface_only,
        } => {
            gpu.radius = radius;
            gpu.surface_only = u32::from(surface_only);
        }
        EmitterShape::Box { half_extents } => {
            gpu.half_extents = [half_extents.x, half_extents.y, half_extents.z];
        }
        EmitterShape::Cone {
            base_radius,
            height,
        } => {
            gpu.base_radius = base_radius;
            gpu.height = height;
        }
    }
}

/// Copies the budgeted burst arrays and live count into `gpu`.
fn encode_bursts(gpu: &mut GpuQuery, bursts: &[BurstSpawn; MAX_EMITTER_BURSTS], burst_count: u32) {
    for (slot, burst) in bursts.iter().enumerate() {
        gpu.burst_times[slot] = burst.time;
        gpu.burst_counts[slot] = burst.count;
    }
    gpu.burst_count = burst_count;
}

/// Rounds an exact-integer `f32` count payload to the `u32` it encodes.
#[must_use]
fn decode_count(payload: f32) -> u32 {
    // The payload is an exact non-negative integer; `floor(x + 0.5)` rounds it
    // without the banned `round` intrinsic, and the clamp keeps the cast in range.
    let rounded = (payload + 0.5).floor();
    rounded.clamp(0.0, u32::MAX as f32) as u32
}

/// Decodes one packed [`GpuResult`] into the public [`EmitterResult`], selecting
/// the fields the originating `query` variant produced.
fn decode_result(query: &EmitterQuery, raw: &GpuResult) -> EmitterResult {
    match query {
        EmitterQuery::SampleUnitDisk { .. } => EmitterResult::SampleUnitDisk {
            point: [raw.v0, raw.v1],
        },
        EmitterQuery::SampleUnitBall { .. } => EmitterResult::SampleUnitBall {
            point: [raw.v0, raw.v1, raw.v2],
        },
        EmitterQuery::SampleShape { .. } => EmitterResult::SampleShape {
            position: [raw.v0, raw.v1, raw.v2],
            direction: [raw.v3, raw.v4, raw.v5],
        },
        EmitterQuery::Accumulate { .. } => EmitterResult::Accumulate {
            count: decode_count(raw.v0),
            carry: raw.v1,
        },
        EmitterQuery::BurstsInWindow { .. } => EmitterResult::BurstsInWindow {
            count: decode_count(raw.v0),
        },
        EmitterQuery::InheritedVelocity { .. } => EmitterResult::InheritedVelocity {
            velocity: [raw.v0, raw.v1, raw.v2],
        },
        EmitterQuery::BuildSpawn { .. } => EmitterResult::BuildSpawn {
            position: [raw.v0, raw.v1, raw.v2],
            velocity: [raw.v3, raw.v4, raw.v5],
        },
        EmitterQuery::SpawnCount { .. } => EmitterResult::SpawnCount {
            count: decode_count(raw.v0),
            carry: raw.v1,
        },
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

/// A compiled, reusable particle-emission compute pipeline, twinning the `CPU`
/// golden [`emitter`](prism_render_architecture::particle::emitter).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter`；无第三方引擎源码或衍生代码。
pub struct GpuEmitter {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuEmitter {
    /// Compiles the emission kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuEmitter {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_emitter"),
            source: ShaderSource::Wgsl(EMITTER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_emitter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_emitter_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_emitter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuEmitter {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`EmitterResult`] per
    /// input, in order.
    ///
    /// Each result matches the `CPU` golden within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[EmitterQuery]) -> Vec<EmitterResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_emitter_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_emitter_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_emitter_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_emitter_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_emitter_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_emitter_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_emitter_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}
