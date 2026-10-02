//! `wgpu` compute twin of the sub-frame spawn interpolation contract
//! ([`subframe_spawn`](prism_render_architecture::particle::subframe_spawn),
//! particle design §8.2).
//!
//! The `CPU` golden
//! [`subframe_spawn`](prism_render_architecture::particle::subframe_spawn) owns
//! the small, verifiable arithmetic a fast-moving emitter needs to smear its
//! per-frame spawns along the path the emitter swept during the frame: the
//! fractional spawn accumulator that turns a continuous rate into an integer
//! count without dropping the remainder
//! ([`SpawnAccumulator::accumulate`](prism_render_architecture::particle::subframe_spawn::SpawnAccumulator::accumulate)),
//! the centered sub-frame fraction of the `i`-th spawn
//! ([`SubframeSchedule::fraction`](prism_render_architecture::particle::subframe_spawn::SubframeSchedule::fraction)),
//! the linear blend of the emitter's previous and current transform origin
//! ([`interpolate_position`](prism_render_architecture::particle::subframe_spawn::interpolate_position))
//! and of a scalar emitter property
//! ([`interpolate_scalar`](prism_render_architecture::particle::subframe_spawn::interpolate_scalar)),
//! and the one-shot burst crossing test
//! ([`SpawnBurst::is_due`](prism_render_architecture::particle::subframe_spawn::SpawnBurst::is_due)).
//! [`GpuSubframeSpawn`] is the on-device twin: one thread solves one query, so a
//! passing real-device parity test is direct evidence the ported kernel computes
//! the same count, carry, fraction, blends and crossing flag the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced for a batch of
//! independent queries: the integer spawn `count` and the surviving fractional
//! `carry` from
//! [`accumulate`](prism_render_architecture::particle::subframe_spawn::SpawnAccumulator::accumulate),
//! the centered fraction of the `i`-th of `N` spawns from
//! [`fraction`](prism_render_architecture::particle::subframe_spawn::SubframeSchedule::fraction),
//! the interpolated position and scalar from
//! [`interpolate_position`](prism_render_architecture::particle::subframe_spawn::interpolate_position)
//! and
//! [`interpolate_scalar`](prism_render_architecture::particle::subframe_spawn::interpolate_scalar),
//! and the one-shot crossing flag from
//! [`is_due`](prism_render_architecture::particle::subframe_spawn::SpawnBurst::is_due).
//! The variable-length
//! [`fractions`](prism_render_architecture::particle::subframe_spawn::SubframeSchedule::fractions)
//! vector is not twinned separately: it is just the per-index
//! [`fraction`](prism_render_architecture::particle::subframe_spawn::SubframeSchedule::fraction)
//! collected in order, so the single-index twin already covers its logic.
//!
//! # Correctness model
//!
//! The spawn `count` and the crossing flag are discrete classifications: the
//! count is the `floor` of a quantity rejection-sampled away from any integer
//! boundary, and the flag is two ordered comparisons of raw stored inputs, so
//! `CPU` and `GPU` agree exactly and the parity test asserts an exact `==` on
//! both. The `carry`, the fraction and the two interpolations thread through
//! multiplies, adds and one guarded division, so `CPU` and `GPU` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits. The parity test therefore asserts
//! a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every continuous
//! quantity, tight enough to catch a genuinely wrong port yet loose enough to
//! admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A negative or `NaN` `rate` or `dt` must "emit nothing": the reference returns
//! a zero count and an unchanged carry. The kernel reproduces this with the
//! single ordered guard `rate >= 0.0 && dt >= 0.0`, which is `false` for a
//! negative operand and, because every ordered comparison against `NaN` is
//! `false`, also for a `NaN` operand — matching the reference guard without an
//! `f32` equality test. A zero spawn count has no spawn to place, so the
//! fraction falls back to `0.0`, mirroring the reference. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `+ - * /`,
//! ordered comparisons and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `round`, no inverse trigonometry, no `sqrt` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no loop: each thread performs a fixed, bounded sequence of
//! arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::subframe_spawn`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
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
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` sub-frame-spawn kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`subframe_spawn`](prism_render_architecture::particle::subframe_spawn)
/// branch for branch; see the module documentation for the algorithm.
const SUBFRAME_SPAWN_WGSL: &str = r#"
// Sub-frame-spawn twin: one thread per query reproduces the spawn accumulator
// (integer count + surviving carry), the centered sub-frame fraction of the
// i-th spawn, the position and scalar linear interpolations, and the one-shot
// burst crossing flag. It mirrors the CPU golden `particle::subframe_spawn`
// branch for branch, uses only the portable core-WGSL subset (floor, + - * /,
// ordered compares and unsigned index math), needs no sqrt and no transcendental
// call and takes no optional feature, so it runs unmodified on Metal, Vulkan and
// DX12. There is no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::subframe_spawn；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Previous-frame emitter origin for interpolate_position; a pad lane follows.
    prev_pos: vec3<f32>,
    pad_prev: f32,
    // Current-frame emitter origin for interpolate_position; a pad lane follows.
    curr_pos: vec3<f32>,
    pad_curr: f32,
    // Accumulator carry, spawn rate per second and frame dt, plus the sub-frame
    // interpolation fraction shared by both interpolations.
    carry: f32,
    rate: f32,
    dt: f32,
    frac: f32,
    // Previous and current scalar emitter property, the burst instant, and the
    // frame's lower (exclusive) clock time.
    prev_scalar: f32,
    curr_scalar: f32,
    burst_time: f32,
    prev_time: f32,
    // Frame's upper (inclusive) clock time, the fraction divisor (spawn count)
    // and index for fraction(i), and a tail pad word.
    curr_time: f32,
    frac_count: u32,
    frac_index: u32,
    pad_tail: u32,
}

struct Result {
    // interpolate_position blend, with the spawn count packed into w's lane.
    interp_pos: vec3<f32>,
    spawn_count: u32,
    // fraction(i), surviving carry, interpolate_scalar blend, and the is_due flag.
    fraction: f32,
    next_carry: f32,
    interp_scalar: f32,
    is_due: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // accumulate: a negative or NaN rate/dt emits nothing and keeps the carry.
    // `rate >= 0.0 && dt >= 0.0` is false for a negative operand and, because
    // every ordered compare against NaN is false, also for a NaN operand, so the
    // single ordered guard matches the reference `is_nan || < 0.0` guard without
    // an f32 equality test.
    var spawn_count: u32 = 0u;
    var next_carry: f32 = q.carry;
    if (q.rate >= 0.0 && q.dt >= 0.0) {
        let total = q.rate * q.dt + q.carry;
        let whole_f = floor(total);
        // `total` is non-negative here, so the conversion never wraps.
        spawn_count = u32(whole_f);
        next_carry = total - whole_f;
    }

    // fraction(i): the centered fraction (i + 0.5) / N; an empty schedule has no
    // spawn to place and yields 0.0.
    var fraction: f32 = 0.0;
    if (q.frac_count != 0u) {
        fraction = (f32(q.frac_index) + 0.5) / f32(q.frac_count);
    }

    // interpolate_position / interpolate_scalar: `prev + (curr - prev) * frac`,
    // so the endpoints are exact.
    let interp_pos = q.prev_pos + (q.curr_pos - q.prev_pos) * q.frac;
    let interp_scalar = q.prev_scalar + (q.curr_scalar - q.prev_scalar) * q.frac;

    // is_due: the half-open crossing test `prev_time < time <= curr_time`, so a
    // burst fires exactly once as the clock sweeps past it.
    var is_due: u32 = 0u;
    if (q.prev_time < q.burst_time && q.burst_time <= q.curr_time) {
        is_due = 1u;
    }

    var out: Result;
    out.interp_pos = interp_pos;
    out.spawn_count = spawn_count;
    out.fraction = fraction;
    out.next_carry = next_carry;
    out.interp_scalar = interp_scalar;
    out.is_due = is_due;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SUBFRAME_SPAWN_WGSL`].
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
/// Each `vec3` lane carries a trailing pad word so it stays `16`-byte aligned on
/// device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Previous-frame emitter origin.
    prev_pos: [f32; 3],
    /// Pad lane after `prev_pos`.
    pad_prev: f32,
    /// Current-frame emitter origin.
    curr_pos: [f32; 3],
    /// Pad lane after `curr_pos`.
    pad_curr: f32,
    /// Accumulator carry fed to `accumulate`.
    carry: f32,
    /// Spawn rate per second fed to `accumulate`.
    rate: f32,
    /// Frame `dt` fed to `accumulate`.
    dt: f32,
    /// Sub-frame interpolation fraction shared by both interpolations.
    frac: f32,
    /// Previous-frame scalar emitter property.
    prev_scalar: f32,
    /// Current-frame scalar emitter property.
    curr_scalar: f32,
    /// Burst instant fed to `is_due`.
    burst_time: f32,
    /// Frame lower (exclusive) clock time fed to `is_due`.
    prev_time: f32,
    /// Frame upper (inclusive) clock time fed to `is_due`.
    curr_time: f32,
    /// Fraction divisor (the frame's spawn count) fed to `fraction`.
    frac_count: u32,
    /// Fraction index `i` fed to `fraction`.
    frac_index: u32,
    /// Tail padding word.
    pad_tail: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `interpolate_position` blend.
    interp_pos: [f32; 3],
    /// Integer spawn count from `accumulate`.
    spawn_count: u32,
    /// Centered sub-frame fraction from `fraction`.
    fraction: f32,
    /// Surviving fractional carry from `accumulate`.
    next_carry: f32,
    /// `interpolate_scalar` blend.
    interp_scalar: f32,
    /// `1` when the burst instant is crossed this frame, `0` otherwise.
    is_due: u32,
}

/// One query for the sub-frame-spawn twin: the accumulator inputs, the fraction
/// index and divisor, the two interpolation endpoints with their shared
/// fraction, and the burst-crossing times.
///
/// The twinned routines are independent, so a single query exercises every one
/// of them at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubframeSpawnQuery {
    /// Accumulator carry fed to
    /// [`SpawnAccumulator::accumulate`](prism_render_architecture::particle::subframe_spawn::SpawnAccumulator::accumulate).
    pub carry: f32,
    /// Spawn rate per second fed to `accumulate`.
    pub rate: f32,
    /// Frame `dt` fed to `accumulate`.
    pub dt: f32,
    /// Spawn count divisor fed to
    /// [`SubframeSchedule::fraction`](prism_render_architecture::particle::subframe_spawn::SubframeSchedule::fraction).
    pub frac_count: u32,
    /// Spawn index `i` fed to `fraction`.
    pub frac_index: u32,
    /// Previous-frame emitter origin fed to
    /// [`interpolate_position`](prism_render_architecture::particle::subframe_spawn::interpolate_position).
    pub prev_pos: [f32; 3],
    /// Current-frame emitter origin fed to `interpolate_position`.
    pub curr_pos: [f32; 3],
    /// Sub-frame fraction shared by `interpolate_position` and
    /// [`interpolate_scalar`](prism_render_architecture::particle::subframe_spawn::interpolate_scalar).
    pub frac: f32,
    /// Previous-frame scalar property fed to `interpolate_scalar`.
    pub prev_scalar: f32,
    /// Current-frame scalar property fed to `interpolate_scalar`.
    pub curr_scalar: f32,
    /// Burst instant fed to
    /// [`SpawnBurst::is_due`](prism_render_architecture::particle::subframe_spawn::SpawnBurst::is_due).
    pub burst_time: f32,
    /// Frame lower (exclusive) clock time fed to `is_due`.
    pub prev_time: f32,
    /// Frame upper (inclusive) clock time fed to `is_due`.
    pub curr_time: f32,
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports across its twinned routines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubframeSpawnResult {
    /// Integer spawn count, matching the first element of the
    /// [`accumulate`](prism_render_architecture::particle::subframe_spawn::SpawnAccumulator::accumulate)
    /// tuple.
    pub spawn_count: u32,
    /// Surviving fractional carry, matching the returned accumulator's `carry`.
    pub next_carry: f32,
    /// Centered sub-frame fraction, matching
    /// [`SubframeSchedule::fraction`](prism_render_architecture::particle::subframe_spawn::SubframeSchedule::fraction).
    pub fraction: f32,
    /// Interpolated position, matching
    /// [`interpolate_position`](prism_render_architecture::particle::subframe_spawn::interpolate_position).
    pub interp_pos: [f32; 3],
    /// Interpolated scalar, matching
    /// [`interpolate_scalar`](prism_render_architecture::particle::subframe_spawn::interpolate_scalar).
    pub interp_scalar: f32,
    /// Burst-crossing flag, matching
    /// [`SpawnBurst::is_due`](prism_render_architecture::particle::subframe_spawn::SpawnBurst::is_due).
    pub is_due: bool,
}

/// Encodes one [`SubframeSpawnQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SubframeSpawnQuery) -> GpuQuery {
    GpuQuery {
        prev_pos: q.prev_pos,
        pad_prev: 0.0,
        curr_pos: q.curr_pos,
        pad_curr: 0.0,
        carry: q.carry,
        rate: q.rate,
        dt: q.dt,
        frac: q.frac,
        prev_scalar: q.prev_scalar,
        curr_scalar: q.curr_scalar,
        burst_time: q.burst_time,
        prev_time: q.prev_time,
        curr_time: q.curr_time,
        frac_count: q.frac_count,
        frac_index: q.frac_index,
        pad_tail: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SubframeSpawnResult`],
/// turning the crossing flag back into a [`bool`].
fn decode_result(raw: &GpuResult) -> SubframeSpawnResult {
    SubframeSpawnResult {
        spawn_count: raw.spawn_count,
        next_carry: raw.next_carry,
        fraction: raw.fraction,
        interp_pos: raw.interp_pos,
        interp_scalar: raw.interp_scalar,
        is_due: raw.is_due != 0,
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

/// A compiled, reusable sub-frame-spawn compute pipeline, twinning the `CPU`
/// golden
/// [`subframe_spawn`](prism_render_architecture::particle::subframe_spawn).
pub struct GpuSubframeSpawn {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSubframeSpawn {
    /// Compiles the sub-frame-spawn kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSubframeSpawn {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_subframe_spawn"),
            source: ShaderSource::Wgsl(SUBFRAME_SPAWN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_subframe_spawn_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_subframe_spawn_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_subframe_spawn_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSubframeSpawn {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SubframeSpawnResult`]
    /// per input, in order.
    ///
    /// The spawn count and the crossing flag equal the reference exactly for
    /// inputs clear of the integer and boundary thresholds; the carry, the
    /// fraction and the two interpolations match to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SubframeSpawnQuery],
    ) -> Vec<SubframeSpawnResult> {
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
            label: Some("prism_volumetric_subframe_spawn_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_subframe_spawn_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_subframe_spawn_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_subframe_spawn_bind_group"),
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
            label: Some("prism_volumetric_subframe_spawn_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_subframe_spawn_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_subframe_spawn_pass"),
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

        raw.iter().map(decode_result).collect()
    }
}
