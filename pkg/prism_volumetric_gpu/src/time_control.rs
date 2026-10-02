//! `wgpu` compute twin of the particle time-dilation *policy* contracts
//! ([`time_control`](prism_render_architecture::particle::time_control),
//! particle design §25, §26).
//!
//! The `CPU` golden
//! [`time_control`](prism_render_architecture::particle::time_control) owns the
//! pure per-emitter clock math a `VFX` stack needs to run one emitter in
//! slow-motion while the rest of the scene runs at full speed: the composed
//! [`EmitterTimeControl::effective_scale`](prism_render_architecture::particle::time_control::EmitterTimeControl::effective_scale),
//! the clamped
//! [`EmitterTimeControl::effective_dt`](prism_render_architecture::particle::time_control::EmitterTimeControl::effective_dt),
//! the
//! [`EmitterTimeControl::is_effectively_paused`](prism_render_architecture::particle::time_control::EmitterTimeControl::is_effectively_paused)
//! test, the `CFL`-style
//! [`substep_count`](prism_render_architecture::particle::time_control::substep_count),
//! the
//! [`scaled_spawn_rate`](prism_render_architecture::particle::time_control::scaled_spawn_rate)
//! and the
//! [`scaled_age_delta`](prism_render_architecture::particle::time_control::scaled_age_delta).
//! [`GpuTimeControl`] is the on-device twin: one thread per query reproduces
//! every one of those derived quantities, so a passing real-device parity test
//! is direct evidence the ported kernel applies the same dilation, the same
//! pause gate and the same substep rounding the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! The twin reproduces the *pure functions* only; the mutable
//! `FixedStepAccumulator` advance state machine is deliberately out of scope.
//! For one packed [`TimeControlQuery`] the kernel emits one
//! [`TimeControlResult`] holding the effective scale, the effective `dt`, the
//! scaled spawn rate, the scaled age delta, the clamped substep count and the
//! effectively-paused flag. The reference's `effective_scale` pause gate (a
//! `true` `paused` flag forces exactly `0.0`), the `clamp_non_negative` guard on
//! `effective_dt` and `scaled_spawn_rate`, and the `substep_count` ceiling —
//! integer truncation plus one extra step only when a real remainder survives
//! the `CMP_EPS` tolerance, clamped to `max_substeps`, floored at one step for a
//! non-empty frame, and the single-capped-step fallback when `max_substep_dt`
//! is non-positive — are all mirrored branch for branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, `min`,
//! comparison, `+ - * /` and unsigned bit arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no `floor` (the integer part comes from a
//! truncating `u32` conversion exactly as the reference's saturating `as u32`
//! does) and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, compares and
//! one divide, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. The continuous outputs are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32`
//! fields, while the integer `substep_count` and the `is_effectively_paused`
//! flag are compared exactly (`==`) because they carry no floating-point slack.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`time_control`](prism_render_architecture::particle::time_control); no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::time_control::EmitterTimeControl;
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

/// The portable core-`WGSL` time-control kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`time_control`](prism_render_architecture::particle::time_control) branch
/// for branch; see the module documentation for the algorithm.
const TIME_CONTROL_WGSL: &str = r#"
// Particle time-control twin: one thread per query reproduces the emitter's
// effective time scale, effective dt, scaled spawn rate, scaled age delta,
// CFL-style substep count and effectively-paused flag. It mirrors the CPU
// golden `particle::time_control` branch for branch, uses only the portable
// core-WGSL subset (max/min, comparison and + - * / plus unsigned bit math),
// needs no floor (the integer part comes from a truncating u32 conversion, as
// the reference's saturating `as u32` does) and takes no optional feature, so
// it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::time_control; no
// third-party engine source or derived code.

// Absolute tolerance for every "is this zero / did a remainder survive"
// comparison, matching the reference `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Scene-wide dilation, already clamped >= 0 on the host.
    global_scale: f32,
    // Per-emitter dilation, already clamped >= 0 on the host.
    local_scale: f32,
    // Hard pause flag: non-zero forces the effective scale to 0.
    paused: u32,
    // Raw frame dt in seconds before dilation.
    raw_dt: f32,
    // Base spawn rate in particles per second before dilation.
    base_rate: f32,
    // Stability ceiling: largest stable substep dt.
    max_substep_dt: f32,
    // Hard cap on the substep count.
    max_substeps: u32,
    // Padding word rounding the struct to a 32-byte stride.
    pad: u32,
}

struct Result {
    // effective_scale.
    effective_scale: f32,
    // effective_dt(raw_dt).
    effective_dt: f32,
    // scaled_spawn_rate(base_rate).
    scaled_spawn_rate: f32,
    // scaled_age_delta(raw_dt), equal to effective_dt.
    scaled_age_delta: f32,
    // substep_count(effective_dt, max_substep_dt, max_substeps).
    substep_count: u32,
    // is_effectively_paused as a u32 bool (1 = paused).
    is_effectively_paused: u32,
    // Padding words rounding the struct to a 32-byte stride.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamps a scalar into [0, +inf): negative inputs collapse to 0, mirroring the
// reference `clamp_non_negative` for the finite, non-NaN domain the fixtures
// stay inside.
fn clamp_non_negative(value: f32) -> f32 {
    return max(value, 0.0);
}

// The CFL-style substep count for a scaled dt, mirroring the reference
// `substep_count`: integer truncation plus one extra step only when a real
// remainder survives CMP_EPS, clamped to `max_substeps`, floored at one step
// for a non-empty frame, with a single-capped-step fallback when there is no
// usable ceiling.
fn substep_count(effective_dt: f32, max_substep_dt: f32, max_substeps: u32) -> u32 {
    if (effective_dt <= CMP_EPS) {
        return 0u;
    }
    if (max_substep_dt <= CMP_EPS) {
        // No usable ceiling: run one step, still honoring the hard cap.
        return min(max_substeps, 1u);
    }

    var n: u32 = u32(effective_dt / max_substep_dt);
    if (f32(n) * max_substep_dt + CMP_EPS < effective_dt) {
        n = n + 1u;
    }
    if (n == 0u) {
        n = 1u;
    }
    return min(n, max_substeps);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // effective_scale: 0 when paused, else the product of global and local.
    var effective_scale: f32 = q.global_scale * q.local_scale;
    if (q.paused != 0u) {
        effective_scale = 0.0;
    }

    // effective_dt and scaled_spawn_rate are the clamped scaled quantities.
    let effective_dt = clamp_non_negative(q.raw_dt * effective_scale);
    let scaled_spawn_rate = clamp_non_negative(q.base_rate * effective_scale);

    // is_effectively_paused: the effective scale is at or below CMP_EPS.
    var paused_flag: u32 = 0u;
    if (effective_scale <= CMP_EPS) {
        paused_flag = 1u;
    }

    var out: Result;
    out.effective_scale = effective_scale;
    out.effective_dt = effective_dt;
    out.scaled_spawn_rate = scaled_spawn_rate;
    out.scaled_age_delta = effective_dt;
    out.substep_count = substep_count(effective_dt, q.max_substep_dt, q.max_substeps);
    out.is_effectively_paused = paused_flag;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// One time-control query: the emitter's composed dilation state plus the raw
/// frame `dt`, base spawn rate and substep schedule limits the reference
/// consumes.
///
/// These are exactly the inputs the golden
/// [`EmitterTimeControl::effective_scale`](prism_render_architecture::particle::time_control::EmitterTimeControl::effective_scale),
/// [`EmitterTimeControl::effective_dt`](prism_render_architecture::particle::time_control::EmitterTimeControl::effective_dt),
/// [`substep_count`](prism_render_architecture::particle::time_control::substep_count),
/// [`scaled_spawn_rate`](prism_render_architecture::particle::time_control::scaled_spawn_rate)
/// and
/// [`scaled_age_delta`](prism_render_architecture::particle::time_control::scaled_age_delta)
/// take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeControlQuery {
    /// The composed per-emitter dilation state (global scale, local scale and
    /// pause flag).
    pub control: EmitterTimeControl,
    /// The raw frame `dt` in seconds, before dilation.
    pub raw_dt: f32,
    /// The base spawn rate in particles per second, before dilation.
    pub base_rate: f32,
    /// The stability ceiling: the largest substep `dt` that stays stable.
    pub max_substep_dt: f32,
    /// The hard cap on the number of substeps.
    pub max_substeps: u32,
}

impl TimeControlQuery {
    /// Builds a query from an emitter control state and its frame scalars.
    #[must_use]
    pub const fn new(
        control: EmitterTimeControl,
        raw_dt: f32,
        base_rate: f32,
        max_substep_dt: f32,
        max_substeps: u32,
    ) -> TimeControlQuery {
        TimeControlQuery {
            control,
            raw_dt,
            base_rate,
            max_substep_dt,
            max_substeps,
        }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports across its twinned pure functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeControlResult {
    /// The effective time scale, matching
    /// [`EmitterTimeControl::effective_scale`](prism_render_architecture::particle::time_control::EmitterTimeControl::effective_scale).
    pub effective_scale: f32,
    /// The effective frame `dt`, matching
    /// [`EmitterTimeControl::effective_dt`](prism_render_architecture::particle::time_control::EmitterTimeControl::effective_dt).
    pub effective_dt: f32,
    /// The dilated spawn rate, matching
    /// [`scaled_spawn_rate`](prism_render_architecture::particle::time_control::scaled_spawn_rate).
    pub scaled_spawn_rate: f32,
    /// The dilated age increment, matching
    /// [`scaled_age_delta`](prism_render_architecture::particle::time_control::scaled_age_delta).
    pub scaled_age_delta: f32,
    /// The `CFL`-style substep count, matching
    /// [`substep_count`](prism_render_architecture::particle::time_control::substep_count).
    pub substep_count: u32,
    /// Whether the emitter will not advance this frame, matching
    /// [`EmitterTimeControl::is_effectively_paused`](prism_render_architecture::particle::time_control::EmitterTimeControl::is_effectively_paused).
    pub is_effectively_paused: bool,
}

/// `repr(C)` `std430` layout of one packed query: eight scalar words
/// (`global_scale`, `local_scale`, `paused`, `raw_dt`, `base_rate`,
/// `max_substep_dt`, `max_substeps`, `pad`) totalling `32` bytes, matching the
/// `WGSL` `Query` struct word for word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Scene-wide dilation, already clamped `>= 0`.
    global_scale: f32,
    /// Per-emitter dilation, already clamped `>= 0`.
    local_scale: f32,
    /// Hard pause flag: non-zero forces the effective scale to zero.
    paused: u32,
    /// Raw frame `dt` in seconds before dilation.
    raw_dt: f32,
    /// Base spawn rate in particles per second before dilation.
    base_rate: f32,
    /// Stability ceiling: largest stable substep `dt`.
    max_substep_dt: f32,
    /// Hard cap on the substep count.
    max_substeps: u32,
    /// Padding word rounding the struct to a `32`-byte stride.
    pad: u32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image, applying the host-side
    /// `TimeScale` clamp through the accessors so the on-device factors are
    /// already non-negative.
    fn new(query: &TimeControlQuery) -> GpuQuery {
        GpuQuery {
            global_scale: query.control.global_scale.get(),
            local_scale: query.control.local_scale.get(),
            paused: u32::from(query.control.paused),
            raw_dt: query.raw_dt,
            base_rate: query.base_rate,
            max_substep_dt: query.max_substep_dt,
            max_substeps: query.max_substeps,
            pad: 0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: eight scalar words
/// (`effective_scale`, `effective_dt`, `scaled_spawn_rate`, `scaled_age_delta`,
/// `substep_count`, `is_effectively_paused`, two pad words) totalling `32`
/// bytes, matching the `WGSL` `Result` struct word for word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Effective time scale.
    effective_scale: f32,
    /// Effective frame `dt`.
    effective_dt: f32,
    /// Dilated spawn rate.
    scaled_spawn_rate: f32,
    /// Dilated age increment.
    scaled_age_delta: f32,
    /// Substep count.
    substep_count: u32,
    /// Effectively-paused flag as a `u32` bool.
    is_effectively_paused: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable time-control compute pipeline.
pub struct GpuTimeControl {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTimeControl {
    /// Compiles the time-control kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTimeControl {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_time_control"),
            source: ShaderSource::Wgsl(TIME_CONTROL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_time_control_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_time_control_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_time_control_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTimeControl {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`TimeControlResult`] per
    /// input, in order.
    ///
    /// The continuous fields of each result equal the reference answers to
    /// within the tolerance documented on this module, while `substep_count`
    /// and `is_effectively_paused` match exactly. An empty input returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[TimeControlQuery]) -> Vec<TimeControlResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_time_control_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_time_control_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_time_control_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_time_control_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_time_control_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_time_control_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_time_control_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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

/// Decodes one packed [`GpuResult`] into the public [`TimeControlResult`].
fn decode_result(raw: &GpuResult) -> TimeControlResult {
    TimeControlResult {
        effective_scale: raw.effective_scale,
        effective_dt: raw.effective_dt,
        scaled_spawn_rate: raw.scaled_spawn_rate,
        scaled_age_delta: raw.scaled_age_delta,
        substep_count: raw.substep_count,
        is_effectively_paused: raw.is_effectively_paused != 0,
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
