//! `wgpu` compute twin of the adaptive-quality decision mapping
//! ([`QualityBounds::decision_at`](prism_render_architecture::quality::controller::QualityBounds::decision_at)).
//!
//! The reference adaptive-quality controller steers a single scalar quality
//! level `q` in `[0, 1]` and maps it through a per-knob
//! [`QualityBounds`](prism_render_architecture::quality::controller::QualityBounds)
//! range into a concrete
//! [`QualityDecision`](prism_render_architecture::quality::QualityDecision):
//! render scale, geometry error tolerance, shadow page budget, and
//! global-illumination ray density. That mapping is the stateless, monotone,
//! transcendental-free core of the loop, and is exactly what
//! [`QualityBounds::decision_at`](prism_render_architecture::quality::controller::QualityBounds::decision_at)
//! computes.
//!
//! [`GpuQualityDecision`] is the on-device twin that runs one thread per
//! `(bounds, q)` sample and reproduces that mapping step for step, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same decision the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one `(bounds, q)` sample the kernel emits:
//!
//! * `render_scale` = `lerp(render_scale.0, render_scale.1, clamp_unit(q))`;
//! * `geometry_error_pixels` =
//!   `lerp(geometry_error_pixels.0, geometry_error_pixels.1, clamp_unit(q))`;
//! * `shadow_page_budget` =
//!   `lerp_u32(shadow_page_budget.0, shadow_page_budget.1, clamp_unit(q))`,
//!   the `u32`-endpoint lerp rounded to the nearest integer and floored at `0`;
//! * `gi_ray_scale` = `lerp(gi_ray_scale.0, gi_ray_scale.1, clamp_unit(q))`.
//!
//! Here `clamp_unit(q)` clamps `q` into `[0, 1]` and resolves a `NaN` to `0`,
//! and `lerp(a, b, t) = a + (b - a) * t`. Every coefficient and ordering is
//! copied verbatim from the golden, so the only source of `CPU`/`GPU`
//! divergence is a last-place rounding difference in a shared `+ - * /`
//! sequence; the integer budget is bit-identical because both sides round the
//! same `f32` the same way.
//!
//! # What is not twinned (host-only)
//!
//! The stateful proportional-integral feedback loop
//! ([`AdaptiveQualityController`](prism_render_architecture::quality::controller::AdaptiveQualityController)) —
//! its integral accumulator, anti-windup clamp, deadband hysteresis and the
//! per-frame quality update — stays on the host. Only the pure, stateless
//! `decision_at` mapping is twinned here.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `min`,
//! `max`, `floor` and the ordered comparisons — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, inverse trigonometry, `smoothstep`, `round` or `sqrt`, and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no loop: each thread performs a fixed, bounded sequence of
//! arithmetic, so the kernel provably terminates.
//!
//! The `clamp_unit` `NaN` guard is expressed without an `f32` equality: the
//! low branch is taken when `!(q >= 0.0)`, which is `true` for both a negative
//! `q` and a `NaN` (every ordered comparison with `NaN` is `false`), matching
//! the golden `q.is_nan() || q < 0.0` fallback exactly. The `lerp_u32` rounding
//! is `floor(value + 0.5)`, which equals the golden's round-half-away-from-zero
//! for the non-negative values a `[0, 1]` interpolation between non-negative
//! endpoints can produce.
//!
//! # Correctness model
//!
//! Every continuous output is a shared `+ - * /` sequence with no reordering,
//! so `CPU` and `GPU` agree to within a last-place rounding slack. The integer
//! budget is derived by the same `as f32` / round / `as u32` chain on both
//! sides, so it matches bit-for-bit. The parity test asserts
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every continuous output and an
//! exact `==` on the budget.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::quality::controller`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` quality-decision kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `decide` mirrors
/// the `CPU` golden
/// [`QualityBounds::decision_at`](prism_render_architecture::quality::controller::QualityBounds::decision_at)
/// closed form; see the module documentation for the algorithm.
const QUALITY_DECISION_WGSL: &str = r#"
// Adaptive-quality decision twin: one thread maps one (bounds, q) sample through
// clamp_unit + lerp into a render scale, geometry error, shadow page budget and
// GI ray scale, mirroring the CPU golden
// `quality::controller::QualityBounds::decision_at` with only + - * /, min, max,
// floor and ordered comparisons.
//
// Provenance: 孪生自本仓 prism_render_architecture::quality::controller；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of samples in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // render_scale endpoints (lowest / highest quality).
    rs0: f32,
    rs1: f32,
    // geometry_error_pixels endpoints.
    ge0: f32,
    ge1: f32,
    // shadow_page_budget endpoints (u32).
    sb0: u32,
    sb1: u32,
    // gi_ray_scale endpoints.
    gi0: f32,
    gi1: f32,
    // Requested quality level, clamped on device.
    q: f32,
}

struct Res {
    render_scale: f32,
    geometry_error_pixels: f32,
    shadow_page_budget: u32,
    gi_ray_scale: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// Clamp to [0, 1], resolving NaN to 0, mirroring the golden `clamp_unit`.
// `!(x >= 0.0)` is true for a negative x and for a NaN, matching the golden
// `x.is_nan() || x < 0.0` low branch without an f32 equality.
fn clamp_unit(x: f32) -> f32 {
    var r: f32 = x;
    if (!(x >= 0.0)) {
        r = 0.0;
    } else if (x > 1.0) {
        r = 1.0;
    }
    return r;
}

// Linear interpolation a + (b - a) * t, mirroring the golden `lerp`.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

@compute @workgroup_size(64)
fn decide(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let qd = queries[idx];
    let q = clamp_unit(qd.q);

    var out: Res;
    out.render_scale = lerp(qd.rs0, qd.rs1, q);
    out.geometry_error_pixels = lerp(qd.ge0, qd.ge1, q);
    out.gi_ray_scale = lerp(qd.gi0, qd.gi1, q);

    // lerp_u32: interpolate the u32 endpoints as f32, round to nearest with
    // floor(value + 0.5) (equals the golden round-half-away-from-zero for the
    // non-negative values here), and floor at zero before the integer cast.
    let budget = lerp(f32(qd.sb0), f32(qd.sb1), q);
    var rounded: f32 = floor(budget + 0.5);
    if (!(rounded >= 0.0)) {
        rounded = 0.0;
    }
    out.shadow_page_budget = u32(rounded);

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`QUALITY_DECISION_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid samples in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one decision query: the four knob endpoint
/// pairs and the requested quality level, a `36`-byte scalar-packed stride
/// matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Render-scale lowest-quality endpoint.
    rs0: f32,
    /// Render-scale highest-quality endpoint.
    rs1: f32,
    /// Geometry-error lowest-quality endpoint.
    ge0: f32,
    /// Geometry-error highest-quality endpoint.
    ge1: f32,
    /// Shadow-page-budget lowest-quality endpoint.
    sb0: u32,
    /// Shadow-page-budget highest-quality endpoint.
    sb1: u32,
    /// `GI` ray-scale lowest-quality endpoint.
    gi0: f32,
    /// `GI` ray-scale highest-quality endpoint.
    gi1: f32,
    /// Requested quality level.
    q: f32,
}

/// `repr(C)` `std430` layout of one decision result, matching the `WGSL` `Res`
/// struct: the three interpolated `f32` knobs and the integer shadow page
/// budget, a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Interpolated render scale.
    render_scale: f32,
    /// Interpolated geometry error in pixels.
    geometry_error_pixels: f32,
    /// Interpolated shadow page budget.
    shadow_page_budget: u32,
    /// Interpolated `GI` ray scale.
    gi_ray_scale: f32,
}

/// One decision query: a per-knob
/// [`QualityBounds`](prism_render_architecture::quality::controller::QualityBounds)
/// range (each field a lowest- / highest-quality endpoint pair) plus the
/// requested quality level `q`.
///
/// `q` is clamped into `[0, 1]` on device, so an out-of-range or `NaN` value is
/// resolved exactly as the golden
/// [`QualityBounds::decision_at`](prism_render_architecture::quality::controller::QualityBounds::decision_at)
/// resolves it.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::quality::controller`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityDecisionQuery {
    /// Render-scale endpoints at lowest / highest quality.
    pub render_scale: (f32, f32),
    /// Geometry-error-in-pixels endpoints at lowest / highest quality.
    pub geometry_error_pixels: (f32, f32),
    /// Shadow-page-budget endpoints at lowest / highest quality.
    pub shadow_page_budget: (u32, u32),
    /// `GI` ray-scale endpoints at lowest / highest quality.
    pub gi_ray_scale: (f32, f32),
    /// Requested quality level in `[0, 1]` (clamped on device).
    pub q: f32,
}

impl QualityDecisionQuery {
    /// Builds a query from the four knob endpoint pairs and a quality level.
    #[must_use]
    pub const fn new(
        render_scale: (f32, f32),
        geometry_error_pixels: (f32, f32),
        shadow_page_budget: (u32, u32),
        gi_ray_scale: (f32, f32),
        q: f32,
    ) -> QualityDecisionQuery {
        QualityDecisionQuery {
            render_scale,
            geometry_error_pixels,
            shadow_page_budget,
            gi_ray_scale,
            q,
        }
    }
}

/// One sample's resolved quality decision, mirroring the `CPU` golden
/// [`QualityDecision`](prism_render_architecture::quality::QualityDecision).
///
/// `render_scale`, `geometry_error_pixels` and `gi_ray_scale` are the
/// [`lerp`](prism_render_architecture::quality) of their endpoints at the
/// clamped quality level; `shadow_page_budget` is the rounded, zero-floored
/// `lerp_u32` of its `u32` endpoints.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::quality::controller`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityDecisionResult {
    /// Render scale in `(0, 1]` for the temporal upsampler.
    pub render_scale: f32,
    /// Tolerated screen-space geometry error in pixels.
    pub geometry_error_pixels: f32,
    /// Number of shadow pages the allocator may keep resident.
    pub shadow_page_budget: u32,
    /// `GI` ray density scale in `(0, 1]`.
    pub gi_ray_scale: f32,
}

/// Encodes one [`QualityDecisionQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &QualityDecisionQuery) -> GpuQuery {
    GpuQuery {
        rs0: q.render_scale.0,
        rs1: q.render_scale.1,
        ge0: q.geometry_error_pixels.0,
        ge1: q.geometry_error_pixels.1,
        sb0: q.shadow_page_budget.0,
        sb1: q.shadow_page_budget.1,
        gi0: q.gi_ray_scale.0,
        gi1: q.gi_ray_scale.1,
        q: q.q,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`QualityDecisionResult`].
fn decode_result(raw: &GpuResult) -> QualityDecisionResult {
    QualityDecisionResult {
        render_scale: raw.render_scale,
        geometry_error_pixels: raw.geometry_error_pixels,
        shadow_page_budget: raw.shadow_page_budget,
        gi_ray_scale: raw.gi_ray_scale,
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

/// A compiled, reusable quality-decision compute pipeline, twinning the `CPU`
/// golden
/// [`QualityBounds::decision_at`](prism_render_architecture::quality::controller::QualityBounds::decision_at).
pub struct GpuQualityDecision {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQualityDecision {
    /// Compiles the quality-decision kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQualityDecision {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quality_decision"),
            source: ShaderSource::Wgsl(QUALITY_DECISION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quality_decision_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quality_decision_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quality_decision_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("decide"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQualityDecision {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every sample in `queries` and returns one
    /// [`QualityDecisionResult`] per input, in order.
    ///
    /// Each continuous field equals the matching `CPU` golden
    /// [`QualityBounds::decision_at`](prism_render_architecture::quality::controller::QualityBounds::decision_at)
    /// knob to within a last-place rounding slack, and the integer budget
    /// matches bit-for-bit. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[QualityDecisionQuery],
    ) -> Vec<QualityDecisionResult> {
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
            label: Some("prism_volumetric_quality_decision_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quality_decision_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quality_decision_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quality_decision_bind_group"),
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
            label: Some("prism_volumetric_quality_decision_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quality_decision_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quality_decision_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, flattened to a 1-D dispatch.
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
