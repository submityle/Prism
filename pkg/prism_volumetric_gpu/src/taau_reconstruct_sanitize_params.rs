//! `wgpu` compute twin of the temporal-resolve parameter sanitizer inside the
//! reconstructor
//! ([`reconstruct`](prism_render_architecture::temporal_upscale::reconstruct)).
//!
//! The `CPU` golden
//! [`ResolveParams::sanitized`](prism_render_architecture::temporal_upscale::reconstruct::ResolveParams::sanitized)
//! clamps the two resolve tunables into a safe working range: `variance_gamma`
//! is forced non-negative (a negative variance box is meaningless; `0` clips
//! straight to the mean) and `max_confidence` is floored at `1` so there is
//! always at least the current frame of accumulation. A `NaN` on either field
//! resolves to that field's default
//! ([`DEFAULT_VARIANCE_GAMMA`](prism_render_architecture::temporal_upscale::reconstruct::DEFAULT_VARIANCE_GAMMA)
//! and
//! [`DEFAULT_MAX_CONFIDENCE`](prism_render_architecture::temporal_upscale::reconstruct::DEFAULT_MAX_CONFIDENCE)).
//! It is a stateless, closed-form predicate with no transcendental calls, no
//! `64`-bit integers, and no loops, so it ports to the device directly.
//!
//! [`GpuTaauReconstructSanitizeParams`] is the on-device twin of that function:
//! one thread resolves one query, reproducing the two threshold branches. A
//! passing real-device parity test is direct evidence the ported kernel folds
//! the exact same edge cases the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For one query holding a raw `variance_gamma` and `max_confidence` the kernel
//! reproduces:
//! - `variance_gamma = if raw >= 0.0 { raw } else { DEFAULT_VARIANCE_GAMMA }`;
//! - `max_confidence = if raw >= 1.0 { raw } else { DEFAULT_MAX_CONFIDENCE }`.
//!
//! The `>=` comparisons fold a `NaN` to the default on each axis exactly like
//! the reference, since a `NaN` compares false against every bound.
//!
//! # What stays on the host
//!
//! Nothing of this function stays on the host; it is fully closed-form. The
//! sibling composite resolve
//! ([`resolve`](prism_render_architecture::temporal_upscale::reconstruct::resolve))
//! chains tone-map, `YCoCg`, neighborhood clip, and lock state and is twinned by
//! other modules, not here. An empty batch short-circuits on the host, since a
//! storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The kernel performs only comparisons and copies — no arithmetic that could
//! drift — so a passing fixture reproduces the reference branch exactly. The
//! parity test still asserts the shared continuous tolerance (`abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3`) on both outputs, which here is satisfied with zero
//! difference on every finite fixture and admits the finite default a `NaN`
//! input folds to.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `f32` comparisons and
//! copies — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse
//! trigonometry, no `sqrt`, no `round`, and no `64`-bit integers. No optional
//! device feature is required, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::reconstruct`；无第三方引擎源码或衍生代码。
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

/// Default variance-box half-width the sanitizer substitutes for a negative or
/// `NaN` `variance_gamma`, aligned with the golden
/// [`DEFAULT_VARIANCE_GAMMA`](prism_render_architecture::temporal_upscale::reconstruct::DEFAULT_VARIANCE_GAMMA).
pub const DEFAULT_VARIANCE_GAMMA: f32 = 1.0;

/// Default confidence cap the sanitizer substitutes for a sub-`1` or `NaN`
/// `max_confidence`, aligned with the golden
/// [`DEFAULT_MAX_CONFIDENCE`](prism_render_architecture::temporal_upscale::reconstruct::DEFAULT_MAX_CONFIDENCE).
pub const DEFAULT_MAX_CONFIDENCE: f32 = 16.0;

/// The portable core-`WGSL` resolve-parameter sanitizer kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`ResolveParams::sanitized`](prism_render_architecture::temporal_upscale::reconstruct::ResolveParams::sanitized);
/// see the module documentation for the algorithm.
const TAAU_RECONSTRUCT_SANITIZE_PARAMS_WGSL: &str = r#"
// Resolve-parameter sanitizer twin: one thread clamps a raw variance_gamma and
// max_confidence into their safe working range, mirroring the CPU golden
// `temporal_upscale::reconstruct::ResolveParams::sanitized` with only f32
// comparisons and copies. A NaN folds to the per-field default because it
// compares false against every bound.
//
// Provenance: 孪生自本仓 prism_render_architecture::temporal_upscale::reconstruct；无第三方
// 引擎源码或衍生代码。

const DEFAULT_VARIANCE_GAMMA: f32 = 1.0;
const DEFAULT_MAX_CONFIDENCE: f32 = 16.0;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Raw variance-box half-width before sanitizing.
    variance_gamma: f32,
    // Raw confidence cap before sanitizing.
    max_confidence: f32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Sanitized variance-box half-width (non-negative; NaN -> default).
    variance_gamma: f32,
    // Sanitized confidence cap (>= 1; NaN -> default).
    max_confidence: f32,
    pad0: u32,
    pad1: u32,
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

    // A negative box is meaningless; NaN compares false against the bound, so
    // both the negative and NaN cases fold to the default.
    var variance_gamma = DEFAULT_VARIANCE_GAMMA;
    if (q.variance_gamma >= 0.0) {
        variance_gamma = q.variance_gamma;
    }
    // Floor the cap at 1 so there is always at least the current frame; NaN
    // again folds to the default.
    var max_confidence = DEFAULT_MAX_CONFIDENCE;
    if (q.max_confidence >= 1.0) {
        max_confidence = q.max_confidence;
    }

    var out: Result;
    out.variance_gamma = variance_gamma;
    out.max_confidence = max_confidence;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TAAU_RECONSTRUCT_SANITIZE_PARAMS_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the raw `variance_gamma` and
/// `max_confidence` plus two pad words, a `16`-byte stride matching the `WGSL`
/// `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Raw variance-box half-width before sanitizing.
    variance_gamma: f32,
    /// Raw confidence cap before sanitizing.
    max_confidence: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the sanitized `variance_gamma` and `max_confidence` plus two pad words, a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Sanitized variance-box half-width.
    variance_gamma: f32,
    /// Sanitized confidence cap.
    max_confidence: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One sanitize query for the twin: the raw `variance_gamma` and
/// `max_confidence` tunables straight off a resolve configuration, before any
/// clamping.
///
/// The host enqueues one [`TaauReconstructSanitizeParamsQuery`] per query,
/// mirroring the input of the reference
/// [`ResolveParams::sanitized`](prism_render_architecture::temporal_upscale::reconstruct::ResolveParams::sanitized).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauReconstructSanitizeParamsQuery {
    /// Raw variance-box half-width in standard deviations, before sanitizing.
    pub variance_gamma: f32,
    /// Raw maximum accumulated confidence, before sanitizing.
    pub max_confidence: f32,
}

impl TaauReconstructSanitizeParamsQuery {
    /// Builds a query from a raw `variance_gamma` and `max_confidence`.
    #[must_use]
    pub const fn new(
        variance_gamma: f32,
        max_confidence: f32,
    ) -> TaauReconstructSanitizeParamsQuery {
        TaauReconstructSanitizeParamsQuery {
            variance_gamma,
            max_confidence,
        }
    }
}

/// One sanitized resolve configuration, mirroring the reference
/// [`ResolveParams::sanitized`](prism_render_architecture::temporal_upscale::reconstruct::ResolveParams::sanitized).
///
/// `variance_gamma` is non-negative and `max_confidence` is at least `1`; a
/// `NaN` input on either field resolves to that field's default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauReconstructSanitizeParamsResult {
    /// Sanitized variance-box half-width (non-negative).
    pub variance_gamma: f32,
    /// Sanitized maximum accumulated confidence (at least `1`).
    pub max_confidence: f32,
}

/// Encodes one [`TaauReconstructSanitizeParamsQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &TaauReconstructSanitizeParamsQuery) -> GpuQuery {
    GpuQuery {
        variance_gamma: q.variance_gamma,
        max_confidence: q.max_confidence,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`TaauReconstructSanitizeParamsResult`].
fn decode_result(raw: &GpuResult) -> TaauReconstructSanitizeParamsResult {
    TaauReconstructSanitizeParamsResult {
        variance_gamma: raw.variance_gamma,
        max_confidence: raw.max_confidence,
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

/// A compiled, reusable resolve-parameter sanitizer compute pipeline, twinning
/// the `CPU` golden
/// [`ResolveParams::sanitized`](prism_render_architecture::temporal_upscale::reconstruct::ResolveParams::sanitized).
pub struct GpuTaauReconstructSanitizeParams {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTaauReconstructSanitizeParams {
    /// Compiles the sanitizer kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTaauReconstructSanitizeParams {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_params"),
            source: ShaderSource::Wgsl(TAAU_RECONSTRUCT_SANITIZE_PARAMS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_params_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_params_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_params_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTaauReconstructSanitizeParams {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`TaauReconstructSanitizeParamsResult`] per input, in order.
    ///
    /// Each output matches the reference exactly on finite inputs and folds a
    /// `NaN` to the documented default. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TaauReconstructSanitizeParamsQuery],
    ) -> Vec<TaauReconstructSanitizeParamsResult> {
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
            label: Some("prism_volumetric_taau_reconstruct_sanitize_params_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_params_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_params_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_params_bind_group"),
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
            label: Some("prism_volumetric_taau_reconstruct_sanitize_params_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_params_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_taau_reconstruct_sanitize_params_pass"),
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
