//! `wgpu` compute twin of the volumetric cloud `coverage_remap` and the shared
//! `remap` scalar it is built on
//! ([`volumetric`](prism_render_architecture::volumetric), cloud modeling
//! design §4/§6).
//!
//! The `CPU` golden exposes two pure scalar maps. The shared
//! [`remap`](prism_render_architecture::volumetric::math) rescales `x` from an
//! input range `[in_lo, in_hi]` to an output range `[out_lo, out_hi]`, guarding
//! a collapsed input span (`|in_hi - in_lo| < EPS`, `EPS = 1e-6`) by taking the
//! interpolant as zero so it never divides by (near) zero. The cloud
//! [`coverage_remap`](prism_render_architecture::volumetric::modeling) saturates
//! both the base shape and the coverage into `[0, 1]`, then returns
//! `saturate(remap(base, 1 - cov, 1, 0, 1))`, so that raising coverage
//! monotonically lifts more of the shape field above the cloud threshold.
//!
//! Both are fixed, bounded sequences of `clamp` and `+ - * /` arithmetic with
//! no loop, no container and no transcendental, so each is a clean
//! one-thread-per-query device twin.
//!
//! [`GpuCloudCoverageRemap`] evaluates both maps for one query per thread,
//! reproducing the reference's exact closed form so a passing real-device
//! parity test is direct evidence the ported kernel rescales the same way the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For a query `(x, in_lo, in_hi, out_lo, out_hi, base_shape, coverage)` the
//! twin returns two values:
//!
//! - `remap_value = remap(x, in_lo, in_hi, out_lo, out_hi)`, with the collapsed
//!   span guarded to a zero interpolant,
//! - `coverage_remap_value = saturate(remap(saturate(base_shape),
//!   1 - saturate(coverage), 1, 0, 1))`.
//!
//! # What stays on the host
//!
//! Nothing of the two maps themselves stays on the host: each is a pure
//! fixed-width scalar map. The surrounding noise-field evaluation, detail
//! erosion and height shaping in
//! [`volumetric`](prism_render_architecture::volumetric) are not part of this
//! twin; the host enqueues one query per evaluation it wants, so a storage
//! buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Both outputs are continuous values threaded through a subtract, a divide and
//! a `clamp`, so `CPU` and `GPU` are not bit-exact: a `GPU` divide or fused
//! multiply-add may land a few units in the last place from the scalar
//! reference. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-5` or `rel_diff <= 1e-4`) on each output, tight enough to
//! catch a genuinely wrong port (a dropped guard, a swapped bound, a wrong
//! `1 - cov`) yet loose enough to admit a legal last-place difference. The
//! `|span| < EPS` guard is a magnitude comparison, so both sides take the same
//! branch for fixtures kept clear of the `EPS` threshold.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp` and
//! `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, `sqrt`, `tan`, no
//! inverse trigonometry, no `round` and no `ceil`, and no wide integer type. No
//! optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. There is no loop: each thread performs a fixed, bounded
//! sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::volumetric`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` cloud coverage remap kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`remap`](prism_render_architecture::volumetric::math) and
/// [`coverage_remap`](prism_render_architecture::volumetric::modeling) closed
/// forms; see the module documentation for the algorithm.
const CLOUD_COVERAGE_REMAP_WGSL: &str = r#"
// Cloud coverage remap twin: one thread rescales one query through the shared
// remap and the cloud coverage_remap, mirroring the CPU golden
// `volumetric::math::remap` and `volumetric::modeling::coverage_remap` closed
// forms with only clamp and + - * /. It owns no noise evaluation and no detail
// erosion; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::volumetric；无第三方引擎源码或
// 衍生代码。

// Span-collapse epsilon below which remap takes a zero interpolant, matching
// the reference `EPS`.
const EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Value to rescale through the shared remap.
    x: f32,
    // Input range lower bound.
    in_lo: f32,
    // Input range upper bound.
    in_hi: f32,
    // Output range lower bound.
    out_lo: f32,
    // Output range upper bound.
    out_hi: f32,
    // Unsaturated base shape fed to coverage_remap.
    base_shape: f32,
    // Unsaturated coverage fed to coverage_remap.
    coverage: f32,
    pad0: f32,
}

struct Outcome {
    // remap(x, in_lo, in_hi, out_lo, out_hi).
    remap_value: f32,
    // coverage_remap(base_shape, coverage).
    coverage_remap_value: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outcome>;

// Clamp into [0, 1]; the reference `saturate`. Named `sat` to avoid shadowing
// the WGSL built-in `saturate`.
fn sat(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

// Rescale x from [in_lo, in_hi] to [out_lo, out_hi]; a collapsed input span
// (|span| < EPS) takes a zero interpolant so no divide-by-(near)-zero occurs.
fn remap(x: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    let span = in_hi - in_lo;
    var t: f32 = 0.0;
    if (abs(span) >= EPS) {
        t = (x - in_lo) / span;
    }
    return out_lo + (out_hi - out_lo) * t;
}

// Cloud coverage remap: saturate both inputs, then lift the shape above the
// coverage-driven threshold and saturate the result.
fn coverage_remap(base_shape: f32, coverage: f32) -> f32 {
    let base = sat(base_shape);
    let cov = sat(coverage);
    return sat(remap(base, 1.0 - cov, 1.0, 0.0, 1.0));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Outcome;
    out.remap_value = remap(q.x, q.in_lo, q.in_hi, q.out_lo, q.out_hi);
    out.coverage_remap_value = coverage_remap(q.base_shape, q.coverage);
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CLOUD_COVERAGE_REMAP_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the five `remap` arguments plus the
/// two `coverage_remap` arguments and one pad word to a `32`-byte stride,
/// matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Value to rescale through the shared `remap`.
    x: f32,
    /// Input range lower bound.
    in_lo: f32,
    /// Input range upper bound.
    in_hi: f32,
    /// Output range lower bound.
    out_lo: f32,
    /// Output range upper bound.
    out_hi: f32,
    /// Unsaturated base shape fed to `coverage_remap`.
    base_shape: f32,
    /// Unsaturated coverage fed to `coverage_remap`.
    coverage: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Outcome`
/// struct: the two mapped values plus two pad words to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `remap(x, in_lo, in_hi, out_lo, out_hi)`.
    remap_value: f32,
    /// `coverage_remap(base_shape, coverage)`.
    coverage_remap_value: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One query for the cloud coverage remap twin: the five `remap` arguments plus
/// the base shape and coverage for `coverage_remap`.
///
/// The host owns the surrounding noise evaluation and detail erosion, and
/// enqueues one [`CloudCoverageRemapQuery`] per evaluation it wants, matching
/// the reference
/// [`remap`](prism_render_architecture::volumetric::math) and
/// [`coverage_remap`](prism_render_architecture::volumetric::modeling).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudCoverageRemapQuery {
    /// Value to rescale through the shared `remap`.
    pub x: f32,
    /// Input range lower bound.
    pub in_lo: f32,
    /// Input range upper bound.
    pub in_hi: f32,
    /// Output range lower bound.
    pub out_lo: f32,
    /// Output range upper bound.
    pub out_hi: f32,
    /// Unsaturated base shape fed to `coverage_remap`.
    pub base_shape: f32,
    /// Unsaturated coverage fed to `coverage_remap`.
    pub coverage: f32,
}

impl CloudCoverageRemapQuery {
    /// Builds a query from the five `remap` arguments and the two
    /// `coverage_remap` arguments.
    #[must_use]
    pub const fn new(
        x: f32,
        in_lo: f32,
        in_hi: f32,
        out_lo: f32,
        out_hi: f32,
        base_shape: f32,
        coverage: f32,
    ) -> CloudCoverageRemapQuery {
        CloudCoverageRemapQuery {
            x,
            in_lo,
            in_hi,
            out_lo,
            out_hi,
            base_shape,
            coverage,
        }
    }
}

/// One resolved query, mirroring the two scalar outputs the reference
/// [`remap`](prism_render_architecture::volumetric::math) and
/// [`coverage_remap`](prism_render_architecture::volumetric::modeling) produce.
///
/// The field names carry a `_value` suffix so they avoid reserved words.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudCoverageRemapResult {
    /// `remap(x, in_lo, in_hi, out_lo, out_hi)`.
    pub remap_value: f32,
    /// `coverage_remap(base_shape, coverage)`.
    pub coverage_remap_value: f32,
}

/// Encodes one [`CloudCoverageRemapQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &CloudCoverageRemapQuery) -> GpuQuery {
    GpuQuery {
        x: q.x,
        in_lo: q.in_lo,
        in_hi: q.in_hi,
        out_lo: q.out_lo,
        out_hi: q.out_hi,
        base_shape: q.base_shape,
        coverage: q.coverage,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`CloudCoverageRemapResult`].
fn decode_result(raw: &GpuResult) -> CloudCoverageRemapResult {
    CloudCoverageRemapResult {
        remap_value: raw.remap_value,
        coverage_remap_value: raw.coverage_remap_value,
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

/// A compiled, reusable cloud coverage remap compute pipeline, twinning the
/// `CPU` golden
/// [`remap`](prism_render_architecture::volumetric::math) and
/// [`coverage_remap`](prism_render_architecture::volumetric::modeling).
pub struct GpuCloudCoverageRemap {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCloudCoverageRemap {
    /// Compiles the cloud coverage remap kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCloudCoverageRemap {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloud_coverage_remap"),
            source: ShaderSource::Wgsl(CLOUD_COVERAGE_REMAP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloud_coverage_remap_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloud_coverage_remap_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloud_coverage_remap_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCloudCoverageRemap {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one
    /// [`CloudCoverageRemapResult`] per input, in order.
    ///
    /// Both outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CloudCoverageRemapQuery],
    ) -> Vec<CloudCoverageRemapResult> {
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
            label: Some("prism_volumetric_cloud_coverage_remap_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloud_coverage_remap_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloud_coverage_remap_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloud_coverage_remap_bind_group"),
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
            label: Some("prism_volumetric_cloud_coverage_remap_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloud_coverage_remap_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloud_coverage_remap_pass"),
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
