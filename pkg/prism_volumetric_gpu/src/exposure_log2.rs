//! `wgpu` compute twin of the cheap binary-`log2` ratio approximation and the
//! auto-exposure average-luminance division used by the ray-scene footprint and
//! luminance-histogram stages.
//!
//! Two independent `CPU` reference formulas are reproduced here, each ported
//! faithfully by its exact closed form (never by a native `log2`):
//!
//! - `log2_linear(ratio)`: a deliberately cheap approximation of `log2`. For a
//!   non-finite or non-positive `ratio` it is `0`; otherwise it normalizes the
//!   `mantissa` into `[1, 2)` by halving it while it is `>= 2` (incrementing the
//!   `exponent`) and doubling it while it is `< 1` (decrementing the
//!   `exponent`), then returns `exponent + (mantissa - 1)`. That trailing
//!   `mantissa - 1` is a linear-in-mantissa stand-in for the true fractional
//!   `log2`, which is why the result is exact on powers of two and only
//!   approximate between them.
//! - `exposure_from_average(avg_lum, key)`: the auto-exposure scale `key /
//!   denom`, where `denom` is `avg_lum` guarded to at least `EPS = 1e-6` so a
//!   black frame cannot divide by zero.
//!
//! [`GpuExposureLog2`] is the on-device twin of exactly those two formulas. One
//! thread solves one whole query: it runs the two normalization `while` loops to
//! reproduce `log2_linear`, then forms the guarded exposure division, so a
//! passing real-device parity test is direct evidence the ported kernel computes
//! the same values the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one query the twin reproduces two scalars: the `log2_value` from the
//! ratio normalization and the `exposure` from the guarded average-luminance
//! division. Both are pure functions of their inputs.
//!
//! # What stays on the host
//!
//! Nothing stateful. The host guarantees each `ratio` is finite and positive
//! and each `avg_lum` is non-negative, then marshals the batch of
//! `(ratio, avg_lum, key)` tuples into the device query buffer and reads the two
//! scalars back.
//!
//! # Correctness model
//!
//! The formulas thread through only `+ - * /`, comparisons, and bounded `while`
//! loops with no transcendental and no reorderable reduction, so the `CPU` and
//! `GPU` evaluate the same expression in the same iteration order. They are not
//! bit-exact — a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate — so results are compared within `abs_diff <= 1e-5` or
//! `rel_diff <= 1e-4`. The `while` loops multiply by the exactly representable
//! `0.5` and `2.0`, so the loop trip count matches on both paths for the finite,
//! positive ratios the host supplies.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — the four arithmetic
//! operators, comparisons, and `while` loops — with no `sin`, `cos`, `exp`,
//! `log`, `log2`, `pow`, `tan`, no inverse trigonometry, no `round`, no `sqrt`,
//! and no optional device feature, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::footprint` 与 `prism_render_architecture::particle::luminance_hist`；无第三方引擎源码或衍生代码。
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
/// used across this crate's one-thread-per-element kernels; here one element is
/// one whole `log2`-and-exposure evaluation.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` cheap-`log2` and exposure kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` goldens `ray_scene::footprint::log2_linear` and
/// `particle::luminance_hist::exposure_from_average`; see the module
/// documentation for the algorithm.
const EXPOSURE_LOG2_WGSL: &str = r#"
// Cheap-log2 ratio approximation and auto-exposure division twin: one thread
// solves one query, mirroring the CPU goldens `ray_scene::footprint::log2_linear`
// and `particle::luminance_hist::exposure_from_average` using only the four
// arithmetic operators, comparisons, and bounded while loops.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::footprint 与
// prism_render_architecture::particle::luminance_hist；无第三方引擎源码或衍生代码。

const EXPOSURE_EPS: f32 = 1e-6;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    ratio: f32,
    avg_lum: f32,
    key: f32,
    pad0: u32,
}

struct ExposureResult {
    log2_value: f32,
    exposure: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<ExposureResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // log2_linear: normalize mantissa into [1, 2) by halving while >= 2 and
    // doubling while < 1, accumulating the exponent, then add (mantissa - 1) as
    // the cheap linear-in-mantissa fractional stand-in. The host guarantees a
    // finite, positive ratio, so the non-finite/non-positive guard is a no-op.
    var mantissa = q.ratio;
    var exponent = 0.0;
    while (mantissa >= 2.0) {
        mantissa = mantissa * 0.5;
        exponent = exponent + 1.0;
    }
    while (mantissa < 1.0) {
        mantissa = mantissa * 2.0;
        exponent = exponent - 1.0;
    }
    let log2_value = exponent + (mantissa - 1.0);

    // exposure_from_average: key / avg_lum with avg_lum guarded to at least EPS.
    var denom = q.avg_lum;
    if (denom < EXPOSURE_EPS) {
        denom = EXPOSURE_EPS;
    }
    let exposure = q.key / denom;

    var out: ExposureResult;
    out.log2_value = log2_value;
    out.exposure = exposure;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in [`EXPOSURE_LOG2_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the ratio, average luminance, and exposure key plus one pad word to a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Linear ratio fed to the cheap `log2` approximation (finite, positive).
    ratio: f32,
    /// Average scene luminance for the exposure division (non-negative).
    avg_lum: f32,
    /// Exposure key numerator.
    key: f32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `ExposureResult`
/// struct: the two scalars plus two pad words to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Cheap `log2` approximation of the ratio.
    log2_value: f32,
    /// Auto-exposure scale `key / max(avg_lum, EPS)`.
    exposure: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One combined query: the linear `ratio` for the cheap `log2`, and the
/// `avg_lum` and `key` for the exposure division.
///
/// The host guarantees `ratio` is finite and positive and `avg_lum` is
/// non-negative, matching the preconditions the references assume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureLog2Query {
    /// Linear ratio fed to the cheap `log2` approximation.
    pub ratio: f32,
    /// Average scene luminance for the exposure division.
    pub avg_lum: f32,
    /// Exposure key numerator.
    pub key: f32,
}

/// One resolved pair, mirroring the references `log2_linear` and
/// `exposure_from_average`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureLog2Result {
    /// Cheap `log2` approximation of `ratio`.
    pub log2_value: f32,
    /// Auto-exposure scale `key / max(avg_lum, EPS)`.
    pub exposure: f32,
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

/// A compiled, reusable cheap-`log2` and exposure compute pipeline, twinning the
/// `CPU` goldens `ray_scene::footprint::log2_linear` and
/// `particle::luminance_hist::exposure_from_average`.
pub struct GpuExposureLog2 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuExposureLog2 {
    /// Compiles the cheap-`log2` and exposure kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuExposureLog2 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_exposure_log2"),
            source: ShaderSource::Wgsl(EXPOSURE_LOG2_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_exposure_log2_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_exposure_log2_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_exposure_log2_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuExposureLog2 {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one
    /// [`ExposureLog2Result`] per input, in order.
    ///
    /// Each scalar equals the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ExposureLog2Query],
    ) -> Vec<ExposureLog2Result> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let encoded: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                ratio: q.ratio,
                avg_lum: q.avg_lum,
                key: q.key,
                pad0: 0,
            })
            .collect();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_exposure_log2_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_exposure_log2_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_exposure_log2_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_exposure_log2_bind_group"),
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
            label: Some("prism_volumetric_exposure_log2_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_exposure_log2_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_exposure_log2_pass"),
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

        raw.iter()
            .map(|r| ExposureLog2Result {
                log2_value: r.log2_value,
                exposure: r.exposure,
            })
            .collect()
    }
}
