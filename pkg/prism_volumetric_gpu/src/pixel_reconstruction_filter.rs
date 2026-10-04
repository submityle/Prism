//! `wgpu` compute twin of the pixel reconstruction filter warp of the reference
//! tracer (`prism_render_architecture::reference_pt::filter`).
//!
//! A path tracer takes one or more sub-pixel jitter samples per pixel. A naive
//! renderer gives every sample the same weight — a box filter — which lets
//! high-frequency detail alias across pixel edges. Production renderers instead
//! reconstruct each pixel with a smooth, wider kernel. Rather than splatting
//! each sample across several pixels, the reference *importance samples* the
//! kernel: it warps the uniform `[0, 1)` jitter through the filter's inverse
//! cumulative distribution so the sample positions already carry the filter's
//! shape. Averaging the radiance of those samples is then exactly a filtered
//! estimate, and the per-pixel streaming of the film is preserved.
//!
//! This module is the on-device twin of that oracle's stateless, `RNG`-free
//! `PixelFilter::warp`, together with the `tent_offset` inverse `CDF` it calls:
//!
//! - `Box` (kind `0`): the jitter passes through unchanged, keeping the warped
//!   position in `[0, 1)`.
//! - `Tent` (kind `1`): a radius-one triangular kernel centred on the pixel
//!   midpoint `0.5`, spanning `[-0.5, 1.5]` so the kernel reaches one pixel into
//!   each neighbour. Each axis is warped independently by `tent_offset`.
//!
//! [`GpuPixelFilter`] warps one jitter sample per thread, reproducing the
//! reference's exact closed form — only comparisons and `sqrt`, no
//! transcendental — so a passing real-device parity test is direct evidence the
//! ported kernel computes the same warp the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`PixelFilterQuery`] — the filter `kind` plus the two
//! uniform jitter coordinates in `[0, 1)` — and writes one
//! [`PixelFilterResult`] holding the warped `(wx, wy)` position and a `valid`
//! flag. For the box filter the kernel copies the jitter through verbatim; for
//! the tent filter it applies `0.5 + tent_offset(u)` on each axis, where
//! `tent_offset(u)` is `sqrt(2 u) - 1` on the lower half and
//! `1 - sqrt(2 - 2 u)` on the upper half.
//!
//! # What stays on the host
//!
//! The sampler that draws the uniform jitter (which needs an `RNG`), the film
//! accumulation and the filter selection all stay on the host; the device sees
//! only the stateless, fixed-width warp, one sample at a time, so a storage
//! buffer is never zero-sized.
//!
//! # Correctness model
//!
//! The tent branch threads through `sqrt`, so the `CPU` and `GPU` are not
//! bit-exact: a `GPU` `sqrt` may land a few units in the last place from the
//! scalar reference. The parity test asserts the tent outputs within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative floor `1e-6`). The box
//! branch performs no arithmetic, so its pass-through is asserted within
//! `abs_diff <= 1e-6`. The discrete `valid` word is asserted exactly: `1` when
//! `kind < 2`, else `0` with a zeroed position.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, ordered
//! comparisons, `select`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round` and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::filter`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` pixel-filter kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `reference_pt::filter::{tent_offset, PixelFilter::warp}`.
const PIXEL_RECONSTRUCTION_FILTER_WGSL: &str = r#"
// Pixel reconstruction filter twin: one thread warps one uniform jitter sample
// through the selected filter's inverse CDF, mirroring the CPU golden
// `reference_pt::filter::{tent_offset, PixelFilter::warp}` with only ordered
// comparisons, sqrt, products and quotients. The RNG sampler and the film
// accumulation stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::filter；无第三方
// 引擎源码或衍生代码。

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Query {
    kind: u32,
    pad0: u32,
    jitter_x: f32,
    jitter_y: f32,
};

struct Outcome {
    wx: f32,
    wy: f32,
    valid: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outcome>;

// Warps a uniform `u` in [0, 1) into a radius-one tent offset in [-1, 1] via the
// tent's inverse CDF: a shifted square root on each half.
fn tent_offset(u: f32) -> f32 {
    // Split at the tent's midpoint; each half inverts a parabolic CDF.
    return select(1.0 - sqrt(2.0 - 2.0 * u), sqrt(2.0 * u) - 1.0, u < 0.5);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var wx: f32 = 0.0;
    var wy: f32 = 0.0;
    var valid: u32 = 0u;

    if (q.kind == 0u) {
        // Box: the jitter passes through unchanged, no arithmetic.
        wx = q.jitter_x;
        wy = q.jitter_y;
        valid = 1u;
    } else if (q.kind == 1u) {
        // Tent: centre on the pixel midpoint and warp each axis independently.
        wx = 0.5 + tent_offset(q.jitter_x);
        wy = 0.5 + tent_offset(q.jitter_y);
        valid = 1u;
    }

    var out: Outcome;
    out.wx = wx;
    out.wy = wy;
    out.valid = valid;
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte-aligned uniform struct matching `Params` in
/// [`PIXEL_RECONSTRUCTION_FILTER_WGSL`].
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
/// the filter discriminant, a pad word and the two jitter coordinates, a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Filter kind: `0` for box, `1` for tent.
    kind: u32,
    /// Padding word keeping the float pair `8`-byte aligned.
    pad0: u32,
    /// Uniform jitter `x` in `[0, 1)`.
    jitter_x: f32,
    /// Uniform jitter `y` in `[0, 1)`.
    jitter_y: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Outcome`
/// struct: the warped position, the `valid` word and one pad word to a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Warped position, `x` axis.
    wx: f32,
    /// Warped position, `y` axis.
    wy: f32,
    /// `1` when the filter kind is recognised, else `0`.
    valid: u32,
    /// Padding word.
    pad0: u32,
}

/// One query for the pixel-filter twin: the filter kind and the two uniform
/// jitter coordinates.
///
/// `kind` selects the filter: `0` is the box filter (identity pass-through),
/// `1` is the radius-one tent filter. Any other value is an unrecognised kind
/// and yields `valid = 0` with a zeroed position. `jitter_x` and `jitter_y` are
/// the uniform `[0, 1)` sub-pixel coordinates drawn by the host sampler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelFilterQuery {
    /// Filter kind: `0` for box, `1` for tent.
    pub kind: u32,
    /// Uniform jitter `x` in `[0, 1)`.
    pub jitter_x: f32,
    /// Uniform jitter `y` in `[0, 1)`.
    pub jitter_y: f32,
}

impl PixelFilterQuery {
    /// Builds a query from the filter kind and the two jitter coordinates.
    #[must_use]
    pub const fn new(kind: u32, jitter_x: f32, jitter_y: f32) -> PixelFilterQuery {
        PixelFilterQuery {
            kind,
            jitter_x,
            jitter_y,
        }
    }
}

/// One resolved query of the pixel-filter twin: the warped sub-pixel position
/// and the degeneracy flag.
///
/// `wx` and `wy` are the warped position in pixel units relative to the pixel's
/// lower corner — `[0, 1)` for the box filter, `[-0.5, 1.5]` for the tent.
/// `valid` is `1` when the filter kind is recognised (`kind < 2`), else `0`, in
/// which case the position is zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelFilterResult {
    /// Warped position, `x` axis.
    pub wx: f32,
    /// Warped position, `y` axis.
    pub wy: f32,
    /// `1` when the filter kind is recognised, else `0`.
    pub valid: u32,
}

/// Encodes one public [`PixelFilterQuery`] into its packed `std430` form.
fn encode_query(q: &PixelFilterQuery) -> GpuQuery {
    GpuQuery {
        kind: q.kind,
        pad0: 0,
        jitter_x: q.jitter_x,
        jitter_y: q.jitter_y,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`PixelFilterResult`].
fn decode_result(raw: &GpuResult) -> PixelFilterResult {
    PixelFilterResult {
        wx: raw.wx,
        wy: raw.wy,
        valid: raw.valid,
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

/// A compiled, reusable pixel-filter compute pipeline, twinning the `CPU`
/// golden `reference_pt::filter::{tent_offset, PixelFilter::warp}`.
pub struct GpuPixelFilter {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPixelFilter {
    /// Compiles the pixel-filter kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPixelFilter {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_pixel_reconstruction_filter"),
            source: ShaderSource::Wgsl(PIXEL_RECONSTRUCTION_FILTER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_pixel_reconstruction_filter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_pixel_reconstruction_filter_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_pixel_reconstruction_filter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPixelFilter {
            module,
            layout,
            pipeline,
        }
    }

    /// Warps every query in `queries` and returns one [`PixelFilterResult`] per
    /// input, in order.
    ///
    /// The tent outputs match the reference to within the tolerance documented
    /// on this module; the box outputs are a verbatim pass-through. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[PixelFilterQuery],
    ) -> Vec<PixelFilterResult> {
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
            label: Some("prism_volumetric_pixel_reconstruction_filter_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_pixel_reconstruction_filter_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_pixel_reconstruction_filter_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_pixel_reconstruction_filter_bind_group"),
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
            label: Some("prism_volumetric_pixel_reconstruction_filter_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_pixel_reconstruction_filter_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_pixel_reconstruction_filter_pass"),
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
