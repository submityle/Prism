//! `wgpu` compute twin of the temporal-upscale render <-> display pixel mapping
//! ([`resolution`](prism_render_architecture::temporal_upscale::resolution)).
//!
//! A temporal upsampler renders at a fraction of the display resolution and
//! reconstructs the full image from jittered history, so every frame carries an
//! exact affine mapping between the **render** space the `GPU` rasterizes into
//! and the **display** space the reconstructor outputs. The `CPU` golden
//! [`UpscaleResolution`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution)
//! owns that mapping; [`GpuTaauResolutionMap`] is the on-device twin of its
//! per-coordinate affine arithmetic, so a passing real-device parity test is
//! direct evidence the ported kernel computes the same pixel transforms the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Given the integer render/display dimensions (derived host-side by
//! [`from_display`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::from_display))
//! and a set of coordinates, one thread reproduces the whole affine surface:
//!
//! - [`render_to_display`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::render_to_display):
//!   `(c + 0.5) * (display / render) - 0.5` per axis, pixel-center convention;
//! - [`display_to_render`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::display_to_render):
//!   `(c + 0.5) * effective_scale - 0.5` per axis;
//! - [`jitter_to_clip`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::jitter_to_clip):
//!   `jitter * (2 / render)` per axis, the vertical axis negated;
//! - [`effective_scale_x`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::effective_scale_x)
//!   and [`effective_scale_y`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::effective_scale_y):
//!   `render / display` per axis.
//!
//! Each quantity is a divide, an add and a multiply on the dimensions and the
//! input coordinate, so a `GPU` kernel reproduces every output within a few
//! units in the last place of the scalar reference.
//!
//! # What stays on the host
//!
//! The render-dimension derivation
//! ([`from_display`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::from_display))
//! — the render-scale validation, the `round`-to-nearest-pixel and the `max(1)`
//! floor — stays on the host: it is one-time integer setup, not per-coordinate
//! work, and it uses `round`, which the device subset omits. The pixel-count
//! helper
//! ([`render_pixel_count`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::render_pixel_count))
//! returns a `u64` and is therefore never twinned. The host enqueues one
//! [`TaauResolutionMapQuery`] per coordinate sample carrying the already-derived
//! integer dimensions, so a storage buffer is never zero-sized; an empty batch
//! short-circuits on the host with no dispatch.
//!
//! # Correctness model
//!
//! Every output is a finite affine combination of positive dimensions (each
//! guaranteed `>= 1` by the host derivation) and the input coordinate, so there
//! is no discrete branch to flip: the `CPU` and `GPU` agree within a legal
//! last-place divide difference, and the parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on each component.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /` and an
//! unsigned-to-float conversion — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, no inverse trigonometry, no `sqrt`, no `round`. No optional device
//! feature is required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! There is no loop: each thread performs a fixed, bounded sequence of
//! arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::resolution`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` temporal-upscale resolution-mapping kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`UpscaleResolution`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution)
/// per-coordinate affine maps; see the module documentation for the algorithm.
const TAAU_RESOLUTION_MAP_WGSL: &str = r#"
// Temporal-upscale resolution-mapping twin: one thread reproduces the render
// <-> display affine maps (render_to_display, display_to_render, jitter_to_clip)
// and the effective scales for one coordinate sample, mirroring the CPU golden
// `temporal_upscale::resolution::UpscaleResolution` with only + - * / and an
// unsigned-to-float convert. It owns no render-dimension derivation (round,
// max(1), scale validation) and no u64 pixel count; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::temporal_upscale::resolution；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of samples in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Already-derived integer dimensions (all >= 1 by host construction).
    render_width: u32,
    render_height: u32,
    display_width: u32,
    display_height: u32,
    // Render-space coordinate to map to display space.
    render_x: f32,
    render_y: f32,
    // Display-space coordinate to map to render space.
    display_x: f32,
    display_y: f32,
    // Sub-pixel jitter offset (render pixels) to convert to clip space.
    jitter_x: f32,
    jitter_y: f32,
    pad0: f32,
    pad1: f32,
}

struct Result {
    // render_to_display(render_x, render_y)
    rd_x: f32,
    rd_y: f32,
    // display_to_render(display_x, display_y)
    dr_x: f32,
    dr_y: f32,
    // jitter_to_clip(jitter_x, jitter_y)
    jc_x: f32,
    jc_y: f32,
    // effective_scale_x, effective_scale_y
    eff_x: f32,
    eff_y: f32,
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

    let rw = f32(q.render_width);
    let rh = f32(q.render_height);
    let dw = f32(q.display_width);
    let dh = f32(q.display_height);

    // render_to_display: forward map with the display/render ratio, pixel-center
    // convention (+0.5 then -0.5).
    let inv_x = dw / rw;
    let inv_y = dh / rh;
    let rd_x = (q.render_x + 0.5) * inv_x - 0.5;
    let rd_y = (q.render_y + 0.5) * inv_y - 0.5;

    // effective scales: the realized render/display ratio after host rounding.
    let eff_x = rw / dw;
    let eff_y = rh / dh;

    // display_to_render: inverse map with the effective scale.
    let dr_x = (q.display_x + 0.5) * eff_x - 0.5;
    let dr_y = (q.display_y + 0.5) * eff_y - 0.5;

    // jitter_to_clip: pixels -> NDC (2 units span render_width); flip y.
    let ndc_per_pixel_x = 2.0 / rw;
    let ndc_per_pixel_y = 2.0 / rh;
    let jc_x = q.jitter_x * ndc_per_pixel_x;
    let jc_y = -q.jitter_y * ndc_per_pixel_y;

    var out: Result;
    out.rd_x = rd_x;
    out.rd_y = rd_y;
    out.dr_x = dr_x;
    out.dr_y = dr_y;
    out.jc_x = jc_x;
    out.jc_y = jc_y;
    out.eff_x = eff_x;
    out.eff_y = eff_y;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TAAU_RESOLUTION_MAP_WGSL`].
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

/// `repr(C)` `std430` layout of one mapping query: the four integer dimensions
/// and the three coordinate pairs, plus two pad words to a `48`-byte
/// (`16`-byte-multiple) stride matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Render width in pixels.
    render_width: u32,
    /// Render height in pixels.
    render_height: u32,
    /// Display width in pixels.
    display_width: u32,
    /// Display height in pixels.
    display_height: u32,
    /// Render-space `x` to map to display space.
    render_x: f32,
    /// Render-space `y` to map to display space.
    render_y: f32,
    /// Display-space `x` to map to render space.
    display_x: f32,
    /// Display-space `y` to map to render space.
    display_y: f32,
    /// Jitter `x` (render pixels) to convert to clip space.
    jitter_x: f32,
    /// Jitter `y` (render pixels) to convert to clip space.
    jitter_y: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one mapping result: the three mapped coordinate
/// pairs and the two effective scales, a `32`-byte (`16`-byte-multiple) stride
/// matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `render_to_display` `x`.
    rd_x: f32,
    /// `render_to_display` `y`.
    rd_y: f32,
    /// `display_to_render` `x`.
    dr_x: f32,
    /// `display_to_render` `y`.
    dr_y: f32,
    /// `jitter_to_clip` `x`.
    jc_x: f32,
    /// `jitter_to_clip` `y`.
    jc_y: f32,
    /// `effective_scale_x`.
    eff_x: f32,
    /// `effective_scale_y`.
    eff_y: f32,
}

/// One per-sample query for the temporal-upscale resolution-mapping twin: the
/// already-derived integer render/display dimensions and the three coordinate
/// pairs to map.
///
/// The host owns the render-dimension derivation
/// ([`from_display`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution::from_display))
/// and passes the resulting integer dimensions here, matching the reference
/// which stores them once per resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauResolutionMapQuery {
    /// Render (input) width in pixels (`>= 1`).
    pub render_width: u32,
    /// Render (input) height in pixels (`>= 1`).
    pub render_height: u32,
    /// Display (output) width in pixels (`>= 1`).
    pub display_width: u32,
    /// Display (output) height in pixels (`>= 1`).
    pub display_height: u32,
    /// Render-space `x` to map to display space.
    pub render_x: f32,
    /// Render-space `y` to map to display space.
    pub render_y: f32,
    /// Display-space `x` to map to render space.
    pub display_x: f32,
    /// Display-space `y` to map to render space.
    pub display_y: f32,
    /// Jitter `x` (render pixels) to convert to clip space.
    pub jitter_x: f32,
    /// Jitter `y` (render pixels) to convert to clip space.
    pub jitter_y: f32,
}

impl TaauResolutionMapQuery {
    /// Builds a query from the integer dimensions and the three coordinate
    /// pairs.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the reference's full per-coordinate affine input set"
    )]
    pub const fn new(
        render_width: u32,
        render_height: u32,
        display_width: u32,
        display_height: u32,
        render_x: f32,
        render_y: f32,
        display_x: f32,
        display_y: f32,
        jitter_x: f32,
        jitter_y: f32,
    ) -> TaauResolutionMapQuery {
        TaauResolutionMapQuery {
            render_width,
            render_height,
            display_width,
            display_height,
            render_x,
            render_y,
            display_x,
            display_y,
            jitter_x,
            jitter_y,
        }
    }
}

/// One resolved set of affine maps, mirroring the arrays and scalars the golden
/// [`UpscaleResolution`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution)
/// returns for one coordinate sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauResolutionMapResult {
    /// `render_to_display(render_x, render_y)`.
    pub render_to_display: [f32; 2],
    /// `display_to_render(display_x, display_y)`.
    pub display_to_render: [f32; 2],
    /// `jitter_to_clip(jitter_x, jitter_y)`.
    pub jitter_to_clip: [f32; 2],
    /// `effective_scale_x()`.
    pub effective_scale_x: f32,
    /// `effective_scale_y()`.
    pub effective_scale_y: f32,
}

/// Encodes one [`TaauResolutionMapQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &TaauResolutionMapQuery) -> GpuQuery {
    GpuQuery {
        render_width: q.render_width,
        render_height: q.render_height,
        display_width: q.display_width,
        display_height: q.display_height,
        render_x: q.render_x,
        render_y: q.render_y,
        display_x: q.display_x,
        display_y: q.display_y,
        jitter_x: q.jitter_x,
        jitter_y: q.jitter_y,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`TaauResolutionMapResult`].
fn decode_result(raw: &GpuResult) -> TaauResolutionMapResult {
    TaauResolutionMapResult {
        render_to_display: [raw.rd_x, raw.rd_y],
        display_to_render: [raw.dr_x, raw.dr_y],
        jitter_to_clip: [raw.jc_x, raw.jc_y],
        effective_scale_x: raw.eff_x,
        effective_scale_y: raw.eff_y,
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

/// A compiled, reusable temporal-upscale resolution-mapping compute pipeline,
/// twinning the `CPU` golden
/// [`UpscaleResolution`](prism_render_architecture::temporal_upscale::resolution::UpscaleResolution).
pub struct GpuTaauResolutionMap {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTaauResolutionMap {
    /// Compiles the resolution-mapping kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTaauResolutionMap {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_taau_resolution_map"),
            source: ShaderSource::Wgsl(TAAU_RESOLUTION_MAP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_taau_resolution_map_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_taau_resolution_map_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_taau_resolution_map_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTaauResolutionMap {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every sample in `queries` and returns one
    /// [`TaauResolutionMapResult`] per input, in order.
    ///
    /// Each component equals the reference within a legal last-place divide
    /// difference, as documented on this module. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TaauResolutionMapQuery],
    ) -> Vec<TaauResolutionMapResult> {
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
            label: Some("prism_volumetric_taau_resolution_map_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_taau_resolution_map_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_taau_resolution_map_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_taau_resolution_map_bind_group"),
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
            label: Some("prism_volumetric_taau_resolution_map_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_taau_resolution_map_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_taau_resolution_map_pass"),
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
