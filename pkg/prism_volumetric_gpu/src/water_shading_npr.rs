//! `wgpu` compute twin of the stateless stylized (`NPR`) water-lighting response
//! [`plan_npr`](prism_render_architecture::water::shading::npr::plan_npr) and its
//! toon color-ramp primitive
//! [`quantize_ramp`](prism_render_architecture::water::shading::npr::quantize_ramp).
//!
//! The stylized water frontend reinterprets the resolved lighting into a toon
//! look: it ramp-quantizes the water color into discrete bands, snaps the
//! specular highlight into a hard on/off toon block, draws a hand-painted
//! shoreline foam edge that fades with distance to shore, carries the caustic
//! coverage through as `halftone` dot coverage with its screen-space scale, and
//! turns the surface flow speed into a saturating flow-line intensity. Every
//! quantity is a pure clamp, divide, compare or band snap with no
//! floating-point transcendental, so the port is faithful.
//!
//! [`GpuWaterShadingNpr`] is the on-device twin of that one response planner.
//! One thread solves one query, reproducing the reference's per-channel ramp
//! quantization, its toon threshold, its guarded foam-edge fade and its clamped
//! flow line, so a passing real-device parity test is direct evidence the
//! ported kernel computes the same stylized response the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces every field of the reference response:
//!
//! * `ramp_color`: each `RGB` channel is `quantize_ramp(channel, color_bands)` —
//!   the channel is clamped to `0..=1`, scaled by `max(color_bands, 1)`,
//!   truncated to an integer band index, divided back by the band count and
//!   capped at `1`, a non-decreasing step function.
//! * `toon_specular`: `1` when `specular_intensity >= specular_threshold`,
//!   else `0`.
//! * `foam_edge`: when `foam_edge_width > EPS`, the fade
//!   `clamp(1 - dist_to_shore / foam_edge_width, 0, 1)`; otherwise `0`.
//! * `halftone_coverage`: `clamp(caustic_intensity, 0, 1)`.
//! * `halftone_scale`: the `halftone_scale` parameter carried through verbatim.
//! * `flow_line`: `clamp(flow_speed * flow_line_gain, 0, 1)`.
//!
//! The reference `EPS` guard constant (`1e-6`) is reproduced as a `WGSL` `const`.
//!
//! # What stays on the host
//!
//! The reference [`SurfaceShadingInputs`](prism_render_architecture::water::shading::SurfaceShadingInputs)
//! carries several fields (`cos_view`, `ior`, `jacobian`, `depth`,
//! `ssr_confidence`, `ray_budget`) that
//! [`plan_npr`](prism_render_architecture::water::shading::npr::plan_npr) never
//! reads; they drive the `PBR` and hybrid frontends instead, so the twin omits
//! them and the host supplies only the inputs the stylized response consumes.
//! The surrounding frontend selection and the shared advanced base services are
//! stateful host infrastructure and are never dispatched.
//!
//! # Correctness model
//!
//! Every output is a pure `f32` map built from `clamp`, multiply, divide,
//! ordered comparison and an integer band truncation, so `CPU` and `GPU` agree
//! to within floating-point tolerance and the parity test asserts each field
//! with an absolute-or-relative closeness check. The only discrete decisions —
//! the band truncation and the toon threshold — are kept away from their exact
//! boundaries by the fixtures so a last-place rounding difference cannot flip a
//! band or the toon block.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`, `max`,
//! multiply, divide, ordered comparison and the `u32`/`f32` numeric
//! conversions — with no `sin`, `cos`, `exp`, `log`, `pow`, no inverse
//! trigonometry, no `sqrt` and no `u64`. No optional device feature is required,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each
//! thread performs a fixed, bounded sequence of arithmetic, so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::shading::npr`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` stylized-water-response kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`plan_npr`](prism_render_architecture::water::shading::npr::plan_npr); see
/// the module documentation for the algorithm.
const WATER_SHADING_NPR_WGSL: &str = r#"
// Stylized (NPR) water-lighting response twin: one thread resolves one surface
// sample into a toon response — per-channel ramp-quantized color, a hard toon
// specular block, a guarded shoreline foam-edge fade, halftone caustic coverage
// with its scale, and a saturating flow line — mirroring the CPU golden
// `water::shading::npr::plan_npr` with only clamp, multiply, divide, ordered
// comparison and integer band truncation. It owns no frontend selection or
// shared base services; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::shading::npr；无第三方
// 引擎源码或衍生代码。

// Reference guard epsilon (water::EPS), below which the foam edge is disabled.
const EPS: f32 = 1e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // NprShadingParams: toon color-ramp band count.
    color_bands: u32,
    // NprShadingParams: specular intensity at or above which the toon block is on.
    specular_threshold: f32,
    // NprShadingParams: shore distance over which the foam edge fades out.
    foam_edge_width: f32,
    // NprShadingParams: halftone screen-space dot scale, carried through.
    halftone_scale: f32,
    // NprShadingParams: flow-line intensity gain per unit flow speed.
    flow_line_gain: f32,
    // SurfaceShadingInputs: base water color channels.
    water_color_r: f32,
    water_color_g: f32,
    water_color_b: f32,
    // SurfaceShadingInputs: raw specular highlight intensity.
    specular_intensity: f32,
    // SurfaceShadingInputs: raw caustic coverage.
    caustic_intensity: f32,
    // SurfaceShadingInputs: surface flow speed.
    flow_speed: f32,
    // SurfaceShadingInputs: distance to the shoreline.
    dist_to_shore: f32,
}

struct Result {
    // Ramp-quantized stylized water color.
    ramp_r: f32,
    ramp_g: f32,
    ramp_b: f32,
    // Toon specular block, 1 when the highlight is on, 0 otherwise.
    toon_specular: f32,
    // Hand-drawn shoreline foam edge coverage.
    foam_edge: f32,
    // Halftone caustic dot coverage.
    halftone_coverage: f32,
    // Screen-space dot scale carried through for the halftone pattern.
    halftone_scale: f32,
    // Flow-aligned stylization line intensity.
    flow_line: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Quantize `t` into `bands` discrete steps: clamp to 0..=1, scale by the band
// count (floored to 1), truncate to a band index, divide back and cap at 1.
fn quantize_ramp(t: f32, bands: u32) -> f32 {
    let b = f32(max(bands, 1u));
    let idx = u32(clamp(t, 0.0, 1.0) * b);
    return min(f32(idx) / b, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;

    // Ramp-quantize each color channel into the toon bands.
    out.ramp_r = quantize_ramp(q.water_color_r, q.color_bands);
    out.ramp_g = quantize_ramp(q.water_color_g, q.color_bands);
    out.ramp_b = quantize_ramp(q.water_color_b, q.color_bands);

    // Toon specular: a hard on/off block at the threshold.
    if (q.specular_intensity >= q.specular_threshold) {
        out.toon_specular = 1.0;
    } else {
        out.toon_specular = 0.0;
    }

    // Shoreline foam edge: fade with distance to shore, disabled when the edge
    // width collapses to (or below) the guard epsilon. The division is reached
    // only on the guarded branch, so the width is strictly positive.
    if (q.foam_edge_width > EPS) {
        out.foam_edge = clamp(1.0 - q.dist_to_shore / q.foam_edge_width, 0.0, 1.0);
    } else {
        out.foam_edge = 0.0;
    }

    // Halftone caustic coverage, clamped to 0..=1, with its scale carried through.
    out.halftone_coverage = clamp(q.caustic_intensity, 0.0, 1.0);
    out.halftone_scale = q.halftone_scale;

    // Flow-aligned line intensity, saturating at 1.
    out.flow_line = clamp(q.flow_speed * q.flow_line_gain, 0.0, 1.0);

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_SHADING_NPR_WGSL`].
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

/// `repr(C)` `std430` layout of one stylized-response query, matching the `WGSL`
/// `Query` struct: the five `NprShadingParams` fields the planner reads followed
/// by the seven `SurfaceShadingInputs` scalars it consumes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Toon color-ramp band count (the golden `color_bands`).
    color_bands: u32,
    /// Toon specular threshold (the golden `specular_threshold`).
    specular_threshold: f32,
    /// Foam-edge fade width (the golden `foam_edge_width`).
    foam_edge_width: f32,
    /// Halftone dot scale carried through (the golden `halftone_scale`).
    halftone_scale: f32,
    /// Flow-line gain per unit speed (the golden `flow_line_gain`).
    flow_line_gain: f32,
    /// Base water color red channel (the golden `water_color.r`).
    water_color_r: f32,
    /// Base water color green channel (the golden `water_color.g`).
    water_color_g: f32,
    /// Base water color blue channel (the golden `water_color.b`).
    water_color_b: f32,
    /// Raw specular highlight intensity (the golden `specular_intensity`).
    specular_intensity: f32,
    /// Raw caustic coverage (the golden `caustic_intensity`).
    caustic_intensity: f32,
    /// Surface flow speed (the golden `flow_speed`).
    flow_speed: f32,
    /// Distance to the shoreline (the golden `dist_to_shore`).
    dist_to_shore: f32,
}

/// `repr(C)` `std430` layout of one stylized-response result, matching the
/// `WGSL` `Result` struct: the three ramp-quantized color channels plus the five
/// scalar response terms.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Ramp-quantized red channel.
    ramp_r: f32,
    /// Ramp-quantized green channel.
    ramp_g: f32,
    /// Ramp-quantized blue channel.
    ramp_b: f32,
    /// Toon specular block (`1` on, `0` off).
    toon_specular: f32,
    /// Shoreline foam edge coverage.
    foam_edge: f32,
    /// Halftone caustic dot coverage.
    halftone_coverage: f32,
    /// Halftone dot scale carried through.
    halftone_scale: f32,
    /// Flow-aligned line intensity.
    flow_line: f32,
}

/// One stylized-water-response query: the five
/// [`NprShadingParams`](prism_render_architecture::water::shading::NprShadingParams)
/// fields the planner reads plus the seven
/// [`SurfaceShadingInputs`](prism_render_architecture::water::shading::SurfaceShadingInputs)
/// scalars it consumes, mirroring the inputs
/// [`plan_npr`](prism_render_architecture::water::shading::npr::plan_npr) reads.
///
/// The host supplies only the consumed inputs; the unused `cos_view`, `ior`,
/// `jacobian`, `depth`, `ssr_confidence` and `ray_budget` fields never affect
/// the stylized response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterShadingNprQuery {
    /// Toon color-ramp band count (the golden `color_bands`).
    pub color_bands: u32,
    /// Toon specular threshold (the golden `specular_threshold`).
    pub specular_threshold: f32,
    /// Foam-edge fade width (the golden `foam_edge_width`).
    pub foam_edge_width: f32,
    /// Halftone dot scale carried through (the golden `halftone_scale`).
    pub halftone_scale: f32,
    /// Flow-line gain per unit speed (the golden `flow_line_gain`).
    pub flow_line_gain: f32,
    /// Base water color red channel (the golden `water_color.r`).
    pub water_color_r: f32,
    /// Base water color green channel (the golden `water_color.g`).
    pub water_color_g: f32,
    /// Base water color blue channel (the golden `water_color.b`).
    pub water_color_b: f32,
    /// Raw specular highlight intensity (the golden `specular_intensity`).
    pub specular_intensity: f32,
    /// Raw caustic coverage (the golden `caustic_intensity`).
    pub caustic_intensity: f32,
    /// Surface flow speed (the golden `flow_speed`).
    pub flow_speed: f32,
    /// Distance to the shoreline (the golden `dist_to_shore`).
    pub dist_to_shore: f32,
}

impl WaterShadingNprQuery {
    /// Builds a query from the stylized tuning parameters and the consumed
    /// surface inputs.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query flattens the golden params and inputs into one record"
    )]
    pub const fn new(
        color_bands: u32,
        specular_threshold: f32,
        foam_edge_width: f32,
        halftone_scale: f32,
        flow_line_gain: f32,
        water_color_r: f32,
        water_color_g: f32,
        water_color_b: f32,
        specular_intensity: f32,
        caustic_intensity: f32,
        flow_speed: f32,
        dist_to_shore: f32,
    ) -> WaterShadingNprQuery {
        WaterShadingNprQuery {
            color_bands,
            specular_threshold,
            foam_edge_width,
            halftone_scale,
            flow_line_gain,
            water_color_r,
            water_color_g,
            water_color_b,
            specular_intensity,
            caustic_intensity,
            flow_speed,
            dist_to_shore,
        }
    }
}

/// One resolved stylized-water response, mirroring the reference
/// [`NprResponse`](prism_render_architecture::water::shading::NprResponse) with
/// its `ramp_color` flattened into three channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterShadingNprResult {
    /// Ramp-quantized red channel (the golden `ramp_color.r`).
    pub ramp_r: f32,
    /// Ramp-quantized green channel (the golden `ramp_color.g`).
    pub ramp_g: f32,
    /// Ramp-quantized blue channel (the golden `ramp_color.b`).
    pub ramp_b: f32,
    /// Toon specular block (the golden `toon_specular`).
    pub toon_specular: f32,
    /// Shoreline foam edge coverage (the golden `foam_edge`).
    pub foam_edge: f32,
    /// Halftone caustic dot coverage (the golden `halftone_coverage`).
    pub halftone_coverage: f32,
    /// Halftone dot scale carried through (the golden `halftone_scale`).
    pub halftone_scale: f32,
    /// Flow-aligned line intensity (the golden `flow_line`).
    pub flow_line: f32,
}

/// Encodes one [`WaterShadingNprQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterShadingNprQuery) -> GpuQuery {
    GpuQuery {
        color_bands: q.color_bands,
        specular_threshold: q.specular_threshold,
        foam_edge_width: q.foam_edge_width,
        halftone_scale: q.halftone_scale,
        flow_line_gain: q.flow_line_gain,
        water_color_r: q.water_color_r,
        water_color_g: q.water_color_g,
        water_color_b: q.water_color_b,
        specular_intensity: q.specular_intensity,
        caustic_intensity: q.caustic_intensity,
        flow_speed: q.flow_speed,
        dist_to_shore: q.dist_to_shore,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterShadingNprResult`].
fn decode_result(raw: &GpuResult) -> WaterShadingNprResult {
    WaterShadingNprResult {
        ramp_r: raw.ramp_r,
        ramp_g: raw.ramp_g,
        ramp_b: raw.ramp_b,
        toon_specular: raw.toon_specular,
        foam_edge: raw.foam_edge,
        halftone_coverage: raw.halftone_coverage,
        halftone_scale: raw.halftone_scale,
        flow_line: raw.flow_line,
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

/// A compiled, reusable stylized-water-response compute pipeline, twinning the
/// stateless `f32` response of the `CPU` golden
/// [`plan_npr`](prism_render_architecture::water::shading::npr::plan_npr).
pub struct GpuWaterShadingNpr {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterShadingNpr {
    /// Compiles the stylized-water-response kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterShadingNpr {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_shading_npr"),
            source: ShaderSource::Wgsl(WATER_SHADING_NPR_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_shading_npr_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_shading_npr_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_shading_npr_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterShadingNpr {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`WaterShadingNprResult`]
    /// per input, in order.
    ///
    /// Each field equals the reference to within floating-point tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterShadingNprQuery],
    ) -> Vec<WaterShadingNprResult> {
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
            label: Some("prism_volumetric_water_shading_npr_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_shading_npr_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_shading_npr_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_shading_npr_bind_group"),
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
            label: Some("prism_volumetric_water_shading_npr_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_shading_npr_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_shading_npr_pass"),
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
