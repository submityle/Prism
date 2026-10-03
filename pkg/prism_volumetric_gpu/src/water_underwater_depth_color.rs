//! `wgpu` compute twin of the underwater depth colour shift
//! [`depth_color_shift`](prism_render_architecture::water::underwater::depth_color_shift)
//! the water subsystem uses to tint submerged surfaces by depth.
//!
//! Below the surface water is a participating medium: light is absorbed along
//! every path, and because red extinction is largest the red channel collapses
//! first, then green, leaving the deep blue-green cast that reads as
//! underwater. The per-channel shift is a stateless, closed-form, non-negative
//! function, so it ports cleanly to the device: a passing real-device parity
//! run is direct evidence the ported kernel honours the same `Beer-Lambert`
//! attenuation the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread computes one colour query. Each of the three channels is
//! multiplied by its own `Beer-Lambert` transmittance over `depth`, mirroring
//! [`depth_color_shift`](prism_render_architecture::water::underwater::depth_color_shift):
//! `channel.max(0) * clamp(exp_approx(-(ext.max(0) * depth.max(0))), 0, 1)`.
//! The transmittance factor reproduces
//! [`beer_lambert_transmittance`](prism_render_architecture::water::underwater::beer_lambert_transmittance),
//! and the exponential reproduces the crate's monotone
//! [`exp_approx`](prism_render_architecture::water::exp_approx) exactly:
//! `base = 1 + x / 4096`, floored to zero, squared twelve times.
//!
//! # What stays on the host
//!
//! The surrounding underwater passes — the `Henyey-Greenstein` phase, the
//! multiple-scatter boost, god-ray in-scatter, and the visibility test — plus
//! the froxel traversal that supplies per-sample `depth` are stateful,
//! variable-length host work and are not twinned here.
//!
//! # Correctness model
//!
//! Each channel is a short sequence of multiplies, a clamp, and the fixed
//! twelve-step squaring of `exp_approx` — no transcendental built-in — so the
//! `CPU` and `GPU` agree to within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+`, `-`, `*`, `/`,
//! `abs`, `min`, `max`, `clamp`, and a fixed-count loop — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`, and no
//! `round`. Each thread performs a fixed, bounded sequence of arithmetic, so
//! the kernel provably terminates. No optional device feature is required, so
//! it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::underwater`；无第三方引擎源码或衍生代码。
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

/// The inlined `WGSL` twin of
/// [`depth_color_shift`](prism_render_architecture::water::underwater::depth_color_shift),
/// its [`beer_lambert_transmittance`](prism_render_architecture::water::underwater::beer_lambert_transmittance)
/// factor, and the [`exp_approx`](prism_render_architecture::water::exp_approx)
/// exponential, computed with only `+`, `-`, `*`, `/`, `abs`, `min`, `max`,
/// `clamp`, and a fixed twelve-step squaring loop.
const WATER_UNDERWATER_DEPTH_COLOR_WGSL: &str = r#"
// Twin of water::underwater::depth_color_shift. Each channel is scaled by its
// Beer-Lambert transmittance over the depth, with exp_approx inlined exactly
// (base = 1 + x/4096, floored to zero, squared twelve times).
//
// Provenance: 孪生自本仓 prism_render_architecture::water::underwater；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Input linear RGB colour.
    color_r: f32,
    color_g: f32,
    color_b: f32,
    // Per-channel extinction coefficients.
    ext_r: f32,
    ext_g: f32,
    ext_b: f32,
    // Path depth below the surface.
    depth: f32,
    pad0: f32,
}

struct Result {
    // Depth-shifted linear RGB colour (each channel non-negative).
    r: f32,
    g: f32,
    b: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Monotone exponential approximation: base = 1 + x/4096, floored to zero, then
// squared twelve times (2^12 = 4096). Pure add/mul, matching the golden.
fn exp_approx(x: f32) -> f32 {
    var base: f32 = 1.0 + x / 4096.0;
    if (base < 0.0) {
        base = 0.0;
    }
    for (var i: i32 = 0; i < 12; i = i + 1) {
        base = base * base;
    }
    return base;
}

// Beer-Lambert transmittance over `distance` at `extinction`, floored inputs,
// clamped to [0, 1].
fn beer_lambert_transmittance(extinction: f32, distance: f32) -> f32 {
    let e = max(extinction, 0.0);
    let d = max(distance, 0.0);
    return clamp(exp_approx(-(e * d)), 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var out: Result;
    out.r = max(q.color_r, 0.0) * beer_lambert_transmittance(q.ext_r, q.depth);
    out.g = max(q.color_g, 0.0) * beer_lambert_transmittance(q.ext_g, q.depth);
    out.b = max(q.color_b, 0.0) * beer_lambert_transmittance(q.ext_b, q.depth);
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_UNDERWATER_DEPTH_COLOR_WGSL`].
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

/// `repr(C)` `std430` layout of one depth-colour query, matching the `WGSL`
/// `Query` struct: the three colour channels, the three extinction
/// coefficients, the depth, and one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Red component of the input colour.
    color_r: f32,
    /// Green component of the input colour.
    color_g: f32,
    /// Blue component of the input colour.
    color_b: f32,
    /// Red-channel extinction coefficient.
    ext_r: f32,
    /// Green-channel extinction coefficient.
    ext_g: f32,
    /// Blue-channel extinction coefficient.
    ext_b: f32,
    /// Path depth below the surface.
    depth: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one depth-colour result, matching the `WGSL`
/// `Result` struct: the three shifted channels plus one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Red component of the shifted colour.
    r: f32,
    /// Green component of the shifted colour.
    g: f32,
    /// Blue component of the shifted colour.
    b: f32,
    /// Padding word.
    pad0: f32,
}

/// One depth-colour query to run on the device, mirroring the inputs of
/// [`depth_color_shift`](prism_render_architecture::water::underwater::depth_color_shift).
///
/// Each channel of `color` is attenuated by its own extinction coefficient over
/// `depth`; see the module documentation for the arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterUnderwaterDepthColorQuery {
    /// Red component of the input linear RGB colour.
    pub color_r: f32,
    /// Green component of the input linear RGB colour.
    pub color_g: f32,
    /// Blue component of the input linear RGB colour.
    pub color_b: f32,
    /// Red-channel extinction coefficient.
    pub ext_r: f32,
    /// Green-channel extinction coefficient.
    pub ext_g: f32,
    /// Blue-channel extinction coefficient.
    pub ext_b: f32,
    /// Path depth below the surface.
    pub depth: f32,
}

/// One resolved depth-colour result, mirroring the golden shifted colour (each
/// channel non-negative and no brighter than its input).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterUnderwaterDepthColorResult {
    /// Red component of the shifted colour.
    pub r: f32,
    /// Green component of the shifted colour.
    pub g: f32,
    /// Blue component of the shifted colour.
    pub b: f32,
}

/// Encodes one [`WaterUnderwaterDepthColorQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &WaterUnderwaterDepthColorQuery) -> GpuQuery {
    GpuQuery {
        color_r: q.color_r,
        color_g: q.color_g,
        color_b: q.color_b,
        ext_r: q.ext_r,
        ext_g: q.ext_g,
        ext_b: q.ext_b,
        depth: q.depth,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterUnderwaterDepthColorResult`].
fn decode_result(raw: &GpuResult) -> WaterUnderwaterDepthColorResult {
    WaterUnderwaterDepthColorResult {
        r: raw.r,
        g: raw.g,
        b: raw.b,
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

/// A compiled, reusable depth-colour compute pipeline, twinning the numeric
/// core of the `CPU` golden
/// [`depth_color_shift`](prism_render_architecture::water::underwater::depth_color_shift).
pub struct GpuWaterUnderwaterDepthColor {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterUnderwaterDepthColor {
    /// Compiles the depth-colour kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterUnderwaterDepthColor {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_underwater_depth_color"),
            source: ShaderSource::Wgsl(WATER_UNDERWATER_DEPTH_COLOR_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_underwater_depth_color_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_underwater_depth_color_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_underwater_depth_color_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterUnderwaterDepthColor {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries` and returns one
    /// [`WaterUnderwaterDepthColorResult`] per input, in order.
    ///
    /// The outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterUnderwaterDepthColorQuery],
    ) -> Vec<WaterUnderwaterDepthColorResult> {
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
            label: Some("prism_volumetric_water_underwater_depth_color_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_underwater_depth_color_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_underwater_depth_color_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_underwater_depth_color_bind_group"),
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
            label: Some("prism_volumetric_water_underwater_depth_color_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_underwater_depth_color_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_underwater_depth_color_pass"),
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
