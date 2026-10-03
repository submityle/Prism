//! `wgpu` compute twin of the underwater single/multiple-scatter primitives
//! ([`underwater`](prism_render_architecture::water::underwater)).
//!
//! Below the surface, water is a participating medium: light is absorbed and
//! scattered along every path. The `CPU` golden module derives the pure
//! parameter math feeding the froxel volume. This twin ports its three
//! stateless, per-point numeric kernels so a volumetric pass can evaluate them
//! on-device:
//!
//! * [`henyey_greenstein`](prism_render_architecture::water::underwater::henyey_greenstein)
//!   — the single-lobe `Henyey-Greenstein` scattering phase function,
//!   `(1 - g^2) / (4*PI * (1 + g^2 - 2*g*cos_theta)^(3/2))`, with the `3/2`
//!   power evaluated as `d * sqrt(d)` so only `sqrt` is used.
//! * [`multiple_scatter_boost`](prism_render_architecture::water::underwater::multiple_scatter_boost)
//!   — the bounded geometric-series amplification `single / (1 - albedo)`.
//! * [`godray_inscatter`](prism_render_architecture::water::underwater::godray_inscatter)
//!   — the god-ray in-scattered radiance
//!   `surface_light * scatter_albedo * (1 - transmittance)` over a shaft, where
//!   the transmittance is the `Beer-Lambert` term
//!   [`beer_lambert_transmittance`](prism_render_architecture::water::underwater::beer_lambert_transmittance).
//!
//! [`GpuWaterUnderwaterScatter`] is the on-device twin: one thread evaluates one
//! query, reproducing all three reference closed forms exactly, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same scatter response the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Each [`WaterUnderwaterScatterQuery`] carries the per-point drivers for all
//! three operations; the kernel returns one [`WaterUnderwaterScatterResult`]
//! holding the phase value, the multiple-scatter boost, and the god-ray
//! in-scatter together.
//!
//! The `Beer-Lambert` transmittance threads through the shared monotone
//! `exp_approx` envelope, which is reproduced inline in the kernel with only
//! add and multiply (`base = 1 + x / 4096`, floored at zero, squared twelve
//! times), bit-for-bit matching the reference's definition. No `exp`, `log`,
//! `pow` or inverse trigonometry is used; the only non-arithmetic built-in is
//! `sqrt`.
//!
//! # What stays on the host
//!
//! The per-channel colour primitives
//! ([`depth_color_shift`](prism_render_architecture::water::underwater::depth_color_shift))
//! and the visibility predicate
//! ([`is_visible`](prism_render_architecture::water::underwater::is_visible))
//! are not ported by this scalar twin. The empty-batch short-circuit also stays
//! on the host, since a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! All three outputs are continuous, so for fixtures chosen clear of the lobe
//! and boost singularities (asymmetry `g` away from `±0.999`, albedo away from
//! `0.999`) the `CPU` and `GPU` agree to a tight tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`); the small slack admits a legal
//! last-place `sqrt` / divide / squaring difference while still catching a
//! genuinely wrong port.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `sqrt`, a bounded `for` loop and `+ - * /` — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, no inverse trigonometry, no `round`, and no bare f32
//! equality. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. The only loop is the fixed twelve-step
//! squaring in `exp_approx`, so the kernel provably terminates.
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

/// The portable core-`WGSL` underwater-scatter kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` evaluates
/// the three `CPU` golden
/// [`underwater`](prism_render_architecture::water::underwater) closed forms per
/// query; see the module documentation for the algorithm.
const WATER_UNDERWATER_SCATTER_WGSL: &str = r#"
// wgpu compute twin of prism_render_architecture::water::underwater: the three
// stateless per-point scatter kernels (henyey_greenstein, multiple_scatter_boost,
// godray_inscatter), with only clamp, min, max, sqrt, a bounded for loop and
// + - * /. The per-channel colour shift and the visibility predicate stay on the
// host.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::underwater；无第三方
// 引擎源码或衍生代码。

// Shared non-negative epsilon, matching water::EPS.
const EPS: f32 = 1e-6;
// Shared circle constant, matching water::PI (core::f32::consts::PI rounded to
// the nearest f32).
const PI: f32 = 3.14159265358979;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // henyey_greenstein drivers.
    cos_theta: f32,
    g: f32,
    // multiple_scatter_boost drivers.
    single: f32,
    albedo: f32,
    // godray_inscatter drivers.
    surface_light: f32,
    scatter_albedo: f32,
    extinction: f32,
    path_length: f32,
}

struct Result {
    // henyey_greenstein phase value.
    phase: f32,
    // multiple_scatter_boost amplified radiance.
    scatter_boost: f32,
    // godray_inscatter in-scattered radiance.
    inscatter: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Monotone exponential envelope, reproduced exactly from water::exp_approx:
// base = 1 + x / 4096, floored at zero, squared twelve times (2^12 = 4096).
// Only add and multiply, so it is portable and deterministic.
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

// Beer-Lambert transmittance over a path, floored inputs, clamped to 0..=1.
fn beer_lambert_transmittance(extinction: f32, distance: f32) -> f32 {
    let e = max(extinction, 0.0);
    let d = max(distance, 0.0);
    return clamp(exp_approx(-(e * d)), 0.0, 1.0);
}

// Henyey-Greenstein phase: (1 - g^2) / (4*PI * denom_base^(3/2)), with the 3/2
// power as denom_base * sqrt(denom_base). g clamped just inside (-1, 1).
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let gg = clamp(g, -0.999, 0.999);
    let c = clamp(cos_theta, -1.0, 1.0);
    let denom_base = max(1.0 + gg * gg - 2.0 * gg * c, EPS);
    let denom = 4.0 * PI * denom_base * sqrt(denom_base);
    return (1.0 - gg * gg) / denom;
}

// Bounded multiple-scattering boost: single / (1 - albedo), albedo clamped
// below one and single floored at zero.
fn multiple_scatter_boost(single: f32, albedo: f32) -> f32 {
    let a = clamp(albedo, 0.0, 0.999);
    return max(single, 0.0) / (1.0 - a);
}

// God-ray in-scatter: surface_light * scatter_albedo * (1 - transmittance).
fn godray_inscatter(
    surface_light: f32,
    scatter_albedo: f32,
    extinction: f32,
    path_length: f32,
) -> f32 {
    let light = max(surface_light, 0.0);
    let albedo = clamp(scatter_albedo, 0.0, 1.0);
    let t = beer_lambert_transmittance(extinction, path_length);
    return light * albedo * (1.0 - t);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.phase = henyey_greenstein(q.cos_theta, q.g);
    out.scatter_boost = multiple_scatter_boost(q.single, q.albedo);
    out.inscatter = godray_inscatter(q.surface_light, q.scatter_albedo, q.extinction, q.path_length);
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words,
/// filling a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_UNDERWATER_SCATTER_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// eight `f32` drivers, one for each argument the three twinned kernels read.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Phase cosine `cos_theta`.
    cos_theta: f32,
    /// Phase asymmetry `g`.
    g: f32,
    /// Single-scatter radiance `single`.
    single: f32,
    /// Single-scatter albedo `albedo`.
    albedo: f32,
    /// Surface light `surface_light`.
    surface_light: f32,
    /// God-ray scatter albedo `scatter_albedo`.
    scatter_albedo: f32,
    /// Extinction coefficient `extinction`.
    extinction: f32,
    /// Shaft path length `path_length`.
    path_length: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the three scalar outputs and one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `henyey_greenstein` phase value.
    phase: f32,
    /// `multiple_scatter_boost` amplified radiance.
    scatter_boost: f32,
    /// `godray_inscatter` in-scattered radiance.
    inscatter: f32,
    /// Padding word.
    pad0: f32,
}

/// One underwater-scatter query: the per-point drivers for all three kernels.
///
/// Mirrors the arguments the matching `CPU` golden
/// [`underwater`](prism_render_architecture::water::underwater) functions read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterUnderwaterScatterQuery {
    /// Phase cosine `cos_theta` (twins `henyey_greenstein`).
    pub cos_theta: f32,
    /// Phase asymmetry `g` in `(-1, 1)` (twins `henyey_greenstein`).
    pub g: f32,
    /// Single-scatter radiance `single` (twins `multiple_scatter_boost`).
    pub single: f32,
    /// Single-scatter albedo `albedo` (twins `multiple_scatter_boost`).
    pub albedo: f32,
    /// Surface light `surface_light` (twins `godray_inscatter`).
    pub surface_light: f32,
    /// God-ray scatter albedo `scatter_albedo` (twins `godray_inscatter`).
    pub scatter_albedo: f32,
    /// Extinction coefficient `extinction` (twins `godray_inscatter`).
    pub extinction: f32,
    /// Shaft path length `path_length` (twins `godray_inscatter`).
    pub path_length: f32,
}

/// One underwater-scatter result, mirroring the values the matching `CPU`
/// golden returns.
///
/// Holds the `henyey_greenstein` phase, the `multiple_scatter_boost` radiance,
/// and the `godray_inscatter` in-scatter together, one per input query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterUnderwaterScatterResult {
    /// Output of `henyey_greenstein`.
    pub phase: f32,
    /// Output of `multiple_scatter_boost`.
    pub scatter_boost: f32,
    /// Output of `godray_inscatter`.
    pub inscatter: f32,
}

/// Encodes one [`WaterUnderwaterScatterQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &WaterUnderwaterScatterQuery) -> GpuQuery {
    GpuQuery {
        cos_theta: q.cos_theta,
        g: q.g,
        single: q.single,
        albedo: q.albedo,
        surface_light: q.surface_light,
        scatter_albedo: q.scatter_albedo,
        extinction: q.extinction,
        path_length: q.path_length,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterUnderwaterScatterResult`].
fn decode_result(raw: &GpuResult) -> WaterUnderwaterScatterResult {
    WaterUnderwaterScatterResult {
        phase: raw.phase,
        scatter_boost: raw.scatter_boost,
        inscatter: raw.inscatter,
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

/// A compiled, reusable underwater-scatter compute pipeline, twinning the three
/// stateless per-point kernels of the `CPU` golden
/// [`underwater`](prism_render_architecture::water::underwater).
pub struct GpuWaterUnderwaterScatter {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterUnderwaterScatter {
    /// Compiles the underwater-scatter kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterUnderwaterScatter {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_underwater_scatter"),
            source: ShaderSource::Wgsl(WATER_UNDERWATER_SCATTER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_underwater_scatter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_underwater_scatter_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_underwater_scatter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterUnderwaterScatter {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one
    /// [`WaterUnderwaterScatterResult`] per input, in order.
    ///
    /// Each result equals the matching `CPU` golden
    /// [`underwater`](prism_render_architecture::water::underwater) outcome
    /// within the tolerance documented on this module. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterUnderwaterScatterQuery],
    ) -> Vec<WaterUnderwaterScatterResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_underwater_scatter_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_underwater_scatter_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_underwater_scatter_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_underwater_scatter_bind_group"),
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
            label: Some("prism_volumetric_water_underwater_scatter_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_underwater_scatter_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_underwater_scatter_pass"),
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
