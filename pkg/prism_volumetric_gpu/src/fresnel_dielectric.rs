//! `wgpu` compute twin of the smooth-dielectric Fresnel reflectance
//! (`crate`-external golden `reference_pt::dielectric::fresnel_dielectric`).
//!
//! A dielectric interface — glass, water, a clear coat — reflects a
//! view-angle-dependent fraction of the incident energy and transmits the rest.
//! The golden `fresnel_dielectric` returns that unpolarized reflectance for an
//! interface between two real refractive indices, including the total internal
//! reflection (`TIR`) regime where all energy is reflected.
//!
//! # What is twinned
//!
//! One thread resolves one query. For an incident cosine `cos_i` and the two
//! indices `eta_i` (incident side) and `eta_t` (transmitted side), the kernel
//! clamps `cos_i` into `0..=1`, forms the relative index `eta = eta_i / eta_t`,
//! and applies Snell's law in squared-sine form:
//! `sin2_t = eta * eta * max(1 - cos_i * cos_i, 0)`. When `sin2_t >= 1` the
//! interface is past the critical angle and the reflectance is exactly `1`;
//! otherwise it forms `cos_t = sqrt(max(1 - sin2_t, 0))`, the parallel and
//! perpendicular amplitude coefficients `r_parl` and `r_perp`, and returns the
//! mean of their squares. The twin spells out the same closed form with the
//! same ordered compares as the reference, so a passing real-device parity test
//! is direct evidence the ported kernel reflects identically, not merely that
//! the shader compiles.
//!
//! # What stays on the host
//!
//! Nothing of the per-query math stays on the host: the whole evaluation is a
//! fixed, bounded sequence of arithmetic and one `sqrt` that runs on device.
//! The host only flattens the query batch into a `std430` storage buffer and
//! short-circuits an empty batch (a storage buffer cannot be zero-sized).
//!
//! # Correctness model
//!
//! The reflectance is a *continuous* quantity, so the parity test compares with
//! an absolute-or-relative tolerance (`abs <= 1e-5 || rel <= 1e-4`). The `TIR`
//! branch folds into that continuous output — the reflectance is exactly `1`
//! there — and fixtures sit well clear of the critical angle so the branch
//! resolves the same way on both sides.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `max`,
//! `sqrt`, `+ - * /` and ordered comparisons — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `round` and no
//! `u64`/`u16`/`i64`/`f64`. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread
//! performs a fixed, bounded sequence of arithmetic, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture` 的 `reference_pt::dielectric::fresnel_dielectric`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Fresnel-reflectance kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the golden `reference_pt::dielectric::fresnel_dielectric`; see the module
/// documentation for the algorithm.
const FRESNEL_DIELECTRIC_WGSL: &str = r#"
// Smooth-dielectric Fresnel reflectance twin: one thread evaluates one
// interface, mirroring the CPU golden
// `reference_pt::dielectric::fresnel_dielectric` with only clamp/max/sqrt,
// + - * / and ordered comparisons. Past the critical angle (sin2_t >= 1) the
// reflectance is exactly 1 (total internal reflection).
//
// Provenance: 孪生自本仓 prism_render_architecture 的
// reference_pt::dielectric::fresnel_dielectric；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Cosine of the incident angle on the incident side (clamped to 0..=1).
    cos_i: f32,
    // Refractive index of the medium the light arrives through.
    eta_i: f32,
    // Refractive index of the medium the light would enter.
    eta_t: f32,
    pad0: f32,
}

struct FresnelResult {
    // Unpolarized reflectance in 0..=1 (exactly 1 under total internal
    // reflection).
    reflectance: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<FresnelResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Clamp the incident cosine, then Snell's law in squared-sine form.
    let cos_i = clamp(q.cos_i, 0.0, 1.0);
    let eta = q.eta_i / q.eta_t;
    let sin2_i = max(1.0 - cos_i * cos_i, 0.0);
    let sin2_t = eta * eta * sin2_i;

    // Default to total internal reflection; refine when below the critical
    // angle. Ordered compare `< 1.0` mirrors the reference `>= 1.0` early-out.
    var reflectance: f32 = 1.0;
    if (sin2_t < 1.0) {
        let cos_t = sqrt(max(1.0 - sin2_t, 0.0));
        let r_parl = (q.eta_t * cos_i - q.eta_i * cos_t)
            / (q.eta_t * cos_i + q.eta_i * cos_t);
        let r_perp = (q.eta_i * cos_i - q.eta_t * cos_t)
            / (q.eta_i * cos_i + q.eta_t * cos_t);
        reflectance = 0.5 * (r_parl * r_parl + r_perp * r_perp);
    }

    var out: FresnelResult;
    out.reflectance = reflectance;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`FRESNEL_DIELECTRIC_WGSL`].
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

/// `repr(C)` `std430` layout of one Fresnel query: the incident cosine and the
/// two refractive indices, matching the `WGSL` `Query` struct's `16`-byte
/// stride (three `f32` plus one pad word).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Cosine of the incident angle on the incident side.
    cos_i: f32,
    /// Refractive index of the incident medium.
    eta_i: f32,
    /// Refractive index of the transmitted medium.
    eta_t: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one Fresnel result, matching the `WGSL`
/// `FresnelResult` struct: the reflectance plus three pad words to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Unpolarized reflectance in `0..=1`.
    reflectance: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// One Fresnel query: the incident cosine `cos_i` and the two refractive
/// indices `eta_i` (incident side) and `eta_t` (transmitted side).
///
/// The fields mirror the golden `reference_pt::dielectric::fresnel_dielectric`
/// arguments; the host enqueues one query per interface sample, and an empty
/// batch is short-circuited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FresnelDielectricQuery {
    /// Cosine of the incident angle on the incident side.
    pub cos_i: f32,
    /// Refractive index of the incident medium.
    pub eta_i: f32,
    /// Refractive index of the transmitted medium.
    pub eta_t: f32,
}

impl FresnelDielectricQuery {
    /// Builds a query from an incident cosine and the two refractive indices.
    #[must_use]
    pub const fn new(cos_i: f32, eta_i: f32, eta_t: f32) -> FresnelDielectricQuery {
        FresnelDielectricQuery {
            cos_i,
            eta_i,
            eta_t,
        }
    }
}

/// One resolved Fresnel reflectance: the unpolarized fraction of energy
/// reflected at the interface (exactly `1` under total internal reflection).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FresnelDielectricResult {
    /// Unpolarized reflectance in `0..=1`.
    pub reflectance: f32,
}

/// Encodes one [`FresnelDielectricQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &FresnelDielectricQuery) -> GpuQuery {
    GpuQuery {
        cos_i: q.cos_i,
        eta_i: q.eta_i,
        eta_t: q.eta_t,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`FresnelDielectricResult`].
fn decode_result(raw: &GpuResult) -> FresnelDielectricResult {
    FresnelDielectricResult {
        reflectance: raw.reflectance,
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

/// A compiled, reusable Fresnel-reflectance compute pipeline, twinning the
/// golden `reference_pt::dielectric::fresnel_dielectric`.
pub struct GpuFresnelDielectric {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFresnelDielectric {
    /// Compiles the Fresnel-reflectance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFresnelDielectric {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fresnel_dielectric"),
            source: ShaderSource::Wgsl(FRESNEL_DIELECTRIC_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fresnel_dielectric_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fresnel_dielectric_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fresnel_dielectric_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFresnelDielectric {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`FresnelDielectricResult`] per input, in order.
    ///
    /// The reflectance equals the reference within floating-point tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[FresnelDielectricQuery],
    ) -> Vec<FresnelDielectricResult> {
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
            label: Some("prism_volumetric_fresnel_dielectric_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fresnel_dielectric_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fresnel_dielectric_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fresnel_dielectric_bind_group"),
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
            label: Some("prism_volumetric_fresnel_dielectric_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fresnel_dielectric_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fresnel_dielectric_pass"),
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
