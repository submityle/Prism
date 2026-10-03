//! `wgpu` compute twin of the physically based water shading planner
//! ([`plan_pbr`](prism_render_architecture::water::shading::pbr::plan_pbr)).
//!
//! The water `PBR` frontend resolves, per surface sample, a complete lighting
//! response from a static [`PbrShadingParams`](prism_render_architecture::water::shading::PbrShadingParams)
//! tuning slice and the per-view
//! [`SurfaceShadingInputs`](prism_render_architecture::water::shading::SurfaceShadingInputs):
//! the base reflectance `F0` (either an explicit override or the `Schlick`
//! relation from the index of refraction), the `Schlick`/`Fresnel` reflectance
//! at the view angle, the three-tier `SSR` -> `RT` -> probe reflection routing,
//! the micro-surface foam mask from the surface `Jacobian`, the foam-roughened
//! `GGX` roughness, and the grazing subsurface back-transmission. The golden
//! [`plan_pbr`](prism_render_architecture::water::shading::pbr::plan_pbr) is a
//! pure, deterministic function of its two argument structs, so a passing
//! real-device parity test is direct evidence the ported planner reproduces the
//! reference response, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread resolves one surface sample. The thread reproduces the golden's
//! closed form field by field:
//!
//! - `F0` is `f0_override` when it exceeds the shared epsilon, otherwise the
//!   dielectric relation `((ior - 1) / (ior + 1))^2`, the device mirror of the
//!   golden's `f0_from_ior`;
//! - the `Fresnel` term is `F0 + (1 - F0) * (1 - cos)^5` with `cos` clamped to
//!   `0..=1` and the fifth power expanded to an explicit product, the device
//!   mirror of the golden's `fresnel_schlick`;
//! - the reflection tier is `0` (`ScreenSpace`) when `ssr_confidence` meets
//!   `ssr_min_confidence`, else `1` (`RayTraced`) when `ray_budget` meets
//!   `rt_min_budget`, else `2` (`Probe`);
//! - the foam mask ramps `clamp((foam_fold_threshold - jacobian) /
//!   foam_fold_threshold, 0, 1)` when the fold threshold exceeds the epsilon,
//!   otherwise `0`;
//! - the specular roughness is `clamp(base + foam * (1 - base), 0, 1)` with the
//!   base clamped to `0..=1`;
//! - the subsurface transmission is `clamp(grazing_transmission * (1 - cos),
//!   0, 1)`.
//!
//! The reflection-tier code is the discriminant order of
//! [`ReflectionTier`](prism_render_architecture::water::shading::ReflectionTier):
//! `0` = `ScreenSpace`, `1` = `RayTraced`, `2` = `Probe`. It is a chain of
//! ordered comparisons and so is reproduced exactly.
//!
//! # What stays on the host
//!
//! Only the per-sample numeric planner is twinned. The variable-length frame of
//! samples, the choice of which samples to shade, and the sibling `npr` and
//! `hybrid` frontends stay on the host. The device returns one response per
//! input sample; the host uses it exactly as it would use the return value of
//! [`plan_pbr`](prism_render_architecture::water::shading::pbr::plan_pbr). An
//! empty sample batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! Every continuous field is a bounded sequence of `+`, `-`, `*`, `/` and
//! `clamp` with no transcendental call, so the `CPU` and `GPU` agree to within
//! the documented tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`). The
//! reflection-tier code is a chain of ordered comparisons and is asserted with
//! `==`. The parity test pins each field against the golden
//! [`plan_pbr`](prism_render_architecture::water::shading::pbr::plan_pbr).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+`, `-`, `*`, `/`,
//! `clamp` and ordered compares — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, no inverse trigonometry, no `sqrt`, no `round`, and no `u64`. No
//! optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. There is no loop: each thread performs a fixed, bounded
//! sequence of work, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::shading::pbr::plan_pbr`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` water-`PBR` planner kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`plan_pbr`](prism_render_architecture::water::shading::pbr::plan_pbr)
/// per-sample response; see the module documentation for the algorithm.
const WATER_SHADING_PBR_WGSL: &str = r#"
// Water PBR shading planner twin: one thread resolves one surface sample,
// mirroring the CPU golden `water::shading::pbr::plan_pbr`. It derives F0 from
// an override or the dielectric IOR relation, the Schlick Fresnel term (fifth
// power as an explicit product), the SSR -> RT -> probe reflection tier, the
// Jacobian foam mask, the foam-roughened GGX roughness, and the grazing
// subsurface transmission. Only +, -, *, /, clamp and ordered compares; no
// floating-point ==, no u64, no transcendental, no sqrt. The variable-length
// frame of samples and the sibling NPR/hybrid frontends stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::shading::pbr::plan_pbr；
// 无第三方引擎源码或衍生代码。

// Shared epsilon matching `water::EPS`, guarding the override and fold branches.
const EPS: f32 = 1.0e-6;

struct Params {
    // Number of surface samples in the storage array; threads past this return.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Explicit base reflectance F0; when at or below EPS, derive F0 from IOR.
    f0_override: f32,
    // Water index of refraction, expected > 1.
    ior: f32,
    // Cosine of the view/normal angle; clamped to 0..=1 before use.
    cos_view: f32,
    // Surface Jacobian driving the micro-surface foam mask.
    jacobian: f32,
    // Jacobian at or above which no foam appears; mask ramps as it drops to 0.
    foam_fold_threshold: f32,
    // Base GGX roughness for calm water; clamped to 0..=1.
    base_roughness: f32,
    // Grazing subsurface back-transmission strength.
    grazing_transmission: f32,
    // Screen-space reflection hit confidence.
    ssr_confidence: f32,
    // Minimum screen-space confidence to accept the SSR tier.
    ssr_min_confidence: f32,
    // Ray-tracing budget availability.
    ray_budget: f32,
    // Minimum ray budget to accept the RT tier when SSR misses.
    rt_min_budget: f32,
    pad0: f32,
}

struct Result {
    // Base reflectance F0 used by the Fresnel term.
    f0: f32,
    // Schlick Fresnel reflectance at the view angle.
    fresnel: f32,
    // Micro-surface foam coverage from the Jacobian fold.
    foam_mask: f32,
    // Effective GGX roughness after foam roughening.
    specular_roughness: f32,
    // Grazing subsurface back-transmission.
    subsurface_transmission: f32,
    // Reflection tier code: 0 = ScreenSpace, 1 = RayTraced, 2 = Probe.
    reflection_tier: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Dielectric base reflectance: ((ior - 1) / (ior + 1))^2.
fn f0_from_ior(ior: f32) -> f32 {
    let n = (ior - 1.0) / (ior + 1.0);
    return n * n;
}

// Schlick Fresnel: F0 + (1 - F0) * (1 - cos)^5, the fifth power an explicit
// product so no `pow` is used.
fn fresnel_schlick(f0: f32, cos: f32) -> f32 {
    let c = clamp(cos, 0.0, 1.0);
    let one_minus = 1.0 - c;
    let p2 = one_minus * one_minus;
    let p5 = p2 * p2 * one_minus;
    return f0 + (1.0 - f0) * p5;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.count) {
        return;
    }
    let q = queries[tid];

    // F0: explicit override above EPS, else the dielectric IOR relation.
    var f0: f32;
    if (q.f0_override > EPS) {
        f0 = q.f0_override;
    } else {
        f0 = f0_from_ior(q.ior);
    }

    let cos_view = clamp(q.cos_view, 0.0, 1.0);
    let fresnel = fresnel_schlick(f0, cos_view);

    // Reflection tier: SSR when confident, else RT when budgeted, else probe.
    var tier: u32;
    if (q.ssr_confidence >= q.ssr_min_confidence) {
        tier = 0u;
    } else if (q.ray_budget >= q.rt_min_budget) {
        tier = 1u;
    } else {
        tier = 2u;
    }

    // Foam mask: ramps as the Jacobian folds below the threshold.
    var foam_mask: f32;
    if (q.foam_fold_threshold > EPS) {
        foam_mask = clamp(
            (q.foam_fold_threshold - q.jacobian) / q.foam_fold_threshold,
            0.0,
            1.0,
        );
    } else {
        foam_mask = 0.0;
    }

    let base_roughness = clamp(q.base_roughness, 0.0, 1.0);
    let specular_roughness = clamp(
        base_roughness + foam_mask * (1.0 - base_roughness),
        0.0,
        1.0,
    );
    let subsurface_transmission = clamp(q.grazing_transmission * (1.0 - cos_view), 0.0, 1.0);

    var out: Result;
    out.f0 = f0;
    out.fresnel = fresnel;
    out.foam_mask = foam_mask;
    out.specular_roughness = specular_roughness;
    out.subsurface_transmission = subsurface_transmission;
    out.reflection_tier = tier;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[tid] = out;
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_SHADING_PBR_WGSL`].
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

/// `repr(C)` `std430` layout of one surface sample: the eleven planner inputs
/// plus one pad word to a `48`-byte, `16`-aligned stride matching the `WGSL`
/// `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Explicit base reflectance override.
    f0_override: f32,
    /// Water index of refraction.
    ior: f32,
    /// Cosine of the view/normal angle.
    cos_view: f32,
    /// Surface `Jacobian`.
    jacobian: f32,
    /// `Jacobian` fold threshold.
    foam_fold_threshold: f32,
    /// Base `GGX` roughness.
    base_roughness: f32,
    /// Grazing subsurface transmission strength.
    grazing_transmission: f32,
    /// Screen-space reflection confidence.
    ssr_confidence: f32,
    /// Minimum confidence for the `SSR` tier.
    ssr_min_confidence: f32,
    /// Ray-tracing budget availability.
    ray_budget: f32,
    /// Minimum budget for the `RT` tier.
    rt_min_budget: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one resolved response: the five continuous
/// fields and the reflection-tier code plus two pad words to a `32`-byte,
/// `16`-aligned stride matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Base reflectance `F0`.
    f0: f32,
    /// `Schlick` `Fresnel` reflectance.
    fresnel: f32,
    /// Micro-surface foam coverage.
    foam_mask: f32,
    /// Foam-roughened `GGX` roughness.
    specular_roughness: f32,
    /// Grazing subsurface transmission.
    subsurface_transmission: f32,
    /// Reflection-tier code (`0` = `ScreenSpace`, `1` = `RayTraced`, `2` =
    /// `Probe`).
    reflection_tier: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One surface-sample query for the twin: the full set of inputs the golden
/// [`plan_pbr`](prism_render_architecture::water::shading::pbr::plan_pbr) reads
/// from its
/// [`PbrShadingParams`](prism_render_architecture::water::shading::PbrShadingParams)
/// and
/// [`SurfaceShadingInputs`](prism_render_architecture::water::shading::SurfaceShadingInputs)
/// arguments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterShadingPbrQuery {
    /// Explicit base reflectance override; derive `F0` from `ior` when at or
    /// below the shared epsilon.
    pub f0_override: f32,
    /// Water index of refraction, expected `> 1`.
    pub ior: f32,
    /// Cosine of the view/normal angle; clamped to `0..=1`.
    pub cos_view: f32,
    /// Surface `Jacobian` driving the foam mask.
    pub jacobian: f32,
    /// `Jacobian` at or above which no foam appears.
    pub foam_fold_threshold: f32,
    /// Base `GGX` roughness for calm water.
    pub base_roughness: f32,
    /// Grazing subsurface back-transmission strength.
    pub grazing_transmission: f32,
    /// Screen-space reflection hit confidence.
    pub ssr_confidence: f32,
    /// Minimum confidence to accept the `SSR` tier.
    pub ssr_min_confidence: f32,
    /// Ray-tracing budget availability.
    pub ray_budget: f32,
    /// Minimum budget to accept the `RT` tier.
    pub rt_min_budget: f32,
}

/// One resolved water-`PBR` response, mirroring the golden
/// [`PbrResponse`](prism_render_architecture::water::shading::PbrResponse).
///
/// `reflection_tier` is the discriminant order of
/// [`ReflectionTier`](prism_render_architecture::water::shading::ReflectionTier):
/// `0` = `ScreenSpace`, `1` = `RayTraced`, `2` = `Probe`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterShadingPbrResult {
    /// Base reflectance `F0` used by the `Fresnel` term.
    pub f0: f32,
    /// `Schlick` `Fresnel` reflectance at the view angle.
    pub fresnel: f32,
    /// Micro-surface foam coverage from the `Jacobian` fold.
    pub foam_mask: f32,
    /// Effective `GGX` roughness after foam roughening.
    pub specular_roughness: f32,
    /// Grazing subsurface back-transmission.
    pub subsurface_transmission: f32,
    /// Selected reflection tier code.
    pub reflection_tier: u32,
}

/// Encodes one [`WaterShadingPbrQuery`] into its `std430` [`GpuQuery`].
fn encode_query(q: &WaterShadingPbrQuery) -> GpuQuery {
    GpuQuery {
        f0_override: q.f0_override,
        ior: q.ior,
        cos_view: q.cos_view,
        jacobian: q.jacobian,
        foam_fold_threshold: q.foam_fold_threshold,
        base_roughness: q.base_roughness,
        grazing_transmission: q.grazing_transmission,
        ssr_confidence: q.ssr_confidence,
        ssr_min_confidence: q.ssr_min_confidence,
        ray_budget: q.ray_budget,
        rt_min_budget: q.rt_min_budget,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterShadingPbrResult`].
fn decode_result(raw: &GpuResult) -> WaterShadingPbrResult {
    WaterShadingPbrResult {
        f0: raw.f0,
        fresnel: raw.fresnel,
        foam_mask: raw.foam_mask,
        specular_roughness: raw.specular_roughness,
        subsurface_transmission: raw.subsurface_transmission,
        reflection_tier: raw.reflection_tier,
    }
}

/// Builds a read-only or read-write storage/uniform binding layout entry.
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

/// A compiled, reusable water-`PBR` planner compute pipeline, twinning the `CPU`
/// golden [`plan_pbr`](prism_render_architecture::water::shading::pbr::plan_pbr).
pub struct GpuWaterShadingPbr {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterShadingPbr {
    /// Compiles the planner kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterShadingPbr {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_shading_pbr"),
            source: ShaderSource::Wgsl(WATER_SHADING_PBR_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_shading_pbr_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_shading_pbr_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_shading_pbr_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterShadingPbr {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every surface sample and returns one [`WaterShadingPbrResult`]
    /// per input, in order.
    ///
    /// Each result equals the reference's per-sample response: the continuous
    /// fields within tolerance and the reflection-tier code exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterShadingPbrQuery],
    ) -> Vec<WaterShadingPbrResult> {
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
            label: Some("prism_volumetric_water_shading_pbr_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_shading_pbr_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_shading_pbr_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_shading_pbr_bind_group"),
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
            label: Some("prism_volumetric_water_shading_pbr_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_shading_pbr_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_shading_pbr_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per surface sample, flattened to a 1-D dispatch.
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
