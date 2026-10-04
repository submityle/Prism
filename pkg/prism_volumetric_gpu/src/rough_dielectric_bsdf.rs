//! `wgpu` compute twin of the rough-dielectric microfacet `BSDF` evaluation and
//! probability density of the reference tracer
//! (`prism_render_architecture::reference_pt::rough_dielectric`).
//!
//! A rough dielectric is frosted glass: the smooth reflect-or-transmit split of
//! a dielectric interface, blurred by a `GGX` (Trowbridge-Reitz) microfacet
//! distribution. Three physical pieces drive each lobe: the `GGX` normal
//! distribution `D(wm)`, the height-correlated Smith masking-shadowing `G2`,
//! and the unpolarized dielectric `Fresnel` reflectance `F` that splits energy
//! between the reflected and transmitted microfacet lobes. The reflected lobe
//! uses the ordinary half vector `wm proportional to wi + wo`; the transmitted
//! lobe uses the generalized half vector `wm proportional to etap*wi + wo`
//! whose change-of-variables Jacobian carries the
//! `1 / (etap*wi.wm + wo.wm)^2` compression, with an additional `1 / etap^2`
//! radiance-mode solid-angle compression.
//!
//! This module is the on-device twin of that oracle's stateless, `RNG`-free
//! core:
//!
//! - `evaluate`: the full `BSDF` value `f(wo, wi)` for a fixed direction pair,
//!   returning the per-channel reflected or transmitted tint scaled by the
//!   microfacet lobe.
//! - `pdf`: the solid-angle density the importance sampler would assign to
//!   `(wo, wi)`, a proper mixture of the visible-normal density, the
//!   reflect/refract Jacobian and the `Fresnel`-proportional lobe-selection
//!   probability.
//!
//! [`GpuRoughDielectric`] evaluates both for one direction pair per thread,
//! reproducing the reference's exact closed form — only `sqrt`, `abs`, `clamp`,
//! products and quotients, no transcendental — so a passing real-device parity
//! test is direct evidence the ported kernel computes the same scattering the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`RoughDielectricQuery`] — the relative index `ior`,
//! the perceptual `roughness`, the per-channel reflected and transmitted tints,
//! and the view, light and shading-normal directions — and writes one
//! [`RoughDielectricResult`] holding the three-channel `BSDF` value, the scalar
//! `pdf`, a `valid` flag and the `reflect_flag` lobe discriminant. The kernel
//! forms the signed cosines off the dot products with the shading normal,
//! chooses the reflected or transmitted lobe, builds the (generalized) half
//! vector, applies the `GGX` distribution, the Smith `G2`/`G1` terms and the
//! dielectric `Fresnel`, then assembles the lobe value and the mixture density.
//!
//! # What stays on the host
//!
//! The visible-normal importance sampler (which needs an `RNG`), the path
//! throughput assembly and the material graph all stay on the host; the device
//! sees only the stateless, fixed-width `evaluate`/`pdf` pair, one direction
//! pair at a time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Both outputs thread through `sqrt`, products and quotients, so the `CPU` and
//! `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units in the
//! last place from the scalar reference. The parity test asserts each continuous
//! output within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative floor
//! `1e-6`), and the discrete `valid` and `reflect_flag` words exactly. Grazing
//! incidence, a degenerate half vector and a back-facing microfacet all
//! short-circuit to a zero value, a zero density and `valid = 0`, matching the
//! reference's early returns.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`,
//! `clamp`, `min`, `max`, `select`, `+ - * /` and unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry,
//! no `round` and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::rough_dielectric`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` rough-dielectric kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `reference_pt::rough_dielectric::RoughDielectric::{evaluate,
/// pdf}` closed forms; see the module documentation for the algorithm.
const ROUGH_DIELECTRIC_BSDF_WGSL: &str = r#"
// Rough-dielectric BSDF twin: one thread computes one direction pair's
// microfacet reflect/transmit BSDF value plus its mixture probability density,
// mirroring the CPU golden
// `reference_pt::rough_dielectric::RoughDielectric::{evaluate, pdf}` with only
// sqrt, abs, clamp, products, quotients and select. The RNG sampler and the
// path assembly stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::rough_dielectric；无第三方
// 引擎源码或衍生代码。

// A direction cosine below this magnitude is a grazing degeneracy.
const COS_EPS: f32 = 1.0e-8;
// Squared-length floor below which a vector is treated as the zero vector.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Smallest GGX width; below this the lobe is a numerical mirror.
const MIN_ALPHA: f32 = 1.0e-3;
// Reciprocal of pi, matching the host oracle's literal.
const INV_PI: f32 = 0.3183098861837907;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Relative index of refraction eta_t / eta_i (interior over exterior).
    ior: f32,
    // Perceptual roughness in [0, 1], remapped to alpha = roughness^2.
    roughness: f32,
    // Per-channel reflected-lobe tint.
    reflectance_x: f32,
    reflectance_y: f32,
    reflectance_z: f32,
    // Per-channel transmitted-lobe tint.
    transmittance_x: f32,
    transmittance_y: f32,
    transmittance_z: f32,
    // View direction.
    wo_x: f32,
    wo_y: f32,
    wo_z: f32,
    // Light direction.
    wi_x: f32,
    wi_y: f32,
    wi_z: f32,
    // Shading normal (the frame +z axis).
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Outcome {
    // Three-channel BSDF value f(wo, wi).
    value_x: f32,
    value_y: f32,
    value_z: f32,
    // Solid-angle probability density.
    pdf: f32,
    // 1 when the direction pair is non-degenerate, else 0.
    valid: u32,
    // 1 when the lobe is a reflection (same side of the normal), else 0.
    reflect_flag: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outcome>;

// Returns x squared, spelled out so the banned powi/powf are avoided.
fn sqr(x: f32) -> f32 {
    return x * x;
}

// Returns the unit vector along v, or the zero vector when v is numerically the
// zero vector, so normalization never yields NaN.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    let scaled = v * (1.0 / sqrt(max(len_sq, EPS_LEN_SQ)));
    return select(vec3<f32>(0.0, 0.0, 0.0), scaled, len_sq > EPS_LEN_SQ);
}

// Flips a so it lies in the same hemisphere as the reference direction.
fn faced_toward(a: vec3<f32>, reference: vec3<f32>) -> vec3<f32> {
    return select(a, -a, dot(a, reference) < 0.0);
}

// The GGX normal distribution D for a half vector whose cosine to the normal is
// cos_h (passed as an absolute cosine). Zero for a back-facing half vector.
fn ggx_distribution(alpha: f32, cos_h: f32) -> f32 {
    if (cos_h <= 0.0) {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let c2 = cos_h * cos_h;
    let denom = c2 * (a2 - 1.0) + 1.0;
    return a2 * INV_PI / (denom * denom);
}

// The Smith Lambda auxiliary for a direction whose cosine to the normal is
// cos_w. Normal incidence returns zero; callers guard |cos_w| >= COS_EPS.
fn ggx_lambda(alpha: f32, cos_w: f32) -> f32 {
    let c = abs(cos_w);
    if (c >= 1.0) {
        return 0.0;
    }
    let c2 = c * c;
    let tan2 = (1.0 - c2) / c2;
    let a2 = alpha * alpha;
    return 0.5 * (sqrt(1.0 + a2 * tan2) - 1.0);
}

// The Smith single-direction masking term G1 in [0, 1].
fn ggx_g1(alpha: f32, cos_w: f32) -> f32 {
    return 1.0 / (1.0 + ggx_lambda(alpha, cos_w));
}

// The height-correlated Smith masking-shadowing term G2.
fn ggx_g2(alpha: f32, cos_o: f32, cos_i: f32) -> f32 {
    return 1.0 / (1.0 + ggx_lambda(alpha, cos_o) + ggx_lambda(alpha, cos_i));
}

// The unpolarized Fresnel reflectance of a dielectric interface in the relative
// index form. A negative cos_i strikes the back of the microfacet, inverting
// the index ratio; beyond the critical angle it returns 1 (total internal
// reflection).
fn fr_dielectric(cos_i_in: f32, eta_in: f32) -> f32 {
    var cos_i = clamp(cos_i_in, -1.0, 1.0);
    var eta = eta_in;
    if (cos_i < 0.0) {
        eta = 1.0 / eta;
        cos_i = -cos_i;
    }
    let sin2_i = max(1.0 - cos_i * cos_i, 0.0);
    let sin2_t = sin2_i / (eta * eta);
    if (sin2_t >= 1.0) {
        return 1.0;
    }
    let cos_t = sqrt(max(1.0 - sin2_t, 0.0));
    let r_parl = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (r_parl * r_parl + r_perp * r_perp);
}

// The visible-normal (VNDF) solid-angle density of the microfacet normal wm for
// the view direction wo. Callers guard |cos_o| >= COS_EPS.
fn visible_normal_pdf(alpha: f32, wo: vec3<f32>, wm: vec3<f32>, normal: vec3<f32>) -> f32 {
    let cos_o = dot(normal, wo);
    if (abs(cos_o) < COS_EPS) {
        return 0.0;
    }
    let d = ggx_distribution(alpha, abs(dot(normal, wm)));
    let g1 = ggx_g1(alpha, cos_o);
    return d * g1 * abs(dot(wo, wm)) / abs(cos_o);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let reflectance = vec3<f32>(q.reflectance_x, q.reflectance_y, q.reflectance_z);
    let transmittance = vec3<f32>(q.transmittance_x, q.transmittance_y, q.transmittance_z);
    let wo = vec3<f32>(q.wo_x, q.wo_y, q.wo_z);
    let wi = vec3<f32>(q.wi_x, q.wi_y, q.wi_z);
    let normal = vec3<f32>(q.normal_x, q.normal_y, q.normal_z);
    let ior = q.ior;

    let rough = clamp(q.roughness, 0.0, 1.0);
    let alpha = max(rough * rough, MIN_ALPHA);

    let cos_o = dot(normal, wo);
    let cos_i = dot(normal, wi);
    let reflect = cos_i * cos_o > 0.0;

    var value = vec3<f32>(0.0, 0.0, 0.0);
    var pdf: f32 = 0.0;
    var valid: u32 = 0u;

    if (abs(cos_o) >= COS_EPS && abs(cos_i) >= COS_EPS) {
        // etap: 1 for reflection; ior or 1/ior for transmission by the side wo
        // sits on.
        let etap_trans = select(1.0 / ior, ior, cos_o > 0.0);
        let etap = select(etap_trans, 1.0, reflect);

        let wm_raw = wi * etap + wo;
        let wm_len_sq = dot(wm_raw, wm_raw);
        if (wm_len_sq > EPS_LEN_SQ) {
            let wm = faced_toward(normalize_or_zero(wm_raw), normal);
            // Discard pairs that lie behind the chosen microfacet.
            let behind = dot(wm, wi) * cos_i < 0.0 || dot(wm, wo) * cos_o < 0.0;
            if (!behind) {
                valid = 1u;
                let d = ggx_distribution(alpha, abs(dot(normal, wm)));
                let g2 = ggx_g2(alpha, cos_o, cos_i);
                let f = fr_dielectric(dot(wo, wm), ior);

                if (reflect) {
                    let denom = 4.0 * abs(cos_i * cos_o);
                    if (denom > 0.0) {
                        value = reflectance * (d * g2 * f / denom);
                    }
                } else {
                    let denom = sqr(dot(wi, wm) + dot(wo, wm) / etap) * cos_i * cos_o;
                    if (abs(denom) >= COS_EPS) {
                        let ft = d * (1.0 - f) * g2
                            * abs(dot(wi, wm) * dot(wo, wm) / denom) / sqr(etap);
                        value = transmittance * ft;
                    }
                }

                let pr = f;
                let pt = 1.0 - f;
                if (pr + pt > 0.0) {
                    let vndf = visible_normal_pdf(alpha, wo, wm, normal);
                    if (reflect) {
                        let woh = abs(dot(wo, wm));
                        if (woh >= COS_EPS) {
                            pdf = vndf / (4.0 * woh) * pr / (pr + pt);
                        }
                    } else {
                        let denom2 = sqr(dot(wi, wm) + dot(wo, wm) / etap);
                        if (denom2 > 0.0) {
                            let dwm_dwi = abs(dot(wi, wm)) / denom2;
                            pdf = vndf * dwm_dwi * pt / (pr + pt);
                        }
                    }
                }
            }
        }
    }

    var out: Outcome;
    out.value_x = value.x;
    out.value_y = value.y;
    out.value_z = value.z;
    out.pdf = pdf;
    out.valid = valid;
    out.reflect_flag = select(0u, 1u, reflect);
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte-aligned uniform struct matching `Params` in
/// [`ROUGH_DIELECTRIC_BSDF_WGSL`].
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
/// the relative index, roughness, two tints and three directions, plus three
/// pad words to an `80`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Relative index of refraction `eta_t / eta_i`.
    ior: f32,
    /// Perceptual roughness in `[0, 1]`.
    roughness: f32,
    /// Reflected-lobe tint, `x` channel.
    reflectance_x: f32,
    /// Reflected-lobe tint, `y` channel.
    reflectance_y: f32,
    /// Reflected-lobe tint, `z` channel.
    reflectance_z: f32,
    /// Transmitted-lobe tint, `x` channel.
    transmittance_x: f32,
    /// Transmitted-lobe tint, `y` channel.
    transmittance_y: f32,
    /// Transmitted-lobe tint, `z` channel.
    transmittance_z: f32,
    /// View direction `x`.
    wo_x: f32,
    /// View direction `y`.
    wo_y: f32,
    /// View direction `z`.
    wo_z: f32,
    /// Light direction `x`.
    wi_x: f32,
    /// Light direction `y`.
    wi_y: f32,
    /// Light direction `z`.
    wi_z: f32,
    /// Shading normal `x`.
    normal_x: f32,
    /// Shading normal `y`.
    normal_y: f32,
    /// Shading normal `z`.
    normal_z: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Outcome`
/// struct: the three-channel value, the density, the `valid` and `reflect_flag`
/// words, plus two pad words to a `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `BSDF` value, `x` channel.
    value_x: f32,
    /// `BSDF` value, `y` channel.
    value_y: f32,
    /// `BSDF` value, `z` channel.
    value_z: f32,
    /// Solid-angle probability density.
    pdf: f32,
    /// `1` when the direction pair is non-degenerate, else `0`.
    valid: u32,
    /// `1` when the lobe is a reflection, else `0`.
    reflect_flag: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One query for the rough-dielectric twin: the interface's relative index and
/// roughness, the reflected and transmitted tints, and the view, light and
/// shading-normal directions.
///
/// `ior` is the relative index `eta_t / eta_i`; `roughness` is remapped by the
/// kernel to the `GGX` width `alpha = roughness^2`. The direction triples are
/// taken as supplied (the host normalizes them); the kernel reads the signed
/// cosines off the dot products with `normal`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoughDielectricQuery {
    /// Relative index of refraction `eta_t / eta_i`.
    pub ior: f32,
    /// Perceptual roughness in `[0, 1]`.
    pub roughness: f32,
    /// Per-channel reflected-lobe tint.
    pub reflectance: [f32; 3],
    /// Per-channel transmitted-lobe tint.
    pub transmittance: [f32; 3],
    /// View direction.
    pub wo: [f32; 3],
    /// Light direction.
    pub wi: [f32; 3],
    /// Shading normal (the frame `+z` axis).
    pub normal: [f32; 3],
}

impl RoughDielectricQuery {
    /// Builds a query from the interface parameters and the three directions.
    #[must_use]
    pub const fn new(
        ior: f32,
        roughness: f32,
        reflectance: [f32; 3],
        transmittance: [f32; 3],
        wo: [f32; 3],
        wi: [f32; 3],
        normal: [f32; 3],
    ) -> RoughDielectricQuery {
        RoughDielectricQuery {
            ior,
            roughness,
            reflectance,
            transmittance,
            wo,
            wi,
            normal,
        }
    }
}

/// One resolved query of the rough-dielectric twin: the three-channel `BSDF`
/// value, the mixture density, the degeneracy flag and the lobe discriminant.
///
/// `value` is `RoughDielectric::evaluate`; `pdf` is `RoughDielectric::pdf`;
/// `valid` is `0` when the direction pair short-circuits (grazing, a degenerate
/// half vector or a back-facing microfacet), in which case `value` is zero and
/// `pdf` is zero. `reflect_flag` is `1` when `wo` and `wi` lie on the same side
/// of `normal` (a reflection lobe), else `0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoughDielectricResult {
    /// Three-channel `BSDF` value.
    pub value: [f32; 3],
    /// Solid-angle probability density.
    pub pdf: f32,
    /// `1` when the direction pair is non-degenerate, else `0`.
    pub valid: u32,
    /// `1` when the lobe is a reflection, else `0`.
    pub reflect_flag: u32,
}

/// Encodes one [`RoughDielectricQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &RoughDielectricQuery) -> GpuQuery {
    GpuQuery {
        ior: q.ior,
        roughness: q.roughness,
        reflectance_x: q.reflectance[0],
        reflectance_y: q.reflectance[1],
        reflectance_z: q.reflectance[2],
        transmittance_x: q.transmittance[0],
        transmittance_y: q.transmittance[1],
        transmittance_z: q.transmittance[2],
        wo_x: q.wo[0],
        wo_y: q.wo[1],
        wo_z: q.wo[2],
        wi_x: q.wi[0],
        wi_y: q.wi[1],
        wi_z: q.wi[2],
        normal_x: q.normal[0],
        normal_y: q.normal[1],
        normal_z: q.normal[2],
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`RoughDielectricResult`].
fn decode_result(raw: &GpuResult) -> RoughDielectricResult {
    RoughDielectricResult {
        value: [raw.value_x, raw.value_y, raw.value_z],
        pdf: raw.pdf,
        valid: raw.valid,
        reflect_flag: raw.reflect_flag,
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

/// A compiled, reusable rough-dielectric compute pipeline, twinning the `CPU`
/// golden `reference_pt::rough_dielectric::RoughDielectric::{evaluate, pdf}`.
pub struct GpuRoughDielectric {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRoughDielectric {
    /// Compiles the rough-dielectric kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRoughDielectric {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_rough_dielectric_bsdf"),
            source: ShaderSource::Wgsl(ROUGH_DIELECTRIC_BSDF_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_rough_dielectric_bsdf_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_rough_dielectric_bsdf_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_rough_dielectric_bsdf_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRoughDielectric {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`RoughDielectricResult`] per input, in order.
    ///
    /// The values and densities match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RoughDielectricQuery],
    ) -> Vec<RoughDielectricResult> {
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
            label: Some("prism_volumetric_rough_dielectric_bsdf_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rough_dielectric_bsdf_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_rough_dielectric_bsdf_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_rough_dielectric_bsdf_bind_group"),
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
            label: Some("prism_volumetric_rough_dielectric_bsdf_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_rough_dielectric_bsdf_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_rough_dielectric_bsdf_pass"),
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
