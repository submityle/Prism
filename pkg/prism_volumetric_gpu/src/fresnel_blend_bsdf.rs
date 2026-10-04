//! `wgpu` compute twin of the coupled diffuse-specular Ashikhmin-Shirley
//! reflection model
//! ([`fresnel_blend`](prism_render_architecture::reference_pt::fresnel_blend)).
//!
//! A clear-coated or painted "plastic" surface is a dielectric specular layer
//! over a diffuse substrate. The two layers are not independent: energy the
//! specular layer reflects (a view-angle-dependent `Fresnel` fraction) never
//! reaches the diffuse base, so treating them as two additive lobes
//! over-counts energy at grazing angles. Ashikhmin and Shirley derived a single
//! reciprocal, energy-conserving `BRDF` that couples them: the diffuse term is
//! attenuated by `(1 - R_s)` and by a factor that vanishes as either direction
//! grazes, exactly where the specular `Fresnel` term saturates to one. The
//! specular lobe is the shared isotropic GGX microfacet distribution with a
//! Schlick `Fresnel` tint.
//!
//! This module is the on-device twin of the `evaluate` and `pdf` closed forms
//! only; the stochastic `sample` routine (which needs a random stream) stays on
//! the host. [`GpuFresnelBlend`] evaluates one query per thread, reproducing
//! the reference's exact arithmetic — only `sqrt`, products, quotients and
//! clamps, with the Schlick quintics spelled out as repeated multiplications —
//! so a passing real-device parity test is direct evidence the ported kernel
//! computes the same `BRDF` value and density the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`FresnelBlendQuery`] — the per-channel diffuse and
//! specular reflectances, the perceptual roughness, and the view, light and
//! normal unit vectors — and writes one [`FresnelBlendResult`] holding the
//! coupled `BRDF` value `value`, the two-strategy mixture density `pdf`, and a
//! `valid` flag. The kernel mirrors
//! [`FresnelBlend::evaluate`](prism_render_architecture::reference_pt::fresnel_blend::FresnelBlend::evaluate)
//! and
//! [`FresnelBlend::pdf`](prism_render_architecture::reference_pt::fresnel_blend::FresnelBlend::pdf):
//! the Ashikhmin-Shirley diffuse term, the GGX specular term sharing the same
//! Schlick `Fresnel`, the cosine-hemisphere diffuse density and the GGX
//! visible-normal reflection density, balanced at probability `0.5` each.
//!
//! # What stays on the host
//!
//! The importance `sample` routine, the random stream, the visible-normal
//! half-vector warp and the integrator's path assembly all stay on the host;
//! the device sees only the stateless `evaluate` + `pdf` pair, one query at a
//! time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Both outputs thread through `sqrt`, products and quotients, so the `CPU` and
//! `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units in
//! the last place from the scalar reference. The parity test asserts the value
//! channels and the density within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`
//! (relative floored at `1e-6`), tight enough to catch a wrong port yet loose
//! enough to admit a legal last-place difference. The `valid` flag — set when
//! both the view and light cosines are positive — is compared exactly.
//!
//! # Degenerate inputs
//!
//! When either direction is below the surface (`cos_o <= 0` or `cos_i <= 0`)
//! the kernel reports a zero value, a zero density and `valid = 0`, matching
//! the reference early return. When the half vector `wo + wi` degenerates to
//! the zero vector (back-to-back directions) only the specular term and the
//! specular density drop to zero while the diffuse term survives, again exactly
//! as the reference does; `valid` stays `1` because the diffuse response is
//! still physical. Every division is guarded by an ordered comparison so no
//! `NaN` escapes.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `clamp`,
//! `min`, `max`, `select`, `dot`, `+ - * /` and unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry,
//! no `round` and no `f32` remainder, and no `f64`/`u64`/`u16`/`i64`/`i16`. It
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::fresnel_blend`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` coupled diffuse-specular kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`FresnelBlend::evaluate`](prism_render_architecture::reference_pt::fresnel_blend::FresnelBlend::evaluate)
/// and
/// [`FresnelBlend::pdf`](prism_render_architecture::reference_pt::fresnel_blend::FresnelBlend::pdf)
/// closed forms; see the module documentation for the algorithm.
const FRESNEL_BLEND_BSDF_WGSL: &str = r#"
// Coupled diffuse-specular (Ashikhmin-Shirley Fresnel blend) twin: one thread
// evaluates one query's BRDF value and two-strategy mixture density, mirroring
// the CPU golden `reference_pt::fresnel_blend::FresnelBlend::{evaluate, pdf}`
// with the shared GGX microfacet core and a Schlick Fresnel tint. Only sqrt,
// products, quotients, clamps and guarded selects appear. The stochastic sample
// routine stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::fresnel_blend；无第三方
// 引擎源码或衍生代码。

// 28 / (23 * pi): the Ashikhmin-Shirley diffuse normalization constant.
const DIFFUSE_NORM: f32 = 28.0 / (23.0 * 3.14159265359);
// Probability of the diffuse sampling strategy in the two-lobe mixture.
const DIFFUSE_SAMPLE_PROBABILITY: f32 = 0.5;
// Reciprocal of pi, used by the cosine-hemisphere and GGX densities.
const INV_PI: f32 = 1.0 / 3.14159265359;
// Smallest GGX width; below this the lobe is numerically a perfect mirror.
const MIN_ALPHA: f32 = 1.0e-3;
// Squared-length threshold below which a vector is treated as the zero vector.
const EPS_LEN_SQ: f32 = 1.0e-12;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Per-channel diffuse (substrate) reflectance R_d.
    diffuse_r: f32,
    diffuse_g: f32,
    diffuse_b: f32,
    // Per-channel specular normal-incidence reflectance R_s (the F0).
    specular_r: f32,
    specular_g: f32,
    specular_b: f32,
    // Perceptual roughness in [0, 1], remapped to alpha = roughness^2.
    roughness: f32,
    // View direction wo (unit).
    wo_x: f32,
    wo_y: f32,
    wo_z: f32,
    // Light direction wi (unit).
    wi_x: f32,
    wi_y: f32,
    wi_z: f32,
    // Surface normal (unit).
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
}

struct Blend {
    // Coupled BRDF value per channel.
    value_r: f32,
    value_g: f32,
    value_b: f32,
    // Two-strategy mixture solid-angle density.
    pdf: f32,
    // 1 when both the view and light cosines are positive, else 0.
    valid: u32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Blend>;

// Unit vector along v, or the zero vector when v is (numerically) zero, so the
// division never yields a NaN. Mirrors the reference `Vec3::normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    let inv = 1.0 / sqrt(max(len_sq, EPS_LEN_SQ));
    let scaled = v * inv;
    return select(vec3<f32>(0.0, 0.0, 0.0), scaled, len_sq > EPS_LEN_SQ);
}

// Raises x to the fifth power without a transcendental call.
fn quintic(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

// Schlick's per-channel Fresnel: f0 + (1 - f0) * (1 - cos)^5.
fn fresnel_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    let c = clamp(1.0 - cos_theta, 0.0, 1.0);
    let c2 = c * c;
    let c5 = c2 * c2 * c;
    let one = vec3<f32>(1.0, 1.0, 1.0);
    return f0 + (one - f0) * c5;
}

// The GGX normal distribution D(h) for a half vector with cosine cos_h to the
// macroscopic normal; zero for a back-facing half vector.
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
// cos_w. Grazing drives it toward infinity; normal incidence returns zero.
fn ggx_lambda(alpha: f32, cos_w: f32) -> f32 {
    let c = abs(cos_w);
    if (c >= 1.0) {
        return 0.0;
    }
    let c2 = c * c;
    let tan2 = (1.0 - c2) / max(c2, EPS_LEN_SQ);
    let a2 = alpha * alpha;
    return 0.5 * (sqrt(1.0 + a2 * tan2) - 1.0);
}

// The Smith single-direction masking term G1(w) in [0, 1].
fn ggx_g1(alpha: f32, cos_w: f32) -> f32 {
    return 1.0 / (1.0 + ggx_lambda(alpha, cos_w));
}

// The GGX visible-normal reflection density G1(wo) * D(h) / (4 cos_o).
fn ggx_reflection_pdf(alpha: f32, cos_o: f32, cos_h: f32) -> f32 {
    if (cos_o <= 0.0) {
        return 0.0;
    }
    return ggx_g1(alpha, cos_o) * ggx_distribution(alpha, cos_h) / (4.0 * cos_o);
}

// The cosine-weighted hemisphere density max(n . wi, 0) / pi.
fn cosine_hemisphere_pdf(n: vec3<f32>, wi: vec3<f32>) -> f32 {
    return max(dot(n, wi), 0.0) * INV_PI;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let diffuse_rgb = vec3<f32>(q.diffuse_r, q.diffuse_g, q.diffuse_b);
    let specular_rgb = vec3<f32>(q.specular_r, q.specular_g, q.specular_b);
    let wo = vec3<f32>(q.wo_x, q.wo_y, q.wo_z);
    let wi = vec3<f32>(q.wi_x, q.wi_y, q.wi_z);
    let normal = vec3<f32>(q.normal_x, q.normal_y, q.normal_z);

    // GGX width: alpha = clamp(roughness, 0, 1)^2, clamped up to MIN_ALPHA.
    let r = clamp(q.roughness, 0.0, 1.0);
    let alpha = max(r * r, MIN_ALPHA);

    let cos_o = dot(normal, wo);
    let cos_i = dot(normal, wi);

    var out: Blend;
    out.value_r = 0.0;
    out.value_g = 0.0;
    out.value_b = 0.0;
    out.pdf = 0.0;
    out.valid = 0u;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;

    if (cos_o > 0.0 && cos_i > 0.0) {
        out.valid = 1u;

        // Ashikhmin-Shirley diffuse term: Rd (1 - Rs) coupled by a factor that
        // falls to zero as either direction grazes.
        let fi = 1.0 - quintic(1.0 - 0.5 * cos_i);
        let fo = 1.0 - quintic(1.0 - 0.5 * cos_o);
        let one = vec3<f32>(1.0, 1.0, 1.0);
        let diffuse = diffuse_rgb * (one - specular_rgb) * (DIFFUSE_NORM * fi * fo);

        // GGX microfacet specular term with a Schlick Fresnel tint. The half
        // vector degenerates (and specular drops to zero) only when wo + wi is
        // the zero vector.
        let sum = wo + wi;
        let micro_h = normalize_or_zero(sum);
        let half_len_sq = dot(micro_h, micro_h);
        var specular = vec3<f32>(0.0, 0.0, 0.0);
        if (half_len_sq > EPS_LEN_SQ) {
            let cos_h = dot(normal, micro_h);
            if (cos_h > 0.0) {
                let woh = dot(wo, micro_h);
                let denom = 4.0 * abs(woh) * max(cos_o, cos_i);
                if (denom > 0.0) {
                    let d = ggx_distribution(alpha, cos_h);
                    let fresnel = fresnel_schlick(specular_rgb, max(woh, 0.0));
                    specular = fresnel * (d / denom);
                }
            }
        }
        let value = diffuse + specular;
        out.value_r = value.x;
        out.value_g = value.y;
        out.value_b = value.z;

        // Two-strategy mixture density: 0.5 cosine diffuse + 0.5 GGX specular.
        let diffuse_pdf = cosine_hemisphere_pdf(normal, wi);
        var specular_pdf = 0.0;
        if (half_len_sq > EPS_LEN_SQ) {
            specular_pdf = ggx_reflection_pdf(alpha, cos_o, dot(normal, micro_h));
        }
        out.pdf = DIFFUSE_SAMPLE_PROBABILITY * diffuse_pdf
            + (1.0 - DIFFUSE_SAMPLE_PROBABILITY) * specular_pdf;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`FRESNEL_BLEND_BSDF_WGSL`].
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
/// the per-channel diffuse and specular reflectances, the roughness, and the
/// three unit vectors, flattened to scalars and padded to an `80`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Red diffuse reflectance.
    diffuse_r: f32,
    /// Green diffuse reflectance.
    diffuse_g: f32,
    /// Blue diffuse reflectance.
    diffuse_b: f32,
    /// Red specular reflectance.
    specular_r: f32,
    /// Green specular reflectance.
    specular_g: f32,
    /// Blue specular reflectance.
    specular_b: f32,
    /// Perceptual roughness.
    roughness: f32,
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
    /// Normal `x`.
    normal_x: f32,
    /// Normal `y`.
    normal_y: f32,
    /// Normal `z`.
    normal_z: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
    /// Padding word.
    pad3: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Blend` struct:
/// the coupled value, the density and the `valid` flag, padded to a `32`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Red coupled value.
    value_r: f32,
    /// Green coupled value.
    value_g: f32,
    /// Blue coupled value.
    value_b: f32,
    /// Mixture density.
    pdf: f32,
    /// Validity flag (`1` or `0`).
    valid: u32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// One query for the coupled diffuse-specular twin: the per-channel diffuse and
/// specular reflectances, the perceptual roughness, and the view, light and
/// normal unit vectors.
///
/// `diffuse` is the substrate reflectance `R_d`, `specular` the dielectric
/// normal-incidence reflectance `R_s` (the `F0`), `roughness` the perceptual
/// roughness remapped to `alpha = roughness^2`. `wo`, `wi` and `normal` are
/// unit vectors in the viewer hemisphere. The host owns the stochastic sampler
/// and enqueues one [`FresnelBlendQuery`] per evaluation it needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FresnelBlendQuery {
    /// Per-channel diffuse reflectance `R_d`.
    pub diffuse: [f32; 3],
    /// Per-channel specular reflectance `R_s`.
    pub specular: [f32; 3],
    /// Perceptual roughness in `[0, 1]`.
    pub roughness: f32,
    /// View direction `wo` (unit).
    pub wo: [f32; 3],
    /// Light direction `wi` (unit).
    pub wi: [f32; 3],
    /// Surface normal (unit).
    pub normal: [f32; 3],
}

impl FresnelBlendQuery {
    /// Builds a query from the reflectances, roughness and the three vectors.
    #[must_use]
    pub const fn new(
        diffuse: [f32; 3],
        specular: [f32; 3],
        roughness: f32,
        wo: [f32; 3],
        wi: [f32; 3],
        normal: [f32; 3],
    ) -> FresnelBlendQuery {
        FresnelBlendQuery {
            diffuse,
            specular,
            roughness,
            wo,
            wi,
            normal,
        }
    }
}

/// One resolved query of the coupled diffuse-specular twin: the coupled `BRDF`
/// value, the two-strategy mixture density, and the validity flag.
///
/// `value` is
/// [`FresnelBlend::evaluate`](prism_render_architecture::reference_pt::fresnel_blend::FresnelBlend::evaluate)
/// and `pdf` is
/// [`FresnelBlend::pdf`](prism_render_architecture::reference_pt::fresnel_blend::FresnelBlend::pdf);
/// `valid` is `1` when both the view and light cosines are positive, else `0`
/// (in which case `value` and `pdf` are zero).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FresnelBlendResult {
    /// Coupled `BRDF` value per channel.
    pub value: [f32; 3],
    /// Two-strategy mixture solid-angle density.
    pub pdf: f32,
    /// `1` when both cosines are positive, else `0`.
    pub valid: u32,
}

/// Encodes one [`FresnelBlendQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &FresnelBlendQuery) -> GpuQuery {
    GpuQuery {
        diffuse_r: q.diffuse[0],
        diffuse_g: q.diffuse[1],
        diffuse_b: q.diffuse[2],
        specular_r: q.specular[0],
        specular_g: q.specular[1],
        specular_b: q.specular[2],
        roughness: q.roughness,
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
        pad3: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`FresnelBlendResult`].
fn decode_result(raw: &GpuResult) -> FresnelBlendResult {
    FresnelBlendResult {
        value: [raw.value_r, raw.value_g, raw.value_b],
        pdf: raw.pdf,
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

/// A compiled, reusable coupled diffuse-specular compute pipeline, twinning the
/// `CPU` golden
/// [`FresnelBlend::evaluate`](prism_render_architecture::reference_pt::fresnel_blend::FresnelBlend::evaluate)
/// and
/// [`FresnelBlend::pdf`](prism_render_architecture::reference_pt::fresnel_blend::FresnelBlend::pdf).
pub struct GpuFresnelBlend {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFresnelBlend {
    /// Compiles the coupled diffuse-specular kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFresnelBlend {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fresnel_blend"),
            source: ShaderSource::Wgsl(FRESNEL_BLEND_BSDF_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fresnel_blend_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fresnel_blend_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fresnel_blend_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFresnelBlend {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one
    /// [`FresnelBlendResult`] per input, in order.
    ///
    /// The values and densities match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[FresnelBlendQuery],
    ) -> Vec<FresnelBlendResult> {
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
            label: Some("prism_volumetric_fresnel_blend_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fresnel_blend_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fresnel_blend_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fresnel_blend_bind_group"),
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
            label: Some("prism_volumetric_fresnel_blend_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fresnel_blend_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fresnel_blend_pass"),
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
