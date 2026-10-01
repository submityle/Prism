//! `wgpu` compute twin of the Cook-Torrance microfacet specular `BRDF` core
//! ([`microfacet_ggx`](prism_render_architecture::particle::microfacet_ggx),
//! design sections 16-17, "`GGX` + 多散近似").
//!
//! Lit sprite and mesh particles route through the shared `PBR` closure whose
//! single-scattering specular term is the `GGX` / Trowbridge-Reitz lobe
//! `f_spec = D * V * F`. The `CPU` golden
//! [`microfacet_ggx`](prism_render_architecture::particle::microfacet_ggx) owns
//! that math; [`GpuMicrofacetGgx`] is the on-device twin that runs one thread
//! per sample and reproduces the same scalars and `RGB` triples the reference
//! produces. A passing real-device parity test is therefore direct evidence the
//! ported kernels evaluate the same rational / radical algebra the reference
//! does, not merely that the shaders compile.
//!
//! # What is twinned
//!
//! Eight closed-form terms are reproduced, one `WGSL` entry point each:
//!
//! * [`clamp_alpha`](prism_render_architecture::particle::microfacet_ggx::clamp_alpha)
//!   — the linear-roughness floor/ceiling.
//! * [`alpha_from_perceptual_roughness`](prism_render_architecture::particle::microfacet_ggx::alpha_from_perceptual_roughness)
//!   — the `alpha = perceptual^2` squaring remap.
//! * [`ggx_distribution`](prism_render_architecture::particle::microfacet_ggx::ggx_distribution)
//!   — the `GGX` normal distribution function `D`, a pure rational function.
//! * [`smith_g2_height_correlated`](prism_render_architecture::particle::microfacet_ggx::smith_g2_height_correlated)
//!   — the height-correlated `Smith` masking-shadowing `G2` (uses `sqrt`).
//! * [`visibility_smith_ggx_correlated`](prism_render_architecture::particle::microfacet_ggx::visibility_smith_ggx_correlated)
//!   — the folded visibility `V = G2 / (4 * NoL * NoV)`.
//! * [`fresnel_schlick_f0_rgb`](prism_render_architecture::particle::microfacet_ggx::fresnel_schlick_f0_rgb)
//!   — the three-channel `Schlick` `Fresnel`.
//! * [`specular_ggx_scalar`](prism_render_architecture::particle::microfacet_ggx::specular_ggx_scalar)
//!   — the assembled scalar lobe `D * V * F`.
//! * [`specular_dvf_rgb`](prism_render_architecture::particle::microfacet_ggx::specular_dvf_rgb)
//!   — the three-channel assembled lobe.
//!
//! The host feeds pre-computed cosines (`NoH`, `NoL`, `NoV`, `VoH`), the linear
//! roughness `alpha` and the per-channel `F0` through one packed sample array,
//! so the vector-to-cosine reduction
//! [`MicrofacetDirs::from_vectors`](prism_render_architecture::particle::microfacet_ggx::MicrofacetDirs::from_vectors)
//! stays on the host and the device twins only the transcendental-free scalar
//! algebra.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `clamp`, `min`,
//! `+ - * /`, `sqrt` and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan` or optional device feature, so they run
//! unmodified on `Metal`, `Vulkan` and `DX12`. The reference's fifth-power
//! `Fresnel` term is reproduced as an explicit chain of multiplies
//! (`x2 = x * x; x5 = x2 * x2 * x`), never a `pow` call, and the circle
//! constant is a compile-time literal that rounds to the same `f32` as
//! [`core::f32::consts::PI`], not a transcendental call.
//!
//! # Correctness model
//!
//! Each output is a fixed, non-reorderable sequence of multiplies, adds, one or
//! two divides and a `sqrt`, so `CPU` and `GPU` evaluate the same closed form.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, and the `GGX` denominator `(NoH^2 * (alpha^2 - 1) + 1)^2`,
//! the `Smith` radicals and the divides each shed a few units in the last
//! place. The parity test therefore asserts `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, tight enough to catch a genuinely wrong port (a swapped
//! term, a dropped `sqrt`, a wrong normalizer, a missing clamp) yet loose
//! enough to admit legal fused multiply-add contraction and the fifth-power
//! regrouping.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Cook-Torrance / `GGX` (Trowbridge-Reitz) microfacet
//! specular with the height-correlated `Smith` visibility of `Heitz` and the
//! `Schlick` `Fresnel` approximation, plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// used across this crate's kernels; the samples of one call are flattened to a
/// single linear index so the dispatch stays one-dimensional.
const WORKGROUP_SIZE: u32 = 64;

/// Scalar components in an `RGB` result (`Fresnel` and the `D * V * F` lobe are
/// three-channel). The device packs them as a flat `count * 3` `f32` array so a
/// storage buffer needs no `vec3` alignment padding.
const RGB_CHANNELS: usize = 3;

/// The portable core-`WGSL` microfacet-`GGX` kernels, embedded inline so the
/// twin ships as a single source file (there is no external `.wesl`). Every
/// entry point mirrors its `CPU` golden counterpart in
/// [`microfacet_ggx`](prism_render_architecture::particle::microfacet_ggx) term
/// for term; see the module documentation for the algorithm.
const MICROFACET_GGX_WGSL: &str = r#"
// Cook-Torrance microfacet GGX twin: one thread per sample evaluates one of the
// eight closed-form terms (clamp_alpha, alpha_from_perceptual_roughness, the
// GGX distribution D, the height-correlated Smith G2 and visibility V, the
// three-channel Schlick Fresnel, and the assembled scalar / RGB D*V*F lobes).
// All share one packed sample array and use only the portable core-WGSL subset
// (clamp/min, + - * /, sqrt and unsigned index math) with the fifth power of
// the Fresnel term written as an explicit multiply chain, so they take no
// optional feature and run unmodified on Metal, Vulkan and DX12. They mirror
// the CPU golden `particle::microfacet_ggx`.
//
// Provenance: standard Cook-Torrance / GGX microfacet specular with the
// height-correlated Smith visibility of Heitz and the Schlick Fresnel; no
// third-party engine source or derived code.

struct Params {
    // Number of samples (one thread each).
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Sample {
    // Cosine between normal and half vector, `NoH`.
    n_dot_h: f32,
    // Cosine between normal and light, `NoL`.
    n_dot_l: f32,
    // Cosine between normal and view, `NoV`.
    n_dot_v: f32,
    // Cosine between view (== light) and half vector, `VoH`.
    v_dot_h: f32,
    // Linear roughness `alpha` (also the scalar input for the roughness
    // kernels: `clamp_alpha` reads it as `alpha`, `alpha_from_perceptual`
    // reads it as a perceptual roughness).
    alpha: f32,
    // Per-channel reflectance at normal incidence, `F0`. The scalar specular
    // kernel uses `f0_r` as its scalar `F0`.
    f0_r: f32,
    f0_g: f32,
    f0_b: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> samples: array<Sample>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Circle constant, used only to normalize the GGX NDF. This literal rounds to
// the same f32 as the reference's `core::f32::consts::PI`; it is a compile-time
// constant, not a transcendental call.
const PI: f32 = 3.14159265358979323846264338327950288;
// Smallest linear roughness the NDF is evaluated at (mirror-limit floor).
const MIN_ALPHA: f32 = 1.0e-4;
// Generic denominator guard: quotients below this collapse to 0.0.
const MIN_DENOM: f32 = 1.0e-7;

// Clamps a scalar into the 0..=1 range (used for cosine terms).
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Clamps a linear roughness to [MIN_ALPHA, 1], mirroring `clamp_alpha`.
fn clamp_alpha(alpha: f32) -> f32 {
    return clamp(alpha, MIN_ALPHA, 1.0);
}

// `alpha = perceptual^2`, mirroring `alpha_from_perceptual_roughness`
// (perceptual is clamped to 0..=1 first).
fn alpha_from_perceptual(perceptual: f32) -> f32 {
    let p = clamp01(perceptual);
    return p * p;
}

// The GGX / Trowbridge-Reitz distribution `D`, mirroring `ggx_distribution`.
// The denominator kernel is grouped as `NoH^2 * alpha^2 + (1 - NoH^2)` (not the
// algebraically equal `NoH^2 * (alpha^2 - 1) + 1`) so the tiny `alpha^2` at the
// NoH = 1 peak is not cancelled away, exactly as the reference does.
fn ggx_distribution(n_dot_h: f32, alpha: f32) -> f32 {
    let a = clamp_alpha(alpha);
    let a2 = a * a;
    let noh = clamp01(n_dot_h);
    let noh2 = noh * noh;
    let kernel = noh2 * a2 + (1.0 - noh2);
    let denom = PI * kernel * kernel;
    return a2 / denom;
}

// Height-correlated Smith G2 of Heitz, mirroring
// `smith_g2_height_correlated`.
fn smith_g2(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    let a = clamp_alpha(alpha);
    let a2 = a * a;
    let nl = clamp01(n_dot_l);
    let nv = clamp01(n_dot_v);
    let lambda_v = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    let lambda_l = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    let denom = lambda_v + lambda_l;
    if (denom < MIN_DENOM) {
        return 0.0;
    }
    return (2.0 * nl * nv) / denom;
}

// Height-correlated Smith visibility `V = 0.5 / (Lambda_v + Lambda_l)`,
// mirroring `visibility_smith_ggx_correlated`.
fn visibility(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    let a = clamp_alpha(alpha);
    let a2 = a * a;
    let nl = clamp01(n_dot_l);
    let nv = clamp01(n_dot_v);
    let lambda_v = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    let lambda_l = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    let denom = 2.0 * (lambda_v + lambda_l);
    if (denom < MIN_DENOM) {
        return 0.0;
    }
    return 1.0 / denom;
}

// Scalar Schlick Fresnel `f0 + (1 - f0) * (1 - cos)^5`, mirroring
// `fresnel_rim::fresnel_schlick`. The fifth power is a hand-rolled multiply
// chain `x2 = x * x; x5 = x2 * x2 * x`, never a `pow` call.
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let cos = clamp01(cos_theta);
    let one_minus_cos = 1.0 - cos;
    let x2 = one_minus_cos * one_minus_cos;
    let x5 = x2 * x2 * one_minus_cos;
    return f0 + (1.0 - f0) * x5;
}

@compute @workgroup_size(64)
fn clamp_alpha_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    results[i] = clamp_alpha(samples[i].alpha);
}

@compute @workgroup_size(64)
fn alpha_from_perceptual_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    results[i] = alpha_from_perceptual(samples[i].alpha);
}

@compute @workgroup_size(64)
fn distribution_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let s = samples[i];
    results[i] = ggx_distribution(s.n_dot_h, s.alpha);
}

@compute @workgroup_size(64)
fn smith_g2_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let s = samples[i];
    results[i] = smith_g2(s.n_dot_l, s.n_dot_v, s.alpha);
}

@compute @workgroup_size(64)
fn visibility_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let s = samples[i];
    results[i] = visibility(s.n_dot_l, s.n_dot_v, s.alpha);
}

@compute @workgroup_size(64)
fn fresnel_rgb_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let s = samples[i];
    let base = i * 3u;
    results[base] = fresnel_schlick(s.v_dot_h, s.f0_r);
    results[base + 1u] = fresnel_schlick(s.v_dot_h, s.f0_g);
    results[base + 2u] = fresnel_schlick(s.v_dot_h, s.f0_b);
}

@compute @workgroup_size(64)
fn specular_scalar_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let s = samples[i];
    let d = ggx_distribution(s.n_dot_h, s.alpha);
    let vis = visibility(s.n_dot_l, s.n_dot_v, s.alpha);
    let f = fresnel_schlick(s.v_dot_h, s.f0_r);
    // specular_dvf_scalar order: `d * vis * f`.
    results[i] = d * vis * f;
}

@compute @workgroup_size(64)
fn specular_dvf_rgb_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let s = samples[i];
    let d = ggx_distribution(s.n_dot_h, s.alpha);
    let vis = visibility(s.n_dot_l, s.n_dot_v, s.alpha);
    // Reference assembly: `scale = d * vis`, then per-channel Fresnel * scale.
    let scale = d * vis;
    let base = i * 3u;
    results[base] = fresnel_schlick(s.v_dot_h, s.f0_r) * scale;
    results[base + 1u] = fresnel_schlick(s.v_dot_h, s.f0_g) * scale;
    results[base + 2u] = fresnel_schlick(s.v_dot_h, s.f0_b) * scale;
}
"#;

/// One microfacet sample: the pre-computed clamped cosines, the linear
/// roughness and the per-channel reflectance a shading point needs.
///
/// The cosines mirror the reference
/// [`MicrofacetDirs`](prism_render_architecture::particle::microfacet_ggx::MicrofacetDirs):
/// `NoH`, `NoL`, `NoV` and `VoH` (`== LoH`). `alpha` is the linear roughness,
/// and `f0` is the linear-`RGB` reflectance at normal incidence. The device
/// kernels clamp every cosine and the roughness exactly as the reference does,
/// so unclamped inputs are accepted and sanitized identically on both sides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MicrofacetSample {
    /// Cosine between the surface normal and the half vector, `NoH`.
    pub n_dot_h: f32,
    /// Cosine between the surface normal and the light direction, `NoL`.
    pub n_dot_l: f32,
    /// Cosine between the surface normal and the view direction, `NoV`.
    pub n_dot_v: f32,
    /// Cosine between the view (equivalently light) direction and the half
    /// vector, `VoH`.
    pub v_dot_h: f32,
    /// Linear roughness `alpha`.
    pub alpha: f32,
    /// Per-channel reflectance at normal incidence, `F0`. The scalar specular
    /// evaluation uses the first (red) channel as its scalar `F0`.
    pub f0: [f32; 3],
}

/// One sample as uploaded. `32`-byte `repr(C)` matching `Sample` in
/// [`MICROFACET_GGX_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSample {
    n_dot_h: f32,
    n_dot_l: f32,
    n_dot_v: f32,
    v_dot_h: f32,
    alpha: f32,
    f0_r: f32,
    f0_g: f32,
    f0_b: f32,
}

impl GpuSample {
    /// Packs a [`MicrofacetSample`] into the device layout.
    fn from_sample(s: &MicrofacetSample) -> GpuSample {
        GpuSample {
            n_dot_h: s.n_dot_h,
            n_dot_l: s.n_dot_l,
            n_dot_v: s.n_dot_v,
            v_dot_h: s.v_dot_h,
            alpha: s.alpha,
            f0_r: s.f0[0],
            f0_g: s.f0[1],
            f0_b: s.f0[2],
        }
    }

    /// Packs a lone scalar into the `alpha` field (other fields zeroed) for the
    /// roughness-only kernels [`GpuMicrofacetGgx::eval_clamp_alpha`] and
    /// [`GpuMicrofacetGgx::eval_alpha_from_perceptual_roughness`].
    fn from_scalar(alpha: f32) -> GpuSample {
        GpuSample {
            n_dot_h: 0.0,
            n_dot_l: 0.0,
            n_dot_v: 0.0,
            v_dot_h: 0.0,
            alpha,
            f0_r: 0.0,
            f0_g: 0.0,
            f0_b: 0.0,
        }
    }
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// [`MICROFACET_GGX_WGSL`]: the sample count plus three pad words — `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable microfacet-`GGX` pipeline set: one compute pipeline per
/// twinned term, all sharing one bind-group layout and shader module.
pub struct GpuMicrofacetGgx {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_clamp_alpha: ComputePipeline,
    pipeline_alpha_from_perceptual: ComputePipeline,
    pipeline_distribution: ComputePipeline,
    pipeline_smith_g2: ComputePipeline,
    pipeline_visibility: ComputePipeline,
    pipeline_fresnel_rgb: ComputePipeline,
    pipeline_specular_scalar: ComputePipeline,
    pipeline_specular_dvf_rgb: ComputePipeline,
}

impl GpuMicrofacetGgx {
    /// Compiles the microfacet-`GGX` kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMicrofacetGgx {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_microfacet_ggx"),
            source: ShaderSource::Wgsl(MICROFACET_GGX_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_microfacet_ggx_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_microfacet_ggx_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str, label: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipeline_clamp_alpha = make(
            "clamp_alpha_main",
            "prism_volumetric_microfacet_ggx_clamp_alpha_pipeline",
        );
        let pipeline_alpha_from_perceptual = make(
            "alpha_from_perceptual_main",
            "prism_volumetric_microfacet_ggx_alpha_from_perceptual_pipeline",
        );
        let pipeline_distribution = make(
            "distribution_main",
            "prism_volumetric_microfacet_ggx_distribution_pipeline",
        );
        let pipeline_smith_g2 = make(
            "smith_g2_main",
            "prism_volumetric_microfacet_ggx_smith_g2_pipeline",
        );
        let pipeline_visibility = make(
            "visibility_main",
            "prism_volumetric_microfacet_ggx_visibility_pipeline",
        );
        let pipeline_fresnel_rgb = make(
            "fresnel_rgb_main",
            "prism_volumetric_microfacet_ggx_fresnel_rgb_pipeline",
        );
        let pipeline_specular_scalar = make(
            "specular_scalar_main",
            "prism_volumetric_microfacet_ggx_specular_scalar_pipeline",
        );
        let pipeline_specular_dvf_rgb = make(
            "specular_dvf_rgb_main",
            "prism_volumetric_microfacet_ggx_specular_dvf_rgb_pipeline",
        );
        GpuMicrofacetGgx {
            module,
            layout,
            pipeline_clamp_alpha,
            pipeline_alpha_from_perceptual,
            pipeline_distribution,
            pipeline_smith_g2,
            pipeline_visibility,
            pipeline_fresnel_rgb,
            pipeline_specular_scalar,
            pipeline_specular_dvf_rgb,
        }
    }

    /// Clamps every roughness in `alphas` to `[MIN_ALPHA, 1]`, matching
    /// [`clamp_alpha`](prism_render_architecture::particle::microfacet_ggx::clamp_alpha).
    ///
    /// Returns one value per input in order; an empty slice yields an empty
    /// result with no dispatch (a storage buffer cannot be zero-sized).
    #[must_use]
    pub fn eval_clamp_alpha(&self, ctx: &GpuContext, alphas: &[f32]) -> Vec<f32> {
        if alphas.is_empty() {
            return Vec::new();
        }
        let gpu_samples: Vec<GpuSample> =
            alphas.iter().map(|&a| GpuSample::from_scalar(a)).collect();
        self.run(ctx, &self.pipeline_clamp_alpha, &gpu_samples, 1)
    }

    /// Maps every perceptual roughness in `perceptuals` to the linear roughness
    /// `alpha = perceptual^2`, matching
    /// [`alpha_from_perceptual_roughness`](prism_render_architecture::particle::microfacet_ggx::alpha_from_perceptual_roughness).
    ///
    /// Returns one value per input in order; an empty slice yields an empty
    /// result with no dispatch.
    #[must_use]
    pub fn eval_alpha_from_perceptual_roughness(
        &self,
        ctx: &GpuContext,
        perceptuals: &[f32],
    ) -> Vec<f32> {
        if perceptuals.is_empty() {
            return Vec::new();
        }
        let gpu_samples: Vec<GpuSample> = perceptuals
            .iter()
            .map(|&p| GpuSample::from_scalar(p))
            .collect();
        self.run(ctx, &self.pipeline_alpha_from_perceptual, &gpu_samples, 1)
    }

    /// Evaluates the `GGX` distribution `D` for every sample, matching
    /// [`ggx_distribution`](prism_render_architecture::particle::microfacet_ggx::ggx_distribution)`(s.n_dot_h, s.alpha)`.
    ///
    /// Returns one value per sample in order; an empty slice yields an empty
    /// result with no dispatch.
    #[must_use]
    pub fn eval_distribution(&self, ctx: &GpuContext, samples: &[MicrofacetSample]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }
        let gpu_samples = pack(samples);
        self.run(ctx, &self.pipeline_distribution, &gpu_samples, 1)
    }

    /// Evaluates the height-correlated `Smith` `G2` for every sample, matching
    /// [`smith_g2_height_correlated`](prism_render_architecture::particle::microfacet_ggx::smith_g2_height_correlated)`(s.n_dot_l, s.n_dot_v, s.alpha)`.
    ///
    /// Returns one value per sample in order; an empty slice yields an empty
    /// result with no dispatch.
    #[must_use]
    pub fn eval_smith_g2(&self, ctx: &GpuContext, samples: &[MicrofacetSample]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }
        let gpu_samples = pack(samples);
        self.run(ctx, &self.pipeline_smith_g2, &gpu_samples, 1)
    }

    /// Evaluates the height-correlated `Smith` visibility `V` for every sample,
    /// matching
    /// [`visibility_smith_ggx_correlated`](prism_render_architecture::particle::microfacet_ggx::visibility_smith_ggx_correlated)`(s.n_dot_l, s.n_dot_v, s.alpha)`.
    ///
    /// Returns one value per sample in order; an empty slice yields an empty
    /// result with no dispatch.
    #[must_use]
    pub fn eval_visibility(&self, ctx: &GpuContext, samples: &[MicrofacetSample]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }
        let gpu_samples = pack(samples);
        self.run(ctx, &self.pipeline_visibility, &gpu_samples, 1)
    }

    /// Evaluates the three-channel `Schlick` `Fresnel` for every sample,
    /// matching
    /// [`fresnel_schlick_f0_rgb`](prism_render_architecture::particle::microfacet_ggx::fresnel_schlick_f0_rgb)`(s.v_dot_h, s.f0)`.
    ///
    /// Returns one `RGB` triple per sample in order; an empty slice yields an
    /// empty result with no dispatch.
    #[must_use]
    pub fn eval_fresnel_rgb(
        &self,
        ctx: &GpuContext,
        samples: &[MicrofacetSample],
    ) -> Vec<[f32; 3]> {
        if samples.is_empty() {
            return Vec::new();
        }
        let gpu_samples = pack(samples);
        let flat = self.run(ctx, &self.pipeline_fresnel_rgb, &gpu_samples, RGB_CHANNELS);
        unflatten_rgb(&flat)
    }

    /// Evaluates the assembled scalar lobe `D * V * F` for every sample,
    /// matching
    /// [`specular_ggx_scalar`](prism_render_architecture::particle::microfacet_ggx::specular_ggx_scalar)
    /// with the sample's `f0[0]` as the scalar `F0`.
    ///
    /// Returns one value per sample in order; an empty slice yields an empty
    /// result with no dispatch.
    #[must_use]
    pub fn eval_specular_scalar(&self, ctx: &GpuContext, samples: &[MicrofacetSample]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }
        let gpu_samples = pack(samples);
        self.run(ctx, &self.pipeline_specular_scalar, &gpu_samples, 1)
    }

    /// Evaluates the three-channel assembled lobe `D * V * F` for every sample,
    /// matching
    /// [`specular_dvf_rgb`](prism_render_architecture::particle::microfacet_ggx::specular_dvf_rgb)`(dirs, s.alpha, s.f0)`.
    ///
    /// Returns one `RGB` triple per sample in order; an empty slice yields an
    /// empty result with no dispatch.
    #[must_use]
    pub fn eval_specular_dvf_rgb(
        &self,
        ctx: &GpuContext,
        samples: &[MicrofacetSample],
    ) -> Vec<[f32; 3]> {
        if samples.is_empty() {
            return Vec::new();
        }
        let gpu_samples = pack(samples);
        let flat = self.run(
            ctx,
            &self.pipeline_specular_dvf_rgb,
            &gpu_samples,
            RGB_CHANNELS,
        );
        unflatten_rgb(&flat)
    }

    /// Uploads `gpu_samples`, dispatches `pipeline` one thread per sample and
    /// reads back `gpu_samples.len() * components` `f32` results.
    ///
    /// Callers guarantee `gpu_samples` is non-empty (every public entry point
    /// early-returns on an empty input), so the storage buffers are never
    /// zero-sized.
    fn run(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        gpu_samples: &[GpuSample],
        components: usize,
    ) -> Vec<f32> {
        let device = ctx.device();
        let count = gpu_samples.len();

        let gpu_params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let out_bytes = (count * components * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_microfacet_ggx_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let samples_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_microfacet_ggx_samples"),
            contents: bytemuck::cast_slice(gpu_samples),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_microfacet_ggx_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_microfacet_ggx_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_microfacet_ggx_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: samples_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_microfacet_ggx_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_microfacet_ggx_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(values.len(), count * components);
        values
    }
}

/// Packs a slice of [`MicrofacetSample`] into the device layout.
fn pack(samples: &[MicrofacetSample]) -> Vec<GpuSample> {
    samples.iter().map(GpuSample::from_sample).collect()
}

/// Regroups a flat `count * 3` `f32` buffer into `RGB` triples.
fn unflatten_rgb(flat: &[f32]) -> Vec<[f32; 3]> {
    flat.chunks_exact(RGB_CHANNELS)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
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
