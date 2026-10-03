//! `wgpu` compute twin of the fused water-optics planning step
//! ([`plan_optics`](prism_render_architecture::water::optics::plan_optics)).
//!
//! The `CPU` golden
//! [`plan_optics`](prism_render_architecture::water::optics::plan_optics) takes a
//! static [`OpticsProfile`](prism_render_architecture::water::optics::OpticsProfile)
//! and per-view
//! [`OpticsInputs`](prism_render_architecture::water::optics::OpticsInputs) and
//! fuses the spectral-dispersion stage with the underwater-transport stage into a
//! single deterministic
//! [`OpticsPlan`](prism_render_architecture::water::optics::OpticsPlan): the
//! per-channel refractive indices, the screen-space dispersion offsets, the
//! depth-attenuated color, the single-scatter phase, the godray inscatter, the
//! bounded multiple-scatter boost, and a coarse visibility flag. Every input is a
//! scalar (no slices), so the entire fused chain lives inside one thread.
//!
//! [`GpuWaterOpticsPlan`] is the on-device twin of that whole chain: one thread
//! computes one sample's complete plan, reproducing the reference's exact closed
//! form so that a passing real-device parity test is direct evidence the ported
//! kernel computes the same optics plan the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For one `(profile, inputs)` sample the kernel reproduces, in order:
//! - `spectral_iors(a, b)`: `cauchy_ior(a, b, lambda) = a + b / max(lambda, EPS)^2`
//!   sampled at the red/green/blue reference wavelengths `0.700/0.546/0.440` um.
//! - `dispersion_offsets(iors, sin_incidence, strength)`: per channel
//!   `channel_transmitted_sine(s, ior) * max(strength, 0)` where
//!   `channel_transmitted_sine(s, ior) = clamp(clamp(s, 0, 1) / max(ior, EPS), 0, 1)`.
//! - `depth_color_shift(color, extinction, depth)`: per channel
//!   `max(color, 0) * beer_lambert_transmittance(extinction, depth)`.
//! - `henyey_greenstein(cos_scatter, asymmetry_g)` phase.
//! - `godray_inscatter(surface_light, scatter_albedo, extinction.g, shaft_length)`.
//! - `single = beer_lambert_transmittance(extinction.g, view_depth) *
//!   max(surface_light, 0) * phase`, then
//!   `multiple_scatter_boost(single, scatter_albedo) =
//!   max(single, 0) / (1 - clamp(scatter_albedo, 0, 0.999))`.
//! - `visible = beer_lambert_transmittance(extinction.g, view_depth) >=
//!   clamp(visibility_threshold, 0, 1)`.
//!
//! The `Beer-Lambert` transmittance uses the crate's monotone `exp_approx`
//! (`base = 1 + x / 4096`, floored to zero, squared `12` times), replicated
//! verbatim so the device matches the host transport math.
//!
//! # What stays on the host
//!
//! Nothing of the fused chain stays host-side: every scalar step is portable and
//! runs on device. The host only owns the trivial empty-batch short-circuit (a
//! storage buffer cannot be zero-sized) and the packing of the public
//! [`WaterOpticsPlanQuery`] into its `std430` slot.
//!
//! # Correctness model
//!
//! The nine continuous outputs thread through `exp_approx`, a `sqrt` in the phase
//! denominator, and chained multiplies, so the `CPU` and `GPU` are not bit-exact:
//! a device `sqrt` or a `12`-step squaring may land a few units in the last place
//! from the scalar reference. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on each continuous output, tight
//! enough to catch a genuinely wrong port yet loose enough to admit a legal
//! last-place difference. The `visible` flag is a magnitude comparison, so for
//! fixtures chosen with a clear transmittance-vs-threshold margin the `CPU` and
//! `GPU` agree exactly and the flag is asserted with `==`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `clamp`,
//! `sqrt`, `+ - * /`, a fixed `12`-iteration loop, and unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round`, `smoothstep` or `cbrt`, and no `u64`/`u16`/`i64`/`f64`. No optional
//! device feature is required, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::optics`（融合 `dispersion` 与 `underwater`）；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` water-optics planning kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`plan_optics`](prism_render_architecture::water::optics::plan_optics) fused
/// chain; see the module documentation for the algorithm.
const WATER_OPTICS_PLAN_WGSL: &str = r#"
// Water-optics planning twin: one thread fuses the spectral-dispersion stage and
// the underwater-transport stage into one deterministic plan, mirroring the CPU
// golden `water::optics::plan_optics` closed form with only min/max/clamp/sqrt,
// + - * /, and a replicated monotone exp_approx. It owns no variable-length work.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::optics（融合 dispersion
// 与 underwater）；无第三方引擎源码或衍生代码。

const EPS: f32 = 1.0e-6;
const PI: f32 = 3.14159265358979;
const WL_R: f32 = 0.700;
const WL_G: f32 = 0.546;
const WL_B: f32 = 0.440;

struct Params {
    // Number of samples in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    cauchy_a: f32,
    cauchy_b: f32,
    refraction_strength: f32,
    extinction_r: f32,
    extinction_g: f32,
    extinction_b: f32,
    scatter_albedo: f32,
    asymmetry_g: f32,
    visibility_threshold: f32,
    sin_incidence: f32,
    view_depth: f32,
    surface_color_r: f32,
    surface_color_g: f32,
    surface_color_b: f32,
    surface_light: f32,
    cos_scatter: f32,
    shaft_length: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    ior_r: f32,
    ior_g: f32,
    ior_b: f32,
    disp0: f32,
    disp1: f32,
    disp2: f32,
    depth_r: f32,
    depth_g: f32,
    depth_b: f32,
    phase: f32,
    inscatter: f32,
    scatter_boost: f32,
    visible: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Monotone exponential approximation: base = 1 + x / 4096, floored to zero, then
// squared 12 times (2^12 = 4096). Replicated from the crate's `exp_approx`.
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

// Cauchy refractive-index law n(lambda) = a + b / max(lambda, EPS)^2.
fn cauchy_ior(a: f32, b: f32, wavelength_um: f32) -> f32 {
    let lambda = max(wavelength_um, EPS);
    return a + b / (lambda * lambda);
}

// Transmitted-ray sine for one channel: clamp(clamp(s, 0, 1) / max(ior, EPS), 0, 1).
fn channel_transmitted_sine(sin_incidence: f32, ior: f32) -> f32 {
    let n = max(ior, EPS);
    return clamp(clamp(sin_incidence, 0.0, 1.0) / n, 0.0, 1.0);
}

// Beer-Lambert transmittance exp(-max(e,0) * max(d,0)) via exp_approx, in 0..=1.
fn beer_lambert_transmittance(extinction: f32, distance: f32) -> f32 {
    let e = max(extinction, 0.0);
    let d = max(distance, 0.0);
    return clamp(exp_approx(-(e * d)), 0.0, 1.0);
}

// Henyey-Greenstein single-lobe phase; the 3/2 power is d * sqrt(d).
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let gg = clamp(g, -0.999, 0.999);
    let c = clamp(cos_theta, -1.0, 1.0);
    let denom_base = max(1.0 + gg * gg - 2.0 * gg * c, EPS);
    let denom = 4.0 * PI * denom_base * sqrt(denom_base);
    return (1.0 - gg * gg) / denom;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Spectral IOR via the Cauchy law at the three reference wavelengths.
    let ior_r = cauchy_ior(q.cauchy_a, q.cauchy_b, WL_R);
    let ior_g = cauchy_ior(q.cauchy_a, q.cauchy_b, WL_G);
    let ior_b = cauchy_ior(q.cauchy_a, q.cauchy_b, WL_B);

    // Screen-space dispersion offsets: transmitted sine times refraction gain.
    let gain = max(q.refraction_strength, 0.0);
    let disp0 = channel_transmitted_sine(q.sin_incidence, ior_r) * gain;
    let disp1 = channel_transmitted_sine(q.sin_incidence, ior_g) * gain;
    let disp2 = channel_transmitted_sine(q.sin_incidence, ior_b) * gain;

    // Depth-attenuated surface color via per-channel Beer-Lambert transmittance.
    let depth_r = max(q.surface_color_r, 0.0)
        * beer_lambert_transmittance(q.extinction_r, q.view_depth);
    let depth_g = max(q.surface_color_g, 0.0)
        * beer_lambert_transmittance(q.extinction_g, q.view_depth);
    let depth_b = max(q.surface_color_b, 0.0)
        * beer_lambert_transmittance(q.extinction_b, q.view_depth);

    // Single-scatter phase for the view/scatter angle.
    let phase = henyey_greenstein(q.cos_scatter, q.asymmetry_g);

    // Godray inscatter along the shaft: light * albedo * (1 - transmittance).
    let light = max(q.surface_light, 0.0);
    let albedo = clamp(q.scatter_albedo, 0.0, 1.0);
    let shaft_tr = beer_lambert_transmittance(q.extinction_g, q.shaft_length);
    let inscatter = light * albedo * (1.0 - shaft_tr);

    // Single scatter and its bounded multiple-scatter boost.
    let view_tr = beer_lambert_transmittance(q.extinction_g, q.view_depth);
    let single = view_tr * max(q.surface_light, 0.0) * phase;
    let boost_albedo = clamp(q.scatter_albedo, 0.0, 0.999);
    let scatter_boost = max(single, 0.0) / (1.0 - boost_albedo);

    // Coarse visibility: transmittance at view depth against the threshold.
    let threshold = clamp(q.visibility_threshold, 0.0, 1.0);
    var visible: u32 = 0u;
    if (view_tr >= threshold) {
        visible = 1u;
    }

    var out: Result;
    out.ior_r = ior_r;
    out.ior_g = ior_g;
    out.ior_b = ior_b;
    out.disp0 = disp0;
    out.disp1 = disp1;
    out.disp2 = disp2;
    out.depth_r = depth_r;
    out.depth_g = depth_g;
    out.depth_b = depth_b;
    out.phase = phase;
    out.inscatter = inscatter;
    out.scatter_boost = scatter_boost;
    out.visible = visible;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`WATER_OPTICS_PLAN_WGSL`].
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

/// `repr(C)` `std430` layout of one optics query: the full
/// [`OpticsProfile`](prism_render_architecture::water::optics::OpticsProfile) and
/// [`OpticsInputs`](prism_render_architecture::water::optics::OpticsInputs) fields
/// flattened to scalars, plus three pad words to an `80`-byte stride, matching the
/// `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `Cauchy` constant term `A`.
    cauchy_a: f32,
    /// `Cauchy` dispersion term `B`.
    cauchy_b: f32,
    /// Screen-space dispersion gain.
    refraction_strength: f32,
    /// Red extinction coefficient.
    extinction_r: f32,
    /// Green extinction coefficient.
    extinction_g: f32,
    /// Blue extinction coefficient.
    extinction_b: f32,
    /// Single-scattering albedo.
    scatter_albedo: f32,
    /// `Henyey-Greenstein` asymmetry `g`.
    asymmetry_g: f32,
    /// Transmittance visibility threshold.
    visibility_threshold: f32,
    /// Sine of the incidence angle.
    sin_incidence: f32,
    /// View-ray depth through the medium.
    view_depth: f32,
    /// Red surface color.
    surface_color_r: f32,
    /// Green surface color.
    surface_color_g: f32,
    /// Blue surface color.
    surface_color_b: f32,
    /// Incident surface light intensity.
    surface_light: f32,
    /// Cosine of the scattering angle.
    cos_scatter: f32,
    /// Length of the godray light shaft.
    shaft_length: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one optics result, matching the `WGSL` `Result`
/// struct: nine continuous outputs, the phase, inscatter and scatter boost, the
/// `visible` word, and three pad words to a `64`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Red `IOR`.
    ior_r: f32,
    /// Green `IOR`.
    ior_g: f32,
    /// Blue `IOR`.
    ior_b: f32,
    /// Red dispersion offset.
    disp0: f32,
    /// Green dispersion offset.
    disp1: f32,
    /// Blue dispersion offset.
    disp2: f32,
    /// Red depth-attenuated color.
    depth_r: f32,
    /// Green depth-attenuated color.
    depth_g: f32,
    /// Blue depth-attenuated color.
    depth_b: f32,
    /// `Henyey-Greenstein` phase value.
    phase: f32,
    /// Godray inscatter.
    inscatter: f32,
    /// Multiple-scatter boosted single scatter.
    scatter_boost: f32,
    /// `1` when geometry stays visible, `0` otherwise.
    visible: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One optics query flattening the static
/// [`OpticsProfile`](prism_render_architecture::water::optics::OpticsProfile) and
/// per-view
/// [`OpticsInputs`](prism_render_architecture::water::optics::OpticsInputs) fields
/// into scalars for the fused water-optics twin.
///
/// Mirrors the two struct inputs of the reference
/// [`plan_optics`](prism_render_architecture::water::optics::plan_optics): the
/// `Cauchy` coefficients and extinction/scattering/visibility scalars of the
/// profile, and the incidence, depth, color, light, scatter cosine and shaft
/// length of the per-view inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterOpticsPlanQuery {
    /// `Cauchy` constant term `A` (baseline `IOR`).
    pub cauchy_a: f32,
    /// `Cauchy` dispersion term `B`.
    pub cauchy_b: f32,
    /// Screen-space dispersion gain `refraction_strength`.
    pub refraction_strength: f32,
    /// Red `Beer-Lambert` extinction coefficient.
    pub extinction_r: f32,
    /// Green `Beer-Lambert` extinction coefficient.
    pub extinction_g: f32,
    /// Blue `Beer-Lambert` extinction coefficient.
    pub extinction_b: f32,
    /// Single-scattering albedo.
    pub scatter_albedo: f32,
    /// `Henyey-Greenstein` asymmetry parameter `g`.
    pub asymmetry_g: f32,
    /// Transmittance threshold below which geometry is invisible.
    pub visibility_threshold: f32,
    /// Sine of the incidence angle at the interface.
    pub sin_incidence: f32,
    /// View-ray depth through the medium.
    pub view_depth: f32,
    /// Red surface color prior to depth attenuation.
    pub surface_color_r: f32,
    /// Green surface color prior to depth attenuation.
    pub surface_color_g: f32,
    /// Blue surface color prior to depth attenuation.
    pub surface_color_b: f32,
    /// Incident surface light intensity.
    pub surface_light: f32,
    /// Cosine of the scattering angle for the phase function.
    pub cos_scatter: f32,
    /// Length of the light shaft for godray inscatter.
    pub shaft_length: f32,
}

/// One resolved optics plan, mirroring the reference
/// [`OpticsPlan`](prism_render_architecture::water::optics::OpticsPlan).
///
/// The three `IOR` values and three dispersion offsets come from the spectral
/// stage; `depth_color_*`, `phase`, `inscatter` and `scatter_boost` come from the
/// transport stage; `visible` is the coarse visibility flag at `view_depth`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterOpticsPlanResult {
    /// Red `IOR` from the `Cauchy` law.
    pub ior_r: f32,
    /// Green `IOR` from the `Cauchy` law.
    pub ior_g: f32,
    /// Blue `IOR` from the `Cauchy` law.
    pub ior_b: f32,
    /// Non-negative per-channel screen-space refraction offsets.
    pub dispersion_offsets: [f32; 3],
    /// Red depth-attenuated color after the `Beer-Lambert` shift.
    pub depth_color_r: f32,
    /// Green depth-attenuated color after the `Beer-Lambert` shift.
    pub depth_color_g: f32,
    /// Blue depth-attenuated color after the `Beer-Lambert` shift.
    pub depth_color_b: f32,
    /// `Henyey-Greenstein` phase value for the scatter angle.
    pub phase: f32,
    /// Godray inscatter contribution along the shaft.
    pub inscatter: f32,
    /// Multiple-scatter boosted single-scatter estimate.
    pub scatter_boost: f32,
    /// Whether geometry at `view_depth` remains visible.
    pub visible: bool,
}

/// Encodes one [`WaterOpticsPlanQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterOpticsPlanQuery) -> GpuQuery {
    GpuQuery {
        cauchy_a: q.cauchy_a,
        cauchy_b: q.cauchy_b,
        refraction_strength: q.refraction_strength,
        extinction_r: q.extinction_r,
        extinction_g: q.extinction_g,
        extinction_b: q.extinction_b,
        scatter_albedo: q.scatter_albedo,
        asymmetry_g: q.asymmetry_g,
        visibility_threshold: q.visibility_threshold,
        sin_incidence: q.sin_incidence,
        view_depth: q.view_depth,
        surface_color_r: q.surface_color_r,
        surface_color_g: q.surface_color_g,
        surface_color_b: q.surface_color_b,
        surface_light: q.surface_light,
        cos_scatter: q.cos_scatter,
        shaft_length: q.shaft_length,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterOpticsPlanResult`],
/// turning the `visible` word back into a [`bool`].
fn decode_result(raw: &GpuResult) -> WaterOpticsPlanResult {
    WaterOpticsPlanResult {
        ior_r: raw.ior_r,
        ior_g: raw.ior_g,
        ior_b: raw.ior_b,
        dispersion_offsets: [raw.disp0, raw.disp1, raw.disp2],
        depth_color_r: raw.depth_r,
        depth_color_g: raw.depth_g,
        depth_color_b: raw.depth_b,
        phase: raw.phase,
        inscatter: raw.inscatter,
        scatter_boost: raw.scatter_boost,
        visible: raw.visible != 0,
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

/// A compiled, reusable water-optics planning compute pipeline, twinning the
/// `CPU` golden
/// [`plan_optics`](prism_render_architecture::water::optics::plan_optics) fused
/// chain.
pub struct GpuWaterOpticsPlan {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterOpticsPlan {
    /// Compiles the water-optics planning kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterOpticsPlan {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_optics_plan"),
            source: ShaderSource::Wgsl(WATER_OPTICS_PLAN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_optics_plan_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_optics_plan_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_optics_plan_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterOpticsPlan {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every sample in `queries` and returns one [`WaterOpticsPlanResult`]
    /// per input, in order.
    ///
    /// The continuous outputs match the reference within the tolerance documented
    /// on this module; the `visible` flag equals the reference exactly for
    /// fixtures clear of a transmittance-vs-threshold tie. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterOpticsPlanQuery],
    ) -> Vec<WaterOpticsPlanResult> {
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
            label: Some("prism_volumetric_water_optics_plan_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_optics_plan_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_optics_plan_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_optics_plan_bind_group"),
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
            label: Some("prism_volumetric_water_optics_plan_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_optics_plan_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_optics_plan_pass"),
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
