//! `wgpu` compute twin of the exact complex-index-of-refraction `Fresnel`
//! reflectance of a conductor interface
//! ([`conductor`](prism_render_architecture::reference_pt::conductor)).
//!
//! A real metal's reflectance is governed by a wavelength-dependent *complex*
//! index of refraction `n~ = eta + i*k` (the real index `eta` and the
//! extinction coefficient `k`). The `CPU` golden evaluates the unpolarized
//! `Fresnel` reflectance of such an interface per red/green/blue channel, which
//! reproduces both a metal's base colour at normal incidence and its
//! characteristic off-axis hue drift toward a white edge highlight at grazing.
//! This module is the on-device twin of that spectral oracle:
//!
//! - [`fresnel_conductor`](prism_render_architecture::reference_pt::conductor::fresnel_conductor):
//!   the per-channel exact reflectance at a given incidence cosine, each channel
//!   the mean `0.5 * (R_s + R_p)` of the squared `s`- and `p`-polarized
//!   amplitudes, clamped to `[0, 1]`.
//! - [`average_fresnel_conductor`](prism_render_architecture::reference_pt::conductor::average_fresnel_conductor):
//!   the hemispherical cosine-weighted average
//!   `F_avg = 2 * integral_0^1 F(mu) mu d mu`, evaluated by `32`-node midpoint
//!   quadrature.
//!
//! [`GpuConductorFresnel`] evaluates both for one query per thread, reproducing
//! the reference's exact closed form — only `sqrt`, products and quotients, no
//! transcendental — so a passing real-device parity test is direct evidence the
//! ported kernel computes the same reflectance the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`ConductorFresnelQuery`] — the three-channel complex
//! index `eta + i*k` and the incidence cosine — and writes one
//! [`ConductorFresnelResult`] holding the per-channel reflectance at that cosine
//! and the per-channel hemispherical average. The channel kernel decomposes the
//! complex transmitted cosine into `a^2 + b^2` and its real part `a` by real
//! square roots, forms the `s`-polarized reflectance and the `p`-polarized
//! reflectance relative to it, then averages and clamps. The average loops the
//! same kernel over `32` midpoint nodes with the cosine weight
//! `2 * mu / 32`.
//!
//! # What stays on the host
//!
//! The `GGX` microfacet lobe, the visible-normal importance sampling, the
//! masking-shadowing term and the `BRDF` assembly all stay on the host; the
//! device sees only the stateless, fixed-width `Fresnel` evaluation, one query
//! at a time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Both outputs thread through `sqrt`, products and quotients, so the `CPU` and
//! `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units in the
//! last place from the scalar reference. The parity test asserts the direct
//! reflectance within `abs_diff <= 1e-5` or `rel_diff <= 1e-4` and the
//! `32`-node quadrature average within the looser `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (the sum of thirty-two evaluations accumulates more
//! last-place slack), tight enough to catch a genuinely wrong port yet loose
//! enough to admit a legal last-place difference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `clamp`,
//! `min`, `max`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and
//! no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::conductor`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` conductor-`Fresnel` kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`fresnel_conductor`](prism_render_architecture::reference_pt::conductor::fresnel_conductor)
/// and
/// [`average_fresnel_conductor`](prism_render_architecture::reference_pt::conductor::average_fresnel_conductor)
/// closed forms; see the module documentation for the algorithm.
const CONDUCTOR_FRESNEL_WGSL: &str = r#"
// Conductor Fresnel twin: one thread computes one query's per-channel exact
// reflectance at the incidence cosine plus the per-channel 32-node midpoint
// quadrature hemispherical average, mirroring the CPU golden
// `reference_pt::conductor::{fresnel_conductor, average_fresnel_conductor}`
// with only sqrt, products, quotients and a clamp. The GGX lobe and the BRDF
// assembly stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::conductor；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Red/green/blue real index of refraction eta.
    eta_r: f32,
    eta_g: f32,
    eta_b: f32,
    // Red/green/blue extinction coefficient k.
    k_r: f32,
    k_g: f32,
    k_b: f32,
    // Cosine of the incidence angle.
    cos_theta_i: f32,
    pad0: f32,
}

struct Reflectance {
    // Per-channel exact reflectance at the query incidence cosine.
    fresnel_r: f32,
    fresnel_g: f32,
    fresnel_b: f32,
    // Per-channel hemispherical cosine-weighted average reflectance.
    avg_r: f32,
    avg_g: f32,
    avg_b: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Reflectance>;

// Exact unpolarized Fresnel reflectance of one wavelength channel of a
// conductor interface, from the incidence cosine and the complex index
// eta + i*k. Only sqrt, products, quotients and a clamp appear.
fn fresnel_conductor_channel(cos_theta_i: f32, eta: f32, k: f32) -> f32 {
    let cos_i = clamp(cos_theta_i, 0.0, 1.0);
    let cos2 = cos_i * cos_i;
    let sin2 = 1.0 - cos2;
    let eta2 = eta * eta;
    let k2 = k * k;
    let t0 = eta2 - k2 - sin2;
    let a2_plus_b2 = sqrt(max(t0 * t0 + 4.0 * eta2 * k2, 0.0));
    let a = sqrt(max(0.5 * (a2_plus_b2 + t0), 0.0));
    // s-polarized reflectance.
    let t1 = a2_plus_b2 + cos2;
    let t2 = 2.0 * a * cos_i;
    let denom_s = t1 + t2;
    var r_s: f32 = 1.0;
    if (denom_s > 0.0) {
        r_s = (t1 - t2) / denom_s;
    }
    // p-polarized reflectance, expressed relative to r_s.
    let t3 = cos2 * a2_plus_b2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let denom_p = t3 + t4;
    var r_p: f32 = r_s;
    if (denom_p > 0.0) {
        r_p = r_s * (t3 - t4) / denom_p;
    }
    return clamp(0.5 * (r_s + r_p), 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Reflectance;

    // Direct per-channel reflectance at the query incidence cosine.
    out.fresnel_r = fresnel_conductor_channel(q.cos_theta_i, q.eta_r, q.k_r);
    out.fresnel_g = fresnel_conductor_channel(q.cos_theta_i, q.eta_g, q.k_g);
    out.fresnel_b = fresnel_conductor_channel(q.cos_theta_i, q.eta_b, q.k_b);

    // Hemispherical cosine-weighted average by 32-node midpoint quadrature:
    // F_avg = sum over i of F(mu_i) * weight_i, mu_i = (i + 0.5) / 32,
    // weight_i = 2 * mu_i / 32.
    let nodes: u32 = 32u;
    let inv_nodes = 1.0 / 32.0;
    var acc_r: f32 = 0.0;
    var acc_g: f32 = 0.0;
    var acc_b: f32 = 0.0;
    for (var i: u32 = 0u; i < nodes; i = i + 1u) {
        let mu = (f32(i) + 0.5) * inv_nodes;
        let weight = 2.0 * mu * inv_nodes;
        acc_r = acc_r + fresnel_conductor_channel(mu, q.eta_r, q.k_r) * weight;
        acc_g = acc_g + fresnel_conductor_channel(mu, q.eta_g, q.k_g) * weight;
        acc_b = acc_b + fresnel_conductor_channel(mu, q.eta_b, q.k_b) * weight;
    }
    out.avg_r = acc_r;
    out.avg_g = acc_g;
    out.avg_b = acc_b;

    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CONDUCTOR_FRESNEL_WGSL`].
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
/// the three-channel complex index and the incidence cosine, plus one pad word
/// to a `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Red channel real index `eta`.
    eta_r: f32,
    /// Green channel real index `eta`.
    eta_g: f32,
    /// Blue channel real index `eta`.
    eta_b: f32,
    /// Red channel extinction coefficient `k`.
    k_r: f32,
    /// Green channel extinction coefficient `k`.
    k_g: f32,
    /// Blue channel extinction coefficient `k`.
    k_b: f32,
    /// Cosine of the incidence angle.
    cos_theta_i: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Reflectance`
/// struct: the per-channel direct reflectance and the per-channel average, plus
/// two pad words to a `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Red channel direct reflectance.
    fresnel_r: f32,
    /// Green channel direct reflectance.
    fresnel_g: f32,
    /// Blue channel direct reflectance.
    fresnel_b: f32,
    /// Red channel hemispherical average.
    avg_r: f32,
    /// Green channel hemispherical average.
    avg_g: f32,
    /// Blue channel hemispherical average.
    avg_b: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One query for the conductor-`Fresnel` twin: the three-channel complex index
/// of refraction `eta + i*k` and the cosine of the incidence angle.
///
/// `eta_*` are the per-channel real indices and `k_*` the per-channel
/// extinction coefficients; `cos_theta_i` is clamped to `[0, 1]` by the kernel.
/// The host owns the microfacet lobe and enqueues one
/// [`ConductorFresnelQuery`] per spectral evaluation it needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConductorFresnelQuery {
    /// Red channel real index `eta`.
    pub eta_r: f32,
    /// Green channel real index `eta`.
    pub eta_g: f32,
    /// Blue channel real index `eta`.
    pub eta_b: f32,
    /// Red channel extinction coefficient `k`.
    pub k_r: f32,
    /// Green channel extinction coefficient `k`.
    pub k_g: f32,
    /// Blue channel extinction coefficient `k`.
    pub k_b: f32,
    /// Cosine of the incidence angle.
    pub cos_theta_i: f32,
}

impl ConductorFresnelQuery {
    /// Builds a query from the three-channel complex index and incidence cosine.
    #[must_use]
    pub const fn new(
        eta_r: f32,
        eta_g: f32,
        eta_b: f32,
        k_r: f32,
        k_g: f32,
        k_b: f32,
        cos_theta_i: f32,
    ) -> ConductorFresnelQuery {
        ConductorFresnelQuery {
            eta_r,
            eta_g,
            eta_b,
            k_r,
            k_g,
            k_b,
            cos_theta_i,
        }
    }
}

/// One resolved query of the conductor-`Fresnel` twin: the per-channel exact
/// reflectance at the query incidence cosine and the per-channel hemispherical
/// cosine-weighted average.
///
/// `fresnel_*` are
/// [`fresnel_conductor`](prism_render_architecture::reference_pt::conductor::fresnel_conductor);
/// `avg_*` are
/// [`average_fresnel_conductor`](prism_render_architecture::reference_pt::conductor::average_fresnel_conductor).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConductorFresnelResult {
    /// Red channel direct reflectance at `cos_theta_i`.
    pub fresnel_r: f32,
    /// Green channel direct reflectance at `cos_theta_i`.
    pub fresnel_g: f32,
    /// Blue channel direct reflectance at `cos_theta_i`.
    pub fresnel_b: f32,
    /// Red channel hemispherical average reflectance.
    pub avg_r: f32,
    /// Green channel hemispherical average reflectance.
    pub avg_g: f32,
    /// Blue channel hemispherical average reflectance.
    pub avg_b: f32,
}

/// Encodes one [`ConductorFresnelQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ConductorFresnelQuery) -> GpuQuery {
    GpuQuery {
        eta_r: q.eta_r,
        eta_g: q.eta_g,
        eta_b: q.eta_b,
        k_r: q.k_r,
        k_g: q.k_g,
        k_b: q.k_b,
        cos_theta_i: q.cos_theta_i,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ConductorFresnelResult`].
fn decode_result(raw: &GpuResult) -> ConductorFresnelResult {
    ConductorFresnelResult {
        fresnel_r: raw.fresnel_r,
        fresnel_g: raw.fresnel_g,
        fresnel_b: raw.fresnel_b,
        avg_r: raw.avg_r,
        avg_g: raw.avg_g,
        avg_b: raw.avg_b,
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

/// A compiled, reusable conductor-`Fresnel` compute pipeline, twinning the
/// `CPU` golden
/// [`fresnel_conductor`](prism_render_architecture::reference_pt::conductor::fresnel_conductor)
/// and
/// [`average_fresnel_conductor`](prism_render_architecture::reference_pt::conductor::average_fresnel_conductor).
pub struct GpuConductorFresnel {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuConductorFresnel {
    /// Compiles the conductor-`Fresnel` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuConductorFresnel {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_conductor_fresnel"),
            source: ShaderSource::Wgsl(CONDUCTOR_FRESNEL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_conductor_fresnel_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_conductor_fresnel_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_conductor_fresnel_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuConductorFresnel {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`ConductorFresnelResult`] per input, in order.
    ///
    /// The reflectances match the reference to within the tolerance documented
    /// on this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ConductorFresnelQuery],
    ) -> Vec<ConductorFresnelResult> {
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
            label: Some("prism_volumetric_conductor_fresnel_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_conductor_fresnel_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_conductor_fresnel_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_conductor_fresnel_bind_group"),
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
            label: Some("prism_volumetric_conductor_fresnel_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_conductor_fresnel_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_conductor_fresnel_pass"),
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
