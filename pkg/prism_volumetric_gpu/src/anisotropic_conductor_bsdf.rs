//! `wgpu` compute twin of the anisotropic rough-conductor `BRDF` evaluation and
//! probability density from the `CPU` golden
//! `prism_render_architecture::reference_pt::conductor_aniso::AnisoConductor`.
//!
//! Brushed aluminium, satin steel and the grooves on a machined bezel stretch a
//! specular highlight into a streak aligned with the grain. The reference
//! `AnisoConductor` pairs the exact complex-index `Fresnel` reflectance with an
//! anisotropic `GGX` microfacet lobe to serve as an offline oracle for that
//! brushed-metal look. This module ports the two stateless, no-`RNG` entry
//! points — `evaluate` and `pdf` — onto the device: the importance-sampling
//! `sample` method, which needs a random stream, is deliberately *not* twinned.
//!
//! [`GpuAnisoConductor`] is the on-device twin: one thread solves one query, so
//! a passing real-device parity test is direct evidence the ported kernel
//! computes the same `BRDF` value and density, and takes the same
//! below-horizon / degenerate-half-vector branch the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the reference closed form
//! `f_r = F * D * G2 / (4 cos_o cos_i)` and the matching reflection density
//! `G1(wo) * D(h) / (4 wo.z)`. It independently replicates every dependency the
//! golden composes: the Disney / `UE` roughness-anisotropy remap
//! `from_roughness_anisotropy`, the elliptical `GGX` `distribution`, the Smith
//! `lambda` auxiliary feeding `g1`/`g2`, the Duff tangent-basis construction,
//! the local-frame projection, and the per-channel complex-index `Fresnel`
//! reflectance. There is no loop: each thread performs a fixed, bounded
//! sequence of arithmetic plus a handful of `sqrt` calls, so the kernel
//! provably terminates.
//!
//! # Correctness model
//!
//! Every quantity threads through multiplies, adds, guarded divisions and
//! `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous quantity, tight enough to catch a genuinely wrong port yet
//! loose enough to admit legal fused multiply-add contraction. The discrete
//! `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! When either direction lies at or below the surface (`cos_o <= 0` or
//! `cos_i <= 0`), or the half vector degenerates (its pre-normalization squared
//! length is at or below `EPS_LEN_SQ`, or its local `z` is non-positive), the
//! reference returns a zero value and zero density; the twin reports `valid = 0`
//! with cleared outputs. The Duff basis has a removable singularity as
//! `normal.z -> -1`, so fixtures and the sweep keep the shading normal away from
//! that pole. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `select`, `sqrt`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no inverse trigonometry, no
//! `smoothstep`, no `round`, no `copysign` and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::conductor_aniso`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` anisotropic rough-conductor kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `AnisoConductor::evaluate` and `AnisoConductor::pdf`
/// branch for branch; see the module documentation for the algorithm.
const ANISOTROPIC_CONDUCTOR_BSDF_WGSL: &str = r#"
// Anisotropic rough-conductor twin: one thread per query reproduces the BRDF
// value F*D*G2/(4 cos_o cos_i) and the reflection density G1(wo)*D(h)/(4 wo.z)
// that AnisoConductor::evaluate and ::pdf derive from a complex index eta+i*k,
// a roughness/anisotropy pair, and the (wo, wi, normal) directions. It mirrors
// the CPU golden branch for branch, uses only the portable core-WGSL subset
// (min/max/clamp/select/sqrt and + - * / plus unsigned index math), takes no
// optional feature, and has no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::conductor_aniso；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Complex index real part eta (per channel).
    ex: f32,
    ey: f32,
    ez: f32,
    // Complex index extinction k (per channel).
    kx: f32,
    ky: f32,
    kz: f32,
    // Perceptual roughness in [0, 1] and anisotropy in [0, 1].
    roughness: f32,
    anisotropy: f32,
    // Outgoing direction (away from the surface), world space.
    wox: f32,
    woy: f32,
    woz: f32,
    // Incoming direction (away from the surface), world space.
    wix: f32,
    wiy: f32,
    wiz: f32,
    // Shading normal, world space.
    nx: f32,
    ny: f32,
    nz: f32,
}

struct Result {
    // BRDF value f_r(wo, wi), per channel; zero when invalid.
    vx: f32,
    vy: f32,
    vz: f32,
    // Solid-angle reflection density; zero when invalid.
    pdf: f32,
    // 1 when the pair reflects above the surface, 0 for a degenerate query.
    valid: u32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const INV_PI: f32 = 0.31830988618;
const MIN_ALPHA: f32 = 0.001;
const EPS_LEN_SQ: f32 = 1e-12;

// Disney / UE (Burley) roughness-anisotropy remap into the two GGX widths.
fn alpha_from_roughness_anisotropy(roughness: f32, anisotropy: f32) -> vec2<f32> {
    let r = clamp(roughness, 0.0, 1.0);
    let alpha = r * r;
    let aniso = clamp(anisotropy, 0.0, 1.0);
    let aspect = sqrt(max(1.0 - 0.9 * aniso, 1.0e-4));
    let ax = max(alpha / aspect, MIN_ALPHA);
    let ay = max(alpha * aspect, MIN_ALPHA);
    return vec2<f32>(ax, ay);
}

// Anisotropic GGX normal distribution D(h); zero for a back-facing half vector.
fn ggx_distribution(h: vec3<f32>, ax: f32, ay: f32) -> f32 {
    if (h.z <= 0.0) {
        return 0.0;
    }
    let hx = h.x / ax;
    let hy = h.y / ay;
    let hz = h.z;
    let q = hx * hx + hy * hy + hz * hz;
    return INV_PI / (ax * ay * q * q);
}

// Smith Lambda auxiliary for a local-frame direction.
fn ggx_lambda(w: vec3<f32>, ax: f32, ay: f32) -> f32 {
    let cz = abs(w.z);
    if (cz >= 1.0) {
        return 0.0;
    }
    let axx = ax * w.x;
    let ayy = ay * w.y;
    let numer = axx * axx + ayy * ayy;
    if (numer <= 0.0) {
        return 0.0;
    }
    let ratio = numer / (cz * cz);
    return 0.5 * (sqrt(1.0 + ratio) - 1.0);
}

fn ggx_g1(w: vec3<f32>, ax: f32, ay: f32) -> f32 {
    return 1.0 / (1.0 + ggx_lambda(w, ax, ay));
}

fn ggx_g2(wo: vec3<f32>, wi: vec3<f32>, ax: f32, ay: f32) -> f32 {
    return 1.0 / (1.0 + ggx_lambda(wo, ax, ay) + ggx_lambda(wi, ax, ay));
}

fn ggx_reflection_pdf(wo: vec3<f32>, h: vec3<f32>, ax: f32, ay: f32) -> f32 {
    if (wo.z <= 0.0) {
        return 0.0;
    }
    return ggx_g1(wo, ax, ay) * ggx_distribution(h, ax, ay) / (4.0 * wo.z);
}

// Exact unpolarized conductor Fresnel reflectance for one channel.
fn fresnel_channel(cos_theta: f32, eta: f32, k: f32) -> f32 {
    let cos_i = clamp(cos_theta, 0.0, 1.0);
    let cos2 = cos_i * cos_i;
    let sin2 = 1.0 - cos2;
    let eta2 = eta * eta;
    let k2 = k * k;
    let t0 = eta2 - k2 - sin2;
    let a2b2 = sqrt(max(t0 * t0 + 4.0 * eta2 * k2, 0.0));
    let a = sqrt(max(0.5 * (a2b2 + t0), 0.0));
    let t1 = a2b2 + cos2;
    let t2 = 2.0 * a * cos_i;
    let denom_s = t1 + t2;
    let r_s = select(1.0, (t1 - t2) / denom_s, denom_s > 0.0);
    let t3 = cos2 * a2b2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let denom_p = t3 + t4;
    let r_p = select(r_s, r_s * (t3 - t4) / denom_p, denom_p > 0.0);
    return clamp(0.5 * (r_s + r_p), 0.0, 1.0);
}

fn fresnel_conductor(eta: vec3<f32>, k: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    return vec3<f32>(
        fresnel_channel(cos_theta, eta.x, k.x),
        fresnel_channel(cos_theta, eta.y, k.y),
        fresnel_channel(cos_theta, eta.z, k.z),
    );
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let eta = vec3<f32>(q.ex, q.ey, q.ez);
    let k = vec3<f32>(q.kx, q.ky, q.kz);
    let wo = vec3<f32>(q.wox, q.woy, q.woz);
    let wi = vec3<f32>(q.wix, q.wiy, q.wiz);
    let normal = vec3<f32>(q.nx, q.ny, q.nz);

    let aniso = alpha_from_roughness_anisotropy(q.roughness, q.anisotropy);
    let ax = aniso.x;
    let ay = aniso.y;

    var out: Result;
    out.vx = 0.0;
    out.vy = 0.0;
    out.vz = 0.0;
    out.pdf = 0.0;
    out.valid = 0u;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;

    let cos_o = dot(normal, wo);
    let cos_i = dot(normal, wi);
    if (cos_o <= 0.0 || cos_i <= 0.0) {
        results[idx] = out;
        return;
    }

    // Branchless Duff tangent basis around the (unit) shading normal. The sign
    // uses an ordered compare instead of copysign so it stays in the core
    // subset; normal.z near -1 is the removable singularity avoided by callers.
    let sign_z = select(-1.0, 1.0, normal.z >= 0.0);
    let a_duff = -1.0 / (sign_z + normal.z);
    let b_duff = normal.x * normal.y * a_duff;
    let tangent = vec3<f32>(
        1.0 + sign_z * normal.x * normal.x * a_duff,
        sign_z * b_duff,
        -sign_z * normal.x,
    );
    let bitangent = vec3<f32>(b_duff, sign_z + normal.y * normal.y * a_duff, -normal.y);

    let wo_local = vec3<f32>(dot(wo, tangent), dot(wo, bitangent), dot(wo, normal));
    let wi_local = vec3<f32>(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));

    let half_sum = wo_local + wi_local;
    let half_len_sq = dot(half_sum, half_sum);
    let half_valid = half_len_sq > EPS_LEN_SQ;
    let inv_len = select(0.0, 1.0 / sqrt(half_len_sq), half_valid);
    let half_local = half_sum * inv_len;
    if (!half_valid || half_local.z <= 0.0) {
        results[idx] = out;
        return;
    }

    let d = ggx_distribution(half_local, ax, ay);
    let g2 = ggx_g2(wo_local, wi_local, ax, ay);
    let woh = max(dot(wo_local, half_local), 0.0);
    let fresnel = fresnel_conductor(eta, k, woh);
    let value = fresnel * (d * g2 / (4.0 * cos_o * cos_i));
    let pdf = ggx_reflection_pdf(wo_local, half_local, ax, ay);

    out.vx = value.x;
    out.vy = value.y;
    out.vz = value.z;
    out.pdf = pdf;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`ANISOTROPIC_CONDUCTOR_BSDF_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The `vec3` inputs are flattened to scalar lanes so the layout never trips a
/// `16`-byte vector-alignment rule; the kernel rebuilds each `vec3<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    ex: f32,
    ey: f32,
    ez: f32,
    kx: f32,
    ky: f32,
    kz: f32,
    roughness: f32,
    anisotropy: f32,
    wox: f32,
    woy: f32,
    woz: f32,
    wix: f32,
    wiy: f32,
    wiz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    vx: f32,
    vy: f32,
    vz: f32,
    pdf: f32,
    valid: u32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// One query for the anisotropic rough-conductor twin: the metal's complex
/// index `eta + i*k`, the perceptual `roughness`/`anisotropy` pair, and the
/// outgoing, incoming and normal directions.
///
/// Both the `BRDF` value and the reflection density are derived from this one
/// tuple, so a single query exercises the whole twinned core at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnisoConductorQuery {
    /// Per-channel real index of refraction `eta`.
    pub eta: [f32; 3],
    /// Per-channel extinction coefficient `k`.
    pub k: [f32; 3],
    /// Perceptual roughness in `[0, 1]`.
    pub roughness: f32,
    /// Perceptual anisotropy in `[0, 1]`.
    pub anisotropy: f32,
    /// Outgoing direction (away from the surface), expected unit length.
    pub wo: [f32; 3],
    /// Incoming direction (away from the surface), expected unit length.
    pub wi: [f32; 3],
    /// Shading normal, expected unit length.
    pub normal: [f32; 3],
}

impl AnisoConductorQuery {
    /// Builds a query from the complex index, the roughness-anisotropy pair and
    /// the three directions.
    #[must_use]
    pub fn new(
        eta: [f32; 3],
        k: [f32; 3],
        roughness: f32,
        anisotropy: f32,
        wo: [f32; 3],
        wi: [f32; 3],
        normal: [f32; 3],
    ) -> AnisoConductorQuery {
        AnisoConductorQuery {
            eta,
            k,
            roughness,
            anisotropy,
            wo,
            wi,
            normal,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `AnisoConductor::evaluate` and `AnisoConductor::pdf` outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnisoConductorResult {
    /// The `BRDF` value `f_r(wo, wi)` per channel; cleared to zero when invalid.
    pub value: [f32; 3],
    /// The solid-angle reflection density; zero when invalid.
    pub pdf: f32,
    /// `1` when the pair reflects above the surface, `0` for a degenerate query
    /// (either direction below the horizon or a degenerate half vector).
    pub valid: u32,
}

/// Encodes one [`AnisoConductorQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &AnisoConductorQuery) -> GpuQuery {
    GpuQuery {
        ex: q.eta[0],
        ey: q.eta[1],
        ez: q.eta[2],
        kx: q.k[0],
        ky: q.k[1],
        kz: q.k[2],
        roughness: q.roughness,
        anisotropy: q.anisotropy,
        wox: q.wo[0],
        woy: q.wo[1],
        woz: q.wo[2],
        wix: q.wi[0],
        wiy: q.wi[1],
        wiz: q.wi[2],
        nx: q.normal[0],
        ny: q.normal[1],
        nz: q.normal[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`AnisoConductorResult`].
fn decode_result(raw: &GpuResult) -> AnisoConductorResult {
    AnisoConductorResult {
        value: [raw.vx, raw.vy, raw.vz],
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

/// A compiled, reusable anisotropic rough-conductor compute pipeline, twinning
/// the `CPU` golden `AnisoConductor::evaluate` and `AnisoConductor::pdf`.
pub struct GpuAnisoConductor {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAnisoConductor {
    /// Compiles the anisotropic rough-conductor kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAnisoConductor {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_anisotropic_conductor_bsdf"),
            source: ShaderSource::Wgsl(ANISOTROPIC_CONDUCTOR_BSDF_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_anisotropic_conductor_bsdf_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_anisotropic_conductor_bsdf_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_anisotropic_conductor_bsdf_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAnisoConductor {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`AnisoConductorResult`]
    /// per input, in order.
    ///
    /// Each continuous channel matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[AnisoConductorQuery],
    ) -> Vec<AnisoConductorResult> {
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
            label: Some("prism_volumetric_anisotropic_conductor_bsdf_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_anisotropic_conductor_bsdf_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_anisotropic_conductor_bsdf_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_anisotropic_conductor_bsdf_bind_group"),
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
            label: Some("prism_volumetric_anisotropic_conductor_bsdf_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_anisotropic_conductor_bsdf_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_anisotropic_conductor_bsdf_pass"),
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
