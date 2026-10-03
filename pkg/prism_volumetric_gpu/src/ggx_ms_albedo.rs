//! `wgpu` compute twin of the `GGX` multiple-scattering directional-albedo
//! lookup from the reference path tracer's energy-compensation stack
//! ([`ggx_energy`](prism_render_architecture::reference_pt::ggx_energy)).
//!
//! The `CPU` golden bakes the white-`Fresnel` single-scatter directional albedo
//! `E(cos_theta, alpha)` into a `16x16` lookup table (`alpha` rows by
//! `cos_theta` columns) plus a `16`-entry cosine-weighted average-albedo table
//! `E_avg(alpha)`, both regenerated and furnace-checked by the golden's own
//! `Monte Carlo` self-test. At run time it reads the table with only multiplies,
//! adds and a clamp — no transcendental call — bilinearly interpolating
//! `directional_albedo`, linearly interpolating `average_albedo`, and composing
//! the scalar Kulla-Conty multiple-scattering lobe
//! `f_ms = (1 - E(mu_o)) * (1 - E(mu_i)) / (pi * (1 - E_avg))`.
//!
//! [`GpuGgxMsAlbedo`] is the on-device twin of that lookup-and-compose core: one
//! thread evaluates one query and emits the two directional albedos plus the
//! multiple-scattering lobe, so a passing real-device parity test is direct
//! evidence the ported kernel reads the same baked table and performs the same
//! bilinear interpolation the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For a query `(cos_o, cos_i, alpha)` the kernel reproduces three reference
//! values: `directional_albedo(cos_o, alpha)`, `directional_albedo(cos_i,
//! alpha)` and `multiscatter_lobe(cos_o, cos_i, alpha)`. The device carries its
//! own faithful copy of the baked `ALBEDO` and `AVG_ALBEDO` tables and the
//! `axis_weights` node-coordinate math (`t = clamp(x * 16 - 0.5, 0, 15)`,
//! `lo = floor(t)`, `hi = min(lo + 1, 15)`, `frac = t - lo`), so no reference
//! crate call is made from either side: the host oracle and the `WGSL` kernel
//! each re-derive the answer independently from the same literals.
//!
//! # What stays on the host
//!
//! The golden's `Monte Carlo` table regeneration, the per-channel `Fresnel`
//! tint `F_ms` and any variable-length aggregation are not part of this scalar
//! lookup core and stay off-device. The host enqueues one query per shaded
//! sample; an empty batch short-circuits with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! Every output threads through a clamp, a floor-based node split and a chain of
//! multiplies, adds and one divide. The piecewise-linear interpolation is
//! globally continuous — at a node boundary one side's `frac` tends to zero
//! while the other's tends to one, and both evaluate to the same node value —
//! so a unit-in-the-last-place disagreement in `floor` between `CPU` and `GPU`
//! cannot move the result discontinuously. The parity test therefore asserts a
//! tolerance (`abs_diff <= 1e-5` or `rel_diff <= 1e-4`) on each continuous
//! output, tight enough to catch a genuinely wrong port (a transposed table
//! index, a dropped `1 -`, a wrong axis) yet loose enough to admit a legal
//! last-place difference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `min`,
//! `max`, `clamp`, `+ - * /` and signed index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round`, no
//! `cbrt` and no `smoothstep`: the integer part is `floor(t)` and the fraction
//! is `t - floor(t)`. No `64`-bit or `16`-bit integer type is used; indices are
//! `i32`. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. Each thread performs a fixed, bounded sequence
//! of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::ggx_energy`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `GGX` multiple-scattering directional-albedo lookup
/// kernel, embedded inline so the twin ships as a single source file. The single
/// entry point `solve` mirrors the `CPU` golden
/// [`ggx_energy`](prism_render_architecture::reference_pt::ggx_energy) lookup
/// core; see the module documentation for the algorithm. The baked `ALBEDO` and
/// `AVG_ALBEDO` tables are reproduced here verbatim from the reference literals.
const GGX_MS_ALBEDO_WGSL: &str = r#"
// GGX multiple-scattering directional-albedo lookup twin: one thread reads the
// baked 16x16 single-scatter albedo table (alpha by cos_theta) and the 16-entry
// cosine-weighted average table, bilinearly interpolates the directional
// albedos for cos_o and cos_i, and composes the scalar Kulla-Conty
// multiple-scattering lobe, using only floor/min/max/clamp and + - * /. The
// Monte Carlo table regeneration and the per-channel Fresnel tint stay host-side.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::ggx_energy；无第三方
// 引擎源码或衍生代码。

const PI: f32 = 3.1415927;

// Baked single-scatter directional albedo E(mu, alpha) under a white (F = 1)
// Fresnel term, flattened row-major as ALBEDO[a * 16 + c] where a is the alpha
// node and c is the cos_theta node; both are cell centres at (i + 0.5) / 16.
var<private> ALBEDO: array<f32, 256> = array<f32, 256>(
    0.89053, 0.94328, 0.97589, 0.98818, 0.99206, 0.99439, 0.99607, 0.99685, 0.99744, 0.99806, 0.99817, 0.99817, 0.99858, 0.99886, 0.99885, 0.99904,
    0.93451, 0.88498, 0.89344, 0.91887, 0.93761, 0.95195, 0.96259, 0.96975, 0.97565, 0.97898, 0.98269, 0.98372, 0.98662, 0.98678, 0.98818, 0.98973,
    0.94954, 0.88988, 0.87099, 0.87141, 0.88501, 0.89715, 0.91246, 0.92474, 0.93381, 0.94122, 0.94994, 0.95394, 0.95931, 0.96293, 0.96593, 0.96903,
    0.95385, 0.89693, 0.86369, 0.85375, 0.85025, 0.85593, 0.86688, 0.87682, 0.88543, 0.89518, 0.90309, 0.91140, 0.91951, 0.92397, 0.93080, 0.93376,
    0.95163, 0.89433, 0.86047, 0.83623, 0.82537, 0.82326, 0.82529, 0.83233, 0.83790, 0.84373, 0.85562, 0.86116, 0.87050, 0.87715, 0.88401, 0.88939,
    0.94932, 0.88946, 0.84891, 0.82225, 0.80592, 0.79468, 0.79105, 0.79034, 0.79559, 0.80038, 0.80365, 0.80971, 0.81667, 0.82446, 0.82961, 0.83616,
    0.94567, 0.88040, 0.83628, 0.80512, 0.78428, 0.76896, 0.76043, 0.75496, 0.75359, 0.75547, 0.75319, 0.75845, 0.76217, 0.76699, 0.77263, 0.77446,
    0.94142, 0.87175, 0.82330, 0.79018, 0.76237, 0.74273, 0.73166, 0.72252, 0.71253, 0.70897, 0.70770, 0.70686, 0.70679, 0.70787, 0.71245, 0.71734,
    0.93671, 0.86121, 0.80691, 0.76759, 0.73937, 0.72034, 0.69780, 0.68750, 0.67520, 0.66904, 0.66345, 0.66195, 0.65407, 0.65477, 0.65289, 0.65404,
    0.92915, 0.84826, 0.79412, 0.75430, 0.71678, 0.69452, 0.66872, 0.65459, 0.64058, 0.62912, 0.62176, 0.61331, 0.60612, 0.60542, 0.60027, 0.59651,
    0.92425, 0.83505, 0.77650, 0.73150, 0.69712, 0.66509, 0.64125, 0.61974, 0.60482, 0.59115, 0.57870, 0.56560, 0.56079, 0.55360, 0.54713, 0.54434,
    0.91756, 0.82414, 0.75863, 0.71164, 0.67160, 0.63993, 0.61433, 0.59389, 0.57071, 0.55353, 0.53875, 0.52792, 0.51651, 0.50474, 0.50088, 0.49233,
    0.91143, 0.81112, 0.74357, 0.68803, 0.65290, 0.61420, 0.58876, 0.55983, 0.53882, 0.51787, 0.50314, 0.48608, 0.47485, 0.46401, 0.45315, 0.44662,
    0.90468, 0.79955, 0.73071, 0.67325, 0.62805, 0.59150, 0.55480, 0.52925, 0.50862, 0.48782, 0.46760, 0.45411, 0.43806, 0.42644, 0.41378, 0.40239,
    0.89889, 0.78596, 0.71155, 0.65008, 0.60615, 0.56783, 0.53284, 0.50402, 0.47753, 0.45741, 0.43591, 0.41926, 0.39868, 0.38978, 0.37722, 0.36418,
    0.89346, 0.77409, 0.69544, 0.63218, 0.58563, 0.54142, 0.50747, 0.47595, 0.44834, 0.42990, 0.40566, 0.38615, 0.36961, 0.35676, 0.34430, 0.32918
);

// Baked cosine-weighted average albedo E_avg(alpha) for each alpha node.
var<private> AVG_ALBEDO: array<f32, 16> = array<f32, 16>(
    0.99608, 0.97482, 0.94267, 0.90348, 0.86012, 0.81500, 0.76850, 0.72253, 0.67764, 0.63568, 0.59390, 0.55526, 0.51834, 0.48476, 0.45266, 0.42318
);

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Viewing-direction cosine.
    cos_o: f32,
    // Incident-direction cosine.
    cos_i: f32,
    // GGX roughness alpha.
    alpha: f32,
    pad0: f32,
}

struct Result {
    // Directional albedo E(cos_o, alpha).
    dir_albedo_o: f32,
    // Directional albedo E(cos_i, alpha).
    dir_albedo_i: f32,
    // Scalar Kulla-Conty multiple-scattering lobe.
    ms_lobe: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Lower node index along one axis for the continuous coordinate x in [0, 1].
fn axis_lo(x: f32) -> i32 {
    let t = clamp(x * 16.0 - 0.5, 0.0, 15.0);
    return i32(floor(t));
}

// Fractional node position along one axis.
fn axis_frac(x: f32) -> f32 {
    let t = clamp(x * 16.0 - 0.5, 0.0, 15.0);
    return t - floor(t);
}

// Single-scatter directional albedo, bilinearly interpolated from ALBEDO and
// clamped to [0, 1].
fn directional_albedo(cos_theta: f32, alpha: f32) -> f32 {
    let a_lo = axis_lo(alpha);
    let a_hi = min(a_lo + 1, 15);
    let a_f = axis_frac(alpha);
    let c_lo = axis_lo(cos_theta);
    let c_hi = min(c_lo + 1, 15);
    let c_f = axis_frac(cos_theta);
    let v00 = ALBEDO[a_lo * 16 + c_lo];
    let v01 = ALBEDO[a_lo * 16 + c_hi];
    let v10 = ALBEDO[a_hi * 16 + c_lo];
    let v11 = ALBEDO[a_hi * 16 + c_hi];
    let row_lo = v00 + (v01 - v00) * c_f;
    let row_hi = v10 + (v11 - v10) * c_f;
    return clamp(row_lo + (row_hi - row_lo) * a_f, 0.0, 1.0);
}

// Cosine-weighted average albedo, linearly interpolated and clamped to [0, 1].
fn average_albedo(alpha: f32) -> f32 {
    let lo = axis_lo(alpha);
    let hi = min(lo + 1, 15);
    let f = axis_frac(alpha);
    let v = AVG_ALBEDO[lo] + (AVG_ALBEDO[hi] - AVG_ALBEDO[lo]) * f;
    return clamp(v, 0.0, 1.0);
}

// Scalar Kulla-Conty multiple-scattering lobe, zero once E_avg reaches one.
fn multiscatter_lobe(cos_o: f32, cos_i: f32, alpha: f32) -> f32 {
    let e_avg = average_albedo(alpha);
    let denom = 1.0 - e_avg;
    if (denom <= 1.0e-4) {
        return 0.0;
    }
    let e_o = directional_albedo(cos_o, alpha);
    let e_i = directional_albedo(cos_i, alpha);
    return (1.0 - e_o) * (1.0 - e_i) / (PI * denom);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var out: Result;
    out.dir_albedo_o = directional_albedo(q.cos_o, q.alpha);
    out.dir_albedo_i = directional_albedo(q.cos_i, q.alpha);
    out.ms_lobe = multiscatter_lobe(q.cos_o, q.cos_i, q.alpha);
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`GGX_MS_ALBEDO_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the two cosines and the roughness
/// `alpha` plus one pad word to a `16`-byte stride, matching the `WGSL` `Query`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Viewing-direction cosine `cos_o`.
    cos_o: f32,
    /// Incident-direction cosine `cos_i`.
    cos_i: f32,
    /// `GGX` roughness `alpha`.
    alpha: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the two directional albedos, the multiple-scattering lobe, and one pad word to
/// a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Directional albedo `E(cos_o, alpha)`.
    dir_albedo_o: f32,
    /// Directional albedo `E(cos_i, alpha)`.
    dir_albedo_i: f32,
    /// Scalar multiple-scattering lobe `f_ms`.
    ms_lobe: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the `GGX` multiple-scattering directional-albedo twin: the two
/// direction cosines and the roughness `alpha`.
///
/// The host enqueues one [`GgxMsAlbedoQuery`] per shaded sample, mirroring the
/// reference lookup
/// [`directional_albedo`](prism_render_architecture::reference_pt::ggx_energy::directional_albedo)
/// and
/// [`multiscatter_lobe`](prism_render_architecture::reference_pt::ggx_energy::multiscatter_lobe)
/// inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxMsAlbedoQuery {
    /// Viewing-direction cosine `cos_o`.
    pub cos_o: f32,
    /// Incident-direction cosine `cos_i`.
    pub cos_i: f32,
    /// `GGX` roughness `alpha`.
    pub alpha: f32,
}

impl GgxMsAlbedoQuery {
    /// Builds a query from the two cosines and the roughness `alpha`.
    #[must_use]
    pub const fn new(cos_o: f32, cos_i: f32, alpha: f32) -> GgxMsAlbedoQuery {
        GgxMsAlbedoQuery {
            cos_o,
            cos_i,
            alpha,
        }
    }
}

/// One resolved lookup of the `GGX` multiple-scattering directional-albedo twin,
/// mirroring the reference
/// [`directional_albedo`](prism_render_architecture::reference_pt::ggx_energy::directional_albedo)
/// and
/// [`multiscatter_lobe`](prism_render_architecture::reference_pt::ggx_energy::multiscatter_lobe)
/// outputs.
///
/// `dir_albedo_o` and `dir_albedo_i` are the single-scatter directional albedos
/// for `cos_o` and `cos_i` at the query's `alpha`; `ms_lobe` is the scalar
/// Kulla-Conty multiple-scattering lobe composed from them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxMsAlbedoResult {
    /// Directional albedo `E(cos_o, alpha)`.
    pub dir_albedo_o: f32,
    /// Directional albedo `E(cos_i, alpha)`.
    pub dir_albedo_i: f32,
    /// Scalar multiple-scattering lobe `f_ms`.
    pub ms_lobe: f32,
}

/// Encodes one [`GgxMsAlbedoQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &GgxMsAlbedoQuery) -> GpuQuery {
    GpuQuery {
        cos_o: q.cos_o,
        cos_i: q.cos_i,
        alpha: q.alpha,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GgxMsAlbedoResult`].
fn decode_result(raw: &GpuResult) -> GgxMsAlbedoResult {
    GgxMsAlbedoResult {
        dir_albedo_o: raw.dir_albedo_o,
        dir_albedo_i: raw.dir_albedo_i,
        ms_lobe: raw.ms_lobe,
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

/// A compiled, reusable `GGX` multiple-scattering directional-albedo compute
/// pipeline, twinning the lookup-and-compose core of the `CPU` golden
/// [`ggx_energy`](prism_render_architecture::reference_pt::ggx_energy).
pub struct GpuGgxMsAlbedo {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGgxMsAlbedo {
    /// Compiles the `GGX` multiple-scattering albedo kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGgxMsAlbedo {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ggx_ms_albedo"),
            source: ShaderSource::Wgsl(GGX_MS_ALBEDO_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ggx_ms_albedo_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ggx_ms_albedo_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ggx_ms_albedo_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGgxMsAlbedo {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`GgxMsAlbedoResult`] per
    /// input, in order.
    ///
    /// Each directional albedo and the multiple-scattering lobe match the
    /// reference to within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GgxMsAlbedoQuery],
    ) -> Vec<GgxMsAlbedoResult> {
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
            label: Some("prism_volumetric_ggx_ms_albedo_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ggx_ms_albedo_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ggx_ms_albedo_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ggx_ms_albedo_bind_group"),
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
            label: Some("prism_volumetric_ggx_ms_albedo_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ggx_ms_albedo_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ggx_ms_albedo_pass"),
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
