//! `wgpu` compute twin of the split-sum environment-`BRDF` (`DFG`) analytic
//! primitives
//! ([`env_brdf_split_sum`](prism_render_architecture::particle::env_brdf_split_sum),
//! design §17 "`PBR` 粒子与体积着色", §22), exposing the closed-form,
//! transcendental-free pieces of the Lazarov / Karis `EnvBRDFApprox` fit one
//! pure-numeric routine at a time.
//!
//! The golden module provides two ways to obtain the integrated environment
//! `BRDF` `(scale, bias)`: a baked `DfgLut` bilinear table lookup, and a
//! table-free analytic polynomial fit. The analytic path is pure closed-form
//! algebra — a short polynomial plus a transcendental-free `2^(-9.28·NoV)`
//! surrogate recovered by integer-power squaring — so it ports cleanly to a
//! compute kernel. This twin takes the per-sample view: a single `solve` kernel
//! with an **operation-code dispatch** evaluates exactly one golden analytic
//! routine per lane, so a parity test can pin each building block on a real
//! device.
//!
//! # Why op-code dispatch
//!
//! Each twinned routine is a small, independently testable numeric kernel.
//! Rather than compile one pipeline per routine, every
//! [`EnvBrdfSplitSumQuery`] carries an operation code, and the single `solve`
//! kernel branches on it (an `if` / `else if` ladder on an unsigned code, an
//! exact integer compare). One thread handles one query; the batch may freely
//! mix operations. This keeps a single shader module and bind-group layout while
//! still surfacing every primitive to the parity suite.
//!
//! # What is twinned
//!
//! The pure-numeric analytic routines, each closed-form and transcendental-free:
//!
//! * `exp2_grazing_falloff` — the `2^(-9.28·NoV)` grazing term, reconstructed as
//!   `(2^(-9.28·NoV / 32))^32` from a truncated base-two series raised to the
//!   32nd power by five squarings (no `exp2`).
//! * `env_brdf_approx` — the Lazarov / Karis `(scale, bias)` polynomial fit.
//! * `env_brdf_specular` — the scalar reconstruction `F0·scale + bias` for a
//!   grayscale `F0`.
//! * `env_brdf_specular_rgb` — the per-channel reconstruction for a tinted
//!   (metal) `F0`.
//! * `prefilter_mip_lod` — the linear roughness-to-`mip`-`LOD` map for the
//!   split-sum prefiltered-colour lookup.
//!
//! # Left on the host (not twinned)
//!
//! The baked [`DfgLut`](prism_render_architecture::particle::env_brdf_split_sum::DfgLut)
//! bilinear table lookup, its `std430` packing (`to_std430`, `buffer_bytes`) and
//! the underlying hemispherical Monte-Carlo bake all stay on the host: the table
//! is a variable-length grid gathered with dynamic indices and the bake is a
//! transcendental integration, neither of which has a fixed-size on-device
//! counterpart here. The prefiltered-environment-map `mip` chain itself is a
//! separate texturing stage outside this numeric twin.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `+ - * /` and integer/`f32` value conversions — and no `exp`, `pow`,
//! `sin` / `cos` or any optional device feature, so it runs unmodified on Metal,
//! Vulkan and DX12. The golden `squared_k_times` integer-power squaring is
//! reproduced as the same doubling loop; no transcendental appears.
//!
//! # Correctness model
//!
//! Every twinned routine is fixed closed-form algebra, so `CPU` and `GPU`
//! evaluate the same expression. They are not bit-exact in general: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts an absolute-or-relative tolerance on the continuous `(scale, bias)`,
//! reconstruction and `LOD` values.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::env_brdf_split_sum`；
//! Lazarov / Karis split-sum environment-`BRDF` analytic fit (Dimitar Lazarov,
//! *Physically Based Lighting in Call of Duty: Black Ops 2*, 2013; Brian Karis
//! mobile `PBR` notes) plus `wgpu` compute dispatch; no third-party engine
//! source or derived code.
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::env_brdf_split_sum::{
    env_brdf_approx, env_brdf_specular, env_brdf_specular_rgb, exp2_grazing_falloff,
    prefilter_mip_lod, EnvBrdfTerms,
};
use prism_render_architecture::particle::Vec3;
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
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

// Operation codes shared by the host encoder and the `solve` kernel. Each tags
// one golden analytic routine; the kernel branches on the code with an exact
// integer compare.
const OP_EXP2_GRAZING: u32 = 0;
const OP_APPROX: u32 = 1;
const OP_SPECULAR: u32 = 2;
const OP_SPECULAR_RGB: u32 = 3;
const OP_MIP_LOD: u32 = 4;

/// The portable core-`WGSL` split-sum environment-`BRDF` analytic kernel,
/// embedded inline so the twin ships as a single source file. One thread
/// evaluates one query, branching on its operation code; see the module
/// documentation for the op-dispatch rationale.
const ENV_BRDF_SPLIT_SUM_WGSL: &str = r#"
// Split-sum environment-BRDF analytic primitive twin: one thread per query
// evaluates a single golden routine selected by an operation code. It mirrors
// the CPU golden `particle::env_brdf_split_sum`, uses only the portable
// core-WGSL subset (min/max/clamp and + - * / plus integer/f32 value
// conversions) and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: Lazarov / Karis split-sum environment-BRDF analytic fit; no
// third-party engine source or derived code.

struct Params {
    // Number of valid lanes in the batch, one thread each.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query's packed inputs. 48-byte std430 stride matching the host GpuQuery.
struct Query {
    // Operation code selecting the golden routine.
    op: u32,
    // Prefilter mip-chain level count for the LOD map.
    mip_count: u32,
    ipad0: u32,
    ipad1: u32,
    // Scalar inputs: (n_dot_v or roughness, roughness, f0, pad).
    args: vec4<f32>,
    // Per-channel reflectance F0 for the RGB reconstruction (r, g, b, pad).
    f0_rgb: vec4<f32>,
}

// One query's packed outputs. 16-byte std430 stride matching the host GpuResult.
struct Res {
    // Scalar in x; (scale, bias) in xy; the RGB reconstruction in xyz.
    v: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

const OP_EXP2_GRAZING: u32 = 0u;
const OP_APPROX: u32 = 1u;
const OP_SPECULAR: u32 = 2u;
const OP_SPECULAR_RGB: u32 = 3u;
const OP_MIP_LOD: u32 = 4u;

// `ln(2)` and its truncated-series powers, matching the golden f32 constants.
const LN2: f32 = 0.6931471805599453;
const LN2_SQ_HALF: f32 = LN2 * LN2 / 2.0;
const LN2_CUBE_SIXTH: f32 = LN2 * LN2 * LN2 / 6.0;

// Clamp a scalar into 0..=1 without branching on equality, matching the golden
// `clamp01`.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Square `base` `k` times, i.e. raise it to the 2^k-th power, matching the
// golden `squared_k_times` (integer-power squaring, no `pow`).
fn squared_k_times(base: f32, k: u32) -> f32 {
    var acc = base;
    var remaining = k;
    loop {
        if (remaining == 0u) {
            break;
        }
        acc = acc * acc;
        remaining = remaining - 1u;
    }
    return acc;
}

// `2^(-9.28 * NoV)` for `n_dot_v` in 0..=1 without a transcendental, matching
// the golden `exp2_grazing_falloff`: a truncated base-two series on the scaled
// exponent, raised back to the 32nd power by five squarings.
fn exp2_grazing_falloff(n_dot_v: f32) -> f32 {
    let exponent_magnitude = 9.28 * clamp01(n_dot_v);
    let s = exponent_magnitude * (1.0 / 32.0);
    let base = 1.0 - LN2 * s + LN2_SQ_HALF * s * s - LN2_CUBE_SIXTH * s * s * s;
    return squared_k_times(base, 5u);
}

// The Lazarov / Karis analytic `(scale, bias)` fit, matching the golden
// `env_brdf_approx`.
fn env_brdf_approx(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let n = clamp01(n_dot_v);
    let rough = clamp01(roughness);
    let rx = 1.0 - rough;
    let ry = rough * -0.0275 + 0.0425;
    let rz = rough * -0.572 + 1.04;
    let rw = rough * 0.022 + -0.04;
    let grazing = min(rx * rx, exp2_grazing_falloff(n));
    let a = grazing * rx + ry;
    return vec2<f32>(-1.04 * a + rz, 1.04 * a + rw);
}

// Linear roughness-to-mip-LOD map, matching the golden `prefilter_mip_lod`: a
// degenerate 0- or 1-level chain always samples LOD 0.
fn prefilter_mip_lod(roughness: f32, mip_count: u32) -> f32 {
    if (mip_count <= 1u) {
        return 0.0;
    }
    let max_lod = f32(mip_count - 1u);
    return clamp01(roughness) * max_lod;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane = gid.x;
    if (lane >= params.count) {
        return;
    }
    let q = queries[lane];
    var out: Res;
    out.v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let op = q.op;
    if (op == OP_EXP2_GRAZING) {
        out.v.x = exp2_grazing_falloff(q.args.x);
    } else if (op == OP_APPROX) {
        let t = env_brdf_approx(q.args.x, q.args.y);
        out.v.x = t.x;
        out.v.y = t.y;
    } else if (op == OP_SPECULAR) {
        let t = env_brdf_approx(q.args.x, q.args.y);
        out.v.x = q.args.z * t.x + t.y;
    } else if (op == OP_SPECULAR_RGB) {
        let t = env_brdf_approx(q.args.x, q.args.y);
        out.v.x = q.f0_rgb.x * t.x + t.y;
        out.v.y = q.f0_rgb.y * t.x + t.y;
        out.v.z = q.f0_rgb.z * t.x + t.y;
    } else if (op == OP_MIP_LOD) {
        out.v.x = prefilter_mip_lod(q.args.x, q.mip_count);
    }
    results[lane] = out;
}
"#;

/// One split-sum environment-`BRDF` analytic query: a tagged request to evaluate
/// a single golden routine on the device.
///
/// Each variant names one analytic routine of the `CPU` golden
/// [`env_brdf_split_sum`](prism_render_architecture::particle::env_brdf_split_sum)
/// and carries just that routine's operands. Holds `f32` operands, so it derives
/// only [`Clone`], [`Copy`], [`Debug`] and [`PartialEq`] (no [`Eq`] / [`Hash`]).
/// Provenance: query tagging for the `env_brdf_split_sum` twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnvBrdfSplitSumQuery {
    /// The `2^(-9.28·NoV)` grazing falloff for one view cosine, mirroring the
    /// golden `exp2_grazing_falloff`. `Provenance:` `exp2_grazing_falloff`.
    Exp2GrazingFalloff {
        /// The view cosine `NoV` (clamped to `0..=1`).
        n_dot_v: f32,
    },
    /// The Lazarov / Karis `(scale, bias)` fit for one `(NoV, roughness)`,
    /// mirroring the golden `env_brdf_approx`. `Provenance:` `env_brdf_approx`.
    EnvBrdfApprox {
        /// The view cosine `NoV` (clamped to `0..=1`).
        n_dot_v: f32,
        /// The perceptual `roughness` (clamped to `0..=1`).
        roughness: f32,
    },
    /// The scalar specular reconstruction `F0·scale + bias` for a grayscale
    /// `F0`, mirroring the golden `env_brdf_specular`.
    /// `Provenance:` `env_brdf_specular`.
    EnvBrdfSpecular {
        /// The view cosine `NoV` (clamped to `0..=1`).
        n_dot_v: f32,
        /// The perceptual `roughness` (clamped to `0..=1`).
        roughness: f32,
        /// The grayscale reflectance at normal incidence `F0`.
        f0: f32,
    },
    /// The per-channel specular reconstruction for a tinted `F0`, mirroring the
    /// golden `env_brdf_specular_rgb`. `Provenance:` `env_brdf_specular_rgb`.
    EnvBrdfSpecularRgb {
        /// The view cosine `NoV` (clamped to `0..=1`).
        n_dot_v: f32,
        /// The perceptual `roughness` (clamped to `0..=1`).
        roughness: f32,
        /// The per-channel reflectance at normal incidence `F0` as `[r, g, b]`.
        f0: [f32; 3],
    },
    /// The roughness-to-`mip`-`LOD` map for the prefiltered-colour lookup,
    /// mirroring the golden `prefilter_mip_lod`.
    /// `Provenance:` `prefilter_mip_lod`.
    PrefilterMipLod {
        /// The perceptual `roughness` (clamped to `0..=1`).
        roughness: f32,
        /// The prefiltered-map `mip`-chain level count.
        mip_count: u32,
    },
}

/// The result of one split-sum environment-`BRDF` analytic query, mirroring the
/// golden return of the routine the query names.
///
/// Provenance: result tagging for the `env_brdf_split_sum` twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnvBrdfSplitSumResult {
    /// A scalar result (grazing falloff, scalar reconstruction, `mip` `LOD`).
    Scalar {
        /// The scalar value.
        value: f32,
    },
    /// The analytic `(scale, bias)` term pair.
    Terms {
        /// The multiplicative `scale` applied to `F0`.
        scale: f32,
        /// The additive `bias`.
        bias: f32,
    },
    /// A three-channel `RGB` specular reconstruction.
    Vector {
        /// The `[r, g, b]` components.
        v: [f32; 3],
    },
}

/// The `CPU` golden verdict for one query, delegating to the matching routine of
/// [`env_brdf_split_sum`](prism_render_architecture::particle::env_brdf_split_sum)
/// so callers (and the parity test) can pin the twin lane for lane.
///
/// Every twinned routine is a published `pub` function of the golden module, so
/// each arm calls it directly rather than reimplementing the formula.
///
/// Provenance: `CPU` reference for the `env_brdf_split_sum` twin, 孪生自本仓
/// `prism_render_architecture::particle::env_brdf_split_sum`.
#[must_use]
pub fn cpu_reference(query: &EnvBrdfSplitSumQuery) -> EnvBrdfSplitSumResult {
    match query {
        EnvBrdfSplitSumQuery::Exp2GrazingFalloff { n_dot_v } => EnvBrdfSplitSumResult::Scalar {
            value: exp2_grazing_falloff(*n_dot_v),
        },
        EnvBrdfSplitSumQuery::EnvBrdfApprox { n_dot_v, roughness } => {
            let terms: EnvBrdfTerms = env_brdf_approx(*n_dot_v, *roughness);
            EnvBrdfSplitSumResult::Terms {
                scale: terms.scale,
                bias: terms.bias,
            }
        }
        EnvBrdfSplitSumQuery::EnvBrdfSpecular {
            n_dot_v,
            roughness,
            f0,
        } => EnvBrdfSplitSumResult::Scalar {
            value: env_brdf_specular(*n_dot_v, *roughness, *f0),
        },
        EnvBrdfSplitSumQuery::EnvBrdfSpecularRgb {
            n_dot_v,
            roughness,
            f0,
        } => {
            let v: Vec3 =
                env_brdf_specular_rgb(*n_dot_v, *roughness, Vec3::new(f0[0], f0[1], f0[2]));
            EnvBrdfSplitSumResult::Vector { v: [v.x, v.y, v.z] }
        }
        EnvBrdfSplitSumQuery::PrefilterMipLod {
            roughness,
            mip_count,
        } => EnvBrdfSplitSumResult::Scalar {
            value: prefilter_mip_lod(*roughness, *mip_count as usize),
        },
    }
}

/// Uniform dispatch parameters. `repr(C)` `std430` layout matching `Params` in
/// [`ENV_BRDF_SPLIT_SUM_WGSL`]: the lane count and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One query as uploaded. `48`-byte `std430` stride matching `Query` in the
/// shader: four leading `u32` codes and two operand `vec4`s.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    op: u32,
    mip_count: u32,
    ipad0: u32,
    ipad1: u32,
    args: [f32; 4],
    f0_rgb: [f32; 4],
}

/// One result as read back. `16`-byte `std430` stride matching `Res` in the
/// shader: a single output `vec4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    v: [f32; 4],
}

/// Encodes one query into its packed `GpuQuery`, placing each operand in the
/// slot the kernel reads for that operation code.
fn encode(query: &EnvBrdfSplitSumQuery) -> GpuQuery {
    let mut g = GpuQuery::zeroed();
    match query {
        EnvBrdfSplitSumQuery::Exp2GrazingFalloff { n_dot_v } => {
            g.op = OP_EXP2_GRAZING;
            g.args = [*n_dot_v, 0.0, 0.0, 0.0];
        }
        EnvBrdfSplitSumQuery::EnvBrdfApprox { n_dot_v, roughness } => {
            g.op = OP_APPROX;
            g.args = [*n_dot_v, *roughness, 0.0, 0.0];
        }
        EnvBrdfSplitSumQuery::EnvBrdfSpecular {
            n_dot_v,
            roughness,
            f0,
        } => {
            g.op = OP_SPECULAR;
            g.args = [*n_dot_v, *roughness, *f0, 0.0];
        }
        EnvBrdfSplitSumQuery::EnvBrdfSpecularRgb {
            n_dot_v,
            roughness,
            f0,
        } => {
            g.op = OP_SPECULAR_RGB;
            g.args = [*n_dot_v, *roughness, 0.0, 0.0];
            g.f0_rgb = [f0[0], f0[1], f0[2], 0.0];
        }
        EnvBrdfSplitSumQuery::PrefilterMipLod {
            roughness,
            mip_count,
        } => {
            g.op = OP_MIP_LOD;
            g.mip_count = *mip_count;
            g.args = [*roughness, 0.0, 0.0, 0.0];
        }
    }
    g
}

/// Decodes one raw `GpuResult` into the typed result the query shape implies.
fn decode(query: &EnvBrdfSplitSumQuery, r: &GpuResult) -> EnvBrdfSplitSumResult {
    match query {
        EnvBrdfSplitSumQuery::EnvBrdfApprox { .. } => EnvBrdfSplitSumResult::Terms {
            scale: r.v[0],
            bias: r.v[1],
        },
        EnvBrdfSplitSumQuery::EnvBrdfSpecularRgb { .. } => EnvBrdfSplitSumResult::Vector {
            v: [r.v[0], r.v[1], r.v[2]],
        },
        EnvBrdfSplitSumQuery::Exp2GrazingFalloff { .. }
        | EnvBrdfSplitSumQuery::EnvBrdfSpecular { .. }
        | EnvBrdfSplitSumQuery::PrefilterMipLod { .. } => {
            EnvBrdfSplitSumResult::Scalar { value: r.v[0] }
        }
    }
}

/// A compiled, reusable split-sum environment-`BRDF` analytic-evaluation
/// pipeline.
///
/// Provenance: `wgpu` compute twin of
/// `prism_render_architecture::particle::env_brdf_split_sum`.
pub struct GpuEnvBrdfSplitSum {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuEnvBrdfSplitSum {
    /// Compiles the split-sum environment-`BRDF` analytic kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required. Provenance: pipeline construction for the
    /// `env_brdf_split_sum` twin.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuEnvBrdfSplitSum {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_env_brdf_split_sum"),
            source: ShaderSource::Wgsl(ENV_BRDF_SPLIT_SUM_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_env_brdf_split_sum_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_env_brdf_split_sum_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_env_brdf_split_sum_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuEnvBrdfSplitSum {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one
    /// [`EnvBrdfSplitSumResult`] per query in input order.
    ///
    /// Each lane reproduces the golden analytic routine its query names, to
    /// within the tolerance documented on this module. An empty `queries` slice
    /// yields an empty result — storage buffers cannot be zero-sized, so it is
    /// handled by an early return. Provenance: primitive evaluation for the
    /// `env_brdf_split_sum` twin.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[EnvBrdfSplitSumQuery],
    ) -> Vec<EnvBrdfSplitSumResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries.iter().map(encode).collect();
        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_env_brdf_split_sum_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_env_brdf_split_sum_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_env_brdf_split_sum_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_env_brdf_split_sum_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_env_brdf_split_sum_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_env_brdf_split_sum_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_env_brdf_split_sum_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, in workgroups of 64 (the kernel's size).
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        queries
            .iter()
            .zip(gpu_results.iter())
            .map(|(query, raw)| decode(query, raw))
            .collect()
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
