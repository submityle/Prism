//! `wgpu` compute twin of the separable `Smith`-`GGX` masking-shadowing family
//! of the particle microfacet `BRDF`
//! ([`microfacet_ggx`](prism_render_architecture::particle::microfacet_ggx)).
//!
//! The Cook-Torrance specular lobe multiplies a normal distribution `D`, a
//! geometric masking-shadowing term `G`, and a `Fresnel` term `F`, then divides
//! by the foreshortening factor `4 * NoL * NoV`. This module is the on-device
//! twin of the reference's *separable* (uncorrelated) `Smith` branch of the
//! masking-shadowing term, parameterized by a precomputed roughness remap `k`:
//!
//! - [`schlick_ggx_g1`](prism_render_architecture::particle::microfacet_ggx::schlick_ggx_g1):
//!   the one-sided `Schlick`-`GGX` term `G1(NoX, k) = NoX / (NoX * (1 - k) + k)`,
//!   with its `NoX` clamped to `[0, 1]`.
//! - [`smith_g_separable`](prism_render_architecture::particle::microfacet_ggx::smith_g_separable):
//!   the full separable masking-shadowing `G = G1(NoL, k) * G1(NoV, k)`.
//! - [`visibility_smith_ggx_separable`](prism_render_architecture::particle::microfacet_ggx::visibility_smith_ggx_separable):
//!   the visibility `V = G / (4 * NoL * NoV)`, the explicit-denominator companion
//!   that assembles the lobe as `D * V * F`.
//!
//! [`GpuGgxSmithVisibility`] evaluates the whole cohesive family for one query
//! per thread, so each sub-term is independently pinned, reproducing the
//! reference's exact closed form — only products, quotients and guarded
//! divisions, no `sqrt`, no transcendental — so a passing real-device parity
//! test is direct evidence the ported kernel computes the same masking the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`GgxSmithVisibilityQuery`] — the light cosine `NoL`,
//! the view cosine `NoV` and the precomputed roughness remap `k` — and writes
//! one [`GgxSmithVisibilityResult`] holding the one-sided `G1(NoL)` and
//! `G1(NoV)`, their product `G`, and the visibility `V`. The kernel clamps the
//! cosines to `[0, 1]`, forms each `G1` as a guarded rational, multiplies them,
//! and divides by the guarded `4 * NoL * NoV`.
//!
//! # What stays on the host
//!
//! The roughness remaps `smith_k_direct` / `smith_k_ibl` that produce `k`, the
//! `GGX` normal distribution, the `Fresnel` term, the height-correlated
//! visibility and the full `BRDF` assembly all stay on the host; the device
//! sees only the stateless, fixed-width separable masking evaluation, one query
//! at a time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Both the `G1` terms and the visibility thread through products and guarded
//! quotients, so the `CPU` and `GPU` are not bit-exact: a `GPU` divide may land
//! a few units in the last place from the scalar reference. The parity test
//! asserts each output within `abs_diff <= 1e-5` or `rel_diff <= 1e-4` (a floor
//! of `1e-6` on the relative denominator), tight enough to catch a genuinely
//! wrong port yet loose enough to admit a legal last-place difference.
//!
//! # Degenerate region
//!
//! Both the `G1` denominator `NoX * (1 - k) + k` and the visibility denominator
//! `4 * NoL * NoV` are guarded against a `MIN_DENOM` of `1e-7`: a divisor below
//! that collapses the quotient to `0.0` rather than producing a `NaN`. Grazing
//! light or view (`NoL -> 0` or `NoV -> 0`) drives `4 * NoL * NoV` under the
//! guard, so the visibility returns a clean `0.0`; a parity fixture pins that
//! guarded branch explicitly. The randomized sweep draws the cosines away from
//! zero so the division is well-conditioned.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round`, no `sqrt`
//! and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::microfacet_ggx`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` separable `Smith`-`GGX` masking kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`schlick_ggx_g1`](prism_render_architecture::particle::microfacet_ggx::schlick_ggx_g1),
/// [`smith_g_separable`](prism_render_architecture::particle::microfacet_ggx::smith_g_separable)
/// and
/// [`visibility_smith_ggx_separable`](prism_render_architecture::particle::microfacet_ggx::visibility_smith_ggx_separable)
/// closed forms; see the module documentation for the algorithm.
const GGX_SMITH_VISIBILITY_WGSL: &str = r#"
// Separable Smith-GGX visibility twin: one thread computes one query's one-sided
// G1(NoL) and G1(NoV), their product G = G1(NoL) * G1(NoV), and the visibility
// V = G / (4 * NoL * NoV), mirroring the CPU golden
// `particle::microfacet_ggx::{schlick_ggx_g1, smith_g_separable,
// visibility_smith_ggx_separable}` with only products, quotients, a clamp and a
// denominator guard. The GGX NDF, Fresnel and BRDF assembly stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::microfacet_ggx；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Cosine of the light angle NoL.
    n_dot_l: f32,
    // Cosine of the view angle NoV.
    n_dot_v: f32,
    // Precomputed Smith roughness remap k.
    k: f32,
    pad0: f32,
}

struct Visibility {
    // One-sided Schlick-GGX masking G1(NoL, k).
    g1_l: f32,
    // One-sided Schlick-GGX masking G1(NoV, k).
    g1_v: f32,
    // Separable masking-shadowing G = G1(NoL) * G1(NoV).
    g_separable: f32,
    // Visibility V = G / (4 * NoL * NoV).
    visibility: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Visibility>;

// Generic denominator guard matching the reference MIN_DENOM: quotients whose
// divisor falls below this collapse to 0.0 rather than dividing by near-zero.
const MIN_DENOM: f32 = 1.0e-7;

// Clamps a scalar into the 0..=1 range (used for the cosine terms).
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// One-sided Schlick-GGX masking term G1(NoX, k) = NoX / (NoX * (1 - k) + k),
// with NoX clamped to 0..=1 and the denominator guarded against zero.
fn schlick_ggx_g1(n_dot_x: f32, k: f32) -> f32 {
    let x = clamp01(n_dot_x);
    let denom = x * (1.0 - k) + k;
    if (denom < MIN_DENOM) {
        return 0.0;
    }
    return x / denom;
}

// Separable (uncorrelated) Smith masking-shadowing G = G1(NoL) * G1(NoV).
fn smith_g_separable(n_dot_l: f32, n_dot_v: f32, k: f32) -> f32 {
    return schlick_ggx_g1(n_dot_l, k) * schlick_ggx_g1(n_dot_v, k);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let nl = clamp01(q.n_dot_l);
    let nv = clamp01(q.n_dot_v);

    var out: Visibility;
    out.g1_l = schlick_ggx_g1(nl, q.k);
    out.g1_v = schlick_ggx_g1(nv, q.k);
    out.g_separable = smith_g_separable(nl, nv, q.k);

    let denom = 4.0 * nl * nv;
    if (denom < MIN_DENOM) {
        out.visibility = 0.0;
    } else {
        out.visibility = out.g_separable / denom;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`GGX_SMITH_VISIBILITY_WGSL`].
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
/// the two cosines and the roughness remap, plus one pad word to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Cosine of the light angle `NoL`.
    n_dot_l: f32,
    /// Cosine of the view angle `NoV`.
    n_dot_v: f32,
    /// Precomputed `Smith` roughness remap `k`.
    k: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Visibility`
/// struct: the two one-sided masking terms, their product and the visibility.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// One-sided `Schlick`-`GGX` masking `G1(NoL, k)`.
    g1_l: f32,
    /// One-sided `Schlick`-`GGX` masking `G1(NoV, k)`.
    g1_v: f32,
    /// Separable masking-shadowing `G = G1(NoL) * G1(NoV)`.
    g_separable: f32,
    /// Visibility `V = G / (4 * NoL * NoV)`.
    visibility: f32,
}

/// One query for the separable `Smith`-`GGX` visibility twin: the light cosine
/// `NoL`, the view cosine `NoV` and the precomputed roughness remap `k`.
///
/// `n_dot_l` and `n_dot_v` are clamped to `[0, 1]` by the kernel. `k` is a
/// precomputed `Smith` remap from
/// [`smith_k_direct`](prism_render_architecture::particle::microfacet_ggx::smith_k_direct)
/// or
/// [`smith_k_ibl`](prism_render_architecture::particle::microfacet_ggx::smith_k_ibl);
/// the host owns that remap and the lobe assembly and enqueues one
/// [`GgxSmithVisibilityQuery`] per evaluation it needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxSmithVisibilityQuery {
    /// Cosine of the light angle `NoL`.
    pub n_dot_l: f32,
    /// Cosine of the view angle `NoV`.
    pub n_dot_v: f32,
    /// Precomputed `Smith` roughness remap `k`.
    pub k: f32,
}

impl GgxSmithVisibilityQuery {
    /// Builds a query from the two cosines and the precomputed roughness remap.
    #[must_use]
    pub const fn new(n_dot_l: f32, n_dot_v: f32, k: f32) -> GgxSmithVisibilityQuery {
        GgxSmithVisibilityQuery {
            n_dot_l,
            n_dot_v,
            k,
        }
    }
}

/// One resolved query of the separable `Smith`-`GGX` visibility twin: the two
/// one-sided masking terms, their product and the visibility.
///
/// `g1_l` / `g1_v` are
/// [`schlick_ggx_g1`](prism_render_architecture::particle::microfacet_ggx::schlick_ggx_g1)
/// of `NoL` / `NoV`; `g_separable` is
/// [`smith_g_separable`](prism_render_architecture::particle::microfacet_ggx::smith_g_separable);
/// `visibility` is
/// [`visibility_smith_ggx_separable`](prism_render_architecture::particle::microfacet_ggx::visibility_smith_ggx_separable).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxSmithVisibilityResult {
    /// One-sided `Schlick`-`GGX` masking `G1(NoL, k)`.
    pub g1_l: f32,
    /// One-sided `Schlick`-`GGX` masking `G1(NoV, k)`.
    pub g1_v: f32,
    /// Separable masking-shadowing `G = G1(NoL) * G1(NoV)`.
    pub g_separable: f32,
    /// Visibility `V = G / (4 * NoL * NoV)`.
    pub visibility: f32,
}

/// Encodes one [`GgxSmithVisibilityQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &GgxSmithVisibilityQuery) -> GpuQuery {
    GpuQuery {
        n_dot_l: q.n_dot_l,
        n_dot_v: q.n_dot_v,
        k: q.k,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GgxSmithVisibilityResult`].
fn decode_result(raw: &GpuResult) -> GgxSmithVisibilityResult {
    GgxSmithVisibilityResult {
        g1_l: raw.g1_l,
        g1_v: raw.g1_v,
        g_separable: raw.g_separable,
        visibility: raw.visibility,
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

/// A compiled, reusable separable `Smith`-`GGX` visibility compute pipeline,
/// twinning the `CPU` golden
/// [`schlick_ggx_g1`](prism_render_architecture::particle::microfacet_ggx::schlick_ggx_g1),
/// [`smith_g_separable`](prism_render_architecture::particle::microfacet_ggx::smith_g_separable)
/// and
/// [`visibility_smith_ggx_separable`](prism_render_architecture::particle::microfacet_ggx::visibility_smith_ggx_separable).
pub struct GpuGgxSmithVisibility {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGgxSmithVisibility {
    /// Compiles the separable `Smith`-`GGX` visibility kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGgxSmithVisibility {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ggx_smith_visibility"),
            source: ShaderSource::Wgsl(GGX_SMITH_VISIBILITY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ggx_smith_visibility_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ggx_smith_visibility_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ggx_smith_visibility_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGgxSmithVisibility {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`GgxSmithVisibilityResult`] per input, in order.
    ///
    /// The masking terms match the reference to within the tolerance documented
    /// on this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GgxSmithVisibilityQuery],
    ) -> Vec<GgxSmithVisibilityResult> {
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
            label: Some("prism_volumetric_ggx_smith_visibility_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ggx_smith_visibility_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ggx_smith_visibility_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ggx_smith_visibility_bind_group"),
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
            label: Some("prism_volumetric_ggx_smith_visibility_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ggx_smith_visibility_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ggx_smith_visibility_pass"),
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
