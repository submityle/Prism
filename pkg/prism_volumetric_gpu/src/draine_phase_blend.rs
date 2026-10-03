//! `wgpu` compute twin of the volumetric `Draine` and `HG-Draine` scattering
//! phase functions
//! ([`draine_phase`](prism_render_architecture::volumetric::scatter::draine_phase)
//! and
//! [`hg_draine_phase`](prism_render_architecture::volumetric::scatter::hg_draine_phase)).
//!
//! The `CPU` golden phases live in
//! [`prism_render_architecture::volumetric::scatter`]: the `Draine`
//! (`Jendersie` and `d'Eon` 2023) phase multiplies the normalized
//! `Henyey-Greenstein` shape by `1 + alpha*u^2` to sharpen the forward `Mie`
//! peak while a matching normalization keeps the `4*PI` solid-angle integral at
//! one, and the `HG-Draine` blend is the convex mix
//! `lerp(draine, hg, saturate(hg_weight))` of that `Draine` lobe with a plain
//! `HG` lobe at the same scattering cosine. Both carry the isotropic
//! `1 / (4*PI)` leading factor; both clamp the scattering cosine to `[-1, 1]`
//! and the asymmetry `g` to `[-0.999, 0.999]` so the `1 + g^2 - 2*g*u`
//! denominator never collapses.
//!
//! [`GpuDrainePhaseBlend`] is the on-device twin of exactly those two phases.
//! One thread solves one whole query: it evaluates the inlined `hg_phase`
//! helper and the `draine_phase` closed form, then forms the `HG-Draine` blend,
//! reproducing the reference's clamps, the `max(denom, EPS)` guard, the
//! `denom * sqrt(denom)` pure-`sqrt` form of `denom^1.5`, the
//! `norm = 1 + alpha*(1 + 2*g^2)/3` normalization, and the `saturate(hg_weight)`
//! convex blend. A passing real-device parity test is therefore direct evidence
//! the ported kernel computes the same phase values the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one query the twin reproduces two scalar phase values:
//! `draine = draine_phase(cos_theta, g_draine, alpha)` and
//! `hg_draine = hg_draine_phase(cos_theta, g_hg, g_draine, alpha, hg_weight)`.
//! The `hg_phase` helper is twinned only as an inlined `WGSL` function in the
//! service of the blend; it is not surfaced as a separate parity target.
//!
//! # What stays on the host
//!
//! Nothing stateful: these phases are pure functions of their arguments. The
//! host only marshals the batch of `(cos_theta, g_hg, g_draine, alpha,
//! hg_weight)` tuples into the device query buffer and reads the two phase
//! values back.
//!
//! # Correctness model
//!
//! The phases thread through only `+ - * /`, `clamp`, `max`, and `sqrt`, with no
//! transcendental and no reorderable reduction, so the `CPU` and `GPU` evaluate
//! the same expression. They are not bit-exact — a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, and `sqrt` rounding can
//! differ in the last place — so results are compared within
//! `abs_diff <= 1e-5` or `rel_diff <= 1e-4`. The `INV_FOUR_PI` constant is
//! formed as `1.0 / (4.0 * PI)` on both sides from the same `PI` so the shared
//! normalization is computed, not an approximated literal.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `max`,
//! `sqrt`, and the four arithmetic operators — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `round`, and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::volumetric::scatter`；无第三方引擎源码或衍生代码。
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
/// used across this crate's one-thread-per-element kernels; here one element is
/// one whole `Draine` and `HG-Draine` phase evaluation.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` `Draine` and `HG-Draine` phase kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` goldens
/// [`draine_phase`](prism_render_architecture::volumetric::scatter::draine_phase)
/// and
/// [`hg_draine_phase`](prism_render_architecture::volumetric::scatter::hg_draine_phase);
/// see the module documentation for the algorithm.
const DRAINE_PHASE_BLEND_WGSL: &str = r#"
// Draine and HG-Draine volumetric scattering phase twin: one thread evaluates
// one query, mirroring the CPU goldens `volumetric::scatter::draine_phase` and
// `volumetric::scatter::hg_draine_phase` using only clamp, max, sqrt and the
// four arithmetic operators.
//
// Provenance: 孪生自本仓 prism_render_architecture::volumetric::scatter；无第三方引擎源码或衍生代码。

const PI: f32 = 3.14159274;
const INV_FOUR_PI: f32 = 1.0 / (4.0 * PI);
const MAX_ABS_G: f32 = 0.999;
const EPS: f32 = 1e-6;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    cos_theta: f32,
    g_hg: f32,
    g_draine: f32,
    alpha: f32,
    hg_weight: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct PhaseResult {
    draine: f32,
    hg_draine: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<PhaseResult>;

fn hg_phase(cos_theta: f32, g_in: f32) -> f32 {
    let u = clamp(cos_theta, -1.0, 1.0);
    let g = clamp(g_in, -MAX_ABS_G, MAX_ABS_G);
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * u, EPS);
    return INV_FOUR_PI * (1.0 - g2) / (denom * sqrt(denom));
}

fn draine_phase(cos_theta: f32, g_in: f32, alpha_in: f32) -> f32 {
    let u = clamp(cos_theta, -1.0, 1.0);
    let g = clamp(g_in, -MAX_ABS_G, MAX_ABS_G);
    let alpha = max(alpha_in, 0.0);
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * u, EPS);
    let norm = 1.0 + alpha * (1.0 + 2.0 * g2) / 3.0;
    return INV_FOUR_PI * (1.0 - g2) * (1.0 + alpha * u * u) / (norm * denom * sqrt(denom));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let draine = draine_phase(q.cos_theta, q.g_draine, q.alpha);
    let hg = hg_phase(q.cos_theta, q.g_hg);
    // Convex HG-Draine blend lerp(draine, hg, saturate(hg_weight)) with the
    // weight clamped to [0, 1] exactly as the reference does.
    let w = clamp(q.hg_weight, 0.0, 1.0);
    let hg_draine = draine + (hg - draine) * w;

    var out: PhaseResult;
    out.draine = draine;
    out.hg_draine = hg_draine;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`DRAINE_PHASE_BLEND_WGSL`].
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
/// the five phase arguments plus three pad words to a `32`-byte stride (a
/// `16`-byte multiple as `std430` requires for an array element).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Cosine of the scattering angle, clamped to `[-1, 1]` in the kernel.
    cos_theta: f32,
    /// `HG`-lobe asymmetry, clamped to `[-0.999, 0.999]` in the kernel.
    g_hg: f32,
    /// `Draine`-lobe asymmetry, clamped to `[-0.999, 0.999]` in the kernel.
    g_draine: f32,
    /// `Draine` shape parameter, clamped non-negative in the kernel.
    alpha: f32,
    /// `HG` weight of the `HG-Draine` blend, clamped to `[0, 1]` in the kernel.
    hg_weight: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `PhaseResult`
/// struct: the two phase values plus two pad words to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `Draine` phase value.
    draine: f32,
    /// `HG-Draine` blended phase value.
    hg_draine: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One phase-evaluation query: the scattering cosine and the asymmetry, shape,
/// and blend parameters driving the `Draine` and `HG-Draine` phases.
///
/// All arguments are passed through verbatim; the kernel applies the same
/// clamps the reference does (`cos_theta` to `[-1, 1]`, `g_hg` and `g_draine`
/// to `[-0.999, 0.999]`, `alpha` to non-negative, `hg_weight` to `[0, 1]`), so
/// out-of-range inputs are twinned exactly rather than rejected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrainePhaseBlendQuery {
    /// Cosine of the scattering angle (`+1` forward, `-1` back scatter).
    pub cos_theta: f32,
    /// `HG`-lobe asymmetry parameter.
    pub g_hg: f32,
    /// `Draine`-lobe asymmetry parameter.
    pub g_draine: f32,
    /// `Draine` forward-peak shape parameter.
    pub alpha: f32,
    /// `HG` weight of the `HG-Draine` convex blend.
    pub hg_weight: f32,
}

/// One resolved phase pair, mirroring the references
/// [`draine_phase`](prism_render_architecture::volumetric::scatter::draine_phase)
/// and
/// [`hg_draine_phase`](prism_render_architecture::volumetric::scatter::hg_draine_phase).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrainePhaseBlendResult {
    /// `Draine` phase value for `(cos_theta, g_draine, alpha)`.
    pub draine: f32,
    /// `HG-Draine` blended phase value.
    pub hg_draine: f32,
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

/// A compiled, reusable `Draine` and `HG-Draine` phase compute pipeline,
/// twinning the `CPU` goldens
/// [`draine_phase`](prism_render_architecture::volumetric::scatter::draine_phase)
/// and
/// [`hg_draine_phase`](prism_render_architecture::volumetric::scatter::hg_draine_phase).
pub struct GpuDrainePhaseBlend {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDrainePhaseBlend {
    /// Compiles the `Draine` and `HG-Draine` phase kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDrainePhaseBlend {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_draine_phase_blend"),
            source: ShaderSource::Wgsl(DRAINE_PHASE_BLEND_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_draine_phase_blend_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_draine_phase_blend_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_draine_phase_blend_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDrainePhaseBlend {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one
    /// [`DrainePhaseBlendResult`] per input, in order.
    ///
    /// Each phase value equals the reference to within the tolerance documented
    /// on this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[DrainePhaseBlendQuery],
    ) -> Vec<DrainePhaseBlendResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let encoded: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                cos_theta: q.cos_theta,
                g_hg: q.g_hg,
                g_draine: q.g_draine,
                alpha: q.alpha,
                hg_weight: q.hg_weight,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            })
            .collect();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_draine_phase_blend_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_draine_phase_blend_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_draine_phase_blend_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_draine_phase_blend_bind_group"),
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
            label: Some("prism_volumetric_draine_phase_blend_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_draine_phase_blend_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_draine_phase_blend_pass"),
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

        raw.iter()
            .map(|r| DrainePhaseBlendResult {
                draine: r.draine,
                hg_draine: r.hg_draine,
            })
            .collect()
    }
}
