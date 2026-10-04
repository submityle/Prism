//! `wgpu` compute twin of the screen-coverage cloth level-of-detail selection
//! and sim-budget resolution from the `CPU` golden
//! `prism_render_architecture::cloth::lod`.
//!
//! A garment cannot solve every sim-mesh vertex at every distance: a hero
//! character filling the screen earns the full solve, while the same character
//! across the map collapses to a reduced solve or a plain skinned shell. The
//! reference maps a cloth piece's screen coverage onto a discrete
//! `ClothLodTier`, clamps that tier to be no finer than the piece's authored
//! `native_form`, resolves how many sim vertices and constraints the tier
//! keeps, and offers a hysteretic variant that holds the current tier inside a
//! symmetric dead-band so a garment hovering on a threshold does not pop frame
//! to frame. This module ports those stateless, closed-form classifiers onto
//! the device: one thread resolves one query, computing *both* the stateless
//! resolve (its tier plus sim-vertex / constraint budget) and the hysteretic
//! tier in the same invocation.
//!
//! [`GpuClothLod`] is the on-device twin: a passing real-device parity test is
//! direct evidence the kernel takes the same ordered-threshold branch and the
//! same integer decimation the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! * `select_cloth_lod_tier(coverage, thresholds)` — the ordered coverage
//!   classifier: at or above `reduced_sim_below` it is `FullSim (0)`, else at or
//!   above `skinned_below` it is `ReducedSim (1)`, else `SkinnedProxy (2)`.
//! * `cloth_lod_budget(piece, tier)` — the single decimation rule: full sim
//!   keeps the authored counts, reduced sim keeps `max(count / 4, 1)` of each,
//!   and the skinned proxy keeps none.
//! * `resolve_cloth_lod` — the stateless gate: the coverage-selected tier is
//!   coarsened to be no finer than `native_form` (`coarser_of` is the maximum
//!   coarseness index), then charged through `cloth_lod_budget`.
//! * `select_cloth_lod_tier_hysteretic(coverage, thresholds, hysteresis,
//!   current)` — the pop-free gate: a symmetric band around each boundary holds
//!   the `current` tier until coverage falls a full band below (to coarsen) or
//!   rises a full band above (to refine); `hysteresis == 0` reduces it exactly
//!   to the stateless classifier.
//!
//! The `Vec`-producing `bin_cloth_lod` and the `cloth_deformation_request`
//! builder are deliberately *not* twinned, since the former is a stateful
//! bucket scatter and the latter emits a scheduler request type.
//!
//! # Tier encoding
//!
//! Tiers are encoded as their coarseness index: `FullSim = 0`,
//! `ReducedSim = 1`, `SkinnedProxy = 2`, so `coarser_of` is a plain `max` of
//! two indices and every comparison stays integer.
//!
//! # Correctness model
//!
//! Every quantity threads through ordered float compares, integer division and
//! `max`, so the result is a discrete tier and two integer budgets. The parity
//! test compares every field exactly: there is no continuous channel to admit a
//! fused multiply-add drift. A query whose `current` or `native_form` encodes an
//! out-of-range tier (`> 2`) is rejected: the kernel reports `valid = 0` with
//! cleared outputs, mirrored by the host oracle.
//!
//! # Degenerate inputs
//!
//! A zero hysteresis band collapses both edges of every boundary onto the
//! authored threshold, so the hysteretic tier equals the stateless tier for
//! every `current`. A non-positive sim-vertex or constraint count simply
//! decimates to the `max(_, 1)` floor on a simulated tier. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `select`, `max`, unsigned integer division and arithmetic — with no `sin`,
//! `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no float modulo, no bare
//! float equality and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::lod`；无第三方引擎源码或衍生代码。
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

/// Tier encoding for full-resolution simulation (finest, coarseness `0`).
pub const TIER_FULL_SIM: u32 = 0;
/// Tier encoding for reduced-resolution simulation (coarseness `1`).
pub const TIER_REDUCED_SIM: u32 = 1;
/// Tier encoding for the static skinned proxy (coarsest, coarseness `2`).
pub const TIER_SKINNED_PROXY: u32 = 2;

/// The portable core-`WGSL` cloth-LOD kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden `resolve_cloth_lod` and `select_cloth_lod_tier_hysteretic` branch for
/// branch; see the module documentation for the algorithm.
const CLOTH_LOD_WGSL: &str = r#"
// Cloth-LOD twin: one thread per query reproduces the stateless resolve (its
// coverage-selected tier clamped to native_form, plus the sim-vertex and
// constraint budget) and the hysteretic tier of cloth::lod. Tiers are encoded
// as their coarseness index: FullSim = 0, ReducedSim = 1, SkinnedProxy = 2.
// Provenance: 孪生自本仓 prism_render_architecture::cloth::lod；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Screen coverage in [0, 1].
    coverage: f32,
    // Below this coverage, drop from full sim to reduced sim.
    reduced_sim_below: f32,
    // Below this coverage, drop from reduced sim to a skinned proxy.
    skinned_below: f32,
    // Symmetric coverage dead-band for the hysteretic gate.
    hysteresis: f32,
    // Current tier for the hysteretic gate (0/1/2).
    current: u32,
    // Finest tier the piece has geometry for (0/1/2); clamps the stateless tier.
    native_form: u32,
    // Authored sim-mesh vertex count at the finest tier.
    sim_vertex_count: u32,
    // Authored constraint count at the finest tier.
    constraint_count: u32,
}

struct Result {
    // Stateless resolved tier (coverage-selected, clamped to native_form).
    tier_stateless: u32,
    // Sim vertices solved at the stateless tier.
    sim_vertices: u32,
    // Constraints solved at the stateless tier.
    constraints: u32,
    // Hysteretic tier given the current tier and dead-band.
    tier_hysteretic: u32,
    // 1 for a well-formed query, 0 when current or native_form exceeds 2.
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Ordered coverage classifier: at or above reduced_sim_below -> FullSim (0),
// else at or above skinned_below -> ReducedSim (1), else SkinnedProxy (2).
fn select_tier(coverage: f32, reduced_sim_below: f32, skinned_below: f32) -> u32 {
    let full = coverage >= reduced_sim_below;
    let reduced = coverage >= skinned_below;
    return select(select(2u, 1u, reduced), 0u, full);
}

// Single decimation rule: full sim keeps the authored counts, reduced sim keeps
// max(count / 4, 1) of each, the skinned proxy keeps none.
fn budget_sim_vertices(tier: u32, sim_vertex_count: u32) -> u32 {
    let reduced = max(sim_vertex_count / 4u, 1u);
    return select(select(0u, reduced, tier == 1u), sim_vertex_count, tier == 0u);
}

fn budget_constraints(tier: u32, constraint_count: u32) -> u32 {
    let reduced = max(constraint_count / 4u, 1u);
    return select(select(0u, reduced, tier == 1u), constraint_count, tier == 0u);
}

// Hysteretic classifier: a symmetric band around each boundary holds the
// current tier until coverage falls a full band below (to coarsen) or rises a
// full band above (to refine). band == 0 reduces this to select_tier exactly.
fn select_tier_hysteretic(
    coverage: f32,
    reduced_sim_below: f32,
    skinned_below: f32,
    hysteresis: f32,
    current: u32,
) -> u32 {
    let band = max(hysteresis, 0.0);
    let reduced_down = reduced_sim_below - band;
    let reduced_up = reduced_sim_below + band;
    let skinned_down = skinned_below - band;
    let skinned_up = skinned_below + band;

    if (current == 0u) {
        // FullSim.
        let coarse = select(1u, 2u, coverage < skinned_down);
        return select(0u, coarse, coverage < reduced_down);
    }
    if (current == 1u) {
        // ReducedSim.
        let lower = select(1u, 2u, coverage < skinned_down);
        return select(lower, 0u, coverage >= reduced_up);
    }
    // SkinnedProxy.
    let upper = select(2u, 1u, coverage >= skinned_up);
    return select(upper, 0u, coverage >= reduced_up);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.tier_stateless = 0u;
    out.sim_vertices = 0u;
    out.constraints = 0u;
    out.tier_hysteretic = 0u;
    out.valid = 0u;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    // Reject an out-of-range current or native_form tier.
    if (q.current > 2u || q.native_form > 2u) {
        results[idx] = out;
        return;
    }

    let selected = select_tier(q.coverage, q.reduced_sim_below, q.skinned_below);
    // coarser_of is the maximum coarseness index.
    let tier = max(selected, q.native_form);

    out.tier_stateless = tier;
    out.sim_vertices = budget_sim_vertices(tier, q.sim_vertex_count);
    out.constraints = budget_constraints(tier, q.constraint_count);
    out.tier_hysteretic = select_tier_hysteretic(
        q.coverage,
        q.reduced_sim_below,
        q.skinned_below,
        q.hysteresis,
        q.current,
    );
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` layout of the dispatch parameters.
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
/// Four `f32` lanes followed by four `u32` lanes keep the stride a flat `32`
/// bytes with no vector-alignment padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    coverage: f32,
    reduced_sim_below: f32,
    skinned_below: f32,
    hysteresis: f32,
    current: u32,
    native_form: u32,
    sim_vertex_count: u32,
    constraint_count: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Five payload words plus three padding words keep the stride a flat
/// `32` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    tier_stateless: u32,
    sim_vertices: u32,
    constraints: u32,
    tier_hysteretic: u32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One cloth-LOD query: the screen coverage and the authored thresholds that
/// drive the stateless classifier, the hysteresis band and current tier for the
/// hysteretic classifier, the piece's `native_form` clamp, and the authored sim
/// counts the budget rule decimates.
///
/// A single query exercises the whole twinned core at once: it yields both the
/// stateless resolve (tier plus budget) and the hysteretic tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothLodQuery {
    /// Screen coverage in `[0, 1]`.
    pub coverage: f32,
    /// Below this coverage, drop from full sim to reduced sim.
    pub reduced_sim_below: f32,
    /// Below this coverage, drop from reduced sim to a skinned proxy.
    pub skinned_below: f32,
    /// Symmetric coverage dead-band for the hysteretic gate.
    pub hysteresis: f32,
    /// Current tier for the hysteretic gate (`0`/`1`/`2`).
    pub current: u32,
    /// Finest tier the piece has geometry for (`0`/`1`/`2`).
    pub native_form: u32,
    /// Authored sim-mesh vertex count at the finest tier.
    pub sim_vertex_count: u32,
    /// Authored constraint count at the finest tier.
    pub constraint_count: u32,
}

impl ClothLodQuery {
    /// Builds a cloth-LOD query from its coverage, thresholds, hysteresis state
    /// and authored counts.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query mirrors the golden thresholds, hysteresis state and piece counts verbatim"
    )]
    pub fn new(
        coverage: f32,
        reduced_sim_below: f32,
        skinned_below: f32,
        hysteresis: f32,
        current: u32,
        native_form: u32,
        sim_vertex_count: u32,
        constraint_count: u32,
    ) -> ClothLodQuery {
        ClothLodQuery {
            coverage,
            reduced_sim_below,
            skinned_below,
            hysteresis,
            current,
            native_form,
            sim_vertex_count,
            constraint_count,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `resolve_cloth_lod` decision plus the `select_cloth_lod_tier_hysteretic`
/// tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClothLodResult {
    /// Stateless resolved tier (coverage-selected, clamped to `native_form`).
    pub tier_stateless: u32,
    /// Sim vertices solved at the stateless tier.
    pub sim_vertices: u32,
    /// Constraints solved at the stateless tier.
    pub constraints: u32,
    /// Hysteretic tier given the current tier and dead-band.
    pub tier_hysteretic: u32,
    /// `1` for a well-formed query, `0` when `current` or `native_form`
    /// exceeds `2`.
    pub valid: u32,
}

/// Encodes one [`ClothLodQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ClothLodQuery) -> GpuQuery {
    GpuQuery {
        coverage: q.coverage,
        reduced_sim_below: q.reduced_sim_below,
        skinned_below: q.skinned_below,
        hysteresis: q.hysteresis,
        current: q.current,
        native_form: q.native_form,
        sim_vertex_count: q.sim_vertex_count,
        constraint_count: q.constraint_count,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ClothLodResult`].
fn decode_result(raw: &GpuResult) -> ClothLodResult {
    ClothLodResult {
        tier_stateless: raw.tier_stateless,
        sim_vertices: raw.sim_vertices,
        constraints: raw.constraints,
        tier_hysteretic: raw.tier_hysteretic,
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

/// A compiled, reusable cloth-LOD compute pipeline, twinning the `CPU` golden
/// `resolve_cloth_lod` and `select_cloth_lod_tier_hysteretic`.
pub struct GpuClothLod {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothLod {
    /// Compiles the cloth-LOD kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothLod {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_lod"),
            source: ShaderSource::Wgsl(CLOTH_LOD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_lod_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_lod_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_lod_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothLod {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`ClothLodResult`] per
    /// input, in order.
    ///
    /// Every field matches the reference exactly; there is no continuous
    /// channel. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[ClothLodQuery]) -> Vec<ClothLodResult> {
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
            label: Some("prism_volumetric_cloth_lod_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_lod_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_lod_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_lod_bind_group"),
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
            label: Some("prism_volumetric_cloth_lod_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_lod_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_lod_pass"),
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
