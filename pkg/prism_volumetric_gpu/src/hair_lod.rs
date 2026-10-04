//! `wgpu` compute twin of the screen-coverage hair LOD selection from the `CPU`
//! golden `prism_render_architecture::hair::lod`.
//!
//! A groom cannot render every strand at every distance: a character filling
//! the screen needs full strands, while the same character across the map
//! should collapse to cards or a mesh shell. The golden module maps a hair
//! group's projected screen coverage to a discrete [`HairLodTier`] and then
//! resolves how many render strands and control points per strand that tier
//! keeps. This twin ports the two pure, deterministic classification functions:
//! `select_hair_lod_tier` and `resolve_hair_lod`. One thread solves one query.
//!
//! The caller supplies coverage as a screen fraction in `0..=1`, so there is no
//! projection or transcendental math anywhere in the kernel. The tier is picked
//! by an ordered coverage ladder, then clamped to be no finer than the group's
//! authored `native_form` (a card-authored groom is never promoted to strands
//! it does not own). Strand budgets decimate by fixed integer factors, so the
//! result is exactly reproducible frame to frame and bit-exact with the golden.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, independently of the golden crate:
//! `select_hair_lod_tier(coverage, thresholds)` — the coverage ladder
//! `Strands` / `ReducedStrands` / `Cards` / `Mesh` (tier indices `0..=3`); and
//! `resolve_hair_lod(group, coverage, thresholds)` — the coarser-of clamp
//! against `native_form` plus the per-tier strand/segment budget. `Strands`
//! keeps the authored counts; `ReducedStrands` decimates to a quarter of the
//! strands and half the control points (never below one); `Cards` and `Mesh`
//! keep no per-strand geometry (`0`, `0`). There is no loop: each thread
//! performs a fixed, bounded sequence of ordered compares, selects and integer
//! divisions, so the kernel provably terminates.
//!
//! # Correctness model
//!
//! Every output is a discrete integer (the tier index, the strand and segment
//! budgets, the validity flag), so the parity test compares every field with an
//! exact `==`: there is no floating-point result to tolerance. The coverage
//! ladder uses ordered `>=` comparisons with `select`, which match the golden
//! `if`/`else if` chain exactly on the same `f32` inputs.
//!
//! # Degenerate inputs
//!
//! A `native_form` greater than `3` is not a valid tier index; such a query is
//! rejected with `valid = 0` and all outputs cleared to `0`, and the host
//! oracle mirrors that rejection. Every other query (any coverage, any
//! threshold triple, even thresholds that violate the coarsest-last invariant)
//! classifies deterministically with `valid = 1`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `select`, `max`,
//! integer `u32` arithmetic and ordered `f32` compares — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, no `round`, no float `%`, and no `u64` / `i64` / `f64`,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The coverage ladder
//! avoids bare float equality entirely by using ordered `>=` comparisons.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::lod`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` hair LOD kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden `select_hair_lod_tier` and `resolve_hair_lod`; see the module
/// documentation.
const HAIR_LOD_WGSL: &str = r#"
// Hair LOD twin: one thread per query maps a screen coverage to a discrete tier
// through an ordered coverage ladder, clamps that tier to be no finer than the
// group's authored native_form, then resolves the per-tier strand and segment
// budget with integer decimation. It mirrors the CPU golden exactly and uses
// only the portable core subset.

struct Params {
    // Number of valid queries in the input and output buffers.
    count: u32,
    // Padding words so the uniform struct fills 16 bytes.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Projected screen coverage fraction in [0, 1].
    coverage: f32,
    // Below this coverage, drop from full strands to reduced strands.
    reduced_strands_below: f32,
    // Below this coverage, drop from reduced strands to cards.
    cards_below: f32,
    // Below this coverage, drop from cards to a static mesh shell.
    mesh_below: f32,
    // Finest authored tier index (0..=3); a larger value rejects the query.
    native_form: u32,
    // Authored render-strand count at the finest tier.
    max_render_strands: u32,
    // Authored control points per strand at the finest tier.
    segments_per_strand: u32,
    // Padding word to a 32-byte stride.
    pad0: u32,
}

struct Result {
    // Resolved coarsest-of tier index (0..=3).
    tier: u32,
    // Render strands kept at this tier (0 for card/mesh proxies).
    render_strands: u32,
    // Control points per strand at this tier (0 for card/mesh proxies).
    segments_per_strand: u32,
    // 1 for a classified query, 0 when native_form is out of range.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Classifies a coverage into a tier index via the ordered coverage ladder,
// mirroring the golden if / else-if chain with select on ordered compares.
fn select_tier(coverage: f32, reduced_below: f32, cards_below: f32, mesh_below: f32) -> u32 {
    // Default coarsest; progressively refine toward finer tiers.
    var tier: u32 = 3u;
    tier = select(tier, 2u, coverage >= mesh_below);
    tier = select(tier, 1u, coverage >= cards_below);
    tier = select(tier, 0u, coverage >= reduced_below);
    return tier;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    // native_form must be a valid tier index 0..=3; otherwise reject.
    if (q.native_form > 3u) {
        out.tier = 0u;
        out.render_strands = 0u;
        out.segments_per_strand = 0u;
        out.valid = 0u;
        results[idx] = out;
        return;
    }

    let coverage_tier = select_tier(
        q.coverage,
        q.reduced_strands_below,
        q.cards_below,
        q.mesh_below,
    );
    // coarser_of: the tier with the larger coarseness index wins.
    let tier = max(coverage_tier, q.native_form);

    // Per-tier strand/segment budget. Strands (0) keep authored counts;
    // ReducedStrands (1) decimate by integer factors, never below one; Cards (2)
    // and Mesh (3) keep no per-strand geometry.
    let reduced_strands = max(q.max_render_strands / 4u, 1u);
    let reduced_segments = max(q.segments_per_strand / 2u, 1u);

    var render_strands: u32 = 0u;
    var segments: u32 = 0u;
    render_strands = select(render_strands, reduced_strands, tier == 1u);
    segments = select(segments, reduced_segments, tier == 1u);
    render_strands = select(render_strands, q.max_render_strands, tier == 0u);
    segments = select(segments, q.segments_per_strand, tier == 0u);

    out.tier = tier;
    out.render_strands = render_strands;
    out.segments_per_strand = segments;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in the kernel.
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
/// Four `f32` thresholds followed by three `u32` budgets and one trailing pad
/// word give a fixed `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    coverage: f32,
    reduced_strands_below: f32,
    cards_below: f32,
    mesh_below: f32,
    native_form: u32,
    max_render_strands: u32,
    segments_per_strand: u32,
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Four `u32` words give a fixed `16`-byte stride with no padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    tier: u32,
    render_strands: u32,
    segments_per_strand: u32,
    valid: u32,
}

/// One query for the hair LOD twin: the projected screen coverage, the three
/// coverage thresholds, the group's authored finest tier, and the authored
/// render-strand and segment budgets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairLodQuery {
    /// Projected screen coverage fraction in `0..=1`.
    pub coverage: f32,
    /// Below this coverage, drop from full strands to reduced strands.
    pub reduced_strands_below: f32,
    /// Below this coverage, drop from reduced strands to cards.
    pub cards_below: f32,
    /// Below this coverage, drop from cards to a static mesh shell.
    pub mesh_below: f32,
    /// Finest authored tier index (`0` = `Strands` .. `3` = `Mesh`); a larger
    /// value rejects the query.
    pub native_form: u32,
    /// Authored render-strand count at the finest tier.
    pub max_render_strands: u32,
    /// Authored control points per strand at the finest tier.
    pub segments_per_strand: u32,
}

impl HairLodQuery {
    /// Builds a query from the coverage, the three thresholds, the authored
    /// `native_form` tier index and the authored strand/segment budgets.
    #[must_use]
    pub fn new(
        coverage: f32,
        reduced_strands_below: f32,
        cards_below: f32,
        mesh_below: f32,
        native_form: u32,
        max_render_strands: u32,
        segments_per_strand: u32,
    ) -> HairLodQuery {
        HairLodQuery {
            coverage,
            reduced_strands_below,
            cards_below,
            mesh_below,
            native_form,
            max_render_strands,
            segments_per_strand,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `select_hair_lod_tier` and `resolve_hair_lod`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HairLodResult {
    /// Resolved coarsest-of tier index (`0` = `Strands` .. `3` = `Mesh`).
    pub tier: u32,
    /// Render strands kept at this tier (`0` for card/mesh proxies).
    pub render_strands: u32,
    /// Control points per strand at this tier (`0` for card/mesh proxies).
    pub segments_per_strand: u32,
    /// `1` for a classified query, `0` when `native_form` is out of range.
    pub valid: u32,
}

/// Encodes one [`HairLodQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &HairLodQuery) -> GpuQuery {
    GpuQuery {
        coverage: q.coverage,
        reduced_strands_below: q.reduced_strands_below,
        cards_below: q.cards_below,
        mesh_below: q.mesh_below,
        native_form: q.native_form,
        max_render_strands: q.max_render_strands,
        segments_per_strand: q.segments_per_strand,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`HairLodResult`].
fn decode_result(raw: &GpuResult) -> HairLodResult {
    HairLodResult {
        tier: raw.tier,
        render_strands: raw.render_strands,
        segments_per_strand: raw.segments_per_strand,
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

/// A compiled, reusable hair LOD compute pipeline, twinning the `CPU` golden
/// `select_hair_lod_tier` and `resolve_hair_lod`.
pub struct GpuHairLod {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairLod {
    /// Compiles the hair LOD kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairLod {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hair_lod"),
            source: ShaderSource::Wgsl(HAIR_LOD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hair_lod_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hair_lod_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hair_lod_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairLod {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`HairLodResult`] per
    /// input, in order.
    ///
    /// Every field matches the reference exactly, since all outputs are
    /// discrete integers. An empty `queries` batch returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[HairLodQuery]) -> Vec<HairLodResult> {
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
            label: Some("prism_volumetric_hair_lod_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_lod_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_lod_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hair_lod_bind_group"),
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
            label: Some("prism_volumetric_hair_lod_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hair_lod_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hair_lod_pass"),
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
