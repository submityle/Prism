//! `wgpu` compute twin of the hair LOD cross-fade resolver from the `CPU`
//! golden `prism_render_architecture::hair::transition::resolve_hair_lod_transition`.
//!
//! A groom cannot pop between detail tiers in a single frame without reading as
//! a flicker. The reference spreads each tier switch across a small *coverage
//! band* around the hard threshold and reports a [`HairLodTransition`] of a
//! finer tier fading out and a coarser tier fading in. This module ports the
//! stateless, no-`RNG` resolver onto the device: one thread resolves one query,
//! so a passing real-device parity test is direct evidence the kernel takes the
//! same nearest-threshold branch and computes the same blend the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `resolve_hair_lod_transition`: the hard
//! tier pick `select_hair_lod_tier`, the `native_form` clamp via `coarser_of`,
//! and the fixed three-boundary nearest-threshold scan that picks the band the
//! coverage is inside and the linear `blend` across it. The per-strand
//! screen-door `strand_survives_dither` is deliberately *not* twinned: it hashes
//! with a `splitmix64`-style `u64` mixer the portable core-`WGSL` subset cannot
//! express.
//!
//! # Tier encoding
//!
//! Tiers are the `u32` coarseness rank of the reference `HairLodTier`:
//! `Strands = 0`, `ReducedStrands = 1`, `Cards = 2`, `Mesh = 3`. `coarser_of`
//! is then the integer `max` of two ranks, exactly as the reference compares
//! `coarseness()`.
//!
//! # Correctness model
//!
//! The only continuous output is `blend`, a single clamped divide, so the parity
//! test asserts `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on
//! it and compares the discrete `from_tier`, `to_tier`, `is_cross_fading` and
//! `valid` fields exactly. `is_finite(x)` is replicated identically on host and
//! device as `(x == x) && (abs(x) < 3.4e38)`: the `x == x` self-compare is the
//! portable `NaN` test (a `NaN` is the only value unequal to itself), and the
//! magnitude guard rejects the infinities, so no non-finite input slips into
//! the band arithmetic.
//!
//! # Degenerate inputs
//!
//! A `native_form` outside `0..=3` has no reference tier, so the twin reports
//! `valid = 0` with cleared outputs and the host mirrors that by skipping the
//! continuous comparison. A non-positive or non-finite `band`, or a non-finite
//! `coverage`, degrades to the hard tier pick exactly as the reference does. An
//! empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, `clamp`,
//! `abs`, ordered compares, integer `==`, `+ - * /` and unsigned index math —
//! with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no `f32`
//! remainder and no `u64`/`i64`/`f64`, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The one `f32` `==` is the documented `NaN` self-compare;
//! every other equality is on `u32` tier ranks.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::transition`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` hair LOD cross-fade kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `resolve_hair_lod_transition` branch for branch; see the
/// module documentation for the algorithm.
const HAIR_LOD_TRANSITION_WGSL: &str = r#"
// Hair LOD cross-fade twin: one thread per query reproduces
// resolve_hair_lod_transition. It picks the hard tier via select_hair_lod_tier,
// clamps both tiers no finer than native_form through coarser_of (integer max
// of coarseness ranks), and runs a fixed three-boundary nearest-threshold scan
// that chooses the band the coverage is inside and the linear blend across it.
// It uses only the portable core-WGSL subset (max/clamp/abs, ordered compares,
// integer ==, + - * / and unsigned index math), takes no optional feature, and
// has no unbounded loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::hair::transition；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Screen coverage of the groom this frame, a fraction in [0, 1].
    coverage: f32,
    // Below this coverage, drop from full strands to reduced strands.
    reduced_strands_below: f32,
    // Below this coverage, drop from reduced strands to cards.
    cards_below: f32,
    // Below this coverage, drop from cards to a static mesh shell.
    mesh_below: f32,
    // Half-width of the dissolve band on each side of a threshold.
    band: f32,
    // Coarseness rank the groom was authored at; tiers never go finer.
    native_form: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Finer tier (coarseness rank) being faded out.
    from_tier: u32,
    // Coarser tier (coarseness rank) being faded in.
    to_tier: u32,
    // Cross-fade position in [0, 1]; 0 when settled.
    blend: f32,
    // 1 when from_tier != to_tier, else 0.
    is_cross_fading: u32,
    // 1 when native_form is a real tier (0..=3), else 0 with cleared outputs.
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Largest magnitude accepted as finite; the two f32 infinities exceed it.
const F32_MAX_FINITE: f32 = 3.4e38;

// Portable finiteness test: x == x is the NaN self-compare (a NaN is the only
// value unequal to itself) and the magnitude guard rejects the infinities.
fn is_finite(x: f32) -> bool {
    return (x == x) && (abs(x) < F32_MAX_FINITE);
}

// Hard coverage-to-tier classification, mirroring select_hair_lod_tier.
fn select_tier(cov: f32, rsb: f32, cb: f32, mb: f32) -> u32 {
    if (cov >= rsb) {
        return 0u;
    }
    if (cov >= cb) {
        return 1u;
    }
    if (cov >= mb) {
        return 2u;
    }
    return 3u;
}

// Running best candidate across the three-boundary scan.
struct Best {
    // 1 once a boundary has claimed this coverage, else 0.
    found: u32,
    // Distance from coverage to the claimed threshold.
    dist: f32,
    // Finer tier rank at or above the threshold.
    finer: u32,
    // Coarser tier rank below the threshold.
    coarser: u32,
    // Linear blend across the band.
    blend: f32,
}

// Folds one boundary into the running best, keeping the nearest threshold whose
// band the coverage lies inside (first-seen wins on an exact distance tie).
fn consider(best: Best, threshold: f32, finer: u32, coarser: u32, cov: f32, band: f32) -> Best {
    if (!is_finite(threshold)) {
        return best;
    }
    let distance = abs(cov - threshold);
    if (distance >= band) {
        return best;
    }
    let blend = clamp((threshold + band - cov) / (2.0 * band), 0.0, 1.0);
    let is_closer = (best.found == 0u) || (distance < best.dist);
    if (is_closer) {
        var nb: Best;
        nb.found = 1u;
        nb.dist = distance;
        nb.finer = finer;
        nb.coarser = coarser;
        nb.blend = blend;
        return nb;
    }
    return best;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.from_tier = 0u;
    out.to_tier = 0u;
    out.blend = 0.0;
    out.is_cross_fading = 0u;
    out.valid = 0u;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    // A native_form outside the tier range has no reference answer.
    if (q.native_form > 3u) {
        results[idx] = out;
        return;
    }
    out.valid = 1u;

    let native = q.native_form;
    let cov = q.coverage;
    let band = q.band;

    // The hard tier pick, clamped no finer than native_form. Computed before the
    // degrade check so a NaN coverage classifies to Mesh exactly as the golden.
    let hard_tier = max(
        select_tier(cov, q.reduced_strands_below, q.cards_below, q.mesh_below),
        native,
    );

    // Non-positive or non-finite band, or non-finite coverage, disables fading.
    if (band <= 0.0 || !is_finite(band) || !is_finite(cov)) {
        out.from_tier = hard_tier;
        out.to_tier = hard_tier;
        out.blend = 0.0;
        out.is_cross_fading = 0u;
        results[idx] = out;
        return;
    }

    // Fixed three-boundary unrolled scan for the nearest band.
    var best: Best;
    best.found = 0u;
    best.dist = 0.0;
    best.finer = 0u;
    best.coarser = 0u;
    best.blend = 0.0;
    best = consider(best, q.reduced_strands_below, 0u, 1u, cov, band);
    best = consider(best, q.cards_below, 1u, 2u, cov, band);
    best = consider(best, q.mesh_below, 2u, 3u, cov, band);

    if (best.found == 0u) {
        out.from_tier = hard_tier;
        out.to_tier = hard_tier;
        out.blend = 0.0;
        out.is_cross_fading = 0u;
        results[idx] = out;
        return;
    }

    let lod_from = max(best.finer, native);
    let to = max(best.coarser, native);
    if (lod_from == to) {
        // native_form collapsed the band: no visible fade.
        out.from_tier = lod_from;
        out.to_tier = lod_from;
        out.blend = 0.0;
        out.is_cross_fading = 0u;
    } else {
        out.from_tier = lod_from;
        out.to_tier = to;
        out.blend = best.blend;
        out.is_cross_fading = 1u;
    }
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`HAIR_LOD_TRANSITION_WGSL`].
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
/// All scalar lanes, so the layout is a flat `32`-byte stride with no vector
/// alignment rule to trip; the two trailing pad words round the `u32` tail up to
/// a `16`-byte multiple so a batch of two or more packs contiguously.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    coverage: f32,
    reduced_strands_below: f32,
    cards_below: f32,
    mesh_below: f32,
    band: f32,
    native_form: u32,
    pad0: u32,
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Five live words plus three pad words give a flat `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    from_tier: u32,
    to_tier: u32,
    blend: f32,
    is_cross_fading: u32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One query for the hair LOD cross-fade twin: the groom's screen `coverage`,
/// the three descending tier thresholds, the dissolve `band` half-width and the
/// authored `native_form` coarseness rank.
///
/// The whole twinned resolver is driven by this one tuple, so a single query
/// exercises the hard pick, the `native_form` clamp and the nearest-band scan at
/// once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairLodTransitionQuery {
    /// Screen coverage of the groom, a fraction in `[0, 1]`.
    pub coverage: f32,
    /// Below this coverage, drop from full strands to reduced strands.
    pub reduced_strands_below: f32,
    /// Below this coverage, drop from reduced strands to cards.
    pub cards_below: f32,
    /// Below this coverage, drop from cards to a static mesh shell.
    pub mesh_below: f32,
    /// Half-width of the dissolve band on each side of a threshold; a
    /// non-positive `band` disables cross-fading.
    pub band: f32,
    /// Coarseness rank the groom was authored at (`Strands = 0` ..= `Mesh = 3`);
    /// resolved tiers never go finer than this.
    pub native_form: u32,
}

impl HairLodTransitionQuery {
    /// Builds a query from the coverage, the three thresholds, the band and the
    /// authored tier rank.
    #[must_use]
    pub fn new(
        coverage: f32,
        reduced_strands_below: f32,
        cards_below: f32,
        mesh_below: f32,
        band: f32,
        native_form: u32,
    ) -> HairLodTransitionQuery {
        HairLodTransitionQuery {
            coverage,
            reduced_strands_below,
            cards_below,
            mesh_below,
            band,
            native_form,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `resolve_hair_lod_transition` output.
///
/// Tiers are coarseness ranks (`Strands = 0` ..= `Mesh = 3`). When `from_tier`
/// equals `to_tier` the groom is settled at that single tier and `blend` is `0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairLodTransitionResult {
    /// Finer tier (coarseness rank) being faded out.
    pub from_tier: u32,
    /// Coarser tier (coarseness rank) being faded in.
    pub to_tier: u32,
    /// Cross-fade position in `0..=1`; `0` when settled.
    pub blend: f32,
    /// `1` when `from_tier != to_tier`, else `0`.
    pub is_cross_fading: u32,
    /// `1` when `native_form` is a real tier (`0..=3`); `0` with cleared outputs
    /// otherwise.
    pub valid: u32,
}

/// Encodes one [`HairLodTransitionQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &HairLodTransitionQuery) -> GpuQuery {
    GpuQuery {
        coverage: q.coverage,
        reduced_strands_below: q.reduced_strands_below,
        cards_below: q.cards_below,
        mesh_below: q.mesh_below,
        band: q.band,
        native_form: q.native_form,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`HairLodTransitionResult`].
fn decode_result(raw: &GpuResult) -> HairLodTransitionResult {
    HairLodTransitionResult {
        from_tier: raw.from_tier,
        to_tier: raw.to_tier,
        blend: raw.blend,
        is_cross_fading: raw.is_cross_fading,
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

/// A compiled, reusable hair LOD cross-fade compute pipeline, twinning the `CPU`
/// golden `resolve_hair_lod_transition`.
pub struct GpuHairLodTransition {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairLodTransition {
    /// Compiles the hair LOD cross-fade kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairLodTransition {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hair_lod_transition"),
            source: ShaderSource::Wgsl(HAIR_LOD_TRANSITION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hair_lod_transition_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hair_lod_transition_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hair_lod_transition_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairLodTransition {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`HairLodTransitionResult`] per input, in order.
    ///
    /// The `blend` channel matches the reference to within the tolerance
    /// documented on this module; the discrete tier and flag fields match
    /// exactly. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HairLodTransitionQuery],
    ) -> Vec<HairLodTransitionResult> {
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
            label: Some("prism_volumetric_hair_lod_transition_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_lod_transition_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_lod_transition_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hair_lod_transition_bind_group"),
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
            label: Some("prism_volumetric_hair_lod_transition_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hair_lod_transition_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hair_lod_transition_pass"),
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
