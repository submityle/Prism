//! `wgpu` compute twin of the particle LOD / quality-ladder decision contract
//! ([`lod`](prism_render_architecture::particle::lod), particle design §28).
//!
//! The `CPU` golden [`lod`](prism_render_architecture::particle::lod) owns the
//! small, fully discrete policy the particle render loop consumes before it
//! builds its indirect passes: it maps a screen-coverage fraction to a
//! [`ParticleLodTier`](prism_render_architecture::particle::lod::ParticleLodTier)
//! ([`select_particle_lod_tier`](prism_render_architecture::particle::lod::select_particle_lod_tier)),
//! clamps that tier against an emitter's authored native form
//! ([`ParticleLodTier::coarser_of`](prism_render_architecture::particle::lod::ParticleLodTier::coarser_of))
//! and reports whether the result still simulates
//! ([`ParticleLodTier::is_simulated`](prism_render_architecture::particle::lod::ParticleLodTier::is_simulated)),
//! clamps a requested quality to a platform ceiling
//! ([`resolve_quality`](prism_render_architecture::particle::lod::resolve_quality)),
//! applies the integer budget divisor
//! ([`budget_for_quality`](prism_render_architecture::particle::lod::budget_for_quality)
//! over
//! [`ParticleQuality::particle_divisor`](prism_render_architecture::particle::lod::ParticleQuality::particle_divisor)),
//! and walks the degradation staircase one rung
//! ([`ParticleQuality::degrade`](prism_render_architecture::particle::lod::ParticleQuality::degrade))
//! or several
//! ([`ParticleQuality::degrade_steps`](prism_render_architecture::particle::lod::ParticleQuality::degrade_steps)).
//! [`GpuParticleLod`] is the on-device twin: one thread resolves one query, so a
//! passing real-device parity test is direct evidence the ported kernel takes
//! the same branches and computes the same integer budgets the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced for a batch of
//! independent queries: the coverage-selected tier, the native-form clamp, the
//! `is_simulated` predicate, the platform-clamped quality, that quality's
//! divisor and the resulting particle budget (with the `.max(1)` floor and the
//! zero-budget short circuit), the single degradation rung (an [`Option`]
//! mirrored by a `u32` flag), and the saturating multi-step degradation.
//!
//! # Correctness model
//!
//! Every output is a tier code, a quality code, a `bool` or a `u32` integer
//! budget / divisor built from integer arithmetic and `f32` *threshold
//! comparisons* whose results are discrete classifications. There is no
//! continuous `f32` output, so `CPU` and `GPU` agree bit-exactly and the parity
//! test asserts an exact `==` on every field. The only `f32` operations are the
//! three `>=` coverage comparisons inside
//! [`select_particle_lod_tier`](prism_render_architecture::particle::lod::select_particle_lod_tier);
//! the fixtures keep every coverage at least `0.05` clear of each threshold so
//! the `>=` verdict can never straddle a boundary where a `GPU` fused rounding
//! might disagree with the reference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned integer
//! arithmetic, `min`, `max`, `f32` comparison and a bounded `loop` — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `sqrt`, no inverse trigonometry and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The single `loop` steps at most `steps` rungs and breaks at the
//! lowest quality, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::lod`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::lod::{ParticleLodTier, ParticleQuality, PlatformTier};
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

/// Tier code for [`ParticleLodTier::Full`](prism_render_architecture::particle::lod::ParticleLodTier::Full).
const TIER_FULL: u32 = 0;
/// Tier code for [`ParticleLodTier::Reduced`](prism_render_architecture::particle::lod::ParticleLodTier::Reduced).
const TIER_REDUCED: u32 = 1;
/// Tier code for [`ParticleLodTier::Impostor`](prism_render_architecture::particle::lod::ParticleLodTier::Impostor).
const TIER_IMPOSTOR: u32 = 2;
/// Tier code for [`ParticleLodTier::Culled`](prism_render_architecture::particle::lod::ParticleLodTier::Culled).
const TIER_CULLED: u32 = 3;

/// Quality code for [`ParticleQuality::Low`](prism_render_architecture::particle::lod::ParticleQuality::Low).
const QUALITY_LOW: u32 = 0;
/// Quality code for [`ParticleQuality::Medium`](prism_render_architecture::particle::lod::ParticleQuality::Medium).
const QUALITY_MEDIUM: u32 = 1;
/// Quality code for [`ParticleQuality::High`](prism_render_architecture::particle::lod::ParticleQuality::High).
const QUALITY_HIGH: u32 = 2;
/// Quality code for [`ParticleQuality::Ultra`](prism_render_architecture::particle::lod::ParticleQuality::Ultra).
const QUALITY_ULTRA: u32 = 3;

/// Platform code for [`PlatformTier::Mobile`](prism_render_architecture::particle::lod::PlatformTier::Mobile).
const PLATFORM_MOBILE: u32 = 0;
/// Platform code for [`PlatformTier::Console`](prism_render_architecture::particle::lod::PlatformTier::Console).
const PLATFORM_CONSOLE: u32 = 1;
/// Platform code for [`PlatformTier::Desktop`](prism_render_architecture::particle::lod::PlatformTier::Desktop).
const PLATFORM_DESKTOP: u32 = 2;
/// Platform code for [`PlatformTier::HighEnd`](prism_render_architecture::particle::lod::PlatformTier::HighEnd).
const PLATFORM_HIGH_END: u32 = 3;

/// The portable core-`WGSL` particle-LOD decision kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `resolve` mirrors
/// the `CPU` golden
/// [`lod`](prism_render_architecture::particle::lod) branch for branch; see the
/// module documentation for the policy.
const LOD_WGSL: &str = r#"
// Particle-LOD decision twin: one thread per query reproduces the
// coverage-selected tier, the native-form clamp, the is_simulated predicate, the
// platform-clamped quality, its divisor and particle budget, the single
// degradation rung and the saturating multi-step degradation. It mirrors the CPU
// golden `particle::lod` branch for branch, uses only the portable core-WGSL
// subset (unsigned integer arithmetic, min/max, f32 comparison and one bounded
// loop), needs no sqrt and no transcendental call and takes no optional feature,
// so it runs unmodified on Metal, Vulkan and DX12. The loop steps at most `steps`
// rungs and breaks at the lowest quality, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::lod；无第三方引擎源码
// 或衍生代码。

const TIER_FULL: u32 = 0u;
const TIER_REDUCED: u32 = 1u;
const TIER_IMPOSTOR: u32 = 2u;
const TIER_CULLED: u32 = 3u;

const QUALITY_LOW: u32 = 0u;
const QUALITY_MEDIUM: u32 = 1u;
const QUALITY_HIGH: u32 = 2u;
const QUALITY_ULTRA: u32 = 3u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Screen-coverage fraction and the three descending tier boundaries.
    coverage: f32,
    reduced_below: f32,
    impostor_below: f32,
    cull_below: f32,
    // Authored native (coarsest) form the coverage tier is clamped against, the
    // requested quality, the platform ceiling and the authored particle budget.
    native_form: u32,
    requested_quality: u32,
    platform: u32,
    max_particles: u32,
    // Number of degradation rungs for the saturating staircase; pad to 16 bytes.
    degrade_steps: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // select_particle_lod_tier verdict.
    selected_tier: u32,
    // selected_tier.coarser_of(native_form).
    coarser_tier: u32,
    // coarser_tier.is_simulated() as a 0/1 flag.
    simulated: u32,
    // resolve_quality(requested, platform).
    resolved_quality: u32,
    // resolved_quality.particle_divisor().
    divisor: u32,
    // budget_for_quality(max_particles, resolved_quality).
    budget: u32,
    // requested_quality.degrade(): flag is 1 when Some, 0 at Low; the code is the
    // degraded quality when the flag is 1, else the input left unchanged.
    degrade_flag: u32,
    degraded_quality: u32,
    // requested_quality.degrade_steps(degrade_steps), saturating at Low.
    degrade_steps_quality: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The platform's quality ceiling: Mobile->Low, Console->Medium, Desktop->High,
// HighEnd->Ultra. Since both codes share the same 0..3 ranking the ceiling code
// equals the platform code, but the branches are spelled out to mirror
// `PlatformTier::max_quality`.
fn max_quality_of(platform: u32) -> u32 {
    if (platform == 0u) {
        return QUALITY_LOW;
    }
    if (platform == 1u) {
        return QUALITY_MEDIUM;
    }
    if (platform == 2u) {
        return QUALITY_HIGH;
    }
    return QUALITY_ULTRA;
}

// The integer budget divisor: Ultra 1, High 2, Medium 4, Low 8. Mirrors
// `ParticleQuality::particle_divisor`.
fn divisor_of(quality: u32) -> u32 {
    if (quality == QUALITY_ULTRA) {
        return 1u;
    }
    if (quality == QUALITY_HIGH) {
        return 2u;
    }
    if (quality == QUALITY_MEDIUM) {
        return 4u;
    }
    return 8u;
}

@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // select_particle_lod_tier: three descending `>=` comparisons. The fixtures
    // keep coverage clear of each boundary so the verdict is unambiguous.
    var selected_tier: u32 = TIER_CULLED;
    if (q.coverage >= q.reduced_below) {
        selected_tier = TIER_FULL;
    } else if (q.coverage >= q.impostor_below) {
        selected_tier = TIER_REDUCED;
    } else if (q.coverage >= q.cull_below) {
        selected_tier = TIER_IMPOSTOR;
    } else {
        selected_tier = TIER_CULLED;
    }

    // coarser_of: the higher-rank (larger-code) tier. The rank equals the code,
    // so max picks the coarser form, matching `ParticleLodTier::coarser_of`.
    let coarser_tier = max(selected_tier, q.native_form);

    // is_simulated: Full or Reduced only (codes 0 and 1).
    var simulated: u32 = 0u;
    if (coarser_tier == TIER_FULL || coarser_tier == TIER_REDUCED) {
        simulated = 1u;
    }

    // resolve_quality: clamp the requested quality to the platform ceiling. The
    // rank equals the code, so min picks the lower, matching the reference.
    let cap = max_quality_of(q.platform);
    let resolved_quality = min(q.requested_quality, cap);

    // budget_for_quality: a zero budget stays zero; any positive budget keeps at
    // least one particle after the integer divisor.
    let divisor = divisor_of(resolved_quality);
    var budget: u32 = 0u;
    if (q.max_particles != 0u) {
        budget = max(q.max_particles / divisor, 1u);
    }

    // degrade(): one rung down, None at Low. The codes descend by one, so a rung
    // is a saturating decrement with a presence flag mirroring the Option.
    var degrade_flag: u32 = 0u;
    var degraded_quality: u32 = q.requested_quality;
    if (q.requested_quality != QUALITY_LOW) {
        degrade_flag = 1u;
        degraded_quality = q.requested_quality - 1u;
    }

    // degrade_steps(): step down `steps` rungs, saturating at Low. The loop runs
    // at most `steps` iterations and breaks once Low is reached.
    var staircase: u32 = q.requested_quality;
    var i: u32 = 0u;
    loop {
        if (i >= q.degrade_steps) {
            break;
        }
        if (staircase == QUALITY_LOW) {
            break;
        }
        staircase = staircase - 1u;
        i = i + 1u;
    }

    var out: Result;
    out.selected_tier = selected_tier;
    out.coarser_tier = coarser_tier;
    out.simulated = simulated;
    out.resolved_quality = resolved_quality;
    out.divisor = divisor;
    out.budget = budget;
    out.degrade_flag = degrade_flag;
    out.degraded_quality = degraded_quality;
    out.degrade_steps_quality = staircase;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`LOD_WGSL`].
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
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Screen-coverage fraction.
    coverage: f32,
    /// Drop-to-reduced boundary.
    reduced_below: f32,
    /// Drop-to-impostor boundary.
    impostor_below: f32,
    /// Cull boundary.
    cull_below: f32,
    /// Authored native-form tier code.
    native_form: u32,
    /// Requested quality code.
    requested_quality: u32,
    /// Platform tier code.
    platform: u32,
    /// Authored maximum particle budget.
    max_particles: u32,
    /// Number of degradation rungs for the staircase.
    degrade_steps: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `select_particle_lod_tier` verdict code.
    selected_tier: u32,
    /// Native-form-clamped tier code.
    coarser_tier: u32,
    /// `1` when the clamped tier simulates, `0` otherwise.
    simulated: u32,
    /// Platform-clamped quality code.
    resolved_quality: u32,
    /// Integer budget divisor at the resolved quality.
    divisor: u32,
    /// Particle budget after the divisor and `.max(1)` floor.
    budget: u32,
    /// `1` when `degrade` returned `Some`, `0` at the lowest quality.
    degrade_flag: u32,
    /// Degraded quality code (valid when `degrade_flag` is `1`).
    degraded_quality: u32,
    /// Saturating multi-step degradation result code.
    degrade_steps_quality: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query for the particle-LOD twin: a screen coverage and tier boundaries, an
/// authored native form, a requested quality, a platform ceiling, an authored
/// budget and a degradation-step count.
///
/// The tier selection, the quality clamp, the budget and the two degradation
/// paths are independent, so a single query exercises every twinned decision at
/// once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleLodQuery {
    /// Screen-coverage fraction in `0..=1`.
    pub coverage: f32,
    /// Below this coverage, drop from full to reduced simulation.
    pub reduced_below: f32,
    /// Below this coverage, drop from reduced simulation to an impostor.
    pub impostor_below: f32,
    /// Below this coverage, cull the emitter entirely.
    pub cull_below: f32,
    /// Authored coarsest form the coverage tier is clamped against.
    pub native_form: ParticleLodTier,
    /// Requested rendering quality before the platform clamp.
    pub requested_quality: ParticleQuality,
    /// Hardware class whose ceiling clamps the requested quality.
    pub platform: PlatformTier,
    /// Authored maximum live particle count the divisor scales.
    pub max_particles: u32,
    /// Number of rungs to step down the degradation staircase.
    pub degrade_steps: u32,
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports across its twinned decisions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleLodResult {
    /// Coverage-selected tier, matching
    /// [`select_particle_lod_tier`](prism_render_architecture::particle::lod::select_particle_lod_tier).
    pub selected_tier: ParticleLodTier,
    /// Native-form-clamped tier, matching
    /// [`ParticleLodTier::coarser_of`](prism_render_architecture::particle::lod::ParticleLodTier::coarser_of).
    pub coarser_tier: ParticleLodTier,
    /// Whether the clamped tier simulates, matching
    /// [`ParticleLodTier::is_simulated`](prism_render_architecture::particle::lod::ParticleLodTier::is_simulated).
    pub simulated: bool,
    /// Platform-clamped quality, matching
    /// [`resolve_quality`](prism_render_architecture::particle::lod::resolve_quality).
    pub resolved_quality: ParticleQuality,
    /// Integer divisor at the resolved quality, matching
    /// [`ParticleQuality::particle_divisor`](prism_render_architecture::particle::lod::ParticleQuality::particle_divisor).
    pub divisor: u32,
    /// Particle budget, matching
    /// [`budget_for_quality`](prism_render_architecture::particle::lod::budget_for_quality).
    pub budget: u32,
    /// One degradation rung, matching
    /// [`ParticleQuality::degrade`](prism_render_architecture::particle::lod::ParticleQuality::degrade).
    pub degraded: Option<ParticleQuality>,
    /// Saturating multi-step degradation, matching
    /// [`ParticleQuality::degrade_steps`](prism_render_architecture::particle::lod::ParticleQuality::degrade_steps).
    pub degrade_steps_quality: ParticleQuality,
}

/// Maps a [`ParticleLodTier`] to its `WGSL` tier code.
fn tier_code(tier: ParticleLodTier) -> u32 {
    match tier {
        ParticleLodTier::Full => TIER_FULL,
        ParticleLodTier::Reduced => TIER_REDUCED,
        ParticleLodTier::Impostor => TIER_IMPOSTOR,
        ParticleLodTier::Culled => TIER_CULLED,
    }
}

/// Maps a `WGSL` tier code back to a [`ParticleLodTier`].
///
/// # Panics
///
/// Panics on a code outside `0..=3`, which cannot occur: the kernel only ever
/// writes one of the four tier codes.
fn tier_from_code(code: u32) -> ParticleLodTier {
    match code {
        TIER_FULL => ParticleLodTier::Full,
        TIER_REDUCED => ParticleLodTier::Reduced,
        TIER_IMPOSTOR => ParticleLodTier::Impostor,
        TIER_CULLED => ParticleLodTier::Culled,
        other => panic!("kernel wrote an out-of-range tier code: {other}"),
    }
}

/// Maps a [`ParticleQuality`] to its `WGSL` quality code.
fn quality_code(quality: ParticleQuality) -> u32 {
    match quality {
        ParticleQuality::Low => QUALITY_LOW,
        ParticleQuality::Medium => QUALITY_MEDIUM,
        ParticleQuality::High => QUALITY_HIGH,
        ParticleQuality::Ultra => QUALITY_ULTRA,
    }
}

/// Maps a `WGSL` quality code back to a [`ParticleQuality`].
///
/// # Panics
///
/// Panics on a code outside `0..=3`, which cannot occur: the kernel only ever
/// writes one of the four quality codes.
fn quality_from_code(code: u32) -> ParticleQuality {
    match code {
        QUALITY_LOW => ParticleQuality::Low,
        QUALITY_MEDIUM => ParticleQuality::Medium,
        QUALITY_HIGH => ParticleQuality::High,
        QUALITY_ULTRA => ParticleQuality::Ultra,
        other => panic!("kernel wrote an out-of-range quality code: {other}"),
    }
}

/// Maps a [`PlatformTier`] to its `WGSL` platform code.
fn platform_code(platform: PlatformTier) -> u32 {
    match platform {
        PlatformTier::Mobile => PLATFORM_MOBILE,
        PlatformTier::Console => PLATFORM_CONSOLE,
        PlatformTier::Desktop => PLATFORM_DESKTOP,
        PlatformTier::HighEnd => PLATFORM_HIGH_END,
    }
}

/// Encodes one [`ParticleLodQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ParticleLodQuery) -> GpuQuery {
    GpuQuery {
        coverage: q.coverage,
        reduced_below: q.reduced_below,
        impostor_below: q.impostor_below,
        cull_below: q.cull_below,
        native_form: tier_code(q.native_form),
        requested_quality: quality_code(q.requested_quality),
        platform: platform_code(q.platform),
        max_particles: q.max_particles,
        degrade_steps: q.degrade_steps,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ParticleLodResult`],
/// turning the discrete codes back into enums and the degrade flag into an
/// [`Option`].
fn decode_result(raw: &GpuResult) -> ParticleLodResult {
    ParticleLodResult {
        selected_tier: tier_from_code(raw.selected_tier),
        coarser_tier: tier_from_code(raw.coarser_tier),
        simulated: raw.simulated != 0,
        resolved_quality: quality_from_code(raw.resolved_quality),
        divisor: raw.divisor,
        budget: raw.budget,
        degraded: if raw.degrade_flag == 0 {
            None
        } else {
            Some(quality_from_code(raw.degraded_quality))
        },
        degrade_steps_quality: quality_from_code(raw.degrade_steps_quality),
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

/// A compiled, reusable particle-LOD decision compute pipeline, twinning the
/// `CPU` golden [`lod`](prism_render_architecture::particle::lod).
pub struct GpuParticleLod {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuParticleLod {
    /// Compiles the particle-LOD kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuParticleLod {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_lod"),
            source: ShaderSource::Wgsl(LOD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_lod_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_lod_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_lod_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("resolve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuParticleLod {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one [`ParticleLodResult`]
    /// per input, in order.
    ///
    /// Every field equals the reference exactly: the outputs are tier / quality
    /// codes and integer budgets, and the fixtures keep coverage clear of each
    /// threshold so the discrete classification never straddles a boundary. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ParticleLodQuery],
    ) -> Vec<ParticleLodResult> {
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
            label: Some("prism_volumetric_lod_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_lod_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_lod_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_lod_bind_group"),
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
            label: Some("prism_volumetric_lod_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_lod_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_lod_pass"),
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
