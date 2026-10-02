//! `wgpu` compute twin of the flocking / `Boids` steering contract
//! ([`boids`](prism_render_architecture::particle::boids), particle design
//! §10, §28, §29).
//!
//! The `CPU` golden [`boids`](prism_render_architecture::particle::boids) owns
//! Reynolds' three classic steering rules — *separation*, *alignment* and
//! *cohesion* — plus *goal seeking*, the fixed-order weighted
//! [`combine_forces`](prism_render_architecture::particle::boids::combine_forces)
//! composition, the kinematic clamps
//! ([`clamp_length`](prism_render_architecture::particle::boids::clamp_length),
//! [`limit_turn`](prism_render_architecture::particle::boids::limit_turn)), the
//! semi-implicit [`integrate`](prism_render_architecture::particle::boids::integrate)
//! step, the full [`steer`](prism_render_architecture::particle::boids::steer)
//! pipeline and the spatial-hash cell mapping
//! [`boid_cell`](prism_render_architecture::particle::boids::boid_cell).
//! [`GpuBoids`] is the on-device twin: one thread per boid reproduces every one
//! of those per-element answers, so a passing real-device parity test is direct
//! evidence the ported kernel runs the same steering math and classifies the
//! same degenerate cases the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For each queried boid the kernel reproduces, in the reference's own order:
//! the normalized `separation`, `alignment` and `cohesion` directions over its
//! neighbor list, the optional `goal` direction, the weighted `combined`
//! steering force, the turn-limited `turned` force, the force-clamped `steer`
//! acceleration (the value the golden
//! [`steer`](prism_render_architecture::particle::boids::steer) returns), the
//! speed-clamped `integrated` velocity, and the integer spatial-hash `cell`.
//! Two dedicated probes additionally exercise
//! [`clamp_length`](prism_render_architecture::particle::boids::clamp_length)
//! and [`limit_turn`](prism_render_architecture::particle::boids::limit_turn)
//! on caller-supplied vectors so the clamp branches are covered directly, not
//! only along the steering pipeline.
//!
//! Neighbor discovery itself is *not* twinned: the golden
//! `visible_neighbors` is built on the host-only spatial hash
//! `NeighborGrid` (a stable counting sort plus a 27-cell ball query), which has
//! no fixed-size on-device analogue here. The host therefore gathers each
//! boid's visible neighbors on the `CPU` and feeds the kernel a per-element,
//! fixed-upper-bound index list (padded to [`MAX_NEIGHBORS`]); the kernel only
//! consumes the `neighbor_count` valid entries. The classification/budget map
//! `boids_quality_budget` is likewise left on the host: it is a pure
//! enum-to-struct lookup with no `f32` arithmetic to port.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `floor`, `dot`, `+ - * /` and one `sqrt` for the genuine
//! Euclidean length normalizations — with no `sin`, `cos`, `acos`, `exp`,
//! `log`, `pow` or `tan` and no optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each boid is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and a handful of `sqrt`-guarded normalizations, so `CPU` and `GPU` evaluate
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32`
//! fields, while the integer `cell` is compared exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`boids`](prism_render_architecture::particle::boids); the three-rule
//! flocking model is Reynolds' classic steering behavior re-derived at the
//! algorithm level, with no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::boids::{BoidsLimits, BoidsWeights};
use prism_render_architecture::particle::stages::CellCoord;
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// Fixed upper bound on the neighbors one boid may be fed per step.
///
/// The host gathers each boid's visible neighbors on the `CPU` (the golden
/// `visible_neighbors` uses the host-only `NeighborGrid`) and pads the
/// per-element list to this length; the kernel consumes only the first
/// `neighbor_count` entries, so the pad value is never read. `64` matches the
/// `Ultra` tier neighbor cap of the reference `boids_quality_budget`.
///
/// Provenance: this twin's host/device neighbor-list contract for
/// `separation_force`, `alignment_force` and `cohesion_force`.
pub const MAX_NEIGHBORS: u32 = 64;

/// The portable core-`WGSL` flocking kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden [`boids`](prism_render_architecture::particle::boids) function for
/// function; see the module documentation for the algorithm.
const BOIDS_WGSL: &str = r#"
// Flocking / Boids steering twin: one thread per boid reproduces the three
// Reynolds rules (separation, alignment, cohesion), optional goal seeking, the
// fixed-order weighted combine, the turn and force clamps, the semi-implicit
// integrate step, the full steer pipeline and the integer spatial-hash cell. It
// mirrors the CPU golden `particle::boids` function for function, uses only the
// portable core-WGSL subset (min/max/clamp/abs/floor/dot and + - * / plus one
// sqrt for the Euclidean normalizations) and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::boids; no third-party
// engine source or derived code.

// Squared-length epsilon guarding every normalization and the clamp degeneracy
// test, so no exact == / != is ever written on an f32. Matches the reference
// `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Fixed per-element neighbor-list stride; mirrors the Rust `MAX_NEIGHBORS`.
const MAX_NEIGHBORS: u32 = 64u;

struct Params {
    // Number of boids queried in this dispatch.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Index of this boid into the global positions / velocities arrays.
    boid_index: u32,
    // Non-zero when the optional goal direction is active.
    has_goal: u32,
    // Count of valid entries in this boid's neighbor-index slice.
    neighbor_count: u32,
    pad0: u32,
    // Goal point for goal_force; a pad lane follows.
    goal: vec3<f32>,
    pad1: f32,
    // Per-rule weights: (separation, alignment, cohesion, goal).
    weights: vec4<f32>,
    // Kinematic limits consumed by the steer pipeline.
    max_speed: f32,
    max_force: f32,
    max_turn: f32,
    // Cell size for the boid_cell lattice mapping.
    cell_size: f32,
    // Integration timestep, then the two dedicated-probe scalars.
    dt: f32,
    clamp_max_len: f32,
    turn_max: f32,
    pad2: f32,
    // Dedicated clamp_length probe input; a pad lane follows.
    clamp_input: vec3<f32>,
    pad3: f32,
    // Dedicated limit_turn probe steer vector; a pad lane follows.
    turn_steer: vec3<f32>,
    pad4: f32,
    // Dedicated limit_turn probe velocity vector; a pad lane follows.
    turn_velocity: vec3<f32>,
    pad5: f32,
}

struct Result {
    // separation_force direction; a pad lane follows.
    separation: vec3<f32>,
    pad0: f32,
    // alignment_force direction; a pad lane follows.
    alignment: vec3<f32>,
    pad1: f32,
    // cohesion_force direction; a pad lane follows.
    cohesion: vec3<f32>,
    pad2: f32,
    // goal_force direction (zero when has_goal is clear); a pad lane follows.
    goal_dir: vec3<f32>,
    pad3: f32,
    // combine_forces weighted sum; a pad lane follows.
    combined: vec3<f32>,
    pad4: f32,
    // limit_turn of combined against the boid velocity; a pad lane follows.
    turned: vec3<f32>,
    pad5: f32,
    // steer: clamp_length(turned, max_force); a pad lane follows.
    steer: vec3<f32>,
    pad6: f32,
    // integrate of velocity under steer over dt; a pad lane follows.
    integrated: vec3<f32>,
    pad7: f32,
    // Dedicated clamp_length probe output; a pad lane follows.
    clamp_out: vec3<f32>,
    pad8: f32,
    // Dedicated limit_turn probe output; a pad lane follows.
    turn_out: vec3<f32>,
    pad9: f32,
    // boid_cell integer lattice coordinate; a pad lane follows.
    cell: vec3<i32>,
    pad10: i32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> queries: array<Query>;
@group(0) @binding(4) var<storage, read> neighbors: array<u32>;
@group(0) @binding(5) var<storage, read_write> results: array<Result>;

// Unit vector along v, or zero when v is (numerically) the zero vector, so a
// normalization never yields NaN. Mirrors the reference `normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Clamps a vector's magnitude to max_len (no-op when already shorter); a near
// zero vector or a non-positive limit short-circuits. Mirrors `clamp_length`.
fn clamp_length(v: vec3<f32>, max_len: f32) -> vec3<f32> {
    let len_sq = dot(v, v);
    let max_sq = max_len * max_len;
    if (max_len > 0.0 && len_sq > max_sq && len_sq > EPS_LEN_SQ) {
        return v * (max_len / sqrt(len_sq));
    } else if (max_len > 0.0) {
        return v;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Splits steer into a heading-parallel part (free) and a perpendicular part
// (the turn), clamping the latter to max_turn. A boid with no heading is left
// unchanged. Mirrors the reference `limit_turn`.
fn limit_turn(steer: vec3<f32>, velocity: vec3<f32>, max_turn: f32) -> vec3<f32> {
    let heading = normalize_or_zero(velocity);
    if (dot(heading, heading) <= EPS_LEN_SQ) {
        return steer;
    }
    let along = heading * dot(steer, heading);
    let perpendicular = steer - along;
    return along + clamp_length(perpendicular, max_turn);
}

// Integer lattice cell containing p for a given cell size; a non-positive size
// collapses to the origin cell. Mirrors `NeighborGrid::cell_of` / `boid_cell`.
fn boid_cell(p: vec3<f32>, cell_size: f32) -> vec3<i32> {
    if (cell_size <= 0.0) {
        return vec3<i32>(0, 0, 0);
    }
    let inv = 1.0 / cell_size;
    return vec3<i32>(
        i32(floor(p.x * inv)),
        i32(floor(p.y * inv)),
        i32(floor(p.z * inv)),
    );
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let self_pos = positions[q.boid_index].xyz;
    let self_vel = velocities[q.boid_index].xyz;
    let base = idx * MAX_NEIGHBORS;

    // One pass over the neighbor list accumulates the separation push plus the
    // velocity and position sums the alignment / cohesion rules average.
    var push = vec3<f32>(0.0, 0.0, 0.0);
    var vel_sum = vec3<f32>(0.0, 0.0, 0.0);
    var pos_sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var i: u32 = 0u; i < q.neighbor_count; i = i + 1u) {
        let n = neighbors[base + i];
        let other_pos = positions[n].xyz;
        let other_vel = velocities[n].xyz;
        // separation: inverse-distance-weighted push away from the neighbor.
        let offset = self_pos - other_pos;
        let dist_sq = dot(offset, offset);
        if (dist_sq > EPS_LEN_SQ) {
            let dist = sqrt(dist_sq);
            push = push + offset * (1.0 / (dist * dist));
        }
        vel_sum = vel_sum + other_vel;
        pos_sum = pos_sum + other_pos;
    }
    let separation = normalize_or_zero(push);

    var alignment = vec3<f32>(0.0, 0.0, 0.0);
    var cohesion = vec3<f32>(0.0, 0.0, 0.0);
    if (q.neighbor_count > 0u) {
        let inv = 1.0 / f32(q.neighbor_count);
        alignment = normalize_or_zero(vel_sum * inv);
        let centroid = pos_sum * inv;
        cohesion = normalize_or_zero(centroid - self_pos);
    }

    var goal_dir = vec3<f32>(0.0, 0.0, 0.0);
    if (q.has_goal != 0u) {
        goal_dir = normalize_or_zero(q.goal - self_pos);
    }

    // combine_forces: fixed separation, alignment, cohesion, goal order.
    let w = q.weights;
    let combined =
        separation * w.x + alignment * w.y + cohesion * w.z + goal_dir * w.w;

    // steer pipeline: turn-limit then force-clamp.
    let turned = limit_turn(combined, self_vel, q.max_turn);
    let steer = clamp_length(turned, q.max_force);

    // integrate: semi-implicit velocity update, speed-clamped.
    let stepped = self_vel + steer * q.dt;
    let integrated = clamp_length(stepped, q.max_speed);

    // Dedicated clamp / turn probes on caller-supplied vectors.
    let clamp_out = clamp_length(q.clamp_input, q.clamp_max_len);
    let turn_out = limit_turn(q.turn_steer, q.turn_velocity, q.turn_max);

    let cell = boid_cell(self_pos, q.cell_size);

    var out: Result;
    out.separation = separation;
    out.pad0 = 0.0;
    out.alignment = alignment;
    out.pad1 = 0.0;
    out.cohesion = cohesion;
    out.pad2 = 0.0;
    out.goal_dir = goal_dir;
    out.pad3 = 0.0;
    out.combined = combined;
    out.pad4 = 0.0;
    out.turned = turned;
    out.pad5 = 0.0;
    out.steer = steer;
    out.pad6 = 0.0;
    out.integrated = integrated;
    out.pad7 = 0.0;
    out.clamp_out = clamp_out;
    out.pad8 = 0.0;
    out.turn_out = turn_out;
    out.pad9 = 0.0;
    out.cell = cell;
    out.pad10 = 0;
    results[idx] = out;
}
"#;

/// One per-boid flocking query.
///
/// Carries the boid's index into the shared `positions` / `velocities` arrays,
/// its host-gathered neighbor list (padded to [`MAX_NEIGHBORS`] on dispatch),
/// an optional `goal` point, the per-rule [`BoidsWeights`], the [`BoidsLimits`]
/// the steer pipeline clamps against, the `cell_size` for the lattice mapping,
/// the integration `dt`, and the two dedicated-probe inputs that exercise
/// `clamp_length` and `limit_turn` directly. Holds a `Vec`, so it derives
/// [`Clone`] (not [`Copy`]).
///
/// Provenance: this twin's host/device query contract for
/// [`boids`](prism_render_architecture::particle::boids).
#[derive(Clone, Debug, PartialEq)]
pub struct BoidsQuery {
    /// Index of this boid into the shared `positions` / `velocities` arrays.
    pub self_index: u32,
    /// Host-gathered neighbor indices; the first [`MAX_NEIGHBORS`] are used.
    pub neighbors: Vec<u32>,
    /// Optional goal point for the goal-seeking rule.
    pub goal: Option<Vec3>,
    /// Per-rule blend weights for the fixed-order combine.
    pub weights: BoidsWeights,
    /// Kinematic limits the steer pipeline clamps against.
    pub limits: BoidsLimits,
    /// Spatial-hash cell size for the `boid_cell` mapping.
    pub cell_size: f32,
    /// Integration timestep.
    pub dt: f32,
    /// Dedicated `clamp_length` probe input vector.
    pub clamp_input: Vec3,
    /// Dedicated `clamp_length` probe magnitude bound.
    pub clamp_max_len: f32,
    /// Dedicated `limit_turn` probe steer vector.
    pub turn_steer: Vec3,
    /// Dedicated `limit_turn` probe velocity (heading) vector.
    pub turn_velocity: Vec3,
    /// Dedicated `limit_turn` probe turn bound.
    pub turn_max: f32,
}

/// The resolved per-boid answer, mirroring every value the reference reports
/// across its twinned steering functions.
///
/// The `f32` vector fields match the reference to the tolerance documented on
/// this module; the integer [`CellCoord`] is exact. Holds `f32` fields, so it
/// derives [`PartialEq`] (not [`Eq`]).
///
/// Provenance: this twin's result record for
/// [`boids`](prism_render_architecture::particle::boids).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoidsResult {
    /// `separation_force` normalized direction.
    pub separation: Vec3,
    /// `alignment_force` normalized direction.
    pub alignment: Vec3,
    /// `cohesion_force` normalized direction.
    pub cohesion: Vec3,
    /// `goal_force` direction ([`Vec3::ZERO`] when no goal is active).
    pub goal_dir: Vec3,
    /// `combine_forces` weighted, fixed-order sum.
    pub combined: Vec3,
    /// `limit_turn` of `combined` against the boid velocity.
    pub turned: Vec3,
    /// `steer`: `clamp_length(turned, max_force)`, the golden steer result.
    pub steer: Vec3,
    /// `integrate` of the velocity under `steer` over `dt`.
    pub integrated: Vec3,
    /// Dedicated `clamp_length` probe output.
    pub clamp_length: Vec3,
    /// Dedicated `limit_turn` probe output.
    pub limit_turn: Vec3,
    /// `boid_cell` integer lattice coordinate.
    pub cell: CellCoord,
}

/// `repr(C)` `std430` image of one packed query: eight `16`-byte slots holding
/// the index / flag / count header, the goal, the weights, the limits and cell
/// size, the timestep and probe scalars, and the three probe vectors — `128`
/// bytes, each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Index of this boid into the global arrays.
    boid_index: u32,
    /// Non-zero when the goal direction is active.
    has_goal: u32,
    /// Count of valid neighbor-list entries.
    neighbor_count: u32,
    /// Padding word.
    pad0: u32,
    /// Goal point.
    goal: [f32; 3],
    /// Padding lane after the goal.
    pad1: f32,
    /// Per-rule weights `(separation, alignment, cohesion, goal)`.
    weights: [f32; 4],
    /// Maximum integrated speed.
    max_speed: f32,
    /// Maximum steering-force magnitude.
    max_force: f32,
    /// Maximum perpendicular turn magnitude.
    max_turn: f32,
    /// Spatial-hash cell size.
    cell_size: f32,
    /// Integration timestep.
    dt: f32,
    /// Dedicated clamp probe magnitude bound.
    clamp_max_len: f32,
    /// Dedicated turn probe bound.
    turn_max: f32,
    /// Padding lane.
    pad2: f32,
    /// Dedicated clamp probe input vector.
    clamp_input: [f32; 3],
    /// Padding lane after the clamp input.
    pad3: f32,
    /// Dedicated turn probe steer vector.
    turn_steer: [f32; 3],
    /// Padding lane after the turn steer.
    pad4: f32,
    /// Dedicated turn probe velocity vector.
    turn_velocity: [f32; 3],
    /// Padding lane after the turn velocity.
    pad5: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image, resolving the optional goal to a
    /// flag plus vector and clamping the neighbor count to [`MAX_NEIGHBORS`].
    fn new(query: &BoidsQuery) -> GpuQuery {
        let (has_goal, goal) = match query.goal {
            Some(target) => (1, [target.x, target.y, target.z]),
            None => (0, [0.0, 0.0, 0.0]),
        };
        let neighbor_count = query.neighbors.len().min(MAX_NEIGHBORS as usize) as u32;
        GpuQuery {
            boid_index: query.self_index,
            has_goal,
            neighbor_count,
            pad0: 0,
            goal,
            pad1: 0.0,
            weights: [
                query.weights.separation,
                query.weights.alignment,
                query.weights.cohesion,
                query.weights.goal,
            ],
            max_speed: query.limits.max_speed,
            max_force: query.limits.max_force,
            max_turn: query.limits.max_turn,
            cell_size: query.cell_size,
            dt: query.dt,
            clamp_max_len: query.clamp_max_len,
            turn_max: query.turn_max,
            pad2: 0.0,
            clamp_input: [
                query.clamp_input.x,
                query.clamp_input.y,
                query.clamp_input.z,
            ],
            pad3: 0.0,
            turn_steer: [query.turn_steer.x, query.turn_steer.y, query.turn_steer.z],
            pad4: 0.0,
            turn_velocity: [
                query.turn_velocity.x,
                query.turn_velocity.y,
                query.turn_velocity.z,
            ],
            pad5: 0.0,
        }
    }
}

/// `repr(C)` `std430` image of one result: ten `f32` `vec3` slots for the
/// twinned directions / forces / velocities, then one `i32` `vec3` slot for the
/// cell — `176` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Separation direction.
    separation: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// Alignment direction.
    alignment: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// Cohesion direction.
    cohesion: [f32; 3],
    /// Padding lane.
    pad2: f32,
    /// Goal direction.
    goal_dir: [f32; 3],
    /// Padding lane.
    pad3: f32,
    /// Combined weighted force.
    combined: [f32; 3],
    /// Padding lane.
    pad4: f32,
    /// Turn-limited force.
    turned: [f32; 3],
    /// Padding lane.
    pad5: f32,
    /// Force-clamped steer acceleration.
    steer: [f32; 3],
    /// Padding lane.
    pad6: f32,
    /// Speed-clamped integrated velocity.
    integrated: [f32; 3],
    /// Padding lane.
    pad7: f32,
    /// Dedicated clamp probe output.
    clamp_out: [f32; 3],
    /// Padding lane.
    pad8: f32,
    /// Dedicated turn probe output.
    turn_out: [f32; 3],
    /// Padding lane.
    pad9: f32,
    /// Integer lattice cell.
    cell: [i32; 3],
    /// Padding lane.
    pad10: i32,
}

/// Uniform parameters for one dispatch: the boid count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of boids in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable flocking compute pipeline.
pub struct GpuBoids {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBoids {
    /// Compiles the flocking kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBoids {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_boids"),
            source: ShaderSource::Wgsl(BOIDS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_boids_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_boids_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_boids_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBoids {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every `query` on-device against the shared `positions` and
    /// `velocities` arrays, returning one [`BoidsResult`] per input, in order.
    ///
    /// Each result equals the reference answers (`separation_force`,
    /// `alignment_force`, `cohesion_force`, `goal_force`, `combine_forces`,
    /// `limit_turn`, `clamp_length`, `steer`, `integrate` and `boid_cell`) to
    /// within the tolerance documented on this module. An empty `queries` input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        velocities: &[Vec3],
        queries: &[BoidsQuery],
    ) -> Vec<BoidsResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed_pos: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let packed_vel: Vec<[f32; 4]> = velocities.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();
        let packed_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();

        let stride = MAX_NEIGHBORS as usize;
        let mut neighbor_flat = vec![0u32; count * stride];
        for (qi, query) in queries.iter().enumerate() {
            let base = qi * stride;
            let n = query.neighbors.len().min(stride);
            neighbor_flat[base..base + n].copy_from_slice(&query.neighbors[..n]);
        }

        let pos_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_boids_positions"),
            contents: bytemuck::cast_slice(&packed_pos),
            usage: BufferUsages::STORAGE,
        });
        let vel_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_boids_velocities"),
            contents: bytemuck::cast_slice(&packed_vel),
            usage: BufferUsages::STORAGE,
        });
        let query_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_boids_queries"),
            contents: bytemuck::cast_slice(&packed_queries),
            usage: BufferUsages::STORAGE,
        });
        let neighbor_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_boids_neighbors"),
            contents: bytemuck::cast_slice(&neighbor_flat),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_boids_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_boids_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_boids_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: pos_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: vel_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: query_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: neighbor_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_boids_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_boids_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_boids_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per boid, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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

/// Decodes one packed [`GpuResult`] into the public [`BoidsResult`].
fn decode_result(raw: &GpuResult) -> BoidsResult {
    BoidsResult {
        separation: vec3_of(raw.separation),
        alignment: vec3_of(raw.alignment),
        cohesion: vec3_of(raw.cohesion),
        goal_dir: vec3_of(raw.goal_dir),
        combined: vec3_of(raw.combined),
        turned: vec3_of(raw.turned),
        steer: vec3_of(raw.steer),
        integrated: vec3_of(raw.integrated),
        clamp_length: vec3_of(raw.clamp_out),
        limit_turn: vec3_of(raw.turn_out),
        cell: CellCoord::new(raw.cell[0], raw.cell[1], raw.cell[2]),
    }
}

/// Rebuilds a [`Vec3`] from a packed three-lane array.
fn vec3_of(raw: [f32; 3]) -> Vec3 {
    Vec3::new(raw[0], raw[1], raw[2])
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
