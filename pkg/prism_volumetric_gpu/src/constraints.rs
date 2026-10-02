//! `wgpu` compute twin of the `XPBD` constraint-solve numeric contract
//! ([`constraints`](prism_render_architecture::particle::constraints),
//! particle design §10 "约束求解/断裂").
//!
//! The `CPU` golden
//! [`constraints`](prism_render_architecture::particle::constraints) owns the
//! closed-form scalar and vector math an `XPBD` solver needs per constraint:
//! the compliance regularizer
//! ([`effective_compliance`](prism_render_architecture::particle::constraints::effective_compliance),
//! [`Compliance::from_stiffness`](prism_render_architecture::particle::constraints::Compliance::from_stiffness),
//! [`Compliance::scaled_for_substep`](prism_render_architecture::particle::constraints::Compliance::scaled_for_substep),
//! [`Compliance::is_rigid`](prism_render_architecture::particle::constraints::Compliance::is_rigid)),
//! the substep schedule
//! ([`SubstepSchedule::substep_dt`](prism_render_architecture::particle::constraints::SubstepSchedule::substep_dt),
//! [`SubstepSchedule::effective_compliance`](prism_render_architecture::particle::constraints::SubstepSchedule::effective_compliance)),
//! the distance-constraint projection
//! ([`project_distance`](prism_render_architecture::particle::constraints::project_distance)),
//! the tearing test
//! ([`constraint_strain`](prism_render_architecture::particle::constraints::constraint_strain),
//! [`should_tear`](prism_render_architecture::particle::constraints::should_tear)),
//! the fracture rigid proxy
//! ([`rigid_from_particles`](prism_render_architecture::particle::constraints::rigid_from_particles)),
//! and the solver-tier selection matrix
//! ([`SolverSelection::select`](prism_render_architecture::particle::constraints::SolverSelection::select),
//! [`ConvergenceContract::for_tier`](prism_render_architecture::particle::constraints::ConvergenceContract::for_tier)).
//! [`GpuConstraints`] is the on-device twin: one thread solves one query, so a
//! passing real-device parity test is direct evidence the ported kernel
//! evaluates the same closed form the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! Every numeric primitive is reproduced through a tagged
//! [`ConstraintsQuery`]: one variant per reference function. The graph-coloring
//! and slice-scanning helpers
//! ([`color_distance_constraints`](prism_render_architecture::particle::constraints::color_distance_constraints),
//! [`batch_constraints`](prism_render_architecture::particle::constraints::batch_constraints),
//! [`tear_scan`](prism_render_architecture::particle::constraints::tear_scan))
//! and the chunk-partition driver
//! ([`plan_fracture`](prism_render_architecture::particle::constraints::plan_fracture))
//! are deliberately *not* twinned: they are host-side batch-index and
//! allocation logic, not per-element kernel math, so there is nothing on device
//! to compare them against. The per-chunk rigid proxy they call is twinned
//! directly via [`ConstraintsQuery::RigidFromParticles`], which the host feeds a
//! fixed-length particle budget (at most [`MAX_RIGID_PARTICLES`]).
//!
//! # Correctness model
//!
//! Every term is a rational function of its inputs with at most one `sqrt` (the
//! projection normal, the rigid proxy's distance), so `CPU` and `GPU` evaluate
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every
//! continuous lane; the boolean rigidity / tear flags and the discrete solver
//! tier are compared exactly.
//!
//! # Degenerate inputs
//!
//! Every division is guarded exactly as the reference guards it: a non-positive
//! substep squared, inverse-mass sum, or total mass falls back to the rigid /
//! zero result rather than dividing, a coincident distance pair returns a zero
//! correction, and a near-zero rest length floors its denominator at `GUARD_EPS`
//! so a pinned rope still reports a finite strain. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `min`, `max`, `dot`, `sqrt` and `+ - * /` — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no inverse trigonometry and no optional device feature,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The rigid proxy's one
//! loop is bounded by [`MAX_RIGID_PARTICLES`], so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::constraints`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::constraints::SolverTier;
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
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::constraints`；无第三方引擎源码或衍生代码。
const WORKGROUP_SIZE: u32 = 64;

/// Upper bound on the particle count one [`ConstraintsQuery::RigidFromParticles`]
/// query carries. The host budgets each fracture chunk into at most this many
/// particles before dispatch, mirroring how
/// [`plan_fracture`](prism_render_architecture::particle::constraints::plan_fracture)
/// feeds contiguous spans to
/// [`rigid_from_particles`](prism_render_architecture::particle::constraints::rigid_from_particles).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::constraints`；无第三方引擎源码或衍生代码。
pub const MAX_RIGID_PARTICLES: usize = 8;

/// The portable core-`WGSL` constraint-solve kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` branches on a
/// per-query op code into the `CPU` golden
/// [`constraints`](prism_render_architecture::particle::constraints) terms; see
/// the module documentation for the algorithm.
const CONSTRAINTS_WGSL: &str = r#"
// XPBD constraint-solve twin: one thread per query reproduces one reference
// numeric function selected by `op`. It mirrors the CPU golden
// `particle::constraints` term for term, uses only the portable core-WGSL
// subset (abs/clamp/min/max/dot/sqrt and + - * /), needs no transcendental call
// and takes no optional feature, so it runs unmodified on Metal, Vulkan and
// DX12. The only loop is bounded by MAX_RIGID_PARTICLES, so the kernel provably
// terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::constraints；
// 无第三方引擎源码或衍生代码。

// Scalar division guard, matching the reference EPS (metres-per-newton, substep
// dt, inverse-mass sum and rest-length floor).
const GUARD_EPS: f32 = 1.0e-9;
// Squared-length floor for a degenerate separation, matching EPS_LEN_SQ.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Finite mass of a (numerically) immovable particle, matching STATIC_MASS.
const STATIC_MASS: f32 = 1.0e9;
// Fixed particle budget of one rigid-proxy query, matching MAX_RIGID_PARTICLES.
const MAX_RIGID_PARTICLES: u32 = 8u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Endpoints of the distance projection (op 6); each vec3 carries a trailing
    // pad lane to stay 16-byte aligned on device.
    pos_a: vec3<f32>,
    pad_a: f32,
    pos_b: vec3<f32>,
    pad_b: f32,
    // Fixed-length rigid-proxy particle budget (op 9).
    rigid_positions: array<vec3<f32>, 8>,
    rigid_inv_masses: array<f32, 8>,
    // Compliance / schedule scalars.
    alpha: f32,
    substep_dt: f32,
    stiffness: f32,
    dt: f32,
    // Distance-projection scalars.
    inv_mass_a: f32,
    inv_mass_b: f32,
    rest: f32,
    alpha_tilde: f32,
    lambda: f32,
    current_length: f32,
    // Tearing scalars.
    strain: f32,
    max_strain: f32,
    // Solver-selection scalars.
    stiff_ratio_threshold: f32,
    contact_density_threshold: f32,
    stiffness_ratio: f32,
    contact_density: f32,
    // Integer controls.
    substeps: u32,
    force_high_stability: u32,
    tier: u32,
    rigid_count: u32,
    // Op classification code (0..=11) plus pad words to fill the slot.
    op: u32,
    pad_q0: f32,
    pad_q1: f32,
    pad_q2: f32,
}

struct Result {
    // Up to seven scalar outputs; the interpretation depends on the query op.
    // For project_distance (op 6): delta_a.xyz, delta_b.xyz, delta_lambda. For
    // rigid (op 9): inv_mass, com.xyz, bounding_radius. For convergence (op 11):
    // unconditionally_stable, parallelizable, cost_multiplier. Every
    // single-valued op leaves its answer in `a`.
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// alpha / substep_dt^2 with a non-positive squared dt guarded to 0; mirrors
// `Compliance::scaled_for_substep` (raw alpha, no clamp).
fn scaled_for_substep(alpha: f32, substep_dt: f32) -> f32 {
    let dt2 = substep_dt * substep_dt;
    if (dt2 > GUARD_EPS) {
        return alpha / dt2;
    }
    return 0.0;
}

// Clamps a raw compliance to be non-negative; mirrors `Compliance::new`.
fn compliance_new(alpha: f32) -> f32 {
    if (alpha > 0.0) {
        return alpha;
    }
    return 0.0;
}

// Free-function effective compliance; mirrors `effective_compliance`
// (clamps alpha via `Compliance::new` first).
fn effective_compliance(alpha: f32, substep_dt: f32) -> f32 {
    return scaled_for_substep(compliance_new(alpha), substep_dt);
}

// Inverse stiffness, infinite stiffness folding to rigid 0; mirrors
// `Compliance::from_stiffness`.
fn from_stiffness(stiffness: f32) -> f32 {
    if (stiffness > GUARD_EPS) {
        return 1.0 / stiffness;
    }
    return 0.0;
}

// Per-substep timestep dt / substeps with zero substeps guarded; mirrors
// `SubstepSchedule::substep_dt`.
fn substep_dt_of(dt: f32, substeps: u32) -> f32 {
    if (substeps == 0u) {
        return 0.0;
    }
    return dt / f32(substeps);
}

// Projects one XPBD distance constraint; mirrors `project_distance`. Writes the
// position deltas and multiplier increment into `out`.
fn project_distance(q: Query, out: ptr<function, Result>) {
    let delta = q.pos_a - q.pos_b;
    let len_sq = dot(delta, delta);
    if (len_sq <= EPS_LEN_SQ) {
        return;
    }
    let len = sqrt(len_sq);
    let normal = delta * (1.0 / len);
    let c = len - q.rest;
    let denom = q.inv_mass_a + q.inv_mass_b + q.alpha_tilde;
    if (denom <= GUARD_EPS) {
        return;
    }
    let delta_lambda = (-c - q.alpha_tilde * q.lambda) / denom;
    let da = normal * (q.inv_mass_a * delta_lambda);
    let db = normal * (-q.inv_mass_b * delta_lambda);
    (*out).a = da.x;
    (*out).b = da.y;
    (*out).c = da.z;
    (*out).d = db.x;
    (*out).e = db.y;
    (*out).f = db.z;
    (*out).g = delta_lambda;
}

// Relative stretch with a near-zero rest length floored; mirrors
// `constraint_strain`.
fn constraint_strain(rest: f32, current_length: f32) -> f32 {
    var denom = rest;
    if (!(rest > GUARD_EPS)) {
        denom = GUARD_EPS;
    }
    return (current_length - rest) / denom;
}

// Rigid-body proxy of a shard; mirrors `rigid_from_particles`. Writes inv_mass,
// center of mass and bounding radius into `out`.
fn rigid_from_particles(q: Query, out: ptr<function, Result>) {
    var total_mass = 0.0;
    var weighted = vec3<f32>(0.0, 0.0, 0.0);
    var i = 0u;
    loop {
        if (i >= q.rigid_count || i >= MAX_RIGID_PARTICLES) {
            break;
        }
        let inv_mass = q.rigid_inv_masses[i];
        var mass = STATIC_MASS;
        if (inv_mass > GUARD_EPS) {
            mass = 1.0 / inv_mass;
        }
        total_mass = total_mass + mass;
        weighted = weighted + q.rigid_positions[i] * mass;
        i = i + 1u;
    }
    var com = vec3<f32>(0.0, 0.0, 0.0);
    if (total_mass > GUARD_EPS) {
        com = weighted * (1.0 / total_mass);
    }
    var radius = 0.0;
    i = 0u;
    loop {
        if (i >= q.rigid_count || i >= MAX_RIGID_PARTICLES) {
            break;
        }
        let diff = q.rigid_positions[i] - com;
        let dist = sqrt(dot(diff, diff));
        if (dist > radius) {
            radius = dist;
        }
        i = i + 1u;
    }
    var inv_mass_out = 0.0;
    if (total_mass > GUARD_EPS) {
        inv_mass_out = 1.0 / total_mass;
    }
    (*out).a = inv_mass_out;
    (*out).b = com.x;
    (*out).c = com.y;
    (*out).d = com.z;
    (*out).e = radius;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.a = 0.0;
    out.b = 0.0;
    out.c = 0.0;
    out.d = 0.0;
    out.e = 0.0;
    out.f = 0.0;
    out.g = 0.0;
    out.pad0 = 0.0;

    if (q.op == 0u) {
        out.a = effective_compliance(q.alpha, q.substep_dt);
    } else if (q.op == 1u) {
        out.a = from_stiffness(q.stiffness);
    } else if (q.op == 2u) {
        out.a = scaled_for_substep(q.alpha, q.substep_dt);
    } else if (q.op == 3u) {
        // is_rigid: alpha <= GUARD_EPS, encoded as 1.0 / 0.0.
        if (q.alpha <= GUARD_EPS) {
            out.a = 1.0;
        } else {
            out.a = 0.0;
        }
    } else if (q.op == 4u) {
        out.a = substep_dt_of(q.dt, q.substeps);
    } else if (q.op == 5u) {
        // Schedule effective compliance: scaled_for_substep with the raw alpha
        // and the schedule's substep dt (no Compliance::new clamp).
        out.a = scaled_for_substep(q.alpha, substep_dt_of(q.dt, q.substeps));
    } else if (q.op == 6u) {
        project_distance(q, &out);
    } else if (q.op == 7u) {
        out.a = constraint_strain(q.rest, q.current_length);
    } else if (q.op == 8u) {
        // should_tear: strain > max_strain, encoded as 1.0 / 0.0.
        if (q.strain > q.max_strain) {
            out.a = 1.0;
        } else {
            out.a = 0.0;
        }
    } else if (q.op == 9u) {
        rigid_from_particles(q, &out);
    } else if (q.op == 10u) {
        // SolverSelection::select: escalate to VBD (1) on force, stiffness or
        // density; else XPBD (0).
        let stiff = q.stiffness_ratio > q.stiff_ratio_threshold;
        let dense = q.contact_density > q.contact_density_threshold;
        if (q.force_high_stability != 0u || stiff || dense) {
            out.a = 1.0;
        } else {
            out.a = 0.0;
        }
    } else {
        // ConvergenceContract::for_tier: XPBD (tier 0) vs VBD (tier 1).
        if (q.tier == 0u) {
            out.a = 0.0;
            out.b = 1.0;
            out.c = 1.0;
        } else {
            out.a = 1.0;
            out.b = 1.0;
            out.c = 3.0;
        }
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CONSTRAINTS_WGSL`].
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
/// Every `vec3` lane carries a trailing pad word so each stays `16`-byte aligned
/// on device; the trailing pad words fill the final `16`-byte slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Distance-projection endpoint `a`.
    pos_a: [f32; 3],
    /// Pad lane after `pos_a`.
    pad_a: f32,
    /// Distance-projection endpoint `b`.
    pos_b: [f32; 3],
    /// Pad lane after `pos_b`.
    pad_b: f32,
    /// Rigid-proxy particle positions, each padded to a `16`-byte lane.
    rigid_positions: [[f32; 4]; MAX_RIGID_PARTICLES],
    /// Rigid-proxy particle inverse masses.
    rigid_inv_masses: [f32; MAX_RIGID_PARTICLES],
    /// Raw compliance `α`.
    alpha: f32,
    /// Per-substep timestep for the standalone scale ops.
    substep_dt: f32,
    /// Stiffness for `from_stiffness`.
    stiffness: f32,
    /// Full-frame timestep for the schedule ops.
    dt: f32,
    /// Inverse mass of endpoint `a`.
    inv_mass_a: f32,
    /// Inverse mass of endpoint `b`.
    inv_mass_b: f32,
    /// Rest length / rest value.
    rest: f32,
    /// Substep-scaled compliance `α̃` for the projection.
    alpha_tilde: f32,
    /// Running Lagrange multiplier `λ`.
    lambda: f32,
    /// Current separation for the strain op.
    current_length: f32,
    /// Relative stretch for the tear test.
    strain: f32,
    /// Tear strain threshold.
    max_strain: f32,
    /// Stiffness-ratio escalation threshold.
    stiff_ratio_threshold: f32,
    /// Contact-density escalation threshold.
    contact_density_threshold: f32,
    /// Island stiffness ratio.
    stiffness_ratio: f32,
    /// Island contact density.
    contact_density: f32,
    /// Number of equal substeps per frame.
    substeps: u32,
    /// Caller demand for high stability (`0`/`1`).
    force_high_stability: u32,
    /// Solver tier code for the convergence lookup (`0` `XPBD`, `1` `VBD`).
    tier: u32,
    /// Live particle count of the rigid-proxy budget.
    rigid_count: u32,
    /// Op classification code (`0..=11`).
    op: u32,
    /// Padding lane.
    pad_q0: f32,
    /// Padding lane.
    pad_q1: f32,
    /// Padding lane.
    pad_q2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: up to seven scalar outputs plus one pad lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// First output lane (the single answer for scalar ops).
    a: f32,
    /// Second output lane.
    b: f32,
    /// Third output lane.
    c: f32,
    /// Fourth output lane.
    d: f32,
    /// Fifth output lane.
    e: f32,
    /// Sixth output lane.
    f: f32,
    /// Seventh output lane.
    g: f32,
    /// Padding lane.
    pad0: f32,
}

/// One tagged query selecting which reference term the kernel evaluates.
///
/// There is one variant per twinned `CPU` golden function; a field a variant
/// does not name is ignored. The discriminant order matches the `u32` op codes
/// the kernel branches on.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::constraints`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstraintsQuery {
    /// The free-function substep-scaled compliance, matching
    /// [`effective_compliance`](prism_render_architecture::particle::constraints::effective_compliance).
    EffectiveCompliance {
        /// Raw compliance `α` (clamped non-negative before scaling).
        alpha: f32,
        /// Per-substep timestep.
        substep_dt: f32,
    },
    /// The stiffness-to-compliance inverse, matching
    /// [`Compliance::from_stiffness`](prism_render_architecture::particle::constraints::Compliance::from_stiffness).
    ComplianceFromStiffness {
        /// Material stiffness (non-positive folds to rigid `0`).
        stiffness: f32,
    },
    /// The raw substep-scaled compliance, matching
    /// [`Compliance::scaled_for_substep`](prism_render_architecture::particle::constraints::Compliance::scaled_for_substep).
    ComplianceScaledForSubstep {
        /// Raw compliance `α` (used directly, no clamp).
        alpha: f32,
        /// Per-substep timestep.
        substep_dt: f32,
    },
    /// The numeric rigidity test, matching
    /// [`Compliance::is_rigid`](prism_render_architecture::particle::constraints::Compliance::is_rigid).
    ComplianceIsRigid {
        /// Raw compliance `α`.
        alpha: f32,
    },
    /// The per-substep timestep, matching
    /// [`SubstepSchedule::substep_dt`](prism_render_architecture::particle::constraints::SubstepSchedule::substep_dt).
    SubstepDt {
        /// Full-frame timestep.
        dt: f32,
        /// Number of equal substeps.
        substeps: u32,
    },
    /// The schedule's effective compliance, matching
    /// [`SubstepSchedule::effective_compliance`](prism_render_architecture::particle::constraints::SubstepSchedule::effective_compliance).
    ScheduleEffectiveCompliance {
        /// Full-frame timestep.
        dt: f32,
        /// Number of equal substeps.
        substeps: u32,
        /// Raw compliance `α` (used directly, no clamp).
        alpha: f32,
    },
    /// The `XPBD` distance-constraint projection, matching
    /// [`project_distance`](prism_render_architecture::particle::constraints::project_distance).
    ProjectDistance {
        /// Position of endpoint `a`.
        pos_a: [f32; 3],
        /// Position of endpoint `b`.
        pos_b: [f32; 3],
        /// Inverse mass of endpoint `a`.
        inv_mass_a: f32,
        /// Inverse mass of endpoint `b`.
        inv_mass_b: f32,
        /// Rest length.
        rest: f32,
        /// Substep-scaled compliance `α̃`.
        alpha_tilde: f32,
        /// Running Lagrange multiplier `λ`.
        lambda: f32,
    },
    /// The relative stretch, matching
    /// [`constraint_strain`](prism_render_architecture::particle::constraints::constraint_strain).
    ConstraintStrain {
        /// Rest length.
        rest: f32,
        /// Current separation.
        current_length: f32,
    },
    /// The tear test, matching
    /// [`should_tear`](prism_render_architecture::particle::constraints::should_tear).
    ShouldTear {
        /// Relative stretch.
        strain: f32,
        /// Tear threshold.
        max_strain: f32,
    },
    /// The fracture rigid proxy, matching
    /// [`rigid_from_particles`](prism_render_architecture::particle::constraints::rigid_from_particles).
    RigidFromParticles {
        /// Particle positions; only the first `count` are live.
        positions: [[f32; 3]; MAX_RIGID_PARTICLES],
        /// Particle inverse masses; only the first `count` are live.
        inv_masses: [f32; MAX_RIGID_PARTICLES],
        /// Number of live particles (at most [`MAX_RIGID_PARTICLES`]).
        count: u32,
    },
    /// The solver-tier selection, matching
    /// [`SolverSelection::select`](prism_render_architecture::particle::constraints::SolverSelection::select).
    SolverSelect {
        /// Stiffness-ratio escalation threshold.
        stiff_ratio_threshold: f32,
        /// Contact-density escalation threshold.
        contact_density_threshold: f32,
        /// Island stiffness ratio.
        stiffness_ratio: f32,
        /// Island contact density.
        contact_density: f32,
        /// Caller demand for high stability.
        force_high_stability: bool,
    },
    /// The convergence contract of a tier, matching
    /// [`ConvergenceContract::for_tier`](prism_render_architecture::particle::constraints::ConvergenceContract::for_tier).
    ConvergenceForTier {
        /// The solver tier to describe.
        tier: SolverTier,
    },
}

impl ConstraintsQuery {
    /// Returns the `u32` op code the kernel branches on for this variant.
    #[must_use]
    const fn code(&self) -> u32 {
        match self {
            ConstraintsQuery::EffectiveCompliance { .. } => 0,
            ConstraintsQuery::ComplianceFromStiffness { .. } => 1,
            ConstraintsQuery::ComplianceScaledForSubstep { .. } => 2,
            ConstraintsQuery::ComplianceIsRigid { .. } => 3,
            ConstraintsQuery::SubstepDt { .. } => 4,
            ConstraintsQuery::ScheduleEffectiveCompliance { .. } => 5,
            ConstraintsQuery::ProjectDistance { .. } => 6,
            ConstraintsQuery::ConstraintStrain { .. } => 7,
            ConstraintsQuery::ShouldTear { .. } => 8,
            ConstraintsQuery::RigidFromParticles { .. } => 9,
            ConstraintsQuery::SolverSelect { .. } => 10,
            ConstraintsQuery::ConvergenceForTier { .. } => 11,
        }
    }
}

/// One resolved answer, tagged to match the query variant that produced it.
///
/// Each variant carries exactly the term(s) the matching [`ConstraintsQuery`]
/// variant selects.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::constraints`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstraintsResult {
    /// The free-function substep-scaled compliance `α̃`.
    EffectiveCompliance {
        /// `α̃ = clamp(α, 0, ∞) / substep_dt²`.
        alpha_tilde: f32,
    },
    /// The stiffness-to-compliance inverse `α`.
    ComplianceFromStiffness {
        /// `α = 1 / stiffness` (or `0` when rigid).
        alpha: f32,
    },
    /// The raw substep-scaled compliance `α̃`.
    ComplianceScaledForSubstep {
        /// `α̃ = α / substep_dt²`.
        alpha_tilde: f32,
    },
    /// The numeric rigidity test.
    ComplianceIsRigid {
        /// Whether the constraint is (numerically) rigid.
        is_rigid: bool,
    },
    /// The per-substep timestep.
    SubstepDt {
        /// `dt / substeps` (or `0` when `substeps == 0`).
        substep_dt: f32,
    },
    /// The schedule's effective compliance `α̃`.
    ScheduleEffectiveCompliance {
        /// `α̃ = α / substep_dt²`.
        alpha_tilde: f32,
    },
    /// The `XPBD` distance-projection correction.
    ProjectDistance {
        /// Position delta for endpoint `a`.
        delta_a: [f32; 3],
        /// Position delta for endpoint `b`.
        delta_b: [f32; 3],
        /// Increment to the accumulated Lagrange multiplier.
        delta_lambda: f32,
    },
    /// The relative stretch.
    ConstraintStrain {
        /// `(current − rest) / max(rest, EPS)`.
        strain: f32,
    },
    /// The tear test.
    ShouldTear {
        /// Whether the strain exceeds the threshold.
        tears: bool,
    },
    /// The fracture rigid proxy.
    RigidFromParticles {
        /// Aggregate inverse mass.
        inv_mass: f32,
        /// Mass-weighted center of mass.
        center_of_mass: [f32; 3],
        /// Bounding-sphere radius about the center of mass.
        bounding_radius: f32,
    },
    /// The selected solver tier.
    SolverSelect {
        /// The chosen tier.
        tier: SolverTier,
    },
    /// The convergence contract of a tier.
    ConvergenceForTier {
        /// Whether the tier is stable for any timestep / stiffness.
        unconditionally_stable: bool,
        /// Whether iterations parallelize across colors / vertex blocks.
        parallelizable: bool,
        /// Relative per-iteration cost against the `XPBD` baseline of `1`.
        cost_multiplier: f32,
    },
}

/// Encodes one [`ConstraintsQuery`] into its `std430` [`GpuQuery`] slot, zeroing
/// the lanes the chosen op does not read.
fn encode_query(query: &ConstraintsQuery) -> GpuQuery {
    let mut gpu = GpuQuery {
        pos_a: [0.0; 3],
        pad_a: 0.0,
        pos_b: [0.0; 3],
        pad_b: 0.0,
        rigid_positions: [[0.0; 4]; MAX_RIGID_PARTICLES],
        rigid_inv_masses: [0.0; MAX_RIGID_PARTICLES],
        alpha: 0.0,
        substep_dt: 0.0,
        stiffness: 0.0,
        dt: 0.0,
        inv_mass_a: 0.0,
        inv_mass_b: 0.0,
        rest: 0.0,
        alpha_tilde: 0.0,
        lambda: 0.0,
        current_length: 0.0,
        strain: 0.0,
        max_strain: 0.0,
        stiff_ratio_threshold: 0.0,
        contact_density_threshold: 0.0,
        stiffness_ratio: 0.0,
        contact_density: 0.0,
        substeps: 0,
        force_high_stability: 0,
        tier: 0,
        rigid_count: 0,
        op: query.code(),
        pad_q0: 0.0,
        pad_q1: 0.0,
        pad_q2: 0.0,
    };
    match *query {
        ConstraintsQuery::EffectiveCompliance { alpha, substep_dt }
        | ConstraintsQuery::ComplianceScaledForSubstep { alpha, substep_dt } => {
            gpu.alpha = alpha;
            gpu.substep_dt = substep_dt;
        }
        ConstraintsQuery::ComplianceFromStiffness { stiffness } => {
            gpu.stiffness = stiffness;
        }
        ConstraintsQuery::ComplianceIsRigid { alpha } => {
            gpu.alpha = alpha;
        }
        ConstraintsQuery::SubstepDt { dt, substeps } => {
            gpu.dt = dt;
            gpu.substeps = substeps;
        }
        ConstraintsQuery::ScheduleEffectiveCompliance {
            dt,
            substeps,
            alpha,
        } => {
            gpu.dt = dt;
            gpu.substeps = substeps;
            gpu.alpha = alpha;
        }
        ConstraintsQuery::ProjectDistance {
            pos_a,
            pos_b,
            inv_mass_a,
            inv_mass_b,
            rest,
            alpha_tilde,
            lambda,
        } => {
            gpu.pos_a = pos_a;
            gpu.pos_b = pos_b;
            gpu.inv_mass_a = inv_mass_a;
            gpu.inv_mass_b = inv_mass_b;
            gpu.rest = rest;
            gpu.alpha_tilde = alpha_tilde;
            gpu.lambda = lambda;
        }
        ConstraintsQuery::ConstraintStrain {
            rest,
            current_length,
        } => {
            gpu.rest = rest;
            gpu.current_length = current_length;
        }
        ConstraintsQuery::ShouldTear { strain, max_strain } => {
            gpu.strain = strain;
            gpu.max_strain = max_strain;
        }
        ConstraintsQuery::RigidFromParticles {
            positions,
            inv_masses,
            count,
        } => {
            for (slot, position) in gpu.rigid_positions.iter_mut().zip(positions.iter()) {
                slot[0] = position[0];
                slot[1] = position[1];
                slot[2] = position[2];
            }
            gpu.rigid_inv_masses = inv_masses;
            gpu.rigid_count = count;
        }
        ConstraintsQuery::SolverSelect {
            stiff_ratio_threshold,
            contact_density_threshold,
            stiffness_ratio,
            contact_density,
            force_high_stability,
        } => {
            gpu.stiff_ratio_threshold = stiff_ratio_threshold;
            gpu.contact_density_threshold = contact_density_threshold;
            gpu.stiffness_ratio = stiffness_ratio;
            gpu.contact_density = contact_density;
            gpu.force_high_stability = u32::from(force_high_stability);
        }
        ConstraintsQuery::ConvergenceForTier { tier } => {
            gpu.tier = tier_code(tier);
        }
    }
    gpu
}

/// Maps a [`SolverTier`] to the `u32` code the kernel and encoder share.
#[must_use]
const fn tier_code(tier: SolverTier) -> u32 {
    match tier {
        SolverTier::Xpbd => 0,
        SolverTier::Vbd => 1,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ConstraintsResult`],
/// selecting the fields the originating `query` variant produced.
fn decode_result(query: &ConstraintsQuery, raw: &GpuResult) -> ConstraintsResult {
    match query {
        ConstraintsQuery::EffectiveCompliance { .. } => {
            ConstraintsResult::EffectiveCompliance { alpha_tilde: raw.a }
        }
        ConstraintsQuery::ComplianceFromStiffness { .. } => {
            ConstraintsResult::ComplianceFromStiffness { alpha: raw.a }
        }
        ConstraintsQuery::ComplianceScaledForSubstep { .. } => {
            ConstraintsResult::ComplianceScaledForSubstep { alpha_tilde: raw.a }
        }
        ConstraintsQuery::ComplianceIsRigid { .. } => ConstraintsResult::ComplianceIsRigid {
            is_rigid: raw.a > 0.5,
        },
        ConstraintsQuery::SubstepDt { .. } => ConstraintsResult::SubstepDt { substep_dt: raw.a },
        ConstraintsQuery::ScheduleEffectiveCompliance { .. } => {
            ConstraintsResult::ScheduleEffectiveCompliance { alpha_tilde: raw.a }
        }
        ConstraintsQuery::ProjectDistance { .. } => ConstraintsResult::ProjectDistance {
            delta_a: [raw.a, raw.b, raw.c],
            delta_b: [raw.d, raw.e, raw.f],
            delta_lambda: raw.g,
        },
        ConstraintsQuery::ConstraintStrain { .. } => {
            ConstraintsResult::ConstraintStrain { strain: raw.a }
        }
        ConstraintsQuery::ShouldTear { .. } => ConstraintsResult::ShouldTear { tears: raw.a > 0.5 },
        ConstraintsQuery::RigidFromParticles { .. } => ConstraintsResult::RigidFromParticles {
            inv_mass: raw.a,
            center_of_mass: [raw.b, raw.c, raw.d],
            bounding_radius: raw.e,
        },
        ConstraintsQuery::SolverSelect { .. } => {
            let tier = if raw.a > 0.5 {
                SolverTier::Vbd
            } else {
                SolverTier::Xpbd
            };
            ConstraintsResult::SolverSelect { tier }
        }
        ConstraintsQuery::ConvergenceForTier { .. } => ConstraintsResult::ConvergenceForTier {
            unconditionally_stable: raw.a > 0.5,
            parallelizable: raw.b > 0.5,
            cost_multiplier: raw.c,
        },
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

/// A compiled, reusable constraint-solve compute pipeline, twinning the `CPU`
/// golden [`constraints`](prism_render_architecture::particle::constraints).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::constraints`；无第三方引擎源码或衍生代码。
pub struct GpuConstraints {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuConstraints {
    /// Compiles the constraint-solve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::constraints`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuConstraints {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_constraints"),
            source: ShaderSource::Wgsl(CONSTRAINTS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_constraints_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_constraints_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_constraints_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuConstraints {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`ConstraintsResult`]
    /// per input, in order.
    ///
    /// Each result matches the `CPU` golden within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::constraints`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ConstraintsQuery],
    ) -> Vec<ConstraintsResult> {
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
            label: Some("prism_volumetric_constraints_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_constraints_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_constraints_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_constraints_bind_group"),
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
            label: Some("prism_volumetric_constraints_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_constraints_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_constraints_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}
