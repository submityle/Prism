//! `wgpu` compute twin of the particle numerical-integration *stability policy*
//! ([`stability`](prism_render_architecture::particle::stability), particle
//! design §25).
//!
//! The `CPU` golden
//! [`stability`](prism_render_architecture::particle::stability) owns the
//! fixed-`dt` substep scheduling, the `CFL`-driven substep recommendation, the
//! anti-explosion velocity / displacement clamps and the integrator selection
//! that keep the per-particle integrators stable. [`GpuStabilityQuery`] packs
//! every scalar, enum code and [`Vec3`](prism_render_architecture::particle::Vec3)
//! input one thread needs, and [`GpuStabilityResult`] carries back every answer
//! the reference computes, so a passing real-device parity test is direct
//! evidence the ported kernel reproduces the same schedule, the same clamps and
//! the same integrator choice the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For each query one thread evaluates, branch for branch:
//! [`plan_substeps`](prism_render_architecture::particle::stability::plan_substeps)
//! and
//! [`plan_substeps_fixed`](prism_render_architecture::particle::stability::plan_substeps_fixed)
//! (with the recomposed
//! [`SubstepPlan::total_dt`](prism_render_architecture::particle::stability::SubstepPlan::total_dt)),
//! the
//! [`XpbdSubstepPlan`](prism_render_architecture::particle::stability::XpbdSubstepPlan)
//! solver-iteration floor and its
//! [`compliance_over_dt_sq`](prism_render_architecture::particle::stability::XpbdSubstepPlan::compliance_over_dt_sq),
//! the
//! [`cfl_number`](prism_render_architecture::particle::stability::cfl_number) and
//! the
//! [`cfl_decision`](prism_render_architecture::particle::stability::cfl_decision)
//! substep recommendation,
//! [`clamp_velocity`](prism_render_architecture::particle::stability::clamp_velocity),
//! [`clamp_displacement`](prism_render_architecture::particle::stability::clamp_displacement)
//! and the combined
//! [`StepLimits::apply`](prism_render_architecture::particle::stability::StepLimits::apply),
//! the
//! [`stable_dt_bound`](prism_render_architecture::particle::stability::stable_dt_bound)
//! stability bound and the
//! [`select_integrator`](prism_render_architecture::particle::stability::select_integrator)
//! choice.
//!
//! # Correctness model
//!
//! The substep counts, the `XPBD` iteration count, the `CFL` clamp flag and the
//! integrator code are discrete classifications, so for inputs clear of the
//! branch ties `CPU` and `GPU` agree exactly and the parity test asserts an
//! exact `==`. The timesteps, the `CFL` number, the compliance, the clamped
//! vectors and the stability bound thread through multiplies, divides and
//! `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every continuous quantity.
//!
//! # Degenerate inputs
//!
//! A non-positive frame or target `dt` folds to a single substep of the whole
//! frame; a non-positive substep `dt` yields a zero `XPBD` compliance rather
//! than dividing by zero; a non-positive `cell_size` yields a zero `CFL`
//! number; a non-positive clamp limit disables that clamp; and a non-positive
//! stiffness has no stability bound and returns [`f32::INFINITY`]. The parity
//! fixtures keep every input clear of the branch ties (see the test module), so
//! `CPU` and `GPU` always take the same branch. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `ceil`, `sqrt`, `abs`, `dot`, `+ - * /`, unsigned index math and one
//! `bitcast` for the infinity sentinel — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no inverse trigonometry and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each
//! thread performs a fixed, bounded sequence of arithmetic, so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::stability`；无第三方引擎源码或衍生代码。
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

use prism_render_architecture::particle::stability::IntegrationNeeds;
use prism_render_architecture::particle::{IntegratorKind, Vec3};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// `WGSL` / host code for semi-implicit Euler, matching the reference
/// [`IntegratorKind::SemiImplicitEuler`](prism_render_architecture::particle::IntegratorKind)
/// enum order.
const INTEGRATOR_EULER: u32 = 0;
/// `WGSL` / host code for position `Verlet`, matching the reference
/// [`IntegratorKind::Verlet`](prism_render_architecture::particle::IntegratorKind)
/// enum order.
const INTEGRATOR_VERLET: u32 = 1;
/// `WGSL` / host code for midpoint `RK2`, matching the reference
/// [`IntegratorKind::Rk2`](prism_render_architecture::particle::IntegratorKind)
/// enum order.
const INTEGRATOR_RK2: u32 = 2;

/// The portable core-`WGSL` stability-policy kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`stability`](prism_render_architecture::particle::stability) branch for
/// branch; see the module documentation for the contract.
const STABILITY_WGSL: &str = r#"
// Stability-policy twin: one thread per query reproduces the fixed-dt substep
// plans, the XPBD compliance, the CFL number and decision, the velocity and
// displacement clamps, the stability bound and the integrator selection. It
// mirrors the CPU golden `particle::stability` branch for branch, uses only the
// portable core-WGSL subset (min/max/ceil/sqrt/abs/dot and + - * / plus
// unsigned index math and one bitcast for the infinity sentinel), needs no
// transcendental call and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. There is no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::stability；无第三方
// 引擎源码或衍生代码。

// Squared-length floor below which a vector is treated as zero before a
// direction-preserving clamp. Matches the reference `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Integrator classification codes, in the reference `IntegratorKind` order.
const INTEGRATOR_EULER: u32 = 0u;
const INTEGRATOR_VERLET: u32 = 1u;
const INTEGRATOR_RK2: u32 = 2u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Velocity shared by clamp_velocity and StepLimits::apply; a pad lane follows.
    velocity: vec3<f32>,
    pad_v: f32,
    // Standalone displacement delta for clamp_displacement; a pad lane follows.
    displacement_delta: vec3<f32>,
    pad_d: f32,
    // Scalar inputs.
    frame_dt: f32,
    target_substep_dt: f32,
    compliance: f32,
    cfl_max_speed: f32,
    cfl_dt: f32,
    cfl_cell_size: f32,
    cfl_limit: f32,
    step_dt: f32,
    step_max_speed: f32,
    step_max_step: f32,
    clamp_vel_max_speed: f32,
    clamp_disp_max_step: f32,
    stiffness: f32,
    needs_stiffness: f32,
    // Unsigned inputs.
    max_substeps: u32,
    fixed_substeps: u32,
    solver_iterations: u32,
    cfl_max_substeps: u32,
    integrator_code: u32,
    needs_positional: u32,
    needs_high_accuracy: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // clamp_velocity result; a pad lane follows.
    clamp_velocity_out: vec3<f32>,
    pad_cv: f32,
    // clamp_displacement result; a pad lane follows.
    clamp_displacement_out: vec3<f32>,
    pad_cd: f32,
    // StepLimits::apply clamped velocity; a pad lane follows.
    step_velocity: vec3<f32>,
    pad_sv: f32,
    // StepLimits::apply clamped displacement; a pad lane follows.
    step_displacement: vec3<f32>,
    pad_sd: f32,
    // Continuous scalar outputs.
    plan_dt: f32,
    plan_total_dt: f32,
    fixed_dt: f32,
    xpbd_compliance: f32,
    cfl_number_out: f32,
    cfl_decision_number: f32,
    stable_dt_bound_out: f32,
    // Discrete outputs.
    plan_substeps: u32,
    fixed_substeps_out: u32,
    xpbd_solver_iterations: u32,
    cfl_decision_substeps: u32,
    cfl_decision_clamped: u32,
    select_integrator_code: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The CFL number max_speed * dt / cell_size; a non-positive cell_size returns
// 0.0 rather than dividing by zero. Mirrors the reference `cfl_number`.
fn cfl_number(max_speed: f32, dt: f32, cell_size: f32) -> f32 {
    if (cell_size > 0.0) {
        return max_speed * dt / cell_size;
    }
    return 0.0;
}

// Clamps a velocity to `max_speed`, preserving direction; a non-positive limit
// disables the clamp. Mirrors the reference `clamp_velocity`.
fn clamp_velocity(velocity: vec3<f32>, max_speed: f32) -> vec3<f32> {
    if (max_speed <= 0.0) {
        return velocity;
    }
    let speed_sq = dot(velocity, velocity);
    if (speed_sq > max_speed * max_speed && speed_sq > EPS_LEN_SQ) {
        return velocity * (max_speed / sqrt(speed_sq));
    }
    return velocity;
}

// Clamps a per-step displacement to `max_step`, preserving direction; a
// non-positive limit disables the clamp. Mirrors the reference
// `clamp_displacement`.
fn clamp_displacement(delta: vec3<f32>, max_step: f32) -> vec3<f32> {
    if (max_step <= 0.0) {
        return delta;
    }
    let len_sq = dot(delta, delta);
    if (len_sq > max_step * max_step && len_sq > EPS_LEN_SQ) {
        return delta * (max_step / sqrt(len_sq));
    }
    return delta;
}

// Approximate maximum stable dt for the integrator code at a given stiffness; a
// non-positive stiffness returns +inf. Mirrors the reference `stable_dt_bound`.
fn stable_dt_bound(code: u32, stiffness: f32) -> f32 {
    if (stiffness <= 0.0) {
        return bitcast<f32>(0x7f800000u);
    }
    var coefficient: f32 = 2.0;
    if (code == INTEGRATOR_RK2) {
        coefficient = 2.0 * sqrt(2.0);
    }
    return coefficient / sqrt(stiffness);
}

// Chooses an integrator code: positional constraints win (Verlet), else high
// accuracy (RK2), else semi-implicit Euler. Mirrors `select_integrator`.
fn select_integrator(positional: u32, high_accuracy: u32) -> u32 {
    if (positional != 0u) {
        return INTEGRATOR_VERLET;
    }
    if (high_accuracy != 0u) {
        return INTEGRATOR_RK2;
    }
    return INTEGRATOR_EULER;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // plan_substeps: ceil(frame_dt / target) clamped to 1..=max_substeps, then
    // the actual fixed dt that recomposes the frame. Degenerate inputs fall back
    // to a single substep of the whole frame.
    let ceiling = max(q.max_substeps, 1u);
    var plan_substeps: u32;
    var plan_dt: f32;
    if (q.frame_dt <= 0.0 || q.target_substep_dt <= 0.0) {
        plan_substeps = 1u;
        plan_dt = max(q.frame_dt, 0.0);
    } else {
        let needed = ceil(q.frame_dt / q.target_substep_dt);
        if (needed >= f32(ceiling)) {
            plan_substeps = ceiling;
        } else {
            plan_substeps = max(u32(needed), 1u);
        }
        plan_dt = q.frame_dt / f32(plan_substeps);
    }
    let plan_total_dt = f32(plan_substeps) * plan_dt;

    // plan_substeps_fixed: at least one substep; a non-positive frame yields a
    // zero dt. The XPBD plan is built on this fixed schedule.
    let fixed_substeps = max(q.fixed_substeps, 1u);
    var fixed_dt: f32 = 0.0;
    if (q.frame_dt > 0.0) {
        fixed_dt = q.frame_dt / f32(fixed_substeps);
    }

    // XpbdSubstepPlan::new forces at least one solver iteration;
    // compliance_over_dt_sq is alpha / dt^2, or 0.0 for a non-positive dt.
    let xpbd_iters = max(q.solver_iterations, 1u);
    var xpbd_compliance: f32 = 0.0;
    if (fixed_dt > 0.0) {
        xpbd_compliance = q.compliance / (fixed_dt * fixed_dt);
    }

    // cfl_number for the whole frame.
    let cfl_num = cfl_number(q.cfl_max_speed, q.cfl_dt, q.cfl_cell_size);

    // cfl_decision: ceil(number / limit) clamped to 1..=cfl_max_substeps, with a
    // clamp flag when the ceiling still cannot satisfy the limit. Degenerate
    // inputs fall back to a single unclamped substep.
    let cfl_ceiling = max(q.cfl_max_substeps, 1u);
    var cfl_substeps: u32 = 1u;
    var cfl_clamped: u32 = 0u;
    if (cfl_num <= q.cfl_limit || q.cfl_limit <= 0.0) {
        cfl_substeps = 1u;
        cfl_clamped = 0u;
    } else {
        let needed_cfl = ceil(cfl_num / q.cfl_limit);
        if (needed_cfl >= f32(cfl_ceiling)) {
            cfl_substeps = cfl_ceiling;
            cfl_clamped = 1u;
        } else {
            cfl_substeps = max(u32(needed_cfl), 1u);
            cfl_clamped = 0u;
        }
    }

    // Standalone clamps.
    let clamped_vel = clamp_velocity(q.velocity, q.clamp_vel_max_speed);
    let clamped_disp = clamp_displacement(q.displacement_delta, q.clamp_disp_max_step);

    // StepLimits::apply: clamp the velocity first, then the dt-scaled
    // displacement of the clamped velocity.
    let step_vel = clamp_velocity(q.velocity, q.step_max_speed);
    let step_disp = clamp_displacement(step_vel * q.step_dt, q.step_max_step);

    // Stability bound and integrator selection.
    let bound = stable_dt_bound(q.integrator_code, q.stiffness);
    let integ = select_integrator(q.needs_positional, q.needs_high_accuracy);

    var out: Result;
    out.clamp_velocity_out = clamped_vel;
    out.pad_cv = 0.0;
    out.clamp_displacement_out = clamped_disp;
    out.pad_cd = 0.0;
    out.step_velocity = step_vel;
    out.pad_sv = 0.0;
    out.step_displacement = step_disp;
    out.pad_sd = 0.0;
    out.plan_dt = plan_dt;
    out.plan_total_dt = plan_total_dt;
    out.fixed_dt = fixed_dt;
    out.xpbd_compliance = xpbd_compliance;
    out.cfl_number_out = cfl_num;
    out.cfl_decision_number = cfl_num;
    out.stable_dt_bound_out = bound;
    out.plan_substeps = plan_substeps;
    out.fixed_substeps_out = fixed_substeps;
    out.xpbd_solver_iterations = xpbd_iters;
    out.cfl_decision_substeps = cfl_substeps;
    out.cfl_decision_clamped = cfl_clamped;
    out.select_integrator_code = integ;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`STABILITY_WGSL`].
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
/// Each `vec3` carries a trailing pad word so it stays `16`-byte aligned on
/// device, and the whole struct is a multiple of `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Velocity for `clamp_velocity` and `StepLimits::apply`.
    velocity: [f32; 3],
    /// Pad lane after `velocity`.
    pad_v: f32,
    /// Delta for the standalone `clamp_displacement`.
    displacement_delta: [f32; 3],
    /// Pad lane after `displacement_delta`.
    pad_d: f32,
    /// Frame `dt` for both substep planners.
    frame_dt: f32,
    /// Target per-substep `dt` for `plan_substeps`.
    target_substep_dt: f32,
    /// Raw `XPBD` compliance `alpha`.
    compliance: f32,
    /// Max speed for the `CFL` number.
    cfl_max_speed: f32,
    /// Step `dt` for the `CFL` number.
    cfl_dt: f32,
    /// Grid cell size for the `CFL` number.
    cfl_cell_size: f32,
    /// `CFL` limit for `cfl_decision`.
    cfl_limit: f32,
    /// `dt` for `StepLimits::apply`.
    step_dt: f32,
    /// `StepLimits` max speed.
    step_max_speed: f32,
    /// `StepLimits` max step.
    step_max_step: f32,
    /// Max speed for the standalone `clamp_velocity`.
    clamp_vel_max_speed: f32,
    /// Max step for the standalone `clamp_displacement`.
    clamp_disp_max_step: f32,
    /// Stiffness for `stable_dt_bound`.
    stiffness: f32,
    /// `IntegrationNeeds` stiffness (unused by the selection; carried for parity).
    needs_stiffness: f32,
    /// Max substeps for `plan_substeps`.
    max_substeps: u32,
    /// Explicit substep count for `plan_substeps_fixed`.
    fixed_substeps: u32,
    /// Solver iterations for `XpbdSubstepPlan::new`.
    solver_iterations: u32,
    /// Max substeps for `cfl_decision`.
    cfl_max_substeps: u32,
    /// Integrator code for `stable_dt_bound`.
    integrator_code: u32,
    /// `IntegrationNeeds` positional-constraints flag as `0` / `1`.
    needs_positional: u32,
    /// `IntegrationNeeds` high-accuracy flag as `0` / `1`.
    needs_high_accuracy: u32,
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
    /// `clamp_velocity` output.
    clamp_velocity_out: [f32; 3],
    /// Pad lane after `clamp_velocity_out`.
    pad_cv: f32,
    /// `clamp_displacement` output.
    clamp_displacement_out: [f32; 3],
    /// Pad lane after `clamp_displacement_out`.
    pad_cd: f32,
    /// `StepLimits::apply` clamped velocity.
    step_velocity: [f32; 3],
    /// Pad lane after `step_velocity`.
    pad_sv: f32,
    /// `StepLimits::apply` clamped displacement.
    step_displacement: [f32; 3],
    /// Pad lane after `step_displacement`.
    pad_sd: f32,
    /// `plan_substeps` fixed per-substep `dt`.
    plan_dt: f32,
    /// `plan_substeps` recomposed total `dt`.
    plan_total_dt: f32,
    /// `plan_substeps_fixed` per-substep `dt`.
    fixed_dt: f32,
    /// `XpbdSubstepPlan::compliance_over_dt_sq` output.
    xpbd_compliance: f32,
    /// `cfl_number` output.
    cfl_number_out: f32,
    /// `cfl_decision` `cfl_number` field (equals `cfl_number_out`).
    cfl_decision_number: f32,
    /// `stable_dt_bound` output.
    stable_dt_bound_out: f32,
    /// `plan_substeps` substep count.
    plan_substeps: u32,
    /// `plan_substeps_fixed` substep count.
    fixed_substeps_out: u32,
    /// `XpbdSubstepPlan` solver-iteration count.
    xpbd_solver_iterations: u32,
    /// `cfl_decision` recommended substep count.
    cfl_decision_substeps: u32,
    /// `cfl_decision` clamp flag as `0` / `1`.
    cfl_decision_clamped: u32,
    /// `select_integrator` chosen integrator code.
    select_integrator_code: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query for the stability twin: every scalar, enum code and
/// [`Vec3`](prism_render_architecture::particle::Vec3) input the twinned
/// functions consume, bundled so a single thread exercises the whole policy at
/// once.
///
/// The fields are grouped by the function they drive; the sub-problems are
/// independent, so a single query covers substep planning, the `CFL` decision,
/// both clamps, the combined step and the integrator selection together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuStabilityQuery {
    /// Velocity for `clamp_velocity` and `StepLimits::apply`.
    pub velocity: Vec3,
    /// Delta for the standalone `clamp_displacement`.
    pub displacement: Vec3,
    /// Frame `dt` fed to both substep planners.
    pub frame_dt: f32,
    /// Target per-substep `dt` for `plan_substeps`.
    pub target_substep_dt: f32,
    /// Substep ceiling for `plan_substeps`.
    pub max_substeps: u32,
    /// Explicit substep count for `plan_substeps_fixed` and the `XPBD` base.
    pub fixed_substeps: u32,
    /// Solver iterations for `XpbdSubstepPlan::new`.
    pub solver_iterations: u32,
    /// Raw `XPBD` compliance `alpha`.
    pub compliance: f32,
    /// Max speed for the `CFL` number and decision.
    pub cfl_max_speed: f32,
    /// Step `dt` for the `CFL` number and decision.
    pub cfl_dt: f32,
    /// Grid cell size for the `CFL` number and decision.
    pub cfl_cell_size: f32,
    /// `CFL` limit for `cfl_decision`.
    pub cfl_limit: f32,
    /// Substep ceiling for `cfl_decision`.
    pub cfl_max_substeps: u32,
    /// `dt` for `StepLimits::apply`.
    pub step_dt: f32,
    /// `StepLimits` max speed.
    pub step_max_speed: f32,
    /// `StepLimits` max step.
    pub step_max_step: f32,
    /// Max speed for the standalone `clamp_velocity`.
    pub clamp_velocity_max_speed: f32,
    /// Max step for the standalone `clamp_displacement`.
    pub clamp_displacement_max_step: f32,
    /// Integrator for `stable_dt_bound`.
    pub integrator: IntegratorKind,
    /// Stiffness for `stable_dt_bound`.
    pub stiffness: f32,
    /// Request fed to `select_integrator`.
    pub needs: IntegrationNeeds,
}

/// One resolved answer for a single query, mirroring every value the reference
/// [`stability`](prism_render_architecture::particle::stability) policy reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuStabilityResult {
    /// `plan_substeps` substep count.
    pub plan_substeps: u32,
    /// `plan_substeps` fixed per-substep `dt`.
    pub plan_dt: f32,
    /// `plan_substeps` recomposed total `dt` (`SubstepPlan::total_dt`).
    pub plan_total_dt: f32,
    /// `plan_substeps_fixed` substep count.
    pub fixed_substeps: u32,
    /// `plan_substeps_fixed` per-substep `dt`.
    pub fixed_dt: f32,
    /// `XpbdSubstepPlan` solver-iteration count.
    pub xpbd_solver_iterations: u32,
    /// `XpbdSubstepPlan::compliance_over_dt_sq` output.
    pub xpbd_compliance_over_dt_sq: f32,
    /// `cfl_number` output.
    pub cfl_number: f32,
    /// `cfl_decision` recommended substep count.
    pub cfl_substeps: u32,
    /// `cfl_decision` clamp flag.
    pub cfl_clamped: bool,
    /// `clamp_velocity` output.
    pub clamped_velocity: Vec3,
    /// `clamp_displacement` output.
    pub clamped_displacement: Vec3,
    /// `StepLimits::apply` clamped velocity.
    pub step_velocity: Vec3,
    /// `StepLimits::apply` clamped displacement.
    pub step_displacement: Vec3,
    /// `stable_dt_bound` output.
    pub stable_dt_bound: f32,
    /// `select_integrator` chosen integrator.
    pub integrator: IntegratorKind,
}

/// Maps an [`IntegratorKind`] to its `WGSL` integrator code.
fn integrator_code(kind: IntegratorKind) -> u32 {
    match kind {
        IntegratorKind::SemiImplicitEuler => INTEGRATOR_EULER,
        IntegratorKind::Verlet => INTEGRATOR_VERLET,
        IntegratorKind::Rk2 => INTEGRATOR_RK2,
    }
}

/// Maps a `WGSL` integrator code back to an [`IntegratorKind`].
///
/// # Panics
///
/// Panics on a code outside `0..=2`, which cannot occur: the kernel only ever
/// writes one of the three integrator codes.
fn integrator_from_code(code: u32) -> IntegratorKind {
    match code {
        INTEGRATOR_EULER => IntegratorKind::SemiImplicitEuler,
        INTEGRATOR_VERLET => IntegratorKind::Verlet,
        INTEGRATOR_RK2 => IntegratorKind::Rk2,
        other => panic!("kernel wrote an out-of-range integrator code: {other}"),
    }
}

/// Encodes one [`GpuStabilityQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &GpuStabilityQuery) -> GpuQuery {
    GpuQuery {
        velocity: [q.velocity.x, q.velocity.y, q.velocity.z],
        pad_v: 0.0,
        displacement_delta: [q.displacement.x, q.displacement.y, q.displacement.z],
        pad_d: 0.0,
        frame_dt: q.frame_dt,
        target_substep_dt: q.target_substep_dt,
        compliance: q.compliance,
        cfl_max_speed: q.cfl_max_speed,
        cfl_dt: q.cfl_dt,
        cfl_cell_size: q.cfl_cell_size,
        cfl_limit: q.cfl_limit,
        step_dt: q.step_dt,
        step_max_speed: q.step_max_speed,
        step_max_step: q.step_max_step,
        clamp_vel_max_speed: q.clamp_velocity_max_speed,
        clamp_disp_max_step: q.clamp_displacement_max_step,
        stiffness: q.stiffness,
        needs_stiffness: q.needs.stiffness,
        max_substeps: q.max_substeps,
        fixed_substeps: q.fixed_substeps,
        solver_iterations: q.solver_iterations,
        cfl_max_substeps: q.cfl_max_substeps,
        integrator_code: integrator_code(q.integrator),
        needs_positional: u32::from(q.needs.positional_constraints),
        needs_high_accuracy: u32::from(q.needs.high_accuracy),
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuStabilityResult`],
/// turning the clamp flag into a `bool` and the integrator code back into an
/// [`IntegratorKind`].
fn decode_result(raw: &GpuResult) -> GpuStabilityResult {
    GpuStabilityResult {
        plan_substeps: raw.plan_substeps,
        plan_dt: raw.plan_dt,
        plan_total_dt: raw.plan_total_dt,
        fixed_substeps: raw.fixed_substeps_out,
        fixed_dt: raw.fixed_dt,
        xpbd_solver_iterations: raw.xpbd_solver_iterations,
        xpbd_compliance_over_dt_sq: raw.xpbd_compliance,
        cfl_number: raw.cfl_number_out,
        cfl_substeps: raw.cfl_decision_substeps,
        cfl_clamped: raw.cfl_decision_clamped != 0,
        clamped_velocity: Vec3::new(
            raw.clamp_velocity_out[0],
            raw.clamp_velocity_out[1],
            raw.clamp_velocity_out[2],
        ),
        clamped_displacement: Vec3::new(
            raw.clamp_displacement_out[0],
            raw.clamp_displacement_out[1],
            raw.clamp_displacement_out[2],
        ),
        step_velocity: Vec3::new(
            raw.step_velocity[0],
            raw.step_velocity[1],
            raw.step_velocity[2],
        ),
        step_displacement: Vec3::new(
            raw.step_displacement[0],
            raw.step_displacement[1],
            raw.step_displacement[2],
        ),
        stable_dt_bound: raw.stable_dt_bound_out,
        integrator: integrator_from_code(raw.select_integrator_code),
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

/// A compiled, reusable stability-policy compute pipeline, twinning the `CPU`
/// golden [`stability`](prism_render_architecture::particle::stability).
pub struct GpuStability {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuStability {
    /// Compiles the stability-policy kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuStability {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_stability"),
            source: ShaderSource::Wgsl(STABILITY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_stability_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_stability_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_stability_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuStability {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`GpuStabilityResult`]
    /// per input, in order.
    ///
    /// The discrete counts, the clamp flag and the integrator choice equal the
    /// reference exactly for inputs clear of the branch ties; the continuous
    /// quantities match to within the tolerance documented on this module. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuStabilityQuery],
    ) -> Vec<GpuStabilityResult> {
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
            label: Some("prism_volumetric_stability_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_stability_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_stability_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_stability_bind_group"),
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
            label: Some("prism_volumetric_stability_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_stability_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_stability_pass"),
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
