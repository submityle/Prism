//! Real-device parity for the numerical-integration stability twin:
//! [`GpuStability`](prism_volumetric_gpu::GpuStability) must reproduce the `CPU`
//! golden [`stability`](prism_render_architecture::particle::stability) across
//! the fixed-`dt` substep plans
//! ([`plan_substeps`](prism_render_architecture::particle::stability::plan_substeps),
//! [`plan_substeps_fixed`](prism_render_architecture::particle::stability::plan_substeps_fixed)
//! and the recomposed
//! [`SubstepPlan::total_dt`](prism_render_architecture::particle::stability::SubstepPlan::total_dt)),
//! the `XPBD` solver-iteration floor and its
//! [`compliance_over_dt_sq`](prism_render_architecture::particle::stability::XpbdSubstepPlan::compliance_over_dt_sq),
//! the [`cfl_number`](prism_render_architecture::particle::stability::cfl_number)
//! and [`cfl_decision`](prism_render_architecture::particle::stability::cfl_decision)
//! substep recommendation, the
//! [`clamp_velocity`](prism_render_architecture::particle::stability::clamp_velocity),
//! [`clamp_displacement`](prism_render_architecture::particle::stability::clamp_displacement)
//! and combined
//! [`StepLimits::apply`](prism_render_architecture::particle::stability::StepLimits::apply)
//! anti-explosion clamps, the
//! [`stable_dt_bound`](prism_render_architecture::particle::stability::stable_dt_bound)
//! stability bound and the
//! [`select_integrator`](prism_render_architecture::particle::stability::select_integrator)
//! choice.
//!
//! The fixtures cover the branches the golden unit tests call out: a frame that
//! rounds up and shrinks the fixed `dt`, a frame whose target forces the substep
//! ceiling, degenerate non-positive `dt` / `target` / `cell_size` inputs, the
//! `CFL` single-step, mid-range and ceiling-clamp branches, active and disabled
//! velocity / displacement clamps, each [`IntegratorKind`] stability bound
//! (including the non-positive-stiffness [`f32::INFINITY`] case), every
//! [`select_integrator`](prism_render_architecture::particle::stability::select_integrator)
//! priority, an empty short-circuit batch and a randomized batch. Every
//! randomized fixture is rejection-sampled clear of its branch ties so `CPU` and
//! `GPU` always take the same branch; the sampler uses only an integer `LCG`, a
//! divide and `sqrt`, never a transcendental call.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The substep counts, the `XPBD` iteration count, the `CFL` clamp flag and the
//! chosen integrator are discrete classifications, so `CPU` and `GPU` must agree
//! exactly: the comparison is an exact `==`. The timesteps, the `CFL` number,
//! the compliance, the clamped vectors and the stability bound thread through
//! multiplies, divides and `sqrt`, so they are compared under tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`); the
//! stability bound additionally matches [`f32::INFINITY`] exactly in the
//! degenerate case.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::stability`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::stability::{
    cfl_decision, cfl_number, clamp_displacement, clamp_velocity, plan_substeps,
    plan_substeps_fixed, select_integrator, stable_dt_bound, IntegrationNeeds, StepLimits,
    XpbdSubstepPlan,
};
use prism_render_architecture::particle::{IntegratorKind, Vec3};
use prism_volumetric_gpu::{GpuContext, GpuStability, GpuStabilityQuery};

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two [`Vec3`] values, lane by lane.
fn approx_vec(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

/// Tolerant comparison of a [`stable_dt_bound`] pair, matching
/// [`f32::INFINITY`] exactly (same sign) in the non-positive-stiffness case and
/// otherwise under the shared tolerance.
fn approx_bound(gpu: f32, cpu: f32) -> bool {
    if cpu.is_infinite() {
        return gpu.is_infinite() && gpu.is_sign_positive() == cpu.is_sign_positive();
    }
    approx(gpu, cpu)
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        range(state, -span, span),
        range(state, -span, span),
        range(state, -span, span),
    )
}

/// Euclidean length, so the rejection sampler needs no library helper beyond
/// `sqrt` (which the house rules permit).
fn length(v: Vec3) -> f32 {
    (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
}

/// `true` when `value` sits within `frac` of `limit`, i.e. too close to a clamp
/// or compare tie to classify identically on both devices.
fn near(value: f32, limit: f32, frac: f32) -> bool {
    (value - limit).abs() < frac * limit.abs().max(REL_FLOOR)
}

/// `true` when `ratio`'s fractional part is near an integer boundary, where the
/// `ceil` in [`plan_substeps`] / [`cfl_decision`] could land differently across
/// devices.
fn frac_near_boundary(ratio: f32) -> bool {
    let frac = ratio - ratio.floor();
    !(0.2..=0.8).contains(&frac)
}

/// A well-conditioned default query, clear of every branch tie. Individual
/// tests override just the fields they exercise.
fn base_query() -> GpuStabilityQuery {
    GpuStabilityQuery {
        velocity: Vec3::new(3.0, 4.0, 0.0),
        displacement: Vec3::new(0.0, 0.0, 2.0),
        frame_dt: 0.1,
        target_substep_dt: 0.023,
        max_substeps: 64,
        fixed_substeps: 4,
        solver_iterations: 3,
        compliance: 1.0e-4,
        cfl_max_speed: 10.0,
        cfl_dt: 0.1,
        cfl_cell_size: 2.0,
        cfl_limit: 1.0,
        cfl_max_substeps: 64,
        step_dt: 0.1,
        step_max_speed: 2.0,
        step_max_step: 0.1,
        clamp_velocity_max_speed: 2.5,
        clamp_displacement_max_step: 1.0,
        integrator: IntegratorKind::SemiImplicitEuler,
        stiffness: 4.0,
        needs: IntegrationNeeds {
            stiffness: 4.0,
            positional_constraints: false,
            high_accuracy: false,
        },
    }
}

/// A rejection-sampled query whose every twinned function lands clear of its
/// branch tie, so the `CPU` and `GPU` take the same branch bit for bit.
fn rand_query(state: &mut u64) -> GpuStabilityQuery {
    loop {
        let frame_dt = range(state, 0.01, 0.21);
        let target_substep_dt = range(state, 0.005, 0.055);
        // The substep ceiling is high enough that `needed` never ties it.
        if frac_near_boundary(frame_dt / target_substep_dt) {
            continue;
        }

        let cfl_max_speed = range(state, 1.0, 50.0);
        let cfl_dt = range(state, 0.005, 0.055);
        let cfl_cell_size = range(state, 0.5, 3.5);
        let cfl_limit = range(state, 0.5, 2.5);
        let number = cfl_max_speed * cfl_dt / cfl_cell_size;
        // Keep the number clear of the limit compare and, when it exceeds the
        // limit, keep the recommended-substep `ceil` clear of its boundary and
        // clear of the ceiling of 64.
        if near(number, cfl_limit, 0.1) {
            continue;
        }
        if number > cfl_limit {
            let ratio = number / cfl_limit;
            if frac_near_boundary(ratio) || ratio.ceil() >= 63.0 {
                continue;
            }
        }

        let velocity = rand_vec(state, 10.0);
        let vlen = length(velocity);
        let displacement = rand_vec(state, 2.0);
        let dlen = length(displacement);
        let clamp_velocity_max_speed = range(state, 1.0, 9.0);
        let clamp_displacement_max_step = range(state, 0.2, 2.2);
        // Both standalone clamps must sit clearly above or below their limit.
        if near(vlen, clamp_velocity_max_speed, 0.1) || near(dlen, clamp_displacement_max_step, 0.1)
        {
            continue;
        }

        let step_dt = range(state, 0.005, 0.055);
        let step_max_speed = range(state, 1.0, 9.0);
        let step_max_step = range(state, 0.1, 2.1);
        // The combined step clamps the velocity first, then the displacement of
        // that clamped velocity; both must land clear of their ties.
        if near(vlen, step_max_speed, 0.1) {
            continue;
        }
        let clamped_speed = vlen.min(step_max_speed);
        if near(clamped_speed * step_dt, step_max_step, 0.1) {
            continue;
        }

        // A positive, finite stiffness keeps the stability bound finite.
        let stiffness = range(state, 1.0, 100.0);
        let code = (lcg(state) * 3.0) as u32;
        let integrator = match code {
            0 => IntegratorKind::SemiImplicitEuler,
            1 => IntegratorKind::Verlet,
            _ => IntegratorKind::Rk2,
        };
        let needs = IntegrationNeeds {
            stiffness,
            positional_constraints: lcg(state) > 0.5,
            high_accuracy: lcg(state) > 0.5,
        };

        return GpuStabilityQuery {
            velocity,
            displacement,
            frame_dt,
            target_substep_dt,
            max_substeps: 64,
            fixed_substeps: 1 + (lcg(state) * 8.0) as u32,
            solver_iterations: 1 + (lcg(state) * 8.0) as u32,
            compliance: range(state, 1.0e-5, 1.0e-3),
            cfl_max_speed,
            cfl_dt,
            cfl_cell_size,
            cfl_limit,
            cfl_max_substeps: 64,
            step_dt,
            step_max_speed,
            step_max_step,
            clamp_velocity_max_speed,
            clamp_displacement_max_step,
            integrator,
            stiffness,
            needs,
        };
    }
}

/// Asserts every twinned answer for one query matches the `CPU` golden.
fn assert_parity(gpu: &GpuStability, ctx: &GpuContext, q: &GpuStabilityQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    let plan = plan_substeps(q.frame_dt, q.target_substep_dt, q.max_substeps);
    assert_eq!(
        g.plan_substeps, plan.substeps,
        "plan_substeps count mismatch"
    );
    assert!(
        approx(g.plan_dt, plan.dt),
        "plan_dt mismatch: gpu {} vs cpu {}",
        g.plan_dt,
        plan.dt
    );
    assert!(
        approx(g.plan_total_dt, plan.total_dt()),
        "plan_total_dt mismatch: gpu {} vs cpu {}",
        g.plan_total_dt,
        plan.total_dt()
    );

    let fixed = plan_substeps_fixed(q.frame_dt, q.fixed_substeps);
    assert_eq!(
        g.fixed_substeps, fixed.substeps,
        "plan_substeps_fixed count mismatch"
    );
    assert!(
        approx(g.fixed_dt, fixed.dt),
        "fixed_dt mismatch: gpu {} vs cpu {}",
        g.fixed_dt,
        fixed.dt
    );

    let xpbd = XpbdSubstepPlan::new(fixed, q.solver_iterations);
    assert_eq!(
        g.xpbd_solver_iterations, xpbd.solver_iterations,
        "xpbd solver_iterations mismatch"
    );
    let cpu_compliance = xpbd.compliance_over_dt_sq(q.compliance);
    assert!(
        approx(g.xpbd_compliance_over_dt_sq, cpu_compliance),
        "xpbd compliance mismatch: gpu {} vs cpu {}",
        g.xpbd_compliance_over_dt_sq,
        cpu_compliance
    );

    let cpu_number = cfl_number(q.cfl_max_speed, q.cfl_dt, q.cfl_cell_size);
    assert!(
        approx(g.cfl_number, cpu_number),
        "cfl_number mismatch: gpu {} vs cpu {}",
        g.cfl_number,
        cpu_number
    );
    let decision = cfl_decision(
        q.cfl_max_speed,
        q.cfl_dt,
        q.cfl_cell_size,
        q.cfl_limit,
        q.cfl_max_substeps,
    );
    assert_eq!(
        g.cfl_substeps, decision.substeps,
        "cfl_decision substeps mismatch"
    );
    assert_eq!(
        g.cfl_clamped, decision.clamped,
        "cfl_decision clamp flag mismatch"
    );

    let cpu_cv = clamp_velocity(q.velocity, q.clamp_velocity_max_speed);
    assert!(
        approx_vec(g.clamped_velocity, cpu_cv),
        "clamp_velocity mismatch: gpu {:?} vs cpu {:?}",
        g.clamped_velocity,
        cpu_cv
    );
    let cpu_cd = clamp_displacement(q.displacement, q.clamp_displacement_max_step);
    assert!(
        approx_vec(g.clamped_displacement, cpu_cd),
        "clamp_displacement mismatch: gpu {:?} vs cpu {:?}",
        g.clamped_displacement,
        cpu_cd
    );

    let step = StepLimits {
        max_speed: q.step_max_speed,
        max_step: q.step_max_step,
    }
    .apply(q.velocity, q.step_dt);
    assert!(
        approx_vec(g.step_velocity, step.velocity),
        "step velocity mismatch: gpu {:?} vs cpu {:?}",
        g.step_velocity,
        step.velocity
    );
    assert!(
        approx_vec(g.step_displacement, step.displacement),
        "step displacement mismatch: gpu {:?} vs cpu {:?}",
        g.step_displacement,
        step.displacement
    );

    let cpu_bound = stable_dt_bound(q.integrator, q.stiffness);
    assert!(
        approx_bound(g.stable_dt_bound, cpu_bound),
        "stable_dt_bound mismatch: gpu {} vs cpu {}",
        g.stable_dt_bound,
        cpu_bound
    );
    assert_eq!(
        g.integrator,
        select_integrator(q.needs),
        "select_integrator mismatch"
    );
}

#[test]
fn plan_rounds_up_and_recomposes_the_frame() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStability::new(&ctx);
    // ceil(0.1 / 0.023) = 5 substeps, dt shrinks to recompose the frame.
    assert_parity(&gpu, &ctx, &base_query());
    // A coarser target forces four substeps of 0.025s.
    let mut coarse = base_query();
    coarse.target_substep_dt = 0.028;
    assert_parity(&gpu, &ctx, &coarse);
}

#[test]
fn plan_clamps_to_the_substep_ceiling() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStability::new(&ctx);
    // A tiny target would need many substeps but the ceiling pins it at eight.
    let mut q = base_query();
    q.frame_dt = 0.1;
    q.target_substep_dt = 0.001;
    q.max_substeps = 8;
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn degenerate_dt_folds_to_a_single_substep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStability::new(&ctx);
    // A non-positive frame dt yields one substep of a clamped-to-zero dt, a zero
    // fixed dt, and a rigid (zero) XPBD compliance without dividing by zero.
    let mut zero_frame = base_query();
    zero_frame.frame_dt = 0.0;
    assert_parity(&gpu, &ctx, &zero_frame);
    let mut neg_frame = base_query();
    neg_frame.frame_dt = -0.05;
    assert_parity(&gpu, &ctx, &neg_frame);
    // A non-positive target also degrades to a single whole-frame substep.
    let mut zero_target = base_query();
    zero_target.target_substep_dt = 0.0;
    assert_parity(&gpu, &ctx, &zero_target);
}

#[test]
fn cfl_single_step_mid_range_and_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStability::new(&ctx);
    // number = 0.5 <= limit 1 => single unclamped substep (base).
    assert_parity(&gpu, &ctx, &base_query());
    // number = 5, limit = 1 => five substeps, under the ceiling.
    let mut mid = base_query();
    mid.cfl_max_speed = 100.0;
    mid.cfl_dt = 0.1;
    mid.cfl_cell_size = 2.0;
    mid.cfl_limit = 1.0;
    mid.cfl_max_substeps = 64;
    assert_parity(&gpu, &ctx, &mid);
    // Same number but a ceiling of four clamps the recommendation.
    let mut clamped = mid;
    clamped.cfl_max_substeps = 4;
    assert_parity(&gpu, &ctx, &clamped);
    // A non-positive limit degrades to a single unclamped substep.
    let mut no_limit = mid;
    no_limit.cfl_limit = 0.0;
    assert_parity(&gpu, &ctx, &no_limit);
}

#[test]
fn cfl_number_zero_for_non_positive_cell() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStability::new(&ctx);
    // A non-positive cell size yields a zero CFL number without dividing by zero.
    let mut q = base_query();
    q.cfl_cell_size = 0.0;
    assert_parity(&gpu, &ctx, &q);
    let mut neg = base_query();
    neg.cfl_cell_size = -2.0;
    assert_parity(&gpu, &ctx, &neg);
}

#[test]
fn clamps_active_and_disabled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStability::new(&ctx);
    // Base clamps both the velocity (len 5 -> 2.5) and the displacement (2 -> 1).
    assert_parity(&gpu, &ctx, &base_query());
    // A velocity under its limit and a displacement under its limit pass through.
    let mut under = base_query();
    under.velocity = Vec3::new(1.0, 0.0, 0.0);
    under.clamp_velocity_max_speed = 2.5;
    under.displacement = Vec3::new(0.0, 0.3, 0.0);
    under.clamp_displacement_max_step = 1.0;
    under.step_max_speed = 5.0;
    under.step_max_step = 1.0;
    assert_parity(&gpu, &ctx, &under);
    // Non-positive limits disable both clamps entirely.
    let mut disabled = base_query();
    disabled.clamp_velocity_max_speed = 0.0;
    disabled.clamp_displacement_max_step = -1.0;
    disabled.step_max_speed = 0.0;
    disabled.step_max_step = 0.0;
    assert_parity(&gpu, &ctx, &disabled);
}

#[test]
fn stable_bound_for_each_integrator() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStability::new(&ctx);
    for kind in [
        IntegratorKind::SemiImplicitEuler,
        IntegratorKind::Verlet,
        IntegratorKind::Rk2,
    ] {
        let mut q = base_query();
        q.integrator = kind;
        q.stiffness = 4.0;
        assert_parity(&gpu, &ctx, &q);
    }
    // A non-positive stiffness has no stability bound and returns +inf.
    let mut zero = base_query();
    zero.stiffness = 0.0;
    assert_parity(&gpu, &ctx, &zero);
    let mut neg = base_query();
    neg.stiffness = -10.0;
    assert_parity(&gpu, &ctx, &neg);
}

#[test]
fn select_integrator_follows_priority() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStability::new(&ctx);
    // Positional constraints win even with high accuracy also requested.
    let mut verlet = base_query();
    verlet.needs = IntegrationNeeds {
        stiffness: 100.0,
        positional_constraints: true,
        high_accuracy: true,
    };
    assert_parity(&gpu, &ctx, &verlet);
    // High accuracy without constraints picks RK2.
    let mut rk2 = base_query();
    rk2.needs = IntegrationNeeds {
        stiffness: 1.0,
        positional_constraints: false,
        high_accuracy: true,
    };
    assert_parity(&gpu, &ctx, &rk2);
    // The cheap default otherwise (base already covers Euler).
    assert_parity(&gpu, &ctx, &base_query());
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStability::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}

#[test]
fn randomized_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStability::new(&ctx);
    // A batch exercises the one-thread-per-query flattening; each result must be
    // independent of its neighbours. The sampler keeps every query clear of its
    // branch ties so CPU and GPU take the same branch bit for bit.
    let mut state = 0x5151_a1b2_c3d4_e5f6_u64;
    let batch: Vec<GpuStabilityQuery> = (0..256).map(|_| rand_query(&mut state)).collect();
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}
