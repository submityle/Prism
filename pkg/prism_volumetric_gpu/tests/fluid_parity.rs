//! Real-device parity for the per-point grid-fluid, combustion and
//! vorticity-confinement twin:
//! [`GpuFluid`](prism_volumetric_gpu::GpuFluid) must reproduce the `CPU` golden
//! [`fluid`](prism_render_architecture::particle::fluid) routine for routine
//! across the two advection `CFL` scalars
//! ([`cfl_number`](prism_render_architecture::particle::fluid::cfl_number),
//! [`stable_timestep`](prism_render_architecture::particle::fluid::stable_timestep)),
//! the three advection updates
//! ([`semi_lagrangian_backtrace`](prism_render_architecture::particle::fluid::semi_lagrangian_backtrace),
//! [`maccormack_corrected`](prism_render_architecture::particle::fluid::maccormack_corrected),
//! [`advect_particle`](prism_render_architecture::particle::fluid::advect_particle)),
//! the trilinear kernel
//! ([`trilinear_weights`](prism_render_architecture::particle::fluid::trilinear_weights),
//! [`trilinear_sample`](prism_render_architecture::particle::fluid::trilinear_sample)),
//! the two central-difference operators
//! ([`central_gradient`](prism_render_architecture::particle::fluid::central_gradient),
//! [`subtract_pressure_gradient`](prism_render_architecture::particle::fluid::subtract_pressure_gradient)),
//! the combustion coupling
//! ([`step_combustion`](prism_render_architecture::particle::fluid::step_combustion),
//! [`buoyancy_force`](prism_render_architecture::particle::fluid::buoyancy_force),
//! [`blackbody_emission`](prism_render_architecture::particle::fluid::blackbody_emission),
//! [`heat_haze_distortion`](prism_render_architecture::particle::fluid::heat_haze_distortion))
//! and the vorticity estimate and confinement force
//! ([`NeighborVelocities::curl`](prism_render_architecture::particle::fluid::NeighborVelocities::curl),
//! [`vorticity_confinement_force`](prism_render_architecture::particle::fluid::vorticity_confinement_force)).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each routine threads through multiplies, adds, guarded divisions and at most
//! one `sqrt` (the confinement normalize), so `CPU` and `GPU` are not bit-exact:
//! a `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`) on every continuous quantity. Every fixture is placed
//! clear of every branch boundary: the `CFL` cell size and `stable_timestep`
//! max speed stay well above the degenerate guard, the trilinear fractions stay
//! in the open interior of `[0, 1]`, the combustion temperature stays far from
//! both the ignition threshold and the ambient point, and the black-body ramp
//! stays below every channel-clamp knee. The fixtures use no transcendental
//! method (only algebraic `sqrt` through the reused vector math), and the random
//! batch draws from a host-side integer `LCG` so it needs no external math
//! library.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::fluid::{
    CombustionParams, CombustionState, EmissionParams, NeighborScalars, NeighborVelocities,
    VorticityParams,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::fluid::cpu_reference;
use prism_volumetric_gpu::{FluidQuery, FluidResult, GpuContext, GpuFluid};

/// Absolute parity bound on every continuous quantity. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes (hot combustion
/// temperatures) where a few units in the last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn approx(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Returns whether two vectors agree component-wise within the parity bound.
fn approx_vec(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

/// Returns whether two combustion states agree channel for channel within the
/// parity bound.
fn states_match(a: &CombustionState, b: &CombustionState) -> bool {
    approx(a.temperature, b.temperature) && approx(a.fuel, b.fuel) && approx(a.smoke, b.smoke)
}

/// Returns whether a `GPU` result matches the `CPU` reference, matching variant
/// against variant and applying the continuous-field bound per lane.
fn results_match(got: &FluidResult, want: &FluidResult) -> bool {
    match (got, want) {
        (FluidResult::Scalar(a), FluidResult::Scalar(b)) => approx(*a, *b),
        (FluidResult::Vector(a), FluidResult::Vector(b)) => approx_vec(*a, *b),
        (FluidResult::Weights(a), FluidResult::Weights(b)) => {
            a.iter().zip(b.iter()).all(|(x, y)| approx(*x, *y))
        }
        (FluidResult::Combustion(a), FluidResult::Combustion(b)) => states_match(a, b),
        _ => false,
    }
}

/// Evaluates a single query on device and asserts it matches the `CPU` golden.
fn check(gpu: &GpuFluid, ctx: &GpuContext, query: FluidQuery) {
    let want = cpu_reference(&query);
    let got = gpu.evaluate(ctx, std::slice::from_ref(&query));
    assert_eq!(got.len(), 1, "one query yields one result");
    assert!(
        results_match(&got[0], &want),
        "GPU result {:?} must match CPU {:?} for {:?}",
        got[0],
        want,
        query
    );
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

/// Draws a uniform `f32` in `[lo, hi)` from the generator.
fn rand_range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Draws a point whose components lie in `[-2, 2)`.
fn rand_point(state: &mut u64) -> Vec3 {
    Vec3::new(
        rand_range(state, -2.0, 2.0),
        rand_range(state, -2.0, 2.0),
        rand_range(state, -2.0, 2.0),
    )
}

/// Draws a direction whose squared length clears `0.1`, so the confinement
/// normalize is well conditioned and never hits the flat-region guard.
fn rand_dir(state: &mut u64) -> Vec3 {
    loop {
        let v = Vec3::new(
            rand_range(state, -1.0, 1.0),
            rand_range(state, -1.0, 1.0),
            rand_range(state, -1.0, 1.0),
        );
        if v.length_squared() > 0.1 {
            return v;
        }
    }
}

/// Builds the six axis-aligned velocity neighbors from random samples.
fn rand_neighbor_velocities(state: &mut u64) -> NeighborVelocities {
    NeighborVelocities {
        x_plus: rand_point(state),
        x_minus: rand_point(state),
        y_plus: rand_point(state),
        y_minus: rand_point(state),
        z_plus: rand_point(state),
        z_minus: rand_point(state),
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFluid::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn cfl_scalars_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFluid::new(&ctx);
    // Cell size well above the degenerate guard: a plain divide.
    check(
        &gpu,
        &ctx,
        FluidQuery::CflNumber {
            max_velocity: 3.5,
            dt: 0.02,
            cell_size: 0.5,
        },
    );
    // Zero cell size exercises the guard: both sides return exactly zero.
    check(
        &gpu,
        &ctx,
        FluidQuery::CflNumber {
            max_velocity: 3.5,
            dt: 0.02,
            cell_size: 0.0,
        },
    );
    // Max speed well above the guard: a plain divide.
    check(
        &gpu,
        &ctx,
        FluidQuery::StableTimestep {
            max_velocity: 4.0,
            cell_size: 0.5,
            cfl_target: 0.9,
        },
    );
    // Zero max speed exercises the guard: nothing moving, zero step.
    check(
        &gpu,
        &ctx,
        FluidQuery::StableTimestep {
            max_velocity: 0.0,
            cell_size: 0.5,
            cfl_target: 0.9,
        },
    );
}

#[test]
fn advection_updates_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFluid::new(&ctx);
    check(
        &gpu,
        &ctx,
        FluidQuery::SemiLagrangianBacktrace {
            pos: Vec3::new(1.0, 2.0, -0.5),
            velocity: Vec3::new(0.5, -1.0, 2.0),
            dt: 0.1,
        },
    );
    check(
        &gpu,
        &ctx,
        FluidQuery::MaccormackCorrected {
            forward: Vec3::new(0.3, 0.6, 0.9),
            original: Vec3::new(0.2, 0.5, 1.0),
            back_advected: Vec3::new(0.25, 0.55, 0.95),
        },
    );
    check(
        &gpu,
        &ctx,
        FluidQuery::AdvectParticle {
            pos: Vec3::new(-1.0, 0.5, 3.0),
            velocity: Vec3::new(2.0, -0.5, 1.0),
            dt: 0.05,
        },
    );
}

#[test]
fn trilinear_kernel_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFluid::new(&ctx);
    // Fractions in the open interior of the cell, away from either face.
    check(
        &gpu,
        &ctx,
        FluidQuery::TrilinearWeights {
            frac: Vec3::new(0.3, 0.6, 0.4),
        },
    );
    let corners = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    let weights = [0.1, 0.05, 0.2, 0.15, 0.1, 0.05, 0.2, 0.15];
    check(&gpu, &ctx, FluidQuery::TrilinearSample { corners, weights });
}

#[test]
fn central_difference_operators_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFluid::new(&ctx);
    check(
        &gpu,
        &ctx,
        FluidQuery::CentralGradient {
            neighbors: NeighborScalars {
                x_plus: 2.0,
                x_minus: 1.0,
                y_plus: 3.0,
                y_minus: -1.0,
                z_plus: 0.5,
                z_minus: -0.5,
            },
            inv_2h: 2.0,
        },
    );
    check(
        &gpu,
        &ctx,
        FluidQuery::SubtractPressureGradient {
            velocity: Vec3::new(1.5, -0.5, 2.0),
            pressure_gradient: Vec3::new(0.5, 0.5, -1.0),
        },
    );
}

#[test]
fn combustion_step_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFluid::new(&ctx);
    let params = CombustionParams {
        ignition_temperature: 500.0,
        burn_rate: 2.0,
        smoke_yield: 0.5,
        heat_yield: 10.0,
        cooling_rate: 0.1,
        ambient_temperature: 300.0,
        buoyancy_alpha: 0.05,
        buoyancy_beta: 0.02,
    };
    // Ignited: temperature far above ignition, burned volume (0.2) far below the
    // remaining fuel (5.0), so neither branch sits on its boundary.
    check(
        &gpu,
        &ctx,
        FluidQuery::StepCombustion {
            state: CombustionState::new(800.0, 5.0, 1.0),
            params,
            dt: 0.1,
        },
    );
    // Cold: temperature far below ignition, so only linear cooling runs.
    check(
        &gpu,
        &ctx,
        FluidQuery::StepCombustion {
            state: CombustionState::new(350.0, 5.0, 1.0),
            params,
            dt: 0.1,
        },
    );
}

#[test]
fn combustion_forces_and_colour_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFluid::new(&ctx);
    let params = CombustionParams {
        ignition_temperature: 500.0,
        burn_rate: 2.0,
        smoke_yield: 0.5,
        heat_yield: 10.0,
        cooling_rate: 0.1,
        ambient_temperature: 300.0,
        buoyancy_alpha: 0.05,
        buoyancy_beta: 0.02,
    };
    check(
        &gpu,
        &ctx,
        FluidQuery::BuoyancyForce {
            state: CombustionState::new(800.0, 5.0, 1.0),
            params,
        },
    );
    // Temperature at `t = 0.4` of the ramp: every channel stays below its clamp
    // knee (`t*1.6 = 0.64 < 1`), so no clamp branch is on its boundary.
    check(
        &gpu,
        &ctx,
        FluidQuery::BlackbodyEmission {
            temperature: 700.0,
            params: EmissionParams {
                low_temperature: 300.0,
                white_temperature: 1300.0,
                intensity_scale: 2.0,
            },
        },
    );
    // Degenerate ramp (zero span) exercises the guard: emission is exactly zero.
    check(
        &gpu,
        &ctx,
        FluidQuery::BlackbodyEmission {
            temperature: 700.0,
            params: EmissionParams {
                low_temperature: 500.0,
                white_temperature: 500.0,
                intensity_scale: 2.0,
            },
        },
    );
    check(
        &gpu,
        &ctx,
        FluidQuery::HeatHazeDistortion {
            temperature_gradient: Vec3::new(0.5, -1.0, 0.25),
            strength: 0.3,
        },
    );
}

#[test]
fn vorticity_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFluid::new(&ctx);
    check(
        &gpu,
        &ctx,
        FluidQuery::Curl {
            neighbors: NeighborVelocities {
                x_plus: Vec3::new(0.0, 1.0, 0.5),
                x_minus: Vec3::new(0.0, -1.0, 0.5),
                y_plus: Vec3::new(0.5, 0.0, 1.0),
                y_minus: Vec3::new(-0.5, 0.0, 1.0),
                z_plus: Vec3::new(1.0, 0.5, 0.0),
                z_minus: Vec3::new(1.0, -0.5, 0.0),
            },
            inv_2h: 2.0,
        },
    );
    // Clear gradient length, well above the normalize guard: a real force.
    check(
        &gpu,
        &ctx,
        FluidQuery::VorticityConfinementForce {
            curl: Vec3::new(0.2, -0.4, 0.6),
            magnitude_gradient: Vec3::new(1.0, 2.0, -1.0),
            params: VorticityParams { epsilon: 0.3 },
            cell_size: 0.5,
        },
    );
    // Flat vorticity region: the zero gradient hits the normalize guard and the
    // force is exactly zero.
    check(
        &gpu,
        &ctx,
        FluidQuery::VorticityConfinementForce {
            curl: Vec3::new(0.2, -0.4, 0.6),
            magnitude_gradient: Vec3::ZERO,
            params: VorticityParams { epsilon: 0.3 },
            cell_size: 0.5,
        },
    );
}

/// Appends one query per twinned routine to `queries`, drawing inputs from the
/// generator and steering combustion through the `ignited` branch when asked so
/// the batch exercises both combustion paths.
fn push_suite(state: &mut u64, ignited: bool, queries: &mut Vec<FluidQuery>) {
    // Cell size and max speed well above the degenerate guards.
    queries.push(FluidQuery::CflNumber {
        max_velocity: rand_range(state, 0.5, 5.0),
        dt: rand_range(state, 0.005, 0.05),
        cell_size: rand_range(state, 0.25, 1.0),
    });
    queries.push(FluidQuery::StableTimestep {
        max_velocity: rand_range(state, 0.5, 5.0),
        cell_size: rand_range(state, 0.25, 1.0),
        cfl_target: rand_range(state, 0.5, 1.0),
    });
    queries.push(FluidQuery::SemiLagrangianBacktrace {
        pos: rand_point(state),
        velocity: rand_point(state),
        dt: rand_range(state, 0.005, 0.05),
    });
    queries.push(FluidQuery::MaccormackCorrected {
        forward: rand_point(state),
        original: rand_point(state),
        back_advected: rand_point(state),
    });
    queries.push(FluidQuery::AdvectParticle {
        pos: rand_point(state),
        velocity: rand_point(state),
        dt: rand_range(state, 0.005, 0.05),
    });
    // Fractions in the open interior of the cell.
    queries.push(FluidQuery::TrilinearWeights {
        frac: Vec3::new(
            rand_range(state, 0.1, 0.9),
            rand_range(state, 0.1, 0.9),
            rand_range(state, 0.1, 0.9),
        ),
    });
    let corners = [
        rand_point(state),
        rand_point(state),
        rand_point(state),
        rand_point(state),
        rand_point(state),
        rand_point(state),
        rand_point(state),
        rand_point(state),
    ];
    let weights = [
        rand_range(state, 0.0, 1.0),
        rand_range(state, 0.0, 1.0),
        rand_range(state, 0.0, 1.0),
        rand_range(state, 0.0, 1.0),
        rand_range(state, 0.0, 1.0),
        rand_range(state, 0.0, 1.0),
        rand_range(state, 0.0, 1.0),
        rand_range(state, 0.0, 1.0),
    ];
    queries.push(FluidQuery::TrilinearSample { corners, weights });
    queries.push(FluidQuery::CentralGradient {
        neighbors: NeighborScalars {
            x_plus: rand_range(state, -2.0, 2.0),
            x_minus: rand_range(state, -2.0, 2.0),
            y_plus: rand_range(state, -2.0, 2.0),
            y_minus: rand_range(state, -2.0, 2.0),
            z_plus: rand_range(state, -2.0, 2.0),
            z_minus: rand_range(state, -2.0, 2.0),
        },
        inv_2h: rand_range(state, 0.5, 4.0),
    });
    queries.push(FluidQuery::SubtractPressureGradient {
        velocity: rand_point(state),
        pressure_gradient: rand_point(state),
    });
    let params = CombustionParams {
        ignition_temperature: 500.0,
        burn_rate: rand_range(state, 1.0, 3.0),
        smoke_yield: rand_range(state, 0.3, 0.7),
        heat_yield: rand_range(state, 5.0, 15.0),
        cooling_rate: rand_range(state, 0.05, 0.2),
        ambient_temperature: 300.0,
        buoyancy_alpha: rand_range(state, 0.02, 0.08),
        buoyancy_beta: rand_range(state, 0.01, 0.04),
    };
    // Ignited temperatures stay far above the ignition threshold; cold ones far
    // below. Fuel stays well above the per-step burn so the clamp is not on its
    // boundary.
    let temperature = if ignited {
        rand_range(state, 700.0, 1000.0)
    } else {
        rand_range(state, 320.0, 420.0)
    };
    let combustion_state = CombustionState::new(
        temperature,
        rand_range(state, 4.0, 8.0),
        rand_range(state, 0.5, 2.0),
    );
    queries.push(FluidQuery::StepCombustion {
        state: combustion_state,
        params,
        dt: rand_range(state, 0.02, 0.08),
    });
    queries.push(FluidQuery::BuoyancyForce {
        state: combustion_state,
        params,
    });
    // Temperature at `t` in `[0.1, 0.5]` of the ramp keeps every channel below
    // its clamp knee.
    let low = 300.0;
    let span = 1000.0;
    let t = rand_range(state, 0.1, 0.5);
    queries.push(FluidQuery::BlackbodyEmission {
        temperature: low + t * span,
        params: EmissionParams {
            low_temperature: low,
            white_temperature: low + span,
            intensity_scale: rand_range(state, 0.5, 2.0),
        },
    });
    queries.push(FluidQuery::HeatHazeDistortion {
        temperature_gradient: rand_point(state),
        strength: rand_range(state, 0.1, 0.5),
    });
    queries.push(FluidQuery::Curl {
        neighbors: rand_neighbor_velocities(state),
        inv_2h: rand_range(state, 0.5, 4.0),
    });
    // Gradient length clears the normalize guard by construction.
    queries.push(FluidQuery::VorticityConfinementForce {
        curl: rand_point(state),
        magnitude_gradient: rand_dir(state),
        params: VorticityParams {
            epsilon: rand_range(state, 0.1, 0.5),
        },
        cell_size: rand_range(state, 0.25, 1.0),
    });
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFluid::new(&ctx);
    let mut state = 0x_f1d1_0f0f_51c3_0001_u64;
    let mut queries = Vec::new();
    for round in 0..24 {
        push_suite(&mut state, round % 2 == 0, &mut queries);
    }
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");

    let mut saw_ignited = false;
    let mut saw_cold = false;
    for (q, g) in queries.iter().zip(got.iter()) {
        let want = cpu_reference(q);
        if let FluidQuery::StepCombustion { state, params, .. } = q {
            if state.temperature >= params.ignition_temperature {
                saw_ignited = true;
            } else {
                saw_cold = true;
            }
        }
        assert!(
            results_match(g, &want),
            "GPU result {g:?} must match CPU {want:?} for {q:?}"
        );
    }
    assert!(
        saw_ignited && saw_cold,
        "the batch must exercise both combustion branches"
    );
}
