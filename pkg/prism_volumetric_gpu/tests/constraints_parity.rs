//! Real-device parity for the `XPBD` constraint-solve twin:
//! [`GpuConstraints`](prism_volumetric_gpu::constraints::GpuConstraints) must
//! reproduce the `CPU` golden
//! [`constraints`](prism_render_architecture::particle::constraints) across
//! every numeric term it exposes: the compliance regularizer
//! ([`effective_compliance`](prism_render_architecture::particle::constraints::effective_compliance),
//! [`Compliance::from_stiffness`](prism_render_architecture::particle::constraints::Compliance::from_stiffness),
//! [`Compliance::scaled_for_substep`](prism_render_architecture::particle::constraints::Compliance::scaled_for_substep),
//! [`Compliance::is_rigid`](prism_render_architecture::particle::constraints::Compliance::is_rigid)),
//! the substep schedule
//! ([`SubstepSchedule::substep_dt`](prism_render_architecture::particle::constraints::SubstepSchedule::substep_dt),
//! [`SubstepSchedule::effective_compliance`](prism_render_architecture::particle::constraints::SubstepSchedule::effective_compliance)),
//! the distance projection
//! ([`project_distance`](prism_render_architecture::particle::constraints::project_distance)),
//! the tearing test
//! ([`constraint_strain`](prism_render_architecture::particle::constraints::constraint_strain),
//! [`should_tear`](prism_render_architecture::particle::constraints::should_tear)),
//! the fracture rigid proxy
//! ([`rigid_from_particles`](prism_render_architecture::particle::constraints::rigid_from_particles)),
//! and the solver-tier matrix
//! ([`SolverSelection::select`](prism_render_architecture::particle::constraints::SolverSelection::select),
//! [`ConvergenceContract::for_tier`](prism_render_architecture::particle::constraints::ConvergenceContract::for_tier)).
//!
//! The fixtures exercise one dedicated query per term plus a randomized mixed
//! batch compared element for element. Every scalar is an interior value held
//! well away from its guard branch: substep timesteps and inverse masses are
//! comfortably positive, rest lengths are far from zero, strains sit clearly on
//! one side of the tear threshold, and the solver thresholds are crossed
//! decisively so no fixture lands on a classification tie.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each term is a rational function of its inputs with at most one `sqrt` (the
//! projection normal, the rigid proxy's distance), so `CPU` and `GPU` evaluate
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every
//! continuous `f32` lane; the boolean rigidity / tear flags and the discrete
//! solver tier are compared exactly.
//!
//! # Conditioning
//!
//! Every fixture is kept clear of the guard cracks: `substep_dt > 0` so no
//! compliance scale divides by zero, `substeps >= 1` so the schedule timestep is
//! finite, inverse masses are nonzero so the projection denominator stays well
//! above `EPS`, distance endpoints are non-coincident so the normal is defined,
//! rest lengths are `>= 0.5` so the strain denominator never floors, strains are
//! separated from the tear threshold on both sides, and the solver stiffness
//! ratio / contact density are driven decisively above or below their thresholds
//! so both tiers are exercised with no tie.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::constraints`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::constraints::{
    constraint_strain, effective_compliance, project_distance, rigid_from_particles, should_tear,
    Compliance, ConvergenceContract, SolverSelection, SolverTier, SubstepSchedule, TearThreshold,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::constraints::{
    ConstraintsQuery, ConstraintsResult, GpuConstraints, MAX_RIGID_PARTICLES,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes (such as a stiff
/// compliance scaled by a tiny squared substep) where a few units in the last
/// place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
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

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// A pseudo-random substep count in `1..=16`, never zero so the schedule
/// timestep stays finite.
fn ranged_substeps(state: &mut u64) -> u32 {
    1 + (lcg(state) * 16.0) as u32
}

/// Converts a packed component triple into a [`Vec3`].
fn v3(c: [f32; 3]) -> Vec3 {
    Vec3::new(c[0], c[1], c[2])
}

/// Recomputes the expected [`ConstraintsResult`] by calling the `CPU` golden
/// directly for `query`.
fn golden_result(query: &ConstraintsQuery) -> ConstraintsResult {
    match *query {
        ConstraintsQuery::EffectiveCompliance { alpha, substep_dt } => {
            ConstraintsResult::EffectiveCompliance {
                alpha_tilde: effective_compliance(alpha, substep_dt),
            }
        }
        ConstraintsQuery::ComplianceFromStiffness { stiffness } => {
            ConstraintsResult::ComplianceFromStiffness {
                alpha: Compliance::from_stiffness(stiffness).alpha,
            }
        }
        ConstraintsQuery::ComplianceScaledForSubstep { alpha, substep_dt } => {
            // The twin uses the raw alpha (no `Compliance::new` clamp), so the
            // reference mirrors it with a direct struct literal.
            ConstraintsResult::ComplianceScaledForSubstep {
                alpha_tilde: Compliance { alpha }.scaled_for_substep(substep_dt),
            }
        }
        ConstraintsQuery::ComplianceIsRigid { alpha } => ConstraintsResult::ComplianceIsRigid {
            is_rigid: Compliance { alpha }.is_rigid(),
        },
        ConstraintsQuery::SubstepDt { dt, substeps } => ConstraintsResult::SubstepDt {
            substep_dt: SubstepSchedule::new(dt, substeps, 1).substep_dt(),
        },
        ConstraintsQuery::ScheduleEffectiveCompliance {
            dt,
            substeps,
            alpha,
        } => ConstraintsResult::ScheduleEffectiveCompliance {
            alpha_tilde: SubstepSchedule::new(dt, substeps, 1)
                .effective_compliance(Compliance { alpha }),
        },
        ConstraintsQuery::ProjectDistance {
            pos_a,
            pos_b,
            inv_mass_a,
            inv_mass_b,
            rest,
            alpha_tilde,
            lambda,
        } => {
            let c = project_distance(
                v3(pos_a),
                v3(pos_b),
                inv_mass_a,
                inv_mass_b,
                rest,
                alpha_tilde,
                lambda,
            );
            ConstraintsResult::ProjectDistance {
                delta_a: [c.delta_a.x, c.delta_a.y, c.delta_a.z],
                delta_b: [c.delta_b.x, c.delta_b.y, c.delta_b.z],
                delta_lambda: c.delta_lambda,
            }
        }
        ConstraintsQuery::ConstraintStrain {
            rest,
            current_length,
        } => ConstraintsResult::ConstraintStrain {
            strain: constraint_strain(rest, current_length),
        },
        ConstraintsQuery::ShouldTear { strain, max_strain } => ConstraintsResult::ShouldTear {
            tears: should_tear(strain, TearThreshold::new(max_strain)),
        },
        ConstraintsQuery::RigidFromParticles {
            positions,
            inv_masses,
            count,
        } => {
            let n = count as usize;
            let pos: Vec<Vec3> = positions[..n].iter().map(|&p| v3(p)).collect();
            let r = rigid_from_particles(&pos, &inv_masses[..n]);
            ConstraintsResult::RigidFromParticles {
                inv_mass: r.inv_mass,
                center_of_mass: [r.center_of_mass.x, r.center_of_mass.y, r.center_of_mass.z],
                bounding_radius: r.bounding_radius,
            }
        }
        ConstraintsQuery::SolverSelect {
            stiff_ratio_threshold,
            contact_density_threshold,
            stiffness_ratio,
            contact_density,
            force_high_stability,
        } => {
            ConstraintsResult::SolverSelect {
                tier: SolverSelection::new(stiff_ratio_threshold, contact_density_threshold)
                    .select(stiffness_ratio, contact_density, force_high_stability),
            }
        }
        ConstraintsQuery::ConvergenceForTier { tier } => {
            let cc = ConvergenceContract::for_tier(tier);
            ConstraintsResult::ConvergenceForTier {
                unconditionally_stable: cc.unconditionally_stable,
                parallelizable: cc.parallelizable,
                cost_multiplier: cc.cost_multiplier,
            }
        }
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: each lane the
/// variant carries must agree within the parity bound (continuous lanes within
/// tolerance, boolean / tier lanes exactly).
fn pin(idx: usize, query: &ConstraintsQuery, got: &ConstraintsResult) {
    let want = golden_result(query);
    match (got, want) {
        (
            ConstraintsResult::EffectiveCompliance { alpha_tilde: g },
            ConstraintsResult::EffectiveCompliance { alpha_tilde: w },
        ) => {
            assert!(
                close(*g, w),
                "query {idx} effective_compliance: gpu {g} vs cpu {w}"
            );
        }
        (
            ConstraintsResult::ComplianceFromStiffness { alpha: g },
            ConstraintsResult::ComplianceFromStiffness { alpha: w },
        ) => {
            assert!(
                close(*g, w),
                "query {idx} from_stiffness: gpu {g} vs cpu {w}"
            );
        }
        (
            ConstraintsResult::ComplianceScaledForSubstep { alpha_tilde: g },
            ConstraintsResult::ComplianceScaledForSubstep { alpha_tilde: w },
        ) => {
            assert!(
                close(*g, w),
                "query {idx} scaled_for_substep: gpu {g} vs cpu {w}"
            );
        }
        (
            ConstraintsResult::ComplianceIsRigid { is_rigid: g },
            ConstraintsResult::ComplianceIsRigid { is_rigid: w },
        ) => {
            assert_eq!(*g, w, "query {idx} is_rigid: gpu {g} vs cpu {w}");
        }
        (
            ConstraintsResult::SubstepDt { substep_dt: g },
            ConstraintsResult::SubstepDt { substep_dt: w },
        ) => {
            assert!(close(*g, w), "query {idx} substep_dt: gpu {g} vs cpu {w}");
        }
        (
            ConstraintsResult::ScheduleEffectiveCompliance { alpha_tilde: g },
            ConstraintsResult::ScheduleEffectiveCompliance { alpha_tilde: w },
        ) => {
            assert!(
                close(*g, w),
                "query {idx} schedule effective_compliance: gpu {g} vs cpu {w}"
            );
        }
        (
            ConstraintsResult::ProjectDistance {
                delta_a: ga,
                delta_b: gb,
                delta_lambda: gl,
            },
            ConstraintsResult::ProjectDistance {
                delta_a: wa,
                delta_b: wb,
                delta_lambda: wl,
            },
        ) => {
            for k in 0..3 {
                assert!(
                    close(ga[k], wa[k]),
                    "query {idx} project delta_a[{k}]: gpu {} vs cpu {}",
                    ga[k],
                    wa[k]
                );
                assert!(
                    close(gb[k], wb[k]),
                    "query {idx} project delta_b[{k}]: gpu {} vs cpu {}",
                    gb[k],
                    wb[k]
                );
            }
            assert!(
                close(*gl, wl),
                "query {idx} project delta_lambda: gpu {gl} vs cpu {wl}"
            );
        }
        (
            ConstraintsResult::ConstraintStrain { strain: g },
            ConstraintsResult::ConstraintStrain { strain: w },
        ) => {
            assert!(close(*g, w), "query {idx} strain: gpu {g} vs cpu {w}");
        }
        (
            ConstraintsResult::ShouldTear { tears: g },
            ConstraintsResult::ShouldTear { tears: w },
        ) => {
            assert_eq!(*g, w, "query {idx} should_tear: gpu {g} vs cpu {w}");
        }
        (
            ConstraintsResult::RigidFromParticles {
                inv_mass: gm,
                center_of_mass: gc,
                bounding_radius: gr,
            },
            ConstraintsResult::RigidFromParticles {
                inv_mass: wm,
                center_of_mass: wc,
                bounding_radius: wr,
            },
        ) => {
            assert!(
                close(*gm, wm),
                "query {idx} rigid inv_mass: gpu {gm} vs cpu {wm}"
            );
            for k in 0..3 {
                assert!(
                    close(gc[k], wc[k]),
                    "query {idx} rigid com[{k}]: gpu {} vs cpu {}",
                    gc[k],
                    wc[k]
                );
            }
            assert!(
                close(*gr, wr),
                "query {idx} rigid bounding_radius: gpu {gr} vs cpu {wr}"
            );
        }
        (
            ConstraintsResult::SolverSelect { tier: g },
            ConstraintsResult::SolverSelect { tier: w },
        ) => {
            assert_eq!(*g, w, "query {idx} solver tier: gpu {g:?} vs cpu {w:?}");
        }
        (
            ConstraintsResult::ConvergenceForTier {
                unconditionally_stable: gs,
                parallelizable: gp,
                cost_multiplier: gcost,
            },
            ConstraintsResult::ConvergenceForTier {
                unconditionally_stable: ws,
                parallelizable: wp,
                cost_multiplier: wcost,
            },
        ) => {
            assert_eq!(
                *gs, ws,
                "query {idx} convergence unconditionally_stable: gpu {gs} vs cpu {ws}"
            );
            assert_eq!(
                *gp, wp,
                "query {idx} convergence parallelizable: gpu {gp} vs cpu {wp}"
            );
            assert!(
                close(*gcost, wcost),
                "query {idx} convergence cost_multiplier: gpu {gcost} vs cpu {wcost}"
            );
        }
        (g, w) => panic!("query {idx} result variant mismatch: gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuConstraints, queries: &[ConstraintsQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn effective_compliance_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // Interior positive compliances and comfortably positive substep timesteps,
    // so the squared dt stays well above the guard and no branch floors.
    let queries = vec![
        ConstraintsQuery::EffectiveCompliance {
            alpha: 0.01,
            substep_dt: 0.004,
        },
        ConstraintsQuery::EffectiveCompliance {
            alpha: 0.05,
            substep_dt: 0.008,
        },
        ConstraintsQuery::EffectiveCompliance {
            alpha: 0.002,
            substep_dt: 0.016,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn compliance_from_stiffness_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // Stiffnesses comfortably above the guard, so each inverts to a finite,
    // interior compliance rather than folding to the rigid `0` branch.
    let queries = vec![
        ConstraintsQuery::ComplianceFromStiffness { stiffness: 10.0 },
        ConstraintsQuery::ComplianceFromStiffness { stiffness: 125.0 },
        ConstraintsQuery::ComplianceFromStiffness { stiffness: 900.0 },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn compliance_scaled_for_substep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // Positive compliances with positive substep timesteps; the raw alpha is
    // used directly (no clamp), matched by a direct struct literal in the
    // reference.
    let queries = vec![
        ConstraintsQuery::ComplianceScaledForSubstep {
            alpha: 0.02,
            substep_dt: 0.005,
        },
        ConstraintsQuery::ComplianceScaledForSubstep {
            alpha: 0.08,
            substep_dt: 0.012,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn compliance_is_rigid_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // One clearly rigid (`alpha == 0`) and several clearly compliant alphas,
    // all held far from the `EPS = 1e-9` tie so the boolean is unambiguous.
    let queries = vec![
        ConstraintsQuery::ComplianceIsRigid { alpha: 0.0 },
        ConstraintsQuery::ComplianceIsRigid { alpha: 0.001 },
        ConstraintsQuery::ComplianceIsRigid { alpha: 0.5 },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn substep_dt_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // Nonzero substep counts, so the schedule divides rather than taking the
    // degenerate zero branch.
    let queries = vec![
        ConstraintsQuery::SubstepDt {
            dt: 0.016,
            substeps: 4,
        },
        ConstraintsQuery::SubstepDt {
            dt: 0.008,
            substeps: 8,
        },
        ConstraintsQuery::SubstepDt {
            dt: 0.02,
            substeps: 1,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn schedule_effective_compliance_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // Positive compliance, nonzero substeps and a finite frame timestep, so the
    // substep dt is finite and the scale divides.
    let queries = vec![
        ConstraintsQuery::ScheduleEffectiveCompliance {
            dt: 0.016,
            substeps: 4,
            alpha: 0.01,
        },
        ConstraintsQuery::ScheduleEffectiveCompliance {
            dt: 0.012,
            substeps: 6,
            alpha: 0.05,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn project_distance_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // Non-coincident endpoints (separation well above the squared-length guard)
    // and nonzero inverse masses, so the normal is defined and the projection
    // denominator stays comfortably above the guard.
    let queries = vec![
        ConstraintsQuery::ProjectDistance {
            pos_a: [0.0, 0.0, 0.0],
            pos_b: [1.2, 0.0, 0.0],
            inv_mass_a: 1.0,
            inv_mass_b: 1.0,
            rest: 1.0,
            alpha_tilde: 0.05,
            lambda: 0.1,
        },
        ConstraintsQuery::ProjectDistance {
            pos_a: [0.3, 0.4, 0.2],
            pos_b: [-0.5, 0.1, 0.9],
            inv_mass_a: 0.5,
            inv_mass_b: 1.5,
            rest: 0.8,
            alpha_tilde: 0.2,
            lambda: -0.05,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn constraint_strain_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // Rest lengths far above the guard floor so the denominator never floors;
    // current lengths both above and below rest give positive and negative
    // strains.
    let queries = vec![
        ConstraintsQuery::ConstraintStrain {
            rest: 1.0,
            current_length: 1.3,
        },
        ConstraintsQuery::ConstraintStrain {
            rest: 2.0,
            current_length: 1.6,
        },
        ConstraintsQuery::ConstraintStrain {
            rest: 0.5,
            current_length: 0.55,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn should_tear_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // Strains clearly on each side of the threshold, far from the tie, so the
    // boolean is unambiguous on both devices.
    let queries = vec![
        ConstraintsQuery::ShouldTear {
            strain: 0.1,
            max_strain: 0.3,
        },
        ConstraintsQuery::ShouldTear {
            strain: 0.5,
            max_strain: 0.3,
        },
        ConstraintsQuery::ShouldTear {
            strain: 0.05,
            max_strain: 0.2,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn rigid_from_particles_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // A spread-out cluster with nonzero inverse masses, so each mass inverts
    // finitely, the total mass clears the guard and the center of mass / radius
    // are well defined.
    let mut positions = [[0.0_f32; 3]; MAX_RIGID_PARTICLES];
    let mut inv_masses = [0.0_f32; MAX_RIGID_PARTICLES];
    let samples = [
        ([1.0, 0.0, 0.0], 1.0),
        ([-1.0, 0.0, 0.0], 1.0),
        ([0.0, 2.0, 0.0], 0.5),
        ([0.0, -1.0, 1.0], 1.5),
        ([0.5, 0.5, -0.5], 0.8),
    ];
    for (slot, &(p, _m)) in positions.iter_mut().zip(samples.iter()) {
        *slot = p;
    }
    for (slot, &(_, m)) in inv_masses.iter_mut().zip(samples.iter()) {
        *slot = m;
    }
    let queries = vec![
        ConstraintsQuery::RigidFromParticles {
            positions,
            inv_masses,
            count: 5,
        },
        ConstraintsQuery::RigidFromParticles {
            positions,
            inv_masses,
            count: 1,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn solver_select_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    // Thresholds crossed decisively in each direction (and one forced), so every
    // tier choice is unambiguous.
    let queries = vec![
        ConstraintsQuery::SolverSelect {
            stiff_ratio_threshold: 1000.0,
            contact_density_threshold: 8.0,
            stiffness_ratio: 100.0,
            contact_density: 2.0,
            force_high_stability: false,
        },
        ConstraintsQuery::SolverSelect {
            stiff_ratio_threshold: 1000.0,
            contact_density_threshold: 8.0,
            stiffness_ratio: 5000.0,
            contact_density: 2.0,
            force_high_stability: false,
        },
        ConstraintsQuery::SolverSelect {
            stiff_ratio_threshold: 1000.0,
            contact_density_threshold: 8.0,
            stiffness_ratio: 100.0,
            contact_density: 32.0,
            force_high_stability: false,
        },
        ConstraintsQuery::SolverSelect {
            stiff_ratio_threshold: 1000.0,
            contact_density_threshold: 8.0,
            stiffness_ratio: 100.0,
            contact_density: 2.0,
            force_high_stability: true,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn convergence_for_tier_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    let queries = vec![
        ConstraintsQuery::ConvergenceForTier {
            tier: SolverTier::Xpbd,
        },
        ConstraintsQuery::ConvergenceForTier {
            tier: SolverTier::Vbd,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn randomized_mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConstraints::new(&ctx);
    let mut state: u64 = 0x00c0_1157_7a1e_d00d_u64 ^ 0x9e37_79b9_7f4a_7c15;
    let mut queries = Vec::new();
    for _ in 0..16 {
        let alpha = ranged(&mut state, 0.001, 0.1);
        let substep_dt = ranged(&mut state, 0.002, 0.02);
        let substeps = ranged_substeps(&mut state);
        let dt = ranged(&mut state, 0.008, 0.02);
        queries.push(ConstraintsQuery::EffectiveCompliance { alpha, substep_dt });
        queries.push(ConstraintsQuery::ComplianceFromStiffness {
            stiffness: ranged(&mut state, 10.0, 1000.0),
        });
        queries.push(ConstraintsQuery::ComplianceScaledForSubstep { alpha, substep_dt });
        queries.push(ConstraintsQuery::ComplianceIsRigid {
            alpha: ranged(&mut state, 0.001, 0.5),
        });
        queries.push(ConstraintsQuery::SubstepDt { dt, substeps });
        queries.push(ConstraintsQuery::ScheduleEffectiveCompliance {
            dt,
            substeps,
            alpha,
        });
        // Non-coincident endpoints: `b` is offset by a comfortably large span so
        // the separation stays well above the squared-length guard.
        let pos_a = [
            signed(&mut state, 0.5),
            signed(&mut state, 0.5),
            signed(&mut state, 0.5),
        ];
        let pos_b = [
            pos_a[0] + ranged(&mut state, 0.5, 1.5),
            pos_a[1] + ranged(&mut state, 0.5, 1.5),
            pos_a[2] + ranged(&mut state, 0.5, 1.5),
        ];
        queries.push(ConstraintsQuery::ProjectDistance {
            pos_a,
            pos_b,
            inv_mass_a: ranged(&mut state, 0.5, 2.0),
            inv_mass_b: ranged(&mut state, 0.5, 2.0),
            rest: ranged(&mut state, 0.5, 1.5),
            alpha_tilde: ranged(&mut state, 0.01, 0.3),
            lambda: signed(&mut state, 0.2),
        });
        // Rest far from zero; current length kept positive and clearly offset.
        let rest = ranged(&mut state, 0.5, 2.0);
        queries.push(ConstraintsQuery::ConstraintStrain {
            rest,
            current_length: rest + signed(&mut state, 0.3),
        });
        // Strain and threshold separated by a decisive gap on a chosen side.
        let max_strain = ranged(&mut state, 0.2, 0.4);
        let tears = lcg(&mut state) < 0.5;
        let strain = if tears {
            max_strain + ranged(&mut state, 0.1, 0.3)
        } else {
            max_strain - ranged(&mut state, 0.1, 0.15)
        };
        queries.push(ConstraintsQuery::ShouldTear { strain, max_strain });
        // Rigid proxy over a spread-out cluster with nonzero inverse masses.
        let count = 1 + (lcg(&mut state) * (MAX_RIGID_PARTICLES as f32)) as u32;
        let count = count.min(MAX_RIGID_PARTICLES as u32);
        let mut positions = [[0.0_f32; 3]; MAX_RIGID_PARTICLES];
        let mut inv_masses = [0.0_f32; MAX_RIGID_PARTICLES];
        for k in 0..(count as usize) {
            positions[k] = [
                signed(&mut state, 1.5),
                signed(&mut state, 1.5),
                signed(&mut state, 1.5),
            ];
            inv_masses[k] = ranged(&mut state, 0.5, 2.0);
        }
        queries.push(ConstraintsQuery::RigidFromParticles {
            positions,
            inv_masses,
            count,
        });
        // Solver selection: drive the ratio / density decisively to one side.
        let want_vbd = lcg(&mut state) < 0.5;
        let (stiffness_ratio, contact_density) = if want_vbd {
            (
                ranged(&mut state, 2000.0, 5000.0),
                ranged(&mut state, 1.0, 4.0),
            )
        } else {
            (
                ranged(&mut state, 10.0, 500.0),
                ranged(&mut state, 1.0, 4.0),
            )
        };
        queries.push(ConstraintsQuery::SolverSelect {
            stiff_ratio_threshold: 1000.0,
            contact_density_threshold: 8.0,
            stiffness_ratio,
            contact_density,
            force_high_stability: false,
        });
        let tier = if lcg(&mut state) < 0.5 {
            SolverTier::Xpbd
        } else {
            SolverTier::Vbd
        };
        queries.push(ConstraintsQuery::ConvergenceForTier { tier });
    }
    check(&ctx, &gpu, &queries);
}
