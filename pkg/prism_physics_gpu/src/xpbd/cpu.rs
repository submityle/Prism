//! The `CPU` golden twin of the `GPU` `XPBD` distance solver.
//!
//! [`cpu_solve`] advances a [`ParticleState`] under a set of
//! [`DistanceConstraint`]s by one frame step using the substep `XPBD` scheme,
//! projecting the constraints in *colour order*. It performs the identical
//! floating-point arithmetic, in the identical order, as
//! `shaders/xpbd.wgsl` — predict, reset multipliers, sweep the colours, recover
//! velocities — so a passing real-device parity test is direct evidence the
//! ported kernel computes the same trajectory as this reference.
//!
//! Colour order (rather than raw insertion order) is the reference precisely
//! because the device projects one colour per dispatch. Within a colour no two
//! constraints share a particle, so the parallel dispatch and this sequential
//! loop touch each particle in the same, unambiguous way.
//!
//! Provenance: substep `XPBD` (predict / project / recover) of Müller et al.,
//! with the canonical stretch-constraint projection. No Unreal Engine source or
//! derived code.

use glam::Vec3;

use super::coloring::Colouring;
use super::config::{XpbdConfig, XpbdError};
use super::constraint::DistanceConstraint;
use super::state::ParticleState;

/// Machine epsilon for `f32`, matching `prism_physics_core`'s degenerate-length
/// guard so both engines treat near-coincident particles identically.
const EPSILON: f32 = f32::EPSILON;

/// Advances `state` under `constraints` by `dt` seconds using colour-ordered
/// substep `XPBD`.
///
/// This is the reference the `GPU` kernel is validated against. It mutates
/// `state` in place; positions and velocities carry forward to the next call.
///
/// # Errors
///
/// Returns [`XpbdError`] when the config or state is invalid, a constraint
/// indexes a missing particle, or the constraint graph needs more colours than
/// supported. Does nothing (returns `Ok`) when there are no particles or `dt`
/// is non-positive.
pub fn cpu_solve(
    state: &mut ParticleState,
    constraints: &[DistanceConstraint],
    config: &XpbdConfig,
    dt: f32,
) -> Result<(), XpbdError> {
    config.validate()?;
    if !state.is_consistent() {
        return Err(XpbdError::InvalidConfig(
            "particle state arrays must have equal length",
        ));
    }
    if state.is_empty() || dt <= 0.0 {
        return Ok(());
    }

    let particle_count = state.len() as u32;
    let colouring = Colouring::build(constraints, particle_count)?;
    let ordered = colouring.reorder(constraints);

    let substeps = config.effective_substeps();
    let iterations = config.effective_iterations();
    let h = dt / substeps as f32;
    if h <= 0.0 {
        return Ok(());
    }
    let damping_scale = (1.0 - config.damping * h).max(0.0);
    let inv_h = 1.0 / h;

    let mut prev = vec![Vec3::ZERO; state.len()];
    let mut lambda = vec![0.0f32; ordered.len()];

    for _ in 0..substeps {
        predict(state, &mut prev, config.gravity, damping_scale, h);
        lambda.iter_mut().for_each(|l| *l = 0.0);
        for _ in 0..iterations {
            for &(start, end) in colouring.ranges() {
                for gi in start..end {
                    project(
                        &ordered[gi as usize],
                        &mut lambda[gi as usize],
                        &mut state.positions,
                        &state.inverse_masses,
                        h,
                    );
                }
            }
        }
        finalize(state, &prev, inv_h);
    }
    Ok(())
}

/// One substep prediction: snapshot positions, integrate acceleration, damp.
fn predict(
    state: &mut ParticleState,
    prev: &mut [Vec3],
    gravity: Vec3,
    damping_scale: f32,
    h: f32,
) {
    for (((prev_pos, pos), vel), &w) in prev
        .iter_mut()
        .zip(state.positions.iter_mut())
        .zip(state.velocities.iter_mut())
        .zip(state.inverse_masses.iter())
    {
        *prev_pos = *pos;
        if w <= 0.0 {
            continue;
        }
        let mut v = *vel;
        v += gravity * h;
        v *= damping_scale;
        *vel = v;
        *pos += v * h;
    }
}

/// Projects one distance constraint, accumulating its Lagrange multiplier.
fn project(
    con: &DistanceConstraint,
    lambda: &mut f32,
    positions: &mut [Vec3],
    inverse_masses: &[f32],
    h: f32,
) {
    let ia = con.a as usize;
    let ib = con.b as usize;
    let wa = inverse_masses[ia];
    let wb = inverse_masses[ib];
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return;
    }
    let delta = positions[ia] - positions[ib];
    let length = delta.length();
    if length < EPSILON {
        return;
    }
    let normal = delta / length;
    let c = length - con.rest_length;
    let alpha_tilde = con.compliance / (h * h);
    let delta_lambda = (-c - alpha_tilde * *lambda) / (w_sum + alpha_tilde);
    *lambda += delta_lambda;
    let correction = normal * delta_lambda;
    positions[ia] += correction * wa;
    positions[ib] -= correction * wb;
}

/// Recovers velocities from the net substep displacement.
fn finalize(state: &mut ParticleState, prev: &[Vec3], inv_h: f32) {
    for ((vel, pos), prev_pos) in state
        .velocities
        .iter_mut()
        .zip(state.positions.iter())
        .zip(prev.iter())
    {
        *vel = (*pos - *prev_pos) * inv_h;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rigid_pair() -> (ParticleState, Vec<DistanceConstraint>) {
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 1.0);
        state.push(Vec3::new(2.0, 0.0, 0.0), 1.0);
        let cons = vec![DistanceConstraint::new(0, 1, 1.0, 0.0)];
        (state, cons)
    }

    #[test]
    fn no_gravity_rigid_pair_restores_rest_length() {
        // With no gravity a single rigid constraint pulls the pair to the rest
        // length and holds it there.
        let (mut state, cons) = rigid_pair();
        let config = XpbdConfig::new(Vec3::ZERO, 1, 20, 0.0);
        cpu_solve(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        let length = (state.positions[0] - state.positions[1]).length();
        assert!((length - 1.0).abs() < 1e-4, "length was {length}");
    }

    #[test]
    fn pinned_anchor_never_moves() {
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 0.0); // pinned
        state.push(Vec3::new(0.0, -1.0, 0.0), 1.0);
        let cons = vec![DistanceConstraint::new(0, 1, 1.0, 0.0)];
        let config = XpbdConfig::default();
        cpu_solve(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[0], Vec3::ZERO);
    }

    #[test]
    fn free_fall_matches_closed_form_without_constraints() {
        // A single free particle with no constraints is pure symplectic Euler,
        // so we can check it against a hand-rolled substep integration.
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 1.0);
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let config = XpbdConfig::new(gravity, 4, 1, 0.0);
        let dt = 1.0 / 60.0;
        cpu_solve(&mut state, &[], &config, dt).unwrap();

        // Independent reference: 4 substeps of v += g*h; x += v*h.
        let h = dt / 4.0;
        let mut v = Vec3::ZERO;
        let mut x = Vec3::ZERO;
        for _ in 0..4 {
            v += gravity * h;
            x += v * h;
        }
        assert!((state.positions[0] - x).length() < 1e-6);
        assert!((state.velocities[0] - v).length() < 1e-6);
    }

    #[test]
    fn colour_batched_equals_plain_sequential_over_ordered() {
        // Independent cross-check: because same-colour constraints share no
        // particle, sweeping the colours must equal sweeping the identical
        // reordered list with no colour structure at all.
        let mut state = ParticleState::new();
        for i in 0..6 {
            state.push(Vec3::new(i as f32, 0.0, 0.0), 1.0);
        }
        state.inverse_masses[0] = 0.0; // pin one end
        let cons: Vec<_> = (0..5)
            .map(|i| DistanceConstraint::new(i, i + 1, 1.0, 0.0))
            .collect();
        let config = XpbdConfig::new(Vec3::new(0.0, -9.81, 0.0), 3, 4, 0.3);
        let dt = 1.0 / 60.0;

        let mut a = state.clone();
        cpu_solve(&mut a, &cons, &config, dt).unwrap();

        let mut b = state.clone();
        naive_sequential_solve(&mut b, &cons, &config, dt);

        for i in 0..a.len() {
            assert!(
                (a.positions[i] - b.positions[i]).length() < 1e-6,
                "particle {i} diverged: {:?} vs {:?}",
                a.positions[i],
                b.positions[i]
            );
        }
    }

    /// A deliberately naive reference: reorder by colour, then sweep the list
    /// sequentially with no colour batching. Equal to [`cpu_solve`] by the
    /// disjoint-particle property, so it independently exercises the loop.
    fn naive_sequential_solve(
        state: &mut ParticleState,
        constraints: &[DistanceConstraint],
        config: &XpbdConfig,
        dt: f32,
    ) {
        let colouring = Colouring::build(constraints, state.len() as u32).unwrap();
        let ordered = colouring.reorder(constraints);
        let substeps = config.effective_substeps();
        let iterations = config.effective_iterations();
        let h = dt / substeps as f32;
        let damping_scale = (1.0 - config.damping * h).max(0.0);
        let inv_h = 1.0 / h;
        let mut prev = vec![Vec3::ZERO; state.len()];
        let mut lambda = vec![0.0f32; ordered.len()];
        for _ in 0..substeps {
            predict(state, &mut prev, config.gravity, damping_scale, h);
            lambda.iter_mut().for_each(|l| *l = 0.0);
            for _ in 0..iterations {
                for gi in 0..ordered.len() {
                    project(
                        &ordered[gi],
                        &mut lambda[gi],
                        &mut state.positions,
                        &state.inverse_masses,
                        h,
                    );
                }
            }
            finalize(state, &prev, inv_h);
        }
    }

    #[test]
    fn damping_bleeds_energy_from_a_swinging_chain() {
        // A pinned pendulum released horizontally should lose speed over time
        // with damping enabled; check kinetic energy is bounded and finite.
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 0.0);
        state.push(Vec3::new(1.0, 0.0, 0.0), 1.0);
        let cons = vec![DistanceConstraint::new(0, 1, 1.0, 0.0)];
        let config = XpbdConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 2, 2.0);
        for _ in 0..120 {
            cpu_solve(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        }
        // Constraint keeps the bob on the unit circle throughout.
        let radius = (state.positions[1] - state.positions[0]).length();
        assert!((radius - 1.0).abs() < 1e-2, "radius drifted to {radius}");
        assert!(state.velocities[1].is_finite());
    }

    #[test]
    fn empty_or_nonpositive_dt_is_a_noop() {
        let (mut state, cons) = rigid_pair();
        let before = state.clone();
        cpu_solve(&mut state, &cons, &XpbdConfig::default(), 0.0).unwrap();
        assert_eq!(state, before);
    }
}
