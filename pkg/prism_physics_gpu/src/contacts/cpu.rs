//! The `CPU` golden twin of the `GPU` one-sided contact solver.
//!
//! [`cpu_resolve_contacts`] advances a [`ParticleState`] under a set of
//! [`ContactConstraint`]s by one frame step using the same substep `XPBD` scheme
//! as the distance solver — predict, reset multipliers, sweep the colours,
//! recover velocities — but with the projection replaced by the *one-sided*
//! non-penetration rule (see [`project`]). It performs the identical
//! floating-point arithmetic, in the identical order, as
//! `shaders/contacts_resolve.wgsl`, so a passing real-device parity test is
//! direct evidence the ported kernel computes the same trajectory as this
//! reference.
//!
//! The constraints are projected in *colour order* (reusing the distance
//! solver's [`Colouring`]) for exactly the same reason: same-colour constraints
//! share no particle, so the device's one-colour-per-dispatch sweep and this
//! sequential loop touch each particle in the same, unambiguous way.
//!
//! # Scope
//!
//! This is a standalone contact solver: it resolves only the contacts passed to
//! it and does not co-solve distance constraints. Unifying the two constraint
//! families into a single interleaved sweep is deliberately left to a later
//! stage so the proven distance solver stays untouched. The contact set itself
//! is fixed for the whole step — normals and penetration are re-derived from the
//! live positions on every projection, but no new contacts are detected
//! mid-step; that is the standard per-frame detection limit.
//!
//! Provenance: substep `XPBD` with the canonical one-sided contact projection
//! (Müller et al.). No Unreal Engine source or derived code.

use glam::Vec3;

use crate::xpbd::{Colouring, ParticleState, XpbdConfig, XpbdError};

use super::constraint::ContactConstraint;

/// Machine epsilon for `f32`, matching the distance solver's degenerate-length
/// guard so both engines treat near-coincident particles identically.
const EPSILON: f32 = f32::EPSILON;

/// Advances `state` under the non-penetration `contacts` by `dt` seconds using
/// colour-ordered substep `XPBD`.
///
/// This is the reference the `GPU` kernel is validated against. It mutates
/// `state` in place; positions and velocities carry forward to the next call.
///
/// # Errors
///
/// Returns [`XpbdError`] when the config or state is invalid, a contact indexes
/// a missing particle, or the contact graph needs more colours than supported.
/// Does nothing (returns `Ok`) when there are no particles or `dt` is
/// non-positive.
pub fn cpu_resolve_contacts(
    state: &mut ParticleState,
    contacts: &[ContactConstraint],
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
    let colouring = Colouring::build(contacts, particle_count)?;
    let ordered = colouring.reorder(contacts);

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

/// Projects one contact constraint, accumulating its (non-negative) multiplier.
///
/// The two differences from the bidirectional distance projection are the whole
/// of the contact model:
///
/// * **Separated pairs are skipped.** When `c = length - rest >= 0` the spheres
///   are not overlapping, so the inequality is already satisfied and the pair is
///   left untouched — a contact never acts at a distance.
/// * **The multiplier is clamped to be non-negative.** After the usual
///   `delta_lambda` update the running multiplier is floored at `0` and only the
///   *clamped* increment is applied to the positions, so the accumulated
///   impulse can push the pair apart but never pull it together across
///   iterations.
fn project(
    con: &ContactConstraint,
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
    let c = length - con.rest;
    if c >= 0.0 {
        return;
    }
    let normal = delta / length;
    let alpha_tilde = con.compliance / (h * h);
    let delta_lambda = (-c - alpha_tilde * *lambda) / (w_sum + alpha_tilde);
    let new_lambda = (*lambda + delta_lambda).max(0.0);
    let applied = new_lambda - *lambda;
    *lambda = new_lambda;
    let correction = normal * applied;
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

    fn state_pair(xa: f32, xb: f32, wa: f32, wb: f32) -> ParticleState {
        let mut state = ParticleState::new();
        state.push(Vec3::new(xa, 0.0, 0.0), wa);
        state.push(Vec3::new(xb, 0.0, 0.0), wb);
        state
    }

    #[test]
    fn overlapping_pair_is_pushed_to_rest() {
        // Two unit-radius spheres overlapping (centres 1.0 apart, rest 2.0)
        // with no gravity must be pushed out to at least the rest separation.
        let mut state = state_pair(0.0, 1.0, 1.0, 1.0);
        let cons = vec![ContactConstraint::new(0, 1, 2.0, 0.0)];
        let config = XpbdConfig::new(Vec3::ZERO, 1, 20, 0.0);
        cpu_resolve_contacts(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        let sep = (state.positions[0] - state.positions[1]).length();
        assert!(sep >= 2.0 - 1e-3, "separation was {sep}");
    }

    #[test]
    fn separated_pair_is_left_untouched() {
        // Centres 3.0 apart, rest 2.0: already separated, so with no gravity the
        // solver must be a no-op (the one-sided constraint never pulls them in).
        let mut state = state_pair(0.0, 3.0, 1.0, 1.0);
        let before = state.clone();
        let cons = vec![ContactConstraint::new(0, 1, 2.0, 0.0)];
        let config = XpbdConfig::new(Vec3::ZERO, 1, 20, 0.0);
        cpu_resolve_contacts(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        for i in 0..state.len() {
            assert!((state.positions[i] - before.positions[i]).length() < 1e-6);
        }
    }

    #[test]
    fn pinned_particle_never_moves_and_takes_the_whole_push() {
        // Particle 0 pinned (w = 0). All of the separation correction must land
        // on the movable particle 1, and the pin must not drift.
        let mut state = state_pair(0.0, 1.0, 0.0, 1.0);
        let cons = vec![ContactConstraint::new(0, 1, 2.0, 0.0)];
        let config = XpbdConfig::new(Vec3::ZERO, 1, 20, 0.0);
        cpu_resolve_contacts(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[0], Vec3::ZERO);
        let sep = (state.positions[0] - state.positions[1]).length();
        assert!(sep >= 2.0 - 1e-3, "separation was {sep}");
    }

    #[test]
    fn deep_penetration_is_resolved() {
        // Nearly coincident centres (0.1 apart) with a large rest of 2.0: the
        // solver must still separate the pair well past the coincident guard.
        let mut state = state_pair(0.0, 0.1, 1.0, 1.0);
        let cons = vec![ContactConstraint::new(0, 1, 2.0, 0.0)];
        let config = XpbdConfig::new(Vec3::ZERO, 1, 40, 0.0);
        cpu_resolve_contacts(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        let sep = (state.positions[0] - state.positions[1]).length();
        assert!(sep > 1.0, "separation was {sep}");
        assert!(state.positions[0].is_finite());
        assert!(state.positions[1].is_finite());
    }

    #[test]
    fn empty_or_nonpositive_dt_is_a_noop() {
        let mut state = state_pair(0.0, 1.0, 1.0, 1.0);
        let before = state.clone();
        let cons = vec![ContactConstraint::new(0, 1, 2.0, 0.0)];
        cpu_resolve_contacts(&mut state, &cons, &XpbdConfig::default(), 0.0).unwrap();
        assert_eq!(state, before);
    }

    #[test]
    fn no_contacts_is_pure_free_fall() {
        // With no contacts the solver reduces to symplectic-Euler free fall,
        // matching a hand-rolled substep integration.
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 1.0);
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let config = XpbdConfig::new(gravity, 4, 1, 0.0);
        let dt = 1.0 / 60.0;
        cpu_resolve_contacts(&mut state, &[], &config, dt).unwrap();

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
}
