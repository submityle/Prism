//! The `CPU` golden twin of the `GPU` one-sided contact solver.
//!
//! [`cpu_resolve_contacts`] advances a [`ParticleState`] under a set of
//! [`ContactConstraint`]s by one frame step using the same substep `XPBD` scheme
//! as the distance solver — predict, reset multipliers, sweep the colours,
//! recover velocities — but with the projection replaced by the *one-sided*
//! non-penetration rule (see [`project`]) and a trailing velocity-level
//! restitution pass (see [`apply_restitution`]). It performs the identical
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
//! and the substep velocity-level restitution of Müller et al. 2020. No Unreal
//! Engine source or derived code.

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
    let mut vel_pre = vec![Vec3::ZERO; state.len()];
    let mut lambda = vec![0.0f32; ordered.len()];

    for _ in 0..substeps {
        predict(state, &mut prev, config.gravity, damping_scale, h);
        // Snapshot the post-prediction (post-gravity) velocities: the
        // restitution pass restores the *pre-solve* approach speed, so it must
        // read the velocities as they were before the position solve altered
        // them through `finalize`.
        vel_pre.copy_from_slice(&state.velocities);
        lambda.iter_mut().for_each(|l| *l = 0.0);
        for _ in 0..iterations {
            for &(start, end) in colouring.ranges() {
                for gi in start..end {
                    project(
                        &ordered[gi as usize],
                        &mut lambda[gi as usize],
                        &mut state.positions,
                        &prev,
                        &state.inverse_masses,
                        h,
                    );
                }
            }
        }
        finalize(state, &prev, inv_h);
        // Velocity-level restitution, in colour order so no two updates touch a
        // shared particle's velocity at once (mirroring the projection sweep).
        for &(start, end) in colouring.ranges() {
            for gi in start..end {
                apply_restitution(
                    &ordered[gi as usize],
                    &state.positions,
                    &mut state.velocities,
                    &vel_pre,
                    &prev,
                    &state.inverse_masses,
                );
            }
        }
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

/// Projects one contact constraint, accumulating its (non-negative) multiplier
/// and then applying positional Coulomb friction.
///
/// The two differences from the bidirectional distance projection are the whole
/// of the *normal* contact model:
///
/// * **Separated pairs are skipped.** When `c = length - rest >= 0` the spheres
///   are not overlapping, so the inequality is already satisfied and the pair is
///   left untouched — a contact never acts at a distance.
/// * **The multiplier is clamped to be non-negative.** After the usual
///   `delta_lambda` update the running multiplier is floored at `0` and only the
///   *clamped* increment is applied to the positions, so the accumulated
///   impulse can push the pair apart but never pull it together across
///   iterations.
///
/// After the normal correction, [`apply_friction`] adds the tangential Coulomb
/// term (a no-op for a frictionless contact), completing the contact response.
fn project(
    con: &ContactConstraint,
    lambda: &mut f32,
    positions: &mut [Vec3],
    prev: &[Vec3],
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
    apply_friction(con, ia, ib, positions, prev, normal, -c, wa, wb, w_sum);
}

/// Applies the positional Coulomb friction correction for one contact.
///
/// Called immediately after the normal projection with the same `normal`
/// (`b` toward `a`) and the pre-projection penetration `penetration = -c > 0`.
/// It measures the tangential drift the pair accumulated *this substep* from the
/// `prev` snapshot, then either fully cancels it (static, while the drift stays
/// inside the cone `static_friction * penetration`) or clamps it to the dynamic
/// cone `dynamic_friction * penetration`. A frictionless contact (both
/// coefficients `0`) returns immediately, leaving the trajectory identical to
/// the pre-friction solver.
///
/// The correction is split by inverse-mass weight so it cancels the *relative*
/// tangential motion without moving the pair's centre of mass, exactly like the
/// normal projection; the shared `normal` means the correction is purely
/// tangential and never fights the non-penetration solve.
#[expect(
    clippy::too_many_arguments,
    reason = "the projection's live locals are threaded in rather than recomputed so the CPU twin matches the WGSL kernel arithmetic exactly"
)]
fn apply_friction(
    con: &ContactConstraint,
    ia: usize,
    ib: usize,
    positions: &mut [Vec3],
    prev: &[Vec3],
    normal: Vec3,
    penetration: f32,
    wa: f32,
    wb: f32,
    w_sum: f32,
) {
    if con.static_friction <= 0.0 && con.dynamic_friction <= 0.0 {
        return;
    }
    let da = positions[ia] - prev[ia];
    let db = positions[ib] - prev[ib];
    let relative = da - db;
    let normal_amount = relative.dot(normal);
    let tangent = relative - normal * normal_amount;
    let tangent_len = tangent.length();
    if tangent_len < EPSILON {
        return;
    }
    let scale = if tangent_len < con.static_friction * penetration {
        1.0
    } else {
        (con.dynamic_friction * penetration / tangent_len).min(1.0)
    };
    let correction = tangent * scale;
    positions[ia] -= correction * (wa / w_sum);
    positions[ib] += correction * (wb / w_sum);
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

/// Applies the velocity-level restitution correction for one contact.
///
/// Called once per substep *after* [`finalize`] has recovered the post-solve
/// velocities. Restitution restores a fraction of the pre-solve *approach*
/// speed along the contact normal as a post-solve *separating* speed, so a
/// struck pair rebounds rather than sticking: the target relative normal
/// velocity is `max(-restitution * v_n_pre, 0)`, where `v_n_pre` is the
/// relative normal velocity captured right after prediction (`vel_pre`). The
/// current relative normal velocity `v_n` is nudged to that target by a
/// mass-weighted impulse split along the same normal the projection used. The
/// `max(_, 0)` clamp means an already-separating pair (`v_n_pre >= 0`) is never
/// given extra energy, so the pass can only add a bounce, never inject it into
/// a resting stack.
///
/// Two guards keep the pass faithful to the position solve:
///
/// * **Inactive pairs are skipped.** Using the substep-start snapshot `prev`, a
///   pair that was already separated (`c_pre = |prev_a - prev_b| - rest >= 0`)
///   was never a live contact this substep, so it gets no impulse — matching
///   the projection's own separated-pair skip and leaving disjoint pairs
///   untouched.
/// * **An inelastic contact is a no-op.** A zero `restitution` returns
///   immediately, so the pre-restitution trajectory is preserved byte for byte.
///
/// The impulse is split by inverse-mass weight and applied with opposite signs
/// to `a` and `b`, so it changes only the *relative* normal velocity and leaves
/// the pair's momentum untouched, exactly like the normal projection.
fn apply_restitution(
    con: &ContactConstraint,
    positions: &[Vec3],
    velocities: &mut [Vec3],
    vel_pre: &[Vec3],
    prev: &[Vec3],
    inverse_masses: &[f32],
) {
    if con.restitution <= 0.0 {
        return;
    }
    let ia = con.a as usize;
    let ib = con.b as usize;
    let delta_pre = prev[ia] - prev[ib];
    let length_pre = delta_pre.length();
    let c_pre = length_pre - con.rest;
    if c_pre >= 0.0 {
        return;
    }
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
    let relative = velocities[ia] - velocities[ib];
    let v_n = relative.dot(normal);
    let relative_pre = vel_pre[ia] - vel_pre[ib];
    let v_n_pre = relative_pre.dot(normal);
    let target = (-con.restitution * v_n_pre).max(0.0);
    let dv = target - v_n;
    velocities[ia] += normal * dv * (wa / w_sum);
    velocities[ib] -= normal * dv * (wb / w_sum);
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

    /// A helper: one pinned particle at the origin and one overlapping movable
    /// particle offset along +x by `gap` (rest 1.0), the movable one carrying a
    /// purely tangential (+y) initial velocity. The contact normal is x, so all
    /// of the initial motion is tangential — the cleanest possible friction probe.
    fn tangential_pair(gap: f32, vy: f32) -> ParticleState {
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 0.0);
        state.push(Vec3::new(gap, 0.0, 0.0), 1.0);
        state.velocities[1] = Vec3::new(0.0, vy, 0.0);
        state
    }

    /// Runs one frame of a tangential-slide pair and returns the movable
    /// particle's recovered tangential (+y) velocity. The single pinned/​movable
    /// pair unavoidably couples a little tangential motion through the tilting
    /// normal, so the friction tests below compare *against the frictionless
    /// run of the identical scene* rather than an absolute target.
    fn slide_tangential_velocity(
        static_f: f32,
        dynamic_f: f32,
        gap: f32,
        vy: f32,
        substeps: u32,
    ) -> f32 {
        let mut state = tangential_pair(gap, vy);
        let cons = vec![ContactConstraint::new(0, 1, 1.0, 0.0).with_friction(static_f, dynamic_f)];
        let config = XpbdConfig::new(Vec3::ZERO, substeps, 8, 0.0);
        cpu_resolve_contacts(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        state.velocities[1].y
    }

    #[test]
    fn zero_friction_matches_new_constructor_exactly() {
        // A `with_friction(0.0, 0.0)` contact must be bit-for-bit the same
        // trajectory as a plain `new` contact — the friction path is a true
        // no-op at zero, guarding the pre-friction regression suite.
        let cons_plain = vec![ContactConstraint::new(0, 1, 1.0, 0.0)];
        let cons_zero = vec![ContactConstraint::new(0, 1, 1.0, 0.0).with_friction(0.0, 0.0)];
        let config = XpbdConfig::new(Vec3::ZERO, 2, 8, 0.0);
        let mut a = tangential_pair(0.85, 1.0);
        let mut b = tangential_pair(0.85, 1.0);
        cpu_resolve_contacts(&mut a, &cons_plain, &config, 1.0 / 60.0).unwrap();
        cpu_resolve_contacts(&mut b, &cons_zero, &config, 1.0 / 60.0).unwrap();
        assert_eq!(a.positions[1], b.positions[1]);
        assert_eq!(a.velocities[1], b.velocities[1]);
    }

    #[test]
    fn strong_static_friction_kills_most_tangential_motion() {
        // Against the frictionless baseline, a large static coefficient (the
        // substep drift stays well inside the stick cone) must cancel the bulk
        // of the tangential slide — only the small normal-coupling residual of
        // the single pinned pair survives.
        let free = slide_tangential_velocity(0.0, 0.0, 0.85, 1.0, 1);
        let stuck = slide_tangential_velocity(2.0, 2.0, 0.85, 1.0, 1);
        assert!(free > 0.9, "baseline slide unexpectedly small: {free}");
        assert!(
            stuck.abs() < 0.25 * free,
            "static friction left too much tangential motion: {stuck} vs baseline {free}"
        );
    }

    #[test]
    fn dynamic_friction_partially_damps_sliding() {
        // A moderate slide over a deep overlap (gap 0.5, so the pair stays in
        // contact through the frame) lands in the dynamic regime: friction
        // reduces the tangential velocity monotonically in the coefficient, but
        // a bounded cone never fully stops (nor reverses) the slide in one frame.
        let free = slide_tangential_velocity(0.0, 0.0, 0.5, 8.0, 1);
        let weak = slide_tangential_velocity(0.05, 0.05, 0.5, 8.0, 1);
        let strong = slide_tangential_velocity(0.2, 0.2, 0.5, 8.0, 1);
        assert!(weak < free, "weak friction did not damp: {weak} vs {free}");
        assert!(
            strong < weak,
            "stronger friction should damp more: strong {strong} vs weak {weak}"
        );
        assert!(
            strong > 0.0,
            "a bounded dynamic cone must not reverse the slide: {strong}"
        );
    }

    #[test]
    fn friction_never_moves_the_pinned_particle() {
        // The immovable half of a frictional contact must stay put.
        let mut state = tangential_pair(0.85, 5.0);
        let cons = vec![ContactConstraint::new(0, 1, 1.0, 0.0).with_friction(1.0, 1.0)];
        let config = XpbdConfig::new(Vec3::ZERO, 2, 8, 0.0);
        cpu_resolve_contacts(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[0], Vec3::ZERO);
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

    /// A pinned particle 0 at the origin and a movable particle 1 offset along
    /// +x by `gap` (rest 1.0), approaching head-on with normal velocity `vx`
    /// (negative toward the pin). The contact normal is the x-axis, so the
    /// whole interaction is a clean 1-D bounce.
    fn headon_pair(gap: f32, vx: f32) -> ParticleState {
        let mut state = ParticleState::new();
        state.push(Vec3::ZERO, 0.0);
        state.push(Vec3::new(gap, 0.0, 0.0), 1.0);
        state.velocities[1] = Vec3::new(vx, 0.0, 0.0);
        state
    }

    /// Runs one frame of a head-on bounce and returns the movable particle's
    /// recovered +x (separating) velocity. With particle 0 pinned the whole
    /// mass weight lands on particle 1, so the restitution pass sets its normal
    /// velocity exactly to the target `restitution * approach_speed`.
    fn bounce_velocity(restitution: f32, gap: f32, vx: f32) -> f32 {
        let mut state = headon_pair(gap, vx);
        let cons = vec![ContactConstraint::new(0, 1, 1.0, 0.0).with_restitution(restitution)];
        let config = XpbdConfig::new(Vec3::ZERO, 1, 8, 0.0);
        cpu_resolve_contacts(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        state.velocities[1].x
    }

    #[test]
    fn zero_restitution_matches_default_exactly() {
        // A `with_restitution(0.0)` contact must be bit-for-bit the same
        // trajectory as a plain `new` contact — the restitution path is a true
        // no-op at zero, guarding the pre-restitution regression suite.
        let cons_plain = vec![ContactConstraint::new(0, 1, 1.0, 0.0)];
        let cons_zero = vec![ContactConstraint::new(0, 1, 1.0, 0.0).with_restitution(0.0)];
        let config = XpbdConfig::new(Vec3::ZERO, 2, 8, 0.0);
        let mut a = headon_pair(0.9, -1.5);
        let mut b = headon_pair(0.9, -1.5);
        cpu_resolve_contacts(&mut a, &cons_plain, &config, 1.0 / 60.0).unwrap();
        cpu_resolve_contacts(&mut b, &cons_zero, &config, 1.0 / 60.0).unwrap();
        assert_eq!(a.positions[1], b.positions[1]);
        assert_eq!(a.velocities[1], b.velocities[1]);
    }

    #[test]
    fn full_restitution_restores_approach_speed() {
        // A head-on unit approach against a pin: with e = 1 the pair must leave
        // at essentially the incoming normal speed (energy conserved along the
        // normal), the position solve's over-separation corrected to the target.
        let elastic = bounce_velocity(1.0, 0.99, -1.0);
        assert!(
            (elastic - 1.0).abs() < 1e-2,
            "elastic rebound not ~approach speed: {elastic}"
        );
    }

    #[test]
    fn restitution_scales_rebound_monotonically() {
        // Among active coefficients the rebound speed grows with the
        // coefficient: each simply sets the target separation velocity to
        // `e * approach_speed`.
        let quarter = bounce_velocity(0.25, 0.99, -1.0);
        let half = bounce_velocity(0.5, 0.99, -1.0);
        let full = bounce_velocity(1.0, 0.99, -1.0);
        assert!(quarter > 0.0, "no rebound at all: {quarter}");
        assert!(
            half > quarter,
            "half not bouncier than quarter: {half} vs {quarter}"
        );
        assert!(full > half, "full not bounciest: {full} vs {half}");
    }

    #[test]
    fn separated_pair_never_bounces() {
        // Two particles far apart (rest 1.0, 5.0 apart) with e = 1 under
        // gravity: the activity gate (checked on the substep-start snapshot)
        // skips the pair every substep, so both fall exactly as free particles.
        let mut with_contact = ParticleState::new();
        with_contact.push(Vec3::ZERO, 1.0);
        with_contact.push(Vec3::new(5.0, 0.0, 0.0), 1.0);
        let mut free = with_contact.clone();
        let cons = vec![ContactConstraint::new(0, 1, 1.0, 0.0).with_restitution(1.0)];
        let g = XpbdConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 8, 0.0);
        cpu_resolve_contacts(&mut with_contact, &cons, &g, 1.0 / 60.0).unwrap();
        cpu_resolve_contacts(&mut free, &[], &g, 1.0 / 60.0).unwrap();
        for i in 0..free.len() {
            assert!((with_contact.positions[i] - free.positions[i]).length() < 1e-6);
            assert!((with_contact.velocities[i] - free.velocities[i]).length() < 1e-6);
        }
    }

    #[test]
    fn restitution_never_moves_the_pinned_particle() {
        // The immovable half of a bouncing contact must stay put in both
        // position and velocity.
        let mut state = headon_pair(0.9, -3.0);
        let cons = vec![ContactConstraint::new(0, 1, 1.0, 0.0).with_restitution(1.0)];
        let config = XpbdConfig::new(Vec3::ZERO, 2, 8, 0.0);
        cpu_resolve_contacts(&mut state, &cons, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[0], Vec3::ZERO);
        assert_eq!(state.velocities[0], Vec3::ZERO);
    }
}
