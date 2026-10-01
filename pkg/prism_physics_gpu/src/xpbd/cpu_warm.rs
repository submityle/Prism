//! The `CPU` golden twin of the *warm-started* `GPU` `XPBD` distance solver.
//!
//! [`cpu_solve_warm`] advances a [`ParticleState`] under a set of
//! [`DistanceConstraint`]s by one frame step using the substep `XPBD` scheme,
//! projecting the constraints in *colour order* exactly as the cold
//! [`cpu_solve`](super::cpu_solve) does, but *warm-started* from a
//! [`DistanceCache`]: each constraint's Lagrange multiplier is seeded from the
//! previous frame's converged value and its matching position correction is
//! applied up front, so the iterative sweep below starts from the previous
//! frame's solution rather than from rest.
//!
//! The warm path differs from the cold [`cpu_solve`] in exactly two places,
//! each a byte-for-byte-intent twin of the device kernel
//! `shaders/xpbd_warm.wgsl`:
//!
//! 1. the per-substep multiplier reset (`lambda = 0`) is replaced by one
//!    [`apply_warm_start`] pass per colour, which seeds each multiplier and
//!    applies its cached correction before the projection sweeps; and
//! 2. after the final substep the converged multipliers are written back into
//!    the cache with [`DistanceCache::store`], pruning any pair that produced no
//!    constraint this frame.
//!
//! An empty `cache` seeds every constraint to `0`, so the first warmed frame is
//! identical to the cold solve — the keystone the parity test relies on.
//!
//! Provenance: substep `XPBD` (predict / warm-start / project / recover) of
//! Müller et al., with the canonical stretch-constraint projection and the
//! warm-starting of an iterative constraint solver (Müller et al. substep
//! `XPBD`). No Unreal Engine source or derived code.

use glam::Vec3;

use super::coloring::Colouring;
use super::config::{XpbdConfig, XpbdError};
use super::constraint::DistanceConstraint;
use super::cpu::{finalize, predict, project, EPSILON};
use super::state::ParticleState;
use super::warm_start::DistanceCache;

/// Advances `state` under `constraints` by `dt` seconds using colour-ordered
/// substep `XPBD`, warm-starting each constraint from `cache` and re-storing the
/// converged multipliers into it.
///
/// This is the reference the warm-started `GPU` kernel is validated against. It
/// mutates `state` in place and rebuilds `cache` from this frame's constraints;
/// positions, velocities, and the cache all carry forward to the next call.
///
/// # Errors
///
/// Returns [`XpbdError`] when the config or state is invalid, a constraint
/// indexes a missing particle, or the constraint graph needs more colours than
/// supported. Does nothing (returns `Ok`) when there are no particles or `dt`
/// is non-positive.
pub fn cpu_solve_warm(
    state: &mut ParticleState,
    constraints: &[DistanceConstraint],
    config: &XpbdConfig,
    dt: f32,
    cache: &mut DistanceCache,
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

    // The warm solver integrates every particle, so the awake mask is all true;
    // it reuses [`predict`]/[`finalize`] from the cold twin for a single,
    // parity-locked numeric path.
    let awake = vec![true; state.len()];

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

    // The per-constraint seed is read once (the cache does not change mid-frame)
    // and is aligned with `ordered`, so it is re-applied identically every
    // substep alongside that substep's fresh gravity prediction.
    let seed = cache.seed(&ordered);
    let mut prev = vec![Vec3::ZERO; state.len()];
    let mut lambda = vec![0.0f32; ordered.len()];

    for _ in 0..substeps {
        predict(state, &mut prev, config.gravity, damping_scale, h, &awake);
        // Warm-start: seed each multiplier and apply its cached correction in
        // colour order (same-colour constraints share no particle) so the sweep
        // below starts from the previous frame's solution.
        for &(start, end) in colouring.ranges() {
            for gi in start..end {
                apply_warm_start(
                    &ordered[gi as usize],
                    &mut lambda[gi as usize],
                    seed[gi as usize],
                    &mut state.positions,
                    &state.inverse_masses,
                );
            }
        }
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
        finalize(state, &prev, inv_h, &awake);
    }

    // Persist the converged multipliers (aligned with `ordered`) so next frame
    // seeds from them; pairs absent this frame are pruned by the rebuild.
    cache.store(&ordered, &lambda);
    Ok(())
}

/// Seeds one distance constraint's running multiplier from the cache and
/// applies the matching warm-start correction.
///
/// The running multiplier is set to `seed` and the same position correction the
/// projection would have accumulated — `normal * seed`, split by inverse mass —
/// is applied up front. Because the projection updates positions by the
/// *increment* in the multiplier, this pre-application is what makes the seeded
/// multiplier actually move the pair; without it the seed would be a no-op and
/// the "warm" start would be fake.
///
/// Unlike the one-sided contact warm-start, there is **no non-positive skip**: a
/// distance constraint is two-sided, so its multiplier may be negative (a
/// compressed pair pushing apart) and must still be applied. A seed of `0` (a
/// new constraint) therefore applies a zero correction and leaves positions
/// untouched, so a cold cache reproduces the cold solve exactly. The correction
/// uses the same degenerate-length guard and the same `normal * lambda`
/// arithmetic as [`project`], so the seeded state lies on the projection's own
/// trajectory.
fn apply_warm_start(
    con: &DistanceConstraint,
    lambda: &mut f32,
    seed: f32,
    positions: &mut [Vec3],
    inverse_masses: &[f32],
) {
    *lambda = seed;
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
    let correction = normal * seed;
    positions[ia] += correction * wa;
    positions[ib] -= correction * wb;
}

#[cfg(test)]
mod tests {
    use super::super::cpu::cpu_solve;
    use super::*;

    /// A pinned-at-both-ends compressed line: the interior particles are packed
    /// at 0.85 spacing (rest 1.0), so every adjacent pair stays in persistent
    /// compression and the cache fills, while the fixed ends keep the motion
    /// bounded.
    fn confined_line(n: u32) -> (ParticleState, Vec<DistanceConstraint>) {
        let mut state = ParticleState::new();
        for i in 0..n {
            let inv_mass = if i == 0 || i == n - 1 { 0.0 } else { 1.0 };
            state.push(Vec3::new(0.85 * i as f32, 0.0, 0.0), inv_mass);
        }
        let cons = (0..n - 1)
            .map(|i| DistanceConstraint::new(i, i + 1, 1.0, 1.0e-6))
            .collect();
        (state, cons)
    }

    #[test]
    fn warm_empty_cache_matches_cold_solve_exactly() {
        // The keystone parity: a fresh (empty) cache seeds every constraint to 0
        // and applies a zero correction, so the first warmed frame must equal
        // the cold solve bit-for-bit.
        let (initial, cons) = confined_line(8);
        let config = XpbdConfig::new(Vec3::new(0.0, -9.81, 0.0), 4, 8, 0.5);
        let dt = 1.0 / 60.0;

        let mut cold = initial.clone();
        cpu_solve(&mut cold, &cons, &config, dt).unwrap();

        let mut warm = initial.clone();
        let mut cache = DistanceCache::new();
        cpu_solve_warm(&mut warm, &cons, &config, dt, &mut cache).unwrap();

        for i in 0..cold.len() {
            assert_eq!(
                cold.positions[i], warm.positions[i],
                "particle {i} position: cold {:?} vs warm {:?}",
                cold.positions[i], warm.positions[i]
            );
            assert_eq!(
                cold.velocities[i], warm.velocities[i],
                "particle {i} velocity"
            );
        }
    }

    #[test]
    fn warm_caches_a_live_constraints_multiplier() {
        // A compressed line converges to a non-zero multiplier on each live
        // constraint; that value must land in the cache under the pair's key for
        // next frame, proving the store path runs.
        let (mut state, cons) = confined_line(8);
        let config = XpbdConfig::new(Vec3::ZERO, 4, 8, 0.0);
        let mut cache = DistanceCache::new();
        cpu_solve_warm(&mut state, &cons, &config, 1.0 / 60.0, &mut cache).unwrap();
        assert_eq!(cache.len(), cons.len());
        // At least one interior pair must carry a non-zero (here, compression)
        // multiplier.
        let any_live = cons.iter().any(|c| {
            cache
                .get(super::super::warm_start::DistanceKey::from(c))
                .abs()
                > 0.0
        });
        assert!(
            any_live,
            "a compressed line should cache a non-zero multiplier"
        );
    }

    #[test]
    fn warm_start_changes_the_trajectory_once_the_cache_fills() {
        // Frame one is identical to a cold solve (empty cache), but once the
        // cache holds the converged multipliers the seeded correction pre-moves
        // the pairs, so a warmed second frame must differ from a cold second
        // frame started from the same state.
        let (initial, cons) = confined_line(8);
        let config = XpbdConfig::new(Vec3::new(0.0, -9.81, 0.0), 2, 2, 0.1);
        let dt = 1.0 / 60.0;

        // Warm the cache with one frame, then capture the shared post-frame-one
        // state both continuations start from.
        let mut warm = initial.clone();
        let mut cache = DistanceCache::new();
        cpu_solve_warm(&mut warm, &cons, &config, dt, &mut cache).unwrap();
        assert!(!cache.is_empty());
        let after_frame_one = warm.clone();

        // Cold continuation: solve frame two from scratch.
        let mut cold_next = after_frame_one.clone();
        cpu_solve(&mut cold_next, &cons, &config, dt).unwrap();

        // Warm continuation: solve frame two seeded from the filled cache.
        let mut warm_next = after_frame_one.clone();
        cpu_solve_warm(&mut warm_next, &cons, &config, dt, &mut cache).unwrap();

        let diverged = (0..cold_next.len())
            .any(|i| (cold_next.positions[i] - warm_next.positions[i]).length() > 1e-9);
        assert!(
            diverged,
            "a filled cache must change the trajectory versus a cold restart"
        );
    }

    #[test]
    fn warm_cache_prunes_a_departed_constraint() {
        // Solve a live constraint (fills the cache), then a frame with no
        // constraints must empty it.
        let (mut state, cons) = confined_line(4);
        let config = XpbdConfig::new(Vec3::ZERO, 2, 4, 0.0);
        let mut cache = DistanceCache::new();
        cpu_solve_warm(&mut state, &cons, &config, 1.0 / 60.0, &mut cache).unwrap();
        assert!(!cache.is_empty());
        cpu_solve_warm(&mut state, &[], &config, 1.0 / 60.0, &mut cache).unwrap();
        assert!(cache.is_empty());
    }

    #[test]
    fn empty_or_nonpositive_dt_is_a_noop() {
        let (mut state, cons) = confined_line(4);
        let before = state.clone();
        let mut cache = DistanceCache::new();
        cpu_solve_warm(&mut state, &cons, &XpbdConfig::default(), 0.0, &mut cache).unwrap();
        assert_eq!(state, before);
        assert!(cache.is_empty());
    }
}
