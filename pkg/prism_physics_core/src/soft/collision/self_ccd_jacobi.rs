//! Parallel-safe (Jacobi) continuous self-collision for the soft-body kernel.
//!
//! [`super::self_ccd::resolve_self_ccd`] resolves each tunnelling pair *in
//! place* (Gauss-Seidel): a later pair sees an earlier pair's correction, so
//! the pass is inherently serial and its result depends on the pair visitation
//! order. That is exactly the dependency a GPU kernel cannot honour — thousands
//! of pair threads run concurrently and must each read a value no other thread
//! is simultaneously writing.
//!
//! This module is the parallel-safe twin, mirroring the relationship between
//! [`super::self_collision`] and
//! [`super::virtual_particles_jacobi`](super::virtual_particles_jacobi). It
//! reuses the *identical* deterministic broad phase
//! ([`super::self_ccd::collect_self_ccd_candidate_pairs`]) so the two resolvers
//! fold the same candidate set, but the narrow phase is restructured as a
//! classic Jacobi step:
//!
//! 1. **Freeze** a read-only snapshot of the frame-end positions and velocities.
//! 2. **Accumulate**: every candidate pair's time-of-impact resolution is
//!    computed *solely from the frozen snapshot* and written as a per-particle
//!    position delta and velocity delta into own-slot accumulators. Because no
//!    pair reads another pair's output, the pairs may be evaluated in any order
//!    — the GPU can run one thread per particle that sums only its own half of
//!    every incident pair.
//! 3. **Apply**: each particle advances by its accumulated delta once.
//!
//! For any particle that participates in exactly one resolved pair (the common
//! case once a sane `cell_size` keeps buckets small), a single Jacobi iteration
//! reproduces the Gauss-Seidel result bit-for-bit, because the sole delta is
//! `resolved_state - frozen_state`. Clusters where one particle is pulled by
//! several pairs at once differ from Gauss-Seidel — as every Jacobi scheme does
//! — but remain deterministic (the per-particle reduction follows the fixed
//! [`alloc::collections::BTreeSet`] pair order) and converge to the same
//! separated state under repeated iteration.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is a
//! mechanical Jacobi re-expression of the closed-form swept-pair TOI resolution
//! in [`super::self_ccd`], using the standard frozen-snapshot accumulate/apply
//! split shared with the virtual-particle Jacobi pass.

use glam::Vec3;

use crate::math::scalar::Real;

use super::self_ccd::{collect_self_ccd_candidate_pairs, swept_pair_toi, SelfCcdParams};

/// Numerical floor below which `dt` is treated as zero (no velocity recovery),
/// matching [`super::self_ccd`]'s `EPS_REL_MOTION`.
const EPS_DT: Real = 1e-12;

/// One pair's own-slot contribution, expressed as deltas against the frozen
/// snapshot so the two partners can be accumulated independently.
struct PairDelta {
    /// Index of the first particle.
    ia: usize,
    /// Index of the second particle.
    ib: usize,
    /// Position delta for `ia` (zero when `ia` is pinned).
    dp_a: Vec3,
    /// Position delta for `ib` (zero when `ib` is pinned).
    dp_b: Vec3,
    /// Velocity delta for `ia`, present only when the pair exchanges an impulse
    /// and `ia` is free.
    dv_a: Option<Vec3>,
    /// Velocity delta for `ib`, present only when the pair exchanges an impulse
    /// and `ib` is free.
    dv_b: Option<Vec3>,
}

/// Computes a single candidate pair's time-of-impact resolution from the frozen
/// snapshot, returning the per-partner position/velocity deltas, or [`None`]
/// when the pair is jointly immovable or never reaches `thickness`.
///
/// The math is identical to [`super::self_ccd::resolve_self_ccd`]'s narrow
/// phase; the only difference is that the resolved absolute state is returned as
/// a delta against the frozen `curr_*` / `vel_*` inputs so Jacobi accumulation
/// can sum several pairs per particle.
#[expect(
    clippy::too_many_arguments,
    reason = "the frozen snapshot of a pair is eight small Copy scalars/vectors; \
              bundling them into a struct would only move the argument list"
)]
fn pair_delta(
    ia: usize,
    ib: usize,
    prev_a: Vec3,
    curr_a: Vec3,
    prev_b: Vec3,
    curr_b: Vec3,
    vel_a: Vec3,
    vel_b: Vec3,
    wa: Real,
    wb: Real,
    thickness: Real,
    restitution: Real,
    inv_dt: Real,
) -> Option<PairDelta> {
    let wsum = wa + wb;
    if wsum <= 0.0 {
        return None;
    }
    let t = swept_pair_toi(prev_a, curr_a, prev_b, curr_b, thickness)?;

    // Positions at the time of impact.
    let a_c = prev_a + (curr_a - prev_a) * t;
    let b_c = prev_b + (curr_b - prev_b) * t;
    let delta = a_c - b_c;
    let unit = delta.normalize_or_zero();
    let normal = if unit.length_squared() > 0.0 {
        unit
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let penetration = (thickness - delta.length()).max(0.0);
    let inv_wsum = 1.0 / wsum;

    // Resolved absolute positions, snapped to the TOI contact then pushed
    // symmetrically apart by inverse mass. Expressed as deltas vs. the frozen
    // frame-end positions so multiple incident pairs can be summed.
    let pos_a = a_c + normal * (wa * inv_wsum * penetration);
    let pos_b = b_c - normal * (wb * inv_wsum * penetration);
    let dp_a = if wa > 0.0 { pos_a - curr_a } else { Vec3::ZERO };
    let dp_b = if wb > 0.0 { pos_b - curr_b } else { Vec3::ZERO };

    // Normal restitution impulse, recovered from the TOI approach velocity.
    let va_in = (a_c - prev_a) * inv_dt;
    let vb_in = (b_c - prev_b) * inv_dt;
    let vrel_n = (va_in - vb_in).dot(normal);
    let (dv_a, dv_b) = if vrel_n < 0.0 {
        let impulse = -(1.0 + restitution) * vrel_n * inv_wsum;
        let new_va = va_in + normal * (wa * impulse);
        let new_vb = vb_in - normal * (wb * impulse);
        (
            if wa > 0.0 { Some(new_va - vel_a) } else { None },
            if wb > 0.0 { Some(new_vb - vel_b) } else { None },
        )
    } else {
        (None, None)
    };

    Some(PairDelta {
        ia,
        ib,
        dp_a,
        dp_b,
        dv_a,
        dv_b,
    })
}

/// Runs one parallel-safe (Jacobi) continuous self-collision iteration.
///
/// The deterministic twin of [`super::self_ccd::resolve_self_ccd`]: it shares
/// the identical broad phase and per-pair time-of-impact math but reads every
/// pair's inputs from a frozen snapshot of `positions`/`velocities` and applies
/// the summed per-particle correction once at the end, so the result never
/// depends on pair evaluation order and maps directly onto a one-thread-per-pair
/// GPU kernel.
///
/// For a particle in a single resolved pair the outcome is identical to the
/// Gauss-Seidel core; clusters sharing a particle differ (as all Jacobi schemes
/// do) but stay deterministic and converge under repeated iteration.
///
/// A pinned particle (`inverse_mass <= 0`) is never written, a pair of two
/// pinned particles contributes nothing, and a disabled sweep, a non-positive
/// `thickness`, fewer than two particles, or a `prev_positions` /
/// `inverse_masses` slice shorter than `positions` is handled without
/// panicking. `velocities` is written only where the column is long enough.
pub fn resolve_self_ccd_jacobi(
    positions: &mut [Vec3],
    prev_positions: &[Vec3],
    velocities: &mut [Vec3],
    inverse_masses: &[Real],
    params: SelfCcdParams,
    dt: Real,
) {
    let params = params.sanitized();
    if !params.enabled || params.thickness <= 0.0 {
        return;
    }
    let count = positions
        .len()
        .min(prev_positions.len())
        .min(inverse_masses.len());
    if count < 2 {
        return;
    }

    let pairs = collect_self_ccd_candidate_pairs(
        positions,
        prev_positions,
        count,
        params.cell_size,
        params.thickness,
    );
    if pairs.is_empty() {
        return;
    }

    let inv_dt = if dt.abs() <= EPS_DT { 0.0 } else { 1.0 / dt };

    // Freeze the frame-end state: every pair reads from these snapshots, never
    // from the accumulators, which is what makes the pass order-independent.
    let frozen_pos = positions[..count].to_vec();
    let vel_len = velocities.len().min(count);
    let frozen_vel = velocities[..vel_len].to_vec();

    // Own-slot accumulators: `pos_delta[i]` / `vel_delta[i]` sum particle `i`'s
    // half of every incident pair. Iterating `pairs` in BTreeSet order fixes the
    // per-particle float reduction order, so the summation is deterministic.
    let mut pos_delta = alloc::vec![Vec3::ZERO; count];
    let mut vel_delta = alloc::vec![Vec3::ZERO; count];
    let mut vel_touched = alloc::vec![false; count];

    for &(i, j) in &pairs {
        let ia = i as usize;
        let ib = j as usize;
        let wa = inverse_masses[ia].max(0.0);
        let wb = inverse_masses[ib].max(0.0);
        let vel_a = frozen_vel.get(ia).copied().unwrap_or(Vec3::ZERO);
        let vel_b = frozen_vel.get(ib).copied().unwrap_or(Vec3::ZERO);
        let Some(pd) = pair_delta(
            ia,
            ib,
            prev_positions[ia],
            frozen_pos[ia],
            prev_positions[ib],
            frozen_pos[ib],
            vel_a,
            vel_b,
            wa,
            wb,
            params.thickness,
            params.restitution,
            inv_dt,
        ) else {
            continue;
        };

        pos_delta[pd.ia] += pd.dp_a;
        pos_delta[pd.ib] += pd.dp_b;
        if let Some(dv) = pd.dv_a {
            vel_delta[pd.ia] += dv;
            vel_touched[pd.ia] = true;
        }
        if let Some(dv) = pd.dv_b {
            vel_delta[pd.ib] += dv;
            vel_touched[pd.ib] = true;
        }
    }

    // Apply: advance each free particle by its summed delta exactly once.
    for i in 0..count {
        if inverse_masses[i].max(0.0) > 0.0 {
            positions[i] += pos_delta[i];
            if vel_touched[i] && i < velocities.len() {
                velocities[i] += vel_delta[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::self_ccd::resolve_self_ccd;
    use super::*;

    /// Enabled, no-bounce settings for a given `thickness`.
    fn params(thickness: Real) -> SelfCcdParams {
        SelfCcdParams::new(0.2, thickness)
    }

    #[test]
    fn single_pair_matches_gauss_seidel() {
        // A lone tunnelling pair touches no other particle, so one Jacobi
        // iteration must reproduce the Gauss-Seidel core bit-for-bit.
        let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let masses = [1.0, 1.0];
        let p = params(0.4);

        let mut gs_pos = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let mut gs_vel = [Vec3::new(2.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)];
        resolve_self_ccd(&mut gs_pos, &prev, &mut gs_vel, &masses, p, 1.0);

        let mut jac_pos = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let mut jac_vel = [Vec3::new(2.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)];
        resolve_self_ccd_jacobi(&mut jac_pos, &prev, &mut jac_vel, &masses, p, 1.0);

        assert_eq!(jac_pos, gs_pos, "position must match Gauss-Seidel");
        assert_eq!(jac_vel, gs_vel, "velocity must match Gauss-Seidel");
    }

    #[test]
    fn disjoint_pairs_match_gauss_seidel() {
        // Two independent tunnelling pairs far enough apart to never share a
        // cell: no particle is pulled by more than one pair, so Jacobi equals
        // Gauss-Seidel for the whole batch.
        let build = || {
            (
                [
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(-1.0, 0.0, 0.0),
                    Vec3::new(1.0, 50.0, 0.0),
                    Vec3::new(-1.0, 50.0, 0.0),
                ],
                [
                    Vec3::new(-1.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(-1.0, 50.0, 0.0),
                    Vec3::new(1.0, 50.0, 0.0),
                ],
            )
        };
        let masses = [1.0, 1.0, 1.0, 1.0];
        let p = params(0.4);

        let (mut gs_pos, prev) = build();
        let mut gs_vel = [Vec3::ZERO; 4];
        resolve_self_ccd(&mut gs_pos, &prev, &mut gs_vel, &masses, p, 1.0);

        let (mut jac_pos, _) = build();
        let mut jac_vel = [Vec3::ZERO; 4];
        resolve_self_ccd_jacobi(&mut jac_pos, &prev, &mut jac_vel, &masses, p, 1.0);

        assert_eq!(jac_pos, gs_pos);
        assert_eq!(jac_vel, gs_vel);
    }

    #[test]
    fn resolves_a_full_tunnel_through() {
        let thickness = 0.4;
        let mut positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let mut velocities = [Vec3::new(2.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)];
        let masses = [1.0, 1.0];
        resolve_self_ccd_jacobi(
            &mut positions,
            &prev,
            &mut velocities,
            &masses,
            params(thickness),
            1.0,
        );

        let separation = positions[0].distance(positions[1]);
        assert!(
            (separation - thickness).abs() < 1e-4,
            "post-CCD separation {separation} must equal thickness {thickness}"
        );
        assert!(
            positions[0].x <= positions[1].x + 1e-4,
            "particle 0 tunnelled past particle 1"
        );
        let vrel_n = (velocities[0] - velocities[1]).dot(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            vrel_n >= -1e-4,
            "relative normal velocity {vrel_n} must not still be closing"
        );
    }

    #[test]
    fn pinned_partner_stays_put() {
        let thickness = 0.5;
        let pin_pos = Vec3::new(0.0, 0.0, 0.0);
        let mut positions = [pin_pos, Vec3::new(-2.0, 0.0, 0.0)];
        let prev = [pin_pos, Vec3::new(2.0, 0.0, 0.0)];
        let mut velocities = [Vec3::ZERO, Vec3::ZERO];
        let masses = [0.0, 1.0];
        resolve_self_ccd_jacobi(
            &mut positions,
            &prev,
            &mut velocities,
            &masses,
            params(thickness),
            1.0,
        );

        assert_eq!(positions[0], pin_pos, "pinned partner moved");
        assert_eq!(velocities[0], Vec3::ZERO, "pinned partner gained velocity");
        let separation = positions[1].distance(pin_pos);
        assert!(
            (separation - thickness).abs() < 1e-4,
            "free particle must sit exactly `thickness` from the pin, got {separation}"
        );
    }

    #[test]
    fn two_pinned_partners_are_a_no_op() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.1, 0.0, 0.0);
        let mut positions = [a, b];
        let prev = [a, b];
        let mut velocities = [Vec3::ZERO, Vec3::ZERO];
        let masses = [0.0, 0.0];
        resolve_self_ccd_jacobi(&mut positions, &prev, &mut velocities, &masses, params(0.5), 1.0);
        assert_eq!(positions[0], a);
        assert_eq!(positions[1], b);
    }

    #[test]
    fn disabled_sweep_is_a_no_op() {
        let mut positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let before = positions;
        let mut velocities = [Vec3::ZERO, Vec3::ZERO];
        let masses = [1.0, 1.0];
        let mut p = params(0.4);
        p.enabled = false;
        resolve_self_ccd_jacobi(&mut positions, &prev, &mut velocities, &masses, p, 1.0);
        assert_eq!(positions, before);
    }

    #[test]
    fn too_few_particles_is_a_no_op() {
        let mut positions = [Vec3::new(0.0, 0.0, 0.0)];
        let prev = [Vec3::new(0.0, 0.0, 0.0)];
        let mut velocities = [Vec3::ZERO];
        let masses = [1.0];
        resolve_self_ccd_jacobi(&mut positions, &prev, &mut velocities, &masses, params(0.5), 1.0);
        assert_eq!(positions[0], Vec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn short_prev_slice_does_not_panic() {
        let mut positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.1, 0.0, 0.0)];
        let prev = [Vec3::new(0.0, 0.0, 0.0)]; // one short -> count == 1
        let mut velocities = [Vec3::ZERO, Vec3::ZERO];
        let masses = [1.0, 1.0];
        resolve_self_ccd_jacobi(&mut positions, &prev, &mut velocities, &masses, params(0.5), 1.0);
        assert_eq!(positions[1], Vec3::new(0.1, 0.0, 0.0));
    }

    #[test]
    fn coincident_pair_is_finite_and_separated() {
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let mut positions = [origin, origin];
        let prev = [origin, origin];
        let mut velocities = [Vec3::ZERO, Vec3::ZERO];
        let masses = [1.0, 1.0];
        resolve_self_ccd_jacobi(&mut positions, &prev, &mut velocities, &masses, params(0.5), 1.0);
        for p in &positions {
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }
        let separation = positions[0].distance(positions[1]);
        assert!((separation - 0.5).abs() < 1e-4, "got {separation}");
    }

    #[test]
    fn resolution_is_deterministic() {
        let build = || {
            (
                [
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(-1.0, 0.1, 0.0),
                    Vec3::new(0.0, 2.0, 0.0),
                ],
                [
                    Vec3::new(-1.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.1, 0.0),
                    Vec3::new(0.0, 2.0, 0.0),
                ],
            )
        };
        let masses = [1.0, 1.0, 1.0];
        let (mut pa, prev_a) = build();
        let (mut pb, prev_b) = build();
        let mut va = [Vec3::ZERO; 3];
        let mut vb = [Vec3::ZERO; 3];
        resolve_self_ccd_jacobi(&mut pa, &prev_a, &mut va, &masses, params(0.4), 1.0);
        resolve_self_ccd_jacobi(&mut pb, &prev_b, &mut vb, &masses, params(0.4), 1.0);
        assert_eq!(pa, pb);
        assert_eq!(va, vb);
    }

    #[test]
    fn restitution_controls_rebound_speed() {
        let sep_speed = |restitution: Real| -> Real {
            let mut positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
            let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
            let mut velocities = [Vec3::new(2.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)];
            let masses = [1.0, 1.0];
            let mut p = params(0.4);
            p.restitution = restitution;
            resolve_self_ccd_jacobi(&mut positions, &prev, &mut velocities, &masses, p, 1.0);
            let axis = (positions[0] - positions[1]).normalize_or_zero();
            (velocities[0] - velocities[1]).dot(axis)
        };
        let inelastic = sep_speed(0.0);
        let bouncy = sep_speed(1.0);
        assert!(
            bouncy > inelastic + 1e-3,
            "restitution 1 ({bouncy}) must separate faster than 0 ({inelastic})"
        );
    }

    #[test]
    fn dt_zero_leaves_velocities_untouched() {
        let mut positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let mut velocities = [Vec3::new(5.0, 0.0, 0.0), Vec3::new(-5.0, 0.0, 0.0)];
        let before = velocities;
        let masses = [1.0, 1.0];
        resolve_self_ccd_jacobi(&mut positions, &prev, &mut velocities, &masses, params(0.4), 0.0);
        // inv_dt == 0 => va_in == vb_in == 0 => vrel_n == 0, not < 0, no impulse.
        assert_eq!(velocities, before, "dt=0 must not touch velocities");
    }
}
