//! Real-device parity for the flocking / `Boids` steering twin:
//! [`GpuBoids`](prism_volumetric_gpu::boids::GpuBoids) must reproduce the `CPU`
//! golden [`boids`](prism_render_architecture::particle::boids) across every
//! twinned per-boid answer — the normalized [`separation_force`], [`alignment_force`]
//! and [`cohesion_force`] directions, the [`goal_force`] direction, the
//! fixed-order [`combine_forces`] sum, the [`limit_turn`] result, the
//! force-clamped [`steer`] acceleration, the speed-clamped [`integrate`]
//! velocity, the dedicated [`clamp_length`] / [`limit_turn`] probes and the
//! integer [`boid_cell`] lattice coordinate.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each boid is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and a handful of `sqrt`-guarded normalizations, so `CPU` and `GPU` evaluate
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every `f32` field, while
//! the integer `cell` is compared exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from a branch tie: the randomized
//! batch uses large kinematic limits so the turn / force / speed clamps are a
//! no-op on both devices (same branch), while dedicated tests drive the clamp
//! branches with clear margins. Positions carry a fixed fractional offset so
//! `floor(position / cell_size)` is unambiguous and the `cell` match is exact,
//! velocities are plainly nonzero so headings are well-defined, and neighbor
//! lists stay below [`MAX_NEIGHBORS`](prism_volumetric_gpu::boids::MAX_NEIGHBORS)
//! so no truncation diverges. All vectors stay far above the squared-length
//! epsilon so each normalization lands on the same side of its guard.
//!
//! The fixture `RNG` is a host-side `u64` linear-congruential generator; it uses
//! only integer and divide arithmetic (no transcendental, no external math
//! library), so the inputs are themselves bit-reproducible.
//!
//! Provenance: twinned from this repository's
//! [`boids`](prism_render_architecture::particle::boids); the three-rule
//! flocking model is Reynolds' classic steering behavior re-derived at the
//! algorithm level, with no third-party engine source or derived code.

use prism_render_architecture::particle::boids::{
    alignment_force, boid_cell, clamp_length, cohesion_force, combine_forces, goal_force,
    integrate, limit_turn, separation_force, steer, BoidsLimits, BoidsWeights,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::boids::{BoidsQuery, BoidsResult, GpuBoids, MAX_NEIGHBORS};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
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

/// Asserts two vectors agree channel-for-channel within the parity bound.
fn close_vec(label: &str, idx: usize, got: Vec3, want: Vec3) {
    assert!(
        close(got.x, want.x) && close(got.y, want.y) && close(got.z, want.z),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.x,
        got.y,
        got.z,
        want.x,
        want.y,
        want.z
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

/// A `BoidsQuery` seeded with large, no-op kinematic limits and clearly
/// conditioned dedicated probes, so a caller overrides only the fields a given
/// fixture exercises. The default probes land plainly inside their branches:
/// `clamp_input` is short enough to pass through unchanged and `turn_steer` has
/// a perpendicular component well above `turn_max`.
fn base_query(self_index: u32, neighbors: Vec<u32>) -> BoidsQuery {
    BoidsQuery {
        self_index,
        neighbors,
        goal: None,
        weights: BoidsWeights::classic_flock(),
        limits: BoidsLimits::new(100.0, -1.0, 100.0, 100.0, 100.0),
        cell_size: 1.0,
        dt: 0.5,
        clamp_input: Vec3::new(0.3, 0.0, 0.0),
        clamp_max_len: 1.0,
        turn_steer: Vec3::new(1.0, 5.0, 0.0),
        turn_velocity: Vec3::new(2.0, 0.0, 0.0),
        turn_max: 0.5,
    }
}

/// A small, well-conditioned three-boid scene used by the probe-focused tests:
/// distinct lattice positions with a fixed fractional offset and plainly
/// nonzero velocities.
fn small_scene() -> (Vec<Vec3>, Vec<Vec3>) {
    let positions = vec![
        Vec3::new(0.37, 0.41, 0.29),
        Vec3::new(1.37, 0.41, 0.29),
        Vec3::new(0.37, 1.41, 0.29),
    ];
    let velocities = vec![
        Vec3::new(0.11, 0.02, 0.0),
        Vec3::new(0.5, 0.1, 0.0),
        Vec3::new(0.1, 0.5, 0.0),
    ];
    (positions, velocities)
}

/// A distinct-position, nonzero-velocity pool on an integer lattice with a
/// fixed fractional offset, so every pairwise distance is at least one unit
/// (far above the squared-length epsilon) and every `cell` is unambiguous.
fn flock_pool(len: usize) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut positions = Vec::with_capacity(len);
    let mut velocities = Vec::with_capacity(len);
    for i in 0..len {
        let ix = (i % 7) as i32 - 3;
        let iy = ((i / 7) % 5) as i32 - 2;
        let iz = (i % 3) as i32 - 1;
        positions.push(Vec3::new(
            ix as f32 + 0.37,
            iy as f32 + 0.41,
            iz as f32 + 0.29,
        ));
        velocities.push(Vec3::new(
            iz as f32 * 0.5 + 0.11,
            ix as f32 * 0.5 + 0.13,
            iy as f32 * 0.5 + 0.17,
        ));
    }
    (positions, velocities)
}

/// Picks a small set of distinct neighbor indices (never the boid itself) from
/// the pool, drawn from the integer `RNG`. Count stays well below
/// [`MAX_NEIGHBORS`] so the host never truncates a list.
fn pick_neighbors(state: &mut u64, pool_len: usize, self_index: usize) -> Vec<u32> {
    let count = 2 + (lcg(state) * 4.0) as usize;
    let mut out: Vec<u32> = Vec::new();
    while out.len() < count {
        let raw = (lcg(state) * pool_len as f32) as usize;
        let n = raw.min(pool_len - 1);
        if n != self_index && !out.contains(&(n as u32)) {
            out.push(n as u32);
        }
    }
    out
}

/// Pins one `GPU` result against the `CPU` golden for `query`: every twinned
/// steering value is recomputed by the reference and compared within bound, and
/// the integer `cell` is compared exactly. The neighbor slice is truncated to
/// [`MAX_NEIGHBORS`] to mirror the kernel's fixed-stride consumption.
fn pin(idx: usize, positions: &[Vec3], velocities: &[Vec3], query: &BoidsQuery, got: &BoidsResult) {
    let si = query.self_index as usize;
    let used = query.neighbors.len().min(MAX_NEIGHBORS as usize);
    let neighbors = &query.neighbors[..used];

    let want_sep = separation_force(si, positions, neighbors);
    let want_align = alignment_force(velocities, neighbors);
    let want_coh = cohesion_force(si, positions, neighbors);
    let want_goal = match query.goal {
        Some(target) => goal_force(positions[si], target),
        None => Vec3::ZERO,
    };
    let want_combined = combine_forces(want_sep, want_align, want_coh, want_goal, query.weights);
    let want_turned = limit_turn(want_combined, velocities[si], query.limits.max_turn);
    let want_steer = steer(
        si,
        positions,
        velocities,
        neighbors,
        query.goal,
        query.weights,
        query.limits,
    );
    let want_integrated = integrate(velocities[si], want_steer, query.dt, query.limits.max_speed);
    let want_clamp = clamp_length(query.clamp_input, query.clamp_max_len);
    let want_turn = limit_turn(query.turn_steer, query.turn_velocity, query.turn_max);
    let want_cell = boid_cell(positions[si], query.cell_size);

    close_vec("separation", idx, got.separation, want_sep);
    close_vec("alignment", idx, got.alignment, want_align);
    close_vec("cohesion", idx, got.cohesion, want_coh);
    close_vec("goal_dir", idx, got.goal_dir, want_goal);
    close_vec("combined", idx, got.combined, want_combined);
    close_vec("turned", idx, got.turned, want_turned);
    close_vec("steer", idx, got.steer, want_steer);
    close_vec("integrated", idx, got.integrated, want_integrated);
    close_vec("clamp_length", idx, got.clamp_length, want_clamp);
    close_vec("limit_turn", idx, got.limit_turn, want_turn);
    assert_eq!(
        got.cell, want_cell,
        "query {idx} cell: gpu {:?} vs cpu {:?}",
        got.cell, want_cell
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference, element-for-element.
fn check(
    ctx: &GpuContext,
    gpu: &GpuBoids,
    positions: &[Vec3],
    velocities: &[Vec3],
    queries: &[BoidsQuery],
) {
    let got = gpu.eval(ctx, positions, velocities, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, positions, velocities, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[], &[], &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn separation_pushes_away_from_a_close_neighbor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    // Self and one neighbor half a unit apart on +x: the inverse-distance push
    // points cleanly toward -x and normalizes to a unit direction.
    let positions = vec![Vec3::new(0.37, 0.19, 0.23), Vec3::new(0.87, 0.19, 0.23)];
    let velocities = vec![Vec3::new(0.11, 0.0, 0.0), Vec3::new(0.07, 0.0, 0.0)];
    let query = base_query(0, vec![1]);
    check(&ctx, &gpu, &positions, &velocities, &[query]);
}

#[test]
fn alignment_matches_the_mean_neighbor_heading() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    let (positions, velocities) = small_scene();
    // Two neighbors with clearly different, plainly nonzero headings: the mean
    // velocity normalizes to a well-defined alignment direction.
    let query = base_query(0, vec![1, 2]);
    check(&ctx, &gpu, &positions, &velocities, &[query]);
}

#[test]
fn cohesion_steers_toward_the_centroid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    let (positions, velocities) = small_scene();
    // The neighbor centroid sits off the boid, so the cohesion direction is a
    // well-defined unit vector toward it.
    let query = base_query(0, vec![1, 2]);
    check(&ctx, &gpu, &positions, &velocities, &[query]);
}

#[test]
fn goal_force_points_at_the_target() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    let (positions, velocities) = small_scene();
    // Goal-only weights with an empty neighborhood isolate the goal direction:
    // the target is far from the boid, so the direction is unambiguous.
    let mut query = base_query(0, Vec::new());
    query.goal = Some(Vec3::new(5.0, 5.0, 5.0));
    query.weights = BoidsWeights::new(0.0, 0.0, 0.0, 1.0);
    check(&ctx, &gpu, &positions, &velocities, &[query]);
}

#[test]
fn combine_forces_uses_the_fixed_weighted_order() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    let (positions, velocities) = small_scene();
    // Distinct per-rule weights plus an active goal exercise all four terms of
    // the fixed separation / alignment / cohesion / goal sum.
    let mut query = base_query(0, vec![1, 2]);
    query.goal = Some(Vec3::new(5.0, 5.0, 5.0));
    query.weights = BoidsWeights::new(2.0, 3.0, 4.0, 5.0);
    check(&ctx, &gpu, &positions, &velocities, &[query]);
}

#[test]
fn limit_turn_clamps_the_perpendicular_component() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    let (positions, velocities) = small_scene();
    // Heading +x, steer strongly +y: the +y (turn) part (length 5) is capped to
    // turn_max 0.5 while the along-heading part passes through. The 5-vs-0.5
    // margin keeps both devices on the clamp branch.
    let mut query = base_query(0, vec![1, 2]);
    query.turn_steer = Vec3::new(1.0, 5.0, 0.0);
    query.turn_velocity = Vec3::new(2.0, 0.0, 0.0);
    query.turn_max = 0.5;
    check(&ctx, &gpu, &positions, &velocities, &[query]);
}

#[test]
fn limit_turn_is_identity_without_a_heading() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    let (positions, velocities) = small_scene();
    // A zero probe velocity (squared length exactly zero) means no heading, so
    // the turn clamp is skipped and the steer vector passes through unchanged.
    let mut query = base_query(0, vec![1, 2]);
    query.turn_steer = Vec3::new(1.0, 5.0, 0.0);
    query.turn_velocity = Vec3::ZERO;
    query.turn_max = 0.5;
    check(&ctx, &gpu, &positions, &velocities, &[query]);
}

#[test]
fn clamp_length_covers_each_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    let (positions, velocities) = small_scene();
    // Three probes, one per branch, each with a clear margin: a short vector
    // passes through (0.09 vs 1.0), a long one is clamped (25 vs 1.0), and a
    // non-positive limit collapses to zero.
    let mut no_op = base_query(0, vec![1, 2]);
    no_op.clamp_input = Vec3::new(0.3, 0.0, 0.0);
    no_op.clamp_max_len = 1.0;
    let mut clamped = base_query(1, vec![0, 2]);
    clamped.clamp_input = Vec3::new(3.0, 4.0, 0.0);
    clamped.clamp_max_len = 1.0;
    let mut collapsed = base_query(2, vec![0, 1]);
    collapsed.clamp_input = Vec3::new(3.0, 4.0, 0.0);
    collapsed.clamp_max_len = 0.0;
    check(
        &ctx,
        &gpu,
        &positions,
        &velocities,
        &[no_op, clamped, collapsed],
    );
}

#[test]
fn integrate_clamps_to_max_speed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    // No neighbors and no goal make the steering acceleration zero, so the
    // integrated velocity is the current velocity clamped to max_speed. The
    // speed 3 clearly exceeds the cap 2, so both devices take the clamp branch.
    let positions = vec![Vec3::new(0.37, 0.41, 0.29)];
    let velocities = vec![Vec3::new(3.0, 0.0, 0.0)];
    let mut query = base_query(0, Vec::new());
    query.dt = 1.0;
    query.limits = BoidsLimits::new(100.0, -1.0, 2.0, 100.0, 100.0);
    check(&ctx, &gpu, &positions, &velocities, &[query]);
}

#[test]
fn steer_is_bounded_by_max_force() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    // Large rule weights push the raw steering force well past the force cap, so
    // the final steer magnitude is clamped to max_force 0.5 (a 10-vs-0.5 margin
    // keeps both devices on the clamp branch).
    let positions = vec![Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.3, 0.0, 0.0)];
    let velocities = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.07, 0.0, 0.0)];
    let mut query = base_query(0, vec![1]);
    query.goal = Some(Vec3::new(0.1, 0.0, 9.0));
    query.weights = BoidsWeights::new(10.0, 10.0, 10.0, 10.0);
    query.limits = BoidsLimits::new(1.0, -1.0, 5.0, 0.5, 100.0);
    check(&ctx, &gpu, &positions, &velocities, &[query]);
}

#[test]
fn boid_cell_matches_the_lattice_exactly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    // Three boids with fractional-offset positions so each floor is
    // unambiguous: a unit lattice, a non-positive cell size (collapses to the
    // origin), and a coarse lattice with negative coordinates.
    let positions = vec![
        Vec3::new(1.37, -0.63, 2.37),
        Vec3::new(3.1, 4.2, -1.3),
        Vec3::new(-2.63, -0.63, -3.63),
    ];
    let velocities = vec![
        Vec3::new(0.11, 0.2, 0.0),
        Vec3::new(0.2, 0.11, 0.0),
        Vec3::new(0.1, 0.1, 0.3),
    ];
    let mut unit = base_query(0, Vec::new());
    unit.cell_size = 1.0;
    let mut collapsed = base_query(1, Vec::new());
    collapsed.cell_size = 0.0;
    let mut coarse = base_query(2, Vec::new());
    coarse.cell_size = 2.0;
    check(
        &ctx,
        &gpu,
        &positions,
        &velocities,
        &[unit, collapsed, coarse],
    );
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoids::new(&ctx);
    let (positions, velocities) = flock_pool(24);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // Many boids spread across several workgroups, each with a random neighbor
    // subset, random positive weights and an optional goal, all under large
    // no-op clamps so every device stays on the same branch; the per-thread
    // indexing and contiguous storage layout are exercised together.
    let pool_len = positions.len();
    let mut queries = Vec::new();
    for q in 0..150 {
        let self_index = q % pool_len;
        let neighbors = pick_neighbors(&mut state, pool_len, self_index);
        let mut query = base_query(self_index as u32, neighbors);
        query.weights = BoidsWeights::new(
            0.2 + lcg(&mut state) * 2.3,
            0.2 + lcg(&mut state) * 2.3,
            0.2 + lcg(&mut state) * 2.3,
            0.2 + lcg(&mut state) * 2.3,
        );
        let self_pos = positions[self_index];
        query.goal = if lcg(&mut state) > 0.5 {
            Some(self_pos.add(Vec3::new(
                2.0 + lcg(&mut state),
                -1.5 - lcg(&mut state),
                1.5 + lcg(&mut state),
            )))
        } else {
            None
        };
        queries.push(query);
    }
    check(&ctx, &gpu, &positions, &velocities, &queries);
}
