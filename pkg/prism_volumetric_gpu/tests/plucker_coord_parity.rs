//! Real-device parity for the Plücker line-coordinate twin:
//! [`GpuPluckerCoord`](prism_volumetric_gpu::plucker_coord::GpuPluckerCoord)
//! must reproduce the `CPU` golden
//! [`plucker_coord`](prism_render_architecture::particle::plucker_coord)
//! across axis-aligned line pairs, an intersecting (coplanar) pair whose `side`
//! is ~zero, clearly skew pairs with a strongly signed `side`, a pair sharing a
//! construction point, and a randomized batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! subtracts, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units
//! in the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every `f32` field.
//!
//! # Conditioning
//!
//! Fixtures use modest coordinates so the moment cross products and `side`
//! sums stay small enough that the absolute floor absorbs any fused
//! multiply-add slack, keeping the near-zero Plücker residual `u·v` well inside
//! tolerance even where cancellation dominates. Larger-magnitude cases are
//! covered by the signed `side` where the relative bound applies.
//!
//! Provenance: twinned from this repository's
//! [`plucker_coord`](prism_render_architecture::particle::plucker_coord); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::plucker_coord::{side, Line6, Vec3};
use prism_volumetric_gpu::plucker_coord::{GpuPluckerCoord, PluckerQuery, PluckerResult};
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

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        signed(state, span),
        signed(state, span),
        signed(state, span),
    )
}

/// Builds a modestly-scaled random query. The span is small enough that the
/// moment cross products stay well-conditioned, so the near-zero residual
/// `u·v` is absorbed by the absolute floor while `side` still varies widely.
fn rand_query(state: &mut u64) -> PluckerQuery {
    PluckerQuery::new(
        rand_vec(state, 4.0),
        rand_vec(state, 4.0),
        rand_vec(state, 4.0),
        rand_vec(state, 4.0),
    )
}

/// Pins one `GPU` result against the `CPU` golden for `query`: both rebuilt
/// lines (direction and moment), the permuted inner product `side`, and the two
/// Plücker residuals must all agree within bound.
fn pin(idx: usize, query: &PluckerQuery, got: &PluckerResult) {
    let l1 = Line6::from_points(query.first_start, query.first_end);
    let l2 = Line6::from_points(query.second_start, query.second_end);
    let want_side = side(&l1, &l2);
    let want_r1 = l1.moment_orthogonality();
    let want_r2 = l2.moment_orthogonality();

    close_vec("first_u", idx, got.first.u, l1.u);
    close_vec("first_v", idx, got.first.v, l1.v);
    close_vec("second_u", idx, got.second.u, l2.u);
    close_vec("second_v", idx, got.second.v, l2.v);
    assert!(
        close(got.side, want_side),
        "query {idx} side: gpu {} vs cpu {}",
        got.side,
        want_side
    );
    assert!(
        close(got.first_moment_residual, want_r1),
        "query {idx} first_moment_residual: gpu {} vs cpu {}",
        got.first_moment_residual,
        want_r1
    );
    assert!(
        close(got.second_moment_residual, want_r2),
        "query {idx} second_moment_residual: gpu {} vs cpu {}",
        got.second_moment_residual,
        want_r2
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuPluckerCoord, queries: &[PluckerQuery]) {
    let got = gpu.eval(ctx, queries);
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
    let gpu = GpuPluckerCoord::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn axis_aligned_pair_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPluckerCoord::new(&ctx);
    // The x-axis through the origin against a line offset along z: a clean,
    // strongly-signed `side` with simple integer moments.
    let query = PluckerQuery::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn intersecting_pair_is_coplanar() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPluckerCoord::new(&ctx);
    // Two lines that cross at the origin are coplanar, so `side` is ~zero; the
    // absolute floor pins that near-zero value on both devices.
    let query = PluckerQuery::new(
        Vec3::new(-2.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, -3.0, 0.0),
        Vec3::new(0.0, 3.0, 0.0),
    );
    let got = gpu.eval(&ctx, &[query]);
    assert!(got[0].side.abs() <= EPS, "coplanar side should be ~zero");
    check(&ctx, &gpu, &[query]);
}

#[test]
fn skew_pair_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPluckerCoord::new(&ctx);
    // A clearly skew pair with a strongly-signed `side` well away from zero.
    let query = PluckerQuery::new(
        Vec3::new(-3.0, -1.0, 0.0),
        Vec3::new(3.0, 1.0, 0.0),
        Vec3::new(0.0, -2.0, 2.0),
        Vec3::new(1.0, 2.0, 2.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn shared_point_pair_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPluckerCoord::new(&ctx);
    // The two segments share a construction endpoint; the lines meet there and
    // the moments are built from overlapping points, exercising the general
    // algebra with a common vertex.
    let query = PluckerQuery::new(
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(2.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(0.0, 2.0, -1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPluckerCoord::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random pairs,
    // dispatched together so the per-thread indexing and contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        PluckerQuery::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ),
        PluckerQuery::new(
            Vec3::new(-3.0, -1.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
            Vec3::new(0.0, -2.0, 2.0),
            Vec3::new(1.0, 2.0, 2.0),
        ),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_pairs_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPluckerCoord::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins every reported field
    // across many random line geometries.
    let queries: Vec<PluckerQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
