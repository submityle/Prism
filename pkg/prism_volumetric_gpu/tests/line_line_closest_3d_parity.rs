//! Real-device parity for the infinite-line / ray closest-point twin:
//! [`GpuLineLineClosest3d`](prism_volumetric_gpu::line_line_closest_3d::GpuLineLineClosest3d)
//! must reproduce the `CPU` golden
//! [`line_line_closest_3d`](prism_render_architecture::particle::line_line_closest_3d)
//! across orthogonal skew lines (a full-rank solve with a well-conditioned
//! determinant), exactly-parallel lines (an integer-valued determinant that is
//! zero on both devices, exercising the parallel pin), a degenerate-direction
//! query (a zero-length direction forcing a parameter to `0`), a ray pair whose
//! optimum lies behind both origins (both ray parameters clamp to `0`), a ray
//! pair clamped on one side, and a randomized batch of clearly-conditioned skew
//! pairs compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units
//! in the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every `f32` field.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from a branch tie and from the
//! degeneracy cracks: skew pairs keep their directions far from parallel so the
//! determinant sits well above the compare epsilon, the parallel pair uses
//! integer coordinates whose determinant is exactly zero on both devices, the
//! degenerate query has a direction of exactly zero length, and the random ray
//! batch is rejection-sampled so both ray parameters land comfortably positive,
//! clear of every clamp boundary. This keeps `CPU` and `GPU` on the same side
//! of every branch regardless of a few units in the last place of slack.
//!
//! Provenance: twinned from this repository's
//! [`line_line_closest_3d`](prism_render_architecture::particle::line_line_closest_3d);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::line_line_closest_3d::{
    line_line_closest, ray_ray_closest, ClosestLines,
};
use prism_volumetric_gpu::line_line_closest_3d::{
    GpuLineLineClosest3d, LineLineQuery, LineLineResult,
};
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

/// Asserts two points agree lane-by-lane within the parity bound.
fn close_vec(label: &str, idx: usize, got: [f32; 3], want: [f32; 3]) {
    assert!(
        close(got[0], want[0]) && close(got[1], want[1]) && close(got[2], want[2]),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got[0],
        got[1],
        got[2],
        want[0],
        want[1],
        want[2]
    );
}

/// Pins one `ClosestLines` record (`s`, `t`, both points and the distance)
/// against the reference within bound.
fn pin_record(idx: usize, label: &str, got: &ClosestLines, want: &ClosestLines) {
    assert!(
        close(got.s, want.s),
        "query {idx} {label} s: gpu {} vs cpu {}",
        got.s,
        want.s
    );
    assert!(
        close(got.t, want.t),
        "query {idx} {label} t: gpu {} vs cpu {}",
        got.t,
        want.t
    );
    assert!(
        close(got.distance, want.distance),
        "query {idx} {label} distance: gpu {} vs cpu {}",
        got.distance,
        want.distance
    );
    close_vec(
        &format!("{label} point_on_a"),
        idx,
        got.point_on_a,
        want.point_on_a,
    );
    close_vec(
        &format!("{label} point_on_b"),
        idx,
        got.point_on_b,
        want.point_on_b,
    );
}

/// Pins both the line-line and ray-ray records of one `GPU` result against the
/// `CPU` golden for `query`.
fn pin(idx: usize, query: &LineLineQuery, got: &LineLineResult) {
    let want_line = line_line_closest(query.p1, query.d1, query.p2, query.d2);
    let want_ray = ray_ray_closest(query.p1, query.d1, query.p2, query.d2);
    pin_record(idx, "line", &got.line, &want_line);
    pin_record(idx, "ray", &got.ray, &want_ray);
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuLineLineClosest3d, queries: &[LineLineQuery]) {
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
fn rand_vec(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// Euclidean dot product, so the rejection sampler needs no library call.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Builds a clearly-conditioned skew pair by rejection sampling: the directions
/// are drawn well-spread and accepted only when they are far from parallel (so
/// the determinant sits well above the compare epsilon) and the `CPU` ray
/// solver places both ray parameters comfortably positive and interior, clear
/// of every clamp boundary. The line solve is always full-rank for such a pair.
fn skew_query(state: &mut u64) -> LineLineQuery {
    loop {
        let p1 = rand_vec(state, 5.0);
        let d1 = rand_vec(state, 4.0);
        let p2 = rand_vec(state, 5.0);
        let d2 = rand_vec(state, 4.0);

        let a = dot(d1, d1);
        let c = dot(d2, d2);
        // Both directions must be clearly non-degenerate.
        if a < 1.0 || c < 1.0 {
            continue;
        }
        // Directions must be far from parallel so the determinant is large.
        let b = dot(d1, d2);
        let cos_sq = (b * b) / (a * c);
        if cos_sq > 0.64 {
            continue;
        }

        // Both ray parameters must land comfortably positive and interior so
        // neither clamp branch sits near a tie on either device.
        let ray = ray_ray_closest(p1, d1, p2, d2);
        if !(0.1..=20.0).contains(&ray.s) || !(0.1..=20.0).contains(&ray.t) {
            continue;
        }

        return LineLineQuery::new(p1, d1, p2, d2);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLineLineClosest3d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn orthogonal_skew_lines_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLineLineClosest3d::new(&ctx);
    // Line A along +x through the origin; line B along +y raised to z = 2. The
    // mutual perpendicular is the z axis, gap = 2, both feet at the lines'
    // crossing over z. Full-rank determinant well clear of the parallel pin.
    let query = LineLineQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn skew_offset_feet_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLineLineClosest3d::new(&ctx);
    // A along x through the origin; B along y through (3, 0, 5). Nearest foot on
    // A is x = 3, on B is y = 0, gap = 5.
    let query = LineLineQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 5.0],
        [0.0, 1.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn parallel_lines_pin_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLineLineClosest3d::new(&ctx);
    // Both lines along x with integer coordinates: the determinant
    // a*c - b*b = 1*4 - 2*2 = 0 is exactly zero on both devices, so the parallel
    // pin fires identically (s = 0, perpendicular foot on B). Gap = 5.
    let query = LineLineQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [10.0, 4.0, 3.0],
        [2.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn degenerate_direction_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLineLineClosest3d::new(&ctx);
    // Line A has a zero-length direction (squared length exactly zero), so it
    // collapses to its base point and the parallel branch pins s = 0, dropping a
    // perpendicular from p1 onto line B. The query becomes point-to-line.
    let query = LineLineQuery::new(
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn ray_pair_behind_both_origins_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLineLineClosest3d::new(&ctx);
    // Two rays pointing away from each other in x from separated origins: the
    // ray optimum clamps both parameters to 0 (the two origins), while the
    // infinite-line optimum for the same geometry stays unconstrained.
    let query = LineLineQuery::new(
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [4.0, 3.0, 0.0],
        [1.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn ray_pair_clamped_one_side_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLineLineClosest3d::new(&ctx);
    // Ray A points away from B's crossing so its parameter clamps to 0, then B
    // re-projects onto A's origin: exercises the clamp-and-reproject path with
    // the clamp clearly past the boundary.
    let query = LineLineQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-4.0, 2.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLineLineClosest3d::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random skew pairs,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        LineLineQuery::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
        ),
        LineLineQuery::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 4.0, 3.0],
            [2.0, 0.0, 0.0],
        ),
    ];
    for _ in 0..48 {
        queries.push(skew_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_skew_pairs_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLineLineClosest3d::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned skew pairs (several workgroups'
    // worth) pins every reported field across many random line geometries.
    let queries: Vec<LineLineQuery> = (0..200).map(|_| skew_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
