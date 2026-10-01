//! Real-device parity for the oriented-plane / line-intersection twin:
//! [`GpuPlaneLineIntersect`](prism_volumetric_gpu::plane_line_intersect::GpuPlaneLineIntersect)
//! must reproduce the `CPU` golden
//! [`plane_line_intersect`](prism_render_architecture::particle::plane_line_intersect)
//! across a clean infinite-line crossing, a parallel-off-plane line, a
//! coincident line, a segment that crosses with `t` interior, a segment whose
//! crossing falls outside `[0, 1]`, a ray that crosses ahead of its origin, a
//! ray whose crossing is behind it, and a randomized batch of
//! clearly-conditioned crossings compared field-for-field.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The infinite-line classification code and the segment / ray hit flags are
//! integers, compared with an exact `==`. The continuous `f32` fields (the
//! parameters, the crossing points and the signed distance) are not bit-exact:
//! a `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every
//! such field.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from a branch tie and from the
//! degeneracy cracks: the crossing fixtures keep `|normal · dir|` far above the
//! guard epsilon, the parallel and coincident fixtures use exactly-perpendicular
//! integer geometry so the guard fires identically on both devices, and the
//! segment / ray fixtures place `t` clearly inside or clearly outside their
//! acceptance band. The random fixtures are rejection-sampled to keep the
//! determinant large and every `t` clear of its boundary, so `CPU` and `GPU`
//! share every branch regardless of a few units in the last place of slack. All
//! fixture data is built from integer and four-function arithmetic only, with no
//! `f32` transcendental call.
//!
//! Provenance: twinned from this repository's
//! [`plane_line_intersect`](prism_render_architecture::particle::plane_line_intersect);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::plane_line_intersect::{
    line_plane_intersect, ray_plane_intersect, segment_plane_intersect, signed_distance,
    LinePlaneResult,
};
use prism_volumetric_gpu::plane_line_intersect::{
    GpuPlaneLineIntersect, PlaneLineQuery, PlaneLineResult, CLASS_COINCIDENT, CLASS_PARALLEL,
    CLASS_POINT,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// Asserts two 3-vectors agree channel-for-channel within the parity bound.
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

/// A pseudo-random 3-vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// Dot product of two 3-vectors, matching the reference helper without pulling
/// in a math crate.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Builds a clearly-conditioned query by rejection sampling: the infinite line
/// must cross well clear of the parallel guard (`|normal · dir| >= 0.3`), its
/// crossing parameter must sit comfortably off the `t == 0` boundary the ray
/// test turns on, the segment direction must also be well clear of parallel, and
/// the segment crossing parameter must land comfortably interior (`[0.1, 0.9]`).
/// This keeps every random fixture inside the crossing branch and clear of every
/// acceptance boundary, so `CPU` and `GPU` share every branch.
fn crossing_query(state: &mut u64) -> PlaneLineQuery {
    loop {
        let p0 = rand_vec(state, 3.0);
        let dir = rand_vec(state, 2.0);
        let p1 = rand_vec(state, 3.0);
        let normal = rand_vec(state, 2.0);
        let d = signed(state, 2.0);

        let denom_line = dot(normal, dir);
        // The infinite line and ray must be clearly non-parallel.
        if denom_line.abs() < 0.3 {
            continue;
        }
        let seg_dir = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let denom_seg = dot(normal, seg_dir);
        // The segment must also be clearly non-parallel.
        if denom_seg.abs() < 0.3 {
            continue;
        }

        let line_t = (d - dot(normal, p0)) / denom_line;
        // Keep the ray decision clear of the t == 0 boundary either way.
        if line_t.abs() < 0.2 {
            continue;
        }
        let seg_t = (d - dot(normal, p0)) / denom_seg;
        // Keep the segment crossing comfortably interior.
        if !(0.1..=0.9).contains(&seg_t) {
            continue;
        }

        return PlaneLineQuery::new(p0, dir, p1, normal, d);
    }
}

/// Maps a reference [`LinePlaneResult`] to the twin's classification code.
fn line_code(result: &LinePlaneResult) -> u32 {
    match result {
        LinePlaneResult::Point(_, _) => CLASS_POINT,
        LinePlaneResult::Parallel => CLASS_PARALLEL,
        LinePlaneResult::Coincident => CLASS_COINCIDENT,
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the infinite-line
/// classification (and its crossing when present), the segment hit record, the
/// ray hit record and the signed distance of `p0` must all agree.
fn pin(idx: usize, query: &PlaneLineQuery, got: &PlaneLineResult) {
    let want_line = line_plane_intersect(query.p0, query.dir, query.normal, query.d);
    assert_eq!(
        got.line_code,
        line_code(&want_line),
        "query {idx} line_code: gpu {} vs cpu {}",
        got.line_code,
        line_code(&want_line)
    );
    if let LinePlaneResult::Point(want_point, want_t) = want_line {
        assert!(
            close(got.line_t, want_t),
            "query {idx} line_t: gpu {} vs cpu {}",
            got.line_t,
            want_t
        );
        close_vec("line_point", idx, got.line_point, want_point);
    }

    let want_seg = segment_plane_intersect(query.p0, query.p1, query.normal, query.d);
    assert_eq!(
        got.segment_hit,
        want_seg.is_some(),
        "query {idx} segment_hit: gpu {} vs cpu {}",
        got.segment_hit,
        want_seg.is_some()
    );
    if let Some((want_point, want_t)) = want_seg {
        assert!(
            close(got.segment_t, want_t),
            "query {idx} segment_t: gpu {} vs cpu {}",
            got.segment_t,
            want_t
        );
        close_vec("segment_point", idx, got.segment_point, want_point);
    }

    let want_ray = ray_plane_intersect(query.p0, query.dir, query.normal, query.d);
    assert_eq!(
        got.ray_hit,
        want_ray.is_some(),
        "query {idx} ray_hit: gpu {} vs cpu {}",
        got.ray_hit,
        want_ray.is_some()
    );
    if let Some((want_point, want_t)) = want_ray {
        assert!(
            close(got.ray_t, want_t),
            "query {idx} ray_t: gpu {} vs cpu {}",
            got.ray_t,
            want_t
        );
        close_vec("ray_point", idx, got.ray_point, want_point);
    }

    let want_dist = signed_distance(query.p0, query.normal, query.d);
    assert!(
        close(got.signed_distance_p0, want_dist),
        "query {idx} signed_distance_p0: gpu {} vs cpu {}",
        got.signed_distance_p0,
        want_dist
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuPlaneLineIntersect, queries: &[PlaneLineQuery]) {
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
    let gpu = GpuPlaneLineIntersect::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn line_crossing_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneLineIntersect::new(&ctx);
    // Plane x = 0, line along +x from x = -2: a clean crossing at t = 2, origin.
    let query = PlaneLineQuery::new(
        [-2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn line_parallel_off_plane_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneLineIntersect::new(&ctx);
    // Plane y = 0, line along +x at y = 5: exactly perpendicular normal/dir, so
    // the parallel guard fires identically; p0 is off the plane.
    let query = PlaneLineQuery::new(
        [0.0, 5.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 5.0, 0.0],
        [0.0, 1.0, 0.0],
        0.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn line_coincident_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneLineIntersect::new(&ctx);
    // Plane y = 0, line along +x at y = 0: parallel guard fires and p0 lies on
    // the plane, so the classification is coincident.
    let query = PlaneLineQuery::new(
        [-3.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        0.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn segment_crosses_interior_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneLineIntersect::new(&ctx);
    // Plane z = 0, segment from z = -1 to z = 1: crosses at t = 0.5, interior.
    let query = PlaneLineQuery::new(
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        0.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn segment_crossing_outside_range_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneLineIntersect::new(&ctx);
    // Plane z = 0, segment from z = 2 to z = 3: the crossing is at t = -2, well
    // outside [0, 1], so the segment misses while the infinite line crosses.
    let query = PlaneLineQuery::new(
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 1.0],
        0.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn ray_crosses_ahead_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneLineIntersect::new(&ctx);
    // Plane x = 0, ray from x = -2 heading +x: crosses ahead at t = 2.
    let query = PlaneLineQuery::new(
        [-2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn ray_crossing_behind_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneLineIntersect::new(&ctx);
    // Plane x = 0, ray from x = 2 heading +x: the crossing is behind at t = -2,
    // so the ray misses while the infinite line crosses.
    let query = PlaneLineQuery::new(
        [2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneLineIntersect::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random crossings,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned field-for-field.
    let mut queries = vec![
        PlaneLineQuery::new(
            [-2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.0,
        ),
        PlaneLineQuery::new(
            [0.0, 5.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 5.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
        ),
        PlaneLineQuery::new(
            [-3.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            0.0,
        ),
    ];
    for _ in 0..48 {
        queries.push(crossing_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_crossings_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneLineIntersect::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned crossings (several workgroups' worth)
    // pins every reported field across many random plane / line geometries.
    let queries: Vec<PlaneLineQuery> = (0..200).map(|_| crossing_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
