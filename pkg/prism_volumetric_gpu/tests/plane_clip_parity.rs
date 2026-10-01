//! Real-device parity for the half-space (plane) segment-clip twin:
//! [`GpuPlaneClip`](prism_volumetric_gpu::plane_clip::GpuPlaneClip) must
//! reproduce the `CPU` golden
//! [`plane_clip`](prism_render_architecture::particle::plane_clip) across a
//! segment fully inside the half-space, a segment fully outside, a segment that
//! straddles the plane with its `b` endpoint trimmed, a straddle with its `a`
//! endpoint trimmed, a segment parallel to the plane (the `intersect_param`
//! denominator guard fires), a point lying exactly on the plane (classified
//! `On`), and a randomized batch of clearly-conditioned geometries compared
//! field-for-field.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The two endpoint classification codes and the two hit flags are integers /
//! sign decisions, compared with an exact `==`. The continuous `f32` fields (the
//! signed distances, the crossing parameter and the clipped endpoints) are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on
//! every such field.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from a branch tie and from the
//! degeneracy cracks: each endpoint's signed distance stays far from zero (so the
//! `clip_segment` sign test and the `classify` band fire identically on both
//! devices), the parallel fixture uses exactly-perpendicular integer geometry so
//! the denominator guard fires identically, and the straddle fixtures place the
//! crossing `t` clearly interior. The random fixtures are rejection-sampled to
//! keep both signed distances large, the denominator large and `t` clear of its
//! boundaries, so `CPU` and `GPU` share every branch regardless of a few units in
//! the last place of slack. All fixture data is built from integer and
//! four-function arithmetic only, with no `f32` transcendental call.
//!
//! Provenance: twinned from this repository's
//! [`plane_clip`](prism_render_architecture::particle::plane_clip); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::plane_clip::{
    classify, clip_segment, intersect_param, Plane, Side,
};
use prism_volumetric_gpu::plane_clip::{
    GpuPlaneClip, PlaneClipQuery, PlaneClipResult, SIDE_INSIDE, SIDE_ON, SIDE_OUTSIDE,
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

/// The classification band half-width every query passes to `classify`, matching
/// the reference `CMP_EPS`.
const CLASSIFY_EPS: f32 = 1.0e-6;

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

/// Builds a clearly-conditioned query by rejection sampling: both endpoints must
/// sit well clear of the plane (`|signed_distance| >= 0.3`, so the `clip_segment`
/// sign test and the `classify` band never tie), the segment must be clearly
/// non-parallel (`|da - db| >= 0.3`), and the crossing parameter must land either
/// comfortably interior (`[0.1, 0.9]`) or comfortably outside (`<= -0.1` or
/// `>= 1.1`) so the `intersect_param` acceptance test fires identically on both
/// devices.
fn clip_query(state: &mut u64) -> PlaneClipQuery {
    loop {
        let normal = rand_vec(state, 2.0);
        let d = signed(state, 2.0);
        let a = rand_vec(state, 3.0);
        let b = rand_vec(state, 3.0);

        let da = dot(normal, a) + d;
        let db = dot(normal, b) + d;
        // Both endpoints must be clearly off the plane.
        if da.abs() < 0.3 || db.abs() < 0.3 {
            continue;
        }
        let denom = da - db;
        // The segment must be clearly non-parallel.
        if denom.abs() < 0.3 {
            continue;
        }
        let t = da / denom;
        // Keep the crossing clearly interior or clearly outside [0, 1].
        let interior = (0.1..=0.9).contains(&t);
        let exterior = t <= -0.1 || t >= 1.1;
        if !(interior || exterior) {
            continue;
        }

        return PlaneClipQuery::new(normal, d, a, b, CLASSIFY_EPS);
    }
}

/// Maps a reference [`Side`] to the twin's classification code.
fn side_code(side: Side) -> u32 {
    match side {
        Side::Inside => SIDE_INSIDE,
        Side::Outside => SIDE_OUTSIDE,
        Side::On => SIDE_ON,
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the signed
/// distances, both endpoint classifications, the `intersect_param` record and the
/// `clip_segment` survivor must all agree.
fn pin(idx: usize, query: &PlaneClipQuery, got: &PlaneClipResult) {
    let plane = Plane::new(query.normal, query.d);

    let want_da = plane.signed_distance(query.a);
    let want_db = plane.signed_distance(query.b);
    assert!(
        close(got.signed_distance_a, want_da),
        "query {idx} signed_distance_a: gpu {} vs cpu {}",
        got.signed_distance_a,
        want_da
    );
    assert!(
        close(got.signed_distance_b, want_db),
        "query {idx} signed_distance_b: gpu {} vs cpu {}",
        got.signed_distance_b,
        want_db
    );

    let want_ca = side_code(classify(&plane, query.a, query.eps));
    let want_cb = side_code(classify(&plane, query.b, query.eps));
    assert_eq!(
        got.classify_a, want_ca,
        "query {idx} classify_a: gpu {} vs cpu {}",
        got.classify_a, want_ca
    );
    assert_eq!(
        got.classify_b, want_cb,
        "query {idx} classify_b: gpu {} vs cpu {}",
        got.classify_b, want_cb
    );

    let want_t = intersect_param(&plane, query.a, query.b);
    assert_eq!(
        got.intersect_hit,
        want_t.is_some(),
        "query {idx} intersect_hit: gpu {} vs cpu {}",
        got.intersect_hit,
        want_t.is_some()
    );
    if let Some(t) = want_t {
        assert!(
            close(got.intersect_t, t),
            "query {idx} intersect_t: gpu {} vs cpu {}",
            got.intersect_t,
            t
        );
    }

    let want_clip = clip_segment(&plane, query.a, query.b);
    assert_eq!(
        got.clip_hit,
        want_clip.is_some(),
        "query {idx} clip_hit: gpu {} vs cpu {}",
        got.clip_hit,
        want_clip.is_some()
    );
    if let Some((want_a, want_b)) = want_clip {
        close_vec("clip_a", idx, got.clip_a, want_a);
        close_vec("clip_b", idx, got.clip_b, want_b);
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuPlaneClip, queries: &[PlaneClipQuery]) {
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
    let gpu = GpuPlaneClip::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn segment_fully_inside_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneClip::new(&ctx);
    // Inside where x >= -1 (normal +x, d = 1): both endpoints are inside, so the
    // segment is returned unchanged and never crosses.
    let query = PlaneClipQuery::new(
        [1.0, 0.0, 0.0],
        1.0,
        [2.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        CLASSIFY_EPS,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn segment_fully_outside_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneClip::new(&ctx);
    // Inside where x >= 5 (normal +x, d = -5): both endpoints lie at x < 5, so
    // the clip rejects the segment while the infinite crossing is outside [0, 1].
    let query = PlaneClipQuery::new(
        [1.0, 0.0, 0.0],
        -5.0,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        CLASSIFY_EPS,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn segment_straddles_trimming_b_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneClip::new(&ctx);
    // Inside where x >= 0 (normal +x, d = 0): a at x = 1 is inside, b at x = -1 is
    // outside, so b is trimmed to the boundary x = 0 at t = 0.5.
    let query = PlaneClipQuery::new(
        [1.0, 0.0, 0.0],
        0.0,
        [1.0, 0.0, 0.0],
        [-1.0, 2.0, 0.0],
        CLASSIFY_EPS,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn segment_straddles_trimming_a_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneClip::new(&ctx);
    // Inside where x >= 0 (normal +x, d = 0): a at x = -2 is outside, b at x = 2
    // is inside, so a is trimmed to the boundary x = 0 at t = 0.5.
    let query = PlaneClipQuery::new(
        [1.0, 0.0, 0.0],
        0.0,
        [-2.0, 0.0, 0.0],
        [2.0, 4.0, 0.0],
        CLASSIFY_EPS,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn segment_parallel_to_plane_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneClip::new(&ctx);
    // Plane y = 0 (normal +y, d = 0), segment along +x at y = 3: exactly
    // perpendicular normal/direction, so the denominator guard fires and
    // intersect_param misses; both endpoints are inside, so the clip keeps all.
    let query = PlaneClipQuery::new(
        [0.0, 1.0, 0.0],
        0.0,
        [-2.0, 3.0, 0.0],
        [2.0, 3.0, 0.0],
        CLASSIFY_EPS,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn endpoint_on_plane_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneClip::new(&ctx);
    // Plane x = 0 (normal +x, d = 0): a lies exactly on the plane (classified On)
    // while b is clearly inside, built from integer geometry so the signed
    // distance is exactly zero on both devices.
    let query = PlaneClipQuery::new(
        [1.0, 0.0, 0.0],
        0.0,
        [0.0, 2.0, 0.0],
        [3.0, 1.0, 0.0],
        CLASSIFY_EPS,
    );
    let got = gpu.eval(&ctx, &[query]);
    assert_eq!(got.len(), 1);
    assert_eq!(
        got[0].classify_a, SIDE_ON,
        "endpoint on the plane must classify as On"
    );
    pin(0, &query, &got[0]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneClip::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random geometries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned field-for-field.
    let mut queries = vec![
        PlaneClipQuery::new(
            [1.0, 0.0, 0.0],
            0.0,
            [1.0, 0.0, 0.0],
            [-1.0, 2.0, 0.0],
            CLASSIFY_EPS,
        ),
        PlaneClipQuery::new(
            [1.0, 0.0, 0.0],
            -5.0,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            CLASSIFY_EPS,
        ),
        PlaneClipQuery::new(
            [0.0, 1.0, 0.0],
            0.0,
            [-2.0, 3.0, 0.0],
            [2.0, 3.0, 0.0],
            CLASSIFY_EPS,
        ),
    ];
    for _ in 0..48 {
        queries.push(clip_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_segments_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneClip::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned geometries (several workgroups' worth)
    // pins every reported field across many random plane / segment pairs.
    let queries: Vec<PlaneClipQuery> = (0..200).map(|_| clip_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
