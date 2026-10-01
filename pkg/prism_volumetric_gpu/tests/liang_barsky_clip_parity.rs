//! Real-device parity for the `Liang-Barsky` clip twin:
//! [`GpuLiangBarskyClip`](prism_volumetric_gpu::liang_barsky_clip::GpuLiangBarskyClip)
//! must reproduce the `CPU` golden
//! [`clip`](prism_render_architecture::particle::liang_barsky_clip::clip) across
//! segments fully inside the window, segments rejected because the surviving
//! parameter interval is empty, segments parallel to and outside an edge,
//! segments partially clipped against one or more window edges, diagonal threads
//! across opposite corners, boundary-coincident segments, zero-length degenerate
//! segments, inverted windows normalized before clipping, and a randomized
//! integer-coordinate batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The survival boolean (the `clip` `Option` discriminant) is a pure
//! sign/epsilon decision, so the comparison asserts an exact `==` on it. The
//! clipped endpoint coordinates and the entry/exit parameters `t0`/`t1` are a
//! fixed, non-reorderable sequence of multiplies, adds and divides; a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place, so the comparison allows
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on those `f32` values.
//!
//! # Conditioning
//!
//! Every fixture uses integer (or exactly representable) coordinates and window
//! bounds, so each per-edge parallel/sign test is far from the compare epsilon
//! and both devices share each accept/reject/clip branch regardless of a few
//! units in the last place of slack in the clipped coordinates. No fixture uses
//! a transcendental function to synthesize a coordinate.
//!
//! Provenance: twinned from this repository's
//! [`liang_barsky_clip`](prism_render_architecture::particle::liang_barsky_clip);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::liang_barsky_clip::{clip, ClipRect, Segment2};
use prism_volumetric_gpu::liang_barsky_clip::{
    GpuLiangBarskyClip, LiangBarskyQuery, LiangBarskyResult,
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

/// Asserts two 2D points agree channel-for-channel within the parity bound.
fn close_point(label: &str, idx: usize, got: [f32; 2], want: [f32; 2]) {
    assert!(
        close(got[0], want[0]) && close(got[1], want[1]),
        "query {idx} {label}: gpu ({}, {}) vs cpu ({}, {})",
        got[0],
        got[1],
        want[0],
        want[1]
    );
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw state's high bits.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 40) as u32
}

/// A pseudo-random integer coordinate in `[-span, span]`, kept exactly
/// representable in `f32` so every per-edge test stays integer-valued.
fn coord(state: &mut u64, span: i32) -> f32 {
    let modulus = (2 * span + 1) as u32;
    (lcg(state) % modulus) as i32 as f32 - span as f32
}

/// A pseudo-random integer-coordinate point.
fn rand_point(state: &mut u64, span: i32) -> [f32; 2] {
    [coord(state, span), coord(state, span)]
}

/// A pseudo-random normalized clip window with integer bounds and a non-zero
/// extent on both axes.
fn rand_rect(state: &mut u64, span: i32) -> ClipRect {
    let x0 = coord(state, span);
    let x1 = coord(state, span);
    let y0 = coord(state, span);
    let y1 = coord(state, span);
    let xmin = x0.min(x1);
    let xmax = x0.max(x1);
    let ymin = y0.min(y1);
    let ymax = y0.max(y1);
    // Guarantee a non-degenerate window so the clip has an interior to trim to.
    ClipRect::new([xmin, ymin], [xmax + 1.0, ymax + 1.0])
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the survival
/// boolean exactly, the clipped endpoints and `t` parameters to within bound.
fn pin(idx: usize, query: &LiangBarskyQuery, got: &LiangBarskyResult) {
    let want = clip(query.rect, query.seg);

    assert_eq!(
        got.hit,
        want.is_some(),
        "query {idx} hit: gpu {} vs cpu {}",
        got.hit,
        want.is_some()
    );

    if let Some(want) = want {
        let res = got.clipped().expect("gpu reports a surviving segment");
        close_point("p0", idx, res.p0, want.p0);
        close_point("p1", idx, res.p1, want.p1);
        assert!(
            close(res.t0, want.t0),
            "query {idx} t0: gpu {} vs cpu {}",
            res.t0,
            want.t0
        );
        assert!(
            close(res.t1, want.t1),
            "query {idx} t1: gpu {} vs cpu {}",
            res.t1,
            want.t1
        );
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuLiangBarskyClip, queries: &[LiangBarskyQuery]) {
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

/// The shared `[0, 10] x [0, 10]` window used by the deterministic fixtures.
const RECT: ClipRect = ClipRect::new([0.0, 0.0], [10.0, 10.0]);

/// Builds a query from the shared window and a segment's endpoints.
fn q(a: [f32; 2], b: [f32; 2]) -> LiangBarskyQuery {
    LiangBarskyQuery::new(RECT, Segment2::new(a, b))
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn fully_inside_segment_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    // Fully inside: returned unchanged with t0 = 0, t1 = 1.
    check(&ctx, &gpu, &[q([2.0, 3.0], [7.0, 8.0])]);
}

#[test]
fn fully_outside_segments_are_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    let queries = [
        // Fully left of the window.
        q([-5.0, 2.0], [-1.0, 8.0]),
        // Fully right of the window.
        q([11.0, 2.0], [15.0, 8.0]),
        // Fully above the window.
        q([2.0, 11.0], [8.0, 20.0]),
        // Fully below the window.
        q([2.0, -11.0], [8.0, -1.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn parallel_outside_edge_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    let queries = [
        // Horizontal segment above the window: parallel to top/bottom edges and
        // outside, rejected on the p_k == 0, q_k < 0 test.
        q([-5.0, 15.0], [15.0, 15.0]),
        // Vertical segment to the right: parallel to left/right edges, outside.
        q([15.0, -5.0], [15.0, 15.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn single_edge_crossings_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    let queries = [
        // Enters through the left edge, far endpoint inside: t0 = 0.5, t1 = 1.
        q([-5.0, 5.0], [5.0, 5.0]),
        // Starts inside, exits through the right edge: t0 = 0, t1 = 0.5.
        q([5.0, 5.0], [15.0, 5.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn spanning_and_diagonal_threads_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    let queries = [
        // Spans both vertical edges horizontally.
        q([-5.0, 5.0], [15.0, 5.0]),
        // Spans both horizontal edges vertically.
        q([4.0, -5.0], [4.0, 15.0]),
        // Diagonal across two opposite corners.
        q([-5.0, -5.0], [15.0, 15.0]),
        // Diagonal with a known entry parameter: (-2,-2) -> (6,6), t0 = 0.25.
        q([-2.0, -2.0], [6.0, 6.0]),
        // Threads the window entering bottom-left, exiting top-right.
        q([-2.0, -1.0], [12.0, 13.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn boundary_coincident_segments_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    let queries = [
        // Endpoint exactly on the left boundary.
        q([0.0, 5.0], [6.0, 5.0]),
        // Segment lying along the top edge (parallel, inside the slab).
        q([2.0, 10.0], [8.0, 10.0]),
        // Interior horizontal and vertical lines, returned unchanged.
        q([1.0, 6.0], [9.0, 6.0]),
        q([3.0, 1.0], [3.0, 9.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn swapped_endpoints_swap_the_visible_endpoints() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    // Forward and reverse spans of the same geometry; the direction flips, so
    // the reversed clip's endpoints swap. Both must still match the reference.
    let queries = [q([-5.0, 5.0], [15.0, 5.0]), q([15.0, 5.0], [-5.0, 5.0])];
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_point_segments_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    let queries = [
        // Zero-length segment inside the window: survives as a point.
        q([5.0, 5.0], [5.0, 5.0]),
        // Zero-length segment outside the window: rejected.
        q([-5.0, -5.0], [-5.0, -5.0]),
        // Zero-length segment exactly on a corner: survives on the boundary.
        q([0.0, 0.0], [0.0, 0.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn inverted_window_is_normalized_before_clipping() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    // An inverted rectangle is clipped against its swapped-corner equivalent,
    // exactly as the reference normalizes it.
    let inverted = ClipRect::new([10.0, 10.0], [0.0, 0.0]);
    let queries = [LiangBarskyQuery::new(
        inverted,
        Segment2::new([-5.0, 5.0], [15.0, 5.0]),
    )];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    // One batch mixing the deterministic fixtures with a fan of guaranteed
    // crossings, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised, then pinned element-for-
    // element.
    let mut queries = vec![
        q([2.0, 3.0], [7.0, 8.0]),
        q([-5.0, -5.0], [15.0, 15.0]),
        q([-5.0, 2.0], [-1.0, 8.0]),
    ];
    // A fan of segments from the far left into the window at integer heights,
    // each clipped against the left edge.
    for k in 0..16 {
        let y = k as f32;
        queries.push(q([-6.0, y], [6.0, y]));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_integer_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLiangBarskyClip::new(&ctx);
    let mut state = 0x0f0f_1234_dead_beef_u64;
    // Several workgroups' worth of random integer-coordinate windows and
    // segments. Integer bounds are far from the compare epsilon, so every
    // per-edge test is identical on both devices and the two sides share each
    // accept/reject/clip branch.
    let queries: Vec<LiangBarskyQuery> = (0..200)
        .map(|_| {
            let rect = rand_rect(&mut state, 8);
            LiangBarskyQuery::new(
                rect,
                Segment2::new(rand_point(&mut state, 12), rand_point(&mut state, 12)),
            )
        })
        .collect();
    check(&ctx, &gpu, &queries);
}
