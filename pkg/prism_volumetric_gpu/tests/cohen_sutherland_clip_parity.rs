//! Real-device parity for the `Cohen-Sutherland` clip twin:
//! [`GpuCohenSutherlandClip`](prism_volumetric_gpu::cohen_sutherland_clip::GpuCohenSutherlandClip)
//! must reproduce the `CPU` golden
//! [`cohen_sutherland_clip`](prism_render_architecture::particle::cohen_sutherland_clip)
//! across segments fully inside the window, segments trivially rejected because
//! both endpoints share an outside half-plane, segments partially clipped
//! against one, two or three window edges, diagonal threads across opposite
//! corners, boundary-coincident segments, zero-length degenerate segments, and a
//! randomized integer-coordinate batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The survival boolean (the `clip_segment` `Option` discriminant) and both raw
//! endpoint outcodes are pure sign/epsilon decisions, so the comparison asserts
//! an exact `==` on them. The clipped endpoint coordinates are a fixed,
//! non-reorderable sequence of multiplies, adds and divides; a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place, so the comparison allows
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on those `f32` values.
//!
//! # Conditioning
//!
//! Every fixture uses integer (or exactly representable) coordinates and window
//! bounds, so each outcode comparison is an integer far from the compare epsilon
//! and the outcode bits are identical on both devices. This keeps `CPU` and
//! `GPU` on the same side of every accept/reject/clip branch regardless of a few
//! units in the last place of slack in the clipped coordinates. No fixture uses
//! a transcendental function to synthesize a coordinate.
//!
//! Provenance: twinned from this repository's
//! [`cohen_sutherland_clip`](prism_render_architecture::particle::cohen_sutherland_clip);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::cohen_sutherland_clip::{clip_segment, ClipRect};
use prism_volumetric_gpu::cohen_sutherland_clip::{
    ClipSegmentQuery, ClipSegmentResult, GpuCohenSutherlandClip,
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
/// representable in `f32` so every outcode comparison stays integer-valued.
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
    ClipRect::new(xmin, ymin, xmax + 1.0, ymax + 1.0)
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the survival
/// boolean and both outcodes exactly, the clipped endpoints to within bound.
fn pin(idx: usize, query: &ClipSegmentQuery, got: &ClipSegmentResult) {
    let want = clip_segment(&query.rect, query.a, query.b);

    assert_eq!(
        got.hit,
        want.is_some(),
        "query {idx} hit: gpu {} vs cpu {}",
        got.hit,
        want.is_some()
    );

    let want_outcode_a = u32::from(query.rect.outcode(query.a));
    let want_outcode_b = u32::from(query.rect.outcode(query.b));
    assert_eq!(
        got.outcode_a, want_outcode_a,
        "query {idx} outcode_a: gpu {} vs cpu {}",
        got.outcode_a, want_outcode_a
    );
    assert_eq!(
        got.outcode_b, want_outcode_b,
        "query {idx} outcode_b: gpu {} vs cpu {}",
        got.outcode_b, want_outcode_b
    );

    if let Some((wa, wb)) = want {
        let (ga, gb) = got.clipped().expect("gpu reports a surviving segment");
        close_point("clip_a", idx, ga, wa);
        close_point("clip_b", idx, gb, wb);
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuCohenSutherlandClip, queries: &[ClipSegmentQuery]) {
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
const RECT: ClipRect = ClipRect::new(0.0, 0.0, 10.0, 10.0);

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohenSutherlandClip::new(&ctx);
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
    let gpu = GpuCohenSutherlandClip::new(&ctx);
    let query = ClipSegmentQuery::new(RECT, [2.0, 3.0], [7.0, 8.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn trivially_rejected_segments_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohenSutherlandClip::new(&ctx);
    let queries = [
        // Fully left of the window.
        ClipSegmentQuery::new(RECT, [-5.0, 2.0], [-1.0, 8.0]),
        // Fully right of the window.
        ClipSegmentQuery::new(RECT, [11.0, 2.0], [15.0, 8.0]),
        // Fully above the window.
        ClipSegmentQuery::new(RECT, [2.0, 11.0], [8.0, 20.0]),
        // Fully below the window.
        ClipSegmentQuery::new(RECT, [2.0, -11.0], [8.0, -1.0]),
        // Both endpoints share the left half-plane (bitwise AND non-zero).
        ClipSegmentQuery::new(RECT, [-3.0, 1.0], [-1.0, 9.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn single_edge_crossings_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohenSutherlandClip::new(&ctx);
    let queries = [
        // Enters through the left edge, far endpoint inside.
        ClipSegmentQuery::new(RECT, [-5.0, 5.0], [5.0, 5.0]),
        // One endpoint inside, exits through the top edge.
        ClipSegmentQuery::new(RECT, [5.0, 5.0], [5.0, 15.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn two_edge_spans_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohenSutherlandClip::new(&ctx);
    let queries = [
        // Spans both horizontal edges.
        ClipSegmentQuery::new(RECT, [-5.0, 5.0], [15.0, 5.0]),
        // Spans both vertical edges.
        ClipSegmentQuery::new(RECT, [4.0, -5.0], [4.0, 15.0]),
        // Diagonal across two opposite corners.
        ClipSegmentQuery::new(RECT, [-5.0, -5.0], [15.0, 15.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn corner_threads_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohenSutherlandClip::new(&ctx);
    let queries = [
        // Both endpoints outside in opposite corner regions; the line threads
        // the window, entering left and leaving top.
        ClipSegmentQuery::new(RECT, [-2.0, -1.0], [12.0, 13.0]),
        // Clips against the left edge, then the top edge, then the right edge.
        ClipSegmentQuery::new(RECT, [-5.0, 2.0], [15.0, 12.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn boundary_coincident_segments_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohenSutherlandClip::new(&ctx);
    let queries = [
        // Endpoint exactly on the left boundary.
        ClipSegmentQuery::new(RECT, [0.0, 5.0], [6.0, 5.0]),
        // Segment lying along the top edge.
        ClipSegmentQuery::new(RECT, [2.0, 10.0], [8.0, 10.0]),
        // Interior horizontal and vertical lines, returned unchanged.
        ClipSegmentQuery::new(RECT, [1.0, 6.0], [9.0, 6.0]),
        ClipSegmentQuery::new(RECT, [3.0, 1.0], [3.0, 9.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_point_segments_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohenSutherlandClip::new(&ctx);
    let queries = [
        // Zero-length segment inside the window: survives as a point.
        ClipSegmentQuery::new(RECT, [5.0, 5.0], [5.0, 5.0]),
        // Zero-length segment outside the window: rejected.
        ClipSegmentQuery::new(RECT, [15.0, 5.0], [15.0, 5.0]),
        // Zero-length segment exactly on a corner: survives on the boundary.
        ClipSegmentQuery::new(RECT, [0.0, 0.0], [0.0, 0.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohenSutherlandClip::new(&ctx);
    // One batch mixing the deterministic fixtures with a fan of guaranteed
    // crossings, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised, then pinned element-for-
    // element.
    let mut queries = vec![
        ClipSegmentQuery::new(RECT, [2.0, 3.0], [7.0, 8.0]),
        ClipSegmentQuery::new(RECT, [-5.0, -5.0], [15.0, 15.0]),
        ClipSegmentQuery::new(RECT, [-5.0, 2.0], [-1.0, 8.0]),
    ];
    // A fan of segments from the far left into the window at integer heights,
    // each clipped against the left edge.
    for k in 0..16 {
        let y = k as f32;
        queries.push(ClipSegmentQuery::new(RECT, [-6.0, y], [6.0, y]));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_integer_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohenSutherlandClip::new(&ctx);
    let mut state = 0x0f0f_1234_dead_beef_u64;
    // Several workgroups' worth of random integer-coordinate windows and
    // segments. Integer bounds are far from the compare epsilon, so every
    // outcode bit is identical on both devices and the two sides share each
    // accept/reject/clip branch.
    let queries: Vec<ClipSegmentQuery> = (0..200)
        .map(|_| {
            let rect = rand_rect(&mut state, 8);
            ClipSegmentQuery::new(rect, rand_point(&mut state, 12), rand_point(&mut state, 12))
        })
        .collect();
    check(&ctx, &gpu, &queries);
}
