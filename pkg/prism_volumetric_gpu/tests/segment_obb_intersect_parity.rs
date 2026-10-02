//! Real-device parity for the finite-segment vs *oriented bounding box*
//! (`OBB`) slab-clip twin:
//! [`GpuSegmentObbIntersect`](prism_volumetric_gpu::segment_obb_intersect::GpuSegmentObbIntersect)
//! must reproduce the `CPU` golden
//! [`segment_obb_intersect`](prism_render_architecture::particle::segment_obb_intersect::segment_obb_intersect),
//! [`segment_obb_overlaps`](prism_render_architecture::particle::segment_obb_intersect::segment_obb_overlaps)
//! and
//! [`point_in_obb`](prism_render_architecture::particle::segment_obb_intersect::point_in_obb)
//! across an empty batch, an axis-aligned pierce, a parallel-slab inside hit, a
//! parallel-slab clear miss, a segment wholly inside the box, a segment starting
//! inside, a wider box with non-unit half-extents, a zero-length segment, a
//! rotated box exercising the frame transform and a large pseudo-random batch
//! compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of three guarded divisions
//! and a running interval intersection, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the span and hit-point
//! values and asserts an *exact* match on the discrete overlap and containment
//! flags. For the random batch a verdict disagreement is tolerated only when the
//! query sits inside a narrow tie band (a grazing span `|t_exit - t_enter|
//! <= 1e-2`, a segment-endpoint tangency `t_enter` near `1` or `t_exit` near `0`,
//! or an endpoint projection within `1e-2` of a face), the only places where a
//! legal `ULP` perturbation can flip a `<=`/`>` verdict; the named fixtures are
//! all placed clear of such boundaries so they assert exact flags
//! unconditionally.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::segment_obb_intersect`；
//! standard slab-clip finite-segment/`OBB` intersection; no third-party engine
//! source or derived code.

use prism_render_architecture::particle::segment_obb_intersect::{v_dot, v_sub};
use prism_volumetric_gpu::segment_obb_intersect::{
    cpu_reference, GpuSegmentObbIntersect, SegmentObbQuery, SegmentObbResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the span and hit-point values. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Half-width of the tie band inside which a `<=`/`>` verdict can legally flip
/// under a `ULP`-scale perturbation, so a boolean disagreement there is
/// tolerated for the random batch (never for the clear-of-boundary fixtures).
const TIE: f32 = 1.0e-2;

/// The world basis, i.e. an `OBB` that is really an `AABB`.
const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// A fixed `45`-degree basis in the `xy` plane, used to exercise the box-frame
/// transform. The components are a hardcoded unit diagonal so the fixture needs
/// no `f32` transcendental call.
const S: f32 = 0.707_106_77;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Builds one query from a segment and an `OBB`.
fn query(
    p0: [f32; 3],
    p1: [f32; 3],
    center: [f32; 3],
    axes: [[f32; 3]; 3],
    half: [f32; 3],
) -> SegmentObbQuery {
    SegmentObbQuery {
        p0,
        p1,
        center,
        axes,
        half,
    }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: the overlap and both containment flags match exactly, the span
/// endpoints match within tolerance when the segment hits, and the two hit
/// points match within tolerance when the segment hits. Returns the `GPU`
/// verdicts for extra per-test assertions. Use only for fixtures placed clear of
/// every boundary.
fn check(
    ctx: &GpuContext,
    gpu: &GpuSegmentObbIntersect,
    queries: &[SegmentObbQuery],
) -> Vec<SegmentObbResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = cpu_reference(q);
        assert_eq!(
            g.overlaps, want.overlaps,
            "lane {lane}: overlaps gpu {} vs cpu {}",
            g.overlaps, want.overlaps
        );
        assert_eq!(
            g.p0_inside, want.p0_inside,
            "lane {lane}: p0_inside gpu {} vs cpu {}",
            g.p0_inside, want.p0_inside
        );
        assert_eq!(
            g.p1_inside, want.p1_inside,
            "lane {lane}: p1_inside gpu {} vs cpu {}",
            g.p1_inside, want.p1_inside
        );
        if let (Some(gh), Some(wh)) = (g.span, want.span) {
            assert!(
                close(gh.t_enter, wh.t_enter),
                "lane {lane}: t_enter gpu {} vs cpu {}",
                gh.t_enter,
                wh.t_enter
            );
            assert!(
                close(gh.t_exit, wh.t_exit),
                "lane {lane}: t_exit gpu {} vs cpu {}",
                gh.t_exit,
                wh.t_exit
            );
            for k in 0..3 {
                assert!(
                    close(g.enter_point[k], want.enter_point[k]),
                    "lane {lane}: enter_point[{k}] gpu {} vs cpu {}",
                    g.enter_point[k],
                    want.enter_point[k]
                );
                assert!(
                    close(g.exit_point[k], want.exit_point[k]),
                    "lane {lane}: exit_point[{k}] gpu {} vs cpu {}",
                    g.exit_point[k],
                    want.exit_point[k]
                );
            }
        }
    }
    got
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

/// The smallest clearance of a point's per-axis projection to its slab face: a
/// positive value means inside with that much margin, a negative value means
/// outside, and a magnitude within `TIE` marks a containment tie.
fn containment_slack(
    point: [f32; 3],
    center: [f32; 3],
    axes: [[f32; 3]; 3],
    half: [f32; 3],
) -> f32 {
    let m = v_sub(point, center);
    let mut best = f32::INFINITY;
    for (axis, &h) in axes.iter().zip(half.iter()) {
        let slack = h - v_dot(m, *axis).abs();
        best = best.min(slack);
    }
    best
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentObbIntersect::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn axis_aligned_pierce_through_center() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentObbIntersect::new(&ctx);
    // Straight through the unit box along +x: enter at x = -1, exit at x = 1, so
    // the clipped span is [0.25, 0.75]. Both endpoints are far outside the box.
    let q = query(
        [-2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        IDENTITY,
        [1.0, 1.0, 1.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    let span = got[0].span.expect("segment pierces the box");
    assert!(got[0].overlaps, "pierce should overlap");
    assert!(
        !got[0].p0_inside && !got[0].p1_inside,
        "endpoints are outside"
    );
    assert!(close(span.t_enter, 0.25), "t_enter {}", span.t_enter);
    assert!(close(span.t_exit, 0.75), "t_exit {}", span.t_exit);
    assert!(close(got[0].enter_point[0], -1.0), "enter x");
    assert!(close(got[0].exit_point[0], 1.0), "exit x");
}

#[test]
fn parallel_slab_inside_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentObbIntersect::new(&ctx);
    // Travels in y at x = 0.5 (inside the x slab, far from its +/-1 faces) and
    // z = 0: both the x and z axes take the parallel-slab guard with the start
    // inside, and the y slab clips the span to [0.4, 0.6].
    let q = query(
        [0.5, -5.0, 0.0],
        [0.5, 5.0, 0.0],
        [0.0, 0.0, 0.0],
        IDENTITY,
        [1.0, 1.0, 1.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    let span = got[0].span.expect("parallel but inside the slab");
    assert!(close(span.t_enter, 0.4), "t_enter {}", span.t_enter);
    assert!(close(span.t_exit, 0.6), "t_exit {}", span.t_exit);
}

#[test]
fn parallel_slab_outside_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentObbIntersect::new(&ctx);
    // Travels in y at x = 2 — far outside the x slab, so the x-axis parallel
    // guard rejects the whole query. A clear miss on both the overlap flag and
    // the containment flags.
    let q = query(
        [2.0, -5.0, 0.0],
        [2.0, 5.0, 0.0],
        [0.0, 0.0, 0.0],
        IDENTITY,
        [1.0, 1.0, 1.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    assert!(
        !got[0].overlaps,
        "a segment parallel to and outside the x slab must miss"
    );
    assert!(got[0].span.is_none(), "a miss has no span");
    assert!(
        !got[0].p0_inside && !got[0].p1_inside,
        "endpoints are outside"
    );
}

#[test]
fn segment_fully_inside_spans_whole_unit_interval() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentObbIntersect::new(&ctx);
    // Both endpoints lie strictly inside a wide box, so the span clamps to the
    // full [0, 1] range and both containment flags are true.
    let q = query(
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        IDENTITY,
        [2.0, 2.0, 2.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    let span = got[0].span.expect("segment lies inside the box");
    assert!(
        got[0].p0_inside && got[0].p1_inside,
        "both endpoints inside"
    );
    assert!(close(span.t_enter, 0.0), "t_enter {}", span.t_enter);
    assert!(close(span.t_exit, 1.0), "t_exit {}", span.t_exit);
}

#[test]
fn segment_starting_inside_clamps_enter_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentObbIntersect::new(&ctx);
    // Start at the center, exit through the +x face at x = 1: the span clamps
    // its entry to 0 and exits at t = 0.2. Only the start is inside.
    let q = query(
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        IDENTITY,
        [1.0, 1.0, 1.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    let span = got[0].span.expect("start is inside");
    assert!(got[0].p0_inside && !got[0].p1_inside, "only p0 inside");
    assert!(close(span.t_enter, 0.0), "t_enter {}", span.t_enter);
    assert!(close(span.t_exit, 0.2), "t_exit {}", span.t_exit);
}

#[test]
fn non_unit_half_extents_scale_interval() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentObbIntersect::new(&ctx);
    // A box twice as wide on x: enter at x = -2, exit at x = 2 over the
    // [-4, 4] segment, so the span is [0.25, 0.75].
    let q = query(
        [-4.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        IDENTITY,
        [2.0, 1.0, 1.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    let span = got[0].span.expect("wider box on x");
    assert!(close(span.t_enter, 0.25), "t_enter {}", span.t_enter);
    assert!(close(span.t_exit, 0.75), "t_exit {}", span.t_exit);
}

#[test]
fn zero_length_segment_inside_and_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentObbIntersect::new(&ctx);
    // A degenerate point reduces to a containment test: the full [0, 1] span
    // when inside, a miss when outside. Both points are far from any face.
    let inside = query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        IDENTITY,
        [1.0, 1.0, 1.0],
    );
    let outside = query(
        [5.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        IDENTITY,
        [1.0, 1.0, 1.0],
    );
    let got = check(&ctx, &gpu, &[inside, outside]);
    let span = got[0].span.expect("degenerate point inside the box");
    assert!(got[0].p0_inside && got[0].p1_inside, "point is inside");
    assert!(close(span.t_enter, 0.0), "t_enter {}", span.t_enter);
    assert!(close(span.t_exit, 1.0), "t_exit {}", span.t_exit);
    assert!(!got[1].overlaps, "point outside misses");
    assert!(got[1].span.is_none(), "a miss has no span");
}

#[test]
fn rotated_box_transforms_into_frame() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentObbIntersect::new(&ctx);
    // A unit box spun 45 degrees in the xy plane spans world x in
    // [-sqrt(2), sqrt(2)]; a world-x segment from -3 to 3 enters at x = -1.414
    // and exits at x = 1.414, well clear of any tie. This exercises the dot
    // transform into the (non-identity) box frame.
    let axes = [[S, S, 0.0], [-S, S, 0.0], [0.0, 0.0, 1.0]];
    let q = query(
        [-3.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        axes,
        [1.0, 1.0, 1.0],
    );
    let got = check(&ctx, &gpu, &[q]);
    let span = got[0].span.expect("rotated box pierced along world x");
    assert!(got[0].overlaps, "rotated pierce should overlap");
    assert!(span.t_enter < span.t_exit, "ordered span");
    assert!(
        span.t_enter > 0.2 && span.t_enter < 0.3,
        "enter {}",
        span.t_enter
    );
    assert!(
        span.t_exit > 0.7 && span.t_exit < 0.8,
        "exit {}",
        span.t_exit
    );
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentObbIntersect::new(&ctx);
    let mut state = 0x_5e97_0bb1_2357_0001_u64;

    let mut saw_overlap = false;
    let mut saw_miss = false;
    let mut saw_inside = false;

    let rotated = [[S, S, 0.0], [-S, S, 0.0], [0.0, 0.0, 1.0]];

    for round in 0u32..8 {
        // Alternate the box frame so both the identity and transformed paths are
        // exercised across the batch.
        let axes = if round % 2 == 0 { IDENTITY } else { rotated };
        let mut queries = Vec::with_capacity(128);
        for _ in 0..128 {
            let center = [
                lcg(&mut state) * 6.0 - 3.0,
                lcg(&mut state) * 6.0 - 3.0,
                lcg(&mut state) * 6.0 - 3.0,
            ];
            let half = [
                0.5 + lcg(&mut state) * 1.5,
                0.5 + lcg(&mut state) * 1.5,
                0.5 + lcg(&mut state) * 1.5,
            ];
            let p0 = [
                center[0] + lcg(&mut state) * 10.0 - 5.0,
                center[1] + lcg(&mut state) * 10.0 - 5.0,
                center[2] + lcg(&mut state) * 10.0 - 5.0,
            ];
            let p1 = [
                center[0] + lcg(&mut state) * 10.0 - 5.0,
                center[1] + lcg(&mut state) * 10.0 - 5.0,
                center[2] + lcg(&mut state) * 10.0 - 5.0,
            ];
            queries.push(query(p0, p1, center, axes, half));
        }

        let got = gpu.eval(&ctx, &queries);
        assert_eq!(got.len(), queries.len(), "one result per query");
        for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
            let want = cpu_reference(q);

            // The overlap flag matches exactly unless the query sits in a tie
            // band: a grazing span, or a segment-endpoint tangency where the
            // clip clamps t_enter near 1 or t_exit near 0 — the only places a
            // ULP perturbation can legally flip the <= / > verdict.
            if g.overlaps != want.overlaps {
                let near = match (g.span, want.span) {
                    (Some(h), _) | (_, Some(h)) => {
                        (h.t_exit - h.t_enter).abs() <= TIE
                            || (1.0 - h.t_enter).abs() <= TIE
                            || h.t_exit.abs() <= TIE
                    }
                    _ => false,
                };
                assert!(
                    near,
                    "lane {lane}: overlaps disagreement off the tie band: gpu {} cpu {}",
                    g.overlaps, want.overlaps
                );
            }

            // Each containment flag matches exactly unless the endpoint's
            // projection sits within TIE of a face.
            if g.p0_inside != want.p0_inside {
                let slack = containment_slack(q.p0, q.center, q.axes, q.half);
                assert!(
                    slack.abs() <= TIE,
                    "lane {lane}: p0_inside disagreement off the tie band: gpu {} cpu {} slack {slack}",
                    g.p0_inside, want.p0_inside
                );
            }
            if g.p1_inside != want.p1_inside {
                let slack = containment_slack(q.p1, q.center, q.axes, q.half);
                assert!(
                    slack.abs() <= TIE,
                    "lane {lane}: p1_inside disagreement off the tie band: gpu {} cpu {} slack {slack}",
                    g.p1_inside, want.p1_inside
                );
            }

            // Values are compared only where both devices agree on a hit.
            if let (Some(gh), Some(wh)) = (g.span, want.span) {
                assert!(
                    close(gh.t_enter, wh.t_enter),
                    "lane {lane}: t_enter gpu {} vs cpu {}",
                    gh.t_enter,
                    wh.t_enter
                );
                assert!(
                    close(gh.t_exit, wh.t_exit),
                    "lane {lane}: t_exit gpu {} vs cpu {}",
                    gh.t_exit,
                    wh.t_exit
                );
                for k in 0..3 {
                    assert!(
                        close(g.enter_point[k], want.enter_point[k]),
                        "lane {lane}: enter_point[{k}] gpu {} vs cpu {}",
                        g.enter_point[k],
                        want.enter_point[k]
                    );
                    assert!(
                        close(g.exit_point[k], want.exit_point[k]),
                        "lane {lane}: exit_point[{k}] gpu {} vs cpu {}",
                        g.exit_point[k],
                        want.exit_point[k]
                    );
                }
            }

            saw_overlap |= want.overlaps;
            saw_miss |= !want.overlaps;
            saw_inside |= want.p0_inside || want.p1_inside;
        }
    }

    // A large random spread must exercise both verdict classes and at least one
    // interior endpoint, so the test is not trivially passing on a degenerate
    // batch.
    assert!(
        saw_overlap && saw_miss && saw_inside,
        "random batch should produce overlaps, misses and interior endpoints"
    );
}
