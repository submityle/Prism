//! Real-device parity for the 2D `Sutherland-Hodgman` convex-window clip twin:
//! [`GpuSutherlandHodgman2d`](prism_volumetric_gpu::sutherland_hodgman_2d::GpuSutherlandHodgman2d)
//! must reproduce the `CPU` golden
//! [`clip_polygon`](prism_render_architecture::particle::sutherland_hodgman_2d::clip_polygon)
//! across an empty batch, a subject fully inside the window, a square clipped by
//! a square, a triangle-window corner clip, an offset window, a far window that
//! removes the subject, identical polygons, degenerate (`< 3` vertex) inputs, a
//! many-crossing hexagon clip and a large pseudo-random batch of axis-aligned
//! rectangle pairs, compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable walk of cross products and linear
//! interpolations, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The comparison therefore asserts an *exact* match on
//! the surviving vertex count and the per-vertex order, yet a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the vertex coordinates. Every
//! fixture is placed clear of the inside/parallel boundary (the random batch
//! reject-samples so no subject vertex lies near a clip edge line) so the
//! discrete vertex count never flips under a legal perturbation.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sutherland_hodgman_2d`；
//! classic `Sutherland-Hodgman` convex-window polygon clip; no third-party
//! engine source or derived code.

use prism_volumetric_gpu::sutherland_hodgman_2d::{
    cpu_reference, GpuSutherlandHodgman2d, SutherlandHodgman2dQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the vertex coordinates. A `GPU` may fuse a
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

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A rectangle `[x0, x1] x [y0, y1]` wound counter-clockwise, mirroring the
/// `square` helper in the golden test module.
fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<[f32; 2]> {
    Vec::from([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
}

/// Builds one clip query from a subject ring and a convex window ring.
fn query(subject: Vec<[f32; 2]>, clip: Vec<[f32; 2]>) -> SutherlandHodgman2dQuery {
    SutherlandHodgman2dQuery { subject, clip }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: the surviving vertex count matches exactly and every vertex
/// matches in order within tolerance. Use only for fixtures placed clear of
/// every inside/parallel boundary.
fn check(ctx: &GpuContext, gpu: &GpuSutherlandHodgman2d, queries: &[SutherlandHodgman2dQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = cpu_reference(q);
        assert_eq!(
            g.verts.len(),
            want.len(),
            "lane {lane}: vertex count gpu {} vs cpu {}",
            g.verts.len(),
            want.len()
        );
        for (k, (gv, wv)) in g.verts.iter().zip(want.iter()).enumerate() {
            assert!(
                close(gv[0], wv[0]) && close(gv[1], wv[1]),
                "lane {lane} vertex {k}: gpu [{}, {}] vs cpu [{}, {}]",
                gv[0],
                gv[1],
                wv[0],
                wv[1]
            );
        }
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

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSutherlandHodgman2d::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn subject_fully_inside_is_unchanged() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSutherlandHodgman2d::new(&ctx);
    // A small square well inside a larger window: no edge clips, so the four
    // vertices survive in their rotated ring order.
    let q = query(rect(1.0, 1.0, 3.0, 3.0), rect(0.0, 0.0, 4.0, 4.0));
    let got = gpu.eval(&ctx, std::slice::from_ref(&q));
    assert_eq!(
        got[0].verts.len(),
        4,
        "an interior square keeps four vertices"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn square_clips_square_to_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSutherlandHodgman2d::new(&ctx);
    // A larger square cropped to a smaller window: the result is their
    // overlap rectangle, four vertices.
    let q = query(rect(0.0, 0.0, 4.0, 4.0), rect(1.0, 1.0, 3.0, 3.0));
    let got = gpu.eval(&ctx, std::slice::from_ref(&q));
    assert_eq!(
        got[0].verts.len(),
        4,
        "overlap of two squares is a rectangle"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn triangle_window_corner_clip() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSutherlandHodgman2d::new(&ctx);
    // A square cropped by a right-triangle window recovering the lower-left
    // half, a three-vertex ring.
    let q = query(
        rect(0.0, 0.0, 4.0, 4.0),
        Vec::from([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]),
    );
    let got = gpu.eval(&ctx, std::slice::from_ref(&q));
    assert_eq!(got[0].verts.len(), 3, "a triangle window yields a triangle");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn offset_window_clips_to_intersection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSutherlandHodgman2d::new(&ctx);
    // An offset window overlapping only part of the subject.
    let q = query(rect(0.0, 0.0, 4.0, 4.0), rect(-1.0, 1.0, 2.0, 5.0));
    check(&ctx, &gpu, &[q]);
}

#[test]
fn far_window_removes_subject() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSutherlandHodgman2d::new(&ctx);
    // A window far from the subject clips everything away.
    let q = query(rect(0.0, 0.0, 1.0, 1.0), rect(5.0, 5.0, 9.0, 9.0));
    let got = gpu.eval(&ctx, std::slice::from_ref(&q));
    assert!(got[0].verts.is_empty(), "a far window removes the subject");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn identical_polygons_survive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSutherlandHodgman2d::new(&ctx);
    // Clipping a square by an identical square: every vertex sits exactly on a
    // window edge and counts as inside (integer coordinates make the signed side
    // exactly zero on both devices), so all four survive.
    let q = query(rect(0.0, 0.0, 4.0, 4.0), rect(0.0, 0.0, 4.0, 4.0));
    let got = gpu.eval(&ctx, std::slice::from_ref(&q));
    assert_eq!(
        got[0].verts.len(),
        4,
        "identical squares keep four vertices"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn degenerate_subject_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSutherlandHodgman2d::new(&ctx);
    // A two-vertex subject has no face; the clip is empty.
    let q = query(
        Vec::from([[1.0, 1.0], [2.0, 2.0]]),
        rect(0.0, 0.0, 4.0, 4.0),
    );
    let got = gpu.eval(&ctx, std::slice::from_ref(&q));
    assert!(got[0].verts.is_empty(), "a two-vertex subject yields empty");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn degenerate_window_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSutherlandHodgman2d::new(&ctx);
    // A two-vertex window has no half-plane set; the clip is empty.
    let q = query(
        rect(0.0, 0.0, 4.0, 4.0),
        Vec::from([[0.0, 0.0], [1.0, 1.0]]),
    );
    let got = gpu.eval(&ctx, std::slice::from_ref(&q));
    assert!(got[0].verts.is_empty(), "a two-vertex window yields empty");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn hexagon_poking_out_yields_many_crossings() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSutherlandHodgman2d::new(&ctx);
    // A convex hexagon whose top vertex pokes above the window's top edge: the
    // single out-of-window vertex is replaced by two boundary crossings, so the
    // ring grows to seven vertices. Every other vertex is at least two units
    // clear of all four window edges, far from any inside/parallel boundary.
    let subject = Vec::from([
        [2.0, 5.0],
        [5.0, 2.0],
        [8.0, 5.0],
        [8.0, 8.0],
        [5.0, 13.0],
        [2.0, 8.0],
    ]);
    let window = rect(0.0, 0.0, 10.0, 10.0);
    let q = query(subject, window);
    let got = gpu.eval(&ctx, std::slice::from_ref(&q));
    assert_eq!(
        got[0].verts.len(),
        7,
        "one poking vertex becomes two crossings, growing the ring to seven"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn random_rectangle_pairs_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSutherlandHodgman2d::new(&ctx);

    let mut state = 0x5eed_1234_abcd_ef01u64;
    let mut queries: Vec<SutherlandHodgman2dQuery> = Vec::new();
    let mut saw_full = false;
    let mut saw_empty = false;
    let mut saw_clip = false;

    // Reject-sampled axis-aligned rectangle pairs: the clip window is a square
    // and the subject is a rectangle whose every corner is at least `GAP` from
    // all four window edge lines, so no subject vertex lands in the
    // inside/parallel band and the vertex count is unambiguous.
    const GAP: f32 = 0.25;
    while queries.len() < 160 {
        let cx = lcg(&mut state) * 4.0 - 2.0;
        let cy = lcg(&mut state) * 4.0 - 2.0;
        let h = 1.5 + lcg(&mut state) * 1.5;
        let left = cx - h;
        let right = cx + h;
        let bottom = cy - h;
        let top = cy + h;

        let sx = lcg(&mut state) * 8.0 - 4.0;
        let sy = lcg(&mut state) * 8.0 - 4.0;
        let shx = 0.5 + lcg(&mut state) * 1.5;
        let shy = 0.5 + lcg(&mut state) * 1.5;
        let sx0 = sx - shx;
        let sx1 = sx + shx;
        let sy0 = sy - shy;
        let sy1 = sy + shy;

        // Reject when any subject corner coordinate sits within GAP of a window
        // edge line, which is the only configuration that could make a vertex's
        // inside/outside verdict ambiguous.
        let near_x = |x: f32| (x - left).abs() < GAP || (x - right).abs() < GAP;
        let near_y = |y: f32| (y - bottom).abs() < GAP || (y - top).abs() < GAP;
        if near_x(sx0) || near_x(sx1) || near_y(sy0) || near_y(sy1) {
            continue;
        }

        let subject = rect(sx0, sy0, sx1, sy1);
        let window = rect(left, bottom, right, top);
        let out = cpu_reference(&query(subject.clone(), window.clone()));
        match out.len() {
            0 => saw_empty = true,
            4 => {
                if sx0 > left && sx1 < right && sy0 > bottom && sy1 < top {
                    saw_full = true;
                } else {
                    saw_clip = true;
                }
            }
            _ => saw_clip = true,
        }
        queries.push(query(subject, window));
    }

    check(&ctx, &gpu, &queries);

    // A healthy spread must exercise fully-inside, clipped and removed cases, so
    // the batch is not trivially passing on one class.
    assert!(
        saw_full && saw_empty && saw_clip,
        "random batch should produce fully-inside, clipped and removed results"
    );
}
